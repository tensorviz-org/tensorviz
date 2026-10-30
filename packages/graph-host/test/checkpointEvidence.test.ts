import * as assert from 'node:assert/strict';
import { readCheckpointEvidence, checkpointEvidencePage } from '@tensorviz/graph-host/checkpointEvidence';
import { captureRunEvidence, inspectEvidence } from '@tensorviz/graph-host/mcpRunEvidence';

import { checkpointFixture } from './checkpointFixture';

const request = { sourceDigest: 'a'.repeat(64), checkpoint: { file: 'weights.pt', strict: true } };

suite('checkpoint evidence contract', () => {
  test('failed loads keep stored and registered metadata separate from runtime success', () => {
    const evidence = captureRunEvidence({ ok: false, reason: 'Load failed', detail: '', exitCode: 3, envelope: { protocol_version: 1, result: null,
      worker: { checkpointCompatibility: checkpointFixture() }, spans: {}, error: null } }, '/ws', undefined, request);
    const page = inspectEvidence({}, evidence, 'checkpoint', 0, 1) as any;
    assert.equal(page.checkpoint.loadStatus, 'failed'); assert.equal(page.items[0].shape, 'different');
    assert.deepEqual(page.items[0].registered.shape, [2, 3]); assert.deepEqual(page.items[0].stored.shape, [2, 4]);
    assert.equal((evidence.worker as any).checkpointCompatibility, undefined, 'do not echo an unpaged payload in setup');
    assert.equal(page.nextOffset, null);
    assert.equal(page.checkpoint.counts.shapeDifferences, 1);
    assert.equal(page.checkpoint.counts.dtypeDifferences, 0);
    assert.match(page.detail, /Historical/); assert.match(page.detail, /does not verify/);
  });

  test('success is not inferred from metadata; mismatched dtype can accompany successful loading', () => {
    const value = checkpointFixture(); value.loadStatus = 'succeeded';
    value.entries[0].stored.shape = [2, 3]; value.entries[0].shape = 'same';
    value.entries[0].stored.dtype = 'torch.float64'; value.entries[0].dtype = 'different';
    const evidence = captureRunEvidence({ ok: true, exitCode: 0, envelope: { protocol_version: 1,
      result: { diagnostics: { execution: { checkpointCompatibility: value } } }, worker: {}, spans: {}, error: null } }, '/ws', undefined, request);
    assert.equal((inspectEvidence({}, evidence, 'checkpoint', 0, 1) as any).checkpoint.loadStatus, 'succeeded');
    assert.equal((evidence.setup as any).checkpointCompatibility, undefined);
  });

  test('wrong source/checkpoint, forged comparisons, duplicates and unknown fields are refused', () => {
    for (const change of ['source', 'file', 'key', 'strict', 'digest', 'shape', 'presence', 'kind', 'complete', 'duplicate', 'extra', 'unbounded']) {
      const value: any = checkpointFixture();
      if (change === 'source') value.sourceDigest = 'c'.repeat(64);
      if (change === 'file') value.checkpoint.file = 'other.pt';
      if (change === 'key') value.checkpoint.key = 'other';
      if (change === 'strict') value.checkpoint.strict = false;
      if (change === 'digest') value.checkpoint.digest = 'invalid';
      if (change === 'shape') value.entries[0].shape = 'same';
      if (change === 'presence') value.entries[0].presence = 'registered_only';
      if (change === 'kind') value.entries[0].registered.kind = 'stored';
      if (change === 'complete') value.omissions = ['entry_limit'];
      if (change === 'duplicate') value.entries.push(value.entries[0]);
      if (change === 'extra') value.private = 'not exposed';
      if (change === 'unbounded') value.entries[0].name = 'x'.repeat(1024 * 1024);
      assert.equal(readCheckpointEvidence(value, request), undefined, change);
    }
    assert.equal(readCheckpointEvidence(checkpointFixture()), undefined);
    assert.equal(readCheckpointEvidence(checkpointFixture(), { sourceDigest: request.sourceDigest }), undefined);
    assert.equal(checkpointEvidencePage(undefined, 0, 10).checkpoint, null);
  });

  test('large bounded reports page every entry without inlining the inventory in summary', () => {
    const report: any = checkpointFixture();
    report.entries = Array.from({length:200}, (_, i) => ({...report.entries[0], name:`${i}-${'x'.repeat(1900)}`}));
    const parsed = readCheckpointEvidence(report, request)!; assert.ok(parsed);
    let offset = 0, seen = 0;
    for (;;) {
      const page = checkpointEvidencePage(parsed, offset, 200);
      assert.ok(Buffer.byteLength(JSON.stringify(page)) < 64 * 1024);
      assert.ok(page.items!.length > 0); seen += page.items!.length;
      assert.equal((page.checkpoint as any).entries, undefined);
      assert.equal(page.checkpoint!.counts.shapeDifferences, 200, 'counts cover retained rows, not only this page');
      if (page.nextOffset == null) break;
      assert.ok(page.nextOffset > offset); offset = page.nextOffset;
    }
    assert.equal(seen, 200);
  });
});
