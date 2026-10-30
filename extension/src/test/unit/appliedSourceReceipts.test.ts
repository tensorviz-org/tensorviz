import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { AppliedSourceReceipts, APPLIED_SOURCE_LIMITS } from '../../appliedSourceReceipts';
import type { AppliedSourceEvidence } from '../../sourceReconciliation';

function evidence(): AppliedSourceEvidence {
  const { current: _current, ...input } = JSON.parse(readFileSync(join(__dirname, '../../../../tests/fixtures/contracts/source-reconciliation.json'), 'utf8')).request;
  return input;
}
const target = { workspaceFolder: '/work', file: 'model.py', cls: 'Model', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };

suite('confirmed native Apply receipts', () => {
  test('preparation is not Apply; only the exact reservation confirms immutable private evidence', () => {
    let now = 10;
    const store = new AppliedSourceReceipts(APPLIED_SOURCE_LIMITS, () => now), input = evidence(), base = structuredClone(input);
    const selected = { ...target }, reservation = store.reserve('review', selected, input)!;
    assert.equal(store.summary('review'), null); assert.equal(store.read('review'), undefined);
    store.confirm('other', reservation); store.confirm('review', 'wrong');
    assert.equal(store.summary('review'), null);
    assert.equal(store.reserve('review', selected, input), undefined);
    input.reviewedSource = 'changed'; input.sources['model.py'] = 'changed'; selected.cls = 'Other';
    now = 20; store.confirm('review', reservation);
    const first = store.read('review')!;
    assert.deepEqual(first.evidence, base); assert.deepEqual(first.target, target);
    assert.equal(first.receipt.appliedAt, 20); assert.equal(first.receipt.unavailable, null);
    assert.ok(!JSON.stringify(store.summary('review')).includes(base.reviewedSource));
    assert.equal(first.evidence!.version, 1);
    (first.evidence as AppliedSourceEvidence).reviewedSource = 'changed'; first.target.cls = 'Other'; first.receipt.file = 'other.py';
    assert.deepEqual(store.read('review')!.evidence, base); assert.equal(store.summary('review')!.file, 'model.py');
    now = 30; store.confirm('review', reservation); store.discard('review', reservation);
    assert.equal(store.summary('review')!.appliedAt, 20, 'late callbacks cannot renew retention');
  });

  test('Discard removes pending evidence; old confirmation cannot confirm a replacement request', () => {
    const store = new AppliedSourceReceipts(), first = store.reserve('review', target, evidence())!;
    store.discard('review', first); store.confirm('review', first);
    assert.equal(store.read('review'), undefined);
    const second = store.reserve('review', target, evidence())!;
    store.confirm('review', first); store.discard('review', first);
    assert.equal(store.summary('review'), null);
    store.confirm('review', second); assert.ok(store.summary('review'));
  });

  for (const state of ['pending', 'applied']) {
    test(`forget and dispose permanently revoke ${state} receipts even when Apply completes late`, () => {
      const store = new AppliedSourceReceipts(), reservation = store.reserve('review', target, evidence())!;
      if (state === 'applied') store.confirm('review', reservation);
      store.forget('review'); store.confirm('review', reservation);
      assert.equal(store.read('review'), undefined);
      const second = store.reserve('review', target, evidence())!;
      store.dispose(); store.confirm('review', second);
      assert.equal(store.reserve('review', target, evidence()), undefined);
      assert.equal(store.read('review'), undefined);
    });
  }

  test('pending and confirmed entries expire without a late completion reviving them', () => {
    let now = 0;
    const store = new AppliedSourceReceipts({ ...APPLIED_SOURCE_LIMITS, ttlMs: 100 }, () => now);
    const pending = store.reserve('pending', target, evidence())!;
    const applied = store.reserve('applied', target, evidence())!;
    now = 50; store.confirm('applied', applied);
    now = 100; store.confirm('pending', pending);
    assert.equal(store.read('pending'), undefined); assert.ok(store.read('applied'));
    now = 150; assert.equal(store.read('applied'), undefined);
    store.confirm('applied', applied); assert.equal(store.summary('applied'), null);
  });

  test('record eviction cannot be reversed by a late Apply', () => {
    const store = new AppliedSourceReceipts({ ...APPLIED_SOURCE_LIMITS, records: 1 });
    const first = store.reserve('first', target, evidence())!;
    const second = store.reserve('second', target, evidence())!;
    store.confirm('first', first); assert.equal(store.read('first'), undefined);
    store.confirm('second', second); assert.ok(store.read('second'));
  });

  test('oversized evidence keeps the confirmed receipt with an explicit limit', () => {
    const store = new AppliedSourceReceipts({ ...APPLIED_SOURCE_LIMITS, evidenceBytes: 1 });
    const id = store.reserve('review', target, evidence())!;
    store.confirm('review', id);
    assert.equal(store.summary('review')!.unavailable, 'too_large');
    assert.equal(store.read('review')!.evidence, undefined);
  });

  test('total bytes evict old source/capture bodies without losing the native Apply observation', () => {
    const input = evidence(), bytes = Buffer.byteLength(JSON.stringify(input));
    const store = new AppliedSourceReceipts({ ...APPLIED_SOURCE_LIMITS, totalBytes: bytes });
    const first = store.reserve('first', target, input)!; store.confirm('first', first);
    const second = store.reserve('second', target, input)!; store.confirm('second', second);
    assert.equal(store.summary('first')!.unavailable, 'evicted');
    assert.equal(store.read('first')!.evidence, undefined);
    assert.deepEqual(store.read('second')!.evidence, input);
  });
});
