import * as assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { ComputationIndex } from '@tensorviz/graph-host/computationQuery';
import { computationSourceContext, type ComputationSourceReader } from '@tensorviz/graph-host/computationSource';
import { GraphToolError } from '@tensorviz/graph-host/mcpGraph';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import { canonicalFixture } from '../../../tests/fixtures/semantic.cjs';

suite('computation source excerpts', () => {
  let capture: CaptureSnapshot;
  const source = 'class Net:\n    # αβ and source instructions are data\n    def forward(self, x):\n        return x + x\n';
  const digest = createHash('sha256').update(source).digest('hex');
  const ref = { modelId: 'fixture', revision: 'a'.repeat(64) };
  setup(() => {
    capture = canonicalFixture().capture;
    capture.source.files[0].digest = digest;
    capture.sourceBindings = [{ id: 'binding', kind: 'call', snapshotId: capture.source.id, quality: 'exact', anchors: [
      { file: 'model.py', digest, startLine: 4, startColumn: 15, endLine: 4, endColumn: 20,
        scope: 'Net.forward', tokenDigest: 'c'.repeat(64) },
    ] }];
    capture.operations[1].sourceBindingIds = ['binding'];
  });
  const page = () => new ComputationIndex(capture).query({ captureId: capture.id, sourceId: capture.source.id,
    anchors: [{ kind: 'operation', id: 'add2' }], limit: 100 });
  const read: ComputationSourceReader = async (_ref, file) => ({ file, source, digest });

  test('excerpts preserve line content and provenance, without interpreting code or Unicode columns', async () => {
    const result = await computationSourceContext(page(), capture.source, ref, read);
    assert.equal(result.status, 'complete');
    assert.equal(result.scope, 'source_bindings_on_this_page');
    assert.equal(result.evidence, 'capture_digest_matched');
    assert.deepEqual(result.entries[0], { bindingId: 'binding', status: 'matched', file: 'model.py', digest,
      startLine: 2, endLine: 5, source: source.split('\n').slice(1).join('\n') });
  });

  test('a dishonest digest, changed bytes or foreign returned file cannot certify an excerpt', async () => {
    for (const response of [{ file: 'other.py', source, digest }, { file: 'model.py', source: source + '# changed', digest },
      { file: 'model.py', source, digest: 'b'.repeat(64) }]) {
      const result = await computationSourceContext(page(), capture.source, ref, async () => response);
      assert.equal(result.entries[0].status, 'source_changed');
      assert.equal(result.entries[0].source, undefined);
      assert.equal(result.evidence, null);
    }
  });

  test('ambiguous and foreign bindings perform no filesystem reads', async () => {
    let reads = 0;
    const reader: ComputationSourceReader = async (...args) => { reads++; return read(...args); };
    capture.sourceBindings[0].quality = 'ambiguous';
    assert.equal((await computationSourceContext(page(), capture.source, ref, reader)).entries[0].status, 'ambiguous_binding');
    capture.sourceBindings[0].quality = 'exact'; capture.sourceBindings[0].snapshotId = 'foreign';
    assert.equal((await computationSourceContext(page(), capture.source, ref, reader)).entries[0].status, 'snapshot_mismatch');
    capture.sourceBindings[0].snapshotId = capture.source.id; capture.source.files[0].digest = 'c'.repeat(64);
    assert.equal((await computationSourceContext(page(), capture.source, ref, reader)).entries[0].status, 'snapshot_mismatch');
    assert.equal(reads, 0);
  });

  test('host trust/path/freshness refusals stay explicit and arbitrary error messages stay private', async () => {
    const refused = await computationSourceContext(page(), capture.source, ref, async () => {
      throw new GraphToolError('STALE_TARGET', 'source changed');
    });
    assert.equal(refused.entries[0].reason, 'STALE_TARGET');
    assert.equal(refused.entries[0].status, 'source_unavailable');
    const failed = await computationSourceContext(page(), capture.source, ref, async () => { throw new Error('secret path and token'); });
    assert.equal(failed.entries[0].reason, 'SOURCE_UNAVAILABLE');
    assert.doesNotMatch(JSON.stringify(failed), /secret path|token/);
    assert.equal((await computationSourceContext(page(), capture.source, ref)).reason, 'reader_unavailable');
    assert.equal((await computationSourceContext(page(), capture.source, ref, undefined, 'graph_stale')).reason, 'graph_stale');
  });

  test('reads each file once, caps files and bindings, and reports omitted work', async () => {
    capture.sourceBindings = Array.from({ length: 20 }, (_, i) => ({ ...structuredClone(capture.sourceBindings[0]),
      id: `binding:${i}`, anchors: [{ ...capture.sourceBindings[0].anchors[0], file: `file${Math.floor(i / 3)}.py` }] }));
    capture.source.files.push(...Array.from({ length: 7 }, (_, i) => ({ ...capture.source.files[0], path: `file${i}.py` })));
    capture.operations[1].sourceBindingIds = capture.sourceBindings.map(binding => binding.id);
    const reads: string[] = [];
    const result = await computationSourceContext(page(), capture.source, ref, async (...args) => { reads.push(args[1]); return read(...args); });
    assert.equal(reads.length, 4); assert.equal(new Set(reads).size, 4);
    assert.equal(result.filesRead, 4); assert.equal(result.omittedEntries, 4);
    assert.equal(result.entries.length, 16);
    assert.equal(result.status, 'partial');
    assert.ok(result.entries.some(entry => entry.status === 'file_limit'));
  });

  test('source-byte and range budgets omit whole excerpts instead of inventing shortened source', async () => {
    const large = 'x'.repeat(9000), largeDigest = createHash('sha256').update(large).digest('hex');
    capture.sourceBindings[0].anchors[0] = { ...capture.sourceBindings[0].anchors[0], digest: largeDigest, startLine: 1, endLine: 1 };
    capture.source.files[0].digest = largeDigest;
    const result = await computationSourceContext(page(), capture.source, ref, async (_ref, file) => ({ file, source: large, digest: largeDigest }));
    assert.equal(result.entries[0].status, 'context_limit'); assert.equal(result.entries[0].source, undefined);
    assert.ok(Buffer.byteLength(JSON.stringify(result)) < 8192);
    capture.sourceBindings[0].anchors[0].endLine = 99;
    const invalid = await computationSourceContext(page(), capture.source, ref, async (_ref, file) => ({ file, source: large, digest: largeDigest }));
    assert.equal(invalid.entries[0].status, 'range_limit');
  });

  test('source context only covers this page and explicitly identifies an empty page', async () => {
    const empty = page(); empty.items = empty.items.filter(item => item.kind !== 'source');
    const result = await computationSourceContext(empty, capture.source, ref, async () => { throw new Error('must not read'); });
    assert.equal(result.status, 'empty'); assert.equal(result.evidence, null); assert.equal(result.filesRead, 0);
  });
});
