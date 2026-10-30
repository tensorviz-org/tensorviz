import * as assert from 'node:assert/strict';
import { AttemptLedger } from '@tensorviz/graph-host/attemptLedger';
import { CaptureInspector } from '@tensorviz/graph-host/captureInspection';
import { ComputationIndex } from '@tensorviz/graph-host/computationQuery';
import { analyzeComputationChange } from '@tensorviz/graph-host/computationChange';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import type { TraceOutcome } from '@tensorviz/graph-host/traceWorkerCore';
import { canonicalFixture } from '../../../tests/fixtures/semantic.cjs';

const target = { workspaceFolder: '/ws', file: 'model.py', cls: 'Net',
  documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
class CountingLedger extends AttemptLedger {
  reads = 0;
  override capture(id: string) { this.reads++; return super.capture(id); }
}
function retain(ledger: AttemptLedger, id: string, options: {
  displayed?: boolean; partial?: boolean; prepare?: (capture: CaptureSnapshot) => void;
} = {}) {
  const { capture, plan } = canonicalFixture(id);
  capture.id = `capture:${id}`;
  if (options.partial) { capture.status = 'partial'; capture.coverage.omissions = ['operation_limit']; }
  options.prepare?.(capture); plan.source = structuredClone(capture.source);
  if (capture.backend === 'jaxpr') plan.execution = { ...plan.execution, framework: 'jax' };
  const handle = ledger.begin(id, target, 'manual', null);
  handle.resolved({ plan, applicabilityDigest: capture.planDigest });
  const outcome: TraceOutcome = { ok: true, exitCode: 0, envelope: { protocol_version: 1,
    result: { diagnostics: { semantic_capture: capture } }, spans: {}, error: null, worker: {} } };
  handle.captured(outcome); handle.finish('passed');
  if (options.displayed !== false) ledger.projected(target, outcome.envelope);
  assert.equal(ledger.captureIdentity(capture.id)?.record.format, 'semantic', 'the fixture must pass actual capture admission');
  return capture;
}
const query = (capture: CaptureSnapshot) => ({ captureId: capture.id, sourceId: capture.source.id,
  anchors: [{ kind: 'value', id: 'h' }], direction: 'downstream', depth: 20, limit: 100 });
const change = (capture: CaptureSnapshot) => ({ captureId: capture.id, sourceId: capture.source.id,
  changes: [{ valueId: 'h', shape: [2, 8] }], depth: 20, limit: 100 });
const find = (capture: CaptureSnapshot) => ({ captureId: capture.id, sourceId: capture.source.id, text: 'add' });

suite('immutable retained computation index reuse', () => {
  test('query pages and analysis match cold results while sharing one parsed payload', () => {
    const ledger = new CountingLedger(), capture = retain(ledger, 'first');
    const inspector = new CaptureInspector(ledger), cold = new CaptureInspector(ledger, 0);
    const freshQuery = cold.queryComputation(target, query(capture));
    const freshAnalysis = cold.analyzeComputation(target, change(capture));
    const freshFind = cold.findComputation(target, find(capture));
    ledger.reads = 0;
    assert.deepEqual(inspector.findComputation(target, find(capture)), freshFind);
    assert.deepEqual(inspector.queryComputation(target, query(capture)), freshQuery);
    assert.deepEqual(inspector.analyzeComputation(target, change(capture)), freshAnalysis);
    assert.deepEqual(inspector.queryComputation(target, query(capture)), freshQuery);
    const page = { ...query(capture), offset: 2, limit: 2 };
    const warmPage = inspector.queryComputation(target, page);
    assert.equal(ledger.reads, 1, 'paging and shape analysis must not deserialize the capture again');
    assert.deepEqual(warmPage, cold.queryComputation(target, page));
    ledger.reads = 0; inspector.list(target); inspector.assertRetained(target, capture.id, capture.source.id);
    assert.equal(ledger.reads, 0, 'metadata and post-read ownership checks need no payload parse');
  });

  test('returned records, edges, source and analysis cannot poison a subsequent query', () => {
    const ledger = new CountingLedger(), capture = retain(ledger, 'first'), inspector = new CaptureInspector(ledger);
    const first = inspector.queryComputation(target, query(capture)), expected = structuredClone(first);
    const analysis = inspector.analyzeComputation(target, change(capture)), expectedAnalysis = structuredClone(analysis);
    const found = inspector.findComputation(target, find(capture)), expectedFound = structuredClone(found);
    (found.items[0].record as any).target = 'poison'; found.items[0].matchedFields.push('poison');
    first.source.files[0].digest = 'b'.repeat(64);
    first.page.evidence.omissions.push('opaque_kernel');
    const operation = first.page.items.find(item => item.kind === 'entity' && item.ref.id === 'add1');
    assert.ok(operation && 'record' in operation); (operation.record as any).target = 'poison';
    const edge = first.page.items.find(item => item.kind === 'edge');
    assert.ok(edge && edge.kind === 'edge'); edge.source.id = 'poison'; edge.path.push('poison');
    analysis.analysis.changes[0].shape[0] = 999;
    analysis.analysis.values[0].derivedShape!.push(999);
    assert.deepEqual(inspector.queryComputation(target, query(capture)), expected);
    assert.deepEqual(inspector.analyzeComputation(target, change(capture)), expectedAnalysis);
    assert.deepEqual(inspector.findComputation(target, find(capture)), expectedFound);
    assert.equal(ledger.reads, 1);
    assert.deepEqual(inspector.retained(target, capture.id, capture.source.id), capture);
  });

  test('warm entries still enforce model and source ownership and remain historical after saved changes', () => {
    const ledger = new CountingLedger(), capture = retain(ledger, 'first'), inspector = new CaptureInspector(ledger);
    const first = inspector.queryComputation(target, query(capture));
    const found = inspector.findComputation(target, find(capture));
    for (const method of ['queryComputation', 'analyzeComputation', 'findComputation'] as const) {
      const args = method === 'queryComputation' ? query(capture) : method === 'analyzeComputation' ? change(capture) : find(capture);
      assert.throws(() => inspector[method]({ ...target, cls: 'Other' }, args), { code: 'UNKNOWN_CAPTURE' });
      assert.throws(() => inspector[method](target, { ...args, sourceId: 'other' }), { code: 'SOURCE_MISMATCH' });
    }
    ledger.begin('failed-new-source', { ...target, documentVersion: 2, sourceDigest: 'b'.repeat(64) }, 'manual', null).finish('failed');
    assert.deepEqual(inspector.queryComputation({ ...target, documentVersion: 2 }, query(capture)), first);
    assert.deepEqual(inspector.findComputation({ ...target, documentVersion: 2 }, find(capture)), found);
    assert.equal(first.page.evidence.appliesToCurrentRuntime, 'not_established');
    assert.equal(ledger.reads, 1);
    ledger.begin('retarget', { ...target, cls: 'Other' }, 'manual', null);
    assert.throws(() => inspector.queryComputation(target, query(capture)), { code: 'UNKNOWN_CAPTURE' });
    assert.throws(() => inspector.analyzeComputation(target, change(capture)), { code: 'UNKNOWN_CAPTURE' });
    assert.throws(() => inspector.findComputation(target, find(capture)), { code: 'UNKNOWN_CAPTURE' });
  });

  test('eviction and reinsertion with identical external IDs never reuse the old computation', () => {
    const ledger = new CountingLedger(), capture = retain(ledger, 'same'), inspector = new CaptureInspector(ledger);
    const oldToken = ledger.captureIdentity(capture.id)!.token;
    const first = inspector.queryComputation(target, query(capture));
    const record = ledger.captureIdentity(capture.id)!.record; record.unavailable = 'retention_limit';
    assert.equal(ledger.captureIdentity(capture.id)!.record.unavailable, null, 'identity metadata is owned by the caller');
    retain(ledger, 'replacement');
    assert.throws(() => inspector.queryComputation(target, query(capture)), { code: 'UNKNOWN_CAPTURE' });
    assert.throws(() => inspector.findComputation(target, find(capture)), { code: 'UNKNOWN_CAPTURE' });
    const next = retain(ledger, 'same', { prepare: value => { value.operations[0].arguments.args[1] = { kind: 'int', value: '2' }; } });
    assert.equal(next.source.id, capture.source.id);
    assert.notEqual(ledger.captureIdentity(next.id)!.token, oldToken);
    const second = inspector.queryComputation(target, query(next));
    assert.notDeepEqual(second, first);
    assert.deepEqual(second.page, new ComputationIndex(next).query(query(next)));
    assert.deepEqual(inspector.findComputation(target, find(next)), new ComputationIndex(next).find(find(next)));
    assert.equal(ledger.reads, 2);
  });

  test('the two-entry bound evicts the least recently queried still-retained capture', () => {
    const ledger = new CountingLedger(), inspector = new CaptureInspector(ledger);
    const first = retain(ledger, 'first');
    const second = retain(ledger, 'second', { partial: true, displayed: false });
    const third = retain(ledger, 'third', { displayed: false });
    assert.equal(inspector.list(target).captures.length, 3, 'all three ledger slots must remain live');
    inspector.queryComputation(target, query(first)); inspector.queryComputation(target, query(second));
    inspector.queryComputation(target, query(first)); inspector.queryComputation(target, query(third));
    assert.equal(ledger.reads, 3);
    inspector.queryComputation(target, query(first)); assert.equal(ledger.reads, 3);
    inspector.queryComputation(target, query(second)); assert.equal(ledger.reads, 4);
  });

  test('existing execution-reuse bookkeeping preserves index identity without relabelling evidence', () => {
    const ledger = new CountingLedger(), capture = retain(ledger, 'first'), inspector = new CaptureInspector(ledger);
    const first = inspector.queryComputation(target, query(capture)), token = ledger.captureIdentity(capture.id)!.token;
    const plan = canonicalFixture(capture.attemptId).plan;
    const attempt = ledger.begin('reused', target, 'manual', null);
    attempt.resolved({ plan: canonicalFixture('reused').plan, applicabilityDigest: capture.planDigest });
    attempt.captured({ ok: true, exitCode: 0, envelope: { protocol_version: 1,
      result: { diagnostics: { semantic_capture: capture } }, spans: {}, error: null, worker: {} },
    reused: { attemptId: capture.attemptId, captureId: capture.id, plan, planDigest: capture.planDigest!, capturedAt: 1 } });
    attempt.finish('passed');
    assert.equal(ledger.captureIdentity(capture.id)!.token, token);
    assert.deepEqual(ledger.captureIdentity(capture.id)!.record.reusedBy, ['reused']);
    assert.deepEqual(inspector.queryComputation(target, query(capture)), first);
    assert.equal(first.page.evidence.appliesToCurrentRuntime, 'not_established');
    assert.equal(ledger.reads, 1);
  });

  test('serialized-input cap counts the sum and oversized inputs take the uncached path', () => {
    const ledger = new CountingLedger(), first = retain(ledger, 'first');
    const second = retain(ledger, 'second', { partial: true, displayed: false });
    const budget = Math.max(...[first, second].map(capture => ledger.captureIdentity(capture.id)!.record.retainedBytes));
    const inspector = new CaptureInspector(ledger, budget);
    inspector.queryComputation(target, query(first)); inspector.queryComputation(target, query(second));
    inspector.queryComputation(target, query(second)); assert.equal(ledger.reads, 2);
    inspector.queryComputation(target, query(first)); assert.equal(ledger.reads, 3);
    const uncached = new CaptureInspector(ledger, 1);
    const result = uncached.queryComputation(target, query(first));
    assert.deepEqual(uncached.queryComputation(target, query(first)), result);
    assert.equal(ledger.reads, 5, 'a capture over the cap is still queryable without retention');
  });

  test('five minutes of inactivity expires an index while each query refreshes its idle time', () => {
    const ledger = new CountingLedger(), capture = retain(ledger, 'first'), inspector = new CaptureInspector(ledger);
    const actualNow = Date.now; let now = actualNow();
    try {
      Date.now = () => now;
      const first = inspector.queryComputation(target, query(capture));
      now += 299_999; assert.deepEqual(inspector.queryComputation(target, query(capture)), first);
      now += 299_999; inspector.analyzeComputation(target, change(capture)); assert.equal(ledger.reads, 1);
      now += 300_000; assert.deepEqual(inspector.queryComputation(target, query(capture)), first);
      assert.equal(ledger.reads, 2);
    } finally { Date.now = actualNow; }
  });

  test('cached partial JAX evidence retains omissions and unsupported dimension results', () => {
    const ledger = new CountingLedger(), capture = retain(ledger, 'partial', { partial: true,
      prepare: value => { value.backend = 'jaxpr'; value.coverage.scope = 'traced_program'; value.torchVersion = null; value.jaxVersion = '0.10.2'; } });
    const inspector = new CaptureInspector(ledger), first = inspector.queryComputation(target, query(capture));
    assert.equal(first.page.evidence.status, 'partial');
    assert.equal(first.page.evidence.scope, 'traced_program');
    assert.deepEqual(first.page.evidence.omissions, ['operation_limit']);
    const analysis = inspector.analyzeComputation(target, change(capture));
    assert.deepEqual(analysis.analysis.summary, { compatible: 0, incompatible: 0, unknown: 2 });
    assert.ok(analysis.analysis.checks.every(check => check.reason === 'runtime_aten_capture_required'));
    assert.deepEqual(inspector.queryComputation(target, query(capture)), first); assert.equal(ledger.reads, 1);
  });

  test('a prepared analysis index must own the exact capture even when all external IDs agree', () => {
    const capture = canonicalFixture().capture, index = new ComputationIndex(capture);
    assert.deepEqual(analyzeComputationChange(capture, change(capture), index), analyzeComputationChange(capture, change(capture)));
    assert.throws(() => analyzeComputationChange(structuredClone(capture), change(capture), index), { code: 'INVALID_CAPTURE' });
  });
});
