import * as assert from 'node:assert/strict';
import { AttemptLedger } from '@tensorviz/graph-host/attemptLedger';
import type { ResolvedExecutionPlan } from '@tensorviz/graph-host/executionPlanCore';
import type { LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import type { TraceOutcome, WorkerEnvelope } from '@tensorviz/graph-host/traceWorkerCore';
import { canonicalFixture, automaticFixture } from '../../../tests/fixtures/semantic.cjs';

const target: LocalBrowserTargetIdentity = { workspaceFolder: '/ws', file: 'model.py', cls: 'Net',
  sourceDigest: 'a'.repeat(64), documentVersion: 1, backendRevision: 1 };
const result = (ok = true): TraceOutcome => ({ ok, exitCode: ok ? 0 : 4, reason: 'forward', detail: '',
  envelope: { protocol_version: 1, result: { graph: { nodes: [{ id: 'observed' }] } }, spans: {},
    error: ok ? null : { type: 'AssertionError', message: 'private model detail', frames: [] }, worker: {} } });

suite('semantic attempt and retained capture ledger', () => {
  test('prepared calls retain immutable source and capture identity without inventing an executable plan', () => {
    const ledger = new AttemptLedger();
    const { capture, preparation } = automaticFixture('auto');
    const attempt = ledger.begin('auto', target, 'automatic', null);
    attempt.prepared({ preparation, applicabilityDigest: capture.preparationDigest });
    preparation.source.files[0].digest = 'b'.repeat(64);
    attempt.resolved({ plan: canonicalFixture('auto').plan, applicabilityDigest: 'b'.repeat(64) });
    const outcome = result();
    outcome.envelope!.result = { diagnostics: { semantic_capture: capture } };
    attempt.captured(outcome); ledger.projected(target, outcome.envelope); attempt.finish('passed');
    const state = ledger.state(target), retained = ledger.capture(capture.id)!;
    assert.equal(state.latestAttempt!.sourceId, capture.source.id);
    assert.equal(state.latestAttempt!.planDigest, null);
    assert.equal(state.lastCompleteCaptureId, capture.id);
    assert.equal(retained.record.preparationDigest, capture.preparationDigest);
    assert.deepEqual(retained.preparation?.source, capture.source);
    assert.deepEqual(ledger.capturePreparation(target), retained.preparation);
    assert.equal(ledger.executionPlan(target), undefined);
    assert.equal(ledger.attribution(target)!.setup, null);
    ledger.begin('refused', target, 'manual', null).finish('refused');
    assert.equal(ledger.capturePreparation(target), undefined);
    assert.equal(ledger.state(target).displayedCaptureId, capture.id);
  });
  test('complete, partial and displayed canonical evidence retain independent slots', () => {
    const ledger = new AttemptLedger();
    const run = (id: string, partial = false) => {
      const { capture, plan } = canonicalFixture(id);
      capture.id = `capture:${id}`;
      if (partial) { capture.status = 'partial'; capture.coverage.omissions = ['operation_limit']; }
      const handle = ledger.begin(id, target, 'manual', null);
      handle.resolved({ plan, applicabilityDigest: capture.planDigest });
      const outcome = result();
      outcome.envelope!.result = { diagnostics: { semantic_capture: capture } };
      handle.captured(outcome); handle.finish('passed');
      return { capture, outcome };
    };
    const first = run('first');
    ledger.projected(target, first.outcome.envelope);
    const second = run('second', true);
    const state = ledger.state(target);
    assert.equal(state.lastCompleteCaptureId, first.capture.id);
    assert.equal(state.lastPartialCaptureId, second.capture.id);
    assert.equal(state.displayedCaptureId, first.capture.id);
    assert.deepEqual(state.latestAttempt!.captureIds, [second.capture.id]);
    assert.equal(ledger.capture(first.capture.id)!.record.format, 'semantic');
    assert.equal(ledger.capture(second.capture.id)!.record.forwardStatus, 'passed', 'partial capture does not mean a failed forward');
    ledger.projected(target, second.outcome.envelope);
    assert.equal(ledger.state(target).displayedCaptureId, second.capture.id);
    assert.equal(ledger.state(target).lastCompleteCaptureId, first.capture.id);
  });

  test('rejected semantic payloads stay explicitly legacy and never occupy the complete slot', () => {
    const ledger = new AttemptLedger();
    const { capture, plan } = canonicalFixture('one');
    const handle = ledger.begin('one', target, 'manual', null);
    handle.resolved({ plan, applicabilityDigest: capture.planDigest });
    capture.source.files[0].digest = 'b'.repeat(64);
    const outcome = result();
    outcome.envelope!.result = { diagnostics: { semantic_capture: capture } };
    handle.captured(outcome); handle.finish('passed');
    const state = ledger.state(target);
    assert.equal(state.lastCompleteCaptureId, null);
    const record = ledger.capture(state.lastPartialCaptureId!)!.record;
    assert.equal(record.format, 'legacy');
    assert.equal(record.semanticRejection, 'invalid_capture');
  });

  test('a duplicate capture ID cannot overwrite evidence displayed from a different attempt', () => {
    const ledger = new AttemptLedger();
    for (const id of ['one', 'two']) {
      const { capture, plan } = canonicalFixture(id);
      const handle = ledger.begin(id, target, 'manual', null);
      handle.resolved({ plan, applicabilityDigest: capture.planDigest });
      const outcome = result();
      outcome.envelope!.result = { diagnostics: { semantic_capture: capture } };
      handle.captured(outcome); handle.finish('passed');
      if (id === 'one') ledger.projected(target, outcome.envelope);
    }
    const state = ledger.state(target);
    assert.equal(ledger.capture(state.displayedCaptureId!)!.record.attemptId, 'one');
    assert.equal(ledger.capture(state.lastPartialCaptureId!)!.record.semanticRejection, 'invalid_capture');
  });
  test('timeout and refusal do not overwrite displayed evidence or invent resolved setup', () => {
    const ledger = new AttemptLedger();
    const first = ledger.begin('one', target, 'manual', null), captured = result();
    first.captured(captured);
    ledger.projected(target, captured.envelope);
    first.finish('passed');
    const id = ledger.state(target).displayedCaptureId!;
    assert.equal(ledger.capture(id)!.record.format, 'legacy');
    assert.equal(ledger.state(target).lastCompleteCaptureId, null, 'legacy success does not prove complete semantics');
    const second = ledger.begin('two', target, 'manual', null);
    second.captured({ ok: false, timedOut: true, reason: 'timeout', detail: '' });
    second.finish('failed');
    assert.equal(ledger.state(target).latestAttempt!.status, 'timed_out');
    assert.deepEqual(ledger.state(target).latestAttempt!.captureIds, []);
    assert.equal(ledger.state(target).displayedCaptureId, id);
    const third = ledger.begin('three', target, 'manual', null);
    third.finish('refused');
    assert.equal(ledger.state(target).latestAttempt!.status, 'needs_setup');
    assert.equal(ledger.state(target).latestAttempt!.planDigest, null);
    assert.equal(ledger.state(target).lastPartialCaptureId, id);
  });

  test('projection failure retains its new capture separately from the graph on screen', () => {
    const ledger = new AttemptLedger();
    const one = ledger.begin('one', target, 'manual', null), first = result();
    one.captured(first); ledger.projected(target, first.envelope); one.finish('passed');
    const displayed = ledger.state(target).displayedCaptureId!;
    const two = ledger.begin('two', target, 'mcp', { inputs: [] });
    two.captured(result(false)); two.phase('projection'); two.finish('refused');
    const state = ledger.state(target);
    assert.equal(state.displayedCaptureId, displayed);
    assert.notEqual(state.lastPartialCaptureId, displayed);
    assert.equal(state.latestAttempt!.phase, 'projection');
    assert.equal(state.latestAttempt!.status, 'failed');
    const retained = ledger.capture(state.lastPartialCaptureId!)!;
    retained.envelope!.result = null;
    assert.ok(ledger.capture(state.lastPartialCaptureId!)!.envelope!.result, 'inspection cannot mutate retained evidence');
    ledger.projected({ ...target, sourceDigest: 'b'.repeat(64) });
    assert.equal(ledger.state(target).displayedCaptureId, null, 'source-only refresh clears measured identity');
    assert.equal(ledger.state(target).lastPartialCaptureId, state.lastPartialCaptureId);
  });

  test('old completions and retargeting never move newer capture slots', () => {
    const ledger = new AttemptLedger();
    const old = ledger.begin('old', target, 'manual', null);
    const latest = ledger.begin('latest', target, 'mcp', null), fresh = result();
    latest.captured(fresh); ledger.projected(target, fresh.envelope); latest.finish('passed');
    const state = ledger.state(target);
    old.captured(result()); old.finish('passed');
    assert.deepEqual(ledger.state(target), state);
    const other = { ...target, cls: 'Other' };
    ledger.begin('other', other, 'automatic', null);
    assert.equal(ledger.capture(state.displayedCaptureId!), undefined);
    assert.equal(ledger.state(target).latestAttempt, null);
    assert.equal(ledger.state(other).displayedCaptureId, null);
  });

  test('a newer busy refusal preserves the independently completing displayed capture', () => {
    const ledger = new AttemptLedger();
    const running = ledger.begin('running', target, 'manual', null);
    ledger.begin('busy', target, 'manual', null).finish('refused');
    const measured = result();
    running.captured(measured); ledger.projected(target, measured.envelope); running.finish('passed');
    const state = ledger.state(target);
    assert.equal(state.latestAttempt!.id, 'busy');
    assert.equal(state.latestAttempt!.status, 'needs_setup');
    assert.equal(ledger.capture(state.displayedCaptureId!)!.record.attemptId, 'running');
  });

  test('finished attempts are immutable and resolved source/plan identity is copied once', () => {
    const ledger = new AttemptLedger();
    const handle = ledger.begin('one', target, 'manual', { setup: 'profile' });
    const plan = { source: { id: 'source:resolved' } } as ResolvedExecutionPlan;
    handle.resolved({ plan, applicabilityDigest: 'c'.repeat(64) });
    plan.source.id = 'mutated';
    handle.resolved({ plan, applicabilityDigest: 'd'.repeat(64) });
    handle.finish('cancelled');
    const state = ledger.state(target);
    handle.phase('projection'); handle.captured(result()); handle.finish('passed');
    assert.deepEqual(ledger.state(target), state);
    assert.equal(state.latestAttempt!.sourceId, 'source:resolved');
    assert.equal(state.latestAttempt!.planDigest, 'c'.repeat(64));
    assert.equal(ledger.executionPlan(target)!.source.id, 'source:resolved', 'resolved plan survives a stopped attempt with no capture');
  });

  test('retention is bounded without losing the fact that a capture existed', () => {
    const ledger = new AttemptLedger(512);
    const capture = result(), envelope = capture.envelope as WorkerEnvelope;
    envelope.result = { oversized: 'x'.repeat(1024) };
    const attempt = ledger.begin('one', target, 'manual', null);
    attempt.captured(capture); attempt.finish('failed');
    const id = ledger.state(target).lastPartialCaptureId!;
    assert.equal(ledger.capture(id)!.record.unavailable, 'retention_limit');
    assert.equal(ledger.capture(id)!.record.retainedBytes, 0);
    assert.equal(ledger.capture(id)!.envelope, undefined);
    for (let index = 0; index < 5; index++) {
      const newer = ledger.begin(`run:${index}`, target, 'manual', null);
      newer.captured(result()); newer.finish('passed');
    }
    assert.equal(ledger.capture(id), undefined, 'only the current complete/partial/displayed slots retain data');
  });

  suite('execution cache: a reused capture', () => {
    const fresh = (ledger: AttemptLedger, id: string) => {
      const { capture, plan } = canonicalFixture(id);
      capture.id = `capture:${id}`;
      const handle = ledger.begin(id, target, 'manual', null);
      handle.resolved({ plan, applicabilityDigest: capture.planDigest });
      const outcome = result();
      outcome.envelope!.result = { diagnostics: { semantic_capture: capture } };
      handle.captured(outcome); ledger.projected(target, outcome.envelope); handle.finish('passed');
      return { capture, plan, outcome };
    };
    const reuse = (ledger: AttemptLedger, id: string, original: { capture: ReturnType<typeof canonicalFixture>['capture']; plan: ResolvedExecutionPlan },
      override: Partial<{ captureId: string; planDigest: string }> = {}) => {
      const handle = ledger.begin(id, target, 'manual', null);
      // The new attempt resolves its own plan (new ids, same digest).
      handle.resolved({ plan: { ...canonicalFixture(id).plan }, applicabilityDigest: original.capture.planDigest });
      const envelope: WorkerEnvelope = { protocol_version: 1, spans: {}, error: null, worker: {},
        result: { diagnostics: { semantic_capture: structuredClone(original.capture) } } };
      const outcome: TraceOutcome = { ok: true, exitCode: 0, envelope, reused: { attemptId: original.capture.attemptId,
        captureId: override.captureId ?? original.capture.id, plan: structuredClone(original.plan),
        planDigest: override.planDigest ?? original.capture.planDigest, capturedAt: 1 } };
      handle.captured(outcome); ledger.projected(target, envelope); handle.finish('passed');
      return { handle, envelope };
    };

    test('keeps the original attempt identity and records which attempt reused it', () => {
      const ledger = new AttemptLedger();
      const a = fresh(ledger, 'first');
      reuse(ledger, 'second', a);
      const state = ledger.state(target), attempt = state.latestAttempt!;
      assert.equal(attempt.id, 'second');
      assert.deepEqual(attempt.captureIds, [a.capture.id]);
      assert.equal(attempt.reusedCaptureId, a.capture.id);
      assert.equal(attempt.status, 'succeeded');
      assert.equal(state.lastCompleteCaptureId, a.capture.id);
      assert.equal(state.displayedCaptureId, a.capture.id, 'the reused capture is the displayed evidence');
      const retained = ledger.capture(a.capture.id)!;
      assert.equal(retained.record.format, 'semantic', 'a reused id is never downgraded to legacy');
      assert.equal(retained.record.attemptId, 'first');
      assert.deepEqual(retained.record.reusedBy, ['second']);
      assert.equal(retained.plan!.id, 'first', 'replay/evidence use the plan that produced the capture');
      const evidence = ledger.attribution(target)!;
      assert.equal(evidence.displayedCapture!.attemptId, 'first');
      assert.equal(evidence.displayedCapture!.format, 'semantic');
    });

    test('a reused capture no longer retained (A -> B -> A) is inserted with its original identity', () => {
      const ledger = new AttemptLedger();
      const a = fresh(ledger, 'a1');
      fresh(ledger, 'b');
      fresh(ledger, 'b2');                    // A's capture leaves every slot
      assert.equal(ledger.capture(a.capture.id), undefined);
      reuse(ledger, 'a2', a);
      const retained = ledger.capture(a.capture.id)!;
      assert.equal(retained.record.attemptId, 'a1');
      assert.equal(retained.record.sourceId, a.plan.source.id);
      assert.equal(retained.record.status, 'complete');
      assert.equal(retained.plan!.id, 'a1');
      assert.equal(ledger.state(target).latestAttempt!.reusedCaptureId, a.capture.id);
    });

    test('a reuse claim that does not match the payload is not trusted as reuse', () => {
      const ledger = new AttemptLedger();
      const a = fresh(ledger, 'first');
      reuse(ledger, 'liar', a, { captureId: 'capture:other' });
      const attempt = ledger.state(target).latestAttempt!;
      assert.equal(attempt.reusedCaptureId ?? null, null);
      assert.notEqual(attempt.captureIds[0], a.capture.id);
      assert.equal(ledger.capture(attempt.captureIds[0])!.record.format, 'legacy');
      reuse(ledger, 'other-digest', a, { planDigest: 'f'.repeat(64) });
      assert.equal(ledger.state(target).latestAttempt!.reusedCaptureId ?? null, null);
    });
  });
});
