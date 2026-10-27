import * as assert from 'node:assert/strict';
import { ValidationSummaryStore } from '@tensorviz/graph-host/validationSummary';
import { LocalBrowserValidation } from '../../localBrowserValidation';
import type { LocalBrowserSync } from '../../localBrowserSync';
import { isLocalBrowserMessage, type LocalBrowserTargetIdentity, type LocalBrowserValidationSummary, type ProjectionPayload } from '@tensorviz/contracts/protocol';
import type { ResolvedExecutionPlan } from '../../executionPlan';
import type { TraceOutcome } from '../../worker';

const target: LocalBrowserTargetIdentity = { workspaceFolder: '/ws', file: 'model.py', cls: 'Net',
  sourceDigest: 'a'.repeat(64), documentVersion: 1, backendRevision: 1 };

suite('host validation summaries', () => {
  test('a no-capture failure does not borrow the preceding measured revision', () => {
    const store = new ValidationSummaryStore(() => {});
    store.begin('one', target)('passed', 'Measured.', { target, revision: 1 });
    store.begin('two', target)('failed', 'Timed out.');
    assert.deepEqual(store.snapshot(target), { runId: 'two', target, status: 'failed', detail: 'Timed out.',
      evidence: store.ledger.attribution(target) });
    const copy = store.snapshot(target)!;
    copy.target.sourceDigest = 'b'.repeat(64);
    assert.equal(store.snapshot(target)!.target.sourceDigest, target.sourceDigest);
    assert.equal(store.snapshot({ ...target, cls: 'Other' }), undefined);
  });

  test('older progress and late completion cannot overwrite the latest request', () => {
    const published: LocalBrowserValidationSummary[] = [];
    const store = new ValidationSummaryStore(summary => published.push(summary));
    const old = store.begin('one', target);
    store.begin('two', target)('refused', 'Busy.');
    old('passed', 'Late success.', { target, revision: 2 });
    assert.equal(store.snapshot(target)!.runId, 'two');
    assert.equal(published.length, 4, 'late evidence refreshes attribution without overwriting the latest verdict');
    assert.equal(published[3].runId, 'two');
  });

  test('manual results retain only the projection actually produced by their run', async () => {
    const store = new ValidationSummaryStore(() => {});
    const next = { ...target, backendRevision: 2 };
    const sync = { validate: async (_target: unknown, _signal: unknown, _inputs: unknown, _progress: unknown,
      measured: (target: LocalBrowserTargetIdentity, projection: ProjectionPayload) => void) => {
      measured(next, { revision: 2 } as ProjectionPayload);
      return { status: 'failed', detail: 'Partial forward.' };
    } } as unknown as LocalBrowserSync;
    const profile = { kind: 'model', model: 'model:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
      init: { args: [], kwargs: {} }, inputs: { args: [], kwargs: {} }, limits: { timeout_ms: 1000 } };
    const manager = new LocalBrowserValidation(sync, () => {}, {
      snapshot: async (_root, selected) => (selected
        ? { guards: {}, resolution: { status: 'resolved', spec: { profile, identity: { profile: selected } } } }
        : { profiles: ['one'], profileDefinitions: { one: profile }, guards: {}, resolution: { status: 'needs_setup', requirements: [
          { profile: null, field: ['profiles'], code: 'ambiguous', phase: 'static', reason: 'Choose.', remedy: 'Choose.' },
        ] } }) as never,
      choose: async () => undefined,
    }, store);
    manager.handle({ type: 'validateRequest', protocolVersion: 1, workspaceSessionId: 'session', messageId: 'm', requestId: 'run', target });
    assert.equal(store.snapshot(target)!.status, 'running');
    await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(store.snapshot(next), { runId: 'run', target: next, status: 'failed', detail: 'Partial forward.', projectionRevision: 2,
      evidence: store.ledger.attribution(next) });
  });

  test('timeout attribution keeps prior forward evidence and exposes only safe setup labels', () => {
    const published: LocalBrowserValidationSummary[] = [];
    const store = new ValidationSummaryStore(summary => published.push(summary));
    const first = store.begin('one', target, 'mcp');
    const capture: TraceOutcome = { ok: true, exitCode: 0, envelope: { protocol_version: 1,
      result: { graph: {} }, spans: {}, error: null, worker: {} } };
    first.attempt.captured(capture); store.ledger.projected(target, capture.envelope); first('passed', 'Measured.', { target, revision: 1 });
    const second = store.begin('two', target, 'manual');
    second.attempt.resolved({ applicabilityDigest: 'b'.repeat(64), plan: {
      source: { id: 'source:two' }, origin: 'profile', profileName: 'tiny',
      execution: { mode: 'eval', device: 'cpu', inputs: { privateData: 'secret-value' } },
      environment: { executable: '/private/path/python' },
    } as unknown as ResolvedExecutionPlan });
    second.attempt.captured({ ok: false, timedOut: true, reason: 'timeout', detail: '' });
    second('failed', 'Timed out.');
    const evidence = store.snapshot(target)!.evidence!;
    assert.equal(evidence.attemptStatus, 'timed_out');
    assert.equal(evidence.displayedCapture!.attemptId, 'one');
    assert.equal(evidence.displayedCapture!.forwardStatus, 'passed');
    assert.equal(evidence.displayedCapture!.status, 'partial');
    assert.equal(evidence.lastCompleteCaptureId, null);
    assert.deepEqual(evidence.setup, { origin: 'profile', profileName: 'tiny', mode: 'eval', device: 'cpu' });
    assert.ok(!JSON.stringify(published).includes('secret-value'));
    assert.ok(!JSON.stringify(published).includes('/private/path'));
    for (const summary of published) assert.ok(isLocalBrowserMessage({ type: 'validationState', protocolVersion: 1,
      workspaceSessionId: 'session', messageId: 'message', ...summary }), 'real host emissions satisfy the wire contract');
    evidence.setup!.profileName = 'changed';
    assert.equal(store.snapshot(target)!.evidence!.setup!.profileName, 'tiny');
    store.ledger.projected(target);
    assert.equal(store.snapshot(target)!.evidence!.displayedCapture, null, 'source-only refresh clears displayed attribution');
  });

  test('an older successful worker refreshes capture attribution after a newer busy refusal', () => {
    const published: LocalBrowserValidationSummary[] = [];
    const store = new ValidationSummaryStore(summary => published.push(summary));
    const old = store.begin('old', target);
    store.begin('busy', target)('refused', 'Busy.');
    const capture: TraceOutcome = { ok: true, exitCode: 0, envelope: { protocol_version: 1,
      result: { graph: {} }, spans: {}, error: null, worker: {} } };
    old.attempt.captured(capture); store.ledger.projected(target, capture.envelope); old('passed', 'Measured.', { target, revision: 2 });
    assert.equal(published.at(-1)!.runId, 'busy');
    assert.equal(published.at(-1)!.status, 'refused');
    assert.equal(published.at(-1)!.evidence!.displayedCapture!.attemptId, 'old');
    assert.equal(published.at(-1)!.projectionRevision, undefined);
  });
});
