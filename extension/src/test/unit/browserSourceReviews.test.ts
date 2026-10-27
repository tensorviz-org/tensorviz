import * as assert from 'node:assert/strict';
import { BrowserSourceReviews } from '../../browserSourceReviews';
import { CandidateHistory } from '../../candidateHistory';
import { CandidateReview } from '../../candidateReview';
import { GraphToolError } from '@tensorviz/graph-host/mcpGraph';
import type { CandidateExecution } from '../../executionPlan';
import type { CandidateRunResult } from '@tensorviz/graph-host/candidateExecution';
import type { SourceProposalOutcome } from '../../localBrowserSourceEdits';
import { isLocalBrowserMessage, type CandidateAction, type LocalBrowserSourceEditRequestMessage } from '@tensorviz/contracts/protocol';
import { canonicalFixture, appliedFixture } from '../../../../tests/fixtures/semantic.cjs';

const target = { workspaceFolder: '/ws', file: 'model.py', cls: 'Model', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
const tick = () => new Promise<void>(resolve => setImmediate(resolve));
function request(id = 'proposal', viewId = '1'): LocalBrowserSourceEditRequestMessage {
  return { type: 'sourceEditRequest', protocolVersion: 1, workspaceSessionId: 'session', messageId: 'request',
    requestId: id, viewId, target, edit: { kind: 'remove_layer', nodeId: 'fc' } };
}
function result(): CandidateRunResult {
  const { capture, plan } = canonicalFixture('candidate:browser');
  return { evidence: { execution: { id: plan.id, plan } as CandidateExecution, capture, baseCurrent: true, candidateCurrent: true },
    outcome: { ok: true, executionCurrent: true, exitCode: 0, envelope: { protocol_version: 1, result: {}, error: null, worker: {}, spans: {} } } };
}
function harness() {
  const history = new CandidateHistory(), published: any[] = [], seen: string[] = [];
  let candidate: CandidateReview | undefined, finish!: (value: SourceProposalOutcome) => void, calls = 0;
  let execute = async (_signal: AbortSignal) => result();
  let inspectApplied = async (_scope: string) => appliedFixture();
  const service = new BrowserSourceReviews({
    async propose(message, signal, onReview) {
      seen.push(message.requestId);
      const metadata = { title: 'Review', file: 'model.py', beforeDigest: 'a'.repeat(64), afterDigest: 'b'.repeat(64) };
      candidate = new CandidateReview(canonicalFixture().capture, signal!, () => true,
        async abort => { calls++; return execute(abort); }, { history, requestId: message.requestId });
      candidate.onDidChange(value => onReview?.({ ...metadata, candidate: value }));
      onReview?.({ ...metadata, candidate: candidate.snapshot() });
      const outcome = await new Promise<SourceProposalOutcome>(resolve => {
        finish = resolve; signal!.addEventListener('abort', () => resolve({ status: 'cancelled', detail: 'Stopped.' }), { once: true });
      });
      await candidate.finish(); return outcome;
    },
    runCandidate(scope) {
      if (scope !== seen.at(-1) || !candidate?.available) throw new GraphToolError('CANDIDATE_UNAVAILABLE', 'No review.');
      void candidate.run(); return history.get(scope, candidate.snapshot().runId!);
    }, getCandidate: (scope, id) => history.get(scope, id), listCandidates: scope => history.list(scope),
    inspectCandidate: (scope, id, query) => history.inspect(scope, id, query),
    cancelCandidate(scope, id) { const record = history.get(scope, id); if (scope === seen.at(-1) && candidate?.snapshot().runId === id) candidate.cancel(); return record; },
    forgetCandidates: scope => history.forget(scope),
    inspectAppliedSource: scope => inspectApplied(scope),
  }, message => {
    assert.ok(isLocalBrowserMessage({ protocolVersion: 1, workspaceSessionId: 'session', messageId: 'reply', ...message }), JSON.stringify(message));
    published.push(message);
  });
  const control = (action: CandidateAction, overrides = {}) => {
    service.control({ type: 'candidateRequest', protocolVersion: 1, workspaceSessionId: 'session', messageId: 'command',
      target, viewId: '1', proposalId: 'proposal', requestId: 'candidate-read', action, ...overrides });
    return published.at(-1);
  };
  return { service, history, published, seen, control, get calls() { return calls; },
    finish: () => finish({ status: 'applied', detail: 'Applied explicitly.' }), setRunner: (fn: typeof execute) => { execute = fn; },
    setAppliedReader: (fn: typeof inspectApplied) => { inspectApplied = fn; } };
}

suite('browser candidate source review ownership', () => {
  test('applied-source reads require confirmed Apply and stay separate from candidate history', async () => {
    const h = harness(); h.service.handle(request()); let inspected = 0;
    h.setAppliedReader(async scope => { inspected++; assert.equal(scope, h.seen[0]); return appliedFixture(); });
    assert.equal(h.control({ kind: 'reconcile' }).result, null); assert.equal(inspected, 0);
    h.control({ kind: 'run', retryId: 'one' }); await tick(); h.finish(); await tick();
    const historical = h.history.list(h.seen[0]);
    h.control({ kind: 'reconcile' }); await tick();
    assert.deepEqual(h.published.at(-1).result.applied, appliedFixture());
    assert.deepEqual(h.published.at(-1).result.candidates, []);
    assert.deepEqual(h.history.list(h.seen[0]), historical); assert.equal(inspected, 1); assert.equal(h.calls, 1);
    h.service.clear();
  });

  for (const removal of ['disconnect', 'expiry']) test(`${removal} suppresses an applied-source reply that completes late`, async () => {
    const h = harness(); h.service.handle(request()); h.finish(); await tick();
    let release!: (value: ReturnType<typeof appliedFixture>) => void;
    h.setAppliedReader(async () => new Promise(resolve => { release = resolve; }));
    h.control({ kind: 'reconcile' }); const before = h.published.length;
    const now = Date.now;
    try {
      if (removal === 'disconnect') h.service.cancelOwner('1');
      else { const later = now() + 31 * 60_000; Date.now = () => later; }
      release(appliedFixture()); await tick(); assert.equal(h.published.length, before);
    } finally { Date.now = now; h.service.clear(); }
  });

  test('scoped semantic evidence survives Apply and remains private to its owning view', async () => {
    const h = harness();
    h.setRunner(async () => {
      const value = result();
      value.evidence!.postconditions = { version: 1, executionId: value.evidence!.execution.id, captureId: value.evidence!.capture!.id,
        recipe: 'binary_scalar_v1', scope: 'observed_expression', status: 'verified', reason: 'observed_requested_literal',
        bindingId: 'binding:literal', observedOperations: 1 };
      return value;
    });
    h.service.handle(request()); h.control({ kind: 'run', retryId: 'semantic' }); await tick();
    const before = h.control({ kind: 'list' }).result.candidates[0];
    assert.equal(before.summary.semanticPostconditions, 'verified');
    assert.equal(before.summary.postconditions.scope, 'observed_expression');
    h.finish(); await tick();
    const after = h.control({ kind: 'inspect', runId: before.runId }).result.candidates[0];
    assert.equal(after.review, 'historical'); assert.deepEqual(after.summary.postconditions, before.summary.postconditions);
    assert.equal(h.control({ kind: 'inspect', runId: before.runId }, { viewId: '2' }).result, null);
    h.service.clear();
  });

  test('preparation is static and host source IDs cannot collide with MCP or browser IDs', async () => {
    const h = harness(); h.service.handle(request('mcp-guessed'));
    assert.equal(h.calls, 0); assert.match(h.seen[0], /^browser:/); assert.notEqual(h.seen[0], 'mcp-guessed');
    assert.equal(h.published[0].type, 'sourceEditReview');
    h.service.handle(request('mcp-guessed', '2')); assert.equal(h.published.at(-1).status, 'refused');
    assert.equal(h.seen.length, 1); h.service.clear(); await tick();
  });
  test('every control resolves view/session/proposal/target ownership before executing or reading', async () => {
    const h = harness(); h.service.handle(request());
    for (const action of [{ kind: 'run', retryId: 'retry' }, { kind: 'list' }, { kind: 'reconcile' }, { kind: 'cancel', runId: 'guess' }, { kind: 'inspect', runId: 'guess' }] as CandidateAction[]) {
      for (const changed of [{ viewId: '2' }, { workspaceSessionId: 'other' }, { proposalId: 'other' }, { target: { ...target, documentVersion: 2 } }]) {
        assert.equal(h.control(action, changed).result, null);
      }
    }
    assert.equal(h.calls, 0); h.service.clear(); await tick();
  });
  test('explicit run retries share identity after Apply and never reexecute evicted results', async () => {
    const h = harness(); h.service.handle(request());
    const first = h.control({ kind: 'run', retryId: 'retry' }).result.candidates[0];
    assert.equal(first.summary.runtime, 'running');
    assert.equal(h.control({ kind: 'run', retryId: 'retry' }).result.candidates[0].runId, first.runId);
    await tick(); assert.equal(h.calls, 1); h.finish(); await tick();
    assert.equal(h.published.at(-1).status, 'applied');
    const record = h.control({ kind: 'run', retryId: 'retry' }).result.candidates[0];
    assert.equal(record.review, 'historical');
    assert.equal(h.control({ kind: 'run', retryId: 'new' }).result, null);
    const inspect = h.control({ kind: 'inspect', runId: record.runId, query: { ...record.capture, path: ['operations'], offset: 0, limit: 1 } });
    assert.equal(inspect.result.page.items.length, 1); assert.equal(inspect.result.page.nextOffset, 1);
    h.history.clear(); assert.equal(h.control({ kind: 'run', retryId: 'retry' }).result, null);
    assert.equal(h.calls, 1); h.service.clear();
  });
  test('disconnect cancels its worker and late completion cannot recreate evidence or publish replies', async () => {
    const h = harness(); let release!: (value: CandidateRunResult) => void, signal: AbortSignal | undefined;
    h.setRunner(abort => { signal = abort; return new Promise(resolve => { release = resolve; }); });
    h.service.handle(request()); h.control({ kind: 'run', retryId: 'retry' }); await tick();
    h.service.cancelOwner('2'); assert.equal(signal!.aborted, false);
    h.service.cancelOwner('1'); assert.equal(signal!.aborted, true);
    const before = h.published.length; release(result()); await tick();
    assert.equal(h.published.length, before); assert.deepEqual(h.history.list(h.seen[0]), []);
    assert.equal(h.control({ kind: 'list' }).result, null);
  });
  test('legacy source requests remain reviewable but cannot grant candidate access', async () => {
    const h = harness(), message = request(); delete message.viewId;
    h.service.handle(message); assert.equal(h.published.length, 0);
    assert.equal(h.control({ kind: 'run', retryId: 'retry' }).result, null);
    h.finish(); await tick(); assert.equal(h.published.at(-1).status, 'applied'); h.service.clear();
  });
  test('cancellation retains a historical summary until expiry and cannot replay a proposal', async () => {
    const h = harness(); h.service.handle(request()); h.control({ kind: 'run', retryId: 'retry' }); await tick();
    h.service.cancel('proposal'); await tick();
    assert.equal(h.control({ kind: 'list' }).result.candidates[0].review, 'historical');
    h.service.handle(request()); assert.equal(h.seen.length, 1); assert.equal(h.published.at(-1).status, 'cancelled');
    const now = Date.now;
    try { const later = now() + 31 * 60_000; Date.now = () => later; assert.equal(h.control({ kind: 'list' }).result, null); }
    finally { Date.now = now; }
    assert.deepEqual(h.history.list(h.seen[0]), []); h.service.clear();
  });
});
