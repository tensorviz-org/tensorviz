import * as assert from 'node:assert/strict';
import { CandidateReview, candidateReviewText } from '../../candidateReview';
import type { CandidateRunResult } from '@tensorviz/graph-host/candidateExecution';
import type { CandidateExecution } from '../../executionPlan';
import { canonicalFixture } from '../../../../tests/fixtures/semantic.cjs';

function outcome(): CandidateRunResult {
  return { outcome: { ok: true, exitCode: 0, executionCurrent: true,
    recheckExecution: async () => true, envelope: { protocol_version: 1, result: {}, error: null, spans: {}, worker: {} } },
    evidence: { baseCurrent: true, candidateCurrent: true, execution: { id: 'candidate:test' } as CandidateExecution,
      capture: canonicalFixture().capture } };
}

suite('reviewed candidate result state', () => {
  test('does not execute during review construction or inspection', async () => {
    const review = new CandidateReview(canonicalFixture().capture, new AbortController().signal,
      () => true, async () => { assert.fail('execution needs the explicit run action'); });
    assert.equal(review.snapshot().runtime, 'not_run');
    assert.equal((await review.refresh()).runtime, 'not_run');
    assert.match(candidateReviewText(review.snapshot()), /Semantic postconditions: not checked/);
    await review.finish(); await review.run();
  });

  test('successful forward, incomplete capture, changed state and postconditions remain independent', async () => {
    const base = canonicalFixture().capture, result = outcome();
    base.inputDigest = 'a'.repeat(64); base.stateDigest = 'b'.repeat(64);
    result.evidence!.capture!.inputDigest = base.inputDigest;
    result.evidence!.capture!.stateDigest = 'c'.repeat(64);
    result.evidence!.capture!.status = 'partial';
    const review = new CandidateReview(base, new AbortController().signal, () => true, async () => result);
    const state = await review.run();
    assert.equal(state.runtime, 'succeeded'); assert.equal(state.capture, 'partial');
    assert.equal(state.inputs, 'same'); assert.equal(state.state, 'different');
    assert.equal(state.semanticPostconditions, 'not_checked');
    state.runtime = 'failed'; assert.equal(review.snapshot().runtime, 'succeeded');
    await review.finish();
  });

  test('missing input/state identities cannot establish equivalence', async () => {
    const base = canonicalFixture().capture, result = outcome();
    base.inputDigest = base.stateDigest = null;
    result.evidence!.capture!.inputDigest = result.evidence!.capture!.stateDigest = null;
    const review = new CandidateReview(base, new AbortController().signal, () => true, async () => result);
    assert.equal((await review.run()).inputs, 'unavailable');
    assert.equal(review.snapshot().state, 'unavailable'); await review.finish();
  });

  test('semantic evidence is immutable and remains scoped when source becomes stale', async () => {
    const result = outcome(); let current = true;
    result.evidence!.postconditions = { version: 1, executionId: 'candidate:test', captureId: result.evidence!.capture!.id,
      scope: 'observed_expression', recipe: 'binary_scalar_v1', status: 'verified', reason: 'observed_requested_literal',
      bindingId: 'binding:literal', observedOperations: 3 };
    const review = new CandidateReview(canonicalFixture().capture, new AbortController().signal, () => current, async () => result);
    await review.run(); current = false;
    result.evidence!.postconditions.observedOperations = 99;
    const summary = review.snapshot();
    assert.equal(summary.freshness, 'stale'); assert.equal(summary.semanticPostconditions, 'verified');
    assert.equal(summary.postconditions!.observedOperations, 3);
    assert.match(candidateReviewText(summary), /3 captured occurrences. Other executions are unverified/);
    await review.finish();
  });

  test('runtime success cannot override a failed semantic check', async () => {
    const result = outcome();
    result.evidence!.postconditions = { version: 1, executionId: 'candidate:test', captureId: result.evidence!.capture!.id,
      scope: 'observed_expression', recipe: 'binary_scalar_v1', status: 'failed', reason: 'observed_literal_mismatch',
      bindingId: 'binding:literal', observedOperations: 1 };
    const review = new CandidateReview(canonicalFixture().capture, new AbortController().signal, () => true, async () => result);
    const summary = await review.run();
    assert.equal(summary.runtime, 'succeeded'); assert.equal(summary.semanticPostconditions, 'failed');
    assert.match(candidateReviewText(summary), /Recorded literal differs from the request/);
    await review.finish();
  });

  test('a failed forward can still have current partial evidence', async () => {
    const result = outcome();
    result.outcome = { ...result.outcome, ok: false, reason: 'failure', detail: '', exitCode: 4 };
    result.evidence!.capture!.status = 'partial';
    const review = new CandidateReview(canonicalFixture().capture, new AbortController().signal, () => true, async () => result);
    assert.equal((await review.run()).runtime, 'failed');
    assert.equal(review.snapshot().freshness, 'current'); await review.finish();
  });

  for (const missing of ['evidence', 'executionCurrent', 'recheck'] as const) test(`${missing} cannot certify current validation`, async () => {
    const result = outcome();
    if (missing === 'evidence') delete result.evidence;
    if (missing === 'executionCurrent') delete result.outcome.executionCurrent;
    if (missing === 'recheck') delete result.outcome.recheckExecution;
    const review = new CandidateReview(canonicalFixture().capture, new AbortController().signal, () => true, async () => result);
    await review.run();
    assert.notEqual((await review.refresh()).freshness, 'current');
    if (missing === 'evidence') assert.equal(review.snapshot().runtime, 'refused');
    await review.finish();
  });

  test('changed source and changed execution setup latch stale state', async () => {
    for (const change of ['source', 'environment']) {
      let current = true;
      const result = outcome(); result.outcome.recheckExecution = async () => current;
      const review = new CandidateReview(canonicalFixture().capture, new AbortController().signal,
        () => change === 'source' ? current : true, async () => result);
      await review.run(); current = false;
      assert.equal((await review.refresh()).freshness, 'stale');
      current = true; assert.equal((await review.refresh()).freshness, 'stale'); await review.finish();
    }
  });

  test('repeated clicks share one run, including synchronous result listeners', async () => {
    let calls = 0, release!: (result: CandidateRunResult) => void;
    const gate = new Promise<CandidateRunResult>(resolve => { release = resolve; });
    const review = new CandidateReview(canonicalFixture().capture, new AbortController().signal, () => true,
      () => { calls++; return gate; });
    let repeated: Promise<unknown> | undefined;
    review.onDidChange(state => { if (state.runtime === 'running') repeated = review.run(); });
    const first = review.run(); assert.equal(first, review.run()); assert.equal(first, repeated);
    await Promise.resolve(); assert.equal(calls, 1); release(outcome()); await first; await review.finish();
  });

  for (const action of ['cancel', 'parent', 'finish']) test(`${action} waits for cleanup and prevents a late success`, async () => {
    const parent = new AbortController(); let stopped = false, release!: () => void;
    const gate = new Promise<void>(resolve => { release = resolve; });
    const review = new CandidateReview(canonicalFixture().capture, parent.signal, () => true, async signal => {
      signal.addEventListener('abort', () => { stopped = true; });
      await gate; return outcome();
    });
    const pending = review.run(); await Promise.resolve();
    let finished = false;
    const closing = action === 'finish' ? review.finish().then(() => { finished = true; }) : undefined;
    if (action === 'cancel') review.cancel();
    if (action === 'parent') parent.abort();
    assert.equal(stopped, true); assert.equal(finished, false);
    release(); assert.equal((await pending).runtime, 'cancelled');
    await closing; await review.finish();
  });

  test('arbitrary worker errors do not enter the result document', async () => {
    const review = new CandidateReview(canonicalFixture().capture, new AbortController().signal, () => true,
      async () => { throw new Error('private model source and credential'); });
    assert.equal((await review.run()).runtime, 'refused');
    assert.doesNotMatch(candidateReviewText(review.snapshot()), /private|credential/); await review.finish();
  });

  test('cancellation cannot hide a failed cleanup', async () => {
    let release!: (result: CandidateRunResult) => void;
    const pending = new Promise<CandidateRunResult>(resolve => { release = resolve; });
    const review = new CandidateReview(canonicalFixture().capture, new AbortController().signal, () => true, () => pending);
    const running = review.run(); review.cancel();
    release({ outcome: { ok: false, reason: 'cleanup', detail: '' }, cleanupFailed: true });
    assert.equal((await running).cleanup, 'failed');
    assert.equal(review.snapshot().runtime, 'refused');
    assert.match(candidateReviewText(review.snapshot()), /Private temporary files may remain/); await review.finish();
  });

  test('a later run cannot erase failed cleanup and make the review applyable', async () => {
    let calls = 0;
    const review = new CandidateReview(canonicalFixture().capture, new AbortController().signal, () => true,
      async () => ++calls === 1 ? { outcome: { ok: false, reason: 'cleanup', detail: '' }, cleanupFailed: true } : outcome());
    await review.run();
    assert.equal(review.available, false);
    assert.equal((await review.run()).cleanup, 'failed');
    assert.equal(calls, 1);
    await review.finish();
    assert.equal(review.snapshot().cleanup, 'failed');
  });
});
