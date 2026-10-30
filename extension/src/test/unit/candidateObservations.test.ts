import * as assert from 'node:assert/strict';
import { CandidateHistory, CANDIDATE_HISTORY_LIMITS } from '../../candidateHistory';
import { CandidateReview } from '../../candidateReview';
import { candidateObservationFixture } from '../candidateObservationFixture';
import { canonicalFixture } from '../../../../tests/fixtures/semantic.cjs';

const initial = () => new CandidateReview(canonicalFixture().capture, new AbortController().signal,
  () => true, async () => { throw new Error('Inspection cannot execute'); }).snapshot();
const passed = () => ({ ...initial(), runtime: 'succeeded' as const });
suite('bounded exact-candidate observations', () => {
  test('immutable output metadata pages survive closure while raw values stay private', () => {
    const history = new CandidateHistory(), value = candidateObservationFixture(), id = history.begin('owner', initial());
    assert.throws(() => history.inspectObservations('owner', id), /finish/);
    history.complete('owner', id, passed(), value); history.close('owner');
    (value.observations!.execution.outputs as any).leaves[0].path = 'poison';
    const first: any = history.inspectObservations('owner', id, { section: 'outputs', limit: 1 });
    assert.equal(first.evidence.leaves.items[0].path, '$[0]');
    assert.equal(first.evidence.leaves.nextOffset, 1); assert.equal(first.comparison, null);
    assert.equal(first.baselineUnavailable, 'not_retained'); assert.doesNotMatch(JSON.stringify(first), /AAAAQA|base64/);
    first.evidence.leaves.items[0].path = 'mutated';
    assert.equal((history.inspectObservations('owner', id, { section: 'outputs' }).evidence.leaves as any).items[0].path, '$[0]');
    const next: any = history.inspectObservations('owner', id, { section: 'outputs', offset: 1, limit: 1 });
    assert.equal(next.evidence.leaves.items[0].path, '$[1]');
    assert.throws(() => history.inspectObservations('other', id), /no longer retained/);
    assert.throws(() => history.inspectObservations('owner', id, { limit: 101 }), /bounded/);
    assert.throws(() => history.inspectObservations('owner', id, { atol: -1 }), /bounded/);
    history.forget('owner'); history.complete('owner', id, passed(), value);
    assert.throws(() => history.inspectObservations('owner', id), /no longer retained/);
  });
  for (const field of ['sourceDigest', 'planDigest', 'resolvedPlan'] as const) test(`refuses foreign ${field} while keeping the runtime summary`, () => {
    const history = new CandidateHistory(), value = candidateObservationFixture(), id = history.begin('owner', initial());
    value.observations!.setup[field] = field === 'resolvedPlan' ? {} : 'b'.repeat(64);
    history.complete('owner', id, passed(), value);
    assert.equal(history.get('owner', id).summary.runtime, 'succeeded');
    assert.throws(() => history.inspectObservations('owner', id), /invalid_observations/);
  });
  test('a baseline needs the original attempt, source, workspace and plan identity', () => {
    const value = candidateObservationFixture(), prior = candidateObservationFixture('baseline').observations!;
    Object.assign(value.execution, { baseAttemptId: prior.plan.id, baseSourceId: prior.plan.source.id,
      baseWorkspaceId: prior.plan.source.workspaceId, basePlanDigest: prior.planDigest });
    const history = new CandidateHistory();
    for (const field of [null, 'baseAttemptId', 'baseSourceId', 'baseWorkspaceId', 'basePlanDigest'] as const) {
      const candidate = structuredClone(value); if (field) (candidate.execution as any)[field] = 'foreign';
      const id = history.begin('owner', initial()); history.complete('owner', id, passed(), candidate, prior);
      const output: any = history.inspectObservations('owner', id, { section: 'outputs' });
      assert.equal(output.baselineUnavailable, field ? 'identity_mismatch' : null);
      assert.equal(output.comparison?.allClose ?? null, field ? null : true);
    }
  });
  test('training and optimizer diagnostics cannot be requested from a candidate', () => {
    const history = new CandidateHistory(), id = history.begin('owner', initial());
    history.complete('owner', id, passed(), candidateObservationFixture());
    for (const section of ['training', 'gradients', 'optimizer', 'experiment']) {
      assert.throws(() => history.inspectObservations('owner', id, { section } as never), /supported observation section/);
    }
  });

  test('observation payloads obey per-payload/total limits, expiry and permanent eviction', () => {
    const value = candidateObservationFixture(); delete value.capture;
    const small = new CandidateHistory({ ...CANDIDATE_HISTORY_LIMITS, captureBytes: 1 });
    const rejected = small.begin('owner', initial()); small.complete('owner', rejected, passed(), value);
    assert.throws(() => small.inspectObservations('owner', rejected), /too_large/);
    let now = 0;
    const size = Buffer.byteLength(JSON.stringify({ candidate: value.observations, baselineUnavailable: 'not_retained' }));
    const history = new CandidateHistory({ ...CANDIDATE_HISTORY_LIMITS, totalBytes: size, ttlMs: 100 }, () => now);
    const first = history.begin('owner', initial()); history.complete('owner', first, passed(), value);
    const second = history.begin('owner', initial()); history.complete('owner', second, passed(), value);
    assert.throws(() => history.inspectObservations('owner', first), /evicted/);
    history.complete('owner', first, passed(), value);
    assert.throws(() => history.inspectObservations('owner', first), /evicted/);
    assert.ok(history.inspectObservations('owner', second));
    now = 100; assert.throws(() => history.inspectObservations('owner', second), /no longer retained/);
  });
});
