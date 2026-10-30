import * as assert from 'node:assert/strict';
import { CandidateHistory, CANDIDATE_HISTORY_LIMITS } from '../../candidateHistory';
import { CandidateReview } from '../../candidateReview';
import type { CandidateEvidence, CandidateRunResult } from '@tensorviz/graph-host/candidateExecution';
import type { CandidateExecution } from '../../executionPlan';
import { canonicalFixture } from '../../../../tests/fixtures/semantic.cjs';
import { candidateCheckpointFixture } from '../candidateCheckpointFixture';

const initial = () => new CandidateReview(canonicalFixture().capture, new AbortController().signal,
  () => true, async () => { throw new Error('never execute'); }).snapshot();
function evidence(): CandidateEvidence {
  const { capture, plan } = canonicalFixture('candidate:retained');
  return { execution: { id: plan.id, plan } as CandidateExecution, capture, baseCurrent: true, candidateCurrent: true };
}
function result(): CandidateRunResult {
  return { evidence: evidence(), outcome: { ok: true, exitCode: 0, executionCurrent: true,
    recheckExecution: async () => true, envelope: { protocol_version: 1, result: { private: 'never retain' }, error: null, spans: {}, worker: {} } } };
}
const query = (value: CandidateEvidence) => ({ captureId: value.capture!.id, sourceId: value.capture!.source.id, path: [], offset: 0, limit: 50 });

suite('bounded private candidate history', () => {
  test('load failures retain immutable checkpoint pages without a forward capture', () => {
    const history = new CandidateHistory(), value = candidateCheckpointFixture(), id = history.begin('owner', initial());
    history.complete('owner', id, { ...initial(), runtime: 'failed' }, value);
    value.checkpoint.entries[0].name = 'mutated';
    history.close('owner');
    const first = history.inspectCheckpoint('owner', id, { limit: 1 });
    assert.equal(first.checkpoint!.loadStatus, 'failed'); assert.equal(first.items![0].name, 'left.weight');
    assert.equal(first.checkpoint!.counts.shapeDifferences, 2); assert.equal(first.nextOffset, 1);
    const second = history.inspectCheckpoint('owner', id, { offset: first.nextOffset!, limit: 1 });
    assert.equal(second.items![0].name, 'right.weight'); assert.equal(second.nextOffset, null);
    first.items![0].name = 'reader_mutation';
    assert.equal(history.inspectCheckpoint('owner', id).items![0].name, 'left.weight');
    assert.equal(history.get('owner', id).capture, null); assert.equal(history.get('owner', id).review, 'historical');
    assert.throws(() => history.inspectCheckpoint('other', id), /no longer retained/);
    assert.throws(() => history.inspectCheckpoint('owner', id, { offset: -1 }), /bounded checkpoint/);
    assert.throws(() => history.inspectCheckpoint('owner', id, { limit: 201 }), /bounded checkpoint/);
    history.forget('owner'); assert.throws(() => history.inspectCheckpoint('owner', id), /no longer retained/);
    history.complete('owner', id, initial(), value); assert.deepEqual(history.list('owner'), []);
  });

  test('checkpoint-only reports share the byte limit and cannot be revived after eviction or expiry', () => {
    let now = 0;
    const value = candidateCheckpointFixture(), size = Buffer.byteLength(JSON.stringify(value.checkpoint));
    const history = new CandidateHistory({ ...CANDIDATE_HISTORY_LIMITS, totalBytes: size, ttlMs: 100 }, () => now);
    const first = history.begin('one', initial()); history.complete('one', first, initial(), value);
    const second = history.begin('two', initial()); history.complete('two', second, initial(), value);
    assert.throws(() => history.inspectCheckpoint('one', first), /evicted/);
    assert.equal(history.inspectCheckpoint('two', second).checkpoint!.loadStatus, 'failed');
    assert.equal(history.get('one', first).unavailable, 'not_captured', 'checkpoint eviction must not invent a capture');
    now = 100; assert.throws(() => history.inspectCheckpoint('two', second), /no longer retained/);
  });

  for (const field of ['source', 'resource', 'comparison']) test(`retention refuses forged checkpoint ${field}`, () => {
    const history = new CandidateHistory(), value = candidateCheckpointFixture(), id = history.begin('one', initial());
    if (field === 'source') value.checkpoint.sourceDigest = '0'.repeat(64);
    if (field === 'resource') value.checkpoint.checkpoint.digest = '0'.repeat(64);
    if (field === 'comparison') value.checkpoint.entries[0].shape = 'same';
    history.complete('one', id, { ...initial(), runtime: 'failed' }, value);
    assert.equal(history.get('one', id).summary.runtime, 'failed');
    assert.throws(() => history.inspectCheckpoint('one', id), /invalid_checkpoint/);
  });

  test('retains immutable canonical pages separately from worker envelopes and plans', () => {
    const history = new CandidateHistory(), value = evidence(), original = structuredClone(value);
    const id = history.begin('one', initial()); history.complete('one', id, initial(), value);
    value.capture!.operations.length = 0;
    const record = history.get('one', id); record.summary.diagnostics.push('mutated');
    const page = history.inspect('one', id, { ...query(original), path: ['operations'] });
    assert.equal(page.total, original.capture!.operations.length);
    assert.deepEqual(history.get('one', id).summary.diagnostics, []);
    assert.equal(history.get('one', id).executionId, original.execution.id);
    assert.doesNotMatch(JSON.stringify(history.list('one')), /recheckExecution|private|"plan"/);
  });
  test('wrong proposal and mismatched capture identities cannot read a retained record', () => {
    const history = new CandidateHistory(), value = evidence(), id = history.begin('owner', initial());
    history.complete('owner', id, initial(), value);
    assert.deepEqual(history.list('other'), []);
    assert.throws(() => history.get('other', id), /no longer retained/);
    assert.throws(() => history.inspect('other', id, query(value)), /no longer retained/);
    assert.throws(() => history.inspect('owner', id, { ...query(value), sourceId: 'another-source' }), /exact capture/);
    history.forget('other'); assert.equal(history.list('owner').length, 1);
  });
  for (const bad of ['structure', 'identity', 'references']) test(`refuses ${bad} corruption without hiding the runtime summary`, () => {
    const history = new CandidateHistory(), value = evidence(), id = history.begin('one', initial());
    if (bad === 'structure') (value.capture as any).status = 'made_up';
    if (bad === 'identity') value.execution.id = 'different';
    if (bad === 'references') value.capture!.operations[0].arguments.args = [{ kind: 'value', id: 'absent' }];
    history.complete('one', id, { ...initial(), runtime: 'failed' }, value);
    assert.equal(history.get('one', id).unavailable, 'invalid_capture');
    assert.equal(history.get('one', id).summary.runtime, 'failed');
    assert.throws(() => history.inspect('one', id, query(value)), /unavailable/);
  });
  test('per-capture and total byte limits preserve honest summaries after payload eviction', () => {
    const value = evidence(), size = Buffer.byteLength(JSON.stringify(value.capture));
    const small = new CandidateHistory({ ...CANDIDATE_HISTORY_LIMITS, captureBytes: size - 1 });
    const tooBig = small.begin('one', initial()); small.complete('one', tooBig, initial(), value);
    assert.equal(small.get('one', tooBig).unavailable, 'too_large');
    const history = new CandidateHistory({ ...CANDIDATE_HISTORY_LIMITS, totalBytes: size });
    const first = history.begin('one', initial()); history.complete('one', first, initial(), value);
    const second = history.begin('two', initial()); history.complete('two', second, initial(), value);
    assert.equal(history.get('one', first).unavailable, 'evicted');
    assert.ok(history.get('one', first).capture);
    assert.equal(history.get('two', second).unavailable, null);
    assert.throws(() => history.inspect('one', first, query(value)), /unavailable/);
  });
  test('record count and fixed TTL remain bounded on reads and late completion', () => {
    let now = 0;
    const history = new CandidateHistory({ ...CANDIDATE_HISTORY_LIMITS, records: 2, ttlMs: 100 }, () => now);
    const first = history.begin('one', initial()); history.complete('one', first, initial(), evidence());
    history.begin('one', initial()); const last = history.begin('one', initial());
    assert.throws(() => history.get('one', first), /no longer retained/);
    assert.equal(history.list('one').length, 2);
    now = 100; assert.deepEqual(history.list('one'), []);
    history.complete('one', last, initial(), evidence()); assert.deepEqual(history.list('one'), []);
  });
  for (const action of ['forget', 'clear'] as const) test(`${action} prevents resurrection by late completion, refresh or close`, () => {
    const history = new CandidateHistory(), id = history.begin('one', initial());
    if (action === 'forget') history.forget('one'); else history.clear();
    history.complete('one', id, initial(), evidence()); history.update('one', id, initial()); history.close('one');
    assert.deepEqual(history.list('one'), []);
  });
  test('native reruns get distinct records, while duplicate clicks share the current run', async () => {
    const history = new CandidateHistory(); let runs = 0, current = true;
    const review = new CandidateReview(canonicalFixture().capture, new AbortController().signal, () => current,
      async () => { runs++; return result(); }, { history, requestId: 'one' });
    const running = review.run(), first = review.snapshot().runId!;
    assert.equal(review.run(), running); assert.equal(history.get('one', first).summary.runtime, 'running');
    await running; await review.run(); const second = review.snapshot().runId!;
    assert.notEqual(first, second); assert.equal(runs, 2);
    await review.finish(); current = false; review.snapshot();
    assert.equal(review.available, false); await review.run(); assert.equal(runs, 2);
    assert.deepEqual(history.list('one').map(item => item.review), ['historical', 'historical']);
    assert.equal(history.get('one', second).summary.freshness, 'stale');
    assert.ok(history.inspect('one', second, query(evidence())).items.length);
  });
  test('cancellation drops late evidence; cleanup failure remains visible after close', async () => {
    for (const cleanupFailed of [false, true]) {
      const history = new CandidateHistory(); let release!: (value: CandidateRunResult) => void;
      const review = new CandidateReview(canonicalFixture().capture, new AbortController().signal, () => true,
        () => new Promise(resolve => { release = resolve; }), { history, requestId: 'one' });
      const pending = review.run(); await Promise.resolve(); const closing = review.finish();
      release({ ...result(), ...(cleanupFailed ? { cleanupFailed: true as const } : {}) }); await pending; await closing;
      const record = history.list('one')[0];
      assert.equal(record.summary.runtime, cleanupFailed ? 'refused' : 'cancelled');
      assert.equal(record.summary.cleanup, cleanupFailed ? 'failed' : 'complete');
      assert.equal(record.capture, null); assert.equal(record.review, 'historical');
    }
  });

  test('a terminal listener starting a rerun cannot relabel the preceding completion', async () => {
    const history = new CandidateHistory(); let next: Promise<unknown> | undefined;
    const review = new CandidateReview(canonicalFixture().capture, new AbortController().signal, () => true,
      async () => result(), { history, requestId: 'one' });
    let rerun = false;
    review.onDidChange(value => {
      if (value.runtime === 'succeeded' && !rerun) { rerun = true; next = review.run(); }
    });
    const pending = review.run(), first = review.snapshot().runId;
    const completed = await pending; await next;
    assert.equal(completed.runId, first); assert.equal(completed.runtime, 'succeeded');
    assert.notEqual(review.snapshot().runId, first);
    await review.finish();
  });
});
