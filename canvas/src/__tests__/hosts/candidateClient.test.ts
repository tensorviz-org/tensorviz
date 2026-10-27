import { createLocalBrowserCandidateClient } from '../../ide/localBrowserCandidateClient';
import { createLocalBrowserSourceEditClient } from '../../ide/localBrowserSourceEditClient';
import type { LocalBrowserViewTransport } from '../../ide/localBrowserViewTransport';
import type { CandidateRecord, CandidateReviewSnapshot, AppliedSourceObservation } from '../../ide/protocol.generated';
import corpus from '../../../../tests/fixtures/contracts/local-browser-protocol.json';
const applied = (): AppliedSourceObservation => JSON.parse(JSON.stringify(corpus)).valid
  .find((item: { messageId: string }) => item.messageId === 'applied-source-checked').result.applied;

const target = { workspaceFolder: '/work', file: 'model.py', cls: 'Model', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
const summary: CandidateReviewSnapshot = { runId: 'run1', staticCheck: 'passed', runtime: 'succeeded', freshness: 'current',
  capture: 'partial', inputs: 'same', state: 'different', cleanup: 'complete', semanticPostconditions: 'not_checked', diagnostics: [] };
const record: CandidateRecord = { runId: 'run1', startedAt: 1, finishedAt: 2, review: 'open', summary,
  executionId: 'exec1', capture: { captureId: 'capture1', sourceId: 'source1' }, unavailable: null };
function harness() {
  const messages = new Set<(value: any) => void>(), statuses = new Set<(value: any) => void>();
  const send = jest.fn(value => value);
  const transport = { status: { phase: 'connected', hostLink: 'connected' }, session: { viewId: '1' }, send,
    onMessage(fn: any) { messages.add(fn); return () => messages.delete(fn); },
    onStatusChange(fn: any) { statuses.add(fn); return () => statuses.delete(fn); },
  } as unknown as LocalBrowserViewTransport;
  const wire = createLocalBrowserCandidateClient(transport);
  const message = (value: any) => messages.forEach(fn => fn(value));
  const respond = (overrides = {}) => message({ ...send.mock.calls[send.mock.calls.length - 1][0],
    result: { candidates: [record], page: null }, detail: 'Candidate evidence.', ...overrides, type: 'candidateResult' });
  return { ...wire, transport, send, message, respond, status: (value: any) => statuses.forEach(fn => fn(value)) };
}
afterEach(() => jest.useRealTimers());

test('source preparation grants no execution and adds the broker connection identity', async () => {
  const h = harness(), source = createLocalBrowserSourceEditClient(h.transport, () => target);
  const proposal = source.propose({ kind: 'remove_layer', nodeId: 'fc' });
  expect(h.send.mock.calls[0][0]).toMatchObject({ type: 'sourceEditRequest', viewId: '1' });
  h.message({ type: 'sourceEditReview', requestId: proposal.requestId, target, candidate: { ...summary, runtime: 'not_run' } });
  expect(source.candidates!.snapshot().reviews[0].prepared).toBe(true);
  expect(h.send).toHaveBeenCalledTimes(1);
  h.message({ type: 'sourceEditResult', requestId: proposal.requestId, target, status: 'applied', detail: 'Applied.' });
  expect((await proposal.promise).status).toBe('applied');
  expect(source.candidates!.snapshot().reviews[0].open).toBe(false);
  source.dispose(); h.client.dispose();
});

test('lost run receipts reuse the same execution key across closure; confirmed new runs use a new key', async () => {
  jest.useFakeTimers(); const h = harness(); h.begin('p', target);
  const first = h.client.run('p'), initial = h.send.mock.calls[0][0];
  jest.advanceTimersByTime(10_000); expect((await first).candidates).toBeNull();
  expect(h.client.snapshot().reviews[0].retryPending).toBe(true);
  h.end('p'); const retry = h.client.run('p'), retried = h.send.mock.calls[1][0];
  expect(retried.action.retryId).toBe(initial.action.retryId); expect(retried.requestId).not.toBe(initial.requestId);
  h.respond(); await retry;
  expect(h.client.snapshot().reviews[0].retryPending).toBeUndefined();
  const next = h.client.run('p'); expect(h.send.mock.calls[2][0].action.retryId).not.toBe(initial.action.retryId);
  h.respond({ result: null, detail: 'Review closed.' }); await next; h.client.dispose();
});

test('foreign request/view/target/proposal replies cannot settle or attach candidate evidence', async () => {
  const h = harness(); h.begin('p', target); const pending = h.client.run('p'); let settled = false;
  void pending.then(() => { settled = true; });
  for (const wrong of [{ viewId: '2' }, { requestId: 'unknown' }, { proposalId: 'other' }, { target: { ...target, documentVersion: 2 } }]) h.respond(wrong);
  await Promise.resolve(); expect(settled).toBe(false); expect(h.client.snapshot().reviews[0].records).toEqual([]);
  h.respond(); await pending; expect(h.client.snapshot().reviews[0].records[0].runId).toBe('run1'); h.client.dispose();
});

test('historical reads keep the original target and check capture, entity and path identity', async () => {
  const h = harness(); h.begin('p', target); h.end('p');
  h.message({ type: 'activeTarget', target: { ...target, cls: 'Other' } });
  const query = { ...record.capture!, path: ['operations'], offset: 0, limit: 1 };
  const pending = h.client.request('p', { kind: 'inspect', runId: 'run1', query });
  expect(h.send.mock.calls[0][0].target).toEqual(target);
  const page = { ...record.capture!, path: ['operations'], node: { key: null, kind: 'array', count: 0 }, items: [], total: 0, nextOffset: null };
  h.respond({ result: { candidates: [record], page: { ...page, sourceId: 'other' } } });
  expect(h.client.snapshot().busy).toBe(true);
  h.respond({ result: { candidates: [record], page } }); expect((await pending).page).toEqual(page); h.client.dispose();
});

test('disconnect clears connection-owned evidence and ignores late execution replies', async () => {
  const h = harness(); h.begin('p', target); const pending = h.client.run('p');
  h.status({ phase: 'reconnecting', hostLink: 'disconnected' });
  expect((await pending).candidates).toBeNull(); h.respond();
  expect(h.client.snapshot().reviews).toEqual([]); h.client.dispose();
});

test('closing a capture read cannot cancel a running execution request', async () => {
  const h = harness(); h.begin('p', target); const run = h.client.run('p');
  h.client.cancelRead(); expect(h.client.snapshot().busy).toBe(true);
  h.respond(); await run;
  const read = h.client.request('p', { kind: 'list' }); h.client.cancelRead();
  expect((await read).candidates).toBeNull(); expect(h.send).toHaveBeenCalledTimes(2);
  expect(h.send.mock.calls.every(([message]) => message.type === 'candidateRequest')).toBe(true); h.client.dispose();
});

test('summaries are copied and local proposal history is bounded', () => {
  const h = harness(); for (let i = 0; i < 40; i++) { h.begin(`p${i}`, target); h.end(`p${i}`); }
  const state = h.client.snapshot(); expect(state.reviews.length).toBe(32); expect(state.reviews[0].id).toBe('p8');
  state.reviews[0].target.file = 'changed'; expect(h.client.snapshot().reviews[0].target.file).toBe('model.py'); h.client.dispose();
});

test('applied-source reads have their own result and never rewrite candidate history', async () => {
  const h = harness(); h.begin('p', target, true);
  const run = h.client.run('p'); h.respond(); await run;
  h.message({ type: 'sourceEditResult', requestId: 'p', target, status: 'applied', detail: 'Applied.' });
  const before = h.client.snapshot().reviews[0].records;
  const reading = h.client.request('p', { kind: 'reconcile' });
  h.respond({ result: { candidates: [], page: null, applied: applied() } });
  expect((await reading).applied).toEqual(applied());
  const item = h.client.snapshot().reviews[0];
  expect(item.outcome).toBe('applied'); expect(item.captured).toBe(true); expect(item.appliedStale).toBe(false);
  expect(item.records).toEqual(before); expect(item.candidate).toEqual(summary);
  expect(h.send.mock.calls[1][0].action).toEqual({ kind: 'reconcile' }); h.client.dispose();
});

test('source or validation messages mark applied checks historical, including changes while a reply is in flight', async () => {
  const h = harness(); h.begin('p', target, true); h.end('p');
  const first = h.client.request('p', { kind: 'reconcile' });
  h.message({ type: 'syncState', target, state: 'stale' });
  h.respond({ result: { candidates: [], page: null, applied: applied() } }); await first;
  expect(h.client.snapshot().reviews[0].appliedStale).toBe(true);
  const next = h.client.request('p', { kind: 'reconcile' }); h.respond({ result: { candidates: [], page: null, applied: applied() } }); await next;
  expect(h.client.snapshot().reviews[0].appliedStale).toBe(false);
  h.message({ type: 'validationState', target, status: 'running' });
  expect(h.client.snapshot().reviews[0].appliedStale).toBe(true);
  expect(h.client.snapshot().reviews[0].applied).toEqual(applied()); h.client.dispose();
});

test('applied-source evidence cannot arrive on an unrelated candidate action or with candidate records', async () => {
  const h = harness(); h.begin('p', target);
  const listing = h.client.request('p', { kind: 'list' });
  h.respond({ result: { candidates: [], page: null, applied: applied() } });
  expect(h.client.snapshot().busy).toBe(true); expect(h.client.snapshot().reviews[0].applied).toBeUndefined();
  h.respond(); await listing;
  const reading = h.client.request('p', { kind: 'reconcile' });
  h.respond({ result: { candidates: [record], page: null, applied: applied() } });
  expect(h.client.snapshot().busy).toBe(true);
  h.respond({ result: { candidates: [], page: null, applied: applied() } }); await reading; h.client.dispose();
});

test('applied-source reads have a bounded longer deadline and cancellation never executes or applies', async () => {
  jest.useFakeTimers(); const h = harness(); h.begin('p', target);
  const reading = h.client.request('p', { kind: 'reconcile' });
  jest.advanceTimersByTime(10_000); expect(h.client.snapshot().busy).toBe(true);
  jest.advanceTimersByTime(85_000); expect((await reading).applied).toBeUndefined();
  const next = h.client.request('p', { kind: 'reconcile' }); h.client.cancelRead();
  expect((await next).applied).toBeUndefined();
  h.respond({ result: { candidates: [], page: null, applied: applied() } });
  expect(h.client.snapshot().reviews[0].applied).toBeUndefined();
  expect(h.send.mock.calls.every(([value]) => value.action.kind === 'reconcile')).toBe(true); h.client.dispose();
});
