import type { CandidateAction, CandidateRecord, CandidateReviewSnapshot, CaptureInspectionPage, CandidateCheckpointPage, LocalBrowserTargetIdentity, AppliedSourceObservation } from './protocol.generated';
import { newLocalBrowserMessageId } from './protocol.generated';
import type { LocalBrowserViewTransport } from './localBrowserViewTransport';
import { sameValidationTarget } from './localBrowserValidationClient';

export interface BrowserCandidateReview {
  id: string; target: LocalBrowserTargetIdentity; viewId: string;
  open: boolean; prepared: boolean; candidate: CandidateReviewSnapshot | null;
  records: CandidateRecord[]; detail: string;
  retryPending?: true;
  outcome?: 'applied' | 'cancelled' | 'refused';
  applied?: AppliedSourceObservation;
  appliedStale?: boolean;
  captured?: boolean;
}
export interface BrowserCandidateState { reviews: BrowserCandidateReview[]; selectedId: string | null; busy: boolean }
export interface CandidateReply { candidates: CandidateRecord[] | null; page: CaptureInspectionPage | null; detail: string; applied?: AppliedSourceObservation; checkpoint?: CandidateCheckpointPage }
export interface LocalBrowserCandidateClient {
  snapshot(): BrowserCandidateState;
  onChange(listener: () => void): () => void;
  select(id: string): void;
  run(id: string): Promise<CandidateReply>;
  request(id: string, action: CandidateAction): Promise<CandidateReply>;
  cancelRead(): void;
  dispose(): void;
}
const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value));

/** Owns correlation and safe retry identity; all execution authority stays on the host. */
export function createLocalBrowserCandidateClient(transport: LocalBrowserViewTransport) {
  const reviews = new Map<string, BrowserCandidateReview>(), retries = new Map<string, string>();
  const listeners = new Set<() => void>();
  let selectedId: string | null = null, disposed = false, sourceRevision = 0;
  let pending: { id: string; review: BrowserCandidateReview; action: CandidateAction;
    sourceRevision: number;
    timer: ReturnType<typeof setTimeout>; resolve(reply: CandidateReply): void } | null = null;
  const emit = () => listeners.forEach(listener => listener());
  const settle = (reply: CandidateReply) => {
    const item = pending; pending = null;
    if (item) { clearTimeout(item.timer); item.resolve(reply); emit(); }
  };
  const cancelRead = () => {
    if (pending?.action.kind === 'inspect' || pending?.action.kind === 'checkpoint' || pending?.action.kind === 'list' || pending?.action.kind === 'reconcile') {
      settle({ candidates: null, page: null, detail: 'Candidate inspection closed.' });
    }
  };
  const offMessage = transport.onMessage(message => {
    if (['activeTarget', 'projection', 'syncState', 'staleState', 'validationState', 'snapshot'].includes(message.type)) {
      sourceRevision++;
      reviews.forEach(item => { if (item.applied) item.appliedStale = true; });
      emit(); return;
    }
    if (message.type === 'sourceEditReview' || message.type === 'sourceEditResult') {
      const item = reviews.get(message.requestId);
      if (!item || !sameValidationTarget(item.target, message.target)) return;
      if (message.type === 'sourceEditReview' && item.open) {
        item.prepared = true; item.candidate = clone(message.candidate);
      } else if (message.type === 'sourceEditResult') { item.open = false; item.outcome = message.status; item.detail = message.detail; }
      emit(); return;
    }
    if (message.type !== 'candidateResult' || !pending || message.requestId !== pending.id
      || message.proposalId !== pending.review.id || message.viewId !== pending.review.viewId
      || !sameValidationTarget(message.target, pending.review.target)) return;
    const query = pending.action.kind === 'inspect' ? pending.action.query : undefined, page = message.result?.page;
    if (page && (!query || page.captureId !== query.captureId || page.sourceId !== query.sourceId
      || JSON.stringify(page.path) !== JSON.stringify(query.path) || JSON.stringify(page.entity ?? null) !== JSON.stringify(query.entity ?? null))) return;
    const action = pending.action, item = pending.review, records = message.result?.candidates;
    const checkpoint = message.result?.checkpoint;
    if (checkpoint && (action.kind !== 'checkpoint' || checkpoint.runId !== action.runId || checkpoint.offset !== action.offset
      || (checkpoint.nextOffset !== null && checkpoint.nextOffset > action.offset + action.limit))) return;
    if (action.kind === 'checkpoint' && message.result && (!checkpoint || page !== null)) return;
    const applied = message.result?.applied;
    if (applied && action.kind !== 'reconcile') return;
    if (action.kind === 'reconcile' && message.result && (!applied || records?.length !== 0 || page !== null)) return;
    if (applied) { item.applied = clone(applied); item.appliedStale = pending.sourceRevision !== sourceRevision; }
    if (records) {
      if ('runId' in action && (records.length !== 1 || records[0].runId !== action.runId)) return;
      if (action.kind === 'run' && records.length !== 1) return;
      if (action.kind === 'list') item.records = clone(records);
      else for (const record of records) item.records = [...item.records.filter(old => old.runId !== record.runId), clone(record)].slice(-32);
      const latest = records.find(record => record.runId === item.candidate?.runId);
      if (latest) item.candidate = clone(latest.summary);
      if (action.kind === 'run') { retries.delete(item.id); item.candidate = clone(records[0].summary); }
    }
    item.detail = message.detail;
    settle({ candidates: records ?? null, page: page ?? null, detail: message.detail, ...(applied ? { applied: clone(applied) } : {}),
      ...(checkpoint ? { checkpoint: clone(checkpoint) } : {}) });
  });
  const offStatus = transport.onStatusChange(status => {
    if (status.phase === 'connected' && status.hostLink === 'connected') return;
    reviews.clear(); retries.clear(); selectedId = null;
    settle({ candidates: null, page: null, detail: 'The connection ended. Candidate history belonged to that connection.' }); emit();
  });
  const request = (id: string, action: CandidateAction): Promise<CandidateReply> => {
    const review = reviews.get(id);
    if (disposed || pending || !review || transport.session?.viewId !== review.viewId
      || transport.status.phase !== 'connected' || transport.status.hostLink !== 'connected') {
      return Promise.resolve({ candidates: null, page: null, detail: 'Connect to the original review and wait for the current candidate request.' });
    }
    const requestId = newLocalBrowserMessageId();
    const promise = new Promise<CandidateReply>(resolve => {
      pending = { id: requestId, review, action: clone(action), sourceRevision, resolve, timer: setTimeout(() => {
        review.detail = action.kind === 'run' ? 'No run receipt arrived. Retry run receipt to inspect the same execution.' : 'Candidate request timed out. Refresh its result.';
        settle({ candidates: null, page: null, detail: review.detail });
      }, action.kind === 'reconcile' ? 95_000 : 10_000) };
    });
    emit();
    try {
      if (!transport.send({ type: 'candidateRequest', viewId: review.viewId, requestId, proposalId: id, target: review.target, action })) {
        settle({ candidates: null, page: null, detail: 'Candidate request could not reach VS Code.' });
      }
    } catch { settle({ candidates: null, page: null, detail: 'Candidate request could not start. Retry the same run receipt if needed.' }); }
    return promise;
  };
  const client: LocalBrowserCandidateClient = {
    snapshot: () => clone({ reviews: Array.from(reviews.values()).map(item => ({ ...item, ...(retries.has(item.id) ? { retryPending: true as const } : {}) })), selectedId, busy: !!pending }),
    onChange(listener) { listeners.add(listener); return () => { listeners.delete(listener); }; },
    select(id) { if (reviews.has(id)) { selectedId = id; emit(); } },
    run(id) { const retryId = retries.get(id) ?? newLocalBrowserMessageId(); retries.set(id, retryId); return request(id, { kind: 'run', retryId }); },
    request, cancelRead,
    dispose() {
      disposed = true; reviews.clear(); retries.clear(); listeners.clear();
      settle({ candidates: null, page: null, detail: 'Candidate inspection closed.' }); offMessage(); offStatus();
    },
  };
  return { client,
    begin(id: string, target: LocalBrowserTargetIdentity, captured = false) {
      const viewId = transport.session?.viewId;
      if (!viewId) return;
      if (reviews.size >= 32) { const old = reviews.keys().next().value!; reviews.delete(old); retries.delete(old); }
      reviews.set(id, { id, target: clone(target), viewId, captured, open: true, prepared: false, candidate: null, records: [], detail: '' });
      selectedId = id; emit();
    },
    end(id: string) { const item = reviews.get(id); if (item) { item.open = false; emit(); } },
  };
}
