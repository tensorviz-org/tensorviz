/** Private browser proposal ownership. The broker verifies every supplied view ID. */
import { randomUUID } from 'node:crypto';
import type { LocalBrowserSourceEdits, SourceProposalOutcome, SourceProposalReview } from './localBrowserSourceEdits';
import type { LocalBrowserHostPublish } from '@tensorviz/graph-host/localBrowserHostTransport';
import type { LocalBrowserCandidateRequestMessage, LocalBrowserSourceEditRequestMessage } from '@tensorviz/contracts/protocol';
import { sameTargetIdentity } from './localBrowserSync';
import { GraphToolError } from '@tensorviz/graph-host/mcpGraph';

interface Review {
  request: LocalBrowserSourceEditRequestMessage;
  sourceId: string; controller: AbortController; finishedAt?: number;
  outcome?: SourceProposalOutcome; review?: SourceProposalReview;
  retries: Map<string, string>;
}
type Source = Pick<LocalBrowserSourceEdits, 'propose' | 'runCandidate' | 'cancelCandidate' | 'getCandidate'
  | 'listCandidates' | 'inspectCandidate' | 'inspectCandidateCheckpointText' | 'forgetCandidates' | 'inspectAppliedSource'>;

export class BrowserSourceReviews {
  private readonly reviews = new Map<string, Review>();
  constructor(private readonly source: Source, private readonly publish: (message: LocalBrowserHostPublish) => unknown) {}

  handle(request: LocalBrowserSourceEditRequestMessage): void {
    this.prune();
    const old = this.reviews.get(request.requestId);
    if (old) {
      if (old.request.viewId !== request.viewId || old.request.workspaceSessionId !== request.workspaceSessionId
        || !sameTargetIdentity(old.request.target, request.target)) {
        this.publish({ type: 'sourceEditResult', requestId: request.requestId, target: request.target,
          status: 'refused', detail: 'This request ID is unavailable. Prepare a new source proposal.' });
      } else if (old.outcome) this.publish({ type: 'sourceEditResult', requestId: request.requestId, target: request.target, ...old.outcome });
      else if (old.review) this.report(old, old.review);
      return;
    }
    if (this.reviews.size >= 32) {
      const completed = [...this.reviews.values()].find(item => item.finishedAt !== undefined);
      if (completed) this.remove(completed);
    }
    if (this.reviews.size >= 32) {
      this.publish({ type: 'sourceEditResult', requestId: request.requestId, target: request.target,
        status: 'refused', detail: 'Finish the current source reviews before preparing another proposal.' });
      return;
    }
    const item: Review = { request: structuredClone(request), sourceId: `browser:${randomUUID()}`,
      controller: new AbortController(), retries: new Map() };
    this.reviews.set(request.requestId, item);
    void this.execute(item);
  }

  cancel(requestId: string): void { this.reviews.get(requestId)?.controller.abort(); }
  ownsRequest(requestId: string | undefined): boolean {
    return requestId !== undefined && [...this.reviews.values()].some(item =>
      item.sourceId === requestId && item.finishedAt === undefined && !item.controller.signal.aborted);
  }
  cancelOwner(viewId: string): void {
    for (const item of this.reviews.values()) if (item.request.viewId === viewId) this.remove(item);
  }
  clear(): void { for (const item of this.reviews.values()) this.remove(item); }

  control(message: LocalBrowserCandidateRequestMessage): void {
    let result: Extract<LocalBrowserHostPublish, { type: 'candidateResult' }>['result'] = null;
    let detail = 'Candidate result. This describes the reviewed copy, not the live editor buffer.';
    try {
      this.prune();
      const item = this.reviews.get(message.proposalId);
      if (!item || item.request.viewId !== message.viewId || item.request.workspaceSessionId !== message.workspaceSessionId
        || !sameTargetIdentity(item.request.target, message.target)) {
        throw new GraphToolError('UNKNOWN_PROPOSAL', 'This source review is unavailable in this browser connection.');
      }
      const action = message.action;
      if (action.kind === 'reconcile') {
        if (item.outcome?.status !== 'applied') throw new GraphToolError('APPLIED_SOURCE_UNAVAILABLE', 'Apply the reviewed source in VS Code before checking it.');
        void this.reconcile(message, item); return;
      }
      if (action.kind === 'run') {
        const prior = item.retries.get(action.retryId);
        if (prior) result = { candidates: [this.source.getCandidate(item.sourceId, prior)], page: null };
        else {
          if (!item.review || item.finishedAt !== undefined || item.controller.signal.aborted) throw new GraphToolError('CANDIDATE_UNAVAILABLE', 'Run a candidate only while its native source review is open.');
          if (item.retries.size >= 128) throw new GraphToolError('REQUEST_LIMIT', 'This proposal has reached its candidate request limit.');
          const record = this.source.runCandidate(item.sourceId);
          item.retries.set(action.retryId, record.runId); result = { candidates: [record], page: null };
        }
      } else if (action.kind === 'cancel') result = { candidates: [this.source.cancelCandidate(item.sourceId, action.runId)], page: null };
      else if (action.kind === 'list') result = { candidates: this.source.listCandidates(item.sourceId), page: null };
      else if (action.kind === 'checkpoint') result = { candidates: [this.source.getCandidate(item.sourceId, action.runId)], page: null,
        checkpoint: this.source.inspectCandidateCheckpointText(item.sourceId, action.runId, { offset: action.offset, limit: action.limit }) };
      else result = { candidates: [this.source.getCandidate(item.sourceId, action.runId)],
        page: action.query ? this.source.inspectCandidate(item.sourceId, action.runId, action.query) : null };
    } catch (error) {
      detail = error instanceof GraphToolError ? error.message : 'The candidate request could not complete. Inspect its result before retrying.';
    }
    this.publish({ type: 'candidateResult', viewId: message.viewId, requestId: message.requestId,
      proposalId: message.proposalId, target: message.target, result, detail });
  }

  private async reconcile(message: LocalBrowserCandidateRequestMessage, item: Review): Promise<void> {
    let result: Extract<LocalBrowserHostPublish, { type: 'candidateResult' }>['result'] = null;
    let detail = 'Applied-source observation. Source state, current setup and recorded effects are separate checks.';
    try {
      const applied = await this.source.inspectAppliedSource(item.sourceId, item.controller.signal);
      result = { candidates: [], page: null, applied };
    } catch (error) {
      detail = error instanceof GraphToolError ? error.message : 'Applied-source inspection could not complete. Check the saved source and try again.';
    }
    this.prune();
    if (item.controller.signal.aborted || this.reviews.get(message.proposalId) !== item) return;
    this.publish({ type: 'candidateResult', viewId: message.viewId, requestId: message.requestId,
      proposalId: message.proposalId, target: message.target, result, detail });
  }

  private report(item: Review, review: SourceProposalReview): void {
    if (item.controller.signal.aborted || this.reviews.get(item.request.requestId) !== item) return;
    item.review = structuredClone(review);
    if (item.request.viewId) this.publish({ type: 'sourceEditReview', requestId: item.request.requestId,
      target: item.request.target, candidate: review.candidate ?? null });
  }
  private async execute(item: Review): Promise<void> {
    try {
      item.outcome = await this.source.propose({ requestId: item.sourceId, target: item.request.target, edit: item.request.edit },
        item.controller.signal, review => this.report(item, review));
    } catch { item.outcome = { status: 'refused', detail: 'The source proposal could not complete. Inspect the VS Code buffer before retrying.' }; }
    item.finishedAt = Date.now();
    if (this.reviews.get(item.request.requestId) === item) this.publish({ type: 'sourceEditResult',
      requestId: item.request.requestId, target: item.request.target, ...item.outcome });
  }
  private remove(item: Review): void {
    item.controller.abort(); this.source.forgetCandidates(item.sourceId);
    this.reviews.delete(item.request.requestId);
  }
  private prune(): void {
    for (const item of this.reviews.values()) if (item.finishedAt !== undefined && Date.now() - item.finishedAt >= 30 * 60_000) this.remove(item);
  }
}
