import { ModelGraph, SavedGraph } from '../types';
import { GraphDocumentV2, GraphLayout } from '../types/graphDocument';
import { modelGraphToDocumentV2 } from './graphDocumentAdapter';

export type GraphSaveState = 'idle' | 'saving' | 'saved' | 'offline' | 'conflict';

export interface GraphSaveRequest {
  graph: ModelGraph;
  name: string;
  description?: string;
  author: 'user' | 'agent';
  opSummary?: string;
  createKey?: string;
  operations: Array<Record<string, unknown>>;
  layout: GraphLayout;
}

export interface GraphSaveMetadata {
  expectedRevision?: number;
  author?: 'user' | 'agent';
  opSummary?: string;
  createKey?: string;
  operations?: Array<Record<string, unknown>>;
  layout?: GraphLayout;
}

export type PersistGraph = (graph: ModelGraph, name: string, description?: string,
  id?: string, meta?: GraphSaveMetadata) => Promise<SavedGraph>;
type Waiter = {
  resolve: (saved: SavedGraph | null) => void;
  reject: (error: unknown) => void;
};
type QueuedSave = {
  request: GraphSaveRequest;
  fingerprint: string;
  waiters: Waiter[];
};
type CommitBoundary = Pick<GraphSaveRequest, 'author' | 'opSummary'>;

export interface GraphSaveQueueCallbacks {
  onState?: (state: GraphSaveState) => void;
  onCreated?: (saved: SavedGraph) => void;
  onSaved?: (saved: SavedGraph, request: GraphSaveRequest) => void;
  onPendingChanged?: (state: {
    graphId?: string;
    revision: number;
    requests: GraphSaveRequest[];
  }) => void;
}

export interface GraphSaveEnqueueOptions {
  /**
   * Authority boundaries (such as starting an AI turn) cannot wait for a
   * human-triggered offline retry. Reject them immediately when recovery is
   * already required; background autosaves may keep journaling by default.
   */
  rejectIfBlocked?: boolean;
}

export function graphSaveDocument(
  request: GraphSaveRequest,
  graphId: string,
  revision: number
): GraphDocumentV2 {
  return modelGraphToDocumentV2(request.graph, {
    graphId,
    name: request.name,
    description: request.description,
    revision,
    operations: request.operations,
    layout: request.layout,
  });
}

export function graphSaveFingerprint(request: GraphSaveRequest): string {
  const document = graphSaveDocument(request, 'authority', 0);
  document.semantic.graph_id = '';
  document.view.graph_id = '';
  return JSON.stringify(document);
}

function sameCommitBoundary(
  request: CommitBoundary,
  boundary: CommitBoundary | null
): boolean {
  return boundary === null
    || (request.author === boundary.author && request.opSummary === boundary.opSummary);
}

function canCoalesceAutosaves(previous: GraphSaveRequest, next: GraphSaveRequest): boolean {
  return previous.author === 'user'
    && next.author === 'user'
    && previous.opSummary === undefined
    && next.opSummary === undefined;
}

function appendQueuedSave(queue: QueuedSave[], queued: QueuedSave): void {
  const tail = queue[queue.length - 1];
  if (tail && canCoalesceAutosaves(tail.request, queued.request)) {
    tail.request = queued.request;
    tail.fingerprint = queued.fingerprint;
    tail.waiters.push(...queued.waiters);
    return;
  }
  queue.push(queued);
}

/**
 * The first failed request may have reached the server even when its response
 * did not reach the browser. Never coalesce a later autosave into that
 * ambiguous head: recovery needs its exact snapshot to determine whether that
 * revision landed. Requests known not to have been sent may still coalesce at
 * the tail of the blocked backlog.
 */
function appendBehindPossiblySentHead(queue: QueuedSave[], queued: QueuedSave): void {
  const tail = queue[queue.length - 1];
  if (
    queue.length > 1
    && tail
    && canCoalesceAutosaves(tail.request, queued.request)
  ) {
    tail.request = queued.request;
    tail.fingerprint = queued.fingerprint;
    tail.waiters.push(...queued.waiters);
    return;
  }
  queue.push(queued);
}

export class RevisionedGraphSaveQueue {
  private graphId: string | undefined;
  private revision = 0;
  private lastFingerprint: string | null = null;
  private lastBoundary: CommitBoundary | null = null;
  private pending: QueuedSave[] = [];
  private failed: QueuedSave[] = [];
  private failedRetryable = false;
  private inFlight = false;
  private active: QueuedSave | null = null;
  private activeGeneration = -1;
  private generation = 0;
  private callbacksActive = true;

  constructor(
    private readonly persist: PersistGraph,
    private readonly callbacks: GraphSaveQueueCallbacks = {}
  ) {}

  private outstanding(): QueuedSave[] {
    const active = this.active && this.activeGeneration === this.generation
      ? [this.active]
      : [];
    return [...active, ...this.failed, ...this.pending];
  }

  private emitPending(): void {
    if (!this.callbacksActive) return;
    try {
      this.callbacks.onPendingChanged?.({
        graphId: this.graphId,
        revision: this.revision,
        requests: this.outstanding().map(({ request }) => request),
      });
    } catch (error) {
      console.warn('Failed to update graph crash journal:', error);
    }
  }

  private clearActive(queued: QueuedSave): void {
    if (this.active === queued) {
      this.active = null;
      this.activeGeneration = -1;
    }
  }

  private emitState(state: GraphSaveState): void {
    if (this.callbacksActive) this.callbacks.onState?.(state);
  }

  /** Resume lifecycle callbacks for this queue. Queues are active by default. */
  activateCallbacks(): void {
    this.callbacksActive = true;
  }

  /**
   * Retire callbacks without cancelling persistence or settlement. This keeps a
   * save already accepted by the server internally consistent while preventing
   * an unmounted editor from navigating or rewriting another session's journal.
   */
  retireCallbacks(): void {
    this.callbacksActive = false;
  }

  private matchesAuthoritativeSnapshot(
    request: GraphSaveRequest,
    fingerprint: string
  ): boolean {
    return fingerprint === this.lastFingerprint
      && sameCommitBoundary(request, this.lastBoundary)
      && !this.inFlight
      && this.pending.length === 0
      && this.failed.length === 0;
  }

  configure(graphId: string | undefined, revision: number, baselineFingerprint: string | null): void {
    this.generation += 1;
    [...this.pending, ...this.failed].forEach((queued) => {
      queued.waiters.forEach(({ resolve }) => resolve(null));
    });
    this.graphId = graphId;
    this.revision = revision;
    this.lastFingerprint = baselineFingerprint;
    // A hydrated baseline has no commit attribution. Treat it as compatible
    // with an identical first request; successful commits set a real boundary.
    this.lastBoundary = null;
    this.pending = [];
    this.failed = [];
    this.failedRetryable = false;
  }

  /**
   * True while any edit has not yet reached the server: one in flight, one
   * queued behind it, or one parked in the failed backlog after a conflict.
   *
   * Exposed for the unload guard. The commit lands roughly 4s after an edit
   * (debounce plus round trip), and closing the tab inside that window used to
   * destroy the edit with no warning and no recovery — while reload and in-app
   * navigation were both covered by the draft queue, which taught users the
   * "Saved" chip could be trusted.
   */
  hasUncommittedWork(): boolean {
    return this.inFlight || this.pending.length > 0 || this.failed.length > 0;
  }

  authority(): { graphId?: string; revision: number } {
    return { graphId: this.graphId, revision: this.revision };
  }

  needsSave(request: GraphSaveRequest): boolean {
    return !this.matchesAuthoritativeSnapshot(request, graphSaveFingerprint(request));
  }

  /**
   * Whether the document differs from the latest accepted or queued snapshot,
   * ignoring author attribution. Used only to flush real user edits before an
   * agent boundary; normal enqueue still treats authorship as significant.
   */
  needsSnapshotSave(request: GraphSaveRequest): boolean {
    const fingerprint = graphSaveFingerprint(request);
    const outstanding = this.outstanding();
    const latest = outstanding[outstanding.length - 1];
    return fingerprint !== (latest?.fingerprint ?? this.lastFingerprint);
  }

  enqueue(
    request: GraphSaveRequest,
    options: GraphSaveEnqueueOptions = {}
  ): Promise<SavedGraph | null> {
    const fingerprint = graphSaveFingerprint(request);
    if (this.matchesAuthoritativeSnapshot(request, fingerprint)) {
      return Promise.resolve(null);
    }
    if (this.failed.length > 0 && options.rejectIfBlocked) {
      return Promise.reject(new Error(
        this.failedRetryable
          ? 'Graph saving is offline. Retry the save before starting an AI turn.'
          : 'The saved graph has a revision conflict. Reload it before starting an AI turn.'
      ));
    }
    return new Promise((resolve, reject) => {
      const queued = {
        request,
        fingerprint,
        waiters: [{ resolve, reject }],
      };
      if (this.failed.length > 0) {
        appendBehindPossiblySentHead(this.failed, queued);
        this.emitPending();
        return;
      }
      appendQueuedSave(this.pending, queued);
      this.emitPending();
      void this.drain();
    });
  }

  retry(): void {
    if (this.failed.length === 0 || !this.failedRetryable || this.inFlight) return;
    // Preserve the ambiguous failed head exactly. Re-coalescing while moving it
    // back to pending could replace that snapshot with a later autosave.
    this.pending = [...this.failed, ...this.pending];
    this.failed = [];
    this.failedRetryable = false;
    void this.drain();
  }

  private async drain(): Promise<void> {
    if (this.inFlight) return;
    this.inFlight = true;
    try {
      while (this.pending.length > 0) {
        const queued = this.pending.shift()!;
        const saveGeneration = this.generation;
        this.active = queued;
        this.activeGeneration = saveGeneration;
        if (
          queued.fingerprint === this.lastFingerprint
          && sameCommitBoundary(queued.request, this.lastBoundary)
        ) {
          this.clearActive(queued);
          this.emitPending();
          queued.waiters.forEach(({ resolve }) => resolve(null));
          continue;
        }

        this.emitState('saving');
        try {
          const saved = await this.persist(
            queued.request.graph,
            queued.request.name,
            queued.request.description,
            this.graphId,
            {
              ...(this.graphId ? { expectedRevision: this.revision } : {}),
              author: queued.request.author,
              opSummary: queued.request.opSummary,
              createKey: queued.request.createKey,
              operations: queued.request.operations,
              layout: queued.request.layout,
            }
          );
          if (saveGeneration !== this.generation) {
            this.clearActive(queued);
            this.emitPending();
            queued.waiters.forEach(({ resolve }) => resolve(null));
            continue;
          }
          const created = !this.graphId || saved.id !== this.graphId;
          this.graphId = saved.id;
          this.revision = saved.currentRevision;
          this.lastFingerprint = queued.fingerprint;
          this.lastBoundary = {
            author: queued.request.author,
            opSummary: queued.request.opSummary,
          };
          this.failed = [];
          this.failedRetryable = false;
          if (created && this.callbacksActive) this.callbacks.onCreated?.(saved);
          if (this.callbacksActive) this.callbacks.onSaved?.(saved, queued.request);
          this.clearActive(queued);
          this.emitPending();
          this.emitState('saved');
          queued.waiters.forEach(({ resolve }) => resolve(saved));
        } catch (error) {
          if (saveGeneration !== this.generation) {
            this.clearActive(queued);
            this.emitPending();
            queued.waiters.forEach(({ resolve }) => resolve(null));
            continue;
          }
          const message = error instanceof Error ? error.message : String(error);
          const conflict = message.includes('(409)') || message.includes('graph_revision_conflict');
          this.failedRetryable = !conflict;
          const backlog = [queued, ...this.pending];
          this.pending = [];
          backlog.forEach((item) => {
            item.waiters.forEach(({ reject }) => reject(error));
            item.waiters = [];
          });
          this.clearActive(queued);
          this.failed = backlog;
          this.emitPending();
          this.emitState(conflict ? 'conflict' : 'offline');
          break;
        }
      }
    } finally {
      this.inFlight = false;
    }
  }
}
