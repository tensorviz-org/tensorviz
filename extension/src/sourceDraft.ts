import { emptyReviewProjection, projectReviewSnapshot } from '@tensorviz/graph-host/graphReviewCore';
import type { SourceSet } from '@tensorviz/graph-host/workspaceSourceCore';
import type { GraphReviewRequest, GraphReviewState, IdeMultifileProjectionResponse, LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import type { LocalBrowserHostPublish } from '@tensorviz/graph-host/localBrowserHostTransport';
import type { CodeArtifact } from './sourceChanges';

export interface SourceDraftInput {
  id: string; target: LocalBrowserTargetIdentity; artifact: CodeArtifact; sources: SourceSet;
}
export interface SourceDraftHandle { show(reveal: () => Promise<void>, decide: (approved: boolean) => void): void; dispose(): void }

/** A proposal owns this temporary view. It never replaces the live projection. */
export class SourceDrafts {
  private pending?: { state?: GraphReviewState; reveal?: () => Promise<void>; decide?: (approved: boolean) => void; dispose(): void };
  constructor(private readonly options: {
    endpoint(): { origin: string; token: string } | undefined;
    publish(message: LocalBrowserHostPublish): unknown;
    changed(listener: () => void): { dispose(): void };
    fetchImpl?: typeof fetch;
  }) {}
  get current() { return this.pending?.reveal ? this.pending.state : undefined; }
  get active() { return !!this.pending; }
  replay(): void { if (this.current) this.options.publish({ type: 'graphReviewState', state: this.current }); }
  handle(request: GraphReviewRequest): void {
    if (request.reviewId !== this.current?.id) { this.replay(); return; }
    if (request.action === 'reveal') void this.pending?.reveal?.();
    else if (request.action === 'discardDraft' || (request.action === 'acceptDraft' && !this.current?.stale)) {
      const decide = this.pending?.decide;
      if (this.pending) this.pending.decide = undefined;
      decide?.(request.action === 'acceptDraft');
    } else if (request.action === 'acceptDraft') this.replay();
  }
  dispose(): void { this.pending?.dispose(); }

  start(input: SourceDraftInput, signal: AbortSignal, current: () => boolean): SourceDraftHandle {
    this.dispose();
    const controller = new AbortController();
    const entry: NonNullable<SourceDrafts['pending']> = { dispose: () => {
      controller.abort(); clearTimeout(timeout); changed.dispose(); signal.removeEventListener('abort', entry.dispose);
      if (this.pending !== entry) return;
      this.pending = undefined;
      this.options.publish({ type: 'graphReviewState', state: null });
    } };
    this.pending = entry;
    const changed = this.options.changed(() => {
      if (entry.state && !entry.state.stale && !current()) { entry.state = { ...entry.state, stale: true }; this.replay(); }
    });
    const timeout = setTimeout(() => controller.abort(), 30_000);
    signal.addEventListener('abort', entry.dispose, { once: true });
    const endpoint = this.options.endpoint();
    const request = async (body: unknown): Promise<IdeMultifileProjectionResponse> => {
      if (!endpoint) throw new Error('backend unavailable');
      const response = await (this.options.fetchImpl ?? fetch)(`${endpoint.origin}/api/ide/multifile/project`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', 'X-tensorViz-Token': endpoint.token },
        body: JSON.stringify(body), signal: controller.signal,
      });
      if (!response.ok) throw new Error('draft projection unavailable');
      return response.json();
    };
    const target = { file: input.target.file, cls: input.target.cls, project: '' };
    const after = { ...input.sources, entries: input.sources.entries.map(source => source.file === input.artifact.path
      ? { ...source, source: input.artifact.after, digest: input.artifact.after_digest } : source) };
    // Start alongside the native diff. Late replies can never resurrect a closed proposal.
    void Promise.all([projectReviewSnapshot(target, input.sources, request), projectReviewSnapshot(target, after, request)])
      .then(([before, after]) => ({ before, after, unavailable: false }))
      .catch(() => ({ before: emptyReviewProjection(target.cls), after: emptyReviewProjection(target.cls), unavailable: true }))
      .then(result => {
        clearTimeout(timeout);
        if (this.pending !== entry || signal.aborted) return;
        entry.state = { id: input.id, proposal: { id: input.id, ...(result.unavailable ? { unavailable: true } : {}) },
          beforeLabel: 'Current code', afterLabel: 'Proposed code', before: result.before, after: result.after,
          changedFiles: [input.artifact.path], stale: !current(), title: input.target.cls,
          liveContext: { workspaceFolder: input.target.workspaceFolder, file: input.target.file, cls: input.target.cls } };
        this.replay();
      });
    if (signal.aborted) entry.dispose();
    return { show: (reveal, decide) => { if (this.pending === entry && !signal.aborted) { entry.reveal = reveal; entry.decide = decide; this.replay(); } }, dispose: entry.dispose };
  }
}
