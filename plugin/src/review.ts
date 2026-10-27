import { randomUUID } from 'node:crypto';
import { realpath } from 'node:fs/promises';
import path from 'node:path';
import { discoverReview, gitSnapshot, workingSnapshotCore, reviewCoverage,
  type ReviewInventory, type ReviewSession } from '@tensorviz/graph-host/reviewSnapshotsCore';
import { reviewGit } from '@tensorviz/graph-host/gitReviewSources';
import { emptyReviewProjection, projectReviewSnapshot } from '@tensorviz/graph-host/graphReviewCore';
import type { GraphReviewState, LocalBrowserMessage, LocalBrowserTargetIdentity, IdeMultifileProjectionResponse } from '@tensorviz/contracts/protocol';
import type { LocalBrowserHostPublish, LocalBrowserViewLink } from '@tensorviz/graph-host/localBrowserHostTransport';
import type { McpViewRegistry } from '@tensorviz/graph-host/mcpViews';
import { HostError, relativeInside, workspaceRoot } from './workspace';

interface ReviewArgs { workspaceRoot?: string; base?: string; file?: string; cls?: string; reviewId?: string; viewId?: string }
interface Session { data: ReviewSession; inventory: ReviewInventory; state: GraphReviewState;
  args: ReviewArgs; link?: LocalBrowserViewLink; viewId?: string }

/** Saved source review owns neither the live model nor execution authority. */
export class StandaloneReview {
  private session?: Session;
  private preparing = false;
  private generation = 0;
  private checking?: Promise<void>;
  private timer?: ReturnType<typeof setInterval>;
  private readonly receipts = new Map<string, { reviewId: string; mode: string }>();
  constructor(private readonly host: {
    prepare(root: string, signal: AbortSignal): Promise<void>;
    post<T>(route: string, body: unknown, signal: AbortSignal): Promise<T>;
    connect(signal: AbortSignal): Promise<void>;
    view(signal: AbortSignal): Promise<LocalBrowserViewLink>;
    views: McpViewRegistry;
    publish(message: LocalBrowserHostPublish): unknown;
    target(): LocalBrowserTargetIdentity | undefined;
    busy(): boolean;
    signal: AbortSignal;
  }) {}
  get active(): boolean { return !!this.session; }
  get busy(): boolean { return this.preparing; }
  replay(): void { this.host.publish({ type: 'graphReviewState', state: this.session?.state ?? null }); }

  async invoke(args: ReviewArgs, signal: AbortSignal): Promise<Record<string, unknown>> {
    signal = AbortSignal.any([signal, this.host.signal]);
    if (args.reviewId) {
      if (args.reviewId !== this.session?.state.id) throw new HostError('STALE_REVIEW', 'This comparison was closed or replaced. Open the intended comparison again.');
      await this.checkFreshness();
      return this.result(false);
    }
    if (this.preparing || this.host.busy()) throw new HostError('BUSY', 'Wait for capture or the current comparison to finish.');
    this.preparing = true;
    const generation = ++this.generation;
    try {
      const root = await workspaceRoot(args.workspaceRoot);
      let repository: string;
      try { repository = await realpath((await reviewGit(root, ['rev-parse', '--show-toplevel'])).trim()); }
      catch { throw new HostError('GIT_COMPARISON_UNAVAILABLE', 'Source comparison requires Git on PATH and a local checkout with a commit. No GitHub push is needed.'); }
      if (repository !== root) throw new HostError('WORKSPACE_REQUIRED', 'Supply the exact Git checkout root as workspaceRoot for source comparison.');
      const file = args.file ? relativeInside(root, path.resolve(root, args.file)) : undefined;
      if (file && !file.endsWith('.py')) throw new HostError('PYTHON_FILE_REQUIRED', 'Select a Python model file.');
      await this.host.prepare(root, signal);
      const data: ReviewSession = { title: 'Local model changes', folder: root,
        before: await gitSnapshot(root, args.base ?? 'HEAD'), after: await workingSnapshotCore(root),
        preferredFile: file, preferredClass: args.cls,
        refresh: async () => { throw new Error('Use the plugin review refresh action.'); } };
      const inventory = await discoverReview(data, { request: body => this.host.post('/api/ide/review/index', body, signal) });
      const requested = inventory.models.find(model => [model.before, model.after].some(candidate => candidate
        && (!file || candidate.file === file) && (!args.cls || candidate.cls === args.cls)));
      if (args.cls && !requested && inventory.models.length) throw new HostError('MODEL_NOT_DISCOVERED', 'The requested class is not an affected model. Inspect the returned workspace candidates and source coverage.');
      const state = await this.state(data, inventory, requested?.id ?? inventory.models[0]?.id, signal);
      signal.throwIfAborted();
      if (generation !== this.generation) throw new HostError('CANCELLED', 'The comparison was replaced.');
      const views = this.host.views.list();
      if (args.viewId && !views.some(view => view.viewId === args.viewId)) throw new HostError('VIEW_DISCONNECTED', 'The selected browser view is no longer connected.');
      const viewId = args.viewId ?? (views.length === 1 ? views[0].viewId : undefined);
      this.session = { data, inventory, state, args: { ...args, workspaceRoot: root }, viewId };
      this.receipts.clear();
      await this.host.connect(signal);
      if (!viewId) this.session.link = await this.host.view(signal);
      await this.checkFreshness();
      this.replay();
      if (this.timer) clearInterval(this.timer);
      this.timer = setInterval(() => { void this.checkFreshness(); }, 2500);
      this.timer.unref();
      return this.result(true);
    } catch (error) {
      if (error instanceof HostError) throw error;
      signal.throwIfAborted();
      throw new HostError('GIT_COMPARISON_UNAVAILABLE', error instanceof Error ? error.message.slice(0, 512) : 'The saved Git comparison could not be read.');
    } finally { this.preparing = false; }
  }

  private async state(data: ReviewSession, inventory: ReviewInventory, id: string | undefined, signal: AbortSignal): Promise<GraphReviewState> {
    const model = inventory.models.find(item => item.id === id), failures: string[] = [];
    const project = async (side: 'before' | 'after') => {
      const target = model?.[side];
      if (!target) return emptyReviewProjection(model?.label ?? 'Local comparison');
      try { return await projectReviewSnapshot(target, inventory.sources[side].get(target.project)!,
        body => this.host.post<IdeMultifileProjectionResponse>('/api/ide/multifile/project', body, signal)); }
      catch (error) {
        signal.throwIfAborted();
        failures.push(`${side}: ${target.file}::${target.cls}: ${error instanceof Error ? error.message : 'Source analysis unavailable.'}`);
        return { ...emptyReviewProjection(target.cls), truncated: true };
      }
    };
    const [before, after] = await Promise.all([project('before'), project('after')]);
    const live = this.host.target(), target = model?.after ?? model?.before;
    const liveContext = live && live.workspaceFolder === data.folder && live.file === target?.file && live.cls === target.cls
      ? { workspaceFolder: live.workspaceFolder, file: live.file, cls: live.cls } : undefined;
    const coverage = reviewCoverage(inventory, model);
    return { id: randomUUID(), workspaceRoot: data.folder, title: data.title,
      beforeLabel: data.before.label, afterLabel: data.after.label, before, after, stale: false,
      changedFiles: inventory.changedFiles.filter(file => file.endsWith('.py')),
      models: inventory.models.map(item => ({ id: item.id, label: item.label, file: (item.after ?? item.before)!.file, status: item.status })),
      selectedModel: id, modelStatus: failures.length ? 'unresolved' : model?.status, liveContext,
      notes: [...coverage.notes, ...failures], repositoryNotes: coverage.repositoryNotes,
      ...(model?.status === 'renamed' && model.before && model.after ? { renamedFiles: { [model.before.file]: model.after.file } } : {}) };
  }

  async checkFreshness(): Promise<void> {
    if (this.checking) return this.checking;
    const session = this.session;
    if (!session || session.state.stale) return;
    this.checking = (async () => {
      let unchanged = false;
      try {
        const now = await workingSnapshotCore(session.data.folder!);
        unchanged = now.truncated === session.data.after.truncated && now.files.size === session.data.after.files.size
          && [...now.files].every(([file, digest]) => session.data.after.files.get(file) === digest);
      } catch { /* An unreadable working tree cannot certify freshness. */ }
      if (!unchanged && this.session === session) { session.state = { ...session.state, stale: true }; this.receipts.clear(); this.replay(); }
    })();
    try { await this.checking; } finally { this.checking = undefined; }
  }

  private result(includeLink: boolean): Record<string, unknown> {
    const session = this.session!, state = session.state;
    const viewId = session.viewId ?? (session.link && this.host.views.viewForHandoff(session.link.handoffId));
    const connected = !!viewId && this.host.views.list().some(view => view.viewId === viewId);
    const expired = !connected && !!session.link && Date.parse(session.link.expiresAt) <= Date.now();
    const receipt = viewId ? this.receipts.get(viewId) : undefined;
    const displayed = !state.stale && connected && receipt?.reviewId === state.id && receipt.mode === 'diff';
    const otherMode = connected && receipt?.reviewId === state.id && receipt.mode !== 'diff';
    const empty = !state.models?.length ? 'no_models' : [state.before, state.after].every(projection =>
      !(projection.document as { semantic?: { nodes?: unknown[] } }).semantic?.nodes?.length) ? 'no_graph' : undefined;
    return { reviewId: state.id, status: state.stale ? 'stale' : empty ?? (displayed ? 'ready' : expired ? 'display_unavailable' : otherMode ? 'awaiting_diff' : 'awaiting_browser'), displayed,
      comparison: { before: state.beforeLabel, after: state.afterLabel, executed: false },
      selectedModel: state.selectedModel ?? null, models: state.models, changedFiles: state.changedFiles,
      coverage: { notes: state.notes, repositoryNotes: state.repositoryNotes, incomplete: state.modelStatus === 'unresolved'
        || !!state.before.truncated || !!state.after.truncated || !!state.notes?.length },
      ...(viewId ? { viewId } : {}),
      ...(includeLink && session.link ? { url: session.link.pageUrl, linkExpiresAt: session.link.expiresAt } : {}),
      nextAction: { kind: state.stale ? 'refresh' : empty ? 'inspect_source' : expired ? 'refresh' : displayed ? 'inspect' : otherMode ? 'show_diff' : includeLink && session.link ? 'open_browser' : 'poll',
        instruction: state.stale ? 'Saved files changed. Refresh this comparison before describing it as current.'
          : empty ? 'No comparable model graph is available. Inspect the returned coverage notes and saved source; polling cannot produce a graph.'
          : expired ? 'The browser link expired. Start open_graph_diff again for a fresh link; the source comparison did not execute either revision.'
          : displayed ? 'The exact browser rendered this source comparison. Retained capture shapes are context, not execution of either revision.'
          : otherMode ? 'Select Diff in the connected browser view, then poll this reviewId to confirm the comparison rendered.'
            : 'Open the private URL once if returned, then poll this reviewId. Do not claim the diff is visible until displayed is true; report browser errors separately.' } };
  }

  async handle(message: LocalBrowserMessage): Promise<void> {
    if (message.type === 'viewState' && 'viewId' in message) this.receipts.delete(message.viewId);
    if (message.type === 'graphReviewApplied') {
      const connected = new Set(this.host.views.list().map(view => view.viewId));
      for (const id of this.receipts.keys()) if (!connected.has(id)) this.receipts.delete(id);
      if (message.reviewId === this.session?.state.id && connected.has(message.viewId)) {
        this.receipts.set(message.viewId, message);
        if (this.receipts.size > 32) this.receipts.delete(this.receipts.keys().next().value!);
      }
      return;
    }
    const session = this.session;
    if (message.type !== 'graphReviewRequest' || !session || message.reviewId !== session.state.id) return;
    if (message.action === 'close') { this.close(); return; }
    if (this.preparing || this.host.busy()) return;
    if (message.action === 'refresh') {
      await this.invoke({ ...session.args, viewId: session.viewId ?? (session.link && this.host.views.viewForHandoff(session.link.handoffId)) }, this.host.signal);
    } else if (message.action === 'selectModel' && session.inventory.models.some(model => model.id === message.modelId)) {
      this.preparing = true;
      try {
        const state = await this.state(session.data, session.inventory, message.modelId, this.host.signal);
        if (this.session === session) { session.state = { ...state, stale: session.state.stale }; this.receipts.clear(); this.replay(); }
      } finally { this.preparing = false; }
    }
  }

  close(): void {
    ++this.generation; this.session = undefined; this.receipts.clear();
    if (this.timer) clearInterval(this.timer);
    this.replay();
  }

  reportError(error: unknown): void {
    if (!this.session) return;
    const detail = error instanceof Error ? error.message.slice(0, 512) : 'Source review could not complete.';
    this.session.state = { ...this.session.state, notes: [...this.session.state.notes ?? [], detail].slice(-200) };
    this.replay();
  }
}
