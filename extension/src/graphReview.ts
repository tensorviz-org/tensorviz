import * as vscode from 'vscode';
import { emptyReviewProjection, projectReviewSnapshot, reviewAnalysisError } from '@tensorviz/graph-host/graphReviewCore';
export { emptyReviewProjection } from '@tensorviz/graph-host/graphReviewCore';
import { randomUUID } from 'node:crypto';
import * as path from 'node:path';
import { changedReviewFiles, gitReviewSourceSet, resolveReviewRevision, reviewGit } from '@tensorviz/graph-host/gitReviewSources';
import { collectSourceSet, SourceSet, vscodeSourceIO } from './workspaceSources';
import type { GraphReviewRequest, GraphReviewState, IdeMultifileProjectionResponse, IdeMultifileSpan, LocalBrowserTargetIdentity, ProjectionPayload } from '@tensorviz/contracts/protocol';
import type { LocalBrowserHostPublish } from '@tensorviz/graph-host/localBrowserHostTransport';
import { discoverReview, reviewCoverage, reviewDirectory, reviewFilePath, ReviewInventory, ReviewModelRoot, ReviewSession } from './reviewSnapshots';
import { GraphToolError } from '@tensorviz/graph-host/mcpGraph';

export interface GraphReviewSpec { before: string; after: string; mergeBase?: boolean }
interface Review {
  state: GraphReviewState; target: LocalBrowserTargetIdentity; spec: GraphReviewSpec;
  sources: { before: SourceSet; after: SourceSet };
}
const scheme = 'tensorviz-git-review';
/** A file's identity through directory aliases; unset when it cannot be resolved. */
const onDisk = (folder: string, file: string) => { try { return reviewFilePath(path.join(folder, file)); } catch { return undefined; } };

/** Window-owned immutable review. Webviews can request only cached node IDs. */
export class GraphReview implements vscode.Disposable {
  private review?: Review;
  private generation = 0;
  private opening = false;
  private sourceEpoch = 0;
  private session?: { data: ReviewSession; inventory: ReviewInventory; id: string; selected?: string; updated?: boolean; stale?: boolean };
  private updateTimer?: ReturnType<typeof setInterval>;
  private following = 0;
  private revealing = 0;
  private watcher?: vscode.Disposable;
  private documents = new Map<string, { text: string; side: 'before' | 'after'; file: string; reviewId: string; sessionId?: string }>();
  private readonly subscriptions: vscode.Disposable[];
  constructor(private readonly options: {
    target(): LocalBrowserTargetIdentity | undefined;
    endpoint(): { origin: string; token: string } | undefined;
    publish(message: LocalBrowserHostPublish): unknown;
  }) {
    this.subscriptions = [
      vscode.workspace.registerTextDocumentContentProvider(scheme, { provideTextDocumentContent: uri => this.documents.get(uri.toString())?.text ?? '' }),
      vscode.window.onDidChangeTextEditorSelection(event => {
        if (this.session) { void this.followEditor(event.textEditor, event.selections[0]?.active.line); return; }
        const document = this.documents.get(event.textEditor.document.uri.toString());
        const review = this.review;
        if (!review || document?.reviewId !== review.state.id) return;
        const line = event.selections[0]?.active.line;
        if (line === undefined) return;
        const entries = Object.entries(review.state[document.side].spanmap) as [string, IdeMultifileSpan][];
        const matches = entries.filter(([, span]) => span.file === document.file && span.start[0] <= line + 1 && span.end[0] >= line + 1)
          .sort((a, b) => (a[1].end[0] - a[1].start[0]) - (b[1].end[0] - b[1].start[0]));
        if (matches.length) this.options.publish({ type: 'graphReviewFocus', reviewId: review.state.id, side: document.side,
          nodeIds: matches.filter(([, span]) => span.start[0] === matches[0][1].start[0] && span.end[0] === matches[0][1].end[0]).slice(0, 100).map(([id]) => id) });
      }),
      vscode.window.onDidChangeActiveTextEditor(editor => {
        if (editor && this.session) void this.followEditor(editor, editor.selection.active.line);
      }),
      vscode.workspace.onDidChangeTextDocument(event => this.sourceChanged(event.document.uri)),
      vscode.workspace.onDidSaveTextDocument(document => this.sourceChanged(document.uri)),
      vscode.workspace.onDidCreateFiles(event => event.files.forEach(uri => this.sourceChanged(uri))),
      vscode.workspace.onDidDeleteFiles(event => event.files.forEach(uri => this.sourceChanged(uri))),
      vscode.workspace.onDidRenameFiles(event => event.files.forEach(item => { this.sourceChanged(item.oldUri); this.sourceChanged(item.newUri); })),
    ];
  }
  ownsDiff(input: vscode.TabInputTextDiff): boolean {
    return [input.original, input.modified].every(uri => {
      const document = this.documents.get(uri.toString());
      return !!document && (this.session ? document.sessionId === this.session.id : document.reviewId === this.review?.state.id);
    });
  }
  get current(): GraphReviewState | undefined { return this.review?.state; }
  get active(): boolean { return !!this.review; }
  replay(): void { this.options.publish({ type: 'graphReviewState', state: this.review?.state ?? null }); }
  private sourceChanged(uri: vscode.Uri): void {
    this.sourceEpoch++;
    const review = this.review;
    if (!review || (!this.session?.data.after.working && review.spec.after !== 'WORKTREE') || uri.scheme !== 'file' || review.state.stale) return;
    const relative = path.relative(reviewDirectory(review.target.workspaceFolder), reviewFilePath(uri.fsPath));
    if (relative.startsWith('..') || path.isAbsolute(relative) || !relative.endsWith('.py')) return;
    if (this.session) this.session.stale = true;
    review.state = { ...review.state, stale: true };
    this.replay();
  }
  async handle(request: GraphReviewRequest): Promise<void> {
    if (request.action === 'open') return this.open();
    if (request.reviewId !== this.review?.state.id) return;
    if (request.action === 'close') return this.close();
    if (request.action === 'refresh') {
      if (this.session) {
        const current = this.session;
        try { const next = await current.data.refresh(); if (this.session === current) await this.openSession(next, current.selected); }
        catch (error) { void vscode.window.showErrorMessage(`Graph review could not refresh. ${(error as Error).message}`); }
        return;
      }
      return this.open(this.review!.spec);
    }
    if (request.action === 'selectModel' && request.modelId) return this.selectModel(request.modelId);
    if (request.action === 'reveal' && request.side) await this.reveal(request.side, request.nodeId);
  }
  async openSession(data: ReviewSession, preferredModel?: string, signal?: AbortSignal): Promise<GraphReviewState | undefined> {
    if (signal?.aborted) return;
    const generation = ++this.generation;
    const sourceEpoch = this.sourceEpoch;
    const endpoint = this.options.endpoint();
    if (!endpoint) throw new Error('The model backend is unavailable.');
    const inventory = await vscode.window.withProgress({ location: vscode.ProgressLocation.Window, title: 'Finding affected models…' },
      () => discoverReview(data, endpoint));
    if (generation !== this.generation || signal?.aborted) return;
    const requested = data.preferredClass && inventory.models.find(model => [model.after, model.before]
      .some(root => root?.file === data.preferredFile && root?.cls === data.preferredClass));
    if (data.preferredClass && !requested && inventory.models.length) {
      throw new GraphToolError('MODEL_NOT_DISCOVERED', `No affected model named ${data.preferredClass} was discovered in ${data.preferredFile}. Choose a discovered model or check review coverage.`);
    }
    await this.closeTabs();
    if (generation !== this.generation || signal?.aborted) return;
    this.watcher?.dispose();
    if (this.updateTimer) clearInterval(this.updateTimer);
    const session = this.session = { data, inventory, id: randomUUID(), stale: !!data.after.working && sourceEpoch !== this.sourceEpoch };
    if (data.after.working && data.folder) {
      const watcher = vscode.workspace.createFileSystemWatcher(new vscode.RelativePattern(data.folder, '**/*.py'));
      this.watcher = vscode.Disposable.from(watcher, watcher.onDidChange(uri => this.sourceChanged(uri)), watcher.onDidCreate(uri => this.sourceChanged(uri)), watcher.onDidDelete(uri => this.sourceChanged(uri)));
    }
    const cursor = data.preferredLocation;
    const atCursor = cursor && inventory.models.find(model => {
      const root = model[cursor.side];
      return root?.file === cursor.file && root.line <= cursor.line && root.endLine >= cursor.line;
    });
    const inFile = inventory.models.find(model => model.after?.file === data.preferredFile || model.before?.file === data.preferredFile);
    const previousReview = this.review;
    await this.selectModel(inventory.models.find(model => model.id === preferredModel)?.id ?? (requested ? requested.id : undefined)
      ?? atCursor?.id ?? inFile?.id ?? inventory.models[0]?.id, true, signal);
    if (this.session !== session || signal?.aborted || this.review === previousReview) return;
    if (data.updated) {
      this.updateTimer = setInterval(() => void this.checkUpdates(), 60_000);
      this.updateTimer.unref?.();
      void this.checkUpdates();
    }
    return this.review?.state;
  }
  private async checkUpdates(): Promise<void> {
    const session = this.session;
    if (!session?.data.updated || this.review?.state.updated) return;
    try {
      const updated = await session.data.updated();
      if (updated && this.session === session && this.review) { session.updated = true; this.review.state = { ...this.review.state, updated }; this.replay(); }
    } catch {
      if (this.session === session && this.review) {
        const note = 'Could not check for PR updates. Refresh to retry.';
        if (!this.review.state.notes?.includes(note)) { this.review.state.notes = [...this.review.state.notes ?? [], note]; this.replay(); }
      }
    }
  }
  private async selectModel(id?: string, reveal = true, signal?: AbortSignal): Promise<void> {
    const session = this.session;
    if (!session) return;
    const model = session.inventory.models.find(item => item.id === id);
    if (id && !model) return;
    const generation = ++this.generation;
    try {
      const failures: string[] = [];
      const project = async (side: 'before' | 'after') => {
        const root = model?.[side];
        if (!root) return emptyReviewProjection(model?.label ?? 'Pull request');
        try { return await this.projectSnapshot(root, session.inventory.sources[side].get(root.project)!); }
        catch (error) {
          failures.push(`${side}: ${root.file}::${root.cls}: ${(error as Error).message}`);
          return { ...emptyReviewProjection(root.cls), truncated: true };
        }
      };
      const [before, after] = await Promise.all([project('before'), project('after')]);
      if (this.session !== session || this.generation !== generation || signal?.aborted) return;
      session.selected = id;
      const root = model?.after ?? model?.before;
      const live = this.options.target();
      // The Canvas traces a nested project from its own root (modelProjectRoot);
      // review names files from the Git toplevel. The same file on disk is the same model.
      const liveFile = live && onDisk(live.workspaceFolder, live.file);
      const liveContext = live && root && session.data.folder && live.cls === root.cls && liveFile && liveFile === onDisk(session.data.folder, root.file)
        ? { workspaceFolder: live.workspaceFolder, file: live.file, cls: live.cls } : undefined;
      const coverage = reviewCoverage(session.inventory, model);
      const expandedSources = (side: 'before' | 'after'): SourceSet => {
        const item = model?.[side];
        if (!item) return { entries: [], truncated: false };
        const set = session.inventory.sources[side].get(item.project)!;
        return { ...set, entries: set.entries.map(entry => ({ ...entry, file: item.project ? `${item.project}/${entry.file}` : entry.file })) };
      };
      this.review = { target: { workspaceFolder: session.data.folder ?? '', file: root?.file ?? '', cls: root?.cls ?? '',
        documentVersion: 0, sourceDigest: '', backendRevision: 0 }, spec: { before: session.data.before.label, after: session.data.after.label },
        sources: { before: expandedSources('before'), after: expandedSources('after') }, state: {
          id: randomUUID(), title: session.data.title, beforeLabel: session.data.before.label, afterLabel: session.data.after.label,
          before, after, changedFiles: session.inventory.changedFiles.filter(file => file.endsWith('.py')), stale: !!session.stale,
          models: session.inventory.models.map(item => ({ id: item.id, label: item.label, file: (item.after ?? item.before)!.file, status: item.status })),
          selectedModel: id, modelStatus: failures.length ? 'unresolved' : model?.status,
          liveContext,
          notes: [...coverage.notes, ...failures], repositoryNotes: coverage.repositoryNotes, updated: session.updated,
          ...(model?.status === 'renamed' && model.before && model.after ? { renamedFiles: { [model.before.file]: model.after.file } } : {}),
        } };
      this.replay();
      if (reveal && model) await this.reveal(model.after ? 'after' : 'before');
    } catch (error) {
      if (generation === this.generation) void vscode.window.showErrorMessage(`Graph review could not open this model. ${(error as Error).message}`);
    }
  }
  private async projectSnapshot(root: ReviewModelRoot, sources: SourceSet): Promise<ProjectionPayload> {
    return projectReviewSnapshot(root, sources, body => this.requestProjection(body));
  }
  private async followEditor(editor: vscode.TextEditor, line: number | undefined): Promise<void> {
    const session = this.session;
    if (!session || line === undefined || this.revealing) return;
    const stored = this.documents.get(editor.document.uri.toString());
    const location = stored?.sessionId === session.id ? stored : session.data.locate?.(editor.document.uri);
    if (!location) return;
    const request = ++this.following;
    try {
      if (await session.data[location.side].read(location.file) !== editor.document.getText() || this.session !== session || request !== this.following || this.revealing) return;
      const choices = session.inventory.models.filter(model => {
        const root = model[location.side];
        return root?.file === location.file && root.line <= line + 1 && root.endLine >= line + 1;
      });
      if (choices.length === 1 && session.selected !== choices[0].id) await this.selectModel(choices[0].id, false);
      if (this.session !== session || request !== this.following || !this.review) return;
      const review = this.review;
      const matches = (Object.entries(review.state[location.side].spanmap) as [string, IdeMultifileSpan][])
        .filter(([, span]) => span.file === location.file && span.start[0] <= line + 1 && span.end[0] >= line + 1)
        .sort((a, b) => a[1].end[0] - a[1].start[0] - (b[1].end[0] - b[1].start[0]));
      if (matches.length) setTimeout(() => {
        if (this.review === review) this.options.publish({ type: 'graphReviewFocus', reviewId: review.state.id, side: location.side,
          nodeIds: matches.filter(([, span]) => span.start[0] === matches[0][1].start[0] && span.end[0] === matches[0][1].end[0]).slice(0, 100).map(([id]) => id) });
      }, 50);
    } catch { /* A document from a different or superseded snapshot cannot focus this review. */ }
  }
  async open(spec?: GraphReviewSpec): Promise<void> {
    if (this.opening) return;
    const target = this.options.target();
    if (!target) { void vscode.window.showInformationMessage('Open a model as a graph before reviewing its changes.'); return; }
    this.opening = true;
    const generation = ++this.generation;
    try {
      if (!spec) {
        const choice = await vscode.window.showQuickPick([
          { label: 'Working changes', description: 'HEAD → current files, including unsaved Python buffers', value: 'working' },
          { label: 'Branch changes', description: 'Merge base → HEAD, for reviewing a pull request', value: 'branch' },
          { label: 'Two revisions', description: 'Choose any two local Git revisions', value: 'revisions' },
        ], { title: 'Review model changes', placeHolder: 'Choose what to compare in Canvas and the native code diff' });
        if (!choice) return;
        if (choice.value === 'working') spec = { before: 'HEAD', after: 'WORKTREE' };
        else {
          const before = await vscode.window.showInputBox({ title: choice.value === 'branch' ? 'Pull request base branch' : 'Before revision', value: 'main', prompt: 'A local branch, tag, or commit. Git history will not be modified.' });
          if (!before) return;
          const after = choice.value === 'branch' ? 'HEAD' : await vscode.window.showInputBox({ title: 'After revision', value: 'HEAD' });
          if (!after) return;
          spec = { before, after, mergeBase: choice.value === 'branch' };
        }
      }
      const selectedSpec = spec;
      this.watcher?.dispose();
      const watcher = vscode.workspace.createFileSystemWatcher(new vscode.RelativePattern(target.workspaceFolder, '**/*.py'));
      this.watcher = vscode.Disposable.from(watcher, watcher.onDidChange(uri => this.sourceChanged(uri)), watcher.onDidCreate(uri => this.sourceChanged(uri)), watcher.onDidDelete(uri => this.sourceChanged(uri)));
      const sourceEpoch = this.sourceEpoch;
      const next = await vscode.window.withProgress({ location: vscode.ProgressLocation.Window, title: 'Comparing model revisions…' }, async () => {
        let beforeCommit = await resolveReviewRevision(target.workspaceFolder, selectedSpec.before);
        const afterCommit = selectedSpec.after === 'WORKTREE' ? undefined : await resolveReviewRevision(target.workspaceFolder, selectedSpec.after);
        if (selectedSpec.mergeBase && afterCommit) beforeCommit = (await reviewGit(target.workspaceFolder, ['merge-base', beforeCommit, afterCommit])).trim();
        const before = await gitReviewSourceSet(target.workspaceFolder, target.file, beforeCommit);
        const after = afterCommit ? await gitReviewSourceSet(target.workspaceFolder, target.file, afterCommit)
          : await collectSourceSet(target.file, vscodeSourceIO(target.workspaceFolder));
        const [left, right] = await Promise.all([this.project(target, before), this.project(target, after)]);
        return { target, spec: selectedSpec, sources: { before, after }, state: {
          id: randomUUID(), beforeLabel: `${selectedSpec.mergeBase ? 'base' : selectedSpec.before} · ${beforeCommit.slice(0, 7)}`,
          afterLabel: afterCommit ? `${selectedSpec.after} · ${afterCommit.slice(0, 7)}` : 'Working changes',
          before: left, after: right, changedFiles: changedReviewFiles(before, after), stale: false as boolean,
          liveContext: { workspaceFolder: target.workspaceFolder, file: target.file, cls: target.cls },
        } } satisfies Review;
      });
      if (generation !== this.generation || this.options.target()?.file !== target.file || this.options.target()?.cls !== target.cls
        || this.options.target()?.workspaceFolder !== target.workspaceFolder) return;
      await this.closeTabs();
      this.session = undefined;
      if (this.updateTimer) clearInterval(this.updateTimer);
      if (selectedSpec.after === 'WORKTREE' && sourceEpoch !== this.sourceEpoch) next.state.stale = true;
      this.review = next;
      this.replay();
      await this.reveal('after');
    } catch (error) {
      if (generation === this.generation) void vscode.window.showErrorMessage(`Graph review could not open. ${error instanceof Error ? error.message : 'Please retry.'}`);
    } finally { this.opening = false; }
  }
  private async project(target: LocalBrowserTargetIdentity, sources: SourceSet): Promise<ProjectionPayload> {
    return projectReviewSnapshot({ file: target.file, cls: target.cls, project: '' }, sources, body => this.requestProjection(body));
  }
  private async requestProjection(body: unknown): Promise<IdeMultifileProjectionResponse> {
    const endpoint = this.options.endpoint();
    if (!endpoint) throw new Error('The model backend is unavailable. Reopen the model.');
    const result = await fetch(`${endpoint.origin}/api/ide/multifile/project`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-tensorViz-Token': endpoint.token },
      body: JSON.stringify(body), signal: AbortSignal.timeout(30_000),
    });
    if (!result.ok) {
      const root = (body as { root?: { file?: string } }).root?.file || 'source';
      throw new Error(reviewAnalysisError(root, result.status, await result.text()));
    }
    return await result.json() as IdeMultifileProjectionResponse;
  }
  private uri(side: 'before' | 'after', file: string): vscode.Uri {
    const review = this.review!;
    const uri = vscode.Uri.from({ scheme, authority: review.state.id, path: `/${side}/${file}` });
    this.documents.set(uri.toString(), { text: review.sources[side].entries.find(entry => entry.file === file)?.source ?? '', side, file, reviewId: review.state.id, sessionId: this.session?.id });
    return uri;
  }
  private async reveal(side: 'before' | 'after', nodeId?: string): Promise<void> {
    const review = this.review;
    if (!review) return;
    const root = this.session?.inventory.models.find(item => item.id === this.session?.selected)?.[side];
    const span = nodeId ? review.state[side].spanmap[nodeId] as IdeMultifileSpan | undefined
      : root ? { file: root.file, start: [root.line, 0], end: [root.line, 0] } as IdeMultifileSpan : undefined;
    if (nodeId && !span) return;
    const file = span?.file ?? review.target.file;
    if (!review.sources.before.entries.some(entry => entry.file === file) && !review.sources.after.entries.some(entry => entry.file === file)) return;
    const model = this.session?.inventory.models.find(item => item.id === this.session?.selected);
    const isRoot = file === model?.before?.file || file === model?.after?.file;
    const leftFile = isRoot ? model?.before?.file ?? file : file, rightFile = isRoot ? model?.after?.file ?? file : file;
    const editor = this.session?.data.editorDiff;
    const reuse = editor && editor.beforeFile === leftFile && editor.afterFile === rightFile;
    const left = reuse ? editor.original : this.uri('before', leftFile), right = reuse ? editor.modified : this.uri('after', rightFile);
    this.revealing++;
    try {
      await vscode.commands.executeCommand('vscode.diff', left, right, `${file} (${review.state.beforeLabel} ↔ ${review.state.afterLabel})`, {
        preview: true, preserveFocus: true, viewColumn: vscode.ViewColumn.One,
        ...(span && side === 'after' ? { selection: new vscode.Range(span.start[0] - 1, span.start[1], span.end[0] - 1, span.end[1]) } : {}),
      });
      if (span) {
        const editor = vscode.window.visibleTextEditors.find(item => item.document.uri.toString() === (side === 'before' ? left : right).toString());
        if (editor) {
          const range = new vscode.Range(span.start[0] - 1, span.start[1], span.end[0] - 1, span.end[1]);
          editor.selection = new vscode.Selection(range.start, range.end);
          editor.revealRange(range, vscode.TextEditorRevealType.InCenterIfOutsideViewport);
        }
      }
    } finally { this.revealing--; }
  }
  private async closeTabs(): Promise<void> {
    const tabs = vscode.window.tabGroups.all.flatMap(group => group.tabs).filter(tab => tab.input instanceof vscode.TabInputTextDiff && tab.input.original.scheme === scheme);
    if (tabs.length) await vscode.window.tabGroups.close(tabs, true);
    this.documents.clear();
  }
  async close(): Promise<void> { this.watcher?.dispose(); this.watcher = undefined; if (this.updateTimer) clearInterval(this.updateTimer); ++this.generation; this.session = undefined; this.review = undefined; this.replay(); await this.closeTabs(); }
  dispose(): void { this.watcher?.dispose(); if (this.updateTimer) clearInterval(this.updateTimer); ++this.generation; this.session = undefined; this.review = undefined; this.subscriptions.forEach(item => item.dispose()); this.documents.clear(); }
}

