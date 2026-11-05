import * as vscode from 'vscode';
import { randomUUID } from 'node:crypto';
import { CanvasBridge, errorHtml, shellHtml, ShellTarget } from './shell';
import { LocalBrowserHostPublish } from '@tensorviz/graph-host/localBrowserHostTransport';
import { canvasHomeHtml, canvasHomeParts, CanvasHomeState } from './canvasHome';
import type { CanvasHomeSetup } from './canvasHomeSetup';
import { OPEN_AS_GRAPH } from './commandIds';
import { selectedTextUri } from './editorSelection';
import { NEW_MODEL_COMMAND } from './views';
import {
  isLocalBrowserMessage, LOCAL_BROWSER_PROTOCOL_VERSION, LOCAL_BROWSER_SEQUENCED_MESSAGE_TYPES,
  LocalBrowserMessage, newLocalBrowserMessageId,
} from '@tensorviz/contracts/protocol';

export const SIDEBAR_CANVAS_VIEW = 'tensorviz.sidebarCanvas';
export const SHOW_SIDEBAR_CANVAS = 'tensorviz.showSidebarCanvas';
export const CANVAS_REQUESTED = 'tensorviz.canvasRequested';
export const CANVAS_VISIBLE = 'tensorviz.canvasVisible';

/** The provider owns presentation, never the graph, review, validation or backend.
 * prepare starts only the service; it must not select or execute a model. */
export interface SidebarCanvasOptions {
  prepare(): Promise<ShellTarget | undefined>;
  home?(): CanvasHomeState;
  checkHome?(signal: AbortSignal): Promise<CanvasHomeSetup>;
  /** SSH forwarding addresses are not permanent. Re-resolve on explicit show. */
  refreshTarget?: boolean;
  receive(message: LocalBrowserMessage): void;
  replay(): void;
  viewConnection?(viewId: string, connected: boolean): void;
  channel: vscode.OutputChannel;
}

const VIEW_REQUESTS = new Set([
  'agentRequest', 'codeFocusRequest',
  'graphReviewRequest', 'snapshotRequest', 'revealNode', 'sourceEditRequest', 'sourceEditCancel',
  'validateRequest', 'validateCancel', 'projectFocus', 'projectExpand', 'projectDeclare',
  'viewSelection', 'viewFocusResult', 'ack', 'error',
  'candidateRequest', 'captureInspectRequest', 'captureExportRequest', 'captureExportCancel',
  'graphSetupRequest',
]);
const SEQUENCED = new Set<string>(LOCAL_BROWSER_SEQUENCED_MESSAGE_TYPES);

export class SidebarCanvas implements vscode.WebviewViewProvider, vscode.Disposable {
  private view?: vscode.WebviewView;
  private bridge?: CanvasBridge;
  private subscriptions: vscode.Disposable[] = [];
  private target?: ShellTarget;
  private starting?: Promise<void>;
  private disposed = false;
  private ready = false;
  private dirty = false;
  private failure = false;
  private requested = false;
  private visibilityGeneration = 0;
  private sequence = 0;
  private sessionId = randomUUID();
  private readonly viewId = randomUUID();
  private connected = false;
  // A view can disappear while its manual validation or native review completes.
  // Keep the latest receipts as well as the controller's authoritative snapshot.
  private readonly receipts = new Map<string, LocalBrowserHostPublish>();
  private presentation: unknown;
  private checkpoint?: { requestId: string; finish(): void };
  private modelKey?: string;
  private homeState?: CanvasHomeState;
  private homeRendered = false;
  private contentVisible = false;
  private reviewVisible = false;
  private projectVisible = false;
  private homeCheck?: AbortController;
  private homeRetry?: { run(): Promise<unknown>; label: string };

  constructor(private readonly options: SidebarCanvasOptions) {}

  get requestedVisible(): boolean { return this.requested; }
  get visible(): boolean { return this.view?.visible ?? false; }
  get connectedViewId(): string | undefined { return this.connected ? this.viewId : undefined; }
  /** Read-only host evidence; renderer delivery is asserted separately in Electron. */
  inspectForTest() {
    return { visible: this.visible, requested: this.requested, ready: this.ready,
      homeVisible: this.visible && !!this.homeState && !this.contentVisible,
      sessionId: this.sessionId, viewId: this.viewId, presentation: this.presentation,
      receipts: [...this.receipts.values()], messages: this.bridge?.received ?? [] };
  }
  async handleForTest(message: Parameters<CanvasBridge['handleForTest']>[0]): Promise<boolean> {
    if (!this.bridge) return false;
    await this.bridge.handleForTest(message);
    return true;
  }

  register(): vscode.Disposable {
    return vscode.window.registerWebviewViewProvider(SIDEBAR_CANVAS_VIEW, this,
      { webviewOptions: { retainContextWhenHidden: true } });
  }

  /** Native entry and ordinary reveal never acquire a runtime. */
  async reveal(): Promise<void> {
    if (this.disposed) return;
    this.visibilityGeneration++;
    this.requested = true;
    await vscode.commands.executeCommand('setContext', CANVAS_REQUESTED, true);
    // Focus the VIEW, wherever the user moved it. Never reset a container layout.
    if (!this.requested || this.disposed) return;
    await vscode.commands.executeCommand(`${SIDEBAR_CANVAS_VIEW}.focus`);
    if (this.target && !this.failure && this.options.refreshTarget) await this.start(true);
  }

  /** Explicit graph/review actions prepare the existing service owner. */
  async show(): Promise<boolean> {
    await this.reveal();
    if (this.disposed || !this.requested) return false;
    await this.start();
    return !this.disposed && !!this.target && !this.failure;
  }

  async hide(): Promise<void> {
    const generation = ++this.visibilityGeneration;
    this.requested = false;
    // A round trip captures the editor's current camera even when its batched
    // store listener / 100 ms gesture checkpoint has not fired yet.
    if (this.ready && this.visible) await this.capturePresentation();
    if (generation !== this.visibilityGeneration || this.requested || this.disposed) return;
    await vscode.commands.executeCommand('setContext', CANVAS_REQUESTED, false);
    if (!this.visible || generation !== this.visibilityGeneration || this.disposed) return;
    // Hide this view in its current container without closing another extension.
    // The unconditional contribution remains registered; focus restores it.
    await vscode.commands.executeCommand(`${SIDEBAR_CANVAS_VIEW}.removeView`);
  }

  private capturePresentation(): Promise<void> {
    this.checkpoint?.finish();
    return new Promise(resolve => {
      const requestId = randomUUID();
      const timer = setTimeout(() => {
        this.options.channel.appendLine('[canvas] Presentation checkpoint unavailable; retaining the last checkpoint.');
        finish();
      }, 1000);
      const finish = () => { clearTimeout(timer); this.checkpoint = undefined; resolve(); };
      this.checkpoint = { requestId, finish };
      this.send({ type: 'canvasCheckpoint', requestId });
    });
  }

  resolveWebviewView(view: vscode.WebviewView): void {
    if (this.disposed) return;
    this.detach();
    this.view = view;
    view.title = 'tensorViz';
    this.subscriptions.push(view.onDidDispose(() => {
      if (this.view === view) { this.detach(); this.updateVisibility(); }
    }), view.onDidChangeVisibility(() => {
      this.updateVisibility();
      if (view.visible && this.ready && this.dirty) this.replay();
    }), view.webview.onDidReceiveMessage(message => {
      if (message?.type === 'homeReady') {
        this.refreshHome();
        this.send({ type: 'canvasContent', visible: this.contentVisible });
        return;
      }
      if (this.contentVisible || this.homeState?.pending) return;
      if (message?.type === 'homeAction') {
        void this.homeAction(message.action);
      } else if (message?.type === 'homeModel' && Number.isInteger(message.index) && message.index >= 0) {
        const model = this.homeState?.models[message.index];
        if (model) void this.runHomeRequest({ label: `Opening ${model.name}…`,
          run: async () => vscode.commands.executeCommand(OPEN_AS_GRAPH, { uri: model.uri, cls: model.name }) }, true);
      }
    }));
    this.updateVisibility();
    if (this.target && !this.failure) this.mount();
    else if (this.failure) this.setError();
    this.refreshHome(true);
  }

  refreshHome(recheck = false): void {
    if (!this.view || this.contentVisible || this.disposed) return;
    this.homeState = { ...this.homeState, ...(this.options.home?.() ?? { hasWorkspace: false, indexed: false, models: [] }) };
    if (this.homeRendered) this.send({ type: 'canvasHomeState', html: canvasHomeParts(this.homeState).body });
    else {
      this.view.webview.options = { enableScripts: true };
      this.view.webview.html = canvasHomeHtml(this.homeState);
      this.homeRendered = true;
    }
    if (this.options.checkHome && (recheck || !this.homeState.setup)) void this.checkHome();
  }

  private async checkHome(): Promise<void> {
    this.homeCheck?.abort();
    const controller = this.homeCheck = new AbortController();
    this.homeState = { ...this.homeState!, setup: { phase: 'checking', detail: 'Checking the tensorViz runtime. You can choose a model or PR while checks run.' } };
    this.refreshHome();
    try {
      const setup = await this.options.checkHome!(controller.signal);
      if (controller.signal.aborted || this.disposed) return;
      this.homeState = { ...this.homeState!, setup };
    } catch {
      if (controller.signal.aborted || this.disposed) return;
      this.homeState = { ...this.homeState!, setup: { phase: 'unavailable', detail: 'The tensorViz runtime check could not complete. Inspect the log or check again.' } };
    }
    this.refreshHome();
  }

  private async runHomeRequest(request: { run(): Promise<unknown>; label: string }, remember = false): Promise<void> {
    if (this.disposed || this.homeState?.pending) return;
    const opening = remember || request === this.homeRetry;
    if (remember) this.homeRetry = request;
    this.homeState = { ...this.homeState!, pending: request.label, error: undefined, retryAvailable: !!this.homeRetry };
    this.refreshHome();
    try { await request.run(); }
    catch { this.homeState = { ...this.homeState!, error: 'Opening did not complete. Retry or inspect the tensorViz log.' }; }
    finally {
      if (!this.disposed) {
        if (opening && !this.homeState?.error) this.homeRetry = undefined;
        this.homeState = { ...this.homeState!, pending: undefined, retryAvailable: !!this.homeRetry };
        this.refreshHome(true);
      }
    }
  }

  private async homeAction(action: unknown): Promise<void> {
    if (action === 'open' || action === 'pr') {
      const target = action === 'open' ? { uri: selectedTextUri() } : undefined;
      await this.runHomeRequest({ label: action === 'pr' ? 'Opening pull request…' : 'Opening your model…',
        run: async () => action === 'pr' ? vscode.commands.executeCommand('tensorviz.openPullRequest')
          : vscode.commands.executeCommand(OPEN_AS_GRAPH, target) }, true);
    } else if (action === 'starter') await this.runHomeRequest({ label: 'Choosing a starter model…',
      run: async () => vscode.commands.executeCommand(NEW_MODEL_COMMAND) });
    else if (action === 'retry' && this.homeRetry) await this.runHomeRequest(this.homeRetry);
    else if (action === 'setup') await this.runHomeRequest({ label: 'Preparing tensorViz…', run: () => this.show() });
    else if (action === 'check') this.refreshHome(true);
    else if (action === 'log') this.options.channel.show();
    else if (action === 'trust') await vscode.commands.executeCommand('workbench.trust.manage');
    else if (action === 'folder') await vscode.commands.executeCommand('workbench.action.files.openFolder');
  }

  private updateVisibility(): void {
    void vscode.commands.executeCommand('setContext', CANVAS_VISIBLE, this.visible);
    this.updateConnection();
  }

  private updateConnection(): void {
    const connected = this.ready && this.visible;
    if (connected === this.connected) return;
    this.connected = connected;
    this.options.viewConnection?.(this.viewId, connected);
    if (connected) this.send({ type: 'canvasVisible' });
  }

  private start(refresh = false): Promise<void> {
    if (this.disposed || (this.target && !this.failure && !refresh)) return Promise.resolve();
    if (this.starting) return this.starting;
    if (!this.contentVisible) {
      this.homeState = { ...this.homeState!, error: undefined,
        pending: this.homeState?.pending ?? 'Preparing tensorViz…' };
      this.refreshHome();
    }
    this.starting = (async () => {
      try {
        const target = await this.options.prepare();
        if (this.disposed) return;
        if (!target) { this.setError(); return; }
        if (!this.failure && this.ready && this.target?.appUrl === target.appUrl && this.target.appOrigin === target.appOrigin) return;
        if (this.ready && this.visible) await this.capturePresentation();
        if (this.disposed) return;
        this.target = target;
        this.failure = false;
        this.mount();
      } catch {
        // Backend/preflight owns detailed diagnostics. Do not echo source,
        // credentials or arbitrary exception text into the webview.
        if (!this.disposed) this.setError();
      }
    })().finally(() => {
      this.starting = undefined;
      if (this.homeState?.pending === 'Preparing tensorViz…') {
        this.homeState = { ...this.homeState, pending: undefined };
        this.refreshHome(true);
      }
    });
    return this.starting;
  }

  setError(): void {
    this.failure = true;
    this.ready = false;
    this.updateConnection();
    if (!this.contentVisible) {
      this.homeState = { ...this.homeState!, pending: undefined,
        retryAvailable: !!this.homeRetry,
        error: 'tensorViz could not start. Complete runtime setup or retry.' + (this.homeRetry ? ' Your selected opening is retained.' : '') };
      this.refreshHome();
      return;
    }
    if (this.view) {
      this.homeRendered = false;
      this.view.webview.html = errorHtml('Canvas could not connect to the backend.',
        'Check the tensorViz output and selected Python environment, then run tensorViz: Open as Graph to retry.', '');
    }
  }

  /** Environment changes replace the service without changing Canvas visibility or its model. */
  async prepareBackendRestart(): Promise<void> {
    if (this.ready && this.visible) await this.capturePresentation();
    this.bridge?.dispose();
    this.bridge = undefined;
    this.target = undefined;
    this.ready = false;
    this.updateConnection();
    if (!this.contentVisible) this.refreshHome();
    else if (this.view) {
      this.homeRendered = false;
      this.view.webview.html = errorHtml('Switching Python environment…',
        'Canvas will reconnect when the backend is ready.', '');
    }
  }

  async reconnect(): Promise<void> {
    if (this.view) await this.start(true);
  }

  private mount(): void {
    const view = this.view, target = this.target;
    if (!view || !target) return;
    this.bridge?.dispose();
    this.ready = false;
    this.updateConnection();
    this.sessionId = randomUUID();
    view.webview.options = CanvasBridge.options(target.port, target.forwarded);
    this.bridge = new CanvasBridge(view.webview, target, this.options.channel);
    this.bridge.onMessage(raw => {
      const message = raw as unknown as { type: string; protocolVersion?: number; message?: unknown; state?: unknown; requestId?: string };
      if (message.type === 'canvasConnect') {
        if (message.protocolVersion !== LOCAL_BROWSER_PROTOCOL_VERSION) { this.setError(); return true; }
        this.ready = true;
        this.updateConnection();
        this.send({ type: 'canvasConnected', sessionId: this.sessionId, viewId: this.viewId, presentation: this.presentation });
        this.replay();
        this.send({ type: 'canvasContent', visible: this.contentVisible });
        return true;
      }
      if (message.type === 'canvasPresentation') {
        if (JSON.stringify(message.state)?.length <= 4 * 1024 * 1024) this.presentation = message.state;
        if (message.requestId === this.checkpoint?.requestId) this.checkpoint?.finish();
        return true;
      }
      if (message.type !== 'canvasMessage') return;
      if (this.ready && isLocalBrowserMessage(message.message, this.sessionId) && VIEW_REQUESTS.has(message.message.type)) {
        if ('viewId' in message.message && (message.message.viewId !== this.viewId || !this.connected)) return true;
        this.options.receive(message.message);
      }
      return true;
    });
    const url = new URL(target.appUrl);
    url.searchParams.set('view', 'canvas');
    this.homeState = { ...this.homeState, ...(this.options.home?.() ?? { hasWorkspace: false, indexed: false, models: [] }) };
    view.webview.html = shellHtml({ ...target, appUrl: url.toString() }, this.homeState, this.contentVisible);
    this.homeRendered = true;
  }

  /** Same typed publication contract as the browser transport; no socket,
   * cookies, credentials, model execution or second controller here. */
  post(message: LocalBrowserHostPublish): boolean {
    if (this.disposed) return false;
    if (message.type === 'focusView' && (!this.connected || message.viewId !== this.viewId)) return false;
    const target = message.type === 'snapshot' ? message.state.activeTarget
      : 'target' in message ? message.target : undefined;
    if (target) {
      const key = JSON.stringify([target.workspaceFolder, target.file, target.cls]);
      if (key !== this.modelKey && ['snapshot', 'activeTarget', 'projection'].includes(message.type)) {
        this.receipts.clear();
        this.modelKey = key;
      }
    } else if (message.type === 'snapshot' && message.state.activeTarget === null) {
      this.receipts.clear();
      this.modelKey = undefined;
    }
    if (message.type === 'snapshot' && message.state.validation) {
      this.receipts.set('validationState', { type: 'validationState', ...message.state.validation });
    }
    if (['validateResult', 'validationState', 'sourceEditResult'].includes(message.type)) this.receipts.set(message.type, message);
    this.deliver(message);
    if (message.type === 'graphReviewState') this.reviewVisible = !!message.state;
    if (message.type === 'projectIndex') this.projectVisible = !!message.index;
    const content = message.type === 'snapshot' ? !!(message.state.activeTarget || message.state.opening || this.reviewVisible || this.projectVisible)
      : message.type === 'activeTarget' || message.type === 'projection' ? !!message.target || this.reviewVisible || this.projectVisible
      : message.type === 'graphReviewState' || message.type === 'projectIndex' ? this.reviewVisible || this.projectVisible || !!this.modelKey
      : undefined;
    if (content !== undefined && content !== this.contentVisible) {
      this.contentVisible = content;
      if (this.homeRendered) this.send({ type: 'canvasContent', visible: content });
      if (content) {
        this.homeCheck?.abort(); this.homeRetry = undefined;
        this.homeState = { ...this.homeState!, error: undefined, retryAvailable: false };
      }
      else this.refreshHome(true);
    }
    return true;
  }

  private deliver(message: LocalBrowserHostPublish): void {
    if (!this.ready || !this.visible) this.dirty = true;
    if (!this.ready) return;
    const full = {
      ...message, protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
      workspaceSessionId: this.sessionId, messageId: newLocalBrowserMessageId(),
      ...(SEQUENCED.has(message.type) ? { hostSequence: ++this.sequence } : {}),
      ...(message.type === 'snapshot' ? { role: 'host' } : {}),
    };
    if (!isLocalBrowserMessage(full, this.sessionId)) throw new Error(`Invalid Canvas publication: ${message.type}`);
    this.send({ type: 'canvasMessage', message: full });
  }

  private send(message: unknown): void {
    const view = this.view;
    if (!view) { this.dirty = true; return; }
    try {
      void Promise.resolve(view.webview.postMessage(message)).then(delivered => {
        if (!delivered && this.view === view) this.dirty = true;
      }, () => { if (this.view === view) this.dirty = true; });
    } catch { this.dirty = true; }
  }

  private replay(): void {
    this.dirty = false;
    this.options.replay();
    for (const message of this.receipts.values()) this.deliver(message);
  }

  private detach(): void {
    this.homeCheck?.abort();
    this.checkpoint?.finish();
    this.bridge?.dispose();
    this.bridge = undefined;
    this.subscriptions.splice(0).forEach(subscription => subscription.dispose());
    this.view = undefined;
    this.homeRendered = false;
    this.ready = false;
    this.updateConnection();
    this.dirty = true;
  }

  dispose(): void {
    this.disposed = true;
    this.homeCheck?.abort();
    this.detach();
    this.receipts.clear();
    this.presentation = undefined;
    this.target = undefined;
    this.updateVisibility();
  }
}
