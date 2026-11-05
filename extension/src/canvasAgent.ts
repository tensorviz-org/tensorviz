import { randomUUID } from 'node:crypto';
import { CANVAS_AGENT_PERMISSION_LIMIT, canvasAgentPermissionPolicyAllows, isCanvasAgentPermissionMode, isCanvasAgentPermissionRequest, isCanvasAgentPermissionPresentation, isCanvasAgentSettings, type CanvasAgentAction, type CanvasAgentPermission,
  type CanvasAgentPermissionPresentation,
  type CanvasAgentPermissionMode,
  type CanvasAgentPermissionRequest, type GraphOpeningState, type CanvasAgentAvailability, type CanvasAgentProvider, type CanvasAgentSelections,
  type CanvasAgentSettings, type CanvasAgentState, type CanvasAgentTurn, type LocalBrowserAgentRequest, type LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import { validateAgentSettings, waitingForAgent } from './canvasAgentRuntime';
import { graphSetupWorkflow } from '@tensorviz/contracts/graphSetupWorkflow';
import { buildGraphSetupContinuationPrompt, buildPermissionContinuationPrompt, buildSetupDiagnosisPrompt, buildCommandResultPrompt } from './canvasAgentPrompts';

export interface AgentRunInput {
  provider: 'codex' | 'claude';
  cwd: string;
  context: string;
  conversationId?: string;
  selectionEvidence?: Record<string, unknown>;
  messages: CanvasAgentTurn[];
  settings?: CanvasAgentSettings;
  intent?: 'open_graph' | 'node_explanation' | 'model_explanation';
}
export type AgentWarmInput = Omit<AgentRunInput, 'messages' | 'context'> & { context?: string; preparation?: { context: string } };
export type AgentRunEvent = { kind: 'text'; text: string } | { kind: 'activity'; text: string }
  | { kind: 'opened_graph'; text: string; target: LocalBrowserTargetIdentity }
  | { kind: 'setup_evidence'; opening: GraphOpeningState };
export type CanvasAgentPermissionRequester = (request: CanvasAgentPermissionRequest, signal: AbortSignal) => Promise<boolean | undefined>;
export interface CanvasAgentRoute {
  provider: CanvasAgentProvider;
  settings: CanvasAgentSettings;
  resolvedModel?: string;
  detail?: string;
  ready: boolean;
  busy: boolean;
}
export interface CanvasAgentOptions {
  prepare(target: LocalBrowserTargetIdentity, nodeIds: string[], draftId: string | undefined, signal: AbortSignal, intent?: 'open_graph', viewId?: string): Promise<{ cwd: string; context: string; selectionEvidence?: Record<string, unknown> }>;
  prepareWarm?(viewId: string, signal: AbortSignal): Promise<{ cwd: string; context: string } | undefined>;
  openingPending?(target: LocalBrowserTargetIdentity): boolean;
  openingComplete?(target: LocalBrowserTargetIdentity): boolean;
  openingProgress?(target: LocalBrowserTargetIdentity, signal: AbortSignal, waitMs?: number): Promise<GraphOpeningState | undefined>;
  cancelOpening?(target: LocalBrowserTargetIdentity): void;
  run(input: AgentRunInput, emit: (event: AgentRunEvent) => void, signal: AbortSignal): Promise<void>;
  publish(viewId: string, state: CanvasAgentState): void;
  clear(preserveEvidence?: boolean): void;
  inspect?(provider: CanvasAgentProvider, refresh: boolean, signal: AbortSignal): Promise<CanvasAgentAvailability>;
  warm?(provider: CanvasAgentProvider, settings: CanvasAgentSettings, signal: AbortSignal, intent?: 'open_graph', prepared?: { cwd: string; context: string }): Promise<void>;
  preferences?: { provider: CanvasAgentProvider; selections: CanvasAgentSelections };
  remember?(preferences: { provider: CanvasAgentProvider; selections: CanvasAgentSelections }): void;
  invalidate?(): void;
  finishTurn?(signal: AbortSignal): void;
  projectWork?: { running(signal?: AbortSignal): boolean; wait(signal: AbortSignal, waitMs: number): Promise<void>; unread(signal: AbortSignal): readonly string[] };
}

/** One window-owned conversation. Hiding a view never cancels its work. */
export class CanvasAgent {
  private state: CanvasAgentState = this.empty();
  private readonly views = new Set<string>();
  private readonly submitted = new Set<string>();
  private setupTarget?: LocalBrowserTargetIdentity;
  private active?: AbortController;
  private operationOwner?: AbortController;
  private publishTimer?: NodeJS.Timeout;
  private disposed = false;
  private discovery?: AbortController;
  private warming?: { controller: AbortController; provider: CanvasAgentProvider; checked: CanvasAgentAvailability };
  private hasPreference = false;
  private warmViewId?: string;
  /** Preparation is a full-context request on the user's account: at most one per conversation. */
  private preparedSession?: string;
  private readonly permissionWaiters = new Set<() => void>();
  private permission?: { receipt: CanvasAgentPermission; controller: AbortController; sessionId: string; target: LocalBrowserTargetIdentity;
    done: Promise<void>; resolve(allowed: boolean): void; cleanup(): void };

  constructor(private readonly options: CanvasAgentOptions) {
    const saved = options.preferences;
    if (saved && ['codex', 'claude'].includes(saved.provider)) {
      this.state.provider = saved.provider; this.hasPreference = true;
      this.state.selections = Object.fromEntries(Object.entries(saved.selections ?? {})
        .filter(([key, value]) => ['codex', 'claude'].includes(key) && isCanvasAgentSettings(value)));
    }
  }

  private empty(): CanvasAgentState {
    return { sessionId: randomUUID(), revision: 0, provider: 'codex', selections: {},
      agents: { codex: { status: 'unchecked', detail: 'Open chat to check the agent.', models: [] },
        claude: { status: 'unchecked', detail: 'Open chat to check the agent.', models: [] } },
      status: 'idle', messages: [], activity: '', permissionMode: 'ask' };
  }

  private publish(): void {
    clearTimeout(this.publishTimer); this.publishTimer = undefined;
    this.state.revision++;
    for (const view of this.views) this.options.publish(view, structuredClone(this.state));
  }

  disconnect(viewId: string): void { this.views.delete(viewId); }

  /** Read account routing without changing settings, discovering or starting a chat. */
  routing(): CanvasAgentRoute {
    const provider = this.state.provider, availability = this.state.agents?.[provider];
    const settings = this.runSettings(provider, this.state.selections?.[provider] ?? {});
    return { provider, settings: { ...settings }, resolvedModel: settings.model ?? availability?.defaultModel,
      ready: !this.disposed && availability?.status === 'ready', detail: availability?.detail, busy: !!this.active || !!this.warming || !!this.discovery };
  }

  /** Explicit source interpretation checks its configured provider without opening or warming chat. */
  async prepareExplanation(signal: AbortSignal): Promise<CanvasAgentRoute> {
    signal.throwIfAborted();
    const route = this.routing();
    if (route.busy || this.disposed) return route;
    const provider = route.provider, selection = JSON.stringify(this.state.selections?.[provider] ?? {});
    let checked = this.state.agents?.[provider];
    if (checked?.status !== 'ready' && this.options.inspect) {
      checked = await this.options.inspect(provider, false, signal);
      signal.throwIfAborted();
      if (this.disposed || this.state.provider !== provider || JSON.stringify(this.state.selections?.[provider] ?? {}) !== selection) {
        throw new Error('The configured agent changed. Explain this model again.');
      }
      if (this.active || this.discovery || this.warming) return this.routing();
      this.state.agents![provider] = checked; this.publish();
    }
    const ready = this.routing();
    if (checked?.status === 'ready') {
      try { validateAgentSettings(checked, ready.settings); }
      catch (error) { return { ...ready, ready: false, detail: error instanceof Error ? error.message : 'Choose an available model in agent settings.' }; }
    }
    return ready;
  }

  /** Tool authority is captured from the active host turn, never supplied by model arguments. */
  captureOperation(): { target: LocalBrowserTargetIdentity; signal: AbortSignal; approve: CanvasAgentPermissionRequester; current(): boolean } {
    const controller = this.active, sessionId = this.state.sessionId, requestId = this.state.requestId;
    const target = this.state.context?.target;
    if (!controller || controller.signal.aborted || !target || !requestId || this.disposed) throw new Error('No Canvas agent task is active.');
    const captured = { ...target };
    const current = () => !this.disposed && this.active === controller && !controller.signal.aborted
      && this.state.sessionId === sessionId && this.state.requestId === requestId && this.samePermissionTarget(this.state.context?.target, captured);
    return { target: captured, signal: controller.signal, current, approve: async (request, signal) => {
      if (!current() || signal.aborted) return false;
      this.operationOwner = controller;
      return this.requestPermission(captured, request, signal);
    } };
  }

  /** Capture once when an operation starts so later stages cannot inherit another turn or native fallback. */
  permissionRequester(target: LocalBrowserTargetIdentity): CanvasAgentPermissionRequester {
    const controller = this.active, sessionId = this.state.sessionId, requestId = this.state.requestId;
    const capturedTarget = { ...target };
    if (controller && (!this.samePermissionTarget(this.state.context?.target, capturedTarget) || !requestId)) {
      throw new Error('The model or source changed. Restart setup for the current model before approving changes.');
    }
    if (controller) this.operationOwner = controller;
    return async (request, signal) => {
      if (this.disposed || signal.aborted || controller?.signal.aborted) return false;
      if (!controller) return undefined;
      if (this.active !== controller || this.state.sessionId !== sessionId || this.state.requestId !== requestId) return false;
      return this.requestPermission(capturedTarget, request, signal);
    };
  }

  /** Only an active agent task uses chat permissions; other hosts retain their existing review. */
  async requestPermission(target: LocalBrowserTargetIdentity, request: CanvasAgentPermissionRequest, signal: AbortSignal): Promise<boolean | undefined> {
    if (this.disposed || signal.aborted) return false;
    const controller = this.active;
    if (!controller) return undefined;
    if (controller.signal.aborted) return false;
    if (!this.samePermissionTarget(this.state.context?.target, target) || !this.state.requestId) {
      throw new Error('The model or source changed. Restart setup for the current model before approving changes.');
    }
    if (!isCanvasAgentPermissionRequest(request)) throw new Error('The setup permission is incomplete or too large to review.');
    if (this.permission) throw new Error('Another setup permission is waiting for your decision.');
    // Freeze authority before asking the agent to explain it. Neither returned data
    // nor later mutations of an installer's plan can change what is approved.
    const receipt: CanvasAgentPermission = { ...structuredClone(request), id: randomUUID(), requestId: this.state.requestId, status: 'pending' };
    return new Promise<boolean>(resolve => {
      let finish!: () => void;
      const done = new Promise<void>(settled => { finish = settled; });
      const abort = () => this.settlePermission('cancelled', receipt.id);
      this.permission = { receipt, controller, sessionId: this.state.sessionId, target: { ...target }, done, resolve: allowed => { resolve(allowed); finish(); },
        cleanup: () => { signal.removeEventListener('abort', abort); controller.signal.removeEventListener('abort', abort); } };
      this.state.permissions = [...(this.state.permissions ?? []).slice(-(CANVAS_AGENT_PERMISSION_LIMIT - 1)), receipt];
      signal.addEventListener('abort', abort, { once: true });
      controller.signal.addEventListener('abort', abort, { once: true });
      this.state.activity = 'Preparing approval…';
      if (signal.aborted || controller.signal.aborted) abort();
      else { this.publish(); for (const wake of this.permissionWaiters) wake(); }
    });
  }

  /** Read-only data for the active provider, never an execution or approval tool. */
  pendingPermission(): CanvasAgentPermission | undefined {
    return this.permissionIsCurrent() ? structuredClone(this.permission!.receipt) : undefined;
  }

  /** Generated copy can explain only this immutable request, once, in its owning turn. */
  presentPermission(id: string, presentation: CanvasAgentPermissionPresentation): void {
    const pending = this.permission;
    if (!pending || pending.receipt.id !== id || !this.permissionIsCurrent()) throw new Error('The permission request is no longer current. Read pending permissions again.');
    if (pending.receipt.presentation) throw new Error('This permission is already presented. Its reviewed explanation cannot be changed.');
    if (!isCanvasAgentPermissionPresentation(presentation)) throw new Error('Provide a concise title, summary, and distinct allow/deny labels without markup or line breaks.');
    pending.receipt.presentation = structuredClone(presentation);
    const mode = this.state.permissionMode ?? 'ask';
    if (this.policyApproves(pending.receipt, mode)) this.settlePermission('approved', id, mode);
    else { this.state.activity = 'Waiting for your approval…'; this.publish(); }
  }

  private samePermissionTarget(current: LocalBrowserTargetIdentity | undefined, target: LocalBrowserTargetIdentity): boolean {
    return !!current && current.workspaceFolder === target.workspaceFolder && current.file === target.file
      && current.cls === target.cls && current.sourceDigest === target.sourceDigest;
  }

  private policyApproves(request: CanvasAgentPermissionRequest, mode: CanvasAgentPermissionMode): boolean {
    return canvasAgentPermissionPolicyAllows(request, mode);
  }

  /** An installer can request wording without changing the opening snapshot. */
  private async waitForOpeningOrPermission(target: LocalBrowserTargetIdentity, signal: AbortSignal): Promise<GraphOpeningState | undefined> {
    const controller = new AbortController();
    const abort = () => controller.abort();
    signal.addEventListener('abort', abort, { once: true });
    let wake!: () => void;
    const permission = new Promise<void>(resolve => { wake = resolve; this.permissionWaiters.add(wake); });
    try {
      signal.throwIfAborted();
      if (!this.permission) await Promise.race([this.options.openingProgress!(target, controller.signal, 10000), permission]);
      signal.throwIfAborted();
      return await this.options.openingProgress!(target, signal, 0);
    } finally {
      controller.abort(); signal.removeEventListener('abort', abort); this.permissionWaiters.delete(wake);
    }
  }

  private permissionIsCurrent(): boolean {
    const pending = this.permission;
    return !!pending && !this.disposed && this.active === pending.controller && !pending.controller.signal.aborted && this.state.sessionId === pending.sessionId
      && this.state.requestId === pending.receipt.requestId && this.samePermissionTarget(this.state.context?.target, pending.target);
  }

  private settlePermission(status: Exclude<CanvasAgentPermission['status'], 'pending'>, id?: string, policyMode?: CanvasAgentPermissionMode): void {
    const pending = this.permission;
    if (!pending || id && pending.receipt.id !== id) return;
    this.permission = undefined;
    pending.cleanup(); pending.receipt.status = status;
    if (status === 'approved') {
      pending.receipt.approval = policyMode ? 'policy' : 'user';
      if (policyMode) pending.receipt.mode = policyMode;
    }
    pending.resolve(status === 'approved');
    this.state.activity = status === 'approved' ? 'Continuing setup…' : status === 'denied' ? 'Setup was not approved.' : 'Permission request cancelled.';
    if (!this.disposed) this.publish();
  }

  /** The setup button starts a host-authored task, using this conversation's saved agent settings. */
  openGraph(opening: GraphOpeningState): void {
    if (this.disposed || this.active) return;
    this.cancelWarm(); this.discovery?.abort(); this.discovery = undefined;
    // Only the abandoned discovery would have settled these; 'checking' disables Send and hides Refresh.
    for (const provider of ['codex', 'claude'] as const) if (this.state.agents![provider].status === 'checking') {
      this.state.agents![provider] = { status: 'unchecked', detail: 'The check was interrupted. Refresh to check the agent.', models: [] };
    }
    const prior = this.state.context?.target;
    if (prior && (prior.workspaceFolder !== opening.target.workspaceFolder || prior.file !== opening.target.file || prior.cls !== opening.target.cls)) {
      this.options.clear();
      const { provider, selections, agents } = this.state;
      this.state = { ...this.empty(), provider, selections, agents };
      this.submitted.clear();
    }
    if (this.submitted.size >= 20) {
      this.state.status = 'error'; this.state.activity = 'This conversation is full. Start a new chat to continue.'; this.publish(); return;
    }
    this.setupTarget = opening.target;
    const requestId = randomUUID();
    this.submitted.add(requestId);
    void this.send({ action: 'send', requestId, provider: this.state.provider, target: opening.target, nodeIds: [],
      prompt: `Set up this project and open the graph for ${opening.target.cls}.` }, 'open_graph');
  }

  handle(message: LocalBrowserAgentRequest): void {
    if (this.disposed) return;
    const knownView = this.views.has(message.viewId);
    if (message.action !== 'permission' && message.action !== 'permission_mode') this.views.add(message.viewId);
    if (['discover', 'configure', 'clear'].includes(message.action)) this.warmViewId = message.viewId;
    // Every accepted or refused action acknowledges its own ID. Background state
    // publications must never be mistaken for acknowledgement of a later action.
    this.state.ack = 'requestId' in message && message.requestId ? message.requestId : message.messageId;
    if (message.action === 'sync') { this.options.publish(message.viewId, structuredClone(this.state)); return; }
    if (message.action === 'discover') { if (!this.active) void this.discover(!!message.refresh); return; }
    if (message.action === 'permission_mode') {
      if (!knownView || !isCanvasAgentPermissionMode(message.mode)) { this.options.publish(message.viewId, structuredClone(this.state)); return; }
      this.state.permissionMode = message.mode;
      const pending = this.permission;
      if (pending && !this.permissionIsCurrent()) { this.settlePermission('cancelled', pending.receipt.id); return; }
      if (pending?.receipt.presentation && this.policyApproves(pending.receipt, message.mode)) this.settlePermission('approved', pending.receipt.id, message.mode);
      else this.publish();
      return;
    }
    if (message.action === 'permission') {
      const pending = this.permission;
      if (pending && !this.permissionIsCurrent()) {
        this.settlePermission('cancelled', pending.receipt.id); return;
      }
      if (knownView && pending?.receipt.id === message.permissionId && ['allow', 'deny'].includes(message.decision)
        && (message.decision === 'deny' || pending.receipt.presentation)) {
        this.settlePermission(message.decision === 'allow' ? 'approved' : 'denied', message.permissionId);
      } else this.options.publish(message.viewId, structuredClone(this.state));
      return;
    }
    if (message.action === 'stop') {
      if (message.requestId === this.state.requestId && this.active) {
        this.active.abort(); this.active = undefined;
        this.options.clear(true);
        this.state.status = 'stopped'; this.state.activity = 'Stopped. Pending agent work was cancelled.';
        this.publish();
      }
      return;
    }
    if (message.action === 'clear') {
      this.active?.abort(); this.active = undefined;
      this.settlePermission('cancelled');
      this.cancelWarm(); this.discovery?.abort(); this.discovery = undefined;
      this.options.clear();
      const { provider, selections, agents, ack } = this.state;
      this.state = { ...this.empty(), provider, selections, agents, ack };
      this.submitted.clear(); this.setupTarget = undefined; this.publish(); void this.discover(false); return;
    }
    if (this.active) { this.options.publish(message.viewId, structuredClone(this.state)); return; }
    if (message.action === 'configure') {
      if (this.state.messages.length && message.provider !== this.state.provider) {
        this.state.activity = 'Start a new chat to switch coding agents.'; this.publish(); return;
      }
      this.cancelWarm();
      this.hasPreference = true; this.state.provider = message.provider;
      this.state.selections = { ...this.state.selections, [message.provider]: { ...message.settings } };
      this.remember(); this.publish(); void this.warm(); return;
    }
    if ((this.discovery || this.warming) && this.state.agents?.[message.provider].status === 'checking') {
      this.state.activity = 'Wait for this agent’s check to finish before sending.'; this.publish(); return;
    }
    if (this.submitted.has(message.requestId)) return;
    if (this.submitted.size >= 20) {
      this.state.status = 'error'; this.state.activity = 'This conversation is full. Start a new chat to continue.'; this.publish(); return;
    }
    this.submitted.add(message.requestId);
    const setup = this.setupTarget;
    const followUp = setup && setup.workspaceFolder === message.target.workspaceFolder && setup.file === message.target.file
      && setup.cls === message.target.cls && setup.sourceDigest === message.target.sourceDigest
      && (this.options.openingPending?.(message.target) ?? !this.options.openingComplete?.(message.target));
    void this.send(message, followUp ? 'open_graph' : undefined, message.viewId);
  }

  private remember(): void {
    this.options.remember?.({ provider: this.state.provider, selections: structuredClone(this.state.selections ?? {}) });
  }

  private async discover(refresh: boolean): Promise<void> {
    if (!this.options.inspect || this.discovery) return;
    this.cancelWarm();
    const controller = this.discovery = new AbortController();
    const current = () => !this.disposed && this.discovery === controller && !controller.signal.aborted;
    const timer = setTimeout(() => controller.abort(), 45_000);
    for (const provider of ['codex', 'claude'] as const) this.state.agents![provider] = { ...this.state.agents![provider], status: 'checking', detail: 'Checking agent and sign-in…' };
    this.publish();
    try {
      await Promise.all((['codex', 'claude'] as const).map(async provider => {
        let checked: CanvasAgentAvailability;
        try { checked = await this.options.inspect!(provider, refresh, controller.signal); }
        catch { checked = { status: 'error', detail: 'Could not check this agent. Refresh to try again.', models: [] }; }
        if (current()) {
          this.state.agents![provider] = checked;
          if (provider === this.state.provider) await this.warm(controller.signal);
          if (current()) this.publish();
        }
      }));
      if (current() && !this.active && !this.hasPreference && !this.state.messages.length) {
        const ready = (['codex', 'claude'] as const).filter(provider => this.state.agents![provider].status === 'ready');
        if (ready.length === 1 && this.state.provider !== ready[0]) {
          this.state.provider = ready[0]; await this.warm(controller.signal);
        }
      }
    } finally {
      clearTimeout(timer);
      if (this.discovery === controller) {
        this.discovery = undefined;
        for (const provider of ['codex', 'claude'] as const) if (this.state.agents![provider].status === 'checking' && this.warming?.provider !== provider) {
          this.state.agents![provider] = { status: 'error', detail: 'The check timed out. Refresh to try again.', models: [] };
        }
        if (!this.disposed) this.publish();
      }
    }
  }

  private runSettings(provider: CanvasAgentProvider, settings: CanvasAgentSettings): CanvasAgentSettings {
    const availability = this.state.agents?.[provider];
    const model = availability?.models.find(model => model.id === (settings.model ?? availability.defaultModel));
    return !settings.effort && model?.defaultEffort ? { ...settings, effort: model.defaultEffort } : settings;
  }

  private cancelWarm(): void {
    if (!this.warming) return;
    const { controller, provider, checked } = this.warming;
    this.warming = undefined;
    if (this.state.agents?.[provider].status === 'checking') this.state.agents[provider] = checked;
    controller.abort();
  }

  private async warm(signal?: AbortSignal): Promise<void> {
    if (!this.options.warm || this.active || this.disposed) return;
    const provider = this.state.provider, checked = this.state.agents![provider];
    const settings = this.state.selections?.[provider] ?? {};
    try { validateAgentSettings(checked, settings); } catch { return; }
    this.cancelWarm();
    const controller = new AbortController();
    const warming = this.warming = { controller, provider, checked };
    const abort = () => controller.abort();
    signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) abort();
    const timer = setTimeout(() => controller.abort(), 45_000);
    this.state.agents![provider] = { ...checked, status: 'checking', detail: 'Preparing the agent connection…' };
    this.publish();
    try {
      const setup = this.setupTarget;
      const intent = setup && (this.options.openingPending?.(setup) ?? !this.options.openingComplete?.(setup)) ? 'open_graph' : undefined;
      const sessionId = this.state.sessionId;
      // Speculative context can be unavailable or become stale while opening.
      // Keep connection readiness usable; Send performs the authoritative checks.
      const prepared = !intent && !this.state.messages.length && this.warmViewId && this.preparedSession !== sessionId
        ? await this.options.prepareWarm?.(this.warmViewId, controller.signal).catch(() => undefined) : undefined;
      if (this.warming !== warming) return;
      controller.signal.throwIfAborted();
      await this.options.warm(provider, this.runSettings(provider, settings), controller.signal, intent, prepared);
      // A connected warm-up has sent its preparation. Sessions replaced for settings, Refresh
      // or a timeout take their context from Send instead; one cancelled earlier sent nothing.
      if (prepared) this.preparedSession = sessionId;
      if (this.warming === warming) {
        this.state.agents![provider] = controller.signal.aborted
          ? { ...checked, status: 'error', detail: 'The connection timed out. Refresh to try again.' } : checked;
      }
    } catch (error) {
      // The timeout aborts the connection, which rejects with a cancellation rather than resolving.
      if (this.warming === warming) this.state.agents![provider] = { ...checked, status: 'error',
        detail: controller.signal.aborted ? 'The connection timed out. Refresh to try again.'
          : error instanceof Error ? error.message.slice(0, 2048) : 'Could not prepare the agent connection. Refresh to try again.' };
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      if (this.warming === warming) { this.warming = undefined; this.publish(); }
    }
  }

  private async send(message: Extract<CanvasAgentAction, { action: 'send' }>, intent?: 'open_graph', viewId?: string): Promise<void> {
    const controller = new AbortController();
    this.active = controller;
    if (intent) controller.signal.addEventListener('abort', () => this.options.cancelOpening?.(message.target), { once: true });
    const current = () => !this.disposed && this.active === controller && !controller.signal.aborted;
    let started = false;
    try {
      if (!message.prompt.trim()) throw new Error('Enter a message first.');
      const prior = this.state.context?.target;
      if (prior && (prior.workspaceFolder !== message.target.workspaceFolder || prior.file !== message.target.file || prior.cls !== message.target.cls)) {
        throw new Error('This chat belongs to another model. Start a new chat for the selected model.');
      }
      if (this.state.messages.length && message.provider !== this.state.provider) throw new Error('Start a new chat to switch coding agents.');
      this.state.provider = message.provider;
      this.state.requestId = message.requestId;
      this.state.status = 'running'; this.state.activity = 'Preparing your request…'; this.state.failure = undefined;
      this.state.context = { target: { ...message.target }, nodeIds: [...message.nodeIds] };
      this.publish();
      let prepared = await this.options.prepare(message.target, message.nodeIds, message.draftId, controller.signal, intent, viewId);
      if (!current()) return;
      // First use should work with the installed agent even if the chat was never opened.
      if (intent && !this.hasPreference && this.options.inspect) {
        const providers = ['codex', 'claude'] as const;
        const checked = await Promise.all(providers.map(provider => this.options.inspect!(provider, false, controller.signal)));
        if (!current()) return;
        providers.forEach((provider, index) => { this.state.agents![provider] = checked[index]; });
        const ready = providers.filter((_, index) => checked[index].status === 'ready');
        if (ready.length === 1) message = { ...message, provider: ready[0] };
        this.state.provider = message.provider;
      }
      const settings = { ...(message.settings ?? this.state.selections?.[message.provider] ?? {}) };
      let runSettings = settings;
      if (this.options.inspect) {
        // A hostile or stale view cannot bypass readiness or advertise invented settings.
        const checked = await this.options.inspect(message.provider, false, controller.signal);
        if (!current()) return;
        this.state.agents![message.provider] = checked;
        validateAgentSettings(checked, settings);
        runSettings = this.runSettings(message.provider, settings);
      }
      if (intent) prepared = await this.options.prepare(message.target, message.nodeIds, message.draftId, controller.signal, intent, viewId);
      if (!current()) return;
      let opening = await this.options.openingProgress?.(message.target, controller.signal, 0);
      if (!current()) return;
      if (intent && this.options.openingProgress && (!opening || !this.samePermissionTarget(opening.target, message.target))) {
        this.state.status = 'idle'; this.state.activity = 'The model setup or source changed. Restart setup for the current model.'; return;
      }
      let openingId = opening?.id;
      this.hasPreference = true;
      this.state.selections = { ...this.state.selections, [message.provider]: settings }; this.remember();
      this.state.messages.push({ id: message.requestId, role: 'user', text: message.prompt.trim(), settings });
      let history = structuredClone(this.state.messages);
      const answer: CanvasAgentTurn = { id: randomUUID(), role: 'assistant', text: '', settings };
      this.state.messages.push(answer); this.state.activity = waitingForAgent(message.provider); this.publish();
      let continuations = 0;
      const presentationAttempts = new Map<string, number>();
      const commandResultAttempts = new Map<string, number>();
      const observedFailures = new Set<string>();
      const failureKey = (state: GraphOpeningState) => JSON.stringify([state.id, state.target.sourceDigest, state.failure]);
      for (;;) {
        const before = opening && graphSetupWorkflow(opening);
        started = true;
        let acceptingEvents = true;
        try {
          await this.options.run({ provider: message.provider, ...prepared, conversationId: this.state.sessionId, settings: runSettings, messages: history, ...(intent ? { intent } : {}) }, event => {
            if (!acceptingEvents || !current()) return;
            if (event.kind === 'setup_evidence') {
              if (this.samePermissionTarget(event.opening.target, message.target) && event.opening.failure?.diagnostic) {
                observedFailures.add(failureKey(event.opening));
              }
              return;
            }
            if (event.kind === 'opened_graph') {
              if (event.target.workspaceFolder === message.target.workspaceFolder) {
                if (this.permission && !this.samePermissionTarget(this.permission.target, event.target)) this.settlePermission('cancelled');
                this.state.context = { target: { ...event.target }, nodeIds: [] };
                this.setupTarget = undefined;
                this.publish();
              }
              return;
            }
            const firstText = event.kind === 'text' && !answer.text && !!event.text;
            if (event.kind === 'text') {
              answer.text = (answer.text + event.text).slice(0, 32000);
              if (this.state.activity === waitingForAgent(message.provider)) this.state.activity = 'Writing reply…';
            }
            else this.state.activity = this.permission ? this.permission.receipt.presentation ? 'Waiting for your approval…' : 'Preparing approval…' : event.text.slice(0, 2048);
            if (event.kind === 'activity' || firstText) this.publish();
            else if (!this.publishTimer) this.publishTimer = setTimeout(() => this.publish(), 80);
          }, controller.signal);
        } finally {
          acceptingEvents = false;
        }
        started = false;
        if (!current()) return;
        const ownsSetup = !!intent || this.operationOwner === controller || this.state.permissions?.some(receipt => receipt.requestId === message.requestId);
        // Host operations run in the background so the same provider can explain
        // newly resolved plans before waiting for the user's decision.
        for (;;) {
          if (this.permission && !this.permission.receipt.presentation) break;
          if (this.permission) await this.permission.done;
          if (!current()) return;
          if (this.options.projectWork?.running(controller.signal)) {
            await this.options.projectWork.wait(controller.signal, 10000);
            continue;
          }
          if (!ownsSetup || !this.options.openingProgress) break;
          opening = await this.options.openingProgress(message.target, controller.signal, 0);
          if (!current()) return;
          if (!opening || (openingId && opening.id !== openingId) || !this.samePermissionTarget(opening.target, message.target)
            || !this.samePermissionTarget(this.state.context?.target, message.target)) {
            this.state.status = 'idle'; this.state.activity = 'The model setup or source changed. Restart setup for the current model.'; return;
          }
          openingId ??= opening.id;
          const workflow = graphSetupWorkflow(opening);
          if (workflow.nextStep.kind !== 'wait') break;
          this.state.activity = workflow.stages.capture === 'running' ? 'Capturing the graph…' : 'Preparing project Python…';
          this.publish();
          opening = await this.waitForOpeningOrPermission(message.target, controller.signal);
          if (!current()) return;
        }
        const pending = this.pendingPermission();
        if (pending && !pending.presentation) {
          if (!this.samePermissionTarget(this.permission?.target, message.target)) {
            throw new Error('The model or source changed. Restart setup for the current model before approving changes.');
          }
          const attempts = presentationAttempts.get(pending.id) ?? 0;
          if (attempts >= 2 || presentationAttempts.size >= CANVAS_AGENT_PERMISSION_LIMIT) {
            this.settlePermission('cancelled', pending.id);
            throw new Error('The agent could not explain the pending operation. Setup stopped before that operation was approved. Retry setup to continue.');
          }
          presentationAttempts.set(pending.id, attempts + 1);
          prepared = await this.options.prepare(message.target, message.nodeIds, message.draftId, controller.signal, intent, viewId);
          if (!current()) return;
          history = [...structuredClone(this.state.messages), { id: randomUUID(), role: 'user', text: buildPermissionContinuationPrompt(pending) }];
          if (answer.text) answer.text += '\n\n';
          this.state.activity = 'Preparing approval…'; this.publish();
          continue;
        }
        const unreadCommands = this.options.projectWork?.unread(controller.signal) ?? [];
        if (unreadCommands.length) {
          const jobId = unreadCommands[0], attempts = commandResultAttempts.get(jobId) ?? 0;
          if (attempts >= 2 || commandResultAttempts.size >= 8) throw new Error('The command finished, but the agent did not inspect its result. Ask it to inspect the completed command before continuing.');
          commandResultAttempts.set(jobId, attempts + 1);
          prepared = await this.options.prepare(message.target, message.nodeIds, message.draftId, controller.signal, intent, viewId);
          if (!current()) return;
          history = [...structuredClone(this.state.messages), { id: randomUUID(), role: 'user', text: buildCommandResultPrompt(jobId) }];
          if (answer.text) answer.text += '\n\n';
          this.state.activity = 'Inspecting command result…'; this.publish();
          continue;
        }
        if (opening?.phase === 'failed' && opening.failure?.diagnostic && !observedFailures.has(failureKey(opening))) {
          const declined = this.state.permissions?.some(receipt => receipt.requestId === message.requestId && receipt.status === 'denied');
          if (declined || observedFailures.size >= 4) break;
          observedFailures.add(failureKey(opening));
          prepared = await this.options.prepare(message.target, message.nodeIds, message.draftId, controller.signal, intent, viewId);
          if (!current()) return;
          history = [...structuredClone(this.state.messages), { id: randomUUID(), role: 'user', text: buildSetupDiagnosisPrompt(opening) }];
          if (answer.text) answer.text += '\n\n';
          this.state.activity = 'Inspecting setup failure…'; this.publish();
          continue;
        }
        if (!ownsSetup || !this.options.openingProgress || !before || !opening) break;
        const workflow = graphSetupWorkflow(opening);
        const advancedToInputs = before.stages.python !== 'ready' && workflow.stages.python === 'ready' && workflow.nextStep.kind === 'inputs';
        const declined = this.state.permissions?.some(receipt => receipt.requestId === message.requestId && receipt.status === 'denied');
        if (!advancedToInputs || declined || continuations >= 4) break;
        // Resume only a proven host-stage transition. An unchanged input stage may
        // be the agent asking about user intent, and must never cause an auto retry.
        prepared = await this.options.prepare(message.target, message.nodeIds, message.draftId, controller.signal, intent, viewId);
        if (!current()) return;
        const verified = await this.options.openingProgress(message.target, controller.signal, 0);
        if (!current()) return;
        if (!verified || verified.id !== openingId || !this.samePermissionTarget(verified.target, message.target)) {
          this.state.status = 'idle'; this.state.activity = 'The model setup or source changed. Restart setup for the current model.'; return;
        }
        opening = verified;
        if (graphSetupWorkflow(opening).nextStep.kind !== 'inputs') break;
        history = [...structuredClone(this.state.messages), { id: randomUUID(), role: 'user', text: buildGraphSetupContinuationPrompt(workflow) }];
        continuations++;
        if (answer.text) answer.text += '\n\n';
        this.state.activity = 'Preparing model inputs…'; this.publish();
      }
      if (!current()) return;
      if (intent && !(opening ? graphSetupWorkflow(opening).nextStep.kind === 'complete' : this.options.openingComplete?.(message.target))) {
        this.state.status = opening?.phase === 'failed' ? 'error' : 'idle';
        this.state.failure = opening?.phase === 'failed' ? structuredClone(opening.failure) : undefined;
        const next = opening && graphSetupWorkflow(opening).nextStep;
        this.state.activity = next?.kind === 'execution_approval' ? 'Setup is ready for your execution approval.'
          : next?.kind === 'blocked' ? opening!.detail
            : 'Setup needs your input. Reply here to continue.';
        return;
      }
      this.state.status = 'idle'; this.state.activity = '';
    } catch (error) {
      if (!current()) return;
      this.state.status = 'error';
      this.state.activity = error instanceof Error ? error.message.slice(0, 2048) : 'The coding agent could not complete this request.';
      if (started) {
        this.options.invalidate?.();
        this.state.agents![message.provider] = { ...this.state.agents![message.provider], status: 'error', detail: 'The last request failed. Refresh to check the agent connection and sign-in.' };
      }
    } finally {
      if (this.permission?.controller === controller) this.settlePermission('cancelled');
      if (this.operationOwner === controller) this.operationOwner = undefined;
      if (current()) { this.active = undefined; this.publish(); }
      this.options.finishTurn?.(controller.signal);
    }
  }

  dispose(): void {
    this.disposed = true; this.active?.abort(); this.active = undefined;
    this.settlePermission('cancelled');
    this.cancelWarm();
    this.discovery?.abort();
    clearTimeout(this.publishTimer);
    this.views.clear(); this.options.clear();
  }
}
