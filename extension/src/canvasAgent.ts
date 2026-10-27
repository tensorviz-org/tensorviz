import { randomUUID } from 'node:crypto';
import { isCanvasAgentSettings, type CanvasAgentAction, type GraphOpeningState, type CanvasAgentAvailability, type CanvasAgentProvider, type CanvasAgentSelections,
  type CanvasAgentSettings, type CanvasAgentState, type CanvasAgentTurn, type LocalBrowserAgentRequest, type LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import { validateAgentSettings } from './canvasAgentRuntime';

export interface AgentRunInput {
  provider: 'codex' | 'claude';
  cwd: string;
  context: string;
  messages: CanvasAgentTurn[];
  settings?: CanvasAgentSettings;
  intent?: 'open_graph';
}
export type AgentRunEvent = { kind: 'text'; text: string } | { kind: 'activity'; text: string };
export interface CanvasAgentOptions {
  prepare(target: LocalBrowserTargetIdentity, nodeIds: string[], draftId: string | undefined, signal: AbortSignal, intent?: 'open_graph'): Promise<{ cwd: string; context: string }>;
  openingPending?(target: LocalBrowserTargetIdentity): boolean;
  openingComplete?(target: LocalBrowserTargetIdentity): boolean;
  cancelOpening?(target: LocalBrowserTargetIdentity): void;
  run(input: AgentRunInput, emit: (event: AgentRunEvent) => void, signal: AbortSignal): Promise<void>;
  publish(viewId: string, state: CanvasAgentState): void;
  clear(): void;
  inspect?(provider: CanvasAgentProvider, refresh: boolean, signal: AbortSignal): Promise<CanvasAgentAvailability>;
  preferences?: { provider: CanvasAgentProvider; selections: CanvasAgentSelections };
  remember?(preferences: { provider: CanvasAgentProvider; selections: CanvasAgentSelections }): void;
  invalidate?(): void;
  finishTurn?(signal: AbortSignal): void;
}

/** One window-owned conversation. Hiding a view never cancels its work. */
export class CanvasAgent {
  private state: CanvasAgentState = this.empty();
  private readonly views = new Set<string>();
  private readonly submitted = new Set<string>();
  private setupTarget?: LocalBrowserTargetIdentity;
  private active?: AbortController;
  private publishTimer?: NodeJS.Timeout;
  private disposed = false;
  private discovery?: AbortController;
  private hasPreference = false;

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
      status: 'idle', messages: [], activity: '' };
  }

  private publish(): void {
    clearTimeout(this.publishTimer); this.publishTimer = undefined;
    this.state.revision++;
    for (const view of this.views) this.options.publish(view, structuredClone(this.state));
  }

  disconnect(viewId: string): void { this.views.delete(viewId); }

  /** The setup button starts a host-authored task, using this conversation's saved agent settings. */
  openGraph(opening: GraphOpeningState): void {
    if (this.disposed || this.active) return;
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
      prompt: `Set up and open the captured graph for ${opening.target.cls} in ${opening.target.file}, preserving my current edited values. Explain your recommended setup and ask me about missing intent when it matters. Save a complete setup; saving runs it once to capture the graph, so do not start other runs.` }, 'open_graph');
  }

  handle(message: LocalBrowserAgentRequest): void {
    if (this.disposed) return;
    this.views.add(message.viewId);
    // Every accepted or refused action acknowledges its own ID. Background state
    // publications must never be mistaken for acknowledgement of a later action.
    this.state.ack = 'requestId' in message && message.requestId ? message.requestId : message.messageId;
    if (message.action === 'sync') { this.options.publish(message.viewId, structuredClone(this.state)); return; }
    if (message.action === 'discover') { if (!this.active) void this.discover(!!message.refresh); return; }
    if (message.action === 'stop') {
      if (message.requestId === this.state.requestId && this.active) {
        this.active.abort(); this.active = undefined;
        this.options.clear();
        this.state.status = 'stopped'; this.state.activity = 'Stopped. Pending agent work was cancelled.';
        this.publish();
      }
      return;
    }
    if (this.active) { this.options.publish(message.viewId, structuredClone(this.state)); return; }
    if (message.action === 'configure') {
      if (this.state.messages.length && message.provider !== this.state.provider) {
        this.state.activity = 'Start a new chat to switch coding agents.'; this.publish(); return;
      }
      this.hasPreference = true; this.state.provider = message.provider;
      this.state.selections = { ...this.state.selections, [message.provider]: { ...message.settings } };
      this.remember(); this.publish(); return;
    }
    if (message.action === 'clear') {
      this.options.clear();
      const { provider, selections, agents, ack } = this.state;
      this.state = { ...this.empty(), provider, selections, agents, ack };
      this.submitted.clear(); this.setupTarget = undefined; this.publish(); return;
    }
    if (this.discovery && this.state.agents?.[message.provider].status === 'checking') {
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
    void this.send(message, followUp ? 'open_graph' : undefined);
  }

  private remember(): void {
    this.options.remember?.({ provider: this.state.provider, selections: structuredClone(this.state.selections ?? {}) });
  }

  private async discover(refresh: boolean): Promise<void> {
    if (!this.options.inspect || this.discovery) return;
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
        if (current()) { this.state.agents![provider] = checked; this.publish(); }
      }));
      if (current() && !this.active && !this.hasPreference && !this.state.messages.length) {
        const ready = (['codex', 'claude'] as const).filter(provider => this.state.agents![provider].status === 'ready');
        if (ready.length === 1) this.state.provider = ready[0];
      }
    } finally {
      clearTimeout(timer);
      if (this.discovery === controller) {
        this.discovery = undefined;
        for (const provider of ['codex', 'claude'] as const) if (this.state.agents![provider].status === 'checking') {
          this.state.agents![provider] = { status: 'error', detail: 'The check timed out. Refresh to try again.', models: [] };
        }
        if (!this.disposed) this.publish();
      }
    }
  }

  private async send(message: Extract<CanvasAgentAction, { action: 'send' }>, intent?: 'open_graph'): Promise<void> {
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
      this.state.status = 'running'; this.state.activity = 'Connecting to your coding agent…';
      this.state.context = { target: { ...message.target }, nodeIds: [...message.nodeIds] };
      this.publish();
      let prepared = await this.options.prepare(message.target, message.nodeIds, message.draftId, controller.signal, intent);
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
        const model = checked.models.find(model => model.id === settings.model);
        if (settings.model && !settings.effort && model?.defaultEffort) runSettings = { ...settings, effort: model.defaultEffort };
      }
      if (intent) prepared = await this.options.prepare(message.target, message.nodeIds, message.draftId, controller.signal, intent);
      if (!current()) return;
      this.hasPreference = true;
      this.state.selections = { ...this.state.selections, [message.provider]: settings }; this.remember();
      this.state.messages.push({ id: message.requestId, role: 'user', text: message.prompt.trim(), settings });
      const history = structuredClone(this.state.messages);
      const answer: CanvasAgentTurn = { id: randomUUID(), role: 'assistant', text: '', settings };
      this.state.messages.push(answer); this.publish();
      started = true;
      await this.options.run({ provider: message.provider, ...prepared, settings: runSettings, messages: history, ...(intent ? { intent } : {}) }, event => {
        if (!current()) return;
        if (event.kind === 'text') answer.text = (answer.text + event.text).slice(0, 32000);
        else this.state.activity = event.text.slice(0, 2048);
        if (event.kind === 'activity') this.publish();
        else if (!this.publishTimer) this.publishTimer = setTimeout(() => this.publish(), 80);
      }, controller.signal);
      if (!current()) return;
      if (intent && !this.options.openingComplete?.(message.target)) {
        this.state.status = 'idle';
        this.state.activity = 'Setup needs your input. Check the agent reply and continue the conversation or return to model setup.';
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
      if (current()) { this.active = undefined; this.publish(); }
      this.options.finishTurn?.(controller.signal);
    }
  }

  dispose(): void {
    this.disposed = true; this.active?.abort(); this.active = undefined;
    this.discovery?.abort();
    clearTimeout(this.publishTimer);
    this.views.clear(); this.options.clear();
  }
}
