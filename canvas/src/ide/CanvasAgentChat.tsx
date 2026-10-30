import React from 'react';
import { useChat } from '@ai-sdk/react';
import { CanvasAgentChatTransport, agentUIMessages, type AgentUIMessage } from './canvasAgentChatTransport';
import { ArrowUp, ChevronDown, Copy, Crosshair, Ellipsis, LockKeyhole, MessageCircle, Minus, SlidersHorizontal, Square, SquarePen } from 'lucide-react';
import { newLocalBrowserMessageId, type CanvasAgentAction, type CanvasAgentAvailability, type CanvasAgentModel, type CanvasAgentProvider, type CanvasAgentSettings, type CanvasAgentState, type LocalBrowserTargetIdentity } from './protocol.generated';
import type { LocalBrowserViewTransport } from './localBrowserViewTransport';
import { bridgeActive, readClipboardText, writeClipboardText } from './bridge';
import './CanvasAgentChat.css';

interface Props {
  transport: LocalBrowserViewTransport | null;
  target: LocalBrowserTargetIdentity | null;
  connected: boolean;
  /** The pending source draft: Send revises it (never applies it); a stale draft cannot be revised. */
  draftReview?: { id: string; stale: boolean; canRevise?: boolean };
}

/** Window event that opens the chat at its prompt, keeping the typed text (e.g. Continue editing a draft). */
export const AGENT_COMPOSE_EVENT = 'tensorviz-agent-compose';

const AGENT_NAMES: Record<CanvasAgentProvider, string> = { codex: 'Codex', claude: 'Claude Code' };
/** A local edit stays visible (and is what Send uses) until the host acknowledges its requestId. */
type ConfigEdit = { provider: CanvasAgentProvider; settings: CanvasAgentSettings; requestId: string };
/** Only for hosts that predate `ack`: an exact selection echo stands in for the acknowledgement. */
const echoes = (edit: ConfigEdit, state: CanvasAgentState) => state.provider === edit.provider
  && (state.selections?.[edit.provider]?.model ?? '') === (edit.settings.model ?? '')
  && (state.selections?.[edit.provider]?.effort ?? '') === (edit.settings.effort ?? '');
function readinessText(status: CanvasAgentAvailability['status'], name: string) {
  switch (status) {
    case 'ready': return `${name} ready`;
    case 'checking': return `Checking ${name}…`;
    case 'notInstalled': return `${name} isn't installed where VS Code runs.`;
    case 'signInRequired': return `Sign in to ${name} in a terminal, then refresh.`;
    case 'error': return `Couldn't check ${name}. Refresh to try again.`;
    default: return `${name} hasn't been checked yet.`;
  }
}
/** What a turn asked for. Defaults stay unresolved: the agent's default then may not be its default now. */
function requestedLabel(settings: CanvasAgentSettings, models: CanvasAgentModel[]) {
  const model = !settings.model ? 'default model' : settings.model === 'default' ? 'recommended model'
    : models.find(entry => entry.id === settings.model)?.label ?? settings.model;
  return `${model} · ${settings.effort || 'default effort'}`;
}

/** The host owns execution, history, agent readiness and saved settings; open/minimized and the draft are view state. */
export function CanvasAgentChat({ transport, target, connected, draftReview }: Props) {
  const [open, setOpen] = React.useState(false);
  const [draft, setDraft] = React.useState('');
  const [edit, setEdit] = React.useState<ConfigEdit | null>(null);
  const [settingsOpen, setSettingsOpen] = React.useState(false);
  const [state, setState] = React.useState<CanvasAgentState | null>(null);
  const [selection, setSelection] = React.useState<string[]>([]);
  const [nodeLabels, setNodeLabels] = React.useState<Record<string, string>>({});
  const [pending, setPending] = React.useState<string | null>(null);
  const [unread, setUnread] = React.useState(false);
  const [announcementText, setAnnouncementText] = React.useState('');
  const providerHintId = React.useId();
  const settingsId = React.useId();
  const blockerId = React.useId();
  const draftBlockerId = React.useId();
  const [composeRequest, setComposeRequest] = React.useState(0);
  const [notice, setNotice] = React.useState<{ text: string; error?: boolean } | null>(null);
  const [confirmClear, setConfirmClear] = React.useState(false);
  const stateRef = React.useRef(state);
  const submitted = React.useRef<{ id: string; text: string; context: { target: LocalBrowserTargetIdentity; nodeIds: string[] } } | null>(null);
  const openRef = React.useRef(open); openRef.current = open;
  const editRef = React.useRef(edit); editRef.current = edit;
  const syncId = React.useRef<{ id: string; pendingId?: string } | null>(null);
  const input = React.useRef<HTMLTextAreaElement>(null);
  const newChat = React.useRef<HTMLButtonElement>(null);
  const bubble = React.useRef<HTMLButtonElement>(null);
  const settingsToggle = React.useRef<HTMLButtonElement>(null);
  const settingsPanel = React.useRef<HTMLDivElement>(null);
  const focusSettings = React.useRef(false);
  const history = React.useRef<HTMLDivElement>(null);
  const follow = React.useRef(true);
  const copyText = React.useCallback((text: string) => {
    void (bridgeActive() ? writeClipboardText(text) : navigator.clipboard.writeText(text)).then(() => {
      if (openRef.current) setNotice({ text: 'Copied.' });
    }, () => { if (openRef.current) setNotice({ text: 'Could not copy. Try again.', error: true }); });
  }, []);
  React.useEffect(() => {
    if (!open) return;
    const copySelection = (event: KeyboardEvent) => {
      if (!bridgeActive() || event.altKey || event.shiftKey || event.isComposing || !(event.metaKey || event.ctrlKey)
        || event.key.toLowerCase() !== 'c') return;
      const selection = window.getSelection();
      if (!selection?.anchorNode || !selection.focusNode || !history.current?.contains(selection.anchorNode)
        || !history.current.contains(selection.focusNode) || !selection.toString()) return;
      event.preventDefault(); event.stopPropagation(); copyText(selection.toString());
    };
    document.addEventListener('keydown', copySelection, true);
    return () => document.removeEventListener('keydown', copySelection, true);
  }, [open, copyText]);
  const sendAction = React.useCallback((action: CanvasAgentAction) => {
    const viewId = transport?.session?.viewId;
    return (viewId && transport?.send({ type: 'agentRequest', viewId, ...action })) || null;
  }, [transport]);
  const sendRef = React.useRef(sendAction); sendRef.current = sendAction;
  const chatTransport = React.useMemo(() => new CanvasAgentChatTransport(action => {
    const sent = !!sendRef.current(action);
    if (action.action === 'send') {
      if (sent) { setPending(action.requestId); follow.current = true; setSettingsOpen(false); input.current?.focus(); }
      else setNotice({ text: 'Canvas is reconnecting. Your draft is saved; try again when connected.' });
    }
    return sent;
  }), []);
  const chat = useChat<AgentUIMessage>({ transport: chatTransport, onError: error => {
    setPending(null); setNotice({ text: error.message, error: true });
    if (stateRef.current) setMessages(agentUIMessages(stateRef.current));
  } });
  const { setMessages } = chat;
  const running = state?.status === 'running' || pending !== null || chat.status === 'submitted' || chat.status === 'streaming';
  React.useEffect(() => {
    if (state && !chatTransport.streaming) setMessages(agentUIMessages(state));
  }, [state, chat.status, chatTransport, setMessages]);
  React.useEffect(() => () => chatTransport.disconnect(), [chatTransport]);

  React.useEffect(() => {
    if (!transport) return;
    const sync = () => {
      if (transport.status.phase !== 'connected' || transport.status.hostLink !== 'connected') return;
      setNotice(null);
      const request = sendAction({ action: 'sync' });
      syncId.current = request ? { id: request.messageId, pendingId: submitted.current?.id } : null;
      // A configure lost in the disconnect is replayed with its requestId, never silently dropped.
      const pendingEdit = editRef.current;
      if (pendingEdit) sendAction({ action: 'configure', ...pendingEdit });
      if (openRef.current) sendAction({ action: 'discover' });
    };
    const off = transport.onMessage(message => {
      if (message.type !== 'agentState' || message.viewId !== transport.session?.viewId) return;
      const previous = stateRef.current;
      if (previous?.sessionId === message.state.sessionId && previous.revision > message.state.revision) return;
      chatTransport.accept(message.state);
      if (!openRef.current && previous?.status === 'running' && message.state.status !== 'running') setUnread(true);
      if (previous?.status === 'running' && message.state.status !== 'running') {
        setAnnouncementText(message.state.status === 'error' ? 'Agent needs attention' : message.state.status === 'stopped' ? 'Agent stopped' : 'Agent reply ready');
      } else if (message.state.status === 'running' || !message.state.messages.length) setAnnouncementText('');
      const checked = message.state.agents?.[message.state.provider]?.status;
      // An interrupted check (Not checked) has no verdict to announce.
      if (openRef.current && previous?.agents?.[message.state.provider]?.status === 'checking' && checked && checked !== 'checking' && checked !== 'unchecked') {
        setAnnouncementText(checked === 'ready' ? `${AGENT_NAMES[message.state.provider]} ready` : `${AGENT_NAMES[message.state.provider]} needs attention`);
      }
      stateRef.current = message.state; setState(message.state);
      const sent = submitted.current;
      if (sent && message.state.messages.some(turn => turn.id === sent.id)) {
        setDraft(value => value.trim() === sent.text ? '' : value);
        submitted.current = null;
      }
      // Background publications repeat the last ack, so only an exact match settles a local request.
      // A newer edit outlives an older ack; an acknowledged edit yields to the host even if it was refused.
      const { ack, messages } = message.state;
      const resynced = ack && ack === syncId.current?.id ? syncId.current : null;
      if (resynced) {
        syncId.current = null;
        if (resynced.pendingId && !messages.some(turn => turn.id === resynced.pendingId)) chatTransport.disconnect();
      }
      setEdit(current => current && (ack ? ack === current.requestId : echoes(current, message.state)) ? null : current);
      setPending(current => current && (resynced?.pendingId === current || ack === current || messages.some(turn => turn.id === current)) ? null : current);
    });
    const offStatus = transport.onStatusChange(sync);
    sync();
    return () => { off(); offStatus(); };
  }, [transport, sendAction, chatTransport]);

  React.useEffect(() => {
    setSelection([]); setNodeLabels({});
    const onSelection = (event: Event) => {
      const detail = (event as CustomEvent).detail;
      const ids = detail?.nodeIds;
      if (Array.isArray(ids)) setSelection(ids.filter(id => typeof id === 'string'));
      if (detail?.nodeLabels && typeof detail.nodeLabels === 'object') {
        const labels = Object.fromEntries(Object.entries(detail.nodeLabels).filter(([, label]) => typeof label === 'string')) as Record<string, string>;
        setNodeLabels(previous => ({ ...previous, ...labels }));
      }
    };
    window.addEventListener('tensorviz-selection', onSelection);
    window.dispatchEvent(new CustomEvent('builder-command', { detail: { command: 'report-selection' } }));
    return () => window.removeEventListener('tensorviz-selection', onSelection);
  }, [target?.workspaceFolder, target?.file, target?.cls]);

  React.useEffect(() => {
    if (open && follow.current && history.current) history.current.scrollTop = history.current.scrollHeight;
  }, [open, state, settingsOpen]);
  React.useEffect(() => {
    const compose = (event: Event) => {
      if (!openRef.current && !(event as CustomEvent).detail?.opening) sendAction({ action: 'discover' });
      setOpen(true); setUnread(false); setSettingsOpen(false); setConfirmClear(false); setComposeRequest(count => count + 1);
    };
    window.addEventListener(AGENT_COMPOSE_EVENT, compose);
    return () => window.removeEventListener(AGENT_COMPOSE_EVENT, compose);
  }, [sendAction]);
  React.useEffect(() => { if (open) input.current?.focus(); }, [open]);
  // Continue where the text ends, not at its start.
  React.useEffect(() => {
    const field = input.current;
    if (composeRequest && field) { field.focus(); field.setSelectionRange(field.value.length, field.value.length); }
  }, [composeRequest]);
  React.useEffect(() => {
    if (!settingsOpen) return;
    if (focusSettings.current) settingsPanel.current?.querySelector<HTMLElement>('select:not(:disabled)')?.focus({ preventScroll: true });
    settingsPanel.current?.scrollIntoView?.({ block: 'nearest' });
  }, [settingsOpen]);

  // Everything below describes the visible configuration; Send uses exactly this.
  const provider = edit?.provider ?? state?.provider ?? 'codex';
  const settings = (edit ? edit.settings : state?.selections?.[provider]) ?? {};
  const name = AGENT_NAMES[provider];
  const agent = state?.agents?.[provider];
  const readiness = !connected ? null : agent?.status ?? 'unchecked';
  const ready = readiness === 'ready';
  const models = agent?.models ?? [];
  const selectedModel = models.find(model => model.id === settings.model);
  const effortModel = models.find(model => model.id === (settings.model ?? agent?.defaultModel));
  const defaultModel = models.find(model => model.id === agent?.defaultModel);
  // Claude's `default` alias is what the empty choice names, so it is listed only when explicitly saved.
  const listedModels = models.filter(model => provider !== 'claude' || model.id !== 'default' || settings.model === 'default');
  const defaultLabel = defaultModel ? `Default (${defaultModel.label})` : 'Default';
  const modelDescription = (settings.model ? selectedModel : defaultModel)?.description;
  const efforts = effortModel?.efforts ?? [];
  const missingModel = ready && !!settings.model && !selectedModel;
  const missingEffort = ready && !missingModel && !!settings.effort && !efforts.includes(settings.effort);
  const showEffort = efforts.length > 0 || !!settings.effort;
  const blocker = !readiness ? '' : !ready ? readinessText(readiness, name)
    : missingModel ? `Saved model “${settings.model}” isn't offered by ${name} now. Choose another model to send.`
    : missingEffort ? `Saved effort “${settings.effort}” isn't available for ${effortModel?.label ?? 'this model'}. Choose another effort to send.` : '';
  const draftBlocker = draftReview?.stale ? 'Source changed since this draft. Discard it to send a new request.' : '';
  const configDisabled = running || !connected;
  const configure = (change: Omit<ConfigEdit, 'requestId'>) => {
    const next = { ...change, requestId: newLocalBrowserMessageId() };
    if (sendAction({ action: 'configure', ...next })) { setEdit(next); setNotice(null); }
    else setNotice({ text: 'Canvas is reconnecting. Agent settings were not changed.', error: true });
  };

  const minimize = () => { setOpen(false); setSettingsOpen(false); requestAnimationFrame(() => bubble.current?.focus()); };
  const closeSettings = (destination: HTMLElement | null = settingsToggle.current) => {
    setSettingsOpen(false);
    requestAnimationFrame(() => {
      destination?.focus({ preventScroll: true }); destination?.scrollIntoView?.({ block: 'nearest' });
    });
  };
  const submit = () => {
    if (running || !draft.trim() || !target || !connected || selection.length > 100 || blocker || draftBlocker) return;
    const requestId = newLocalBrowserMessageId();
    setNotice(null);
    submitted.current = { id: requestId, text: draft.trim(), context: { target: { ...target }, nodeIds: [...selection] } };
    void chat.sendMessage({ id: requestId, parts: [{ type: 'text', text: draft.trim() }] }, { body: { action: {
      action: 'send', requestId, provider, prompt: draft.trim(), settings: { ...settings }, target, nodeIds: selection,
      ...(draftReview ? { draftId: draftReview.id } : {}),
    } } });
  };
  const pinned = pending ? submitted.current?.context : state?.status === 'running' ? state.context : undefined;
  const contextTarget = pinned?.target ?? target;
  const nodes = pinned?.nodeIds ?? selection;
  const context = !contextTarget ? 'No model' : nodes.length === 1 ? nodeLabels[nodes[0]] || 'Selected layer' : nodes.length ? `${nodes.length} selected layers` : contextTarget.cls;
  const unavailable = !connected ? 'Reconnecting to VS Code…' : !target ? 'Open a model to chat with your agent.' : selection.length > 100 ? 'Select at most 100 layers.' : '';
  const activity = notice?.text || (!connected ? '' : unavailable || state?.activity) || '';
  const activityError = notice ? !!notice.error : !unavailable && state?.status === 'error';
  const cancelClear = () => { setConfirmClear(false); requestAnimationFrame(() => newChat.current?.focus()); };
  const announcement = <span className="canvas-agent-announcement" role="status">{announcementText}</span>;
  const statusWord = !readiness ? 'Reconnecting' : readiness === 'checking' ? 'Checking…' : readiness === 'unchecked' ? 'Not checked'
    : blocker ? 'Needs attention' : 'Ready';
  const tone = statusWord === 'Ready' ? 'ready' : readiness === 'checking' ? 'checking' : statusWord === 'Needs attention' ? 'attention' : 'idle';
  const modelLabel = settings.model ? selectedModel?.label ?? settings.model : defaultLabel;
  const summary = [name, modelLabel, settings.effort || (showEffort ? 'Default effort' : '')].filter(Boolean);
  const readinessRow = readiness && <div className={`canvas-agent-readiness${tone === 'attention' ? ' canvas-agent-error' : ''}`} title={agent?.detail || undefined}>
    {/* One caption line; the full account/limits/review statement is in its tooltip. */}
    <span id={blockerId} title={blocker ? undefined : `Uses your ${name} account. Ready doesn’t guarantee model access or remaining limits; Send checks them. Source edits open in VS Code for review.`}>
      {blocker || 'Your account · access and limits checked on Send'}</span>
    {readiness !== 'checking' && <button type="button" className="canvas-agent-refresh" disabled={running} onClick={() => sendAction({ action: 'discover', refresh: true })}>Refresh</button>}
  </div>;
  // Each reply names the settings it requested, on the first reply and whenever they change.
  const turnModels = state ? state.agents?.[state.provider]?.models ?? [] : [];
  let lastRequested = '';
  const turns = chat.messages.map(message => ({ id: message.id, role: message.role,
    text: message.parts.map(part => part.type === 'text' ? part.text : '').join(''), settings: message.metadata?.settings,
  })).filter(message => message.text).map(message => {
    const requested = message.role === 'assistant' && message.settings ? requestedLabel(message.settings, turnModels) : '';
    const shown = requested !== lastRequested ? requested : '';
    if (requested) lastRequested = requested;
    return { message, requested: shown };
  });

  if (!open) return <>{announcement}<button ref={bubble} type="button" className="canvas-agent-bubble" aria-label={`Open agent chat${running ? ' · working' : unread ? state?.status === 'error' ? ' · needs attention' : ' · new reply' : ''}`}
    title={running ? 'Agent working' : unread ? 'New agent reply' : 'Chat with your coding agent'}
    onClick={() => { setOpen(true); setUnread(false); sendAction({ action: 'discover' }); }}>
    {running ? <Ellipsis size={22} /> : <MessageCircle size={21} />}
    {unread && <span className={`canvas-agent-unread${state?.status === 'error' ? ' canvas-agent-unread-error' : ''}`} />}
  </button></>;

  const hasHistory = !!state?.messages.length;
  return <>{announcement}<section className="canvas-agent-chat" aria-label="Canvas chat" onKeyDown={event => {
    event.stopPropagation();
    if (event.key === 'Escape' && !event.nativeEvent.isComposing) {
      if (confirmClear) cancelClear();
      else if (settingsOpen) closeSettings(settingsPanel.current?.contains(document.activeElement) || document.activeElement === settingsToggle.current ? settingsToggle.current : null);
      else minimize();
    }
  }}>
    {/* One surface: the bar names what the request is about and holds the window controls; sections open beneath it. */}
    <header className={`canvas-agent-bar${confirmClear || settingsOpen || hasHistory ? ' canvas-agent-bar-divided' : ''}`}>
      <div className="canvas-agent-context" title={`${contextTarget?.file ?? ''} · ${context} · ${pinned ? 'Request context pinned' : nodes.length ? 'Model + selection attached' : 'Model attached'}`}>
        {pinned ? <LockKeyhole size={13} /> : <Crosshair size={13} />}<span>{context}</span>
      </div>
      <span className={`canvas-agent-status canvas-agent-status-${tone}`}><span className={`canvas-agent-dot canvas-agent-dot-${tone}`} />{statusWord}</span>
      <button ref={settingsToggle} type="button" className="canvas-agent-settings-toggle" aria-expanded={settingsOpen} aria-controls={settingsOpen ? settingsId : undefined}
        aria-label={`Agent settings: ${[statusWord, ...summary].join(', ')}`} title={['Agent settings', statusWord, ...summary].join(' · ')}
        onMouseDown={event => event.preventDefault()}
        onClick={event => {
          if (settingsOpen) closeSettings(event.detail === 0 ? settingsToggle.current : input.current);
          else { focusSettings.current = event.detail === 0; setSettingsOpen(true); }
        }}><SlidersHorizontal size={16} /></button>
      {/* A refused first request can still bind the chat to a model; keep its reset reachable. */}
      {(hasHistory || state?.context) && <button ref={newChat} type="button" title="New chat" aria-label="New agent chat" disabled={running} onClick={() => setConfirmClear(true)}><SquarePen size={15} /></button>}
      <button type="button" title="Minimize chat" aria-label="Minimize agent chat" onClick={minimize}><Minus size={16} /></button>
    </header>
    {confirmClear && <div className="canvas-agent-clear" role="alert"><span>Clear this conversation, draft and pending proposals?</span><button type="button" onClick={() => {
      if (sendAction({ action: 'clear' })) { setDraft(''); setNotice(null); setConfirmClear(false); input.current?.focus(); }
    }}>Clear</button><button type="button" autoFocus onClick={cancelClear}>Cancel</button></div>}
    {/* Preference rows: each row is its native select; the label and chevron are drawn over it and let clicks through. */}
    {settingsOpen && <div ref={settingsPanel} id={settingsId} className="canvas-agent-settings" role="group" aria-label="Agent settings">
      <div className="canvas-agent-setting">
        <label htmlFor={`${settingsId}-agent`}>Agent</label>
        <select id={`${settingsId}-agent`} value={provider} disabled={configDisabled || !!state?.messages.length}
          aria-describedby={state?.messages.length ? providerHintId : undefined}
          onChange={event => {
            const next = event.target.value as CanvasAgentProvider;
            configure({ provider: next, settings: state?.selections?.[next] ?? {} });
          }}>
          <option value="codex">Codex</option><option value="claude">Claude Code</option>
        </select>
        <ChevronDown size={14} />
      </div>
      {!!state?.messages.length && <span id={providerHintId} className="canvas-agent-settings-hint">Start a new chat to switch agents.</span>}
      <div className="canvas-agent-setting">
        <label htmlFor={`${settingsId}-model`}>Model</label>
        {/* The description is on demand: the row's tooltip, and the select's accessible description. */}
        <select id={`${settingsId}-model`} value={settings.model ?? ''} disabled={configDisabled} aria-invalid={missingModel || undefined}
          title={modelDescription ? `${modelLabel} — ${modelDescription}` : modelLabel}
          aria-describedby={modelDescription ? `${settingsId}-model-description` : undefined}
          onChange={event => configure({ provider, settings: event.target.value ? { model: event.target.value } : {} })}>
          <option value="">{defaultLabel}</option>
          {settings.model && !selectedModel && <option value={settings.model}>{ready ? `${settings.model} (unavailable)` : settings.model}</option>}
          {listedModels.map(model => <option key={model.id} value={model.id} title={model.description}>{model.label}</option>)}
        </select>
        <ChevronDown size={14} />
        {modelDescription && <span id={`${settingsId}-model-description`} className="canvas-agent-announcement">{modelDescription}</span>}
      </div>
      {showEffort && <div className="canvas-agent-setting">
        <label htmlFor={`${settingsId}-effort`}>Effort</label>
        <select id={`${settingsId}-effort`} value={settings.effort ?? ''} disabled={configDisabled} aria-invalid={missingEffort || undefined}
          onChange={event => configure({ provider, settings: { ...(settings.model ? { model: settings.model } : {}), ...(event.target.value ? { effort: event.target.value } : {}) } })}>
          <option value="">{effortModel?.defaultEffort ? `Default (${effortModel.defaultEffort})` : 'Default'}</option>
          {settings.effort && !efforts.includes(settings.effort) && <option value={settings.effort}>{ready ? `${settings.effort} (unavailable)` : settings.effort}</option>}
          {efforts.map(effort => <option key={effort} value={effort}>{effort}</option>)}
        </select>
        <ChevronDown size={14} />
      </div>}
      {readinessRow}
    </div>}
    {!!state?.messages.length && <div ref={history} className="canvas-agent-history" role="log" tabIndex={0} aria-label="Agent conversation" aria-live={running ? 'off' : 'polite'} aria-busy={running}
      onScroll={() => { const el = history.current; if (el) follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 32; }}>
      {turns.map(({ message, requested }) => <div key={message.id} className={`canvas-agent-turn canvas-agent-${message.role}`}>
        <span>{message.role === 'user' ? 'You' : AGENT_NAMES[state.provider]}
          {requested && <span className="canvas-agent-turn-settings" title="Model and effort requested for this reply"> · {requested}</span>}</span>
        {message.role === 'assistant' && <button type="button" className="canvas-agent-copy" aria-label="Copy agent response"
          title="Copy response" onClick={() => copyText(message.text)}><Copy size={12} /></button>}
        <p>{message.text.replace(/\n{3,}/g, '\n\n')}</p>
      </div>)}
    </div>}
    {activity && (activityError ? <p key="error" className="canvas-agent-activity canvas-agent-error" role="alert">{activity}</p>
      : <p key="status" className="canvas-agent-activity" role="status" aria-live={running ? 'off' : 'polite'}>{activity}</p>)}
    {draftBlocker && <p id={draftBlockerId} className="canvas-agent-activity">{draftBlocker}</p>}
    {!settingsOpen && blocker && (readiness === 'checking'
      ? <span id={blockerId} className="canvas-agent-announcement">{blocker}</span>
      : readinessRow)}
    <div className={`canvas-agent-composer${hasHistory ? ' canvas-agent-composer-compact' : ''}`}>
      <textarea ref={input} aria-label="Message to coding agent" placeholder={draftReview?.canRevise === false ? 'Ask about this draft…' : draftReview ? 'Describe a revision to the draft…' : 'Ask or describe a change…'} value={draft} maxLength={16000} rows={hasHistory ? 2 : 3}
        aria-describedby={[blocker && blockerId, draftBlocker && draftBlockerId].filter(Boolean).join(' ') || undefined}
        onChange={event => setDraft(event.target.value)} onKeyDown={event => {
          if (bridgeActive() && (event.metaKey || event.ctrlKey) && !event.altKey && event.key.toLowerCase() === 'v') {
            event.preventDefault();
            const field = event.currentTarget;
            const { value, selectionStart: start, selectionEnd: end } = field;
            void readClipboardText().then(text => {
              // A delayed native reply belongs only to this still-focused, unchanged selection.
              if (input.current !== field || document.activeElement !== field || field.value !== value ||
                  field.selectionStart !== start || field.selectionEnd !== end) return;
              if (!text) return;
              const inserted = text.replace(/\r\n?/g, '\n').slice(0, Math.max(0, 16000 - value.length + end - start));
              if (!inserted) return;
              setNotice(null);
              // Use the textarea's native edit so caret, onChange and Undo/Redo stay intact.
              if (!document.execCommand('insertText', false, inserted)) setNotice({ text: 'Could not paste. Try again.', error: true });
            }, () => {
              if (input.current === field && document.activeElement === field && field.value === value &&
                  field.selectionStart === start && field.selectionEnd === end) setNotice({ text: 'Could not paste. Try again.', error: true });
            });
          }
          if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); submit(); }
        }} />
      {running ? <button type="button" className="canvas-agent-stop" aria-label="Stop agent" title="Stop agent" onClick={() => {
        const requestId = pending ?? state?.requestId;
        if (requestId) {
          if (sendAction({ action: 'stop', requestId })) input.current?.focus();
          else setNotice({ text: 'Could not stop while disconnected. Reconnect to check the agent and stop it.', error: true });
        }
      }}><Square size={11} fill="currentColor" /></button> : <button type="button" className="canvas-agent-send" aria-label="Send request" title={blocker || draftBlocker || (draftReview && draftReview.canRevise !== false ? 'Revise the draft' : 'Send request')} disabled={!draft.trim() || !!unavailable || !!blocker || !!draftBlocker} onClick={submit}><ArrowUp size={16} /></button>}
    </div>
  </section></>;
}
