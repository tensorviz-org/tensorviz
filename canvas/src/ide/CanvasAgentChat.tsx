import React from 'react';
import { useChat } from '@ai-sdk/react';
import { CanvasAgentChatTransport, agentUIMessages, type AgentUIMessage } from './canvasAgentChatTransport';
import { ArrowUp, Check, ChevronDown, Circle, Copy, Crosshair, LoaderCircle, LockKeyhole, MessageCircle, Minus, RefreshCw, Shield, ShieldCheck, ShieldQuestion, SlidersHorizontal, Square, SquarePen, TriangleAlert, WifiOff, X } from 'lucide-react';
import CanvasTooltip from '../components/CanvasTooltip';
import CanvasPopover from './CanvasPopover';
import { newLocalBrowserMessageId, type CanvasAgentAction, type CanvasAgentAvailability, type CanvasAgentModel, type CanvasAgentPermission, type CanvasAgentPermissionMode, type CanvasAgentProvider, type CanvasAgentSettings, type CanvasAgentState, type LocalBrowserTargetIdentity } from './protocol.generated';
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
type PermissionDecision = { messageId: string; sessionId: string; requestId: string };
type PermissionModeChoice = { messageId: string; sessionId: string; mode: CanvasAgentPermissionMode };
type ConnectionCheck = { messageId: string; sessionId: string; provider: CanvasAgentProvider; started: boolean };
const PERMISSION_MODES = [
  { value: 'ask', label: 'Ask every time', description: 'Review each environment change.', icon: ShieldQuestion },
  { value: 'auto_setup', label: 'Auto setup', description: 'Approve new isolated Python and its packages. Builds and existing environments still ask.', icon: ShieldCheck },
  { value: 'full_setup', label: 'Full setup access', description: 'Approve environment setup, including builds and version changes.', icon: Shield },
] as const;
const PERMISSION_MODE_REVIEW = 'Commands and source edits always ask.';

function PermissionModeOptions({ value, onChoose }: { value: CanvasAgentPermissionMode; onChoose(mode: CanvasAgentPermissionMode): void }) {
  const [hint, setHint] = React.useState(value);
  const id = React.useId();
  return <div className="local-canvas-menu-section canvas-agent-mode-options" onPointerLeave={() => setHint(value)}>
    {PERMISSION_MODES.map(mode => <React.Fragment key={mode.value}>
      <button type="button" role="menuitemradio" aria-label={mode.label} aria-checked={value === mode.value} aria-describedby={`${id}-${mode.value}`}
        className="local-canvas-menu-item" tabIndex={value === mode.value ? 0 : -1}
        onFocus={() => setHint(mode.value)} onPointerEnter={() => setHint(mode.value)} onClick={() => onChoose(mode.value)}>
        <mode.icon size={15} aria-hidden="true" /><span>{mode.label}</span>{value === mode.value && <Check size={14} aria-hidden="true" />}
      </button>
      <span id={`${id}-${mode.value}`} className="canvas-agent-announcement">{mode.description} {PERMISSION_MODE_REVIEW}</span>
    </React.Fragment>)}
    <div className="canvas-agent-mode-explanation" aria-hidden="true">
      <p>{PERMISSION_MODES.find(mode => mode.value === hint)?.description}</p>
      <p>{PERMISSION_MODE_REVIEW}</p>
    </div>
  </div>;
}
const pendingPermissions = (state: CanvasAgentState | null): CanvasAgentPermission[] => state?.status === 'running'
  ? state.permissions?.filter(permission => permission.status === 'pending' && permission.requestId === state.requestId) ?? [] : [];
/** A request needs an agent explanation before it becomes a user decision. */
const activePermissions = (state: CanvasAgentState | null): CanvasAgentPermission[] => pendingPermissions(state).filter(permission => !!permission.presentation);
const permissionDetails = ({ operation, scope, effects, facts }: CanvasAgentPermission): string => JSON.stringify({ operation, scope, effects, facts }, null, 2);
const permissionResult = (permission: CanvasAgentPermission) => permission.status === 'approved' ? permission.approval === 'policy' ? 'Auto-approved' : 'Allowed'
  : permission.status === 'denied' ? 'Denied' : permission.status === 'cancelled' ? 'Cancelled' : 'No longer active';
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
  const [connectionCheck, setConnectionCheck] = React.useState<'idle' | 'requested' | 'complete'>('idle');
  const connectionCheckRef = React.useRef<ConnectionCheck | null>(null);
  const providerHintId = React.useId();
  const settingsId = React.useId();
  const blockerId = React.useId();
  const taskDetailId = React.useId();
  const draftBlockerId = React.useId();
  const [composeRequest, setComposeRequest] = React.useState(0);
  const [notice, setNotice] = React.useState<{ text: string; error?: boolean } | null>(null);
  const [confirmClear, setConfirmClear] = React.useState(false);
  const [permissionDecisions, setPermissionDecisions] = React.useState<Record<string, PermissionDecision>>({});
  const permissionDecisionsRef = React.useRef(permissionDecisions);
  const [permissionFocus, setPermissionFocus] = React.useState<string | null>(null);
  const permissionHeading = React.useRef<HTMLHeadingElement>(null);
  const focusedPermission = React.useRef<{ id: string; node: HTMLHeadingElement } | null>(null);
  const permissionLabelId = React.useId();
  const [permissionModeChoice, setPermissionModeChoice] = React.useState<PermissionModeChoice | null>(null);
  const permissionModeChoiceRef = React.useRef(permissionModeChoice);
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
    if (!connected) { connectionCheckRef.current = null; setConnectionCheck('idle'); }
  }, [connected]);
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
      if (openRef.current && !activePermissions(stateRef.current).length) sendAction({ action: 'discover' });
    };
    const off = transport.onMessage(message => {
      if (message.type !== 'agentState' || message.viewId !== transport.session?.viewId) return;
      const previous = stateRef.current;
      if (previous?.sessionId === message.state.sessionId && previous.revision > message.state.revision) return;
      if (previous && (previous.sessionId !== message.state.sessionId || previous.provider !== message.state.provider)) {
        connectionCheckRef.current = null; setConnectionCheck('idle');
      }
      const recheck = connectionCheckRef.current;
      if (recheck) {
        if (recheck.sessionId !== message.state.sessionId || recheck.provider !== message.state.provider) {
          connectionCheckRef.current = null; setConnectionCheck('idle');
        } else {
          const availability = message.state.agents?.[recheck.provider]?.status;
          if (message.state.ack === recheck.messageId && availability === 'checking') recheck.started = true;
          else if (recheck.started && availability !== 'checking') {
            connectionCheckRef.current = null;
            setConnectionCheck(availability === 'ready' ? 'complete' : 'idle');
          }
        }
      }
      const nextPermissions = activePermissions(message.state);
      const newlyPending = nextPermissions.find(permission => previous?.sessionId !== message.state.sessionId
        || !activePermissions(previous).some(prior => prior.id === permission.id));
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
      const modeChoice = permissionModeChoiceRef.current;
      if (modeChoice && (modeChoice.sessionId !== message.state.sessionId || ack === modeChoice.messageId || resynced)) {
        permissionModeChoiceRef.current = null; setPermissionModeChoice(null);
        if (ack === modeChoice.messageId) {
          setAnnouncementText(`Setup permissions: ${PERMISSION_MODES.find(mode => mode.value === (message.state.permissionMode ?? 'ask'))?.label}`);
          if ((message.state.permissionMode ?? 'ask') !== modeChoice.mode && message.state.activity) setNotice({ text: message.state.activity, error: true });
        }
      }
      // Consent is never replayed on reconnect. A synced snapshot settles a lost local
      // dispatch; only a fresh user click may retry a still-pending permission.
      const decisions = Object.fromEntries(Object.entries(permissionDecisionsRef.current).filter(([id, decision]) =>
        decision.sessionId === message.state.sessionId && decision.requestId === message.state.requestId
        && nextPermissions.some(permission => permission.id === id) && ack !== decision.messageId && !resynced));
      permissionDecisionsRef.current = decisions; setPermissionDecisions(decisions);
      if (newlyPending) {
        setOpen(true); setSettingsOpen(false); setConfirmClear(false); setUnread(false); setNotice(null);
        follow.current = true; setPermissionFocus(newlyPending.id);
        setAnnouncementText(`Approval needed: ${newlyPending.presentation?.title}`);
      } else {
        const settled = message.state.permissions?.find(permission => permission.status !== 'pending'
          && (previous?.permissions?.some(prior => prior.id === permission.id && prior.status === 'pending')
            || (permission.approval === 'policy' && previous?.sessionId === message.state.sessionId
              && !previous.permissions?.some(prior => prior.id === permission.id))));
        if (settled) setAnnouncementText(`${settled.presentation?.title ?? settled.operation}: ${permissionResult(settled).toLowerCase()}`);
      }
      const requestedEdit = editRef.current;
      if (requestedEdit && ack === requestedEdit.requestId && !echoes(requestedEdit, message.state) && message.state.activity) {
        setNotice({ text: message.state.activity, error: true });
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
  React.useEffect(() => {
    const heading = permissionHeading.current;
    if (!open || !permissionFocus || !heading || (focusedPermission.current?.id === permissionFocus && focusedPermission.current.node === heading)) return;
    // The SDK can attach restored turns one render after the host's permission.
    // If that moves the card, focus its new neutral heading once, never Allow.
    heading.focus({ preventScroll: true }); heading.scrollIntoView?.({ block: 'nearest' });
    focusedPermission.current = { id: permissionFocus, node: heading };
  }, [open, permissionFocus, chat.messages]);
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
  const checkingConnection = readiness === 'checking' || connectionCheck === 'requested';
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
  const blocker = !readiness ? '' : checkingConnection ? readinessText('checking', name) : !ready ? readinessText(readiness, name)
    : missingModel ? `Saved model “${settings.model}” isn't offered by ${name} now. Choose another model to send.`
    : missingEffort ? `Saved effort “${settings.effort}” isn't available for ${effortModel?.label ?? 'this model'}. Choose another effort to send.` : '';
  const draftBlocker = draftReview?.stale ? 'Source changed since this draft. Discard it to send a new request.' : '';
  const configDisabled = running || !connected;
  const permissionMode = state?.permissionMode ?? 'ask';
  const permissionModeOption = PERMISSION_MODES.find(mode => mode.value === permissionMode) ?? PERMISSION_MODES[0];
  const PermissionModeIcon = permissionModeOption.icon;
  const choosePermissionMode = (mode: CanvasAgentPermissionMode) => {
    const live = stateRef.current;
    if (!connected || !live || permissionModeChoiceRef.current || mode === (live.permissionMode ?? 'ask')) return;
    const sent = sendAction({ action: 'permission_mode', mode });
    if (!sent) { setNotice({ text: 'Could not change setup permissions. Reconnect and try again.', error: true }); return; }
    const choice = { messageId: sent.messageId, sessionId: live.sessionId, mode };
    permissionModeChoiceRef.current = choice; setPermissionModeChoice(choice); setNotice(null);
  };
  const configure = (change: Omit<ConfigEdit, 'requestId'>) => {
    connectionCheckRef.current = null; setConnectionCheck('idle');
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
  const permissions = state?.permissions ?? [];
  const currentPermissions = activePermissions(state);
  const needsApproval = currentPermissions.length > 0;
  // Generic progress belongs to the task indicator. Keep only actionable notices
  // and real errors below the conversation; a finished reply must not look busy.
  const activity = notice?.text || (!connected ? '' : unavailable || (state?.status === 'error' ? state.activity : '')) || '';
  const activityError = notice ? !!notice.error : !unavailable && state?.status === 'error';
  const collapseError = activityError && state?.messages.some(message => message.role === 'assistant' && message.text.trim()) && !notice;
  const failureDiagnostic = connected && state?.status === 'error' && !notice ? state.failure?.diagnostic : undefined;
  const cancelClear = () => { setConfirmClear(false); requestAnimationFrame(() => newChat.current?.focus()); };
  const announcement = <span className="canvas-agent-announcement" role="status">{announcementText}</span>;
  const taskState = !connected ? 'disconnected' : needsApproval ? 'approval' : state?.status === 'error' || notice?.error ? 'error'
    : running ? 'running' : state?.status === 'stopped' ? 'stopped' : checkingConnection ? 'checking' : blocker ? 'unavailable' : 'idle';
  const taskLabels = { disconnected: 'Reconnecting to VS Code', approval: 'Waiting for your approval', error: 'Agent needs attention',
    running: 'Agent running', stopped: 'Agent stopped', checking: 'Checking agent connection', unavailable: 'Agent unavailable', idle: 'Agent idle' };
  const taskLabel = taskLabels[taskState];
  const taskDetail = taskState === 'approval' ? taskLabel : [taskLabel, connected && (notice?.error ? notice.text : state?.activity || blocker)].filter(Boolean).join(' · ');
  const TaskIcon = taskState === 'approval' ? ShieldCheck : taskState === 'running' || taskState === 'checking' ? LoaderCircle
    : taskState === 'error' || taskState === 'unavailable' ? TriangleAlert : taskState === 'disconnected' ? WifiOff : taskState === 'stopped' ? Square : Circle;
  const taskAnimated = taskState === 'running' || taskState === 'checking';
  const generationIndicator = taskAnimated && <span className="canvas-agent-generation" role="img" aria-label={taskLabel}
    aria-describedby={taskDetailId} title={taskDetail} tabIndex={0}>
    <span className="canvas-agent-generation-dots" aria-hidden="true"><i /><i /><i /><i /></span>
    <span id={taskDetailId} className="canvas-agent-announcement">{taskDetail}</span>
  </span>;
  // Provider readiness belongs to settings, independently of the task's outcome.
  const statusWord = !readiness ? 'Reconnecting' : checkingConnection ? 'Checking…' : readiness === 'unchecked' ? 'Not checked'
    : blocker ? 'Needs attention' : 'Ready';
  const tone = taskState === 'error' || taskState === 'unavailable' ? 'attention' : taskAnimated || needsApproval ? 'active' : 'idle';
  const modelLabel = settings.model ? selectedModel?.label ?? settings.model : defaultLabel;
  const summary = [name, modelLabel, settings.effort || (showEffort ? 'Default effort' : '')].filter(Boolean);
  const rechecking = checkingConnection;
  const readinessRow = readiness && <div className={`canvas-agent-readiness${blocker && !rechecking ? ' canvas-agent-error' : ''}`} title={agent?.detail || undefined}>
    <span id={blockerId}>
      {blocker || <CanvasTooltip text={`Uses your ${name} account. Model access and limits are checked when you send. Source edits open in VS Code for review.`}>
        <span tabIndex={0}>{connectionCheck === 'complete' ? 'Connection checked' : 'Connected'}</span>
      </CanvasTooltip>}</span>
    <CanvasTooltip text={rechecking ? 'Checking connection and available models…' : 'Recheck connection and available models'}>
      <button type="button" className="canvas-agent-refresh" aria-label="Recheck agent connection" disabled={running || !connected || rechecking}
        onClick={() => {
          if (!state) return;
          const sent = sendAction({ action: 'discover', refresh: true });
          if (!sent) { setNotice({ text: 'Could not recheck the connection. Reconnect and try again.', error: true }); return; }
          connectionCheckRef.current = { messageId: sent.messageId, sessionId: state.sessionId, provider, started: false };
          setConnectionCheck('requested');
        }}><RefreshCw size={14} className={rechecking ? 'canvas-agent-task-spin' : undefined} aria-hidden="true" /></button>
    </CanvasTooltip>
  </div>;
  // Each reply names the settings it requested, on the first reply and whenever they change.
  const turnModels = state ? state.agents?.[state.provider]?.models ?? [] : [];
  let lastRequested = '';
  const turns = chat.messages.map(message => ({ id: message.id, role: message.role,
    text: message.parts.map(part => part.type === 'text' ? part.text : '').join(''), settings: message.metadata?.settings,
  })).filter(message => message.text.trim()).map(message => {
    const requested = message.role === 'assistant' && message.settings ? requestedLabel(message.settings, turnModels) : '';
    const shown = requested !== lastRequested ? requested : '';
    if (requested) lastRequested = requested;
    return { message, requested: shown };
  });
  const decidePermission = (permission: CanvasAgentPermission, decision: 'allow' | 'deny') => {
    const live = stateRef.current;
    if (!connected || permissionDecisionsRef.current[permission.id] || !activePermissions(live).some(current => current.id === permission.id)) return;
    const sent = sendAction({ action: 'permission', permissionId: permission.id, decision });
    if (!sent || !live) { setNotice({ text: 'Could not send your decision. Reconnect and try again.', error: true }); return; }
    const decisions = { ...permissionDecisionsRef.current,
      [permission.id]: { messageId: sent.messageId, sessionId: live.sessionId, requestId: permission.requestId } };
    permissionDecisionsRef.current = decisions; setPermissionDecisions(decisions);
    setNotice(null);
  };
  const renderPermission = (permission: CanvasAgentPermission) => {
    const active = pendingPermissions(state).some(current => current.id === permission.id);
    const presentation = permission.presentation;
    const details = permissionDetails(permission);
    if (!active) return <details key={permission.id} className="canvas-agent-permission-receipt">
      <summary>{permission.status === 'approved' ? <Check size={12} aria-hidden="true" /> : <X size={12} aria-hidden="true" />}
        <span>{presentation?.title ?? permission.operation} · {permissionResult(permission)}</span></summary>
      {presentation && <p>{presentation.summary}</p>}
      {permission.approval === 'policy' && <p>Approved by {PERMISSION_MODES.find(mode => mode.value === permission.mode)?.label ?? 'this chat’s setup permissions'}.</p>}
      <pre>{details}</pre>
    </details>;
    const titleId = `${permissionLabelId}-${permission.id}-title`;
    const summaryId = `${permissionLabelId}-${permission.id}-summary`;
    const deciding = !!permissionDecisions[permission.id];
    return <section key={permission.id} className={`canvas-agent-permission${presentation ? '' : ' canvas-agent-permission-preparing'}`} aria-labelledby={titleId} aria-busy={!presentation}>
      <h3 ref={permissionFocus === permission.id && presentation ? permissionHeading : undefined} id={titleId} tabIndex={presentation ? -1 : undefined} aria-describedby={presentation ? summaryId : undefined}>
        <ShieldCheck size={14} aria-hidden="true" />{presentation?.title ?? 'Preparing approval…'}</h3>
      {presentation && <p id={summaryId}>{presentation.summary}</p>}
      <details><summary>Details</summary><pre>{details}</pre></details>
      {presentation && <div className="canvas-agent-permission-actions">
        <button type="button" className="canvas-agent-permission-allow" disabled={!connected || deciding} onClick={() => decidePermission(permission, 'allow')}>{presentation.allowLabel}</button>
        <button type="button" disabled={!connected || deciding} onClick={() => decidePermission(permission, 'deny')}>{presentation.denyLabel}</button>
        {deciding && <span role="status">Sending decision…</span>}
      </div>}
      {presentation && !connected && <p className="canvas-agent-permission-offline" role="status">Reconnect to respond. Nothing is approved while disconnected.</p>}
    </section>;
  };
  const turnPermissionIds = new Set(turns.filter(turn => turn.message.role === 'user').map(turn => turn.message.id));
  let permissionRequestId: string | undefined;

  if (!open) return <>{announcement}<button ref={bubble} type="button" className={`canvas-agent-bubble${needsApproval ? ' canvas-agent-bubble-approval' : ''}`} aria-label={`Open agent chat${needsApproval ? ' · needs approval' : running ? ' · working' : unread ? state?.status === 'error' ? ' · needs attention' : ' · new reply' : ''}`}
    title={needsApproval ? 'Your approval is needed' : running ? 'Agent working' : unread ? 'New agent reply' : 'Chat with your coding agent'}
    onClick={() => { setOpen(true); setUnread(false); if (!needsApproval) sendAction({ action: 'discover' }); }}>
    {needsApproval ? <ShieldCheck size={20} aria-hidden="true" /> : running ? <LoaderCircle size={20} className="canvas-agent-task-spin" aria-hidden="true" /> : <MessageCircle size={21} aria-hidden="true" />}
    {unread && !needsApproval && <span className={`canvas-agent-unread${state?.status === 'error' ? ' canvas-agent-unread-error' : ''}`} />}
  </button></>;

  const hasHistory = !!state?.messages.length || permissions.length > 0;
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
      {!taskAnimated && taskState !== 'idle' && <span className={`canvas-agent-status canvas-agent-status-${tone}`} data-state={taskState} role="img" aria-label={taskLabel} aria-describedby={taskDetailId} title={taskDetail} tabIndex={0}>
        <TaskIcon size={15} fill={taskState === 'stopped' ? 'currentColor' : 'none'} aria-hidden="true" />
        <span id={taskDetailId} className="canvas-agent-announcement">{taskDetail}</span>
      </span>}
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
      <div className="canvas-agent-setting" title={state?.messages.length ? 'Start a new chat to switch agents.' : undefined}>
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
      {!!state?.messages.length && <span id={providerHintId} className="canvas-agent-announcement">Start a new chat to switch agents.</span>}
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
    {hasHistory && <div ref={history} className="canvas-agent-history" role="log" tabIndex={0} aria-label="Agent conversation" aria-live={running ? 'off' : 'polite'} aria-busy={running && !needsApproval}
      onScroll={() => { const el = history.current; if (el) follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 32; }}>
      {turns.map(({ message, requested }, index) => {
        if (message.role === 'user') permissionRequestId = message.id;
        const turnPermissions = index === turns.length - 1 || turns[index + 1].message.role === 'user'
          ? permissions.filter(permission => permission.requestId === permissionRequestId) : [];
        return <React.Fragment key={message.id}><div className={`canvas-agent-turn canvas-agent-${message.role}`}>
        <span>{message.role === 'user' ? 'You' : AGENT_NAMES[state?.provider ?? provider]}
          {requested && <span className="canvas-agent-turn-settings" title="Model and effort requested for this reply"> · {requested}</span>}</span>
          {message.role === 'assistant' && <button type="button" className="canvas-agent-copy" aria-label="Copy agent response"
          title="Copy response" onClick={() => copyText(message.text)}><Copy size={12} /></button>}
        <p>{message.text.replace(/\n{3,}/g, '\n\n')}</p>
      </div>{turnPermissions.map(renderPermission)}</React.Fragment>; })}
      {permissions.filter(permission => !turnPermissionIds.has(permission.requestId)).map(renderPermission)}
      {generationIndicator}
    </div>}
    {!hasHistory && generationIndicator && <div className="canvas-agent-preparing">{generationIndicator}</div>}
    {activity && (activityError ? collapseError
      ? null
      : <p key="error" className="canvas-agent-activity canvas-agent-error" role="alert">{activity}</p>
      : <p key="status" className="canvas-agent-activity" role="status" aria-live={running ? 'off' : 'polite'}>{activity}</p>)}
    {((collapseError && activity) || failureDiagnostic) && <details className="canvas-agent-issue">
      <summary><TriangleAlert size={13} aria-hidden="true" />Error details</summary>
      {collapseError && activity && <p role="alert">{activity}</p>}
      {failureDiagnostic && <div className="canvas-agent-diagnostic">
        <h4>Traceback</h4><pre tabIndex={0} aria-label="Setup traceback">{failureDiagnostic.traceback}</pre>
        <h4>Observed runtime</h4><pre tabIndex={0} aria-label="Observed setup runtime">{JSON.stringify(failureDiagnostic.runtime, null, 2)}</pre>
        {failureDiagnostic.truncated && <p>Evidence truncated.</p>}
      </div>}
    </details>}
    {draftBlocker && <p id={draftBlockerId} className="canvas-agent-activity">{draftBlocker}</p>}
    {!settingsOpen && blocker && (checkingConnection
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
      <div className="canvas-agent-composer-tools">
        <CanvasPopover label={`Setup permissions: ${permissionModeOption.label}`} title="Setup permissions" role="menu" width={252}
          help={permissionModeChoice ? 'Updating permissions…' : `${permissionModeOption.label}. ${permissionModeOption.description} ${PERMISSION_MODE_REVIEW} This chat only; new chats reset to Ask every time.`}
          className={`canvas-agent-mode-toggle canvas-agent-mode-${permissionMode}`} panelClassName="canvas-agent-mode-menu"
          disabled={!connected || !state || !!permissionModeChoice}
          trigger={permissionModeChoice ? <LoaderCircle size={15} className="canvas-agent-task-spin" aria-hidden="true" /> : <PermissionModeIcon size={16} aria-hidden="true" />}>
          {close => <PermissionModeOptions value={permissionMode} onChoose={mode => { choosePermissionMode(mode); close(); }} />}
        </CanvasPopover>
        <span className="canvas-agent-composer-space" />
        {running ? <button type="button" className="canvas-agent-stop" aria-label="Stop agent" title="Stop agent" onClick={() => {
        const requestId = pending ?? state?.requestId;
        if (requestId) {
          if (sendAction({ action: 'stop', requestId })) input.current?.focus();
          else setNotice({ text: 'Could not stop while disconnected. Reconnect to check the agent and stop it.', error: true });
        }
        }}><Square size={11} fill="currentColor" /></button> : <button type="button" className="canvas-agent-send" aria-label="Send request" title={blocker || draftBlocker || (draftReview && draftReview.canRevise !== false ? 'Revise the draft' : 'Send request')} disabled={!draft.trim() || !!unavailable || !!blocker || !!draftBlocker} onClick={submit}><ArrowUp size={16} /></button>}
      </div>
      {permissionModeChoice && <span className="canvas-agent-announcement" role="status">Updating permissions…</span>}
    </div>
  </section></>;
}
