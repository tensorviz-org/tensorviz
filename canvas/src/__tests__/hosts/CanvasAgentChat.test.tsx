import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { AGENT_COMPOSE_EVENT, CanvasAgentChat } from '../../ide/CanvasAgentChat';
import type { CanvasAgentAvailability, CanvasAgentState } from '../../ide/protocol.generated';
import type { LocalBrowserViewTransport } from '../../ide/localBrowserViewTransport';

const target = { workspaceFolder: '/project', file: 'model.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
const models = [
  { id: 'gpt-a', label: 'GPT A', description: 'Balanced', efforts: ['low', 'high'], defaultEffort: 'high' },
  { id: 'gpt-b', label: 'GPT B', description: 'Fast', efforts: [] },
];
const codex = (changes: Partial<CanvasAgentAvailability> = {}): CanvasAgentAvailability => ({ status: 'ready', detail: '', models, defaultModel: 'gpt-a', ...changes });
// The host names Claude's `default` alias by the model it resolves to.
const claudeModels = [
  { id: 'default', label: 'Opus 5.5', description: 'Opus 5.5 · Most capable for complex work', efforts: ['low', 'high'] },
  { id: 'sonnet', label: 'Sonnet 5.5', description: 'Sonnet 5.5 · Everyday tasks', efforts: [] },
];
const agents = (changes: Partial<CanvasAgentAvailability> = {}) => ({ codex: codex(changes),
  claude: { status: 'ready' as const, detail: '', models: claudeModels, defaultModel: 'default' } });

function fixture(props: Pick<React.ComponentProps<typeof CanvasAgentChat>, 'draftReview'> = {}) {
  let receive: (message: any) => void = () => {};
  let statusChanged: () => void = () => {};
  const status = { phase: 'connected', hostLink: 'connected' };
  let messageId = 0;
  // Like the transport, return the sent envelope with its messageId.
  const send = jest.fn((message: any) => ({ ...message, messageId: `message-${++messageId}` }));
  const transport = { session: { viewId: 'view' }, status, send,
    onMessage: (callback: (message: any) => void) => { receive = callback; return () => {}; },
    onStatusChange: (callback: () => void) => { statusChanged = callback; return () => {}; } } as unknown as LocalBrowserViewTransport;
  const view = render(<CanvasAgentChat transport={transport} target={target} connected {...props} />);
  const state = (changes: Partial<CanvasAgentState> = {}) => act(() => receive({ type: 'agentState', viewId: 'view',
    state: { sessionId: 'chat', revision: 1, provider: 'codex', agents: agents(), status: 'idle', activity: '', messages: [], ...changes } }));
  const sent = (action: string) => send.mock.calls.map(call => call[0]).filter(message => message.action === action);
  const link = (connected: boolean) => {
    status.phase = connected ? 'connected' : 'reconnecting';
    view.rerender(<CanvasAgentChat transport={transport} target={target} connected={connected} {...props} />);
    act(() => statusChanged());
  };
  const lastSyncId = () => send.mock.results.map(result => result.value).filter(message => message.action === 'sync').slice(-1)[0].messageId;
  const draftReview = (draft: typeof props.draftReview) => {
    props = { ...props, draftReview: draft };
    view.rerender(<CanvasAgentChat transport={transport} target={target} connected={status.phase === 'connected'} {...props} />);
  };
  return { send, state, sent, link, lastSyncId, draftReview };
}
const openChat = () => fireEvent.click(screen.getByRole('button', { name: /^Open agent chat/ }));
const settingsToggle = () => screen.getByRole('button', { name: /^Agent settings/ });
const openSettings = () => fireEvent.click(settingsToggle());
const settingsPanel = () => screen.queryByRole('group', { name: 'Agent settings' });
const sendButton = () => screen.getByRole('button', { name: 'Send request' });
const type = (text: string) => fireEvent.change(screen.getByRole('textbox'), { target: { value: text } });
const chooseModel = (model: string) => fireEvent.change(screen.getByRole('combobox', { name: 'Model' }), { target: { value: model } });

test('Use Agent opens progress without requiring a typed prompt or racing host discovery', () => {
  const { sent, state } = fixture();
  act(() => window.dispatchEvent(new CustomEvent(AGENT_COMPOSE_EVENT, { detail: { opening: true } })));
  expect(screen.getByRole('region', { name: 'Canvas chat' })).toBeVisible();
  expect(sent('discover')).toHaveLength(0);
  expect(sent('send')).toHaveLength(0);
  state({ status: 'running', requestId: 'automatic-opening', activity: 'Preparing your graph…',
    messages: [{ id: 'automatic-opening', role: 'user', text: 'Complete setup and open the graph.' }] });
  expect(screen.getByText('Preparing your graph…')).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Stop agent' }));
  expect(sent('stop')).toEqual([expect.objectContaining({ requestId: 'automatic-opening' })]);
});

test('starts as an icon; minimizing preserves the draft and never sends', () => {
  const { send } = fixture();
  expect(screen.queryByRole('textbox')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Open agent chat' }));
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Explain this layer' } });
  fireEvent.click(screen.getByRole('button', { name: 'Minimize agent chat' }));
  fireEvent.click(screen.getByRole('button', { name: 'Open agent chat' }));
  expect(screen.getByRole('textbox')).toHaveValue('Explain this layer');
  expect(send.mock.calls.filter((call: any[]) => call[0]?.action === 'send')).toHaveLength(0);
});

test('submits the current selection, keeps a refused draft, and does not auto-open on reply', () => {
  const { send, state } = fixture();
  fireEvent.click(screen.getByRole('button', { name: 'Open agent chat' }));
  state();
  act(() => window.dispatchEvent(new CustomEvent('tensorviz-selection', { detail: { nodeIds: ['activation'], nodeLabels: { activation: 'act' } } })));
  expect(screen.getByText('act')).toBeVisible();
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Explain this layer' } });
  fireEvent.click(screen.getByRole('button', { name: 'Send request' }));
  const request = (send.mock.calls as unknown as any[][]).find(call => call[0].action === 'send')![0];
  expect(request.nodeIds).toEqual(['activation']); expect(request.target).toEqual(target);
  state({ status: 'error', activity: 'Trust this workspace first.', ack: request.requestId });
  expect(screen.getByRole('textbox')).toHaveValue('Explain this layer');
  state({ revision: 2, status: 'running', ack: request.requestId, requestId: request.requestId, context: { target, nodeIds: ['activation'] },
    messages: [{ id: request.requestId, role: 'user', text: 'Explain this layer' }] });
  expect(screen.getByRole('textbox')).toHaveValue('');
  fireEvent.click(screen.getByRole('button', { name: 'Minimize agent chat' }));
  state({ revision: 3, messages: [{ id: 'reply', role: 'assistant', text: 'ReLU preserves the shape.' }] });
  expect(screen.queryByRole('textbox')).toBeNull();
  expect(screen.getByRole('status')).toHaveTextContent('Agent reply ready');
  fireEvent.click(screen.getByRole('button', { name: 'Open agent chat · new reply' }));
  expect(screen.getByText('ReLU preserves the shape.')).toBeVisible();
});

test('New chat confirms removal; Escape does not erase the draft', () => {
  const { send, state } = fixture();
  fireEvent.click(screen.getByRole('button', { name: 'Open agent chat' }));
  state({ messages: [{ id: 'u', role: 'user', text: 'Earlier question' }] });
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Keep this draft' } });
  fireEvent.click(screen.getByRole('button', { name: 'New agent chat' }));
  expect(screen.getByText('Clear this conversation, draft and pending proposals?')).toBeVisible();
  expect(send.mock.calls.filter((call: any[]) => call[0]?.action === 'clear')).toHaveLength(0);
  fireEvent.keyDown(screen.getByRole('button', { name: 'Cancel' }), { key: 'Escape' });
  expect(screen.queryByRole('alert')).toBeNull();
  expect(screen.getByRole('textbox')).toBeVisible();
  fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Escape' });
  fireEvent.click(screen.getByRole('button', { name: 'Open agent chat' }));
  expect(screen.getByRole('textbox')).toHaveValue('Keep this draft');
});

test('New chat appears only once there is a conversation, and clearing it returns focus to the prompt', () => {
  const { sent, state } = fixture();
  openChat();
  state();
  type('Only a draft');
  // A draft alone is not a conversation; no destructive control is offered for it.
  expect(screen.queryByRole('button', { name: 'New agent chat' })).toBeNull();
  state({ revision: 2, messages: [{ id: 'u', role: 'user', text: 'hi' }, { id: 'a', role: 'assistant', text: 'Hello.' }] });
  fireEvent.click(screen.getByRole('button', { name: 'New agent chat' }));
  expect(sent('clear')).toHaveLength(0);
  fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
  expect(sent('clear')).toHaveLength(1);
  expect(screen.getByRole('textbox')).toHaveValue('');
  // The confirmation and New chat both disappear, so focus must land somewhere useful.
  expect(screen.getByRole('textbox')).toHaveFocus();
  state({ revision: 3, messages: [] });
  expect(screen.queryByRole('button', { name: 'New agent chat' })).toBeNull();
});

test('a refused first request keeps New chat available to release its model context', () => {
  const { sent, state } = fixture();
  openChat();
  state({ status: 'error', context: { target, nodeIds: [] }, messages: [], activity: 'The source changed before the request started.' });
  type('Keep this draft until I confirm');
  fireEvent.click(screen.getByRole('button', { name: 'New agent chat' }));
  expect(sent('clear')).toHaveLength(0);
  fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
  expect(sent('clear')).toHaveLength(1);
  expect(screen.getByRole('textbox')).toHaveFocus();
});

test('checking keeps the idle layout quiet while describing why Send is disabled', () => {
  const { state } = fixture();
  openChat();
  state({ agents: agents({ status: 'checking' }) });
  type('Inspect the model');
  expect(screen.getByText('Checking…')).toBeVisible();
  expect(screen.getByRole('textbox')).toHaveAccessibleDescription('Checking Codex…');
  expect(sendButton()).toBeDisabled();
  expect(document.querySelector('.canvas-agent-readiness')).toBeNull();
});

test('the idle surface shows no boilerplate; a blocked agent’s settings show only the fix, not the account note', () => {
  fixture();
  openChat();
  expect(screen.getByRole('region', { name: 'Canvas chat' })).toBeVisible();
  expect(screen.queryByText(/Your account/)).toBeNull();
  openSettings();
  expect(within(settingsPanel()!).getByText('Codex hasn\'t been checked yet.')).toBeVisible();
  expect(within(settingsPanel()!).queryByText(/account/)).toBeNull();
});

test('completion is announced while chat stays open', () => {
  const { state } = fixture();
  fireEvent.click(screen.getByRole('button', { name: 'Open agent chat' }));
  state({ status: 'running', messages: [{ id: 'answer', role: 'assistant', text: 'Inspecting…' }] });
  state({ revision: 2, status: 'idle', messages: [{ id: 'answer', role: 'assistant', text: 'Ready for review.' }] });
  expect(screen.getByRole('status')).toHaveTextContent('Agent reply ready');
  openSettings();
  expect(screen.getByRole('combobox', { name: 'Agent' })).toBeDisabled();
  expect(screen.getByRole('combobox', { name: 'Agent' })).toHaveAccessibleDescription('Start a new chat to switch agents.');
});

test('discovers on open and on reconnect while open, never on collapsed mount or while minimized', () => {
  const { sent, state, link } = fixture();
  expect(sent('sync')).toHaveLength(1);
  expect(sent('discover')).toHaveLength(0);
  link(false); link(true);
  expect(sent('discover')).toHaveLength(0);
  openChat();
  expect(sent('discover')).toEqual([expect.objectContaining({ action: 'discover' })]);
  expect(sent('discover')[0].refresh).toBeUndefined();
  state({ agents: agents({ status: 'checking' }) });
  expect(screen.queryByRole('button', { name: 'Refresh' })).toBeNull();
  state({ revision: 2 });
  expect(screen.getByRole('status')).toHaveTextContent('Codex ready');
  link(false);
  expect(settingsToggle()).toHaveAccessibleName(/^Agent settings: Reconnecting, Codex/);
  link(true);
  expect(sent('sync')).toHaveLength(3);
  expect(sent('discover')).toHaveLength(2);
});

test.each([
  ['notInstalled', "Codex isn't installed where VS Code runs."],
  ['signInRequired', 'Sign in to Codex in a terminal, then refresh.'],
  ['error', "Couldn't check Codex. Refresh to try again."],
] as const)('%s blocks Send but keeps composing and offers an explicit refresh', (status, message) => {
  const { sent, state } = fixture();
  openChat();
  state({ agents: agents({ status, detail: 'Host detail for this check', models: [] }) });
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Still typing' } });
  expect(screen.getByRole('textbox')).toHaveValue('Still typing');
  expect(screen.getByRole('textbox')).toHaveAccessibleDescription(message);
  expect(sendButton()).toBeDisabled();
  expect(screen.getByText(message).parentElement).toHaveAttribute('title', 'Host detail for this check');
  expect(settingsToggle()).toHaveAccessibleName(/^Agent settings: Needs attention, Codex/);
  expect(screen.getByText('Needs attention')).toBeVisible();
  // The problem is explained outside the collapsed settings; they never open on their own.
  expect(settingsPanel()).toBeNull();
  fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });
  expect(sent('send')).toHaveLength(0);
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  expect(sent('discover').slice(-1)[0]).toEqual(expect.objectContaining({ refresh: true }));
  state({ revision: 2 });
  expect(sendButton()).toBeEnabled();
});

test('a remembered ready state never shows Ready without a host or before the host reports agents', () => {
  const { state, link } = fixture();
  openChat();
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Hi' } });
  state({ agents: undefined });
  expect(sendButton()).toBeDisabled();
  state({ revision: 2 });
  expect(sendButton()).toBeEnabled();
  link(false);
  expect(sendButton()).toBeDisabled();
  expect(screen.queryByText('Codex ready')).toBeNull();
});

test('model and effort edits configure the host, keep the draft, and Send uses the visible choice before the echo', () => {
  const { sent, state } = fixture();
  openChat();
  state({ selections: { codex: { model: 'gpt-a', effort: 'low' } } });
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Keep me' } });
  openSettings();
  expect(screen.getByRole('combobox', { name: 'Agent' })).toHaveFocus();
  expect(screen.getAllByRole('option').map(option => option.textContent)).toEqual(expect.arrayContaining(['Default (high)', 'low', 'high']));
  fireEvent.change(screen.getByRole('combobox', { name: 'Model' }), { target: { value: 'gpt-b' } });
  expect(sent('configure')).toEqual([expect.objectContaining({ provider: 'codex', settings: { model: 'gpt-b' } })]);
  // GPT B advertises no efforts, so the stale "low" is dropped rather than carried over.
  expect(screen.queryByRole('combobox', { name: 'Effort' })).toBeNull();
  expect(screen.getByRole('textbox')).toHaveValue('Keep me');
  // A background publication repeating an earlier ack is not this edit's acknowledgement.
  state({ revision: 2, ack: 'earlier', agents: agents(), selections: { codex: { model: 'gpt-a', effort: 'low' } } });
  expect(screen.getByRole('combobox', { name: 'Model' })).toHaveValue('gpt-b');
  fireEvent.click(sendButton());
  expect(sent('send')[0]).toEqual(expect.objectContaining({ provider: 'codex', settings: { model: 'gpt-b' } }));
});

test('changing model resets effort to that model’s default; agent default uses the advertised default model', () => {
  const { sent, state } = fixture();
  openChat();
  state({ selections: { codex: { model: 'gpt-b' } } });
  openSettings();
  expect(screen.queryByRole('combobox', { name: 'Effort' })).toBeNull();
  fireEvent.change(screen.getByRole('combobox', { name: 'Model' }), { target: { value: '' } });
  expect(sent('configure').slice(-1)[0].settings).toEqual({});
  const effort = screen.getByRole('combobox', { name: 'Effort' });
  expect(effort).toHaveValue('');
  fireEvent.change(effort, { target: { value: 'low' } });
  expect(sent('configure').slice(-1)[0].settings).toEqual({ effort: 'low' });
  expect(settingsToggle()).toHaveAccessibleName('Agent settings: Ready, Codex, Default (GPT A), low');
  // Choosing a model and then an effort keeps the settings open.
  expect(settingsPanel()).toBeVisible();
});

test('a saved model or effort the agent no longer offers stays visible and blocks Send until replaced', () => {
  const { sent, state } = fixture();
  openChat();
  state({ selections: { codex: { model: 'gpt-old' } } });
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Go' } });
  const reason = 'Saved model “gpt-old” isn\'t offered by Codex now. Choose another model to send.';
  expect(screen.getByText(reason)).toBeVisible();
  expect(sendButton()).toBeDisabled();
  openSettings();
  expect(screen.getByRole('combobox', { name: 'Model' })).toHaveValue('gpt-old');
  expect(screen.getByRole('option', { name: 'gpt-old (unavailable)' })).toBeInTheDocument();
  expect(sent('configure')).toHaveLength(0);
  fireEvent.change(screen.getByRole('combobox', { name: 'Model' }), { target: { value: 'gpt-a' } });
  expect(sendButton()).toBeEnabled();
  state({ revision: 2, ack: sent('configure')[0].requestId, selections: { codex: { model: 'gpt-a' } } });
  state({ revision: 3, selections: { codex: { model: 'gpt-a', effort: 'max' } } });
  expect(screen.getByRole('combobox', { name: 'Effort' })).toHaveValue('max');
  expect(sendButton()).toBeDisabled();
  expect(screen.getByRole('textbox')).toHaveAccessibleDescription('Saved effort “max” isn\'t available for GPT A. Choose another effort to send.');
});

test('provider can change only on an empty idle chat, and settings lock while running', () => {
  const { sent, state } = fixture();
  openChat();
  state({ selections: { claude: { model: 'sonnet' } } });
  openSettings();
  fireEvent.change(screen.getByRole('combobox', { name: 'Agent' }), { target: { value: 'claude' } });
  expect(sent('configure')).toEqual([expect.objectContaining({ provider: 'claude', settings: { model: 'sonnet' } })]);
  expect(settingsToggle()).toHaveAccessibleName('Agent settings: Ready, Claude Code, Sonnet 5.5');
  state({ revision: 2, ack: sent('configure')[0].requestId, provider: 'claude', status: 'running', selections: { claude: { model: 'sonnet' } }, messages: [{ id: 'u', role: 'user', text: 'hi' }] });
  for (const name of ['Agent', 'Model']) expect(screen.getByRole('combobox', { name })).toBeDisabled();
});

test('a configure lost to a disconnect is replayed on reconnect', () => {
  const { sent, state, link } = fixture();
  openChat();
  state();
  openSettings();
  fireEvent.change(screen.getByRole('combobox', { name: 'Model' }), { target: { value: 'gpt-b' } });
  link(false);
  expect(screen.getByRole('combobox', { name: 'Model' })).toBeDisabled();
  link(true);
  expect(sent('configure')).toHaveLength(2);
  expect(sent('configure')[1]).toEqual(sent('configure')[0]);
  expect(sent('configure')[1].requestId).toEqual(expect.any(String));
  state({ revision: 2, ack: sent('configure')[0].requestId, selections: { codex: { model: 'gpt-b' } } });
  link(false); link(true);
  expect(sent('configure')).toHaveLength(2);
});

test('Escape closes the settings panel before minimizing', () => {
  fixture();
  openChat();
  openSettings();
  fireEvent.keyDown(screen.getByRole('combobox', { name: 'Model' }), { key: 'Escape' });
  expect(screen.queryByRole('group', { name: 'Agent settings' })).toBeNull();
  expect(screen.getByRole('textbox')).toBeVisible();
  expect(screen.getByRole('button', { name: /^Agent settings/ })).toHaveAttribute('aria-expanded', 'false');
});

test('a stale settings acknowledgement or background update never ends a pending send', () => {
  const { sent, state } = fixture();
  openChat();
  state({ ack: 'mount' });
  openSettings();
  chooseModel('gpt-b');
  const configureId = sent('configure')[0].requestId;
  type('Go');
  fireEvent.click(sendButton());
  const sendId = sent('send')[0].requestId;
  // The host's reply to the earlier configure, then a discovery publication that repeats its ack.
  state({ revision: 2, ack: configureId, selections: { codex: { model: 'gpt-b' } } });
  state({ revision: 3, ack: configureId, selections: { codex: { model: 'gpt-b' } }, agents: agents() });
  expect(screen.getByRole('button', { name: 'Stop agent' })).toBeVisible();
  // Settings closed on Send; they can still be inspected while the request is pending, but not changed.
  expect(settingsPanel()).toBeNull();
  openSettings();
  expect(screen.getByRole('combobox', { name: 'Model' })).toBeDisabled();
  expect(screen.getByRole('combobox', { name: 'Model' })).toHaveValue('gpt-b');
  fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });
  expect(sent('send')).toHaveLength(1);
  state({ revision: 4, ack: sendId, status: 'error', activity: 'Wait for the agent check to finish before sending.', selections: { codex: { model: 'gpt-b' } } });
  expect(sendButton()).toBeEnabled();
  expect(screen.getByRole('textbox')).toHaveValue('Go');
});

test('an older settings acknowledgement never drops a newer local edit', () => {
  const { sent, state } = fixture();
  openChat();
  state({ ack: 'mount' });
  openSettings();
  chooseModel('gpt-b');
  chooseModel('gpt-a');
  const [first, second] = sent('configure').map(message => message.requestId);
  expect(first).not.toEqual(second);
  state({ revision: 2, ack: first, selections: { codex: { model: 'gpt-b' } } });
  expect(screen.getByRole('combobox', { name: 'Model' })).toHaveValue('gpt-a');
  state({ revision: 3, ack: second, selections: { codex: { model: 'gpt-a' } } });
  expect(screen.getByRole('combobox', { name: 'Model' })).toHaveValue('gpt-a');
  type('Go');
  fireEvent.click(sendButton());
  expect(sent('send')[0].settings).toEqual({ model: 'gpt-a' });
});

test('a refused settings change yields to the host, and Send uses the host’s choice', () => {
  const { sent, state } = fixture();
  openChat();
  state({ ack: 'mount', selections: { codex: { model: 'gpt-a' } } });
  openSettings();
  fireEvent.change(screen.getByRole('combobox', { name: 'Agent' }), { target: { value: 'claude' } });
  expect(settingsToggle()).toHaveAccessibleName(/Claude Code/);
  state({ revision: 2, ack: sent('configure')[0].requestId, activity: 'Start a new chat to switch coding agents.', selections: { codex: { model: 'gpt-a' } } });
  expect(screen.getByRole('combobox', { name: 'Agent' })).toHaveValue('codex');
  expect(screen.getByText('Start a new chat to switch coding agents.')).toBeVisible();
  type('Go');
  fireEvent.click(sendButton());
  expect(sent('send')[0]).toEqual(expect.objectContaining({ provider: 'codex', settings: { model: 'gpt-a' } }));
});

test('a send lost to a disconnect is released only by the first sync reply after reconnecting', () => {
  const { sent, state, link, lastSyncId } = fixture();
  openChat();
  state({ ack: 'mount' });
  type('Go');
  fireEvent.click(sendButton());
  link(false); link(true);
  state({ revision: 2, ack: 'mount' });
  expect(screen.getByRole('button', { name: 'Stop agent' })).toBeVisible();
  state({ revision: 2, ack: lastSyncId() });
  expect(sendButton()).toBeEnabled();
  expect(screen.getByRole('textbox')).toHaveValue('Go');
  expect(sent('send')).toHaveLength(1);
});

test('a delayed sync cannot release a send made after that sync was requested', () => {
  const { sent, state, link, lastSyncId } = fixture();
  openChat();
  state({ ack: 'mount' });
  link(false); link(true);
  const earlierSync = lastSyncId();
  type('Go');
  fireEvent.click(sendButton());
  state({ revision: 2, ack: earlierSync });
  expect(screen.getByRole('button', { name: 'Stop agent' })).toBeVisible();
  fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });
  expect(sent('send')).toHaveLength(1);
});

test('Claude’s default alias is offered once, as Default with its actual model and an on-demand description', () => {
  const { sent, state } = fixture();
  openChat();
  state({ provider: 'claude' });
  openSettings();
  const model = screen.getByRole('combobox', { name: 'Model' });
  expect(within(model).getAllByRole('option').map(option => option.textContent)).toEqual(['Default (Opus 5.5)', 'Sonnet 5.5']);
  expect(model).toHaveAccessibleDescription('Opus 5.5 · Most capable for complex work');
  expect(model).toHaveAttribute('title', 'Default (Opus 5.5) — Opus 5.5 · Most capable for complex work');
  expect(settingsToggle()).toHaveAccessibleName('Agent settings: Ready, Claude Code, Default (Opus 5.5), Default effort');
  chooseModel('sonnet');
  expect(model).toHaveAccessibleDescription('Sonnet 5.5 · Everyday tasks');
  // A model saved explicitly as the alias (here by another view) stays visible and selectable.
  state({ revision: 2, ack: sent('configure')[0].requestId, provider: 'claude', selections: { claude: { model: 'default' } } });
  expect(within(model).getAllByRole('option').map(option => option.textContent)).toEqual(['Default (Opus 5.5)', 'Opus 5.5', 'Sonnet 5.5']);
  expect(model).toHaveValue('default');
});

test('settings sit behind an icon-only toggle beside a short status; the full choice is in its name and tooltip only', () => {
  const { state, link } = fixture();
  openChat();
  state({ agents: agents({ status: 'checking' }) });
  expect(screen.getByText('Checking…')).toBeVisible();
  state({ revision: 2, provider: 'claude' });
  expect(screen.getByText('Ready')).toBeVisible();
  // A long default model name never appears as on-screen text while settings are collapsed.
  expect(settingsToggle()).toHaveTextContent(/^$/);
  expect(screen.queryByText(/Opus 5\.5/)).toBeNull();
  expect(settingsToggle()).toHaveAccessibleName('Agent settings: Ready, Claude Code, Default (Opus 5.5), Default effort');
  expect(settingsToggle()).toHaveAttribute('title', 'Agent settings · Ready · Claude Code · Default (Opus 5.5) · Default effort');
  link(false);
  expect(screen.getByText('Reconnecting')).toBeVisible();
});

test('settings start collapsed and open and close from the toggle', () => {
  const { state } = fixture();
  openChat();
  state();
  expect(settingsPanel()).toBeNull();
  expect(screen.queryByRole('combobox')).toBeNull();
  expect(settingsToggle()).toHaveAttribute('aria-expanded', 'false');
  openSettings();
  expect(settingsToggle()).toHaveAttribute('aria-expanded', 'true');
  expect(settingsToggle()).toHaveAttribute('aria-controls', settingsPanel()!.id);
  for (const name of ['Agent', 'Model', 'Effort']) expect(screen.getByRole('combobox', { name })).toBeVisible();
  openSettings();
  expect(settingsPanel()).toBeNull();
  expect(settingsToggle()).toHaveAttribute('aria-expanded', 'false');
  // Background publications and errors never open them.
  state({ revision: 2, status: 'error', activity: 'Agent failed.', agents: agents({ status: 'error' }) });
  expect(settingsPanel()).toBeNull();
});

test('minimizing and reopening collapses settings but keeps the draft and the chosen model', () => {
  const { state } = fixture();
  openChat();
  state({ ack: 'mount' });
  type('Keep me');
  openSettings();
  chooseModel('gpt-b');
  fireEvent.click(screen.getByRole('button', { name: 'Minimize agent chat' }));
  openChat();
  expect(settingsPanel()).toBeNull();
  expect(screen.getByRole('textbox')).toHaveValue('Keep me');
  expect(settingsToggle()).toHaveAccessibleName(/, GPT B$/);
  openSettings();
  expect(screen.getByRole('combobox', { name: 'Model' })).toHaveValue('gpt-b');
});

test('a dispatched Send collapses settings; a Send that cannot be dispatched leaves them open', () => {
  const { send, sent, state } = fixture();
  openChat();
  state({ ack: 'mount' });
  type('Go');
  openSettings();
  chooseModel('gpt-b');
  send.mockReturnValueOnce(null as any);
  fireEvent.click(sendButton());
  expect(settingsPanel()).toBeVisible();
  expect(screen.getByText('Canvas is reconnecting. Your draft is saved; try again when connected.')).toBeVisible();
  fireEvent.click(sendButton());
  expect(sent('send').slice(-1)[0].settings).toEqual({ model: 'gpt-b' });
  expect(settingsPanel()).toBeNull();
  expect(screen.getByRole('textbox')).toHaveFocus();
  // The draft stays until the host records the turn.
  expect(screen.getByRole('textbox')).toHaveValue('Go');
});

test('readiness and Refresh live in the open settings panel, and outside it only for a collapsed blocker', () => {
  const { sent, state } = fixture();
  openChat();
  state();
  expect(screen.queryByRole('button', { name: 'Refresh' })).toBeNull();
  openSettings();
  const panel = screen.getByRole('group', { name: 'Agent settings' });
  // Ready is not an access or quota guarantee; the one-line caption says so, its tooltip in full.
  const caption = within(panel).getByText('Your account · access and limits checked on Send');
  expect(caption).toBeVisible();
  expect(caption).toHaveAttribute('title', expect.stringMatching(/doesn’t guarantee model access or remaining limits.*Source edits open in VS Code for review/));
  expect(within(panel).queryByText(/Codex ready/)).toBeNull();
  expect(within(panel).getByRole('combobox', { name: 'Model' })).toHaveAccessibleDescription('Balanced');
  fireEvent.click(within(panel).getByRole('button', { name: 'Refresh' }));
  expect(sent('discover').slice(-1)[0]).toEqual(expect.objectContaining({ refresh: true }));
  state({ revision: 2, agents: agents({ status: 'signInRequired', models: [] }) });
  expect(screen.getAllByRole('button', { name: 'Refresh' })).toHaveLength(1);
  expect(within(screen.getByRole('group', { name: 'Agent settings' })).getByText('Sign in to Codex in a terminal, then refresh.')).toBeVisible();
  expect(screen.getByRole('textbox')).toHaveAccessibleDescription('Sign in to Codex in a terminal, then refresh.');
});

test('keyboard opening and closing settings keeps the panel and then its toggle in view', async () => {
  const original = Object.getOwnPropertyDescriptor(Element.prototype, 'scrollIntoView');
  const scrolled: Element[] = [];
  Object.defineProperty(Element.prototype, 'scrollIntoView', { configurable: true, value(this: Element) { scrolled.push(this); } });
  try {
    fixture();
    openChat();
    openSettings();
    expect(screen.getByRole('combobox', { name: 'Agent' })).toHaveFocus();
    expect(scrolled.slice(-1)[0]).toBe(screen.getByRole('group', { name: 'Agent settings' }));
    openSettings();
    await act(() => new Promise<void>(resolve => requestAnimationFrame(() => resolve())));
    expect(scrolled.slice(-1)[0]).toBe(settingsToggle());
    expect(settingsToggle()).toHaveFocus();
  } finally {
    if (original) Object.defineProperty(Element.prototype, 'scrollIntoView', original);
    else delete (Element.prototype as Partial<Element>).scrollIntoView;
  }
});

test('mouse settings inspection preserves typing focus and cannot silently change an agent', async () => {
  const { sent, state } = fixture();
  openChat(); state(); type('Keep writing');
  fireEvent.mouseDown(settingsToggle());
  fireEvent.click(settingsToggle(), { detail: 1 });
  expect(screen.getByRole('textbox')).toHaveFocus();
  expect(screen.getByRole('combobox', { name: 'Agent' })).not.toHaveFocus();
  expect(sent('configure')).toHaveLength(0);
  fireEvent.click(settingsToggle(), { detail: 1 });
  await act(() => new Promise<void>(resolve => requestAnimationFrame(() => resolve())));
  expect(screen.getByRole('textbox')).toHaveFocus();
  expect(screen.getByRole('textbox')).toHaveValue('Keep writing');
  fireEvent.click(settingsToggle(), { detail: 1 });
  fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Escape' });
  await act(() => new Promise<void>(resolve => requestAnimationFrame(() => resolve())));
  expect(settingsPanel()).toBeNull();
  expect(screen.getByRole('textbox')).toHaveFocus();
  expect(sent('configure')).toHaveLength(0);
});

test('Stop addresses the active request and returns focus to the next draft', () => {
  const { sent, state } = fixture();
  openChat();
  state({ status: 'running', requestId: 'active-turn', messages: [{ id: 'u', role: 'user', text: 'Explain' }] });
  screen.getByRole('button', { name: 'Stop agent' }).focus();
  fireEvent.click(screen.getByRole('button', { name: 'Stop agent' }));
  expect(sent('stop')).toEqual([expect.objectContaining({ requestId: 'active-turn' })]);
  expect(screen.getByRole('textbox')).toHaveFocus();
});

test('replies name the requested model and effort on the first reply and when they change', () => {
  const { state } = fixture();
  openChat();
  const turn = (id: string, role: 'user' | 'assistant', settings: { model?: string; effort?: string }) => ({ id, role, text: id, settings });
  state({ messages: [
    turn('u1', 'user', {}), turn('a1', 'assistant', {}),
    turn('u2', 'user', {}), turn('a2', 'assistant', {}),
    turn('u3', 'user', { model: 'gpt-a', effort: 'low' }), turn('a3', 'assistant', { model: 'gpt-a', effort: 'low' }),
  ] });
  // Defaults are labelled as requested, not resolved to today's default (GPT A).
  expect(screen.getAllByTitle('Model and effort requested for this reply').map(label => label.textContent))
    .toEqual([' · default model · default effort', ' · GPT A · low']);
});

describe('while a source draft is open', () => {
  const current = { id: 'proposal-1', stale: false };

  test('Enter revises the draft: the request names it, and nothing applies it', () => {
    const { send, sent, state } = fixture({ draftReview: current });
    openChat(); state();
    act(() => window.dispatchEvent(new CustomEvent('tensorviz-selection', { detail: { nodeIds: ['review:after:fc2'], nodeLabels: { 'review:after:fc2': 'fc2' } } })));
    expect(screen.getByRole('textbox')).toHaveAttribute('placeholder', 'Describe a revision to the draft…');
    type('add a sigmoid at end');
    expect(sendButton()).toBeEnabled();
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });
    expect(sent('send')).toEqual([expect.objectContaining({ prompt: 'add a sigmoid at end', draftId: 'proposal-1', target, nodeIds: ['review:after:fc2'] })]);
    // Revision is a send; there is no apply on the agent path.
    expect(send.mock.calls.map(([message]: any[]) => message.action).filter((action: string) => !['sync', 'discover', 'send'].includes(action))).toEqual([]);
  });

  test('Shift+Enter adds a line and an IME composition Enter only commits text', () => {
    const { sent, state } = fixture({ draftReview: current });
    openChat(); state();
    type('first line');
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter', shiftKey: true });
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter', isComposing: true });
    expect(sent('send')).toHaveLength(0);
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });
    expect(sent('send')).toHaveLength(1);
  });

  test('a stale draft blocks revision with a visible reason; a closed draft sends without a draft id', () => {
    const { sent, state, draftReview } = fixture({ draftReview: { id: 'proposal-1', stale: true } });
    openChat(); state();
    type('add dropout');
    expect(sendButton()).toBeDisabled();
    expect(screen.getByText('Source changed since this draft. Discard it to send a new request.')).toBeVisible();
    expect(screen.getByRole('textbox')).toHaveAccessibleDescription('Source changed since this draft. Discard it to send a new request.');
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });
    expect(sent('send')).toHaveLength(0);
    draftReview(undefined);
    expect(screen.queryByText(/Source changed since this draft/)).toBeNull();
    expect(screen.getByRole('textbox')).toHaveValue('add dropout');
    fireEvent.click(sendButton());
    expect(sent('send')).toHaveLength(1);
    expect(sent('send')[0]).not.toHaveProperty('draftId');
  });

  test('Continue editing opens the chat at the prompt, settings collapsed and text kept', () => {
    const { sent, state } = fixture({ draftReview: current });
    openChat(); state({ messages: [{ id: 'u', role: 'user', text: 'Add fc2' }, { id: 'a', role: 'assistant', text: 'Proposed fc2.' }] });
    type('add a sigmoid');
    openSettings();
    fireEvent.click(screen.getByRole('button', { name: 'Minimize agent chat' }));
    expect(screen.queryByRole('textbox')).toBeNull();
    const discovered = sent('discover').length;
    act(() => window.dispatchEvent(new CustomEvent(AGENT_COMPOSE_EVENT)));
    const input = screen.getByRole('textbox') as HTMLTextAreaElement;
    expect(input).toHaveValue('add a sigmoid');
    expect(input).toHaveFocus();
    expect(input.selectionStart).toBe('add a sigmoid'.length);
    expect(settingsPanel()).toBeNull();
    expect(screen.getByText('Proposed fc2.')).toBeVisible();
    expect(sent('discover')).toHaveLength(discovered + 1);
    // Already open: focus returns to the prompt from anywhere in the chat, with nothing sent.
    openSettings();
    act(() => window.dispatchEvent(new CustomEvent(AGENT_COMPOSE_EVENT)));
    expect(settingsPanel()).toBeNull();
    expect(screen.getByRole('textbox')).toHaveFocus();
    expect(screen.getByRole('textbox')).toHaveValue('add a sigmoid');
    expect(sent('send')).toHaveLength(0);
  });
});
