import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { AGENT_COMPOSE_EVENT, CanvasAgentChat } from '../../ide/CanvasAgentChat';
import type { CanvasAgentAvailability, CanvasAgentPermission, CanvasAgentState } from '../../ide/protocol.generated';
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
  return { send, state, sent, link, lastSyncId, draftReview, unmount: view.unmount };
}
const openChat = () => fireEvent.click(screen.getByRole('button', { name: /^Open agent chat/ }));
const settingsToggle = () => screen.getByRole('button', { name: /^Agent settings/ });
const openSettings = () => fireEvent.click(settingsToggle());
const settingsPanel = () => screen.queryByRole('group', { name: 'Agent settings' });
const sendButton = () => screen.getByRole('button', { name: 'Send request' });
const recheckButton = () => screen.getByRole('button', { name: 'Recheck agent connection' });
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
  expect(screen.getByRole('img', { name: 'Agent running' })).toHaveAttribute('title', 'Agent running · Preparing your graph…');
  expect(screen.queryByText('Preparing your graph…')).toBeNull();
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

test('submits the current selection, keeps a refused draft, and does not auto-open on reply', async () => {
  const { send, state } = fixture();
  fireEvent.click(screen.getByRole('button', { name: 'Open agent chat' }));
  state();
  act(() => window.dispatchEvent(new CustomEvent('tensorviz-selection', { detail: { nodeIds: ['activation'], nodeLabels: { activation: 'act' } } })));
  expect(screen.getByText('act')).toBeVisible();
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Explain this layer' } });
  fireEvent.click(screen.getByRole('button', { name: 'Send request' }));
  const request = (send.mock.calls as unknown as any[][]).find(call => call[0].action === 'send')![0];
  expect(request.nodeIds).toEqual(['activation']); expect(request.target).toEqual(target);
  await act(async () => { state({ status: 'error', activity: 'Trust this workspace first.', ack: request.requestId }); });
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
  expect(screen.getByRole('img', { name: 'Checking agent connection' })).toBeVisible();
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
  expect(screen.getByRole('combobox', { name: 'Agent' }).parentElement).toHaveAttribute('title', 'Start a new chat to switch agents.');
  expect(screen.getByText('Start a new chat to switch agents.')).toHaveClass('canvas-agent-announcement');
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
  expect(screen.queryByRole('button', { name: 'Recheck agent connection' })).toBeNull();
  state({ revision: 2 });
  expect(screen.getByRole('status')).toHaveTextContent('Codex ready');
  link(false);
  expect(settingsToggle()).toHaveAccessibleName(/^Agent settings: Reconnecting, Codex/);
  link(true);
  expect(sent('sync')).toHaveLength(3);
  expect(sent('discover')).toHaveLength(2);
});

test('a check that ends without a verdict is not announced as a problem; a failed check is', () => {
  const { state } = fixture();
  openChat();
  state({ agents: agents({ status: 'checking' }) });
  state({ revision: 2, agents: agents({ status: 'signInRequired', detail: 'Not signed in', models: [] }) });
  expect(screen.getByRole('status')).toHaveTextContent('Codex needs attention');
  state({ revision: 3, agents: agents({ status: 'checking' }) });
  // Use Agent interrupts the check and its setup task checks the agent again; the visible status is Not checked.
  state({ revision: 4, status: 'running', requestId: 'setup', activity: 'Preparing your request…',
    agents: agents({ status: 'unchecked', detail: 'The check was interrupted. Refresh to check the agent.', models: [] }) });
  expect(screen.queryByText('Codex needs attention')).toBeNull();
  expect(screen.getByRole('img', { name: 'Agent running' })).toBeVisible();
  expect(screen.queryByText('Not checked')).toBeNull();
});

test.each([
  ['notInstalled', "Codex isn't installed where VS Code runs."],
  ['signInRequired', 'Sign in to Codex in a terminal, then refresh.'],
  ['error', "Couldn't check Codex. Refresh to try again."],
] as const)('%s blocks Send but keeps composing and offers an explicit refresh', (status, message) => {
  const { sent, state, send } = fixture();
  openChat();
  state({ agents: agents({ status, detail: 'Host detail for this check', models: [] }) });
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Still typing' } });
  expect(screen.getByRole('textbox')).toHaveValue('Still typing');
  expect(screen.getByRole('textbox')).toHaveAccessibleDescription(message);
  expect(sendButton()).toBeDisabled();
  expect(screen.getByText(message).parentElement).toHaveAttribute('title', 'Host detail for this check');
  expect(settingsToggle()).toHaveAccessibleName(/^Agent settings: Needs attention, Codex/);
  expect(screen.getByRole('img', { name: 'Agent unavailable' })).toBeVisible();
  // The problem is explained outside the collapsed settings; they never open on their own.
  expect(settingsPanel()).toBeNull();
  fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });
  expect(sent('send')).toHaveLength(0);
  fireEvent.click(screen.getByRole('button', { name: 'Recheck agent connection' }));
  expect(sent('discover').slice(-1)[0]).toEqual(expect.objectContaining({ refresh: true }));
  const ack = send.mock.results.at(-1)!.value.messageId;
  state({ revision: 2, ack, agents: agents({ status: 'checking' }) });
  state({ revision: 3, ack });
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

test.each(['codex', 'claude'] as const)('%s shows the host medium default without replacing an explicit effort', provider => {
  const { state } = fixture();
  const available = agents();
  available[provider] = { ...available[provider], defaultModel: 'medium-model',
    models: [{ id: 'medium-model', label: 'Medium model', description: '', efforts: ['medium', 'high'], defaultEffort: 'medium' }] };
  openChat(); state({ provider, agents: available }); openSettings();
  const effort = screen.getByRole('combobox', { name: 'Effort' });
  expect(effort).toHaveValue('');
  expect(within(effort).getByRole('option', { name: 'Default (medium)' })).toBeVisible();
  state({ revision: 2, provider, agents: available, selections: { [provider]: { effort: 'high' } } });
  expect(effort).toHaveValue('high');
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

test('a stale settings acknowledgement or background update never ends a pending send', async () => {
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
  await act(async () => { state({ revision: 4, ack: sendId, status: 'error', activity: 'Wait for the agent check to finish before sending.', selections: { codex: { model: 'gpt-b' } } }); });
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

test('a send lost to a disconnect is released only by the first sync reply after reconnecting', async () => {
  const { sent, state, link, lastSyncId } = fixture();
  openChat();
  state({ ack: 'mount' });
  type('Go');
  fireEvent.click(sendButton());
  link(false); link(true);
  state({ revision: 2, ack: 'mount' });
  expect(screen.getByRole('button', { name: 'Stop agent' })).toBeVisible();
  await act(async () => { state({ revision: 2, ack: lastSyncId() }); });
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

test('checking appears in the reply area; idle stays quiet and readiness stays in settings descriptions', () => {
  const { state, link } = fixture();
  openChat();
  state({ agents: agents({ status: 'checking' }) });
  expect(screen.getByRole('img', { name: 'Checking agent connection' })).toBeVisible();
  state({ revision: 2, provider: 'claude' });
  expect(screen.queryByRole('img', { name: 'Agent idle' })).toBeNull();
  expect(document.querySelector('.canvas-agent-generation')).toBeNull();
  expect(screen.queryByText('Ready')).toBeNull();
  // A long default model name never appears as on-screen text while settings are collapsed.
  expect(settingsToggle()).toHaveTextContent(/^$/);
  expect(screen.queryByText(/Opus 5\.5/)).toBeNull();
  expect(settingsToggle()).toHaveAccessibleName('Agent settings: Ready, Claude Code, Default (Opus 5.5), Default effort');
  expect(settingsToggle()).toHaveAttribute('title', 'Agent settings · Ready · Claude Code · Default (Opus 5.5) · Default effort');
  link(false);
  expect(screen.getByRole('img', { name: 'Reconnecting to VS Code' })).toBeVisible();
});

test('settings start collapsed and open and close from the toggle', () => {
  const { state } = fixture();
  openChat();
  state();
  expect(settingsPanel()).toBeNull();
  expect(screen.queryByRole('combobox', { name: 'Agent' })).toBeNull();
  expect(screen.queryByRole('combobox', { name: 'Model' })).toBeNull();
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

test('a dispatched Send collapses settings; a Send that cannot be dispatched leaves them open', async () => {
  const { send, sent, state } = fixture();
  openChat();
  state({ ack: 'mount' });
  type('Go');
  openSettings();
  chooseModel('gpt-b');
  send.mockReturnValueOnce(null as any);
  await act(async () => { fireEvent.click(sendButton()); });
  expect(settingsPanel()).toBeVisible();
  expect(screen.getByText('Canvas is reconnecting. Your draft is saved; try again when connected.')).toBeVisible();
  fireEvent.click(sendButton());
  expect(sent('send').slice(-1)[0].settings).toEqual({ model: 'gpt-b' });
  expect(settingsPanel()).toBeNull();
  expect(screen.getByRole('textbox')).toHaveFocus();
  // The draft stays until the host records the turn.
  expect(screen.getByRole('textbox')).toHaveValue('Go');
});

test('connection status and recheck live in settings, with account details available on focus', () => {
  const { sent, state, send } = fixture();
  openChat();
  state();
  expect(screen.queryByRole('button', { name: 'Recheck agent connection' })).toBeNull();
  openSettings();
  const panel = screen.getByRole('group', { name: 'Agent settings' });
  const caption = within(panel).getByText('Connected');
  expect(caption).toBeVisible();
  expect(caption).toHaveAttribute('tabindex', '0');
  act(() => caption.focus());
  expect(caption).toHaveAccessibleDescription('Uses your Codex account. Model access and limits are checked when you send. Source edits open in VS Code for review.');
  expect(within(panel).queryByText(/Codex ready/)).toBeNull();
  expect(within(panel).getByRole('combobox', { name: 'Model' })).toHaveAccessibleDescription('Balanced');
  fireEvent.click(within(panel).getByRole('button', { name: 'Recheck agent connection' }));
  expect(sent('discover').slice(-1)[0]).toEqual(expect.objectContaining({ refresh: true }));
  const ack = send.mock.results.at(-1)!.value.messageId;
  state({ revision: 2, ack, agents: agents({ status: 'checking' }) });
  state({ revision: 3, ack, agents: agents({ status: 'signInRequired', models: [] }) });
  expect(screen.getAllByRole('button', { name: 'Recheck agent connection' })).toHaveLength(1);
  expect(within(screen.getByRole('group', { name: 'Agent settings' })).getByText('Sign in to Codex in a terminal, then refresh.')).toBeVisible();
  expect(screen.getByRole('textbox')).toHaveAccessibleDescription('Sign in to Codex in a terminal, then refresh.');
});

test('recheck starts only on its host acknowledgement and settles when that observed check finishes', () => {
  const { state, send } = fixture(); openChat(); state(); openSettings(); type('Keep my question');
  fireEvent.click(recheckButton()); const dispatch = send.mock.results.at(-1)!.value;
  expect(recheckButton()).toBeDisabled(); expect(recheckButton().querySelector('svg')).toHaveClass('canvas-agent-task-spin');
  expect(screen.queryByText('Connection checked')).toBeNull();
  state({ revision: 2, ack: dispatch.messageId });
  expect(recheckButton()).toBeDisabled(); expect(screen.queryByText('Connection checked')).toBeNull();
  state({ revision: 3, ack: 'unrelated', agents: agents({ status: 'checking' }) });
  state({ revision: 4, ack: dispatch.messageId });
  expect(recheckButton()).toBeDisabled(); expect(screen.queryByText('Connection checked')).toBeNull();
  state({ revision: 5, ack: dispatch.messageId, agents: agents({ status: 'checking' }) });
  expect(screen.getByText('Checking Codex…')).toBeVisible();
  state({ revision: 4, ack: dispatch.messageId });
  expect(screen.queryByText('Connection checked')).toBeNull();
  // Another control can replace the host's last ack while this observed check runs.
  state({ revision: 6, ack: 'another-control-ack' });
  expect(screen.getByText('Connection checked')).toBeVisible(); expect(recheckButton()).toBeEnabled();
  expect(recheckButton().querySelector('svg')).not.toHaveClass('canvas-agent-task-spin');
  expect(screen.getByRole('textbox')).toHaveValue('Keep my question');
});

test('a recheck that ends with an unavailable agent never reports a successful connection check', () => {
  const { state, send } = fixture(); openChat(); state(); openSettings();
  fireEvent.click(recheckButton()); const ack = send.mock.results.at(-1)!.value.messageId;
  state({ revision: 2, ack, agents: agents({ status: 'checking' }) });
  state({ revision: 3, ack, agents: agents({ status: 'signInRequired', models: [] }) });
  expect(screen.queryByText('Connection checked')).toBeNull(); expect(recheckButton()).toBeEnabled();
  expect(screen.getByText('Sign in to Codex in a terminal, then refresh.')).toBeVisible(); expect(sendButton()).toBeDisabled();
});

test.each(['provider', 'session', 'configuration', 'disconnect'] as const)('a completed connection check is cleared after a %s change', change => {
  const { state, send, link } = fixture(); openChat(); state(); openSettings();
  fireEvent.click(recheckButton()); const ack = send.mock.results.at(-1)!.value.messageId;
  state({ revision: 2, ack, agents: agents({ status: 'checking' }) });
  state({ revision: 3, ack }); expect(screen.getByText('Connection checked')).toBeVisible();
  if (change === 'provider') state({ revision: 4, provider: 'claude' });
  else if (change === 'session') state({ sessionId: 'new-chat', revision: 0 });
  else if (change === 'configuration') chooseModel('gpt-b');
  else { link(false); link(true); }
  expect(screen.queryByText('Connection checked')).toBeNull();
  expect(screen.getByText('Connected')).toBeVisible(); expect(recheckButton()).toBeEnabled();
});

test('checking before a conversation uses the reply area rather than a header spinner', () => {
  const { state } = fixture(); openChat(); state({ agents: agents({ status: 'checking' }) });
  const indicator = screen.getByRole('img', { name: 'Checking agent connection' });
  expect(indicator.closest('.canvas-agent-preparing')).not.toBeNull(); expect(indicator).toHaveAttribute('tabindex', '0');
  expect(document.querySelector('.canvas-agent-bar')).not.toContainElement(indicator);
  expect(screen.queryByRole('log')).toBeNull(); expect(indicator.querySelectorAll('.canvas-agent-generation-dots > i')).toHaveLength(4);
  state({ revision: 2 }); expect(document.querySelector('.canvas-agent-generation')).toBeNull();
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

const permission = (changes: Partial<CanvasAgentPermission> = {}): CanvasAgentPermission => ({
  id: 'install-1', requestId: 'setup-turn', operation: 'python.packages.install', scope: 'isolated_environment', effects: ['install'], status: 'pending',
  facts: { python: '/storage/tensorviz/runtime/venv/bin/python', packages: ['numpy==2.2.6', 'scipy==1.15.3'], preserveExistingTorch: true },
  presentation: { title: 'Install project libraries', summary: 'Install the reviewed libraries in TensorViz Python.', allowLabel: 'Install', denyLabel: 'Deny' }, ...changes,
});
const approvalState = (changes: Partial<CanvasAgentState> = {}): Partial<CanvasAgentState> => ({
  status: 'running', requestId: 'setup-turn', activity: 'Waiting for Codex…', context: { target, nodeIds: [] },
  messages: [{ id: 'setup-turn', role: 'user', text: 'Set up this project.' }], permissions: [permission()], ...changes,
});

test('a host permission opens this chat with a short explanation, neutral focus and exact details on demand', () => {
  const { state, sent } = fixture();
  state(approvalState());
  const card = screen.getByRole('region', { name: 'Install project libraries' });
  expect(screen.getByRole('region', { name: 'Canvas chat' })).toBeVisible();
  expect(within(card).getByRole('heading', { name: 'Install project libraries' })).toHaveFocus();
  expect(within(card).getByText('Install the reviewed libraries in TensorViz Python.')).toBeVisible();
  expect(within(card).getByRole('button', { name: 'Install' })).not.toHaveFocus();
  expect(within(card).getByText(/numpy==2\.2\.6/)).not.toBeVisible();
  expect(screen.getByRole('img', { name: 'Waiting for your approval' })).toBeVisible();
  expect(document.querySelector('.canvas-agent-activity')).toBeNull();
  expect(screen.queryByText('Waiting for Codex…')).toBeNull();
  expect(screen.queryByText('Ready')).toBeNull();
  fireEvent.click(within(card).getByText('Details'));
  expect(within(card).getByText(/numpy==2\.2\.6/)).toBeVisible();
  expect(sent('permission')).toHaveLength(0);
  expect(sent('discover')).toHaveLength(0);
});

test('an unexplained operation stays working without presenting consent or stealing focus', () => {
  const { state, sent } = fixture();
  state(approvalState({ permissions: [permission({ presentation: undefined })] }));
  expect(screen.queryByRole('region', { name: 'Canvas chat' })).toBeNull();
  expect(screen.getByRole('button', { name: 'Open agent chat · working' })).toBeVisible();
  openChat();
  const card = screen.getByRole('region', { name: 'Preparing approval…' });
  expect(card).toHaveAttribute('aria-busy', 'true');
  expect(screen.getByRole('textbox')).toHaveFocus();
  expect(screen.getByRole('img', { name: 'Agent running' })).toBeVisible();
  expect(screen.queryByText('Needs approval')).toBeNull();
  expect(within(card).queryByRole('button')).toBeNull();
  expect(screen.getByRole('button', { name: 'Stop agent' })).toBeEnabled();
  fireEvent.click(within(card).getByText('Details'));
  expect(within(card).getByText(/python\.packages\.install/)).toBeVisible();
  expect(sent('permission')).toHaveLength(0);
});

test('agent presentation turns the prepared operation into a focused user decision with dynamic labels', () => {
  const { state, sent } = fixture();
  state(approvalState({ permissions: [permission({ presentation: undefined })] }));
  state(approvalState({ revision: 2, permissions: [permission({ presentation: {
    title: 'Prepare VideoJev inference', summary: 'These packages let VideoJev load for the requested graph.',
    allowLabel: 'Prepare VideoJev', denyLabel: 'Leave Python as it is',
  } })] }));
  const card = screen.getByRole('region', { name: 'Prepare VideoJev inference' });
  expect(within(card).getByRole('heading', { name: 'Prepare VideoJev inference' })).toHaveFocus();
  expect(card).toHaveAttribute('aria-busy', 'false');
  expect(within(card).getByRole('button', { name: 'Prepare VideoJev' })).toBeEnabled();
  fireEvent.click(within(card).getByText('Details'));
  expect(within(card).getByText(/python\.packages\.install/)).toBeVisible();
  expect(within(card).getByText(/numpy==2\.2\.6/)).toBeVisible();
  fireEvent.click(within(card).getByRole('button', { name: 'Leave Python as it is' }));
  expect(sent('permission')).toEqual([expect.objectContaining({ permissionId: 'install-1', decision: 'deny' })]);
});

test.each([['allow', 'Install', 'approved', 'Allowed'], ['deny', 'Deny', 'denied', 'Denied']] as const)(
  '%s sends the exact pending permission while the agent is running and waits for host acknowledgement', (decision, label, status, result) => {
    const { state, send, sent } = fixture();
    state(approvalState());
    fireEvent.click(screen.getByRole('button', { name: label }));
    const dispatch = send.mock.results[send.mock.results.length - 1].value;
    expect(sent('permission')).toEqual([expect.objectContaining({ action: 'permission', permissionId: 'install-1', decision })]);
    expect(screen.getByRole('button', { name: 'Install' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Deny' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: label }));
    expect(sent('permission')).toHaveLength(1);
    state(approvalState({ revision: 2, ack: 'unrelated-background-update' }));
    expect(screen.getByRole('button', { name: 'Install' })).toBeDisabled();
    state(approvalState({ revision: 1, ack: dispatch.messageId }));
    expect(screen.getByRole('button', { name: 'Install' })).toBeDisabled();
    state(approvalState({ revision: 3, ack: dispatch.messageId, permissions: [permission({ status })], activity: 'Continuing project setup…' }));
    expect(screen.queryByRole('button', { name: 'Install' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Deny' })).toBeNull();
    expect(screen.getByText(`Install project libraries · ${result}`)).toBeVisible();
    expect(screen.getByRole('img', { name: 'Agent running' })).toBeVisible();
    expect(screen.queryByText('Ready')).toBeNull();
    expect(sent('send')).toHaveLength(0);
  });

test('minimizing preserves the draft and pending permission with an attention shield', () => {
  const { state } = fixture();
  openChat(); state(); type('Keep my unfinished question');
  fireEvent.click(screen.getByRole('button', { name: 'Minimize agent chat' }));
  state(approvalState({ revision: 2 }));
  expect(screen.getByRole('textbox')).toHaveValue('Keep my unfinished question');
  fireEvent.click(screen.getByRole('button', { name: 'Minimize agent chat' }));
  expect(screen.getByRole('button', { name: 'Open agent chat · needs approval' })).toBeVisible();
  expect(screen.getByRole('button', { name: 'Open agent chat · needs approval' })).toHaveClass('canvas-agent-bubble-approval');
  state(approvalState({ revision: 3, activity: 'Still waiting for approval…' }));
  expect(screen.queryByRole('region', { name: 'Canvas chat' })).toBeNull();
  openChat();
  expect(screen.getByRole('textbox')).toHaveValue('Keep my unfinished question');
  expect(screen.getByRole('button', { name: 'Install' })).toBeEnabled();
});

test('disconnect disables consent, and reconnect reconciles a lost decision without replaying it', () => {
  const { state, link, sent, lastSyncId } = fixture();
  state(approvalState());
  link(false);
  expect(screen.getByRole('button', { name: 'Install' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Deny' })).toBeDisabled();
  expect(screen.getByText('Reconnect to respond. Nothing is approved while disconnected.')).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Install' }));
  expect(sent('permission')).toHaveLength(0);
  link(true); state(approvalState({ revision: 2, ack: lastSyncId() }));
  fireEvent.click(screen.getByRole('button', { name: 'Install' }));
  expect(sent('permission')).toHaveLength(1);
  link(false); link(true);
  expect(screen.getByRole('button', { name: 'Install' })).toBeDisabled();
  state(approvalState({ revision: 3, ack: lastSyncId() }));
  expect(screen.getByRole('button', { name: 'Install' })).toBeEnabled();
  expect(sent('permission')).toHaveLength(1);
  fireEvent.click(screen.getByRole('button', { name: 'Deny' }));
  expect(sent('permission')).toHaveLength(2);
  expect(sent('permission')[1]).toEqual(expect.objectContaining({ permissionId: 'install-1', decision: 'deny' }));
});

test('a stopped or different task cannot offer actions for a stale pending permission', () => {
  const { state, sent } = fixture();
  state(approvalState());
  state(approvalState({ revision: 2, status: 'stopped', activity: 'Agent stopped.' }));
  expect(screen.queryByRole('button', { name: 'Install' })).toBeNull();
  expect(screen.getByText('Install project libraries · No longer active')).toBeVisible();
  state(approvalState({ revision: 3, requestId: 'another-turn' }));
  expect(screen.queryByRole('button', { name: 'Install' })).toBeNull();
  expect(screen.queryByText('Needs approval')).toBeNull();
  expect(sent('permission')).toHaveLength(0);
});

test('permission details are plain text and a restored request works without a retained message', () => {
  const { state } = fixture();
  const details = '<script>approveEverything()</script>\n/path/with/a/very/long/environment/name';
  state(approvalState({ messages: [], permissions: [permission({ operation: 'python.environment.create', effects: ['create'], facts: { destination: details }, presentation: { title: 'Create project Python', summary: 'Keep this model in a separate Python.', allowLabel: 'Allow', denyLabel: 'Keep current Python' } })] }));
  expect(screen.getByRole('log', { name: 'Agent conversation' })).toBeVisible();
  fireEvent.click(screen.getByText('Details'));
  expect(screen.getByText((_text, element) => element?.tagName === 'PRE' && !!element.textContent?.includes('<script>approveEverything()</script>'))).toBeVisible();
  expect(document.querySelector('.canvas-agent-permission script')).toBeNull();
  expect(screen.getByRole('button', { name: 'Allow' })).toBeEnabled();
});

const permissionModeButton = () => screen.getByRole('button', { name: /^Setup permissions:/ });
const permissionModeMenu = () => screen.getByRole('menu', { name: 'Setup permissions' });
const permissionModeLabels = { ask: 'Ask every time', auto_setup: 'Auto setup', full_setup: 'Full setup access' };
const choosePermissionMode = (mode: keyof typeof permissionModeLabels) => {
  fireEvent.click(permissionModeButton());
  fireEvent.click(screen.getByRole('menuitemradio', { name: permissionModeLabels[mode] }));
};

test('setup permissions stay beside composer controls with concise explanations on demand', () => {
  const { state, sent } = fixture(); openChat(); state(); type('Keep this question');
  const toggle = permissionModeButton();
  expect(toggle).toHaveAccessibleName('Setup permissions: Ask every time');
  expect(toggle.closest('.canvas-agent-composer-tools')).toContainElement(sendButton());
  expect(settingsPanel()).toBeNull();
  expect(screen.queryByRole('menu')).toBeNull();
  expect(toggle).not.toHaveTextContent('Ask every time');
  fireEvent.focus(toggle);
  expect(toggle).toHaveAccessibleDescription(/Ask every time\. Review each environment change\. Commands and source edits always ask\. This chat only/);
  fireEvent.click(toggle);
  const menu = permissionModeMenu(), items = within(menu).getAllByRole('menuitemradio');
  expect(items).toHaveLength(3);
  expect(within(menu).getByRole('menuitemradio', { name: 'Ask every time' })).toHaveAttribute('aria-checked', 'true');
  expect(within(menu).getByRole('menuitemradio', { name: 'Auto setup' })).toHaveAccessibleDescription('Approve new isolated Python and its packages. Builds and existing environments still ask. Commands and source edits always ask.');
  fireEvent.pointerEnter(within(menu).getByRole('menuitemradio', { name: 'Auto setup' }));
  expect(within(menu).getByText('Approve new isolated Python and its packages. Builds and existing environments still ask.')).toBeVisible();
  fireEvent.pointerLeave(menu.querySelector('.canvas-agent-mode-options')!);
  expect(within(menu).getByText('Review each environment change.')).toBeVisible();
  fireEvent.click(within(menu).getByRole('menuitemradio', { name: 'Ask every time' }));
  expect(screen.queryByRole('menu')).toBeNull();
  expect(toggle).toHaveFocus();
  expect(screen.getByRole('textbox')).toHaveValue('Keep this question');
  expect(sent('permission_mode')).toHaveLength(0);
});

test('permission menu keyboard navigation inspects modes without changing access and Escape returns to its trigger', () => {
  const { state, sent } = fixture(); openChat(); state({ permissionMode: 'auto_setup' }); type('Keep my setup question');
  const toggle = permissionModeButton();
  act(() => toggle.focus()); fireEvent.keyDown(toggle, { key: 'ArrowDown' });
  const menu = permissionModeMenu();
  const ask = within(menu).getByRole('menuitemradio', { name: 'Ask every time' });
  const auto = within(menu).getByRole('menuitemradio', { name: 'Auto setup' });
  const full = within(menu).getByRole('menuitemradio', { name: 'Full setup access' });
  expect(auto).toHaveFocus(); expect(auto).toHaveAttribute('aria-checked', 'true');
  fireEvent.keyDown(auto, { key: 'ArrowDown' }); expect(full).toHaveFocus();
  expect(within(menu).getByText('Approve environment setup, including builds and version changes.')).toBeVisible();
  fireEvent.keyDown(full, { key: 'ArrowDown' }); expect(ask).toHaveFocus();
  fireEvent.keyDown(ask, { key: 'ArrowUp' }); expect(full).toHaveFocus();
  fireEvent.keyDown(full, { key: 'Home' }); expect(ask).toHaveFocus();
  fireEvent.keyDown(ask, { key: 'End' }); expect(full).toHaveFocus();
  expect(auto).toHaveAttribute('aria-checked', 'true'); expect(full).toHaveAttribute('aria-checked', 'false');
  expect(sent('permission_mode')).toHaveLength(0);
  fireEvent.keyDown(full, { key: 'Escape' });
  expect(screen.queryByRole('menu')).toBeNull(); expect(toggle).toHaveFocus();
  expect(screen.getByRole('region', { name: 'Canvas chat' })).toBeVisible();
  expect(screen.getByRole('textbox')).toHaveValue('Keep my setup question');
});

test('permission menu dismisses on pointer or focus moving outside without sending or losing the draft', () => {
  const { state, sent } = fixture(); openChat(); state(); type('Keep my draft');
  const input = screen.getByRole('textbox');
  fireEvent.click(permissionModeButton()); expect(permissionModeMenu()).toBeVisible();
  fireEvent.pointerDown(input); expect(screen.queryByRole('menu')).toBeNull();
  fireEvent.click(permissionModeButton()); expect(permissionModeMenu()).toBeVisible();
  act(() => input.focus()); expect(screen.queryByRole('menu')).toBeNull(); expect(input).toHaveFocus();
  expect(input).toHaveValue('Keep my draft'); expect(sent('permission_mode')).toHaveLength(0);
});

test('permission menu cannot open before host state and closes its portal on disconnect or unmount', () => {
  const { state, link, unmount, sent } = fixture(); openChat();
  expect(permissionModeButton()).toBeDisabled(); fireEvent.click(permissionModeButton()); expect(screen.queryByRole('menu')).toBeNull();
  state(); fireEvent.click(permissionModeButton()); expect(permissionModeMenu()).toBeVisible();
  link(false); expect(screen.queryByRole('menu')).toBeNull(); expect(permissionModeButton()).toBeDisabled();
  link(true); fireEvent.click(permissionModeButton()); expect(permissionModeMenu()).toBeVisible();
  unmount(); expect(screen.queryByRole('menu')).toBeNull(); expect(sent('permission_mode')).toHaveLength(0);
});

test('mode changes work during a pending task and await the exact host acknowledgement', () => {
  const { state, send, sent } = fixture(); state(approvalState());
  expect(permissionModeButton()).toBeEnabled();
  choosePermissionMode('auto_setup');
  const dispatch = send.mock.results[send.mock.results.length - 1].value;
  expect(sent('permission_mode')).toEqual([expect.objectContaining({ mode: 'auto_setup' })]);
  expect(permissionModeButton()).toHaveAccessibleName('Setup permissions: Ask every time');
  expect(permissionModeButton()).toBeDisabled();
  expect(permissionModeButton().querySelector('svg')).toHaveClass('canvas-agent-task-spin');
  expect(screen.getByText('Updating permissions…', { selector: '.canvas-agent-announcement' })).toHaveAttribute('role', 'status');
  expect(document.querySelector('.canvas-agent-activity')?.textContent ?? '').not.toContain('Updating permissions…');
  expect(screen.queryByRole('menu')).toBeNull();
  expect(screen.getByRole('button', { name: 'Install' })).toBeEnabled();
  fireEvent.click(permissionModeButton());
  expect(screen.queryByRole('menu')).toBeNull();
  expect(sent('permission_mode')).toHaveLength(1);
  state(approvalState({ revision: 2, permissionMode: 'auto_setup', ack: 'unrelated-background-ack' }));
  expect(permissionModeButton()).toHaveAccessibleName('Setup permissions: Auto setup');
  expect(permissionModeButton()).toBeDisabled();
  state(approvalState({ revision: 1, ack: dispatch.messageId }));
  expect(permissionModeButton()).toBeDisabled();
  state(approvalState({ revision: 3, permissionMode: 'auto_setup', ack: dispatch.messageId,
    permissions: [permission({ status: 'approved', approval: 'policy', mode: 'auto_setup' })], activity: 'Preparing inputs…' }));
  expect(permissionModeButton()).toBeEnabled();
  expect(permissionModeButton()).toHaveAccessibleName('Setup permissions: Auto setup');
  expect(screen.queryByText('Updating permissions…')).toBeNull();
  expect(screen.queryByRole('button', { name: 'Install' })).toBeNull();
  expect(screen.getByText('Install project libraries · Auto-approved')).toBeVisible();
  expect(screen.queryByText('Install project libraries · Allowed')).toBeNull();
  expect(screen.getByRole('img', { name: 'Agent running' })).toBeVisible();
  fireEvent.click(screen.getByText('Install project libraries · Auto-approved'));
  expect(screen.getByText('Approved by Auto setup.')).toBeVisible();
  expect(sent('permission')).toHaveLength(0);
});

test('a refused mode change yields to the host without optimistic access', () => {
  const { state, send } = fixture(); openChat(); state();
  choosePermissionMode('full_setup');
  const dispatch = send.mock.results[send.mock.results.length - 1].value;
  expect(permissionModeButton()).toHaveAccessibleName('Setup permissions: Ask every time');
  state({ revision: 2, ack: dispatch.messageId, activity: 'Trust this workspace before changing setup permissions.' });
  expect(permissionModeButton()).toHaveAccessibleName('Setup permissions: Ask every time');
  expect(permissionModeButton()).toBeEnabled();
  expect(screen.getByText('Trust this workspace before changing setup permissions.')).toBeVisible();
});

test('offline mode choice is disabled and a lost change is reconciled without reconnect replay', () => {
  const { state, link, sent, lastSyncId } = fixture(); openChat(); state();
  choosePermissionMode('full_setup');
  link(false);
  expect(permissionModeButton()).toBeDisabled();
  fireEvent.click(permissionModeButton()); expect(screen.queryByRole('menu')).toBeNull();
  expect(sent('permission_mode')).toHaveLength(1);
  link(true);
  expect(permissionModeButton()).toBeDisabled();
  expect(sent('permission_mode')).toHaveLength(1);
  state({ revision: 2, ack: lastSyncId(), permissionMode: 'ask' });
  expect(permissionModeButton()).toBeEnabled();
  expect(permissionModeButton()).toHaveAccessibleName('Setup permissions: Ask every time');
  expect(sent('permission_mode')).toHaveLength(1);
  choosePermissionMode('auto_setup');
  expect(sent('permission_mode')).toHaveLength(2);
});

test('new chat resets permissions only when host state confirms the new session', () => {
  const { state, sent } = fixture(); openChat();
  state({ permissionMode: 'full_setup', messages: [{ id: 'u', role: 'user', text: 'Earlier setup' }] });
  fireEvent.click(screen.getByRole('button', { name: 'New agent chat' }));
  fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
  expect(sent('clear')).toHaveLength(1);
  expect(permissionModeButton()).toHaveAccessibleName('Setup permissions: Full setup access');
  state({ sessionId: 'new-chat', revision: 0 });
  expect(permissionModeButton()).toHaveAccessibleName('Setup permissions: Ask every time');
  expect(sent('permission_mode')).toHaveLength(0);
});

test('policy receipts identify automatic approval while legacy user receipts remain Allowed', () => {
  const { state } = fixture(); openChat();
  state({ permissionMode: 'full_setup' });
  state({ revision: 2, permissionMode: 'full_setup', permissions: [
    permission({ id: 'automatic', status: 'approved', approval: 'policy', mode: 'full_setup' }),
    permission({ id: 'legacy', status: 'approved', presentation: { title: 'Create Python', summary: 'Keep this model in a separate Python.', allowLabel: 'Create', denyLabel: 'Cancel' } }),
  ] });
  expect(screen.getByText('Install project libraries · Auto-approved')).toBeVisible();
  expect(screen.getByRole('status')).toHaveTextContent('Install project libraries: auto-approved');
  expect(screen.getByText('Create Python · Allowed')).toBeVisible();
  const receipt = screen.getByText('Install project libraries · Auto-approved').closest('details')!;
  fireEvent.click(within(receipt).getByText('Install project libraries · Auto-approved'));
  expect(screen.getByText('Approved by Full setup access.')).toBeVisible();
  expect(within(receipt).getByText(/numpy==2\.2\.6/)).toBeVisible();
  expect(screen.queryByRole('button', { name: 'Install' })).toBeNull();
});


test('generation stays at the end of the conversation and disappears when stopped, idle or awaiting approval', () => {
  const { state } = fixture(); openChat();
  state({ status: 'running', requestId: 'work', activity: 'Waiting for Codex…', messages: [{ id: 'work', role: 'user', text: 'Inspect this model' }] });
  const working = screen.getByRole('img', { name: 'Agent running' });
  expect(working).toHaveClass('canvas-agent-generation');
  expect(working.querySelectorAll('.canvas-agent-generation-dots > i')).toHaveLength(4);
  expect(screen.getByRole('log', { name: 'Agent conversation' }).lastElementChild).toBe(working);
  expect(document.querySelector('.canvas-agent-bar')).not.toContainElement(working);
  expect(screen.queryByText('Waiting for Codex…')).toBeNull();
  expect(screen.getByRole('button', { name: 'Stop agent' })).toBeEnabled();
  state({ revision: 2, status: 'stopped', activity: 'Agent stopped.', messages: [{ id: 'work', role: 'user', text: 'Inspect this model' }] });
  expect(screen.getByRole('img', { name: 'Agent stopped' })).toHaveAttribute('data-state', 'stopped');
  expect(screen.getByRole('img', { name: 'Agent stopped' }).querySelector('svg')).not.toHaveClass('canvas-agent-task-spin');
  expect(document.querySelector('.canvas-agent-generation')).toBeNull();
  expect(screen.queryByText('Agent stopped.')).toBeNull();
  expect(screen.queryByRole('button', { name: 'Stop agent' })).toBeNull();
  state({ revision: 3, status: 'idle', activity: '', messages: [{ id: 'answer', role: 'assistant', text: 'This source call scores candidate answers.' }] });
  expect(screen.queryByRole('img', { name: 'Agent idle' })).toBeNull();
  expect(document.querySelector('.canvas-agent-generation')).toBeNull();
  expect(screen.queryByText('Ready')).toBeNull();
  state(approvalState({ revision: 4 }));
  expect(screen.getByRole('img', { name: 'Waiting for your approval' })).toBeVisible();
  expect(document.querySelector('.canvas-agent-generation')).toBeNull();
});

test('provider readiness cannot turn a failed task green or duplicate its error under a reply', () => {
  const { state } = fixture(); openChat();
  const failure = 'The model imports a CUDA-only package unavailable on this Apple Silicon host.';
  state({ status: 'error', activity: failure, agents: agents(), messages: [{ id: 'answer', role: 'assistant', text: 'The import requires CUDA. This Mac has no CUDA device.' }] });
  const icon = screen.getByRole('img', { name: 'Agent needs attention' });
  expect(icon).toHaveAttribute('title', `Agent needs attention · ${failure}`);
  expect(screen.queryByText('Ready')).toBeNull();
  expect(screen.getByText(failure)).not.toBeVisible();
  fireEvent.click(screen.getByText('Error details'));
  expect(screen.getByRole('alert')).toHaveTextContent(failure);
  expect(screen.getByRole('button', { name: 'Send request' })).toBeVisible();
  openSettings();
  expect(screen.getByText('Connected')).toBeVisible();
});

test('an error before the first reply stays directly visible and actionable', () => {
  const { state } = fixture(); openChat();
  state({ status: 'error', activity: 'Trust this workspace before executing project code.', messages: [] });
  expect(screen.getByRole('alert')).toHaveTextContent('Trust this workspace before executing project code.');
  expect(screen.queryByText('Error details')).toBeNull();
  expect(screen.getByRole('img', { name: 'Agent needs attention' })).toBeVisible();
});

test('retained setup evidence is collapsed until requested and exposes the exact traceback and observed runtime', () => {
  const { state } = fixture(); openChat();
  const message = "cannot import name 'ALL_ATTENTION_FUNCTIONS' from 'transformers.modeling_utils'";
  const traceback = `Traceback (most recent call last):\n  File "/project/timejev/asymmetric_video_memory.py", line 9, in <module>\nImportError: ${message}`;
  const runtime = { platform: 'darwin', architecture: 'arm64', python: '3.12.13', torch: '2.6.0', cuda: null, cudaAvailable: false, mpsAvailable: true };
  state({ status: 'error', activity: `ImportError: ${message}`, messages: [{ id: 'answer', role: 'assistant', text: 'The installed library does not export the symbol this model imports.' }],
    failure: { kind: 'environment', code: 'MODEL_IMPORT_FAILED', importFailure: true, diagnostic: {
      exceptionType: 'ImportError', message, traceback, causes: [{ exceptionType: 'ImportError', message }], runtime, truncated: false,
    } } });
  const trace = screen.getByLabelText('Setup traceback'), observed = screen.getByLabelText('Observed setup runtime');
  expect(trace).not.toBeVisible(); expect(observed).not.toBeVisible();
  expect(screen.getByText('Error details').closest('details')).not.toHaveAttribute('open');
  fireEvent.click(screen.getByText('Error details'));
  expect(trace).toBeVisible(); expect(trace.textContent).toBe(traceback); expect(trace).toHaveAttribute('tabindex', '0');
  expect(observed).toBeVisible(); expect(observed.textContent).toBe(JSON.stringify(runtime, null, 2));
  expect(observed).toHaveAttribute('tabindex', '0');
  expect(screen.getAllByRole('alert')).toHaveLength(1); expect(screen.getByRole('alert')).toHaveTextContent(message);
  state({ revision: 2, status: 'running', requestId: 'next-turn', activity: 'Investigating…' });
  expect(screen.queryByLabelText('Setup traceback')).toBeNull(); expect(screen.queryByText('Error details')).toBeNull();
});

test('a retained failure before any reply keeps its short error visible and its evidence expandable', () => {
  const { state } = fixture(); openChat();
  state({ status: 'error', activity: 'Model import failed.', failure: { kind: 'environment', code: 'MODEL_IMPORT_FAILED', diagnostic: {
    exceptionType: 'ImportError', message: 'Missing export.', traceback: 'ImportError: Missing export.', causes: [],
    runtime: { platform: 'darwin', architecture: 'arm64', python: '3.12.13', torch: null, cuda: null, cudaAvailable: null, mpsAvailable: null }, truncated: true,
  } } });
  expect(screen.getByRole('alert')).toHaveTextContent('Model import failed.');
  expect(screen.getByLabelText('Setup traceback')).not.toBeVisible();
  fireEvent.click(screen.getByText('Error details'));
  expect(screen.getByLabelText('Setup traceback')).toBeVisible(); expect(screen.getByText('Evidence truncated.')).toBeVisible();
  expect(screen.getAllByRole('alert')).toHaveLength(1);
});
