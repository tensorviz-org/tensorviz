import { createPanelTransport } from '../../plugin/panelTransport';
import type { LocalBrowserSessionInfo, LocalBrowserTargetIdentity } from '../../ide/protocol.generated';

const session = { protocolVersion: 1, role: 'view', workspaceSessionId: 'test-session', viewId: 'test-view',
  capabilities: {} } as LocalBrowserSessionInfo;
const target = { workspaceFolder: '/project', file: 'model.py', cls: 'Tiny', documentVersion: 1,
  sourceDigest: 'a'.repeat(64), backendRevision: 1 } as LocalBrowserTargetIdentity;
const selection = { type: 'viewSelection' as const, viewId: session.viewId!, target, nodeIds: ['layer'], edgeIds: [] };
const packet = (events: unknown[] = []) => ({ content: [], _meta: { panel: { session, events, waitingForModel: false,
  status: { phase: 'connected', hostLink: 'connected', error: null } } } });
const tick = async () => { for (let i = 0; i < 10; i++) await Promise.resolve(); };

beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

test('polling does not repeat connection callbacks or replay acknowledged events', async () => {
  const event = { sequence: 1, wire: JSON.stringify({ type: 'snapshot', protocolVersion: 1, workspaceSessionId: 'test-session',
    messageId: 'snapshot-1', role: 'host', hostSequence: 1, state: { activeTarget: null, projection: null, traced: false,
      greyed: false, connectionState: 'connected', syncState: null } }) };
  const app = { callServerTool: jest.fn().mockResolvedValue(packet([event])), updateModelContext: jest.fn().mockResolvedValue({}) };
  const transport = createPanelTransport(app, 'panel', jest.fn(), jest.fn());
  const onStatus = jest.fn(), onMessage = jest.fn();
  transport.onStatusChange(onStatus); transport.onMessage(onMessage);
  await transport.connect(); await tick();
  jest.advanceTimersByTime(350); await tick();
  expect(transport.status.phase).toBe('connected');
  expect(onMessage).toHaveBeenCalledTimes(1);
  expect(onStatus).toHaveBeenCalledTimes(2); // connecting, connected
  expect(app.callServerTool.mock.calls[1][0].arguments.after).toBe(1);
  expect(app.callServerTool.mock.calls.slice(0, 2).map(([args]) => args.name)).toEqual(['graph_panel_read', 'graph_panel_read']);
  transport.dispose();
  expect(app.callServerTool.mock.calls[2][0].name).toBe('detach_graph_panel');
});

test('uncertain setup writes stop with visible recovery and are never replayed', async () => {
  const app = { callServerTool: jest.fn().mockResolvedValue(packet()), updateModelContext: jest.fn().mockResolvedValue({}) };
  const waiting = jest.fn(), transport = createPanelTransport(app, 'panel', waiting, jest.fn());
  await transport.connect(); await tick();
  app.callServerTool.mockRejectedValueOnce(new Error('lost response'));
  transport.send({ type: 'graphSetupRequest', target, openingId: 'opening', requestId: 'request', action: { kind: 'retry' } });
  await tick(); jest.advanceTimersByTime(2000); await tick();
  expect(transport.status.phase).toBe('ended');
  expect(waiting).toHaveBeenLastCalledWith(false);
  expect(app.callServerTool.mock.calls.filter(([args]) => args.name === 'graph_panel_exchange')).toHaveLength(1);
  expect(app.callServerTool).toHaveBeenLastCalledWith({ name: 'close_graph_panel', arguments: expect.objectContaining({ panelId: 'panel' }) });
  transport.dispose();
  expect(app.callServerTool).toHaveBeenCalledTimes(3); // failed views cannot leave a current display receipt behind
});

test('a temporary idle read failure reconnects without closing or rerunning the model', async () => {
  const event = { sequence: 1, wire: JSON.stringify({ type: 'snapshot', protocolVersion: 1, workspaceSessionId: 'test-session',
    messageId: 'snapshot-1', role: 'host', hostSequence: 1, state: { activeTarget: null, projection: null, traced: false,
      greyed: false, connectionState: 'connected', syncState: null } }) };
  const app = { callServerTool: jest.fn().mockResolvedValue(packet([event])), updateModelContext: jest.fn() };
  const transport = createPanelTransport(app, 'panel', jest.fn(), jest.fn()), listener = jest.fn();
  transport.onMessage(listener);
  await transport.connect(); await tick();
  app.callServerTool.mockRejectedValueOnce(new Error('chat bridge temporarily unavailable'));
  jest.advanceTimersByTime(350); await tick();
  expect(transport.status.phase).toBe('reconnecting');
  expect(transport.send({ type: 'graphSetupRequest', target, openingId: 'opening', requestId: 'request', action: { kind: 'retry' } })).toBeNull();
  jest.advanceTimersByTime(500); await tick();
  expect(transport.status.phase).toBe('connected');
  expect(listener).toHaveBeenCalledTimes(1);
  expect(app.callServerTool.mock.calls.map(([args]) => args.name)).toEqual(['graph_panel_read', 'graph_panel_read', 'graph_panel_read']);
  expect(app.callServerTool.mock.calls[2][0].arguments.after).toBe(1);
  transport.dispose();
});

test('idle read retries back off, stop on disposal, and release the frame once', async () => {
  const app = { callServerTool: jest.fn().mockResolvedValue(packet()), updateModelContext: jest.fn() };
  const transport = createPanelTransport(app, 'panel', jest.fn(), jest.fn());
  await transport.connect(); await tick();
  app.callServerTool.mockRejectedValue(new Error('temporarily offline'));
  jest.advanceTimersByTime(350); await tick();
  for (const delay of [500, 1000, 2000, 5000, 5000]) {
    const calls = app.callServerTool.mock.calls.length;
    jest.advanceTimersByTime(delay - 1); await tick();
    expect(app.callServerTool).toHaveBeenCalledTimes(calls);
    jest.advanceTimersByTime(1); await tick();
    expect(app.callServerTool).toHaveBeenCalledTimes(calls + 1);
  }
  transport.dispose();
  const calls = app.callServerTool.mock.calls.length;
  jest.advanceTimersByTime(60_000); await tick();
  expect(app.callServerTool).toHaveBeenCalledTimes(calls);
  expect(app.callServerTool.mock.calls.filter(([args]) => args.name === 'detach_graph_panel')).toHaveLength(1);
});

test('an explicitly expired panel and a declined idle read end without retrying', async () => {
  for (const failure of [
    () => Promise.resolve({ isError: true, content: [], structuredContent: { error: { code: 'PANEL_CLOSED' } } }),
    () => Promise.reject(Object.assign(new Error('MCP error -32000: Declined by the user.'), { code: -32000 })),
  ]) {
    const app = { callServerTool: jest.fn().mockResolvedValue(packet()), updateModelContext: jest.fn() };
    const transport = createPanelTransport(app, 'panel', jest.fn(), jest.fn());
    await transport.connect(); await tick();
    app.callServerTool.mockImplementationOnce(failure);
    jest.advanceTimersByTime(350); await tick();
    expect(transport.status.phase).toBe('ended');
    const calls = app.callServerTool.mock.calls.length;
    jest.advanceTimersByTime(60_000); await tick();
    expect(app.callServerTool).toHaveBeenCalledTimes(calls);
    transport.dispose();
  }
});

test('host teardown waits for detachment and each new mount owns a distinct connection', async () => {
  const app = { callServerTool: jest.fn().mockResolvedValue(packet()), updateModelContext: jest.fn() };
  const first = createPanelTransport(app, 'panel', jest.fn(), jest.fn());
  await first.connect(); await tick();
  const firstId = app.callServerTool.mock.calls[0][0].arguments.clientId;
  expect(typeof firstId).toBe('string');
  let finish!: () => void;
  app.callServerTool.mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; }));
  let detached = false;
  const pending = first.detach().then(() => { detached = true; });
  await tick(); expect(detached).toBe(false);
  expect(app.callServerTool).toHaveBeenLastCalledWith({ name: 'detach_graph_panel', arguments: { panelId: 'panel', clientId: firstId } });
  finish(); await pending;
  const second = createPanelTransport(app, 'panel', jest.fn(), jest.fn());
  await second.connect(); await tick();
  expect(app.callServerTool.mock.calls[2][0].arguments.clientId).not.toBe(firstId);
  second.dispose();
});

test('declining a T3 action keeps the current graph connected without replaying the action', async () => {
  const app = { callServerTool: jest.fn().mockResolvedValue(packet()), updateModelContext: jest.fn() };
  const notice = jest.fn(), transport = createPanelTransport(app, 'panel', jest.fn(), notice);
  await transport.connect(); await tick();
  app.callServerTool.mockRejectedValueOnce(Object.assign(new Error('MCP error -32000: Declined by the user.'), { code: -32000 }));
  transport.send({ type: 'graphSetupRequest', target, openingId: 'opening', requestId: 'request', action: { kind: 'retry' } });
  await tick(); jest.advanceTimersByTime(350); await tick();
  expect(transport.status.phase).toBe('connected');
  expect(app.callServerTool.mock.calls.filter(([args]) => args.name === 'graph_panel_exchange')).toHaveLength(1);
  expect(app.callServerTool).toHaveBeenLastCalledWith(expect.objectContaining({ name: 'graph_panel_read' }));
  expect(notice).toHaveBeenCalledWith('Action cancelled. The current graph is still available.');
  transport.dispose();
});

test('selection context carries exact identity and exposes a context failure', async () => {
  const app = { callServerTool: jest.fn().mockResolvedValue(packet()), updateModelContext: jest.fn().mockRejectedValue(new Error('unavailable')) };
  const notice = jest.fn(), transport = createPanelTransport(app, 'panel', jest.fn(), notice);
  await transport.connect(); await tick(); transport.send(selection); await tick();
  expect(app.callServerTool.mock.calls[1][0].name).toBe('graph_panel_read');
  expect(app.updateModelContext).toHaveBeenCalledWith({ structuredContent: { tensorviz: { viewId: 'test-view', target,
    nodeIds: ['layer'], edgeIds: [], tooLarge: false } } });
  expect(notice).toHaveBeenCalledWith(expect.stringContaining('could not update'));
  transport.dispose();
});

test('foreign-session or out-of-order data is rejected before rendering', async () => {
  for (const event of [{ sequence: 2, wire: '{}' }, { sequence: 1, wire: JSON.stringify({ type: 'snapshot', workspaceSessionId: 'other' }) }]) {
    const app = { callServerTool: jest.fn().mockResolvedValue(packet([event])), updateModelContext: jest.fn() };
    const transport = createPanelTransport(app, 'panel', jest.fn(), jest.fn()), listener = jest.fn();
    transport.onMessage(listener); await transport.connect(); await tick();
    expect(listener).not.toHaveBeenCalled(); expect(transport.status.phase).toBe('ended'); transport.dispose();
  }
});
