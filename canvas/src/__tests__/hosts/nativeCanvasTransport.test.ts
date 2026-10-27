import { createNativeCanvasTransport } from '../../ide/nativeCanvasTransport';
import { LOCAL_BROWSER_PROTOCOL_VERSION } from '../../ide/protocol.generated';
import { captureNativePresentation, restoreNativePresentation } from '../../ide/nativeCanvasPresentation';
import { configureCanvasTheme, getCanvasTheme, setCanvasThemePreference } from '../../styles/canvasTheme';
jest.mock('../../ide/nativeCanvasPresentation', () => ({ restoreNativePresentation: jest.fn(), captureNativePresentation: jest.fn() }));

const receive = (data: unknown, origin = 'vscode-webview://canvas', source: Window | null = window.parent) =>
  window.dispatchEvent(new MessageEvent('message', { data, origin, source }));
const envelope = { protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION, workspaceSessionId: 'native', messageId: 'm1' };

test('host theme comes only from a connected parent and respects the user override', async () => {
  localStorage.clear(); configureCanvasTheme('host');
  const post = jest.spyOn(window.parent, 'postMessage').mockImplementation(() => {});
  const transport = createNativeCanvasTransport();
  try {
    receive({ type: 'canvasTheme', theme: 'light' });
    expect(getCanvasTheme()).toBe('dark');
    const connected = transport.connect();
    receive({ type: 'canvasConnected', sessionId: 'native', viewId: 'v', theme: 'light' });
    await connected;
    expect(getCanvasTheme()).toBe('light');
    receive({ type: 'canvasTheme', theme: 'dark' }, 'https://other.example');
    receive({ type: 'canvasTheme', theme: 'dark' }, 'vscode-webview://canvas', null);
    receive({ type: 'canvasTheme', theme: 'invalid' });
    expect(getCanvasTheme()).toBe('light');
    receive({ type: 'canvasTheme', theme: 'dark' });
    expect(getCanvasTheme()).toBe('dark');
    setCanvasThemePreference('light');
    receive({ type: 'canvasTheme', theme: 'dark' });
    expect(getCanvasTheme()).toBe('light');
  } finally { transport.dispose(); post.mockRestore(); localStorage.clear(); }
});

test('native connection uses the parent bridge, validates the session and disposes without cancelling work', async () => {
  const post = jest.spyOn(window.parent, 'postMessage').mockImplementation(() => {});
  const transport = createNativeCanvasTransport();
  const messages: unknown[] = [];
  transport.onMessage(message => messages.push(message));
  const connected = transport.connect();
  expect(transport.connect()).toBe(connected);
  expect(post).toHaveBeenCalledWith({ type: 'canvasConnect', protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION }, '*');
  receive({ type: 'canvasConnected', sessionId: 'wrong' }, 'https://attacker.example', null);
  expect(transport.session).toBeNull();
  receive({ type: 'canvasConnected', sessionId: 'native', viewId: 'native-view', presentation: { camera: 'retained' } });
  await connected;
  expect(restoreNativePresentation).toHaveBeenCalledWith({ camera: 'retained' });
  const message = { ...envelope, type: 'snapshot', role: 'host', hostSequence: 1,
    state: { activeTarget: null, projection: null, traced: false, greyed: false, syncState: null, connectionState: 'connected' } };
  receive({ type: 'canvasMessage', message }, 'https://attacker.example');
  receive({ type: 'canvasMessage', message: { ...message, workspaceSessionId: 'other' } });
  expect(messages).toEqual([]);
  expect(transport.protocolViolations).toBe(1);
  receive({ type: 'canvasMessage', message });
  expect(messages).toEqual([message]);
  post.mockClear();
  const target = { workspaceFolder: '/ws', file: 'net.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 2 };
  expect(transport.send({ type: 'validateRequest', requestId: 'manual', target })).not.toBeNull();
  expect(post.mock.calls[0][1]).toBe('vscode-webview://canvas');
  post.mockClear(); transport.dispose();
  expect(post).not.toHaveBeenCalled();
  expect(transport.send({ type: 'validateCancel', requestId: 'manual', target })).toBeNull();
  receive({ type: 'canvasMessage', message });
  expect(messages).toEqual([message]);
  post.mockRestore();
});

test('a missing host times out visibly and dispose cancels the handshake timer', async () => {
  jest.useFakeTimers();
  const post = jest.spyOn(window.parent, 'postMessage').mockImplementation(() => {});
  const transport = createNativeCanvasTransport();
  const failed = expect(transport.connect()).rejects.toThrow('did not answer');
  jest.advanceTimersByTime(15_000); await failed;
  expect(transport.status.phase).toBe('ended');
  transport.dispose(); post.mockRestore(); jest.useRealTimers();
});

test('native MCP selection and exact focus receipt use the assigned view; showing requests fresh selection only', async () => {
  const { createLocalBrowserViewControl } = await import('../../ide/localBrowserViewControl');
  const post = jest.spyOn(window.parent, 'postMessage').mockImplementation(() => {});
  const transport = createNativeCanvasTransport();
  const target = { workspaceFolder: '/ws', file: 'net.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 2 };
  const control = createLocalBrowserViewControl(transport, () => target);
  let finish: ((ok: boolean) => void) | undefined;
  const events: string[] = [];
  const builder = (event: Event) => {
    const detail = (event as CustomEvent).detail;
    events.push(detail.command);
    if (detail.command === 'focus-selection') finish = detail.payload.finish;
    if (detail.command === 'report-selection') window.dispatchEvent(new CustomEvent('tensorviz-selection', { detail: { nodeIds: ['n'], edgeIds: [] } }));
  };
  window.addEventListener('builder-command', builder);
  try {
    const connected = transport.connect();
    receive({ type: 'canvasConnected', sessionId: 'native' });
    expect(transport.session).toBeNull();
    receive({ type: 'canvasConnected', sessionId: 'native', viewId: 'assigned' });
    await connected;
    expect(transport.session?.viewId).toBe('assigned');
    const sent = () => post.mock.calls.map(call => call[0]?.message).filter(Boolean);
    expect(sent()).toEqual([expect.objectContaining({ type: 'viewSelection', viewId: 'assigned', target, nodeIds: ['n'] })]);
    post.mockClear(); events.length = 0;
    receive({ type: 'canvasVisible' });
    expect(events).toEqual(['report-selection']);
    expect(sent()[0]).toMatchObject({ type: 'viewSelection', viewId: 'assigned' });
    post.mockClear(); events.length = 0;
    const request = { ...envelope, type: 'focusView', target, requestId: 'focus', viewId: 'other', nodeIds: ['n'], edgeIds: [] };
    receive({ type: 'canvasMessage', message: request });
    expect(events).toEqual([]);
    receive({ type: 'canvasMessage', message: { ...request, viewId: 'assigned' } });
    expect(events).toEqual(['focus-selection']);
    expect(sent()).toEqual([]);
    finish!(true);
    expect(sent()).toEqual([expect.objectContaining({ type: 'viewFocusResult', viewId: 'assigned', target, requestId: 'focus', ok: true })]);
  } finally {
    window.removeEventListener('builder-command', builder);
    control.dispose(); transport.dispose(); post.mockRestore();
  }
});

test('checkpoint captures current presentation only for the authenticated parent', async () => {
  const post = jest.spyOn(window.parent, 'postMessage').mockImplementation(() => {});
  const transport = createNativeCanvasTransport();
  const connected = transport.connect();
  receive({ type: 'canvasConnected', sessionId: 'native', viewId: 'v' });
  await connected;
  post.mockClear();
  (captureNativePresentation as jest.Mock).mockReturnValue({ graphId: 'latest' });
  receive({ type: 'canvasCheckpoint', requestId: 'bad' }, 'https://other.example');
  receive({ type: 'canvasCheckpoint', requestId: 'bad' }, 'vscode-webview://canvas', null);
  expect(post).not.toHaveBeenCalled();
  receive({ type: 'canvasCheckpoint', requestId: 'hide' });
  expect(post).toHaveBeenCalledWith({ type: 'canvasPresentation', requestId: 'hide', state: { graphId: 'latest' } }, 'vscode-webview://canvas');
  transport.dispose(); post.mockRestore();
});
