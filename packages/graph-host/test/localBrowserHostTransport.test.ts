import * as assert from 'node:assert/strict';

import {
  LocalBrowserHostTransport,
  LocalBrowserSessionError,
  createLocalBrowserHostTransport,
  mintLocalBrowserSession,
  mintLocalBrowserViewLink,
} from '@tensorviz/graph-host/localBrowserHostTransport';
import { LocalBrowserHostTransportOptions } from '@tensorviz/graph-host/localBrowserHostTransport';
import {
  LocalWebSocket,
  LocalWebSocketCloseEvent,
  LocalWebSocketHandlers,
} from '@tensorviz/graph-host/localBrowserSocket';
import {
  LOCAL_BROWSER_COOKIE_NAME,
  LOCAL_BROWSER_PROTOCOL_VERSION,
  LOCAL_BROWSER_SESSIONS_PATH,
  LOCAL_BROWSER_WEBSOCKET_PATH,
  decodeLocalBrowserFragment,
} from '@tensorviz/contracts/protocol';

/**
 * The host transport against a scripted fake socket and a scripted fetch — the
 * broker's wire behavior (handshake, ordering epoch, terminal close codes) is
 * replayed exactly as `backend/services/ide/local_browser_broker.py` does
 * it, and the Python broker suite holds that file against the real contract.
 */

const SESSION_ID = 'ws-session-1';
const ORIGIN = 'http://127.0.0.1:4000';

const CAPABILITIES = {
  readOnly: true,
  canNavigateSource: true,
  canWriteSource: false,
  canRunAgent: false,
  canEditGraph: false,
};

const MINT_BODY = {
  workspaceSessionId: SESSION_ID,
  launchOrigin: ORIGIN,
  expiresAt: '2026-09-08T20:00:00Z',
  protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
  capabilities: CAPABILITIES,
  viewCredential: 'view-secret',
  hostCookieKey: 'host-secret',
  cookieName: LOCAL_BROWSER_COOKIE_NAME,
};

const TARGET = {
  workspaceFolder: 'workspace-a',
  file: 'model.py',
  cls: 'Net',
  documentVersion: 3,
  sourceDigest: 'a'.repeat(64),
  backendRevision: 7,
};

function mintResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

suite('mintLocalBrowserSession', () => {
  test('posts launch origin to the mint route with the per-launch token', async () => {
    const calls: Array<{ url: string; init: RequestInit }> = [];
    const material = await mintLocalBrowserSession({
      backendBaseUrl: 'http://127.0.0.1:4000/',
      launchToken: 'launch-secret',
      launchOrigin: ORIGIN,
      ttlSeconds: 60,
      fetchImpl: async (url, init) => {
        calls.push({ url: String(url), init: init ?? {} });
        return mintResponse(200, MINT_BODY);
      },
    });
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, `http://127.0.0.1:4000${LOCAL_BROWSER_SESSIONS_PATH}`);
    const headers = calls[0].init.headers as Record<string, string>;
    assert.equal(headers['X-tensorViz-Token'], 'launch-secret');
    assert.deepEqual(JSON.parse(String(calls[0].init.body)), { launchOrigin: ORIGIN, ttlSeconds: 60 });
    assert.equal(material.workspaceSessionId, SESSION_ID);
    assert.equal(material.hostCookieKey, 'host-secret');
    assert.equal(material.viewCredential, 'view-secret');
  });

  test('the page URL carries the session in the fragment, and nothing else secret', () => {
    return mintLocalBrowserSession({
      backendBaseUrl: ORIGIN,
      launchToken: 'launch-secret',
      launchOrigin: ORIGIN,
      fetchImpl: async () => mintResponse(200, MINT_BODY),
    }).then((material) => {
      const url = new URL(material.pageUrl);
      assert.equal(url.origin, ORIGIN);
      assert.equal(url.pathname, '/');
      assert.equal(url.search, '');
      assert.deepEqual(decodeLocalBrowserFragment(url.hash), {
        workspaceSessionId: SESSION_ID,
        credential: 'view-secret',
      });
      // The host cookie key never enters the browser flow.
      assert.ok(!material.pageUrl.includes('host-secret'));
    });
  });

  test('a 401 says the token was rejected and a 404 says the backend is not ours', async () => {
    const unauthorized = await mintLocalBrowserSession({
      backendBaseUrl: ORIGIN,
      launchToken: 'launch-secret',
      launchOrigin: ORIGIN,
      fetchImpl: async () => mintResponse(401, { detail: 'nope' }),
    }).catch((error) => error);
    assert.ok(unauthorized instanceof LocalBrowserSessionError);
    assert.equal(unauthorized.status, 401);
    assert.match(unauthorized.message, /token/);
    assert.ok(!unauthorized.message.includes('launch-secret'));

    const missing = await mintLocalBrowserSession({
      backendBaseUrl: ORIGIN,
      launchToken: 'launch-secret',
      launchOrigin: ORIGIN,
      fetchImpl: async () => mintResponse(404, {}),
    }).catch((error) => error);
    assert.equal(missing.status, 404);
    assert.match(missing.message, /extension mode/);
  });

  test('a malformed mint response is a loud contract error, not an undefined session', async () => {
    const error = await mintLocalBrowserSession({
      backendBaseUrl: ORIGIN,
      launchToken: 'launch-secret',
      launchOrigin: ORIGIN,
      fetchImpl: async () => mintResponse(200, { workspaceSessionId: SESSION_ID }),
    }).catch((mintError) => mintError);
    assert.ok(error instanceof LocalBrowserSessionError);
    assert.match(error.message, /contract/);

    const unreachable = await mintLocalBrowserSession({
      backendBaseUrl: ORIGIN,
      launchToken: 'launch-secret',
      launchOrigin: ORIGIN,
      fetchImpl: async () => {
        throw new Error('ECONNREFUSED');
      },
    }).catch((mintError) => mintError);
    assert.ok(unreachable instanceof LocalBrowserSessionError);
    assert.equal(unreachable.status, null);
  });
});

class FakeSocket implements LocalWebSocket {
  sent: string[] = [];
  closeCalls: Array<{ code: number; reason: string }> = [];
  closed = false;

  constructor(readonly url: string, readonly handlers: LocalWebSocketHandlers, readonly headers: Record<string, string>) {}

  sendText(text: string): boolean {
    if (this.closed) return false;
    this.sent.push(text);
    return true;
  }

  close(code = 1000, reason = ''): void {
    this.closed = true;
    this.closeCalls.push({ code, reason });
  }

  /** Test drivers replaying the broker's wire behavior. */
  driverOpen(): void {
    this.handlers.onOpen();
  }
  driverMessage(message: unknown): void {
    this.handlers.onText(JSON.stringify(message));
  }
  driverClose(event: LocalWebSocketCloseEvent): void {
    this.handlers.onClose(event);
  }
}

class FakeBroker {
  sockets: FakeSocket[] = [];

  connectSocket: NonNullable<LocalBrowserHostTransportOptions['connectSocket']> = (
    url,
    handlers,
    options,
  ) => {
    const socket = new FakeSocket(url, handlers, options.headers ?? {});
    this.sockets.push(socket);
    return socket;
  };

  get last(): FakeSocket {
    const socket = this.sockets[this.sockets.length - 1];
    assert.ok(socket, 'expected a socket attempt');
    return socket;
  }

  driverAcceptHost(socket: FakeSocket): void {
    socket.driverMessage({
      type: 'connected',
      protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
      workspaceSessionId: SESSION_ID,
      messageId: 'broker-1',
      role: 'host',
      session: {
        workspaceSessionId: SESSION_ID,
        role: 'host',
        protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
        capabilities: CAPABILITIES,
        expiresAt: '2026-09-08T20:00:00Z',
      },
    });
  }
}

// Reconnect timers in the transport are setTimeout(..., 0..2ms), and Node runs
// timer phases only once a threshold has elapsed — a settle() made of bare
// setImmediate rounds can complete ten loop turns in under a millisecond, so
// the timer never fires and the test races itself. Real-time rounds instead.
async function settle(rounds = 5): Promise<void> {
  for (let i = 0; i < rounds; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 2));
  }
}

function makeTransport(launchOrigin = ORIGIN): {
  broker: FakeBroker;
  transport: LocalBrowserHostTransport;
  statuses: Array<{ phase: string; epoch: number; error: string | null }>;
  received: unknown[];
  logs: string[];
} {
  const broker = new FakeBroker();
  const statuses: Array<{ phase: string; epoch: number; error: string | null }> = [];
  const received: unknown[] = [];
  const logs: string[] = [];
  let id = 0;
  const transport = createLocalBrowserHostTransport({
    backendBaseUrl: ORIGIN,
    session: {
      workspaceSessionId: SESSION_ID,
      launchOrigin,
      expiresAt: '2026-09-08T20:00:00Z',
      hostCookieKey: 'host-secret',
      viewCredential: 'view-secret',
      pageUrl: `${launchOrigin}/#session=${SESSION_ID}&credential=view-secret`,
    },
    connectSocket: broker.connectSocket,
    messageId: () => `test-mid-${++id}`,
    reconnectDelaysMs: [0, 0, 0, 0],
    log: (line) => logs.push(line),
  });
  transport.onStatusChange((status) => statuses.push({ ...status }));
  transport.onMessage((message) => received.push(message));
  return { broker, transport, statuses, received, logs };
}

suite('localBrowserHostTransport', () => {
  test('SSH host socket stays on remote loopback with the exact desktop launch origin', async () => {
    const external = 'http://127.0.0.1:53128';
    const { broker, transport } = makeTransport(external);
    const connected = transport.connect();
    broker.last.driverOpen(); broker.driverAcceptHost(broker.last); await connected;
    assert.equal(broker.last.url, `ws://127.0.0.1:4000${LOCAL_BROWSER_WEBSOCKET_PATH}`);
    assert.equal(broker.last.headers.Origin, external);
    assert.equal(broker.last.headers.Cookie, `${LOCAL_BROWSER_COOKIE_NAME}=host-secret`);
    transport.dispose();
  });
  test('connects to the broker socket with the host cookie and exact origin headers', async () => {
    const { broker, transport } = makeTransport();
    const connected = transport.connect();
    broker.last.driverOpen();
    broker.driverAcceptHost(broker.last);
    await connected;

    assert.equal(broker.last.url, `ws://127.0.0.1:4000${LOCAL_BROWSER_WEBSOCKET_PATH}`);
    assert.equal(broker.last.headers['Cookie'], `${LOCAL_BROWSER_COOKIE_NAME}=host-secret`);
    assert.equal(broker.last.headers['Origin'], ORIGIN);
    const handshake = JSON.parse(broker.last.sent[0]);
    assert.deepEqual(handshake, {
      type: 'hostHandshake',
      viewControl: true,
      protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
      workspaceSessionId: SESSION_ID,
      messageId: 'test-mid-1',
      role: 'host',
    });
    assert.equal(transport.status.phase, 'connected');
    assert.equal(transport.status.epoch, 1);
    transport.dispose();
  });

  test('publish owns envelope and strictly increasing hostSequence across types', async () => {
    const { broker, transport } = makeTransport();
    const connected = transport.connect();
    broker.last.driverOpen();
    broker.driverAcceptHost(broker.last);
    await connected;

    const first = transport.publish({ type: 'activeTarget', target: TARGET });
    const second = transport.publish({
      type: 'projection',
      target: TARGET,
      projection: { nodes: [] },
      traced: false,
      greyed: false,
    });
    const third = transport.publish({ type: 'ack', target: TARGET, ackId: 'm-12', ok: true });
    // Source-edit replies are per-request traffic, not ordered state: they ride the
    // envelope without a hostSequence.
    const fourth = transport.publish({
      type: 'sourceEditResult',
      target: TARGET,
      requestId: 'ask-1',
      status: 'applied', detail: 'the authoritative explanation',
    });
    const fifth = transport.publish({
      type: 'validateResult',
      target: TARGET,
      requestId: 'ask-2',
      status: 'refused',
      detail: 'the backend did not answer the question',
    });

    assert.ok(first && second && third && fourth && fifth);
    const wire = broker.last.sent.slice(1).map((text) => JSON.parse(text));
    assert.deepEqual(
      wire.map((message) => [message.type, message.hostSequence ?? null, message.messageId]),
      [
        ['activeTarget', 1, 'test-mid-2'],
        ['projection', 2, 'test-mid-3'],
        ['ack', null, 'test-mid-4'],
        ['sourceEditResult', null, 'test-mid-5'],
        ['validateResult', null, 'test-mid-6'],
      ],
    );
    for (const message of wire) {
      assert.equal(message.protocolVersion, 1);
      assert.equal(message.workspaceSessionId, SESSION_ID);
    }
    transport.dispose();
  });

  test('snapshot gets the host role marker; an impossible publish is a local bug and throws', async () => {
    const { broker, transport } = makeTransport();
    const connected = transport.connect();
    broker.last.driverOpen();
    broker.driverAcceptHost(broker.last);
    await connected;

    const snapshot = transport.publish({
      type: 'snapshot',
      state: {
        activeTarget: null,
        projection: null,
        traced: false,
        greyed: false,
        syncState: null,
        connectionState: 'connected',
      },
    });
    assert.ok(snapshot);
    const wire = JSON.parse(broker.last.sent[1]);
    assert.equal(wire.role, 'host');
    assert.equal(wire.hostSequence, 1);

    // Empty reason, empty ackId, or a malformed digest violate the contract;
    // the transport throws at the call site rather than sending a message the
    // broker would refuse.
    assert.throws(() =>
      transport.publish({ type: 'syncState', target: TARGET, state: 'stale', reason: '' }),
    );
    assert.throws(() => transport.publish({ type: 'ack', target: TARGET, ackId: '', ok: true }));
    assert.throws(() =>
      transport.publish({ type: 'activeTarget', target: { ...TARGET, sourceDigest: 'short' } }),
    );
    assert.throws(() =>
      transport.publish({ type: 'sourceEditResult', target: TARGET, requestId: '', status: 'applied', detail: 'late' }),
    );
    transport.dispose();
  });

  test('publish returns null while not connected rather than queueing across epochs', async () => {
    const { broker, transport } = makeTransport();
    assert.equal(transport.publish({ type: 'activeTarget', target: TARGET }), null);
    const connected = transport.connect();
    broker.last.driverOpen();
    broker.driverAcceptHost(broker.last);
    await connected;

    broker.last.driverClose({ kind: 'closed', code: 1006, reason: '' });
    await settle();
    assert.equal(transport.status.phase, 'reconnecting');
    assert.equal(transport.publish({ type: 'activeTarget', target: TARGET }), null);
    transport.dispose();
  });

  test('reconnect restarts the ordering epoch after broker accept', async () => {
    const { broker, transport, statuses } = makeTransport();
    const connected = transport.connect();
    broker.last.driverOpen();
    broker.driverAcceptHost(broker.last);
    await connected;
    transport.publish({ type: 'activeTarget', target: TARGET });
    transport.publish({ type: 'focusNode', target: TARGET, nodeId: 'n1' });

    broker.last.driverClose({ kind: 'closed', code: 1006, reason: '' });
    await settle();
    assert.equal(broker.sockets.length, 2);
    broker.last.driverOpen();
    broker.driverAcceptHost(broker.last);
    await settle();

    assert.equal(transport.status.phase, 'connected');
    assert.equal(transport.status.epoch, 2);
    transport.publish({ type: 'focusNode', target: TARGET, nodeId: 'n2' });
    const wire = JSON.parse(broker.last.sent[1]); // after this socket's handshake frame
    assert.equal(wire.hostSequence, 1, 'a new epoch restarts hostSequence');
    assert.deepEqual(
      statuses.map((status) => status.phase),
      ['connecting', 'connected', 'reconnecting', 'connected'],
    );
    transport.dispose();
  });

  test('model explanation progress publishes and start/cancel reach the host without sequencing', async () => {
    const { broker, transport, received } = makeTransport();
    const connected = transport.connect();
    broker.last.driverOpen(); broker.driverAcceptHost(broker.last); await connected;
    try {
      const state = { requestId: 'explain-model', revision: 'revision', status: 'loading' as const, completed: 0, total: 2 };
      const published = transport.publish({ type: 'modelExplanation', viewId: 'canvas-1', target: TARGET, state });
      assert.ok(published && published.type === 'modelExplanation');
      assert.deepEqual(published.state, state);
      assert.equal('hostSequence' in published, false);
      assert.deepEqual(JSON.parse(broker.last.sent[1]), published);
      const envelope = { protocolVersion: 1, workspaceSessionId: SESSION_ID, messageId: 'explain-start', viewId: 'canvas-1', target: TARGET, requestId: 'explain-model' };
      const start = { ...envelope, type: 'modelExplanationRequest', action: 'start', nodeIds: ['encoder', 'head'] };
      const cancel = { ...start, messageId: 'explain-cancel', action: 'cancel', nodeIds: [] };
      broker.last.driverMessage(start); broker.last.driverMessage(cancel);
      assert.deepEqual(received, [start, cancel]);
      broker.last.driverMessage({ ...start, workspaceSessionId: 'another-session' });
      broker.last.driverMessage({ ...start, nodeIds: ['encoder', 'encoder'] });
      assert.deepEqual(received, [start, cancel]);
      assert.equal(transport.protocolViolations, 2);
    } finally { transport.dispose(); }
  });

  test('MCP view discovery, selection and focus receipts reach the host registry', async () => {
    const { broker, transport, received } = makeTransport();
    const connected = transport.connect();
    broker.last.driverOpen();
    broker.driverAcceptHost(broker.last);
    await connected;
    const envelope = { protocolVersion: 1, workspaceSessionId: SESSION_ID, messageId: 'view-event', viewId: 'canvas-1' };
    const frames = [
      { ...envelope, type: 'viewState', connected: true },
      { ...envelope, type: 'viewSelection', target: TARGET, nodeIds: ['fc'], edgeIds: [] },
      { ...envelope, type: 'viewFocusResult', target: TARGET, requestId: 'focus-1', ok: true, detail: 'Focused.' },
      { ...envelope, type: 'graphReviewApplied', reviewId: 'review-1', mode: 'diff' },
      { protocolVersion: 1, workspaceSessionId: SESSION_ID, messageId: 'review-close', type: 'graphReviewRequest', reviewId: 'review-1', action: 'close' },
    ];
    try {
      for (const frame of frames) broker.last.driverMessage(frame);
      assert.deepEqual(received, frames, 'View tools must receive the broker roster and canvas receipts');
      broker.last.driverMessage({ ...frames[1], workspaceSessionId: 'another-window' });
      broker.last.driverMessage({ ...envelope, type: 'focusView', target: TARGET, requestId: 'focus-2', nodeIds: ['fc'], edgeIds: [] });
      assert.deepEqual(received, frames, 'Reject cross-window and outbound-only focus commands');
      assert.equal(transport.protocolViolations, 2);
    } finally { transport.dispose(); }
  });

  test('MCP tab presence clears the initial empty roster without hiding registry events', async () => {
    const { broker, transport, received } = makeTransport();
    const connected = transport.connect();
    broker.last.driverOpen();
    broker.driverAcceptHost(broker.last);
    await connected;
    const envelope = { type: 'viewState', protocolVersion: 1, workspaceSessionId: SESSION_ID };
    try {
      broker.last.driverMessage({ ...envelope, messageId: 'empty', state: 'detached' });
      assert.equal(transport.viewPresence, 'detached');
      broker.last.driverMessage({ ...envelope, messageId: 'first', viewId: 'one', connected: true });
      assert.equal(transport.viewPresence, 'attached', 'Opening another model should reuse the connected canvas');
      broker.last.driverMessage({ ...envelope, messageId: 'second', viewId: 'two', connected: true });
      broker.last.driverMessage({ ...envelope, messageId: 'first-closed', viewId: 'one', connected: false });
      assert.equal(transport.viewPresence, 'attached', 'Closing one of two tabs must not mark the roster empty');
      broker.last.driverMessage({ ...envelope, messageId: 'last-closed', viewId: 'two', connected: false });
      broker.last.driverMessage({ ...envelope, messageId: 'empty-again', state: 'detached' });
      assert.equal(transport.viewPresence, 'detached');
      assert.equal(received.length, 4, 'All per-tab changes still reach the MCP registry');
      assert.equal(transport.protocolViolations, 0);
    } finally { transport.dispose(); }
  });

  test('view-originated messages arrive; invalid or misdirected frames are violations', async () => {
    const { broker, transport, received } = makeTransport();
    const connected = transport.connect();
    broker.last.driverOpen();
    broker.driverAcceptHost(broker.last);
    await connected;

    broker.last.driverMessage({
      type: 'revealNode',
      protocolVersion: 1,
      workspaceSessionId: SESSION_ID,
      messageId: 'v-1',
      target: TARGET,
      nodeId: 'node-1',
    });
    broker.last.driverMessage({
      type: 'snapshotRequest',
      protocolVersion: 1,
      workspaceSessionId: SESSION_ID,
      messageId: 'v-2',
      role: 'view',
      lastHostSequence: 0,
    });
    broker.last.driverMessage({
      type: 'sourceEditRequest',
      protocolVersion: 1,
      workspaceSessionId: SESSION_ID,
      messageId: 'v-3',
      target: TARGET,
      requestId: 'ask-1',
      edit: { kind: 'set_param', nodeId: 'enc1', param: 'out_features', value: 8 },
    });
    broker.last.driverMessage({
      type: 'sourceEditCancel',
      protocolVersion: 1,
      workspaceSessionId: SESSION_ID,
      messageId: 'v-4',
      target: TARGET,
      requestId: 'ask-1',
    });
    for (const type of ['validateRequest', 'validateCancel']) {
      broker.last.driverMessage({ type, protocolVersion: 1, workspaceSessionId: SESSION_ID,
        messageId: `v-${type}`, target: TARGET, requestId: 'validation-1' });
    }
    assert.equal(received.length, 6);
    assert.equal((received[4] as { type: string }).type, 'validateRequest');
    assert.equal((received[5] as { type: string }).type, 'validateCancel');
    assert.equal((received[0] as { type: string }).type, 'revealNode');
    assert.equal((received[1] as { type: string }).type, 'snapshotRequest');
    assert.equal((received[2] as { type: string }).type, 'sourceEditRequest');
    assert.equal((received[3] as { type: string }).type, 'sourceEditCancel');

    // Cross-session, malformed JSON, and host-own types relayed to the host.
    broker.last.driverMessage({ type: 'revealNode', protocolVersion: 1, workspaceSessionId: 'other', messageId: 'x', target: TARGET, nodeId: 'n' });
    broker.last.handlers.onText('not json');
    broker.last.driverMessage({
      type: 'projection',
      protocolVersion: 1,
      workspaceSessionId: SESSION_ID,
      messageId: 'x-2',
      hostSequence: 99,
      target: TARGET,
      projection: {},
      traced: false,
      greyed: false,
    });
    // An ask reply is host-own traffic: a broker handing one TO the host is a
    // routing bug, never a deliverable message.
    broker.last.driverMessage({
      type: 'sourceEditResult',
      protocolVersion: 1,
      workspaceSessionId: SESSION_ID,
      messageId: 'x-3',
      target: TARGET,
      requestId: 'ask-1',
      status: 'applied', detail: 'misdirected',
    });
    assert.equal(received.length, 6, 'nothing invalid or misdirected is delivered');
    assert.equal(transport.protocolViolations, 4);
    transport.dispose();
  });

  test('a broker terminal close ends the transport without reconnecting', async () => {
    const { broker, transport, logs } = makeTransport();
    const connected = transport.connect();
    broker.last.driverOpen();
    broker.driverAcceptHost(broker.last);
    await connected;

    broker.last.driverClose({ kind: 'closed', code: 4402, reason: 'CREDENTIAL_EXPIRED' });
    await settle();
    assert.equal(transport.status.phase, 'ended');
    assert.equal(transport.status.error, 'CREDENTIAL_EXPIRED');
    assert.equal(broker.sockets.length, 1, 'no reconnect after expiry');
    assert.ok(logs.some((line) => line.includes('CREDENTIAL_EXPIRED')));
  });

  test('an authentication upgrade refusal ends the session', async () => {
    const { broker, transport } = makeTransport();
    const connected = transport.connect();
    broker.last.driverClose({ kind: 'refused', status: 403 });
    await assert.rejects(connected, /INVALID_CREDENTIALS/);
    assert.equal(transport.status.phase, 'ended');
    assert.equal(transport.status.error, 'INVALID_CREDENTIALS');
    transport.dispose();
  });

  for (const status of [404, 500, 101, null]) {
    test(`upgrade failure ${status} explains the connection problem without blaming credentials`, async () => {
      const { broker, transport, logs } = makeTransport();
      const connected = transport.connect();
      broker.last.driverClose({ kind: 'refused', status });
      await assert.rejects(connected, error => {
        assert.ok(error instanceof Error);
        assert.match(error.message, /WebSocket/);
        assert.match(error.message, /Python Environment|Repair runtime/);
        assert.ok(!error.message.includes('INVALID_CREDENTIALS'));
        if (status !== null) assert.ok(error.message.includes(`HTTP ${status}`));
        return true;
      });
      assert.equal(transport.status.phase, 'ended');
      assert.equal(transport.status.error, 'WEBSOCKET_UNAVAILABLE');
      assert.equal(broker.sockets.length, 1);
      assert.ok(!logs.join(' ').includes('host-secret'));
      transport.dispose();
    });
  }

  test('dispose closes politely once and ignores a late socket close', async () => {
    const { broker, transport } = makeTransport();
    const connected = transport.connect();
    broker.last.driverOpen();
    broker.driverAcceptHost(broker.last);
    await connected;

    transport.dispose();
    transport.dispose();
    assert.equal(transport.status.phase, 'ended');
    assert.equal(transport.status.error, null);
    assert.deepEqual(broker.last.closeCalls, [{ code: 1000, reason: 'host transport ended' }]);

    broker.last.driverClose({ kind: 'closed', code: 1006, reason: '' });
    await settle();
    assert.equal(broker.sockets.length, 1, 'dispose blocks reconnect');
  });

  test('dispose before the first connect settles the connect promise as rejected', async () => {
    const { transport } = makeTransport();
    const pending = transport.connect();
    transport.dispose();
    await assert.rejects(pending, /disposed/);
  });

  test('the broker\'s view roster reaches the host and never counts as a violation', async () => {
    // The reopen bug: the host cannot see a browser tab close, because closing it
    // leaves the host's own socket connected. `viewState` is the only way it learns,
    // and it is broker-owned and host-bound — so it must be consumed here rather than
    // forwarded to view-message listeners or counted as misdirected.
    const { broker, transport, received } = makeTransport();
    const connected = transport.connect();
    broker.last.driverOpen();
    broker.driverAcceptHost(broker.last);
    await connected;

    assert.equal(transport.viewPresence, 'unknown', 'nothing said yet is not "the tab is gone"');

    broker.last.driverMessage({
      type: 'viewState',
      protocolVersion: 1,
      workspaceSessionId: SESSION_ID,
      messageId: 'vs-1',
      state: 'detached',
    });
    assert.equal(transport.viewPresence, 'detached');

    broker.last.driverMessage({
      type: 'viewState',
      protocolVersion: 1,
      workspaceSessionId: SESSION_ID,
      messageId: 'vs-2',
      state: 'attached',
    });
    assert.equal(transport.viewPresence, 'attached');

    assert.equal(transport.protocolViolations, 0, 'a host-bound broker frame is not misdirected');
    assert.equal(received.length, 0, 'and it is not view traffic either');
    transport.dispose();
  });
});

suite('agent browser link mint', () => {
  const session = { ...MINT_BODY, pageUrl: `${ORIGIN}/#root-secret` };
  const good = () => ({ workspaceSessionId: SESSION_ID, launchOrigin: ORIGIN, handoffId: 'handoff-1234567890',
    viewCredential: 'single-use-1234567890', expiresAt: new Date(Date.now() + 120000).toISOString() });
  test('uses the launch token privately, rejects redirects and returns only scoped view material', async () => {
    const link = await mintLocalBrowserViewLink({ backendBaseUrl: ORIGIN, launchToken: 'launch-secret', session,
      signal: new AbortController().signal, fetchImpl: async (url, init) => {
        assert.equal(String(url), `${ORIGIN}/local/sessions/${SESSION_ID}/view-links`);
        assert.equal(init?.redirect, 'error');
        assert.equal((init?.headers as any)['X-tensorViz-Token'], 'launch-secret');
        return mintResponse(200, good());
      } });
    assert.match(link.pageUrl, /single-use/);
    assert.doesNotMatch(JSON.stringify(link), /launch-secret|host-secret|view-secret|root-secret/);
  });
  test('refuses mismatched sessions, origins, expiry and malformed credentials without echoing them', async () => {
    for (const override of [{ workspaceSessionId: 'other' }, { launchOrigin: 'https://remote.example' },
      { expiresAt: new Date(0).toISOString() }, { viewCredential: 'secret?bad' }, { handoffId: '' }]) {
      await assert.rejects(mintLocalBrowserViewLink({ backendBaseUrl: ORIGIN, launchToken: 'launch-secret', session,
        signal: new AbortController().signal, fetchImpl: async () => mintResponse(200, { ...good(), ...override }) }),
        error => error instanceof LocalBrowserSessionError && !/secret|remote.example|other/.test(error.message));
    }
  });
});
