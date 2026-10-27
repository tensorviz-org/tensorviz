/**
 * The browser view transport against a scripted fake socket and a scripted
 * fetch — the broker's wire behavior (handshake acceptance, ordering epochs,
 * terminal close codes) is replayed exactly as
 * `backend/services/ide/local_browser_broker.py` does it, and the Python
 * broker suite holds that file against the real contract.
 *
 * The properties worth pinning here, in descending order of blast radius:
 *
 *  1. the fragment credential is erased from the address bar before it is
 *     traded, is traded exactly once, and is never persisted — only the
 *     session-routing id survives in storage, which is what makes a browser
 *     refresh able to re-attach;
 *  2. sequenced host state never renders out of order: everything is held
 *     until the snapshot restores one atomic state, replays at or below the
 *     watermark are dropped, and a host restart resets the epoch instead of
 *     mixing two hosts' sequence numbers;
 *  3. a terminal broker close code (the 44xx band) ends the page honestly and
 *     stops the reconnect loop, while a transient drop reconnects with the
 *     watermark intact, so a wifi blip cannot silently rewind the graph.
 */
import {
  LOCAL_BROWSER_BOOTSTRAP_PATH,
  LOCAL_BROWSER_CLOSE_CODES,
  LOCAL_BROWSER_MESSAGE_TYPES,
  LOCAL_BROWSER_PROTOCOL_VERSION,
  LOCAL_BROWSER_SEQUENCED_MESSAGE_TYPES,
  LOCAL_BROWSER_WEBSOCKET_PATH,
  encodeLocalBrowserFragment,
} from '../../ide/protocol.generated';
import {
  LOCAL_BROWSER_SESSION_STORAGE_KEY,
  LocalBrowserSocketLike,
  LocalBrowserViewLocation,
  LocalBrowserViewMessage,
  LocalBrowserViewStatus,
  createLocalBrowserViewTransport,
} from '../../ide/localBrowserViewTransport';
import { gzipSync } from 'node:zlib';

beforeAll(() => {
  Object.assign(global, { DecompressionStream: require('node:stream/web').DecompressionStream });
  global.TextDecoder = require('node:util').TextDecoder;
});

const compressed = (message: unknown) => JSON.stringify({
  encoding: 'tensorviz-gzip-json-v1', data: gzipSync(JSON.stringify(message)).toString('base64'),
});

const SESSION_ID = 'ws-session-1';
const CREDENTIAL = 'view-secret';

const CAPABILITIES = {
  readOnly: true,
  canNavigateSource: true,
  canWriteSource: false,
  canRunAgent: false,
  canEditGraph: false,
};

const TARGET = {
  workspaceFolder: 'workspace-a',
  file: 'model.py',
  cls: 'Net',
  documentVersion: 3,
  sourceDigest: 'a'.repeat(64),
  backendRevision: 7,
};

const EMPTY_STATE = {
  activeTarget: null,
  projection: null,
  traced: false,
  greyed: false,
  connectionState: 'connected',
  syncState: null,
};

function sessionInfo(overrides: Record<string, unknown> = {}) {
  return {
    workspaceSessionId: SESSION_ID,
    role: 'view',
    protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
    capabilities: CAPABILITIES,
    expiresAt: '2026-09-08T20:00:00Z',
    ...overrides,
  };
}

function frame(type: string, fields: Record<string, unknown>): Record<string, unknown> {
  return {
    type,
    protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
    workspaceSessionId: SESSION_ID,
    messageId: `srv-${type}`,
    ...fields,
  };
}

const connectedFrame = () => frame('connected', { role: 'view', session: sessionInfo() });
const snapshotFrame = (hostSequence: number) =>
  frame('snapshot', { role: 'host', hostSequence, state: EMPTY_STATE });
const projectionFrame = (hostSequence: number) =>
  frame('projection', { hostSequence, target: TARGET, projection: { nodes: [] }, traced: false, greyed: false });
const connectionStateFrame = (state: 'connected' | 'disconnected') =>
  frame('connectionState', { state });

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

class FakeSocket implements LocalBrowserSocketLike {
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onclose: ((event: { code: number; reason: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  sent: string[] = [];
  closes: Array<{ code?: number; reason?: string }> = [];

  constructor(readonly url: string) {}

  send(data: string): void {
    this.sent.push(data);
  }

  close(code?: number, reason?: string): void {
    this.closes.push({ code, reason });
  }

  serverOpen(): void {
    this.onopen?.();
  }

  serverSend(message: Record<string, unknown> | string): void {
    this.onmessage?.({ data: typeof message === 'string' ? message : JSON.stringify(message) });
  }

  serverClose(code: number, reason = ''): void {
    this.onclose?.({ code, reason });
  }
}

/** The transport's timers are real (backoff), so a jitter-free flush is made of
 *  real setTimeout rounds rather than bare microtask yields. */
async function settle(rounds = 5): Promise<void> {
  for (let round = 0; round < rounds; round += 1) {
    await new Promise((resolve) => setTimeout(resolve, 2));
  }
}

interface HarnessOptions {
  hash?: string;
  storedId?: string;
  fetchImpl?: (input: string, init?: RequestInit) => Promise<Response>;
}

function makeHarness(options: HarnessOptions = {}) {
  const events: string[] = [];
  const sockets: FakeSocket[] = [];
  const messages: LocalBrowserViewMessage[] = [];
  const statuses: LocalBrowserViewStatus[] = [];
  const store = new Map<string, string>();
  if (options.storedId !== undefined) {
    store.set(LOCAL_BROWSER_SESSION_STORAGE_KEY, options.storedId);
  }
  const where: LocalBrowserViewLocation = {
    hash: options.hash ?? '',
    origin: 'http://127.0.0.1:3000',
    pathname: '/',
    search: '',
  };
  let nextId = 0;
  const transport = createLocalBrowserViewTransport({
    location: () => where,
    replaceState: (url) => {
      events.push(`replace:${url}`);
      where.hash = '';
    },
    fetchImpl: async (input, init) => {
      events.push(`fetch:${input}`);
      if (typeof init?.body === 'string') events.push(`body:${init.body}`);
      return options.fetchImpl ? options.fetchImpl(input, init) : jsonResponse(200, sessionInfo());
    },
    connectSocket: (url) => {
      events.push(`socket:${url}`);
      const socket = new FakeSocket(url);
      sockets.push(socket);
      return socket;
    },
    storage: {
      getItem: (key) => (store.has(key) ? store.get(key)! : null),
      setItem: (key, value) => void store.set(key, value),
      removeItem: (key) => void store.delete(key),
    },
    messageId: () => `m-${(nextId += 1)}`,
    reconnectDelaysMs: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  });
  transport.onMessage((message) => messages.push(message));
  transport.onStatusChange((status) => statuses.push(status));
  return { transport, sockets, messages, statuses, events, store, where };
}

/** Connect and complete one handshake epoch; returns the handshake the broker saw. */
async function attach(harness: ReturnType<typeof makeHarness>) {
  const pending = harness.transport.connect();
  await settle();
  expect(harness.sockets).toHaveLength(1);
  harness.sockets[0].serverOpen();
  const handshake = JSON.parse(harness.sockets[0].sent[0]);
  harness.sockets[0].serverSend(connectedFrame());
  await pending;
  await settle();
  return handshake;
}

describe('local browser view transport', () => {
  it('keeps compressed and plain frames ordered and ignores inflation from a closed epoch', async () => {
    const harness = makeHarness({ storedId: SESSION_ID }); await attach(harness);
    const socket = harness.sockets[0];
    socket.serverSend(compressed(snapshotFrame(1)));
    socket.serverSend(projectionFrame(2));
    socket.serverSend(compressed(projectionFrame(3)));
    socket.serverSend(projectionFrame(4));
    await settle(20);
    expect(harness.messages.map(message => (message as { hostSequence: number }).hostSequence)).toEqual([1, 2, 3, 4]);
    socket.serverSend(compressed(projectionFrame(5)));
    socket.serverClose(1006);
    await settle(20);
    expect(harness.messages).toHaveLength(4);
    harness.transport.dispose();
  });

  it('ends on failed inflation without publishing a following ready state', async () => {
    const harness = makeHarness({ storedId: SESSION_ID }); await attach(harness);
    harness.sockets[0].serverSend('{"encoding":"tensorviz-gzip-json-v1","data":"eA=="}');
    harness.sockets[0].serverSend(snapshotFrame(1));
    await settle(20);
    expect(harness.messages).toEqual([]);
    expect(harness.transport.status).toMatchObject({ phase: 'ended', error: 'INVALID_MESSAGE' });
  });

  it('erases the fragment before trading the credential, and persists the session id only', async () => {
    const harness = makeHarness({ hash: encodeLocalBrowserFragment(SESSION_ID, CREDENTIAL) });

    const handshake = await attach(harness);

    const replaceAt = harness.events.indexOf('replace:/');
    const fetchAt = harness.events.indexOf(`fetch:${LOCAL_BROWSER_BOOTSTRAP_PATH}`);
    expect(replaceAt).toBeGreaterThanOrEqual(0);
    expect(fetchAt).toBeGreaterThan(replaceAt);
    expect(harness.where.hash).toBe('');
    const bodyEvent = harness.events.find((event) => event.startsWith('body:'));
    expect(JSON.parse(bodyEvent!.slice('body:'.length))).toEqual({
      workspaceSessionId: SESSION_ID,
      credential: CREDENTIAL,
    });
    expect(harness.sockets[0].url).toBe(`ws://127.0.0.1:3000${LOCAL_BROWSER_WEBSOCKET_PATH}`);
    expect(handshake).toMatchObject({
      type: 'viewHandshake',
      role: 'view',
      workspaceSessionId: SESSION_ID,
      protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
    });
    expect(JSON.parse(harness.sockets[0].sent[1])).toMatchObject({
      type: 'snapshotRequest',
      role: 'view',
      lastHostSequence: 0,
    });
    expect(harness.store.get(LOCAL_BROWSER_SESSION_STORAGE_KEY)).toBe(SESSION_ID);
    for (const value of Array.from(harness.store.values())) {
      expect(value).not.toContain(CREDENTIAL);
    }
  });

  it('re-attaches on refresh from the remembered session id, with no fragment and no fetch', async () => {
    const harness = makeHarness({ storedId: SESSION_ID });

    const handshake = await attach(harness);

    expect(handshake.workspaceSessionId).toBe(SESSION_ID);
    expect(harness.events.some((event) => event.startsWith('fetch:'))).toBe(false);
    expect(harness.events.some((event) => event.startsWith('replace:'))).toBe(false);
    expect(JSON.parse(harness.sockets[0].sent[1])).toMatchObject({
      type: 'snapshotRequest',
      lastHostSequence: 0,
    });
  });

  it('ends with INVALID_CREDENTIALS when neither a fragment nor a remembered id exists', async () => {
    const harness = makeHarness();

    await expect(harness.transport.connect()).rejects.toThrow('INVALID_CREDENTIALS');

    expect(harness.sockets).toHaveLength(0);
    expect(harness.events.some((event) => event.startsWith('fetch:') || event.startsWith('socket:'))).toBe(false);
    expect(harness.transport.status).toEqual({
      phase: 'ended',
      hostLink: 'unknown',
      error: 'INVALID_CREDENTIALS',
    });
  });

  it('maps a bootstrap 410 to CREDENTIAL_EXPIRED and clears the remembered id', async () => {
    const harness = makeHarness({
      hash: encodeLocalBrowserFragment(SESSION_ID, CREDENTIAL),
      storedId: 'ws-stale',
      fetchImpl: async () => jsonResponse(410, { detail: 'expired' }),
    });

    await expect(harness.transport.connect()).rejects.toThrow('CREDENTIAL_EXPIRED');

    expect(harness.transport.status.error).toBe('CREDENTIAL_EXPIRED');
    expect(harness.store.has(LOCAL_BROWSER_SESSION_STORAGE_KEY)).toBe(false);
    expect(harness.sockets).toHaveLength(0);
  });

  it('treats a bootstrap payload on the wrong protocol version as a terminal mismatch', async () => {
    const harness = makeHarness({
      hash: encodeLocalBrowserFragment(SESSION_ID, CREDENTIAL),
      fetchImpl: async () =>
        jsonResponse(200, sessionInfo({ protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION + 1 })),
    });

    await expect(harness.transport.connect()).rejects.toThrow('PROTOCOL_VERSION_MISMATCH');

    expect(harness.transport.status).toEqual({
      phase: 'ended',
      hostLink: 'unknown',
      error: 'PROTOCOL_VERSION_MISMATCH',
    });
    expect(harness.sockets).toHaveLength(0);
  });

  it('delivers host MCP validation feedback and refuses malformed execution claims', async () => {
    const harness = makeHarness({ storedId: SESSION_ID }); await attach(harness);
    const valid = frame('validationState', { runId: 'run-1', target: TARGET, status: 'passed', detail: 'Measured.', projectionRevision: 7 });
    harness.sockets[0].serverSend(valid);
    expect(harness.messages).toEqual([valid]);
    harness.sockets[0].serverSend({ ...valid, status: 'running', messageId: 'bad-progress' });
    expect(harness.messages).toHaveLength(1);
    expect(harness.transport.protocolViolations).toBe(1);
  });

  it('delivers browser source review while refusing a host-injected render receipt', async () => {
    const harness = makeHarness({ storedId: SESSION_ID }); await attach(harness);
    const review = frame('graphReviewState', { state: { id: 'review-1', beforeLabel: 'HEAD', afterLabel: 'Working',
      before: {}, after: {}, changedFiles: ['model.py'], stale: false } });
    const focus = frame('graphReviewFocus', { reviewId: 'review-1', side: 'after', nodeIds: ['fc'] });
    harness.sockets[0].serverSend(review); harness.sockets[0].serverSend(focus);
    expect(harness.messages).toEqual([review, focus]);
    harness.sockets[0].serverSend(frame('graphReviewApplied', { reviewId: 'review-1', viewId: 'view-1', mode: 'diff' }));
    expect(harness.messages).toHaveLength(2);
    expect(harness.transport.protocolViolations).toBe(1);
  });

  it('holds sequenced state until the snapshot, drops replays at the watermark', async () => {
    const harness = makeHarness({ storedId: SESSION_ID });
    await attach(harness);

    harness.sockets[0].serverSend(projectionFrame(5));
    expect(harness.messages).toHaveLength(0);

    harness.sockets[0].serverSend(snapshotFrame(5));
    expect(harness.messages.map((message) => message.type)).toEqual(['snapshot']);

    harness.sockets[0].serverSend(projectionFrame(5));
    harness.sockets[0].serverSend(projectionFrame(7));
    expect(harness.messages.map((message) => message.type)).toEqual(['snapshot', 'projection']);
    expect((harness.messages[1] as { hostSequence: number }).hostSequence).toBe(7);
    expect(harness.transport.protocolViolations).toBe(0);
  });

  it('resets the sequence epoch when the host restarts and re-requests from zero', async () => {
    const harness = makeHarness({ storedId: SESSION_ID });
    await attach(harness);
    harness.sockets[0].serverSend(snapshotFrame(5));

    harness.sockets[0].serverSend(connectionStateFrame('disconnected'));
    expect(harness.transport.status.hostLink).toBe('disconnected');
    // A straggler from the dead epoch is held rather than merged into the next one.
    harness.sockets[0].serverSend(projectionFrame(6));
    expect(harness.messages).toHaveLength(1);

    harness.sockets[0].serverSend(connectionStateFrame('connected'));
    const lastSent = JSON.parse(harness.sockets[0].sent[harness.sockets[0].sent.length - 1]);
    expect(lastSent).toMatchObject({ type: 'snapshotRequest', lastHostSequence: 0 });

    harness.sockets[0].serverSend(snapshotFrame(1));
    expect(harness.messages.map((message) => message.type)).toEqual(['snapshot', 'snapshot']);
  });

  it('reconnects after a transient drop with the watermark intact', async () => {
    const harness = makeHarness({ storedId: SESSION_ID });
    await attach(harness);
    harness.sockets[0].serverSend(snapshotFrame(5));
    harness.sockets[0].serverSend(projectionFrame(7));

    harness.sockets[0].serverClose(1006, 'abnormal closure');
    expect(harness.transport.status.phase).toBe('reconnecting');

    await settle();
    expect(harness.sockets).toHaveLength(2);
    // The reconnect reuses the session it already had; no new bootstrap, no new fragment.
    expect(harness.events.filter((event) => event.startsWith('socket:'))).toHaveLength(2);
    expect(harness.events.filter((event) => event.startsWith('fetch:'))).toHaveLength(0);

    harness.sockets[1].serverOpen();
    harness.sockets[1].serverSend(connectedFrame());
    await settle();

    const resent = harness.sockets[1].sent.map((data) => JSON.parse(data));
    expect(resent[0]).toMatchObject({ type: 'viewHandshake', workspaceSessionId: SESSION_ID });
    expect(resent[1]).toMatchObject({ type: 'snapshotRequest', lastHostSequence: 7 });
    expect(harness.transport.status.phase).toBe('connected');
  });

  it('ends without retry when the broker closes with a terminal credential code', async () => {
    const harness = makeHarness({ storedId: SESSION_ID });
    const pending = harness.transport.connect();
    await settle();
    harness.sockets[0].serverOpen();

    harness.sockets[0].serverClose(LOCAL_BROWSER_CLOSE_CODES.CREDENTIAL_EXPIRED, 'credential expired');

    await expect(pending).rejects.toThrow('CREDENTIAL_EXPIRED');
    await settle();
    expect(harness.sockets).toHaveLength(1);
    expect(harness.transport.status).toEqual({
      phase: 'ended',
      hostLink: 'unknown',
      error: 'CREDENTIAL_EXPIRED',
    });
    expect(harness.store.has(LOCAL_BROWSER_SESSION_STORAGE_KEY)).toBe(false);
  });

  it('builds outbound envelopes itself and refuses to send while disconnected', async () => {
    const harness = makeHarness({ storedId: SESSION_ID });
    expect(harness.transport.send({ type: 'revealNode', target: TARGET, nodeId: 'fc1' })).toBeNull();

    await attach(harness);
    const sent = harness.transport.send({ type: 'revealNode', target: TARGET, nodeId: 'fc1' });

    expect(sent).toMatchObject({
      type: 'revealNode',
      target: TARGET,
      nodeId: 'fc1',
      protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
      workspaceSessionId: SESSION_ID,
    });
    expect(sent).not.toHaveProperty('role');
    expect(sent).not.toHaveProperty('hostSequence');
    expect(JSON.parse(harness.sockets[0].sent[harness.sockets[0].sent.length - 1])).toEqual(sent);

    // Source-edit traffic frames the same way: the view's question and cancel carry
    // the envelope, never a role or a sequence claim.
    const asked = harness.transport.send({
      type: 'sourceEditRequest',
      target: TARGET,
      requestId: 'ask-1',
      edit: { kind: 'set_param', nodeId: 'enc1', param: 'out_features', value: 8 },
    });
    expect(asked).toMatchObject({
      type: 'sourceEditRequest',
      target: TARGET,
      requestId: 'ask-1',
      edit: { kind: 'set_param', nodeId: 'enc1', param: 'out_features', value: 8 },
      protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
      workspaceSessionId: SESSION_ID,
    });
    expect(asked).not.toHaveProperty('role');
    expect(harness.transport.send({ type: 'sourceEditCancel', target: TARGET, requestId: 'ask-1' })).not.toBeNull();
    // An empty question or an answer-shaped ask is a local bug: loud at the
    // call site, never a frame the broker would have to refuse.
    expect(() =>
      harness.transport.send({
        type: 'sourceEditRequest',
        target: TARGET,
        requestId: 'ask-2',
        edit: { kind: 'set_param', nodeId: 'enc1', param: '', value: 8 },
      }),
    ).toThrow('invalid sourceEditRequest');
  });

  it('counts contract violations without letting them past the boundary', async () => {
    const harness = makeHarness({ storedId: SESSION_ID });
    await attach(harness);
    harness.sockets[0].serverSend(snapshotFrame(5));
    expect(harness.transport.protocolViolations).toBe(0);

    // Not JSON at all.
    harness.sockets[0].serverSend('{not json');
    // Well-formed, but for somebody else's workspace session.
    harness.sockets[0].serverSend({ ...projectionFrame(6), workspaceSessionId: 'ws-other' });
    // A valid handshake-only message that must never arrive in-band.
    harness.sockets[0].serverSend(frame('hostHandshake', { role: 'host' }));
    // View-originated ask traffic echoing back down: the broker would never do
    // this, so both count as routing drift.
    harness.sockets[0].serverSend(
      frame('sourceEditRequest', { target: TARGET, requestId: 'ask-9', edit: { kind: 'remove_layer', nodeId: 'enc1' } }),
    );
    harness.sockets[0].serverSend(frame('sourceEditCancel', { target: TARGET, requestId: 'ask-9' }));
    expect(harness.transport.protocolViolations).toBe(5);
    expect(harness.messages.map((message) => message.type)).toEqual(['snapshot']);

    // A boundary replay is boring, not a violation: sequence and contract agree it is dead.
    harness.sockets[0].serverSend(projectionFrame(5));
    expect(harness.transport.protocolViolations).toBe(5);
    expect(harness.messages).toHaveLength(1);
  });

  it('delivers source-edit and validation replies unsequenced, even while a snapshot is still outstanding', async () => {
    const harness = makeHarness({ storedId: SESSION_ID });
    await attach(harness);

    // The snapshot to anchor the epoch has not arrived yet; sequenced state
    // would be held, but a question's reply is per-request traffic and must
    // not queue behind host state.
    harness.sockets[0].serverSend(
      frame('sourceEditResult', { target: TARGET, requestId: 'ask-1', status: 'applied', detail: 'enc1 encodes the input' }),
    );
    harness.sockets[0].serverSend(
      frame('validateResult', { target: TARGET, requestId: 'ask-2', status: 'refused', detail: 'no answer right now' }),
    );

    expect(harness.messages.map((message) => message.type)).toEqual(['sourceEditResult', 'validateResult']);
    expect(harness.transport.protocolViolations).toBe(0);
  });

  it('treats a second acceptance inside one epoch as drift, not a re-handshake', async () => {
    const harness = makeHarness({ storedId: SESSION_ID });
    await attach(harness);
    expect(harness.sockets[0].sent).toHaveLength(2);

    harness.sockets[0].serverSend(connectedFrame());

    expect(harness.transport.protocolViolations).toBe(1);
    // One snapshot request per epoch; a duplicate acceptance must not mint another.
    expect(harness.sockets[0].sent).toHaveLength(2);
    expect(harness.messages).toHaveLength(0);
  });

  it('rejects a pending connect on dispose and closes the socket politely exactly once', async () => {
    const harness = makeHarness({ storedId: SESSION_ID });
    const pending = harness.transport.connect();
    await settle();
    expect(harness.sockets).toHaveLength(1);

    harness.transport.dispose();

    await expect(pending).rejects.toThrow('ended');
    expect(harness.sockets[0].closes).toEqual([{ code: 1000, reason: 'page done' }]);
    expect(harness.transport.status).toEqual({ phase: 'ended', hostLink: 'unknown', error: null });

    harness.transport.dispose();
    expect(harness.sockets[0].closes).toHaveLength(1);

    // Disposal is one-way: a page that disposed can never dial again, even if a
    // caller (a remounting component, say) asks it to.
    const fresh = makeHarness({ storedId: SESSION_ID });
    fresh.transport.dispose();
    await expect(fresh.transport.connect()).rejects.toThrow('already disposed');
    expect(fresh.sockets).toHaveLength(0);
  });

  it('pins the sequenced-types contract the watermark logic depends on', () => {
    // snapshot resets the anchor; the other five are gated by it. Reordering or
    // growing this list is a broker + transports + contract change, never a
    // drive-by edit.
    expect([...LOCAL_BROWSER_SEQUENCED_MESSAGE_TYPES]).toEqual([
      'graphOpening',
      'snapshot',
      'activeTarget',
      'projection',
      // Ordered host state for the project plane: a document must not
      // arrive before the index it was drawn from.
      'projectIndex',
      'projectDocument',
      'syncState',
      'staleState',
      'focusNode',
    ]);
    for (const type of LOCAL_BROWSER_SEQUENCED_MESSAGE_TYPES) {
      expect(LOCAL_BROWSER_MESSAGE_TYPES).toContain(type);
    }
    // Source-edit traffic is registered but unsequenced by construction: replies match
    // by requestId, never by a host epoch.
    for (const type of ['sourceEditRequest', 'sourceEditCancel', 'sourceEditResult', 'validateResult']) {
      expect(LOCAL_BROWSER_MESSAGE_TYPES).toContain(type);
      expect(LOCAL_BROWSER_SEQUENCED_MESSAGE_TYPES).not.toContain(type);
    }
  });
});

describe('project wire and page lifecycle', () => {
  const project = { workspaceFolder: '/work', projectRoot: '', experimentId: 'train', indexDigest: 'a'.repeat(64), indexRevision: 1 };
  const index = { experiments: [], roots: [], unreached: [], truncated: false };
  it('sends all project intents, drops malformed/echoed traffic, and restores the surface on reload/reconnect', async () => {
    const { EMPTY_LOCAL_BROWSER_PAGE_STATE, reduceLocalBrowserViewMessage } = await import('../../ide/localBrowserState');
    for (const reload of [false, true]) {
      const h = makeHarness(reload ? { storedId: SESSION_ID } : { hash: encodeLocalBrowserFragment(SESSION_ID, CREDENTIAL) });
      let page = EMPTY_LOCAL_BROWSER_PAGE_STATE;
      h.transport.onMessage(message => { page = reduceLocalBrowserViewMessage(page, message); });
      await attach(h);
      const socket = h.sockets[0];
      const publish = (seq: number) => {
        socket.serverSend(frame('projectIndex', { hostSequence: seq, project, index, truncated: false, indexed: 2, discovered: 3 }));
        socket.serverSend(frame('projectDocument', { hostSequence: seq + 1, project,
          document: { experiment_id: 'train', nodes: [{ id: 'net' }], edges: [] }, spanmap: {}, expanded: ['net'] }));
      };
      publish(1);
      expect(page.project).toBeNull(); // No project frame can escape the snapshot barrier.
      socket.serverSend(snapshotFrame(3));
      publish(4);
      expect(page.project).toEqual(project);
      expect(page.activeTarget).toBeNull();
      expect(page.expanded).toEqual(['net']);
      expect(page.projectDiscovered).toBe(3);
      h.transport.send({ type: 'projectFocus', project, experimentId: 'other' });
      h.transport.send({ type: 'projectExpand', project, nodeId: 'net', expanded: false });
      h.transport.send({ type: 'projectDeclare', project, declaration: 'edge', args: ['net', 'trainer'] });
      expect(socket.sent.slice(-3).map(text => JSON.parse(text).type)).toEqual(['projectFocus', 'projectExpand', 'projectDeclare']);
      expect(socket.sent.slice(-3).every(text => !('hostSequence' in JSON.parse(text)))).toBe(true);
      socket.serverSend(frame('projectFocus', { project, experimentId: 'other' }));
      socket.serverSend(frame('projectDocument', { hostSequence: 6, project, document: {}, spanmap: {}, expanded: ['net', 'net'] }));
      expect(h.transport.protocolViolations).toBe(2);
      expect(page.expanded).toEqual(['net']);
      socket.serverSend(frame('activeTarget', { hostSequence: 7, target: TARGET }));
      expect(page.projectIndex).toBeNull();
      expect(page.expanded).toEqual([]);
      publish(8);
      expect(page.activeTarget).toBeNull();
      socket.serverSend(connectionStateFrame('disconnected'));
      socket.serverSend(connectionStateFrame('connected'));
      socket.serverSend(snapshotFrame(1));
      publish(2);
      expect(page.project).toEqual(project);
      expect(page.syncState).toBe('source-only');
      h.transport.dispose();
    }
  });
});

describe('agent browser handoff', () => {
  it('keeps only the non-secret tab hint, restores it on reload and never bootstraps twice', async () => {
    const first = makeHarness({ hash: encodeLocalBrowserFragment(SESSION_ID, CREDENTIAL),
      fetchImpl: async () => jsonResponse(200, sessionInfo({ handoffId: 'handoff-1' })) });
    expect((await attach(first)).handoffId).toBe('handoff-1');
    expect(first.where.hash).toBe('');
    expect([...first.store.values()].join()).not.toContain(CREDENTIAL);
    const second = makeHarness({ storedId: SESSION_ID });
    for (const [key, value] of first.store) second.store.set(key, value);
    first.transport.dispose();
    expect((await attach(second)).handoffId).toBe('handoff-1');
    expect(second.events.some(event => event.startsWith('fetch:'))).toBe(false);
    second.transport.dispose();
  });

  it('does not reuse a handoff from another session or an ordinary new browser link', async () => {
    for (const hash of ['', encodeLocalBrowserFragment(SESSION_ID, CREDENTIAL)]) {
      const harness = makeHarness({ storedId: SESSION_ID, hash });
      harness.store.set(`${LOCAL_BROWSER_SESSION_STORAGE_KEY}.handoff`, JSON.stringify(['old-session', 'old-handoff']));
      expect((await attach(harness)).handoffId).toBeUndefined();
      harness.transport.dispose();
    }
  });

  it('rejects malformed correlation and mismatched bootstrap session before connecting', async () => {
    for (const override of [{ handoffId: '' }, { handoffId: 12 }, { workspaceSessionId: 'another-session' }]) {
      const harness = makeHarness({ hash: encodeLocalBrowserFragment(SESSION_ID, CREDENTIAL),
        fetchImpl: async () => jsonResponse(200, sessionInfo(override)) });
      await expect(harness.transport.connect()).rejects.toThrow('PROTOCOL_VERSION_MISMATCH');
      expect(harness.sockets).toHaveLength(0);
    }
  });
});
