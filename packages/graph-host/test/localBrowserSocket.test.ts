import * as assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { Duplex } from 'node:stream';

import {
  LocalWebSocket,
  LocalWebSocketCloseEvent,
  connectLocalWebSocket,
} from '@tensorviz/graph-host/localBrowserSocket';

/**
 * The socket client held against an independent, deliberately minimal
 * server-side implementation of RFC 6455 written for this suite — the two
 * share no code, so the handshake, masking, fragmentation and close semantics
 * are actually inter-tested, not self-confirming. The endpoints are an
 * in-memory duplex pair (via the client's `dial` seam) rather than a real
 * TCP accept: restricted sandboxes refuse listen(), and the bytes on the
 * wire are identical either way.
 */

const WS_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';

interface CapturedHandshake {
  requestLine: string;
  headers: Record<string, string>;
}

interface ServerInboundText {
  kind: 'text';
  text: string;
}
interface ServerInboundClose {
  kind: 'close';
  code: number;
  reason: string;
}
interface ServerInboundPong {
  kind: 'pong';
  payload: Buffer;
}
type ServerInbound = ServerInboundText | ServerInboundClose | ServerInboundPong;

interface TestServerApi {
  handshake: CapturedHandshake | null;
  /** Next client frame(s). The queue never blocks: tests are scripted. */
  incoming: ServerInbound[];
  waiters: Array<(frame: ServerInbound) => void>;
  /** The already-connected endpoint the client under test should dial. */
  dial(): Duplex;
  sendText(text: string): void;
  /** Server-side fragmentation: the client must reassemble these into one message. */
  sendTextFragmented(first: string, second: string): void;
  /** Server-side violation: a second data message before the unfinished one closes. */
  sendInterleavedText(first: string, second: string): void;
  sendPing(payload: string): void;
  sendClose(code: number, reason: string): void;
  /** FIN toward the client with no close frame: the link simply died. */
  dropLink(): void;
  stop(): void;
  nextFrame(): Promise<ServerInbound>;
}

function encodeServerFrame(opcode: number, payload: Buffer, fin = true): Buffer {
  let header: Buffer;
  if (payload.length < 126) {
    header = Buffer.from([(fin ? 0x80 : 0) | opcode, payload.length]);
  } else if (payload.length < 65536) {
    header = Buffer.alloc(4);
    header[0] = (fin ? 0x80 : 0) | opcode;
    header[1] = 126;
    header.writeUInt16BE(payload.length, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = (fin ? 0x80 : 0) | opcode;
    header[1] = 127;
    header.writeBigUInt64BE(BigInt(payload.length), 2);
  }
  return Buffer.concat([header, payload]);
}

/** Two endpoints wired back-to-back: bytes written at one end emerge as data
 *  at the other, no kernel involved. Delivery is deferred to the next tick:
 *  real transports never deliver inside _write, and the double should not be
 *  more re-entrant than the thing it models. destroy() tears down only its
 *  own end, mirroring how a transport close affects exactly one side —
 *  Duplex.from / duplexify cannot be used here because it aborts (throws)
 *  across both. */
function socketPair(): { clientSide: Duplex; serverSide: Duplex } {
  const peers: { clientSide?: Duplex; serverSide?: Duplex } = {};
  const relay = (target: Duplex | undefined, chunk: Buffer): void => {
    setImmediate(() => {
      if (target && !target.destroyed) {
        target.push(chunk);
      }
    });
  };
  const clientSide = new Duplex({
    read() {},
    write(chunk: Buffer, _encoding, callback) {
      relay(peers.serverSide, chunk);
      callback();
    },
  });
  const serverSide = new Duplex({
    read() {},
    write(chunk: Buffer, _encoding, callback) {
      relay(peers.clientSide, chunk);
      callback();
    },
  });
  peers.clientSide = clientSide;
  peers.serverSide = serverSide;
  return { clientSide, serverSide };
}

/** A bare WS server speaking over the in-memory pair. `deny` answers the
 *  upgrade with HTTP 403 and nothing else, which is what starlette does for
 *  the broker's pre-accept refusals. */
function startTestServer(options: { deny?: boolean } = {}): TestServerApi {
  const pair = socketPair();
  const broker = pair.serverSide;
  const api: TestServerApi = {
    handshake: null,
    incoming: [],
    waiters: [],
    dial: () => pair.clientSide,
    sendText: (text) => {
      broker.write(encodeServerFrame(0x1, Buffer.from(text, 'utf8')));
    },
    sendTextFragmented: (first, second) => {
      broker.write(encodeServerFrame(0x1, Buffer.from(first, 'utf8'), false));
      broker.write(encodeServerFrame(0x0, Buffer.from(second, 'utf8'), true));
    },
    sendInterleavedText: (first, second) => {
      broker.write(encodeServerFrame(0x1, Buffer.from(first, 'utf8'), false));
      broker.write(encodeServerFrame(0x1, Buffer.from(second, 'utf8'), true));
    },
    sendPing: (payload) => {
      broker.write(encodeServerFrame(0x9, Buffer.from(payload, 'utf8')));
    },
    sendClose: (code, reason) => {
      const reasonBytes = Buffer.from(reason, 'utf8');
      const payload = Buffer.allocUnsafe(2 + reasonBytes.length);
      payload.writeUInt16BE(code, 0);
      reasonBytes.copy(payload, 2);
      broker.write(encodeServerFrame(0x8, payload));
    },
    dropLink: () => {
      // FIN toward the client without a close frame.
      if (!pair.clientSide.destroyed) {
        pair.clientSide.push(null);
      }
    },
    stop: () => broker.destroy(),
    nextFrame() {
      const queued = this.incoming.shift();
      if (queued) return Promise.resolve(queued);
      return new Promise((resolve) => this.waiters.push(resolve));
    },
  };

  let handshook = false;
  let received: Buffer = Buffer.alloc(0);
  broker.on('data', (chunk: Buffer) => {
    received = received.length === 0 ? chunk : Buffer.concat([received, chunk]);
    if (!handshook) {
      const boundary = received.indexOf('\r\n\r\n');
      if (boundary === -1) return;
      const head = received.subarray(0, boundary).toString('latin1');
      received = received.subarray(boundary + 4);
      const lines = head.split('\r\n');
      const headers: Record<string, string> = {};
      for (const line of lines.slice(1)) {
        const colon = line.indexOf(':');
        if (colon > 0) {
          headers[line.slice(0, colon).trim().toLowerCase()] = line.slice(colon + 1).trim();
        }
      }
      api.handshake = { requestLine: lines[0], headers };
      if (options.deny) {
        broker.end('HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n');
        return;
      }
      const key = headers['sec-websocket-key'];
      assert.ok(key, 'client must send Sec-WebSocket-Key');
      const accept = createHash('sha1').update(key + WS_GUID).digest('base64');
      broker.write(
        `HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`,
      );
      handshook = true;
      // Fall through: bytes after the boundary are already frame bytes.
    }
    for (;;) {
      if (received.length < 2) return;
      const opcode = received[0] & 0x0f;
      const masked = (received[1] & 0x80) !== 0;
      let length = received[1] & 0x7f;
      let headerLength = 2;
      if (length === 126) {
        if (received.length < 4) return;
        length = received.readUInt16BE(2);
        headerLength = 4;
      } else if (length === 127) {
        if (received.length < 10) return;
        length = Number(received.readBigUInt64BE(2));
        headerLength = 10;
      }
      const maskOffset = headerLength;
      if (masked) headerLength += 4;
      if (received.length < headerLength + length) return;
      assert.ok(masked, 'client frames must be masked per RFC 6455');
      const mask = received.subarray(maskOffset, maskOffset + 4);
      const payload = Buffer.allocUnsafe(length);
      for (let i = 0; i < length; i += 1) {
        payload[i] = received[maskOffset + 4 + i] ^ mask[i & 3];
      }
      received = received.subarray(headerLength + length);
      // The client under test never fragments, so text frames deliver directly.
      if (opcode === 0x1) {
        deliver({ kind: 'text', text: payload.toString('utf8') });
      } else if (opcode === 0xa) {
        deliver({ kind: 'pong', payload });
      } else if (opcode === 0x8) {
        const code = payload.length >= 2 ? payload.readUInt16BE(0) : 1005;
        const reason = payload.length > 2 ? payload.subarray(2).toString('utf8') : '';
        deliver({ kind: 'close', code, reason });
        broker.end(encodeServerFrame(0x8, payload));
      }
    }

    function deliver(frame: ServerInbound): void {
      const waiter = api.waiters.shift();
      if (waiter) {
        waiter(frame);
      } else {
        api.incoming.push(frame);
      }
    }
  });

  return api;
}

/** Connects the client over the in-memory pair; resolves once the upgrade accepted. */
function connectClient(
  server: TestServerApi,
  headers: Record<string, string> = {},
): Promise<{ ws: LocalWebSocket; texts: string[]; closeEvent: Promise<LocalWebSocketCloseEvent> }> {
  const texts: string[] = [];
  let resolveClose!: (event: LocalWebSocketCloseEvent) => void;
  let resolveOpen!: () => void;
  const closeEvent = new Promise<LocalWebSocketCloseEvent>((resolve) => {
    resolveClose = resolve;
  });
  const opened = new Promise<void>((resolve) => {
    resolveOpen = resolve;
  });
  const ws = connectLocalWebSocket(
    'ws://127.0.0.1:8787/local/ws',
    {
      onOpen: () => {
        resolveOpen();
      },
      onText: (text) => texts.push(text),
      onClose: (event) => resolveClose(event),
    },
    { headers, dial: server.dial },
  );
  return opened.then(() => ({ ws, texts, closeEvent }));
}

suite('localBrowserSocket', function () {
  this.timeout(5000);

  let server: TestServerApi;

  setup(() => {
    server = startTestServer();
  });

  teardown(() => {
    server.stop();
  });

  test('performs the upgrade with the auth headers and closes politely', async () => {
    const client = await connectClient(server, {
      Cookie: 'tensorviz_local_browser=host-key',
      Origin: 'http://127.0.0.1:1234',
    });
    assert.equal(server.handshake?.requestLine, 'GET /local/ws HTTP/1.1');
    assert.equal(server.handshake?.headers['host'], '127.0.0.1:8787');
    assert.equal(server.handshake?.headers['cookie'], 'tensorviz_local_browser=host-key');
    assert.equal(server.handshake?.headers['origin'], 'http://127.0.0.1:1234');
    assert.ok(server.handshake?.headers['sec-websocket-key']);
    assert.equal(server.handshake?.headers['sec-websocket-version'], '13');

    client.ws.close(1000, 'done');
    const closeFrame = await server.nextFrame();
    assert.deepEqual({ kind: closeFrame.kind, code: (closeFrame as ServerInboundClose).code }, {
      kind: 'close',
      code: 1000,
    });
    const event = await client.closeEvent;
    assert.deepEqual(event, { kind: 'closed', code: 1000, reason: 'done' });
  });

  test('delivers text both ways with client masking enforced server-side', async () => {
    const client = await connectClient(server);
    assert.ok(client.ws.sendText('{"type":"ping"}'));
    const inbound = await server.nextFrame();
    assert.deepEqual(inbound, { kind: 'text', text: '{"type":"ping"}' });

    server.sendText('{"type":"pong"}');
    await eventually(() => assert.deepEqual(client.texts, ['{"type":"pong"}']));
  });

  test('reassembles a fragmented server message into one text event', async () => {
    const client = await connectClient(server);
    server.sendTextFragmented('{"part":"one', '","rest":true}');
    await eventually(() => assert.deepEqual(client.texts, ['{"part":"one","rest":true}']));
  });

  test('refuses a new data message that interrupts an unfinished fragmented one', async () => {
    // RFC 6455 forbids interleaving data messages: after a fin=false text the
    // only lawful data frame is the continuation. Silently dropping the first
    // message would corrupt broker traffic, so this is a hard protocol close.
    const client = await connectClient(server);
    server.sendInterleavedText('{"part":"one', '{"other":true}');
    const inbound = await server.nextFrame();
    assert.equal(inbound.kind, 'close');
    assert.equal((inbound as ServerInboundClose).code, 1002);
    const event = await client.closeEvent;
    assert.deepEqual(event, { kind: 'closed', code: 1002, reason: 'message interrupted mid-fragment' });
    assert.deepEqual(client.texts, []);
  });

  test('answers a ping with a pong carrying the same payload', async () => {
    await connectClient(server);
    server.sendPing('beacon');
    const inbound = await server.nextFrame();
    assert.equal(inbound.kind, 'pong');
    assert.equal((inbound as ServerInboundPong).payload.toString('utf8'), 'beacon');
  });

  test('round-trips payloads past the 16-bit and 64-bit length boundaries', async () => {
    const client = await connectClient(server);
    const medium = 'm'.repeat(70_000); // 16-bit path client->server
    const large = 'L'.repeat(100_000); // 64-bit path server->client
    assert.ok(client.ws.sendText(medium));
    const inbound = await server.nextFrame();
    assert.equal(inbound.kind, 'text');
    assert.equal((inbound as ServerInboundText).text.length, 70_000);

    server.sendText(large);
    await eventually(() => {
      assert.equal(client.texts.length, 1);
      assert.equal(client.texts[0].length, 100_000);
    });
  });

  test('relays a server close frame with code and reason, echoing close', async () => {
    const client = await connectClient(server);
    server.sendClose(4402, 'CREDENTIAL_EXPIRED');
    const event = await client.closeEvent;
    assert.deepEqual(event, { kind: 'closed', code: 4402, reason: 'CREDENTIAL_EXPIRED' });
    const echoed = await server.nextFrame();
    assert.equal(echoed.kind, 'close');
    assert.equal((echoed as ServerInboundClose).code, 4402);
  });

  test('reports an abrupt link drop as 1006', async () => {
    const client = await connectClient(server);
    server.dropLink();
    const event = await client.closeEvent;
    assert.equal(event.kind, 'closed');
    assert.equal((event as { code: number }).code, 1006);
  });

  test('reports an upgrade denial with its HTTP status and never opens', async () => {
    const deniedServer = startTestServer({ deny: true });
    try {
      const closeEvent = await new Promise<LocalWebSocketCloseEvent>((resolve) => {
        connectLocalWebSocket(
          'ws://127.0.0.1:8787/local/ws',
          {
            onOpen: () => assert.fail('a denied upgrade must not open'),
            onText: () => assert.fail('a denied upgrade must not deliver text'),
            onClose: resolve,
          },
          { dial: deniedServer.dial },
        );
      });
      assert.deepEqual(closeEvent, { kind: 'refused', status: 403 });
    } finally {
      deniedServer.stop();
    }
  });

  test('reports a transport error before the upgrade as a network error, not a refusal', async () => {
    // A write-swallowing dead transport. NOT a bare PassThrough: that one is a
    // loopback and would echo the upgrade request back into the handshake parser.
    const broken = new Duplex({
      read() {},
      write(_chunk: Buffer, _encoding, callback) {
        callback();
      },
    });
    const event = await new Promise<LocalWebSocketCloseEvent>((resolve) => {
      connectLocalWebSocket(
        'ws://127.0.0.1:8787/local/ws',
        { onOpen: () => assert.fail('unreachable'), onText: () => {}, onClose: resolve },
        { dial: () => broken },
      );
      broken.emit('error', Object.assign(new Error('connect ECONNREFUSED'), { code: 'ECONNREFUSED' }));
    });
    assert.equal(event.kind, 'error');
  });
});

async function eventually(check: () => void): Promise<void> {
  const deadline = Date.now() + 2000;
  for (;;) {
    try {
      check();
      return;
    } catch (error) {
      if (Date.now() > deadline) throw error;
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
  }
}
