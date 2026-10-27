/** A minimal RFC 6455 client for the local-browser broker socket.
 *
 * The extension host runs Node 18, which has no global WebSocket, and the
 * extension ships zero runtime dependencies (see src/test/bootstrap.ts), so
 * the handshake and the frame codec live here. Scope is deliberately narrow:
 * text frames only, no permessage-deflate (we never offer it, so the server
 * may not use it), no subprotocols, HTTP/1.1 only. That is everything the
 * loopback broker speaks and nothing more.
 *
 * Security posture: custom headers (Cookie, Origin) are the point of this
 * module — the broker authenticates both roles on them
 * (backend/services/ide/local_browser_broker.py `_authenticate_socket`).
 * They are written to the socket and never error-logged. Client frames are
 * masked as the RFC requires, a masked server frame is a protocol error (a
 * server must not mask), and the payload cap matches the broker's 16 MiB so a
 * runaway peer is cut off locally first.
 */

import { createHash, randomBytes } from 'node:crypto';
import * as net from 'node:net';
import * as tls from 'node:tls';
import { Duplex } from 'node:stream';

const WS_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';

/** The broker refuses frames past 16 MiB before parsing; the client shares that
 *  bound so an incoming frame can never grow past it either. */
const MAX_PAYLOAD_BYTES = 16 * 1024 * 1024;
const HEADER_LIMIT_BYTES = 16 * 1024;
const DEFAULT_TIMEOUT_MS = 10_000;
/** After we send a close frame the peer is expected to answer and hang up;
 *  past this grace the client takes the socket down itself. */
const CLOSE_REPLY_GRACE_MS = 500;

const OPCODE_CONTINUATION = 0x0;
const OPCODE_TEXT = 0x1;
const OPCODE_BINARY = 0x2;
const OPCODE_CLOSE = 0x8;
const OPCODE_PING = 0x9;
const OPCODE_PONG = 0xa;

/** How the connection ended. `refused` means the upgrade itself was answered
 *  and denied (or the peer hung up mid-handshake) — a policy decision, not a
 *  network fault; `error` is a network fault before the socket ever opened;
 *  `closed` is any end after a successful handshake, carrying the WS close
 *  code (1006 when the TCP link simply died). */
export type LocalWebSocketCloseEvent =
  | { kind: 'refused'; status: number | null }
  | { kind: 'error'; message: string }
  | { kind: 'closed'; code: number; reason: string };

export interface LocalWebSocketHandlers {
  onOpen(): void;
  onText(text: string): void;
  /** Fires exactly once, from the first failure or close onward. */
  onClose(event: LocalWebSocketCloseEvent): void;
}

export interface LocalWebSocketOptions {
  headers?: Record<string, string>;
  timeoutMs?: number;
  maxPayloadBytes?: number;
  /** Test seam: an already-established full-duplex byte stream in place of the
   *  TCP/TLS connect, letting the codec tests run without binding a listener
   *  (restricted sandboxes refuse listen()). The client skips the connect-event
   *  wait and writes the HTTP upgrade into the stream immediately. */
  dial?: () => Duplex;
}

export interface LocalWebSocket {
  /** False when the socket is not open; the frame is skipped. */
  sendText(text: string): boolean;
  close(code?: number, reason?: string): void;
}

export function connectLocalWebSocket(
  url: string,
  handlers: LocalWebSocketHandlers,
  options: LocalWebSocketOptions = {},
): LocalWebSocket {
  const target = new URL(url);
  const secure = target.protocol === 'wss:' || target.protocol === 'https:';
  if (!secure && target.protocol !== 'ws:' && target.protocol !== 'http:') {
    throw new Error(`localBrowserSocket: unsupported protocol ${target.protocol}`);
  }
  const port = Number(target.port) || (secure ? 443 : 80);
  const maxPayload = options.maxPayloadBytes ?? MAX_PAYLOAD_BYTES;
  const key = randomBytes(16).toString('base64');

  let opened = false;
  let settled = false;
  let closeSent = false;
  let handshakeText = '';
  let pending: Buffer = Buffer.alloc(0);
  let messageChunks: Buffer[] | null = null;

  const timer = setTimeout(() => {
    if (!opened) settle({ kind: 'error', message: 'websocket upgrade timed out' });
  }, options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  timer.unref();

  const socket: Duplex = options.dial
    ? options.dial()
    : secure
      ? tls.connect({ host: target.hostname, port, servername: target.hostname })
      : net.connect({ host: target.hostname, port });

  function settle(event: LocalWebSocketCloseEvent): void {
    if (settled) return;
    settled = true;
    clearTimeout(timer);
    socket.removeAllListeners();
    socket.destroy();
    handlers.onClose(event);
  }

  function sendFrame(opcode: number, payload: Buffer): boolean {
    if (settled || socket.destroyed) return false;
    const mask = randomBytes(4);
    let header: Buffer;
    if (payload.length < 126) {
      header = Buffer.from([0x80 | opcode, 0x80 | payload.length]);
    } else if (payload.length < 65536) {
      header = Buffer.alloc(4);
      header[0] = 0x80 | opcode;
      header[1] = 0x80 | 126;
      header.writeUInt16BE(payload.length, 2);
    } else {
      header = Buffer.alloc(10);
      header[0] = 0x80 | opcode;
      header[1] = 0x80 | 127;
      header.writeBigUInt64BE(BigInt(payload.length), 2);
    }
    const masked = Buffer.allocUnsafe(payload.length);
    for (let i = 0; i < payload.length; i += 1) {
      masked[i] = payload[i] ^ mask[i & 3];
    }
    try {
      socket.write(Buffer.concat([header, mask, masked]));
    } catch {
      return false;
    }
    return true;
  }

  function protocolFailure(code: number, reason: string): void {
    sendFrame(OPCODE_CLOSE, encodeClosePayload(code, reason));
    settle({ kind: 'closed', code, reason });
  }

  function onHandshakeChunk(chunk: Buffer): void {
    handshakeText += chunk.toString('latin1');
    if (handshakeText.length > HEADER_LIMIT_BYTES) {
      settle({ kind: 'refused', status: null });
      return;
    }
    const boundary = handshakeText.indexOf('\r\n\r\n');
    if (boundary === -1) return;
    const head = handshakeText.slice(0, boundary);
    const statusMatch = /^HTTP\/\d\.\d (\d{3})/.exec(head);
    const status = statusMatch ? Number(statusMatch[1]) : null;
    const accept = createHash('sha1').update(key + WS_GUID).digest('base64');
    // Exact string comparison: the hash is base64, and base64's '+' is a regex
    // quantifier — interpolating it would refuse a valid upgrade whenever the
    // handshake hash contains one (about a third of handshakes).
    const acceptHeader = /^sec-websocket-accept:[ \t]*([^\r\n]+)/im.exec(head)?.[1].trim();
    if (status !== 101 || acceptHeader !== accept) {
      settle({ kind: 'refused', status });
      return;
    }
    // Anything the server already pipelined past the header boundary is frame bytes.
    pending = Buffer.from(handshakeText.slice(boundary + 4), 'latin1');
    opened = true;
    clearTimeout(timer);
    handlers.onOpen();
    drainFrames();
  }

  function drainFrames(): void {
    for (;;) {
      if (pending.length < 2) return;
      const fin = (pending[0] & 0x80) !== 0;
      const opcode = pending[0] & 0x0f;
      const masked = (pending[1] & 0x80) !== 0;
      let length = pending[1] & 0x7f;
      let headerLength = 2;
      if (length === 126) {
        if (pending.length < 4) return;
        length = pending.readUInt16BE(2);
        headerLength = 4;
      } else if (length === 127) {
        if (pending.length < 10) return;
        const big = pending.readBigUInt64BE(2);
        if (big > BigInt(maxPayload)) {
          protocolFailure(1009, 'message too large');
          return;
        }
        length = Number(big);
        headerLength = 10;
      }
      if (opcode >= 0x8 && (!fin || length > 125)) {
        protocolFailure(1002, 'invalid control frame');
        return;
      }
      if (masked) {
        protocolFailure(1002, 'server frames must not be masked');
        return;
      }
      if (opcode !== OPCODE_CLOSE && length > maxPayload) {
        protocolFailure(1009, 'message too large');
        return;
      }
      if (pending.length < headerLength + length) return;
      const payload = pending.subarray(headerLength, headerLength + length);
      pending = pending.subarray(headerLength + length);
      if (!handleFrame(fin, opcode, payload)) return;
    }
  }

  function handleFrame(fin: boolean, opcode: number, payload: Buffer): boolean {
    switch (opcode) {
      case OPCODE_TEXT:
        if (messageChunks !== null) {
          // A new data message while the fragmented one has no continuation
          // yet is a protocol violation, not a reorder the codec may absorb.
          protocolFailure(1002, 'message interrupted mid-fragment');
          return false;
        }
        if (fin) {
          handlers.onText(payload.toString('utf8'));
        } else {
          messageChunks = [payload];
        }
        return true;
      case OPCODE_CONTINUATION:
        if (messageChunks === null) {
          protocolFailure(1002, 'continuation without a message');
          return false;
        }
        messageChunks.push(payload);
        if (fin) {
          const joined = Buffer.concat(messageChunks);
          messageChunks = null;
          if (joined.length > maxPayload) {
            protocolFailure(1009, 'message too large');
            return false;
          }
          handlers.onText(joined.toString('utf8'));
        }
        return true;
      case OPCODE_BINARY:
        protocolFailure(1003, 'binary frames are not the protocol');
        return false;
      case OPCODE_PING:
        sendFrame(OPCODE_PONG, payload);
        return true;
      case OPCODE_PONG:
        return true;
      case OPCODE_CLOSE: {
        const code = payload.length >= 2 ? payload.readUInt16BE(0) : 1005;
        const reason = payload.length > 2 ? payload.subarray(2).toString('utf8') : '';
        if (!closeSent) {
          closeSent = true;
          sendFrame(OPCODE_CLOSE, payload);
        }
        settle({ kind: 'closed', code, reason });
        return false;
      }
      default:
        protocolFailure(1002, 'unknown opcode');
        return false;
    }
  }

  socket.on('error', (error: NodeJS.ErrnoException) => {
    const message = error.code ?? error.message ?? 'socket error';
    if (!opened) {
      settle({ kind: 'error', message });
    } else {
      settle({ kind: 'closed', code: 1006, reason: message });
    }
  });
  socket.on('end', () => {
    if (!opened) {
      settle({ kind: 'refused', status: null });
    } else {
      settle({ kind: 'closed', code: 1006, reason: '' });
    }
  });
  socket.on('data', (chunk: Buffer) => {
    if (!opened) {
      onHandshakeChunk(chunk);
      return;
    }
    pending = pending.length === 0 ? chunk : Buffer.concat([pending, chunk]);
    drainFrames();
  });
  const writeUpgradeRequest = (): void => {
    const request = [
      `GET ${target.pathname}${target.search} HTTP/1.1`,
      `Host: ${target.host}`,
      'Upgrade: websocket',
      'Connection: Upgrade',
      `Sec-WebSocket-Key: ${key}`,
      'Sec-WebSocket-Version: 13',
      ...Object.entries(options.headers ?? {}).map(([name, value]) => `${name}: ${value}`),
      '\r\n',
    ].join('\r\n');
    socket.write(request);
  };
  if (options.dial) {
    // The injected stream is already open; there is no connect event to await.
    writeUpgradeRequest();
  } else {
    socket.once(secure ? 'secureConnect' : 'connect', writeUpgradeRequest);
  }

  return {
    sendText(text: string): boolean {
      if (!opened || closeSent) return false;
      return sendFrame(OPCODE_TEXT, Buffer.from(text, 'utf8'));
    },
    close(code = 1000, reason = ''): void {
      if (closeSent || settled) return;
      closeSent = true;
      if (opened) {
        sendFrame(OPCODE_CLOSE, encodeClosePayload(code, reason));
        setTimeout(() => settle({ kind: 'closed', code, reason }), CLOSE_REPLY_GRACE_MS).unref();
      } else {
        settle({ kind: 'closed', code, reason });
      }
    },
  };
}

function encodeClosePayload(code: number, reason: string): Buffer {
  const reasonBytes = Buffer.from(reason, 'utf8');
  const payload = Buffer.allocUnsafe(2 + reasonBytes.length);
  payload.writeUInt16BE(code, 0);
  reasonBytes.copy(payload, 2);
  return payload;
}
