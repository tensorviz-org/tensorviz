/** A loopback registry and request bus between VS Code extension hosts.
 *
 *  One window owns the single MCP endpoint; every other window that has a live
 *  graph announces itself here. The MCP host reads the registry, connects to
 *  each peer's ephemeral loopback listener and routes a tool call for that
 *  model back to the window that owns it. No bearer credential is invented:
 *  each listener is bound to 127.0.0.1 and additionally checks a per-host
 *  random token recorded in the shared registry, mirroring the MCP endpoint's
 *  own loopback posture. The registry lives in the extension's shared storage
 *  directory, so it is only visible to this VS Code user.
 */

import { randomBytes, randomUUID } from 'node:crypto';
import * as fs from 'node:fs/promises';
import * as net from 'node:net';
import * as path from 'node:path';

export const HOST_BUS_PROTOCOL = 2;
const MAX_MESSAGE_BYTES = 8 * 1024 * 1024;
const REQUEST_TIMEOUT_MS = 45_000;
const HEARTBEAT_MS = 10_000;

export interface HostAnnouncement {
  protocol: number;
  hostId: string;
  pid: number;
  workspaceFolder: string | null;
  port: number;
  token: string;
  startedAt: number;
}

export type BusMethod = 'describe' | 'call' | 'release' | 'sync';

interface BusRequest {
  id: string;
  token: string;
  method: BusMethod;
  params?: unknown;
}

type BusReply =
  | { id: string; ok: true; result: unknown }
  | { id: string; ok: false; error: { code: string; message: string } };

/** A structured failure that survives the wire with its code intact. */
export class BusError extends Error {
  constructor(readonly code: string, message: string) { super(message); }
}

export type BusHandler = (method: BusMethod, params: unknown, signal: AbortSignal) => Promise<unknown>;

function alive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; }
  catch (error) { return (error as NodeJS.ErrnoException).code === 'EPERM'; }
}

/** The on-disk rendezvous point shared by every window of this user. */
export class HostRegistry {
  constructor(private readonly dir: string) {}

  private file(hostId: string): string {
    const safe = /^[A-Za-z0-9_-]{1,64}$/.test(hostId) ? hostId : Buffer.from(hostId).toString('hex').slice(0, 64);
    return path.join(this.dir, `${safe}.json`);
  }

  async announce(record: HostAnnouncement): Promise<void> {
    await fs.mkdir(this.dir, { recursive: true, mode: 0o700 });
    const file = this.file(record.hostId), temporary = `${file}.${randomUUID()}.tmp`;
    try {
      await fs.writeFile(temporary, JSON.stringify(record), { mode: 0o600, flag: 'wx' });
      await fs.rename(temporary, file);
    } finally { await fs.rm(temporary, { force: true }); }
  }

  async withdraw(hostId: string): Promise<void> {
    await fs.rm(this.file(hostId), { force: true });
  }

  /** Live peers only: a record whose process is gone is removed, not returned. */
  async list(): Promise<HostAnnouncement[]> {
    let names: string[];
    try { names = await fs.readdir(this.dir); } catch { return []; }
    const records: HostAnnouncement[] = [];
    for (const name of names) {
      if (!name.endsWith('.json')) continue;
      const file = path.join(this.dir, name);
      try {
        const record = JSON.parse(await fs.readFile(file, 'utf8')) as HostAnnouncement;
        if (record.protocol !== HOST_BUS_PROTOCOL || typeof record.hostId !== 'string'
          || typeof record.port !== 'number' || typeof record.token !== 'string') continue;
        if (!alive(record.pid)) { await fs.rm(file, { force: true }); continue; }
        records.push(record);
      } catch { /* A half-written or unreadable record is not a host. */ }
    }
    return records;
  }
}

function encode(reply: BusReply): string {
  return `${JSON.stringify(reply)}\n`;
}

/** One window's private request listener. */
export class HostBusServer {
  readonly token = randomBytes(32).toString('hex');
  private server?: net.Server;
  private port?: number;
  private readonly sockets = new Set<net.Socket>();

  constructor(private readonly handler: BusHandler) {}

  async start(): Promise<number> {
    if (this.port) return this.port;
    const server = net.createServer(socket => this.accept(socket));
    await new Promise<void>((resolve, reject) => {
      const failed = (error: Error) => { server.off('listening', ready); reject(error); };
      const ready = () => { server.off('error', failed); resolve(); };
      server.once('error', failed);
      server.once('listening', ready);
      server.listen(0, '127.0.0.1');
    });
    const address = server.address();
    if (!address || typeof address === 'string') { server.close(); throw new Error('hostBus: did not bind a TCP port.'); }
    this.server = server;
    this.port = address.port;
    return this.port;
  }

  get bound(): { port: number; token: string } | undefined {
    return this.port === undefined ? undefined : { port: this.port, token: this.token };
  }

  private accept(socket: net.Socket): void {
    this.sockets.add(socket);
    socket.on('close', () => this.sockets.delete(socket));
    socket.setEncoding('utf8');
    let buffer = '';
    socket.on('data', (chunk: string) => {
      buffer += chunk;
      if (buffer.length > MAX_MESSAGE_BYTES) { socket.destroy(); return; }
      for (let index = buffer.indexOf('\n'); index !== -1; index = buffer.indexOf('\n')) {
        const line = buffer.slice(0, index);
        buffer = buffer.slice(index + 1);
        if (line.trim() !== '') void this.dispatch(line, socket);
      }
    });
    socket.on('error', () => socket.destroy());
  }

  private async dispatch(line: string, socket: net.Socket): Promise<void> {
    let request: BusRequest;
    try { request = JSON.parse(line) as BusRequest; }
    catch { return; }
    const id = typeof request?.id === 'string' ? request.id : '';
    if (!request || request.token !== this.token || !['describe', 'call', 'release', 'sync'].includes(request.method)) {
      this.write(socket, { id, ok: false, error: { code: 'UNAUTHORIZED', message: 'Unknown host bus token.' } });
      return;
    }
    const controller = new AbortController();
    const abort = () => controller.abort();
    socket.once('close', abort);
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    timer.unref();
    try {
      const result = await this.handler(request.method, request.params, controller.signal);
      this.write(socket, { id, ok: true, result });
    } catch (error) {
      const code = error instanceof BusError ? error.code : 'BUS_ERROR';
      const message = error instanceof Error ? error.message : 'The host bus request failed.';
      this.write(socket, { id, ok: false, error: { code, message } });
    } finally { clearTimeout(timer); socket.off('close', abort); }
  }

  private write(socket: net.Socket, reply: BusReply): void {
    if (!socket.destroyed) socket.write(encode(reply));
  }

  async close(): Promise<void> {
    const server = this.server;
    this.server = undefined;
    this.port = undefined;
    for (const socket of this.sockets) socket.destroy();
    if (server) await new Promise<void>(resolve => server.close(() => resolve()));
  }
}

/** A one-shot peer call used by the MCP host when a model lives elsewhere. */
export function callHost(target: { port: number; token: string }, method: BusMethod, params: unknown, timeoutMs = REQUEST_TIMEOUT_MS, signal?: AbortSignal): Promise<unknown> {
  return new Promise<unknown>((resolve, reject) => {
    const id = randomUUID();
    let buffer = '';
    let settled = false;
    const socket = net.connect({ host: '127.0.0.1', port: target.port });
    const abort = () => finish(new BusError('CANCELLED', 'The MCP request was cancelled.'));
    const timer = setTimeout(() => finish(new BusError('BUS_TIMEOUT', 'The owning window did not answer in time.')), timeoutMs);
    function finish(error?: Error, value?: unknown): void {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      socket.removeAllListeners();
      socket.destroy();
      if (error) reject(error); else resolve(value);
    }
    socket.setEncoding('utf8');
    socket.on('connect', () => socket.write(`${JSON.stringify({ id, token: target.token, method, params })}\n`));
    socket.on('data', (chunk: string) => {
      buffer += chunk;
      if (buffer.length > MAX_MESSAGE_BYTES) { finish(new BusError('BUS_ERROR', 'The host bus reply is too large.')); return; }
      for (let index = buffer.indexOf('\n'); index !== -1; index = buffer.indexOf('\n')) {
        const line = buffer.slice(0, index);
        buffer = buffer.slice(index + 1);
        let reply: BusReply;
        try { reply = JSON.parse(line) as BusReply; } catch { continue; }
        if (reply.id !== id) continue;
        if (reply.ok) finish(undefined, reply.result);
        else finish(new BusError(reply.error.code, reply.error.message));
        return;
      }
    });
    socket.on('error', error => finish(new BusError('BUS_UNREACHABLE', (error as NodeJS.ErrnoException).code ?? 'The owning window is unavailable.')));
    socket.on('end', () => finish(new BusError('BUS_UNREACHABLE', 'The owning window closed the connection.')));
    signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) abort();
  });
}

/** Owns one host's listener and its registry record for the window's lifetime. */
export class HostNode {
  readonly registry: HostRegistry;
  readonly bus: HostBusServer;
  private timer?: ReturnType<typeof setInterval>;
  private heartbeat: Promise<void> = Promise.resolve();

  constructor(private readonly options: {
    hostId: string;
    workspaceFolder: string | null;
    registryDir: string;
    handler: BusHandler;
  }) {
    this.registry = new HostRegistry(options.registryDir);
    this.bus = new HostBusServer(options.handler);
  }

  async start(): Promise<void> {
    const port = await this.bus.start();
    const record: HostAnnouncement = { protocol: HOST_BUS_PROTOCOL, hostId: this.options.hostId, pid: process.pid,
      workspaceFolder: this.options.workspaceFolder, port, token: this.bus.token, startedAt: Date.now() };
    await this.registry.announce(record);
    this.timer = setInterval(() => {
      this.heartbeat = this.heartbeat.then(() => this.registry.announce(record)).catch(() => {});
    }, HEARTBEAT_MS);
    this.timer.unref();
  }

  async stop(): Promise<void> {
    if (this.timer) clearInterval(this.timer);
    this.timer = undefined;
    await this.heartbeat;
    await this.registry.withdraw(this.options.hostId).catch(() => {});
    await this.bus.close();
  }
}
