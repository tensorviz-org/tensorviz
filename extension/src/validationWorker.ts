/** Authenticated launch primitive owned by the shared validation controller. */
import { ChildProcess, execFileSync, spawn } from 'node:child_process';
import { randomBytes, timingSafeEqual } from 'node:crypto';
import * as fs from 'node:fs/promises';
import * as net from 'node:net';
import * as os from 'node:os';
import * as path from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import * as vscode from 'vscode';
import type { ExecutionRecord, RunSpec } from '@tensorviz/contracts/protocol';

const BOOTSTRAP_ENV = 'TENSORVIZ_VALIDATION_BOOTSTRAP';
const MAX_FRAME = 4 * 1024 * 1024 + 4096;
const MAX_LOG = 64 * 1024;

/** Incremental parser: inspect each length before allocating a frame body. */
export class ValidationFrames {
  private header = Buffer.alloc(4);
  private headerUsed = 0;
  private body: Buffer | undefined;
  private bodyUsed = 0;
  constructor(private readonly maximum: () => number, private readonly receive: (value: unknown) => void) {}
  push(chunk: Buffer): void {
    let offset = 0;
    while (offset < chunk.length) {
      if (!this.body) {
        const count = Math.min(4 - this.headerUsed, chunk.length - offset);
        chunk.copy(this.header, this.headerUsed, offset, offset + count);
        this.headerUsed += count;
        offset += count;
        if (this.headerUsed < 4) { continue; }
        const size = this.header.readUInt32BE();
        if (!size || size > this.maximum()) { throw new Error('frame_limit'); }
        this.body = Buffer.alloc(size);
        this.bodyUsed = 0;
      }
      const count = Math.min(this.body.length - this.bodyUsed, chunk.length - offset);
      chunk.copy(this.body, this.bodyUsed, offset, offset + count);
      this.bodyUsed += count;
      offset += count;
      if (this.bodyUsed === this.body.length) {
        const text = new TextDecoder('utf-8', { fatal: true }).decode(this.body);
        this.body = undefined;
        this.headerUsed = 0;
        this.receive(JSON.parse(text));
      }
    }
  }
  get partial(): boolean { return this.headerUsed !== 0 || this.body !== undefined; }
}

export interface ValidationWorkerResult {
  record: ExecutionRecord;
  /** Host-only local import reconciliation, independent of the record outcome. */
  dependenciesCurrent?: boolean;
  /** Bounded user logs, host only. Never use these as browser diagnostics. */
  stdout: string;
  stderr: string;
}

function stopped(spec: RunSpec, state: 'failed' | 'cancelled' | 'timed_out', code: string): ExecutionRecord {
  return {
    version: 1, identity: spec.identity, sequence: 1, limits: spec.profile.limits!,
    process: { state, failure: { phase: state === 'failed' ? 'transport' : 'resource', code, source: null } },
    capture: { state: 'incomplete', rootInvocationId: null,
      reason: state === 'cancelled' ? 'cancelled' : state === 'timed_out' ? 'timeout' : 'transport_error' },
    context: null, modules: [], invocations: [], tensors: [], edges: [], requirements: [],
  };
}

/** Launch only from a host-authorized Run action. No interpreter probing/fallback. */
export class ValidationWorker implements vscode.Disposable {
  private active: (() => void) | undefined;
  constructor(private readonly backendDir: string) {}
  cancel(): void { this.active?.(); }
  dispose(): void { this.cancel(); }

  async run(spec: RunSpec, workspaceRoot: string, authorized: boolean, receive?: (record: ExecutionRecord, dependenciesCurrent?: boolean) => void, beforeLaunch?: () => Promise<boolean>): Promise<ValidationWorkerResult> {
    if (!authorized || !vscode.workspace.isTrusted) { throw new Error('Validation requires an authorized Run in a trusted workspace.'); }
    if (this.active) { throw new Error('A validation worker is already active.'); }
    const limits = spec.profile.limits;
    if (!limits || limits.max_events === undefined || limits.max_tensors === undefined
        || limits.max_record_bytes === undefined || limits.timeout_ms === undefined || spec.profile.cwd === undefined) {
      throw new Error('A materialized RunSpec is required.');
    }
    const { max_events, max_tensors, max_record_bytes, timeout_ms } = limits;
    let child: ChildProcess | undefined;
    let socket: net.Socket | undefined;
    let directory: string | undefined;
    let stop: 'cancelled' | 'timed_out' | undefined;
    let timer: NodeJS.Timeout | undefined;
    let killTimer: NodeJS.Timeout | undefined;
    let stdout = Buffer.alloc(0), stderr = Buffer.alloc(0);
    let latest: ExecutionRecord | undefined;
    let transportError = false;
    let spawnFailed = false;
    let authenticated = false;
    let ended = false;
    let closed = false;
    let count = 0;
    let dependenciesCurrent = true;
    const sockets = new Set<net.Socket>();
    const token = randomBytes(32).toString('hex');
    const terminate = (): void => {
      if (!child?.pid) { return; }
      const pid = child.pid;
      if (process.platform === 'win32') {
        // taskkill /T handles descendant processes; child.kill() alone does not.
        const killer = spawn('taskkill', ['/PID', String(pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore', timeout: 1000 });
        const fallback = () => { child?.kill(); child?.stdout?.destroy(); child?.stderr?.destroy(); };
        killer.on('error', fallback);
        killer.on('close', code => { if (code !== 0) fallback(); });
      } else {
        try { process.kill(-pid, 'SIGTERM'); } catch { /* exited */ }
        killTimer ??= setTimeout(() => {
          try { process.kill(-pid, 'SIGKILL'); } catch { /* exited */ }
          if (!closed) { child?.stdout?.destroy(); child?.stderr?.destroy(); }
        }, 250);
      }
    };
    this.active = () => { stop ??= 'cancelled'; terminate(); };
    const server = net.createServer(connection => {
      // One pending connection at a time; reject excess clients without buffering.
      if (socket || sockets.size) { connection.destroy(); return; }
      sockets.add(connection);
      let hello = false;
      const reject = (): void => {
        if (hello) { transportError = true; }
        connection.destroy();
      };
      const frames = new ValidationFrames(
        () => hello ? Math.min(MAX_FRAME, max_record_bytes + 4096) : 1024,
        value => {
          const message = value as { version?: number; type?: string; token?: string; runId?: string; record?: ExecutionRecord; dependenciesCurrent?: boolean };
          if (!message || message.version !== 1) { throw new Error('invalid_frame'); }
          if (!hello) {
            if (Object.keys(message).some(key => !['version', 'type', 'runId', 'token'].includes(key))
                || message.type !== 'hello' || message.runId !== spec.identity.runId
                || typeof message.token !== 'string' || !/^[0-9a-f]{64}$/.test(message.token)
                || !timingSafeEqual(Buffer.from(token), Buffer.from(message.token))) { throw new Error('authentication'); }
            hello = true;
            authenticated = true;
            socket = connection;
            connection.setTimeout(0);
            return;
          }
          const record = message.record;
          if (Object.keys(message).some(key => !['version', 'type', 'record', 'dependenciesCurrent'].includes(key))
              || typeof message.dependenciesCurrent !== 'boolean'
              || message.type !== 'record' || !record || record.version !== 1 || ++count > 8
              || !Number.isSafeInteger(record.sequence) || record.sequence < 0
              || (latest && record.sequence <= latest.sequence)
              || !isDeepStrictEqual(record.identity, spec.identity)
              || !isDeepStrictEqual(record.limits, spec.profile.limits)
              || Buffer.byteLength(JSON.stringify(record)) > max_record_bytes
              || !record.capture || !record.process || !Array.isArray(record.invocations)
              || !Array.isArray(record.tensors) || !Array.isArray(record.modules)
              || !Array.isArray(record.edges) || !Array.isArray(record.requirements)
              || record.invocations.length > max_events
              || record.tensors.length > max_tensors) { throw new Error('invalid_record'); }
          // Python validates relational constraints before sending; the controller
          // independently binds identity, saved revisions and freshness.
          latest = structuredClone(record);
          dependenciesCurrent &&= message.dependenciesCurrent !== false;
          receive?.(structuredClone(record), dependenciesCurrent);
        });
      connection.setTimeout(1000, reject);
      connection.on('data', chunk => { try { frames.push(chunk); } catch { reject(); } });
      connection.on('error', reject);
      connection.on('end', () => {
        if (hello) { ended = true; if (frames.partial || !latest) { transportError = true; } }
      });
      connection.on('close', () => {
        sockets.delete(connection);
        if (hello && !ended) { transportError = true; }
      });
    });
    try {
      await new Promise<void>((resolve, reject) => {
        server.once('error', reject);
        server.listen(0, '127.0.0.1', () => { server.removeListener('error', reject); resolve(); });
      });
      server.on('error', () => { transportError = true; terminate(); });
      directory = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-validation-'));
      if (process.platform === 'win32') {
        execFileSync('icacls', [directory, '/inheritance:r', '/grant:r', `${os.userInfo().username}:(OI)(CI)F`], { stdio: 'ignore' });
      } else {
        await fs.chmod(directory, 0o700);
      }
      const bootstrap = path.join(directory, 'bootstrap.json');
      const body = JSON.stringify({ version: 1, spec, workspaceRoot: await fs.realpath(workspaceRoot), resultChannel: {
        host: '127.0.0.1', port: (server.address() as net.AddressInfo).port, token,
      } });
      if (Buffer.byteLength(body) > 16 * 1024 * 1024) { throw new Error('bootstrap_limit'); }
      await fs.writeFile(bootstrap, body, { flag: 'wx', mode: 0o600 });
      const root = await fs.realpath(workspaceRoot);
      const cwd = await fs.realpath(path.resolve(root, spec.profile.cwd));
      const relative = path.relative(root, cwd);
      if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) { throw new Error('cwd_outside_workspace'); }
      let script: string | undefined;
      if (spec.profile.kind === 'script') {
        script = path.resolve(root, spec.profile.script);
        const scriptRelative = path.relative(root, await fs.realpath(script));
        if (scriptRelative === '..' || scriptRelative.startsWith(`..${path.sep}`) || path.isAbsolute(scriptRelative)
            || !(await fs.stat(script)).isFile()) { throw new Error('script_outside_workspace'); }
      }
      if (stop) { return { record: stopped(spec, stop, stop), stdout: '', stderr: '' }; }
      if (!vscode.workspace.isTrusted) { throw new Error('workspace_trust_revoked'); }
      if (beforeLaunch && !await beforeLaunch()) { throw new Error('stale_before_launch'); }
      if (stop || !vscode.workspace.isTrusted) { throw new Error('cancelled_before_launch'); }
      // Both modes keep their own directory/PYTHONPATH and import namespace. Loading
      // recorder internals through `services` would shadow ordinary user packages.
      // An empty, process-private prefix prevents timestamp-valid user .pyc reads.
      // -B keeps it empty, bounds disk usage, and leaves all user caches untouched.
      const args = ['-B', '-X', `pycache_prefix=${path.join(directory, 'pycache')}`,
        path.join(this.backendDir, 'services/validation/script_bootstrap.py'), ...(script ? [script] : [])];
      const env: NodeJS.ProcessEnv = { ...process.env, [BOOTSTRAP_ENV]: bootstrap, PYTHONDONTWRITEBYTECODE: '1' };
      child = spawn(spec.interpreter.executable, args, {
        cwd, detached: process.platform !== 'win32', windowsHide: true,
        env,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      // OS-observed process state, before Python can finish a capture. This makes
      // even a slow import/forward visibly cancellable without inventing evidence.
      child.once('spawn', () => receive?.({
        version: 1, identity: spec.identity, sequence: 0, limits,
        process: { state: 'running' }, capture: { state: 'waiting' },
        context: null, modules: [], invocations: [], tensors: [], edges: [], requirements: [],
      }));
      timer = setTimeout(() => { stop ??= 'timed_out'; terminate(); }, timeout_ms);
      child.stdout!.on('data', (chunk: Buffer) => {
        if (stdout.length < MAX_LOG) { stdout = Buffer.concat([stdout, chunk.subarray(0, MAX_LOG - stdout.length)]); }
      });
      child.stderr!.on('data', (chunk: Buffer) => {
        if (stderr.length < MAX_LOG) { stderr = Buffer.concat([stderr, chunk.subarray(0, MAX_LOG - stderr.length)]); }
      });
      const code = await new Promise<number | null>(resolve => {
        child!.once('error', () => { spawnFailed = true; resolve(null); });
        child!.once('close', value => { closed = true; resolve(value); });
      });
      // Drain already queued TCP bytes after process close, with a fixed deadline.
      if (socket && !ended && !socket.destroyed) {
        await new Promise<void>(resolve => {
          const drain = setTimeout(resolve, 100);
          socket!.once('close', () => { clearTimeout(drain); resolve(); });
        });
      }
      if (stop) {
        const record = latest ?? stopped(spec, stop, stop);
        record.sequence += 1;
        record.process = stopped(spec, stop, stop).process;
        record.capture = { state: 'incomplete', rootInvocationId: 'rootInvocationId' in record.capture ? record.capture.rootInvocationId : null,
          reason: stop === 'cancelled' ? 'cancelled' : 'timeout' };
        return { record, stdout: stdout.toString(), stderr: stderr.toString() };
      }
      if (spawnFailed) { throw new Error('launch_failed'); }
      if (!authenticated || !ended || transportError || !latest || code === null) {
        const record = latest ? structuredClone(latest) : stopped(spec, 'failed', 'transport_error');
        record.sequence += 1;
        record.process = code === null ? stopped(spec, 'failed', 'transport_error').process : { state: 'exited', exitCode: code };
        record.capture = { state: 'incomplete', rootInvocationId: 'rootInvocationId' in record.capture ? record.capture.rootInvocationId : null, reason: 'transport_error' };
        return { record, stdout: stdout.toString(), stderr: stderr.toString() };
      }
      latest.sequence += 1;
      // OS exit is authoritative even if an early record claimed success.
      if (latest.process.state === 'running' || latest.process.state === 'exited' || code !== 0) { latest.process = { state: 'exited', exitCode: code }; }
      if (latest.capture.state === 'waiting' || latest.capture.state === 'recording') {
        latest.capture = { state: 'incomplete', rootInvocationId: null, reason: 'transport_error' };
      }
      return { record: latest, dependenciesCurrent, stdout: stdout.toString(), stderr: stderr.toString() };
    } catch {
      terminate();
      const record = stopped(spec, stop ?? 'failed', stop ?? 'launch_failed');
      record.sequence = (latest?.sequence ?? 0) + 1;
      if (!stop) { record.process = { state: 'failed', failure: { phase: 'setup', code: 'launch_failed', source: null } }; }
      return { record, stdout: stdout.toString(), stderr: stderr.toString() };
    } finally {
      if (timer) { clearTimeout(timer); }
      // Do not cancel escalation: a descendant can outlive the Python parent.
      killTimer?.unref();
      for (const connection of sockets) { connection.destroy(); }
      await new Promise<void>(resolve => server.close(() => resolve()));
      if (directory) { await fs.rm(directory, { recursive: true, force: true }); }
      this.active = undefined;
    }
  }
}
