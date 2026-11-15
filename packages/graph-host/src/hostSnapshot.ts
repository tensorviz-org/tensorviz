import { spawn, type ChildProcess } from 'node:child_process';
import { backendPythonArgs } from './backendPython';

const ENTRY = 'services/validation/host_snapshot.py';
const TIMEOUT_MS = 30000;
const OUTPUT_LIMIT = 32 * 1024 * 1024;
const FAILED = '{"error":"snapshot_failed"}';
/** A helper nobody has asked for this long is stopped; the next snapshot starts one. */
export let SNAPSHOT_HELPER_IDLE_MS = 5 * 60_000;

/** Common isolated reader for validation profiles and explicit execution plans.
 *
 *  A snapshot is answered by a long-lived helper (`host_snapshot.py --serve`) when
 *  one is free, so each call skips interpreter start and the contract imports that
 *  dominated a one-shot call. The helper never imports project code and resolves
 *  every request from disk, so keeping it alive cannot serve a stale answer. Any
 *  doubt about the helper — busy, cancelled, timed out, crashed, oversized output —
 *  falls back to exactly the one-shot spawn this always used. */
export function readHostSnapshot(python: string, backendDir: string, request: object, signal: AbortSignal): Promise<unknown> {
  if (signal.aborted) return Promise.reject(new Error('cancelled'));
  const env = snapshotEnvironment();
  const key = JSON.stringify([python, backendDir, Object.entries(env).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))]);
  let helper = helpers.get(key);
  if (helper && !helper.usable()) { helper.stop(); helpers.delete(key); helper = undefined; }
  if (!helper) {
    try { helper = new SnapshotHelper(python, backendDir, env); } catch { return oneShot(python, backendDir, env, request, signal); }
    helpers.set(key, helper);
  }
  if (helper.busy) return oneShot(python, backendDir, env, request, signal);
  const used = helper;
  return used.ask(JSON.stringify(request), signal).then(parse, (error: Error) => {
    helpers.delete(key);
    // A cancellation or timeout ends the call as it always did: the one-shot path
    // killed its process and reported snapshot_failed. Only a helper that broke on
    // its own is retried the old way.
    if (error.message === 'cancelled' || error.message === 'timeout') throw new Error('snapshot_failed');
    return oneShot(python, backendDir, env, request, signal);
  });
}

/** Stop every helper (extension deactivation, tests). */
export function disposeSnapshotHelpers(): void {
  for (const helper of helpers.values()) helper.stop();
  helpers.clear();
}

const helpers = new Map<string, SnapshotHelper>();

function snapshotEnvironment(): NodeJS.ProcessEnv {
  return { ...process.env, PYTHONDONTWRITEBYTECODE: '1' };
}

function parse(text: string): unknown {
  if (text === FAILED) throw new Error('snapshot_failed');
  try { return JSON.parse(text); } catch { throw new Error('snapshot_failed'); }
}

type Referable = { ref?(): unknown; unref?(): unknown };
function setReferenced(child: ChildProcess, referenced: boolean): void {
  for (const handle of [child, child.stdin, child.stdout, child.stderr] as (Referable | null)[]) {
    if (referenced) handle?.ref?.(); else handle?.unref?.();
  }
}

class SnapshotHelper {
  private readonly child: ChildProcess;
  private buffer = Buffer.alloc(0);
  private broken = false;
  private idle: ReturnType<typeof setTimeout> | undefined;
  private pending: { resolve(text: string): void; reject(error: Error): void } | undefined;
  busy = false;

  constructor(python: string, backendDir: string, env: NodeJS.ProcessEnv) {
    this.child = spawn(python, backendPythonArgs(backendDir, ENTRY, ['--serve']), {
      cwd: backendDir, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'], env,
    });
    this.child.stdin?.on('error', () => this.fail());
    this.child.stdout?.on('data', (chunk: Buffer) => this.receive(chunk));
    this.child.stderr?.resume();
    this.child.once('error', () => this.fail());
    this.child.once('exit', () => this.fail());
    // An idle helper must never keep the host alive; its stdin closing ends it.
    setReferenced(this.child, false);
    this.arm();
  }

  usable(): boolean { return !this.broken && this.child.exitCode === null && this.child.signalCode === null; }

  ask(line: string, signal: AbortSignal): Promise<string> {
    this.busy = true;
    if (this.idle) clearTimeout(this.idle);
    setReferenced(this.child, true);
    return new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => finish(new Error('timeout')), TIMEOUT_MS);
      const abort = () => finish(new Error('cancelled'));
      const finish = (error?: Error, text?: string) => {
        if (!this.pending) return;
        this.pending = undefined;
        clearTimeout(timer); signal.removeEventListener('abort', abort);
        if (error) { this.stop(); reject(error); return; }
        this.busy = false;
        setReferenced(this.child, false);
        this.arm();
        resolve(text!);
      };
      this.pending = { resolve: text => finish(undefined, text), reject: error => finish(error) };
      signal.addEventListener('abort', abort, { once: true });
      this.child.stdin?.write(line + '\n');
    });
  }

  stop(): void {
    this.broken = true;
    if (this.idle) clearTimeout(this.idle);
    // A Windows venv launcher can exit before its interpreter. EOF also stops
    // that interpreter; keep its pipes referenced until they have closed.
    setReferenced(this.child, true);
    this.child.stdin?.end();
    this.child.kill('SIGKILL');
  }

  private arm(): void {
    this.idle = setTimeout(() => {
      this.stop();
      for (const [key, value] of helpers) if (value === this) helpers.delete(key);
    }, SNAPSHOT_HELPER_IDLE_MS);
    this.idle.unref?.();
  }

  private receive(chunk: Buffer): void {
    if (!this.pending || this.buffer.length + chunk.length > OUTPUT_LIMIT) { this.fail(); return; }
    this.buffer = Buffer.concat([this.buffer, chunk]);
    const end = this.buffer.indexOf(10);
    if (end < 0) return;
    // Exactly one line per request: bytes after it mean the protocol is broken.
    if (end !== this.buffer.length - 1) { this.fail(); return; }
    const text = this.buffer.subarray(0, end).toString();
    this.buffer = Buffer.alloc(0);
    this.pending.resolve(text);
  }

  private fail(): void {
    const pending = this.pending;
    this.broken = true;
    if (pending) pending.reject(new Error('helper_failed'));
    else this.stop();
  }
}

/** The original one process per snapshot. */
function oneShot(python: string, backendDir: string, env: NodeJS.ProcessEnv, request: object, signal: AbortSignal): Promise<unknown> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) { reject(new Error('cancelled')); return; }
    const child = spawn(python, backendPythonArgs(backendDir, ENTRY), {
      cwd: backendDir, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'], env,
    });
    let output = Buffer.alloc(0);
    const stop = () => child.kill('SIGKILL');
    const timer = setTimeout(stop, TIMEOUT_MS);
    signal.addEventListener('abort', stop, { once: true });
    child.stdout.on('data', (chunk: Buffer) => {
      if (output.length + chunk.length > OUTPUT_LIMIT) stop();
      else output = Buffer.concat([output, chunk]);
    });
    child.stderr.resume();
    child.stdin.on('error', () => { /* close/error below owns the result */ });
    child.stdin.end(JSON.stringify(request));
    child.once('error', reject);
    child.once('close', code => {
      clearTimeout(timer); signal.removeEventListener('abort', stop);
      try {
        if (code !== 0 || signal.aborted) throw new Error('snapshot_failed');
        resolve(JSON.parse(output.toString()));
      } catch { reject(new Error('snapshot_failed')); }
    });
  });
}
