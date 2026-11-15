import { randomBytes } from 'node:crypto';
import { ChildProcess, spawn, spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as vscode from 'vscode';

import { BackendLaunch, HOST, backendEnv, backendOrigin, backendReload } from './config';
import { BACKEND_RESTART_LIMIT, planBackendExit } from './backendExit';
import { MAX_WIRE_BYTES } from '@tensorviz/graph-host/localBrowserWire';

const TAIL_LINES = 40;

/** Backend readiness budget, ported from start.sh:80-99 (60 x 1s). */
const BACKEND_ATTEMPTS = 60;

export class StackError extends Error {
  constructor(message: string, readonly detail: string) {
    super(message);
  }
}

type ProbeResult = { ok: true } | { ok: false; why: string };

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function get(
  url: string,
  timeoutMs: number,
  headers: Record<string, string> = {},
): Promise<{ status: number; body: string }> {
  const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
  return { status: res.status, body: await res.text() };
}

/** Count layer types the same way run-e2e.mjs:86-90 does — the endpoint answers
 *  `{"layers": [...]}` (server_v2.py:980-983), but tolerate a bare array too.
 *
 *  Exported only so src/test/unit/probes.parity.test.ts can hold it against the .mjs copy;
 *  the duplication itself is deliberate until the shared-module sync lands (task.md M2). */
export function countLayerTypes(body: string): number {
  try {
    const payload: unknown = JSON.parse(body);
    if (Array.isArray(payload)) {
      return payload.length;
    }
    if (payload && typeof payload === 'object') {
      return Object.values(payload).reduce<number>(
        (n, v) => n + (Array.isArray(v) ? v.length : 0),
        0,
      );
    }
    return 0;
  } catch {
    return 0;
  }
}

/** Kill the child's whole process group.
 *
 * uvicorn is spawned in its own group so a stray grandchild (a trace worker, from
 * M4) cannot outlive it. No tree-kill dependency: it shells out to ps/taskkill to
 * do what a negative-PID group kill does natively, and zero runtime deps means
 * nothing to audit. */
function killTree(child: ChildProcess): void {
  if (child.exitCode !== null || child.signalCode !== null || child.pid === undefined) {
    return;
  }
  const pid = child.pid;
  if (process.platform === 'win32') {
    spawnSync('taskkill', ['/pid', String(pid), '/T', '/F'], { windowsHide: true });
    return;
  }
  try {
    process.kill(-pid, 'SIGTERM');
  } catch {
    return; // already gone
  }
  const escalate = setTimeout(() => {
    if (child.exitCode === null) {
      try {
        process.kill(-pid, 'SIGKILL');
      } catch {
        /* already gone */
      }
    }
  }, 5000);
  escalate.unref();
}

/** Private extension state; never write the launch credential into the source tree. */
function writeLockFile(stateDir: string, launch: BackendLaunch, pid: number | undefined): string {
  const dir = path.join(stateDir, '.tensorviz');
  fs.mkdirSync(dir, { recursive: true });
  const lock = path.join(dir, 'lock');
  fs.writeFileSync(
    lock,
    JSON.stringify({ port: launch.port, pid, token: launch.token, startedAt: new Date().toISOString() }, null, 2),
    { mode: 0o600 },
  );
  // writeFileSync only applies `mode` when it creates the file, so an existing
  // lock from a previous launch would keep whatever permissions it had.
  fs.chmodSync(lock, 0o600);
  return lock;
}

/** 32 bytes of CSPRNG, hex. Regenerated per launch: a token that outlived the
 *  process it authorised would be a credential sitting in a file for no reason. */
export function newLaunchToken(): string {
  return randomBytes(32).toString('hex');
}

/** The uvicorn child: one process, one workspace, one loopback port.
 *
 * This used to be a two-child `Stack` (uvicorn plus `npm start`). Stage A
 * (task.md D2) deleted the second child: the backend serves the built React app
 * from TENSORVIZ_STATIC_DIR, so there is no dev server to compile, no ~700 MB of
 * webpack, no 30–60 s cold start and no Node requirement on the user's machine.
 */
export class BackendProcess implements vscode.Disposable {
  private proc: ChildProcess | undefined;
  private readonly tail: string[] = [];
  private shuttingDown = false;
  private restarts = 0;
  private stateDir?: string;

  /** Fires when the backend dies *after* readiness. Without this, killing uvicorn
   *  leaves a fully-rendered canvas whose every action fails — the trap
   *  start.sh:28-36 was written about. */
  onUnexpectedExit: ((message: string, detail: string) => void) | undefined;
  /** Fires before an in-process restart. The canvas stays attached to this process. */
  onRestarting: ((message: string, detail: string) => void) | undefined;

  constructor(
    readonly backendDir: string,
    readonly interpreter: string,
    private readonly channel: vscode.OutputChannel,
    readonly launch: BackendLaunch,
  ) {}

  get origin(): string {
    return backendOrigin(this.launch.port);
  }

  private get authHeaders(): Record<string, string> {
    return { 'X-tensorViz-Token': this.launch.token };
  }

  private spawnBackend(): ChildProcess {
    const args = [
      // No --reload, unlike start.sh:60-68: the reloader is a supervisor that forks
      // a worker, doubling the tree-kill surface for a benefit the panel
      // close/reopen cycle already provides. --host 127.0.0.1, not 0.0.0.0,
      // because auth is forced off here.
      '-m',
      'uvicorn',
      'server_v2:app',
      '--host',
      HOST,
      '--port',
      String(this.launch.port),
      // The broker's frame bound; uvicorn's default is 16 MiB.
      '--ws-max-size',
      String(MAX_WIRE_BYTES),
    ];
    const cwd = this.backendDir;
    if (backendReload(process.env)) {
      // The dev loop (ide/dev.sh) opts in. The supervisor forks a worker, but both sit in the
      // process group killTree takes down, so the tradeoff above is a second pid, not a leak.
      args.push('--reload', '--reload-dir', cwd);
    }
    this.channel.appendLine(`[backend] spawn: ${this.interpreter} ${args.join(' ')}`);
    this.channel.appendLine(`[backend] cwd:   ${cwd}`);
    this.channel.appendLine(`[backend] serving the app from ${this.launch.staticDir}`);
    const proc = spawn(this.interpreter, args, {
      cwd,
      env: backendEnv(process.env, this.launch) as NodeJS.ProcessEnv,
      stdio: ['ignore', 'pipe', 'pipe'],
      // Own process group so killTree can take the whole tree down. The tradeoff is
      // that children survive an extension-host SIGKILL (which skips deactivate) —
      // with a per-launch port that leaks a process, not a port everything else needs.
      detached: process.platform !== 'win32',
      windowsHide: true,
    });

    const consume = (buf: Buffer): void => {
      for (const line of buf.toString().split('\n')) {
        if (!line.trim()) {
          continue;
        }
        this.channel.appendLine(`[backend] ${line}`);
        this.tail.push(line);
        if (this.tail.length > TAIL_LINES) {
          this.tail.shift();
        }
      }
    };
    proc.stdout?.on('data', consume);
    proc.stderr?.on('data', consume);

    proc.on('exit', (code, signal) => {
      this.channel.appendLine(`[backend] exited (code ${code}, signal ${signal})`);
      if (this.shuttingDown || this.proc !== proc) return;
      const plan = planBackendExit(code, signal, this.restarts);
      const detail = this.tail.join('\n');
      if (!plan.restart || !this.stateDir) {
        this.onUnexpectedExit?.(plan.notice, detail);
        return;
      }
      this.restarts += 1;
      const message = `${plan.notice} Restarting it (${this.restarts}/${BACKEND_RESTART_LIMIT}).`;
      this.channel.appendLine(`[backend] ${message}`);
      this.onRestarting?.(message, detail);
      const next = this.spawnBackend();
      this.proc = next;
      const stateDir = this.stateDir;
      void this.waitFor(next, () => this.probeHealth(), () => {}, () => this.shuttingDown).then(() => {
        if (this.shuttingDown || this.proc !== next) return;
        writeLockFile(stateDir, this.launch, next.pid);
        this.channel.appendLine(`[backend] restarted on ${this.origin}`);
      }).catch((error: Error) => {
        if (this.shuttingDown || this.proc !== next) return;
        this.onUnexpectedExit?.(`${plan.notice} The restart did not become ready. ${error.message}`, this.tail.join('\n'));
      });
    });
    return proc;
  }

  private async probeHealth(): Promise<ProbeResult> {
    try {
      const res = await get(`${this.origin}/health`, 2000);
      if (res.status !== 200) return { ok: false, why: `/health returned ${res.status}` };
      return { ok: true };
    } catch (err) {
      return { ok: false, why: `/health unreachable (${(err as Error).message})` };
    }
  }

  /** Liveness is checked every tick, so a backend that dies at t=2s reports at
   *  t=2s rather than at the timeout — the fail-fast behaviour of start.sh:84-88,
   *  which checks `kill -0 $backend_pid` on every iteration. */
  private async waitFor(
    proc: ChildProcess,
    probe: () => Promise<ProbeResult>,
    report: (message: string) => void,
    isCancelled: () => boolean,
  ): Promise<void> {
    let lastWhy = 'no attempt completed';
    for (let i = 1; i <= BACKEND_ATTEMPTS; i++) {
      if (isCancelled()) {
        throw new StackError('Startup cancelled.', this.tail.join('\n'));
      }
      if (proc.exitCode !== null || proc.signalCode !== null) {
        throw new StackError(
          `The backend exited before becoming ready (code ${proc.exitCode}, signal ${proc.signalCode}).`,
          this.tail.join('\n'),
        );
      }
      const result = await probe();
      if (result.ok) {
        return;
      }
      lastWhy = result.why;
      report(`Waiting for the backend… (${i}/${BACKEND_ATTEMPTS})`);
      await sleep(1000);
    }
    throw new StackError(
      `The backend did not become ready within ${BACKEND_ATTEMPTS} seconds: ${lastWhy}`,
      this.tail.join('\n'),
    );
  }

  /** `stateDir` is VS Code-owned storage for this window's backend state. */
  async start(
    stateDir: string,
    report: (message: string) => void,
    isCancelled: () => boolean,
  ): Promise<void> {
    this.stateDir = stateDir;
    report('Starting the backend…');
    const proc = this.spawnBackend();
    this.proc = proc;

    let healthBody = '';
    await this.waitFor(
      proc,
      async () => {
        try {
          const res = await get(`${this.origin}/health`, 2000);
          if (res.status !== 200) {
            return { ok: false, why: `/health returned ${res.status}` };
          }
          healthBody = res.body;
          return { ok: true };
        } catch (err) {
          return { ok: false, why: `/health unreachable (${(err as Error).message})` };
        }
      },
      report,
      isCancelled,
    );

    // A backend that came up without the static mount would render a 404 in the
    // panel and nothing else, so it is refused here rather than behind an empty
    // iframe. `capabilities` is server_v2's own account of what it is serving.
    const capabilities = this.capabilities(healthBody);
    if (!capabilities.static) {
      throw new StackError(
        'The backend started without the tensorViz app, so the panel would have nothing to show.',
        `TENSORVIZ_STATIC_DIR=${this.launch.staticDir}\n/health said capabilities.static=false\n\n` +
          'Run `node extension/scripts/build-webview.mjs` to build the app into the extension.\n\n' +
          this.tail.join('\n'),
      );
    }

    // A backend that is listening but not serving the model plane must fail here
    // rather than behind a rendered canvas. Same probe and same reasoning as
    // run-e2e.mjs:82-95 — with the launch token, which every /api/* route needs.
    report('Checking the backend is serving layer types…');
    const layerTypes = await get(`${this.origin}/api/layer-types`, 8000, this.authHeaders).catch(
      (err: Error) => ({ status: 0, body: err.message }),
    );
    if (layerTypes.status !== 200 || countLayerTypes(layerTypes.body) === 0) {
      throw new StackError(
        '/api/layer-types returned no layer types — the backend is up but not serving the model plane.',
        `status ${layerTypes.status}\n${layerTypes.body.slice(0, 400)}\n\n${this.tail.join('\n')}`,
      );
    }

    const lock = writeLockFile(stateDir, this.launch, proc.pid);
    this.channel.appendLine(`[backend] ready on ${this.origin} (lock: ${lock})`);
  }

  /** The `environment` block /health reports, for the status bar. Undefined when
   *  the readout could not be parsed — the status bar says so rather than
   *  inventing a version. */
  environment(healthBody: string): { torch?: string; device?: string; python?: string } | undefined {
    try {
      return (JSON.parse(healthBody) as { environment?: Record<string, string> }).environment;
    } catch {
      return undefined;
    }
  }

  private capabilities(healthBody: string): { ide: boolean; static: boolean } {
    try {
      const parsed = JSON.parse(healthBody) as { capabilities?: { ide?: boolean; static?: boolean } };
      return { ide: Boolean(parsed.capabilities?.ide), static: Boolean(parsed.capabilities?.static) };
    } catch {
      return { ide: false, static: false };
    }
  }

  /** Read /health once more, for callers that want the environment readout. */
  async health(): Promise<string> {
    try {
      return (await get(`${this.origin}/health`, 2000)).body;
    } catch {
      return '';
    }
  }

  dispose(): void {
    this.shuttingDown = true;
    if (this.proc) {
      killTree(this.proc);
    }
    this.channel.appendLine('backend stopped');
  }
}
