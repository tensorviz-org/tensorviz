/** The managed Python runtime the tensorViz IDE provisions for its own backend.
 *
 * The fork's promise (task.md D11 / F3) is that the backend needs no environment of the
 * user's. `ide/scripts/fetch-runtime.sh` stages a relocatable standalone CPython into
 * `resources/app/runtime/` and `ide/build.sh` copies `backend/` to
 * `resources/app/backend/`; this module turns that pair into a working interpreter the first
 * time the canvas is opened, and re-checks it on every launch afterwards.
 *
 * Three rules shape everything below.
 *
 * 1. The venv is created from a COPY of the runtime under the extension's globalStorage, never
 *    from `resources/app/runtime/` itself. A venv records its base interpreter as an absolute
 *    path in `pyvenv.cfg home=`; if that path is inside the .app, the first drag from the
 *    downloaded disk image to /Applications breaks every launch after it. The copy also
 *    survives an IDE update replacing the bundle underneath a running window.
 *
 * 2. globalStorage, not `dataFolderName`. `dataFolderName` is where extensions are installed,
 *    which is emphatically not ours to write ~850 MB into; `context.globalStorageUri` is the
 *    per-extension directory VS Code hands out for exactly this.
 *
 * 3. In desktop Remote SSH the workspace extension provisions on the SSH host, using
 *    that host's manifest asset and globalStorage. Other remote kinds are unsupported.
 *
 * 4. Stock VS Code has no bundle either, and it is NOT a fallback to the user's environment:
 *    it acquires the same pinned python-build-standalone interpreter itself. The canonical
 *    pins live in runtime/python/manifest.json (the SAME values fetch-runtime.sh carries,
 *    with a drift test in runtimeManifest.test.ts). The runtime manifest's SHA-256 is verified
 *    while streaming to a `.part` file, the archive is inspected member by member and
 *    extracted to a sticky staging directory, and only a verified interpreter is renamed into
 *    service. A first install never downloads anything without the disclosed consent the
 *    preflight gate asks for, and nothing downloads at activation at all.
 *
 * The stock path downloads CPython and service dependencies on first launch. Model
 * frameworks are installed only through a separately reviewed capture setup.
 */

import { ChildProcess, spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as vscode from 'vscode';

import {
  downloadVerified,
  extractPythonRuntime,
  Transport,
  verifyRuntimeInterpreter,
} from '@tensorviz/graph-host/runtimeAcquire';
import { loadRuntimeManifest, RuntimeManifest, selectRuntimeAsset } from '@tensorviz/graph-host/runtimeManifest';
import { unsupportedRemote } from './remoteSupport';

/** Written into `resources/app/runtime/` by ide/scripts/fetch-runtime.sh. Its presence is how
 *  the extension tells "I am the built-in of a fork build" from "I am installed in stock VS
 *  Code", without asking product.json anything. */
export const RUNTIME_STAMP = '.tensorviz-runtime.json';

/** Written into the provisioned copy. Records what it was built from, so an IDE update that
 *  ships a newer interpreter rebuilds it; changed dependencies are reconciled additively
 *  before the stamp is renewed. */
export const PROVISION_STAMP = '.tensorviz-provisioned.json';

/** Service dependencies only. Model frameworks belong to the separately reviewed
 *  capture environment, never to managed service provisioning. */
export const REQUIREMENTS = 'requirements-standalone.txt';

/** The oldest Python the backend supports. torch 2.14.1's wheels start at 3.11, and the
 *  reported launch crash was a Conda Python 3.7 that passed the old three-module probe and
 *  then died at startup importing dotenv. The gate runs before any third-party import so the
 *  answer is "your Python is too old", not whatever an incompatible torch happens to say. */
export const MIN_BACKEND_PYTHON: readonly [number, number] = [3, 11];

/** What makes an interpreter able to serve the backend. The application import, not a
 *  hand-maintained module list: the list this replaced (`import sys, torch, uvicorn,
 *  fastapi`) passed on exactly the interpreter that crashed, because python-dotenv — a
 *  transitive startup dependency — was never imported until uvicorn loaded server_v2. The
 *  probe runs with the BACKEND directory as cwd, so `import server_v2` resolves the same
 *  staged files and sibling packages the real launch will load, and any import that would
 *  fail at startup fails here, before a process is spawned and a traceback is shown.
 *  Loading the actual server configuration also verifies WebSocket support: uvicorn
 *  can import and serve HTTP without any optional WebSocket implementation.
 *
 *  Exported so tests can pin the contract instead of re-deriving it: a probe that does not
 *  gate the version and import the app is the old bug. preflight.ts re-exports it: the
 *  launch-time resolution and the provision-time verification are the same contract. */
export const BACKEND_PROBE = [
  'import sys',
  'print(sys.version.split()[0])',
  `if sys.version_info[:2] < (${MIN_BACKEND_PYTHON[0]}, ${MIN_BACKEND_PYTHON[1]}):`,
  '    print("the tensorViz backend needs Python ' +
    `${MIN_BACKEND_PYTHON[0]}.${MIN_BACKEND_PYTHON[1]} or newer; ` +
    'this is " + sys.version.split()[0], file=sys.stderr)',
  '    raise SystemExit(17)',
  'import server_v2',
  'from uvicorn import Config',
  'config = Config(server_v2.app, log_config=None)',
  'config.load()',
  'if config.ws_protocol_class is None:',
  '    raise RuntimeError("WebSocket support is missing; install websockets or repair the tensorViz runtime.")',
].join('\n');

/** A cold `import torch` off a spinning disk genuinely takes tens of seconds, the same
 *  reasoning as preflight's interpreter probe budget. */
const BACKEND_PROBE_TIMEOUT_MS = 90_000;
const VENV_TIMEOUT_MS = 180_000;
/** torch alone is a 69–190 MB wheel and the closure is ~780 MB installed; on a slow link the
 *  install legitimately runs for a long time. The budget exists so a wedged pip eventually
 *  reports instead of hanging the first-run experience forever. */
const PIP_TIMEOUT_MS = 45 * 60_000;

/** Enough of the child's output to explain a failure without pasting a whole pip run. */
const TAIL_LINES = 60;

function exists(p: string): boolean {
  try {
    fs.statSync(p);
    return true;
  } catch {
    return false;
  }
}

/** Where the interpreter is inside a python-build-standalone tree. */
export function runtimeInterpreter(runtimeDir: string): string {
  return process.platform === 'win32'
    ? path.join(runtimeDir, 'python.exe')
    : path.join(runtimeDir, 'bin', 'python3');
}

/** Where the interpreter is inside a venv. */
export function venvInterpreter(venvDir: string): string {
  return process.platform === 'win32'
    ? path.join(venvDir, 'Scripts', 'python.exe')
    : path.join(venvDir, 'bin', 'python');
}

/** The two directories ide/build.sh stages next to the built-in.
 *
 * A built-in lives at `<app>/extensions/tensorviz`, so the app directory is two levels up —
 * `<app>/resources/app` on Linux and Windows, `<...>.app/Contents/Resources/app` on macOS. In
 * stock VS Code the extension is under `~/.vscode/extensions/` and neither directory exists,
 * which is the same answer by a different route: no managed runtime here. */
export function bundledLayout(extensionPath: string): { runtime: string; backend: string } {
  const app = path.resolve(extensionPath, '..', '..');
  return { runtime: path.join(app, 'runtime'), backend: path.join(app, 'backend') };
}

export interface ManagedRuntimeOptions {
  /** `context.globalStorageUri.fsPath`. */
  globalStorageDir: string;
  /** `context.extensionPath`. */
  extensionPath: string;
  /** `vscode.env.remoteName` — undefined for a local window. */
  remoteName: string | undefined;
  /** The directory the staged backend lives in right now — the requirements the runtime is
   *  provisioned from and the cwd its verification probe imports with. A thunk, because it is
   *  resolveBackendDir's answer and that can change while a window is open ("tensorviz.repoPath"
   *  edits, a newly opened canvas checkout). */
  backendDir?: () => string | undefined;
  /** Test seam: the download transport. Production makes real HTTPS requests; the unit suite
   *  must be able to run a stock-install provision with fabricated bytes. */
  transport?: Transport;
  /** Test seam: load the pinned manifest from this FILE instead of `<extensionPath>/
   *  runtime-manifest.json`. Electron legs re-pin a copy to their fabricated archive so the
   *  installed extension's real pin never moves. */
  manifestPath?: string;
}

/** Where the interpreter itself comes from. A bundled runtime was staged by
 *  ide/scripts/fetch-runtime.sh into the app; a download runtime is fetched, pinned and
 *  verified from the manifest at first use by the packaged stock extension. Both provision
 *  into the same globalStorage layout afterwards, and everything downstream of provisioning
 *  (venv, pip, the backend probe) is genuinely identical code. */
export type RuntimeSource = BundledRuntimeSource | DownloadRuntimeSource;

export interface BundledRuntimeSource {
  kind: 'bundled';
}

export interface DownloadRuntimeSource {
  kind: 'download';
  manifest: RuntimeManifest;
  /** The pinned asset filename, e.g. `cpython-3.12.14+20260901-aarch64-apple-darwin-….tar.gz`. */
  asset: string;
  /** The HTTPS URL the archive streams from (redirect-bounded, checksum-pinned). */
  url: string;
  sha256: string;
  /** Where the interpreter sits inside the extracted tree, e.g. `bin/python3`. */
  interpreter: string;
  /** The pinned CPython version, e.g. `3.12.14` — checked against `--version` of what arrived. */
  python: string;
}

export type RuntimeStatus =
  /** There is no managed runtime to use here at all, and that is not an error. */
  | { state: 'unavailable'; why: string }
  /** A runtime is bundled but nothing has been provisioned from it yet. */
  | { state: 'absent'; why: string }
  /** Provisioned, but it can no longer import what the backend needs. */
  | { state: 'broken'; why: string }
  | { state: 'ready'; interpreter: string };

type MeasureSetup = <T>(stage: string, operation: () => Promise<T>) => Promise<T>;

export type ProvisionResult = (
  | { ok: true; interpreter: string; elapsedMs: number }
  | { ok: false; message: string; detail: string; cancelled?: boolean }
) & { timingsMs?: Record<string, number> };

export type Report = (message: string) => void;

export interface ProvisionOptions {
  /** Skip the "another window already did it" check after the lock is taken. Repair passes
   *  this: the user asked for a rebuild, so a healthy leftover is not a reason to stop. */
  force?: boolean;
  /** How long to wait for another extension host's setup before giving up. Tests shrink it. */
  lockWaitMs?: number;
}

/** `<root>/lock` — a directory, because mkdir is the primitive that fails atomically when
 *  another host got there first (on every filesystem we ship on). */
const LOCK_DIRNAME = 'lock';
/** Owner record inside the lock directory: who holds it and since when. */
const LOCK_OWNER = 'owner.json';
/** A first install downloads CPython and service wheels; allow slow connections. */
const LOCK_WAIT_DEFAULT_MS = 15 * 60_000;
const LOCK_POLL_MS = 250;
/** When the owner record is missing or corrupt, only age can condemn the lock: a process can
 *  die between mkdir and the owner write, but a healthy holder rewrites nothing for hours. */
const LOCK_STALE_MS = 60 * 60_000;

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/** fs.rmSync with the blast radius proven: every deletion this module performs lives under
 *  the tensorViz globalStorage runtime root, and a bug that computes a wrong path must throw
 *  HERE rather than remove something of the user's. The root itself is never a target. */
export function confinedRm(root: string, target: string, opts: fs.RmOptions = {}): void {
  const resolvedRoot = path.resolve(root);
  const resolvedTarget = path.resolve(target);
  if (resolvedTarget === resolvedRoot || !resolvedTarget.startsWith(resolvedRoot + path.sep)) {
    throw new Error(
      `refusing to remove ${resolvedTarget}: it is not inside the runtime root ${resolvedRoot}`,
    );
  }
  fs.rmSync(resolvedTarget, opts);
}


interface RunResult {
  code: number | null;
  tail: string;
  error: NodeJS.ErrnoException | undefined;
  timedOut: boolean;
  cancelled: boolean;
}

/** Run a child to completion, streaming its last line to `report` and keeping a tail for the
 *  failure message. Cancellation is polled rather than wired to an AbortSignal because the
 *  caller's cancellation source is a VS Code CancellationToken owned by the progress
 *  notification, and a poll keeps this module free of any vscode import on the hot path. */
function run(
  cmd: string,
  args: string[],
  opts: {
    timeoutMs: number;
    label: string;
    report?: Report;
    isCancelled?: () => boolean;
    /** The backend directory a probe imports from. Defaults to the extension host's cwd. */
    cwd?: string;
    /** The environment helper starts pip children; cancel the whole process tree. */
    killTree?: boolean;
  },
): Promise<RunResult> {
  return new Promise<RunResult>((resolve) => {
    let child: ChildProcess;
    // A stray PYTHONHOME or PYTHONPATH from the user's shell points a self-contained
    // interpreter at somebody else's standard library, and the resulting ImportError names a
    // module rather than the cause. The managed runtime is ours end to end; nothing of the
    // user's environment has any business in it.
    const env: NodeJS.ProcessEnv = {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      PIP_DISABLE_PIP_VERSION_CHECK: '1',
    };
    delete env.PYTHONHOME;
    delete env.PYTHONPATH;
    try {
      child = spawn(cmd, args, {
        cwd: opts.cwd,
        detached: opts.killTree && process.platform !== 'win32',
        env,
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
      });
    } catch (err) {
      resolve({
        code: null,
        tail: '',
        error: err as NodeJS.ErrnoException,
        timedOut: false,
        cancelled: false,
      });
      return;
    }

    const tail: string[] = [];
    let timedOut = false;
    let cancelled = false;
    let settled = false;

    const consume = (buf: Buffer): void => {
      for (const line of buf.toString().split('\n')) {
        const text = line.trim();
        if (!text) {
          continue;
        }
        tail.push(text);
        if (tail.length > TAIL_LINES) {
          tail.shift();
        }
        // pip's progress bars are carriage-return animations; truncating keeps the
        // notification from reflowing on every frame.
        opts.report?.(`${opts.label}: ${text.slice(0, 90)}`);
      }
    };
    child.stdout?.on('data', consume);
    child.stderr?.on('data', consume);

    const stop = (): void => {
      if (!opts.killTree || !child.pid) { child.kill('SIGKILL'); return; }
      try {
        if (process.platform === 'win32') {
          const killer = spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { windowsHide: true });
          killer.on('error', () => child.kill('SIGKILL'));
        } else process.kill(-child.pid, 'SIGKILL');
      } catch { /* The process tree already exited. */ }
    };

    const timer = setTimeout(() => {
      timedOut = true;
      stop();
    }, opts.timeoutMs);
    timer.unref?.();

    const poll = opts.isCancelled
      ? setInterval(() => {
          if (opts.isCancelled?.()) {
            cancelled = true;
            stop();
          }
        }, 250)
      : undefined;
    poll?.unref?.();

    const settle = (code: number | null, error: NodeJS.ErrnoException | undefined): void => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      if (poll) {
        clearInterval(poll);
      }
      resolve({ code, tail: tail.join('\n'), error, timedOut, cancelled });
    };
    child.on('error', (err) => settle(null, err as NodeJS.ErrnoException));
    child.on('close', (code) => settle(code, undefined));
  });
}

function why(r: RunResult): string {
  if (r.cancelled) {
    return 'cancelled';
  }
  if (r.timedOut) {
    return 'no answer within the budget';
  }
  if (r.error?.code === 'ENOENT') {
    return 'the interpreter is not there';
  }
  if (r.error) {
    return r.error.message;
  }
  return r.tail.split('\n').pop() || `exit ${r.code}`;
}

/** Hash the service requirements; capture pins must not trigger service setup. */
function requirementsDigest(backendDir: string): string {
  const hash = createHash('sha256');
  for (const name of [REQUIREMENTS]) {
    hash.update(fs.readFileSync(path.join(backendDir, name)));
  }
  return hash.digest('hex');
}

/** Stream the cached archive through sha256 before deciding a download is unneeded. Bytes
 *  get read regardless — the alternative is trusting a path exists, and a truncated cache
 *  entry would silently satisfy it. */
function sha256File(file: string): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const hash = createHash('sha256');
    const reader = fs.createReadStream(file);
    reader.on('data', (chunk) => hash.update(chunk));
    reader.on('error', reject);
    reader.on('end', () => resolve(hash.digest('hex')));
  });
}

/** `check()`'s Python probe, memoised for the window's lifetime.
 *
 *  The probe spawns an interpreter and imports the whole backend application — the
 *  single most expensive thing on the Validate path — and `managedBackendRuntime`
 *  builds a FRESH `ManagedRuntime` per call, so an instance field would never hit.
 *  Hence module scope.
 *
 *  The provision stamp is the invalidation token rather than a `clear()` someone has
 *  to remember: provisioning and repair both rewrite it (`stampProvision`), so a
 *  reprovisioned or repaired runtime lands on a different key and re-probes by
 *  construction. An interpreter mended by hand outside those flows keeps the old
 *  answer until the window reloads, which is the documented limit of this cache. */
const probeCache = new Map<string, Promise<RuntimeStatus>>();

/** Forget every memoised probe. For tests, which build many runtimes over one process. */
export function resetRuntimeProbeCache(): void {
  probeCache.clear();
}

export class ManagedRuntime {
  constructor(private readonly options: ManagedRuntimeOptions) {}

  /** Policy-owned copy under globalStorage, separate from older live runtimes. */
  get root(): string {
    // A new CPU policy gets a new namespace. Never repair an older live
    // runtime in place merely because this build changed its dependency policy.
    return path.join(this.options.globalStorageDir, 'runtime-cpu2141-v1');
  }

  /** The copied standalone interpreter the venv is built from. */
  get base(): string {
    return path.join(this.root, 'python');
  }

  get venv(): string {
    return path.join(this.root, 'venv');
  }

  get stampPath(): string {
    return path.join(this.root, PROVISION_STAMP);
  }

  /** Verified download cache. Survives a failed provisioning: the next attempt must not
   *  re-download 100+ MB because a later pip step hiccuped. */
  get downloads(): string {
    return path.join(this.root, 'downloads');
  }

  /** What preflight would hand to `uvicorn`. */
  get interpreter(): string {
    return venvInterpreter(this.venv);
  }

  get bundled(): { runtime: string; backend: string } {
    return bundledLayout(this.options.extensionPath);
  }

  /** Whether a managed runtime is even a possibility in this window, and if it is, where the
   *  interpreter comes from. Stock VS Code, including desktop SSH, answers yes via
   *  the pinned download source — resolving the answer loads the manifest and asks the
   *  backendDir thunk, but touches neither the network nor the workspace. */
  availability(): { ok: true; source: RuntimeSource } | { ok: false; why: string } {
    const unsupported = unsupportedRemote(this.options.remoteName);
    if (unsupported) return { ok: false, why: unsupported };
    const { runtime, backend } = this.bundled;
    if (exists(path.join(runtime, RUNTIME_STAMP))) {
      if (!exists(path.join(backend, REQUIREMENTS))) {
        return { ok: false, why: `no backend is bundled with this build (looked in ${backend})` };
      }
      return { ok: true, source: { kind: 'bundled' } };
    }
    // Stock VS Code: nothing bundled, so the pinned standalone Python is a verified download.
    let manifest: RuntimeManifest;
    try {
      manifest = loadRuntimeManifest(
        this.options.extensionPath,
        this.options.manifestPath,
      );
    } catch (err) {
      return { ok: false, why: `no pinned runtime manifest (${String(err)})` };
    }
    const selection = selectRuntimeAsset(manifest);
    if (!selection.ok) {
      return { ok: false, why: selection.message };
    }
    // No backend, no runtime: the requirements the venv is provisioned from and the
    // application the verify probe imports are both read from this directory.
    const backendDir = this.options.backendDir?.();
    if (!backendDir || !exists(path.join(backendDir, REQUIREMENTS))) {
      return {
        ok: false,
        why:
          'no backend directory was resolved to provision the managed runtime against ' +
          '(resolveBackendDir found none)',
      };
    }
    return {
      ok: true,
      source: {
        kind: 'download',
        manifest,
        asset: selection.asset,
        url: selection.url,
        sha256: selection.sha256,
        interpreter: selection.pin.interpreter,
        python: manifest.python,
      },
    };
  }

  /** A provision stamp exists only when consent was given and an install finished — the gate
   *  preflight's first-install consent prompt reads. Everything else (repair, explicit
   *  re-install) is not a fresh decision the user never made. */
  get previouslyProvisioned(): boolean {
    return exists(this.stampPath);
  }

  /** The directory the backend probe imports from for this source. availability() has already
   *  proven it for the download kind; the thunk is re-read, because "tensorviz.repoPath" can
   *  change the answer between two calls inside one window. */
  private backendDirFor(source: RuntimeSource): string | undefined {
    if (source.kind === 'bundled') {
      return this.bundled.backend;
    }
    return this.options.backendDir?.();
  }

  /** Run on every launch, per F3. `ready` is the only state that yields an interpreter;
   *  `absent` and `broken` are both repairable and are what routes the user to
   *  `tensorviz.repairRuntime`. */
  async check(): Promise<RuntimeStatus> {
    const available = this.availability();
    if (!available.ok) {
      return { state: 'unavailable', why: available.why };
    }
    if (!exists(this.interpreter)) {
      return { state: 'absent', why: `${this.interpreter} does not exist yet` };
    }
    const stale = this.staleReason();
    if (stale) {
      return { state: 'absent', why: stale };
    }
    const backendDir = this.backendDirFor(available.source);
    if (!backendDir) {
      return { state: 'absent', why: 'no backend directory was resolved for the check' };
    }
    // Everything above is a file-system answer and stays uncached; only the spawn is
    // memoised. The key carries the stamp, so a reprovision or repair re-probes.
    const key = [this.root, this.interpreter, this.provisionStampToken()].join('\u0000');
    const cached = probeCache.get(key);
    if (cached) {
      return cached;
    }
    const pending = run(this.interpreter, ['-c', BACKEND_PROBE], {
      timeoutMs: BACKEND_PROBE_TIMEOUT_MS,
      label: 'runtime check',
      cwd: backendDir,
    }).then((probe): RuntimeStatus =>
      probe.code === 0
        ? { state: 'ready', interpreter: this.interpreter }
        : { state: 'broken', why: `the backend application probe failed: ${why(probe)}` },
    );
    // A rejected promise must never be the cached answer, or one transient spawn
    // failure poisons the window.
    pending.catch(() => probeCache.delete(key));
    probeCache.set(key, pending);
    return pending;
  }

  /** The provision stamp verbatim, or a marker when it cannot be read. Only ever a
   *  cache key — `staleReason` is what actually judges the stamp's contents. */
  private provisionStampToken(): string {
    try {
      return fs.readFileSync(path.join(this.root, PROVISION_STAMP), 'utf8');
    } catch {
      return '\u0000unstamped';
    }
  }

  /** Why the provisioned copy no longer matches the bundle, or undefined when it does.
   *  Changed dependencies need reconciliation before this stamp can certify the venv. */
  private staleReason(): string | undefined {
    let provisioned: { runtime?: string; requirements?: string };
    try {
      provisioned = JSON.parse(fs.readFileSync(path.join(this.root, PROVISION_STAMP), 'utf8'));
    } catch {
      return 'the provisioned runtime carries no stamp, so what it was built from is unknown';
    }
    let wanted: { runtime: string; requirements: string };
     try {
      wanted = this.pinnedStamp();
     } catch (err) {
      // availability() proved the two files it needs are there, so anything that lands here is
      // a corrupt or truncated bundle. Reprovisioning cannot fix that, but saying so beats
      // throwing out of a launch-time check.
      return `this build's runtime stamp could not be read (${String(err)})`;
    }
    if (provisioned.runtime !== wanted.runtime) {
      return 'this build ships a different Python than the provisioned copy was made from';
    }
    if (provisioned.requirements !== wanted.requirements) {
      return "this build's backend dependencies differ from the provisioned copy's";
    }
    return undefined;
  }

  /** What THIS build's pins say the provisioned copy must be: the sha256 of the interpreter —
   *  for the bundled source the stamp ide/scripts/fetch-runtime.sh wrote, for the download
   *  source the manifest's pin — and a digest of the two requirements files the venv is built
   *  from. Public because it is the whole of the staleness contract, and a test that had to
   *  recompute it could not fail when it changed. */
  pinnedStamp(): { runtime: string; requirements: string } {
    const available = this.availability();
    if (!available.ok) {
      throw new Error(`no pinned stamp for an unavailable runtime (${available.why})`);
    }
    const source = available.source;
    const backendDir = this.backendDirFor(source);
    if (source.kind === 'bundled') {
      const stamp = JSON.parse(
        fs.readFileSync(path.join(this.bundled.runtime, RUNTIME_STAMP), 'utf8'),
      ) as { sha256?: string };
      return {
        runtime: String(stamp.sha256),
        requirements: requirementsDigest(this.bundled.backend),
      };
    }
    // availability() proved it; throwing here would be a program bug, not a user state.
    if (!backendDir) {
      throw new Error('no backend directory was resolved for the pinned stamp');
    }
    return { runtime: source.sha256, requirements: requirementsDigest(backendDir) };
  }

  /** Reuse an existing runtime, adding missing dependencies with all installed versions
   * frozen. First installs, changed Python pins and explicit `tensorViz: Repair runtime`
   * build a replacement in staging and promote it only after verification. Dependency
   * conflicts fail without silently rebuilding or replacing Torch.
   *
   * Concurrency: one holder per machine (the lock directory — it arbitrates two windows AND
   * two invocations in the same window), and the loser of a race adopts the winner's runtime
   * instead of rebuilding it — unless this is a forced repair, where the user asked for a
   * rebuild and a healthy leftover is no reason to stop. */
  async provision(
    report: Report,
    isCancelled: () => boolean = () => false,
    options: ProvisionOptions = {},
  ): Promise<ProvisionResult> {
    const available = this.availability();
    if (!available.ok) {
      return {
        ok: false,
        message: 'tensorViz has no managed Python runtime to install here.',
        detail: available.why,
      };
    }
    const started = Date.now();
    const timingsMs: Record<string, number> = {};
    const measure: MeasureSetup = async (stage, operation) => {
      const begin = Date.now();
      try { return await operation(); }
      finally {
        timingsMs[stage] = (timingsMs[stage] ?? 0) + Date.now() - begin;
        report(`${stage}: ${(timingsMs[stage] / 1000).toFixed(1)}s`);
      }
    };
    const result = await this.provisionExclusive(report, isCancelled, options, available.source, measure);
    timingsMs.total = Date.now() - started;
    // Local evidence only. This contains no project paths, source or network URLs.
    fs.mkdirSync(this.root, { recursive: true });
    fs.writeFileSync(path.join(this.root, 'last-setup.json'), JSON.stringify({
      version: 1, recordedAt: new Date().toISOString(), platform: process.platform,
      arch: process.arch, source: available.source.kind,
      outcome: result.ok ? 'ready' : result.cancelled ? 'cancelled' : 'failed', timingsMs,
    }, null, 2));
    return { ...result, timingsMs };
  }

  /** The body of provision() once this window owns the only attempt for this root. */
  private async provisionExclusive(
    report: Report,
    isCancelled: () => boolean,
    options: ProvisionOptions,
    source: RuntimeSource,
    measure: MeasureSetup,
  ): Promise<ProvisionResult> {
    const started = Date.now();
    const lock = await measure('lock', () => this.acquireLock(isCancelled, options.lockWaitMs ?? LOCK_WAIT_DEFAULT_MS));
    if (!lock.ok) {
      if (lock.reason === 'cancelled') {
        return this.cancelledResult('before its turn at the setup lock');
      }
      return {
        ok: false,
        message: 'Another tensorViz runtime setup is already running in another window.',
        detail:
          'It did not finish within the wait budget. Close that window or wait for it, then ' +
          'retry; if that window crashed, the lock is released automatically on the next attempt.',
      };
    }
    try {
      // The window that went second arrives here after the first finished: adopt, don't redo.
      if (!options.force) {
        const status = await measure('reuse-check', () => this.check());
        if (status.state === 'ready') {
          return { ok: true, interpreter: status.interpreter, elapsedMs: Date.now() - started };
        }
      }
      if (!options.force && exists(this.interpreter)) {
        let provisioned: { runtime?: string } = {};
        try {
          provisioned = JSON.parse(this.provisionStampToken()) ?? {};
        } catch { /* Unknown provenance needs a rebuild. */ }
        const pinned = this.pinnedStamp();
        if (provisioned.runtime === pinned.runtime) {
          return await this.updateDependencies(source, report, isCancelled, started, pinned, measure);
        }
      }
      fs.mkdirSync(this.root, { recursive: true });
      // Leftovers of a host that died mid-build (process kill, power loss) are safe to remove:
      // staging-* never entered service and old-* was already replaced when its owner died.
      for (const entry of fs.readdirSync(this.root)) {
        if (/^(staging|old)-/.test(entry)) {
          confinedRm(this.root, path.join(this.root, entry), { recursive: true, force: true });
        }
      }

      report('Preparing the tensorViz runtime…');
      // A half-built staging copy must NOT read as previously consented on the next attempt,
      // and only the swap grants that consent — so the old stamp survives the whole build.
      const stagingBase = path.join(this.root, `staging-python-${process.pid}`);
      const stagingVenv = path.join(this.root, `staging-venv-${process.pid}`);
      try {
        if (source.kind === 'bundled') {
          // verbatimSymlinks is the whole point: python-build-standalone ships bin/python3 as
          // a relative symlink to python3.12, and resolving it (fs.cp's default) would write
          // an absolute path back into the app bundle — the relocation bug this copy avoids.
          fs.cpSync(this.bundled.runtime, stagingBase, {
            recursive: true,
            verbatimSymlinks: true,
          });
        } else {
          const acquired = await this.acquireDownload(source, report, isCancelled, stagingBase, measure);
          if (!acquired.ok) {
            return acquired;
          }
        }
        if (isCancelled()) {
          return this.cancelledResult('after the interpreter was staged');
        }

        const backendDir = this.backendDirFor(source);
        if (!backendDir) {
          return {
            ok: false,
            message: 'tensorViz could not resolve the backend directory to provision against.',
            detail:
              'The backend directory was there when availability was checked and is gone now. ' +
              'Re-run the command; if this persists, tensorViz needs a relaunch.',
          };
        }

        report('Creating the runtime environment…');
        const venv = await measure('environment', () => run(runtimeInterpreter(stagingBase), ['-m', 'venv', stagingVenv], {
          timeoutMs: VENV_TIMEOUT_MS,
          label: 'venv',
          report,
          isCancelled,
        }));
        if (venv.code !== 0) {
          return this.failure('create its runtime environment', venv);
        }

        const stagedInterpreter = venvInterpreter(stagingVenv);
        report('Installing the TensorViz service dependencies (model frameworks are separate)…');
        const install = await measure('dependencies', () => run(
          stagedInterpreter,
          ['-m', 'pip', 'install', '--no-input', '-r', path.join(backendDir, REQUIREMENTS)],
          { timeoutMs: PIP_TIMEOUT_MS, label: 'pip', report, isCancelled },
        ));
        if (install.code !== 0) {
          return this.failure('install the backend dependencies', install);
        }

        report('Verifying the runtime: importing the backend application…');
        const verify = await measure('backend-check', () => run(stagedInterpreter, ['-c', BACKEND_PROBE], {
          timeoutMs: BACKEND_PROBE_TIMEOUT_MS,
          label: 'verify',
          cwd: backendDir,
          report,
          isCancelled,
        }));
        if (verify.code !== 0) {
          return this.failure('import the backend application (import server_v2)', verify);
        }

        // The venv was built under staging names: pyvenv.cfg's home= and every bin/ script
        // quote those paths, and the renames below are about to invalidate them. Rewrite the
        // self-references BEFORE the swap so the promoted copy is internally consistent.
        this.rewriteVenvSelfPaths(stagingVenv, stagingBase);

        const swapped = this.swapIntoService(stagingBase, stagingVenv);
        if (!swapped.ok) {
          return swapped;
        }

        this.stampProvision(this.pinnedStamp());
        return { ok: true, interpreter: this.interpreter, elapsedMs: Date.now() - started };
      } finally {
        // Whatever the outcome above, staging belongs to no one once the attempt is over; on
        // success the renames already made these no-ops.
        confinedRm(this.root, stagingBase, { recursive: true, force: true });
        confinedRm(this.root, stagingVenv, { recursive: true, force: true });
      }
    } catch (err) {
      return {
        ok: false,
        message: 'tensorViz could not provision its Python runtime.',
        detail: String(err),
      };
    } finally {
      lock.release();
    }
  }

  /** Caller holds the same cross-window setup lock as a fresh provision. A failure leaves
   * the old stamp and environment in place; cancellation may leave some new packages. */
  private async updateDependencies(
    source: RuntimeSource, report: Report, isCancelled: () => boolean, started: number,
    pinned: { runtime: string; requirements: string },
    measure: MeasureSetup,
  ): Promise<ProvisionResult> {
    if (isCancelled()) return this.cancelledResult('before updating dependencies');
    const backendDir = this.backendDirFor(source);
    if (!backendDir) return { ok: false, message: 'tensorViz could not find its backend.', detail: 'Reopen the graph and retry.' };
    const directory = fs.mkdtempSync(path.join(this.root, 'dependency-update-'));
    try {
      report('Reusing the tensorViz environment; adding missing libraries without replacing installed versions…');
      const update = await measure('dependencies', () => run(this.interpreter, [
        path.join(backendDir, 'services', 'ide', 'environment_setup.py'), 'update-managed', directory,
      ], { timeoutMs: PIP_TIMEOUT_MS, label: 'dependency update', cwd: backendDir,
        report, isCancelled, killTree: true }));
      if (update.code !== 0) return this.failure('update its existing environment', update);
      if (isCancelled()) return this.cancelledResult('during the dependency update');
      report('Verifying the existing environment and WebSocket support…');
      const verify = await measure('backend-check', () => run(this.interpreter, ['-c', BACKEND_PROBE], {
        timeoutMs: BACKEND_PROBE_TIMEOUT_MS, label: 'verify', cwd: backendDir, report, isCancelled,
      }));
      if (verify.code !== 0) return this.failure('verify its existing environment', verify);
      if (isCancelled()) return this.cancelledResult('during verification');
      if (JSON.stringify(this.pinnedStamp()) !== JSON.stringify(pinned)) {
        return { ok: false, message: 'tensorViz dependencies changed during setup.', detail: 'Reopen the graph to check them again.' };
      }
      this.stampProvision(pinned);
      return { ok: true, interpreter: this.interpreter, elapsedMs: Date.now() - started };
    } finally {
      confinedRm(this.root, directory, { recursive: true, force: true });
    }
  }

  private stampProvision(pinned: { runtime: string; requirements: string }): void {
    fs.writeFileSync(this.stampPath, JSON.stringify({ ...pinned, provisionedAt: new Date().toISOString() }, null, 2));
  }

  /** Exclusive access across every extension host on this machine: mkdir's EEXIST is the
   *  arbitration. A holder that never releases (its process died) is condemned by the pid in
   *  its owner record; a corrupt or half-written record is condemned by age alone. Bounded
   *  wait — the caller's user is watching a progress notification, so "forever" is not a
   *  statement this function is allowed to make. */
  private async acquireLock(
    isCancelled: () => boolean,
    waitMs: number,
  ): Promise<{ ok: true; release: () => void } | { ok: false; reason: 'cancelled' | 'timeout' }> {
    const lockDir = path.join(this.root, LOCK_DIRNAME);
    const ownerPath = path.join(lockDir, LOCK_OWNER);
    const deadline = Date.now() + waitMs;
    fs.mkdirSync(this.root, { recursive: true });
    for (;;) {
      try {
        fs.mkdirSync(lockDir);
        fs.writeFileSync(
          ownerPath,
          JSON.stringify({ pid: process.pid, started: new Date().toISOString() }),
        );
        let released = false;
        return {
          ok: true,
          release: () => {
            if (released) {
              return;
            }
            released = true;
            confinedRm(this.root, lockDir, { recursive: true, force: true });
          },
        };
      } catch (err) {
        if ((err as NodeJS.ErrnoException).code !== 'EEXIST') {
          throw err;
        }
      }
      if (this.lockIsStale(lockDir, ownerPath)) {
        confinedRm(this.root, lockDir, { recursive: true, force: true });
        continue;
      }
      if (isCancelled()) {
        return { ok: false, reason: 'cancelled' };
      }
      if (Date.now() >= deadline) {
        return { ok: false, reason: 'timeout' };
      }
      await sleep(LOCK_POLL_MS);
    }
  }

  /** The holder is dead only when its recorded pid no longer exists on this machine — ESRCH.
   *  EPERM means alive-but-not-ours, which is still a live holder from our point of view. An
   *  owner record that cannot be read gets no such proof of life, so only unmistakable age
   *  condemns it: the lock directory predating LOCK_STALE_MS. */
  private lockIsStale(lockDir: string, ownerPath: string): boolean {
    try {
      const owner = JSON.parse(fs.readFileSync(ownerPath, 'utf8')) as { pid?: number };
      if (typeof owner.pid === 'number' && owner.pid > 0) {
        try {
          process.kill(owner.pid, 0);
          return false;
        } catch (err) {
          return (err as NodeJS.ErrnoException).code === 'ESRCH';
        }
      }
    } catch {
      // No readable owner: fall through to the age check.
    }
    try {
      return Date.now() - fs.statSync(lockDir).mtimeMs > LOCK_STALE_MS;
    } catch {
      // The holder released between EEXIST and stat; calling it stale lets the retry resolve
      // the race either way without another poll.
      return true;
    }
  }

  /** Promote the verified staging build over whatever is serving now. Each move is a
   *  same-directory rename — atomic on every filesystem we ship on — and a move that fails
   *  restores what it set aside, so a throw mid-swap with a previous runtime in service never
   *  strands the user without an interpreter. */
  private swapIntoService(
    stagingBase: string,
    stagingVenv: string,
  ): { ok: true } | { ok: false; message: string; detail: string } {
    const moves: Array<[string, string]> = [
      [stagingVenv, this.venv],
      [stagingBase, this.base],
    ];
    const retired: string[] = [];
    for (const [staging, target] of moves) {
      const aside = path.join(this.root, `old-${path.basename(target)}-${process.pid}`);
      let setAside = false;
      try {
        if (exists(target)) {
          fs.renameSync(target, aside);
          setAside = true;
        }
        fs.renameSync(staging, target);
        if (setAside) {
          retired.push(aside);
        }
      } catch (err) {
        if (setAside) {
          try {
            fs.renameSync(aside, target);
          } catch {
            // Both copies are on disk; the launch-time probe routes the user to repair and
            // the failure message below says what happened.
          }
        }
        return {
          ok: false,
          message: 'tensorViz could not put its verified runtime into service.',
          detail: `${staging}\n  ->  ${target}\n\n${String(err)}`,
        };
      }
    }
    for (const stale of retired) {
      confinedRm(this.root, stale, { recursive: true, force: true });
    }
    return { ok: true };
  }

  /** Point a staged venv at the paths it will live at: pyvenv.cfg's home= plus the bin/ (or
   *  Scripts/) entry points that quote their own location (activate, pip). Symlink CONTENT is
   *  never rewritten — relative shims (python -> python3) are location-independent — but an
   *  ABSOLUTE symlink target can quote the staging base (bin/python3 -> <staging-base>/bin/
   *  python3, exactly what `python -m venv` creates on POSIX); renaming the staging tree into
   *  service would leave that dangling, so those targets are re-pointed. Copied binaries are
   *  not text and are skipped. */
  private rewriteVenvSelfPaths(stagingVenv: string, stagingBase: string): void {
    const pairs: Array<[string, string]> = [
      [stagingVenv, this.venv],
      [stagingBase, this.base],
    ];
    const files = [path.join(stagingVenv, 'pyvenv.cfg')];
    const scripts = path.join(stagingVenv, process.platform === 'win32' ? 'Scripts' : 'bin');
    if (exists(scripts)) {
      for (const entry of fs.readdirSync(scripts)) {
        files.push(path.join(scripts, entry));
      }
    }
    for (const file of files) {
      try {
        if (fs.lstatSync(file).isSymbolicLink()) {
          let target: string;
          try {
            target = fs.readlinkSync(file);
          } catch {
            continue;
          }
          if (!path.isAbsolute(target)) {
            continue;
          }
          let relocated = target;
          for (const [from, to] of pairs) {
            relocated = relocated.split(from).join(to);
          }
          if (relocated !== target) {
            const replacement = `${file}.relocating-${process.pid}`;
            fs.symlinkSync(relocated, replacement);
            fs.renameSync(replacement, file);
          }
          continue;
        }
        const text = fs.readFileSync(file, 'utf8');
        if (text.includes('\0')) {
          continue;
        }
        let rewritten = text;
        for (const [from, to] of pairs) {
          rewritten = rewritten.split(from).join(to);
        }
        if (rewritten !== text) {
          fs.writeFileSync(file, rewritten);
        }
      } catch {
        // A file that resists reading as UTF-8 text is not a text self-reference; skip it.
      }
    }
  }

  private cancelledResult(when: string): ProvisionResult {
    return {
      ok: false,
      message: 'Setting up the tensorViz runtime was cancelled.',
      cancelled: true,
      detail:
        `It stopped ${when}. Any previously working runtime was left in service. ` +
        'Run "tensorViz: Repair runtime" to start over.',
    };
  }

  /** Download, verify and extract the interpreter INTO destDir, with the download cache
   *  keeping retries idempotent.
   *
   * Nothing is trusted because it is leniently named: the cached archive is re-hashed before
   * it is reused, destDir starts empty and failure removes it wholesale, and the extracted
   * interpreter answers --version with the manifest's python before this returns ok. The
   * runtime currently IN SERVICE is out of the blast radius the whole time: destDir is a
   * staging name that swapIntoService renames only after the whole provision verifies. */
  private async acquireDownload(
    source: DownloadRuntimeSource,
    report: Report,
    isCancelled: () => boolean,
    destDir: string,
    measure: MeasureSetup,
  ): Promise<{ ok: true } | { ok: false; message: string; detail: string }> {
    const archive = path.join(this.downloads, source.asset);
    const mb = Math.round(source.manifest.maxDownloadBytes / 1024 / 1024);
    if (fs.existsSync(archive) && (await sha256File(archive)) === source.sha256) {
      report('Using the previously downloaded runtime archive (checksum re-verified).');
    } else {
      report(`Downloading the tensorViz Python runtime (at most ~${mb} MB, checksum pinned)…`);
      const downloaded = await measure('download', () => downloadVerified({
        url: source.url,
        sha256: source.sha256,
        maxBytes: source.manifest.maxDownloadBytes,
        destPath: archive,
        transport: this.options.transport,
        report,
        isCancelled,
      }));
      if (!downloaded.ok) {
        return downloaded;
      }
    }

    confinedRm(this.root, destDir, { recursive: true, force: true });
    report('Unpacking the runtime (every archive member is inspected before it lands)…');
    const extracted = await measure('unpack', () => extractPythonRuntime({
      archive,
      destDir,
      // The contents land directly under the staging root — the extractor drops the asset's
      // single top-level `python/` directory on the way out (the exact shape
      // fetch-runtime.sh makes with tar's --strip-components=1).
      expectedInterpreter: source.interpreter,
    }));
    if (!extracted.ok) {
      confinedRm(this.root, destDir, { recursive: true, force: true });
      return extracted;
    }

    const stagedInterpreter = path.join(destDir, ...source.interpreter.split('/'));
    const verified = await measure('python-check', () => verifyRuntimeInterpreter(stagedInterpreter, source.python));
    if (!verified.ok) {
      confinedRm(this.root, destDir, { recursive: true, force: true });
      return {
        ok: false,
        message: 'The downloaded runtime did not verify after extraction.',
        detail: verified.why,
      };
    }

    report(`Runtime staged: CPython ${source.python} (checksum pinned and version-checked).`);
    return { ok: true };
  }

  private failure(what: string, r: RunResult): ProvisionResult {
    if (r.cancelled) {
      return {
        ok: false,
        message: 'Setting up the tensorViz runtime was cancelled.',
        cancelled: true,
        detail:
          `It stopped while trying to ${what}. ` +
          'Run "tensorViz: Repair runtime" to start over.',
      };
    }
    return {
      ok: false,
      message: `tensorViz could not ${what}.`,
      detail: `${why(r)}\n\n${r.tail}`,
    };
  }
}

/** The one place `vscode.env.remoteName` is read. Everything else takes it as a value, so the
 *  remote rule is a unit test rather than something only a remote session can prove.
 *
 *  `options.backendDir` is the answer resolveBackendDir currently gives, as a thunk: the
 *  packaged backend, the fork's staged backend, or a developer override. The extension wires
 *  it up because runtime.ts must not import preflight.ts (preflight already imports this). */
export function managedRuntimeFor(
  context: vscode.ExtensionContext,
  options?: {
    backendDir?: () => string | undefined;
    transport?: Transport;
    manifestPath?: string;
  },
): ManagedRuntime {
  return new ManagedRuntime({
    globalStorageDir: context.globalStorageUri.fsPath,
    extensionPath: context.extensionPath,
    remoteName: vscode.env.remoteName,
    backendDir: options?.backendDir,
    transport: options?.transport,
    manifestPath: options?.manifestPath,
  });
}

