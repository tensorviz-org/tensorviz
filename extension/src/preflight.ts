import { ChildProcess, spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as net from 'node:net';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { runtimeDiagnostic } from '@tensorviz/graph-host/runtimeAcquire';

import { HOST, MARKER_FILES } from './config';
import {
  BACKEND_PROBE,
  bundledLayout,
  DownloadRuntimeSource,
  ManagedRuntime,
  MIN_BACKEND_PYTHON,
  Report,
  RUNTIME_STAMP,
} from './runtime';

// The probe owns its home in runtime.ts: provision-time verification and launch-time
// resolution are one rule, so it is defined once. Re-exported because preflight's own
// tests pin the contract textually through this module.
export { BACKEND_PROBE, MIN_BACKEND_PYTHON };

export const REPO_PATH_MEMENTO = 'tensorviz.repoPath';

function isFile(p: string): boolean {
  try {
    return fs.statSync(p).isFile();
  } catch {
    return false;
  }
}

/** The first marker file that is missing under `dir`, or undefined if it validates. */
function missingMarker(dir: string): string | undefined {
  return MARKER_FILES.find((marker) => !isFile(path.join(dir, marker)));
}

function block(lines: string[]): string {
  return ['', ...lines.map((l) => (l ? `  ${l}` : '')), ''].join('\n');
}

// ---------------------------------------------------------------------------
// Repo discovery
// ---------------------------------------------------------------------------

export type BackendResolution =
  | { ok: true; backendDir: string; source: string }
  | { ok: false; message: string; detail: string; offerPicker: boolean };

/** Where a candidate backend directory can come from, in the order they are tried. */
interface BackendSource {
  /** Shown in the failure detail, one row per source, whether or not it was found. */
  readonly label: string;
  /** The directory to test, or undefined when this source does not apply here. */
  readonly dir: string | undefined;
  /** Human phrasing for the success log line. */
  readonly source: string;
  /** A source the user configured by hand must not fall through when it is wrong. */
  readonly hardFail?: boolean;
}

/** Resolve the directory holding `server_v2.py`, validating every candidate before it wins.
 *
 * Precedence, first hit wins (task.md P0, D21):
 *   1. the bundle — a tensorViz IDE build stages the backend beside the built-in extension
 *   2. the backend staged inside the vsix, for a stock VS Code / Cursor install
 *   3. the `tensorviz.repoPath` setting — a DEVELOPER OVERRIDE, hard fail if set and wrong
 *   4. an open workspace folder that happens to be a canvas checkout
 *
 * Rows 1 and 2 are what make "open any folder and it works" true: neither depends on the
 * user's workspace at all, so there is nothing to configure and no checkout to clone. Row 3
 * exists because the Electron suites and the owner run from a checkout; no user is told about
 * it, and its setting description says so. Row 4 keeps the plain "open the canvas repo in
 * VS Code" flow working without a setting.
 *
 * Row 3 must not fall through: if the user typed a path and it is wrong, starting a backend
 * from a *different* tree is the worst outcome available, because everything comes up fine and
 * is merely wrong.
 *
 * Deliberately NOT a candidate: path.dirname(context.extensionPath). It can only ever fire in
 * the F5 dev host (in a packaged .vsix that path is under ~/.vscode/extensions), which would
 * make the dev host a materially different program from the shipped one. */
export function resolveBackendDir(context: vscode.ExtensionContext): BackendResolution {
  const configured = vscode.workspace.getConfiguration('tensorviz').get<string>('repoPath', '').trim();
  const bundled = bundledLayout(context.extensionPath);
  const remembered = context.globalState.get<string>(REPO_PATH_MEMENTO);
  const sources: BackendSource[] = [
    {
      label: '1. the backend bundled with this app',
      // The stamp is what distinguishes a real tensorViz IDE bundle from two directories
      // that merely happen to sit at the right relative path.
      dir: isFile(path.join(bundled.runtime, RUNTIME_STAMP)) ? bundled.backend : undefined,
      source: 'the backend bundled with tensorViz IDE',
    },
    {
      label: '2. the backend inside this extension',
      dir: path.join(context.extensionPath, 'backend'),
      source: 'the backend packaged inside the extension',
    },
    {
      label: '3. the "tensorviz.repoPath" setting',
      dir: configured ? path.join(configured, 'backend') : undefined,
      source: 'the "tensorviz.repoPath" setting',
      hardFail: true,
    },
    ...(vscode.workspace.workspaceFolders ?? []).map((folder, index) => ({
      label: `${4 + index}. the open workspace folder`,
      dir: path.join(folder.uri.fsPath, 'backend'),
      source: 'the open workspace folder',
    })),
    {
      label: 'last. the folder you picked last time',
      dir: remembered ? path.join(remembered, 'backend') : undefined,
      source: 'the folder you picked last time',
    },
  ];

  for (const candidate of sources) {
    if (!candidate.dir) {
      continue;
    }
    const missing = missingMarker(candidate.dir);
    if (!missing) {
      return { ok: true, backendDir: candidate.dir, source: candidate.source };
    }
    if (candidate.hardFail) {
      return {
        ok: false,
        offerPicker: false,
        message:
          '"tensorviz.repoPath" does not point at a canvas checkout, so tensorViz will not guess a different one.',
        detail: block([
          `tensorviz.repoPath = ${configured}`,
          '',
          `missing: ${path.join('backend', missing)}`,
          '',
          'This setting is a developer override. Correct it, or clear it to let',
          'tensorViz use the backend it ships with.',
        ]),
      };
    }
  }

  if (remembered) {
    // Re-validated above and not accepted. globalState has no user-facing editor, so
    // without this a remembered-but-wrong path would be both wrong and unfixable.
    void context.globalState.update(REPO_PATH_MEMENTO, undefined);
  }

  return {
    ok: false,
    offerPicker: true,
    message: 'tensorViz could not find a backend to start.',
    detail: block([
      'It looked, in order, at:',
      '',
      ...sources.map((candidate) => `  ${candidate.label}\n${' '.repeat(6)}${candidate.dir ?? 'not applicable here'}`),
      '',
      'A usable backend directory is one that contains:',
      '',
      `    ${MARKER_FILES.join('\n    ')}`,
      '',
      'This build should ship one. If you are running from a checkout, set',
      '"tensorviz.repoPath" to its absolute path.',
    ]),
  };
}

/** Ask the user for the checkout and remember it. Only ever called from the command handler. */
export async function pickRepoPath(context: vscode.ExtensionContext): Promise<string | undefined> {
  const picked = await vscode.window.showOpenDialog({
    canSelectFiles: false,
    canSelectFolders: true,
    canSelectMany: false,
    openLabel: 'Use this checkout',
    title: 'Select your canvas checkout',
  });
  const dir = picked?.[0]?.fsPath;
  if (!dir) {
    return undefined;
  }
  const missing = missingMarker(path.join(dir, 'backend'));
  if (missing) {
    void vscode.window.showErrorMessage(
      `${dir} is not a canvas checkout (missing ${path.join('backend', missing)}).`,
    );
    return undefined;
  }
  await context.globalState.update(REPO_PATH_MEMENTO, dir);
  return dir;
}

// ---------------------------------------------------------------------------
// Ports
// ---------------------------------------------------------------------------

/** Ask the OS for a free loopback port and hand the number to uvicorn.
 *
 * Port 0 binds an ephemeral port; reading it back and closing the socket is the
 * standard way to allocate one for a child that cannot report the port it chose
 * (uvicorn prints it, but only after it is already listening, and the webview
 * needs the number to build its iframe URL and its CSP before that).
 *
 * There is a TOCTOU window between the close and uvicorn's own bind. It is the
 * same window start.sh:39-44 has with lsof, and the failure mode is loud: uvicorn
 * exits with EADDRINUSE and the readiness loop reports its output.
 */
export function reserveLoopbackPort(): Promise<number> {
  return new Promise<number>((resolve, reject) => {
    const server = net.createServer();
    server.once('error', reject);
    server.listen(0, HOST, () => {
      const address = server.address();
      if (typeof address === 'string' || address === null) {
        server.close(() => reject(new Error('Could not allocate a loopback port.')));
        return;
      }
      const { port } = address;
      server.close(() => resolve(port));
    });
  });
}

// ---------------------------------------------------------------------------
// Subprocess probes
// ---------------------------------------------------------------------------

/** What a probe produced. `timedOut` is separate from `error` because a probe killed at the
 *  budget and a probe that never started are different things to say to the user. */
export interface ProbeOutcome {
  code: number | null;
  stdout: string;
  stderr: string;
  error: NodeJS.ErrnoException | undefined;
  timedOut: boolean;
}

/** How a probe process is run. A seam for unit tests: resolution, precedence and error
 *  shaping can then be exercised with fabricated outcomes instead of real interpreters. */
export type ProbeRunner = (
  cmd: string,
  args: string[],
  opts: { cwd?: string; timeoutMs: number; env?: NodeJS.ProcessEnv; signal?: AbortSignal },
) => Promise<ProbeOutcome>;

/** Run a short-lived command without blocking the extension host.
 *
 * Async on purpose. The synchronous child_process API this replaced blocked the extension
 * host's event loop for the whole budget — up to 90 s of a frozen window per candidate while a
 * cold `import torch` ran. Everything else is kept: same combined stdout+stderr, same budget,
 * same windowsHide. `signal` lets a caller cancel probes it no longer needs; an aborted child
 * reports `error.code === 'ABORT_ERR'`. */
function defaultProbeRunner(
  cmd: string,
  args: string[],
  opts: { cwd?: string; timeoutMs: number; env?: NodeJS.ProcessEnv; signal?: AbortSignal },
): Promise<ProbeOutcome> {
  return new Promise<ProbeOutcome>((resolve) => {
    let child: ChildProcess;
    try {
      child = spawn(cmd, args, {
        cwd: opts.cwd,
        env: opts.env,
        signal: opts.signal,
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
      });
    } catch (err) {
      resolve({ code: null, stdout: '', stderr: '', error: err as NodeJS.ErrnoException, timedOut: false });
      return;
    }

    let stdout = '';
    let stderr = '';
    let timedOut = false;
    let settled = false;
    child.stdout?.on('data', (b: Buffer) => {
      stdout += b.toString();
    });
    child.stderr?.on('data', (b: Buffer) => {
      stderr += b.toString();
    });

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill('SIGKILL');
    }, opts.timeoutMs);
    // A probe must never hold the host process open past its own work.
    timer.unref?.();

    const settle = (code: number | null, error: NodeJS.ErrnoException | undefined): void => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      resolve({ code, stdout, stderr, error, timedOut });
    };
    child.on('error', (err) => settle(null, err as NodeJS.ErrnoException));
    child.on('close', (code) => settle(code, undefined));
  });
}

// ---------------------------------------------------------------------------
// Python interpreter
// ---------------------------------------------------------------------------

/** The service prefers its owned runtime. Development fallbacks never use the
 * project's capture setting or the Python extension's active environment. */
export async function interpreterCandidates(backendDir: string, managed?: string): Promise<string[]> {
  // The venv lives beside the code, and "beside" moved in P0: callers now pass a BACKEND
  // directory — `<repo>/backend` from a checkout, `<extension>/backend` from the .vsix —
  // where they used to pass the repository root. docs/local-setup.md tells people to build the
  // venv at the checkout root, so probe the owning checkout as well as the directory itself;
  // otherwise the one venv the documentation asks for is the one interpreter never tried.
  // Only a directory named `backend` has a checkout above it worth probing — that is what
  // repoRootFor decides — so a caller that passes a repository root, as every caller did
  // before P0, still gets exactly one venv candidate and the precedence tests below are
  // unchanged by this.
  const venvs = [backendDir, repoRootFor(backendDir)].map((root) =>
    process.platform === 'win32'
      ? path.join(root, 'venv', 'Scripts', 'python.exe')
      : path.join(root, 'venv', 'bin', 'python'),
  );
  const candidates = [
    managed,
    process.env.BACKEND_PYTHON,
    ...venvs,
    'python3',
    'python',
  ];
  const seen = new Set<string>();
  return candidates.filter((c): c is string => {
    if (typeof c !== 'string' || c.length === 0 || seen.has(c)) return false;
    seen.add(c);
    return true;
  });
}

/** The repository root that owns a backend directory, for advice a user can paste.
 *
 *  `<repo>/backend` -> `<repo>`. A path that is not that directory is its own root. */
export function repoRootFor(backendDir: string): string {
  return path.basename(backendDir) === 'backend' ? path.dirname(backendDir) : backendDir;
}

/** The module a probe failed on, so the message can name it.
 *
 *  The probe imports the backend application itself, so Python's own error already names the
 *  first module missing — catch it rather than sending people to install a package they
 *  already have (ide-plan §15 item 5's bug: saying "cannot import torch" whichever failed). */
export function missingModuleFrom(reason: string): string | undefined {
  return /No module named '([\w.]+)'/.exec(reason)?.[1]?.split('.')[0];
}

/** Strip a raw Python traceback from text headed for a notification.
 *
 *  Notifications are read by people, not debugged from: a traceback's frames belong in the
 *  output log (the full text is always written there by the caller), while the notification
 *  keeps whatever came before the traceback plus its final line — the actual error — and a
 *  pointer at the log. "Failed: ModuleNotFoundError: No module named 'dotenv'" is something a
 *  person can act on; twenty frames of site-packages paths are not. */
export function withoutTraceback(text: string): string {
  const marker = /^\s*Traceback \(most recent call last\):/m.exec(text);
  if (!marker) {
    return text;
  }
  const kept = text.slice(0, marker.index).replace(/\s+$/, '');
  const lastLine = text.replace(/\s+$/, '').split('\n').pop() ?? '';
  return `${kept ? `${kept}\n\n` : ''}${lastLine}\n\nThe full output is in the tensorViz log.`;
}

export type BackendInterpreterResult =
  | { ok: true; interpreter: string; banner: string }
  /** `repairRuntime` means a managed runtime exists here and failed, so the caller should
   *  offer `tensorviz.repairRuntime` rather than only the "install torch yourself" advice. */
  | { ok: false; message: string; detail: string; repairRuntime?: boolean; cancelled?: true };

/** 90s, not 10s: a cold `import torch` on a spinning disk genuinely takes tens of seconds. */
const INTERPRETER_PROBE_TIMEOUT_MS = 90_000;

/** Two at a time. Serial probing costs 90 s per dead candidate on the worst path (five
 *  candidates, none of them the right one); unbounded parallelism starts five simultaneous
 *  `import torch`, which on a laptop is enough memory pressure to make every one of them slower
 *  than it would have been alone. */
const INTERPRETER_PROBE_CONCURRENCY = 2;

function probeFailureReason(r: ProbeOutcome): string {
  if (r.timedOut) {
    return `no answer within ${INTERPRETER_PROBE_TIMEOUT_MS / 1000}s`;
  }
  if (r.error?.code === 'ENOENT') {
    return 'not on PATH';
  }
  if (r.error) {
    return r.error.message;
  }
  return r.stderr.trim().split('\n').pop() || `exit ${r.code}`;
}

export interface BackendInterpreterOptions {
  /** The managed backend runtime (task.md F3). Absent in stock VS Code and in every remote
   *  session, where the user's own environments are the fallback. */
  runtime?: ManagedRuntime;
  /** Progress for the provisioning step, which downloads PyTorch and takes minutes. */
  report?: Report;
  isCancelled?: () => boolean;
  /** Test seam: run probes without real interpreters. Production uses the default runner. */
  probeRunner?: ProbeRunner;
}

/** Native modal dismissal provides Cancel; only Install authorizes setup. */
async function firstInstallConsent(
  source: DownloadRuntimeSource,
): Promise<string | undefined> {
  const mb = Math.round(source.manifest.maxDownloadBytes / 1024 / 1024);
  return vscode.window.showInformationMessage(
    'Install tensorViz’s runtime to open graphs?',
    { modal: true, detail:
      'Setup may take a few minutes. Your project environment is never touched.\n\n' +
      `Python download: at most ~${mb} MB, plus dependency downloads. ` +
      'Estimated disk use: about 850 MB in VS Code’s extension storage. ' +
      `Python ${source.python} is pinned and checksum-verified.` },
    'Install',
  );
}

/** Bring the managed runtime to a usable state, or explain in one line why there is none.
 *
 * `unavailable` is the ordinary answer in stock VS Code and in every remote session, so it is
 * reported as a note and never as a failure. `absent` and `broken` both enter setup:
 * previously provisioned environments reuse their Python and add missing libraries without
 * replacing installed versions. A provisioning failure is also only a note: the user's own
 * environment is still a perfectly good way to run the backend, and refusing to look at it would turn a
 * recoverable first-run download problem into "tensorViz does not start". */
export async function managedInterpreter(
  options: BackendInterpreterOptions,
): Promise<{ interpreter?: string; note: string; repairable: boolean; cancelled?: boolean; failure?: string }> {
  const runtime = options.runtime;
  if (!runtime) {
    return { note: 'managed runtime: not offered by this build', repairable: false };
  }
  const status = await runtime.check();
  if (status.state === 'ready') {
    return { interpreter: status.interpreter, note: 'managed runtime: ready', repairable: true };
  }
  if (status.state === 'unavailable') {
    return { note: `managed runtime: none here (${status.why})`, repairable: false };
  }

  // A first install via download is a NETWORK action the user must take with the size
  // disclosed; merely opening VS Code may not spend a gigabyte. A previously provisioned
  // runtime proves consent was given when it was installed, so managed dependency maintenance
  // never re-asks. The bundled source never asks: its interpreter is already on disk and the
  // pip step is the pre-existing fork flow.
  if (status.state === 'absent' && !runtime.previouslyProvisioned) {
    const available = runtime.availability();
    if (available.ok && available.source.kind === 'download') {
      const picked = await firstInstallConsent(available.source);
      if (picked !== 'Install') {
        return {
          note: 'managed runtime: first-run install declined (the answer was not Install)',
          cancelled: true,
          repairable: true,
        };
      }
    }
  }

  const report = options.report ?? ((): void => {});
  report('Setting up the tensorViz runtime…');
  const provisioned = await runtime.provision(report, options.isCancelled);
  if (provisioned.ok) {
    return {
      interpreter: provisioned.interpreter,
      note: `managed runtime: provisioned in ${Math.round(provisioned.elapsedMs / 1000)}s (${status.why})`,
      repairable: true,
    };
  }
  return {
    note: runtimeDiagnostic(`managed runtime: ${provisioned.message}\n${provisioned.detail}`),
    failure: provisioned.message,
    cancelled: provisioned.cancelled,
    repairable: true,
  };
}

/** Resolve the BACKEND interpreter and validate it in one pass. Validation is not optional
 *  for any source — the whole reason Makefile:29-40 exists is that plain `python` usually is
 *  not the env with torch.
 *
 *  This is the backend role only. The trace interpreter — which Python runs the user's model
 *  — is resolved separately (worker.ts), from different sources, for a different purpose, and
 *  a failure there must never reach this function.
 *
 *  Probes run concurrently but the RESULT is still strictly by precedence: a success at index i
 *  cancels only the probes after i, and any probe already in flight before i still gets to
 *  win. Otherwise `python3` finishing first would quietly beat the managed runtime, which is
 *  the one thing interpreterCandidates' order exists to prevent. */
export async function resolveBackendInterpreter(
  backendDir: string,
  options: BackendInterpreterOptions = {},
): Promise<BackendInterpreterResult> {
  const managed = await managedInterpreter(options);
  if (managed.cancelled || options.isCancelled?.()) return {
    ok: false, cancelled: true, message: 'Service setup cancelled.', detail: managed.note,
  };
  const candidates = await interpreterCandidates(backendDir, managed.interpreter);
  const runProbe = options.probeRunner ?? defaultProbeRunner;
  const outcomes = new Map<number, ProbeOutcome>();
  const inFlight = new Map<number, AbortController>();
  let best = Number.POSITIVE_INFINITY;
  let next = 0;

  const worker = async (): Promise<void> => {
    for (;;) {
      const index = next++;
      // Nothing left, or a higher-precedence candidate already won.
      if (index >= candidates.length || index > best) {
        return;
      }
      const controller = new AbortController();
      inFlight.set(index, controller);
      const r = await runProbe(candidates[index], ['-c', BACKEND_PROBE], {
        cwd: backendDir,
        timeoutMs: INTERPRETER_PROBE_TIMEOUT_MS,
        env: { ...process.env, KMP_DUPLICATE_LIB_OK: 'TRUE' },
        signal: controller.signal,
      });
      inFlight.delete(index);
      outcomes.set(index, r);
      if (r.code === 0 && index < best) {
        best = index;
        for (const [i, c] of inFlight) {
          if (i > index) {
            c.abort();
          }
        }
      }
    }
  };

  await Promise.all(
    Array.from({ length: Math.min(INTERPRETER_PROBE_CONCURRENCY, candidates.length) }, worker),
  );

  if (best < candidates.length) {
    const winner = candidates[best];
    return {
      ok: true,
      interpreter: winner,
      // Adapted from Makefile:41's success line. Which runtime answered is the claim the
      // status bar and every bug report depend on, so the managed one says so out loud.
      banner:
        `python OK: ${winner} (${outcomes.get(best)?.stdout.trim() ?? ''})` +
        (winner === managed.interpreter ? ' — tensorViz managed runtime' : ''),
    };
  }

  // No success means nothing was cancelled, so every candidate has a real outcome to report —
  // all of them at once, in precedence order, as the serial version did.
  const tried = candidates.map(
    (candidate, i) =>
      `${candidate}${candidate === managed.interpreter ? '  (tensorViz managed backend runtime)' : ''}` +
      `  ->  ${probeFailureReason(outcomes.get(i)!)}`,
  );
  const lastError = candidates.length
    ? probeFailureReason(outcomes.get(candidates.length - 1)!)
    : 'no interpreter candidates';

  const missing = missingModuleFrom(lastError);

  return {
    ok: false,
    message: managed.failure ?? (missing
      ? `No Python interpreter here can run the tensorViz backend runtime: ${missing} does not import.`
      : 'The tensorViz service could not start.'),
    repairRuntime: managed.repairable,
    detail: runtimeDiagnostic(block([
      managed.note,
      '',
      ...(managed.repairable ? [
        'Choose Repair runtime to retry tensorViz setup, or Show Log for diagnostics.',
        'If downloading failed, check access to GitHub and its release assets, then retry.',
        'Your project Python is separate and has not been changed by this setup.',
        '',
      ] : [
        'This host has no supported managed runtime. Use a supported host or a compatible service interpreter.',
        'The service needs Python 3.11+ and the packaged backend dependencies; project Python is selected separately.',
        '',
      ]),
      'Interpreters tried, in order:',
      '',
      ...tried.map((t) => `    ${t}`),
      '',
      `last error: ${lastError}`,
    ])),
  };
}
