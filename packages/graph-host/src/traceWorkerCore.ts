import { ChildProcess, spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import * as path from 'node:path';
import { backendPythonArgs, nativeBackend } from './backendPython';
import { CAPTURE_ENVELOPE_MAX_BYTES } from './captureBudgets';
import { tensorInputsSchema, type TensorInputs } from './tensorInputs';
import { executionSpecSchema, executionJobSchema, type ExecutionJob, type ExecutionSpec } from './executionSpec';
import { resolveExecutionPlan, resolveCapturePreparation, resolveRetainedPlan, executionPlanFile,
  candidateRequestFile, EXECUTION_PLAN_ENV, CANDIDATE_REQUEST_ENV,
  type ExecutionSnapshot, type PreparationSnapshot, type CandidateRequest, type ResolvedExecutionPlan } from './executionPlanCore';
import { candidateRequest, candidateEvidence, type CandidateEvidence, type CandidateInput, type CandidateRunResult } from './candidateExecution';
import { sameExecutionGuards } from './executionGuard';
import { ExecutionResultStore, dropEnvironmentFiles, emptyStats, outcomeUnstorable, requestIneligible, servableEntry, shadowDifference, storedRunMeta,
  resultKey, resultKeyParts, DEFAULT_MAX_SIZE_MB, type ExecutionCacheMode, type ExecutionCacheSettings, type ExecutionCacheStats,
  type ReusedCapture } from './executionCache';
import type { RunSpec } from '@tensorviz/contracts/protocol';
import type { components as SemanticComponents } from '@tensorviz/contracts/semanticContracts.generated';

/**
 * The trace worker: the user's model, executed in the user's interpreter.
 *
 * This is the only place tensorViz runs a line of the workspace's code, and the
 * shape of the module follows from that (task.md §8.5, D3, D11):
 *
 *  * it is a **subprocess**, never an import — the backend must not execute a file
 *    that is half-written by definition, and a segfault in somebody's custom CUDA
 *    kernel must not take the editor's backend with it;
 *  * it runs in the **user's** interpreter (ms-python's active environment), not
 *    the backend's, because a graph traced against the wrong torch is fiction;
 *  * it runs only after the user has said yes **once per workspace**, and every run
 *    is cancellable and bounded by a timeout;
 *  * it writes nothing: the extension persists the worker's stdout, which is what
 *    makes `.tensorviz/cache` the extension's file and not the worker's.
 *
 * The stdout contract is task.md §6.3 and a version mismatch is refused loudly:
 * an envelope from another build is not something to interpret optimistically, it
 * is a mismatched install to tell the user about.
 */

/** Envelope version this extension understands (task.md §6.3). Kept in lockstep
 *  with `services/ingest/worker.py::PROTOCOL_VERSION` and
 *  `services/ide/projection.py::WORKER_PROTOCOL_VERSION`. */
export const WORKER_PROTOCOL_VERSION = 1;

/** Exit codes the worker promises. */
export const EXIT = { ok: 0, usage: 2, construct: 3, trace: 4 } as const;

/** A cold `import torch` plus a forward. Generous, because the alternative to
 *  waiting is telling the user their model failed when it did not. */
export const DEFAULT_TIMEOUT_MS = 180_000;

/** Where the worker's `--roots` and PYTHONPATH come from. */
/** The worker's exit code for "the class could not be constructed".
 *
 * Named because it is the one exit the host reacts to differently: it is not a bug in the
 * user's forward, it is a constructor that needs arguments, and it has an answer
 * (`# tensorviz: init=`) the user can be handed instead of a log to read. */
export const WORKER_EXIT_CONSTRUCT_FAILED = 3;

export interface TraceRequest {
  /** Private constructor/input preparation; never relayed as captured graph evidence. */
  prepareInputs?: boolean;
  prepareConfigDefaults?: boolean;
  /** Absolute path of the .py file defining `cls`. */
  file: string;
  /** Required for `--kind model` (the default) and `--kind train_step`; a data
   *  pipeline has no class. */
  cls?: string;
  /** `--input`: shapes ("2,8", "2,8;2") or a Python expression. Optional.
   *
   *  It OUTRANKS the traced class's `# tensorviz: input=` inside
   *  `services/ingest/service.py::ingest_source`. No host sets it today: the Canvas
   *  passes `undefined`, and the remembered `input` in `inputMemory.ts` that once
   *  filled it is no longer read. */
  input?: string;
  /** Bounded positional tensor descriptions, used by MCP instead of expressions. */
  tensorInputs?: TensorInputs;
  /** Explicit reproducible inputs, model setup and device; workspace is host-owned. */
  execution?: ExecutionSpec;
  job?: ExecutionJob;
  workspace?: string;
  tier?: 'real' | 'fake';
  depth?: number;
  /** Which kind of program to trace (task.md M9). Omitted means `model`, so every
   *  call written before M9 asks for exactly what it used to. */
  kind?: 'model' | 'data' | 'train_step';
  /** `--callable`: `name` for a function in `file`, or `other.py:name` for one
   *  beside it. Required by `--kind data` and `--kind train_step`. */
  callable?: string;
  /** `--overrides`: JSON of constructor keyword arguments, from a
   *  `# tensorviz: init=` declaration (task.md P0 / §6.1). The flag already existed;
   *  nothing in the worker changes. */
  overrides?: string;
  /**
   * `--steps N`: run the traced `train_step` N times and keep the series (P4 Watch).
   *
   * Only `--kind train_step` accepts it and the worker refuses N > 200 as a usage
   * error before importing anything of the user's. Omitted means one step, which is
   * exactly what every call written before Watch asked for.
   */
  steps?: number;
  /**
   * How long THIS request may run, overriding the worker's own budget.
   *
   * Watch is the reason it exists: one timeout for every recipe is a number that
   * either kills a 200-step watch or lets a hung forward sit for twelve minutes.
   * The caller that composes the approval computes it, so the seconds the modal
   * promises are the seconds enforced here — the alternative is a consent dialog
   * quoting a number nothing uses.
   */
  timeoutMs?: number;
  /** Host-owned saved plan. Rechecked after probing, immediately before spawn. */
  expectedInterpreter?: string;
  beforeLaunch?: () => Promise<boolean>;
  profileSpec?: RunSpec;
  /** Shared host attempt identity for the plan and its canonical captures. */
  attemptId?: string;
  /** Host lifecycle observes the immutable plan even if execution later times out. */
  onResolvedPlan?: (snapshot: Pick<ExecutionSnapshot, 'plan' | 'applicabilityDigest'>) => void;
  onCapturePrepared?: (snapshot: Pick<PreparationSnapshot, 'preparation' | 'applicabilityDigest'>) => void;
  /** Opt in to the execution cache (`executionCache.ts`). Only interactive Canvas runs set
   *  it; MCP and candidate runs never do. In shadow mode the worker
   *  still runs; the cache is only compared with the fresh result, never served. In `on`
   *  mode `allow` may return a stored identical result without launching the worker;
   *  `bypass` (Run fresh) always runs and replaces the stored entry. */
  reuse?: 'allow' | 'bypass';
}

export interface WorkerFrame {
  file: string;
  line: number;
  name: string;
}

export interface WorkerError {
  type: string;
  message: string;
  frames: WorkerFrame[];
  node_id?: string | null;
  cause_types?: string[];
  missing_module?: string;
  diagnostic?: SemanticComponents['schemas']['Diagnostic'];
  /** Fake-tier automatic captures only: the run stopped because the model needs tensor
   *  values (`services/ingest/trace.py::fake_tensor_limitation`). */
  fakeTensor?: FakeTensorLimitation;
}

/** Why a fake-tensor (shape-only) run could not run the model. */
export interface FakeTensorLimitation {
  reason: 'data_dependent' | 'dynamic_output_shape' | 'unsupported_operator' | 'unsupported_fake_tensor' | 'tensor_subclass';
  errorType: string;
  operation: string;
  phase: 'construction' | 'forward';
}

/** The §6.3 envelope. `result` and `spans` are opaque here: the extension relays
 *  them to `/api/ide/project` and never models the graph's schema. */
export interface WorkerEnvelope {
  protocol_version: number;
  result: unknown | null;
  spans: Record<string, unknown>;
  error: WorkerError | null;
  worker: { python?: string; torch?: string; elapsed_ms?: number; interpreter?: string; platform?: string;
    planId?: string; planDigest?: string; candidateExecution?: unknown; candidateBaseCurrent?: boolean;
    candidatePlanCurrent?: boolean; candidatePlanDigest?: string; candidatePostconditions?: unknown;
    checkpointCompatibility?: unknown };
}

export type TraceOutcome = (
  /** `reused`: no worker ran; `envelope` is an earlier attempt's identical result
   *  (execution cache, `on` mode). Its capture keeps that attempt's ids. */
  | { ok: true; envelope: WorkerEnvelope; exitCode: number; executionCurrent?: boolean; reused?: ReusedCapture }
  /** `envelope` is present whenever the worker managed to print one, even on a
   *  failure — a partial graph after a forward error is the useful half. */
  | { ok: false; reason: string; detail: string; envelope?: WorkerEnvelope; exitCode?: number; timedOut?: boolean; cancelled?: boolean; executionCurrent?: boolean }
) & { /** Host-only saved-setup recheck after asynchronous graph projection. */
  recheckExecution?: (signal: AbortSignal) => Promise<boolean>;
  /** Set when a fake-tier automatic capture stopped on a fake-tensor limitation and this
   *  outcome is the single real-tensor re-run that replaced it. */
  fakeTensorFallback?: FakeTensorLimitation };

/** Parse and validate the worker's stdout.
 *
 * Pure, and separately tested, because this is where a mismatched install has to
 * be caught: an envelope from a different protocol version may be *shaped* right
 * and mean something else, and merging it would put invented shapes on the canvas.
 */
export function parseEnvelope(stdout: string): { ok: true; envelope: WorkerEnvelope } | { ok: false; reason: string; detail: string } {
  const text = stdout.trim();
  if (!text) {
    return { ok: false, reason: 'The trace worker produced no output.', detail: 'stdout was empty.' };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    return {
      ok: false,
      reason: 'The trace worker produced output this extension could not read.',
      detail: `${String(error)}\n\nFirst 500 characters:\n${text.slice(0, 500)}`,
    };
  }
  const envelope = parsed as WorkerEnvelope;
  if (!envelope || typeof envelope !== 'object' || typeof envelope.protocol_version !== 'number') {
    return {
      ok: false,
      reason: 'The trace worker produced output this extension could not read.',
      detail: 'the JSON carries no protocol_version.',
    };
  }
  if (envelope.protocol_version !== WORKER_PROTOCOL_VERSION) {
    return {
      ok: false,
      reason: `tensorViz cannot read this trace worker (protocol ${envelope.protocol_version}).`,
      detail:
        `This extension speaks protocol ${WORKER_PROTOCOL_VERSION}. The extension and the ` +
        'tensorViz checkout it runs against are from different builds; update whichever is older.',
    };
  }
  return { ok: true, envelope };
}

/** The one-line remedy for a user environment without the worker's two imports. */
export const INSTALL_HINT = 'pip install torch "pydantic>=2"';

const PROBE = 'import torch, pydantic; print(torch.__version__, pydantic.VERSION)';

interface RunResult {
  code: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
  outputLimit?: 'stdout' | 'stderr';
  spawnError?: NodeJS.ErrnoException;
}

/** Spawn options shared by ordinary and pre-started workers, so both get the same
 *  process group and therefore the same `killTree` reach. */
function spawnWorker(command: string, args: string[], cwd: string, env: NodeJS.ProcessEnv): ChildProcess {
  return spawn(command, args, {
    cwd,
    env,
    // Own process group, so cancelling kills anything the user's code spawned
    // (a dataloader's workers are the usual case) and not just python itself.
    detached: process.platform !== 'win32',
    windowsHide: true,
  });
}

/** A standby (`services/ingest/standby.py`) handed to `runProcess` instead of a spawn:
 *  `request` is the single stdin line it is waiting for, and `stderr` is what it
 *  printed while idle (importing torch), attributed to the run as a fresh spawn's
 *  would be. */
export interface PrestartedWorker {
  child: ChildProcess;
  request: string;
  stderr: Buffer[];
}

export function runProcess(
  command: string,
  args: string[],
  options: { cwd: string; env: NodeJS.ProcessEnv; timeoutMs: number; stdoutLimit?: number; stderrLimit?: number;
    prestarted?: PrestartedWorker },
  onChild: (child: ChildProcess) => void,
): Promise<RunResult> {
  return new Promise((resolve) => {
    let child: ChildProcess;
    if (options.prestarted) {
      child = options.prestarted.child;
    } else {
      try {
        child = spawnWorker(command, args, options.cwd, options.env);
      } catch (error) {
        resolve({ code: null, stdout: '', stderr: '', timedOut: false, spawnError: error as NodeJS.ErrnoException });
        return;
      }
    }
    onChild(child);

    const buffers: Record<'stdout' | 'stderr', Buffer[]> = { stdout: [], stderr: [] };
    const sizes = { stdout: 0, stderr: 0 };
    if (options.prestarted) {
      for (const chunk of options.prestarted.stderr) { buffers.stderr.push(chunk); sizes.stderr += chunk.length; }
    }
    const limits = { stdout: options.stdoutLimit ?? CAPTURE_ENVELOPE_MAX_BYTES, stderr: options.stderrLimit ?? 1024 * 1024 };
    let outputLimit: 'stdout' | 'stderr' | undefined;
    let timedOut = false;
    let settled = false;
    const timer = setTimeout(() => {
      timedOut = true;
      killTree(child);
    }, options.timeoutMs);

    const settle = (result: RunResult): void => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      resolve(result);
    };

    const collect = (pipe: 'stdout' | 'stderr', chunk: Buffer) => {
      if (outputLimit) return;
      sizes[pipe] += chunk.length;
      if (sizes[pipe] > limits[pipe]) {
        outputLimit = pipe;
        buffers.stdout.length = buffers.stderr.length = 0;
        killTree(child);
      } else buffers[pipe].push(chunk);
    };
    const result = (code: number | null, spawnError?: NodeJS.ErrnoException): RunResult => ({
      code, stdout: Buffer.concat(buffers.stdout).toString('utf8'), stderr: Buffer.concat(buffers.stderr).toString('utf8'),
      timedOut, outputLimit, spawnError,
    });
    child.stdout?.on('data', (chunk: Buffer) => collect('stdout', chunk));
    child.stderr?.on('data', (chunk: Buffer) => collect('stderr', chunk));
    child.on('error', error => settle(result(null, error as NodeJS.ErrnoException)));
    child.on('close', code => settle(result(code)));
    // Last, once every listener above exists. stdin is deliberately not ended: it
    // is the parent-death pipe, and EOF would make the worker kill its own group.
    if (options.prestarted) child.stdin?.write(options.prestarted.request);
  });
}

/** SIGTERM the whole group, then SIGKILL what is left. Same shape as stack.ts's
 *  backend kill, and for the same reason: a trace can have children. */
const terminations = new WeakMap<ChildProcess, Promise<boolean>>();
class CandidateCleanupError extends Error {}
function killTree(child: ChildProcess): Promise<boolean> {
  // Release the parent-death pipe even when platform termination fails.
  child.stdin?.end();
  const previous = terminations.get(child);
  if (previous) return previous;
  const pid = child.pid;
  if (!pid) {
    return Promise.resolve(true);
  }
  let finish!: (stopped: boolean) => void;
  const pending = new Promise<boolean>(resolve => { finish = resolve; });
  terminations.set(child, pending);
  try {
    if (process.platform === 'win32') {
      const command = spawn('taskkill', ['/pid', String(pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore', timeout: 5000 });
      command.once('close', code => finish(code === 0));
      command.once('error', () => { try { child.kill(); } catch { /* already gone */ } finish(false); });
      return pending;
    } else {
      process.kill(-pid, 'SIGTERM');
    }
  } catch {
    /* already gone */
  }
  setTimeout(() => {
    let stopped = process.platform !== 'win32';
    try {
      if (process.platform !== 'win32') {
        process.kill(-pid, 'SIGKILL');
      }
    } catch (error) {
      stopped = (error as NodeJS.ErrnoException).code === 'ESRCH';
    }
    finish(stopped);
  }, 2000).unref?.();
  return pending;
}

/** Preserve the user's import paths. Private graph bootstraps load backend modules
 * by file and must not expose backend packages that shadow workspace namespaces.
 * Legacy module entry points still need the backend on PYTHONPATH.
 *
 * KMP_DUPLICATE_LIB_OK mirrors Makefile:8 — without it torch plus MKL aborts on
 * some machines, and an abort here reads as "your model crashed".
 */
export function workerEnv(parent: NodeJS.ProcessEnv, backendDir: string, privateBootstrap = false): NodeJS.ProcessEnv {
  const existing = parent.PYTHONPATH;
  return {
    ...parent,
    PYTHONPATH: privateBootstrap ? existing : existing ? `${backendDir}${path.delimiter}${existing}` : backendDir,
    KMP_DUPLICATE_LIB_OK: 'TRUE',
    PYTHONUNBUFFERED: '1',
  };
}

/** An idle standby holds ~195 MB (torch and the capture internals imported). Long enough to cover an editing
 *  session's pauses; after it, the next run simply starts the ordinary way. */
export const STANDBY_IDLE_MS = 5 * 60_000;
/** Importing torch prints little. A standby that prints more than this while idle
 *  is not in the state its key describes, so it is retired rather than used. */
const STANDBY_IDLE_STDERR_LIMIT = 64 * 1024;
/** Per-run values a standby receives with its request instead of at spawn. */
const PER_RUN_ENV = [EXECUTION_PLAN_ENV, CANDIDATE_REQUEST_ENV];

interface Standby {
  child: ChildProcess;
  key: string;
  stderr: Buffer[];
  stderrBytes: number;
  usable: boolean;
  timer: ReturnType<typeof setTimeout> | undefined;
}

/** What a standby was started with. A run may use it only if this is identical:
 *  torch reads thread, allocator and library settings from the environment when it
 *  is imported, which for a standby happened before the run existed. Per-run plan
 *  paths are excluded because they are delivered with the request instead. */
export function standbyKey(interpreter: string, cwd: string, env: NodeJS.ProcessEnv): string {
  const entries = Object.entries(env)
    .filter((entry): entry is [string, string] => typeof entry[1] === 'string' && !PER_RUN_ENV.includes(entry[0]))
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return JSON.stringify([interpreter, cwd, entries]);
}

type Referable = { ref?(): unknown; unref?(): unknown };
function setHandlesRef(child: ChildProcess, referenced: boolean): void {
  for (const handle of [child, child.stdin, child.stdout, child.stderr] as (Referable | null)[]) {
    if (referenced) handle?.ref?.(); else handle?.unref?.();
  }
}

/** The environment a standby is spawned with: the run's, minus per-run paths. */
export function standbyEnvironment(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const result: NodeJS.ProcessEnv = { ...env };
  for (const name of PER_RUN_ENV) delete result[name];
  return result;
}

/** The fake-tensor limitation a finished fake-tier run stopped on, or undefined. A
 *  cancelled or timed-out run, and any other failure, is not one. */
export function fakeTensorLimitation(outcome: TraceOutcome): FakeTensorLimitation | undefined {
  if (outcome.ok || outcome.cancelled || outcome.timedOut) return undefined;
  const limitation = outcome.envelope?.error?.fakeTensor;
  const reasons = ['data_dependent', 'dynamic_output_shape', 'unsupported_operator', 'unsupported_fake_tensor', 'tensor_subclass'];
  return limitation && typeof limitation === 'object' && reasons.includes(limitation.reason)
    && (limitation.phase === 'construction' || limitation.phase === 'forward') ? limitation : undefined;
}

/** Command-line arguments for one request (task.md §6.3). */
export function workerArgs(request: TraceRequest, transport: 'inline' | 'plan' | 'preparation' | 'candidate' = 'inline'): string[] {
  if (request.prepareConfigDefaults && !request.prepareInputs) throw new Error('Config defaults require private input preparation.');
  if (transport === 'candidate') {
    if (request.prepareInputs || !request.workspace || !request.cls || request.execution || request.job || request.input !== undefined
      || request.tensorInputs !== undefined || request.overrides !== undefined || request.callable || request.steps !== undefined
      || request.kind && request.kind !== 'model' || request.tier === 'fake') throw new Error('Invalid candidate request.');
    return ['-m', 'services.ingest.worker', '--candidate-execution', '--workspace', request.workspace,
      '--file', request.file, '--cls', request.cls, '--depth', String(request.depth ?? 2), '--json'];
  }
  const args = ['-m', 'services.ingest.worker', '--file', request.file];
  if (request.prepareInputs) {
    if (transport !== 'plan' || !request.execution || request.job || request.profileSpec || request.overrides) {
      throw new Error('Input preparation requires a private construction plan.');
    }
    args.push('--prepare-inputs');
    if (request.prepareConfigDefaults) args.push('--prepare-config-defaults');
  }
  if (request.cls) {
    args.push('--cls', request.cls);
  }
  // Omitted rather than sent as `--kind model`: the default is the worker's, and a
  // flag this extension always passes is a flag the worker can never change.
  if (request.kind && request.kind !== 'model') {
    args.push('--kind', request.kind);
  }
  if (request.callable) {
    args.push('--callable', request.callable);
  }
  // Only ever sent for the kind that accepts it: the worker exits 2 on `--steps`
  // with any other `--kind`, and a flag this side adds by accident would turn a
  // perfectly good model trace into a usage error.
  if (request.steps !== undefined && request.kind === 'train_step') {
    args.push('--steps', String(request.steps));
  }
  if (request.job && !request.execution) throw new Error('Diagnostic jobs require an explicit execution setup.');
  if (transport === 'preparation') {
    // Both tiers: a fake-tier automatic capture is the same canonical capture, recorded
    // on fake tensors (no values). Explicit execution and candidates stay real-only.
    if (request.execution || !request.workspace || request.kind && request.kind !== 'model') {
      throw new Error('Automatic preparation requires a model and its workspace.');
    }
    args.push('--capture-preparation', '--workspace', request.workspace);
  }
  if (request.execution !== undefined) {
    if (!request.workspace || request.input || request.tensorInputs || request.kind && request.kind !== 'model' || request.tier === 'fake') {
      throw new Error('Explicit execution requires a workspace, real model tier and no legacy input.');
    }
    const execution = JSON.stringify(executionSpecSchema.parse(request.execution));
    if (execution.length > 32_768) throw new Error('The execution request exceeds 32 KiB.');
    args.push(...(transport === 'plan' ? ['--resolved-execution'] : ['--execution', execution]), '--workspace', request.workspace);
    if (request.job) {
      const job = JSON.stringify(executionJobSchema.parse(request.job));
      if (transport !== 'plan') args.push('--diagnostic-job', job);
    }
  }
  if (request.tensorInputs !== undefined) {
    if (request.input || (request.kind && request.kind !== 'model')) throw new Error('Tensor inputs require a model request without --input.');
    const inputs = JSON.stringify(tensorInputsSchema.parse(request.tensorInputs));
    if (transport !== 'preparation') args.push('--tensor-inputs', inputs);
  }
  if (request.input && transport !== 'preparation') {
    args.push('--input', request.input);
  }
  if (request.overrides && transport !== 'preparation') {
    args.push('--overrides', request.overrides);
  }
  args.push('--tier', request.tier ?? 'real');
  args.push('--depth', String(request.depth ?? 2));
  args.push('--json');
  return args;
}

export interface TraceWorkerHost {
  timeoutMs?: number;
  /** Standalone hosts own stdin so launcher death also ends the worker process group. */
  parentPipe?: boolean;
  interpreter(file: string): Promise<string | undefined>;
  trusted(): boolean;
  documents(root: string): import('./executionPlanCore').DocumentStates;
  /** Current execution-cache setting for the run's file (`resource`), or for the window when
   *  `resource` is undefined, read on every run so a change, or a folder-level value, applies
   *  to the next run without a reload. Absent (the standalone plugin host) means off. */
  executionCache?(resource?: string): ExecutionCacheSettings | undefined;
}

/** A shadow lookup prepared before launch: the key and whatever the store held for it. */
interface ShadowLookup { key: string; snapshot: ExecutionSnapshot; stored: string | undefined; meta: string | undefined; args: string[] }

/** Asks the worker for `diagnostics.reuse`; sent only for runs the cache looks up, so a run
 *  with the cache off is launched exactly as before. Part of the key (it is in argv). */
export const REUSE_AUDIT_FLAG = '--reuse-audit';

export class TraceWorkerCore {
  private current: ChildProcess | undefined;
  private preparation: AbortController | undefined;
  /** Invalidates requests that are still resolving or probing their interpreter. */
  private runGeneration = 0;
  /** Cached per interpreter path: probing costs a cold `import torch`, and a save
   *  that has to pay for it twice feels broken. */
  private readonly probed = new Map<string, string | undefined>();
  /** At most one pre-started, single-use worker for the next ordinary run. */
  private standby: Standby | undefined;
  private disposed = false;
  /** Per-window execution-cache store; exists only while the window-level mode is not off. */
  private resultStore: ExecutionResultStore | undefined;
  private cacheCounters: ExecutionCacheStats = emptyStats();
  /** Shadow comparisons run after the caller has its result, one at a time, in order. */
  private shadowPending: Promise<void> = Promise.resolve();

  constructor(
    private readonly backendDir: string,
    private readonly channel: { appendLine(value: string): void },
    private readonly options: TraceWorkerHost,
  ) {}

  /** Stop whatever is running. A newer save has arrived, or the panel closed —
   *  either way the answer in flight is about a file that no longer exists.
   *  An idle standby is not "whatever is running": it has executed nothing. */
  cancel(): void {
    this.runGeneration += 1;
    this.preparation?.abort();
    if (this.current) {
      this.channel.appendLine('[worker] cancelled by a newer request');
      killTree(this.current);
      this.current = undefined;
    }
  }

  dispose(): void {
    this.disposed = true;
    this.cancel();
    this.retireStandby();
    this.resultStore?.clear();
    this.resultStore = undefined;
  }

  /** Resolves when every scheduled shadow comparison has finished (tests). */
  executionCacheIdle(): Promise<void> { return this.shadowPending; }

  /** Shadow-mode counters and store occupancy, for logs and tests. */
  executionCacheStats(): ExecutionCacheStats & { entries: number; bytes: number } {
    return { ...this.cacheCounters, entries: this.resultStore?.size ?? 0, bytes: this.resultStore?.bytes ?? 0 };
  }

  /** "tensorViz: Clear Execution Cache": drop every stored result in this window. Returns
   *  how many were dropped. The next Run of each model executes. */
  clearExecutionCache(): number {
    const entries = this.resultStore?.size ?? 0;
    this.resultStore?.clear();
    if (entries) this.channel.appendLine(`[cache] cleared ${entries} stored result${entries === 1 ? '' : 's'}`);
    return entries;
  }

  /** The store and this run's mode when the cache is not off, else undefined. The one store
   *  serves every folder of the window, so only the window-level setting drops it (when off)
   *  or sizes it; a folder set to off skips the cache for its own runs and leaves the store.
   *  Off does nothing beyond reading the settings. */
  private cacheStore(resource: string): { store: ExecutionResultStore; mode: Exclude<ExecutionCacheMode, 'off'> } | undefined {
    let windowLevel: ExecutionCacheSettings | undefined, settings: ExecutionCacheSettings | undefined;
    try { windowLevel = this.options.executionCache?.(); } catch { windowLevel = undefined; }
    try { settings = this.options.executionCache?.(resource); } catch { settings = undefined; }
    if (windowLevel?.mode !== 'shadow' && windowLevel?.mode !== 'on') {
      if (this.resultStore) { this.resultStore.clear(); this.resultStore = undefined; }
      return undefined;
    }
    const mode = settings?.mode;
    if (mode !== 'shadow' && mode !== 'on') return undefined;
    const megabytes = Number.isFinite(windowLevel!.maxSizeMB) && windowLevel!.maxSizeMB! > 0 ? windowLevel!.maxSizeMB! : DEFAULT_MAX_SIZE_MB;
    this.resultStore ??= new ExecutionResultStore({ maxBytes: megabytes * 1024 * 1024 });
    this.resultStore.resize(megabytes * 1024 * 1024);
    return { store: this.resultStore, mode };
  }

  /** `on` mode, before launch: the stored result to return instead of running, or undefined
   *  (run fresh). An entry that fails re-validation is deleted, so it is never retried. */
  private serveLookup(store: ExecutionResultStore, lookup: ShadowLookup): { envelope: WorkerEnvelope; reused: ReusedCapture } | undefined {
    if (lookup.stored === undefined) return undefined;
    if (lookup.meta === undefined) {
      // Kept for comparison only: its capture was not complete (e.g. a byte-limit partial),
      // so it is never served. The run executes and the comparison still happens.
      this.channel.appendLine('[cache] stored result is comparison-only (capture not complete); running fresh');
      return undefined;
    }
    const served = servableEntry({ text: lookup.stored, meta: lookup.meta }, lookup.snapshot, WORKER_PROTOCOL_VERSION);
    if (served.ok) return served;
    this.cacheCounters.serveRejected += 1;
    store.delete(lookup.key);
    lookup.stored = lookup.meta = undefined;
    this.channel.appendLine(`[cache] stored result not servable (${served.reason}); dropped, running fresh`);
    return undefined;
  }

  /** Before launch: compute the key and read the store. Synchronous (it runs between the
   *  standby claim's preconditions and the claim) and never throws into the run. */
  private shadowLookup(store: ExecutionResultStore, request: TraceRequest, snapshot: ExecutionSnapshot | undefined,
    candidate: boolean, args: string[], env: NodeJS.ProcessEnv, interpreter: string): ShadowLookup | undefined {
    try {
      const ineligible = requestIneligible(request, snapshot, candidate);
      if (ineligible || !snapshot) {
        this.cacheCounters.ineligible += 1;
        this.channel.appendLine(`[cache] not eligible (${ineligible})`);
        return undefined;
      }
      const launched = [...args, REUSE_AUDIT_FLAG];
      const key = resultKey(resultKeyParts({ applicabilityDigest: snapshot.applicabilityDigest, args: launched, env, interpreter,
        backendDir: this.backendDir, protocol: WORKER_PROTOCOL_VERSION }));
      this.cacheCounters.lookups += 1;
      const entry = store.entry(key);
      return { key, snapshot, stored: entry?.text, meta: entry?.meta, args: launched };
    } catch {
      this.channel.appendLine('[cache] key unavailable; run not cached');
      return undefined;
    }
  }

  /** After the run: decide storability now (from the outcome the caller is about to get)
   *  and return the comparison + store/refresh as work that `run` schedules after the
   *  caller has its result, so neither JSON parsing nor the comparison delays it. Both
   *  sides are re-read from text, so a caller mutating its envelope cannot affect it. */
  private shadowRecord(store: ExecutionResultStore, lookup: ShadowLookup, outcome: TraceOutcome, stdout: string): () => void {
    let unstorable: string | undefined;
    try { unstorable = outcomeUnstorable(outcome, lookup.snapshot); } catch { unstorable = 'unreadable'; }
    const finishedAt = Date.now();
    const text = stdout.trim();
    const work = (): void => {
      if (this.resultStore !== store) return;   // mode switched off or disposed meanwhile
      try {
        if (lookup.stored !== undefined) {
          this.cacheCounters.hits += 1;
          if (unstorable) {
            // Serving would have shown a result this run could not reproduce. Drop it.
            this.cacheCounters.hitNotStorable += 1;
            store.delete(lookup.key);
            this.channel.appendLine(`[cache] shadow hit: fresh run not storable (${unstorable})`);
            return;
          }
          const difference = shadowDifference(JSON.parse(lookup.stored), JSON.parse(text));
          if (difference) {
            this.cacheCounters.different += 1;
            this.channel.appendLine(`[cache] shadow hit: DIFFERENT (${difference})`);
          } else {
            this.cacheCounters.identical += 1;
            this.channel.appendLine('[cache] shadow hit: identical');
          }
        } else {
          this.cacheCounters.misses += 1;
          if (unstorable) {
            this.cacheCounters.notStorable += 1;
            this.channel.appendLine(`[cache] shadow miss: not storable (${unstorable})`);
            return;
          }
        }
        // Serving identity, from the run's own (not yet mutated) outcome. Undefined keeps the
        // entry for shadow comparison only; `on` mode will then run fresh.
        let meta: ReturnType<typeof storedRunMeta>;
        try { meta = storedRunMeta({ ...outcome, envelope: JSON.parse(text) } as TraceOutcome, lookup.snapshot, finishedAt); } catch { meta = undefined; }
        if (store.putText(lookup.key, text, meta)) {
          this.cacheCounters.stored += 1;
          if (lookup.stored === undefined) this.channel.appendLine(`[cache] shadow miss: stored ${lookup.key.slice(0, 12)}${meta ? '' : ' (comparison only: capture not complete, never served)'}`);
        } else this.channel.appendLine('[cache] not stored (entry exceeds the size limit)');
      } catch {
        this.channel.appendLine('[cache] shadow comparison failed; run unaffected');
      }
    };
    return work;
  }

  /** Start the next run's process now, so it has imported torch by the time it is
   *  needed. `env` is the run environment without its per-run plan/candidate paths. */
  private prestart(interpreter: string, env: NodeJS.ProcessEnv): void {
    const key = standbyKey(interpreter, this.backendDir, env);
    if (this.disposed || this.standby?.key === key) return;
    this.retireStandby();
    // A backend staged without the standby (an older or partial copy) keeps the
    // ordinary path instead of spawning a Python that can only fail.
    const script = path.join(this.backendDir, 'services/ingest/standby.py');
    if (!existsSync(script)) return;
    let child: ChildProcess;
    try {
      child = spawnWorker(interpreter, [script], this.backendDir, env);
    } catch { return; /* The next run spawns ordinarily and reports any real error. */ }
    const standby: Standby = { child, key, stderr: [], stderrBytes: 0, usable: true, timer: undefined };
    // A write to a standby that has just died must not become an unhandled error.
    child.stdin?.on('error', () => { standby.usable = false; });
    child.stdout?.on('data', () => {
      // stdout is the run's envelope channel; anything before the request is a fault.
      standby.usable = false;
      if (this.standby === standby) this.retireStandby();
    });
    child.stderr?.on('data', (chunk: Buffer) => {
      if (standby.stderrBytes + chunk.length > STANDBY_IDLE_STDERR_LIMIT) {
        standby.usable = false;
        if (this.standby === standby) this.retireStandby();
      } else { standby.stderr.push(chunk); standby.stderrBytes += chunk.length; }
    });
    child.once('error', () => { standby.usable = false; if (this.standby === standby) this.standby = undefined; });
    child.once('exit', () => { standby.usable = false; if (this.standby === standby) this.standby = undefined; });
    standby.timer = setTimeout(() => { if (this.standby === standby) this.retireStandby(); }, STANDBY_IDLE_MS);
    standby.timer.unref?.();
    // An idle standby must never be what keeps its host alive: a host that exits
    // closes the standby's stdin, which ends it. A claimed one is ref'd again below.
    setHandlesRef(child, false);
    this.standby = standby;
  }

  /** Take the standby for this exact run, or undefined (the caller spawns as before).
   *  Synchronous by design: it removes the standby from the slot and detaches its
   *  idle listeners in the same tick `runProcess` attaches the run's, so no output
   *  is lost and two runs can never both take it. A standby that does not match the
   *  run is retired, never repurposed. */
  private claimStandby(interpreter: string, env: NodeJS.ProcessEnv, request: string): PrestartedWorker | undefined {
    const standby = this.standby;
    if (!standby) return undefined;
    this.standby = undefined;
    clearTimeout(standby.timer);
    const { child } = standby;
    if (!standby.usable || child.exitCode !== null || child.signalCode !== null
      || standby.key !== standbyKey(interpreter, this.backendDir, env)) {
      killTree(child);
      return undefined;
    }
    child.stdout?.removeAllListeners('data');
    child.stderr?.removeAllListeners('data');
    setHandlesRef(child, true);
    return { child, request, stderr: standby.stderr };
  }

  private retireStandby(): void {
    const standby = this.standby;
    if (!standby) return;
    this.standby = undefined;
    clearTimeout(standby.timer);
    killTree(standby.child);
  }

  /** Trace `request`, or explain why not. Never throws.
   *
   *  A fake-tier automatic capture that stops on a fake-tensor limitation (the model
   *  reads a value: `.item()`, `nonzero`, `unique`, ...) is re-run ONCE with real
   *  tensors and the same capture preparation, and that real outcome is returned with
   *  `fakeTensorFallback` set. Nothing else is retried, and the real tier never is. */
  async run(request: TraceRequest): Promise<TraceOutcome> {
    let preparation: PreparationSnapshot | undefined;
    const fake = request.tier === 'fake' && !request.execution;
    // runInternal cancels (bumps the generation) synchronously before its first await.
    const generation = this.runGeneration + 1;
    const outcome = await this.runOnce(request, fake ? { prepared: snapshot => { preparation = snapshot; } } : undefined);
    const limitation = fake ? fakeTensorLimitation(outcome) : undefined;
    if (!limitation || !preparation || generation !== this.runGeneration || this.disposed) return outcome;
    this.channel.appendLine(`[worker] fake-tensor run stopped: ${limitation.reason}`
      + `${limitation.operation ? ` (${limitation.operation})` : ''} during ${limitation.phase}; re-running once with real tensors`);
    const real = await this.runOnce({ ...request, tier: 'real' }, { preparation });
    return { ...real, fakeTensorFallback: limitation };
  }

  private async runOnce(request: TraceRequest, reuse?: Parameters<TraceWorkerCore['runInternal']>[3]): Promise<TraceOutcome> {
    let shadow: (() => void) | undefined;
    const outcome = await this.runInternal(request, undefined, work => { shadow = work; }, reuse);
    // Shadow comparison after the caller has its result, never before it.
    if (shadow) {
      const work = shadow;
      this.shadowPending = this.shadowPending.then(() => new Promise<void>(resolve => {
        setImmediate(() => { work(); resolve(); });
      }));
    }
    return outcome;
  }

  /** Static saved-plan/environment check for a retained live capture. This does
   * not probe Torch, start a trace, cancel a running model or save source. */
  async checkCapturedSetup(workspace: string, plan: ResolvedExecutionPlan, planDigest: string,
    signal: AbortSignal): Promise<boolean> {
    const generation = this.runGeneration;
    const current = () => !signal.aborted && this.options.trusted() && generation === this.runGeneration;
    try {
      if (!current()) return false;
      const resource = path.join(workspace, plan.source.modelFile);
      const resolve = this.options.interpreter;
      const python = await resolve(resource);
      if (!current() || python !== plan.interpreter.executable) return false;
      const snapshot = await resolveRetainedPlan(python, this.backendDir, workspace, plan, signal, this.options.documents(workspace));
      return current() && snapshot.applicabilityDigest === planDigest && await resolve(resource) === python && current();
    } catch { return false; }
  }

  /** Explicit private candidate lifecycle; no live attempt callbacks or slots. */
  async runCandidate(request: { workspace: string; file: string; cls: string; depth?: number;
    candidate: CandidateInput; current(): Promise<boolean> }, signal: AbortSignal): Promise<CandidateRunResult> {
    const { workspace, file, cls, depth, candidate: input } = request;
    const targetCurrent = request.current.bind(request);
    this.cancel();
    let generation = this.runGeneration;
    const cancelled = (): CandidateRunResult => ({ outcome: { ok: false, cancelled: true,
      reason: 'Candidate validation was cancelled.', detail: 'No candidate source was applied.' } });
    const abort = () => { if (generation === this.runGeneration) this.cancel(); };
    signal.addEventListener('abort', abort, { once: true });
    try {
      if (signal.aborted) return cancelled();
      const candidate = candidateRequest(input);
      const current = async () => !signal.aborted && this.options.trusted() && await targetCurrent();
      if (path.relative(workspace, file).split(path.sep).join('/') !== candidate.plan.source.modelFile
        || cls !== candidate.plan.source.className || !await current()) throw new Error('candidate_target_changed');
      if (generation !== this.runGeneration || signal.aborted) return cancelled();
      const pending = this.runInternal({ workspace, file, cls,
        depth, expectedInterpreter: candidate.plan.interpreter.executable,
        timeoutMs: candidate.plan.timeoutMs, beforeLaunch: current }, candidate);
      generation = this.runGeneration;
      const { candidate: evidence, ...outcome } = await pending;
      if (signal.aborted || generation !== this.runGeneration) return cancelled();
      // Cleanup can await process termination or filesystem work. Recheck after
      // that boundary before publishing an earlier freshness verdict.
      if (outcome.executionCurrent && outcome.recheckExecution) {
        outcome.executionCurrent = await outcome.recheckExecution(signal);
        if (!outcome.executionCurrent) {
          if (evidence) evidence.baseCurrent = false;
          outcome.recheckExecution = async () => false;
        }
      }
      if (signal.aborted || generation !== this.runGeneration) return cancelled();
      return { outcome, evidence };
    } catch (error) {
      if (error instanceof CandidateCleanupError) return { cleanupFailed: true, outcome: { ok: false,
        reason: 'Candidate cleanup could not be completed.',
        detail: 'Private temporary files may remain. No source change was applied.' } };
      return signal.aborted || generation !== this.runGeneration ? cancelled() : { outcome: { ok: false,
        reason: 'The reviewed candidate or its saved setup is unavailable or changed.',
        detail: 'Refresh the graph and review the candidate again. Nothing was applied.' } };
    } finally { signal.removeEventListener('abort', abort); }
  }

  private async runInternal(request: TraceRequest, candidate?: CandidateRequest,
    deferShadow?: (work: () => void) => void,
    /** Fake-tier fallback only: `prepared` receives the automatic capture preparation;
     *  `preparation` re-runs with that same preparation instead of preparing again, so
     *  the re-run's capture belongs to the attempt's one preparation. */
    reuse?: { prepared?: (snapshot: PreparationSnapshot) => void; preparation?: PreparationSnapshot },
  ): Promise<TraceOutcome & { candidate?: CandidateEvidence }> {
    this.cancel();
    if (!this.options.trusted()) return { ok: false,
      reason: 'Model execution requires host authorization.', detail: 'Nothing was run.' };
    const automatic = !candidate && !request.execution && !!request.workspace && (!request.kind || request.kind === 'model');
    let args: string[];
    try { args = workerArgs(request, candidate ? 'candidate' : automatic ? 'preparation' : 'plan'); }
    catch { return { ok: false, reason: 'Invalid tensor input descriptions.', detail: 'Use bounded shape/dtype/fill descriptions for a model request.' }; }
    const generation = this.runGeneration;
    const superseded = (): TraceOutcome => ({
      ok: false,
      cancelled: true,
      reason: 'The trace was superseded by a newer request.',
      detail: '',
    });
    const resource = request.file;
    const resolve = this.options.interpreter;
    const interpreter = await resolve(resource);
    if (generation !== this.runGeneration) {
      return superseded();
    }
    if (!interpreter) {
      return {
        ok: false,
        reason: 'tensorViz could not find your Python interpreter.',
        detail:
          'Choose Python in TensorViz with Open as Graph or Select Python Environment, then retry.',
      };
    }
    if (request.expectedInterpreter && interpreter !== request.expectedInterpreter) {
      return { ok: false, reason: 'The selected Python environment changed.',
        detail: 'Resolve the saved execution again before running it.' };
    }
    const chosen = interpreter;
    const missing = await this.probe(chosen);
    if (generation !== this.runGeneration) {
      return superseded();
    }
    if (missing) {
      // A failed selected environment is actionable; changing Python would break
      // the saved setup and could substitute CPU Torch for the user's GPU build.
      return {
        ok: false,
        reason: `${chosen} cannot run the trace worker.`,
        detail: `${missing}\n\nInstall the two packages it needs:\n    ${INSTALL_HINT}`,
      };
    }

    const preparation = new AbortController();
    this.preparation = preparation;
    let bootstrap: Awaited<ReturnType<typeof executionPlanFile>> | undefined;
    let executionSnapshot: ExecutionSnapshot | undefined;
    let captureSnapshot: PreparationSnapshot | undefined;
    let candidateSnapshot: ExecutionSnapshot | undefined;
    let childForRun: ChildProcess | undefined;
    let timeoutMs = request.timeoutMs ?? this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const preparePlan = (signal = preparation.signal) => resolveExecutionPlan(chosen, this.backendDir, {
      workspace: request.workspace!, documents: this.options.documents(request.workspace!), file: request.file, cls: request.cls!, execution: request.execution!,
      job: request.job, profileSpec: request.profileSpec, timeoutMs, generation, attemptId: request.attemptId,
    }, signal);
    const prepareCapture = (signal = preparation.signal) => resolveCapturePreparation(chosen, this.backendDir, {
      workspace: request.workspace!, documents: this.options.documents(request.workspace!), file: request.file, cls: request.cls!, timeoutMs, generation,
      attemptId: request.attemptId, captureRequest: { input: request.input ?? null,
        tensorInputs: request.tensorInputs ?? null, overrides: request.overrides ? JSON.parse(request.overrides) : null },
    }, signal);
    const prepareCandidate = (signal = preparation.signal) => resolveRetainedPlan(chosen, this.backendDir,
      request.workspace!, candidate!.plan, signal, this.options.documents(request.workspace!));
    try {
      if (candidate) {
        candidateSnapshot = await prepareCandidate();
        if (candidateSnapshot.applicabilityDigest !== candidate.capture.planDigest) throw new Error('candidate_base_changed');
        bootstrap = await candidateRequestFile(candidate);
      } else if (request.execution) {
        executionSnapshot = await preparePlan();
        timeoutMs = executionSnapshot.plan.timeoutMs;
        bootstrap = await executionPlanFile(executionSnapshot);
        request.onResolvedPlan?.(executionSnapshot);
      } else if (automatic) {
        captureSnapshot = reuse?.preparation ?? await prepareCapture();
        timeoutMs = captureSnapshot.preparation.timeoutMs;
        bootstrap = await executionPlanFile(captureSnapshot);
        reuse?.prepared?.(captureSnapshot);
        request.onCapturePrepared?.(captureSnapshot);
      }
      const compiled = nativeBackend(this.backendDir);
      if (compiled) {
        args = backendPythonArgs(this.backendDir, 'services/ingest/bootstrap.py', args.slice(2));
      } else if (bootstrap) {
        // Do not occupy the user's `services` package while importing their model.
        args.splice(0, 2, path.join(this.backendDir, 'services/ingest/bootstrap.py'));
      }
      if (generation !== this.runGeneration) return superseded();
      if (request.beforeLaunch) {
        let ready = false;
        try { ready = await request.beforeLaunch(); } catch { /* fail closed */ }
        if (generation !== this.runGeneration) return superseded();
        if (!ready) return { ok: false, reason: 'The saved execution changed before launch.',
          detail: 'Save current files and settings, then validate again. Nothing was run.' };
      }
      if (!this.options.trusted() || (candidate || request.prepareInputs) && await resolve(resource) !== chosen) {
        return { ok: false, reason: 'The candidate environment or Workspace Trust changed before launch.',
          detail: 'Review the candidate with the current environment. Nothing was run.' };
      }
      if (generation !== this.runGeneration) return superseded();

      const runEnv: NodeJS.ProcessEnv = { ...workerEnv(process.env, this.backendDir, !!bootstrap || compiled),
        TENSORVIZ_PARENT_PIPE: this.options.parentPipe ? '1' : undefined,
        // Never inherit another run's bootstrap through the parent environment.
        [EXECUTION_PLAN_ENV]: candidate ? undefined : bootstrap?.file,
        [CANDIDATE_REQUEST_ENV]: candidate ? bootstrap?.file : undefined };
      // Execution cache (shadow): key and lookup before launch. Off mode reads the setting
      // and nothing else.
      const cache = this.cacheStore(resource);
      const store = cache?.store;
      const lookup = store && deferShadow ? this.shadowLookup(store, request, executionSnapshot, !!candidate, args, runEnv, chosen) : undefined;
      if (lookup) args = lookup.args;
      // `on` mode: an identical earlier result replaces the launch. Nothing below the
      // worker is skipped by the caller — projection and every freshness check still run.
      const served = cache?.mode === 'on' && lookup && request.reuse === 'allow' && executionSnapshot
        ? this.serveLookup(cache.store, lookup) : undefined;
      if (served && executionSnapshot) {
        const planSnapshot = executionSnapshot;
        // The same post-run check a fresh run gets: the saved setup still resolves to the
        // digest this result was produced under, with the same interpreter.
        const recheckExecution = async (signal: AbortSignal) => {
          if (signal.aborted || generation !== this.runGeneration) return false;
          try {
            return await resolve(resource) === chosen
              && (await preparePlan(signal)).applicabilityDigest === planSnapshot.applicabilityDigest
              && !signal.aborted && generation === this.runGeneration;
          } catch { return false; }
        };
        const executionCurrent = await recheckExecution(preparation.signal);
        if (generation !== this.runGeneration) return superseded();
        if (!this.options.trusted()) return { ok: false, reason: 'Model execution requires host authorization.', detail: 'Nothing was run.' };
        this.cacheCounters.hits += 1;
        this.cacheCounters.served += 1;
        this.channel.appendLine(`[cache] served ${lookup!.key.slice(0, 12)}: reused identical result of attempt `
          + `${served.reused.attemptId} from ${new Date(served.reused.capturedAt).toISOString()} (model not run again)`);
        // `servableEntry` has rechecked the installed files; callers keep the verdict only.
        dropEnvironmentFiles(served.envelope);
        return { ok: true, envelope: served.envelope, exitCode: EXIT.ok, executionCurrent, recheckExecution, reused: served.reused };
      }
      const loggedArgs = args.map((arg, index) => ['--execution', '--diagnostic-job'].includes(args[index - 1]) ? '<structured execution setup>' : arg);
      this.channel.appendLine(`[worker] ${chosen} ${loggedArgs.join(' ')}`);
      let stdout = '';
      const finish = <T extends TraceOutcome>(result: T): T => {
        if (store && lookup) deferShadow!(this.shadowRecord(store, lookup, result, stdout));
        // After `shadowRecord` has its verdict: it stores and compares the stdout text, which
        // keeps the installed-file list for the recheck before serving.
        dropEnvironmentFiles(result.envelope);
        return result;
      };
      // Only the ordinary private-bootstrap launch has a standby equivalent. Candidate
      // runs, the compiled backend and legacy `-m` launches keep spawning as before.
      const standbyEligible = !compiled && !candidate && !!bootstrap
        && args[0] === path.join(this.backendDir, 'services/ingest/bootstrap.py');
      // No await between the claim and runProcess: the handoff must be one tick.
      const prestarted = standbyEligible ? this.claimStandby(chosen, runEnv,
        JSON.stringify({ argv: args.slice(1), env: { [EXECUTION_PLAN_ENV]: bootstrap!.file } }) + '\n') : undefined;
      if (prestarted) this.channel.appendLine('[worker] using pre-started worker');
      const started = Date.now();
      const outcome = await runProcess(
        chosen,
        args,
        {
          cwd: this.backendDir,
          env: runEnv,
          timeoutMs,
          prestarted,
        },
        (child) => {
          childForRun = child;
          if (generation === this.runGeneration) {
            this.current = child;
          } else {
            killTree(child);
          }
        },
      );
      if (standbyEligible && !outcome.spawnError) this.prestart(chosen, standbyEnvironment(runEnv));
      if (this.current === childForRun) {
        this.current = undefined;
      }
      if (generation !== this.runGeneration) {
        return superseded();
      }
      this.channel.appendLine(
        `[worker] exit ${outcome.code} in ${Date.now() - started}ms` +
          (outcome.timedOut ? ' (timed out)' : ''),
      );
      if (outcome.stderr.trim()) {
        this.channel.appendLine(candidate ? '[worker] Candidate stderr withheld; inspect structured diagnostics.' : outcome.stderr.trimEnd());
      }

      if (outcome.outputLimit) return { ok: false, reason: 'The trace worker exceeded its output limit.',
        detail: `The ${outcome.outputLimit} limit was exceeded and the worker was stopped. No complete execution evidence is available.` };
      if (outcome.spawnError) {
        return {
          ok: false,
          reason: `tensorViz could not start ${interpreter}.`,
          detail: outcome.spawnError.message,
        };
      }
      if (outcome.timedOut) {
        return {
          ok: false,
          timedOut: true,
          reason: 'The trace timed out.',
          detail:
            `${request.cls ?? request.callable ?? request.file} did not finish within ` +
            `${Math.round(timeoutMs / 1000)}s and was stopped.`,
        };
      }

      stdout = outcome.stdout;
      const parsed = parseEnvelope(outcome.stdout);
      if (!parsed.ok) {
        return candidate ? { ok: false, reason: 'The candidate worker returned unreadable evidence.',
          detail: 'No candidate result was certified.', exitCode: outcome.code ?? undefined }
          : { ...parsed, exitCode: outcome.code ?? undefined };
      }
      const envelope = parsed.envelope;
      let evidence: CandidateEvidence | undefined;
      let executionCurrent: boolean | undefined;
      let recheckExecution: TraceOutcome['recheckExecution'];
      const snapshot = executionSnapshot ?? captureSnapshot;
      if (candidate && candidateSnapshot) {
        evidence = candidateEvidence(envelope, candidate);
        if (!evidence) return { ok: false, reason: 'The candidate worker could not verify the exact reviewed proposal.',
          detail: 'The candidate or its saved setup was refused. Nothing was applied.', exitCode: outcome.code ?? undefined };
        const workerBaseCurrent = evidence.baseCurrent, workerCandidateCurrent = evidence.candidateCurrent;
        const baseCurrent = async (signal: AbortSignal) => {
          if (!workerBaseCurrent || signal.aborted || generation !== this.runGeneration || !this.options.trusted()) return false;
          try {
            const now = await prepareCandidate(signal);
            return await resolve(resource) === chosen && now.applicabilityDigest === candidateSnapshot!.applicabilityDigest
              && sameExecutionGuards(candidateSnapshot!.guards, now.guards) && !!await request.beforeLaunch?.()
              && !signal.aborted && generation === this.runGeneration;
          } catch { return false; }
        };
        evidence.baseCurrent = await baseCurrent(preparation.signal);
        executionCurrent = evidence.baseCurrent && evidence.candidateCurrent;
        const initiallyCurrent = executionCurrent;
        recheckExecution = async signal => initiallyCurrent && workerCandidateCurrent && await baseCurrent(signal);
        if (generation !== this.runGeneration) return superseded();
      } else if (snapshot && envelope.result) {
        // Keep captured evidence even if its source/config/environment can no longer
        // certify the current setup. No result is relabelled as a different plan.
        const diagnostics = (envelope.result as { diagnostics?: {
          execution?: { planDigest?: string; planCurrent?: boolean };
          automatic_capture?: { preparationDigest?: string; current?: boolean };
        } }).diagnostics;
        const workerCurrent = executionSnapshot
          ? diagnostics?.execution?.planDigest === snapshot.applicabilityDigest && diagnostics.execution.planCurrent === true
          : diagnostics?.automatic_capture?.preparationDigest === snapshot.applicabilityDigest && diagnostics.automatic_capture.current === true;
        const prepare = executionSnapshot ? preparePlan : prepareCapture;
        recheckExecution = async signal => {
          if (!workerCurrent || signal.aborted || generation !== this.runGeneration) return false;
          try {
            return await resolve(resource) === chosen
              && (await prepare(signal)).applicabilityDigest === snapshot.applicabilityDigest
              && !signal.aborted && generation === this.runGeneration;
          } catch { return false; }
        };
        executionCurrent = await recheckExecution(preparation.signal);
        if (generation !== this.runGeneration) return superseded();
      }
      if (outcome.code === EXIT.ok) {
        return finish({ ok: true, envelope, exitCode: outcome.code, executionCurrent, recheckExecution, ...(evidence ? { candidate: evidence } : {}) });
      }
      // A forward that raised still produced everything up to the failure, and that
      // partial graph is exactly what shows the user where it went wrong.
      return finish({
        ok: false,
        reason: candidate ? 'The candidate did not complete successfully.' : envelope.error?.message ?? `The trace worker exited ${outcome.code}.`,
        detail: candidate ? 'Inspect its structured execution diagnostics. Nothing was applied.' : outcome.stderr.trim() || (envelope.error?.type ?? ''),
        envelope,
        exitCode: outcome.code ?? undefined,
        executionCurrent,
        recheckExecution,
        ...(evidence ? { candidate: evidence } : {}),
      });
    } catch {
      return generation !== this.runGeneration ? superseded() : { ok: false,
        reason: 'The saved execution plan could not be resolved.',
        detail: 'Save configuration and source, check the selected environment, then validate again. Nothing was certified.' };
    } finally {
      if (this.preparation === preparation) this.preparation = undefined;
      if (candidate && childForRun && await terminations.get(childForRun) === false) {
        this.channel.appendLine('[worker] Candidate process tree termination failed; private workspace retained.');
        throw new CandidateCleanupError();
      }
      try { await bootstrap?.dispose(); }
      catch {
        this.channel.appendLine('[worker] Could not remove the private temporary execution plan.');
        if (candidate) throw new CandidateCleanupError();
      }
    }
  }

  /** `import torch, pydantic` in the user's interpreter, or the reason it failed. */
  private async probe(interpreter: string): Promise<string | undefined> {
    if (this.probed.has(interpreter)) {
      return this.probed.get(interpreter);
    }
    const generation = this.runGeneration;
    const outcome = await runProcess(
      interpreter,
      ['-c', PROBE],
      { cwd: this.backendDir, env: process.env, timeoutMs: 90_000 },
      // Registered, not discarded. This is a 90s budget that `cancel()` and the host's
      // own abort could not reach, so a first click on a cold interpreter held the
      // Cancel button hostage — and `run()` can probe twice before the trace starts.
      // Same bookkeeping `run()` does, so a probe from a superseded generation is
      // killed rather than left to finish.
      (child) => {
        if (generation === this.runGeneration) {
          this.current = child;
        } else {
          killTree(child);
        }
      },
    );
    let failure: string | undefined;
    if (outcome.outputLimit) {
      failure = 'interpreter probe exceeded its output limit';
    } else if (outcome.spawnError) {
      failure = outcome.spawnError.message;
    } else if (outcome.timedOut) {
      failure = 'no answer within 90s';
    } else if (outcome.code !== 0) {
      failure = outcome.stderr.trim().split('\n').pop() || `exit ${outcome.code}`;
    } else {
      this.channel.appendLine(`[worker] ${interpreter}: torch/pydantic ${outcome.stdout.trim()}`);
    }
    // Cache a VERDICT about the interpreter, never the fact that we stopped asking.
    //
    // The probe registers its child above so Cancel can reach it, which means a newer
    // save kills it — and a killed probe looks exactly like a broken interpreter:
    // `exit null`. Cached, that one superseded save made `run()` refuse EVERY later
    // trace with "<python> cannot run the trace worker" for the rest of the worker's
    // life, with the real interpreter sitting there able to import torch the whole
    // time. It needs only a save while the first probe is still running, which is the
    // normal shape of a cold start: four Electron suites reproduced it, each by
    // opening a file and saving it.
    if (generation === this.runGeneration) {
      this.probed.set(interpreter, failure);
    }
    return failure;
  }
}
