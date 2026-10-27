import { sameTargetIdentity } from '@tensorviz/graph-host/graphIdentity';
export { sameTargetIdentity } from '@tensorviz/graph-host/graphIdentity';
/** Source synchronization and explicit capture for the local graph page.
 * The opening controller owns setup; this service enforces saved source, trust,
 * environment and freshness checks and projects captured evidence.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { isHostFile } from './workspaceFiles';
import { randomUUID } from 'node:crypto';
import type { TensorInputs } from '@tensorviz/graph-host/tensorInputs';
import type { ExecutionSpec, ExecutionJob } from '@tensorviz/graph-host/executionSpec';
import type { ExecutionGuard } from '@tensorviz/graph-host/executionGuard';
import { executionFileIdentities, verifyImportedSources, ExecutionFileError, type ExecutionFileIdentity } from './executionFiles';
import type { LocalBrowserValidationPhase } from '@tensorviz/graph-host/validationSummary';
export type { LocalBrowserValidationPhase } from '@tensorviz/graph-host/validationSummary';

import { declaredInit, overridesArgument } from './initComment';
import { InputMemory, inputMemoryFor } from './inputMemory';
import { LocalBrowserHostPublish } from '@tensorviz/graph-host/localBrowserHostTransport';
import {
  IdeProjectionDiagnostic,
  IdeSourceFileDigest,
  LocalBrowserSnapshotState,
  LocalBrowserSyncStateMessage,
  LocalBrowserTargetIdentity,
  LocalBrowserValidationOutcome,
  ProjectionPayload,
} from '@tensorviz/contracts/protocol';
import { SAVE_COALESCE_MS, ShapeHint, TYPING_DEBOUNCE_MS, mergeSavedPositions, shapeHintsByFile } from './projection';
import { DEFAULT_TIMEOUT_MS, EXIT, TraceOutcome, TraceRequest, WORKER_EXIT_CONSTRUCT_FAILED, WorkerEnvelope, type TraceWorker } from './worker';
import { executionDiagnosticDetail } from './executionDiagnostic';
import { executionFailure } from '@tensorviz/graph-host/executionFailure';
import { reusedResultLabel } from '@tensorviz/graph-host/executionCache';
import { graphDebugContext, type GraphDebugBaseline } from '@tensorviz/graph-host/graphDebug';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import type { AttemptHandle } from '@tensorviz/graph-host/attemptLedger';
import { validationPhaseDetails, type ValidationSummaryStore } from '@tensorviz/graph-host/validationSummary';
import { lookupProfile, type ProfileLookupOptions } from '@tensorviz/graph-host/profileValidation';
import {
  SourceIO,
  multifileProjectionBody,
  relativeInside,
  sha256Text,
  vscodeSourceIO,
} from './workspaceSources';

/** Same spelling as middleware.py's LOCAL_TOKEN_HEADER; declared here so this
 *  module stays free of stack.ts's imports (projection.ts does the same). */
const LOCAL_TOKEN_HEADER = 'X-tensorViz-Token';

/** The network read of a projection gets the same ceiling as the class scan. */
const REQUEST_TIMEOUT_MS = 10_000;

/** Where a projection's saved layout lives, relative to the workspace
 *  retained for compatibility with existing saved layouts. */
const VIEWS_DIR = path.join('.tensorviz', 'views');

/** The multi-file projection response, as
 *  backend/api/routers/ide.py's `/api/ide/multifile/project` returns it
 *  (docs/ide-multifile-projection-contract.md). `trace_error` is typed to the
 *  fields this loop consumes; the rest passes through. */
interface ProjectResponse {
  cls: string;
  document: unknown;
  spanmap: Record<string, unknown>;
  /** The ROOT file's digest — the same digest /api/ide/apply guards edits with. */
  source_digest: string;
  /** The digest over the ENTIRE supplied source set: any dependency edit moves it. */
  projection_digest: string;
  graph_id: string;
  revision: number;
  traced: boolean;
  trace_error?: {
    type?: string;
    message?: string;
    node_id?: string | null;
    file?: string;
    line?: number;
  } | null;
  /** The files the projection consulted, with the digests used — the set this
   *  loop subscribes to after every projection. */
  dependencies: IdeSourceFileDigest[];
  diagnostics: IdeProjectionDiagnostic[];
  truncated: boolean;
}

/** What the sync needs to run the user's model: the worker that spawns it, the
 *  consent gate, and the onSave/tier settings — the same idea as
 *  projection.ts's TraceSupport, minus the parts only the embedded canvas has
 *  (diagnostics, edit spans). Absent means the page stays on the AST tier. */
export interface LocalBrowserTraceSupport {
  worker: { run(request: TraceRequest): Promise<TraceOutcome>; cancel(): void; checkCapturedSetup?: TraceWorker['checkCapturedSetup'] };
  trusted(): Promise<boolean>;
  settings(): { onSave: boolean; tier: 'real' | 'fake' };
  /** Every envelope a trace produced, so it can be written down as a run —
   *  for the browser session the run lives under the TARGET's folder. */
  onEnvelope?: (
    envelope: WorkerEnvelope,
    target: { folder: string; file: string; cls: string },
    /** `reused`: no worker ran (execution cache); the envelope is an earlier run's. */
    run?: { reused: true },
  ) => void;
}

export interface LocalBrowserSyncOptions {
  channel: { appendLine(line: string): void };
  /** The running backend, read per call so a restart on a new port is picked up. */
  origin(): string | undefined;
  /** The per-launch token — a request header, never a URL. */
  token(): string | undefined;
  /** Publishes into the CURRENT session's transport; undefined when no session
   *  is connected yet, which late-joining views are replayed for by
   *  snapshotState(). */
  publish(message: LocalBrowserHostPublish): unknown;
  trace?: LocalBrowserTraceSupport;
  fetchImpl?: typeof fetch;
  debounceMs?: number;
  saveCoalesceMs?: number;
  /** How the workspace's Python source set is enumerated/read, per target
   *  folder; the production default is vscodeSourceIO. Injected in tests, where
   *  there is no filesystem to enumerate. */
  sourceIO?: (folder: string) => SourceIO;
  summaries?: ValidationSummaryStore;
  profiles?: ProfileLookupOptions;
  sourceChanged?(target: LocalBrowserTargetIdentity): void;
}

/** The target the loop is following. `identity` is what messages name: it
 *  starts as step 4's answer (backendRevision 0, "not projected yet") and is
 *  replaced by the identity of every projection that lands. */
interface BrowserActive {
  uri: vscode.Uri;
  rel: string;
  cls: string;
  folder: string;
  identity: LocalBrowserTargetIdentity;
  /** The last projection that succeeded; a buffer that stops parsing greys
   *  this rather than blanking the page. */
  last?: ProjectionPayload;
  debugBaseline?: GraphDebugBaseline;
  /** Present when current source or execution evidence could not be refreshed. */
  staleReason?: string;
  /** The files the last projection consulted (file -> digest), root included.
   *  Empty until the first projection lands: nothing else may trigger a
   *  refresh, and the root itself is always followed regardless. */
  dependencies: Map<string, string>;
  /** Verified runtime imports can include layers the static resolver never reached. */
  capturedSources: Map<string, string>;
  /** Observed output shapes by workspace-relative file, for the inlay hints.
   *  Empty for every source-only projection: only a run measures a shape. */
  shapes: Map<string, ShapeHint[]>;
}

class HttpError extends Error {
  constructor(readonly status: number, readonly body: unknown) {
    super(`HTTP ${status}`);
  }
}

/** Error bodies and exception messages can contain source, paths or secrets.
 * Keep only a structured syntax location; all other copy is host-owned. */
function refreshFailureReason(error: unknown, file: string, timedOut: boolean): string {
  const retry = 'Check tensorViz Output in VS Code, then run “tensorViz: Open as Graph” to retry.';
  if (timedOut) return `Graph refresh timed out. ${retry}`;
  if (error instanceof HttpError && error.status === 400) {
    const detail = (error.body as { detail?: { error?: string; line?: number } })
      ?.detail;
    if (detail?.error === 'syntax_error') {
      const line = Number.isSafeInteger(detail.line) && detail.line! > 0 ? detail.line : 1;
      return `${file}:${line} does not parse. Fix the syntax in VS Code; the graph refreshes after editing.`;
    }
  }
  if (error instanceof HttpError && error.status === 404) {
    return 'The selected class is no longer available. Restore it in VS Code or use “tensorViz: Open as Graph” to choose a model.';
  }
  if (error instanceof HttpError && error.status === 422) {
    return `The source could not be projected as a supported PyTorch module. ${retry}`;
  }
  if (error instanceof HttpError) return `Graph could not refresh (HTTP ${error.status}). ${retry}`;
  if (error instanceof InvalidProjectionResponse) return `Graph refresh received an invalid response. ${retry}`;
  return `Graph could not refresh. ${retry}`;
}

class InvalidProjectionResponse extends Error {}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Validate the envelope consumed here and by the page before replacing the
 * last valid graph. Graph internals remain owned by the backend/canvas schema. */
function isProjectResponse(value: unknown): value is ProjectResponse {
  if (!isRecord(value)) return false;
  const document = value.document;
  return typeof value.cls === 'string' && typeof value.graph_id === 'string' && !!value.graph_id
    && typeof value.source_digest === 'string' && typeof value.projection_digest === 'string'
    && Number.isSafeInteger(value.revision) && (value.revision as number) >= 0
    && typeof value.traced === 'boolean' && typeof value.truncated === 'boolean'
    && isRecord(value.spanmap) && isRecord(document) && isRecord(document.view)
    && isRecord(document.semantic) && Array.isArray(document.semantic.nodes)
    && document.semantic.nodes.every(node => isRecord(node) && typeof node.node_id === 'string'
      && (node.params === undefined || isRecord(node.params)))
    && Array.isArray(value.dependencies) && value.dependencies.every(entry => isRecord(entry)
      && typeof entry.file === 'string' && typeof entry.digest === 'string')
    && Array.isArray(value.diagnostics) && value.diagnostics.every(entry => isRecord(entry)
      && typeof entry.code === 'string' && ['info', 'warning', 'error'].includes(String(entry.severity))
      && typeof entry.file === 'string' && Number.isSafeInteger(entry.line)
      && typeof entry.import_text === 'string' && typeof entry.detail === 'string'
      && (entry.node === undefined || typeof entry.node === 'string'));
}

/** What a view's `revealNode` is answered with; localBrowserGraph.ts turns it
 *  into the contract ack. The refusal codes are the frozen contract's:
 *  `STALE_TARGET` when the page is not showing this window's current state,
 *  `INVALID_MESSAGE` when the request names nothing the current graph has. */
export type LocalBrowserRevealResult =
  | { ok: true }
  | { ok: false; errorCode: 'STALE_TARGET' | 'INVALID_MESSAGE'; detail: string };

/** A span as the parse/project response carries it (task.md §6.2). It arrives
 *  over the wire as unknown, and navigation only accepts exactly this shape. */
interface NodeSpan {
  file: string;
  start: [number, number];
  end: [number, number];
  file_digest?: string;
}

function asNodeSpan(value: unknown): NodeSpan | undefined {
  const span = value as Partial<NodeSpan> | null;
  if (
    typeof span !== 'object' ||
    span === null ||
    typeof span.file !== 'string' ||
    !Array.isArray(span.start) ||
    !Array.isArray(span.end) ||
    typeof span.start[0] !== 'number' ||
    typeof span.start[1] !== 'number' ||
    typeof span.end[0] !== 'number' ||
    typeof span.end[1] !== 'number'
  ) {
    return undefined;
  }
  return span as NodeSpan;
}

/** The nodes whose spans in `file` cover `line` (1-based). Innermost wins: a
 *  statement nested inside a compound one is the more specific answer —
 *  projection.ts's rule. A MULTI-file projection adds one new case to the old
 *  rule: tied innermost widths happen when several instances share one
 *  declaration (`enc1.fc` and `enc2.fc` are both Block's `self.fc = ...` line),
 *  and answering with just one would be a silent guess — so all ties are
 *  returned, sorted for determinism, and the page disambiguates. */
function nodesAtLine(spanmap: Record<string, unknown>, file: string, line: number): string[] {
  let best: string[] = [];
  let bestWidth = Number.MAX_SAFE_INTEGER;
  for (const [nodeId, raw] of Object.entries(spanmap)) {
    const span = asNodeSpan(raw);
    if (!span || span.file !== file || line < span.start[0] || line > span.end[0]) {
      continue;
    }
    const width = span.end[0] - span.start[0];
    if (width < bestWidth) {
      best = [nodeId];
      bestWidth = width;
    } else if (width === bestWidth) {
      best.push(nodeId);
    }
  }
  return best.sort();
}

/** The projection's own rule for paths, checked here too: navigation may aim an
 *  editor only at a normalised workspace-relative POSIX path — the browser never
 *  supplies one (spans come from THIS window's projection), so anything else is
 *  a contract violation, refused loudly rather than corrected. */
function isNormalizedRelativePath(file: string): boolean {
  if (file.includes('\\')) {
    return false;
  }
  if (file.startsWith('/') || file.startsWith('~') || /^[A-Za-z]:/.test(file)) {
    return false;
  }
  return file.split('/').every((segment) => segment !== '' && segment !== '.' && segment !== '..');
}

/** Why the model would not build, in words the user can act on.
 *
 * Exit 3 means "construction raised", which is a much wider net than "the constructor
 * needs arguments" — an import the environment does not have lands here too. Saying
 * "supply its constructor arguments with # tensorviz: init=" to someone whose real
 * problem is a missing package sends them to edit a file that was never wrong; a model
 * whose every argument already has a default gets told to supply arguments it does not
 * need. The worker already records the missing module and the exception type
 * (services/ingest/worker.py), so the refusal can name the actual cause.
 *
 * Exception TEXT still stays in the Output channel: it can carry source, paths and
 * credentials. A module name and an exception class are safe and are the two facts that
 * decide what the user does next.
 */
export function constructionRefusal(envelope: WorkerEnvelope | undefined, cls: string): string {
  const error = envelope?.error;
  const diagnosed = executionDiagnosticDetail(error?.diagnostic, cls);
  if (diagnosed) return diagnosed;
  const missing = typeof error?.missing_module === 'string'
    && /^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$/.test(error.missing_module)
    && error.missing_module.length <= 256 ? error.missing_module : '';
  if (missing) {
    return `${cls} could not be constructed: its code imports "${missing}", which the selected Python environment does not have. `
      + `Install the missing dependency in TensorViz's selected Python environment, then retry.`;
  }
  if (typeof error?.type === 'string' && /^[A-Za-z_][A-Za-z0-9_]{0,127}$/.test(error.type)) {
    return `${cls} could not be constructed: ${error.type} was raised while building it. `
      + `Check its tensorviz.toml model profile and tensorViz Output in VS Code for the traceback.`;
  }
  return `${cls} could not be constructed. Check its tensorviz.toml model profile and tensorViz Output in VS Code.`;
}

export class LocalBrowserSync implements vscode.Disposable {
  private readonly disposables: vscode.Disposable[] = [];
  private active: BrowserActive | undefined;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private saveTimer: ReturnType<typeof setTimeout> | undefined;
  /** Bumped per refresh/retarget so a slow answer for an older buffer cannot
   *  overwrite a newer projection (projection.ts's pattern). */
  private generation = 0;
  private traceGeneration = 0;
  /** The one projection request owned by this sync instance. A newer edit,
   *  retarget, or dispose aborts it immediately; generation checks remain the
   *  second line of defence for fetch implementations that ignore AbortSignal. */
  private projectionAbort: AbortController | undefined;
  private initialProjection: Promise<void> = Promise.resolve();
  /** Trace bookkeeping the test API reads (the navigation Electron leg): true
   *  while a save-coalesced trace is armed but not fired, plus a counter of
   *  traceNow invocations in flight — a re-target can overlap a superseded
   *  trace with a newer one. While either says a trace is pending, the
   *  identity the page shows can still roll its backendRevision. */
  private saveTraceScheduled = false;
  private tracesInFlight = 0;
  private manualValidation: AbortController | undefined;
  private automaticValidation: AbortController | undefined;
  /** The cursor's current node set as a joined key (one line can cover several
   *  instances after the multi-file projection): `focusNode` fires on entry, not
   *  on every keystroke within one statement, and a `revealNode` the sync just
   *  answered sets it first so that selection event cannot echo straight back
   *  at the page that asked (projection.ts's discipline). */
  private focused = '';
  /** Fires whenever the observed shapes change, so the inlay-hints provider can
   *  ask for them again. */
  private readonly shapesChanged = new vscode.EventEmitter<void>();
  readonly onDidChangeShapes = this.shapesChanged.event;

  constructor(private readonly options: LocalBrowserSyncOptions) {
    this.disposables.push(
      vscode.workspace.onDidChangeTextDocument((event) => {
        if (!isHostFile(event.document.uri)) return;
        const active = this.active;
        if (!active) {
          return;
        }
        const rel = relativeInside(active.folder, event.document.uri.fsPath);
        // Follow the root, static dependencies and verified runtime imports.
        if (event.contentChanges?.length === 0) return;
        if (rel && (rel === active.rel || active.dependencies.has(rel) || active.capturedSources.has(rel))) {
          this.schedule();
        }
      }),
      // Save, not change: executing the user's model on every keystroke would run
      // half-written code continuously, and the trace is the expensive tier.
      vscode.workspace.onDidSaveTextDocument((document) => {
        if (!isHostFile(document.uri)) return;
        const active = this.active;
        if (!active) {
          return;
        }
        const rel = relativeInside(active.folder, document.uri.fsPath);
        // A dependency save is a model change too. scheduleTrace runs only when
        // the host still opts in. The deprecated traceOnSave setting never does.
        if (rel && (rel === active.rel || active.dependencies.has(rel) || active.capturedSources.has(rel))) {
          this.scheduleTrace();
        }
      }),
      // line → node: the mirror of revealNode. Only the target's own document
      // speaks, and only while a projection's spanmap exists to read.
      vscode.window.onDidChangeTextEditorSelection((event) => this.onCursor(event)),
    );
  }

  /** The identity views and the test API should anchor on: the base identity
   *  until the first projection lands, then the identity of the last
   *  projection (real backend revision, real digested text). */
  currentTarget(): LocalBrowserTargetIdentity | undefined {
    return this.active?.identity;
  }

  /** A same-model command can reuse the view only while its source is unchanged. */
  async sourcesCurrent(): Promise<boolean> {
    const active = this.active, generation = this.generation;
    if (!active || active.staleReason || this.timer) return false;
    const io = (this.options.sourceIO ?? vscodeSourceIO)(active.folder);
    const sources = new Map([...active.dependencies, ...active.capturedSources]);
    sources.set(active.rel, active.identity.sourceDigest);
    try {
      for (const [file, digest] of sources) {
        const document = io.openDocument(file);
        if (sha256Text(document ? document.getText() : await io.read(file)) !== digest) return false;
        if (document && active.capturedSources.has(file) && sha256Text(await io.read(file)) !== digest) return false;
      }
    } catch { return false; }
    return this.active === active && generation === this.generation && !active.staleReason && !this.timer;
  }

  /** Observed shapes for `uri`, for the inlay-hints provider. Every file the
   *  projection reached can carry them, not only the class's own: a layer
   *  declared in an imported module is measured by the same run. */
  shapesFor(uri: vscode.Uri): ShapeHint[] {
    if (!isHostFile(uri)) return [];
    const active = this.active;
    if (!active) {
      return [];
    }
    const rel = relativeInside(active.folder, uri.fsPath);
    return (rel && active.shapes.get(rel)) || [];
  }

  /** Focus must wait for in-flight source projection before decorating an editor. */
  focusContext(target: LocalBrowserTargetIdentity): ReturnType<LocalBrowserSync['sourceEditContext']> {
    if (this.timer || this.projectionAbort) return undefined;
    return this.sourceEditContext(target);
  }

  /** Host-only source-edit context. Never accepts browser paths or source. */
  sourceEditContext(target: LocalBrowserTargetIdentity): {
    projection: ProjectionPayload;
    dependencies: Map<string, string>;
  } | undefined {
    const active = this.active;
    if (!active?.last || active.staleReason || !sameTargetIdentity(target, active.identity)) return undefined;
    return { projection: active.last, dependencies: new Map(active.dependencies) };
  }

  /** True while a trace this loop scheduled has not concluded — armed but not
   *  run yet, or still running. False is terminal for the identity the page
   *  shows: the trace landed as runtime-traced, or this buffer honestly stays
   *  source-only — and nothing here can roll backendRevision after it. */
  tracePending(): boolean {
    return !!this.manualValidation || this.saveTraceScheduled || this.tracesInFlight > 0;
  }

  cancelValidation(): void {
    this.automaticValidation?.abort();
    if (this.manualValidation && !this.manualValidation.signal.aborted) {
      this.manualValidation.abort();
      this.options.trace?.worker.cancel();
    }
  }

  /** A click validates the saved source represented by this projection, never an older disk copy. */
  async validate(target: LocalBrowserTargetIdentity, signal: AbortSignal, tensorInputs?: TensorInputs, progress?: (phase: LocalBrowserValidationPhase) => void,
    measured?: (target: LocalBrowserTargetIdentity, projection: ProjectionPayload) => void,
    captured?: (outcome: TraceOutcome) => void, execution?: ExecutionSpec, job?: ExecutionJob,
    /** The saved profile's own `limits.timeout_ms`, so a run is bounded the way its
     *  author asked rather than by this module's blanket ceiling. */
    timeoutMs?: number, guard?: ExecutionGuard, attempt?: AttemptHandle, reviewedDevice?: string,
    /** Opt this run in to the execution cache (interactive Canvas runs only; never MCP).
     *  In shadow mode it changes nothing about the run. In `on` mode
     *  `allow` may return an identical earlier result without running; `bypass` (Run fresh)
     *  always runs. Everything after the worker below runs either way. */
    reuse?: 'allow' | 'bypass'): Promise<LocalBrowserValidationOutcome> {
    const deviceOverride = execution && reviewedDevice !== undefined && execution.device !== reviewedDevice;
    if (execution && reviewedDevice !== undefined) execution = { ...execution, device: reviewedDevice };
    const refused = (detail: string, failure?: LocalBrowserValidationOutcome['failure']): LocalBrowserValidationOutcome =>
      ({ status: 'refused', detail, ...(failure ? { failure } : {}) });
    const cancelled = (): LocalBrowserValidationOutcome => ({ status: 'cancelled', detail: 'Validation stopped or the source changed. Refresh and validate again.' });
    const active = this.active, support = this.options.trace;
    if (!active || !sameTargetIdentity(target, active.identity)) return refused('The selected source changed. Reopen the current model.');
    if (!execution && !tensorInputs) return refused('Choose explicit model setup and inputs with Open as Graph. No candidate shapes were tried.');
    if (!execution && !this.sourceEditContext(target)) return refused('Wait for a current graph before validating.');
    const startingStaleReason = active.staleReason;
    const targetCurrent = () => sameTargetIdentity(target, active.identity) && (execution !== undefined || !!this.sourceEditContext(target));
    if (!support) return refused('Tracing is unavailable. Select a compatible Python environment in VS Code.');
    const scheduled = signal === this.automaticValidation?.signal;
    if (this.tracePending() && !scheduled) return refused('Another trace is running. Wait for it to finish and try again.');
    const controller = new AbortController();
    this.manualValidation = controller;
    const abort = () => {
      // A late cancel from a superseded caller cannot stop the newer worker.
      if (controller.signal.aborted) return;
      controller.abort();
      if (this.manualValidation === controller) support.worker.cancel();
    };
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort();
    // TraceWorker owns the execution deadline (including process-tree termination).
    // A second wall-clock cutoff here would consume the profile's run budget during
    // interpreter probing and reject valid long runs. Projection has its own bound.
    const dependencies = new Map(active.dependencies);
    dependencies.set(active.rel, target.sourceDigest);
    let executionFiles: ExecutionFileIdentity[] = [];
    const io = (this.options.sourceIO ?? vscodeSourceIO)(active.folder);
    const current = () => !controller.signal.aborted && this.active === active
      && (!active.staleReason || active.staleReason === startingStaleReason);
    const checkSources = async (): Promise<string | undefined> => {
      for (const [file, digest] of dependencies) {
        const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(active.folder, file)));
        if (doc.isDirty) return 'Save the model and its changed dependencies in VS Code, then refresh the graph.';
        if (sha256Text(doc.getText()) !== digest || sha256Text(await io.read(file)) !== digest) {
          return 'Source changed after this graph was projected. Save the files, wait for refresh, then validate again.';
        }
      }
      return undefined;
    };
    const setupCurrent = async () => !guard || await guard.current(controller.signal);
    const staleSetup = 'Saved configuration, dependencies or the selected environment changed. Save current settings and validate again; previous measurements do not validate this setup.';
    try {
      if (!await support.trusted()) return refused('Trust this workspace in VS Code before running validation.');
      if (execution) {
        const dirtyPython = vscode.workspace.textDocuments.some(doc => {
          if (!isHostFile(doc.uri)) return false;
          const relative = path.relative(active.folder, doc.uri.fsPath);
          return doc.isDirty && /\.pyw?$/i.test(relative) && !path.isAbsolute(relative) && relative !== '..' && !relative.startsWith(`..${path.sep}`);
        });
        if (dirtyPython) return refused('Save changed Python files in this workspace before execution; setup functions may import them.');
        executionFiles = await executionFileIdentities(active.folder, execution, controller.signal);
        for (const recipe of [execution.setup]) {
          if (!recipe) continue;
          const setupDoc = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(active.folder, recipe.file)));
          dependencies.set(recipe.file, sha256Text(setupDoc.getText()));
        }
      }
      const problem = await checkSources();
      if (!current()) return cancelled();
      if (problem) return refused(problem);
      if (!targetCurrent()) return refused('The source changed. Refresh and try again.');
      const source = (await vscode.workspace.openTextDocument(active.uri)).getText();
      if (!current()) return cancelled();
      if (!await setupCurrent()) return current() ? refused(staleSetup) : cancelled();
      // Same input the save-driven trace uses, so "Validate" measures the model the
      // canvas is showing and not a different one. It does not harvest: a click on
      // the page must not make this window write a file.
      progress?.('tracing');
      attempt?.phase('environment');
      // With an explicit execution there is nothing left to guess, and nothing may be
      // allowed to creep back in: no remembered input, and no `# tensorviz: init=`
      // override — the profile's own `init` travels inside `execution`.
      const outcome = await support.worker.run({ tensorInputs: execution ? undefined : tensorInputs,
        execution, job, workspace: active.folder,
        expectedInterpreter: guard?.spec.interpreter.executable,
        // A device choice is an explicit inline invocation. The original saved
        // guard still checks all source/config/environment identities before launch.
        profileSpec: deviceOverride ? undefined : guard?.spec,
        attemptId: attempt?.snapshot().id,
        onResolvedPlan: snapshot => attempt?.resolved(snapshot),
        onCapturePrepared: snapshot => attempt?.prepared(snapshot),
        beforeLaunch: async () => {
          if (!current() || !await support.trusted() || !await setupCurrent()) return false;
          if (execution && JSON.stringify(executionFiles) !== JSON.stringify(await executionFileIdentities(active.folder, execution, controller.signal))) return false;
          const changed = await checkSources();
          return !changed && current() && targetCurrent() && await support.trusted();
        },
        timeoutMs: timeoutMs ?? DEFAULT_TIMEOUT_MS, file: active.uri.fsPath, cls: active.cls,
        tier: execution ? 'real' : support.settings().tier,
        ...(execution ? {} : { overrides: overridesArgument(declaredInit(source, active.cls)) }),
        ...(reuse && execution && !job ? { reuse } : {}),
        input: undefined });
      attempt?.captured(outcome);
      captured?.(outcome);
      if (!current()) return cancelled();
      if (!outcome.ok && outcome.cancelled) return cancelled();
      if (outcome.executionCurrent === false) return refused(staleSetup);
      if (!await setupCurrent()) return current() ? refused(staleSetup) : cancelled();
      if (!outcome.ok && outcome.timedOut) return { status: 'failed', failure: { kind: 'capture' },
        detail: 'Validation timed out at its execution limit and was stopped. Check the model or the profile timeout in tensorviz.toml, then retry.' };
      if (execution && JSON.stringify(executionFiles) !== JSON.stringify(await executionFileIdentities(active.folder, execution, controller.signal))) {
        return refused('An input, checkpoint or setup file changed during execution. The captured run is retained but cannot validate the current setup.');
      }
      const changed = await checkSources();
      if (!current()) return cancelled();
      if (changed) return refused(changed);
      const facts = (outcome.envelope?.result as { diagnostics?: {
        execution?: { importedSources?: unknown }; automatic_capture?: { importedSources?: unknown };
      } } | null)?.diagnostics;
      const prepared = facts?.automatic_capture;
      const imported = (execution ? facts?.execution : prepared)?.importedSources;
      if ((execution || prepared) && outcome.envelope?.result) {
        await verifyImportedSources(active.folder, imported, async file =>
          (await vscode.workspace.openTextDocument(vscode.Uri.file(file))).isDirty, controller.signal);
      }
      if (!current()) return cancelled();
      if (!await support.trusted()) return refused('Workspace Trust changed. Validation results were discarded.');
      if (!current()) return cancelled();
      const envelope = outcome.envelope;
      const failedForward = outcome.exitCode === EXIT.trace && !!envelope?.error;
      if (!envelope?.result || (!outcome.ok && !failedForward)) {
        // User exceptions can contain arbitrary source, paths and credentials. Keep the page's
        // failure actionable and leave Python details in the existing host Output channel.
        if (outcome.exitCode === WORKER_EXIT_CONSTRUCT_FAILED) return refused(constructionRefusal(envelope, active.cls), executionFailure(envelope?.error?.diagnostic));
        return { status: 'failed', failure: executionFailure(envelope?.error?.diagnostic), detail: executionDiagnosticDetail(envelope?.error?.diagnostic, active.cls, envelope?.error?.node_id) ?? (outcome.envelope?.error
          ? 'Validation failed during the forward pass. Check the model input, tensor dimensions and layer parameters.'
          : 'The trace could not complete. Check the selected Python environment and tensorViz Output in VS Code, then retry.') };
      }
      const before = active.last;
      progress?.('projecting');
      attempt?.phase('projection');
      const mergeFailure = await this.refresh(active, undefined, envelope, current);
      if (mergeFailure && !controller.signal.aborted && this.active === active) {
        return refused(`The trace finished, but the graph could not refresh. ${mergeFailure}`);
      }
      try {
        if (execution || prepared) await verifyImportedSources(active.folder, imported, async file =>
          (await vscode.workspace.openTextDocument(vscode.Uri.file(file))).isDirty, controller.signal);
      } catch (error) {
        if (current()) this.publishStale(active, staleSetup);
        return controller.signal.aborted ? cancelled() : refused(error instanceof ExecutionFileError ? error.message : staleSetup);
      }
      if (!await setupCurrent() || outcome.recheckExecution && !await outcome.recheckExecution(controller.signal)) {
        // Refresh may already have published the capture. Never leave it green when
        // a config/environment edit races the merge.
        if (current()) this.publishStale(active, staleSetup);
        return controller.signal.aborted ? cancelled() : refused(staleSetup);
      }
      const changedAfterMerge = await checkSources();
      if (!current()) return cancelled();
      if (changedAfterMerge) return refused(changedAfterMerge);
      if (active.last === before || !active.last?.traced) return refused('The trace finished, but the graph could not refresh. Reopen the graph and retry.');
      active.capturedSources = new Map(Array.isArray(imported)
        ? (imported as ExecutionFileIdentity[]).map(entry => [entry.file, entry.digest]) : []);
      const reused = outcome.ok ? outcome.reused : undefined;
      support.onEnvelope?.(envelope, { folder: active.folder, file: active.rel, cls: active.cls }, reused ? { reused: true } : undefined);
      measured?.({ ...active.identity }, active.last);
      if (envelope.error || !outcome.ok) return { status: 'failed', failure: executionFailure(envelope.error?.diagnostic),
        detail: executionDiagnosticDetail(envelope.error?.diagnostic, active.cls, envelope.error?.node_id)
          ?? 'Validation failed during the forward pass. Measured shapes are shown up to the failure; check tensorViz Output in VS Code.' };
      if (reused) return { status: 'passed', reused: { capturedAt: reused.capturedAt },
        detail: `${reusedResultLabel(reused.capturedAt)} Validation passed for the traced input. Other inputs and branches have not been checked.` };
      return { status: 'passed', detail: 'Validation passed for the traced input. Other inputs and branches have not been checked.' };
    } catch (error) {
      return current() ? refused(error instanceof ExecutionFileError ? error.message
        : 'Validation could not complete. Check tensorViz Output in VS Code and retry.') : cancelled();
    } finally {
      signal.removeEventListener('abort', abort);
      if (this.manualValidation === controller) this.manualValidation = undefined;
    }
  }

  /** Leave the class surface and cancel its pending projection/trace work. */
  clearTarget(): void {
    this.cancelValidation();
    this.supersedeProjection();
    // A queued callback reads this.active when it fires. Leaving either timer
    // alive would refresh or execute the NEW model for the OLD target's edit.
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = undefined;
    }
    if (this.saveTimer) {
      clearTimeout(this.saveTimer);
      this.saveTimer = undefined;
    }
    this.saveTraceScheduled = false;
    // A trace for the old class is CPU spent on a graph nobody is watching.
    this.options.trace?.worker.cancel();
    this.traceGeneration += 1;
    // New graph, new node ids: "the cursor entered a node" re-arms from scratch.
    this.focused = '';
    this.active = undefined;
    this.shapesChanged.fire();
  }

  /** Prepare source correspondence. The opening controller owns any subsequent execution. */
  setTarget(uri: vscode.Uri, identity: LocalBrowserTargetIdentity): void {
    const previous = this.active;
    const sameModel = previous?.folder === identity.workspaceFolder && previous.rel === identity.file && previous.cls === identity.cls;
    this.clearTarget();
    this.active = {
      uri,
      rel: identity.file,
      cls: identity.cls,
      folder: identity.workspaceFolder,
      identity,
      ...(sameModel ? { debugBaseline: previous.debugBaseline,
        last: previous.last ? { ...previous.last, greyed: true } : undefined,
        staleReason: 'Refreshing saved source. Previous measurements are out of date.' } : {}),
      dependencies: new Map(),
      capturedSources: new Map(),
      shapes: new Map(),
    };
    // The previous class's measured shapes are not this class's: drop them before
    // the first projection of the new target lands.
    this.shapesChanged.fire();
    this.initialProjection = this.refreshNow().catch((error) => {
      this.options.channel.appendLine(`[local-browser sync] refresh failed: ${String(error)}`);
    });
  }

  async readyTarget(target: LocalBrowserTargetIdentity, signal: AbortSignal): Promise<LocalBrowserTargetIdentity | undefined> {
    await this.initialProjection;
    if (signal.aborted) return undefined;
    const current = this.currentTarget();
    return current && current.workspaceFolder === target.workspaceFolder && current.file === target.file && current.cls === target.cls
      && current.sourceDigest === target.sourceDigest && current.documentVersion === target.documentVersion ? { ...current } : undefined;
  }

  /** The ordered state a reconnecting view is restored from, or undefined
   *  while no projection or failure exists yet — the caller falls back to the
   *  contract's minimal replay (empty snapshot, then the active target). */
  snapshotState(): LocalBrowserSnapshotState | undefined {
    const active = this.active;
    if (!active || (!active.last && !active.staleReason)) {
      return undefined;
    }
    if (active.staleReason) {
      return {
        activeTarget: active.identity,
        projection: active.last ? { ...active.last, greyed: true } : null,
        traced: active.last?.traced ?? false,
        greyed: true,
        syncState: 'stale',
        reason: active.staleReason,
        connectionState: 'connected',
      };
    }
    if (!active.last) return undefined;
    return {
      activeTarget: active.identity,
      projection: active.last,
      traced: active.last.traced,
      greyed: false,
      syncState: active.last.traced ? 'runtime-traced' : 'source-only',
      connectionState: 'connected',
    };
  }

  /** A view asked to reveal a node's line (extension-webpage step 6).
   *
   *  The page names a node plus the exact identity it is showing; only when
   *  that identity IS this window's current one is the answer trustworthy —
   *  anything older, or another target entirely, is refused as STALE_TARGET.
   *  The span that moves the cursor is this window's cached one, resolved from
   *  the node id alone: the browser never names a path or a line, so nothing
   *  from the page can aim an editor at an arbitrary file. The span's file —
   *  whichever dependency the projection attached it to, not always the root —
   *  is re-validated anyway: normalised workspace-relative spelling, membership
   *  of the projection's own dependency set, digest agreement. */
  async revealNode(
    target: LocalBrowserTargetIdentity,
    nodeId: string,
  ): Promise<LocalBrowserRevealResult> {
    const active = this.active;
    if (!active || !sameTargetIdentity(target, active.identity)) {
      return {
        ok: false,
        errorCode: 'STALE_TARGET',
        detail:
          'the page is not showing this window’s current target state; ' +
          'wait for the refresh and click again',
      };
    }
    if (!active.last) {
      return {
        ok: false,
        errorCode: 'INVALID_MESSAGE',
        detail: 'no projection exists for this target yet',
      };
    }
    const span = asNodeSpan(active.last.spanmap[nodeId]);
    if (!span) {
      return {
        ok: false,
        errorCode: 'INVALID_MESSAGE',
        detail: `no node named ${nodeId} in the graph the page is showing`,
      };
    }
    if (!isNormalizedRelativePath(span.file)) {
      return {
        ok: false,
        errorCode: 'INVALID_MESSAGE',
        detail: `the span for ${nodeId} is not a workspace-relative path; nothing will be opened`,
      };
    }
    const dependencyDigest = active.dependencies.get(span.file);
    if (!dependencyDigest) {
      return {
        ok: false,
        errorCode: 'INVALID_MESSAGE',
        detail: `${nodeId} points at ${span.file}, outside the open projection's dependency set`,
      };
    }
    // The span's file digest is the digest of the buffer the projection read.
    // Both sides of this check came down in ONE response, so a disagreement is a
    // contract violation, not a race.
    const spanDigest = (active.last.spanmap[nodeId] as { file_digest?: unknown }).file_digest;
    if (typeof spanDigest !== 'string' || spanDigest !== dependencyDigest) {
      return {
        ok: false,
        errorCode: 'INVALID_MESSAGE',
        detail: `the span for ${nodeId} belongs to an older projection of ${span.file}`,
      };
    }
    // Belt after the rule: even a well-formed span must resolve INSIDE the
    // target's folder before an editor may be pointed at it.
    if (relativeInside(active.folder, path.join(active.folder, span.file)) !== span.file) {
      return {
        ok: false,
        errorCode: 'INVALID_MESSAGE',
        detail: `${span.file} does not stay inside this window's workspace`,
      };
    }
    try {
      const document = await vscode.workspace.openTextDocument(
        vscode.Uri.file(path.join(active.folder, span.file)),
      );
      if (sha256Text(document.getText()) !== dependencyDigest) {
        this.schedule();
        return {
          ok: false,
          errorCode: 'STALE_TARGET',
          detail: `${span.file} changed after this projection; wait for the refresh and click again`,
        };
      }
      const editor = await vscode.window.showTextDocument(document, {
        viewColumn: vscode.ViewColumn.One,
        preserveFocus: false,
        preview: false,
      });
      // Spans are 1-based lines and 0-based columns (Python's own convention);
      // vscode.Position is 0-based in both — projection.ts's reading.
      const start = new vscode.Position(Math.max(0, span.start[0] - 1), Math.max(0, span.start[1]));
      // Suppress the echo FIRST: the selection event this move raises must not
      // bounce a focusNode straight back at the page that just asked. The key is
      // the whole node set at the destination line — the same computation
      // onCursor is about to redo.
      this.focused = nodesAtLine(active.last.spanmap, span.file, span.start[0]).join('\n');
      editor.selection = new vscode.Selection(start, start);
      editor.revealRange(
        new vscode.Range(start, start),
        vscode.TextEditorRevealType.InCenterIfOutsideViewport,
      );
    } catch (error) {
      this.options.channel.appendLine(
        `[local-browser sync] could not reveal ${nodeId}: ${String(error)}`,
      );
      return {
        ok: false,
        errorCode: 'INVALID_MESSAGE',
        detail: `could not open ${span.file}`,
      };
    }
    return { ok: true };
  }

  /** line → nodes: entering a node's span focuses it on the page — in ANY file
   *  the projection consulted, not only the root. Strokes inside one span stay
   *  quiet (entry-only, projection.ts's rule), and the spanmap read is the
   *  projection the page is showing — a buffer mid-edit still answers from the
   *  last graph that parsed, dimmed or not. One definition line can be several
   *  instances at once; the page is told all of them, never a guessed one. */
  private onCursor(event: vscode.TextEditorSelectionChangeEvent): void {
    if (!isHostFile(event.textEditor.document.uri)) return;
    const active = this.active;
    if (!active) {
      return;
    }
    const rel = relativeInside(active.folder, event.textEditor.document.uri.fsPath);
    if (!rel || (rel !== active.rel && !active.dependencies.has(rel))) {
      return;
    }
    const line = event.selections[0]?.active.line ?? 0;
    const matches = active.last ? nodesAtLine(active.last.spanmap, rel, line + 1) : [];
    const key = matches.join('\n');
    if (key === this.focused) {
      return;
    }
    this.focused = key;
    if (matches.length) {
      this.options.publish({
        type: 'focusNode',
        target: active.identity,
        nodeId: matches[0],
        nodeIds: matches,
      });
    }
  }

  private schedule(): void {
    if (this.active) this.options.sourceChanged?.(this.active.identity);
    this.cancelValidation();
    // Invalidate and abort now, not after the debounce: an old response must
    // not flash while the user is still typing the newer source.
    this.supersedeProjection();
    if (this.timer) {
      clearTimeout(this.timer);
    }
    this.timer = setTimeout(() => {
      this.timer = undefined;
      this.refreshNow().catch((error) => {
        this.options.channel.appendLine(`[local-browser sync] refresh failed: ${String(error)}`);
      });
    }, this.options.debounceMs ?? TYPING_DEBOUNCE_MS);
  }

  private scheduleTrace(): void {
    const support = this.options.trace;
    if (!support || !this.active || (this.manualValidation && !this.manualValidation.signal.aborted && !this.automaticValidation) || !support.settings().onSave) {
      return;
    }
    if (this.saveTimer) {
      clearTimeout(this.saveTimer);
    }
    // A newer save superseded the trace in flight: it is running the user's
    // code for a buffer that no longer exists.
    this.cancelValidation();
    support.worker.cancel();
    this.traceGeneration++;
    this.saveTraceScheduled = true;
    this.saveTimer = setTimeout(() => {
      this.saveTimer = undefined;
      this.saveTraceScheduled = false;
      this.tracesInFlight += 1;
      this.traceNow()
        .catch((error) => {
          this.options.channel.appendLine(`[local-browser sync] trace failed: ${String(error)}`);
        })
        .finally(() => {
          this.tracesInFlight -= 1;
        });
    }, this.options.saveCoalesceMs ?? SAVE_COALESCE_MS);
  }

  /** Opted-in saves use the same saved-source/profile gates and lifecycle as
   * manual validation. Only this path may remember a successfully measured input. */
  private async traceNow(): Promise<void> {
    const active = this.active;
    const support = this.options.trace;
    if (!active || !support) return;
    const generation = ++this.traceGeneration;
    const controller = new AbortController();
    this.automaticValidation = controller;
    const current = () => !controller.signal.aborted && generation === this.traceGeneration && this.active === active;
    const observe = this.options.summaries?.begin(randomUUID(), active.identity, 'automatic');
    let result: LocalBrowserValidationOutcome = { status: 'refused', detail: 'Automatic tracing could not start. Use Validate to retry.' };
    let measured: { target: LocalBrowserTargetIdentity; revision: number } | undefined;
    try {
      if (!await support.trusted()) {
        result = { status: 'refused', detail: 'Trust this workspace in VS Code before running validation.' };
        return;
      }
      // A save may outrun the debounced static refresh. Establish a projection of
      // the saved version before entering the shared validation gate.
      let refresh = !active.last || !!active.staleReason;
      const dependencies = new Map(active.dependencies).set(active.rel, active.identity.sourceDigest);
      for (const [file, digest] of dependencies) {
        const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(active.folder, file)));
        refresh ||= sha256Text(doc.getText()) !== digest;
      }
      if (!current()) return;
      if (this.timer) { clearTimeout(this.timer); this.timer = undefined; }
      if (refresh) await this.refresh(active, undefined, undefined, current);
      if (!current()) return;
      const profile = this.options.profiles ? await lookupProfile(this.options.profiles, active.folder,
        { file: active.rel, cls: active.cls }, controller.signal, false) : { ok: false as const, kind: 'none' as const };
      if (!current()) return;
      if (!profile.ok) {
        result = { status: profile.kind === 'cancelled' ? 'cancelled' : 'refused',
          detail: profile.kind === 'none' ? 'Set up this model with Open as Graph before enabling capture on save.' : profile.detail };
        return;
      }
      result = await this.validate({ ...active.identity }, controller.signal, undefined,
        phase => observe?.('running', validationPhaseDetails[phase]),
        (target, projection) => { measured = { target, revision: projection.revision }; },
        undefined,
        profile.ok ? profile.run.execution : undefined, undefined,
        profile.ok ? profile.run.timeoutMs : undefined, profile.ok ? profile.guard : undefined, observe?.attempt);
    } catch {
      result = { status: 'refused', detail: 'Automatic tracing could not complete. Use Open as Graph to inspect or repair setup.' };
    } finally {
      if (!current()) result = { status: 'cancelled', detail: 'Automatic tracing was cancelled or superseded by a source change.' };
      observe?.(result.status, result.detail, measured);
      if (this.automaticValidation === controller) this.automaticValidation = undefined;
    }
  }

  private async refreshNow(): Promise<void> {
    const active = this.active;
    if (active) {
      await this.refresh(active);
    }
  }

  /** Project the whole workspace — merging `trace` when given — and publish.
   *
   * One method for both tiers on purpose, as in projection.ts: everything
   * after the request is identical, and the difference between the AST tier
   * and the traced tier is exactly one field. Since the multi-file task the
   * endpoint is one too: `/api/ide/multifile/project` reads the complete
   * candidate source set — dirty buffers winning over disk — so an imported
   * class's definition reaches the projection; the response's dependency list
   * is what this loop then watches for edits. */
  private async refresh(
    active: BrowserActive,
    source?: string,
    trace?: WorkerEnvelope,
    mayPublish: () => boolean = () => true,
  ): Promise<string | undefined> {
    if (this.active !== active || !mayPublish()) return;
    const origin = this.options.origin();
    const token = this.options.token();
    this.supersedeProjection();
    const generation = this.generation;
    const controller = new AbortController();
    this.projectionAbort = controller;
    let timedOut = false;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    let document = source;
    let version = active.identity.documentVersion;
    try {
      const request = async () => {
        if (!origin || !token) throw new Error('Backend unavailable');
        if (document === undefined) {
          const live = await vscode.workspace.openTextDocument(active.uri);
          document = live.getText();
          version = live.version;
        }
        const body = await multifileProjectionBody({
          io: (this.options.sourceIO ?? vscodeSourceIO)(active.folder),
          file: active.rel,
          cls: active.cls,
          source: document,
          documentVersion: version,
          // What a previous trace measured, so the drawn Input node agrees with the
          // shapes on the cards below it. The backend ranks it below
          // `# tensorviz: input=`, so it is a hint and never an override.
          measuredInputShape: this.inputMemory(active, document).input_shape,
          trace,
        });
        controller.signal.throwIfAborted();
        if (generation !== this.generation || this.active !== active || !mayPublish()) return;
        const response = await this.post(`${origin}/api/ide/multifile/project`, token, body, controller.signal);
        if (response.cls !== active.cls || response.source_digest !== body.root.source_digest
          || response.traced !== (trace !== undefined)) {
          throw new InvalidProjectionResponse();
        }
        return response;
      };
      // Bound source collection and body decoding too. Abort alone is not a
      // deadline when an adapter ignores the signal; its late answer must lose.
      const deadline = new Promise<never>((_resolve, reject) => {
        timeout = setTimeout(() => {
          timedOut = true;
          controller.abort();
          reject(new Error('Projection timeout'));
        }, REQUEST_TIMEOUT_MS);
      });
      const response = await Promise.race([request(), deadline]);
      if (!response || controller.signal.aborted || generation !== this.generation || this.active !== active || !mayPublish()) {
        return;
      }
      // The graph the page is about to show is built from exactly these files:
      // this is the set edits are followed on until the next projection.
      const dependencies = new Map(
        response.dependencies.map((entry) => [entry.file, entry.digest]),
      );
      const traceError = (() => {
        const error = response.trace_error;
        if (!response.traced || !error || typeof error.message !== 'string' || !error.message) {
          return undefined;
        }
        // The server names the innermost WORKSPACE frame (which can now be a
        // dependency's line, not only the root's); display the
        // workspace-relative spelling, and leave the file out when no frame
        // was in the user's code at all.
        const absolute = typeof error.file === 'string' ? error.file.replace(/\\/g, '/') : undefined;
        const rel = absolute
          ? [...dependencies.keys()].find(
              (file) => absolute === file || absolute.endsWith(`/${file}`),
            )
          : undefined;
        return {
          // Exception text can contain source or secrets; only the mapped location
          // crosses to the page. The original envelope stays in the owning host.
          message: 'Forward pass failed.',
          ...(typeof error.node_id === 'string' ? { nodeId: error.node_id } : {}),
          ...(rel ? { file: rel } : {}),
          ...(typeof error.line === 'number' ? { line: error.line } : {}),
        };
      })();
      const saved = this.withSavedPositions(active, response.graph_id, response.document);
      const payload: ProjectionPayload = {
        graphId: response.graph_id,
        revision: response.revision,
        cls: response.cls,
        file: active.rel,
        sourceDigest: response.source_digest,
        traced: response.traced,
        ...(traceError ? { traceError } : {}),
        greyed: false,
        document: saved.document,
        viewRestored: saved.restored,
        spanmap: response.spanmap,
        projectionDigest: response.projection_digest,
        dependencies: response.dependencies,
        diagnostics: response.diagnostics,
        truncated: response.truncated,
      };
      const shapes = shapeHintsByFile(response);
      active.dependencies = dependencies;
      active.last = payload;
      active.shapes = shapes;
      this.shapesChanged.fire();
      active.staleReason = undefined;
      active.identity = {
        ...active.identity,
        documentVersion: version,
        sourceDigest: response.source_digest,
        backendRevision: response.revision,
      };
      this.options.summaries?.ledger.projected(active.identity, trace);
      const captureId = (payload.document as { semantic?: { metadata?: { capture?: { id?: string } } } })?.semantic?.metadata?.capture?.id;
      const retained = captureId ? this.options.summaries?.ledger.capture(captureId) : undefined;
      const capture = retained?.record.format === 'semantic' && !retained.record.unavailable
        ? (retained.envelope?.result as { diagnostics?: { semantic_capture?: CaptureSnapshot } })?.diagnostics?.semantic_capture : undefined;
      if (active.debugBaseline) payload.debug = graphDebugContext(active.debugBaseline, payload, capture);
      if (capture?.status === 'complete' && !capture.invocations.some(call => call.status === 'failed')) {
        const { debug, ...baseline } = payload;
        active.debugBaseline = { projection: baseline, capture };
      }
      this.options.publish({
        type: 'projection',
        target: active.identity,
        projection: payload,
        traced: payload.traced,
        greyed: false,
      });
      const state: LocalBrowserSyncStateMessage['state'] = payload.traced
        ? 'runtime-traced'
        : 'source-only';
      this.options.publish({ type: 'syncState', target: active.identity, state });
    } catch (error) {
      if (generation !== this.generation || this.active !== active || !mayPublish()) {
        return;
      }
      const reason = `${trace ? 'The latest trace could not be merged. ' : ''}${refreshFailureReason(error, active.rel, timedOut)}`;
      this.options.channel.appendLine(`[local-browser sync] ${reason}`);
      this.publishStale(active, reason);
      return reason;
    } finally {
      clearTimeout(timeout);
      if (this.projectionAbort === controller) {
        this.projectionAbort = undefined;
      }
    }
  }

  private publishStale(active: BrowserActive, reason: string): void {
    active.staleReason = reason;
    active.shapes.clear();
    this.shapesChanged.fire();
    if (active.last) this.options.publish({ type: 'staleState', target: active.identity,
      lastValidProjection: { ...active.last, greyed: true }, reason });
    this.options.publish({ type: 'syncState', target: active.identity, state: 'stale', reason });
  }

  /** `<folder>/.tensorviz/views/<rel>#<Class>.view.json` — the same spelling
   *  used by previous releases for the same (file, class) pair. */
  private sidecarPath(active: BrowserActive): string {
    return path.join(active.folder, VIEWS_DIR, `${active.rel}#${active.cls}.view.json`);
  }

  /** What this class is fed, as the sidecar remembers it — and nothing at all when
   *  `source` declares its own input, because the `.py` is the truth
   *  (`inputMemory.ts` owns that rule). */
  private inputMemory(active: BrowserActive, source: string): InputMemory {
    return inputMemoryFor(this.sidecarPath(active), source);
  }

  /** The layout the user arranged, when one was saved — read-only, and only
   *  ids the projection still has (mergeSavedPositions owns that rule). */
  private withSavedPositions(
    active: BrowserActive,
    graphId: string,
    document: unknown,
  ): { document: unknown; restored: boolean } {
    const file = this.sidecarPath(active);
    let saved: { graph_id?: string; node_positions?: Record<string, unknown> };
    try {
      saved = JSON.parse(fs.readFileSync(file, 'utf8'));
    } catch {
      return { document, restored: false };
    }
    return mergeSavedPositions(document, saved, graphId);
  }

  private async post(
    url: string,
    token: string,
    body: unknown,
    signal: AbortSignal,
  ): Promise<ProjectResponse> {
    const response = await (this.options.fetchImpl ?? globalThis.fetch)(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', [LOCAL_TOKEN_HEADER]: token },
      body: JSON.stringify(body),
      signal,
    });
    if (!response.ok) {
      throw new HttpError(response.status, await response.json().catch(() => undefined));
    }
    const value: unknown = await response.json().catch(() => { throw new InvalidProjectionResponse(); });
    if (!isProjectResponse(value)) throw new InvalidProjectionResponse();
    return value;
  }

  dispose(): void {
    this.cancelValidation();
    this.supersedeProjection();
    if (this.timer) {
      clearTimeout(this.timer);
    }
    if (this.saveTimer) {
      clearTimeout(this.saveTimer);
      this.saveTraceScheduled = false;
    }
    this.options.trace?.worker.cancel();
    this.shapesChanged.dispose();
    this.disposables.forEach((item) => item.dispose());
  }

  private supersedeProjection(): void {
    this.generation += 1;
    this.projectionAbort?.abort();
    this.projectionAbort = undefined;
  }
}
