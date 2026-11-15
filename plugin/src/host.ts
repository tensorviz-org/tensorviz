import { randomUUID } from 'node:crypto';
import { setTimeout as delay } from 'node:timers/promises';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { GraphOpeningController, TraceWorkerCore, lookupProfile, multifileProjectionBody, readHostSnapshot, disposeSnapshotHelpers } from '@tensorviz/graph-host';
import { GraphOpeningError, type EnvironmentRecoveryAction } from '@tensorviz/graph-host/graphOpening';
import { inspectModelEnvironment } from '@tensorviz/graph-host/modelEnvironment';
import { executionFailure } from '@tensorviz/graph-host/executionFailure';
import { McpGraphService, GraphToolError, type McpModel } from '@tensorviz/graph-host/mcpGraph';
import { McpOpenGraph, type OpenGraphArgs } from '@tensorviz/graph-host/mcpOpenGraph';
import { McpGraphView } from '@tensorviz/graph-host/mcpGraphView';
import { McpViewRegistry } from '@tensorviz/graph-host/mcpViews';
import type { AttemptHandle } from '@tensorviz/graph-host/attemptLedger';
import { McpValidationRuns } from '@tensorviz/graph-host/mcpValidation';
import { ValidationSummaryStore } from '@tensorviz/graph-host/validationSummary';
import { resolveExecutionPlan, resolveRetainedPlan, type ExecutionSnapshot } from '@tensorviz/graph-host/executionPlanCore';
import { sameExecutionGuards } from '@tensorviz/graph-host/executionGuard';
import type { ExecutionSpec } from '@tensorviz/graph-host/executionSpec';
import type { TraceOutcome } from '@tensorviz/graph-host/traceWorkerCore';
import { CaptureInspector } from '@tensorviz/graph-host/captureInspection';
import { graphDebugContext, type GraphDebugBaseline } from '@tensorviz/graph-host/graphDebug';
import { sameTargetIdentity } from '@tensorviz/graph-host/graphIdentity';
import { type GraphEnvironmentRecovery, type GraphOpeningState, type LocalBrowserTargetIdentity,
  type ProjectionPayload, type LocalBrowserSnapshotState, type LocalBrowserValidationSummary,
  type LocalBrowserCaptureExportRequestMessage, type LocalBrowserGraphPublishRequestMessage,
  type LocalBrowserPublishedGraph } from '@tensorviz/contracts/protocol';
import type { LocalBrowserHostTransport } from '@tensorviz/graph-host/localBrowserHostTransport';
import { PortableExportError } from '@tensorviz/graph-host/portableCapture';
import { WebsiteError } from '@tensorviz/graph-host/websiteAccount';
import type { ValidationPreparation } from '@tensorviz/graph-host/validationController';
import type { ProfileLookup } from '@tensorviz/graph-host/profileValidation';
import type { MultifileProjectionBody } from '@tensorviz/graph-host/workspaceSourceCore';
import { ApprovalGate } from './approval';
import { Backend } from './backend';
import { Environments, type EnvironmentSelection } from './environment';
import { digest, HostError, savedFile, sourceIO, workspaceRoot } from './workspace';
import { StandaloneSetup } from './setup';
import { ProfileChoices } from './profileChoices';
import { installStandaloneDependencies } from './dependencies';
import { inspectEnvironmentCandidates, runEnvironmentHelper } from '@tensorviz/graph-host/projectEnvironmentCore';
import { readSetup, sameSetup } from './setupFiles';
import { ManagedRuntime } from './runtime';
import { collectSourceSet, isNormalizedPythonPath } from '@tensorviz/graph-host/workspaceSourceCore';
import { StandaloneReview } from './review';
import { WebsitePublisher } from './website';

interface Selection {
  root: string;
  file: string;
  cls: string;
  environment?: EnvironmentSelection;
  managedRequested?: boolean;
  profile?: string;
}
interface ProjectionResponse {
  graph_id: string; cls: string; source_digest: string; traced: boolean;
  document: unknown; spanmap: Record<string, unknown>; projection_digest: string;
  dependencies: NonNullable<ProjectionPayload['dependencies']>;
  diagnostics: NonNullable<ProjectionPayload['diagnostics']>; truncated: boolean;
  trace_error?: { message: string; node_id?: string; frames?: { file: string; line: number }[] };
}

interface CaptureInvocation {
  execution: ExecutionSpec;
  timeoutMs: number;
  profile?: Extract<ProfileLookup, { ok: true }>;
  current(signal: AbortSignal): Promise<boolean>;
  facts: Record<string, unknown>;
  resolved?(snapshot: Pick<ExecutionSnapshot, 'plan' | 'applicabilityDigest'>): void;
  captured?(outcome: TraceOutcome): void;
  measured?(target: LocalBrowserTargetIdentity, projection: ProjectionPayload): void;
}

/** Saved-file adapter for the shared opening/capture/evidence state machines. */
export class StandaloneHost {
  readonly id = randomUUID();
  readonly graph: McpGraphService;
  readonly views: McpViewRegistry;
  readonly opening: GraphOpeningController;
  readonly openGraph: McpOpenGraph;
  readonly graphView: McpGraphView;
  readonly environments: Environments;
  readonly setup: StandaloneSetup;
  readonly runtime: ManagedRuntime;
  readonly runs: McpValidationRuns;
  readonly review: StandaloneReview;
  readonly summaries = new ValidationSummaryStore(summary => {
    this.validation = summary;
    this.backend.transport?.publish({ type: 'validationState', ...summary });
  });
  readonly ledger = this.summaries.ledger;
  readonly captures = new CaptureInspector(this.ledger);
  private backend: Backend;
  private servicePython?: string;
  private readonly workerCache = new Set<string>();
  private readonly worker: TraceWorkerCore;
  private selected?: Selection;
  private model?: McpModel;
  private target?: LocalBrowserTargetIdentity;
  private authorized = false;
  private generation = 0;
  private revision = 0;
  private readonly lifetime = new AbortController();
  private readonly inspectedEnvironments = new Map<string, { interpreter: string; fingerprint: string }>();
  private discovering = false;
  private watcher?: ReturnType<typeof setInterval>;
  private checking?: Promise<void>;
  private executionCurrent?: (signal: AbortSignal) => Promise<boolean>;
  private executionSetupDigest?: string | null;
  private lastError?: HostError;
  private validation?: LocalBrowserValidationSummary;
  private debugBaseline?: GraphDebugBaseline;
  private readonly profiles: ProfileChoices;
  private profileChoices: string[] = [];
  /** Publish to tensorviz.dev, asked for by a click in the view; never by a tool. */
  readonly website: WebsitePublisher;
  private publishing?: { requestId: string; controller: AbortController };
  private captureExport?: { requestId: string; controller: AbortController };

  constructor(readonly directory: string, readonly browser: string, readonly approvals: ApprovalGate, storage?: string) {
    this.environments = new Environments(directory, storage);
    this.website = new WebsitePublisher({ storage: this.environments.storage });
    this.profiles = new ProfileChoices(this.environments.storage);
    this.runtime = new ManagedRuntime(directory, this.environments.storage, approvals);
    this.backend = new Backend(directory, browser);
    this.worker = new TraceWorkerCore(directory, { appendLine: () => {} }, {
      parentPipe: true,
      trusted: () => this.authorized && !this.lifetime.signal.aborted,
      interpreter: async () => this.selected?.environment?.interpreter,
      documents: () => ({}),
    });
    this.setup = new StandaloneSetup({ current: target => this.current(target), sources: target => this.sources(target),
      post: (route, body, signal) => this.backend.request(route, body, signal), interpreter: () => this.capturePython(),
      approvals, remember: profile => this.rememberProfile(profile),
      prepare: async (request, signal) => {
        if (this.runs.running) throw new HostError('BUSY', 'Wait for validation or cancel its run before preparing new inputs.');
        signal.throwIfAborted(); this.authorized = true;
        const abort = () => this.worker.cancel();
        signal.addEventListener('abort', abort, { once: true });
        try { return await this.worker.run(request); }
        finally { this.authorized = false; signal.removeEventListener('abort', abort); }
      },
    });
    this.views = new McpViewRegistry(() => this.model, message => this.backend.transport?.publish(message));
    this.graph = new McpGraphService(this.id, () => this.model, () => this.views.list(), undefined, () => this.checkFreshness(true));
    this.review = new StandaloneReview({
      prepare: async (root, signal) => {
        const environment = this.inspectedEnvironments.get(root);
        if (!environment || environment.fingerprint !== await this.environments.fingerprint(environment.interpreter, this.servicePython)) {
          throw new HostError('ENVIRONMENT_REQUIRED', 'Inspect a compatible interpreter for this exact workspace with get_execution_environment before source review. No model capture is needed.');
        }
        await this.backend.start(await this.service(root, signal), signal);
      },
      post: (route, body, signal) => this.backend.request(route, body, signal),
      connect: signal => this.connectBrowser(signal), view: signal => this.backend.viewLink(signal), views: this.views,
      withBrowserAccess: result => this.backend.withBrowserAccess(result),
      publish: message => this.backend.transport?.publish(message), target: () => this.model?.target,
      busy: () => this.runs.running || ['resolving', 'saving', 'capturing'].includes(this.opening.state?.phase ?? ''),
      signal: this.lifetime.signal,
    });
    this.runs = new McpValidationRuns(() => this.model, async (target, signal, _inputs, progress, measured, captured, execution, job, attempt) => {
      if (job) return { status: 'refused', detail: 'Training diagnostics are unavailable in this host.' };
      if (this.review.busy) return { status: 'refused', detail: 'Wait for source review preparation before validation.' };
      if (['resolving', 'saving', 'capturing'].includes(this.opening.state?.phase ?? '')) {
        return { status: 'refused', detail: 'Graph opening is still running. Wait for it to finish or cancel it first.' };
      }
      const selected = this.selected;
      if (!selected?.environment || !this.current(target)) return { status: 'refused', detail: 'Open the exact saved model and choose Python before validation.' };
      this.review.close();
      const request = { workspace: selected.root, file: path.join(selected.root, target.file), cls: target.cls,
        execution, documents: {}, timeoutMs: 180_000, generation: this.generation, attemptId: attempt?.snapshot().id };
      const resolve = (signal: AbortSignal) => resolveExecutionPlan(selected.environment!.interpreter, this.directory, request, signal);
      const frozen = await resolve(signal);
      const current = async (signal: AbortSignal) => {
        if (signal.aborted || !this.current(target)) return false;
        const now = await resolve(signal);
        return frozen.applicabilityDigest === now.applicabilityDigest && sameExecutionGuards(frozen.guards, now.guards);
      };
      const result = await this.executeCapture(target, {
        execution, timeoutMs: request.timeoutMs, current,
        facts: { execution: frozen.plan, applicabilityDigest: frozen.applicabilityDigest },
        resolved: snapshot => {
          if (snapshot.applicabilityDigest !== frozen.applicabilityDigest) throw new HostError('SOURCE_CHANGED', 'The approved invocation changed before execution.');
        }, measured, captured,
      }, signal, detail => progress(detail.startsWith('Capturing') ? 'tracing'
        : detail.startsWith('Updating') ? 'projecting' : 'checking'), attempt);
      return result.outcome;
    }, message => this.backend.transport?.publish(message), this.summaries);
    this.opening = new GraphOpeningController({
      current: target => this.current(target),
      environment: (target, signal, progress) => {
        progress('Checking Python environments…');
        return this.setupOperation(() => this.checkEnvironment(target, signal));
      },
      recoverEnvironment: (target, environment, action, signal) => this.setupOperation(() => this.recoverEnvironment(target, environment, action, signal)),
      hasConfig: async root => { try { await fs.stat(path.join(root, 'tensorviz.toml')); return true; } catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false; throw error; } },
      lookup: (target, signal) => this.setupOperation(() => this.profile(target, signal)),
      discover: (target, signal, draft) => this.setupOperation(() => this.setup.discover(target, signal, draft)),
      save: (target, draft, signal) => this.setupOperation(() => this.setup.save(target, draft, signal)),
      inferInputs: (target, draft, signal) => this.setupOperation(() => this.setup.inferInputs(target, draft, signal)),
      capture: (target, profile, signal, progress) => this.capture(target, profile, signal, progress),
      recover: async () => undefined,
      parent: async () => { throw new GraphOpeningError('Open the parent with its explicit file and class.'); },
      publish: state => this.publishOpening(state),
    });
    this.openGraph = new McpOpenGraph({
      open: async (args, signal) => {
        try { return await this.open(args, signal); }
        catch (error) {
          if (error instanceof HostError && !signal.aborted) this.lastError = error;
          throw error instanceof HostError ? new GraphToolError(error.code, error.message) : error;
        }
      },
      state: () => this.opening.state,
      action: (state, action) => this.opening.action(state.id, randomUUID(), state.target, action),
      model: () => this.graph.list(),
      permission: async signal => {
        this.lastError = undefined;
        try { await this.approvals.reconsider(signal); }
        catch (error) {
          if (error instanceof HostError) { if (!signal.aborted) this.lastError = error; throw new GraphToolError(error.code, error.message); }
          throw error;
        }
      },
      dependencies: async (state, signal) => {
        const selected = this.selected!;
        if (!selected.environment || !this.servicePython) return {};
        const result = await runEnvironmentHelper(this.servicePython, this.directory,
          ['--target-python', this.capturePython(), 'inspect-model', selected.root, path.join(selected.root, state.target.file)], () => signal.aborted, true, true);
        return result as unknown as Record<string, unknown>;
      },
      install: (state, packages, signal) => this.setupOperation(() => this.installDependencies(state, packages, signal)),
    });
    this.graphView = new McpGraphView(this.graph, () => ({
      mcpViews: this.views,
      browserSessionCurrent: id => this.backend.sessionCurrent(id),
      prepareBrowserView: async (_target, signal) => {
        await this.connectBrowser(signal);
        return this.backend.viewLink(signal);
      },
    }));
  }

  private current(target: LocalBrowserTargetIdentity): boolean {
    const selected = this.selected;
    return !this.lifetime.signal.aborted && !!selected && selected.root === target.workspaceFolder
      && selected.file === target.file && selected.cls === target.cls
      && this.target?.sourceDigest === target.sourceDigest;
  }

  private async setupOperation<T>(action: () => Promise<T>): Promise<T> {
    this.lastError = undefined;
    try { return await action(); }
    catch (error) { if (error instanceof HostError) this.lastError = error; throw error; }
  }

  private async service(root: string, signal: AbortSignal): Promise<string> {
    return this.servicePython = await this.runtime.provisionService(root, signal);
  }

  private capturePython(): string {
    if (!this.selected?.environment) throw new HostError('ENVIRONMENT_REQUIRED', 'Choose Python for this model before preparing or executing it.');
    return this.selected.environment.interpreter;
  }

  private rememberInspection(root: string, interpreter: string, fingerprint: string): void {
    this.inspectedEnvironments.delete(root);
    if (this.inspectedEnvironments.size >= 8) this.inspectedEnvironments.delete(this.inspectedEnvironments.keys().next().value!);
    this.inspectedEnvironments.set(root, { interpreter, fingerprint });
  }

  private async checkEnvironment(target: LocalBrowserTargetIdentity, signal: AbortSignal, evidence: {
    refresh?: boolean; missingModules?: string[]; importFailure?: boolean;
  } = {}): Promise<GraphEnvironmentRecovery> {
    const selected = this.selected!;
    await this.sources(target);
    await this.approvals.approve({ operation: 'inspect_environment', workspaceRoot: selected.root,
      interpreter: selected.environment?.interpreter ?? 'No model Python selected',
      detail: 'Check bounded, known Python environments using isolated metadata probes, then inspect the chosen model environment and private capture worker. Candidate discovery does not import this model or install project packages.',
      facts: { file: selected.file, cls: selected.cls, sourceDigest: target.sourceDigest } }, signal);
    const managerPython = await this.service(selected.root, signal);
    await this.backend.start(managerPython, signal);
    await this.connectBrowser(signal);
    await this.preview(target, signal);
    if (selected.managedRequested) {
      selected.environment = { interpreter: await this.runtime.provision(selected.root, signal), source: 'managed' };
      selected.managedRequested = false;
    }
    const managed = await this.runtime.readyCapture(selected.root);
    const collect = async () => inspectEnvironmentCandidates([
      ...(selected.environment ? [{ interpreter: selected.environment.interpreter, source: 'Saved for this folder', selected: true as const,
        ...(selected.environment.source === 'managed' ? { managed: true as const } : {}) }] : []),
      ...(await this.environments.candidates(selected.root)).map(interpreter => ({ interpreter,
        source: interpreter.startsWith(selected.root + path.sep) ? 'Model’s project folder' : 'Known Python environment' })),
      ...(managed ? [{ interpreter: managed, source: 'tensorViz-managed fallback', managed: true as const }] : []),
    ], { backendDir: this.directory, managerPython, signal });
    if (!selected.environment || evidence.refresh) {
      const candidates = await collect(), chosen = candidates.find(candidate => candidate.selected);
      return { reason: chosen?.status === 'unavailable' ? 'unavailable' : 'selection_required', candidates,
        ...(chosen ? { selectedId: chosen.id } : {}), detail: chosen?.status === 'unavailable'
          ? 'The saved Python environment is no longer available. Choose an existing environment.' : 'Choose Python for this model. Detected environments are suggestions; no model was imported.' };
    }
    const [candidate] = await inspectEnvironmentCandidates([{ interpreter: selected.environment.interpreter,
      source: selected.environment.source === 'managed' ? 'tensorViz-managed fallback' : 'Selected for this folder', selected: true,
      ...(selected.environment.source === 'managed' ? { managed: true } : {}) }], { backendDir: this.directory, managerPython, signal });
    const { compatibilityPackages: _copy, ...checked } = await inspectModelEnvironment(candidate, {
      target, backendDir: this.directory, managerPython, signal, workerCache: this.workerCache, ...evidence });
    signal.throwIfAborted();
    if (!this.current(target) || this.selected !== selected || (await savedFile(selected.root, selected.file)).digest !== target.sourceDigest) {
      throw new HostError('SOURCE_CHANGED', 'The selected model changed. Open its current saved source again.');
    }
    if (candidate.status !== 'unavailable') await this.environments.remember(selected.root, candidate.interpreter, !!candidate.managed);
    if (checked.reason === 'ready') {
      if (candidate.fingerprint) this.rememberInspection(selected.root, candidate.interpreter, candidate.fingerprint);
      return checked;
    }
    const alternatives = await collect();
    return { ...checked, candidates: [candidate, ...alternatives.filter(item => item.id !== candidate.id && (!candidate.prefix || item.prefix !== candidate.prefix))].slice(0, 16) };
  }

  private async recoverEnvironment(target: LocalBrowserTargetIdentity, environment: GraphEnvironmentRecovery,
    action: EnvironmentRecoveryAction, signal: AbortSignal): Promise<boolean | GraphEnvironmentRecovery> {
    const selected = this.selected!;
    const check = async () => {
      signal.throwIfAborted();
      if (this.selected !== selected || !this.current(target) || (await savedFile(selected.root, selected.file)).digest !== target.sourceDigest) {
        throw new HostError('SOURCE_CHANGED', 'The model changed while environment setup was open. Open its current saved source again.');
      }
    };
    await check();
    if (action.kind === 'refresh_environments') return this.checkEnvironment(target, signal, { refresh: true });
    if (action.kind === 'select_environment') {
      const candidate = environment.candidates.find(item => item.id === action.candidateId);
      if (!candidate || candidate.status === 'unavailable') return false;
      const [fresh] = await inspectEnvironmentCandidates([candidate], { backendDir: this.directory, managerPython: this.servicePython, signal });
      await check();
      if (fresh.status === 'unavailable' || candidate.fingerprint && candidate.fingerprint !== fresh.fingerprint) {
        return { ...environment, reason: 'selection_required', detail: 'This environment changed while the chooser was open. Refresh the list and choose it again.' };
      }
      selected.environment = { interpreter: candidate.interpreter, source: candidate.managed ? 'managed' : 'explicit' };
      await this.environments.remember(selected.root, candidate.interpreter, !!candidate.managed);
      this.worker.cancel(); this.executionCurrent = undefined; this.executionSetupDigest = undefined;
      return true;
    }
    if (environment.reason !== 'missing_libraries' || !environment.requirements?.length) return false;
    try {
      await this.installDependencies(this.opening.state!, environment.requirements, signal);
      await check();
      return this.checkEnvironment(target, signal);
    } catch (error) {
      await check();
      const observed = await this.checkEnvironment(target, signal);
      return { ...observed, reason: observed.reason === 'ready' ? 'selection_required' : observed.reason,
        detail: `${error instanceof Error ? error.message : 'Library repair did not complete.'} The environment was checked again; existing packages were preserved.`.slice(0, 1024) };
    }
  }

  async inspectEnvironment(args: { workspaceRoot?: string; interpreter?: string }, signal: AbortSignal): Promise<Record<string, unknown>> {
    const root = await workspaceRoot(args.workspaceRoot ?? this.selected?.root);
    const selected = await this.environments.resolve(root, args.interpreter);
    if (!('interpreter' in selected)) return { workspaceRoot: root, ...selected, compatible: false, managedRuntime: this.runtime.availability() };
    const executableFingerprint = await this.environments.executableFingerprint(selected.interpreter);
    await this.approvals.approve({ operation: 'inspect_environment', workspaceRoot: root, interpreter: selected.interpreter,
      detail: 'Read the selected capture Python metadata in isolated mode. Model execution uses this Python; the tool service and installer use a separate TensorViz-owned environment. No model is imported and no project packages are installed by inspection.',
      facts: { interpreterFingerprint: executableFingerprint, backend: 'tensorviz-standalone', protocolVersion: 1 } }, signal);
    const managerPython = await this.service(root, signal);
    if (executableFingerprint !== await this.environments.executableFingerprint(selected.interpreter)) throw new HostError('ENVIRONMENT_CHANGED', 'The interpreter changed during authorization. Inspect the current environment.');
    const result = await this.environments.inspect(selected.interpreter, signal, managerPython) as unknown as Record<string, unknown>;
    const compatible = result.ok === true && Array.isArray(result.missing) && !result.missing.length
      && Array.isArray(result.incompatible) && !result.incompatible.length;
    if (compatible) {
      this.rememberInspection(root, selected.interpreter, String(result.fingerprint));
    }
    return { workspaceRoot: root, ...selected, ...result, compatible };
  }

  async discoverModels(root: string, offset: number, limit: number, signal: AbortSignal): Promise<Record<string, unknown>> {
    if (this.discovering) throw new HostError('BUSY', 'Wait for the current workspace discovery.');
    this.discovering = true;
    try { return await this.scanModels(root, offset, limit, signal); }
    finally { this.discovering = false; }
  }

  private async scanModels(root: string, offset: number, limit: number, signal: AbortSignal): Promise<Record<string, unknown>> {
    root = await workspaceRoot(root);
    const io = sourceIO(root), files = await io.list();
    const first = files.filter(isNormalizedPythonPath).sort()[0];
    if (!first) return { workspaceRoot: root, status: 'ready', models: [], count: 0, nextOffset: null,
      truncated: files.length > 0, notes: files.length ? ['Workspace enumeration reached its bound.'] : [] };
    const environment = this.inspectedEnvironments.get(root);
    if (!environment || environment.fingerprint !== await this.environments.fingerprint(environment.interpreter, this.servicePython)) {
      return { workspaceRoot: root, status: 'environment_required', models: [],
        detail: 'Static class discovery needs an inspected service environment. No project code was imported. Explicit file/class opening remains available.',
        nextAction: { tool: 'get_execution_environment', arguments: { workspaceRoot: root },
          instruction: 'Inspect a compatible interpreter, then repeat list_models with this workspaceRoot.' } };
    }
    const sources = await collectSourceSet(first, { ...io, list: async () => files });
    const service = new Backend(this.directory, this.browser);
    signal = AbortSignal.any([signal, this.lifetime.signal]);
    try {
      await service.start(await this.service(root, signal), signal);
      const indexed = await service.request<{ models: { file: string; cls: string; line: number; endLine: number }[];
        notes: { file: string; detail: string }[]; truncated: boolean }>('/api/ide/review/index',
        { protocol_version: 1, sources: sources.entries, truncated: sources.truncated }, signal);
      for (const entry of sources.entries) if ((await savedFile(root, entry.file)).digest !== entry.digest) {
        throw new HostError('SOURCE_CHANGED', 'Saved source changed during discovery. Retry after saving.');
      }
      if (JSON.stringify(files.sort()) !== JSON.stringify((await io.list()).sort())) {
        throw new HostError('SOURCE_CHANGED', 'Workspace files changed during discovery. Retry after saving.');
      }
      return { workspaceRoot: root, status: 'ready', executed: false,
        models: indexed.models.slice(offset, offset + limit).map(({ file, cls, line, endLine }) => ({ file, cls, line, endLine,
          nextAction: { tool: 'open_graph', arguments: { workspaceRoot: root, file, cls, interpreter: environment.interpreter } } })),
        count: indexed.models.length, nextOffset: offset + limit < indexed.models.length ? offset + limit : null,
        truncated: sources.truncated || indexed.truncated || indexed.notes.length > 200,
        notes: indexed.notes.slice(0, 200) };
    } finally { service.dispose(); }
  }

  private async open(args: OpenGraphArgs, signal: AbortSignal): Promise<GraphOpeningState> {
    if (this.runs.running || this.review.busy) throw new HostError('BUSY', 'Wait for validation or source review before opening another graph.');
    this.review.close();
    this.lastError = undefined;
    this.profileChoices = [];
    const root = await workspaceRoot(args.workspaceRoot);
    if (!args.file || !args.cls) throw new HostError('TARGET_REQUIRED', 'Supply the saved model file and class explicitly.');
    if (args.managedRuntime && args.interpreter) throw new HostError('ENVIRONMENT_AMBIGUOUS', 'Select managedRuntime or an explicit interpreter, not both.');
    const source = await savedFile(root, args.file);
    const resolution = args.managedRuntime ? undefined : await this.environments.resolve(root, args.interpreter);
    const environment = resolution && ('interpreter' in resolution ? resolution : resolution.selected);
    signal.throwIfAborted();
    this.worker.cancel(); this.authorized = false; this.executionCurrent = undefined; this.executionSetupDigest = undefined;
    if (this.watcher) clearInterval(this.watcher);
    this.generation++;
    this.selected = { root, file: source.file, cls: args.cls, environment, profile: args.profile, managedRequested: args.managedRuntime };
    if (this.model && (this.model.target.workspaceFolder !== root || this.model.target.file !== source.file || this.model.target.cls !== args.cls)) {
      this.model = undefined; this.debugBaseline = undefined; this.validation = undefined;
    }
    else if (this.model) { this.model.current = false; this.model.projection.greyed = true; }
    const target = { workspaceFolder: root, file: source.file, cls: args.cls,
      sourceDigest: source.digest, documentVersion: 0, backendRevision: ++this.revision };
    this.target = target;
    // Connect cancellation to the owned opening, including its later asynchronous capture.
    const generation = this.generation;
    const abort = () => { if (this.generation === generation) { this.opening.cancel(); this.worker.cancel(); } };
    signal.addEventListener('abort', abort, { once: true });
    this.watcher = setInterval(() => { void this.checkFreshness(); }, 1500);
    this.watcher.unref();
    try { await this.opening.open(target); }
    finally { signal.removeEventListener('abort', abort); }
    const permissionError = this.lastError as HostError | undefined;
    if (permissionError && ['APPROVAL_DECLINED', 'APPROVAL_CANCELLED'].includes(permissionError.code)) throw permissionError;
    return this.opening.state!;
  }

  private async rememberProfile(profile: string): Promise<void> {
    const selected = this.selected!;
    await this.profiles.remember(selected.root, selected.file, selected.cls, profile);
    selected.profile = profile;
  }

  private async profile(target: LocalBrowserTargetIdentity, signal: AbortSignal): Promise<ProfileLookup> {
    const selected = this.selected!;
    await this.backend.start(await this.service(selected.root, signal), signal);
    await this.preview(target, signal);
    const preferred = selected.profile ?? await this.profiles.read(selected.root, selected.file, selected.cls);
    let choices: string[] = [];
    const result = await lookupProfile({
      hasConfig: async root => { try { await fs.stat(path.join(root, 'tensorviz.toml')); return true; } catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false; throw error; } },
      choose: async names => { choices = names; return undefined; },
      preferred: async () => preferred,
      remember: async (_root, _target, name) => this.rememberProfile(name),
      snapshot: async (root, profile, nextSignal) => await readHostSnapshot(selected.environment!.interpreter, this.directory, {
        root, documents: {}, executable: selected.environment!.interpreter, environmentRevision: 0,
        session: this.id, workspaceId: digest(root), runId: randomUUID(), generation: this.generation, profile,
      }, nextSignal) as ValidationPreparation,
    }, selected.root, { file: target.file, cls: target.cls }, signal);
    if (choices.length && !signal.aborted) {
      this.profileChoices = choices;
      this.lastError = new HostError('PROFILE_REQUIRED', `Several saved profiles match this model. Choose an explicit profile: ${choices.join(', ')}.`);
      throw this.lastError;
    }
    return result;
  }

  private async preview(target: LocalBrowserTargetIdentity, signal: AbortSignal): Promise<void> {
    if (this.model) return; // Retain a previous measured graph until a new capture can replace it.
    try {
      const sources = await this.sources(target);
      const projected = await this.backend.request<ProjectionResponse>('/api/ide/multifile/project', sources, signal);
      if (!this.current(target) || JSON.stringify(sources) !== JSON.stringify(await this.sources(target))) {
        throw new HostError('SOURCE_CHANGED', 'Saved source changed while preparing the source view. Reopen the current model.');
      }
      const projection: ProjectionPayload = { graphId: projected.graph_id, revision: target.backendRevision,
        cls: target.cls, file: target.file, sourceDigest: projected.source_digest, traced: false,
        document: projected.document, spanmap: projected.spanmap, projectionDigest: projected.projection_digest,
        dependencies: projected.dependencies, diagnostics: projected.diagnostics, truncated: projected.truncated,
        greyed: false, viewRestored: false };
      this.model = { target, projection, current: true };
      this.publishSnapshot();
    } catch (error) {
      if (signal.aborted || error instanceof HostError && error.code === 'SOURCE_CHANGED') throw error;
      // Static lowering can be unsupported even when runtime capture succeeds.
      // Setup and execution still provide their own diagnostics and evidence.
    }
  }

  private async installDependencies(state: GraphOpeningState, packages: string[], signal: AbortSignal): Promise<void> {
    const selected = this.selected!;
    this.capturePython();
    const sources = JSON.stringify(await this.sources(state.target));
    const config = await readSetup(selected.root);
    const fingerprint = await this.environments.fingerprint(selected.environment!.interpreter, this.servicePython);
    const facts = await this.inspectEnvironment({ workspaceRoot: selected.root, interpreter: selected.environment!.interpreter }, signal);
    const prefix = (facts.environment as { prefix?: unknown } | undefined)?.prefix;
    if (typeof prefix !== 'string' || !path.isAbsolute(prefix)) throw new HostError('ENVIRONMENT_UNAVAILABLE', 'The exact environment prefix could not be verified; no packages were installed.');
    await installStandaloneDependencies({ python: selected.environment!.interpreter, managerPython: await this.service(selected.root, signal), prefix, workspaceRoot: selected.root,
      backendDir: this.directory, storage: path.join(this.environments.storage, 'temporary'), packages, signal, approvals: this.approvals,
      current: async () => this.selected === selected && this.current(state.target)
        && fingerprint === await this.environments.fingerprint(selected.environment!.interpreter, this.servicePython)
        && sources === JSON.stringify(await this.sources(state.target)) && sameSetup(config, await readSetup(selected.root)),
    });
  }

  private async sources(target: LocalBrowserTargetIdentity): Promise<MultifileProjectionBody> {
    const source = await savedFile(target.workspaceFolder, target.file);
    if (source.digest !== target.sourceDigest) throw new HostError('SOURCE_CHANGED', 'The saved model changed. Reopen it after saving.');
    return multifileProjectionBody({ io: sourceIO(target.workspaceFolder), file: target.file, cls: target.cls, source: source.text });
  }

  private async capture(target: LocalBrowserTargetIdentity, profile: Extract<ProfileLookup, { ok: true }>, signal: AbortSignal, progress: (detail: string) => void) {
    const result = await this.executeCapture(target, { execution: profile.run.execution, timeoutMs: profile.run.timeoutMs ?? 180_000,
      profile, current: profile.guard.current, facts: { profile: profile.guard.spec.profile, identity: profile.guard.spec.identity } }, signal, progress);
    if (!result.hasCapture && result.outcome.failure?.kind === 'environment' && !signal.aborted) {
      const environment = await this.checkEnvironment(result.target, signal, result.outcome.failure);
      if (environment.reason !== 'ready') throw new GraphOpeningError(result.outcome.detail, result.outcome.failure, environment);
    }
    return result;
  }

  private async executeCapture(target: LocalBrowserTargetIdentity, invocation: CaptureInvocation, signal: AbortSignal,
    progress: (detail: string) => void, retainedAttempt?: AttemptHandle) {
    const selected = this.selected!, generation = this.generation;
    const attemptId = retainedAttempt?.snapshot().id ?? randomUUID();
    const attempt = retainedAttempt ?? this.ledger.begin(attemptId, target, 'mcp', invocation.execution);
    let resolved: Pick<ExecutionSnapshot, 'plan' | 'applicabilityDigest'> | undefined;
    const abort = () => this.worker.cancel();
    signal.addEventListener('abort', abort, { once: true });
    try {
      const sources = await this.sources(target);
      const fingerprint = await this.environments.fingerprint(selected.environment!.interpreter, this.servicePython);
      progress('Preparing the saved invocation.');
      await this.approvals.approve({ operation: 'capture', workspaceRoot: selected.root, interpreter: selected.environment!.interpreter,
        detail: 'Import this project, construct the selected model and execute one forward pass. The worker is cancellable but is not an OS sandbox.',
        facts: { file: target.file, cls: target.cls, interpreterFingerprint: fingerprint,
          sourceDigest: target.sourceDigest, ...invocation.facts, timeoutMs: invocation.timeoutMs } }, signal);
      if (!await invocation.current(signal) || fingerprint !== await this.environments.fingerprint(selected.environment!.interpreter, this.servicePython)
        || generation !== this.generation) throw new HostError('SOURCE_CHANGED', 'The approved source, setup or interpreter changed. Reopen the current saved model.');
      this.authorized = true;
      progress('Capturing the approved forward invocation.');
      const result = await this.worker.run({ workspace: selected.root, file: path.join(selected.root, target.file), cls: target.cls,
        tier: 'real', execution: invocation.execution, profileSpec: invocation.profile?.guard.spec, expectedInterpreter: selected.environment!.interpreter,
        attemptId, onResolvedPlan: snapshot => { invocation.resolved?.(snapshot); resolved = snapshot; attempt.resolved(snapshot); },
        timeoutMs: invocation.timeoutMs, beforeLaunch: () => invocation.current(signal) });
      attempt.captured(result);
      invocation.captured?.(result);
      // Explicit MCP runs own their final verdict after projection and freshness
      // checks. Finishing their attempt here makes later progress contradict it.
      if (!retainedAttempt) attempt.finish(result.ok ? 'passed' : result.cancelled ? 'cancelled' : 'failed');
      if (signal.aborted || generation !== this.generation) return { target, hasCapture: false, outcome: { status: 'cancelled' as const, detail: 'Capture was cancelled.' } };
      if (!await invocation.current(signal) || result.executionCurrent === false) throw new HostError('SOURCE_CHANGED', 'Source or setup changed during capture. Refresh after saving.');
      if (result.envelope?.result) {
        progress('Updating measured shapes on the graph.');
        const projected = await this.backend.request<ProjectionResponse>('/api/ide/multifile/project', { ...sources, trace_result: result.envelope }, signal);
        if (!await invocation.current(signal) || !await result.recheckExecution?.(signal) || generation !== this.generation) {
          throw new HostError('SOURCE_CHANGED', 'Source or setup changed before publication. The late result was rejected.');
        }
        // Static expansion can omit imported constants and setup helpers. The
        // idle browser must also track the saved sources used by execution.
        const dependencies = new Map(projected.dependencies.map(entry => [entry.file, entry]));
        for (const file of resolved!.plan.source.files) if (file.role === 'source') {
          dependencies.set(file.path, { file: file.path, digest: file.digest });
        }
        const projection: ProjectionPayload = { graphId: projected.graph_id, revision: ++this.revision,
          cls: target.cls, file: target.file, sourceDigest: projected.source_digest, traced: projected.traced,
          document: projected.document, spanmap: projected.spanmap, projectionDigest: projected.projection_digest,
          dependencies: [...dependencies.values()], diagnostics: projected.diagnostics, truncated: projected.truncated,
          greyed: false, viewRestored: false,
          ...(projected.trace_error ? { traceError: { message: projected.trace_error.message, nodeId: projected.trace_error.node_id } } : {}) };
        target = { ...target, backendRevision: this.revision };
        this.model = { target, projection, current: true };
        this.ledger.projected(target, result.envelope);
        const record = this.ledger.capture(attempt.snapshot().captureIds[0])?.record;
        const capture = record?.format === 'semantic' && !record.unavailable && record.sourceId
          ? this.captures.retained(target, record.id, record.sourceId) : undefined;
        if (this.debugBaseline) projection.debug = graphDebugContext(this.debugBaseline, projection, capture);
        if (capture?.status === 'complete' && !capture.invocations.some(call => call.status === 'failed')) {
          const { debug, ...baseline } = projection;
          this.debugBaseline = { projection: baseline, capture };
        }
        // A completed capture's freshness must not depend on the worker's next
        // generation: a later cancelled run does not change these saved bytes.
        const capturedPlan = resolved!;
        this.executionSetupDigest = capturedPlan.plan.source.configDigest;
        this.executionCurrent = async signal => {
          if (!await invocation.current(signal)) return false;
          const now = await resolveRetainedPlan(selected.environment!.interpreter, this.directory, selected.root,
            capturedPlan.plan, signal, {});
          return now.applicabilityDigest === capturedPlan.applicabilityDigest;
        };
        invocation.measured?.(target, projection);
        this.publishSnapshot();
      }
      return { target, hasCapture: !!this.model?.projection.traced,
        outcome: result.ok ? { status: 'passed' as const, detail: invocation.profile
          ? `Captured one forward pass using saved profile ${invocation.profile.profile}.` : 'Captured one forward pass using the approved explicit invocation.' }
          : { status: result.cancelled ? 'cancelled' as const : 'failed' as const,
            detail: `${result.reason} ${result.detail}`.slice(0, 1024), failure: executionFailure(result.envelope?.error?.diagnostic) } };
    } catch (error) {
      if (!retainedAttempt) attempt.finish(signal.aborted ? 'cancelled' : 'refused');
      if (error instanceof HostError && !signal.aborted && generation === this.generation) this.lastError = error;
      if (error instanceof HostError && error.code === 'SOURCE_CHANGED' && !signal.aborted && generation === this.generation) {
        // Execution guards also cover imported values absent from the static
        // projection's dependencies. Keep that preview visibly stale too.
        if (this.model) { this.model.current = false; this.model.projection.greyed = true; }
        if (!retainedAttempt) this.opening.cancel(error.message, 'source_changed');
        this.publishSnapshot();
      }
      return { target, hasCapture: false, outcome: { status: signal.aborted ? 'cancelled' as const : retainedAttempt ? 'refused' as const : 'failed' as const,
        detail: error instanceof Error ? error.message : 'Capture could not complete.' } };
    } finally {
      this.authorized = false; signal.removeEventListener('abort', abort);
      if (signal.aborted && !retainedAttempt && generation === this.generation && this.opening.state) this.publishOpening(this.opening.state);
    }
  }

  /** SSH forwarding guidance and port notices for a result that carries a browser url. */
  withBrowserAccess<T extends Record<string, unknown>>(result: T): T {
    return this.backend.withBrowserAccess(result);
  }

  async checkFreshness(complete = false): Promise<void> {
    // A tool read must await an in-flight watcher and its own complete guard.
    // Skipping a concurrent check could publish current=true after a saved edit.
    await this.checking;
    if (!this.selected || !this.model?.current || this.lifetime.signal.aborted) return;
    const pending = this.inspectFreshness(complete);
    this.checking = pending;
    try { await pending; } finally { if (this.checking === pending) this.checking = undefined; }
  }

  private async inspectFreshness(complete: boolean): Promise<void> {
    const model = this.model;
    if (!model) return;
    try {
      const paths = model.projection.dependencies ?? [{ file: model.target.file, digest: model.target.sourceDigest }];
      let current = true;
      for (const entry of paths) if ((await savedFile(model.target.workspaceFolder, entry.file)).digest !== entry.digest) { current = false; break; }
      if (current && this.executionSetupDigest !== undefined) {
        const setup = await readSetup(model.target.workspaceFolder);
        current = (setup.identity === null ? null : digest(setup.text)) === this.executionSetupDigest;
      }
      if (current && complete && this.executionCurrent) current = await this.executionCurrent(this.lifetime.signal);
      if (!current && this.model === model) {
        model.current = false; model.projection.greyed = true;
        this.opening.invalidate(model.target, true);
        this.publishSnapshot();
      }
    } catch {
      if (this.model === model) { model.current = false; model.projection.greyed = true; this.opening.invalidate(model.target, true); this.publishSnapshot(); }
    }
  }

  private publishOpening(state: GraphOpeningState): void {
    this.backend.transport?.publish({ type: 'graphOpening', state });
    if (state.phase === 'resolving' && !state.hasCapture && this.model?.current && this.model.projection.traced) {
      // Change inputs retains earlier measurements while preparing a new
      // invocation. Its authorized save must not invalidate that new opening.
      this.model.current = false; this.model.projection.greyed = true;
      this.publishSnapshot();
    }
    const attempt = this.ledger.state(state.target).latestAttempt;
    if (attempt?.finishedAt != null && !state.cancellationReason && ['ready', 'failed', 'cancelled'].includes(state.phase)) {
      this.validation = { target: state.target, runId: attempt.id,
        status: attempt.status === 'succeeded' ? 'passed' : attempt.status === 'cancelled' ? 'cancelled'
          : attempt.status === 'needs_setup' ? 'refused' : 'failed',
        detail: state.detail.slice(0, 512), ...(state.hasCapture && ['succeeded', 'failed', 'timed_out'].includes(attempt.status)
          ? { projectionRevision: state.target.backendRevision } : {}),
        evidence: this.ledger.attribution(state.target) };
      this.backend.transport?.publish({ type: 'validationState', ...this.validation });
    }
  }

  private publishSnapshot(requested = false): void {
    const model = this.model;
    const state: LocalBrowserSnapshotState = model ? {
      activeTarget: model.target, projection: model.projection, traced: model.projection.traced, greyed: !model.current,
      connectionState: 'connected', ...(this.opening.state ? { opening: this.opening.state } : {}),
      ...(this.validation ? { validation: this.validation } : {}),
      ...(model.current ? { syncState: model.projection.traced ? 'runtime-traced' as const : 'source-only' as const }
        : { syncState: 'stale' as const, reason: 'Saved source or setup changed. Refresh through the coding agent to capture the new version.' }),
    } : { activeTarget: null, projection: null, traced: false, greyed: false, connectionState: 'connected', syncState: null,
      ...(this.opening.state ? { opening: this.opening.state } : {}) };
    const transport = this.backend.transport;
    if (requested) transport?.publish({ type: 'snapshot', state });
    else if (model) {
      transport?.publish({ type: 'activeTarget', target: model.target });
      transport?.publish({ type: 'projection', target: model.target, projection: model.projection,
        traced: model.projection.traced, greyed: !model.current });
      transport?.publish({ type: 'syncState', target: model.target, state: model.current ? model.projection.traced ? 'runtime-traced' : 'source-only' : 'stale',
        ...(!model.current ? { reason: 'Saved source or setup changed. Refresh to capture this version.' } : {}) });
    }
    this.review.replay();
  }

  private async connectBrowser(signal: AbortSignal): Promise<void> {
    if (this.backend.transport && this.backend.transport.status.phase !== 'ended') return;
    const transport = await this.backend.connect(signal);
    transport.onMessage(message => {
      this.views.handle(message);
      void this.review.handle(message).catch(error => this.review.reportError(error));
      if (message.type === 'snapshotRequest') this.publishSnapshot(true);
      if (message.type === 'captureInspectRequest') {
        let page = null, detail = 'Retained capture evidence.';
        try {
          if (!this.model || message.target.workspaceFolder !== this.model.target.workspaceFolder || message.target.file !== this.model.target.file
            || message.target.cls !== this.model.target.cls) throw new Error();
          page = this.captures.inspect(message.target, message.query);
        } catch { detail = 'This retained capture is unavailable. Refresh the graph and choose a current capture.'; }
        transport.publish({ type: 'captureInspectResult', target: message.target, requestId: message.requestId, page, detail });
      }
      if (message.type === 'captureExportRequest') void this.exportCapture(transport, message);
      if (message.type === 'captureExportCancel' && this.captureExport?.requestId === message.requestId) this.captureExport.controller.abort();
      if (message.type === 'graphPublishRequest') void this.publishGraph(transport, message);
      if (message.type === 'graphPublishCancel' && this.publishing?.requestId === message.requestId) this.publishing.controller.abort();
      if (!this.review.active && message.type === 'graphSetupRequest' && this.opening.state) {
        // Browser setup writes/capture still go through the same approval gate.
        const target = this.opening.state.target;
        if (message.action.kind === 'retry' && this.selected && message.target.workspaceFolder === target.workspaceFolder
          && message.target.file === target.file && message.target.cls === target.cls) {
          const selected = this.selected;
          try { this.openGraph.invoke(this.id, { workspaceRoot: selected.root, file: selected.file,
            cls: selected.cls, interpreter: selected.environment!.interpreter, profile: selected.profile }); } catch { /* The current opening reports busy. */ }
        } else if (sameTargetIdentity(message.target, target)) void this.opening.action(message.openingId, message.requestId, message.target, message.action);
      }
    });
    transport.onStatusChange(status => { if (status.phase === 'connected') this.publishSnapshot(); });
    this.publishSnapshot();
  }

  /** The displayed capture, detached for a graph file, as the editor's Export graph does. */
  private async exportCapture(transport: LocalBrowserHostTransport, request: LocalBrowserCaptureExportRequestMessage): Promise<void> {
    const reply = (capture: unknown | null, detail: string) => transport.publish({ type: 'captureExportResult',
      target: request.target, requestId: request.requestId, reference: request.reference, capture, detail });
    if (this.captureExport) { reply(null, 'Another capture export is being prepared. Try again when it finishes.'); return; }
    const pending = { requestId: request.requestId, controller: new AbortController() };
    this.captureExport = pending;
    const retained = () => {
      const model = this.model;
      if (!model || !sameTargetIdentity(model.target, request.target)
        || this.captures.list(model.target).displayedCaptureId !== request.reference.captureId) {
        throw new GraphToolError('STALE_TARGET', 'The displayed capture changed. Export the current graph again.');
      }
      return this.captures.retained(model.target, request.reference.captureId, request.reference.sourceId);
    };
    try {
      const portable = await this.backend.portable(retained(), AbortSignal.any([pending.controller.signal, this.lifetime.signal]));
      retained(); // Eviction, retargeting or a newer projection cannot race the reply.
      if (!pending.controller.signal.aborted) reply(portable, 'Portable capture prepared without executing the model.');
    } catch (error) {
      if (!pending.controller.signal.aborted) reply(null,
        error instanceof GraphToolError || error instanceof PortableExportError ? error.message : 'The retained capture could not be exported.');
    } finally { if (this.captureExport === pending) this.captureExport = undefined; }
  }

  /** Publish from the view: the graph file it would export, signed in and uploaded here. */
  private async publishGraph(transport: LocalBrowserHostTransport, request: LocalBrowserGraphPublishRequestMessage): Promise<void> {
    const { target, requestId } = request;
    const result = (status: 'published' | 'failed' | 'cancelled', detail: string, graph: LocalBrowserPublishedGraph | null = null) =>
      transport.publish({ type: 'graphPublishResult', target, requestId, status, detail: detail.slice(0, 512), graph });
    if (this.publishing) { result('failed', 'tensorViz is still publishing your last graph.'); return; }
    const pending = { requestId, controller: new AbortController() };
    this.publishing = pending;
    // A new broker epoch drops this request: stop waiting for a sign-in no view can finish.
    const epoch = transport.status.epoch;
    const offStatus = transport.onStatusChange(status => {
      if (status.phase !== 'connected' || status.epoch !== epoch) pending.controller.abort();
    });
    const signal = AbortSignal.any([pending.controller.signal, this.lifetime.signal]);
    try {
      const graph = await this.website.publish(request.file, {
        signingIn: (signInUrl, detail) => !signal.aborted
          && !!transport.publish({ type: 'graphPublishState', target, requestId, phase: 'signing_in', detail: detail.slice(0, 512), signInUrl }),
        uploading: detail => { transport.publish({ type: 'graphPublishState', target, requestId, phase: 'uploading', detail: detail.slice(0, 512) }); },
      }, signal);
      result('published', `${graph.title} is on tensorViz, visible only to you. Choose who can see it on its Share page.`, graph);
    } catch (error) {
      if (signal.aborted || error instanceof WebsiteError && error.code === 'cancelled') result('cancelled', 'Publishing was cancelled.');
      else result('failed', error instanceof WebsiteError ? error.message : 'Could not publish the graph to tensorViz.');
    } finally {
      offStatus();
      if (this.publishing === pending) this.publishing = undefined;
    }
  }

  async openingView(signal: AbortSignal) {
    if (!this.opening.state || !this.backend.ready) return undefined;
    await this.connectBrowser(signal);
    return this.backend.viewLink(signal);
  }

  approvalProgress(): Record<string, unknown> {
    return this.approvals.pending ? { approval: this.approvals.pending,
      nextAction: { kind: 'approval_required', instruction: this.approvals.pending.method === 'browser'
        ? 'Show the private confirmation URL to the user. Only the user may approve or decline. Poll after their decision; do not approve on their behalf.'
        : 'Wait for the user’s native approval decision, then poll this request.' } } : {};
  }

  async openAndWait(args: OpenGraphArgs, signal: AbortSignal, waitMs = 20_000): Promise<Record<string, unknown>> {
    await this.checkFreshness();
    signal.throwIfAborted();
    let opening = this.openGraph.invoke(this.id, args);
    const requestId = String(opening.requestId), deadline = Date.now() + waitMs;
    try {
      for (;;) {
        if (opening.status === 'cancelled') return opening;
        const failure = recordedToolError(opening.error);
        if (failure && ['OUTSIDE_WORKSPACE', 'WORKSPACE_REQUIRED', 'WORKSPACE_UNAVAILABLE', 'PYTHON_FILE_REQUIRED'].includes(failure.code)) {
          throw new HostError(failure.code, failure.message || 'That file is outside this workspace.');
        }
        const result = { ...opening, ...this.progress(opening.status) };
        if (result.approval || !['opening', 'resolving', 'saving', 'capturing'].includes(String(result.status))
          || Date.now() >= deadline) return result;
        await delay(Math.min(100, deadline - Date.now()), undefined, { signal });
        opening = this.openGraph.invoke(this.id, { requestId });
      }
    } catch (error) {
      if (signal.aborted) {
        try { this.openGraph.invoke(this.id, { requestId, action: 'cancel' }); }
        catch (cancelError) { if (!(cancelError instanceof GraphToolError) || cancelError.code !== 'UNKNOWN_REQUEST') throw cancelError; }
      }
      throw error;
    }
  }

  progress(status?: unknown): Record<string, unknown> {
    const environment = this.opening.state?.environment;
    return { ...(this.runtime.progress ? { runtimeProgress: this.runtime.progress } : {}), ...this.approvalProgress(),
      ...(this.opening.state?.environment?.reason === 'selection_required' ? { managedRuntime: this.runtime.availability() } : {}),
      ...(environment && this.selected && ['selection_required', 'unavailable', 'missing_libraries'].includes(environment.reason)
        && !environment.candidates.some(candidate => candidate.status !== 'unavailable' && candidate.torch) ? {
          managedRuntime: this.runtime.availability(), nextAction: { kind: 'select_environment', tool: 'open_graph',
            arguments: { workspaceRoot: this.selected.root, file: this.selected.file, cls: this.selected.cls,
              ...(this.selected.profile ? { profile: this.selected.profile } : {}) }, requiredArguments: ['interpreter or managedRuntime'],
            instruction: 'No detected environment has the model’s Torch runtime. Select an existing interpreter with the intended Torch/device build, or explicitly request managedRuntime: true for a private CPU fallback if CPU suits this model. Start this exact model with that choice; do not infer Python from your shell or repeatedly retry the unavailable choice.' },
        } : {}),
      ...(this.profileChoices.length && this.selected ? { status: 'blocked', continuationRequired: true,
        profiles: this.profileChoices, nextAction: { kind: 'select_profile', tool: 'open_graph',
          arguments: { workspaceRoot: this.selected.root, file: this.selected.file, cls: this.selected.cls,
            interpreter: this.selected.environment!.interpreter }, requiredArguments: ['profile'],
          instruction: 'Choose one of the existing profiles from repository evidence or ask the user, then start this exact model with profile. No setup was overwritten.' } } : {}),
      ...(this.lastError ? { error: { code: this.lastError.code, message: this.lastError.message },
        ...(['APPROVAL_DECLINED', 'APPROVAL_CANCELLED'].includes(this.lastError.code)
          ? { status: 'blocked', continuationRequired: false } : {}) } : {}) };
  }

  get workspace(): string | undefined { return this.selected?.root; }

  dispose(): void {
    this.lifetime.abort(); this.authorized = false;
    this.website.dispose();
    if (this.watcher) clearInterval(this.watcher);
    this.openGraph.cancelOwner(this.id); this.graphView.cancelOwner(this.id); this.views.cancelOwner(this.id);
    this.runs.clear();
    this.review.close();
    this.opening.clear(); this.worker.dispose(); this.approvals.dispose(); this.backend.dispose();
    disposeSnapshotHelpers();
  }
}

function recordedToolError(value: unknown): { code: string; message: string } | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const record = value as Record<string, unknown>;
  if (typeof record.code !== 'string') return undefined;
  return { code: record.code, message: typeof record.message === 'string' ? record.message : '' };
}
