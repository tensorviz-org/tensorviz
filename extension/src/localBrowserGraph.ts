import { ExecutionReview } from '@tensorviz/graph-host/executionReview';
import { executionResources } from './executionResources';
import { unsupportedRemote } from './remoteSupport';
import { CanvasAgent, type CanvasAgentOptions } from './canvasAgent';
import { isHostFile, matchesHostFile } from './workspaceFiles';
import { GraphReview, type GraphReviewSpec } from './graphReview';
import { SourceDrafts } from './sourceDraft';
import type { ReviewSession } from './reviewSnapshots';
import { McpViewRegistry } from '@tensorviz/graph-host/mcpViews';
import { McpValidationRuns } from '@tensorviz/graph-host/mcpValidation';
import { McpProposals } from './mcpProposals';
import type { TraceWorker } from './worker';
/** Window-owned source projection, validation, review and target authority.
 * Native Canvas and the authenticated external browser share this controller.
 * Selecting a class reads the live buffer; reconnecting replays current state.
 * Source navigation and writes resolve against host-owned identities and spans.
 * Browser credentials stay out of logs and query strings; native authentication
 * remains in CanvasBridge's existing loopback iframe transport.
 */

import { createHash, randomUUID } from 'node:crypto';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { CodeFocus } from './codeFocus';

import { LocalBrowserSync, LocalBrowserTraceSupport, sameTargetIdentity } from './localBrowserSync';
import { LocalBrowserSourceEdits } from './localBrowserSourceEdits';
import { BrowserSourceReviews } from './browserSourceReviews';
import { LocalBrowserValidation } from './localBrowserValidation';
import { ValidationSummaryStore, validationPhaseDetails } from '@tensorviz/graph-host/validationSummary';
import { GraphOpeningController, GraphOpeningError, type EnvironmentRecoveryAction } from '@tensorviz/graph-host/graphOpening';
import { GraphSetupHost } from './graphSetupHost';
import { modelProjectRoot } from './modelProjectRoot';
import { ValidationSetup } from './validationSetup';
import { modelConfigStarter } from './validationConfig';
import { CaptureInspector } from '@tensorviz/graph-host/captureInspection';
import { preparePortableCapture, PortableExportError } from './portableCapture';
import { lookupProfile, type ProfileLookupOptions } from '@tensorviz/graph-host/profileValidation';
import {
  LocalBrowserEmptySnapshotState,
  LocalBrowserMessage,
  LocalBrowserTargetIdentity,
  ProjectionPayload,
  LocalBrowserCaptureExportRequestMessage,
  GraphEnvironmentRecovery,
  isGraphSetupAction,
} from '@tensorviz/contracts/protocol';
import { ShapeHint } from './projection';
import { SourceIO, sha256Text } from './workspaceSources';
import { ProjectSync } from './projectSync';
import { indexWorkspaceFolder } from './projectIndex';
import {
  ProjectSourceIO,
  bodiesFor,
  collectProjectSources,
  vscodeProjectSourceIO,
} from './projectSources';
import { buildProjectDocument } from './projectDocument';
import { GraphToolError, type McpModel } from '@tensorviz/graph-host/mcpGraph';
import {
  LocalBrowserHostTransport,
  LocalBrowserHostPublish,
  LocalBrowserSessionMaterial,
  createLocalBrowserHostTransport,
  mintLocalBrowserSession,
  mintLocalBrowserViewLink,
  type LocalBrowserViewLink,
} from '@tensorviz/graph-host/localBrowserHostTransport';

/** Same spelling as middleware.py's LOCAL_TOKEN_HEADER and projection.ts's
 *  TOKEN_HEADER; declared here so this module stays free of their imports. */
const LOCAL_TOKEN_HEADER = 'X-tensorViz-Token';

type GraphOpenActor = 'human' | 'agent' | 'automatic';

/** The network read of the class list gets the same ceiling as the session mint. */
const BUFFER_CLASSES_TIMEOUT_MS = 10_000;

/** One `nn.Module` class in a buffer, as POST /api/ide/buffer-classes reports
 *  it: 1-based AST spans over the text that was POSTed, not over the disk file. */
export interface BufferClass {
  name: string;
  line: number;
  end_line: number;
  bases: string[];
}

/** The snapshot state a host answers with until the sync's first projection
 *  exists: the legal empty state of the frozen contract, nothing invented. */
export function emptyLocalBrowserSnapshotState(): LocalBrowserEmptySnapshotState {
  return {
    activeTarget: null,
    projection: null,
    traced: false,
    greyed: false,
    syncState: null,
    connectionState: 'connected',
  };
}

/** The running backend, read per call so a restart on a new port is picked up.
 *  The same "endpoint" indirection as projection.ts's BackendEndpoint, extended
 *  by the launch token the session mint proves host identity with. */
export interface LocalBrowserBackendEndpoint {
  origin(): string | undefined;
  launchToken(): string | undefined;
}

export interface LocalBrowserGraphOptions {
  agent?: Pick<CanvasAgentOptions, 'run' | 'clear' | 'inspect' | 'preferences' | 'remember' | 'invalidate'>;
  rememberOpen?(record: { workspaceFolder: string; file: string; cls: string }): void;
  channel: { appendLine(line: string): void };
  backend: LocalBrowserBackendEndpoint;
  /** The origin the page is served from: a dev server when one is configured,
   *  else undefined and the backend's own origin is used. Read per open so a
   *  setting change takes effect without a reload. */
  launchOrigin(): string | undefined;
  /** Resolve the client-visible origin before authentication; host sockets stay internal. */
  resolveLaunchOrigin?(origin: string): Promise<string>;
  /** Starts the backend when it is not running; resolves false when the user
   *  cancelled or startup failed (the start itself already said why). */
  ensureBackend(): Promise<boolean>;
  /** `vscode.env.remoteName`: undefined locally, ssh-remote for desktop SSH. */
  remoteName(): string | undefined;
  /** `vscode.env.openExternal`, or the Electron test seam. */
  openExternal(url: string): Promise<boolean>;
  mintSession?: typeof mintLocalBrowserSession;
  createTransport?: typeof createLocalBrowserHostTransport;
  fetchImpl?: typeof fetch;
  /** Present to make the page a runtime-traced one: the worker that runs the
   *  user's model, the consent gate, and the settings. Absent, the page is the
   *  source tier only, honestly labelled source-only. */
  trace?: LocalBrowserTraceSupport;
  /** Separate worker lifecycle: candidate runs cannot cancel or replace live runs. */
  runCandidate?: TraceWorker['runCandidate'];
  /** How a saved tensorviz.toml profile is found for the class being viewed. Absent —
   *  no interpreter selected — Validate refuses with the reason instead of guessing. */
  profiles?: ProfileLookupOptions;
  selectEnvironment?(target: LocalBrowserTargetIdentity): Promise<boolean>;
  prepareDependencies?(target: LocalBrowserTargetIdentity, signal: AbortSignal, missingModules?: string[], importFailure?: boolean): Promise<boolean>;
  environment?(target: LocalBrowserTargetIdentity, signal: AbortSignal, evidence?: { missingModules?: string[]; importFailure?: boolean }): Promise<GraphEnvironmentRecovery>;
  recoverEnvironment?(target: LocalBrowserTargetIdentity, environment: GraphEnvironmentRecovery,
    action: EnvironmentRecoveryAction, signal: AbortSignal, progress: (detail: string) => void): Promise<boolean | GraphEnvironmentRecovery>;
  /** How the multi-file projection's source set is enumerated/read, per target
   *  folder; defaults to the VS Code implementation. Injected by tests. */
  sourceIO?: (folder: string) => SourceIO;
  /** Injected for tests; the production path enumerates through VS Code. */
  projectSourceIO?: (folder: string) => ProjectSourceIO;
  debounceMs?: number;
  saveCoalesceMs?: number;
}

export class LocalBrowserGraph implements vscode.Disposable {
  private readonly validationSummaries = new ValidationSummaryStore(summary => {
    if (summary.status === 'passed' && summary.evidence?.origin === 'automatic'
      && summary.projectionRevision !== undefined && this.sync.currentTarget()?.backendRevision === summary.projectionRevision) {
      this.opening?.acceptAutomaticCapture(summary.target, summary.detail);
    }
    this.publish({ type: 'validationState', ...summary });
  });
  readonly captureInspector = new CaptureInspector(this.validationSummaries.ledger);
  private host:
    | { session: LocalBrowserSessionMaterial; transport: LocalBrowserHostTransport; origin: string; token: string }
    | undefined;
  /** The session mint+connect+open in flight, so two invocations that land while
   *  the first is still awaiting its mint do not each mint a session and open a
   *  tab. The folded call re-enters after it, sees the host, and retargets. */
  private starting: Promise<void> | undefined;

  private readonly sync: LocalBrowserSync;
  private graphReview?: GraphReview;
  private readonly sourceDrafts: SourceDrafts;
  /** The project half: experiments, focus, expansion. Null until a session opens.
   *  Kept beside `sync` rather than inside it because the two answer different
   *  questions — one about a class, one about the workspace — and conflating them
   *  is what made the class identity hard to extend in the first place. */
  private project: ProjectSync | undefined;
  private projectFolder?: string;
  private projectError?: string;
  private projectOperationGeneration = 0;

  private readonly sourceEdits: LocalBrowserSourceEdits;
  private readonly browserReviews: BrowserSourceReviews;
  private readonly canvasReviews: BrowserSourceReviews;
  private readonly validation: LocalBrowserValidation;
  readonly opening?: GraphOpeningController;
  readonly setupHost?: GraphSetupHost;
  private readonly openingSubscriptions: vscode.Disposable[] = [];
  private waitingForSetupSave = false;
  private readonly agent?: CanvasAgent;
  private readonly executionReview: ExecutionReview;
  private openingActor: GraphOpenActor = 'human';
  private openingSurface: 'browser' | 'canvas' = 'browser';
  private parentFocus?: { file: string; cls: string; attribute: string };
  private captureExport: { request: LocalBrowserCaptureExportRequestMessage; controller: AbortController; native: boolean } | undefined;
  readonly mcpViews: McpViewRegistry;
  private readonly codeFocus: CodeFocus;
  readonly mcpRuns: McpValidationRuns;
  readonly mcpProposals: McpProposals;
  get busy(): boolean { return this.sourceEdits.reviewing || this.mcpRuns.running || !!this.validation.activeRequestId || ['awaiting_execution', 'resolving', 'capturing'].includes(this.opening?.state?.phase ?? ''); }

  constructor(private readonly options: LocalBrowserGraphOptions) {
    this.executionReview = new ExecutionReview({
      current: target => this.openingSourceCurrent(target),
      resources: (interpreter, signal) => executionResources(interpreter, !!options.remoteName(), signal),
      publish: state => {
        const target = this.executionReview.target;
        if (target && this.opening && !this.opening.state) this.opening.sourceOnly(target);
        const opening = this.opening?.state;
        if (target && opening && target.workspaceFolder === opening.target.workspaceFolder && target.file === opening.target.file
          && target.cls === opening.target.cls && target.sourceDigest === opening.target.sourceDigest) this.opening?.execution(state);
        // A host reading that arrives after its run settled updates only the run this opening still shows.
        else if (!target && opening?.execution?.id === state.id) this.opening?.execution(state);
      },
    });
    this.sourceDrafts = new SourceDrafts({
      endpoint: () => { const origin = options.backend.origin(), token = options.backend.launchToken(); return origin && token ? { origin, token } : undefined; },
      publish: message => {
        if (message.type === 'graphReviewState' && message.state) this.executionReview.cancel();
        return this.publish(message.type === 'graphReviewState' && message.state === null && this.graphReview?.current
          ? { ...message, state: this.graphReview.current } : message);
      },
      changed: listener => vscode.workspace.onDidChangeTextDocument(listener), fetchImpl: options.fetchImpl,
    });
    this.sync = new LocalBrowserSync({
      channel: options.channel,
      origin: () => options.backend.origin(),
      token: () => options.backend.launchToken(),
      publish: (message) => {
        if (message.type === 'projection') {
          this.opening?.invalidate(message.target);
          if (message.traced && !message.greyed) this.opening?.acceptMeasuredProjection(message.target);
        }
        const published = this.publish(message);
        if (message.type === 'projection') this.codeFocus?.retry();
        return published ?? undefined;
      },
      // Source saves refresh the projection; computation needs an opening or a reviewed Run.
      trace: options.trace ? { ...options.trace, settings: () => ({ ...options.trace!.settings(), onSave: false }) } : undefined,
      fetchImpl: options.fetchImpl,
      sourceIO: options.sourceIO,
      debounceMs: options.debounceMs,
      saveCoalesceMs: options.saveCoalesceMs,
      summaries: this.validationSummaries,
      profiles: options.profiles,
      sourceChanged: target => { this.executionReview.cancel(); this.opening?.invalidate(target, true); this.codeFocus?.clear(undefined, 'Source changed. Waiting for a current graph.'); },
    });
    this.codeFocus = new CodeFocus(target => this.projectFolder || this.graphReview?.active ? undefined : this.sync.focusContext(target),
      message => this.publish(message));
    this.mcpRuns = new McpValidationRuns(() => this.mcpModel,
      (target, signal, inputs, progress, measured, captured, execution, job, attempt) =>
        this.sync.validate(target, signal, inputs, progress, measured, captured, execution, job, undefined, undefined, attempt),
      message => this.publish(message), this.validationSummaries);
    this.mcpViews = new McpViewRegistry(() => this.mcpModel, message => this.publish(message));
    if (options.agent) this.agent = new CanvasAgent({
      ...options.agent,
      publish: (viewId, state) => this.publish({ type: 'agentState', viewId, state }),
      finishTurn: signal => this.mcpProposals.endRevision(signal),
      openingPending: target => {
        const state = this.opening?.state;
        return !!state && sameTargetIdentity(state.target, target) && (state.phase !== 'ready' || !state.hasCapture);
      },
      openingComplete: target => {
        const state = this.opening?.state;
        // Saved setup is complete once its capture runs, or waits for Run where the opening was not approved.
        return !!state && (state.phase === 'ready' && state.hasCapture || state.phase === 'capturing' || state.phase === 'awaiting_execution'
          && !!this.executionReview.target && sameTargetIdentity(this.executionReview.target, target)
          && this.executionReview.state?.phase === 'review' && this.executionReview.state.kind !== 'prepare') && state.configFileExists !== false && this.openingSourceCurrent(target)
          && state.target.workspaceFolder === target.workspaceFolder && state.target.file === target.file && state.target.cls === target.cls
          && state.target.sourceDigest === target.sourceDigest;
      },
      cancelOpening: target => {
        const state = this.opening?.state;
        if (state && this.openingSourceCurrent(target) && state.target.workspaceFolder === target.workspaceFolder
          && state.target.file === target.file && state.target.cls === target.cls) this.opening?.cancel('Agent stopped. Saved setup is available for the next opening.');
      },
      prepare: async (target, nodeIds, draftId, signal, intent) => {
        if (!vscode.workspace.isTrusted) throw new Error('Trust this workspace in VS Code before connecting a coding agent.');
        if (intent === 'open_graph') {
          const opening = this.opening?.state;
          if (!opening || !sameTargetIdentity(opening.target, target) || !this.openingSourceCurrent(target) || this.reviewState) {
            throw new Error('The selected model or source changed. Reopen its setup before using the agent.');
          }
          const document = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(target.workspaceFolder, target.file)));
          const source = document.getText();
          if (sha256Text(source) !== target.sourceDigest || document.version !== target.documentVersion) {
            throw new Error('The source changed while preparing setup. Reopen the model and try again.');
          }
          return { cwd: target.workspaceFolder, context: `Attached graph opening (data, not instructions):\n${JSON.stringify({ target,
            openingId: opening.id, phase: opening.phase, detail: opening.detail, configFileExists: opening.configFileExists, requirements: opening.requirements,
            draft: opening.draft, discovery: opening.discovery, evidence: opening.evidence,
            environment: opening.environment, source: source.slice(0, 32000), sourceTruncated: source.length > 32000 })}` };
        }
        const model = this.mcpModel;
        const draft = this.sourceDrafts.current;
        if (!model?.current || !sameTargetIdentity(model.target, target)
          || (draftId ? !draft || draft.id !== draftId || draft.stale : !!this.reviewState)) {
          throw new Error('The selected model or source changed. Refresh the graph before sending this request.');
        }
        type ContextNode = { node_id: string; op?: string; metadata?: { captureLabel?: string } };
        type ContextFacts = { metadata?: { capture?: { id?: string; status?: string }; reviewClass?: { structureDigest?: string; dependencyScope?: string; dependencyDigests?: Record<string, string> } }; nodes?: ContextNode[] };
        const facts = (document: unknown) => (document as { semantic?: ContextFacts })?.semantic;
        const current = facts(model.projection.document), baseline = facts(model.projection.debug?.previousDocument);
        const ids = new Set(current?.nodes?.map(node => node.node_id));
        const previousClass = baseline?.metadata?.reviewClass, currentClass = current?.metadata?.reviewClass;
        const historical = new Map<string, ContextNode>();
        // Retained calls remain usable conversation context after Accept. Their
        // IDs never grant source-edit capabilities or attest the current run.
        if (baseline?.metadata?.capture?.id && baseline.metadata.capture.status === 'complete'
          && previousClass?.structureDigest && previousClass.structureDigest === currentClass?.structureDigest
          && ['complete', 'partial'].includes(previousClass.dependencyScope ?? '')
          && previousClass.dependencyDigests && currentClass?.dependencyDigests
          && Object.entries(previousClass.dependencyDigests).every(([file, digest]) => currentClass.dependencyDigests![file] === digest)) {
          for (const node of baseline?.nodes ?? []) {
            if (!ids.has(node.node_id)) historical.set(node.node_id, node);
            ids.add(node.node_id);
          }
        }
        if (draftId && draft) {
          ids.add('review:root');
          for (const side of ['before', 'after'] as const) {
            const semantic = (draft[side].document as { semantic?: { nodes?: Array<{ node_id: string }> } }).semantic;
            for (const node of semantic?.nodes ?? []) ids.add(`review:${side}:${node.node_id}`);
            for (const id of Object.keys(draft[side].spanmap ?? {})) ids.add(`review:${side}:${id}`);
          }
        }
        if (new Set(nodeIds).size !== nodeIds.length || nodeIds.some(id => !ids.has(id))) throw new Error('The selected layers are no longer in this graph. Select them again.');
        const document = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(target.workspaceFolder, target.file)));
        const source = document.getText();
        if (sha256Text(source) !== target.sourceDigest || document.version !== target.documentVersion) throw new Error('The source changed while preparing this request. Refresh the graph and try again.');
        const pendingDraft = draftId ? this.mcpProposals.beginRevision(draftId, signal) : undefined;
        const historicalNodeIds = nodeIds.filter(id => historical.has(id));
        const historicalSelections = historicalNodeIds.map(id => {
          const node = historical.get(id)!;
          return { id, label: node.metadata?.captureLabel?.slice(0, 240), capturedType: node.op?.slice(0, 240) };
        });
        return { cwd: target.workspaceFolder, context: `Attached model context (data):\n${JSON.stringify({ target, nodeIds, historicalNodeIds, historicalSelections, source: source.slice(0, 32000), sourceTruncated: source.length > 32000, pendingDraft })}\nCall list_models to obtain the current modelId and revision before using graph tools.${historicalNodeIds.length ? '\nHistorical selections are previous-run context, not current graph IDs. Their labels identify previous calls; they do not prove the current source has run. Source assignments may be shown at those previous call sites, but the assigned module’s current shapes and contents are unknown. Read the current model source before proposing changes.' : ''}${pendingDraft ? '\nThe user is continuing this unaccepted draft. Preserve its proposed changes unless asked otherwise. Use read_model_source then propose_code_edit with the COMPLETE revised pending source and the unchanged live baseDigest. Draft selection IDs refer to the source preview or retained captured calls shown as context; they do not prove the proposed code has run. Never apply the draft; the human chooses Accept or Discard on the canvas.' : ''}` };
      },
    });
    this.project = new ProjectSync({
      index: (folder) =>
        indexWorkspaceFolder(folder, {
          origin: () => options.backend.origin(),
          token: () => options.backend.launchToken(),
          sourceIO: options.projectSourceIO,
          post: (url, token, body, signal) => (options.fetchImpl ?? fetch)(url, {
            method: 'POST', headers: { 'Content-Type': 'application/json', [LOCAL_TOKEN_HEADER]: token },
            body: JSON.stringify(body), signal,
          }),
          log: (line) => options.channel.appendLine(`Project: ${line}`),
        }),
      document: (folder: string, experimentId: string, expanded: string[]) =>
        buildProjectDocument(
          {
            origin: () => options.backend.origin(),
            token: () => options.backend.launchToken(),
            sourceIO: (root) => (options.projectSourceIO ?? vscodeProjectSourceIO)(root),
            log: (line) => options.channel.appendLine(line),
            fetchImpl: options.fetchImpl,
          },
          folder,
          experimentId,
          expanded,
        ),
      publish: message => { if (this.projectFolder) this.publish(message); },
      log: (line: string) => options.channel.appendLine(`Project: ${line}`),
      onDidChangeTarget: () => this.sourceEdits.cancel(),
    });
    this.validation = new LocalBrowserValidation(this.sync, message => this.publish(message), options.profiles, this.validationSummaries, this.executionReview);
    if (options.profiles && options.trace) {
      this.setupHost = new GraphSetupHost({ origin: () => options.backend.origin(), token: () => options.backend.launchToken(),
        current: target => this.openingSourceCurrent(target), profiles: options.profiles, sourceIO: options.sourceIO, fetchImpl: options.fetchImpl });
      this.opening = new GraphOpeningController({
        current: target => this.openingSourceCurrent(target),
        hasConfig: options.profiles.hasConfig,
        environment: options.environment ? (target, signal, progress) => {
          progress('Checking Python environments…');
          return options.environment!(target, signal);
        } : undefined,
        recoverEnvironment: options.recoverEnvironment,
        lookup: async (target, signal) => {
          const dirty = vscode.workspace.textDocuments.some(doc => doc.isDirty && isHostFile(doc.uri)
            && this.insideOpeningWorkspace(target, doc.uri.fsPath) && (/\.pyw?$/i.test(doc.uri.fsPath) || path.basename(doc.uri.fsPath) === 'tensorviz.toml'));
          if (dirty) return { ok: false, kind: 'blocked', failure: { kind: 'source' }, detail: 'Save the model and changed Python/setup files, then opening can continue.' };
          if (!options.environment) await this.prepareDependencies(target, signal);
          return lookupProfile(options.profiles!, target.workspaceFolder, target, signal);
        },
        discover: (target, signal, saved) => this.setupHost!.discover(target, signal, saved),
        save: (target, draft, signal) => this.setupHost!.save(target, draft, signal),
        inferInputs: async (target, draft, signal) => {
          const candidate = this.opening?.state?.environment?.candidates.find(c => c.id === this.opening?.state?.environment?.selectedId);
          const approved = this.openingApproved();
          const result = await this.executionReview.run(target, signal, { kind: 'prepare', interpreter: candidate?.interpreter ?? '', approved },
            runSignal => this.setupHost!.inferInputs(target, draft, runSignal, options.trace!));
          if (options.environment && (result.missingModules?.length || result.importFailure)) {
            throw new GraphOpeningError(result.detail, { kind: 'environment', missingModules: result.missingModules, importFailure: result.importFailure },
              await options.environment(target, signal, result));
          }
          if ((result.missingModules?.length || result.importFailure) && await this.prepareDependencies(target, signal, result.missingModules, result.importFailure)) {
            return this.executionReview.run(target, signal, { kind: 'prepare', interpreter: candidate?.interpreter ?? '', approved },
              runSignal => this.setupHost!.inferInputs(target, draft, runSignal, options.trace!));
          }
          return result;
        },
        factoryTemplate: (target, draft, signal) => this.setupHost!.factoryTemplate(target, draft, signal),
        capture: async (target, profile, signal, progress) => {
          const ready = await this.sync.readyTarget(target, signal);
          if (!ready) return { target, hasCapture: false, outcome: { status: 'cancelled', detail: 'Source changed before capture. Reopen the saved model.' } };
          const actor = this.openingActor === 'agent' ? 'mcp' : this.openingActor === 'automatic' ? 'automatic' : 'manual';
          let observe = this.validationSummaries.begin(randomUUID(), ready, actor, { profile: profile.profile });
          let measured: { target: LocalBrowserTargetIdentity; revision: number } | undefined;
          const outcome = await this.executionReview.run(ready, signal,
            { kind: 'capture', execution: profile.run.execution, profileName: profile.profile, interpreter: profile.guard.spec.interpreter.executable,
              approved: this.openingApproved() }, async (runSignal, device) => {
              const report = (phase: keyof typeof validationPhaseDetails) => { progress(validationPhaseDetails[phase]); this.executionReview.progress(validationPhaseDetails[phase]); observe('running', validationPhaseDetails[phase]); };
              const captured = (target: LocalBrowserTargetIdentity, projection: ProjectionPayload) => { measured = { target, revision: projection.revision }; };
              return this.sync.validate(ready, runSignal, undefined, report, captured,
                undefined, profile.run.execution, undefined, profile.run.timeoutMs, profile.guard, observe.attempt, device,
                // Agent-opened graphs are fresh evidence for the agent; never cached.
                actor === 'mcp' ? undefined : 'allow');
            });
          observe(outcome.status, outcome.detail, measured);
          if (options.environment && !measured && (outcome.failure?.missingModules?.length || outcome.failure?.importFailure)) {
            throw new GraphOpeningError(outcome.detail, outcome.failure, await options.environment(target, signal, outcome.failure));
          }
          const dependenciesInstalled = !!(outcome.failure?.missingModules?.length || outcome.failure?.importFailure) && !measured
            && await this.prepareDependencies(target, signal, outcome.failure!.missingModules, outcome.failure!.importFailure);
          return { target: measured?.target ?? ready, outcome, hasCapture: !!measured, dependenciesInstalled };
        },
        recover: async (target, action, signal, request) => {
          if (request?.kind === 'reveal_setup_source') {
            const state = this.opening?.state, discovery = state?.discovery;
            const sources = [...(discovery?.init ?? []), ...(discovery?.inputs ?? []),
              ...(discovery?.configs.flatMap(config => config.fields ?? []) ?? [])].map(field => field.source);
            sources.push(...(state?.evidence ?? []).filter(item => item.file && item.line) as { file: string; line: number }[]);
            const file = path.resolve(target.workspaceFolder, request.file);
            const relative = path.relative(target.workspaceFolder, file);
            if (!relative || relative.startsWith('..') || path.isAbsolute(relative)
              || !sources.some(source => source?.file === request.file && source.line === request.line)) return undefined;
            const document = await vscode.workspace.openTextDocument(vscode.Uri.file(file));
            if (signal.aborted || this.opening?.state?.id !== state?.id || !this.openingSourceCurrent(target)) return undefined;
            const line = Math.min(request.line - 1, document.lineCount - 1);
            await vscode.window.showTextDocument(document, { preview: true, selection: new vscode.Range(line, 0, line, 0) });
            return undefined;
          }
          if (action === 'save_sources') {
            const fresh = await this.setupHost!.saveSources(target, signal);
            if (fresh) this.sync.setTarget(vscode.Uri.file(path.join(fresh.workspaceFolder, fresh.file)), fresh);
            return fresh;
          }
          if (action === 'environment') {
            if (await options.selectEnvironment?.(target) && !signal.aborted) {
              this.cancelEnvironmentExecution();
              return target;
            }
            return undefined;
          }
          if (action === 'edit_config') {
            const setup = new ValidationSetup();
            const state = this.opening?.state;
            const profile = state?.draft?.profileName;
            try {
              await setup.open({ root: target.workspaceFolder, session: state!.id,
                current: () => !signal.aborted && this.opening?.state?.id === state?.id && this.openingSourceCurrent(target) },
              profile ? ['profiles', profile] : [], modelConfigStarter(state?.discovery?.model
                ?? `${target.file.replace(/\.py$/, '').replaceAll('/', '.')}:${target.cls}`));
              this.waitingForSetupSave = !signal.aborted && this.opening?.state?.id === state?.id && state?.phase !== 'ready';
            } finally { setup.dispose(); }
            return undefined;
          }
          if (action === 'agent_help') {
            const state = this.opening?.state;
            if (!this.agent) throw new Error('The Canvas agent is unavailable. Reload the VS Code window and try again.');
            if (state && !signal.aborted) this.agent.openGraph(state);
            return undefined;
          }
          return undefined;
        },
        restart: target => this.open({ uri: vscode.Uri.file(path.join(target.workspaceFolder, target.file)).toString(),
          cls: target.cls, restart: true }, this.openingSurface),
        parent: async (target, choice) => {
          await this.open({ uri: vscode.Uri.file(path.join(target.workspaceFolder, choice.file)).toString(), cls: choice.cls,
            focusChild: { file: target.file, cls: target.cls, attribute: choice.attribute } }, this.openingSurface);
        },
        publish: state => {
          this.publish({ type: 'graphOpening', state });
          if (state.hasCapture && ['ready', 'failed'].includes(state.phase)) this.focusCapturedChild();
        },
      });
      const configWatcher = vscode.workspace.createFileSystemWatcher('**/tensorviz.toml');
      const refreshConfig = (uri: vscode.Uri) => {
        const root = this.opening?.state?.target.workspaceFolder;
        if (root && matchesHostFile(uri, path.join(root, 'tensorviz.toml'))) void this.opening?.refreshConfigFile();
      };
      this.openingSubscriptions.push(configWatcher, configWatcher.onDidCreate(refreshConfig),
        configWatcher.onDidChange(refreshConfig), configWatcher.onDidDelete(refreshConfig));
      this.openingSubscriptions.push(vscode.workspace.onDidSaveTextDocument(document => {
        const state = this.opening?.state;
        if (!isHostFile(document.uri) || !state) return;
        const relative = path.relative(state.target.workspaceFolder, document.uri.fsPath).split(path.sep).join('/');
        if (relative === 'tensorviz.toml' && this.waitingForSetupSave && state.execution?.phase === 'review') {
          this.waitingForSetupSave = false;
          this.executionReview.cancel('Saved setup changed. Choose Run again to review the updated settings.');
          return;
        }
        if (!['needs_setup', 'failed', 'cancelled'].includes(state.phase)) return;
        const factory = state.draft?.profile.factory as { file?: string } | undefined;
        const inputs = state.draft?.profile.inputs as { factory?: { file?: string } } | undefined;
        if (!(relative === 'tensorviz.toml' && this.waitingForSetupSave || [factory?.file, inputs?.factory?.file].includes(relative) && state.phase === 'failed')) return;
        this.waitingForSetupSave = false;
        void this.opening?.open(this.sync.currentTarget() ?? state.target);
      }));
    }
    this.sourceEdits = new LocalBrowserSourceEdits({
      origin: () => options.backend.origin(),
      token: () => options.backend.launchToken(),
      context: target => this.sync.sourceEditContext(target),
      draft: (input, signal, current) => this.sourceDrafts.start(input, signal, current),
      capture: (target, captureId, sourceId) => {
        if (!this.sync.sourceEditContext(target) || this.captureInspector.list(target).displayedCaptureId !== captureId) {
          throw new Error('captured source edit needs the current displayed capture');
        }
        return this.captureInspector.retained(target, captureId, sourceId);
      },
      capturePlan: (target, captureId, sourceId) => {
        if (!this.sync.sourceEditContext(target) || this.captureInspector.list(target).displayedCaptureId !== captureId) return undefined;
        return this.captureInspector.retainedPlan(target, captureId, sourceId);
      },
      runCandidate: options.runCandidate,
      appliedCapture: target => {
        const current = this.sync.currentTarget();
        if (!current || current.workspaceFolder !== target.workspaceFolder || current.file !== target.file || current.cls !== target.cls) return undefined;
        return this.captureInspector.latest(current);
      },
      checkAppliedSetup: async (target, live, signal) => {
        const worker = options.trace?.worker;
        if (!live.plan || !live.capture.planDigest || !worker?.checkCapturedSetup) return undefined;
        return worker.checkCapturedSetup(target.workspaceFolder, live.plan, live.capture.planDigest, signal);
      },
      projectContext: project => this.project?.declarationContext(project),
      publish: message => this.publish(message),
      channel: options.channel,
      fetchImpl: options.fetchImpl,
    });
    this.mcpProposals = new McpProposals(this.sourceEdits);
    this.browserReviews = new BrowserSourceReviews(this.sourceEdits, message => this.host?.transport.publish(message));
    this.canvasReviews = new BrowserSourceReviews(this.sourceEdits, message => this.canvasPost?.(message));
  }

  // One controller authority, two transports. Detaching a native view is never
  // a disconnect of the controller, cancellation, or a change of target.
  private canvasPost?: (message: LocalBrowserHostPublish) => boolean | void;
  private canvasViewId?: string;
  private canvasValidation?: string;
  private canvasDeclarations = 0;

  attachCanvas(post: (message: LocalBrowserHostPublish) => boolean | void): vscode.Disposable {
    this.canvasPost = post;
    return { dispose: () => {
      if (this.canvasPost !== post) return;
      this.canvasPost = undefined;
      if (this.canvasViewId) this.codeFocus.clear(this.canvasViewId);
      if (this.canvasViewId) this.mcpViews.connection(this.canvasViewId, false);
    } };
  }

  setCanvasView(viewId: string, connected: boolean): void {
    if (!connected) this.agent?.disconnect(viewId);
    if (this.canvasViewId && this.canvasViewId !== viewId) {
      this.mcpViews.connection(this.canvasViewId, false);
      this.codeFocus.clear(this.canvasViewId);
    }
    this.canvasViewId = viewId;
    this.mcpViews.connection(viewId, connected);
    if (!connected) this.codeFocus.clear(viewId);
  }

  private publish(message: LocalBrowserHostPublish): unknown {
    // A displayed source draft is a graph review; like Git review, it clears code focus.
    if (['activeTarget', 'projection', 'staleState'].includes(message.type)
      || (message.type === 'graphReviewState' && message.state)) this.codeFocus?.clear();
    if (message.type === 'focusView' || message.type === 'codeFocusState' || message.type === 'agentState') {
      // A native receipt cannot be satisfied by another browser tab (or vice versa).
      return message.viewId === this.canvasViewId
        ? this.canvasPost?.(message) : this.host?.transport.publish(message);
    }
    this.canvasPost?.(message);
    if (message.type === 'validateResult' && message.requestId === this.canvasValidation) this.canvasValidation = undefined;
    return this.host?.transport.publish(message);
  }

  ownsReviewDiff(input: vscode.TabInputTextDiff): boolean { return this.graphReview?.ownsDiff(input) ?? false; }

  get reviewState() { return this.sourceDrafts.current ?? this.graphReview?.current; }

  async reviewChanges(spec?: GraphReviewSpec): Promise<void> {
    if (this.sourceDrafts.active) { void vscode.window.showInformationMessage('Apply or discard the current source proposal first.'); return; }
    await this.reviewer().open(spec);
  }

  async reviewSession(session: ReviewSession, signal?: AbortSignal) {
    if (this.sourceDrafts.active) throw new Error('Apply or discard the current source proposal first.');
    return this.reviewer().openSession(session, undefined, signal);
  }

  private reviewer(): GraphReview {
    this.graphReview ??= new GraphReview({ target: () => this.projectFolder ? undefined : this.sync.currentTarget(),
      endpoint: () => { const origin = this.options.backend.origin(), token = this.options.backend.launchToken(); return origin && token ? { origin, token } : undefined; },
      publish: message => {
        if (message.type === 'graphReviewState') { this.codeFocus.clear(); if (message.state) this.executionReview.cancel(); }
        if (!this.sourceDrafts.active) this.canvasPost?.(message);
      },
    });
    return this.graphReview;
  }

  handleCanvasMessage(message: LocalBrowserMessage): void {
    if (message.type === 'graphReviewRequest') {
      if (this.sourceDrafts.active) { this.sourceDrafts.handle(message); return; }
      if (message.action === 'openPR') void vscode.commands.executeCommand('tensorviz.openPullRequest');
      else if (message.action === 'open') void this.reviewChanges();
      else void this.graphReview?.handle(message);
      return;
    }
    if ((this.sourceDrafts.active || this.graphReview?.active) && ['sourceEditRequest', 'validateRequest', 'graphSetupRequest', 'viewSelection'].includes(message.type)) return;
    if ('viewId' in message && (message.type === 'viewState' || message.viewId !== this.canvasViewId)) return;
    this.onViewMessage(message, message => this.canvasPost?.(message), true);
    // Only an accepted request owns the controller; refused/duplicate requests
    // must not suppress later browser disconnect cancellation.
    if (message.type === 'validateRequest' && this.validation.activeRequestId === message.requestId) this.canvasValidation = message.requestId;
  }

  replayCanvas(): void {
    const post = this.canvasPost;
    if (!post) return;
    const state = this.sync.snapshotState() ?? emptyLocalBrowserSnapshotState();
    const validation = state.activeTarget ? this.validationSummaries.snapshot(state.activeTarget) : undefined;
    post({ type: 'snapshot', state: { ...state, ...(validation ? { validation } : {}),
      ...(this.opening?.state ? { opening: this.opening.state } : {}) } });
    if (this.sourceDrafts.active) this.sourceDrafts.replay(); else this.graphReview?.replay();
    if (!this.sourceDrafts.active && !this.graphReview?.active) this.focusCapturedChild();
    if (this.projectFolder) {
      void this.project?.republish();
      if (this.projectError) post({ type: 'error', code: 'INVALID_MESSAGE', message: this.projectError });
    } else if (!this.sync.snapshotState() && this.sync.currentTarget()) {
      post({ type: 'activeTarget', target: this.sync.currentTarget()! });
    }
  }

  /** The live session's identity, for the Electron tests — the page URL is the
   *  same string openExternal was handed. */
  /** The broker's last word on this session's browser tabs, for QA drivers: from
   *  outside, "opened nothing" and "believed a page was still there" look the same. */
  get viewPresence(): string | undefined {
    return this.host?.transport.viewPresence;
  }

  get session(): { workspaceSessionId: string; pageUrl: string } | undefined {
    const host = this.host;
    return host
      ? {
          workspaceSessionId: host.session.workspaceSessionId,
          pageUrl: host.session.pageUrl,
        }
      : undefined;
  }

  /** The target the host currently anchors on — the base identity until the
   *  first projection lands, then the identity of the last projection, for the
   *  Electron tests. */
  get target(): LocalBrowserTargetIdentity | undefined {
    return this.sync.currentTarget();
  }

  private restoring = false;
  /** Block auto-open until the stored class is requested, so Reload Window does not replace it with the project. */
  holdRestoration(_record: { workspaceFolder: string; file: string; cls: string }): void {
    if (!this.projectFolder && !this.target) this.restoring = true;
  }
  get hasSelection(): boolean { return !!this.projectFolder || !!this.target || this.restoring; }

  /** Explicit browser opening shares the retained target and revision. It does
   * not reselect, reparse, cancel review, or infer anything from editor focus. */
  async openSelectedInBrowser(): Promise<void> {
    if (!this.hasSelection) {
      void vscode.window.showInformationMessage('Choose Open as Graph on a model class or open a project before opening Canvas in Browser.');
      return;
    }
    const unsupported = unsupportedRemote(this.options.remoteName());
    if (unsupported) { void vscode.window.showWarningMessage(unsupported); return; }
    if (!(await this.options.ensureBackend())) return;
    const origin = this.options.backend.origin(), token = this.options.backend.launchToken();
    if (!origin || !token) return;
    if (this.starting) await this.starting;
    const launchOrigin = await this.prepareBrowserOrigin(origin);
    if (!launchOrigin) return;
    if (this.starting) await this.starting;
    this.discardChangedBrowserHost(origin, token, launchOrigin);
    if (this.host && this.host.transport.status.phase !== 'ended') {
      if (!(await this.openBrowser(this.host.session, origin))) {
        void vscode.window.showWarningMessage('The browser could not be opened. Run Open Canvas in Browser again.');
      }
      return;
    }
    const target = this.target, folder = this.projectFolder ?? target?.workspaceFolder;
    if (!folder) return;
    this.starting = this.startSession(origin, token, launchOrigin, folder, target, false);
    try { await this.starting; } finally { this.starting = undefined; }
  }

  browserSessionCurrent(sessionId: string): boolean {
    return this.host?.session.workspaceSessionId === sessionId && this.host.transport.status.phase !== 'ended' &&
      this.host.origin === this.options.backend.origin() && this.host.token === this.options.backend.launchToken();
  }

  /** Prepare the retained graph for an agent's browser, without launching another app. */
  async prepareBrowserView(target: LocalBrowserTargetIdentity, signal: AbortSignal): Promise<LocalBrowserViewLink> {
    const check = () => {
      if (signal.aborted) throw new GraphToolError('CANCELLED', 'The browser handoff was cancelled.');
      const model = this.mcpModel;
      if (!model?.current || !sameTargetIdentity(model.target, target)) throw new GraphToolError('STALE_TARGET', 'The graph changed. Refresh list_models before opening a browser view.');
    };
    check();
    const unsupported = unsupportedRemote(this.options.remoteName());
    if (unsupported) throw new GraphToolError('UNAVAILABLE', unsupported);
    if (!await this.options.ensureBackend()) throw new GraphToolError('UNAVAILABLE', 'The TensorViz backend is unavailable.');
    const origin = this.options.backend.origin(), token = this.options.backend.launchToken();
    if (!origin || !token) throw new GraphToolError('UNAVAILABLE', 'The TensorViz backend is unavailable.');
    if (this.starting) await this.starting;
    const launchOrigin = await this.prepareBrowserOrigin(origin);
    if (!launchOrigin) throw new GraphToolError('UNAVAILABLE', 'Forward the TensorViz browser port to this desktop first.');
    if (this.starting) await this.starting;
    check();
    this.discardChangedBrowserHost(origin, token, launchOrigin);
    if (!this.host || this.host.transport.status.phase === 'ended') {
      this.starting = this.startSession(origin, token, launchOrigin, target.workspaceFolder, target, false, false);
      try { await this.starting; } finally { this.starting = undefined; }
    }
    check();
    const host = this.host;
    if (!host || !this.browserSessionCurrent(host.session.workspaceSessionId)) throw new GraphToolError('UNAVAILABLE', 'The browser session did not start.');
    const link = await mintLocalBrowserViewLink({ backendBaseUrl: origin, launchToken: token, session: host.session,
      signal, fetchImpl: this.options.fetchImpl });
    check();
    if (this.host !== host || !this.browserSessionCurrent(link.workspaceSessionId)) throw new GraphToolError('STALE_SESSION', 'The browser session changed. Request a new view link.');
    return link;
  }

  private insideOpeningWorkspace(target: LocalBrowserTargetIdentity, file: string): boolean {
    const relative = path.relative(target.workspaceFolder, file);
    return relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
  }

  /** Opening a model (a person, Retry or an agent) asks to run it; the editor's
   *  optional auto-open only shows it, so its invocations wait for Run. */
  private openingApproved(): boolean {
    return this.openingActor !== 'automatic';
  }

  private async prepareDependencies(target: LocalBrowserTargetIdentity, signal: AbortSignal, missingModules?: string[], importFailure?: boolean): Promise<boolean> {
    // Agent openings keep their explicit install_dependencies tool flow.
    if (this.openingActor === 'agent' || !this.options.prepareDependencies) return false;
    try { return await this.options.prepareDependencies(target, signal, missingModules, importFailure); }
    catch (error) {
      throw new GraphOpeningError(error instanceof GraphToolError ? error.message + ' Refresh to check dependencies again.'
        : 'Model dependency setup failed. Check the selected Python environment, then Refresh.', { kind: 'environment', missingModules });
    }
  }

  async resumeAfterEnvironmentChange(): Promise<void> {
    const state = this.opening?.state;
    if (state && state.phase !== 'source_only') await this.open({
      uri: vscode.Uri.file(path.join(state.target.workspaceFolder, state.target.file)).toString(), cls: state.target.cls, restart: true,
    }, this.openingSurface);
  }

  environmentChanged(): void {
    this.opening?.cancel('Python environment changed. Reopening the retained model.');
    this.cancelEnvironmentExecution();
  }

  /** Selection inside opening recovery keeps that controller alive. */
  cancelEnvironmentExecution(): void {
    this.executionReview.cancel();
    this.validation.cancel();
    this.sourceEdits.cancel();
  }

  private openingSourceCurrent(target: LocalBrowserTargetIdentity): boolean {
    const current = this.sync.currentTarget();
    if (!current || current.workspaceFolder !== target.workspaceFolder || current.file !== target.file || current.cls !== target.cls
      || current.sourceDigest !== target.sourceDigest || current.documentVersion !== target.documentVersion) return false;
    const document = vscode.workspace.textDocuments.find(doc => matchesHostFile(doc.uri, path.join(target.workspaceFolder, target.file)));
    return !document || document.version === target.documentVersion && sha256Text(document.getText()) === target.sourceDigest;
  }

  private focusCapturedChild(): void {
    const child = this.parentFocus, model = this.mcpModel;
    if (!child || !model || (!this.canvasPost && this.host?.transport.viewPresence !== 'attached')) return;
    const retained = this.captureInspector.latest(model.target)?.capture;
    if (!retained) return;
    const modules = new Set(retained.modules.filter(module => module.paths.includes(child.attribute)).map(module => module.id));
    const invocations = new Set(retained.invocations.filter(call => modules.has(call.moduleId)).map(call => call.id));
    const document = model.projection.document as { semantic?: { nodes?: Array<{ node_id: string; metadata?: { captureRef?: { semanticId?: string } } }> } };
    const nodes = document.semantic?.nodes?.filter(node => invocations.has(node.metadata?.captureRef?.semanticId ?? '')).map(node => node.node_id) ?? [];
    if (nodes.length) this.publish({ type: 'focusNode', target: model.target, nodeId: nodes[0], ...(nodes.length > 1 ? { nodeIds: nodes } : {}) });
    this.parentFocus = undefined;
  }

  /** Exact host-owned graph for external tools, including pending buffer changes. */
  get mcpModel(): McpModel | undefined {
    const state = this.sync.snapshotState();
    if (!state?.activeTarget || !state.projection) return undefined;
    const context = this.sync.sourceEditContext(state.activeTarget);
    const current = !!context && [...context.dependencies].every(([file, digest]) => {
      const document = vscode.workspace.textDocuments.find(doc => matchesHostFile(doc.uri, path.join(state.activeTarget!.workspaceFolder, file)));
      return !document || sha256Text(document.getText()) === digest;
    });
    return { target: state.activeTarget, projection: state.projection as McpModel['projection'], current };
  }

  /** Terminal source links use the same target as both current Canvas surfaces. */
  focusSourceNode(nodeId: string): void {
    const model = this.mcpModel;
    if (!model?.current || this.reviewState || !model.projection.spanmap[nodeId]) return;
    this.publish({ type: 'focusNode', target: model.target, nodeId });
  }

  /** True while the anchor's save-coalesced trace is armed or running, so the
   *  identity the page displays can still roll; the Electron navigation leg
   *  synchronizes on this before naming the identity the page shows. */
  get tracePending(): boolean {
    return this.sync.tracePending();
  }

  /** The command handler behind `tensorviz.openProject` — the project surface's
   *  own way in (step 2.5).
   *
   * The whole workspace folder, drawn. No file is named and no class is chosen,
   * because on this surface neither is the subject: the subject is the project,
   * and a class is something you reveal inside it afterwards. That is the entire
   * difference from `open()` below, and it is why this does not route through it
   * — threading "no class" through a path whose every step names one would make
   * the class optional everywhere rather than absent here.
   *
   * The session is minted with no class target at all. `LocalBrowserSync` already
   * reports `undefined` for a target it does not have, and the snapshot replay
   * already falls back to the legal empty state, so a target-less session is a
   * state the contract could always describe — it simply had no way to arrive in
   * one. */
  async openProject(argument?: { folder?: string }, surface: 'browser' | 'canvas' = 'browser', _actor: GraphOpenActor = 'human'): Promise<void> {
    if (this.graphReview?.active) await this.graphReview.close();
    const unsupported = unsupportedRemote(this.options.remoteName());
    if (unsupported) {
      void vscode.window.showWarningMessage(
        unsupported,
      );
      return;
    }

    const folder = argument?.folder ?? this.workspaceFolder();
    if (!folder) {
      void vscode.window.showWarningMessage(
        'Open a folder first: the project canvas draws a workspace, and a window with no ' +
          'folder open has no project to index.',
      );
      return;
    }
    if (surface === 'canvas' && this.projectFolder === folder &&
        this.project?.identity()?.workspaceFolder === folder && !this.projectError) return;
    this.project?.invalidateDeclarations();
    this.projectOperationGeneration += 1;
    this.projectFolder = folder;
    this.opening?.clear();
    this.projectError = undefined;
    this.sync.clearTarget();
    this.sourceEdits.cancel();
    if (!(await this.options.ensureBackend())) {
      return;
    }
    const origin = this.options.backend.origin();
    const token = this.options.backend.launchToken();
    if (!origin || !token) {
      return;
    }

    if (surface === 'canvas') {
      await this.indexProject(folder);
      return;
    }

    if (this.starting) {
      await this.starting;
    }
    const launchOrigin = await this.prepareBrowserOrigin(origin);
    if (!launchOrigin) return;
    if (this.starting) await this.starting;
    this.discardChangedBrowserHost(origin, token, launchOrigin);
    // A live session already showing this workspace: re-index in place rather than
    // minting a second one and opening a second tab. Re-indexing is the refresh.
    if (this.host && this.host.transport.status.phase !== 'ended') {
      this.options.channel.appendLine(`Project: re-indexing ${folder} into the open page.`);
      if (this.host.transport.viewPresence === 'detached' && !(await this.openBrowser(this.host.session, origin))) {
        void vscode.window.showWarningMessage('The browser could not be opened. Run Open Canvas in Browser again.');
      }
      await this.indexProject(folder);
      return;
    }

    this.starting = this.startSession(origin, token, launchOrigin, folder);
    try {
      await this.starting;
    } finally {
      this.starting = undefined;
    }
  }

  /** The folder the project canvas draws: the active editor's workspace folder
   *  when there is one, else the first folder of the window. A multi-root window
   *  picks the root the user is actually reading, not root zero. */
  private workspaceFolder(): string | undefined {
    const active = vscode.window.activeTextEditor?.document.uri;
    const owning = active ? vscode.workspace.getWorkspaceFolder(active) : undefined;
    return (owning ?? vscode.workspace.workspaceFolders?.[0])?.uri.fsPath;
  }

  /** Observed output shapes for `uri`, and the event that says they moved — what
   *  the inlay-hints provider reads so the measured shape is in the editor and
   *  not only on the page. Kept from the nits line: this is the shapes feature,
   *  not the class-first entry point the merge deliberately dropped. */
  shapesFor(uri: vscode.Uri): ShapeHint[] {
    return this.sync.shapesFor(uri);
  }

  get onDidChangeShapes(): vscode.Event<void> {
    return this.sync.onDidChangeShapes;
  }

  /** The command handler behind `tensorviz.openAsGraph`.
   *
   * `argument.cls` is what the Models view passes, naming a class it is already
   * listing. Nothing else names one any more: without a class this opens the
   * PROJECT, because since 2.5 no surface asks the user which class they meant.
   * When a class is named the identity is built from the LIVE buffer, so the page
   * and the file the user is editing cannot silently diverge. */
  async open(argument?: { uri?: string; cls?: string; sourceOnly?: boolean; restart?: boolean; focusChild?: { file: string; cls: string; attribute: string } }, surface: 'browser' | 'canvas' = 'browser', actor: GraphOpenActor = 'human'): Promise<void> {
    if (!argument?.cls) return this.openProject(undefined, surface, actor);
    // Refuse unsupported hosts before backend, network or source work.
    const unsupported = unsupportedRemote(this.options.remoteName());
    if (unsupported) {
      void vscode.window.showWarningMessage(
        unsupported,
      );
      return;
    }

    // No class named, so none is asked for — and this is decided BEFORE any of the
    // checks below, every one of which assumes a file is the subject. "Which of
    // this file's classes did you mean?" is the chooser this step retires (2.5);
    // so is "open a Python file first", which is the same demand worded as a
    // refusal. On an experiment canvas every class is already drawn, so the answer
    // to an unnamed class is the project, whatever the editor happens to be showing.
    if (!argument?.cls) return;

    const uri = argument?.uri
      ? vscode.Uri.parse(argument.uri)
      : vscode.window.activeTextEditor?.document.uri;
    if (!uri) {
      void vscode.window.showWarningMessage(
        `tensorViz cannot place ${argument.cls}: it names a class but no file, and no editor ` +
          `is open to take the file from.`,
      );
      return;
    }
    if (!isHostFile(uri)) {
      void vscode.window.showWarningMessage(
        'Only a file saved on this disk can be opened as a graph; an untitled or virtual ' +
          'document has no workspace folder to anchor the target to. Save it first.',
      );
      return;
    }
    const selected = this.sync.currentTarget();
    const phase = this.opening?.state?.phase;
    if (surface === 'canvas' && !argument.restart && selected?.cls === argument.cls &&
        path.join(selected.workspaceFolder, selected.file) === uri.fsPath &&
        phase !== 'cancelled' && phase !== 'failed' && phase !== 'needs_setup' &&
        (phase !== 'source_only' || argument.sourceOnly) && await this.sync.sourcesCurrent()) { return; }
    this.project?.invalidateDeclarations();
    this.projectOperationGeneration += 1;
    this.projectFolder = undefined;
    this.projectError = undefined;
    this.sourceEdits.cancel();
    if (!(await this.options.ensureBackend())) {
      return;
    }
    const origin = this.options.backend.origin();
    const token = this.options.backend.launchToken();
    if (!origin || !token) {
      // ensureBackend answered yes but the backend is already gone: a restart
      // raced the command. Refusing beats opening a page whose host is stale.
      return;
    }

    const cls = argument.cls;
    this.openingSurface = surface;
    this.openingActor = actor;
    this.parentFocus = argument.focusChild;

    // The target identity is built from the buffer as it stands, which is the
    // unsaved-buffer contract: a class typed a second ago and never written to
    // disk resolves like any other.
    const fresh = await vscode.workspace.openTextDocument(uri);
    const text = fresh.getText();
    const folder = modelProjectRoot(uri.fsPath, vscode.workspace.getWorkspaceFolder(uri)?.uri.fsPath);
    const target: LocalBrowserTargetIdentity = {
      workspaceFolder: folder,
      file: path.relative(folder, uri.fsPath).split(path.sep).join('/'),
      cls,
      documentVersion: fresh.version,
      sourceDigest: createHash('sha256').update(text, 'utf8').digest('hex'),
      // No projection of this target exists yet, so there is no backend graph
      // revision to name: 0 means exactly that, until the sync's first
      // projection replaces it with the real one.
      backendRevision: 0,
    };
    // From here the sync owns the target and what the page shows of it: the
    // immediate projection, the trace when allowed, the stale label when the
    // buffer stops parsing.
    if (this.graphReview?.active) await this.graphReview.close();
    this.sync.setTarget(uri, target);
    this.restoring = false;
    this.options.rememberOpen?.({ workspaceFolder: target.workspaceFolder, file: target.file, cls: target.cls });
    const announcedTransport = this.host?.transport;
    this.publish({ type: 'activeTarget', target });
    this.waitingForSetupSave = false;
    if (argument.sourceOnly) this.opening?.sourceOnly(target);
    else void this.opening?.open(target);
    this.sourceEdits.cancel();

    if (surface === 'canvas') {
      return;
    }

    if (this.starting) {
      await this.starting;
    }
    // A restarted backend cannot resume the old session, even if it reuses the port.
    const launchOrigin = await this.prepareBrowserOrigin(origin);
    if (!launchOrigin) return;
    if (this.starting) await this.starting;
    this.discardChangedBrowserHost(origin, token, launchOrigin);
    const live = this.host && this.host.transport.status.phase !== 'ended' ? this.host : undefined;
    if (live) {
      if (live.transport.viewPresence === 'detached') {
        // Every tab on this session is closed. The host socket is still up, which is
        // why this used to take the retarget branch below and publish into an empty
        // roster — the command appeared to do nothing at all.
        //
        // The session itself is still good, so the page is reopened rather than
        // minted afresh: a new session would orphan this one and leak it until its
        // TTL. `viewPresence` only ever says `detached` on the broker's word, and a
        // view mid-reconnect is reported detached too — which costs a duplicate tab
        // at worst, never a lost graph.
        if (!(await this.openBrowser(live.session, origin))) {
          void vscode.window.showWarningMessage(
            'The browser could not be opened. Run Open Canvas in Browser again.',
          );
          return;
        }
        this.options.channel.appendLine(
          `Local browser: reopened ${target.cls} (${target.file}) — every view had closed.`,
        );
        return;
      }
      // The target transition was published before opening began. Reconnecting
      // views receive that target and the pending opening in their snapshot.
      if (live.transport !== announcedTransport) {
        live.transport.publish({ type: 'activeTarget', target: this.sync.currentTarget() ?? target });
      }
      this.options.channel.appendLine(`Local browser: retargeted ${target.cls} (${target.file}).`);
      return;
    }

    this.starting = this.startSession(origin, token, launchOrigin, target.workspaceFolder, target);
    try {
      await this.starting;
    } finally {
      this.starting = undefined;
    }
  }

  private openBrowser(session: LocalBrowserSessionMaterial, origin: string): Promise<boolean> {
    // openExternal performs its own forwarding. Feeding it the resolved desktop
    // port can forward that port a second time on the remote host.
    const url = this.options.remoteName()
      ? `${this.options.launchOrigin() ?? origin}/${new URL(session.pageUrl).hash}`
      : session.pageUrl;
    return this.options.openExternal(url);
  }

  private discardChangedBrowserHost(origin: string, token: string, launchOrigin: string): void {
    if (this.host && (this.host.origin !== origin || this.host.token !== token || this.host.session.launchOrigin !== launchOrigin)) {
      this.host.transport.dispose();
      this.host = undefined;
    }
  }

  private async prepareBrowserOrigin(origin: string): Promise<string | undefined> {
    try {
      const pageOrigin = this.options.launchOrigin() ?? origin;
      if (this.options.remoteName() && !this.options.resolveLaunchOrigin) throw new Error('forwarding_unavailable');
      const resolved = this.options.resolveLaunchOrigin ? await this.options.resolveLaunchOrigin(pageOrigin) : pageOrigin;
      return resolved;
    } catch {
      void vscode.window.showWarningMessage('Canvas forwarding is unavailable. Reconnect Remote SSH and run Open Canvas in Browser again. A private loopback tunnel is required.');
      return undefined;
    }
  }

  /** Mint the session, arm and connect the host transport, open the page.
   *  Refusals are surfaced here and leave `host` unset, so the next invocation
   *  starts from zero. */
  private async startSession(
    origin: string,
    token: string,
    launchOrigin: string,
    folder: string,
    target?: LocalBrowserTargetIdentity,
    indexProject = true,
    launchBrowser = true,
  ): Promise<void> {
    let session: LocalBrowserSessionMaterial;
    try {
      session = await (this.options.mintSession ?? mintLocalBrowserSession)({
        backendBaseUrl: origin,
        launchToken: token,
        launchOrigin,
      });
    } catch (error) {
      void vscode.window.showWarningMessage(
        `The local browser session could not be created: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
      return;
    }

    const transport = (this.options.createTransport ?? createLocalBrowserHostTransport)({
      backendBaseUrl: origin,
      session,
      log: (line) => this.options.channel.appendLine(`[local-browser host] ${line}`),
    });
    this.host = { session, transport, origin, token };
    transport.onMessage((message) => {
      if ('viewId' in message && message.viewId === this.canvasViewId) return;
      this.onViewMessage(message);
    });
    transport.onStatusChange(status => {
      if (status.phase !== 'connected') {
        // Browser status owns browser views only. MCP runs/proposals belong to
        // their endpoint sessions and are cleared on owner/window teardown.
        this.mcpViews.clear(this.canvasViewId);
        this.codeFocus.disconnectBrowsers(this.canvasViewId);
        // Preserve native UI and MCP reviews; browser-owned UI work still cancels.
        if (!this.canvasReviews.ownsRequest(this.sourceEdits.activeRequestId) && this.canvasDeclarations === 0 &&
            !this.mcpProposals.ownsRequest(this.sourceEdits.activeRequestId)) this.sourceEdits.cancel();
        this.browserReviews.clear();
        if (!this.canvasValidation) this.validation.cancel();
      }
      if (status.phase !== 'connected' && this.host?.transport === transport && !this.captureExport?.native) this.captureExport?.controller.abort();
    });
    // Not awaited: the page must open immediately. A terminal refusal (expired
    // session, wrong role) rejects here, and the next command invocation mints
    // afresh because the stale host is dropped.
    void transport.connect().catch((error: unknown) => {
      if (this.host?.transport === transport) {
        this.host = undefined;
      }
      void vscode.window.showWarningMessage(
        `The local browser session ended before the page could connect to it; ` +
          `run Open Canvas in Browser again to mint a new one (${
            error instanceof Error ? error.message : String(error)
          }).`,
      );
    });

    if (launchBrowser && !(await this.openBrowser(session, origin))) {
      transport.dispose();
      if (this.host?.transport === transport) {
        this.host = undefined;
      }
      void vscode.window.showWarningMessage(
        'The browser could not be opened. The session was disposed of; run Open Canvas in Browser again.',
      );
      return;
    }
    this.options.channel.appendLine(
      target
        ? `Local browser: opened ${target.cls} (${target.file}) in session ${session.workspaceSessionId}.`
        : `Local browser: opened project ${folder} in session ${session.workspaceSessionId}.`,
    );

    if (!target && indexProject) await this.indexProject(folder);
  }

  private async indexProject(folder: string): Promise<void> {
    this.projectOperationGeneration += 1;
    await this.projectOperation(() => this.project!.open(folder));
  }

  /** Never relay backend exceptions, paths or source text to the page or logs. */
  private async projectOperation(operation: () => Promise<unknown>): Promise<void> {
    const folder = this.projectFolder;
    const generation = this.projectOperationGeneration;
    try {
      const result = await operation();
      if (result !== false && generation === this.projectOperationGeneration && this.projectFolder === folder) this.projectError = undefined;
    } catch {
      if (!folder || this.projectFolder !== folder || generation !== this.projectOperationGeneration) return;
      this.project?.invalidateDeclarations();
      this.projectOperationGeneration += 1;
      this.projectError = 'Project graph could not be refreshed. Check the Python environment and backend, then run “tensorViz: Open Project as Graph” in VS Code to retry.';
      this.options.channel.appendLine(this.projectError);
      this.publish({ type: 'error', code: 'INVALID_MESSAGE', message: this.projectError });
      void vscode.window.showWarningMessage(this.projectError);
    }
  }

  private onViewMessage(message: LocalBrowserMessage,
    publish?: (message: LocalBrowserHostPublish) => unknown, native = false): void {
    if (this.sourceDrafts.active) {
      if (message.type === 'graphReviewRequest') { this.sourceDrafts.handle(message); return; }
      if (['sourceEditRequest', 'validateRequest', 'graphSetupRequest', 'viewSelection'].includes(message.type)) return;
      if (message.type === 'viewState' && 'connected' in message && message.connected) this.sourceDrafts.replay();
    }
    if (message.type === 'agentRequest') { this.agent?.handle(message); return; }
    if (message.type === 'viewState' && 'viewId' in message && !message.connected) this.agent?.disconnect(message.viewId);
    const transport = this.host?.transport;
    const reply = publish ?? ((message: LocalBrowserHostPublish) => transport?.publish(message));
    const reviews = native ? this.canvasReviews : this.browserReviews;
    if (message.type === 'graphSetupRequest') {
      if (message.action.kind === 'edit_execution') {
        if (!isGraphSetupAction(message.action)) return;
        const opening = this.opening?.state, review = this.executionReview.state;
        const current = () => !!opening && !!review && this.opening?.state?.id === message.openingId
          && this.executionReview.state?.id === review.id && message.action.kind === 'edit_execution' && message.action.reviewId === review.id
          && !['running', 'cancelling'].includes(this.executionReview.state.phase)
          && sameTargetIdentity(opening.target, message.target) && this.openingSourceCurrent(message.target);
        if (review?.profileName && current()) {
          const setup = new ValidationSetup();
          void setup.reveal({ root: opening!.target.workspaceFolder, session: opening!.id, current },
            ['profiles', review.profileName, ...message.action.field])
            .then(() => { if (current()) this.waitingForSetupSave = true; })
            .catch(() => { void vscode.window.showWarningMessage('Could not open the saved tensorviz.toml. Review model setup and try again.'); })
            .finally(() => setup.dispose());
        }
        return;
      }
      if (message.action.kind === 'execution') {
        if (message.openingId === this.opening?.state?.id) this.executionReview.action(message.target, message.requestId, message.action);
        return;
      }
      if (message.action.kind === 'create_environment') this.options.channel.appendLine(
        `[model environment] Create requested (phase=${this.opening?.state?.phase ?? 'none'}, currentOpening=${message.openingId === this.opening?.state?.id}).`);
      void this.opening?.action(message.openingId, message.requestId, message.target, message.action).then(accepted => {
        if (message.action.kind === 'create_environment') this.options.channel.appendLine(
          `[model environment] Create action finished (accepted=${accepted}, phase=${this.opening?.state?.phase ?? 'none'}).`);
        if (accepted && message.action.kind === 'source_only') {
          this.sync.setTarget(vscode.Uri.file(path.join(message.target.workspaceFolder, message.target.file)), message.target);
        }
      });
      return;
    }
    if (message.type === 'projectFocus') {
      if (this.projectFolder) void this.projectOperation(() => this.project!.focus(message.project, message.experimentId));
      return;
    }
    if (message.type === 'projectExpand') {
      if (this.projectFolder) void this.projectOperation(() => this.project!.expand(message.project, message.nodeId, message.expanded));
      return;
    }
    if (message.type === 'projectDeclare') {
      if (!this.projectFolder) return;
      const folder = this.projectFolder;
      if (native) this.canvasDeclarations++;
      void this.sourceEdits.proposeDeclaration(message).then(async outcome => {
        if (outcome.status === 'applied' && this.projectFolder === folder) await this.indexProject(folder);
        if (outcome.status === 'refused') void vscode.window.showWarningMessage(`tensorViz: ${outcome.detail}`);
        else void vscode.window.showInformationMessage(`tensorViz: ${outcome.detail}`);
      }).catch(() => {
        void vscode.window.showWarningMessage('tensorViz could not complete the declaration review. Inspect the source buffer before retrying.');
      }).finally(() => { if (native) this.canvasDeclarations--; });
      return;
    }
    this.mcpViews.handle(message);
    if (message.type === 'viewState' && 'viewId' in message && !message.connected) this.codeFocus.clear(message.viewId);
    if (message.type === 'viewSelection' && this.mcpViews.displays(message.viewId, message.target)) {
      void this.codeFocus.select(message);
    }
    if (message.type === 'codeFocusRequest') {
      try { void this.codeFocus.configure(this.mcpViews.selection(message.viewId, message.target), message.options); }
      catch { /* A disconnected or obsolete view cannot control an editor. */ }
      return;
    }
    if (message.type === 'viewState' && 'viewId' in message && !message.connected) this.browserReviews.cancelOwner(message.viewId);
    if (message.type === 'candidateRequest') { reviews.control(message); return; }
    if (message.type === 'captureExportRequest') { void this.exportCapture(message, reply, native); return; }
    if (message.type === 'captureExportCancel') {
      const pending = this.captureExport;
      if (pending?.native === native && pending.request.requestId === message.requestId && sameTargetIdentity(pending.request.target, message.target)) pending.controller.abort();
      return;
    }
    if (message.type === 'captureInspectRequest') {
      try {
        const active = this.sync.currentTarget();
        if (!active || this.projectFolder || active.workspaceFolder !== message.target.workspaceFolder
          || active.file !== message.target.file || active.cls !== message.target.cls) {
          throw new GraphToolError('STALE_TARGET', 'The selected model changed. Reopen captured details for this graph.');
        }
        reply({ type: 'captureInspectResult', target: message.target, requestId: message.requestId,
          page: this.captureInspector.inspect(active, message.query), detail: 'Retained capture evidence.' });
      } catch (error) {
        reply({ type: 'captureInspectResult', target: message.target, requestId: message.requestId,
          page: null, detail: error instanceof GraphToolError ? error.message : 'The retained capture could not be inspected.' });
      }
      return;
    }
    if (message.type === 'error') {
      this.options.channel.appendLine(`Local browser: the broker or a view reported ${message.code}.`);
      return;
    }
    if (message.type === 'validateRequest') {
      this.validation.handle(message);
      return;
    }
    if (message.type === 'validateCancel') { this.validation.cancel(message.requestId); return; }
    if (message.type === 'sourceEditRequest') {
      reviews.handle(message);
      return;
    }
    if (message.type === 'sourceEditCancel') {
      reviews.cancel(message.requestId);
      return;
    }
    if (message.type === 'revealNode') {
      // The extension owns the answer: only the sync's cached span can move the
      // cursor, and only when the page names the current identity exactly — a
      // page behind the live state, or aimed at another target, is refused.
      void this.sync
        .revealNode(message.target, message.nodeId)
        .then((result) => {
          // The ack names the asking message's id, per the contract. A session
          // that ended mid-reveal publishes nowhere; publish already refuses.
          reply({
            type: 'ack',
            target: message.target,
            ackId: message.messageId,
            ok: result.ok,
            ...(result.ok ? {} : { errorCode: result.errorCode, detail: result.detail }),
          });
        })
        .catch((error: unknown) => {
          this.options.channel.appendLine(
            `[local-browser host] revealNode failed: ${String(error)}`,
          );
        });
      return;
    }
    if (message.type !== 'snapshotRequest') {
      return;
    }
    if (this.projectFolder) {
      reply({ type: 'snapshot', state: emptyLocalBrowserSnapshotState() });
      void this.project?.republish();
      if (this.projectError) reply({ type: 'error', code: 'INVALID_MESSAGE', message: this.projectError });
      return;
    }
    // The replay the contract pins for a view that (re)connects: one atomic
    // snapshot of the sync's current state. Before any projection exists the
    // state is the legal empty one, followed by the current target as its own
    // ordered message; the live projection lands right after.
    const state = this.sync.snapshotState();
    if (state) {
      const validation = state.activeTarget ? this.validationSummaries.snapshot(state.activeTarget) : undefined;
      reply({ type: 'snapshot', state: { ...state, ...(validation ? { validation } : {}),
        ...(this.opening?.state ? { opening: this.opening.state } : {}) } });
      this.focusCapturedChild();
      return;
    }
    reply({ type: 'snapshot', state: { ...emptyLocalBrowserSnapshotState(),
      ...(this.opening?.state ? { opening: this.opening.state } : {}) } });
    const fallbackTarget = this.sync.currentTarget();
    if (fallbackTarget) {
      reply({ type: 'activeTarget', target: fallbackTarget });
    }
  }

  /** The name errors and the picker's absent-candidate warning use. */
  private relativeName(uri: vscode.Uri): string {
    const folder = vscode.workspace.getWorkspaceFolder(uri)?.uri.fsPath;
    return folder
      ? path.relative(folder, uri.fsPath).split(path.sep).join('/')
      : path.basename(uri.fsPath);
  }

  private async exportCapture(request: LocalBrowserCaptureExportRequestMessage,
    publish: (message: LocalBrowserHostPublish) => unknown, native: boolean): Promise<void> {
    const host = this.host;
    const canvasPost = this.canvasPost;
    const epoch = host?.transport.status.epoch;
    const connected = () => native ? !!canvasPost && this.canvasPost === canvasPost
      : !!host && this.host === host && host.transport.status.epoch === epoch && host.transport.status.phase === 'connected';
    if (!connected()) return;
    const reply = (capture: unknown | null, detail: string) => publish({ type: 'captureExportResult',
      target: request.target, requestId: request.requestId, reference: request.reference, capture, detail });
    if (this.captureExport) { reply(null, 'Another capture export is being prepared. Try again when it finishes.'); return; }
    const pending = { request, controller: new AbortController(), native };
    this.captureExport = pending;
    const retained = () => {
      const active = this.sync.currentTarget();
      if (!connected()
        || this.projectFolder || !active || !sameTargetIdentity(active, request.target)
        || this.captureInspector.list(active).displayedCaptureId !== request.reference.captureId) {
        throw new GraphToolError('STALE_TARGET', 'The displayed capture changed. Export the current graph again.');
      }
      return this.captureInspector.retained(active, request.reference.captureId, request.reference.sourceId);
    };
    try {
      const capture = retained();
      const origin = this.options.backend.origin(), token = this.options.backend.launchToken();
      if (!origin || !token) throw new PortableExportError('Reconnect to VS Code before exporting this capture.');
      const portable = await preparePortableCapture(capture, { origin, token, signal: pending.controller.signal, fetchImpl: this.options.fetchImpl });
      retained(); // Eviction, retargeting or a newer projection cannot race the reply.
      if (!pending.controller.signal.aborted) reply(portable, 'Portable capture prepared without executing the model.');
    } catch (error) {
      if (connected() && !pending.controller.signal.aborted) reply(null,
        error instanceof GraphToolError || error instanceof PortableExportError ? error.message : 'The retained capture could not be exported.');
    } finally { if (this.captureExport === pending) this.captureExport = undefined; }
  }

  dispose(): void {
    this.executionReview.cancel();
    this.agent?.dispose();
    this.opening?.clear();
    for (const disposable of this.openingSubscriptions) disposable.dispose();
    this.captureExport?.controller.abort();
    this.canvasPost = undefined;
    this.project?.invalidateDeclarations();
    this.projectOperationGeneration += 1;
    this.mcpViews.clear();
    this.codeFocus.dispose();
    this.mcpRuns.clear();
    this.mcpProposals.clear();
    this.browserReviews.clear();
    this.canvasReviews.clear();
    this.validation.cancel();
    this.host?.transport.dispose();
    this.host = undefined;
    this.sourceEdits.dispose();
    this.sourceDrafts.dispose();
    this.graphReview?.dispose();
    this.sync.dispose();
  }
}
