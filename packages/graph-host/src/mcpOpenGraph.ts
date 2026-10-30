import { randomUUID } from 'node:crypto';
import * as path from 'node:path';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { graphToolResult, GraphToolError, type JsonObject } from './mcpGraph';
import { INPUT_CONTRACT } from './graphInputContract';
import { isGraphSetupDraft, type GraphOpeningState, type GraphSetupAction } from '@tensorviz/contracts/protocol';
import { graphMcpHelpRequest } from '@tensorviz/contracts/graphHelpRequest';

export const OPEN_GRAPH_CONTINUATION = 'Treat a request to open a graph as a request to finish setup and capture, unless the user explicitly requests source only or cancels. When continuationRequired is true, follow nextAction in the same task without asking the user to repeat the request. Poll progress; inspect repository evidence and submit missing setup; review required dependencies and request native installation approval. Use the environment reported by TensorViz (get_execution_environment is available before opening), never guess from the agent shell. action: install_dependencies installs in that same capture environment and resumes opening. Do not open an environment picker merely because dependencies are missing or installation failed; diagnose and repair that environment first. A source-only preview or needs_setup is not completion. Stop only when validated is true, the user cancels, or a concrete blocker is identified. Report a blocker through action: report_blocker with its reason, observed evidence and nextStep; explain it to the user. Do not end with only "runtime tracing needs setup". Do not repeat unchanged failed actions or bypass declined approval. One successful capture covers only that invocation.';

const blockerSchema = z.object({
  reason: z.string().trim().min(1).max(1024),
  evidence: z.string().trim().min(1).max(2048),
  nextStep: z.string().trim().min(1).max(1024),
}).strict();
type Blocker = z.infer<typeof blockerSchema>;

export interface OpenGraphArgs {
  /** Standalone-only selectors; the companion registration does not advertise them. */
  workspaceRoot?: string;
  interpreter?: string;
  managedRuntime?: boolean;
  profile?: string;
  file?: string;
  cls?: string;
  requestId?: string;
  openingId?: string;
  sourceDigest?: string;
  draft?: Record<string, unknown>;
  action?: 'retry' | 'cancel' | 'environment' | 'request_permission' | 'install_dependencies' | 'report_blocker' | 'select_environment' | 'refresh_environments';
  candidateId?: string;
  packages?: string[];
  blocker?: Blocker;
}
export interface OpenGraphHost {
  open(args: OpenGraphArgs, signal: AbortSignal): Promise<GraphOpeningState>;
  state(): GraphOpeningState | undefined;
  action(state: GraphOpeningState, action: GraphSetupAction): Promise<boolean>;
  model(): JsonObject;
  dependencies?(state: GraphOpeningState, signal: AbortSignal): Promise<JsonObject>;
  install?(state: GraphOpeningState, packages: string[], signal: AbortSignal): Promise<void>;
  focusFailure?(state: GraphOpeningState, owner: string, signal: AbortSignal): Promise<JsonObject>;
  environment?(state: GraphOpeningState, signal: AbortSignal): Promise<boolean>;
  inspectEnvironment?(file: string | undefined, signal: AbortSignal): Promise<JsonObject>;
  permission?(signal: AbortSignal): Promise<void>;
}
interface Request {
  id: string;
  controller: AbortController;
  openingId?: string;
  pending: boolean;
  error?: { code: string; message: string };
  dependencies?: JsonObject;
  dependencyFingerprint?: string;
  failureFocus?: JsonObject;
  target: Pick<OpenGraphArgs, 'file' | 'cls' | 'workspaceRoot' | 'interpreter' | 'profile' | 'managedRuntime'>;
  blocker?: Blocker & { source: 'agent'; sourceDigest?: string };
}

/** Long setup/capture work stays in the host. MCP calls return progress promptly. */
export class McpOpenGraph {
  private readonly requests = new Map<string, Request>();
  private active?: Request;
  constructor(private readonly host: OpenGraphHost) {}

  inspectEnvironment(file: string | undefined, signal: AbortSignal): Promise<JsonObject> {
    if (!this.host.inspectEnvironment) throw new GraphToolError('UNAVAILABLE', 'This host cannot inspect its execution environment.');
    return this.host.inspectEnvironment(file, signal);
  }

  cancelOwner(owner: string): void {
    const request = this.requests.get(owner);
    if (!request) return;
    request.controller.abort();
    const state = this.host.state();
    if (state && state.id === request.openingId && !['ready', 'failed', 'cancelled'].includes(state.phase)) {
      void this.host.action(state, { kind: 'cancel' });
    }
    this.requests.delete(owner);
    if (this.active === request) this.active = undefined;
  }

  invoke(owner: string, args: OpenGraphArgs): JsonObject {
    if (args.packages && args.action !== 'install_dependencies') throw new GraphToolError('INVALID_REQUEST', 'Package requirements need action: install_dependencies.');
    if (args.blocker && args.action !== 'report_blocker') throw new GraphToolError('INVALID_REQUEST', 'A blocker needs action: report_blocker.');
    if (args.candidateId && args.action !== 'select_environment') throw new GraphToolError('INVALID_REQUEST', 'A candidateId needs action: select_environment.');
    let request = this.requests.get(owner);
    if (!args.requestId) {
      if (args.draft || args.action) throw new GraphToolError('REQUEST_REQUIRED', 'Continue the requestId returned by open_graph.');
      const state = this.host.state();
      // A completed capture may still be inspecting dependencies. Its visible
      // retry must supersede that read instead of being rejected as busy.
      const finished = state && state.id === this.active?.openingId && ['ready', 'failed', 'cancelled'].includes(state.phase);
      if (this.active && !finished && (this.active.pending || state && state.id === this.active.openingId
        && ['awaiting_execution', 'resolving', 'saving', 'capturing'].includes(state.phase))) {
        throw new GraphToolError('OPENING_BUSY', 'A graph opening is in progress. Poll its requestId or wait for it to finish.');
      }
      this.cancelOwner(owner);
      request = { id: randomUUID(), controller: new AbortController(), pending: true, target: {
        file: args.file, cls: args.cls, workspaceRoot: args.workspaceRoot, interpreter: args.interpreter, profile: args.profile,
        ...(args.managedRuntime ? { managedRuntime: true } : {}) } };
      this.requests.set(owner, request);
      this.active = request;
      this.work(request, async () => {
        const opened = await this.host.open(args, request!.controller.signal);
        request!.openingId = opened.id;
        if (request!.controller.signal.aborted) await this.host.action(opened, { kind: 'cancel' });
        else await this.inspectDependencies(request!, opened);
      });
    } else {
      if (!request || request.id !== args.requestId) throw new GraphToolError('UNKNOWN_REQUEST', 'This opening belongs to another connection or has expired. Start open_graph again.');
      if (args.file || args.cls || args.workspaceRoot || args.interpreter || args.profile || args.managedRuntime != null) throw new GraphToolError('INVALID_REQUEST', 'Start a new request to select another model, workspace, interpreter or profile.');
      if (args.action === 'cancel') {
        this.cancelOwner(owner);
        return { requestId: request.id, status: 'cancelled', validated: false, continuationRequired: false,
          nextAction: { kind: 'stopped', instruction: 'Opening was cancelled. Do not claim a validated graph.' } };
      }
      if (args.draft || args.action) {
        const state = this.host.state();
        if (request.pending) throw new GraphToolError('OPENING_BUSY', 'Setup is still running. Poll this requestId.');
        if (args.draft && args.action) throw new GraphToolError('INVALID_REQUEST', 'Submit inputs or a recovery action, not both.');
        if (args.action === 'request_permission') {
          if (!this.host.permission || !['APPROVAL_DECLINED', 'APPROVAL_CANCELLED'].includes(request.error?.code ?? '')) {
            throw new GraphToolError('INVALID_REQUEST', 'Permission recovery requires an opening stopped by a permission decision.');
          }
          request.pending = true;
          this.work(request, async () => {
            await this.host.permission!(request!.controller.signal);
            const opened = await this.host.open(request!.target, request!.controller.signal);
            request!.openingId = opened.id;
            if (request!.controller.signal.aborted) await this.host.action(opened, { kind: 'cancel' });
            else await this.inspectDependencies(request!, opened);
          });
          return this.result(request, owner);
        }
        if (args.action === 'report_blocker' && !request.openingId && request.error) {
          this.reportBlocker(request, args.blocker);
          return this.result(request, owner);
        }
        if (!state || state.id !== request.openingId || state.id !== args.openingId || state.target.sourceDigest !== args.sourceDigest) {
          throw new GraphToolError('STALE_TARGET', 'The graph or source changed. Start open_graph again.');
        }
        if (args.draft && !isGraphSetupDraft(args.draft)) throw new GraphToolError('INVALID_SETUP', 'Supply a draft matching inputContract.');
        if (args.action === 'report_blocker') {
          if (!['needs_setup', 'failed', 'source_only'].includes(state.phase) && !(state.phase === 'cancelled' && state.cancellationReason === 'source_changed')) {
            throw new GraphToolError('INVALID_REQUEST', 'Report a blocker only for an incomplete opening waiting for recovery.');
          }
          this.reportBlocker(request, args.blocker, state.target.sourceDigest);
          return this.result(request, owner);
        }
        if (args.draft && !['needs_setup', 'failed'].includes(state.phase)) throw new GraphToolError('OPENING_BUSY', 'This opening is not waiting for inputs.');
        if (args.action === 'install_dependencies' && !['needs_setup', 'failed'].includes(state.phase)) {
          throw new GraphToolError('OPENING_BUSY', 'Wait for this capture to finish before changing dependencies.');
        }
        const inlineInstall = args.action === 'install_dependencies' && state.environment &&
          (state.environment.reason === 'missing_libraries' && state.environment.requirements?.length
            && (!args.packages || JSON.stringify([...args.packages].sort()) === JSON.stringify([...state.environment.requirements].sort()))
            || state.environment.reason === 'incompatible' && state.environment.compatibilityPackages?.length && !args.packages);
        if (args.action === 'select_environment' && !state.environment?.candidates.some(candidate => candidate.id === args.candidateId && candidate.status !== 'unavailable')) {
          throw new GraphToolError('INVALID_REQUEST', 'Select a listed available candidateId from opening.environment. Refresh changed candidates first.');
        }
        const setupAction: GraphSetupAction = args.draft
          ? { kind: 'submit', draft: args.draft as { profile: Record<string, unknown> } }
          : args.action === 'select_environment' ? { kind: 'select_environment', candidateId: args.candidateId! }
          : { kind: args.action === 'install_dependencies' ? inlineInstall ? 'install_dependencies' : 'retry' : args.action! };
        request.pending = true;
        this.work(request, async () => {
          if (args.action === 'environment') {
            if (!await this.host.environment?.(state, request!.controller.signal)) {
              throw new GraphToolError('ENVIRONMENT_NOT_SELECTED', 'Environment selection did not complete. Choose an environment in VS Code and retry.');
            }
            request!.controller.signal.throwIfAborted();
            if (this.host.state()?.id !== state.id) throw new GraphToolError('STALE_TARGET', 'The selected model changed while choosing an environment. Start open_graph again.');
            const opened = await this.host.open({ file: path.join(state.target.workspaceFolder, state.target.file), cls: state.target.cls }, request!.controller.signal);
            request!.openingId = opened.id;
            if (request!.controller.signal.aborted) await this.host.action(opened, { kind: 'cancel' });
            else await this.inspectDependencies(request!, opened);
            return;
          }
          if (args.action === 'install_dependencies' && !inlineInstall) {
            if (!this.host.install || !args.packages?.length) throw new GraphToolError('PACKAGES_REQUIRED', 'Provide PyPI requirements verified from project manifests or package documentation. Do not assume import names equal package names.');
            await this.host.install(state, args.packages, request!.controller.signal);
            await this.inspectDependencies(request!, state);
          }
          request!.controller.signal.throwIfAborted();
          const operation = this.host.action(state, setupAction);
          const expectedId = this.host.state()?.id;
          // Retry creates its new opening synchronously. Bind it now so a
          // disconnect cancels capture while the action is still awaiting it.
          request!.openingId = expectedId;
          const accepted = await operation;
          if (!accepted) throw new GraphToolError('STALE_TARGET', 'The opening could not accept that action. Read its current state.');
          if (this.host.state()?.id !== expectedId) throw new GraphToolError('STALE_TARGET', 'The opening changed during recovery. Call open_graph again for the intended model.');
          request!.openingId = expectedId;
        });
      }
    }
    return this.result(request, owner);
  }

  private work(request: Request, run: () => Promise<void>): void {
    request.error = undefined;
    request.blocker = undefined;
    request.failureFocus = undefined;
    void run().catch(error => {
      request.error = { code: error instanceof GraphToolError ? error.code : 'OPEN_FAILED',
        message: error instanceof GraphToolError ? error.message : 'Could not open the graph. Check the TensorViz environment and saved source.' };
    }).finally(() => { request.pending = false; });
  }

  private async inspectDependencies(request: Request, state: GraphOpeningState): Promise<void> {
    const fingerprint = state.environment?.candidates.find(candidate => candidate.id === state.environment?.selectedId)?.fingerprint;
    const dependencies = await this.host.dependencies?.(state, request.controller.signal);
    request.dependencies = dependencies;
    request.dependencyFingerprint = fingerprint;
  }

  private reportBlocker(request: Request, value: unknown, sourceDigest?: string): void {
    const parsed = blockerSchema.safeParse(value);
    if (!parsed.success) throw new GraphToolError('INVALID_BLOCKER', 'Provide a specific reason, observed evidence (error or source location), and nextStep needed to unblock capture.');
    request.blocker = { ...parsed.data, source: 'agent', sourceDigest };
  }

  private nextAction(request: Request, status: string, current: GraphOpeningState | undefined): JsonObject {
    const identity = { requestId: request.id, ...(current
      ? { openingId: current.id, sourceDigest: current.target.sourceDigest } : {}) };
    const call = (kind: string, instruction: string, args: JsonObject = identity, requiredArguments: string[] = []): JsonObject =>
      ({ kind, tool: 'open_graph', arguments: args, requiredArguments, instruction });
    if (this.host.permission && ['APPROVAL_DECLINED', 'APPROVAL_CANCELLED'].includes(request.error?.code ?? '')) {
      return call('permission_required', 'TensorViz is stopped. If the user requests enabling it, use this action to show their confirmation and resume this opening. This action cannot grant permission. Do not reopen a declined prompt automatically.',
        { requestId: request.id, action: 'request_permission' });
    }
    if (status === 'blocked') return { kind: 'explain_blocker', instruction: 'Explain blocker.reason, evidence and nextStep to the user. Capture is incomplete. Resume only after the blocker is resolved.' };
    if (status === 'source_changed' && current) return call('restart', 'Source changed while opening; this was not a user cancellation. After the edited files finish saving, start a fresh opening of this exact model using its saved setup. If the same interruption repeats without new edits, report_blocker on the interrupted request with that evidence instead of looping.',
      { file: path.join(current.target.workspaceFolder, current.target.file), cls: current.target.cls,
        ...(request.target.workspaceRoot ? { workspaceRoot: request.target.workspaceRoot } : {}),
        ...(request.target.interpreter ? { interpreter: request.target.interpreter } : {}),
        ...(request.target.managedRuntime ? { managedRuntime: true } : {}),
        ...(request.target.profile ? { profile: request.target.profile } : {}) });
    if (status === 'cancelled') return { kind: 'stopped', instruction: 'Opening was cancelled. Do not claim a validated graph or automatically restart a cancelled operation.' };
    if (status === 'ready' && current?.hasCapture) return { kind: 'done', instruction: 'Report the successful capture and its input/mode coverage. This does not validate other inputs or branches.' };
    if (status === 'awaiting_execution') return { kind: 'user_execution', instruction: 'Canvas is waiting for the user to review the execution host, device and workload, then choose Run or Cancel. Show the Canvas and explain that choice. Do not bypass the review or keep polling while waiting for the user. After their decision, poll this requestId.',
      tool: 'open_graph', arguments: { requestId: request.id } };
    if (['opening', 'resolving', 'saving', 'capturing'].includes(status)) {
      return { ...call('poll', 'Wait briefly, then call open_graph with these arguments. Continue until capture succeeds or a concrete blocker is identified.', { requestId: request.id }), pollAfterMs: 1000 };
    }
    if (!current) return call('restart', 'Resolve the returned error or changed target from repository evidence, then start a new open_graph request with the intended file and cls. Do not blindly retry. If recovery requires unavailable information or user action, report_blocker on this request with reason, evidence and nextStep.', { ...request.target });
    if (request.error?.code === 'ENVIRONMENT_REQUIRED') return call('select_environment', 'Choose a working Python environment through the native chooser, then continue this request. If the user declines or no compatible environment is available, report_blocker.', { ...identity, action: 'environment' });
    const environment = current.environment;
    if (environment && environment.reason !== 'ready') {
      if (environment.reason === 'selection_required' || environment.reason === 'unavailable') return call('select_environment',
        'Read opening.environment candidates and choose the model’s intended project Python. Submit its candidateId, or request the native environment chooser. Do not infer Python from your shell or silently substitute a missing saved choice.',
        { ...identity, action: environment.candidates.some(candidate => candidate.status !== 'unavailable') ? 'select_environment' : 'environment' },
        environment.candidates.some(candidate => candidate.status !== 'unavailable') ? ['candidateId'] : []);
      if (environment.reason === 'missing_libraries' && environment.requirements?.length) return call('review_dependencies',
        'These requirements were established from capture libraries and model imports. Request native review of the exact additive plan, then continue this same opening. Existing Torch/build and packages are preserved. Do not repeat declined or unsuccessful repair unchanged.',
        { ...identity, action: 'install_dependencies', packages: environment.requirements });
      return call('inspect_environment_failure',
        'Inspect opening.environment, failure and helpRequest through MCP directly. Choose an appropriate existing environment, investigate unknown imports from project evidence, or request the supported compatibility copy. Do not disguise model errors as missing inputs or retry unchanged failures. Report a concrete blocker if user action or unavailable information is required.');
    }
    if (status === 'needs_setup' && Array.isArray(request.dependencies?.missingModules) && request.dependencies.missingModules.length) {
      return call('review_dependencies', 'Inspect the reported imports and project manifests to establish which packages capture actually requires. For confirmed missing dependencies, fill packages and make this call to request native approval, then continue setup and capture. Optional imports do not require installation. Otherwise submit the completed draft. Do not stop at a dependency list; report_blocker only with concrete evidence and the required next step.', { ...identity, action: 'install_dependencies' }, ['packages']);
    }
    if (status === 'needs_setup') return call('complete_setup', 'Read opening.requirements, unresolved, discovery and the model source. Complete the returned draft using inputContract, then make this call with draft. Missing values are work to resolve, not a final answer. If a required choice or asset cannot be determined, report_blocker with the specific evidence and question or asset needed.', identity, ['draft']);
    if (status === 'source_only') return call('retry', 'Only source is visible. Continue the requested capture through setup; a source-only preview does not complete this task.', { ...identity, action: 'retry' });
    return call('inspect_failure', 'Inspect error, failure, environment and opening.detail. Use get_execution_environment if needed; install required dependencies with install_dependencies in this same capture environment. Do not switch interpreters or open the environment picker just because installation failed. Correct the supported setup or dependency problem before retrying. If capture cannot proceed, report_blocker with reason, observed evidence and nextStep.');
  }

  private result(request: Request, owner: string): JsonObject {
    const state = this.host.state();
    const current = state?.id === request.openingId ? state : undefined;
    const blocker = request.blocker && ((!request.openingId && request.error)
      || current && (['needs_setup', 'failed', 'source_only'].includes(current.phase) || current.phase === 'cancelled' && current.cancellationReason === 'source_changed')
        && current.target.sourceDigest === request.blocker.sourceDigest) ? request.blocker
      : request.error?.code === 'INSTALL_DECLINED' ? { source: 'host', reason: 'Dependency installation was not approved.',
        evidence: request.error.message, nextStep: 'The user must approve the required dependency plan or choose an environment that already has the dependencies. Do not repeat the declined installation.' }
      : request.error?.code === 'ENVIRONMENT_NOT_SELECTED' ? { source: 'host', reason: 'Python environment selection did not complete.',
        evidence: request.error.message, nextStep: 'The user must select a compatible Python environment in VS Code before capture can resume. Do not repeat the environment chooser automatically.' }
      : request.error?.code === 'SOURCE_BUILD_DECLINED' ? { source: 'host', reason: 'Building dependencies from source was not approved.',
        evidence: request.error.message, nextStep: 'The user must approve the source build or provide requirements with compatible wheels. Do not repeat the declined build prompt.' } : undefined;
    const permissionBlocked = this.host.permission && ['APPROVAL_DECLINED', 'APPROVAL_CANCELLED'].includes(request.error?.code ?? '');
    const status = blocker || permissionBlocked ? 'blocked' : request.error ? 'failed' : current?.execution?.phase === 'review' ? 'awaiting_execution' : request.pending ? 'opening'
      : current?.phase === 'cancelled' && current.cancellationReason === 'source_changed' ? 'source_changed' : current?.phase ?? 'superseded';
    const selected = current?.environment?.candidates.find(candidate => candidate.id === current.environment?.selectedId);
    const cached = request.dependencies?.environment as JsonObject | undefined;
    const changed = selected && (request.dependencyFingerprint && request.dependencyFingerprint !== selected.fingerprint
      || cached && (selected.interpreter !== (cached.executable ?? cached.interpreter)
        || selected.python !== cached.python || selected.torch !== cached.torch));
    const environment = changed ? { executable: selected.interpreter, prefix: selected.prefix, python: selected.python,
      torch: selected.torch, architecture: selected.architecture, implementation: selected.implementation, abi: selected.abi } : cached;
    const nextAction = this.nextAction(request, status, current);
    if (current?.phase === 'failed' && current.hasCapture && !request.pending && !request.failureFocus && this.host.focusFailure) {
      request.failureFocus = { status: 'pending' };
      void this.host.focusFailure(current, owner, request.controller.signal).then(result => { request.failureFocus = result; },
        () => { request.failureFocus = { focused: false, detail: 'The Canvas could not confirm focus. Use focus_view on a connected view.' }; });
    }
    return { requestId: request.id, status, validated: status === 'ready' && current?.hasCapture === true,
      continuationRequired: !['blocked', 'cancelled', 'awaiting_execution'].includes(status) && !(status === 'ready' && current?.hasCapture),
      nextAction, next: nextAction.instruction,
      ...(blocker ? { blocker } : {}),
      ...(request.error ? { error: request.error } : {}),
      ...(environment ? { environment } : {}),
      ...(request.dependencies && !changed ? { dependencies: request.dependencies } : {}),
      ...(current && (current.phase === 'failed' || current.environment && current.environment.reason !== 'ready') ? { helpRequest: graphMcpHelpRequest({ opening: current }) } : {}),
      ...(current && request.failureFocus ? { failureFocus: request.failureFocus } : {}),
      ...(current ? { opening: current, ...this.host.model(),
        ...(current.discovery ? { draft: current.draft ?? current.discovery.resolution?.draft ?? null,
          unresolved: current.requirements ?? current.discovery.resolution?.unresolved ?? [], inputContract: INPUT_CONTRACT } : {}) } : {}) };
  }
}

export function registerMcpOpenGraph(server: McpServer, service: McpOpenGraph, owner: string): void {
  server.registerTool('get_execution_environment', {
    description: 'Inspect the selected capture and dependency-install environment: interpreter, source, prefix, Python, Torch and pip versions, platform and architecture. Before selection, returns resolution: selection_required with metadata candidates and an open_graph nextAction; candidates are not install targets. Available before a graph exists. Optional absolute file chooses its resource settings; otherwise uses the current graph or editor. Never installs, selects Python or opens a picker. Use install_dependencies through open_graph after selection; do not infer Python from your shell.',
    inputSchema: { file: z.string().min(1).max(4096).optional() },
    annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
  }, (args, extra) => graphToolResult(() => service.inspectEnvironment(args.file, extra.signal)));
  server.registerTool('open_graph', {
    description: 'Open a PyTorch model in the VS Code Canvas, resolve setup, and capture one validated invocation. Available before any graph exists. Start with an absolute file and optional cls, or use the active Python editor. ' + OPEN_GRAPH_CONTINUATION,
    inputSchema: { file: z.string().min(1).max(4096).optional(), cls: z.string().regex(/^[A-Za-z_]\w*$/).optional(),
      requestId: z.string().uuid().optional(), openingId: z.string().min(1).max(128).optional(),
      sourceDigest: z.string().regex(/^[a-f0-9]{64}$/).optional(), draft: z.record(z.string(), z.json()).optional(),
      action: z.enum(['retry', 'cancel', 'environment', 'install_dependencies', 'report_blocker', 'select_environment', 'refresh_environments']).optional(),
      candidateId: z.string().min(1).max(64).optional(),
      packages: z.array(z.string().min(1).max(200)).min(1).max(32).optional(),
      blocker: blockerSchema.optional().describe('Required with report_blocker: specific reason, observed error or source evidence, and the question, asset or user action needed to continue. Agent-reported evidence is not independently verified.') },
    annotations: { readOnlyHint: false, destructiveHint: true, openWorldHint: true },
  }, args => graphToolResult(() => service.invoke(owner, args)));
}
