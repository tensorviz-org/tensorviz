import { agentGraphContext } from './canvasAgentContext';
import { McpForwarding } from './mcpForwarding';
import { disposeSnapshotHelpers } from '@tensorviz/graph-host/hostSnapshot';
import { openHuggingFace } from './huggingFacePanel';
import { CanvasAgentRuntime, agentExecutable } from './canvasAgentRuntime';
import { CanvasAgentRecording } from './canvasAgentRecording';
import { homedir } from 'node:os';
import { unsupportedRemote } from './remoteSupport';
import { clientOrigin, forwardShellTarget } from './remoteForwarding';
import { registerMcpGraphSetup } from '@tensorviz/graph-host/mcpGraphSetup';
import { registerCanvasAgentPermissionTools } from './canvasAgentPermissionTools';
import { createCanvasAgentProjectTools } from './canvasAgentProjectTools';
import { registerMcpViewTools } from '@tensorviz/graph-host/mcpViews';
import { McpGraphView, registerMcpGraphView } from '@tensorviz/graph-host/mcpGraphView';
import { registerMcpValidationTools } from '@tensorviz/graph-host/mcpValidation';
import { registerMcpCaptureTools } from '@tensorviz/graph-host/mcpCaptures';
import { getModelContext } from '@tensorviz/graph-host/modelContext';
import { registerMcpSourceContextTool } from '@tensorviz/graph-host/mcpSourceContext';
import { registerCanvasCodeTools, registerMcpProposalTools } from './mcpProposals';
import { registerMcpGraphDiffTool } from './mcpGraphDiff';
import { closeMcpConnection, showMcpConnection, updateMcpConnectionStatus } from './mcpConnectionPanel';
import { DEFAULT_MCP_PORT, GraphMcpServer, type McpConnection } from './mcpServer';
import { McpGraphService, GraphToolError } from '@tensorviz/graph-host/mcpGraph';
import { McpOpenGraph, registerMcpOpenGraph } from '@tensorviz/graph-host/mcpOpenGraph';
import { McpLifecycle } from './mcpLifecycle';
import { setUpMcpAgent } from './mcpAgentSetup';
import { ensureModelDependencies, installMcpDependencies, type DependencyInstallOptions } from './mcpDependencies';
import { BusError, HostNode, HostRegistry } from './hostBus';
import { HostGraphHub, LocalGraphProvider, captureTools, type HostToolset } from './mcpHost';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import * as vscode from 'vscode';
import { choosePullRequest, githubReader, loadPullRequest, parsePullRequestUrl } from './githubReview';
import { repositoryForFolder, reviewEditorDiff, reviewFileChanges, workspaceGitFolder } from './editorGraphReview';
import { isHostFile } from './workspaceFiles';
import { runtimeDiagnostic } from '@tensorviz/graph-host/runtimeAcquire';

import { OpenAsGraphLenses } from './openAsGraphLenses';
import { GraphFilePrompt } from './graphFilePrompt';
import { WorkspaceClassIndex, WorkspaceClass } from './workspaceClassIndex';
import { SidebarCanvas } from './sidebarCanvas';
import { OPEN_AS_GRAPH, OPEN_PROJECT, OPEN_PROJECT_IN_BROWSER } from './commandIds';
import { BackendLaunch, devServerUrl, fixedBackendPort } from './config';
import { browserTraceSettings } from './traceSettings';
import { ParseDiagnostics } from './diagnostics';
import { LocalBrowserGraph, type BufferClass } from './localBrowserGraph';
import { graphProfilePreferenceKey } from './graphSetupHost';
import {
  pickRepoPath,
  repoRootFor,
  reserveLoopbackPort,
  resolveBackendDir,
  resolveBackendInterpreter,
  runtimeInstallApprovalPending,
  withoutTraceback,
} from './preflight';
import { registerCanvasCommands } from './graphSurface';
import { pickWorkspaceModel } from './workspaceModelPicker';
import { ShapeInlayHints } from './inlayHints';
import { TOKEN_HEADER } from './projection';
import { CatalogEntry, fetchCatalog, stemFor, templateFor } from './catalog';
import { ManagedRuntime, managedRuntimeFor } from './runtime';
import { ENVIRONMENT_KEY, rememberEnvironment, resolveCaptureEnvironment, selectProjectEnvironment, selectedExecutionEnvironment, inspectExecutionEnvironment, runEnvironmentHelper } from './projectEnvironment';
import { checkCanvasHomeSetup } from './canvasHomeSetup';
import { checkOpeningEnvironment, recoverOpeningEnvironment } from './openingEnvironment';
import { hasValidationConfig, readValidationSnapshot } from './validationHost';
import { Transport } from '@tensorviz/graph-host/runtimeAcquire';
import { VIEW_TYPE, shellTarget } from './shell';
import { OPEN_GRAPH_KEY, graphToRestore, retiredCanvasSerializer, storedOpenGraph, type OpenGraphRecord } from './openGraphState';
import {
  AppMessage,
  GraphOpeningState,
  isGraphOpeningState,
  LocalBrowserTargetIdentity,
} from '@tensorviz/contracts/protocol';
import { BackendProcess, StackError, newLaunchToken } from './stack';
import {
  autoOpenClass,
  declaresModule,
  moduleClasses,
  shouldStartBackend,
} from './activation';
import { registerTracebackLinks } from './tracebackLinks';
import { registerWebsiteCommands } from './websiteCommands';
import { TraceWorker, windowTraceWorker, workspaceTrusted } from './worker';
import {
  MODELS_VIEW,
  MODEL_TEMPLATES,
  classNameFor,
  ModelTemplate,
  ModelsTree,
  TEMPLATES_VIEW,
  TemplatesTree,
  scaffoldModel,
} from './views';

let channel: vscode.OutputChannel;
let sidebarCanvas: SidebarCanvas | undefined;
let backend: BackendProcess | undefined;

/** One automatic backend start per window. A start the user did not ask for must not retry on
 *  every subsequent save; "Open as Graph" and "Repair runtime" remain explicit ways back in. */
let startFailed = false;

/** The start in progress, if any. Three things can ask for a backend in the same second — a
 *  model file being opened, "Open as Graph" — and with a pinned
 *  port (ide/dev.sh) a second spawn is not a spare process but a bind failure that reads as
 *  "the backend crashed". One start at a time; later askers await the same one. */
let backendStarting: Promise<{ repoPath: string; launch: BackendLaunch } | undefined> | undefined;

/** This activation's context, for the handlers that need it and are not called with
 *  one. `resolveBackendDir` reads the extension's own install path, and the panel's
 *  message handler is registered against a view rather than a command, so it has no
 *  context of its own to pass down. */
let extensionContext: vscode.ExtensionContext | undefined;

/** The running backend endpoint, shared by graph and source services. */
const backendEndpoint = {
  origin: () => backend?.origin,
  headers: () =>
    backend ? { [TOKEN_HEADER]: backend.launch.token } : ({} as Record<string, string>),
};

/** The fork's URI scheme, which is how a window knows it is the product rather than an editor
 *  the extension was merely installed into. */
const FORK_URI_SCHEME = 'tensorviz';

/** How catalog.ts reaches the backend: the same origin and per-launch token everything
 *  else uses, plus the two verbs it needs. Kept next to backendEndpoint so a reader sees
 *  one place where the extension decides who it is talking to. */
const catalogHost = {
  origin: () => backend?.origin,
  headers: () => (backend ? { [TOKEN_HEADER]: backend.launch.token } : ({} as Record<string, string>)),
  post: (url: string, headers: Record<string, string>, body: unknown) =>
    backendJson(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify(body),
    }),
  get: (url: string, headers: Record<string, string>) => backendJson(url, { headers }),
};

/** How long the catalogue may take before the picker gives up on it.
 *
 * Generous, because generating a 290-node architecture into source is real work; finite,
 * because the alternative is what this replaced — a fetch with no timeout at all, which
 * left "New model from template" spinning forever behind a progress notification when the
 * backend was not up. A command that cannot fail cannot be cancelled either. */
const CATALOG_TIMEOUT_MS = 60_000;

/** A backend call that always finishes: parsed JSON, or an Error carrying the server's
 *  own words. */
async function backendJson(url: string, init: RequestInit): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CATALOG_TIMEOUT_MS);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    const text = await response.text();
    if (!response.ok) {
      throw new Error(readableBackendError(text, response.status));
    }
    return text ? JSON.parse(text) : undefined;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error(`the backend did not answer within ${CATALOG_TIMEOUT_MS / 1000}s.`);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

/** The server's own message when it has one, rather than a bare status code. */
function readableBackendError(text: string, status: number): string {
  try {
    const parsed = JSON.parse(text) as { detail?: unknown };
    const detail = parsed?.detail;
    if (typeof detail === 'string') return detail;
    if (detail && typeof detail === 'object') {
      const message = (detail as { message?: unknown }).message;
      if (typeof message === 'string') return message;
    }
  } catch {
    // Not JSON. The status is all there is.
  }
  return `the backend answered ${status}.`;
}

const AUTO_OPEN_SETTING = 'autoOpenCanvas';
/** `tensorviz.followEditor` (step 3.4). A setting, not a memento: which file you
 *  are reading is a preference about the camera, and preferences are configured
 *  where the user can find and reverse them. */
const FOLLOW_EDITOR_SETTING = 'followEditor';
let diagnostics: ParseDiagnostics | undefined;
let lenses: WorkspaceClassIndex | undefined;
let inlayHints: ShapeInlayHints | undefined;
/** The external-browser half of "Open as Graph" (extension-webpage step 4):
 *  per-window local-browser session, its host transport, and the target the
 *  page is anchored on. Module-scoped for the same reason as the lens index:
 *  registered once, reused by every command invocation after. */
let browserGraph: LocalBrowserGraph | undefined;
const agentProjectTools = new Map<string, ReturnType<typeof createCanvasAgentProjectTools>>();

/** What the Electron tests hook into. Nothing in the product reads it. */
export interface TestApi {
  messages(): AppMessage[];
  backendOrigin(): string | undefined;
  launchToken(): string | undefined;
  post(message: AppMessage): Promise<boolean>;
  /** The classes the CodeLens provider believes a folder holds. */
  classesIn(root: string): WorkspaceClass[] | undefined;
  /** What the Problems panel shows for a file, without screen-scraping it. */
  diagnostics(uri: vscode.Uri): readonly vscode.Diagnostic[];
  /** The live local-browser session, and the target it currently anchors on —
   *  what step 4's Electron suite reads instead of screen-scraping a browser. */
  localBrowserSession(): { workspaceSessionId: string; pageUrl: string } | undefined;
  localBrowserTarget(): LocalBrowserTargetIdentity | undefined;
  /** What the broker has said about this session's browser tabs. A QA driver cannot
   *  see a page close from outside, and "opened nothing" and "believed a page was
   *  there" look identical without it. */
  localBrowserViewPresence(): string | undefined;
  /** The extension's globalStorage directory, so Electron suites that exercise the managed
   *  runtime can clear or inspect its root without guessing at the profile layout. */
  globalStorageDir(): string | undefined;
  /** Re-run the lazy test preseed after a suite deleted the runtime root to stage a
   *  first-install leg — the later suites must find the seeded ready runtime back. */
  seedManagedRuntimeForTests(): void;
  /** True while the anchor's save-coalesced trace is armed or running: the
   *  identity the page displays can still roll backendRevision, so the
   *  navigation Electron leg must not name an identity yet. False is
   *  terminal — runtime-traced when the trace landed, honestly source-only
   *  when this workspace cannot trace. */
  localBrowserTracePending(): boolean;
  mcpConnection(): McpConnection | undefined;
  sidebarState(): ReturnType<SidebarCanvas['inspectForTest']> | undefined;
  graphReviewState(): import('@tensorviz/contracts/protocol').GraphReviewState | undefined;
  graphOpeningState(): import('@tensorviz/contracts/protocol').GraphOpeningState | undefined;
}

const testApi: TestApi = {
  messages() {
    return sidebarCanvas?.inspectForTest().messages ?? [];
  },
  classesIn(root) {
    return lenses?.classesFor(root);
  },
  diagnostics(uri) {
    return diagnostics?.get(uri) ?? [];
  },
  backendOrigin() {
    return backend?.origin;
  },
  launchToken() {
    return backend?.launch.token;
  },
  /** Deliver a message as if the app had posted it, which is the only way to
   *  exercise the host services without driving the real canvas. */
  async post(message) {
    return sidebarCanvas?.handleForTest(message) ?? false;
  },
  localBrowserSession() {
    return browserGraph?.session;
  },
  localBrowserViewPresence() {
    return browserGraph?.viewPresence;
  },
  globalStorageDir() {
    return extensionContext?.globalStorageUri.fsPath;
  },
  seedManagedRuntimeForTests() {
    if (extensionContext) {
      seedManagedRuntimeForTests(managedBackendRuntime(extensionContext));
    }
  },
  localBrowserTarget() {
    return browserGraph?.target;
  },
  mcpConnection() { return graphMcp?.connection; },
  sidebarState() { return sidebarCanvas?.inspectForTest(); },
  graphReviewState() { return browserGraph?.reviewState; },
  graphOpeningState() { return browserGraph?.opening?.state; },
  localBrowserTracePending() {
    return browserGraph?.tracePending ?? false;
  },
};

let graphMcp: GraphMcpServer | undefined;
let desktopMcp: McpForwarding | undefined;
let mcpStarting: Promise<McpConnection> | undefined;
let mcpLifecycle: McpLifecycle | undefined;
let hostGraphNode: HostNode | undefined;
let hostGraphService: McpGraphService | undefined;
let mcpHostRegistry: HostRegistry | undefined;
let hostRegistryDir = '';
let windowHostId = '';
let mcpOpening: McpOpenGraph | undefined;
let mcpGraphView: McpGraphView | undefined;

function windowOpening(): McpOpenGraph {
  return mcpOpening ??= new McpOpenGraph({
    pending: () => runtimeInstallApprovalPending() ? { kind: 'runtime_install_approval',
      detail: 'Approve the runtime installation in VS Code to continue opening. Run “tensorViz: Repair runtime” if the prompt is hidden.' } : undefined,
    inspectEnvironment: async (file, signal) => {
      const context = extensionContext!;
      if (file && !path.isAbsolute(file)) throw new GraphToolError('ABSOLUTE_PATH_REQUIRED', 'Supply an absolute model file path.');
      const target = browserGraph?.opening?.state?.target;
      const resource = file ? vscode.Uri.file(file) : target ? vscode.Uri.file(path.join(target.workspaceFolder, target.file))
        : vscode.window.activeTextEditor?.document.uri ?? vscode.workspace.workspaceFolders?.[0]?.uri ?? context.globalStorageUri;
      const resolved = resolveBackendDir(context);
      if (!resolved.ok) return { ok: false, detail: 'The bundled TensorViz backend is unavailable. No environment was changed.' };
      return inspectExecutionEnvironment(context, resolved.backendDir, { resource, signal, managerPython: backend?.interpreter,
        readyManaged: async () => {
          const managed = await managedBackendRuntime(context).check();
          return managed.state === 'ready' ? managed.interpreter : undefined;
        } });
    },
    open: async (args, signal) => {
      const active = vscode.window.activeTextEditor?.document;
      if (args.file && !path.isAbsolute(args.file)) throw new GraphToolError('ABSOLUTE_PATH_REQUIRED', 'Supply an absolute model file path.');
      const uri = args.file ? vscode.Uri.file(args.file) : active?.uri;
      if (!uri || !isHostFile(uri) || !/\.pyw?$/i.test(uri.fsPath)) {
        throw new GraphToolError('MODEL_REQUIRED', 'Supply an absolute Python file path or focus a saved Python editor in the intended VS Code window.');
      }
      const folder = vscode.workspace.getWorkspaceFolder(uri);
      if (!folder && uri.fsPath !== active?.uri.fsPath) throw new GraphToolError('WORKSPACE_REQUIRED', 'Open the model folder in VS Code before requesting its graph.');
      if (!vscode.workspace.isTrusted) throw new GraphToolError('WORKSPACE_UNTRUSTED', 'Trust the workspace in VS Code before opening and executing its model.');
      const graph = browserGraph;
      if (!graph?.opening || !sidebarCanvas) throw new GraphToolError('NO_BACKEND', 'The TensorViz backend is unavailable in this installation.');
      if (!await sidebarCanvas.show()) throw new GraphToolError('ENVIRONMENT_REQUIRED', 'Select a working Python environment in VS Code, then call open_graph again.');
      signal.throwIfAborted();
      let cls = args.cls;
      if (!cls) {
        const root = folder?.uri.fsPath ?? path.dirname(uri.fsPath);
        const relative = path.relative(root, uri.fsPath).split(path.sep).join('/');
        const choices = (await workspaceClasses(root)).filter(entry => entry.file === relative);
        signal.throwIfAborted();
        if (choices.length !== 1) throw new GraphToolError('CLASS_REQUIRED',
          choices.length ? `Choose cls from this file: ${choices.map(entry => entry.name).join(', ')}.`
            : 'No unambiguous model class was found. Read the model source and supply cls explicitly.');
        cls = choices[0].name;
      }
      signal.throwIfAborted();
      await graph.open({ uri: uri.toString(), cls, restart: true }, 'canvas', 'agent');
      const state = graph.opening.state;
      if (!state || path.join(state.target.workspaceFolder, state.target.file) !== uri.fsPath || state.target.cls !== cls) {
        throw new GraphToolError('OPEN_FAILED', 'The requested model could not be opened. Check the TensorViz output in VS Code.');
      }
      return state;
    },
    state: () => browserGraph?.opening?.state,
    action: async (state, action) => await browserGraph?.opening?.action(state.id, randomUUID(), state.target, action) ?? false,
    environment: async (state, signal) => !signal.aborted && await checkEnvironmentCommand(extensionContext!, true, false,
      vscode.Uri.file(path.join(state.target.workspaceFolder, state.target.file))) && !signal.aborted,
    model: () => ({ ...windowGraph().list(), failure: browserGraph?.mcpModel?.projection.traceError ?? null }),
    focusFailure: async (state, owner, signal) => {
      const model = browserGraph?.mcpModel, viewId = sidebarCanvas?.connectedViewId;
      const nodeId = model?.projection.traceError?.nodeId;
      if (!model?.current || !nodeId || !viewId || model.target.sourceDigest !== state.target.sourceDigest
        || model.target.cls !== state.target.cls || model.target.file !== state.target.file) {
        return { focused: false, detail: 'No current mapped failure is available in the visible Canvas.' };
      }
      const graph = windowGraph();
      graph.select(graph.require(graph.reference(model)).nodes, [nodeId]);
      return browserGraph!.mcpViews.focus(owner, { viewId, target: model.target, nodeIds: [nodeId], edgeIds: [] }, signal);
    },
    dependencies: async (state, signal) => {
      const context = extensionContext!;
      const resolved = resolveBackendDir(context);
      const file = path.join(state.target.workspaceFolder, state.target.file);
      const selected = await executionEnvironment(context, vscode.Uri.file(file));
      if (!resolved.ok || !selected.interpreter || signal.aborted) return { ok: false, environment: selected, detail: 'The selected TensorViz execution environment is unavailable.' };
      const result = await runEnvironmentHelper(backend?.interpreter ?? selected.interpreter, resolved.backendDir,
        [...(backend ? ['--target-python', selected.interpreter] : []),
          'inspect-model', state.target.workspaceFolder, file], () => signal.aborted);
      return { ...result, environment: { ...result.environment, ...selected } };
    },
    install: async (state, packages, signal) => {
      if (browserGraph?.opening?.state?.id !== state.id) throw new GraphToolError('STALE_TARGET', 'The graph opening changed. Start again.');
      await installMcpDependencies({ ...await modelDependencyOptions(extensionContext!, state.target, signal), packages });
    },
  });
}

async function modelDependencyOptions(context: vscode.ExtensionContext, target: LocalBrowserTargetIdentity,
  signal: AbortSignal): Promise<Omit<DependencyInstallOptions, 'packages'>> {
  const approve = browserGraph?.agentPermissionRequester(target);
  const openingId = browserGraph?.opening?.state?.id;
  const resolved = resolveBackendDir(context);
  const uri = vscode.Uri.file(path.join(target.workspaceFolder, target.file));
  const python = await executionInterpreter(context, uri);
  if (!resolved.ok || !python) throw new GraphToolError('ENVIRONMENT_REQUIRED', 'Select a Python environment before installing model dependencies.');
  return { python, managerPython: backend?.interpreter, backendDir: resolved.backendDir, storage: context.globalStorageUri.fsPath, signal,
    approve,
    current: async () => {
      const live = browserGraph?.opening?.state;
      const document = await vscode.workspace.openTextDocument(uri);
      const dirty = vscode.workspace.textDocuments.some(doc => {
        const relative = path.relative(target.workspaceFolder, doc.uri.fsPath);
        return doc.isDirty && isHostFile(doc.uri) && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative)
          && (/\.pyw?$/i.test(doc.uri.fsPath) || ['tensorviz.toml', 'pyproject.toml', 'requirements.txt'].includes(path.basename(doc.uri.fsPath)));
      });
      return vscode.workspace.isTrusted && !!live && live.id === openingId && live.target.sourceDigest === target.sourceDigest && !dirty
        && !document.isDirty && createHash('sha256').update(document.getText()).digest('hex') === target.sourceDigest
        && await executionInterpreter(context, uri) === python;
    } };
}

/** This window's graph service, shared by the MCP endpoint and the host bus. */
function windowGraph(): McpGraphService {
  return hostGraphService ??= new McpGraphService(windowHostId,
    () => browserGraph?.mcpModel, () => browserGraph?.mcpViews.list() ?? []);
}

/** Bind every tool to the originating MCP connection, including remote calls. */
function windowTools(owner: string, canvasCode = false): HostToolset {
  const surface = browserGraph;
  const graph = windowGraph();
  return captureTools(server => {
    graph.register(server);
    registerMcpOpenGraph(server, windowOpening(), owner);
    registerMcpGraphDiffTool(server, {
      target: () => browserGraph?.mcpModel?.target,
      busy: () => browserGraph?.busy ?? false,
      show: async () => await sidebarCanvas?.show() ?? false,
      review: async (session, signal) => browserGraph?.reviewSession(session, signal),
      current: () => browserGraph?.reviewState,
    });
    registerMcpGraphView(server, mcpGraphView ??= new McpGraphView(graph, () => browserGraph), owner);
    if (surface?.opening && surface.setupHost) registerMcpGraphSetup(server, surface.opening, surface.setupHost,
      canvasCode ? () => surface.agentPermissions.pending() : undefined);
    if (!surface) return;
    registerMcpCaptureTools(server, graph, surface.captureInspector,
      (ref, file) => surface.mcpProposals.readSource(graph, ref, owner, file));
    registerMcpSourceContextTool(server, graph, surface.captureInspector,
      (ref, query) => surface.mcpProposals.querySourceContext(graph, ref, query));
    registerMcpViewTools(server, graph, surface.mcpViews, owner);
    registerMcpValidationTools(server, graph, surface.mcpRuns, owner);
    registerMcpProposalTools(server, graph, surface.mcpProposals, owner);
    if (canvasCode) {
      registerCanvasCodeTools(server, graph, surface.mcpProposals, owner);
      registerCanvasAgentPermissionTools(server, surface.agentPermissions);
      let tools = agentProjectTools.get(owner);
      if (!tools) {
        tools = createCanvasAgentProjectTools({ capture: () => {
          if (!browserGraph) throw new Error('No Canvas agent task is active.');
          return browserGraph.captureAgentOperation();
        } });
        agentProjectTools.set(owner, tools);
      }
      tools.register(server);
    }
  });
}

function releaseMcpOwner(owner: string): void {
  agentProjectTools.get(owner)?.dispose(); agentProjectTools.delete(owner);
  mcpOpening?.cancelOwner(owner);
  mcpGraphView?.cancelOwner(owner);
  browserGraph?.mcpViews.cancelOwner(owner);
  browserGraph?.mcpRuns.cancelOwner(owner);
  browserGraph?.mcpProposals.cancelOwner(owner);
}

function windowDescription() {
  return { ...windowGraph().list(),
    workspaceFolders: (vscode.workspace.workspaceFolders ?? []).map(folder => folder.uri.fsPath),
    activeFile: vscode.window.activeTextEditor?.document.uri.fsPath,
    focused: vscode.window.state.focused,
    mcp: graphMcp?.connection ? { ...graphMcp.connection, sessions: graphMcp.sessionCount } : undefined };
}

function hostRegistry(): HostRegistry {
  return mcpHostRegistry ??= new HostRegistry(hostRegistryDir);
}

function createGraphMcpServer(): GraphMcpServer {
  return new GraphMcpServer((server, sessionId) => {
    const owner = `${windowHostId}/${sessionId}`;
    const tools = windowTools(owner);
    const hub = new HostGraphHub({
      hostId: windowHostId, owner,
      workspaceFolder: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? null,
      registry: hostRegistry(), tools, listLocal: windowDescription,
      remote: new Set(tools.keys()),
    });
    hub.register(server);
    return () => { releaseMcpOwner(owner); void hub.dispose(); };
  }, windowHostId);
}

async function connectGraphMcp(port: number): Promise<McpConnection> {
  const unsupported = unsupportedRemote(vscode.env.remoteName);
  if (unsupported) throw new Error(unsupported);
  if (mcpStarting) return mcpStarting;
  if (graphMcp?.connection) return graphMcp.connection;
  const endpoint = createGraphMcpServer();
  graphMcp = endpoint;
  mcpStarting = endpoint.start(port);
  try { return await mcpStarting; }
  catch (error) {
    await endpoint.close();
    if (graphMcp === endpoint) graphMcp = undefined;
    throw error;
  }
  finally { mcpStarting = undefined; }
}

async function startHostBus(context: vscode.ExtensionContext): Promise<void> {
  const provider = new LocalGraphProvider(windowTools, windowDescription, releaseMcpOwner);
  const node = new HostNode({
    hostId: windowHostId,
    workspaceFolder: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? null,
    registryDir: hostRegistryDir,
    handler: async (method, params, signal) => {
      if (method === 'describe') return provider.list();
      if (method === 'sync') { await mcpLifecycle?.synchronize(); return {}; }
      const { name, args, owner } = (params ?? {}) as { name?: string; args?: unknown; owner?: string };
      if (typeof owner !== 'string' || !/^[\w-]{1,64}\/[\w-]{1,64}$/.test(owner)) {
        throw new BusError('INVALID_OWNER', 'A routed call requires its originating MCP connection.');
      }
      if (method === 'release') { provider.release(owner); return {}; }
      return provider.call(String(name), args, signal, owner);
    },
  });
  await node.start();
  hostGraphNode = node;
  const reap = setInterval(() => {
    void node.registry.list().then(records => provider.reap(new Set(records.map(record => record.hostId))));
  }, 10_000);
  reap.unref();
  context.subscriptions.push({ dispose: () => { clearInterval(reap); provider.reap(new Set()); } });
  context.subscriptions.push({ dispose: () => { void hostGraphNode?.stop(); } });
}

async function stopGraphMcp(): Promise<void> {
  const endpoint = graphMcp;
  if (endpoint) await endpoint.close();
  if (graphMcp === endpoint) graphMcp = undefined;
}

export function activate(context: vscode.ExtensionContext): { __test: TestApi } {
  extensionContext = context;
  channel = vscode.window.createOutputChannel('tensorViz');
  // Every window announces its live graph on a loopback bus so the one window
  // that owns the MCP endpoint can list and read the others. The identity is set
  // synchronously so a failed listener never leaves the window anonymous.
  windowHostId = randomUUID();
  hostRegistryDir = path.join(context.globalStorageUri.fsPath, 'hostbus');
  mcpHostRegistry = new HostRegistry(hostRegistryDir);
  const hostBusReady = startHostBus(context);
  void hostBusReady.catch(error => channel?.appendLine(`[hostbus] ${String(error)}`));
  context.subscriptions.push(channel, { dispose: () => { void stopGraphMcp(); } });
  desktopMcp = new McpForwarding(createGraphMcpServer, clientOrigin);
  context.subscriptions.push({ dispose: () => { void desktopMcp?.close(); } });
  const mcpItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 99);
  mcpItem.command = 'tensorviz.mcpMenu';
  mcpItem.name = 'TensorViz coding agents';
  mcpItem.text = '$(plug) TensorViz: Off';
  mcpItem.tooltip = 'Connect a coding agent to TensorViz';
  mcpItem.show();
  mcpLifecycle = new McpLifecycle({
    directory: context.globalStorageUri.fsPath, hostId: windowHostId, registry: hostRegistry(),
    start: async port => { await hostBusReady; return connectGraphMcp(port); }, stop: stopGraphMcp,
    local: () => graphMcp?.connection ? { ...graphMcp.connection, sessions: graphMcp.sessionCount } : undefined,
    publish: status => {
      updateMcpConnectionStatus(status);
      if (status.state === 'Off') void desktopMcp?.close();
      const sessions = status.sessions + (desktopMcp?.sessionCount ?? 0);
      const state = status.state !== 'Off' && sessions > 0 ? 'Connected' : status.state;
      mcpItem.text = `${state === 'Connected' ? '$(debug-connected)' : '$(plug)'} TensorViz: ${state}`;
      mcpItem.tooltip = status.error ?? (status.url
        ? `${status.url} · ${sessions} agent connection(s). Click for setup or disconnect.`
        : 'Connect a coding agent to TensorViz');
    },
  });
  const helpContext = async (opening?: GraphOpeningState) => {
    if (!isGraphOpeningState(opening) || opening.phase !== 'failed' && opening.phase !== 'needs_setup'
      && !(opening.environment && opening.environment.reason !== 'ready')) return undefined;
    const snapshot = JSON.parse(JSON.stringify(opening)) as GraphOpeningState;
    const resource = vscode.Uri.file(path.join(snapshot.target.workspaceFolder, snapshot.target.file));
    const selected = await selectedExecutionEnvironment(context, resource, async () => {
      const result = await managedBackendRuntime(context).check();
      return result.state === 'ready' ? result.interpreter : undefined;
    });
    return { opening: snapshot, interpreter: selected.interpreter, environmentSource: selected.source };
  };
  const connect = async (failedOpening?: GraphOpeningState, configureAgent = false) => {
    try {
      const help = await helpContext(failedOpening);
      const port = vscode.workspace.getConfiguration('tensorviz').inspect<number>('mcpPort')?.globalValue ?? DEFAULT_MCP_PORT;
      await mcpLifecycle!.setEnabled(true, port);
      const status = mcpLifecycle!.status;
      if (!status.url) throw new Error(status.error ?? 'MCP could not start.');
      const target = help?.opening.target ?? browserGraph?.opening?.state?.target;
      if (vscode.env.remoteName === 'ssh-remote') showMcpConnection(status.url, context, target, 'ssh', undefined, help);
      else if (help && !configureAgent) showMcpConnection(status.url, context, target, 'local', undefined, help);
      else await setUpMcpAgent(status.url, agent => showMcpConnection(status.url!, context, target, 'local', agent, help));
    } catch (error) {
      void vscode.window.showErrorMessage(error instanceof Error ? error.message : 'TensorViz MCP could not start.');
    }
  };
  const forwardMcp = async (failedOpening?: GraphOpeningState) => {
    if (vscode.env.remoteName !== 'ssh-remote') {
      void vscode.window.showWarningMessage(unsupportedRemote(vscode.env.remoteName) ?? 'Use Connect coding agent (MCP) for a local desktop window.');
      return;
    }
    try {
      const help = await helpContext(failedOpening);
      const port = vscode.workspace.getConfiguration('tensorviz').inspect<number>('mcpPort')?.globalValue ?? DEFAULT_MCP_PORT;
      await mcpLifecycle!.setEnabled(true, port);
      if (!mcpLifecycle!.status.url) throw new Error('MCP could not start on the SSH host.');
      const url = await desktopMcp!.open();
      showMcpConnection(url, context, help?.opening.target ?? browserGraph?.opening?.state?.target, 'desktop', undefined, help);
    } catch {
      void vscode.window.showErrorMessage('MCP forwarding could not be established. Reconnect Remote SSH and run Forward MCP to This Desktop again.');
    }
  };
  const disconnect = async () => {
    await mcpLifecycle!.setEnabled(false);
    closeMcpConnection();
    void vscode.window.showInformationMessage('TensorViz MCP disconnected in all windows on this host. Reconnect from the status bar.');
  };
  context.subscriptions.push(mcpItem, { dispose: () => { void mcpLifecycle?.dispose(); } },
    vscode.commands.registerCommand('tensorviz.connectMcp', connect),
    vscode.commands.registerCommand('tensorviz.disconnectMcp', disconnect),
    vscode.commands.registerCommand('tensorviz.forwardMcp', forwardMcp),
    vscode.commands.registerCommand('tensorviz.mcpMenu', async () => {
      const status = mcpLifecycle!.status;
      const choice = await vscode.window.showQuickPick(status.url
        ? ['Set up coding agent…', 'Copy MCP URL', ...(vscode.env.remoteName === 'ssh-remote' ? ['Forward MCP to This Desktop'] : []), 'Disconnect'] : ['Connect coding agent…'],
      { title: `TensorViz MCP · ${status.state}`, placeHolder: status.error ?? (vscode.env.remoteName === 'ssh-remote' ? 'Remote-host agents use the SSH endpoint; desktop agents need explicit forwarding' : 'One connection for all local projects') });
      if (choice === 'Forward MCP to This Desktop') await forwardMcp();
      else if (choice === 'Disconnect') await disconnect();
      else if (choice === 'Copy MCP URL' && status.url) await vscode.env.clipboard.writeText(status.url);
      else if (choice) await connect();
    }),
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('tensorviz.repairRuntime', () => repairRuntime(context)),
    // A Python traceback in the integrated terminal becomes a link to the card that raised.
    registerTracebackLinks({
      workspaceRoot: () => vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
      target: () => {
        const model = browserGraph?.mcpModel;
        if (!model?.current) return undefined;
        return { uri: vscode.Uri.file(path.join(model.target.workspaceFolder, model.target.file)),
          cls: model.target.cls, rel: model.target.file, spanmap: model.projection.spanmap,
          sourceDigest: model.target.sourceDigest };
      },
      reveal: async (frame) => {
        const document = await vscode.workspace.openTextDocument(vscode.Uri.file(frame.file));
        const editor = await vscode.window.showTextDocument(document);
        const line = Math.max(0, frame.line - 1);
        editor.selection = new vscode.Selection(line, 0, line, 0);
        editor.revealRange(new vscode.Range(line, 0, line, 0));
      },
      focusNode: (nodeId) => browserGraph?.focusSourceNode(nodeId),
      reportTrace: (frame, failure) =>
        diagnostics?.reportTrace(vscode.Uri.file(frame.file), failure),
    }),
  );

  // Runtime selection and any installation happen only in backend-needing commands.
  // Compatible project environments are reused; downloads and missing libraries need consent.

  diagnostics = new ParseDiagnostics();
  lenses = new WorkspaceClassIndex((root) => workspaceClasses(root));
  context.subscriptions.push(diagnostics, lenses);
  // The external half of "Open as Graph" (extension-webpage step 4): the window's own
  // local-browser session and native sidebar transport share one controller.
  // Editor panels remain available for graph comparisons.
  //
  // The executed tier behind the page (step 5): the same consent gate and settings the
  // embedded canvas answers to, and a worker of its own so the two surfaces never cancel
  // each other's runs. The interpreter is read against the traced file, not a fixed
  // folder, because a browser session can follow a class in any workspace folder.
  // No modal here: activation must not pop dialogs, and an unresolvable backend is a
  // source-only page, not a dialog — the start flows that ARE user-invoked say why.
  const browserBackend = resolveBackendDir(context);
  const browserInterpreter = (resource: vscode.Uri) => executionInterpreter(context, resource);
  const browserWorker = browserBackend.ok
    ? windowTraceWorker(context, browserBackend.backendDir, channel, { interpreter: browserInterpreter })
    : undefined;
  const candidateWorker = browserBackend.ok
    ? windowTraceWorker(context, browserBackend.backendDir, channel, { interpreter: browserInterpreter }) : undefined;
  // Execution cache (`tensorviz.executionCache.mode`): stored results live in these two
  // workers' memory only, so clearing them is the whole cache for this window.
  context.subscriptions.push(vscode.commands.registerCommand('tensorviz.clearExecutionCache', () => {
    const cleared = (browserWorker?.clearExecutionCache() ?? 0) + (candidateWorker?.clearExecutionCache() ?? 0);
    void vscode.window.showInformationMessage(cleared
      ? `tensorViz: Cleared ${cleared} reused result${cleared === 1 ? '' : 's'}. The next Run of each model executes.`
      : 'tensorViz: The execution cache is empty.');
  }));
  // Resolve saved profiles with the exact worker policy, including an explicitly
  // selected managed runtime. The retained plan prevents switching Python afterward.
  const profileLookup = browserBackend.ok ? {
    hasConfig: hasValidationConfig,
    preferred: async (root: string, target: { file: string; cls: string }) =>
      context.workspaceState.get<string>(graphProfilePreferenceKey(root, target)),
    remember: async (root: string, target: { file: string; cls: string }, name: string) =>
      context.workspaceState.update(graphProfilePreferenceKey(root, target), name),
    snapshot: async (root: string, profile: string | undefined, signal: AbortSignal) => {
      // Absence was checked before interpreter selection; failures here must stop execution.
      const python = await browserInterpreter(vscode.Uri.file(root));
      if (!python) throw new Error('no execution interpreter is available for this folder');
      const documents: Record<string, { version: number; dirty: boolean }> = {};
      for (const doc of vscode.workspace.textDocuments) {
        if (!isHostFile(doc.uri)) continue;
        const relative = path.relative(root, doc.uri.fsPath).split(path.sep).join('/');
        if (relative && !relative.startsWith('..') && !path.isAbsolute(relative)) {
          documents[relative] = { version: doc.version, dirty: doc.isDirty };
        }
      }
      return readValidationSnapshot(python, browserBackend.backendDir, {
        root, documents, executable: python, environmentRevision: 0,
        session: VALIDATION_SESSION_ID, workspaceId: createHash('sha256').update(root).digest('hex'),
        runId: randomUUID(), generation: 1, ...(profile ? { profile } : {}),
      }, signal);
    },
    choose: async (names: string[], signal: AbortSignal) => {
      const cancellation = new vscode.CancellationTokenSource();
      const abort = () => cancellation.cancel();
      signal.addEventListener('abort', abort, { once: true });
      if (signal.aborted) abort();
      try {
        return await vscode.window.showQuickPick(names,
          { title: 'More than one saved profile measures this class', ignoreFocusOut: true }, cancellation.token);
      } finally {
        signal.removeEventListener('abort', abort);
        cancellation.dispose();
      }
    },
  } : undefined;
  const workerCache = new Set<string>();
  const environmentOptions = (target: LocalBrowserTargetIdentity, signal: AbortSignal) => {
    if (!browserBackend.ok) throw new GraphToolError('NO_BACKEND', browserBackend.message);
    return { context, target, signal, backendDir: browserBackend.backendDir, managerPython: backend?.interpreter, workerCache,
      approve: browserGraph?.agentPermissionRequester(target),
      diagnostic: (message: string) => channel.appendLine(message),
      readyManaged: async () => { const result = await managedBackendRuntime(context).check(); return result.state === 'ready' ? result.interpreter : undefined; },
    };
  };
  let agentRecording: CanvasAgentRecording | undefined;
  const recordingDirectory = process.env.TENSORVIZ_AGENT_RECORDING_DIR?.trim();
  if (recordingDirectory) {
    try {
      agentRecording = new CanvasAgentRecording(recordingDirectory, { extensionVersion: context.extension.packageJSON.version,
        extensionBundleSha256: createHash('sha256').update(fs.readFileSync(path.join(context.extensionPath, 'out', 'extension.js'))).digest('hex'),
        platform: process.platform, architecture: process.arch, timingScope: 'local_host_observation',
        modelIdentity: 'requested_settings_only', qualityJudgment: 'ungraded' }, message => channel.appendLine(message));
      context.subscriptions.push(agentRecording);
      channel.appendLine(`Local agent test recording: ${agentRecording.file}`);
    } catch (error) { channel.appendLine(`Agent recording could not start: ${String(error)}`); }
  }
  const observeAgent = agentRecording ? (event: Record<string, unknown>) => agentRecording!.record(event) : undefined;
  const canvasAgent = new CanvasAgentRuntime({ tools: owner => windowTools(owner, true), release: releaseMcpOwner, observe: observeAgent,
    pipeline: () => vscode.workspace.getConfiguration('tensorviz').inspect<'native' | 'acp'>('agent.pipeline')?.globalValue ?? 'native',
    acpAdapter: async provider => {
      const configured = vscode.workspace.getConfiguration('tensorviz').inspect<string>(`agent.${provider}AcpPath`)?.globalValue;
      if (!configured) throw new Error('Set the ACP adapter executable path in TensorViz settings.');
      return agentExecutable(provider, configured);
    },
    executable: provider => agentExecutable(provider, vscode.workspace.getConfiguration('tensorviz').get<string>(`agent.${provider}Path`, '')) });
  context.subscriptions.push(vscode.workspace.onDidChangeConfiguration(event => {
    if (['agent.codexPath', 'agent.claudePath', 'agent.pipeline', 'agent.codexAcpPath', 'agent.claudeAcpPath']
      .some(setting => event.affectsConfiguration(`tensorviz.${setting}`))) canvasAgent.invalidate();
  }));
  browserGraph = new LocalBrowserGraph({
    observeAgent,
    agent: { run: (input, emit, signal) => canvasAgent.run(input, emit, signal), clear: preserveEvidence => canvasAgent.clear(preserveEvidence),
      projectWork: {
        running: signal => [...agentProjectTools.values()].some(tools => tools.running(signal)),
        wait: (signal, waitMs) => [...agentProjectTools.values()].find(tools => tools.running(signal))?.wait(signal, waitMs) ?? Promise.resolve(),
        unread: signal => [...agentProjectTools.values()].flatMap(tools => tools.unread(signal)),
      },
      reference: model => windowGraph().reference(model),
      selection: (reference, nodeIds) => windowGraph().inspectNodes(reference, nodeIds),
      graph: (reference, maxBytes) => agentGraphContext(windowGraph(), reference, maxBytes),
      modelContext: reference => getModelContext(windowGraph(), browserGraph!.captureInspector, reference),
      inspect: (provider, refresh, signal) => {
        if (!vscode.workspace.isTrusted) return Promise.resolve({ status: 'error' as const, detail: 'Trust this workspace in VS Code before connecting a coding agent.', models: [] });
        return canvasAgent.inspect(provider, browserGraph?.target?.workspaceFolder ?? vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? homedir(), refresh, signal);
      },
      warm: (provider, settings, signal, intent, prepared) => {
        if (!vscode.workspace.isTrusted) return Promise.reject(new Error('Trust this workspace in VS Code before connecting a coding agent.'));
        return canvasAgent.warm({ provider, settings, intent, ...(prepared ? { preparation: { context: prepared.context } } : {}),
          cwd: prepared?.cwd ?? browserGraph?.target?.workspaceFolder ?? vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? homedir() }, signal);
      },
      preferences: context.workspaceState.get('tensorviz.canvasAgentPreferences'),
      remember: preferences => { void context.workspaceState.update('tensorviz.canvasAgentPreferences', preferences); },
      invalidate: () => canvasAgent.invalidate() },
    rememberOpen: (record: OpenGraphRecord) => { void context.workspaceState.update(OPEN_GRAPH_KEY, record); },
    channel,
    backend: { origin: () => backend?.origin, launchToken: () => backend?.launch.token },
    // A dev server carries the page while the app is being worked on; shipped, the
    // backend serves it from its own origin, which is also the only origin its
    // session handshake accepts. Same sources shellTarget reads, minus the launch
    // token — the page authenticates with the fragment credential, never a query
    // token, so the URL deliberately stays out of withLaunchToken.
    launchOrigin: () => {
      const dev = devServerUrl(
        vscode.workspace.getConfiguration('tensorviz').get<string>('devServerUrl', ''),
        process.env,
      ).trim();
      return dev === '' ? undefined : dev;
    },
    // Reuse the backend that is already running, exactly like the canvas path at
    // The current Canvas: the graph's sessions are minted against one process, and a
    // respawn here would drown every live session's broker state while this page
    // keeps pointing at the old origin (whose broker then 401s the bootstrap
    // against a session it never minted).
    ensureBackend: async () =>
      backend !== undefined
        ? true
        : (await ensureBackend(context, vscode.ProgressLocation.Notification)) !== undefined &&
          backend !== undefined,
    remoteName: () => vscode.env.remoteName,
    resolveLaunchOrigin: clientOrigin,
    // TENSORVIZ_LOCAL_BROWSER_LINK_FILE is the Electron seam: the environment is
    // read per open, not at activation, so any suite in the shared host can claim
    // the seam by setting it before it fires the command. One URL per line.
    openExternal: async (url) => {
      const linkFile = process.env.TENSORVIZ_LOCAL_BROWSER_LINK_FILE?.trim();
      if (linkFile) {
        fs.mkdirSync(path.dirname(linkFile), { recursive: true });
        fs.appendFileSync(linkFile, `${url}\n`, 'utf8');
        return true;
      }
      return vscode.env.openExternal(vscode.Uri.parse(url));
    },
    profiles: profileLookup,
    environment: (target, signal, evidence) => checkOpeningEnvironment(environmentOptions(target, signal), evidence),
    recoverEnvironment: (target, environment, action, signal, progress) => {
      const openingId = browserGraph?.opening?.state?.id;
      return recoverOpeningEnvironment(environmentOptions(target, signal), environment, action, {
        current: async () => {
          const document = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(target.workspaceFolder, target.file)));
          return browserGraph?.opening?.state?.id === openingId && !document.isDirty
            && createHash('sha256').update(document.getText()).digest('hex') === target.sourceDigest;
        },
        dependencies: () => modelDependencyOptions(context, target, signal),
        changed: () => browserGraph?.cancelEnvironmentExecution(),
        progress,
      });
    },
    selectEnvironment: target => checkEnvironmentCommand(context, true, false,
      vscode.Uri.file(path.join(target.workspaceFolder, target.file))),
    prepareDependencies: async (target, signal, missingModules, importFailure) => ensureModelDependencies({
      ...await modelDependencyOptions(context, target, signal), workspace: target.workspaceFolder,
      file: path.join(target.workspaceFolder, target.file), missingModules, importFailure,
      selectEnvironment: python => rememberEnvironment(context, { kind: 'project', interpreter: python, explicit: true },
        vscode.Uri.file(path.join(target.workspaceFolder, target.file))),
    }),
    runCandidate: candidateWorker?.runCandidate.bind(candidateWorker),
    trace: browserWorker
      ? {
          worker: browserWorker,
          trusted: async () => workspaceTrusted(),
          settings: resource => browserTraceSettings(section => vscode.workspace.getConfiguration('tensorviz', resource).get(section)),
        }
      : undefined,
  });
  let canvasAttachment: vscode.Disposable | undefined;
  sidebarCanvas = new SidebarCanvas({
    channel,
    checkHome: async signal => {
      const resolved = resolveBackendDir(context);
      const unsupported = unsupportedRemote(vscode.env.remoteName);
      if (unsupported) return { phase: 'unavailable', detail: unsupported };
      if (!resolved.ok) return { phase: 'unavailable', detail: 'The packaged tensorViz backend is unavailable. Reinstall the extension or inspect the log.' };
      return checkCanvasHomeSetup({ signal,
        runtime: () => managedBackendRuntime(context).check() });
    },
    home: () => {
      const folders = vscode.workspace.workspaceFolders ?? [];
      return {
        hasWorkspace: folders.length > 0,
        indexed: folders.length > 0 && folders.every(folder => lenses?.classesFor(folder.uri.fsPath) !== undefined),
        models: folders.flatMap(folder => (lenses?.classesFor(folder.uri.fsPath) ?? []).map(model => ({
          name: model.name,
          file: folders.length > 1 ? `${folder.name}/${model.file}` : model.file,
          uri: vscode.Uri.joinPath(folder.uri, model.file).toString(),
        }))),
      };
    },
    refreshTarget: vscode.env.remoteName === 'ssh-remote',
    prepare: async () => {
      const unsupported = unsupportedRemote(vscode.env.remoteName);
      if (unsupported) { void vscode.window.showWarningMessage(unsupported); return undefined; }
      const running = backend
        ? { repoPath: await resolveRepo(context), launch: backend.launch }
        : await ensureBackend(context, vscode.ProgressLocation.Notification);
      if (running?.repoPath && !canvasAttachment) {
        canvasAttachment = browserGraph!.attachCanvas(message => sidebarCanvas?.post(message));
        context.subscriptions.push(canvasAttachment);
      }
      return running?.repoPath
        ? await forwardShellTarget(shellTarget(running.launch.port, running.launch.token, workspaceDir(running.repoPath)))
        : undefined;
    },
    receive: message => browserGraph?.handleCanvasMessage(message),
    replay: () => browserGraph?.replayCanvas(),
    viewConnection: (viewId, connected) => browserGraph?.setCanvasView(viewId, connected),
  });
  context.subscriptions.push(sidebarCanvas, sidebarCanvas.register(),
    lenses.onDidChangeClasses(() => sidebarCanvas?.refreshHome()),
    vscode.workspace.onDidChangeWorkspaceFolders(() => sidebarCanvas?.refreshHome()),
    ...registerCanvasCommands(sidebarCanvas, browserGraph, async target => {
      let uri = target.uri ? vscode.Uri.parse(target.uri) : undefined;
      if (!uri || !isHostFile(uri) || !/\.pyw?$/i.test(uri.fsPath)) {
        try {
          const chosen = await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: 'Finding Python models…' },
            () => pickWorkspaceModel(vscode.workspace.workspaceFolders ?? [], workspaceClasses));
          if (!chosen) return undefined;
          target = chosen;
          uri = vscode.Uri.parse(chosen.uri);
        } catch (error) {
          channel.appendLine(`[models] workspace discovery failed: ${String(error)}`);
          void vscode.window.showErrorMessage('Could not discover workspace models. Check the tensorViz output, then try Open as Graph again.');
          return undefined;
        }
      }
      let cls = target.cls;
      if (!cls) {
        if (!await ensureBackend(context, vscode.ProgressLocation.Window)) return undefined;
        await lenses?.refresh(uri);
        const root = WorkspaceClassIndex.rootFor(uri);
        const relative = path.relative(root, uri.fsPath).split(path.sep).join('/');
        const candidates = (lenses?.classesFor(root) ?? []).filter(entry => entry.file === relative);
        if (candidates.length === 0) {
          const document = await vscode.workspace.openTextDocument(uri);
          const result = await backendJson(`${backend!.origin}/api/ide/buffer-classes`, {
            method: 'POST', headers: { 'Content-Type': 'application/json', ...backendEndpoint.headers() },
            body: JSON.stringify({ file: relative, source: document.getText(), include_unresolved_bases: true }),
          }) as { classes: (BufferClass & { unverified?: boolean })[] };
          if (!result.classes.length) {
            void vscode.window.showWarningMessage('No PyTorch model classes or subclasses were found in this file.');
            return undefined;
          }
          // Choosing an unresolved subclass is explicit. Do not add it to the
          // verified Models index or automatically execute the only candidate.
          const selected = await vscode.window.showQuickPick(result.classes.map(entry => ({
            label: entry.name, description: entry.unverified ? 'Unverified model type' : undefined,
            detail: entry.unverified ? 'Library or custom base; capture will verify the constructed module.' : undefined,
          })), { title: 'Open as Graph', placeHolder: 'Choose the class to capture' });
          cls = selected?.label;
        } else {
          cls = candidates.length === 1 ? candidates[0].name : await vscode.window.showQuickPick(
            candidates.map(entry => entry.name), { title: 'Open as Graph', placeHolder: 'Choose the model to capture' });
        }
        if (!cls) return undefined;
      }
      if (!context.workspaceState.get<boolean>('graphCaptureExplained')) {
        void context.workspaceState.update('graphCaptureExplained', true);
        void vscode.window.showInformationMessage('Open as Graph collects missing setup, then runs the model once on its saved device to capture the graph. Use Run model to run it on another device. Source-only inspection remains available.');
      }
      return { uri: uri.toString(), cls };
    }),
  );
  context.subscriptions.push(vscode.commands.registerCommand('tensorviz.reviewGraphChanges', async (spec?: import('./graphReview').GraphReviewSpec) => {
    if (await sidebarCanvas?.show()) await browserGraph?.reviewChanges(spec);
  }));
  const readGitHubReview = githubReader();
  const reviewCache = path.join(context.globalStorageUri.fsPath, 'pr-reviews');
  let reviewRequest = 0;
  const reviewFile = (kind: 'head' | 'working' | 'staged') =>
    async (argument?: vscode.Uri | { resourceUri?: vscode.Uri }) => {
      const request = ++reviewRequest;
      const uri = argument instanceof vscode.Uri ? argument : argument?.resourceUri ?? vscode.window.activeTextEditor?.document.uri;
      if (!uri || !isHostFile(uri)) { void vscode.window.showInformationMessage('Open a Python file to compare changes.'); return; }
      try {
        const session = await reviewFileChanges(uri, kind);
        if (request === reviewRequest && await sidebarCanvas?.show()) await browserGraph?.reviewSession(session);
      } catch (error) { void vscode.window.showErrorMessage(`Could not open graph diff. ${(error as Error).message}`); }
    };
  context.subscriptions.push(
    vscode.commands.registerCommand('tensorviz.compareWithHead', reviewFile('head')),
    vscode.commands.registerCommand('tensorviz.graphDiffWorking', reviewFile('working')),
    vscode.commands.registerCommand('tensorviz.graphDiffStaged', reviewFile('staged')),
  );
  context.subscriptions.push(vscode.commands.registerCommand('tensorviz.openPullRequest', async (url?: unknown) => {
    const request = ++reviewRequest;
    try {
      const folder = await workspaceGitFolder(vscode.window.activeTextEditor?.document.uri);
      const address = typeof url === 'string' ? parsePullRequestUrl(url)
        : await choosePullRequest(folder ? await repositoryForFolder(folder) : undefined, readGitHubReview);
      if (!address || request !== reviewRequest || !await sidebarCanvas?.show()) return;
      await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: 'Opening pull request as graph diff…' }, async () => {
        const session = await loadPullRequest(address, readGitHubReview, reviewCache);
        if (request === reviewRequest) await browserGraph?.reviewSession(session);
      });
    } catch (error) { void vscode.window.showErrorMessage(`Could not open PR. ${(error as Error).message}`); }
  }), vscode.commands.registerCommand('tensorviz.graphDiffFromEditor', async () => {
    const request = ++reviewRequest;
    const input = vscode.window.tabGroups.activeTabGroup.activeTab?.input;
    if (!(input instanceof vscode.TabInputTextDiff)) { void vscode.window.showInformationMessage('Open a code diff, then choose Graph diff.'); return; }
    try {
      if (browserGraph?.ownsReviewDiff(input)) { await sidebarCanvas?.show(); return; }
      const session = await reviewEditorDiff(input, readGitHubReview, reviewCache);
      if (request === reviewRequest && await sidebarCanvas?.show()) await browserGraph?.reviewSession(session);
    } catch (error) { void vscode.window.showErrorMessage(`Could not open graph diff. ${(error as Error).message}`); }
  }));
  // Both Canvas surfaces share the current graph's measured source hints.
  inlayHints = new ShapeInlayHints(uri => browserGraph?.shapesFor(uri) ?? []);
  inlayHints.refreshOn(browserGraph.onDidChangeShapes);
  context.subscriptions.push(
    inlayHints,
    vscode.languages.registerInlayHintsProvider({ language: 'python', scheme: 'file' }, inlayHints),
  );
  // Keep the per-class action available to reveal the retained sidebar even
  // when hidden, or when an editor-only comparison also draws that class.
  const graphLenses = new OpenAsGraphLenses(lenses);
  context.subscriptions.push(
    graphLenses,
    vscode.languages.registerCodeLensProvider({ language: 'python', scheme: 'file' }, graphLenses),
    new GraphFilePrompt(lenses),
  );
  // Open and save are when a class can appear or disappear. Typing is not: a
  // half-written `class` line would flicker a lens on and off every keystroke.
  //
  // Opening a model file is also what starts the backend (task.md P0). Nothing else does any
  // more: the Models view lists what the backend's AST scan found, and P0 deleted the "Start
  // tensorViz" row because opening a folder with a model in it is the instruction. Without
  // this the row's absence left a fresh window on "Indexing…" for ever, which is worse than
  // the row ever was. `startFailed` makes it one attempt per window: a failure is a diagnostic,
  // one per save is a loop, and saves are exactly what this fires on.
  const reindex = (document: vscode.TextDocument): void => {
    if (document.languageId === 'python' && isHostFile(document.uri)) {
      void lenses?.refresh(document.uri);
    }
    if (
      shouldStartBackend({
        languageId: document.languageId,
        scheme: document.uri.scheme,
        text: document.getText(),
        backendRunning: backend !== undefined,
        trusted: workspaceTrusted(),
        startFailed,
      })
    ) {
      channel.appendLine(`Starting source discovery: ${document.uri.fsPath} declares a Python subclass.`);
      // The status bar, not a notification: nobody asked for this, they opened a file.
      void ensureBackend(context, vscode.ProgressLocation.Window).then((running) => {
        startFailed = running === undefined;
        if (running) {
          // The index could not answer until now, so ask it again for every folder.
          for (const folder of vscode.workspace.workspaceFolders ?? []) {
            void lenses?.refresh(folder.uri);
          }
          void maybeAutoOpenGraph(context, document);
        }
      });
    } else if (
      backend &&
      document.languageId === 'python' &&
      isHostFile(document.uri) &&
      declaresModule(document.getText())
    ) {
      // The backend was already up — started by the Agent panel, or by an earlier file — so
      // the branch above did not run, and the canvas would only ever open for the FIRST model
      // file of the window. Opening a model is the instruction to draw it, every time the
      // canvas is not already on screen. (maybeAutoOpenGraph is what checks that, and the
      // setting.) Only on open, deliberately: closing the canvas focuses the code, and a
      // hook on the active editor would reopen it in the same breath.
      void maybeAutoOpenGraph(context, document);
    }
  };
  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument(reindex),
    vscode.workspace.onDidSaveTextDocument(reindex),
    // Granting trust is the moment the extension is allowed to do anything at all, and
    // the model the user is looking at is ALREADY open by then — so no open event will
    // ever fire for it. Without this, the first run of a new folder is: open it, answer
    // "yes, I trust the authors", and watch nothing happen, with the only way forward
    // being to open some other file. VS Code restarts the extension host on trust, so
    // this runs in the new host over the documents that survived it.
    vscode.workspace.onDidGrantWorkspaceTrust(() => {
      channel.appendLine('Workspace trusted — looking at the files already open.');
      sidebarCanvas?.refreshHome(true);
      for (const document of vscode.workspace.textDocuments) {
        reindex(document);
      }
    }),
  );

  // The same gap at activation. `onStartupFinished` fires after the editors are restored,
  // so a window reopened on a model has it open before this extension exists, and an event
  // that already happened cannot be listened for.
  for (const document of vscode.workspace.textDocuments) {
    reindex(document);
  }
  registerShell(context, lenses);
  // Retired model panels have no trustworthy selected target to restore. Close
  // them without starting another controller; comparison panels keep their own
  // distinct view types and are unaffected.
  context.subscriptions.push(
    vscode.window.registerWebviewPanelSerializer(VIEW_TYPE, retiredCanvasSerializer(() => { void restoreOpenGraph(context); })),
  );
  void restoreOpenGraph(context);
  // The workspace itself is the subject, so the workspace is what is looked at —
  // not the file somebody happens to open (step 2.7). Git does not wait for you to
  // open a tracked file before it notices `.git`, and a project canvas that waits
  // for a Python file is a canvas that has already made the file the subject.
  //
  // Deliberately a CHEAP look first. `.git` is a directory read; an index is a walk
  // of every `.py` plus a Python process, and paying that in a window that merely
  // contains a stray script would be a worse trade than the one it fixes. So: find
  // one model file with `findFiles`, which is VS Code's own indexed search and
  // honours the same excludes the explorer does, and only then start anything.
  void detectProject(context);

  if (!unsupportedRemote(vscode.env.remoteName)) mcpLifecycle.start();
  return { __test: testApi };
}

/** Reopen the class that was on the canvas before Reload Window. The sidebar view
 *  itself comes back empty; the stored target is what fills it. A graph that is
 *  already selected is left alone. */
async function restoreOpenGraph(context: vscode.ExtensionContext): Promise<void> {
  const stored = graphToRestore(storedOpenGraph(context.workspaceState.get(OPEN_GRAPH_KEY)), browserGraph?.hasSelection === true);
  if (!stored) return;
  const folder = vscode.workspace.workspaceFolders?.find(item => item.uri.fsPath === stored.workspaceFolder);
  if (!folder || !browserGraph) return;
  browserGraph.holdRestoration(stored);
  if (await sidebarCanvas?.show()) {
    await browserGraph.open({ uri: vscode.Uri.file(path.join(stored.workspaceFolder, stored.file)).toString(), cls: stored.cls }, 'canvas', 'automatic');
  }
}

/** Is there a project here at all, and if so draw it (step 2.7).
 *
 * The `enabledExplicitly` rule from `autoOpenClass` applies unchanged: the fork
 * draws unasked because that is its whole proposition, and another editor draws
 * when the user has said so. Nothing here starts a backend until that question is
 * already answered yes — the check is a file search, not an index.
 */
async function detectProject(context: vscode.ExtensionContext): Promise<void> {
  const configuration = vscode.workspace.getConfiguration('tensorviz');
  if (!configuration.get<boolean>(AUTO_OPEN_SETTING, true)) {
    return;
  }
  const setting = configuration.inspect<boolean>(AUTO_OPEN_SETTING);
  const explicit =
    setting?.globalValue !== undefined ||
    setting?.workspaceValue !== undefined ||
    setting?.workspaceFolderValue !== undefined;
  if (vscode.env.uriScheme !== FORK_URI_SCHEME && !explicit) {
    return;
  }
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder) {
    return;
  }
  // One match is all the question needs. `**/*.py` with a cap of 1 stops at the
  // first hit rather than enumerating a repository to answer yes/no.
  const found = await vscode.workspace.findFiles(
    new vscode.RelativePattern(folder, '**/*.py'),
    '**/{node_modules,venv,.venv,site-packages}/**',
    1,
  );
  if (found.length === 0) {
    channel.appendLine(`Project: no Python in ${folder.uri.fsPath}; nothing to draw.`);
    return;
  }
  channel.appendLine(`Project: ${folder.uri.fsPath} holds Python; opening the canvas.`);
  if (!browserGraph?.hasSelection && !sidebarCanvas?.requestedVisible) {
    if (await sidebarCanvas?.show()) await browserGraph?.openProject({ folder: folder.uri.fsPath }, 'canvas', 'automatic');
  }
}

/** VS Code does not reliably await an async deactivate, so the group kill must be
 *  synchronous — no awaits on this path. */
export function deactivate(): void {
  for (const tools of agentProjectTools.values()) tools.dispose();
  agentProjectTools.clear();
  disposeSnapshotHelpers();
  void mcpLifecycle?.dispose();
  void stopGraphMcp();
  sidebarCanvas?.dispose();
  sidebarCanvas = undefined;
  browserGraph?.dispose();
  browserGraph = undefined;
  backend?.dispose();
  backend = undefined;
}

/** The `nn.Module` classes under `root`, from the backend's AST scan.
 *
 * Returns nothing while the backend is not running rather than starting it: a
 * CodeLens refresh must never be the thing that spawns a python process. The
 * lenses appear when startup finishes and refreshes the index, without needing
 * the canvas to be opened first. */
async function workspaceClasses(root: string): Promise<WorkspaceClass[]> {
  const origin = backend?.origin;
  if (!origin) {
    // Throw, do not answer `[]`. An empty list is a real answer — this folder has
    // no nn.Module classes — and returning it for "nobody has looked yet" made the
    // Models view say "No nn.Module classes here yet" while staring at a file with
    // one in it. The honest message and its Open Canvas button were already
    // written and were unreachable, because this line told the index it had
    // scanned. `refresh` swallows this and keeps whatever it had.
    throw new Error('the tensorViz backend is not running, so nothing has been scanned');
  }
  const response = await fetch(`${origin}/api/ide/workspace/classes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', [TOKEN_HEADER]: backend!.launch.token },
    body: JSON.stringify({ root }),
  });
  if (!response.ok) {
    throw new Error(`/api/ide/workspace/classes returned ${response.status}`);
  }
  return ((await response.json()) as { classes: WorkspaceClass[] }).classes;
}

/** The legacy embedded open: the canvas first, then the class projected onto it.
 *
 * What "Open as Graph" did before it moved to the external browser
 * (extension-webpage step 4). Graph comparison still uses it; the legacy
 * Electron suites reach it
 * through the test API. */
/**
 * In the fork, draw the file that was just opened, once.
 *
 * P0's last activation item. It runs only after the backend answered, because the class list is
 * the backend's AST scan and guessing at it here would be a second, worse parser. A user who
 * closes the graph is not asked again for this folder — `autoOpenClass` reads that memento, and
 * it is a memento rather than a setting because it records what happened, not a preference.
 */
async function maybeAutoOpenGraph(
  context: vscode.ExtensionContext,
  document: vscode.TextDocument,
): Promise<void> {
  // From the buffer, not from the backend's workspace scan. The scan is the lenses' index and
  // runs on its own; waiting on it here put 2 to 6 seconds between opening a model and seeing
  // it (measured on this repository), for a class name that is in the text we already hold.
  const classes = moduleClasses(document.getText());
  if (classes.length === 0) {
    // Say so. The first version was silent here, and "nothing happened" is not a diagnosis.
    channel.appendLine(
      `Auto-open: no nn.Module class in ${vscode.workspace.asRelativePath(document.uri, false)}.`,
    );
    return;
  }
  // Optional initial opening never replaces an explicitly selected model or
  // project, including when the user hides the Canvas or opens another editor.
  const draw = autoOpenClass({
    isFork: vscode.env.uriScheme === FORK_URI_SCHEME,
    graphShowing: browserGraph?.hasSelection === true || sidebarCanvas?.requestedVisible === true,
    enabled: vscode.workspace.getConfiguration('tensorviz').get<boolean>(AUTO_OPEN_SETTING, true),
    // Written by the user somewhere, at any scope — as opposed to the packaged
    // default, which nobody chose and which must not be read as consent to open
    // a canvas unasked in an editor that is not the fork.
    enabledExplicitly: (() => {
      const setting = vscode.workspace.getConfiguration('tensorviz').inspect<boolean>(AUTO_OPEN_SETTING);
      return (
        setting?.globalValue !== undefined ||
        setting?.workspaceValue !== undefined ||
        setting?.workspaceFolderValue !== undefined
      );
    })(),
    classes,
    // So a helper module can be told from the model. Same buffer the classes came from.
    source: document.getText(),
  });
  if (!draw) {
    return;
  }
  const folder = vscode.workspace.getWorkspaceFolder(document.uri)?.uri.fsPath;
  if (!folder) {
    // A loose file with no folder has no project to draw. The per-class graph is
    // what it can still have, and saying so beats opening nothing in silence.
    channel.appendLine(`Auto-open: ${vscode.workspace.asRelativePath(document.uri, false)} belongs to no workspace folder; drawing ${draw}.`);
    if (await sidebarCanvas?.show()) await browserGraph?.open({ uri: document.uri.toString(), cls: draw }, 'canvas', 'automatic');
    return;
  }
  channel.appendLine(`Auto-open: ${vscode.workspace.asRelativePath(document.uri, false)} declares a model; drawing project ${folder}.`);
  if (await sidebarCanvas?.show()) await browserGraph?.openProject({ folder }, 'canvas', 'automatic');
}

function fail(message: string, detail: string, extraButtons: string[] = []): Thenable<string | undefined> {
  channel.appendLine(message);
  channel.appendLine(detail);
  // The log gets the full text; the notification gets the part a person can act on. A raw
  // traceback in a modal is the reported Python 3.7 crash's exact failure mode.
  return vscode.window.showErrorMessage(
    message,
    { modal: true, detail: withoutTraceback(detail) },
    ...extraButtons,
    'Show Log',
  );
}

/** Where `.tensorviz/` and `tensorviz-exports/` live: the folder the user opened,
 *  which is not necessarily the checkout the backend runs from.
 *
 *  Without an open folder, an explicitly configured valid checkout is the project root even
 *  when the packaged backend wins resolution. Otherwise the fallback is the checkout that owns
 *  the backend directory. P0 changed what callers hold — `<repo>/backend` from a checkout,
 *  `<extension>/backend` from the .vsix — and using the backend path wrote a user's exports into
 *  staged extension files, where nobody would look for them and the next package build deletes
 *  them. */
function allowedWorkspaceRoots(backendDir: string): string {
  const folders = (vscode.workspace.workspaceFolders ?? []).map((folder) => folder.uri.fsPath);
  const primary = workspaceDir(backendDir);
  return [...new Set([primary, ...folders].filter((folder) => folder))].join(path.delimiter);
}

function workspaceDir(backendDir: string): string {
  const openFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (openFolder) {
    return openFolder;
  }
  const configured = vscode.workspace.getConfiguration('tensorviz').get<string>('repoPath', '').trim();
  if (configured && fs.existsSync(path.join(configured, 'backend', 'server_v2.py'))) {
    return configured;
  }
  return repoRootFor(backendDir);
}

/**
 * Bring the backend up, without opening anything.
 *
 * `location` is the difference between the two callers: a user who ran "Open as Graph" is
 * waiting and gets a notification, while a user who merely opened a .py file is not and gets the
 * status bar. Both are cancellable.
 */
function ensureBackend(
  context: vscode.ExtensionContext,
  location: vscode.ProgressLocation,
): Promise<{ repoPath: string; launch: BackendLaunch } | undefined> {
  const unsupported = unsupportedRemote(vscode.env.remoteName);
  if (unsupported) { void vscode.window.showWarningMessage(unsupported); return Promise.resolve(undefined); }
  if (backend) return Promise.resolve({ repoPath: backend.backendDir, launch: backend.launch });
  if (!backendStarting) {
    backendStarting = startBackend(context, location).finally(() => {
      backendStarting = undefined;
    });
  }
  return backendStarting;
}

async function startBackend(
  context: vscode.ExtensionContext,
  location: vscode.ProgressLocation,
): Promise<{ repoPath: string; launch: BackendLaunch } | undefined> {
  const repoPath = await resolveRepo(context);
  if (!repoPath) {
    return undefined;
  }
  channel.appendLine(`Repository: ${repoPath}`);

  // Cancellable: a first launch may be downloading a Python runtime and installing torch
  // into it, which is minutes, not seconds. A modal the user cannot escape is not an option.
  const interpreter = await vscode.window.withProgress(
    {
      location,
      title: 'Checking prerequisites…',
      cancellable: true,
    },
    (progress, token) =>
      runPreflight(
        context,
        repoPath,
        (message) => progress.report({ message }),
        () => token.isCancellationRequested,
      ),
  );
  if (!interpreter) {
    return undefined;
  }

  // Backend state belongs to VS Code's per-workspace storage, never the source tree.
  // Empty windows have no storageUri, so isolate their state under global storage.
  const backendStateDir = context.storageUri?.fsPath ?? path.join(context.globalStorageUri.fsPath, 'backend');
  const launch: BackendLaunch = {
    // Per launch, unless ide/dev.sh pinned one so its CRA proxy has somewhere to forward.
    port: fixedBackendPort(process.env) ?? (await reserveLoopbackPort()),
    // Shipped inside the .vsix by extension/scripts/build-webview.mjs.
    staticDir: path.join(context.extensionPath, 'media', 'app'),
    token: newLaunchToken(),
    stateDbPath: path.join(backendStateDir, '.tensorviz', 'state.sqlite3'),
    workspaceDir: workspaceDir(repoPath),
    allowedRoots: allowedWorkspaceRoots(repoPath),
  };

  // Cancellable: a cold `import torch` can run past a minute, and without this the
  // only way out of a hung spawn would be reloading the window.
  const started = await vscode.window.withProgress(
    {
      location,
      title: 'Starting tensorViz…',
      cancellable: true,
    },
    async (progress, token) => {
      const process = new BackendProcess(repoPath, interpreter, channel, launch);
      try {
        await process.start(
          backendStateDir,
          (message) => progress.report({ message }),
          () => token.isCancellationRequested,
        );
      } catch (err) {
        process.dispose();
        if (token.isCancellationRequested) {
          channel.appendLine('Startup cancelled by the user; the backend was stopped.');
        } else if (err instanceof StackError) {
          void fail(err.message, err.detail);
        } else {
          void fail('tensorViz failed to start.', String(err));
        }
        return undefined;
      }
      return process;
    },
  );
  if (!started) {
    return undefined;
  }

  backend = started;
  // Imported subclasses may be open before Canvas starts the backend. Their
  // earlier scan could not run; refresh now without waiting for another save.
  for (const folder of vscode.workspace.workspaceFolders ?? []) void lenses?.refresh(folder.uri);
  for (const document of vscode.workspace.textDocuments) {
    if (document.languageId === 'python') void lenses?.refresh(document.uri);
  }
  started.onRestarting = (message: string, detail: string) => {
    channel.appendLine(`backend restarting: ${message}\n${detail}`);
    void vscode.window.showWarningMessage(message, 'Show Log').then((pick) => {
      if (pick === 'Show Log') channel.show();
    });
  };
  started.onUnexpectedExit = (message: string, detail: string) => {
    channel.appendLine(`backend died after readiness: ${message}\n${detail}`);
    if (backend !== started) return;
    backend = undefined;
    sidebarCanvas?.setError();
    void vscode.window.showErrorMessage(message, 'Show Log').then((pick) => {
      if (pick === 'Show Log') {
        channel.show();
      }
    });
  };
  return { repoPath, launch };
}

async function resolveRepo(context: vscode.ExtensionContext): Promise<string | undefined> {
  const resolution = resolveBackendDir(context);
  if (resolution.ok) {
    channel.appendLine(`Backend resolved from ${resolution.source}.`);
    return resolution.backendDir;
  }
  const buttons = resolution.offerPicker ? ['Select folder…', 'Open Settings'] : ['Open Settings'];
  const pick = await fail(resolution.message, resolution.detail, buttons);
  if (pick === 'Show Log') {
    channel.show();
  } else if (pick === 'Open Settings') {
    void vscode.commands.executeCommand('workbench.action.openSettings', 'tensorviz.repoPath');
  } else if (pick === 'Select folder…') {
    const picked = await pickRepoPath(context);
    // Every other source resolveBackendDir tries yields the BACKEND directory
    // (`<repo>/backend`, or the staged copy inside the extension), and everything below —
    // the probe's cwd, uvicorn's cwd, the trace worker's cwd — depends on that. The picker
    // validates a checkout root, so join it here rather than making each caller cope.
    return picked ? path.join(picked, 'backend') : undefined;
  }
  return undefined;
}

/** Resolve only an existing execution environment. This path never installs or prompts. */
async function executionInterpreter(context: vscode.ExtensionContext, resource: vscode.Uri): Promise<string | undefined> {
  return (await executionEnvironment(context, resource)).interpreter;
}

function executionEnvironment(context: vscode.ExtensionContext, resource: vscode.Uri) {
  return selectedExecutionEnvironment(context, resource, async () => {
    const managed = await managedBackendRuntime(context).check();
    return managed.state === 'ready' ? managed.interpreter : undefined;
  });
}

/** The managed runtime for the BACKEND role, bound to resolveBackendDir's current answer.
 *
 * The binding is a thunk because the answer genuinely changes inside one window — a
 * "tensorviz.repoPath" edit, a canvas checkout opened mid-session — and the runtime must
 * provision against the backend that will actually run, not the one that resolved at
 * activation. It is wired here because runtime.ts must not import preflight (preflight
 * already imports runtime), and the thunk belongs to exactly one place in the app. */
function managedBackendRuntime(context: vscode.ExtensionContext): ManagedRuntime {
  const runtime = managedRuntimeFor(context, {
    backendDir: () => {
      const resolved = resolveBackendDir(context);
      return resolved.ok ? resolved.backendDir : undefined;
    },
    ...testRuntimeSeams(),
  });
  // Lazy, not activation-time: resolveBackendDir's answer that preseeding has to stamp for
  // only exists once the command asking for a backend computed it.
  seedManagedRuntimeForTests(runtime);
  return runtime;
}

/** Electron-suite seams, consulted only when the variables are set, and only set by
 *  .vscode-test.mjs or a suite itself. TENSORVIZ_TEST_RUNTIME_ARCHIVE names a local .tar.gz
 *  that the "download" streams, and TENSORVIZ_TEST_RUNTIME_MANIFEST names the manifest copy
 *  re-pinned to that archive's checksum — together they make a first-install leg deterministic
 *  and offline without ever touching the shipped pin. */
function testRuntimeSeams(): { manifestPath?: string; transport?: Transport } {
  const manifestPath = process.env.TENSORVIZ_TEST_RUNTIME_MANIFEST?.trim();
  const archive = process.env.TENSORVIZ_TEST_RUNTIME_ARCHIVE?.trim();
  return {
    ...(manifestPath ? { manifestPath } : {}),
    ...(archive
      ? {
          transport: (async () => ({
            status: 200,
            headers: { 'content-length': String(fs.statSync(archive).size) },
            stream: fs.createReadStream(archive),
          })) as Transport,
        }
      : {}),
  };
}

/** Electron-suite seam: fabricate the state first-run provisioning leaves behind — a ready
 *  managed runtime whose interpreter IS the run's TENSORVIZ_TEST_PYTHON, reached through a
 *  directory symlink (the interpreter's own pyvenv.cfg resolves next to it, so the venv's
 *  site-packages survive the indirection). Without it every Electron run would face the
 *  stock first-install consent and dead-lock on an unanswerable modal. The product never
 *  reads this: only .vscode-test.mjs sets the variable, and the runtimeFlow suite flips it
 *  to '0' around the legs that must see stock first-run behaviour. POSIX-only (dir symlinks). */
function seedManagedRuntimeForTests(runtime: ManagedRuntime): void {
  if (process.env.TENSORVIZ_TEST_PRESEED_MANAGED !== '1') {
    return;
  }
  if (extensionContext?.extensionMode === vscode.ExtensionMode.Test &&
      !extensionContext.workspaceState.get(ENVIRONMENT_KEY)) {
    void extensionContext.workspaceState.update(ENVIRONMENT_KEY, { kind: 'managed' });
  }
  const python = process.env.TENSORVIZ_TEST_PYTHON?.trim();
  if (!python || process.platform === 'win32') {
    return;
  }
  try {
    if (!runtime.availability().ok) {
      return;
    }
    if (fs.existsSync(runtime.interpreter) && fs.existsSync(runtime.stampPath)) {
      return;
    }
    fs.mkdirSync(runtime.root, { recursive: true });
    if (!fs.existsSync(runtime.venv)) {
      fs.symlinkSync(path.dirname(path.dirname(python)), runtime.venv, 'dir');
    }
    fs.writeFileSync(
      runtime.stampPath,
      JSON.stringify(
        { ...runtime.pinnedStamp(), provisionedAt: new Date().toISOString() },
        null,
        2,
      ),
    );
    channel.appendLine(`[test] managed runtime preseeded from ${python}`);
  } catch (err) {
    // A failed seed leaves the stock first-run path in place; name it in the log rather than
    // diagnosing a modal that never answers.
    channel.appendLine(`[test] managed runtime preseed failed: ${String(err)}`);
  }
}
/** One prerequisite is left.
 *
 * Stage A (task.md D2) removed the rest: with the React build shipped inside the
 * extension and served by the backend, there is no npm, no Node version to check
 * and no second port to reserve. What remains is the thing the Makefile's own
 * preflight exists for — plain `python` usually is not the environment with torch. */
async function runPreflight(
  context: vscode.ExtensionContext,
  repoPath: string,
  report: (message: string) => void,
  isCancelled: () => boolean = () => false,
): Promise<string | undefined> {
  report('Preparing the tensorViz service. Project Python is selected separately.');
  const interpreter = await resolveBackendInterpreter(repoPath, {
    runtime: managedBackendRuntime(context),
    report,
    isCancelled,
  });
  if (!interpreter.ok) {
    if ('cancelled' in interpreter) return undefined;
    // Only offer the repair when a managed runtime actually exists here; elsewhere the
    // button would promise something the install cannot do.
    const pick = await fail(
      interpreter.message,
      interpreter.detail,
      interpreter.repairRuntime ? ['Repair runtime'] : [],
    );
    if (pick === 'Repair runtime') {
      void vscode.commands.executeCommand('tensorviz.repairRuntime');
    }
    return undefined;
  }
  channel.appendLine(interpreter.banner);
  if (vscode.env.uriScheme === FORK_URI_SCHEME) await pointLanguageServerAtInterpreter(interpreter.interpreter);
  return interpreter.interpreter;
}

/**
 * Tell the Python language server which interpreter this IDE actually uses.
 *
 * The fork ships basedpyright, whose `python.pythonPath` defaults to the bare
 * string `python`. On a normal machine that resolves to a system Python with no
 * torch, so every generated model file opened with nine or more errors on it —
 * `import torch` unresolved, and every `nn.*` member unknown as a consequence.
 * The product was flagging its own output as broken.
 *
 * Preflight has already found an interpreter that can import torch — it is the
 * one the backend runs on — so the only thing missing was telling the analyser.
 *
 * Written at Global scope because the interpreter is a fact about the machine,
 * not about the folder, and writing it per-workspace would put a path into the
 * user's repo. Only written when it is unset or still the default, so a user who
 * has chosen an interpreter keeps it.
 */
async function pointLanguageServerAtInterpreter(interpreterPath: string): Promise<void> {
  const python = vscode.workspace.getConfiguration('python');
  for (const key of ['pythonPath', 'defaultInterpreterPath']) {
    const current = python.inspect<string>(key);
    const chosen = current?.globalValue ?? current?.workspaceValue;
    if (chosen && chosen !== 'python' && chosen !== interpreterPath) {
      continue; // the user picked one; leave it alone
    }
    if (chosen === interpreterPath) {
      continue;
    }
    try {
      await python.update(key, interpreterPath, vscode.ConfigurationTarget.Global);
      channel.appendLine(`[python] ${key} -> ${interpreterPath}`);
    } catch (error) {
      channel.appendLine(`[python] could not set ${key}: ${String(error)}`);
    }
  }
}

// ---------------------------------------------------------------------------
// The activity-bar shell
// ---------------------------------------------------------------------------

/** One id for every profile resolution this window makes; the snapshot keys its
 *  guards by it, and a window must not borrow another window's. */
const VALIDATION_SESSION_ID = randomUUID();

/** The activity-bar container, its views, and the commands they invoke.
 *
 * Everything here is registration only: no process is started and no file is read, which is
 * what makes `onStartupFinished` an acceptable activation event for an extension that is
 * also installed into other people's VS Code. */
function registerShell(context: vscode.ExtensionContext, lenses: WorkspaceClassIndex): void {
  const models = new ModelsTree(lenses);
  const modelsView = vscode.window.createTreeView(MODELS_VIEW, { treeDataProvider: models });
  models.attach(modelsView);
  context.subscriptions.push(models, modelsView);
  context.subscriptions.push(lenses.onDidChangeClasses(() => models.refresh()));
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider(TEMPLATES_VIEW, new TemplatesTree()),
  );

  // Sign in to tensorViz and Add to tensorViz (the Canvas's Publish button).
  registerWebsiteCommands(context, channel);

  context.subscriptions.push(
    vscode.commands.registerCommand('tensorviz.selectEnvironment', () => checkEnvironmentCommand(context, true)),
    vscode.commands.registerCommand('tensorviz.checkEnvironment', () =>
      checkEnvironmentCommand(context),
    ),
    vscode.commands.registerCommand('tensorviz.newModelFromTemplate', (id?: string) =>
      newModelFromTemplate(id),
    ),
    vscode.commands.registerCommand('tensorviz.openHuggingFace', (url?: unknown) => {
      openHuggingFace(context, url);
    }),
    vscode.commands.registerCommand('tensorviz.refreshModels', async () => {
      const folders = vscode.workspace.workspaceFolders ?? [];
      await Promise.all(folders.map((folder) => lenses.refresh(folder.uri)));
      models.refresh();
    }),
  );
}

/** Select project execution Python without restarting the tool-owned service. */
async function checkEnvironmentCommand(context: vscode.ExtensionContext, forcePicker = false, resumeGraph = true,
  resource?: vscode.Uri): Promise<boolean> {
  const target = browserGraph?.opening?.state?.target;
  resource ??= target ? vscode.Uri.file(path.join(target.workspaceFolder, target.file)) : vscode.window.activeTextEditor?.document.uri;
  if (backendStarting) await backendStarting;
  const resolved = resolveBackendDir(context);
  if (!resolved.ok) { void fail(resolved.message, resolved.detail); return false; }
  if (!backend && !await ensureBackend(context, vscode.ProgressLocation.Notification)) return false;
  const result = await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: forcePicker ? 'Choosing Python environment…' : 'Checking Python environment…', cancellable: true },
    (progress, token) => (forcePicker ? selectProjectEnvironment : resolveCaptureEnvironment)(context, resolved.backendDir, {
      runtime: managedBackendRuntime(context), managerPython: backend!.interpreter, forcePicker, resource,
      workspaceRoot: target && path.join(target.workspaceFolder, target.file) === resource?.fsPath ? target.workspaceFolder : undefined,
      report: message => progress.report({ message }), isCancelled: () => token.isCancellationRequested,
    }),
  );
  if (!result.ok) {
    if (!('cancelled' in result)) void fail(result.message, result.detail);
    return false;
  }
  if (resumeGraph && browserGraph?.opening?.state) {
    browserGraph.environmentChanged();
    await browserGraph.resumeAfterEnvironmentChange();
    if (!forcePicker) void vscode.window.showInformationMessage('Project Python is ready. Continuing the same model.');
  } else if (resumeGraph || !forcePicker) {
    void vscode.window.showInformationMessage(result.banner);
  }
  return true;
}

/** Write a starter model file. Invoked with a template id from the Templates view, and
 *  without one from the palette, where the user picks. */
async function newModelFromTemplate(id?: string): Promise<void> {
  let template: ModelTemplate | undefined = MODEL_TEMPLATES.find((entry) => entry.id === id);
  let catalogEntry: CatalogEntry | undefined;
  let askedStem: string | undefined;
  if (!template) {
    // The two local starters first: they need no backend, and each carries the
    // `# tensorviz: input=` hint that makes it draw correctly on the very first open.
    // Everything the backend serves follows, which is the difference between offering
    // two architectures and offering seventy-three.
    // `kind` is taken by VS Code's own QuickPickItemKind, so this says `source`.
    type Pick = vscode.QuickPickItem & { id: string; source: 'local' | 'remote'; entry?: CatalogEntry };
    const local: Pick[] = MODEL_TEMPLATES.map((entry) => ({
      label: entry.label,
      detail: entry.description,
      id: entry.id,
      source: 'local',
    }));
    const catalog = await vscode.window.withProgress(
      { location: vscode.ProgressLocation.Window, title: 'tensorViz: reading the template catalogue…' },
      () => fetchCatalog(catalogHost).catch((error) => {
        // A catalogue we cannot read is not a reason to refuse the two we carry.
        channel.appendLine(`Template catalogue unavailable: ${String(error)}`);
        return [] as CatalogEntry[];
      }),
    );
    const remote: Pick[] = catalog.map((entry) => ({
      label: entry.label,
      detail: entry.description,
      description: entry.category,
      id: entry.id,
      source: 'remote',
      entry,
    }));
    const pick = await vscode.window.showQuickPick<Pick>([...local, ...remote], {
      title: 'New model from template',
      matchOnDetail: true,
      matchOnDescription: true,
      placeHolder:
        catalog.length > 0
          ? `${local.length + catalog.length} architectures — type to filter`
          : 'The backend is not up yet, so only the built-in starters are listed',
    });
    if (!pick) {
      return;
    }
    if (pick.source === 'local') {
      template = MODEL_TEMPLATES.find((entry) => entry.id === pick.id);
    } else {
      catalogEntry = pick.entry;
    }
  }
  if (!template && !catalogEntry) {
    return;
  }
  // A catalogue entry has no body until the backend builds one, and the generator writes
  // the class name INTO the source — so the name has to be settled before the code is
  // asked for. The scaffolder derives it from the file name, so the file name is asked
  // for here and the same derivation is used.
  if (catalogEntry) {
    const root = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    if (!root) {
      void vscode.window.showWarningMessage('Open a folder first — a new model file needs somewhere to go.');
      return;
    }
    const stem = await vscode.window.showInputBox({
      prompt: 'File name for the new model',
      value: stemFor(catalogEntry.id),
    });
    if (!stem) {
      return;
    }
    const cleaned = stem.replace(/\.py$/i, '');
    askedStem = cleaned;
    try {
      template = await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Window, title: `tensorViz: building ${catalogEntry.label}…` },
        () => templateFor(catalogHost, catalogEntry as CatalogEntry, classNameFor(cleaned)),
      );
      template = { ...template, fileStem: cleaned };
    } catch (error) {
      // The backend's own words. Some catalogue architectures use operations the code
      // generator cannot express yet, and "it does not pass validation: unknown layer
      // type 'grn'" is the useful half of that.
      void vscode.window.showWarningMessage(
        `${catalogEntry.label} could not be turned into a file. ${String(error)}`,
      );
      return;
    }
  }

  if (!template) {
    return;
  }
  const chosen: ModelTemplate = template;

  const result = await scaffoldModel(
    {
      workspaceRoot: () => vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
      ask: (prompt, value) => vscode.window.showInputBox({ prompt, value }),
      exists: async (fsPath) => {
        try {
          await vscode.workspace.fs.stat(vscode.Uri.file(fsPath));
          return true;
        } catch {
          return false;
        }
      },
      write: async (fsPath, contents) => {
        await vscode.workspace.fs.writeFile(vscode.Uri.file(fsPath), Buffer.from(contents, 'utf8'));
      },
      open: async (fsPath) => {
        const document = await vscode.workspace.openTextDocument(vscode.Uri.file(fsPath));
        await vscode.window.showTextDocument(document, { preview: false });
      },
      warn: (message) => vscode.window.showWarningMessage(message),
    },
    chosen,
    askedStem,
  );
  if (result.ok) {
    channel.appendLine(`scaffolded ${result.className} in ${result.fsPath}`);
  }
}

/** Rebuild the managed runtime from scratch. Deletes a venv, so the backend using it has to
 *  stop first through the runtime provisioning lifecycle. */
async function repairRuntime(context: vscode.ExtensionContext): Promise<void> {
  const runtime = managedBackendRuntime(context);
  const available = runtime.availability();
  if (!available.ok) {
    void fail('There is no tensorViz managed runtime to repair here.', available.why);
    return;
  }
  const result = await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: 'Repairing the tensorViz runtime…',
      cancellable: true,
    },
    (progress, token) =>
      runtime.provision(
        (message) => progress.report({ message }),
        () => token.isCancellationRequested,
        // A repair is a decision the user just made: a leftover healthy runtime is no
        // reason to skip the rebuild they asked for.
        { force: true },
      ),
  );
  if (result.ok) {
    channel.appendLine(`runtime ready: ${result.interpreter} (${Math.round(result.elapsedMs / 1000)}s)`);
    void vscode.window.showInformationMessage('The tensorViz runtime is ready.');
    return;
  }
  if (result.cancelled) {
    channel.appendLine('Runtime repair cancelled. The existing runtime was preserved.');
    return;
  }
  void fail(result.message, runtimeDiagnostic(result.detail));
}


/** The one running session, so a second Start is refused here rather than by a 409
 *  the user has to read. Cleared when its stream ends. */
