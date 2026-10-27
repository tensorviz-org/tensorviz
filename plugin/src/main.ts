import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { z } from 'zod';
import { ApprovalGate, nativePermissionDecision } from './approval';
import { PluginPermissions } from './permissions';
import { applicationData } from './environment';
import { pinAssets } from './assets';
import { StandaloneHost } from './host';
import { HostError } from './workspace';
import { registerGraphPanel } from './graphPanel';
import { graphToolResult, GraphToolError, modelRefSchema } from '@tensorviz/graph-host/mcpGraph';
import { registerMcpViewTools } from '@tensorviz/graph-host/mcpViews';
import { registerMcpCaptureTools } from '@tensorviz/graph-host/mcpCaptures';
import { registerMcpGraphSetup } from '@tensorviz/graph-host/mcpGraphSetup';
import { registerMcpValidationTools } from '@tensorviz/graph-host/mcpValidation';

declare const TENSORVIZ_VERSION: string;
const version = TENSORVIZ_VERSION;
async function start(): Promise<void> {
  if (process.argv.slice(2).length) {
    if (process.argv.length !== 3 || process.argv[2] !== '--revoke-permission') throw new Error('Supported command: --revoke-permission');
    await new PluginPermissions().save('denied');
    process.stderr.write('TensorViz permission revoked.\n');
    return;
  }
  const installedRoot = 'pkg' in process ? path.dirname(process.execPath) : path.resolve(__dirname, '..');
  const assets = await pinAssets(installedRoot, version);
  process.once('exit', () => assets.dispose());
  const server = new McpServer({ name: 'tensorviz', title: 'tensorViz', version }, {
    instructions: 'tensorViz is a local model host. Discover its capabilities before use. Tool discovery never starts Python or executes a model. Explicit user instructions and read-only requests take priority.',
  });

  const approvals = new ApprovalGate(async (message, signal) => {
    if (!server.server.getClientCapabilities()?.elicitation?.form) return undefined;
    const result = await server.server.elicitInput({ mode: 'form', message,
      requestedSchema: { type: 'object', properties: { confirm: { type: 'boolean', title: 'Allow TensorViz and remember my choice', default: false } }, required: ['confirm'] },
    }, { signal, timeout: 300_000 });
    return nativePermissionDecision(result);
  }, process.env.TENSORVIZ_DATA_DIR ?? applicationData());
  const plugin = assets.directory;
  const host = new StandaloneHost(path.join(plugin, 'backend'), path.join(plugin, 'browser'), approvals, process.env.TENSORVIZ_DATA_DIR);
  const panels = await registerGraphPanel(server, host, plugin, version);
  const result = (action: () => Promise<Record<string, unknown>> | Record<string, unknown>) => graphToolResult(async () => {
    try { return await action(); }
    catch (error) { throw error instanceof HostError ? new GraphToolError(error.code, error.message) : error; }
  });

  server.registerTool('list_models', {
    description: 'Discover host capabilities and the current graph without starting Python. With an explicit workspaceRoot, scan bounded saved Python source using an already inspected service environment; never imports project code, captures or provisions runtimes. Discovery returns its own candidates and continuation separately from the active graph. Empty or partial discovery still permits opening an explicit file/class.',
    inputSchema: { workspaceRoot: z.string().min(1).max(4096).optional(),
      offset: z.number().int().min(0).default(0), limit: z.number().int().min(1).max(200).default(50) },
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, (args, extra) => result(async () => {
    await host.checkFreshness(true);
    return { ...host.graph.list(), host: { kind: 'standalone', version, protocolVersion: 1 },
      capabilities: { capture: true, retainedRuns: true, compareRuns: true, localGraphDiff: true, sourceProposals: false,
        training: false, profiling: false, nativeEditor: false, browser: true, graphPanel: true, analytics: false },
      workspaceSelectionRequired: !host.workspace, ...(host.workspace ? { workspaceRoot: host.workspace } : {}),
      ...(args.workspaceRoot ? { discovery: await host.discoverModels(args.workspaceRoot, args.offset, args.limit, extra.signal) } : {}) };
  }));

  host.graph.register(server, false);
  registerMcpViewTools(server, host.graph, host.views, host.id);
  registerMcpCaptureTools(server, host.graph, host.captures);
  registerMcpGraphSetup(server, host.opening, host.setup);
  registerMcpValidationTools(server, host.graph, host.runs, host.id, { diagnostics: false, progress: () => host.approvalProgress() });
  server.registerTool('open_graph_view', {
    description: 'Display the current graph in a browser. Open the returned private URL once, then poll handoffId for the exact display receipt. Does not recapture.',
    inputSchema: { ...modelRefSchema, handoffId: z.string().min(1).max(128).optional() },
    annotations: { readOnlyHint: false, destructiveHint: false, openWorldHint: false },
  }, (args, extra) => result(async () => { await host.checkFreshness(); host.review.close(); return host.graphView.invoke(host.id, args, extra.signal); }));

  server.registerTool('open_graph_diff', {
    description: 'Prepare and display local Git source changes on the shared graph. Supply exact workspaceRoot and optional base (default HEAD), file/class and connected viewId. Uses an already inspected service environment; neither revision is imported or executed. Saved working files include staged and unstaged edits. Returns a private URL only if a new view is needed. Open it once and poll reviewId; only displayed=true proves the exact browser rendered Diff mode. Retained shapes are context, never measurements of reviewed revisions.',
    inputSchema: { workspaceRoot: z.string().min(1).max(4096).optional(), base: z.string().min(1).max(256).optional(),
      file: z.string().min(1).max(4096).optional(), cls: z.string().regex(/^[A-Za-z_]\w*$/).optional(),
      reviewId: z.string().uuid().optional(), viewId: z.string().min(1).max(128).optional() },
    annotations: { readOnlyHint: false, destructiveHint: false, openWorldHint: false },
  }, (args, extra) => result(() => host.review.invoke({ ...args, workspaceRoot: args.workspaceRoot ?? host.workspace }, extra.signal)));

  const absolute = z.string().min(1).max(4096);
  server.registerTool('open_graph', {
    description: 'Capture a saved PyTorch model in an exact local workspace or Git worktree. Start with workspaceRoot, file, cls and optional interpreter/profile. Waits up to 20 seconds for a short capture and returns early for permission or setup. Poll a returned pending requestId until capture succeeds, a model failure is evidenced, the user declines/cancels, or a concrete blocker is reported. TensorViz asks once for persistent plugin permission if it was not granted during installation. Never approve or change permission records on the user’s behalf. After consent, continue without requesting separate TensorViz approvals. Saved recipes are reused; no shape search or automatic dependency installation occurs.',
    inputSchema: { workspaceRoot: absolute.optional(), file: absolute.optional(), cls: z.string().regex(/^[A-Za-z_]\w*$/).optional(),
      interpreter: absolute.optional(), managedRuntime: z.boolean().optional().describe('Request a managed CPU environment for this checkout under the saved TensorViz permission when no suitable Python exists.'),
      profile: z.string().min(1).max(64).optional(),
      waitMs: z.number().int().min(0).max(25000).optional().describe('Bounded completion wait; default 20000 ms. Use 0 for an immediate progress snapshot.'),
      requestId: z.string().uuid().optional(), openingId: z.string().min(1).max(128).optional(), sourceDigest: z.string().regex(/^[a-f0-9]{64}$/).optional(),
      draft: z.record(z.string(), z.json()).optional(), action: z.enum(['retry', 'cancel', 'request_permission', 'select_environment', 'refresh_environments', 'install_dependencies', 'report_blocker']).optional(),
      candidateId: z.string().min(1).max(64).optional().describe('Use a candidate ID from this opening; arbitrary environment paths are not accepted as selection actions.'),
      packages: z.array(z.string().min(1).max(200)).min(1).max(32).optional(),
      blocker: z.object({ reason: z.string().min(1).max(1024), evidence: z.string().min(1).max(2048), nextStep: z.string().min(1).max(1024) }).strict().optional() },
    annotations: { readOnlyHint: false, destructiveHint: true, openWorldHint: true },
  }, ({ waitMs, ...args }, extra) => result(() => host.openAndWait(args, extra.signal, waitMs)));

  let environmentTask: { id: string; controller: AbortController; value?: Record<string, unknown>; error?: { code: string; message: string } } | undefined;
  server.registerTool('get_execution_environment', {
    description: 'Resolve and inspect the exact capture interpreter before a graph exists. Uses an explicit or saved choice, otherwise returns bounded project/known candidates for selection. Never guesses from shell Python or imports a model. Returns a requestId while user permission or metadata inspection is pending. The separate tool runtime may be prepared under that permission; inspection never installs project packages or saves a new selection.',
    inputSchema: { workspaceRoot: absolute.optional(), interpreter: absolute.optional(), requestId: z.string().uuid().optional() },
    annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
  }, args => result(() => {
    if (!args.requestId) {
      if (environmentTask && !environmentTask.value && !environmentTask.error) throw new HostError('ENVIRONMENT_BUSY', 'Poll the current environment request.');
      const task = { id: randomUUID(), controller: new AbortController() } as NonNullable<typeof environmentTask>;
      environmentTask = task;
      void host.inspectEnvironment(args, task.controller.signal).then(value => { task.value = value; }, error => {
        task.error = { code: error instanceof HostError ? error.code : 'ENVIRONMENT_UNAVAILABLE', message: error instanceof HostError ? error.message : 'The selected environment could not be inspected.' };
      });
    } else if (args.requestId !== environmentTask?.id) throw new HostError('UNKNOWN_REQUEST', 'This environment request has expired or belongs to another host.');
    return { requestId: environmentTask!.id, status: environmentTask!.error ? 'blocked' : environmentTask!.value ? 'ready' : 'checking',
      ...(environmentTask!.value ? { environment: environmentTask!.value } : {}), ...(environmentTask!.error ? { error: environmentTask!.error } : {}),
      ...(approvals.pending ? { approval: approvals.pending } : {}), pollAfterMs: 1000 };
  }));

  async function shutdown(): Promise<void> {
    environmentTask?.controller.abort();
    panels.dispose();
    host.dispose();
    await server.close();
  }
  process.once('SIGINT', () => { void shutdown(); });
  process.once('SIGTERM', () => { void shutdown(); });
  process.stdin.once('end', () => { void shutdown(); });
  server.connect(new StdioServerTransport()).catch(() => {
    process.stderr.write('TensorViz could not start its stdio connection. Node.js 22 or newer is required.\n');
    process.exitCode = 1;
  });

}

void start().catch(error => {
  process.stderr.write(`${error instanceof Error ? error.message : 'TensorViz could not start.'}\n`);
  process.exitCode = 1;
});
