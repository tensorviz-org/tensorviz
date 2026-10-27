import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { GraphToolError, graphToolResult, McpGraphService, modelRefSchema, type ModelRef, type JsonObject } from './mcpGraph';
import type { LocalBrowserViewLink } from './localBrowserHostTransport';
import type { LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import type { McpViewRegistry } from './mcpViews';

interface BrowserViewHost {
  prepareBrowserView(target: LocalBrowserTargetIdentity, signal: AbortSignal): Promise<LocalBrowserViewLink>;
  browserSessionCurrent(sessionId: string): boolean;
  mcpViews: McpViewRegistry;
}
interface Handoff {
  ref: ModelRef;
  controller: AbortController;
  link?: LocalBrowserViewLink;
}

/** Display receipts are independent of capture success. One handoff per MCP owner. */
export class McpGraphView {
  private readonly requests = new Map<string, Handoff>();
  constructor(private readonly graph: McpGraphService, private readonly host: () => BrowserViewHost | undefined) {}

  cancelOwner(owner: string): void {
    this.requests.get(owner)?.controller.abort();
    this.requests.delete(owner);
  }

  async invoke(owner: string, args: ModelRef & { handoffId?: string }, signal: AbortSignal): Promise<JsonObject> {
    const { model } = this.graph.require(args);
    const host = this.host();
    if (!host) throw new GraphToolError('UNAVAILABLE', 'Open the graph in TensorViz first.');
    if (signal.aborted) throw new GraphToolError('CANCELLED', 'The browser handoff was cancelled.');
    let request = this.requests.get(owner);
    if (!args.handoffId) {
      this.cancelOwner(owner);
      request = { ref: { modelId: args.modelId, revision: args.revision }, controller: new AbortController() };
      this.requests.set(owner, request);
      try {
        request.link = await host.prepareBrowserView(model.target, AbortSignal.any([signal, request.controller.signal]));
        this.graph.require(args);
        if (request.controller.signal.aborted || signal.aborted) throw new GraphToolError('CANCELLED', 'The browser handoff was cancelled.');
      } catch (error) {
        if (this.requests.get(owner) === request) this.cancelOwner(owner);
        throw error;
      }
    } else if (!request?.link || request.link.handoffId !== args.handoffId || request.ref.modelId !== args.modelId || request.ref.revision !== args.revision) {
      throw new GraphToolError('UNKNOWN_HANDOFF', 'Use this connection’s handoffId and original model reference, or request a new browser view.');
    }
    const link = request!.link!;
    if (!host.browserSessionCurrent(link.workspaceSessionId)) throw new GraphToolError('STALE_SESSION', 'The browser session ended. Request a new browser view.');
    const viewId = host.mcpViews.viewForHandoff(link.handoffId);
    const displayed = !!viewId && host.mcpViews.displays(viewId, model.target);
    if (!viewId && Date.parse(link.expiresAt) <= Date.now()) throw new GraphToolError('VIEW_LINK_EXPIRED', 'The browser did not connect before the link expired. Request a new view link; do not reopen the old URL.');
    return { ...request!.ref, handoffId: link.handoffId, status: displayed ? 'ready' : viewId ? 'connecting' : 'awaiting_browser', displayed,
      ...(viewId ? { viewId } : {}),
      ...(!args.handoffId ? { url: link.pageUrl, linkExpiresAt: link.expiresAt } : {}),
      nextAction: displayed
        ? { kind: 'inspect', instruction: 'Use this viewId for get_selection and focus_view. This confirms display, not a new capture.' }
        : { kind: args.handoffId ? 'poll' : 'open_browser', retryAfterMs: 1000,
          instruction: args.handoffId
            ? 'Poll with the same model reference and handoffId after the browser opens. Do not open another tab or recapture. If the browser shows an error, report it instead of polling indefinitely.'
            : 'Open url once in a visible browser, then poll with this model reference and handoffId until displayed is true. Keep the URL private. If no browser can be opened, report display as unavailable; MCP graph inspection remains usable.' } };
  }
}

export function registerMcpGraphView(server: McpServer, handoff: McpGraphView, owner: string): void {
  server.registerTool('open_graph_view', {
    description: 'Display the current TensorViz graph in the coding app’s browser panel. Returns a private, single-use browser URL; the agent opens it with its browser tool, then polls handoffId for the exact viewId and display receipt. Reuses the graph without capture or source changes. VS Code stays open as host.',
    inputSchema: { ...modelRefSchema, handoffId: z.string().min(1).max(128).optional() },
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
  }, (args, extra) => graphToolResult(() => handoff.invoke(owner, args, extra.signal)));
}
