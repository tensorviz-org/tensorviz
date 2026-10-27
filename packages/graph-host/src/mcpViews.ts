import { randomUUID } from 'node:crypto';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { GraphToolError, McpGraphService, modelRefSchema, type McpModel } from './mcpGraph';
import type { LocalBrowserHostPublish } from './localBrowserHostTransport';
import { sameTargetIdentity } from './graphIdentity';
import type { LocalBrowserMessage, LocalBrowserTargetIdentity, LocalBrowserViewSelectionMessage } from '@tensorviz/contracts/protocol';

type FocusIntent = { viewId: string; target: LocalBrowserTargetIdentity; nodeIds: string[]; edgeIds: string[] };
interface PendingFocus {
  owner: string;
  intent: FocusIntent;
  finish(error?: GraphToolError): void;
}

/** View ids arrive only from the broker or native host; clients never assign them. */
export class McpViewRegistry {
  private readonly views = new Map<string, LocalBrowserViewSelectionMessage | null>();
  private readonly handoffs = new Map<string, string>();
  private readonly pending = new Map<string, PendingFocus>();
  constructor(private readonly read: () => McpModel | undefined, private readonly publish: (message: LocalBrowserHostPublish) => unknown) {}

  list(): Array<{ viewId: string; current: boolean; selectedNodes: number; selectedEdges: number; tooLarge: boolean }> {
    const model = this.read();
    return [...this.views].map(([viewId, selection]) => ({ viewId,
      current: !!model?.current && !!selection && sameTargetIdentity(model.target, selection.target),
      selectedNodes: selection?.nodeIds.length ?? 0, selectedEdges: selection?.edgeIds.length ?? 0, tooLarge: selection?.tooLarge === true }));
  }

  viewForHandoff(handoffId: string): string | undefined {
    const matches = [...this.handoffs].filter(([, id]) => id === handoffId);
    return matches.length === 1 ? matches[0][0] : undefined;
  }

  displays(viewId: string, target: LocalBrowserTargetIdentity): boolean {
    const selection = this.views.get(viewId);
    return !!selection && sameTargetIdentity(selection.target, target);
  }

  clear(exceptViewId?: string): void {
    for (const viewId of this.views.keys()) if (viewId !== exceptViewId) this.connection(viewId, false);
  }

  connection(viewId: string, connected: boolean): void {
    if (connected) { if (!this.views.has(viewId)) this.views.set(viewId, null); }
    else {
      this.views.delete(viewId);
      this.handoffs.delete(viewId);
      for (const request of [...this.pending.values()]) if (request.intent.viewId === viewId) request.finish(new GraphToolError('VIEW_DISCONNECTED', 'The requested canvas closed or is hidden. Show it and refresh discovery.'));
    }
  }

  cancelOwner(owner: string): void {
    for (const request of [...this.pending.values()]) if (request.owner === owner) request.finish(new GraphToolError('CANCELLED', 'The MCP session ended.'));
  }

  handle(message: LocalBrowserMessage): void {
    if (message.type === 'viewState' && 'viewId' in message) {
      this.connection(message.viewId, message.connected);
      if (message.connected && message.handoffId) this.handoffs.set(message.viewId, message.handoffId);
    } else if (message.type === 'viewSelection' && this.views.has(message.viewId)) {
      this.views.set(message.viewId, message);
    } else if (message.type === 'viewFocusResult') {
      const pending = this.pending.get(message.requestId);
      if (!pending || pending.intent.viewId !== message.viewId || !sameTargetIdentity(pending.intent.target, message.target)) return;
      const model = this.read();
      if (message.ok && model?.current && sameTargetIdentity(model.target, message.target)) {
        this.views.set(message.viewId, { type: 'viewSelection', protocolVersion: message.protocolVersion,
          workspaceSessionId: message.workspaceSessionId, messageId: message.messageId, viewId: message.viewId,
          target: message.target, nodeIds: pending.intent.nodeIds, edgeIds: pending.intent.edgeIds });
      }
      pending.finish(!model?.current || !sameTargetIdentity(model.target, message.target)
        ? new GraphToolError('STALE_TARGET', 'The graph changed while focusing. Refresh discovery.')
        : message.ok ? undefined : new GraphToolError('FOCUS_REFUSED', message.detail));
    }
  }

  selection(viewId: string, target: LocalBrowserTargetIdentity): LocalBrowserViewSelectionMessage {
    if (!this.views.has(viewId)) throw new GraphToolError('VIEW_DISCONNECTED', 'Choose a connected view from list_models.');
    const selected = this.views.get(viewId);
    if (!selected || !sameTargetIdentity(selected.target, target)) throw new GraphToolError('SELECTION_UNAVAILABLE', 'Wait for this canvas to display the current graph, then read its selection again.');
    if (selected.tooLarge) throw new GraphToolError('SELECTION_TOO_LARGE', 'Select at most 100 nodes or connections before reading the selection.');
    return selected;
  }

  focus(owner: string, intent: FocusIntent, signal: AbortSignal): Promise<Record<string, unknown>> {
    if (!this.views.has(intent.viewId)) throw new GraphToolError('VIEW_DISCONNECTED', 'Choose a connected view from list_models.');
    if (signal.aborted) throw new GraphToolError('CANCELLED', 'The focus request was cancelled.');
    if (this.pending.size >= 16) throw new GraphToolError('BUSY', 'Wait for pending canvas focus requests to finish.');
    const requestId = randomUUID();
    return new Promise((resolve, reject) => {
      const finish = (error?: GraphToolError) => {
        if (!this.pending.delete(requestId)) return;
        clearTimeout(timer);
        signal.removeEventListener('abort', abort);
        if (error) reject(error); else resolve({ viewId: intent.viewId, target: intent.target, focused: true });
      };
      const abort = () => finish(new GraphToolError('CANCELLED', 'The focus request was cancelled.'));
      const timer = setTimeout(() => finish(new GraphToolError('VIEW_TIMEOUT', 'The canvas did not confirm focus. Make the requested page visible and retry.')), 5000);
      this.pending.set(requestId, { owner, intent, finish });
      signal.addEventListener('abort', abort, { once: true });
      if (!this.publish({ type: 'focusView', requestId, ...intent })) finish(new GraphToolError('VIEW_DISCONNECTED', 'The canvas host is disconnected.'));
    });
  }
}

export function registerMcpViewTools(server: McpServer, graph: McpGraphService, views: McpViewRegistry, owner: string): void {
  const viewId = z.string().min(1).max(128);
  server.registerTool('get_selection', { description: 'Read the exact canvas view selection, including real semantic connections. Obtain viewId from list_models.',
    inputSchema: { ...modelRefSchema, viewId }, annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
  }, args => graph.toolResult('get_selection', () => {
    const { model, nodes, edges } = graph.require(args);
    const selection = views.selection(args.viewId, model.target);
    graph.select(nodes, selection.nodeIds);
    const edgeIds = new Set(edges.map(edge => edge.edge_id));
    if (selection.edgeIds.some(id => !edgeIds.has(id))) throw new GraphToolError('UNKNOWN_EDGE', 'The selection includes a display-only connection. Expand the relevant module and select a real connection.');
    return { ...graph.reference(model), viewId: args.viewId, nodeIds: selection.nodeIds, edgeIds: selection.edgeIds };
  }));
  server.registerTool('focus_view', { description: 'Select and show nodes, or one real connection, in the specified canvas. Expands ancestor blocks. Changes presentation only and waits for the canvas to confirm.',
    inputSchema: { ...modelRefSchema, viewId, nodeIds: z.array(z.string().min(1).max(512)).max(100).default([]), edgeId: z.string().min(1).max(512).optional() },
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, (args, extra) => graph.toolResult('focus_view', async () => {
    const { model, nodes, edges } = graph.require(args);
    if ((!args.nodeIds.length && !args.edgeId) || (args.nodeIds.length && args.edgeId)) throw new GraphToolError('INVALID_SELECTION', 'Choose nodes or one connection.');
    graph.select(nodes, args.nodeIds);
    if (args.edgeId && !edges.some(edge => edge.edge_id === args.edgeId)) throw new GraphToolError('UNKNOWN_EDGE', 'Use a real edge id from get_graph.');
    return views.focus(owner, { viewId: args.viewId, target: model.target, nodeIds: args.nodeIds, edgeIds: args.edgeId ? [args.edgeId] : [] }, extra.signal);
  }));
}
