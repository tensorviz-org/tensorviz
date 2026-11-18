import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { randomUUID } from 'node:crypto';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import WebSocket from 'ws';
import { z } from 'zod';
import { StandaloneHost } from './host';
import { GraphToolError, graphToolResult, type ModelRef } from '@tensorviz/graph-host/mcpGraph';
import { encodeLocalBrowserMessage, MAX_WIRE_BYTES } from '@tensorviz/graph-host/localBrowserWire';
import { isLocalBrowserMessage, LOCAL_BROWSER_PROTOCOL_VERSION, LOCAL_BROWSER_PUBLISH_FILE_MAX_LENGTH } from '@tensorviz/contracts/protocol';
import { createLocalBrowserViewTransport, type LocalBrowserSocketLike,
  type LocalBrowserViewTransport, type LocalBrowserViewSend } from '@tensorviz/graph-host/viewTransport';

// The existing authenticated browser transport stays on the local server. The
// sandboxed MCP App receives graph messages, never a loopback cookie or URL.
interface Panel {
  id: string;
  touched: number;
  attached: boolean;
  transport?: LocalBrowserViewTransport;
  sequence: number;
  delivered: number;
  bytes: number;
  events: { sequence: number; wire: string }[];
  error?: string;
  exchanging?: boolean;
  exchangeDone?: Promise<void>;
  clientId?: string;
  retiredClientIds: Set<string>;
  controller: AbortController;
}

const VIEW_MESSAGES = new Set(['viewSelection', 'viewFocusResult', 'graphSetupRequest',
  'captureInspectRequest', 'captureInspectCancel', 'captureExportRequest', 'captureExportCancel',
  'graphPublishRequest', 'graphPublishCancel', 'graphReviewRequest', 'graphReviewApplied']);
const VIEW_ID_MESSAGES = new Set(['viewSelection', 'viewFocusResult', 'graphReviewApplied']);
// Publishing sends the graph to tensorviz.dev, so it is never a read.
const READ_ONLY_MESSAGES = new Set(['viewSelection', 'viewFocusResult', 'captureInspectRequest', 'captureInspectCancel',
  'captureExportRequest', 'captureExportCancel']);
const MAX_RETIRED_CLIENT_IDS = 64;
const REQUEST_BYTES = 128 * 1024;
const PANEL_IDLE_TIMEOUT_MS = 60 * 60 * 1000;

export class GraphPanels {
  private readonly panels = new Map<string, Panel>();
  private readonly cleanup = setInterval(() => {
    for (const panel of this.panels.values()) if (Date.now() - panel.touched > PANEL_IDLE_TIMEOUT_MS) this.close(panel.id);
  }, 15_000).unref();

  constructor(private readonly host: StandaloneHost) {}

  open(): string {
    // A tool call is not evidence that a host mounted the app. T3 can attach
    // its bridge later without advertising it at MCP initialize. At capacity,
    // expire only the oldest never-attached opener, not a live or detached
    // frame. Keep distinct IDs: two delayed app mounts must not fight over one.
    if (this.panels.size >= 8) {
      const pending = [...this.panels.values()].find(panel => !panel.attached);
      if (pending) this.close(pending.id);
    }
    if (this.panels.size >= 8) throw new GraphToolError('PANEL_LIMIT', 'Close an unused graph panel before opening another.');
    const id = randomUUID();
    this.panels.set(id, { id, touched: Date.now(), attached: false, sequence: 0, delivered: 0, bytes: 0, events: [], retiredClientIds: new Set(), controller: new AbortController() });
    return id;
  }

  private async connect(panel: Panel, signal: AbortSignal): Promise<void> {
    const model = (this.host.graph.list().models as ModelRef[] | undefined)?.[0];
    const controller = panel.controller;
    const combined = AbortSignal.any([signal, controller.signal]);
    const handoff = model ? await this.host.graphView.invoke(`panel:${panel.id}`, model, combined)
      : await this.host.openingView?.(combined);
    if (!handoff) return;
    if (combined.aborted) throw new GraphToolError('PANEL_CLOSED', 'The graph panel closed.');
    const url = new URL(('pageUrl' in handoff ? handoff.pageUrl : handoff.url) as string);
    if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || !url.port) throw new Error('Invalid local graph origin.');
    let cookie = '';
    const transport = createLocalBrowserViewTransport({
      location: () => url, replaceState: () => {}, storage: null, messageId: randomUUID,
      fetchImpl: async (route, init) => {
        const response = await fetch(new URL(route, url.origin), { ...init, redirect: 'error', signal: controller.signal,
          headers: { ...init?.headers, Origin: url.origin } });
        cookie = response.headers.get('set-cookie')?.split(';')[0] ?? '';
        return response;
      },
      connectSocket: address => new WebSocket(address, { headers: { Origin: url.origin, Cookie: cookie } }) as unknown as LocalBrowserSocketLike,
    });
    panel.transport = transport;
    transport.onMessage(message => {
      if (controller.signal.aborted) return;
      try {
        const wire = encodeLocalBrowserMessage(message);
        const event = { sequence: panel.sequence + 1, wire };
        panel.bytes += Buffer.byteLength(JSON.stringify(event));
        if (panel.events.length >= 256 || panel.bytes > 2 * MAX_WIRE_BYTES) throw new Error();
        panel.sequence++; panel.events.push(event);
      } catch {
        panel.error = 'The graph panel could not keep up with updates. Reopen the panel; the capture is retained.';
        transport.dispose();
      }
    });
    await transport.connect();
    if (controller.signal.aborted) { transport.dispose(); throw new GraphToolError('PANEL_CLOSED', 'The graph panel closed.'); }
  }

  private detach(panel: Panel): void {
    panel.controller.abort(); panel.controller = new AbortController();
    panel.transport?.dispose();
    this.host.graphView.cancelOwner(`panel:${panel.id}`);
    panel.transport = undefined; panel.error = undefined;
    panel.sequence = 0; panel.delivered = 0; panel.bytes = 0; panel.events = [];
  }

  async disconnect(id: string, clientId: string): Promise<void> {
    const panel = this.panels.get(id);
    if (panel?.clientId === clientId) {
      this.detach(panel); panel.touched = Date.now();
      await panel.exchangeDone;
    }
  }

  async exchange(id: string, after: number, messages: Record<string, unknown>[], signal: AbortSignal, clientId?: string): Promise<Record<string, unknown>> {
    const panel = this.panels.get(id);
    if (!panel) throw new GraphToolError('PANEL_CLOSED', 'This graph panel expired or belongs to another connection. Reopen it from this chat.');
    if (panel.exchanging) throw new GraphToolError('PANEL_BUSY', 'Wait for the current panel request to finish.');
    if (clientId && panel.clientId !== clientId &&
        (panel.retiredClientIds.has(clientId) || (panel.clientId && panel.retiredClientIds.size >= MAX_RETIRED_CLIENT_IDS))) {
      throw new GraphToolError('PANEL_REPLACED', 'This panel was replaced or remounted too often. Reopen it from this chat.');
    }
    panel.exchanging = true;
    let finish!: () => void;
    panel.exchangeDone = new Promise<void>(resolve => { finish = resolve; });
    try {
      panel.touched = Date.now();
      if (clientId && panel.clientId !== clientId) {
        if (after || messages.length) throw new GraphToolError('PANEL_REPLACED', 'This panel was reopened in another view.');
        // Keep bounded tombstones: a delayed request from a replaced iframe
        // must never detach the current owner or consume its queued updates.
        if (panel.clientId) panel.retiredClientIds.add(panel.clientId);
        this.detach(panel); panel.clientId = clientId;
      }
      if (after > panel.delivered) throw new GraphToolError('INVALID_CURSOR', 'The panel acknowledged an update it has not received.');
      if (after === 0 && panel.delivered > 0 && panel.events[0]?.sequence !== 1) {
        if (messages.length) throw new GraphToolError('INVALID_CURSOR', 'Reconnect the panel before sending actions.');
        // T3 remounts the iframe on reload. Already acknowledged updates cannot
        // restore it, so request a fresh broker snapshot without recapturing.
        this.detach(panel);
      }
      panel.attached = true;
      while (panel.events[0]?.sequence <= after) panel.bytes -= Buffer.byteLength(JSON.stringify(panel.events.shift()!));
      if (!panel.transport) {
        await this.connect(panel, signal);
      }
      if (panel.error) throw new GraphToolError('PANEL_UNAVAILABLE', panel.error);
      const transport = panel.transport;
      if (messages.length && !transport?.session?.viewId) throw new GraphToolError('PANEL_NOT_READY', 'Wait for the graph panel to connect.');
      // Only Publish carries a file: the gzipped graph, within the website's limit.
      const budget = REQUEST_BYTES + (messages.some(message => message.type === 'graphPublishRequest') ? LOCAL_BROWSER_PUBLISH_FILE_MAX_LENGTH : 0);
      if (Buffer.byteLength(JSON.stringify(messages)) > budget) throw new GraphToolError('REQUEST_LIMIT', 'The panel request is too large.');
      const requests = messages.map(message => {
        if (!VIEW_MESSAGES.has(String(message.type))) throw new GraphToolError('INVALID_PANEL_REQUEST', 'This action is not supported by the graph panel.');
        // View identity always comes from the authenticated broker, never the UI.
        const { role: _role, hostSequence: _sequence, viewId: _viewId, ...body } = message;
        const full = { ...body, ...(VIEW_ID_MESSAGES.has(String(message.type)) ? { viewId: transport!.session!.viewId } : {}),
          protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
          workspaceSessionId: transport!.session!.workspaceSessionId, messageId: randomUUID() };
        if (!isLocalBrowserMessage(full, transport!.session!.workspaceSessionId)) throw new GraphToolError('INVALID_PANEL_REQUEST', 'Invalid graph panel action.');
        return full;
      });
      for (const message of requests) {
        if (!transport!.send(message as LocalBrowserViewSend)) {
          throw new GraphToolError('PANEL_DISCONNECTED', 'The graph panel disconnected. Reopen it from this chat.');
        }
      }
      const events: Panel['events'] = [];
      let bytes = 0;
      for (const event of panel.events) {
        const size = Buffer.byteLength(JSON.stringify(event));
        if (events.length && bytes + size > MAX_WIRE_BYTES) break;
        events.push(event); bytes += size;
      }
      if (events.length) panel.delivered = events[events.length - 1].sequence;
      return { status: transport?.status ?? { phase: 'connecting', hostLink: 'unknown', error: null },
        session: transport?.session ?? null, events, waitingForModel: !transport };
    } finally { panel.exchanging = false; panel.exchangeDone = undefined; finish(); }
  }

  close(id: string, clientId?: string): void {
    if (clientId && this.panels.get(id)?.clientId !== clientId) return;
    this.panels.get(id)?.controller.abort();
    this.panels.get(id)?.transport?.dispose();
    this.host.graphView.cancelOwner(`panel:${id}`);
    this.panels.delete(id);
  }

  dispose(): void {
    clearInterval(this.cleanup);
    for (const id of this.panels.keys()) this.close(id);
  }
}

export async function registerGraphPanel(server: McpServer, host: StandaloneHost, assets: string, version: string): Promise<GraphPanels> {
  const { registerAppTool, RESOURCE_MIME_TYPE } = await import('@modelcontextprotocol/ext-apps/server');
  const { OpenAIUiResourceMetadataSchema, OpenAIUiToolMetadataSchema } = await import('@openai/mcp-extensions/server');
  const panels = new GraphPanels(host);
  const clientId = z.string().min(1).max(128);
  const exchangeSchema = { panelId: z.string().uuid(), clientId: clientId.optional(), after: z.number().int().min(0),
    messages: z.array(z.record(z.string(), z.unknown())).max(32).default([]) };
  const uri = `ui://tensorviz/graph-panel-${version}.html`;
  server.registerResource('graph-panel', uri, { mimeType: RESOURCE_MIME_TYPE }, async () => ({ contents: [{ uri, mimeType: RESOURCE_MIME_TYPE,
    text: await fs.readFile(path.join(assets, 'browser/panel.html'), 'utf8'),
    _meta: { ui: { csp: { connectDomains: [], resourceDomains: [] } },
      'openai/ui': OpenAIUiResourceMetadataSchema.parse({ availableDisplayModes: ['fullscreen'], preferredDisplayMode: 'fullscreen' }) },
  }] }));
  registerAppTool(server, 'open_graph_panel', {
    title: 'Model graph',
    description: 'Open the live TensorViz Canvas beside this conversation in clients supporting MCP Apps. Uses the current capture without rerunning the model. The panel follows updates and shares selection with the agent. An empty panel waits for open_graph. Check list_models for a current view before claiming display. Use open_graph_view in clients without MCP Apps.',
    inputSchema: {},
    annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
    _meta: { ui: { resourceUri: uri }, 'openai/ui': OpenAIUiToolMetadataSchema.parse({ entrypoints: [{ type: 'thread' }], preferredModelDisplayMode: 'fullscreen' }) },
  }, async () => {
    const panelId = panels.open();
    return { content: [{ type: 'text', text: 'Model graph panel requested; display is not confirmed. Use list_models to check the connected view. If no panel appears, use open_graph_view with the current model reference for a browser fallback instead of repeatedly opening panels. Capture and display are verified separately.' }],
      structuredContent: { displayed: false, status: 'awaiting_view', fallbackTool: 'open_graph_view' }, _meta: { panelId } };
  });
  registerAppTool(server, 'graph_panel_exchange', {
    description: 'Private Canvas transport for the TensorViz MCP App. Does not grant execution permission.',
    inputSchema: exchangeSchema,
    annotations: { readOnlyHint: false, destructiveHint: false, openWorldHint: false },
    _meta: { ui: { visibility: ['app'] } },
  }, async (args, extra) => {
    try {
      // UI-only messages keep the existing wire budgets and never enter model context.
      return { content: [], _meta: { panel: await panels.exchange(args.panelId, args.after, args.messages, extra.signal, args.clientId) } };
    } catch (error) { return graphToolResult(() => { throw error; }); }
  });
  registerAppTool(server, 'graph_panel_read', {
    description: 'Read Canvas updates and inspect captured evidence or update view selection. Cannot configure, execute or review model changes.',
    inputSchema: exchangeSchema,
    annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
    _meta: { ui: { visibility: ['app'] } },
  }, async (args, extra) => {
    try {
      // Enforce the read-only boundary before connecting or forwarding any message.
      if (args.messages.some(message => !READ_ONLY_MESSAGES.has(String(message.type)))) {
        throw new GraphToolError('INVALID_PANEL_REQUEST', 'This action requires the graph_panel_exchange tool.');
      }
      return { content: [], _meta: { panel: await panels.exchange(args.panelId, args.after, args.messages, extra.signal, args.clientId) } };
    } catch (error) { return graphToolResult(() => { throw error; }); }
  });
  registerAppTool(server, 'detach_graph_panel', {
    description: 'Disconnect this Canvas frame while retaining its opener briefly for navigation back to the conversation. Does not run a model.',
    inputSchema: { panelId: z.string().uuid(), clientId },
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    _meta: { ui: { visibility: ['app'] } },
  }, async args => { await panels.disconnect(args.panelId, args.clientId); return { content: [] }; });
  registerAppTool(server, 'close_graph_panel', {
    description: 'Release this Canvas view. Does not cancel capture or change source.',
    inputSchema: { panelId: z.string().uuid(), clientId: clientId.optional() },
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    _meta: { ui: { visibility: ['app'] } },
  }, args => { panels.close(args.panelId, args.clientId); return { content: [] }; });
  return panels;
}
