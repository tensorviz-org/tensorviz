/** Routes graph tools to their owning VS Code window, preserving MCP session ownership. */

import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import * as path from 'node:path';
import { z } from 'zod';
import type { CallToolResult } from '@modelcontextprotocol/sdk/types.js';
import { GraphToolError, graphToolResult, object, type JsonObject } from '@tensorviz/graph-host/mcpGraph';
import { BusError, callHost, type HostAnnouncement, type HostRegistry } from './hostBus';

export type HostToolHandler = (args: any, extra: any) => CallToolResult | Promise<CallToolResult>;

export interface HostTool {
  config: any;
  handler: HostToolHandler;
}
export type HostToolset = Map<string, HostTool>;

/** Record a registration pass instead of binding it to a live server. */
export function captureTools(register: (server: McpServer) => void): HostToolset {
  const tools: HostToolset = new Map();
  const sink = {
    registerTool(name: string, config: any, handler: HostToolHandler): void { tools.set(name, { config, handler }); },
  } as unknown as McpServer;
  register(sink);
  return tools;
}

function structured(result: CallToolResult): JsonObject {
  if (result.structuredContent && typeof result.structuredContent === 'object' && !Array.isArray(result.structuredContent)) {
    return result.structuredContent as JsonObject;
  }
  const part = Array.isArray(result.content) ? result.content.find(item => item.type === 'text') : undefined;
  const text = part && 'text' in part ? part.text : undefined;
  try { return text ? JSON.parse(text) as JsonObject : {}; }
  catch { return {}; }
}

/** This window's session-bound tools behind the private host bus. */
export class LocalGraphProvider {
  private readonly owners = new Map<string, number>();
  constructor(private readonly load: (owner: string) => HostToolset, private readonly listLocal: () => JsonObject,
    private readonly cancel: (owner: string) => void = () => {}) {}

  list(): JsonObject { return this.listLocal(); }

  release(owner: string): void { this.owners.delete(owner); this.cancel(owner); }

  reap(liveHosts: ReadonlySet<string>): void {
    for (const [owner, touched] of this.owners) {
      if (!liveHosts.has(owner.split('/')[0]) || Date.now() - touched > 31 * 60_000) this.release(owner);
    }
  }

  async call(name: string, args: unknown, signal: AbortSignal, owner: string): Promise<JsonObject> {
    if (!this.owners.has(owner) && this.owners.size >= 64) throw new BusError('TOO_MANY_SESSIONS', 'Close an unused agent connection.');
    this.owners.set(owner, Date.now());
    const tool = this.load(owner).get(name);
    if (!tool) throw new BusError('UNKNOWN_TOOL', `No tool named ${name} is available in this window.`);
    const parsed = z.object(tool.config.inputSchema ?? {}).safeParse(args ?? {});
    if (!parsed.success) throw new BusError('INVALID_ARGUMENTS', 'Arguments do not match this tool schema.');
    const result = await tool.handler(parsed.data, { signal, sessionId: owner, requestId: 'host-bus' });
    if (result.isError) {
      const error = object(structured(result).error);
      throw new BusError(typeof error.code === 'string' ? error.code : 'TOOL_ERROR',
        typeof error.message === 'string' ? error.message : 'The host tool failed.');
    }
    return structured(result);
  }
}

export interface HostSummary {
  hostId: string;
  workspaceFolder: string | null;
  current: boolean;
  reachable: boolean;
  workspaceFolders?: string[];
  activeFile?: string;
  focused?: boolean;
}

export interface HostGraphHubOptions {
  hostId: string;
  workspaceFolder: string | null;
  registry: HostRegistry;
  tools: HostToolset;
  listLocal: () => JsonObject;
  /** Tools safe to run in another window for a model that window owns. */
  remote: ReadonlySet<string>;
  owner: string;
}

/** The single endpoint's view of every window's live graph. */
export class HostGraphHub {
  private selectedHost?: string;
  private readonly usedPeers = new Map<string, HostAnnouncement>();
  constructor(private readonly options: HostGraphHubOptions) {}

  async dispose(): Promise<void> {
    await Promise.allSettled([...this.usedPeers.values()].map(peer =>
      callHost(peer, 'release', { owner: this.options.owner }, 2000)));
    this.usedPeers.clear();
  }

  private async peers(): Promise<HostAnnouncement[]> {
    const records = await this.options.registry.list();
    return records.filter(record => record.hostId !== this.options.hostId);
  }

  private annotate(list: unknown, hostId: string): JsonObject[] {
    return (Array.isArray(list) ? list : []).map(entry => ({ ...object(entry), hostId }));
  }

  async list(): Promise<JsonObject> {
    const local = this.options.listLocal();
    const models = this.annotate(local.models, this.options.hostId);
    const views = this.annotate(local.views, this.options.hostId);
    const metadata = (data: JsonObject) => ({ workspaceFolders: data.workspaceFolders as string[] | undefined,
      activeFile: data.activeFile as string | undefined, focused: data.focused as boolean | undefined });
    const hosts: HostSummary[] = [{ ...metadata(local), hostId: this.options.hostId, workspaceFolder: this.options.workspaceFolder,
      current: true, reachable: true }];
    for (const peer of await this.peers()) {
      try {
        const remote = await callHost(peer, 'describe', {}, 1000) as JsonObject;
        models.push(...this.annotate(remote.models, peer.hostId));
        views.push(...this.annotate(remote.views, peer.hostId));
        hosts.push({ ...metadata(remote), hostId: peer.hostId, workspaceFolder: peer.workspaceFolder, current: true, reachable: true });
      } catch {
        hosts.push({ hostId: peer.hostId, workspaceFolder: peer.workspaceFolder, current: false, reachable: false });
      }
    }
    return { hostId: this.options.hostId, hosts, models, views,
      detail: 'Discover TensorViz windows even before a graph exists. Use open_graph to select a model; use hostId for an ambiguous window or a follow-up without modelId.' };
  }

  register(server: McpServer): void {
    const sink = server as unknown as { registerTool: (name: string, config: any, handler: HostToolHandler) => void };
    for (const [name, tool] of this.options.tools) {
      sink.registerTool(name, { ...tool.config,
        ...(name === 'list_models' ? { description: 'Discover local TensorViz windows, workspace folders, active files, open model revisions and canvas views. Works before a graph exists. Does not execute or open models.' } : {}),
        inputSchema: { ...tool.config.inputSchema,
        hostId: z.string().min(1).max(64).optional().describe('Owning VS Code window from list_models or open_graph.') } },
      (args, extra) => this.invoke(name, args, extra, tool));
    }
  }

  private async invoke(name: string, args: any, extra: any, tool: HostTool): Promise<CallToolResult> {
    if (name === 'list_models') return graphToolResult(() => this.list());
    return graphToolResult(async () => {
      const routedResult = (result: JsonObject, hostId: string): JsonObject => {
        const next = object(result.nextAction);
        return { ...result, hostId, ...(name === 'open_graph' && next.tool === 'open_graph'
          ? { nextAction: { ...next, arguments: { ...object(next.arguments), hostId } } } : {}) };
      };
      const modelId = args && typeof args.modelId === 'string' ? args.modelId : undefined;
      let owner = modelId ? modelId.split(':')[0] : args?.hostId;
      if (modelId && args?.hostId && owner !== args.hostId) throw new GraphToolError('HOST_MISMATCH', 'modelId and hostId name different windows.');
      if (!owner && ['open_graph', 'open_graph_diff', 'get_execution_environment'].includes(name) && !args?.requestId) {
        const hosts = (await this.list()).hosts as HostSummary[];
        const available = hosts.filter(host => host.reachable);
        let matches = args?.file ? available.filter(host => (host.workspaceFolders ?? [host.workspaceFolder]).some(root => {
          if (!root) return false;
          const relative = path.relative(root, args.file);
          return relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
        }) || host.activeFile === args.file) : available.filter(host => host.focused && host.activeFile);
        if (!matches.length && available.length === 1 && !args?.file) matches = available;
        if (!matches.length && args?.file) throw new GraphToolError('OUTSIDE_WORKSPACE', 'That file is outside the open workspace.');
        if (matches.length !== 1) throw new GraphToolError('HOST_REQUIRED', 'Choose hostId from list_models. Several windows match, or the model folder is not open.');
        owner = matches[0].hostId;
      }
      owner ??= this.selectedHost ?? this.options.hostId;
      if (owner && owner !== this.options.hostId) {
        if (!this.options.remote.has(name)) {
          throw new GraphToolError('REMOTE_UNSUPPORTED',
            'The owning window does not support this routed tool. Update TensorViz in that window.');
        }
        const peer = (await this.options.registry.list()).find(record => record.hostId === owner);
        if (!peer) throw new GraphToolError('NO_MODEL', 'That window is no longer available. Refresh list_models.');
        try {
          this.usedPeers.set(owner, peer);
          const result = await callHost(peer, 'call', { name, args, owner: this.options.owner }, undefined, extra?.signal) as JsonObject;
          this.selectedHost = owner;
          return routedResult(result, owner);
        }
        catch (error) {
          if (error instanceof BusError) throw new GraphToolError(error.code, error.message);
          throw error;
        }
      }
      const result = await tool.handler(args ?? {}, extra);
      if (result.isError) {
        const error = object(structured(result).error);
        throw new GraphToolError(String(error.code ?? 'TOOL_ERROR'), String(error.message ?? 'The tool failed.'));
      }
      this.selectedHost = owner;
      return routedResult(structured(result), owner);
    });
  }
}
