import { createHash } from 'node:crypto';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import type { CallToolResult } from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import type { LocalBrowserTargetIdentity, ProjectionPayload } from '@tensorviz/contracts/protocol';

export interface McpModel {
  target: LocalBrowserTargetIdentity;
  projection: ProjectionPayload;
  current: boolean;
}
export const modelRefSchema = {
  modelId: z.string().min(1).max(128),
  revision: z.string().regex(/^[a-f0-9]{64}$/),
};
export const nodeIdsSchema = z.array(z.string().min(1).max(512)).min(1).max(100);
export type ModelRef = { modelId: string; revision: string };
export type JsonObject = Record<string, unknown>;
export function object(value: unknown): JsonObject {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as JsonObject : {};
}
// The worker stores a single tensor as "1x32x16". Unknown/multiple values stay unknown.
function shape(value: unknown): number[] | null {
  const dims = typeof value === 'string' && /^\d+(?:x\d+)*$/.test(value) ? value.split('x').map(Number) : value;
  return Array.isArray(dims) && dims.every(dim => Number.isSafeInteger(dim) && dim >= 0) ? dims : null;
}
const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const exactCount = (value: unknown): number | null => typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : null;

function captureMetrics(meta: JsonObject, output: number[] | null, greyed: boolean): JsonObject {
  const tensor = object(meta.outputTensor), measured = shape(tensor.shape);
  const current = !greyed && meta.shapeEvidence !== 'stale';
  return {
    parameterCount: current ? exactCount(meta.parameterCount) : null,
    outputTensor: current && meta.shapeEvidence === 'measured' && output && measured
      && output.length === measured.length && output.every((dim, axis) => dim === measured[axis])
      && typeof tensor.dtype === 'string' && /^[a-z][a-z0-9_]{0,63}$/.test(tensor.dtype)
      && exactCount(tensor.logicalBytes) !== null
      ? { shape: measured, dtype: tensor.dtype, logicalBytes: tensor.logicalBytes } : null,
  };
}

export class GraphToolError extends Error {
  constructor(readonly code: string, message: string) { super(message); }
}
/** All tools return standard MCP content and the same machine-readable error shape. */
export async function graphToolResult(action: () => JsonObject | Promise<JsonObject>): Promise<CallToolResult> {
  try {
    const result = await action();
    const text = JSON.stringify(result);
    if (Buffer.byteLength(text) > 512 * 1024) throw new GraphToolError('RESULT_TOO_LARGE', 'Request fewer nodes or a smaller page.');
    return { content: [{ type: 'text', text }], structuredContent: result };
  } catch (error) {
    const result = { error: { code: error instanceof GraphToolError ? error.code : 'UNAVAILABLE',
      message: error instanceof GraphToolError ? error.message : 'The graph operation could not complete. Check the TensorViz host and retry.' } };
    return { isError: true, content: [{ type: 'text', text: JSON.stringify(result) }], structuredContent: result };
  }
}

/** Reads only the owning host's projection; never scans files or executes Python. */
export class McpGraphService {
  private readonly refresh?: () => Promise<void>;
  /** `refreshAfterEmptySlot` keeps a caller that still passes an empty slot before the refresh callback working. */
  constructor(readonly hostId: string, private readonly read: () => McpModel | undefined, private readonly views: () => unknown[] = () => [],
    refresh?: () => Promise<void>, refreshAfterEmptySlot?: () => Promise<void>) {
    this.refresh = refresh ?? refreshAfterEmptySlot;
  }

  async toolResult(_tool: string, action: () => JsonObject | Promise<JsonObject>, _eligible = true): Promise<CallToolResult> {
    return graphToolResult(async () => { await this.refresh?.(); return action(); });
  }

  reference(model: McpModel): ModelRef {
    const target = model.target;
    return {
      modelId: `${this.hostId}:${hash([target.workspaceFolder, target.file, target.cls]).slice(0, 20)}`,
      revision: hash([target.documentVersion, target.sourceDigest, target.backendRevision, model.projection.projectionDigest]),
    };
  }

  list(): JsonObject {
    const model = this.read();
    return { hostId: this.hostId, views: this.views(), models: model ? [{ ...this.reference(model), target: model.target,
      current: model.current, traced: model.projection.traced, truncated: model.projection.truncated ?? false }] : [],
      detail: 'This host exposes its currently open live graph. Use open_graph with a model file and class to select and capture another model.' };
  }

  require(ref: ModelRef): { model: McpModel; nodes: JsonObject[]; edges: JsonObject[] } {
    const model = this.requireModel(ref.modelId);
    const actual = this.reference(model);
    if (!model.current || actual.modelId !== ref.modelId || actual.revision !== ref.revision) {
      throw new GraphToolError('STALE_TARGET', 'The model or source revision changed. Refresh list_models before retrying.');
    }
    const semantic = object(object(model.projection.document).semantic);
    if (!Array.isArray(semantic.nodes) || !Array.isArray(semantic.edges)) throw new GraphToolError('NO_GRAPH', 'Wait for the model projection to finish.');
    return { model, nodes: semantic.nodes.map(object), edges: semantic.edges.map(object) };
  }

  /** Retained evidence may be read while a source refresh is stale. Mutating and
   * focus tools still use require() and its exact current-revision check. */
  requireModel(modelId: string): McpModel {
    const model = this.read();
    if (!model) throw new GraphToolError('NO_MODEL', 'Call open_graph with a model file and class to open and capture it.');
    if (this.reference(model).modelId !== modelId) throw new GraphToolError('STALE_TARGET', 'The selected model changed. Refresh list_models.');
    return model;
  }

  select(nodes: JsonObject[], ids: string[]): JsonObject[] {
    const byId = new Map(nodes.map(node => [node.node_id, node]));
    if (new Set(ids).size !== ids.length || ids.some(id => !byId.has(id))) throw new GraphToolError('UNKNOWN_NODE', 'Use distinct node ids from this graph revision.');
    return ids.map(id => byId.get(id)!);
  }

  node(node: JsonObject, projection: ProjectionPayload): JsonObject {
    const params = object(node.params), meta = object(node.metadata);
    const canvas = object(meta.canvas), data = object(canvas.data);
    const provenance = params._shape_provenance;
    const span = object(projection.spanmap[String(node.node_id)]);
    const source = typeof span.file === 'string' ? { file: span.file, start: span.start, end: span.end, kind: span.kind, digest: span.digest } : null;
    const input = shape(projection.traced ? params._in_shape : meta.inputShape ?? data.inputShape ?? params.input_shape);
    const output = shape(projection.traced
      ? (['observed', 'state', 'declared'].includes(String(provenance)) ? params._out_shape : undefined)
      : meta.outputShape ?? data.outputShape ?? params.output_shape);
    return { nodeId: node.node_id, kind: node.kind, operation: node.op,
      parameters: Object.fromEntries(Object.entries(params).filter(([key]) => !key.startsWith('_'))),
      inputPorts: node.in_ports, outputPorts: node.out_ports,
      ...(meta.captureRef ? { captureRef: meta.captureRef, captureLabel: meta.captureLabel,
        capturedOperationCount: meta.capturedOperationCount, typedPorts: meta.capturePorts,
        sourceEditCapabilities: meta.sourceEditCapabilities ?? [], sourceEditLimitation: meta.sourceEditLimitation ?? null,
        metrics: captureMetrics(meta, output, projection.greyed === true) } : {}),
      parentId: Object.prototype.hasOwnProperty.call(meta, 'parentId') ? meta.parentId : canvas.parentId ?? data.parentId ?? null,
      shapes: { input, output, provenance: ['static', 'observed', 'unreached', 'declared', 'failed',
        'not_observed', 'capture_limited', 'static_only', 'unresolved', 'state'].includes(String(provenance)) ? provenance : 'unknown' },
      source,
    };
  }

  edge(edge: JsonObject): JsonObject {
    return { edgeId: edge.edge_id, sourceNodeId: edge.source_node_id, targetNodeId: edge.target_node_id,
      sourcePort: edge.source_port, targetPort: edge.target_port,
      ...(object(edge.metadata).captureEdge ? { captureEdge: object(edge.metadata).captureEdge } : {}) };
  }

  register(server: McpServer, discovery = true): void {
    const annotations = { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false };
    if (discovery) server.registerTool('list_models', { description: 'Discover the open live model and its exact revision in this VS Code window. Does not execute or open models.', inputSchema: {}, annotations },
      () => this.toolResult('list_models', () => this.list()));
    server.registerTool('get_graph', {
      description: 'Read a graph summary or paginated topology. Optional nodeIds limit the graph to those nodes and their internal connections. Shapes retain provenance.',
      inputSchema: { ...modelRefSchema, mode: z.enum(['summary', 'graph']).default('summary'), nodeIds: nodeIdsSchema.optional(),
        nodeOffset: z.number().int().min(0).default(0), edgeOffset: z.number().int().min(0).default(0), limit: z.number().int().min(1).max(200).default(50) }, annotations,
    }, args => this.toolResult('get_graph', () => {
      const { model, nodes, edges } = this.require(args);
      const selected = args.nodeIds ? this.select(nodes, args.nodeIds) : nodes;
      const kept = new Set(selected.map(node => node.node_id));
      const connections = args.nodeIds ? edges.filter(edge => kept.has(edge.source_node_id) && kept.has(edge.target_node_id)) : edges;
      const base = { ...this.reference(model), target: model.target, traced: model.projection.traced,
        projectionDigest: model.projection.projectionDigest, truncated: model.projection.truncated ?? false,
        nodeCount: selected.length, edgeCount: connections.length,
        ...(object(object(object(model.projection.document).semantic).metadata).capture
          ? { capture: object(object(object(model.projection.document).semantic).metadata).capture } : {}) };
      if (args.mode === 'summary') {
        const operations = new Map<string, number>();
        for (const node of selected) { const op = String(node.op); operations.set(op, (operations.get(op) ?? 0) + 1); }
        return { ...base, operations: Object.fromEntries(operations) };
      }
      return { ...base, nodes: selected.slice(args.nodeOffset, args.nodeOffset + args.limit).map(node => this.node(node, model.projection)),
        edges: connections.slice(args.edgeOffset, args.edgeOffset + args.limit).map(edge => this.edge(edge)),
        nextNodeOffset: args.nodeOffset + args.limit < selected.length ? args.nodeOffset + args.limit : null,
        nextEdgeOffset: args.edgeOffset + args.limit < connections.length ? args.edgeOffset + args.limit : null };
    }, args.nodeOffset === 0 && args.edgeOffset === 0));
    server.registerTool('inspect_nodes', { description: 'Inspect selected layers, source references, tensor-shape provenance and connected edges. Does not execute the model.',
      inputSchema: { ...modelRefSchema, nodeIds: nodeIdsSchema, edgeOffset: z.number().int().min(0).default(0), limit: z.number().int().min(1).max(200).default(50) }, annotations,
    }, args => this.toolResult('inspect_nodes', () => {
      const { model, nodes, edges } = this.require(args);
      const selected = this.select(nodes, args.nodeIds), ids = new Set(args.nodeIds);
      const connections = edges.filter(edge => ids.has(String(edge.source_node_id)) || ids.has(String(edge.target_node_id)));
      return { ...this.reference(model), target: model.target,
        nodes: selected.map(node => this.node(node, model.projection)),
        edges: connections.slice(args.edgeOffset, args.edgeOffset + args.limit).map(edge => this.edge(edge)), edgeCount: connections.length,
        nextEdgeOffset: args.edgeOffset + args.limit < connections.length ? args.edgeOffset + args.limit : null };
    }, args.edgeOffset === 0));
  }
}
