import { ModelEdge, ModelGraph, ModelNode } from '../types';
import {
  ArchitectureNode,
  ArchitectureNodeKind,
  GraphDocumentV2,
  GraphLayout,
} from '../types/graphDocument';
import { getLayerByType } from '../data/layerTypes';
import { residualMetadata } from '../canvas/graphSemantics';
import { classifyPassThroughCalls } from './passThroughPresentation';

const SEMANTIC_DATA_KEYS = [
  'captureRef',
  'captureDiagnostic',
  'capturePorts',
  'captureLabel',
  'captureSourceExpression',
  'captureSliceAxis',
  'capturedOperationCount',
  'sourceEditCapabilities',
  'sourceEditLimitation',
  'layerType',
  'parentId',
  'type',
  'isGroup',
  'moduleInstanceId',
  'moduleTemplateId',
  'moduleEntryNodeId',
  'moduleExitNodeId',
  'moduleLayerCount',
  'moduleCapture',
  'moduleMembers',
  'modulePreviewNodes',
  'modulePreviewEdges',
  'repeatCount',
  'repeatBlockLabel',
  'repeatHidden',
  'shapeEvidence',
  'parameterCount',
  'outputTensor',
] as const;

const VIEW_DATA_KEYS = [
  'captureDetailLevel',
  'operationDetailsShown',
  'passThroughShown',
  'extent',
  'moduleLayoutState',
] as const;

export interface GraphDocumentContext {
  graphId?: string;
  name: string;
  description?: string;
  revision: number;
  operations?: Array<Record<string, unknown>>;
  layout?: GraphLayout;
}

function jsonValue<T>(value: T): T {
  if (Array.isArray(value)) {
    return value
      .filter((item) => typeof item !== 'function' && item !== undefined)
      .map((item) => jsonValue(item)) as T;
  }
  if (value && typeof value === 'object') {
    const clean: Record<string, unknown> = {};
    Object.entries(value as Record<string, unknown>).forEach(([key, item]) => {
      if (typeof item !== 'function' && item !== undefined) clean[key] = jsonValue(item);
    });
    return clean as T;
  }
  return value;
}

function pick(source: Record<string, unknown>, keys: readonly string[]): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  keys.forEach((key) => {
    if (source[key] !== undefined) result[key] = jsonValue(source[key]);
  });
  return result;
}

function nodeKind(node: ModelNode): ArchitectureNodeKind {
  const candidate = String(
    node.data?.layerType?.id || node.data?.type || node.type || ''
  ).toLowerCase();
  if (candidate === 'input') return 'input';
  if (candidate === 'output') return 'output';
  if (node.type === 'moduleNode' || candidate === 'group' || candidate === 'module' || candidate === 'modulenode' || node.data?.isGroup) {
    return 'module';
  }
  return 'layer';
}

function layerTypeForSemanticNode(node: ArchitectureNode): ModelNode['data']['layerType'] {
  const explicit = node.metadata?.layerType;
  const explicitLayerType = explicit && typeof explicit === 'object'
    ? explicit as Record<string, unknown>
    : null;
  const registryKey = String(
    explicitLayerType?.id
      || (node.kind === 'input' || node.kind === 'output' || node.kind === 'module'
        ? node.kind
        : node.op)
  );
  const registered = getLayerByType(registryKey);

  if (explicitLayerType) {
    return jsonValue({
      ...(registered || {}),
      ...explicitLayerType,
    }) as ModelNode['data']['layerType'];
  }
  if (registered) return jsonValue(registered);

  const displayName = node.op.replace(/^(?:torch\.)?nn\./, '') || 'Layer';
  return {
    id: displayName.toLowerCase().replace(/[^a-z0-9_]/g, '') || 'layer',
    name: displayName,
    displayName,
    description: '',
    icon: 'Box',
    category: 'misc',
    parameters: [],
  };
}

function canvasNodeState(node: ModelNode): Record<string, unknown> {
  const source = jsonValue(node as unknown as Record<string, unknown>);
  return pick(source, ['type', 'draggable', 'selectable', 'connectable']);
}

function canvasEdgeState(edge: ModelEdge): Record<string, unknown> {
  const source = jsonValue(edge as unknown as Record<string, unknown>);
  const canvas = pick(source, ['type', 'label', 'animated', 'selectable', 'markerStart', 'markerEnd']);
  const edgeData = source.data && typeof source.data === 'object'
    ? Object.fromEntries(
        Object.entries(source.data as Record<string, unknown>).filter(
          ([key, value]) => !['synthetic', 'diffStatus', 'sourceHandle', 'targetHandle', 'role', 'roleEvidence', 'originalEdgeIds', 'captureEdge'].includes(key)
            && typeof value !== 'function'
            && value !== undefined
        )
      )
    : undefined;
  if (edgeData && Object.keys(edgeData).length > 0) canvas.data = jsonValue(edgeData);
  return canvas;
}

function canvasNodeType(kind: ArchitectureNodeKind): string {
  if (kind === 'input') return 'inputNode';
  if (kind === 'output') return 'outputNode';
  if (kind === 'module') return 'moduleNode';
  return 'layerNode';
}

export function modelGraphToDocumentV2(
  graph: ModelGraph,
  context: GraphDocumentContext
): GraphDocumentV2 {
  const graphId = context.graphId || 'pending';
  const nodePositions: Record<string, { x: number; y: number }> = {};
  const nodeCanvas: Record<string, Record<string, unknown>> = {};
  const nodeData: Record<string, Record<string, unknown>> = {};
  const edgeCanvas: Record<string, Record<string, unknown>> = {};
  const selection: string[] = [];
  const collapsedModules: string[] = [];
  const inputs: Array<Record<string, unknown>> = [];
  const realEdges = graph.edges.filter((edge) => !edge.data?.synthetic);
  const incoming = new Map<string, Set<string>>(), outgoing = new Map<string, Set<string>>();
  for (const edge of realEdges) {
    if (!incoming.has(edge.target)) incoming.set(edge.target, new Set());
    if (!outgoing.has(edge.source)) outgoing.set(edge.source, new Set());
    incoming.get(edge.target)!.add(edge.targetHandle || edge.data?.targetHandle || 'x');
    outgoing.get(edge.source)!.add(edge.sourceHandle || edge.data?.sourceHandle || 'y');
  }

  const nodes = graph.nodes.map((node) => {
    const data = (node.data || {}) as unknown as Record<string, unknown>;
    const kind = nodeKind(node);
    nodePositions[node.id] = {
      x: Number(node.position?.x) || 0,
      y: Number(node.position?.y) || 0,
    };
    nodeCanvas[node.id] = canvasNodeState(node);
    nodeData[node.id] = pick(data, VIEW_DATA_KEYS);
    if (node.selected || node.data?.isSelected) selection.push(node.id);
    if ((kind === 'module' && node.data?.expanded !== true) || node.data?.collapsed) {
      collapsedModules.push(node.id);
    }
    if (kind === 'input') {
      inputs.push({
        name: node.id,
        shape: jsonValue(node.data?.parameters?.shape ?? node.data?.inputShape ?? []),
      });
    }
    return {
      node_id: node.id,
      kind,
      op: String(node.data?.layerType?.name || node.data?.layerType?.id || node.type || 'Unknown'),
      params: jsonValue(node.data?.parameters || {}),
      in_ports: jsonValue(node.data?.inputPorts ?? (kind === 'input' ? [] : Array.from(incoming.get(node.id) ?? ['x']))),
      out_ports: jsonValue(node.data?.outputPorts ?? (kind === 'output' ? [] : Array.from(outgoing.get(node.id) ?? ['y']))),
      metadata: pick(data, SEMANTIC_DATA_KEYS),
    };
  });

  const edges = realEdges.map((edge) => {
    edgeCanvas[edge.id] = canvasEdgeState(edge);
    return {
      edge_id: edge.id,
      source_node_id: edge.source,
      source_port: edge.sourceHandle || edge.data?.sourceHandle || 'y',
      target_node_id: edge.target,
      target_port: edge.targetHandle || edge.data?.targetHandle || 'x',
      metadata: { ...residualMetadata(edge.data), ...(edge.data?.captureEdge ? { captureEdge: jsonValue(edge.data.captureEdge) } : {}) },
    };
  });

  return {
    schemaVersion: 2,
    semantic: {
      graph_id: graphId,
      name: context.name,
      revision: context.revision,
      inputs,
      nodes,
      edges,
      modules: [],
      operations: jsonValue(context.operations || []),
      // Preserve whatever the graph already carried. Rebuilding this object from
      // scratch erased `template_instances` on the first save after an AI build,
      // which is how the canvas knows a large flat-looking graph is really N
      // instantiated blocks.
      metadata: {
        ...jsonValue((graph as { metadata?: Record<string, unknown> }).metadata || {}),
        ...(context.description !== undefined ? { description: context.description } : {}),
      },
    },
    view: {
      graph_id: graphId,
      node_positions: nodePositions,
      collapsed_modules: collapsedModules,
      selection,
      layout: {
        strategy: 'manual',
        ...jsonValue(context.layout || {}),
        node_canvas: nodeCanvas,
        node_data: nodeData,
        edge_canvas: edgeCanvas,
      },
    },
  };
}

function migrationPayload(document: GraphDocumentV2): ModelGraph | null {
  const migration = document.semantic.metadata?.migration;
  if (!migration || typeof migration !== 'object') return null;
  const payload = (migration as Record<string, unknown>).legacyPayload;
  if (!payload || typeof payload !== 'object') return null;
  const graph = payload as Partial<ModelGraph>;
  if (!Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) return null;
  const legacy = jsonValue(graph as ModelGraph);
  const converted = modelGraphToDocumentV2(legacy, {
    graphId: document.semantic.graph_id,
    name: document.semantic.name,
    revision: document.semantic.revision,
    operations: document.semantic.operations,
    layout: document.view.layout,
  });
  const semanticCanvasCore = (semantic: GraphDocumentV2['semantic']) => jsonValue({
    inputs: semantic.inputs,
    nodes: semantic.nodes,
    edges: semantic.edges,
    modules: semantic.modules,
  });
  // The rollback payload is a faithful read projection only until the semantic
  // graph changes. AI commits preserve the historical bytes, so stale payloads
  // must not mask the newly committed server graph.
  return JSON.stringify(semanticCanvasCore(converted.semantic))
    === JSON.stringify(semanticCanvasCore(document.semantic))
    ? legacy
    : null;
}

export function graphDocumentV2ToModelGraph(document: GraphDocumentV2): ModelGraph {
  const legacy = migrationPayload(document);
  if (legacy) return legacy;

  const layout = document.view.layout || {};
  const nodeCanvas = layout.node_canvas || {};
  const nodeData = layout.node_data || {};
  const edgeCanvas = layout.edge_canvas || {};
  const selected = new Set(document.view.selection || []);
  const collapsed = new Set(document.view.collapsed_modules || []);

  const nodes = document.semantic.nodes.map((node) => {
    const semanticData = jsonValue(node.metadata || {});
    const viewData = jsonValue(nodeData[node.node_id] || {});
    const isSelected = selected.has(node.node_id);
    const data = {
      ...semanticData,
      ...viewData,
      // Semantic ownership wins over an older duplicate in the canvas layout.
      ...(Object.prototype.hasOwnProperty.call(semanticData, 'parentId') ? { parentId: semanticData.parentId } : {}),
      layerType: layerTypeForSemanticNode(node),
      parameters: jsonValue(node.params || {}),
      ...(Array.isArray(node.in_ports) ? { inputPorts: [...node.in_ports] } : {}),
      ...(Array.isArray(node.out_ports) ? { outputPorts: [...node.out_ports] } : {}),
      ...(isSelected ? { isSelected: true } : {}),
      ...(node.kind === 'module' ? { expanded: !collapsed.has(node.node_id) } : {}),
    } as unknown as ModelNode['data'];
    return {
      ...(jsonValue(nodeCanvas[node.node_id] || {}) as Partial<ModelNode>),
      id: node.node_id,
      type: (nodeCanvas[node.node_id]?.type as string | undefined)
        || canvasNodeType(node.kind),
      position: document.view.node_positions[node.node_id] || { x: 0, y: 0 },
      ...(isSelected ? { selected: true } : {}),
      data,
    } as ModelNode;
  });

  const edges = document.semantic.edges.map((edge) => ({
    ...(jsonValue(edgeCanvas[edge.edge_id] || {}) as Partial<ModelEdge>),
    id: edge.edge_id,
    source: edge.source_node_id,
    target: edge.target_node_id,
    data: { ...(edgeCanvas[edge.edge_id]?.data as ModelEdge['data']), ...residualMetadata(edge.metadata),
      ...(edge.metadata?.captureEdge ? { captureEdge: jsonValue(edge.metadata.captureEdge) } : {}) },
    ...(edge.source_port !== 'y' ? { sourceHandle: edge.source_port } : {}),
    ...(edge.target_port !== 'x' ? { targetHandle: edge.target_port } : {}),
  } as ModelEdge));
  // Graph-level metadata was being dropped here, which silently cost two things.
  //
  // `template_instances` records which nodes belong to which instantiated block.
  // Without it the canvas cannot know a 107-node AI build is really 12
  // `full_transformer_block`s, so ModelBuilder falls back to autoModularize's
  // repeat detection — measured on a real committed graph, that produced 3
  // op-pattern groups (MultiheadAttention, Embedding, Linear) instead of the 12
  // declared blocks, leaving the model an unreadable 1568x25px ribbon.
  //
  // The outbound direction then rebuilt semantic.metadata from scratch, so the
  // first user save ERASED the record entirely.
  //
  // Scope of that erasure, measured rather than assumed: the backend resolver also
  // reads graph.metadata["template_instances"], but it has a `<instance_id>__`
  // prefix fallback that derives entry/exit from topology, and on this graph it
  // resolved `encoder_block_01` to exactly the declared entry and exit nodes with
  // the metadata gone. So referencing a block by name still works — the loss is
  // the canvas's, which has no such fallback and guesses instead.
  return { nodes: classifyPassThroughCalls(document, nodes, edges), edges, metadata: jsonValue(document.semantic.metadata || {}) };
}

export function isGraphDocumentV2(value: unknown): value is GraphDocumentV2 {
  return Boolean(value && typeof value === 'object' && (value as { schemaVersion?: unknown }).schemaVersion === 2);
}

export function legacyModelGraphToDocumentV2(
  graph: ModelGraph,
  context: GraphDocumentContext
): GraphDocumentV2 {
  const document = modelGraphToDocumentV2(graph, context);
  document.semantic.metadata = {
    ...document.semantic.metadata,
    migration: {
      sourceSchema: 'ModelGraph',
      legacyPayload: jsonValue(graph),
    },
  };
  return document;
}
