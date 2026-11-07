import { ModelGraph } from '../types';
import type { LayoutDirection } from '../canvas/edgeAnchors';
import { ArchitectureNode, GraphDocumentV2 } from '../types/graphDocument';
import { getLayerByType } from '../data/layerTypes';
import { modelGraphToDocumentV2 } from './graphDocumentAdapter';
import { exactCount, outputTensorMetadata, residualMetadata, tensorEvidence, tensorShapes } from '../canvas/graphSemantics';
import { GraphCollaboration, portableSnapshotId, validateCollaboration } from './graphCollaboration';
import { requirePortableCapture, type PortableCapture } from './portableCapture';
import { captureReference, capturePorts, captureEdge, validateCaptureDocument } from './portableCaptureDocument';

export const GRAPH_FILE_MAX_BYTES = 96 * 1024 * 1024;
const MAX_NODES = 20000;
const MAX_EDGES = 50000;
const UNSAFE_KEYS = new Set(['__proto__', 'prototype', 'constructor']);
const PRIVATE_KEY = /^(?:_|(?:source|sourceCode|source_code|code|file|file_path|path|location|token|session|sessionId|credential|secret|api[_-]?key|authorization|password)$)/i;
const LOCAL_PATH = /^(?:\/|[a-z]:[\\/]|\\\\|file:|vscode:)/i;
export type GraphFileStatus = 'static' | 'traced' | 'stale';
export interface GraphFile {
  format: 'tensorviz.graph';
  version: 3 | 4;
  capture?: PortableCapture;
  snapshotId: string;
  collaboration: GraphCollaboration;
  exportedAt: string;
  status: GraphFileStatus;
  warningCount: number;
  document: GraphDocumentV2;
}
export interface GraphFileOptions {
  name: string;
  status: GraphFileStatus;
  warningCount?: number;
  capture?: PortableCapture;
  direction?: LayoutDirection;
}

class GraphFileError extends Error {}
function fail(message: string): never { throw new GraphFileError(message); }
function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail(`Invalid ${label}: expected an object.`);
  return value as Record<string, unknown>;
}
function list(value: unknown, label: string, limit = MAX_NODES): unknown[] {
  if (!Array.isArray(value) || value.length > limit) fail(`Invalid ${label}: too many or missing entries.`);
  return value as unknown[];
}
function text(value: unknown, label: string, max = 256): string {
  if (typeof value !== 'string' || !value.trim() || value.length > max || value.split('').some(char => char.charCodeAt(0) < 32)) fail(`Invalid ${label}.`);
  return value as string;
}
function identifier(value: unknown, label: string): string {
  const id = text(value, label);
  if (UNSAFE_KEYS.has(id) || LOCAL_PATH.test(id)) fail(`Unsafe ${label}.`);
  return id;
}
function coordinate(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || Math.abs(value) > 1e7) fail('Invalid layout coordinate.');
  return value as number;
}

/** Only JSON parameter values cross the file boundary; internal source and
 * session fields are never part of the portable format. */
function parameterValue(value: unknown, depth = 0): unknown {
  if (depth > 8) fail('Parameters are nested too deeply.');
  if (value === null || typeof value === 'boolean') return value;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) fail('Invalid numeric parameter.');
    return value;
  }
  if (typeof value === 'string') {
    if (value.length > 4096) fail('A parameter is too long.');
    return LOCAL_PATH.test(value) ? undefined : value;
  }
  if (Array.isArray(value)) {
    if (value.length > 4096) fail('A parameter array is too large.');
    return value.map(v => parameterValue(v, depth + 1)).filter(v => v !== undefined);
  }
  const object = record(value, 'parameter');
  if (Object.keys(object).length > 256) fail('Too many parameters.');
  const entries: Array<[string, unknown]> = [];
  for (const [key, item] of Object.entries(object)) {
    if (UNSAFE_KEYS.has(key)) fail('Unsafe parameter key.');
    if (PRIVATE_KEY.test(key)) continue;
    text(key, 'parameter name');
    const clean = parameterValue(item, depth + 1);
    if (clean !== undefined) entries.push([key, clean]);
  }
  return Object.fromEntries(entries);
}
function parameters(value: unknown): Record<string, unknown> {
  record(value, 'parameters');
  return parameterValue(value) as Record<string, unknown>;
}
function shape(value: unknown): number[] {
  return list(value, 'shape', 32).map(v => {
    if (typeof v !== 'number' || !Number.isFinite(v) || v < -1 || v > 1e12) fail('Invalid tensor shape.');
    return v as number;
  });
}
function metadata(value: unknown, id: string, kind: string, op: string, captured = false): Record<string, unknown> {
  const input = record(value, 'node metadata');
  const layer = input.layerType === undefined ? {} : record(input.layerType, 'layer type');
  const layerId = layer.id === undefined ? op.replace(/^(?:torch\.)?nn\./, '').toLowerCase() : identifier(layer.id, 'layer type');
  const registered = getLayerByType(layerId);
  const output: Record<string, unknown> = {
    layerType: {
      ...(registered ?? { id: layerId, description: '', category: 'misc', icon: 'Box', parameters: [] }),
      name: layer.name === undefined ? op : text(layer.name, 'layer name', 512),
      displayName: layer.displayName === undefined ? op : text(layer.displayName, 'layer label', 512),
    },
  };
  if (captured) {
    output.captureRef = captureReference(input.captureRef);
    output.capturePorts = capturePorts(input.capturePorts);
    if (input.captureLabel !== undefined) output.captureLabel = text(input.captureLabel, 'capture label', 4096);
    if (['repeatCount', 'repeatHidden', 'moduleCapture'].some(key => input[key] !== undefined)) fail('Captured occurrences cannot be replaced by legacy repeat or depth summaries.');
  } else if (input.captureRef !== undefined || input.capturePorts !== undefined) fail('Captured records require graph file version 4 and their complete capture.');
  if (input.parentId !== undefined) output.parentId = identifier(input.parentId, 'parent');
  if (kind === 'module') {
    output.moduleInstanceId = id;
    output.moduleTemplateId = typeof input.moduleTemplateId === 'string' && input.moduleTemplateId.trim() && !LOCAL_PATH.test(input.moduleTemplateId)
      ? text(input.moduleTemplateId, 'module label') : op;
    output.isGroup = true;
    if (input.moduleCapture !== undefined) {
      const capture = record(input.moduleCapture, 'module capture');
      if (capture.status !== 'depth-limited' || !Number.isSafeInteger(capture.maxDepth) || Number(capture.maxDepth) < 0) {
        fail('Invalid module capture limit.');
      }
      output.moduleCapture = { status: 'depth-limited', maxDepth: capture.maxDepth };
    }
  }
  for (const field of ['inputShape', 'outputShape']) if (input[field] !== undefined) output[field] = shape(input[field]);
  if (['measured', 'inferred', 'stale', 'unknown'].includes(String(input.shapeEvidence))) output.shapeEvidence = input.shapeEvidence;
  if (input.parameterCount !== undefined) {
    if (exactCount(input.parameterCount) === undefined) fail('Invalid exact parameter count.');
    output.parameterCount = input.parameterCount;
  }
  if (input.outputTensor !== undefined) {
    const tensor = outputTensorMetadata(input.outputTensor);
    if (!tensor) fail('Invalid output tensor measurement.');
    output.outputTensor = tensor;
  }
  if (input.repeatCount !== undefined) {
    if (!Number.isSafeInteger(input.repeatCount) || Number(input.repeatCount) < 1 || Number(input.repeatCount) > MAX_NODES) fail('Invalid repeat count.');
    output.repeatCount = input.repeatCount;
  }
  if (input.repeatBlockLabel !== undefined) output.repeatBlockLabel = text(input.repeatBlockLabel, 'repeat label');
  if (input.repeatHidden !== undefined) {
    if (typeof input.repeatHidden !== 'boolean') fail('Invalid repeat visibility.');
    output.repeatHidden = input.repeatHidden;
  }
  if (typeof input.resolutionCode === 'string' && /^[A-Z_]{1,64}$/.test(input.resolutionCode)) {
    output.resolutionCode = input.resolutionCode;
    output.resolutionSeverity = 'warning';
    output.resolutionDetail = 'Some structure was unresolved when this graph was exported.';
    output.moduleStatus = 'warning';
  }
  return output;
}

/** Reconstruct the documented subset instead of trusting arbitrary layout,
 * migration payloads or metadata from a downloaded file. */
function portableDocument(value: unknown, captured = false): GraphDocumentV2 {
  const document = record(value, 'graph document');
  if (document.schemaVersion !== 2) fail('Unsupported graph document version.');
  const semantic = record(document.semantic, 'graph');
  const view = record(document.view, 'view');
  const positions = record(view.node_positions, 'positions');
  const direction = view.layout === undefined ? undefined : record(view.layout, 'layout').direction;
  if (direction !== undefined && direction !== 'horizontal' && direction !== 'vertical') fail('Invalid layout direction.');
  const nodes: ArchitectureNode[] = list(semantic.nodes, 'nodes').map(value => {
    const node = record(value, 'node');
    const id = identifier(node.node_id, 'node id');
    const kind = text(node.kind, 'node kind');
    if (!['input', 'output', 'layer', 'module', 'op'].includes(kind)) fail('Unknown node kind.');
    const op = text(node.op, 'operation', 512);
    return {
      node_id: id, kind: kind as ArchitectureNode['kind'], op, params: captured ? {} : parameters(node.params),
      in_ports: list(node.in_ports, 'input ports', captured ? 32768 : 64).map(p => identifier(p, 'port')),
      out_ports: list(node.out_ports, 'output ports', captured ? 32768 : 64).map(p => identifier(p, 'port')),
      metadata: metadata(node.metadata, id, kind, op, captured),
    };
  });
  if (!nodes.length) fail('This graph file has no nodes.');
  const byId = new Map(nodes.map(n => [n.node_id, n]));
  if (byId.size !== nodes.length) fail('Duplicate node ids.');
  for (const node of nodes) {
    const visited = new Set([node.node_id]);
    let parent = node.metadata.parentId as string | undefined;
    while (parent) {
      if (visited.has(parent)) fail('Module parent cycle.');
      visited.add(parent);
      if (visited.size > 64) fail('Module hierarchy is too deep.');
      const owner = byId.get(parent);
      if (!owner || owner.kind !== 'module') fail('Invalid or missing module parent.');
      parent = owner.metadata.parentId as string | undefined;
    }
  }
  const children = new Map<string, string[]>();
  for (const node of nodes) {
    const parent = node.metadata.parentId as string | undefined;
    if (!parent) continue;
    let members = children.get(parent);
    if (!members) children.set(parent, members = []);
    members.push(node.node_id);
  }
  for (const node of nodes.filter(n => n.kind === 'module')) {
    const members = children.get(node.node_id) ?? [];
    node.metadata.moduleMembers = members;
    const pending = [...members];
    let layers = 0;
    while (pending.length) {
      const nested = children.get(pending.pop()!);
      if (nested?.length) pending.push(...nested); else layers += 1;
    }
    node.metadata.moduleLayerCount = layers;
  }
  const edges = list(semantic.edges, 'edges', MAX_EDGES).map(value => {
    const edge = record(value, 'edge');
    const source = identifier(edge.source_node_id, 'edge source');
    const target = identifier(edge.target_node_id, 'edge target');
    if (!byId.has(source) || !byId.has(target)) fail('An edge refers to a missing node.');
    return {
      edge_id: identifier(edge.edge_id, 'edge id'), source_node_id: source, target_node_id: target,
      source_port: identifier(edge.source_port, 'source port'), target_port: identifier(edge.target_port, 'target port'),
      metadata: captured ? { captureEdge: captureEdge(record(edge.metadata, 'edge metadata').captureEdge) } : residualMetadata(edge.metadata),
    };
  });
  if (new Set(edges.map(e => e.edge_id)).size !== edges.length) fail('Duplicate edge ids.');
  const collapsed = list(view.collapsed_modules, 'collapsed modules').map(v => identifier(v, 'collapsed module'));
  if (collapsed.some(id => byId.get(id)?.kind !== 'module') || new Set(collapsed).size !== collapsed.length) fail('Invalid collapsed modules.');
  const nodePositions = Object.fromEntries(nodes.map(node => {
    const position = record(positions[node.node_id], 'node position');
    return [node.node_id, { x: coordinate(position.x), y: coordinate(position.y) }];
  }));
  return {
    schemaVersion: 2,
    semantic: { graph_id: 'snapshot', name: text(semantic.name, 'model name'), revision: 0, nodes, edges, inputs: [], modules: [], operations: [], metadata: {} },
    view: { graph_id: 'snapshot', node_positions: nodePositions, collapsed_modules: collapsed, selection: [], layout: direction ? { direction } : {} },
  };
}

export function parseGraphFile(contents: string): GraphFile {
  if (contents.length > GRAPH_FILE_MAX_BYTES || new Blob([contents]).size > GRAPH_FILE_MAX_BYTES) fail('Graph files must be smaller than 96 MiB.');
  let value: unknown;
  try {
    value = JSON.parse(contents);
    // Inspect every own key before consuming any parsed content, including
    // ignored metadata and array entries, without a reviver call per scalar.
    const pending = [{ value, depth: 0 }];
    let useLegacyReviver = false;
    scan: while (pending.length) {
      const entry = pending.pop()!;
      if (entry.value === null || typeof entry.value !== 'object') continue;
      // This is a fallback threshold, not a new acceptance/depth limit.
      // Let the legacy traversal decide errors for deep or unsafe inputs.
      if (entry.depth > 128) {
        useLegacyReviver = true;
        break;
      }
      for (const key of Object.keys(entry.value)) {
        if (UNSAFE_KEYS.has(key)) {
          useLegacyReviver = true;
          break scan;
        }
        const child = (entry.value as Record<string, unknown>)[key];
        if (child !== null && typeof child === 'object') pending.push({ value: child, depth: entry.depth + 1 });
      }
    }
    if (useLegacyReviver) {
      pending.length = 0;
      value = undefined;
      value = JSON.parse(contents, (key, item) => {
        if (UNSAFE_KEYS.has(key)) fail('Unsafe object key in graph file.');
        return item;
      });
    }
  } catch (error) {
    if (error instanceof GraphFileError) throw error;
    return fail('This file is not valid JSON. Choose a .tensorviz.json graph file.');
  }
  const file = record(value, 'tensorViz graph file');
  if (file.format !== 'tensorviz.graph') fail('This is not a tensorViz graph file.');
  if (![1, 2, 3, 4].includes(Number(file.version)) || typeof file.version !== 'number') fail('Unsupported graph file version. Use a viewer that supports this version.');
  if (file.version !== 4 && file.capture !== undefined) fail('Canonical captures require graph file version 4.');
  if (file.version === 1 && file.collaboration !== undefined) fail('Annotations require graph file version 2.');
  if (file.version === 2 && file.collaboration !== undefined) {
    const collaboration = record(file.collaboration, 'collaboration');
    const walkthrough = record(collaboration.walkthrough, 'walkthrough');
    const newViewFields = Array.isArray(walkthrough.steps) && walkthrough.steps.some(step => {
      const view = step?.view;
      return view && typeof view === 'object' && ('selectedEdge' in view || 'viewport' in view);
    });
    if ('savedViews' in collaboration || 'openingViewId' in collaboration || newViewFields) fail('Saved views require graph file version 3.');
  }
  const exportedAt = text(file.exportedAt, 'export date');
  if (!Number.isFinite(Date.parse(exportedAt))) fail('Invalid export date.');
  if (!['static', 'traced', 'stale'].includes(String(file.status))) fail('Invalid graph status.');
  if (!Number.isSafeInteger(file.warningCount) || Number(file.warningCount) < 0) fail('Invalid warning count.');
  const capture = file.version === 4 ? requirePortableCapture(file.capture) : undefined;
  if (capture && file.status === 'static') fail('A captured execution cannot claim to be a static graph.');
  const document = portableDocument(file.document, !!capture);
  if (file.status === 'stale') document.semantic.nodes.forEach(node => { node.metadata.shapeEvidence = 'stale'; });
  if (capture) validateCaptureDocument(document, capture.capture);
  return { format: 'tensorviz.graph', version: capture ? 4 : 3, ...(capture ? { capture } : {}), exportedAt, status: file.status as GraphFileStatus, warningCount: Number(file.warningCount), document,
    snapshotId: file.version === 1 ? portableSnapshotId(document, exportedAt) : identifier(file.snapshotId, 'snapshot identity'),
    collaboration: validateCollaboration(file.collaboration) };
}

export function createGraphFile(graph: ModelGraph, options: GraphFileOptions): GraphFile {
  const capture = options.capture && requirePortableCapture(options.capture);
  const captured = graph.nodes.some(node => node.data.captureRef !== undefined);
  if (captured !== !!capture) fail('Export needs the retained canonical capture for this graph. Reconnect to VS Code and try again.');
  if (capture && new Set(graph.nodes.map(node => node.data.captureRef?.sourceId)).size !== 1) fail('This graph mixes source snapshots. Reopen it before exporting.');
  // Start with an allowlist before the generic adapter can copy arbitrary data.
  const safe: ModelGraph = {
    nodes: graph.nodes.map(node => {
      const kind = node.type === 'moduleNode' ? 'module' : node.type === 'inputNode' ? 'input' : node.type === 'outputNode' ? 'output' : 'layer';
      const op = node.data.layerType.name || node.data.layerType.id;
      // Live projections use null/empty sentinels for absent optional metadata.
      // Omit them at export; the portable importer still rejects malformed fields.
      const source: Record<string, unknown> = { ...node.data,
        ...tensorShapes(node.data),
        shapeEvidence: options.status === 'stale' ? 'stale' : captured ? node.data.shapeEvidence : tensorEvidence(node.data) };
      for (const key of ['parentId', 'repeatCount', 'repeatBlockLabel', 'repeatHidden', 'inputShape', 'outputShape']) {
        if (source[key] === null || source[key] === '' || (key === 'repeatCount' && source[key] === 0)) delete source[key];
      }
      if (capture && source.captureRef) source.captureRef = { ...captureReference(source.captureRef), sourceId: capture.capture.source.id };
      const clean = metadata(source, node.id, kind, op, captured);
      return {
        id: node.id, type: node.type, position: node.position,
        data: {
          ...clean,
          layerType: clean.layerType as typeof node.data.layerType,
          parameters: captured ? {} : parameters(node.data.parameters), expanded: node.data.expanded,
          ...(captured ? { inputPorts: node.data.inputPorts, outputPorts: node.data.outputPorts } : {}),
        },
      };
    }),
    edges: graph.edges.filter(e => !e.data?.synthetic).map((e, index) => ({
      id: captured ? e.id : `edge-${index + 1}`, source: e.source, target: e.target, sourceHandle: e.sourceHandle || e.data?.sourceHandle, targetHandle: e.targetHandle || e.data?.targetHandle,
      data: captured ? { captureEdge: captureEdge(e.data?.captureEdge) } : residualMetadata(e.data),
    })),
  };
  const document = modelGraphToDocumentV2(safe, { name: options.name, graphId: 'snapshot', revision: 0 });
  document.view.layout = options.direction ? { direction: options.direction } : {};
  document.semantic.nodes.forEach((node, index) => {
    node.metadata = metadata(safe.nodes[index].data, node.node_id, node.kind, node.op, captured);
  });
  const exportedAt = new Date().toISOString();
  return parseGraphFile(JSON.stringify({ format: 'tensorviz.graph', version: captured ? 4 : 1, exportedAt,
    ...(capture ? { capture, snapshotId: portableSnapshotId(document, exportedAt) } : {}),
    status: options.status, warningCount: options.warningCount ?? 0, document }));
}

export function graphFileName(name: string): string {
  const base = name.replace(/[^a-zA-Z0-9_-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 80) || 'model';
  return `${base}.tensorviz.json`;
}
