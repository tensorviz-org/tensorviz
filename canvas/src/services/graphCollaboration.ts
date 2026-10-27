import type { GraphFile } from './graphFile';
import type { ArchitectureNode, GraphDocumentV2 } from '../types/graphDocument';

export type GraphTarget = { kind: 'node'; nodeId: string } | {
  kind: 'edge'; source: string; target: string; sourcePort: string; targetPort: string;
};
export interface GraphAnchor {
  target: GraphTarget;
  snapshotId: string;
  exportedAt: string;
  modelName: string;
  context: string;
}
export type EdgeTarget = Extract<GraphTarget, { kind: 'edge' }>;
export interface SavedGraphView {
  camera: { x: number; y: number; z: number };
  nodePositions: Record<string, { x: number; y: number }>;
  expandedModuleIds: string[];
  selectedNodeIds: string[];
  selectedEdge?: EdgeTarget;
  viewport?: { width: number; height: number };
}
export interface WalkthroughStep {
  id: string; title: string; note: string; view: SavedGraphView; anchors: GraphAnchor[];
}
export type NamedGraphView = WalkthroughStep;
export interface GraphDiscussion {
  id: string;
  anchor: GraphAnchor;
  resolved: boolean;
  messages: Array<{ id: string; author: string; body: string; createdAt: string }>;
}
export interface GraphCollaboration {
  walkthrough: { title: string; steps: WalkthroughStep[] };
  threads: GraphDiscussion[];
  savedViews: NamedGraphView[];
  openingViewId: string | null;
}
export type ContextStatus = 'unchanged' | 'changed' | 'missing' | 'ambiguous';
type Snapshot = Pick<GraphFile, 'snapshotId' | 'document' | 'exportedAt' | 'capture'>;

const UNSAFE = new Set(['__proto__', 'constructor', 'prototype']);
const bad = (label: string): never => { throw new Error(`Invalid ${label} in graph collaboration.`); };
const object = (value: unknown, label: string): Record<string, unknown> => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return bad(label);
  return value as Record<string, unknown>;
};
const array = (value: unknown, label: string, max: number): unknown[] => {
  if (!Array.isArray(value) || value.length > max) return bad(`${label} (maximum ${max})`);
  return value;
};
function plain(value: unknown, label: string, max: number, empty = false): string {
  if (typeof value !== 'string' || value.length > max || (!empty && !value.trim()) || value.split('').some(c => c.charCodeAt(0) < 32 && !'\n\r\t'.includes(c))) return bad(label);
  return value.replace(/\r\n?/g, '\n');
}
function id(value: unknown): string {
  const result = plain(value, 'identity', 256);
  if (UNSAFE.has(result) || /^(?:\/|[a-z]:[\\/]|\\\\|file:|vscode:)/i.test(result) || /[\r\n\t]/.test(result)) return bad('identity');
  return result;
}
function unique<T extends { id: string }>(items: T[], label: string): T[] {
  if (new Set(items.map(item => item.id)).size !== items.length) return bad(`duplicate ${label}`);
  return items;
}
function ids(value: unknown, label: string, max = 5000): string[] {
  const result = array(value, label, max).map(id);
  if (new Set(result).size !== result.length) return bad(`duplicate ${label}`);
  return result;
}
function coord(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || Math.abs(value) > 1e7) return bad('view coordinate');
  return value;
}
function target(value: unknown): GraphTarget {
  const t = object(value, 'discussion target');
  if (t.kind === 'node') return { kind: 'node', nodeId: id(t.nodeId) };
  if (t.kind === 'edge') return { kind: 'edge', source: id(t.source), target: id(t.target), sourcePort: id(t.sourcePort), targetPort: id(t.targetPort) };
  return bad('discussion target');
}
function anchor(value: unknown): GraphAnchor {
  const a = object(value, 'original context');
  const context = plain(a.context, 'original context', 131072);
  try { JSON.parse(context, (key, item) => { if (UNSAFE.has(key)) bad('original context key'); return item; }); }
  catch { return bad('original context'); }
  const exportedAt = plain(a.exportedAt, 'original snapshot date', 80);
  if (!Number.isFinite(Date.parse(exportedAt))) return bad('original snapshot date');
  return { target: target(a.target), snapshotId: id(a.snapshotId), exportedAt, modelName: plain(a.modelName, 'original model name', 256), context };
}
function savedView(value: unknown): SavedGraphView {
  const v = object(value, 'walkthrough view');
  const camera = object(v.camera, 'view camera');
  const z = coord(camera.z);
  if (z < 0.001 || z > 100) return bad('view zoom');
  const positions = object(v.nodePositions, 'view positions');
  if (Object.keys(positions).length > 5000) return bad('view positions');
  const nodePositions = Object.fromEntries(Object.entries(positions).map(([key, value]) => {
    const p = object(value, 'view position');
    return [id(key), { x: coord(p.x), y: coord(p.y) }];
  }));
  const selectedNodeIds = ids(v.selectedNodeIds, 'selected layers', 100);
  const selectedEdge = v.selectedEdge === undefined ? undefined : target(v.selectedEdge);
  if (selectedEdge && (selectedEdge.kind !== 'edge' || selectedNodeIds.length)) return bad('saved edge selection');
  let viewport: SavedGraphView['viewport'];
  if (v.viewport !== undefined) {
    const size = object(v.viewport, 'capture viewport');
    const width = coord(size.width); const height = coord(size.height);
    if (width <= 0 || height <= 0 || width > 100000 || height > 100000) return bad('capture viewport');
    viewport = { width, height };
  }
  return { camera: { x: coord(camera.x), y: coord(camera.y), z }, nodePositions,
    expandedModuleIds: ids(v.expandedModuleIds, 'expanded modules'), selectedNodeIds,
    ...(selectedEdge?.kind === 'edge' ? { selectedEdge } : {}), ...(viewport ? { viewport } : {}) };

}
export function emptyCollaboration(): GraphCollaboration {
  return { walkthrough: { title: 'Architecture walkthrough', steps: [] }, threads: [], savedViews: [], openingViewId: null };
}
export function validateCollaboration(value: unknown): GraphCollaboration {
  if (value === undefined) return emptyCollaboration();
  const c = object(value, 'annotations');
  const walkthrough = object(c.walkthrough, 'walkthrough');
  const steps = unique(array(walkthrough.steps, 'walkthrough steps', 50).map(value => {
    const step = object(value, 'walkthrough step');
    return { id: id(step.id), title: plain(step.title, 'step title', 160), note: plain(step.note, 'step note', 4000, true),
      view: savedView(step.view), anchors: array(step.anchors, 'step targets', 100).map(anchor) };
  }), 'steps');
  const savedViews = unique(array(c.savedViews ?? [], 'saved views', 50).map(value => {
    const item = object(value, 'saved view');
    const view = savedView(item.view);
    const anchors = array(item.anchors, 'saved view targets', 100).map(anchor);
    const targets: GraphTarget[] = view.selectedEdge ? [view.selectedEdge] : view.selectedNodeIds.map(nodeId => ({ kind: 'node', nodeId }));
    if (!targets.length || anchors.length !== targets.length
      || targets.some(t => anchors.filter(a => canonical(a.target) === canonical(t)).length !== 1)) return bad('saved view targets');
    return { id: id(item.id), title: plain(item.title, 'saved view title', 160), note: plain(item.note ?? '', 'saved view note', 4000, true), view, anchors };
  }), 'saved views');
  const openingViewId = c.openingViewId === undefined || c.openingViewId === null ? null : id(c.openingViewId);
  if (openingViewId && !savedViews.some(view => view.id === openingViewId)) return bad('opening view reference');
  let messageCount = 0;
  const threads = unique(array(c.threads, 'discussion threads', 200).map(value => {
    const t = object(value, 'discussion thread');
    if (typeof t.resolved !== 'boolean') return bad('discussion resolution');
    const messages = unique(array(t.messages, 'discussion messages', 1000).map(value => {
      const m = object(value, 'message');
      const createdAt = plain(m.createdAt, 'message date', 80);
      if (!Number.isFinite(Date.parse(createdAt))) return bad('message date');
      return { id: id(m.id), author: plain(m.author, 'display name', 80), body: plain(m.body, 'message text', 4000), createdAt };
    }), 'messages');
    messageCount += messages.length;
    if (!messages.length || messageCount > 1000) return bad('message count (1–1000 total)');
    return { id: id(t.id), anchor: anchor(t.anchor), resolved: t.resolved, messages };
  }), 'threads');
  return { walkthrough: { title: plain(walkthrough.title, 'walkthrough title', 160), steps }, threads, savedViews, openingViewId };
}

/** Canonical JSON is compared directly for target context: no hash collision can
 * incorrectly mark a changed layer unchanged. Presentation fields are excluded. */
export function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value).filter(([, v]) => v !== undefined).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([k, v]) => `${JSON.stringify(k)}:${canonical(v)}`).join(',')}}`;
  return JSON.stringify(value);
}
function nodeFacts(n: ArchitectureNode): unknown {
  return { id: n.node_id, kind: n.kind, op: n.op, params: n.params, parent: n.metadata.parentId,
    inputShape: n.metadata.inputShape, outputShape: n.metadata.outputShape };
}
export function edgeTarget(edge: GraphDocumentV2['semantic']['edges'][number]): EdgeTarget {
  return { kind: 'edge', source: edge.source_node_id, target: edge.target_node_id, sourcePort: edge.source_port, targetPort: edge.target_port };
}
export function targetLabel(t: GraphTarget): string {
  return t.kind === 'node' ? t.nodeId : `${t.source}:${t.sourcePort} → ${t.target}:${t.targetPort}`;
}
function matchingContext(snapshot: Snapshot, t: GraphTarget): { status: ContextStatus; context?: string } {
  const { nodes, edges } = snapshot.document.semantic;
  const captured = snapshot.capture?.capture;
  const records = captured ? new Map([
    ...captured.operations.map(record => [`operation:${record.id}`, record] as const),
    ...captured.invocations.map(record => [`invocation:${record.id}`, record] as const),
    ...captured.values.map(record => [`value:${record.id}`, record] as const),
  ] as Array<[string, unknown]>) : null;
  const values = captured ? new Map(captured.values.map(value => [value.id, value])) : null;
  const facts = (node: ArchitectureNode): unknown => {
    const ref = node.metadata.captureRef as { kind: string; semanticId: string } | undefined;
    if (!captured || !ref) return nodeFacts(node);
    const ports = node.metadata.capturePorts as Array<{ valueId: string | null }>;
    return { node: nodeFacts(node), reference: ref, ports,
      record: ref.kind === 'input' ? captured.inputs : ref.kind === 'output' ? captured.output : records?.get(`${ref.kind}:${ref.semanticId}`),
      values: Array.from(new Set(ports.flatMap(port => port.valueId === null ? [] : [port.valueId]))).map(id => values?.get(id)),
      modules: ref.kind === 'invocation' ? captured.modules.filter(module => module.id === captured.invocations.find(call => call.id === ref.semanticId)?.moduleId) : [],
      status: captured.status, omissions: captured.coverage.omissions };
  };
  if (t.kind === 'node') {
    const found = nodes.filter(n => n.node_id === t.nodeId);
    if (!found.length) return { status: 'missing' };
    if (found.length !== 1) return { status: 'ambiguous' };
    const memberIds = new Set([t.nodeId]);
    if (found[0].kind === 'module') {
      const children = new Map<string, string[]>();
      nodes.forEach(n => { const parent = n.metadata.parentId; if (typeof parent === 'string') children.set(parent, [...(children.get(parent) ?? []), n.node_id]); });
      const pending = [t.nodeId];
      while (pending.length) for (const child of children.get(pending.pop()!) ?? []) {
        if (!memberIds.has(child)) { memberIds.add(child); pending.push(child); }
      }
    }
    const connections = edges.filter(e => memberIds.has(e.source_node_id) || memberIds.has(e.target_node_id)).map(e => canonical(edgeTarget(e))).sort();
    const contents = nodes.filter(n => n.node_id !== t.nodeId && memberIds.has(n.node_id)).map(facts).map(canonical).sort();
    return { status: 'unchanged', context: canonical({ node: facts(found[0]), contents, connections }) };
  }
  const found = edges.filter(e => canonical(edgeTarget(e)) === canonical(t));
  if (!found.length) return { status: 'missing' };
  if (found.length !== 1) return { status: 'ambiguous' };
  return { status: 'unchanged', context: canonical({ edge: t, endpoints: nodes.filter(n => n.node_id === t.source || n.node_id === t.target).map(facts).map(canonical).sort() }) };
}
export function captureAnchor(snapshot: Snapshot, t: GraphTarget): GraphAnchor {
  const result = matchingContext(snapshot, t);
  if (!result.context) throw new Error(`This target is ${result.status}; choose a unique layer or connection.`);
  return anchor({ target: t, snapshotId: snapshot.snapshotId, exportedAt: snapshot.exportedAt, modelName: snapshot.document.semantic.name, context: result.context });
}
export function anchorStatus(snapshot: Snapshot, a: GraphAnchor): ContextStatus {
  const result = matchingContext(snapshot, a.target);
  if (result.status !== 'unchanged') return result.status;
  return snapshot.document.semantic.name === a.modelName && result.context === a.context ? 'unchanged' : 'changed';
}
export function stepStatus(snapshot: Snapshot, step: WalkthroughStep): ContextStatus {
  const states = step.anchors.map(a => anchorStatus(snapshot, a));
  if (states.includes('missing') || step.view.selectedNodeIds.some(id => !snapshot.document.semantic.nodes.some(n => n.node_id === id))) return 'missing';
  if (states.includes('ambiguous')) return 'ambiguous';
  return states.includes('changed') ? 'changed' : 'unchanged';
}
export function hasAnnotations(file: GraphFile): boolean {
  return file.collaboration.walkthrough.steps.length > 0 || file.collaboration.threads.length > 0 || file.collaboration.savedViews.length > 0;
}
export function updateArchitecture(current: GraphFile, incoming: GraphFile): GraphFile {
  if (hasAnnotations(incoming)) throw new Error('The replacement file contains notes. Open it separately, or export an architecture-only graph. Your current graph and notes are unchanged.');
  return { ...incoming, collaboration: current.collaboration };
}

/** Stable portable label for a snapshot, including legacy files. This is not an
 * authenticity signature; original target context is checked by exact equality. */
export function portableSnapshotId(document: GraphDocumentV2, exportedAt: string): string {
  const value = canonical({ exportedAt, name: document.semantic.name, nodes: document.semantic.nodes.map(nodeFacts).map(canonical).sort(), edges: document.semantic.edges.map(e => canonical(edgeTarget(e))).sort() });
  let a = 2166136261; let b = 3335557771;
  for (let i = 0; i < value.length; i += 1) { const c = value.charCodeAt(i); a = Math.imul(a ^ c, 16777619); b = Math.imul(b ^ c, 2246822519); }
  return `snapshot-${(a >>> 0).toString(16)}${(b >>> 0).toString(16)}`;
}
export function annotationId(): string {
  return typeof globalThis.crypto?.randomUUID === 'function' ? globalThis.crypto.randomUUID() : `note-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}
