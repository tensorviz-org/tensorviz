import type { ModelEdge, ModelGraph, ModelNode } from '../types';
import type { GraphDocumentV2 } from '../types/graphDocument';
import { graphDocumentV2ToModelGraph } from '../services/graphDocumentAdapter';
import { sourceAssignmentsOf, type SourceAssignmentFact } from './sourceAssignmentCards';

/** A retained run's node as context: its evidence is historical, never current. */
export const previousData = (node: ModelNode): ModelNode['data'] => ({ ...node.data,
  debugEvidence: 'previous', shapeEvidence: 'stale', debugChanges: undefined,
  error: undefined, captureDiagnostic: undefined, sourceLocation: undefined,
  sourceEditCapabilities: [], sourceEditLimitation: 'previous_run',
  moduleStatus: undefined, moduleDiagnosticStatus: undefined, moduleDiagnostics: [], moduleDiagnosticCount: 0,
  resolutionDetail: undefined, resolutionSeverity: undefined,
});

export interface AssignmentContext {
  /** Capture and source structure this context was proven against. */
  key: string;
  graph: ModelGraph;
  /** Assignment span id → every captured call of its registered module, in call order. */
  calls: Map<string, string[]>;
}

type Correspondence = { nodeId: string; captureId: string; basis: string; invocationIds: string[] };
type ReviewClass = { structureDigest?: string; dependencyScope?: string; dependencyDigests?: Record<string, string>;
  assignments?: Record<string, SourceAssignmentFact> };
const reviewClassOf = (document: unknown) => (document as GraphDocumentV2 | undefined)?.semantic?.metadata?.reviewClass as ReviewClass | undefined;

// Only these containers are known to index by position or key; a custom
// __getitem__ can return anything, so it never proves a registered path.
const SEQUENCES = new Set(['torch.nn.modules.container.Sequential', 'torch.nn.modules.container.ModuleList']);
const MAPPINGS = new Set(['torch.nn.modules.container.ModuleDict']);

/**
 * Annotate literal nested assignments (`self.blocks[0].relu = nn.GELU()`) on the
 * previous calls at that registered path, inside the hierarchy that actually ran.
 *
 * Fails closed unless the source outside unconditional single-target literal
 * assignments is structurally unchanged and the known source candidates are unchanged,
 * and every assignment that differs from the capture maps through registered
 * paths to calls of one uniquely registered module. These source-only cards lose the old
 * contents and measurements; their recorded boundary keeps the existing flow.
 * Partial dependency coverage and unregistered Python references mean affected
 * calls are unverified. These are source annotations, never a new execution.
 */
export function assignmentContext(capture: unknown, source: unknown, before?: unknown): AssignmentContext | null {
  const document = capture as GraphDocumentV2 | undefined;
  const metadata = document?.semantic?.metadata;
  const run = metadata?.capture as { id?: string; status?: string } | undefined;
  const captured = reviewClassOf(capture), current = reviewClassOf(source);
  if (!run?.id || run.status !== 'complete' || !captured?.structureDigest || captured.structureDigest !== current?.structureDigest) return null;
  if (!['complete', 'partial'].includes(captured.dependencyScope ?? '') || !captured.dependencyDigests || !current?.dependencyDigests
    || Object.entries(captured.dependencyDigests).some(([file, digest]) => current.dependencyDigests![file] !== digest)) return null;
  const graph = graphDocumentV2ToModelGraph(document!);
  const calls = new Map(graph.nodes.filter(node => node.data.captureRef?.kind === 'invocation')
    .map(node => [node.data.captureRef!.semanticId, node]));
  const correspondence = (metadata!.sourceCorrespondence ?? []) as Correspondence[];
  const containers = metadata!.registeredContainers as Record<string, { definition: string; keys: string[] }> | undefined;
  const callsAt = (path: string): ModelNode[] | undefined => {
    const facts = correspondence.filter(fact => fact.nodeId === path);
    const fact = facts.length === 1 ? facts[0] : undefined;
    if (!fact || fact.captureId !== run.id || fact.basis !== 'registered_path' || !fact.invocationIds?.length) return undefined;
    const nodes = fact.invocationIds.map(id => calls.get(id));
    return nodes.every(node => node) ? nodes as ModelNode[] : undefined;
  };
  const resolve = (path: NonNullable<SourceAssignmentFact['path']>): ModelNode[] | undefined => {
    let registered = '';
    for (const segment of path) {
      if (segment.kind === 'index') {
        const receiver = containers?.[registered];
        if (!receiver) return undefined;
        // Numeric indices mean position, not a registration name. Only a complete
        // ordinary numeric sequence proves these are the same (no names or holes).
        if (typeof segment.key === 'number') {
          if (!Number.isInteger(segment.key) || segment.key < 0 || !SEQUENCES.has(receiver.definition)
            || !receiver.keys.every((key, index) => key === String(index)) || segment.key >= receiver.keys.length) return undefined;
        } else if (!MAPPINGS.has(receiver.definition) || !receiver.keys.includes(segment.key)) return undefined;
      } else if (segment.kind !== 'attribute' || typeof segment.key !== 'string') return undefined;
      const key = String(segment.key);
      if (!key || key.includes('.')) return undefined;
      registered = registered ? `${registered}.${key}` : key;
    }
    return registered ? callsAt(registered) : undefined;
  };
  const old = captured.assignments ?? {}, now = current.assignments ?? {};
  const previous = before === undefined ? undefined : sourceAssignmentsOf(before);
  const replaced = new Map<string, { node: ModelNode; key: string; fact: SourceAssignmentFact }>();
  const assigned = new Map<string, string[]>();
  for (const key of Array.from(new Set([...Object.keys(old), ...Object.keys(now)]))) {
    if (old[key]?.digest === now[key]?.digest) continue;
    // A removed assignment leaves the block's own default, which this capture never observed.
    const fact = now[key];
    const nodes = fact && !fact.ambiguous && !fact.conditional && fact.path?.length ? resolve(fact.path) : undefined;
    if (!nodes || nodes.some(node => replaced.has(node.id) || node.data.capturedOperationCount === 0
      || node.data.capturePorts?.some(port => port.direction === 'out' && port.valueId
        && node.data.capturePorts?.some(input => input.direction === 'in' && input.valueId === port.valueId)))) return null;
    for (const node of nodes) replaced.set(node.id, { node, key, fact });
    assigned.set(key, nodes.map(node => node.id));
  }
  if (!replaced.size) return null;

  // The replaced module's recorded contents describe the old class: drop them.
  const children = new Map<string, ModelNode[]>();
  for (const node of graph.nodes) if (node.data.parentId) children.set(node.data.parentId, [...children.get(node.data.parentId) ?? [], node]);
  const owner = new Map<string, string>();
  for (const id of Array.from(replaced.keys())) {
    const stack = [...children.get(id) ?? []];
    while (stack.length) {
      const child = stack.pop()!;
      if (replaced.has(child.id) || owner.has(child.id)) return null;
      owner.set(child.id, id);
      stack.push(...children.get(child.id) ?? []);
    }
  }
  // Each crossing value lands on the call's own recorded port for that value.
  const portOf = (call: string, direction: 'in' | 'out', valueId?: string | null) => {
    const ports = (replaced.get(call)!.node.data.capturePorts ?? []).filter(port => port.direction === direction && !!valueId && port.valueId === valueId);
    return ports.length === 1 ? ports[0].id : undefined;
  };
  const edges: ModelEdge[] = [];
  for (const edge of graph.edges) {
    const from = owner.get(edge.source), to = owner.get(edge.target);
    if (!from && !to) { edges.push(edge); continue; }
    if (from && from === to) continue;
    const captureEdge = edge.data?.captureEdge;
    const sourcePort = from && portOf(from, 'out', captureEdge?.valueId), targetPort = to && portOf(to, 'in', captureEdge?.valueId);
    if (!captureEdge || (from && !sourcePort) || (to && !targetPort)) return null;
    edges.push({ ...edge, ...(from ? { source: from, sourceHandle: sourcePort } : {}), ...(to ? { target: to, targetHandle: targetPort } : {}),
      data: { ...edge.data, captureEdge: { ...captureEdge,
        ...(from ? { sourcePortId: sourcePort!, displaySourcePortId: sourcePort! } : {}),
        ...(to ? { targetPortId: targetPort!, displayTargetPortId: targetPort! } : {}) } } });
  }

  const nodes = graph.nodes.filter(node => !owner.has(node.id)).map(node => {
    const replacement = replaced.get(node.id);
    if (!replacement) return { ...node, data: previousData(node) };
    const { key, fact } = replacement;
    // The label is the source callable as written, not a class this run observed.
    const earlier = previous?.[key];
    const label = previous && earlier ? earlier.digest === fact.digest ? undefined : earlier.title : node.data.layerType.displayName;
    return { ...node, data: { ...previousData(node), debugEvidence: undefined, shapeEvidence: 'unknown' as const,
      layerType: { ...node.data.layerType, name: fact.title, displayName: fact.title, description: `Set in source by ${fact.target}; not run` },
      moduleTemplateId: fact.title, parameters: {}, capturedOperationCount: undefined, capturePassThrough: undefined,
      capturePorts: node.data.capturePorts?.map(port => ({ ...port, shape: null, dtype: null, device: null })),
      captureDisplayPorts: undefined, inputShape: undefined, outputShape: undefined,
      outputTensor: undefined, parameterCount: undefined,
      moduleMembers: [], moduleLayerCount: 0, expandable: false, expanded: false,
      sourceLocation: { file: fact.span.file, line: fact.span.start[0] },
      sourceReplacement: { key, target: fact.target, capturedType: node.data.layerType.displayName,
        coverage: captured.dependencyScope as 'complete' | 'partial', ...(label !== undefined ? { before: label } : {}) },
    } };
  });
  // Several old internal consumers can collapse onto the same recorded boundary.
  const seen = new Set<string>();
  const uniqueEdges = edges.filter(edge => {
    const key = JSON.stringify([edge.source, edge.sourceHandle, edge.target, edge.targetHandle]);
    if (seen.has(key)) return false;
    seen.add(key); return true;
  });
  return { key: `${run.id}:${captured.structureDigest}`, graph: { ...graph, nodes, edges: uniqueEdges }, calls: assigned };
}
