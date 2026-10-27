import type { ModelEdge, ModelNode, ModelNodeData } from '../types';
import { captureRoot, graphDetailLevel } from './graphDetailLevels';

// Tensor layout and slices from one verified source expression only.
// Names alone never establish a safe chain.
const layoutOps = new Set(['view.default', 'reshape.default', '_unsafe_view.default', 'permute.default',
  'transpose.int', 't.default', 'unsqueeze.default', 'squeeze.default', 'squeeze.dim', 'squeeze.dims',
  'flatten.using_ints', 'expand.default'].map(name => `aten.${name}`));

export function deriveOperationGroups(nodes: ModelNode[], edges: ModelEdge[]): ModelNode[] {
  const clean = nodes.map(node => ({ ...node, data: { ...node.data,
    operationGroup: undefined, operationGroupHiddenBy: undefined } }));
  if (!captureRoot(clean) || graphDetailLevel(clean) !== 'layers') return clean;
  const incoming = new Map<string, ModelEdge[]>(), outgoing = new Map<string, ModelEdge[]>();
  for (const edge of edges) if (!edge.data?.synthetic) {
    if (!incoming.has(edge.target)) incoming.set(edge.target, []);
    if (!outgoing.has(edge.source)) outgoing.set(edge.source, []);
    incoming.get(edge.target)!.push(edge); outgoing.get(edge.source)!.push(edge);
  }
  const candidates = new Map(clean.filter(node => {
    const data = node.data, ports = data.capturePorts ?? [];
    const incident = [...(incoming.get(node.id) ?? []), ...(outgoing.get(node.id) ?? [])];
    return data.captureRef?.kind === 'operation' && (layoutOps.has(data.layerType.name)
      || (data.layerType.name === 'aten.slice.Tensor' && !!data.captureSourceExpression && Number.isInteger(data.captureSliceAxis)))
      && data.shapeEvidence === 'measured' && data.parameters._shape_provenance === 'observed'
      && !data.error && !data.captureDiagnostic && !data.diffStatus
      && ports.filter(port => port.direction === 'in' && port.valueId).length === 1
      && ports.filter(port => port.direction === 'out' && port.valueId).length === 1
      && !ports.some(port => port.argumentKind === 'opaque' || port.argumentKind === 'symbolic'
        || port.paths.some(path => path.includes('effects')))
      && incident.every(edge => edge.data?.captureEdge && edge.data.captureEdge.kind !== 'effect');
  }).map(node => [node.id, node]));
  const next = new Map<string, string>(), previous = new Map<string, string>();
  for (const [id, node] of Array.from(candidates)) {
    const outputs = outgoing.get(id) ?? [];
    if (outputs.length !== 1) continue;
    const edge = outputs[0], target = candidates.get(edge.target);
    if (!target || target.id === id || (incoming.get(target.id)?.length ?? 0) !== 1
      || node.data.parentId !== target.data.parentId) continue;
    if (node.data.layerType.name === 'aten.slice.Tensor' || target.data.layerType.name === 'aten.slice.Tensor') {
      if (node.data.layerType.name !== target.data.layerType.name
        || node.data.captureSourceExpression !== target.data.captureSourceExpression
        || target.data.captureSliceAxis! <= node.data.captureSliceAxis!
        || node.data.captureRef?.captureId !== target.data.captureRef?.captureId
        || node.data.captureRef?.sourceId !== target.data.captureRef?.sourceId) continue;
    }
    const record = edge.data!.captureEdge!;
    if (!node.data.capturePorts?.some(p => p.id === record.sourcePortId && p.valueId === record.valueId)
      || !target.data.capturePorts?.some(p => p.id === record.targetPortId && p.valueId === record.valueId)) continue;
    next.set(id, target.id); previous.set(target.id, id);
  }
  for (const [id] of Array.from(candidates)) {
    if (previous.has(id)) continue;
    const chain: ModelNode[] = [];
    const seen = new Set<string>();
    let current: string | undefined = id;
    while (current && !seen.has(current)) {
      seen.add(current); chain.push(candidates.get(current)!); current = next.get(current);
    }
    if (chain.length < 2 || chain.some(node => node.data.operationDetailsShown)) continue;
    const first = chain[0], last = chain[chain.length - 1];
    last.data.operationGroup = { members: chain.map(node => node.id),
      ...(first.data.layerType.name === 'aten.slice.Tensor' ? { kind: 'slice' as const } : {}),
      labels: chain.map(node => node.data.layerType.name.replace(/^aten\./, '').split('.')[0]),
      inputShape: first.data.capturePorts?.find(p => p.direction === 'in' && p.valueId)?.shape ?? undefined,
      outputShape: last.data.capturePorts?.find(p => p.direction === 'out' && p.valueId)?.shape ?? undefined };
    for (const member of chain.slice(0, -1)) member.data.operationGroupHiddenBy = last.id;
  }
  return clean;
}

export function revealOperationGroup(nodes: ModelNode[], id: string): ModelNode[] {
  const target = nodes.find(node => node.id === id);
  const group = target?.data.operationGroup ?? nodes.find(node => node.id === target?.data.operationGroupHiddenBy)?.data.operationGroup;
  const members = new Set(group?.members ?? []);
  return nodes.map(node => members.has(node.id)
    ? { ...node, data: { ...node.data, operationDetailsShown: true } } : node);
}

/** Rendering-only text and boundary shapes; never write these into node data. */
export function operationGroupCard(data: ModelNodeData): ModelNodeData {
  const group = data.operationGroup;
  return !group ? data : { ...data,
    layerType: { ...data.layerType, displayName: group.kind === 'slice' ? 'Slice' : 'Tensor layout' },
    captureLabel: group.kind === 'slice' ? 'Slice' : `${group.members.length} operations · ${group.labels.join(' → ')}`,
    parameters: { _shape_provenance: 'observed', _in_shape: group.inputShape, _out_shape: group.outputShape },
    inputShape: group.inputShape, outputShape: group.outputShape, sourceLocation: undefined,
    sourceEditCapabilities: [],
  };
}
