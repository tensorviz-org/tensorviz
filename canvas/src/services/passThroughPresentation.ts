import type { ModelEdge, ModelGraph, ModelNode } from '../types';
import type { GraphDocumentV2 } from '../types/graphDocument';

const knownTypes = new Set([
  'torch.nn.modules.linear.Identity',
  ...['Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d', 'AlphaDropout', 'FeatureAlphaDropout']
    .map(name => `torch.nn.modules.dropout.${name}`),
]);

/** Derived display facts, never capture records. A missing/uncertain call stays visible. */
export function classifyPassThroughCalls(document: GraphDocumentV2, nodes: ModelNode[], edges: ModelEdge[]): ModelNode[] {
  const capture = document.semantic.metadata?.capture as { status?: string; omissions?: unknown[] } | undefined;
  const complete = capture?.status === 'complete' && Array.isArray(capture.omissions) && capture.omissions.length === 0;
  const parents = new Set(nodes.map(node => node.data.parentId));
  const operations = new Map(document.semantic.nodes.map(node => [node.node_id, node.op]));
  const producers = new Map<string, { sourceNodeId: string; sourcePortId: string }>();
  for (const node of nodes) if (['input', 'value', 'operation'].includes(node.data.captureRef?.kind ?? '')) {
    for (const port of node.data.capturePorts ?? []) if (port.direction === 'out' && port.valueId) {
      producers.set(port.valueId, { sourceNodeId: node.id, sourcePortId: port.id });
    }
  }
  // Reuse the exact producer port already established by canonical value uses.
  for (const edge of edges) if (edge.data?.captureEdge) producers.set(edge.data.captureEdge.valueId,
    { sourceNodeId: edge.source, sourcePortId: edge.data.captureEdge.sourcePortId });
  return nodes.map(node => {
    const data = { ...node.data, capturePassThrough: undefined };
    const ports = data.capturePorts ?? [];
    const input = ports.find(port => port.direction === 'in');
    const output = ports.find(port => port.direction === 'out');
    const inputPath = input?.paths[0];
    const directTensor = input?.paths.length === 1 && inputPath?.length === 3 && inputPath[0] === 'arguments'
      && ['args', 'kwargs'].includes(String(inputPath[1])) && output?.paths.length === 1
      && output.paths[0].length === 1 && output.paths[0][0] === 'output';
    const producer = input?.valueId ? producers.get(input.valueId) : undefined;
    if (complete && data.captureRef?.kind === 'invocation' && data.parentId && !parents.has(node.id)
      && knownTypes.has(operations.get(node.id) ?? '') && data.capturedOperationCount === 0
      && data.shapeEvidence === 'measured' && data.parameters._shape_provenance === 'observed' && !data.error
      && directTensor && ports.length === 2 && input?.argumentKind === 'value' && output?.argumentKind === 'value'
      && input.valueId && input.valueId === output.valueId && producer) {
      return { ...node, data: { ...data, capturePassThrough: { valueId: input.valueId, inputPortId: input.id, ...producer } } };
    }
    return { ...node, data };
  });
}

export const passThroughHidden = (node: ModelNode) => Boolean(node.data.capturePassThrough && !node.data.passThroughShown);

export function showPassThroughCalls(nodes: ModelNode[], shown: boolean): ModelNode[] {
  return nodes.map(node => node.data.capturePassThrough
    ? { ...node, data: { ...node.data, passThroughShown: shown } } : node);
}

/** These annotations identify a recorded input; they do not make the call a producer. */
export function passThroughConnections(nodes: ModelNode[]): ModelGraph['edges'] {
  return nodes.flatMap(node => {
    const fact = node.data.capturePassThrough;
    return fact && node.data.passThroughShown ? [{ id: `passthrough:${node.id}`,
      source: fact.sourceNodeId, target: node.id, sourceHandle: fact.sourcePortId, targetHandle: fact.inputPortId,
      data: { synthetic: true, _relationship: 'unchanged input' } }] : [];
  });
}
