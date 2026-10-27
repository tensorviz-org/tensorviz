/** Navigation from semantic comparison records to validated portable projections. */
import type { GraphFile } from './graphFile';
import { captureReference, capturePorts } from './portableCaptureDocument';
import { canonical, edgeTarget, type GraphTarget } from './graphCollaboration';
import { correspondCaptures, type CaptureCorrespondence } from '../ide/captureCorrespondence.generated';
import type { CaptureRecordChange, CapturedValueUse } from '../ide/captureComparison.generated';

type NodeTarget = Extract<GraphTarget, { kind: 'node' }>;
export function capturedRecordTarget(file: GraphFile, kind: CaptureRecordChange['kind'], id: string): NodeTarget | undefined {
  const capture = file.capture?.capture;
  if (!capture) return undefined;
  const lookup = (kind: string, id: string) => file.document.semantic.nodes.find(node => {
    const ref = captureReference(node.metadata.captureRef); return ref.kind === kind && ref.semanticId === id;
  });
  let node = lookup(kind, id);
  if (kind === 'capture') node = capture.rootInvocationId ? lookup('invocation', capture.rootInvocationId) : lookup('input', capture.id);
  if (kind === 'module') {
    const calls = capture.invocations.filter(call => call.moduleId === id);
    if (calls.length === 1) node = lookup('invocation', calls[0].id);
  }
  if (kind === 'value' && !node) {
    const value = capture.values.find(value => value.id === id);
    node = value?.producer ? lookup('operation', value.producer.operationId)
      : file.document.semantic.nodes.find(node => node.kind === 'input' && capturePorts(node.metadata.capturePorts).some(port => port.valueId === id));
  }
  return node ? { kind: 'node', nodeId: node.node_id } : undefined;
}

export function capturedUseTarget(file: GraphFile, use: CapturedValueUse): GraphTarget | undefined {
  const node = file.document.semantic.nodes.find(node => {
    const ref = captureReference(node.metadata.captureRef); return ref.kind === use.entity.kind && ref.semanticId === use.entity.id;
  });
  if (!node) return undefined;
  const ports = capturePorts(node.metadata.capturePorts).filter(port => port.valueId === use.valueId
    && port.paths.some(path => canonical(path) === canonical(use.path)));
  const edges = ports.length === 1 ? file.document.semantic.edges.filter(edge => edge.target_node_id === node.node_id && edge.target_port === ports[0].id) : [];
  return edges.length === 1 ? edgeTarget(edges[0]) : undefined;
}

function nodePairs(before: GraphFile, after: GraphFile, correspondence: CaptureCorrespondence): Array<{ beforeId: string; afterId: string }> {
  const index = (file: GraphFile) => new Map(file.document.semantic.nodes.map(node => {
    const ref = captureReference(node.metadata.captureRef); return [canonical([ref.kind, ref.semanticId]), node.node_id];
  }));
  const a = index(before), b = index(after);
  const pairs = correspondence.pairs.flatMap(pair => {
    const beforeId = a.get(canonical([pair.kind, pair.beforeId])), afterId = b.get(canonical([pair.kind, pair.afterId]));
    return beforeId && afterId ? [{ beforeId, afterId }] : [];
  });
  for (const kind of ['input', 'output']) {
    const beforeId = a.get(canonical([kind, before.capture!.capture.id])), afterId = b.get(canonical([kind, after.capture!.capture.id]));
    if (beforeId && afterId) pairs.push({ beforeId, afterId });
  }
  return pairs;
}

/** Used only when switching comparison sides. An uncertain counterpart does not
 * override the destination's own saved view with IDs from the other capture. */
export function comparisonCounterpart(before: GraphFile, after: GraphFile, side: 'before' | 'after', target: GraphTarget): GraphTarget | undefined {
  const from = side === 'before' ? before : after, to = side === 'before' ? after : before;
  if (!before.capture && !after.capture) {
    if (target.kind === 'node') return to.document.semantic.nodes.some(node => node.node_id === target.nodeId) ? target : undefined;
    return to.document.semantic.edges.filter(edge => canonical(edgeTarget(edge)) === canonical(target)).length === 1 ? target : undefined;
  }
  if (!before.capture || !after.capture) return undefined;
  const pairs = nodePairs(before, after, correspondCaptures(before.capture.capture, after.capture.capture));
  const map = new Map(pairs.map(pair => side === 'before' ? [pair.beforeId, pair.afterId] : [pair.afterId, pair.beforeId]));
  if (target.kind === 'node') {
    const id = map.get(target.nodeId); return id ? { kind: 'node', nodeId: id } : undefined;
  }
  const original = from.document.semantic.edges.filter(edge => canonical(edgeTarget(edge)) === canonical(target));
  if (original.length !== 1) return undefined;
  const previousNode = from.document.semantic.nodes.find(node => node.node_id === target.target)!;
  const nextNode = to.document.semantic.nodes.find(node => node.node_id === map.get(target.target));
  const previousPort = capturePorts(previousNode.metadata.capturePorts).find(port => port.id === target.targetPort);
  if (!nextNode || !previousPort) return undefined;
  // The destination argument slot persists across a deliberate reconnection.
  // Source equality would incorrectly lose focus when the producer is changed.
  const ports = capturePorts(nextNode.metadata.capturePorts).filter(port => port.direction === 'in' && port.argumentKind === 'value'
    && canonical(port.paths) === canonical(previousPort.paths));
  const matches = ports.length === 1 ? to.document.semantic.edges.filter(edge => edge.target_node_id === nextNode.node_id && edge.target_port === ports[0].id) : [];
  return matches.length === 1 ? edgeTarget(matches[0]) : undefined;
}
