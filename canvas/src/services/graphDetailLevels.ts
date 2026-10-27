import type { ModelNode } from '../types';

export type GraphDetailLevel = 'layers' | 'operations';

export const captureRoot = (nodes: readonly ModelNode[]) => nodes.find(node =>
  node.data.captureRef?.kind === 'invocation' && !node.data.parentId);

export const graphDetailLevel = (nodes: readonly ModelNode[]): GraphDetailLevel =>
  captureRoot(nodes)?.data.captureDetailLevel === 'operations' ? 'operations' : 'layers';

/** Expand composition, stopping at calls whose contents are only operations.
 * Ownership, captured records and argument connections remain unchanged. */
export function expandedDetailNodes(nodes: ModelNode[], level: GraphDetailLevel): ModelNode[] {
  const root = captureRoot(nodes);
  if (!root) return nodes;
  const containers = new Set(nodes.filter(node => node.data.captureRef?.kind === 'invocation')
    .map(node => node.data.parentId));
  const byId = new Map(nodes.map(node => [node.id, node]));
  const failedAncestors = new Set<string>();
  for (const node of nodes) if (node.data.error || node.data.captureDiagnostic) {
    let parent = node.data.parentId;
    const seen = new Set<string>();
    while (parent && !seen.has(parent)) {
      seen.add(parent); failedAncestors.add(parent);
      parent = byId.get(parent)?.data.parentId;
    }
  }
  return nodes.map(node => ({ ...node, data: { ...node.data,
    ...(node.id === root.id ? { captureDetailLevel: level } : {}),
    ...(node.type === 'moduleNode' ? { expanded: level === 'operations' || node.id === root.id
      || containers.has(node.id) || failedAncestors.has(node.id) } : {}),
  } }));
}
