import { ModelNode } from '../types';

/** First-open policy only. Existing/saved views must bypass it. No new groups. */
export function initialOverviewNodes(nodes: ModelNode[]): ModelNode[] {
  if (nodes.filter(node => !node.hidden).length <= 24) return nodes;
  return nodes.map(node => node.type === 'moduleNode' && node.data.expanded && !node.data.captureRef
    ? { ...node, data: { ...node.data, expanded: false } } : node);
}

/** A module's header anchor in page space, including nested frame offsets. */
export function graphNodeOrigin(nodes: readonly ModelNode[], id: string): { x: number; y: number } | undefined {
  const byId = new Map(nodes.map(node => [node.id, node]));
  let node = byId.get(id);
  if (!node) return undefined;
  const point = { ...node.position };
  const seen = new Set([id]);
  while (node.data.parentId || node.parentNode) {
    const parent = node.data.parentId || node.parentNode!;
    if (seen.has(parent)) return undefined;
    seen.add(parent);
    node = byId.get(parent);
    if (!node) return undefined;
    point.x += node.position.x; point.y += node.position.y;
  }
  return point;
}
