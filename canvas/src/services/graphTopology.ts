import type { ModelEdge, ModelNode } from '../types';

/** Reconnect surviving endpoints across a chain of deleted nodes. */
export function healEdgesAfterDeletion(
  nodes: ModelNode[],
  edges: ModelEdge[],
  deletedIds: ReadonlySet<string>,
): ModelEdge[] {
  const activeNodeIds = new Set(
    nodes.map((node) => node.id).filter((nodeId) => !deletedIds.has(nodeId)),
  );
  const healed = edges.filter(
    (edge) => activeNodeIds.has(edge.source) && activeNodeIds.has(edge.target),
  );
  const adjacency = new Map<string, string[]>();
  edges.forEach((edge) => {
    const targets = adjacency.get(edge.source) || [];
    targets.push(edge.target);
    adjacency.set(edge.source, targets);
  });
  const added = new Set(healed.map((edge) => `${edge.source}->${edge.target}`));

  activeNodeIds.forEach((source) => {
    const queue = (adjacency.get(source) || []).filter((target) => deletedIds.has(target));
    const visited = new Set(queue);
    while (queue.length) {
      const current = queue.shift()!;
      (adjacency.get(current) || []).forEach((target) => {
        if (activeNodeIds.has(target)) {
          const edgeId = `${source}->${target}`;
          if (!added.has(edgeId)) {
            added.add(edgeId);
            healed.push({ id: edgeId, source, target, data: {} } as ModelEdge);
          }
        } else if (deletedIds.has(target) && !visited.has(target)) {
          visited.add(target);
          queue.push(target);
        }
      });
    }
  });
  return healed;
}

/**
 * A node's input shape is its (first) upstream node's OUTPUT shape.
 *
 * Both inference paths need this and only one of them had it. `runVerification`
 * (the Validate button) derived it inline; `triggerShapeInference` (Refresh all
 * shapes) set `outputShape` only and left `inputShape` untouched — so after a
 * Refresh, every node's input shape was blank, the Shape Table's INPUT column
 * with it, and anything keyed on `inputShape` silently fell back. The Slice
 * range control is keyed on it, which is how the gap finally surfaced.
 *
 * Shared here so the two paths cannot drift apart again.
 */
export function deriveInputShapes(
  edges: Array<{ source: string; target: string }>,
  shapesByNodeId: Record<string, number[] | undefined> | undefined,
): Map<string, number[]> {
  const byNode = new Map<string, number[]>();
  if (!shapesByNodeId) return byNode;
  for (const edge of edges) {
    const upstream = shapesByNodeId[edge.source];
    // FIRST upstream wins: a merge node has several parents and one input shape
    // slot, and picking a later one would report a shape the node does not see
    // first. Matches the argument order codegen uses (edge insertion order).
    if (Array.isArray(upstream) && !byNode.has(edge.target)) {
      byNode.set(edge.target, upstream);
    }
  }
  return byNode;
}
