/**
 * Kahn topological ordering over a node/edge list.
 *
 * Used by the validation sweep to walk nodes in dependency order (inputs first),
 * mirroring the order the shape engine resolves them. On a cycle (or nodes not
 * reachable by Kahn) the remaining node ids are appended in their input order so
 * the sweep still covers every node — it just isn't a perfect topo order.
 */
export interface TopoEdge {
  source: string;
  target: string;
}

export function topologicalOrder(nodeIds: string[], edges: TopoEdge[]): string[] {
  const present = new Set(nodeIds);
  const succ = new Map<string, string[]>();
  const indeg = new Map<string, number>();
  for (const id of nodeIds) {
    succ.set(id, []);
    indeg.set(id, 0);
  }
  for (const { source, target } of edges) {
    if (!present.has(source) || !present.has(target)) continue;
    succ.get(source)!.push(target);
    indeg.set(target, (indeg.get(target) || 0) + 1);
  }

  // Seed the queue in input order so ties (e.g. multiple inputs) stay stable.
  const queue = nodeIds.filter((id) => (indeg.get(id) || 0) === 0);
  const order: string[] = [];
  const seen = new Set<string>();
  while (queue.length) {
    const id = queue.shift()!;
    if (seen.has(id)) continue;
    seen.add(id);
    order.push(id);
    for (const t of succ.get(id) || []) {
      const d = (indeg.get(t) || 0) - 1;
      indeg.set(t, d);
      if (d <= 0 && !seen.has(t)) queue.push(t);
    }
  }

  // Cycle / unreached fallback: append leftovers in input order.
  if (order.length < nodeIds.length) {
    for (const id of nodeIds) {
      if (!seen.has(id)) order.push(id);
    }
  }
  return order;
}
