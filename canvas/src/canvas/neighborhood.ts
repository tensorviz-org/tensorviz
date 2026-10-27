/** Pure indexed traversal. Layout, camera and source state never enter this model. */
export type TraceDirection = 'local' | 'upstream' | 'downstream';
export interface FocusEdge { id: string; source: string; target: string; role?: string; count?: number; originals?: string[]; relationship?: string }
export function indexConnections(edges: readonly FocusEdge[]) {
  const incoming = new Map<string, FocusEdge[]>();
  const outgoing = new Map<string, FocusEdge[]>();
  for (const edge of edges) {
    if (!incoming.has(edge.target)) incoming.set(edge.target, []);
    if (!outgoing.has(edge.source)) outgoing.set(edge.source, []);
    incoming.get(edge.target)!.push(edge);
    outgoing.get(edge.source)!.push(edge);
  }
  return { incoming, outgoing, edges: new Map(edges.map(edge => [edge.id, edge])) };
}
/** Which of `edgeIds` directly enter or leave the seeds. Only direct incidence
 *  counts: an edge between two selected seeds, or a multi-hop traced edge whose
 *  ends are both outside the selection, is deliberately neither. Arrows on a
 *  crossing canvas are easy to lose, so the renderer colours these two apart. */
export function seedEdgeDirections(
  index: ReturnType<typeof indexConnections>,
  seeds: readonly string[],
  edgeIds: Iterable<string>,
) {
  const seedSet = new Set(seeds);
  const incoming = new Set<string>();
  const outgoing = new Set<string>();
  for (const id of Array.from(edgeIds)) {
    const edge = index.edges.get(id);
    if (!edge) continue;
    const leavesSeed = seedSet.has(edge.source);
    const entersSeed = seedSet.has(edge.target);
    if (leavesSeed && !entersSeed) outgoing.add(id);
    else if (entersSeed && !leavesSeed) incoming.add(id);
  }
  return { incoming, outgoing };
}
export function traceConnections(index: ReturnType<typeof indexConnections>, seeds: readonly string[], direction: TraceDirection, limit = 2000) {
  const nodeIds = new Set(seeds);
  const edgeIds = new Set<string>();
  const queue = Array.from(nodeIds);
  let truncated = false;
  for (let cursor = 0; cursor < queue.length; cursor++) {
    const id = queue[cursor];
    const candidates = direction === 'upstream' ? index.incoming.get(id) ?? []
      : direction === 'downstream' ? index.outgoing.get(id) ?? []
        : [...(index.incoming.get(id) ?? []), ...(index.outgoing.get(id) ?? [])];
    for (const edge of candidates) {
      if (edgeIds.has(edge.id)) continue;
      if (edgeIds.size >= limit) { truncated = true; break; }
      edgeIds.add(edge.id);
      for (const other of [edge.source, edge.target]) if (!nodeIds.has(other)) {
        nodeIds.add(other);
        if (direction !== 'local') queue.push(other);
      }
    }
    if (truncated) break;
  }
  return { nodeIds, edgeIds, truncated };
}
