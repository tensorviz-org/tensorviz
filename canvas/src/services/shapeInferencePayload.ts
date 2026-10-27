/**
 * Build the { nodes, edges } payload sent to /api/validate_shapes for shape
 * inference / canvas validation.
 *
 * Edge-free module frames are not runnable leaves, so they are dropped from the
 * payload. A wired module is replaced only when real children exist; otherwise
 * its opaque boundary and wires must remain. BOUNDARY edges reference the wrapper id
 * (`a -> module` and `module -> b`). Dropping the wrapper without rewiring
 * those edges orphans the module's internal entry/exit child and everything
 * downstream of the module — which the backend validator then flags as
 * "Node has no input connections" / "Node is unreachable from input".
 *
 * So we first rewire each boundary edge onto the module's entry/exit child
 * (mirrors the backend verify path's _flatten_for_torch / _reconstruct_hierarchy
 * in verification_service.py), keeping the flattened chain connected. The
 * module's internal edges already connect its children, so a multi-layer module
 * stays wired entry -> ... -> exit.
 */
import { ModelNode, ModelEdge } from '../types';

const isModuleWrapper = (n: ModelNode): boolean =>
  n.type === 'moduleNode' || !!n.data?.isGroup;

/**
 * For every module wrapper, the child a boundary edge really touches: `entry`
 * for an edge arriving at the wrapper, `exit` for one leaving it.
 *
 * Exported because Ungroup needs the same answer. Ungroup removes the wrapper
 * for real, so its boundary edges must land on the same children this payload
 * has always rewired them to — otherwise ungrouping a module would change the
 * graph the backend validates, which is the one thing it must not do.
 */
export function moduleBoundaryChildren(
  nodes: ModelNode[],
  edges: ModelEdge[],
): Map<string, { entry: string; exit: string }> {
  const boundary = new Map<string, { entry: string; exit: string }>();
  for (const node of nodes) {
    if (!isModuleWrapper(node)) continue;
    const children = nodes.filter((n) => n.data?.parentId === node.id);
    if (children.length === 0) continue;
    const childIds = new Set(children.map((c) => c.id));
    const hasInternalIncoming = new Set<string>();
    const hasInternalOutgoing = new Set<string>();
    for (const e of edges) {
      if (childIds.has(e.source) && childIds.has(e.target)) {
        hasInternalOutgoing.add(e.source);
        hasInternalIncoming.add(e.target);
      }
    }
    const entry =
      node.data?.moduleEntryNodeId ||
      children.find((c) => !hasInternalIncoming.has(c.id))?.id ||
      children[0].id;
    const exit =
      node.data?.moduleExitNodeId ||
      children.find((c) => !hasInternalOutgoing.has(c.id))?.id ||
      children[children.length - 1].id;
    boundary.set(node.id, { entry, exit });
  }
  return boundary;
}

export function buildInferablePayload(
  nodes: ModelNode[],
  allEdges: ModelEdge[],
): { inferableNodes: ModelNode[]; inferableEdges: ModelEdge[] } {
  // Synthetic edges are display-only wrapper connectors added by the reconciler
  // for collapsed modules. They reference module-node ids and must never reach
  // the backend — the real child-targeted edges below carry the true topology.
  const edges = allEdges.filter((e) => !e.data?.synthetic);
  // Map each module wrapper id -> its entry/exit child id. Prefer the explicit
  // entry/exit metadata stamped on the wrapper. When that's absent (some
  // backends emit a module with input_node_id/output_node_id == null), derive
  // it topologically from the module's internal edges: the entry is the child
  // with no incoming edge from a sibling, the exit is the child with no
  // outgoing edge to a sibling. This is order-independent (layout may reorder
  // the flat node list), falling back to first/last child only if ambiguous.
  const boundary = moduleBoundaryChildren(nodes, edges);

  // Only drop edge-free frames or wrappers whose real children replace them.
  // An opaque boundary with no captured children still owns its real wires.
  const incident = new Set(edges.flatMap(edge => [edge.source, edge.target]));
  const inferableNodes = nodes.filter((n) =>
    !isModuleWrapper(n) || (!boundary.has(n.id) && incident.has(n.id)),
  );
  const inferableNodeIds = new Set(inferableNodes.map((n) => n.id));

  const inferableEdges = edges
    .map((edge) => {
      const srcModule = boundary.get(edge.source);
      const tgtModule = boundary.get(edge.target);
      if (!srcModule && !tgtModule) return edge;
      return {
        ...edge,
        source: srcModule ? srcModule.exit : edge.source,
        target: tgtModule ? tgtModule.entry : edge.target,
      };
    })
    .filter(
      (edge) => inferableNodeIds.has(edge.source) && inferableNodeIds.has(edge.target),
    );

  return { inferableNodes, inferableEdges };
}
