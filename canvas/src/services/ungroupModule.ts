/**
 * Ungroup: the inverse of ModelBuilder's `groupSelectionAsModule`.
 *
 * Pressing G collapsed a selection into a module and there was no way back —
 * `grep -rn "ungroup" src` returned one hit, and it was a comment in a test. Undo
 * dies on reload, so a group made yesterday was permanent, and the user guide said
 * as much ("there's no ungroup button yet").
 *
 * ── why boundary edges are the whole problem ─────────────────────────────────
 * Grouping REWRITES the endpoint of every edge that crosses the selection
 * boundary: `input -> conv` becomes `input -> module-17`, and the child's id is
 * not kept anywhere. So removing the wrapper naively leaves an edge pointing at a
 * node that no longer exists, i.e. a silently disconnected graph.
 *
 * The answer already exists in this codebase and is not invented here.
 * `buildInferablePayload` has always had to solve exactly this, because the
 * wrapper is not a runnable leaf and is dropped from every backend payload: it
 * rewires each boundary edge onto the module's entry/exit child, preferring the
 * wrapper's own `moduleEntryNodeId`/`moduleExitNodeId` and otherwise deriving them
 * topologically. Ungroup calls that same function (`moduleBoundaryChildren`), so
 * ungrouping a module yields precisely the flat graph the backend has been
 * validating for that module all along. That is the correctness argument: Ungroup
 * cannot introduce a topology the graph did not already have.
 *
 * The one case neither can recover is a module with two distinct entry children
 * and two edges arriving at the wrapper — the pairing was destroyed at group time.
 * Both edges land on the derived entry. No edge is dropped, and shape inference
 * already read the graph that way, but it is why grouping a fan-in is not a
 * perfectly reversible operation.
 */
import { ModelNode, ModelEdge } from '../types';
import { moduleBoundaryChildren } from './shapeInferencePayload';

export interface UngroupResult {
  nodes: ModelNode[];
  edges: ModelEdge[];
  /** The children handed back to the parent canvas, for the confirmation toast. */
  restored: number;
}

/**
 * Dissolve one module wrapper back into the flat graph.
 *
 * `moduleKey` matches the wrapper's node id OR its `moduleInstanceId`, so the
 * in-canvas module card (which knows only the instance id) and the More menu
 * (which knows the selected node) can both call it.
 *
 * Returns null when the key names no module, or when the module has no children —
 * there is nothing to restore and the caller should say so rather than delete a
 * wrapper and lose its edges.
 */
export function ungroupModuleGraph(
  nodes: ModelNode[],
  edges: ModelEdge[],
  moduleKey: string,
): UngroupResult | null {
  const wrapper = nodes.find(
    (n) => n.type === 'moduleNode'
      && (n.id === moduleKey || String(n.data?.moduleInstanceId || '') === moduleKey),
  );
  if (!wrapper) return null;
  const children = nodes.filter((n) => n.data?.parentId === wrapper.id);
  if (children.length === 0) return null;

  // Synthetic edges are display-only wrapper connectors the reconciler adds for
  // collapsed modules; they are rebuilt on the next reconcile, never restored.
  const realEdges = edges.filter((e) => !e.data?.synthetic);
  const boundary = moduleBoundaryChildren(nodes, realEdges).get(wrapper.id);
  if (!boundary) return null;

  const origin = wrapper.position || { x: 0, y: 0 };
  const nextNodes = nodes
    .filter((n) => n.id !== wrapper.id)
    .map((n) => {
      if (n.data?.parentId !== wrapper.id) return n;
      const local = n.position || { x: 0, y: 0 };
      return {
        ...n,
        // Child positions are relative to the wrapper; on the parent canvas they
        // are absolute. The caller relayouts anyway, but a stale relative
        // position would stack every restored child on the origin if it did not.
        position: { x: origin.x + local.x, y: origin.y + local.y },
        parentNode: undefined,
        extent: undefined,
        expandParent: undefined,
        hidden: false,
        data: {
          ...n.data,
          parentId: undefined,
          // A repeated module (xN) hides its non-canonical instances while
          // grouped. Once the wrapper is gone they are ordinary layers again.
          repeatHidden: undefined,
        },
      } as ModelNode;
    });

  const nextEdges = realEdges.map((e) => {
    if (e.source !== wrapper.id && e.target !== wrapper.id) return e;
    return {
      ...e,
      source: e.source === wrapper.id ? boundary.exit : e.source,
      target: e.target === wrapper.id ? boundary.entry : e.target,
      hidden: false,
    } as ModelEdge;
  });

  return { nodes: nextNodes, edges: nextEdges, restored: children.length };
}
