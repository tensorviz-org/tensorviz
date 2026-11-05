/**
 * Draw a projection onto the canvas without throwing away what is on it.
 *
 * The IDE re-projects the file on every settled keystroke, so this runs tens of
 * times a minute on a graph the user is looking at. `handleImport` — the web
 * product's code-import path — is the wrong shape for that: it clears the canvas
 * and re-runs auto-modularisation and layout, which is correct exactly once and
 * catastrophic on a loop (every card jumps, the camera fights the user, and a
 * position they dragged is gone the moment they type a space).
 *
 * So this MERGES by node id, which is what the D6 identity rules exist for:
 *
 *   * a node id the canvas already has keeps its position, and only its data is
 *     replaced — the file is the truth about structure, the canvas about layout;
 *   * a node id the canvas has never seen is laid out, and only it;
 *   * a node id the projection no longer has is removed.
 *
 * Edges come from the projection wholesale: they are entirely derived from the
 * source, and there is no user state on them to preserve.
 */
import { layoutTopLevel } from '../services/canvasLayout';
import type { LayoutDirection } from '../canvas/edgeAnchors';
import { ModelEdge, ModelGraph, ModelNode } from '../types';
import { initialOverviewNodes } from '../services/graphOverview';

export interface MergeOptions {
  /** A layout from `.tensorviz/views/` was merged into this projection, so the
   *  positions on it are the user's work rather than the tracer's emission
   *  order. Opening such a graph must draw it exactly as they left it. */
  viewRestored?: boolean;
  /** The incoming projection is authoritative about which frames are open.
   *
   *  Normally it is NOT: which module frames the user expanded is canvas state,
   *  like position and selection, and a re-projection on a keystroke must not
   *  collapse a frame they opened. That matters far more once a whole project is
   *  drawn as nested frames than it did for one class, where reopening a frame was
   *  one click — here a keystroke would close the entire project.
   *
   *  Set it when a STRUCTURAL push genuinely knows better: a trace that brought new
   *  children has to open the frame holding them, or the deeper layers stay hidden
   *  until the user collapses and reopens it by hand. */
  trustOpenness?: boolean;
  /** Canvas orientation for any cards this merge places. */
  direction?: LayoutDirection;
}

export interface ProjectionMerge {
  nodes: ModelNode[];
  edges: ModelEdge[];
  /** Ids the canvas did not have before this projection. */
  added: string[];
  /** Ids the canvas had and the projection does not. */
  removed: string[];
}

/** A position no layout would produce, so "was never placed" is decidable. */
function isUnplaced(node: ModelNode): boolean {
  const position = node.position;
  return !position || (position.x === 0 && position.y === 0);
}

/**
 * Merge `incoming` into `existing`.
 *
 * `existing` is the canvas as it stands; the returned nodes are in the
 * projection's order, because that order is the source's and a reader scanning
 * the graph should meet the layers in the order the file declares them.
 */
export function mergeProjection(
  incoming: ModelGraph,
  existing: readonly ModelNode[],
  options: MergeOptions = {},
): ProjectionMerge {
  // Opening the graph is the one moment a whole-graph layout is right, and the
  // only moment the incremental rules above never covered: on an empty canvas
  // every node is new, but the projection brings positions for all of them, so
  // nothing counted as unplaced and dagre never ran. What the user saw was
  // whatever order the tracer emitted nodes in — cards overlapping, edges
  // crossing — and the fix was to reach for the layout button every single time.
  //
  // `viewRestored` is what keeps this from being the destructive change the file
  // header warns about: a layout the user arranged is theirs, and is drawn as
  // they left it. Re-projection on a keystroke is unaffected, because by then the
  // canvas is not empty.
  if (existing.length === 0 && !options.viewRestored && incoming.nodes.length > 0) {
    const overview = initialOverviewNodes(incoming.nodes);
    const laid = layoutTopLevel(overview, incoming.edges, options.direction);
    const arranged = overview.map((node) =>
      laid.has(node.id) ? { ...node, position: laid.get(node.id)! } : node,
    );
    return {
      nodes: arranged,
      edges: incoming.edges,
      added: incoming.nodes.map((node) => node.id),
      removed: [],
    };
  }

  const placed = new Map<string, ModelNode>();
  existing.forEach((node) => placed.set(node.id, node));

  const incomingIds = new Set(incoming.nodes.map((node) => node.id));
  const added: string[] = [];
  const nodes = incoming.nodes.map((node) => {
    const previous = placed.get(node.id);
    if (!previous) {
      added.push(node.id);
      return node;
    }
    // Position from the canvas, everything else from the file. `selected` and
    // `hidden` are canvas state too: a projection must not deselect the card the
    // user is inspecting just because they typed in the editor.
    //
    // `data.expanded` is canvas state for the same reason — which frames are open
    // is something the user decided, not something the source says. Carry it
    // across unless a structural push is explicitly authoritative.
    const openness =
      !options.trustOpenness &&
      node.type === 'moduleNode' &&
      previous.data?.expanded !== undefined
        ? { data: { ...node.data, expanded: previous.data.expanded, passThroughShown: previous.data.passThroughShown } }
        : {};
    return {
      ...node,
      position: isUnplaced(previous) ? node.position : previous.position,
      ...openness,
      data: { ...node.data, ...(openness.data ?? {}),
        passThroughShown: previous.data.passThroughShown ?? node.data.passThroughShown,
        operationDetailsShown: previous.data.operationDetailsShown ?? node.data.operationDetailsShown,
        captureDetailLevel: previous.data.captureDetailLevel ?? node.data.captureDetailLevel },
      ...(previous.selected ? { selected: true } : {}),
      ...(previous.hidden !== undefined ? { hidden: previous.hidden } : {}),
    } as ModelNode;
  });

  const removed = existing
    .map((node) => node.id)
    .filter((id) => !incomingIds.has(id));

  // Only the new cards are placed, and only when the projection did not bring a
  // position for them (a sidecar layout does). layoutTopLevel returns positions
  // for the WHOLE graph — it has to, to place a new node relative to its
  // neighbours — and everything already on the canvas is then kept anyway.
  const newAndUnplaced = new Set(
    added.filter((id) => {
      const node = nodes.find((candidate) => candidate.id === id);
      return !!node && isUnplaced(node);
    }),
  );
  if (newAndUnplaced.size === 0) {
    return { nodes, edges: incoming.edges, added, removed };
  }

  const laid = layoutTopLevel(nodes, incoming.edges, options.direction);
  const settled = nodes.map((node) =>
    newAndUnplaced.has(node.id)
      ? { ...node, position: laid.get(node.id) ?? node.position }
      : node,
  );
  return { nodes: settled, edges: incoming.edges, added, removed };
}
