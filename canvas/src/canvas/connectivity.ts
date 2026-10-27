/**
 * Structural rules for a node connection, shared by the drag-to-connect gesture
 * and the Shift+C keyboard path. Deliberately free of tldraw imports so the
 * rules stay unit-testable without an editor.
 *
 * STRUCTURAL only — direction, self-loops, duplicates, cycles. Tensor-shape
 * compatibility is NOT checked here: /api/validate_shapes runs a real torch
 * forward pass and caches on a sha256 of the whole graph, far too slow to run
 * per pointermove. A connection commits optimistically and Refresh shapes /
 * Verify reports any mismatch afterwards.
 */

export type ConnectionSide = 'source' | 'target';

export type ConnectableKind = 'input' | 'output' | 'layer' | 'module';

export interface ConnectableNode {
  id: string;
  kind: ConnectableKind;
  /** Collapsed module frames only; both are optional on ModelNodeData. */
  moduleEntryNodeId?: string;
  moduleExitNodeId?: string;
}

/** The two fields the rules need off a ModelEdge. */
export interface GraphEdgeLike {
  source: string;
  target: string;
}

export type ConnectionRejection =
  | 'missing-port'
  | 'unresolved-module'
  | 'self-loop'
  | 'duplicate'
  | 'cycle';

export type ConnectionResult =
  | { ok: true; source: string; target: string }
  | { ok: false; reason: ConnectionRejection };

/**
 * The node an outgoing edge actually leaves from, or null when this node has no
 * source port at all. A collapsed module frame must itself stay edge-free — the
 * reconciler owns the synthetic frame edges — so an edge drawn from a frame
 * resolves to the module's EXIT child instead.
 */
export function sourceEndpoint(node: ConnectableNode): string | null {
  if (node.kind === 'output') return null;
  if (node.kind !== 'module') return node.id;
  // ModuleShapeUtil guards its route pill the same way: a frame that cannot name
  // both ends cannot be wired, so it gets no port rather than a broken one.
  return node.moduleEntryNodeId && node.moduleExitNodeId ? node.moduleExitNodeId : null;
}

/** The node an incoming edge actually lands on, or null when there is no target port. */
export function targetEndpoint(node: ConnectableNode): string | null {
  if (node.kind === 'input') return null;
  if (node.kind !== 'module') return node.id;
  return node.moduleEntryNodeId && node.moduleExitNodeId ? node.moduleEntryNodeId : null;
}

/**
 * Which ports a card draws. Direction is enforced structurally: a port that
 * could never produce a legal edge is simply never rendered, so an Input can
 * only be dragged FROM and an Output only dragged TO.
 */
export function hasPort(node: ConnectableNode, side: ConnectionSide): boolean {
  return (side === 'source' ? sourceEndpoint(node) : targetEndpoint(node)) !== null;
}

/** Every node that already reaches `id` (reverse BFS). */
export function ancestorsOf(edges: readonly GraphEdgeLike[], id: string): Set<string> {
  return traverse(edges, id, 'up');
}

/** Every node `id` already reaches (forward BFS). */
export function descendantsOf(edges: readonly GraphEdgeLike[], id: string): Set<string> {
  return traverse(edges, id, 'down');
}

function traverse(edges: readonly GraphEdgeLike[], id: string, direction: 'up' | 'down'): Set<string> {
  const seen = new Set<string>();
  const queue = [id];
  while (queue.length > 0) {
    const current = queue.shift() as string;
    for (let i = 0; i < edges.length; i += 1) {
      const edge = edges[i];
      const from = direction === 'up' ? edge.target : edge.source;
      const to = direction === 'up' ? edge.source : edge.target;
      if (from !== current || seen.has(to)) continue;
      seen.add(to);
      queue.push(to);
    }
  }
  return seen;
}

export interface ConnectionAnchor {
  node: ConnectableNode;
  /** The port the gesture started from. */
  side: ConnectionSide;
  /** See anchorReach — computed once at pointerdown, not per pointermove. */
  reach: ReadonlySet<string>;
}

/**
 * The cycle guard for a gesture: the anchor's ancestors when the drag leaves a
 * source port, its descendants when it leaves a target port. Either way, a
 * candidate inside this set would close a loop. One BFS per gesture — running it
 * per pointermove would walk the whole graph on every mouse sample.
 */
export function anchorReach(
  edges: readonly GraphEdgeLike[],
  node: ConnectableNode,
  side: ConnectionSide,
): Set<string> {
  const endpoint = side === 'source' ? sourceEndpoint(node) : targetEndpoint(node);
  if (!endpoint) return new Set<string>();
  return side === 'source' ? ancestorsOf(edges, endpoint) : descendantsOf(edges, endpoint);
}

const rejectionFor = (node: ConnectableNode): ConnectionRejection =>
  node.kind === 'module' ? 'unresolved-module' : 'missing-port';

/**
 * Can the gesture that started at `anchor` land on `candidate`? Returns the
 * RESOLVED endpoints on success, which is what the committing side needs — for a
 * module frame those are its entry/exit children, not the frame.
 */
export function validateConnection(
  anchor: ConnectionAnchor,
  candidate: ConnectableNode,
  edges: readonly GraphEdgeLike[],
): ConnectionResult {
  const anchorEnd = anchor.side === 'source'
    ? sourceEndpoint(anchor.node)
    : targetEndpoint(anchor.node);
  if (!anchorEnd) return { ok: false, reason: rejectionFor(anchor.node) };

  const candidateEnd = anchor.side === 'source'
    ? targetEndpoint(candidate)
    : sourceEndpoint(candidate);
  if (!candidateEnd) return { ok: false, reason: rejectionFor(candidate) };

  const source = anchor.side === 'source' ? anchorEnd : candidateEnd;
  const target = anchor.side === 'source' ? candidateEnd : anchorEnd;

  if (source === target) return { ok: false, reason: 'self-loop' };
  if (edges.some((edge) => edge.source === source && edge.target === target)) {
    return { ok: false, reason: 'duplicate' };
  }
  // A->B when B already reaches A is the reverse duplicate as well as the
  // shortest cycle, so both are reported here.
  if (anchor.reach.has(candidateEnd)) return { ok: false, reason: 'cycle' };

  return { ok: true, source, target };
}

/** A canvas node's React Flow type as a connectable kind. */
export function kindFromNodeType(nodeType: string | undefined): ConnectableKind {
  if (nodeType === 'inputNode') return 'input';
  if (nodeType === 'outputNode') return 'output';
  if (nodeType === 'moduleNode') return 'module';
  return 'layer';
}

/** Plain-language reason for a refused connection, for the Shift+C toast. */
export function connectionRejectionMessage(reason: ConnectionRejection): string {
  switch (reason) {
    case 'self-loop':
      return 'A layer cannot feed itself.';
    case 'duplicate':
      return 'Those two are already connected in that direction.';
    case 'cycle':
      return 'That would close a loop, and the model has to stay acyclic.';
    case 'unresolved-module':
      return 'This block does not name an entry and exit layer to wire through.';
    default:
      return 'Nothing can feed an Input, and an Output cannot feed anything.';
  }
}
