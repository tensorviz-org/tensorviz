/**
 * The drag-to-connect gesture's transient state and lifecycle.
 *
 * It deliberately creates NO tldraw shape. useTldrawBridge's edges effect
 * deletes every model-edge AND arrow on the page and recreates them whenever any
 * node position changes, so anything drawn as a real shape would be erased
 * mid-drag. Instead the whole gesture lives in one atom, written here and read
 * by ConnectionLayer (mounted through tldraw's OnTheCanvas slot).
 *
 * The listeners are window CAPTURE phase on purpose: tldraw's own pointermove is
 * a document.body BUBBLE listener, so capture runs first and the drag keeps
 * tracking even while the pointer is over a card.
 */
import React, { useContext } from 'react';
import { atom, Editor, TLShape, TLShapeId } from 'tldraw';
import {
  ConnectableKind,
  ConnectableNode,
  ConnectionResult,
  ConnectionSide,
  GraphEdgeLike,
  anchorReach,
  validateConnection,
} from './connectivity';
import { shapeIdToNodeId } from './canvasShapeId';
import { nodeIdToShapeId } from './canvasShapeId';
import { AnchorBox, AnchorSide, anchorPoint, boxCenter, sourceSide, targetSide } from './edgeAnchors';
import { portSides } from './portLayout';

/** Mirrors React Flow's connectionRadius = 20 (@reactflow/core esm/index.js:4096).
 *  Without a snap radius the gesture feels finicky and reads as broken. */
export const SNAP_RADIUS_PX = 20;

/** Past this much screen travel the gesture is a drag rather than the first
 *  click of react-flow's connectOnClick. */
const DRAG_THRESHOLD_PX = 4;

export interface ConnectionCandidate {
  node: ConnectableNode;
  /** Page-space port point the rubber band lands on. */
  point: { x: number; y: number };
  result: ConnectionResult;
}

export interface ConnectionDragState {
  anchor: ConnectableNode;
  anchorSide: ConnectionSide;
  reach: ReadonlySet<string>;
  /** Page-space centre of the port the gesture started from. */
  from: { x: number; y: number };
  /** Page-space pointer position. */
  to: { x: number; y: number };
  candidate: ConnectionCandidate | null;
  moved: boolean;
  /** True after a click-to-connect's first click: the band follows the cursor
   *  until a second click lands. */
  pending: boolean;
}

export const connectionDrag = atom<ConnectionDragState | null>('connectionDrag', null);

/* ── The app-state channel ───────────────────────────────── */

export interface ConnectionGraph {
  /** Logical edges — hidden ones included, synthetic frame edges excluded — so
   *  duplicate and cycle checks see the real topology. */
  edges: readonly GraphEdgeLike[];
  commit(source: string, target: string): void;
}

const EMPTY_GRAPH: ConnectionGraph = { edges: [], commit: () => {} };

/**
 * Handed down as a REF, not a value: ports read the graph only at pointerdown,
 * and a plain context value would re-render every port on every graph edit.
 */
export const ConnectionGraphContext = React.createContext<React.MutableRefObject<ConnectionGraph>>({
  current: EMPTY_GRAPH,
});

export const useConnectionGraph = () => useContext(ConnectionGraphContext);

/* ── Shape → connectable node ────────────────────────────── */

const SHAPE_KIND: Record<string, ConnectableKind> = {
  'input-node': 'input',
  'output-node': 'output',
  'module-node': 'module',
  layer: 'layer',
};

/**
 * A shape's connectable identity, or null for anything that is not a wireable
 * model node (edges, the chat card, an EXPANDED module frame — its children are
 * on the canvas carrying their own ports, so wiring the frame too would double up).
 */
export function connectableFromShape(shape: TLShape | undefined): ConnectableNode | null {
  if (!shape) return null;
  const kind = SHAPE_KIND[shape.type];
  if (!kind) return null;
  const data = (shape as any).props?.nodeData || {};
  if (kind === 'module' && data.expanded) return null;
  return {
    id: shapeIdToNodeId(shape.id),
    kind,
    moduleEntryNodeId: data.moduleEntryNodeId,
    moduleExitNodeId: data.moduleExitNodeId,
  };
}

/**
 * Where a port sits in page space. The same anchors useTldrawBridge uses for the
 * committed edge (a chosen side, four sides now), so the ghost and the real edge
 * land on the same point.
 */
export function portPagePoint(
  editor: Editor,
  shapeId: TLShapeId,
  side: AnchorSide,
): { x: number; y: number } | null {
  const bounds = editor.getShapePageBounds(shapeId);
  if (!bounds) return null;
  return anchorPoint({ x: bounds.x, y: bounds.y, w: bounds.width, h: bounds.height }, side);
}

/** A shape's page box in the AnchorBox shape the anchor rules use. */
function shapeBox(editor: Editor, shapeId: TLShapeId): AnchorBox | null {
  const bounds = editor.getShapePageBounds(shapeId);
  return bounds ? { x: bounds.x, y: bounds.y, w: bounds.width, h: bounds.height } : null;
}

/* ── Candidate hit-testing ───────────────────────────────── */

function findCandidate(
  editor: Editor,
  state: ConnectionDragState,
  page: { x: number; y: number },
  edges: readonly GraphEdgeLike[],
): ConnectionCandidate | null {
  const wanted: ConnectionSide = state.anchorSide === 'source' ? 'target' : 'source';
  // The snap radius is a SCREEN distance, so it stays 20px at every zoom.
  const radius = SNAP_RADIUS_PX / editor.getZoomLevel();
  const anchorBox = shapeBox(editor, nodeIdToShapeId(state.anchor.id));
  const toward = anchorBox ? boxCenter(anchorBox) : state.from;
  let best: ConnectionCandidate | null = null;
  let bestDistance = Infinity;

  const shapes = editor.getCurrentPageShapes();
  for (let i = 0; i < shapes.length; i += 1) {
    const node = connectableFromShape(shapes[i]);
    if (!node) continue;
    const bounds = editor.getShapePageBounds(shapes[i].id);
    if (!bounds) continue;
    const box = { x: bounds.x, y: bounds.y, w: bounds.width, h: bounds.height };
    // The side this card's edge will use, matching the committed edge: its own
    // published side when it has neighbours, otherwise the geometry rule.
    const nodeSides = portSides.get().get(node.id);
    const side: AnchorSide = wanted === 'source'
      ? nodeSides?.out ?? sourceSide(box, toward)
      : nodeSides?.in ?? targetSide(box, toward);
    const point = anchorPoint(box, side);
    const distance = Math.sqrt((point.x - page.x) ** 2 + (point.y - page.y) ** 2);
    // Dropping anywhere on the card counts too — that is how a drop onto an
    // Input can be reported as invalid rather than silently missing.
    if (distance > radius && !bounds.containsPoint(page)) continue;
    if (distance >= bestDistance) continue;
    bestDistance = distance;
    best = {
      node,
      point,
      result: validateConnection(
        { node: state.anchor, side: state.anchorSide, reach: state.reach },
        node,
        edges,
      ),
    };
  }
  return best;
}

/* ── Gesture lifecycle ───────────────────────────────────── */

let session: { dispose: () => void } | null = null;

/** Tear the gesture down without committing. */
export function cancelConnection(): void {
  session?.dispose();
  session = null;
  connectionDrag.set(null);
}

export interface StartConnectionOpts {
  node: ConnectableNode;
  side: ConnectionSide;
  /** Which of the card's four sides the gesture starts from. */
  anchorSide: AnchorSide;
  shapeId: TLShapeId;
  /** Screen point of the initiating pointerdown. The drag threshold measures
   *  from the real press, not the port centre, which the press is never exactly on. */
  origin: { x: number; y: number };
}

export function startConnection(
  editor: Editor,
  graph: React.MutableRefObject<ConnectionGraph>,
  { node, side, anchorSide, shapeId, origin }: StartConnectionOpts,
): void {
  const from = portPagePoint(editor, shapeId, anchorSide);
  if (!from) return;

  cancelConnection();
  connectionDrag.set({
    anchor: node,
    anchorSide: side,
    reach: anchorReach(graph.current.edges, node, side),
    from,
    to: from,
    candidate: null,
    moved: false,
    pending: false,
  });

  const track = (event: PointerEvent) => {
    const state = connectionDrag.get();
    if (!state) return;
    const page = editor.screenToPage({ x: event.clientX, y: event.clientY });
    const travelled =
      Math.sqrt((event.clientX - origin.x) ** 2 + (event.clientY - origin.y) ** 2) > DRAG_THRESHOLD_PX;
    connectionDrag.set({
      ...state,
      to: page,
      moved: state.moved || travelled,
      candidate: findCandidate(editor, state, page, graph.current.edges),
    });
  };

  const settle = (candidate: ConnectionCandidate | null) => {
    if (candidate && candidate.result.ok) {
      graph.current.commit(candidate.result.source, candidate.result.target);
    }
    cancelConnection();
  };

  const onUp = () => {
    const state = connectionDrag.get();
    if (!state || state.pending) return;
    if (!state.moved) {
      // react-flow's connectOnClick: a press that never travelled arms the
      // second click instead of cancelling.
      connectionDrag.set({ ...state, pending: true });
      return;
    }
    settle(state.candidate);
  };

  const onDown = (event: PointerEvent) => {
    const state = connectionDrag.get();
    if (!state || !state.pending) return;
    const page = editor.screenToPage({ x: event.clientX, y: event.clientY });
    const candidate = findCandidate(editor, state, page, graph.current.edges);
    if (candidate && candidate.result.ok) {
      // Consume the click so the port under it does not immediately start a
      // fresh gesture, and so tldraw does not also act on it.
      editor.markEventAsHandled(event);
      event.stopPropagation();
      settle(candidate);
      return;
    }
    // Anywhere else — including empty canvas — cancels silently.
    cancelConnection();
  };

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Escape') cancelConnection();
  };

  window.addEventListener('pointermove', track, true);
  window.addEventListener('pointerup', onUp, true);
  window.addEventListener('pointercancel', cancelConnection, true);
  window.addEventListener('pointerdown', onDown, true);
  window.addEventListener('keydown', onKeyDown, true);

  session = {
    dispose: () => {
      window.removeEventListener('pointermove', track, true);
      window.removeEventListener('pointerup', onUp, true);
      window.removeEventListener('pointercancel', cancelConnection, true);
      window.removeEventListener('pointerdown', onDown, true);
      window.removeEventListener('keydown', onKeyDown, true);
    },
  };
}
