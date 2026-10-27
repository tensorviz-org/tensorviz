/**
 * Inserting a node INTO a connection, by pointing at the connection.
 *
 * Putting a Slice between two layers already worked — select the upstream node,
 * Shift+A, pick the op, and addNode's auto-wire splices it. That stays as the
 * keyboard path. But it is a menu answer to a spatial question, and it has a real
 * flaw: it picks the downstream edge with `edges.find(e => e.source === anchor.id)`
 * — the FIRST one only — so on a node that already fans out, one branch is spliced
 * and the other silently keeps its direct edge. Slicing is the op you reach for
 * *because* you are branching, so that is the wrong failure to ship into.
 *
 * Pointing at the edge removes the ambiguity: source and target are both named,
 * so there is nothing to infer.
 *
 * The hit-testing this needs already exists — distanceToEdgeCurve, findEdgeAtPoint
 * and the curve midpoint were all built for edge-click-to-delete.
 */
import React, { useContext } from 'react';
import { atom } from 'tldraw';

/** The edge under the pointer, with the page-space midpoint of its curve. */
export interface HoveredEdge {
  edgeId: string;
  /** Page space — the t=0.5 point of the curve, which is also the chord midpoint. */
  point: { x: number; y: number };
}

export const hoveredEdge = atom<HoveredEdge | null>('hoveredEdge', null);

/** Set once the user commits to inserting here; drives the inline picker. */
export const insertTarget = atom<HoveredEdge | null>('edgeInsertTarget', null);

export interface EdgeInsertChannel {
  /** Create `layerTypeId` and splice it into `edgeId`, replacing that edge. */
  insert(edgeId: string, layerTypeId: string): void;
}

const NO_CHANNEL: EdgeInsertChannel = { insert: () => {} };

/** A ref, so a graph edit never re-renders the affordance. Same shape as ConnectionGraphContext. */
export const EdgeInsertContext = React.createContext<React.MutableRefObject<EdgeInsertChannel>>({
  current: NO_CHANNEL,
});

export const useEdgeInsert = () => useContext(EdgeInsertContext);

export function cancelEdgeInsert(): void {
  insertTarget.set(null);
}
