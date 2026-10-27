/**
 * The published side map for the port dots.
 *
 * The pure rule lives in `edgeAnchors.computePortSides`; this file only holds the
 * tldraw atom that carries it, so the geometry stays importable and testable
 * without pulling tldraw into a unit test.
 *
 * Stores only the SIDE, never an absolute point: the caller applies it to the
 * card's current box, so the endpoint still tracks a card being dragged even
 * while this map is one sync behind.
 */
import { atom } from 'tldraw';
import { NodePortSides } from './edgeAnchors';

export const portSides = atom<ReadonlyMap<string, NodePortSides>>('portSides', new Map());
