/**
 * Search and overlay state for one canvas.
 *
 * Module-level atoms, matching `selectedEdgeId`: the shape utils read them
 * inside `component()`, which runs under tldraw's `useStateTracking`, so only
 * the cards whose state actually changed re-render. Putting this in React
 * context instead would re-render every shape on every keystroke.
 *
 * `graphNodes` is published at graph sync — the same point that rebuilds the
 * focus adjacency index, and for the same reason: it must not be recomputed on
 * a camera move.
 */
import { atom, computed } from 'tldraw';
import type { ModelEdge, ModelNode } from '../types';
import { searchGraph, SearchResult } from './graphSearch';
import { overlayScale, OverlayMetric, OverlayScale } from './graphOverlay';

export const graphNodes = atom<readonly ModelNode[]>('graphNodes', []);
/** Published with the nodes so the inspector can order a module's children by data flow. */
export const graphEdges = atom<readonly ModelEdge[]>('graphEdges', []);
export const graphSearchQuery = atom<string>('graphSearchQuery', '');
export const graphOverlayMetric = atom<OverlayMetric>('graphOverlayMetric', 'none');

export const graphSearchResult = computed<SearchResult>(
  'graphSearchResult',
  () => searchGraph(graphNodes.get(), graphSearchQuery.get()),
);

export const graphOverlayScale = computed<OverlayScale>(
  'graphOverlayScale',
  () => overlayScale(graphNodes.get(), graphOverlayMetric.get()),
);

/** How one card should present itself under the current query: lit, dimmed, or
 *  untouched because nothing is being asked. */
export function searchStateFor(nodeId: string, result: SearchResult): 'match' | 'dim' | undefined {
  if (!result.query.trim() || result.error) return undefined;
  if (result.hits.length === 0) return undefined;
  return result.visibleIds.has(nodeId) ? 'match' : 'dim';
}
