import { ModelGraph } from '../types';
import type { LayoutDirection } from '../canvas/edgeAnchors';
import { createGraphFile, GraphFile, graphFileName } from '../services/graphFile';
import { LocalBrowserPageState } from './localBrowserState';
import type { ProjectionPayload } from './protocol.generated';
import type { PortableCapture } from '../services/portableCapture';
import type { LocalBrowserPortableClient } from './localBrowserPortableClient';
import { sameValidationTarget } from './localBrowserValidationClient';
import { graphDocumentV2ToModelGraph } from '../services/graphDocumentAdapter';
import type { GraphDocumentV2 } from '../types/graphDocument';

function authoritativeGraph(shown: ModelGraph, projection: ProjectionPayload | null): ModelGraph {
  // Source assignment cards are display-only; they never enter a graph file.
  const graph = { ...shown, nodes: shown.nodes.filter(node => !node.data.sourceAssignment) };
  if (!projection?.debug) return graph;
  const current = graphDocumentV2ToModelGraph(projection.document as GraphDocumentV2);
  const displayed = new Map(graph.nodes.map(node => [node.id, node]));
  return { ...current, nodes: current.nodes.map(node => ({ ...node,
    position: displayed.get(node.id)?.position ?? node.position,
    data: { ...node.data, expanded: displayed.get(node.id)?.data.expanded ?? node.data.expanded } })) };
}

export function canExportGraphFile(page: LocalBrowserPageState): boolean {
  return Boolean(page.syncState && page.projection && page.activeTarget
    && page.activeTarget.cls === page.projection.cls
    && page.activeTarget.file === page.projection.file);
}

export function exportCurrentGraphFile(
  graph: ModelGraph,
  page: LocalBrowserPageState,
  displayed: ProjectionPayload | null,
  connected = true,
  capture?: PortableCapture,
  direction?: LayoutDirection,
): GraphFile {
  const projected = page.projection;
  if (!canExportGraphFile(page) || !projected || !displayed
    || displayed.graphId !== projected.graphId || displayed.sourceDigest !== projected.sourceDigest
    || displayed.revision !== projected.revision) {
    throw new Error('Wait for this model to finish opening before exporting its graph.');
  }
  return createGraphFile(authoritativeGraph(graph, projected), {
    name: projected.cls,
    status: !connected || page.syncState === 'stale' || page.greyed ? 'stale'
      : page.syncState === 'runtime-traced' && page.traced ? 'traced' : 'static',
    warningCount: projected.diagnostics?.length ?? 0,
    capture, direction,
  });
}

export interface GraphExportState { graph: ModelGraph; page: LocalBrowserPageState; displayed: ProjectionPayload | null; connected: boolean; direction?: LayoutDirection }

/** Prepare asynchronously, then recheck exactly what the canvas displays. A
 * newer graph must never be paired with the previous run's portable payload. */
export async function prepareCurrentGraphFile(current: () => GraphExportState, client?: LocalBrowserPortableClient | null): Promise<GraphFile> {
  const start = current();
  const ref = authoritativeGraph(start.graph, start.displayed).nodes.find(node => node.data.captureRef)?.data.captureRef;
  if (!ref) return exportCurrentGraphFile(start.graph, start.page, start.displayed, start.connected, undefined, start.direction);
  if (!client || !start.connected || !start.page.activeTarget || !canExportGraphFile(start.page)) {
    throw new Error('Connect to tensorViz and finish opening this graph before exporting its capture.');
  }
  const result = await client.prepare({ captureId: ref.captureId, sourceId: ref.sourceId });
  if (!result.capture) throw new Error(result.detail);
  const next = current();
  if (!next.connected || !next.page.activeTarget || !sameValidationTarget(start.page.activeTarget, next.page.activeTarget)
    || next.displayed?.graphId !== start.displayed?.graphId || next.displayed?.revision !== start.displayed?.revision
    || next.displayed?.sourceDigest !== start.displayed?.sourceDigest) {
    throw new Error('The graph changed during export. Export its current version again.');
  }
  return exportCurrentGraphFile(next.graph, next.page, next.displayed, next.connected, result.capture, next.direction);
}

export function downloadGraphFile(file: GraphFile): void {
  const url = URL.createObjectURL(new Blob([JSON.stringify(file)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = graphFileName(file.document.semantic.name);
  document.body.appendChild(link);
  try { link.click(); } finally {
    link.remove();
    // Keep the object alive until the browser has started its download.
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
