import React from 'react';
import CanvasTooltip from '../components/CanvasTooltip';
import { track, useEditor } from 'tldraw';
import { nodeIdToShapeId } from './canvasShapeId';
import { cameraDuration } from './reducedMotion';
import { formatOverlayValue, OverlayMetric } from './graphOverlay';
import {
  graphOverlayMetric,
  graphOverlayScale,
  graphSearchQuery,
  graphSearchResult,
  graphNodes,
} from './graphQueryState';
import { scopedNodeLabel } from './nodeScope';
import { operationDisplayName } from './operationDisplayName';

const METRICS: Array<{ id: OverlayMetric; label: string; help: string }> = [
  { id: 'none', label: 'Off', help: 'Use the normal node colors without a metric overlay.' },
  { id: 'params', label: 'Parameters', help: 'Color nodes by their exact parameter counts. Nodes without a current count remain uncolored.' },
  { id: 'activation', label: 'Output tensor bytes', help: 'Color nodes by the measured logical size of their output tensor. This is not peak GPU memory; shared storage can be counted more than once.' },
];

/** Screen-space, like the focus controls: a query is about the whole graph, so
 *  it must stay readable at any zoom. */
export const GraphQueryPanel = track(function GraphQueryPanel() {
  const editor = useEditor();
  const result = graphSearchResult.get();
  const query = graphSearchQuery.get();
  const labels = new Map(graphNodes.get().map(node => [node.id, operationDisplayName(node.data) ?? node.data.captureLabel]));

  /** Move the camera to a hit without changing what is open: a search that
   *  expanded containers to reach a node would undo the reader's own view. */
  const reveal = (nodeId: string) => {
    const bounds = editor.getShapePageBounds(nodeIdToShapeId(nodeId));
    if (!bounds) return;
    editor.select(nodeIdToShapeId(nodeId));
    editor.zoomToBounds(bounds, { targetZoom: 1, inset: 120, animation: { duration: cameraDuration(200) } });
  };

  return (
    <section className="graph-query" aria-label="Search and overlays">
      <div className="graph-query-row">
        <span aria-hidden="true">⌕</span>
        <input
          type="search"
          value={query}
          placeholder="type=Conv2d · shape=256 · attr:padding=1"
          aria-label="Search the graph"
          onChange={event => graphSearchQuery.set(event.target.value)}
        />
        {query.trim() !== '' && (
          <button type="button" aria-label="Clear search" onClick={() => graphSearchQuery.set('')}>✕</button>
        )}
      </div>

      {result.error && <p className="graph-query-error" role="alert">Not a valid pattern — {result.error}</p>}

      {!result.error && query.trim() !== '' && (
        <div className="graph-query-results">
          <p className="graph-query-count">
            {result.hits.length === 0
              ? 'No matches'
              : `${result.hits.length} ${result.hits.length === 1 ? 'match' : 'matches'} in ${result.groups.length} ${result.groups.length === 1 ? 'place' : 'places'}`}
          </p>
          <ul>
            {result.groups.slice(0, 8).map(group => (
              <li key={group.visibleId}>
                <button type="button" onClick={() => reveal(group.visibleId)}>
                  <strong>{labels.get(group.visibleId) ?? group.visibleId}</strong>
                  {/* A closed module reports what it contains instead of being
                      forced open — the count is the hit, until you open it. */}
                  {group.hits.length > 1 || group.hits[0].id !== group.visibleId ? (
                    <span className="graph-query-inside">
                      {group.hits.length} inside · {group.hits.slice(0, 3).map(hit => labels.get(hit.id) ?? scopedNodeLabel(hit.id, group.visibleId)).join(', ')}
                      {group.hits.length > 3 ? '…' : ''}
                    </span>
                  ) : (
                    <span className="graph-query-inside">{group.hits[0].matched}</span>
                  )}
                </button>
              </li>
            ))}
          </ul>
          {result.groups.length > 8 && <p className="graph-query-count">+{result.groups.length - 8} more places</p>}
        </div>
      )}

      <GraphOverlayControls />
    </section>
  );
});

/** Metrics are presentation state and can also be controlled outside tldraw. */
export const GraphOverlayControls = track(function GraphOverlayControls() {
  const scale = graphOverlayScale.get();
  const metric = graphOverlayMetric.get();
  const evidence = metric === 'params' ? 'exact parameter counts' : 'complete output tensor measurements';
  return <>
      <div className="graph-query-overlay" role="group" aria-label="Overlay metric">
        <span>Overlay</span>
        {METRICS.map(entry => (
          <CanvasTooltip key={entry.id} text={entry.help}><button
            type="button"
            aria-pressed={metric === entry.id}
            onClick={() => graphOverlayMetric.set(entry.id)}
          >
            {entry.label}
          </button></CanvasTooltip>
        ))}
      </div>

      {metric !== 'none' && (
        <div className="graph-query-legend">
          {scale.values.size === 0 ? (
            <span>No current {evidence} available.</span>
          ) : (
            <>
              <i className="graph-query-ramp" />
              <span>{formatOverlayValue(scale.min, scale.unit)}</span>
              <span className="graph-query-legend-max">{formatOverlayValue(scale.max, scale.unit)}</span>
              {/* Rule 12: say what was left out rather than colouring a guess. */}
              {scale.unmeasured > 0 && (
                <small>{scale.unmeasured} uncoloured — no current {evidence}</small>
              )}
            </>
          )}
          {metric === 'activation' && <small>Logical footprint of one tensor output per call; not peak memory. Shared storage may be counted more than once.</small>}
        </div>
      )}
    </>;
});
