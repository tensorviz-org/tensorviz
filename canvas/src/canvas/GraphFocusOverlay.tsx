import React from 'react';
import type { ModelNode } from '../types';
import { track, useEditor } from 'tldraw';
import { graphFocus, setTraceDirection } from './graphFocus';
import { selectedEdgeId } from './edgeSelection';
import { isModelShape } from './canvasShapePolicy';
import { cameraDuration } from './reducedMotion';
import { GraphQueryPanel } from './GraphQueryPanel';
import { graphEdges, graphNodes } from './graphQueryState';
import { BlockSummary, FamilyLegend } from './BlockSummary';
import { LocalBrowserContext } from '../ide/localBrowserContext';
import { CaptureDetails } from './CaptureDetails';
import { operationDisplayName } from './operationDisplayName';
import { customCardModel } from './customCard';

/** Optional surface navigation shares the query's flow instead of covering it. */
export const GraphNavigationContext = React.createContext<React.ReactNode>(null);

/** Screen-space controls stay readable even when the entire graph is tiny. */
export const GraphFocusOverlay = track(function GraphFocusOverlay() {
  const navigation = React.useContext(GraphNavigationContext);
  const local = React.useContext(LocalBrowserContext);
  const editor = useEditor();
  const state = graphFocus(editor);
  const focus = state.focus.get();
  const edge = state.index.get().edges.get(selectedEdgeId.get() ?? '');
  // This index exists only for this render, and only when a label is displayed.
  let nodesById: Map<string, ModelNode> | undefined;
  const label = (id: string) => {
    if (!nodesById) nodesById = new Map(graphNodes.get().map(node => [node.id, node]));
    const node = nodesById.get(id);
    return node ? customCardModel(node.data.layerType.id, node.data.parameters)?.title
      ?? operationDisplayName(node.data) ?? node.data.captureLabel ?? id : id;
  };
  const selectedLabel = focus.active && focus.selected.length === 1 ? label(focus.selected[0]) : undefined;
  const selected = focus.selected.length === 1 ? graphNodes.get().find(node => node.id === focus.selected[0]) : undefined;
  const zoom = editor.getEfficientZoomLevel();
  const readable = () => {
    const shape = editor.getSelectedShapes().find(isModelShape) || editor.getCurrentPageShapes().find(isModelShape);
    const bounds = shape && editor.getShapePageBounds(shape.id);
    if (bounds) editor.zoomToBounds(bounds, { targetZoom: 1, inset: 80, animation: { duration: cameraDuration(220) } });
  };
  const claim = (event: React.PointerEvent) => { editor.markEventAsHandled(event); event.stopPropagation(); };
  return <div className="graph-focus-overlay" data-local={Boolean(local)} onPointerDown={claim} onPointerUp={claim}>
    {navigation}
    {!local && <GraphQueryPanel />}
    {!local && <details className="graph-legend">
      <summary>Connection guide</summary>
      <span><i className="graph-legend-solid" /> Data flow · arrow points to destination</span>
      <span><i className="graph-legend-in" /> Feeds the focused layer</span>
      <span><i className="graph-legend-out" /> Fed by the focused layer</span>
      <span><i className="graph-legend-dots" /> Residual / skip · still carries data</span>
      <span><i className="graph-legend-dashes" /> Secondary project relationship</span>
      <span>Mixed ×N combines ordinary and residual paths.</span>
      <span>Comparison: + added · − removed · ~ changed.</span>
      <span>Canvas keys: [ / ] select a layer · Enter opens it · Esc clears focus.</span>
    </details>}
    {!local && <FamilyLegend nodes={graphNodes.get()} />}
    {!local && zoom < .55 && <button className="graph-readable" onClick={readable}>Zoom to readable detail ↗</button>}
    {focus.active && <section className="graph-focus-controls" aria-label="Connection focus">
      <div className="graph-focus-caption"><strong>{focus.selected.length === 1 ? selectedLabel : `${focus.selected.length} layers`}</strong><span>{focus.edgeIds.size} connections{focus.truncated ? ' · limit reached' : ''}</span></div>
      {selected && <BlockSummary node={selected} nodes={graphNodes.get()} edges={graphEdges.get()} />}
      <div role="group" aria-label="Trace direction">
        <button aria-pressed={focus.direction === 'local'} onClick={() => setTraceDirection(editor, 'local')}>Neighbors</button>
        <button aria-label="Trace upstream inputs" aria-pressed={focus.direction === 'upstream'} onClick={() => setTraceDirection(editor, 'upstream')}>← Inputs</button>
        <button aria-label="Trace downstream outputs" aria-pressed={focus.direction === 'downstream'} onClick={() => setTraceDirection(editor, 'downstream')}>Outputs →</button>
      </div>
      {focus.direction !== 'local' && <small>Tracing visible connections · does not run the model</small>}
      {selected?.data.captureRef && local?.captures && <CaptureDetails
        key={`${selected.data.captureRef.captureId}:${selected.id}`} reference={selected.data.captureRef}
        label={selectedLabel ?? selected.id} client={local.captures} stale={local.page.greyed || local.page.syncState === 'stale'} />}
    </section>}
    {edge && !focus.active && <section className="graph-focus-controls" aria-label="Selected connection">
      <strong>{label(edge.source)} → {label(edge.target)}</strong>
      <span>{edge.role === 'residual' ? 'Residual / skip' : edge.role === 'mixed' ? 'Mixed ordinary and residual paths' : edge.relationship || 'Data flow'}{(edge.count ?? 0) > 1 ? ` · ${edge.count} original connections` : ''}</span>
      {edge.originals && <details><summary>Original connections</summary><ul>{edge.originals.map(id => <li key={id}>{id}</li>)}</ul></details>}
    </section>}
  </div>;
});
