import React from 'react';
import { createPortal } from 'react-dom';
import type { ModelNode, ModelNodeData } from '../types';
import { track, useEditor, type TLShapeId } from 'tldraw';
import { graphFocus, setTraceDirection } from './graphFocus';
import { selectedEdgeId } from './edgeSelection';
import { isModelShape } from './canvasShapePolicy';
import { cameraDuration } from './reducedMotion';
import { GraphQueryPanel } from './GraphQueryPanel';
import { graphEdges, graphNodes, graphSceneNodes } from './graphQueryState';
import { FamilyLegend } from './BlockSummary';
import { LocalBrowserContext } from '../ide/localBrowserContext';
import { NodeInspector, type NodeInspectionAction } from './NodeInspector';
import { NodeActionToolbar, nodeToolbarPosition } from './NodeActionToolbar';
import { hasDeclarationEdit } from '../ide/directSourceEdits';
import { canEnterModule } from '../services/moduleFocus';
import { publicLabel } from './nodePresentation';
import { customCardModel } from './customCard';
import { operationDisplayName } from './operationDisplayName';
import { NodeExplanationContext } from './NodeExplanationContext';
import { useNodeExplanationAction } from './useNodeExplanationAction';
import { nodeIdToShapeId } from './canvasShapeId';
import { ViewExplanationOverview } from './ViewExplanationOverview';
import { openNodeParameters } from './nodeParameters';
import { TemplateEditingContext } from './NodeParametersPopover';

/** Optional surface navigation shares the query's flow instead of covering it. */
export const GraphNavigationContext = React.createContext<React.ReactNode>(null);
export const GraphActivationContext = React.createContext<((shapeId: TLShapeId) => void) | undefined>(undefined);

/** Screen-space controls stay readable even when the entire graph is tiny. */
export const GraphFocusOverlay = track(function GraphFocusOverlay() {
  const navigation = React.useContext(GraphNavigationContext);
  const local = React.useContext(LocalBrowserContext);
  const templateEditing = React.useContext(TemplateEditingContext);
  const explanations = React.useContext(NodeExplanationContext);
  const editor = useEditor();
  const activate = React.useContext(GraphActivationContext);
  const [inspecting, setInspecting] = React.useState(false);
  const [inspectionAction, setInspectionAction] = React.useState<NodeInspectionAction>();
  const [tracedSelection, setTracedSelection] = React.useState<string | null>(null);
  const state = graphFocus(editor);
  const focus = state.focus.get();
  const edge = state.index.get().edges.get(selectedEdgeId.get() ?? '');
  // This index exists only for this render, and only when a label is displayed.
  let nodesById: Map<string, ModelNode> | undefined;
  const label = (id: string) => {
    if (!nodesById) {
      nodesById = new Map(graphNodes.get().map(node => [node.id, node]));
      for (const node of graphSceneNodes.get()) if (!nodesById.has(node.id)) nodesById.set(node.id, node);
    }
    const node = nodesById.get(id);
    const explanation = explanations?.annotations?.[id] ?? explanations?.states[id];
    if (explanation?.status === 'ready' && explanation.explanation?.title) return explanation.explanation.title;
    const data = node?.data ?? focusBoundaryCard(id);
    return data ? customCardModel(data.layerType.id, data.parameters)?.title
      ?? operationDisplayName(data) ?? publicLabel(data.captureLabel) : undefined;
  };
  // Module focus draws presentation-only boundary cards ("From …", "To …") that are
  // not nodes of the graph, so graphNodes cannot name them. Read the card itself
  // rather than showing its encoded ID.
  const focusBoundaryCard = (id: string) => {
    const card = (editor.getShape(nodeIdToShapeId(id))?.props as { nodeData?: ModelNodeData } | undefined)?.nodeData;
    return card?.focusBoundary ? card : undefined;
  };
  const selectedLabel = focus.active && focus.selected.length === 1 ? label(focus.selected[0]) : undefined;
  const selected = focus.selected.length === 1 ? graphNodes.get().find(node => node.id === focus.selected[0])
    ?? graphSceneNodes.get().find(node => node.id === focus.selected[0]) : undefined;
  const explainAction = useNodeExplanationAction(selected?.id);
  const sourceLabel = edge && !focus.active ? label(edge.source) : undefined;
  const targetLabel = edge && !focus.active ? label(edge.target) : undefined;
  const selectionKey = focus.selected.join('\0');
  const tracing = tracedSelection === selectionKey;
  React.useEffect(() => {
    if (!selectionKey && inspecting) setInspecting(false);
  }, [selectionKey, inspecting]);
  React.useEffect(() => {
    const container = editor.getContainer?.();
    const inspect = () => { if (selected) { setInspectionAction(undefined); setInspecting(true); } };
    container?.addEventListener('tensorviz-inspect-selection', inspect);
    return () => container?.removeEventListener('tensorviz-inspect-selection', inspect);
  }, [editor, selected]);
  const pendingReveal = React.useRef<number | null>(null);
  React.useEffect(() => () => { if (pendingReveal.current !== null) cancelAnimationFrame(pendingReveal.current); }, []);
  const navigate = (id: string) => {
    const shape = editor.getShape(nodeIdToShapeId(id));
    if (!shape) return;
    editor.select(shape.id);
    const reveal = () => {
      pendingReveal.current = null;
      if (!editor.getShape(shape.id)) return;
      const bounds = editor.getShapePageBounds(shape.id);
      if (bounds) editor.zoomToBounds(bounds, { targetZoom: 1, inset: 72, animation: { duration: cameraDuration(220) } });
    };
    if (pendingReveal.current !== null) cancelAnimationFrame(pendingReveal.current);
    const host = editor.getContainer?.()?.closest('.tldraw-canvas-wrapper');
    if (host && host.getBoundingClientRect().width <= 760) {
      setInspecting(false);
      pendingReveal.current = requestAnimationFrame(reveal);
    } else reveal();
  };
  const canonical = graphNodes.get(), scene = graphSceneNodes.get();
  const enterable = React.useMemo(() => selected?.type === 'moduleNode' && !!activate && canEnterModule(canonical, selected.id),
    [canonical, selected, activate]);
  const openInside = selected && activate && enterable
    ? () => { setInspecting(false); activate(nodeIdToShapeId(selected.id)); } : undefined;
  const inspectionNodes = React.useMemo(() => {
    if (!inspecting) return [];
    const ids = new Set(canonical.map(node => node.id));
    return [...canonical, ...scene.filter(node => !ids.has(node.id))];
  }, [inspecting, canonical, scene]);
  const inspectorHost = editor.getContainer?.().closest('.tldraw-canvas-wrapper');
  const inspector = inspecting && focus.active && selected && <NodeInspector key={selected.id} node={selected} title={selectedLabel}
    nodes={inspectionNodes} edges={graphEdges.get()} onNavigate={navigate} onOpenInside={openInside}
    action={inspectionAction?.nodeId === selected.id ? inspectionAction : undefined}
    explanationAction={explainAction}
    onBack={() => setInspecting(false)} />;
  const inspect = (section: NodeInspectionAction['section'] = 'details') => {
    if (!selected) return;
    setInspectionAction({ nodeId: selected.id, section }); setInspecting(true);
  };
  const boundary = selected && (!!selected.data.focusBoundary || selected.type === 'inputNode' || selected.type === 'outputNode');
  const explainable = selected && !boundary && !!explanations?.model;
  const discussable = selected && !!explanations && local?.sourceAction !== 'copy';
  const captured = selected?.data.captureRef && local?.captures;
  const editable = selected && !boundary && local?.sourceEdits && !local.reviewing && !local.page.greyed
    && (hasDeclarationEdit(selected.data) || selected.data.sourceEditCapabilities?.some(cap => cap.command === 'delete_unary_op'));
  const editableTemplate = selected && !boundary && !local && templateEditing;
  let toolbarPosition = { left: 8, top: 8 };
  let popoverAbove = false;
  if (focus.active && !inspector) {
    const bounds = selected ? editor.getShapePageBounds(nodeIdToShapeId(selected.id)) : editor.getSelectionPageBounds();
    if (bounds) {
      const start = editor.pageToViewport({ x: bounds.x, y: bounds.y });
      const end = editor.pageToViewport({ x: bounds.x + bounds.w, y: bounds.y + bounds.h });
      const actions = 2 + Number(!!selected) + Number(!!explainable) + Number(!!discussable) + Number(!!captured) + Number(!!openInside);
      const viewport = editor.getViewportScreenBounds();
      toolbarPosition = nodeToolbarPosition(start, end, viewport, actions * 32 + 20 + (focus.selected.length > 1 ? 80 : 0));
      popoverAbove = toolbarPosition.top + 160 > viewport.h;
    }
  }
  const toolbar = focus.active && !inspector && <NodeActionToolbar key={selectionKey} position={toolbarPosition} popoverAbove={popoverAbove} label={selectedLabel}
    count={focus.selected.length} onClaim={event => { editor.markEventAsHandled(event); event.stopPropagation(); }}
    onInspect={selected ? () => inspect() : undefined} onEnter={openInside} onEdit={editable ? () => inspect('edit') : undefined}
    onEditParameters={editableTemplate ? () => openNodeParameters(nodeIdToShapeId(selected.id)) : undefined}
    onDiscuss={discussable ? () => {
      window.dispatchEvent(new CustomEvent('builder-command', { detail: { command: 'report-selection' } }));
      window.dispatchEvent(new CustomEvent('tensorviz-agent-compose'));
    } : undefined}
    onCapture={captured ? () => inspect('capture') : undefined}
    onExplain={explainable ? explainAction.run : undefined} explainAction={explainAction}
    tracing={tracing} onTrace={() => setTracedSelection(value => value === selectionKey ? null : selectionKey)} onClear={() => editor.selectNone()}>
    <small>{`${focus.edgeIds.size} ${focus.edgeIds.size === 1 ? 'connection' : 'connections'}`}{focus.truncated ? ' · limit reached' : ''}</small>
    <div role="group" aria-label="Trace direction">
      <button aria-pressed={focus.direction === 'local'} onClick={() => setTraceDirection(editor, 'local')}>Neighbors</button>
      <button aria-label="Trace upstream inputs" aria-pressed={focus.direction === 'upstream'} onClick={() => setTraceDirection(editor, 'upstream')}>← Inputs</button>
      <button aria-label="Trace downstream outputs" aria-pressed={focus.direction === 'downstream'} onClick={() => setTraceDirection(editor, 'downstream')}>Outputs →</button>
    </div>
    {focus.direction !== 'local' && <small>Tracing visible connections · does not run the model</small>}
  </NodeActionToolbar>;
  const zoom = editor.getEfficientZoomLevel();
  const readable = () => {
    const shape = editor.getSelectedShapes().find(isModelShape) || editor.getCurrentPageShapes().find(isModelShape);
    const bounds = shape && editor.getShapePageBounds(shape.id);
    if (bounds) editor.zoomToBounds(bounds, { targetZoom: 1, inset: 80, animation: { duration: cameraDuration(220) } });
  };
  const claim = (event: React.PointerEvent) => { editor.markEventAsHandled(event); event.stopPropagation(); };
  return <><div className="graph-focus-overlay" data-local={Boolean(local)} onPointerDown={claim} onPointerUp={claim}>
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
    <ViewExplanationOverview key={explanations?.model?.state?.requestId} nodes={graphNodes.get()} focused={focus.active}
      selectedNodeIds={focus.selected} onNavigate={id => {
        const shape = editor.getShape(nodeIdToShapeId(id));
        if (shape) editor.select(shape.id);
      }} />
    {!local && zoom < .55 && <button className="graph-readable" onClick={readable}>Zoom to readable detail ↗</button>}

    {edge && !focus.active && <section className="graph-focus-controls" aria-label="Selected connection">
      {sourceLabel && targetLabel && <strong>{sourceLabel} → {targetLabel}</strong>}
      <span>{edge.role === 'residual' ? 'Residual / skip' : edge.role === 'mixed' ? 'Mixed ordinary and residual paths' : edge.relationship || 'Data flow'}{(edge.count ?? 0) > 1 ? ` · ${edge.count} original connections` : ''}</span>
      {edge.originals && <details><summary>Original connections</summary><ul>{edge.originals.map(id => <li key={id}>{id}</li>)}</ul></details>}
    </section>}
  </div>{toolbar}{inspector && (inspectorHost ? createPortal(inspector, inspectorHost) : inspector)}</>;
});
