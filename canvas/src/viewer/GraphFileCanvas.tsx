import { getAssetUrls } from '@tldraw/assets/selfHosted';
import React, { useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { TldrawCanvas } from '../canvas/TldrawCanvas';
import { useTldrawBridge } from '../canvas/useTldrawBridge';
import { reconcileModuleView } from '../services/canvasLayout';
import { revealOperationGroup } from '../services/operationGroups';
import { expandedDetailNodes } from '../services/graphDetailLevels';
import '../components/GraphDetailToggle.css';
import { GraphFile } from '../services/graphFile';
import { useValue } from 'tldraw';
import { selectedEdgeId } from '../canvas/edgeSelection';
import { canonical, edgeTarget, GraphTarget, SavedGraphView, targetLabel } from '../services/graphCollaboration';
import { nodeIdToShapeId, shapeIdToNodeId } from '../canvas/canvasShapeId';
import { ModelEdge, ModelGraph, ModelNode } from '../types';
import { cameraForTarget, cameraForViewport, initialGraphFileView, revealGraphNode, restoreGraphFileView } from './graphFileView';
import { CaptureDetails } from '../canvas/CaptureDetails';
import { createOfflineCaptureClient } from './offlineCaptureClient';
import type { LocalBrowserCaptureClient } from '../ide/localBrowserCaptureClient';
import { PassThroughToggle } from '../components/PassThroughToggle';
import { showPassThroughCalls } from '../services/passThroughPresentation';
import { operationDisplayName } from '../canvas/operationDisplayName';
import { BlockSummary } from '../canvas/BlockSummary';

const viewerAssets = getAssetUrls({ baseUrl: './assets' });

export interface GraphFileCanvasHandle {
  captureView: () => SavedGraphView;
  restoreView: (view: SavedGraphView) => void;
  focusTarget: (target: GraphTarget) => void;
}
interface Props { file: GraphFile; initialView?: SavedGraphView; initialTarget?: GraphTarget; focusPanelWidth?: number; panelOpen?: boolean; onTargetChange?: (target: GraphTarget | null) => void; onDiscuss?: (target: GraphTarget) => void }
const GraphFileCanvas = React.forwardRef<GraphFileCanvasHandle, Props>(function GraphFileCanvas({ file, initialView, initialTarget, focusPanelWidth = 0, panelOpen = false, onTargetChange, onDiscuss }, ref) {
  const direction = file.document.view.layout.direction;
  const [captureClient, setCaptureClient] = useState<LocalBrowserCaptureClient | null>(null);
  useEffect(() => {
    const client = file.capture ? createOfflineCaptureClient(file.capture.capture) : null;
    setCaptureClient(client);
    return () => client?.dispose();
  }, [file.capture]);
  // Initial view is applied once per imported architecture, not on annotation edits.
  const entryView = useRef(initialView);
  const entryTarget = useRef(initialTarget);
  const initial = useMemo(() => {
    let graph = initialGraphFileView({ document: file.document });
    if (entryView.current) graph = restoreGraphFileView(graph, entryView.current, direction);
    const target = entryTarget.current;
    if (target) graph = (target.kind === 'node' ? [target.nodeId] : [target.source, target.target]).reduce((current, id) => revealGraphNode(current, id, direction), graph);
    return graph;
  }, [file.document, direction]);
  const [nodes, setNodes] = useState<ModelNode[]>(initial.nodes);
  const [edges, setEdges] = useState<ModelEdge[]>(initial.edges);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const graphRef = useRef<ModelGraph>({ nodes, edges });
  graphRef.current = { nodes, edges };
  const pendingView = useRef<SavedGraphView | null>(entryTarget.current ? null : entryView.current ?? null);
  const pendingTarget = useRef<GraphTarget | null>(entryTarget.current ?? null);
  const clickedEdgeId = useValue('graph-file-selected-edge', () => selectedEdgeId.get(), []);
  const selectedEdge = file.document.semantic.edges.find(edge => edge.edge_id === clickedEdgeId);
  const selectedTarget = useMemo<GraphTarget | null>(() => selectedEdge ? edgeTarget(selectedEdge) : selectedId ? { kind: 'node', nodeId: selectedId } : null, [selectedEdge, selectedId]);
  useEffect(() => { onTargetChange?.(selectedTarget); }, [selectedTarget, onTargetChange]);
  useEffect(() => () => { selectedEdgeId.set(null); }, []);
  const apply = useCallback((graph: ModelGraph) => { setNodes(graph.nodes); setEdges(graph.edges); }, []);
  const toggle = useCallback((id: string) => {
    const graph = graphRef.current;
    apply(reconcileModuleView(graph.nodes.map(node => node.id === id
      ? { ...node, data: { ...node.data, expanded: !node.data.expanded } } : node), graph.edges, { relayout: true, anchorNodeId: id, preserveChildPositions: true, direction }));
  }, [apply, direction]);
  const resetLayout = useCallback(() => {
    const graph = graphRef.current;
    apply(reconcileModuleView(graph.nodes, graph.edges, { relayout: true, direction }));
  }, [apply, direction]);
  const revealOperations = useCallback((id: string) => {
    const graph = graphRef.current;
    apply(reconcileModuleView(revealOperationGroup(graph.nodes, id), graph.edges,
      { relayout: true, anchorNodeId: id, preserveChildPositions: true, direction }));
  }, [apply, direction]);
  const withActions = useMemo(() => nodes.map(node => node.type === 'moduleNode'
    ? { ...node, data: { ...node.data, onToggleExpand: toggle, onResetLayout: resetLayout } }
    : node.data.operationGroup ? { ...node, data: { ...node.data, onRevealOperations: revealOperations } } : node), [nodes, toggle, resetLayout, revealOperations]);
  const bridge = useTldrawBridge({
    nodes: withActions, edges, direction, setNodes, setEdges,
    onSelectionChange: ids => { setSelectedId(ids[0] ?? null); if (ids.length) selectedEdgeId.set(null); },
    onNodeActivate: id => { if (graphRef.current.nodes.find(n => n.id === id)?.type === 'moduleNode') toggle(id); },
  });
  const { editorReady, fitView } = bridge;
  useImperativeHandle(ref, () => ({
    captureView: () => {
      const editor = bridge.editorRef.current;
      if (!editor) throw new Error('The canvas is still loading. Try again in a moment.');
      const graph = graphRef.current;
      editor.stopCameraAnimation();
      const camera = editor.getCamera();
      const viewport = editor.getViewportScreenBounds();
      const edge = file.document.semantic.edges.find(edge => edge.edge_id === selectedEdgeId.get());
      return { camera: { x: camera.x, y: camera.y, z: camera.z },
        nodePositions: Object.fromEntries(graph.nodes.map(node => [node.id, { ...node.position }])),
        expandedModuleIds: graph.nodes.filter(node => node.type === 'moduleNode' && node.data.expanded).map(node => node.id),
        selectedNodeIds: edge ? [] : editor.getSelectedShapeIds().map(shapeIdToNodeId).filter(id => graph.nodes.some(node => node.id === id)),
        ...(edge ? { selectedEdge: edgeTarget(edge) } : {}),
        viewport: { width: viewport.w, height: viewport.h },
      };
    },
    restoreView: view => { pendingTarget.current = null; pendingView.current = view; selectedEdgeId.set(null); apply(restoreGraphFileView(graphRef.current, view, direction)); },
    focusTarget: target => {
      const ids = target.kind === 'node' ? [target.nodeId] : [target.source, target.target];
      if (ids.some(id => !graphRef.current.nodes.some(node => node.id === id))) throw new Error('This target is missing from the current architecture.');
      if (target.kind === 'edge' && file.document.semantic.edges.filter(edge => canonical(edgeTarget(edge)) === canonical(target)).length !== 1) throw new Error('This connection is missing or ambiguous.');
      pendingView.current = null; pendingTarget.current = target;
      apply(ids.reduce((graph, id) => revealGraphNode(graph, id, direction), graphRef.current));
    },
  }), [bridge.editorRef, apply, file.document, direction]);
  useEffect(() => {
    const view = pendingView.current;
    const target = pendingTarget.current;
    if ((!view && !target) || !editorReady) return;
    const frame = requestAnimationFrame(() => {
      const editor = bridge.editorRef.current;
      if (!editor) return;
      if (view) {
        const ids = view.selectedNodeIds.map(nodeIdToShapeId).filter(id => editor.getShape(id));
        editor.select(...ids);
        const viewport = editor.getViewportScreenBounds();
        editor.setCamera(cameraForViewport(view, { width: viewport.w, height: viewport.h }), { animation: { duration: 0 } });
        const matches = view.selectedEdge ? file.document.semantic.edges.filter(edge => canonical(edgeTarget(edge)) === canonical(view.selectedEdge)) : [];
        selectedEdgeId.set(matches.length === 1 ? matches[0].edge_id : null);
        setSelectedId(view.selectedNodeIds.find(id => graphRef.current.nodes.some(n => n.id === id)) ?? null);
        pendingView.current = null;
      } else if (target) {
        const ids = (target.kind === 'node' ? [target.nodeId] : [target.source, target.target]).map(nodeIdToShapeId).filter(id => editor.getShape(id));
        editor.select(...ids);
        const bounds = editor.getSelectionPageBounds(); const viewport = editor.getViewportScreenBounds();
        if (bounds) editor.setCamera(cameraForTarget(bounds, { width: viewport.w, height: viewport.h }, focusPanelWidth), { animation: { duration: 0 } });
        if (target.kind === 'edge') {
          editor.selectNone(); setSelectedId(null);
          const edge = file.document.semantic.edges.find(edge => canonical(edgeTarget(edge)) === canonical(target));
          selectedEdgeId.set(edge?.edge_id ?? null);
        } else { setSelectedId(target.nodeId); selectedEdgeId.set(null); }
        pendingTarget.current = null;
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [nodes, editorReady, bridge.editorRef, file.document, focusPanelWidth]);
  useEffect(() => {
    if (!editorReady || pendingView.current || pendingTarget.current) return;
    const frame = requestAnimationFrame(fitView);
    return () => cancelAnimationFrame(frame);
  }, [editorReady, fitView]);
  const select = (id: string) => {
    selectedEdgeId.set(null);
    apply(revealGraphNode(graphRef.current, id, direction));
    setSelectedId(id);
    bridge.focusNode(id);
    setQuery('');
  };
  const setExpanded = (expanded: boolean) => {
    const graph = graphRef.current;
    apply(reconcileModuleView((expanded ? expandedDetailNodes(graph.nodes, 'operations') : graph.nodes).map(node => node.type === 'moduleNode'
      ? { ...node, data: { ...node.data, expanded } } : node), graph.edges, { relayout: true, direction }));
    requestAnimationFrame(fitView);
  };
  const selected = nodes.find(node => node.id === selectedId);
  const toggleUnchanged = (shown: boolean) => {
    const graph = graphRef.current;
    apply(reconcileModuleView(showPassThroughCalls(graph.nodes, shown), graph.edges, { relayout: true, direction }));
  };
  const matches = query.trim() ? nodes.filter(node => [node.id, node.data.captureLabel, node.data.layerType.displayName,
    node.data.layerType.name, node.data.layerType.id, operationDisplayName(node.data)].join(' ').toLowerCase().includes(query.trim().toLowerCase())).slice(0, 50) : [];
  const navigation = (
    <nav className="graph-viewer-tools" aria-label="Graph navigation">
      <PassThroughToggle nodes={nodes} onToggle={toggleUnchanged} />
      <div className="graph-viewer-controls">
        <button onClick={fitView}>Fit</button>
        <button aria-label="Zoom in" onClick={() => bridge.editorRef.current?.zoomIn()}>+</button>
        <button aria-label="Zoom out" onClick={() => bridge.editorRef.current?.zoomOut()}>−</button>
        <button onClick={() => setExpanded(false)}>Blocks</button>
        <button onClick={() => setExpanded(true)}>Expand all</button>
      </div>
      {/* Claim search keys before tldraw's native container key handler. */}
      <input aria-label="Search layers" placeholder="Find a layer or module…" value={query}
        onChange={event => setQuery(event.target.value)} onKeyDownCapture={event => {
          if (event.key === 'Escape') { event.stopPropagation(); setQuery(''); }
          if (event.key === 'Enter' && matches[0]) { event.stopPropagation(); select(matches[0].id); }
        }} />
      {query.trim() && <div className="graph-viewer-results" aria-label="Search results">
        {matches.length ? matches.map(node => <button key={node.id} onClick={() => select(node.id)}>
          <span>{operationDisplayName(node.data) ?? node.data.captureLabel ?? node.id}</span><small>{node.data.layerType.displayName}</small>
        </button>) : <p>No matching layers.</p>}
      </div>}
    </nav>
  );
  return (
    <div className="graph-viewer-canvas ui2-reactflow-skin">
      <TldrawCanvas readOnly assetUrls={viewerAssets} sidePanelOpen={panelOpen || Boolean(selected)} navigation={navigation}
        onEditorReady={bridge.handleEditorReady} onSelectionChange={bridge.handleSelectionChange}
        onShapeClick={bridge.handleShapeClick} onShapeDoubleClick={bridge.handleShapeDoubleClick}
        onCanvasClick={bridge.handleCanvasClick} />
      {selectedEdge && !panelOpen && <aside className="graph-viewer-details" aria-label="Connection details">
        <button className="graph-viewer-close" aria-label="Close connection details" onClick={() => selectedEdgeId.set(null)}>×</button>
        <h2>Connection</h2><p className="graph-target-label">{targetLabel(edgeTarget(selectedEdge))}</p>
        <button onClick={() => onDiscuss?.(edgeTarget(selectedEdge))}>Discuss connection</button>
      </aside>}
      {selected && !selectedEdge && !panelOpen && <aside className="graph-viewer-details" aria-label="Layer details">
        <button className="graph-viewer-close" aria-label="Close layer details" onClick={() => { setSelectedId(null); bridge.editorRef.current?.selectNone(); }}>×</button>
        <h2>{operationDisplayName(selected.data) ?? selected.data.captureLabel ?? selected.id}</h2><p>{selected.data.layerType.displayName}</p>
        <BlockSummary node={selected} nodes={nodes} edges={edges} />
        <button onClick={() => onDiscuss?.({ kind: 'node', nodeId: selected.id })}>Discuss layer</button>
        {captureClient && selected.data.captureRef && <CaptureDetails key={selected.id} reference={selected.data.captureRef}
          label={selected.data.captureLabel ?? selected.data.layerType.displayName} client={captureClient} stale={file.status === 'stale'} />}
        {selected.data.resolutionDetail && <p role="note">{selected.data.resolutionDetail}</p>}
        {selected.type === 'moduleNode' && <button onClick={() => toggle(selected.id)}>{selected.data.expanded ? 'Collapse module' : 'Expand module'}</button>}
        <dl>
          {selected.data.inputShape && <><dt>Input shape</dt><dd>{JSON.stringify(selected.data.inputShape)}</dd></>}
          {selected.data.outputShape && <><dt>Output shape</dt><dd>{JSON.stringify(selected.data.outputShape)}</dd></>}
          {Object.entries(selected.data.parameters).filter(([key]) => !key.startsWith('_')).map(([key, value]) => <React.Fragment key={key}><dt>{key}</dt><dd>{typeof value === 'string' ? value : JSON.stringify(value)}</dd></React.Fragment>)}
        </dl>
        {!Object.keys(selected.data.parameters).length && <p>No parameters recorded.</p>}
      </aside>}
    </div>
  );
});
export default GraphFileCanvas;
