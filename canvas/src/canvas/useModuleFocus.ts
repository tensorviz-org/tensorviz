import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { Editor } from 'tldraw';
import type { ModelEdge, ModelNode } from '../types';
import { canEnterModule, focusLayoutKey, layoutModuleFocus, matchingFocusSibling, moduleFocusPath,
  placeModuleFocus, projectModuleFocus, recapturedModuleMap, type FocusLayout, type FocusOrientation } from '../services/moduleFocus';
import { nodeIdToShapeId, shapeIdToNodeId } from './canvasShapeId';
import { selectedEdgeId } from './edgeSelection';
import { nodeCardSize } from './cardMetrics';
import { graphSvgBounds } from '../services/graphSvg';
import { captureFocusFrame, type FocusFrameSnapshot } from './moduleFocusMotion';
import { validModuleFocusPresentation, type FocusCamera, type ModuleFocusPresentation } from './moduleFocusPresentation';

interface ScopeView {
  camera?: FocusCamera;
  selected: string[];
  orientation: FocusOrientation;
  layout?: FocusLayout;
  layoutKey?: string;
  layoutRequest?: { key: string; width: number; height: number; orientation: FocusOrientation };
  positions: Map<string, { x: number; y: number }>;
  restoredDirection?: 'LR' | 'TB';
  measured?: boolean;
  restoredLayoutKey?: string;
}
export interface FocusTransition {
  revision: number;
  from: string | null;
  to: string | null;
  anchorId?: string;
  direction: number;
  snapshot?: FocusFrameSnapshot;
}
const viewKey = (id: string | null) => id ?? '__overview__';
const emptyView = (): ScopeView => ({ selected: [], orientation: 'auto', positions: new Map() });

/** Navigation owns presentation only. The bridge's position writes are diverted
 * into this scope's cache, while canonical selection is handled by its caller. */
export function useModuleFocus({ nodes, edges, setNodes, graphId, sourceKey, enabled, svg = false, onRestoreSelection }: {
  nodes: ModelNode[]; edges: ModelEdge[]; setNodes: React.Dispatch<React.SetStateAction<ModelNode[]>>;
  graphId?: string; enabled: boolean;
  sourceKey?: string;
  svg?: boolean; onRestoreSelection?: (ids: string[]) => void;
}) {
  const [scopeId, setScopeId] = useState<string | null>(null);
  const [editor, setEditor] = useState<Editor | null>(null);
  const [version, setVersion] = useState(0);
  const [cameraRequest, setCameraRequest] = useState(0);
  const [settled, setSettled] = useState(0);
  const [transition, setTransition] = useState<FocusTransition | null>(null);
  const views = useRef(new Map<string, ScopeView>());
  const revision = useRef(0);
  const pendingCamera = useRef(false);
  const restoreCamera = useRef(false);
  const committedNodes = useRef<ModelNode[] | undefined>(undefined);
  const committedScene = useRef<{ scopeId: string | null; graphId?: string } | undefined>(undefined);
  const appliedViewport = useRef<{ width: number; height: number } | undefined>(undefined);
  const previousGraph = useRef(graphId);
  const previousNodes = useRef(nodes);
  const previousSource = useRef(sourceKey);
  const state = useRef({ nodes, edges, scopeId, editor, enabled, svg, onRestoreSelection, graphId });
  state.current = { nodes, edges, scopeId, editor, enabled, svg, onRestoreSelection, graphId };
  const scopeView = (id: string | null) => {
    const key = viewKey(id);
    if (!views.current.has(key)) views.current.set(key, emptyView());
    return views.current.get(key)!;
  };
  const currentPath = useMemo(() => moduleFocusPath(nodes, scopeId), [nodes, scopeId]);
  const lastPath = useRef<string[]>([]);
  // Selection is tldraw state (or the SVG selection overlay). Avoid reconciling
  // and rewriting every visible card just because one canonical flag changed.
  const inputCache = useRef<{ sources: ModelNode[]; inputs: ModelNode[] }>({ sources: [], inputs: [] });
  const inputs = useMemo(() => {
    const previous = inputCache.current;
    const same = (a: Record<string, any>, b: Record<string, any>, omit: string[]) => {
      const keys = Object.keys(a).filter(key => !omit.includes(key));
      return keys.length === Object.keys(b).filter(key => !omit.includes(key)).length && keys.every(key => a[key] === b[key]);
    };
    const next = nodes.map((node, i) => {
      const old = previous.sources[i];
      return old && old.id === node.id && (old === node || (same(old, node, ['selected', 'data']) && same(old.data, node.data, ['isSelected'])))
        ? previous.inputs[i] : { ...node, selected: false, data: { ...node.data, isSelected: false } };
    });
    const inputs = next.length === previous.inputs.length && next.every((node, i) => node === previous.inputs[i]) ? previous.inputs : next;
    inputCache.current = { sources: nodes, inputs }; return inputs;
  }, [nodes]);
  const projection = useMemo(() => scopeId && enabled ? projectModuleFocus(inputs, edges, scopeId) : null, [inputs, edges, scopeId, enabled]);
  const lastFocus = useRef<{ graph: { nodes: ModelNode[]; edges: ModelEdge[] }; path: ModelNode[] } | undefined>(undefined);
  const migrating = Boolean(enabled && scopeId && !projection && previousGraph.current === graphId && lastFocus.current);
  const path = migrating ? lastFocus.current!.path : currentPath;
  const overview = projection ? null : { nodes, edges };
  const overviewNodes = overview?.nodes, overviewEdges = overview?.edges;
  const derivedGraph = useMemo(() => {
    // The revision invalidates memoized positions after scope-local drag/layout edits.
    void version;
    if (!projection || !scopeId) return { nodes: overviewNodes!, edges: overviewEdges! };
    const view = scopeView(scopeId), key = focusLayoutKey(projection);
    if (view.layoutKey !== key || !view.layout) {
      const rect = editor?.getContainer().getBoundingClientRect();
      const viewport = { width: rect?.width || 1000, height: rect?.height || 700 };
      const orientation = view.restoredDirection ?? view.layout?.direction ?? view.orientation;
      view.layout = layoutModuleFocus(projection, viewport, orientation);
      view.layoutRequest = view.measured ? undefined : { key, ...viewport, orientation };
      if (view.restoredLayoutKey && view.restoredLayoutKey !== key) {
        view.positions.clear(); view.camera = undefined;
      }
      view.layoutKey = key; view.restoredLayoutKey = undefined;
    }
    const placed = placeModuleFocus(projection, view.layout);
    return { ...placed, nodes: placed.nodes.map(node => view.positions.has(node.id)
      ? { ...node, position: view.positions.get(node.id)! } : node) };
  }, [projection, scopeId, overviewNodes, overviewEdges, editor, version]);
  // A fresh capture replaces IDs before the migration effect runs. Keep the
  // previous focus frame for this commit instead of rebuilding the whole model.
  const graph = migrating ? lastFocus.current!.graph : derivedGraph;
  if (projection) lastFocus.current = { graph, path };
  const graphRef = useRef(graph); graphRef.current = graph;
  const active = Boolean(scopeId && (projection || migrating));
  const layoutKey = active && scopeId ? scopeView(scopeId).layoutKey : undefined;
  const regularZoomSteps = useMemo(() => editor?.getCameraOptions().zoomSteps, [editor]);
  useEffect(() => {
    if (!editor || !active || !regularZoomSteps) return;
    return () => { editor.setCameraOptions({ zoomSteps: regularZoomSteps }); };
  }, [editor, active, regularZoomSteps]);
  const updateZoomRange = useCallback(() => {
    if (!editor || !active || !regularZoomSteps) return;
    const boxes = graph.nodes.filter(node => !node.hidden).map(node => ({ ...node.position, ...nodeCardSize(node) }));
    if (!boxes.length) return;
    const width = Math.max(...boxes.map(box => box.x + box.w)) - Math.min(...boxes.map(box => box.x));
    const height = Math.max(...boxes.map(box => box.y + box.h)) - Math.min(...boxes.map(box => box.y));
    const rect = editor.getContainer().getBoundingClientRect();
    // Automatic and explicit Fit must cover a huge scope, even below 5%.
    const inset = Math.max(64, Math.min(editor.options?.zoomToFitPadding ?? 128, rect.width * .28));
    // Do not tighten the range mid-drag after Fit; that would move the camera.
    const minimum = Math.min(regularZoomSteps[0], editor.getCameraOptions().zoomSteps[0], Math.max(1, rect.width - inset) / Math.max(1, width),
      Math.max(1, rect.height - inset) / Math.max(1, height));
    editor.setCameraOptions({ zoomSteps: minimum < regularZoomSteps[0] ? [minimum, ...regularZoomSteps] : regularZoomSteps });
  }, [editor, active, regularZoomSteps, graph]);
  useEffect(() => {
    if (!editor || !active) return;
    updateZoomRange(); editor.on('resize', updateZoomRange);
    return () => { editor.off('resize', updateZoomRange); };
  }, [editor, active, updateZoomRange]);
  useLayoutEffect(() => {
    if (!scopeId || !projection || !editor) return;
    const view = scopeView(scopeId);
    if (view.measured) return;
    const rect = editor.getContainer().getBoundingClientRect();
    const orientation = view.restoredDirection ?? view.orientation;
    const key = focusLayoutKey(projection);
    const request = view.layoutRequest;
    // The first render already laid out this scope. Reuse it only when the
    // committed viewport and requested inputs match; changed measurements still
    // take the original measured-layout path.
    if (!view.layout || !request || request.key !== key || request.width !== rect.width
      || request.height !== rect.height || request.orientation !== orientation) {
      view.layout = layoutModuleFocus(projection, { width: rect.width, height: rect.height }, orientation);
    }
    view.layoutRequest = undefined;
    view.measured = true;
    view.restoredDirection = undefined;
    setVersion(value => value + 1);
  }, [scopeId, projection, editor]);

  const checkpoint = useCallback(() => {
    const { scopeId: id, editor: current, nodes: canonical } = state.current;
    if (!current || pendingCamera.current) return;
    const view = scopeView(id), camera = current.getCamera();
    view.camera = { x: camera.x, y: camera.y, z: camera.z };
    const known = new Set(canonical.map(node => node.id));
    view.selected = state.current.svg ? canonical.filter(node => node.selected).map(node => node.id)
      : current.getSelectedShapeIds().map(shapeIdToNodeId).filter(id => known.has(id));
  }, []);

  const navigate = useCallback((id: string | null, options: { immediate?: boolean; direction?: number; nodes?: ModelNode[] } = {}) => {
    const current = state.current;
    if (!current.enabled || id === current.scopeId) return false;
    if (id && !projectModuleFocus(options.nodes ?? current.nodes, current.edges, id)) return false;
    current.editor?.stopCameraAnimation?.();
    checkpoint();
    const oldPath = moduleFocusPath(current.nodes, current.scopeId).map(node => node.id);
    const nextPath = moduleFocusPath(current.nodes, id).map(node => node.id);
    const outward = !id || oldPath.includes(id);
    const anchorId = outward ? oldPath[id ? oldPath.indexOf(id) + 1 : 0] : nextPath[current.scopeId ? nextPath.indexOf(current.scopeId) + 1 : 0];
    if (outward && anchorId) scopeView(id).selected = [anchorId];
    const sequence = ++revision.current;
    restoreCamera.current = false;
    pendingCamera.current = true;
    setTransition(options.immediate || !current.editor ? null : { revision: sequence, from: current.scopeId, to: id,
      anchorId, direction: options.direction || 0, snapshot: captureFocusFrame(current.editor, graphRef.current, anchorId) });
    state.current.scopeId = id;
    selectedEdgeId.set(null);
    setScopeId(id); setVersion(value => value + 1); setCameraRequest(value => value + 1);
    return true;
  }, [checkpoint]);
  const back = useCallback(() => {
    const current = state.current;
    const ancestry = moduleFocusPath(current.nodes, current.scopeId);
    navigate(ancestry.length > 1 ? ancestry[ancestry.length - 2].id : null);
  }, [navigate]);
  const switchSibling = useCallback((ancestor: string, sibling: string) => {
    const current = state.current;
    if (!current.scopeId) return;
    const next = matchingFocusSibling(current.nodes, current.scopeId, ancestor, sibling);
    const owner = current.nodes.find(node => node.id === ancestor)?.data.parentId;
    const siblings = current.nodes.filter(node => node.type === 'moduleNode' && node.data.parentId === owner);
    navigate(next, { direction: Math.sign(siblings.findIndex(node => node.id === sibling) - siblings.findIndex(node => node.id === ancestor)) });
  }, [navigate]);
  const setOrientation = useCallback((orientation: FocusOrientation) => {
    const id = state.current.scopeId;
    if (!id) return;
    checkpoint();
    const view = scopeView(id);
    view.orientation = orientation; view.layout = undefined; view.camera = undefined; view.positions.clear(); view.measured = false;
    pendingCamera.current = true;
    restoreCamera.current = false;
    setTransition(null); setVersion(value => value + 1); setCameraRequest(value => value + 1);
  }, [checkpoint]);

  const refit = useCallback(() => {
    checkpoint();
    restoreCamera.current = false;
    pendingCamera.current = true;
    setTransition(null); setCameraRequest(value => value + 1);
  }, [checkpoint]);

  const setDisplayNodes = useCallback<React.Dispatch<React.SetStateAction<ModelNode[]>>>(update => {
    // A user operation can finish inside the bridge's deferred sync window.
    // Never interpret the departing scene's positions in the target scope.
    if (pendingCamera.current || committedScene.current?.scopeId !== state.current.scopeId
      || committedScene.current?.graphId !== state.current.graphId) return;
    if (!state.current.scopeId) { setNodes(update); return; }
    const previous = graphRef.current.nodes;
    const next = typeof update === 'function' ? update(previous) : update;
    const byId = new Map(previous.map(node => [node.id, node]));
    let changed = false;
    const view = scopeView(state.current.scopeId);
    for (const node of next) {
      const before = byId.get(node.id);
      if (before && (before.position.x !== node.position.x || before.position.y !== node.position.y)) {
        view.positions.set(node.id, { ...node.position }); changed = true;
      }
    }
    if (changed) setVersion(value => value + 1);
  }, [setNodes]);
  const finishTransition = useCallback((sequence?: number) => {
    setTransition(current => sequence === undefined || current?.revision === sequence ? null : current);
  }, []);

  const capturePresentation = useCallback((): ModuleFocusPresentation => {
    checkpoint();
    return { scopeId: state.current.scopeId, views: Array.from(views.current).map(([key, view]) => ({
      id: key === '__overview__' ? null : key, camera: view.camera, selected: [...view.selected],
      orientation: view.orientation, direction: view.layout?.direction, positions: Array.from(view.positions),
      layoutKey: view.layoutKey,
    })) };
  }, [checkpoint]);
  const restorePresentation = useCallback((value: unknown, restoredNodes?: ModelNode[]) => {
    if (!validModuleFocusPresentation(value)) return;
    const source = restoredNodes ?? state.current.nodes;
    const known = new Set(source.map(node => node.id));
    views.current.clear();
    for (const view of value.views) if (view.id === null || known.has(view.id)) {
      views.current.set(viewKey(view.id), { ...view, selected: view.selected.filter(id => known.has(id)),
        positions: new Map(view.positions), restoredDirection: view.direction, restoredLayoutKey: view.layoutKey, layoutRequest: undefined });
    }
    const id = value.scopeId && canEnterModule(source, value.scopeId) ? value.scopeId : null;
    state.current.scopeId = id;
    restoreCamera.current = true;
    pendingCamera.current = true;
    setScopeId(id); setTransition(null); setVersion(value => value + 1); setCameraRequest(value => value + 1);
  }, []);

  // Model replacement clears every view. A same-model source refresh falls back
  // along the previously valid path, never into an unrelated module with a name match.
  useEffect(() => {
    const oldNodes = previousNodes.current; previousNodes.current = nodes;
    const sameSource = Boolean(sourceKey && previousSource.current === sourceKey);
    previousSource.current = sourceKey;
    if (previousGraph.current !== graphId || !enabled) {
      previousGraph.current = graphId; views.current.clear(); lastPath.current = []; lastFocus.current = undefined;
      pendingCamera.current = false; restoreCamera.current = false; applied.current = undefined;
      state.current.scopeId = null; appliedViewport.current = undefined;
      setScopeId(null); setTransition(null); return;
    }
    if (scopeId && !projection) {
      const replacements = recapturedModuleMap(oldNodes, nodes, sameSource);
      checkpoint();
      const migrated = new Map<string, ScopeView>();
      views.current.forEach((view, id) => {
        const next = replacements.get(id);
        if (next) migrated.set(next, { ...view, layout: undefined, layoutKey: undefined,
          restoredDirection: view.layout?.direction, measured: false, selected: [], positions: new Map() });
        else if (id === '__overview__' || nodes.some(node => node.id === id)) migrated.set(id, view);
      });
      views.current = migrated;
      const survivor = [...lastPath.current].reverse().map(id => replacements.get(id) ?? id).find(id => canEnterModule(nodes, id));
      pendingCamera.current = true;
      restoreCamera.current = false;
      setCameraRequest(value => value + 1);
      setScopeId(survivor ?? null); setTransition(null);
    } else lastPath.current = path.map(node => node.id);
  }, [graphId, sourceKey, enabled, scopeId, projection, nodes, path, checkpoint]);

  // Runs after the bridge has committed shapes. Camera reads never use an
  // animated DOM container, and no camera animation races the depth overlay.
  const applied = useRef<{ id: string | null; version: number; layoutKey?: string; svg: boolean } | undefined>(undefined);
  useEffect(() => {
    if (!editor || state.current.scopeId !== scopeId || (!active && !applied.current && !pendingCamera.current)) return;
    if (applied.current?.id === scopeId && applied.current.version === cameraRequest
      && applied.current.layoutKey === layoutKey && applied.current.svg === svg) return;
    checkpoint();
    pendingCamera.current = true;
    let frame = 0;
    const apply = () => {
      if (!state.current.svg && committedNodes.current !== graphRef.current.nodes) { frame = requestAnimationFrame(apply); return; }
      const view = scopeView(scopeId);
      const rect = editor.getContainer().getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) { frame = requestAnimationFrame(apply); return; }
      editor.updateViewportScreenBounds(editor.getContainer());
      updateZoomRange();
      editor.stopCameraAnimation();
      if (restoreCamera.current && view.camera) editor.setCamera(view.camera, { animation: { duration: 0 } });
      else if (!state.current.svg) editor.zoomToFit({ animation: { duration: 0 } });
      else {
        const boxes = graphSvgBounds(graphRef.current.nodes);
        if (boxes.length) {
          const x = Math.min(...boxes.map(box => box.x)), y = Math.min(...boxes.map(box => box.y));
          editor.zoomToBounds({ x, y, w: Math.max(...boxes.map(box => box.x + box.w)) - x,
            h: Math.max(...boxes.map(box => box.y + box.h)) - y }, { inset: 48, animation: { duration: 0 } });
        }
      }
      const ids = view.selected.map(nodeIdToShapeId).filter(id => editor.getShape(id));
      editor.select(...ids);
      if (state.current.svg) state.current.onRestoreSelection?.(view.selected.filter(id => graphRef.current.nodes.some(node => node.id === id)));
      applied.current = { id: scopeId, version: cameraRequest, layoutKey, svg };
      restoreCamera.current = false;
      pendingCamera.current = false;
      appliedViewport.current = { width: rect.width, height: rect.height };
      setSettled(revision.current);
    };
    frame = requestAnimationFrame(apply);
    return () => cancelAnimationFrame(frame);
  }, [editor, active, scopeId, cameraRequest, graph, layoutKey, svg, checkpoint, updateZoomRange]);

  const clearSelection = () => { editor?.selectNone(); state.current.onRestoreSelection?.([]); };
  const markSceneCommitted = useCallback((nodes: ModelNode[]) => {
    committedNodes.current = nodes;
    if (nodes === graphRef.current.nodes) committedScene.current = { scopeId: state.current.scopeId, graphId: state.current.graphId };
  }, []);

  return { active, scopeId, path, graph, graphRef, editor, attachEditor: setEditor, navigate, back, switchSibling,
    setDisplayNodes, setOrientation, orientation: scopeId ? scopeView(scopeId).orientation : 'auto' as FocusOrientation,
    direction: scopeId ? scopeView(scopeId).layout?.direction : undefined, transition, settled, checkpoint,
    isReady: () => !pendingCamera.current && (state.current.svg || (committedScene.current?.scopeId === state.current.scopeId
      && committedScene.current?.graphId === state.current.graphId)),
    consumesResize: (rect: { width: number; height: number }) => {
      const matches = appliedViewport.current?.width === rect.width && appliedViewport.current?.height === rect.height;
      appliedViewport.current = undefined;
      return matches;
    }, refit,
    markSceneCommitted, navigationRevision: () => revision.current,
    finishTransition, capturePresentation, restorePresentation, clearSelection };
}

export type ModuleFocusController = ReturnType<typeof useModuleFocus>;
