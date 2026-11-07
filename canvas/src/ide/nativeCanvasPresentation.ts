import { useEffect, useRef } from 'react';
import type { Editor, TLShapeId } from 'tldraw';
import type { ModelEdge, ModelNode } from '../types';
import { postToHost } from './bridge';
import { edgeIdToShapeId, nodeIdToShapeId } from '../canvas/canvasShapeId';
import { reconcileModuleView } from '../services/canvasLayout';
import { validModuleFocusPresentation, type ModuleFocusPresentation } from '../canvas/moduleFocusPresentation';
import type { LayoutDirection } from '../canvas/edgeAnchors';
import { isLayoutChoice, LayoutChoice } from './layoutDirection';

/** The Layout choice and the direction the saved positions were laid out in. */
export interface NativeLayout { choice?: LayoutChoice; direction?: LayoutDirection }

interface Presentation {
  graphId: string;
  camera: { x: number; y: number; z: number };
  selected: TLShapeId[];
  moduleFocus?: ModuleFocusPresentation;
  nodes: Array<{ id: string; x?: number; y?: number; expanded?: boolean; hidden?: boolean; selected?: boolean; passThroughShown?: boolean; operationDetailsShown?: boolean; captureDetailLevel?: 'layers' | 'operations' }>;
  layout?: NativeLayout;
}
let retained: Presentation | undefined;
let capture: (() => Presentation | undefined) | undefined;
/** Called by the authenticated native transport before the host removes the view. */
export function captureNativePresentation(): Presentation | undefined { return capture?.() ?? retained; }
/** The Layout choice the recreated view should start with, whichever graph it shows. */
export const retainedLayoutChoice = (): LayoutChoice | undefined => retained?.layout?.choice;
export const nativeCanvasMode = () => typeof window !== 'undefined' && new URL(window.location.href).searchParams.get('view') === 'canvas';

/** Only presentation fields are restored. Source graph/parameters and measured
 * evidence always come from the host's current projection. */
export function restoreNativePresentation(value: unknown): void {
  const state = value as Presentation | undefined;
  if (!state || typeof state.graphId !== 'string' || !Array.isArray(state.nodes) ||
      !Array.isArray(state.selected) || !state.selected.every(id => typeof id === 'string') ||
      !state.camera || ![state.camera.x, state.camera.y, state.camera.z].every(Number.isFinite) || state.camera.z <= 0) return;
  if (!state.nodes.every(node => typeof node?.id === 'string' &&
      (node.x === undefined || Number.isFinite(node.x)) && (node.y === undefined || Number.isFinite(node.y)) &&
      (node.captureDetailLevel === undefined || ['layers', 'operations'].includes(node.captureDetailLevel)) &&
      [node.expanded, node.hidden, node.selected, node.passThroughShown, node.operationDetailsShown].every(value => value === undefined || typeof value === 'boolean'))) return;
  if (state.moduleFocus !== undefined && !validModuleFocusPresentation(state.moduleFocus)) return;
  if (state.layout !== undefined && (typeof state.layout !== 'object' || state.layout === null
      || !(state.layout.choice === undefined || isLayoutChoice(state.layout.choice))
      || !(state.layout.direction === undefined || ['horizontal', 'vertical'].includes(state.layout.direction)))) return;
  retained = state;
}

export function useNativeCanvasPresentation(graphId: string | undefined, nodes: ModelNode[],
  setNodes: (update: (nodes: ModelNode[]) => ModelNode[]) => void, editor: Editor | null,
  initialFit: { current: string | null }, edges?: ModelEdge[], setEdges?: (edges: ModelEdge[]) => void,
  focus?: { capturePresentation: () => ModuleFocusPresentation; restorePresentation: (value: unknown, nodes?: ModelNode[]) => void; isReady: () => boolean },
  layout?: NativeLayout, onRestoreLayout?: (layout: NativeLayout) => void) {
  const restored = useRef<string | undefined>(undefined);
  const pendingSelection = useRef<TLShapeId[] | undefined>(undefined);
  const currentNodes = useRef(nodes);
  currentNodes.current = nodes;
  const currentFocus = useRef(focus); currentFocus.current = focus;
  const currentLayout = useRef(layout);
  currentLayout.current = layout;
  const restoreLayout = useRef(onRestoreLayout);
  restoreLayout.current = onRestoreLayout;
  useEffect(() => {
    if (!nativeCanvasMode() || !editor || !graphId || nodes.length === 0) return;
    if (restored.current !== graphId && retained?.graphId === graphId) {
      restored.current = graphId;
      const previous = new Map(retained.nodes.map(node => [node.id, node]));
      initialFit.current = graphId;
      editor.setCamera(retained.camera);
      // Expanded children do not have shapes until the restored nodes commit.
      // Keep their selection until the bridge has created those shapes.
      pendingSelection.current = retained.moduleFocus?.scopeId ? undefined : retained.selected.filter(id => editor.getShape(id)
        || nodes.some(node => nodeIdToShapeId(node.id) === id) || edges?.some(edge => edgeIdToShapeId(edge) === id));
      const restoredNodes = nodes.map(node => {
        const saved = previous.get(node.id);
        if (!saved) return node;
        return { ...node, ...(saved.x !== undefined && saved.y !== undefined ? { position: { x: saved.x, y: saved.y } } : {}),
          ...(saved.hidden !== undefined ? { hidden: saved.hidden } : {}),
          ...(saved.selected !== undefined ? { selected: saved.selected } : {}),
          data: { ...node.data, ...(saved.expanded !== undefined ? { expanded: saved.expanded } : {}),
            passThroughShown: saved.passThroughShown === true, captureDetailLevel: saved.captureDetailLevel,
            operationDetailsShown: saved.operationDetailsShown } };
      });
      // Saved positions keep the direction they were laid out in; the canvas
      // re-lays them out only if its current direction differs.
      const graph = reconcileModuleView(restoredNodes, edges ?? [], { preserveChildPositions: true, direction: retained.layout?.direction });
      setNodes(() => graph.nodes);
      setEdges?.(graph.edges);
      if (retained.moduleFocus) currentFocus.current?.restorePresentation(retained.moduleFocus, graph.nodes);
      restoreLayout.current?.(retained.layout ?? {});
      return;
    }
    if (restored.current !== graphId) pendingSelection.current = undefined;
    restored.current = graphId;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const restoreSelection = () => {
      if (currentFocus.current && !currentFocus.current.isReady()) return;
      const selected = pendingSelection.current;
      if (selected && selected.every(id => editor.getShape(id))) {
        pendingSelection.current = undefined;
        editor.select(...selected);
      }
    };
    const snapshot = () => {
      clearTimeout(timer);
      // A scope change is not checkpointed halfway through camera restoration.
      if (currentFocus.current && !currentFocus.current.isReady()) return;
      restoreSelection();
      const camera = editor.getCamera();
      const state: Presentation = { graphId, selected: [...(pendingSelection.current ?? editor.getSelectedShapeIds())], camera: { x: camera.x, y: camera.y, z: camera.z },
        moduleFocus: currentFocus.current?.capturePresentation(),
        nodes: currentNodes.current.map(node => ({ id: node.id, x: node.position?.x, y: node.position?.y,
          expanded: typeof node.data.expanded === 'boolean' ? node.data.expanded : undefined,
          passThroughShown: node.data.passThroughShown,
          captureDetailLevel: node.data.captureDetailLevel,
          operationDetailsShown: node.data.operationDetailsShown,
          hidden: node.hidden, selected: node.selected })),
        ...(currentLayout.current ? { layout: currentLayout.current } : {}) };
      retained = state;
      return state;
    };
    const checkpoint = () => {
      const state = snapshot();
      if (!state) return retained;
      // Presentation is a shell message, deliberately outside the graph protocol.
      postToHost({ type: 'canvasPresentation', state } as never);
      return state;
    };
    // The authenticated transport publishes this snapshot with its request ID.
    capture = snapshot;
    const save = () => { restoreSelection(); clearTimeout(timer); timer = setTimeout(checkpoint, 100); };
    // Node changes (including expansion) are committed now; don't leave their
    // last gesture pending in a timer that disposal can discard.
    checkpoint();
    const off = editor.store.listen(save, { scope: 'all' });
    window.addEventListener('pointerup', checkpoint);
    window.addEventListener('blur', checkpoint);
    window.addEventListener('pagehide', checkpoint);
    return () => {
      clearTimeout(timer); off();
      window.removeEventListener('pointerup', checkpoint);
      window.removeEventListener('blur', checkpoint);
      window.removeEventListener('pagehide', checkpoint);
      if (capture === snapshot) capture = undefined;
    };
  }, [graphId, nodes, edges, setNodes, setEdges, editor, initialFit]);
}
