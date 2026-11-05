/**
 * useTldrawBridge — synchronises React Flow-style nodes/edges state
 * with a tldraw Editor instance.
 *
 * This is the key migration bridge. It lets the existing ModelBuilder
 * continue to use nodes/edges arrays internally while the canvas renders
 * through tldraw. Changes in either direction are propagated:
 *
 *   nodes/edges → tldraw store  (when the app updates nodes)
 *   tldraw store → nodes/edges  (when the user drags/creates on canvas)
 */
import { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import {
  Editor,
  TLShapeId,
} from 'tldraw';
import { ModelNode, ModelEdge } from '../types';
import { EDGE_SHAPE_TYPE, isManagedCanvasShape, isModelShape } from './canvasShapePolicy';
import { cameraDuration } from './reducedMotion';
import { createEdgeRouter, routedShapeGeometry } from './edgeRouting';
import { shouldCarryModuleChildren } from './dragCarry';
import { nodeCardSize } from './cardMetrics';
import { AnchorBox, LayoutDirection } from './edgeAnchors';
import { portSides } from './portLayout';
import { localBrowserMode } from '../ide/projection';
import { nodeIdToShapeId, edgeIdToShapeId } from './canvasShapeId';
import { updateFocusConnections } from './graphFocus';
import { graphEdges, graphNodes } from './graphQueryState';
import { graphIsReadOnly } from './graphAccess';
import { NodeExplanationContext } from './NodeExplanationContext';
import { explainedGraphPresentation } from './nodeExplanationPresentation';

/* ── Helpers ─────────────────────────────────────────────── */

/** A card's page-space box, with the same width/height fallbacks the endpoint
 *  maths has always used for a shape whose props are not yet measured. */
const shapeBox = (shape: any, fallbackW = 200, fallbackH = 120): AnchorBox => ({
  x: shape.x,
  y: shape.y,
  w: typeof shape.props?.w === 'number' ? shape.props.w : fallbackW,
  h: typeof shape.props?.h === 'number' ? shape.props.h : fallbackH,
});

/**
 * Tldraw intentionally makes its public mutation APIs no-ops in read-only mode.
 * Host projections are still allowed to replace the viewer's in-memory canvas,
 * so briefly lift that guard for one synchronous programmatic sync and restore
 * it before the browser can process another user event.
 */
const withProjectionWriteAccess = (editor: Editor, apply: () => void): void => {
  const restoreReadOnly = editor.getIsReadonly();
  if (restoreReadOnly) {
    editor.updateInstanceState({ isReadonly: false }, { history: 'ignore' });
  }
  try {
    apply();
  } finally {
    if (restoreReadOnly) {
      editor.updateInstanceState({ isReadonly: true }, { history: 'ignore' });
    }
  }
};



/**
 * Map React Flow node type to tldraw shape type.
 */
const rfTypeToTldrawType = (rfType: string): string => {
  switch (rfType) {
    case 'inputNode': return 'input-node';
    case 'outputNode': return 'output-node';
    case 'moduleNode': return 'module-node';
    case 'layerNode':
    default: return 'layer';
  }
};

/* ── Hook ────────────────────────────────────────────────── */

interface UseTldrawBridgeOpts {
  nodes: ModelNode[];
  edges: ModelEdge[];
  /** Search keeps the authoritative graph when the canvas shows one scope. */
  queryNodes?: ModelNode[];
  onNodesSynced?: (nodes: ModelNode[]) => void;
  setNodes: React.Dispatch<React.SetStateAction<ModelNode[]>>;
  setEdges: React.Dispatch<React.SetStateAction<ModelEdge[]>>;
  onSelectionChange?: (nodeIds: string[]) => void;
  /** A real click on a node — never a drag. See handleShapeClick. */
  onNodeClick?: () => void;
  /**
   * A node was ACTIVATED: double-clicked. Carries the node id, which the single
   * click deliberately does not — the inspector only needs "a node was clicked",
   * while "reveal this node's line in the editor" needs to know which one, and
   * separating the two keeps a click-to-inspect from stealing the editor's focus
   * on every selection.
   */
  onNodeActivate?: (nodeId: string) => void;
  /** The direction the nodes were laid out with; port sides and skip lanes follow it. */
  direction?: LayoutDirection;
}

export function useTldrawBridge({
  nodes: sourceNodes,
  edges,
  direction,
  setNodes,
  onSelectionChange,
  onNodeClick,
  onNodeActivate,
  queryNodes,
  onNodesSynced,
}: UseTldrawBridgeOpts) {
  const explanations = useContext(NodeExplanationContext);
  const presentation = useMemo(() => explainedGraphPresentation(sourceNodes, explanations?.annotations ?? explanations?.states ?? {},
    explanations?.annotationsVisible === true), [sourceNodes, explanations?.annotations, explanations?.states, explanations?.annotationsVisible]);
  const nodes = presentation.nodes;
  const offsets = useRef(presentation.offsets); offsets.current = presentation.offsets;
  const synced = useRef(onNodesSynced); synced.current = onNodesSynced;
  useEffect(() => { graphNodes.set(queryNodes ?? sourceNodes); }, [sourceNodes, queryNodes]);
  useEffect(() => { graphEdges.set(edges); }, [edges]);
  const editorRef = useRef<Editor | null>(null);
  const pendingFocusNodeRef = useRef<string | null>(null);
  /**
   * Resolve only actual semantic nodes, excluding edges and chat shapes.
   * Shared lossless shape IDs keep repeated, dotted and flat names distinct.
   */
  const nodeIdForShapeRef = useRef<Map<string, string>>(new Map());
  // Flips true once the tldraw editor mounts. Sync effects depend on it so a
  // graph already present at mount (create-from-template, opening a saved graph)
  // is pushed into the store even when the editor mounts AFTER nodes are set —
  // editorRef is a ref, so mutating it alone would never re-run those effects.
  const [editorReady, setEditorReady] = useState(false);
  // Bumped to re-fire a sync effect that had to bail because a from-editor sync
  // was mid-flight — so an app-originated change (e.g. group-as-module) that
  // arrived during the echo window is retried, not permanently swallowed.
  const [resyncTick, setResyncTick] = useState(0);
  const isSyncingToEditor = useRef(false);
  const isSyncingFromEditor = useRef(false);
  const lastNodesHash = useRef('');
  const lastEdgesHash = useRef('');
  // Last node.data object seen per node id. Producers replace data with a new
  // object when it changes and preserve the reference when it does not, so this
  // is both cheaper and truthful where `JSON.stringify(n.data)` was not.
  const lastNodeDataRef = useRef(new Map<string, unknown>());
  
  const edgesRef = useRef(edges);
  useEffect(() => { edgesRef.current = edges; }, [edges]);
  // Live nodes mirror: the module-drag handler below runs inside a tldraw
  // sideEffect and needs the current parentNode graph without re-registering.
  const routerRef = useRef<ReturnType<typeof createEdgeRouter> | undefined>(undefined);
  const nodesRef = useRef(nodes);
  useEffect(() => { nodesRef.current = nodes; }, [nodes]);

  /**
   * Called once when the tldraw editor mounts.
   */
  const handleEditorReady = useCallback((editor: Editor) => {
    editorRef.current = editor;
    setEditorReady(true); // re-run the sync effects now that the editor exists

    // Listen for store changes from user interactions
    const disposeOperationComplete = editor.sideEffects.registerOperationCompleteHandler((source) => {
      if (source !== 'user' || isSyncingToEditor.current) return;

      // Sync positions back to nodes
      isSyncingFromEditor.current = true;
      try {
        const shapes = editor.getCurrentPageShapes();
        const shapesById = new Map(shapes.map(shape => [shape.id, shape]));
        setNodes((prevNodes) => {
          return prevNodes.map((node) => {
            const shapeId = nodeIdToShapeId(node.id);
            const shape = shapesById.get(shapeId);
            if (!shape) return node;

            // Shapes live in absolute page space, but a module child's
            // node.position is RELATIVE to its module (React Flow sub-flow
            // semantics — see reconcileModuleView). Measure against the parent's
            // CURRENT shape rather than React state, so a module drag that moved
            // frame and children together leaves each child's relative position
            // untouched instead of compounding the offset.
            const parentShape = node.parentNode
              ? shapesById.get(nodeIdToShapeId(node.parentNode as string))
              : undefined;
            const displayedPos = parentShape
              ? { x: shape.x - parentShape.x, y: shape.y - parentShape.y }
              : { x: shape.x, y: shape.y };
            const offset = offsets.current.get(node.id);
            const nextPos = { x: displayedPos.x - (offset?.x ?? 0), y: displayedPos.y - (offset?.y ?? 0) };

            if (
              Math.abs(node.position.x - nextPos.x) > 1 ||
              Math.abs(node.position.y - nextPos.y) > 1
            ) {
              return { ...node, position: nextPos };
            }
            return node;
          });
        });

        const strayShapeIds = shapes
          .filter((shape) => !isManagedCanvasShape(shape))
          .map((shape) => shape.id);

        if (strayShapeIds.length > 0) {
          // history:'ignore' — programmatic cleanup must not land on the user's undo stack.
          editor.run(() => editor.deleteShapes(strayShapeIds), { history: 'ignore' });
          editor.setCurrentTool('select');
        }

      } finally {
        setTimeout(() => {
          isSyncingFromEditor.current = false;
        }, 0);
      }
    });

    // Continuously update connected edges while a node is being dragged
    const disposeAfterChange = editor.sideEffects.registerAfterChangeHandler('shape', (prev, next) => {
      // USER GESTURES ONLY. Everything below reacts to a shape moving by treating
      // the delta as a drag — the module branch carries the frame's children by it.
      // The sync effects below write shape geometry programmatically, and their
      // writes are already absolute and already include each child's own new
      // position (absolutePositionOf resolves module-relative → absolute in the
      // same pass). Letting this handler also fire for those writes added the
      // module's own displacement ON TOP of children that were already correct.
      //
      // That is what put an expanded module's layers outside its frame: expanding
      // a module moves its frame during the relayout, and each child ended up
      // displaced by (twice) that frame delta. Measured on F5-TTS — ConvNeXt-V2's
      // frame moved y 922 → 137 (Δ −785) and its first child landed at −1361
      // instead of 209, exactly 2 × 785 too high; Conv1D's frame moved y
      // 1396 → 1225 (Δ −171) and its first child landed 341px too high. X was
      // untouched in both because neither frame moved on x. The next relayout
      // recomputed positions from the graph and snapped them back, which is why
      // the breakage looked intermittent.
      //
      // The operation-complete handler above already guards on this ref; this one
      // was missing it. Edges do not need this handler during a programmatic sync
      // because the edge sync effect rebuilds every model-edge shape itself.
      if (isSyncingToEditor.current) return;

      // Only care about nodes moving or resizing
      if (prev.x === next.x && prev.y === next.y && (prev as any).props?.w === (next as any).props?.w && (prev as any).props?.h === (next as any).props?.h) return;
      if (!isModelShape(next)) return;

      const movedNodeId = nodeIdForShapeRef.current.get(next.id as string);
      if (!movedNodeId) return;

      // tldraw applies no parent transform to these shapes — they are absolute,
      // because the edge geometry below reads shape.x/shape.y directly — so a
      // module frame must translate its own children or it slides off them.
      // Left on the undo stack (NOT history:'ignore') so undoing a module drag
      // returns the frame AND its children as one reversible step; ignoring
      // history would return the frame and leave the children behind, and the
      // relative-position sync in the store handler would then bake in the
      // displacement.
      const moduleDx = next.x - prev.x;
      const moduleDy = next.y - prev.y;
      if (shouldCarryModuleChildren({
        isSyncingToEditor: isSyncingToEditor.current,
        shapeType: next.type,
        dx: moduleDx,
        dy: moduleDy,
      })) {
        // A child module's after-change handler carries its own children. Moving
        // every descendant here as well would translate nested leaves twice.
        const childUpdates = nodesRef.current
          .filter((nd) => nd.parentNode === movedNodeId && !nd.hidden)
          .map((nd) => editor.getShape(nodeIdToShapeId(nd.id)) as any)
          .filter(Boolean)
          .map((sh) => ({ id: sh.id, type: sh.type, x: sh.x + moduleDx, y: sh.y + moduleDy }));
        if (childUpdates.length > 0) {
          editor.run(() => editor.updateShapes(childUpdates), { ignoreShapeLock: true });
        }
      }
      // Moving a sibling into a skip path affects that path even when the
      // sibling is not one of its endpoints. Use the drawn bounds as a cheap
      // broad phase; the router still checks the exact sibling geometry.
      const movedBoxes = [shapeBox(prev), shapeBox(next)];
      const connectedEdges = edgesRef.current.filter(edge => {
        if (edge.hidden) return false;
        if (edge.source === movedNodeId || edge.target === movedNodeId) return true;
        const shape = editor.getShape(edgeIdToShapeId(edge));
        if (!shape) return false;
        const box = shapeBox(shape);
        return movedBoxes.some(moved => moved.x < box.x + box.w && moved.x + moved.w > box.x
          && moved.y < box.y + box.h && moved.y + moved.h > box.y);
      });
      if (connectedEdges.length === 0) return;

      const boxFor = (id: string) => {
        const shape = editor.getShape(nodeIdToShapeId(id)) as any;
        return shape ? shapeBox(shape) : undefined;
      };
      const shapesToUpdate: any[] = [];
      for (const edge of connectedEdges) {
        if (edge.hidden) continue;
        const edgeShapeId = edgeIdToShapeId(edge);
        if (!editor.getShape(edgeShapeId)) continue;

        const route = routerRef.current?.route(edge, boxFor);
        if (!route) continue;
        const geometry = routedShapeGeometry(route);
        const previous = editor.getShape(edgeShapeId) as any;
        if (previous.x === geometry.x && previous.y === geometry.y
          && Object.entries(geometry.props).every(([key, value]) => previous.props[key] === value)) continue;
        shapesToUpdate.push({ id: edgeShapeId, type: EDGE_SHAPE_TYPE, ...geometry });
      }

      if (shapesToUpdate.length > 0) {
        // Edge repositioning during a drag is a derived sync of the user's move,
        // not a separate undoable action — keep it off the undo stack.
        // Edge shapes are created with isLocked: true (see the edges effect
        // below) and tldraw's updateShapes silently DROPS partials for locked
        // shapes, so this batch must also opt out of the lock or the whole
        // reposition is a no-op and edges visibly lag behind the node.
        editor.run(() => editor.updateShapes(shapesToUpdate), {
          history: 'ignore',
          ignoreShapeLock: true,
        });
      }
    });

    // MUST return a teardown. tldraw treats onMount's return value as the
    // teardown and calls it on unmount — and React StrictMode deliberately
    // mounts, unmounts and remounts, invoking onMount twice for the same editor.
    // registerAfterChangeHandler/registerOperationCompleteHandler PUSH onto an
    // array and do not de-duplicate, so without this every handler above was
    // registered TWICE and ran twice per change.
    //
    // For the module branch that meant a frame's children were carried by the
    // frame's delta twice: dragging an expanded ConvNeXt-V2 frame by (197, -145)
    // moved its 7 children by (395, -291) and slid every one of them out of the
    // frame. Same doubling on the programmatic path during expand.
    // The read-only page may move nodes — layout is presentation state — but it may
    // not delete them: the graph is a projection of source this page cannot write, so
    // a removed card would be a lie about the file until the next projection. tldraw's
    // isReadonly used to cover this, and covered dragging with it.
    //
    // Only a USER gesture is refused. The sync effects below delete stray and replaced
    // shapes as a matter of course, and they bracket those writes with
    // isSyncingToEditor — the same discriminator the two handlers above already use.
    const disposeBeforeDelete = editor.sideEffects.registerBeforeDeleteHandler(
      'shape',
      (shape, source) => {
        if (source !== 'user' || isSyncingToEditor.current || !(localBrowserMode() || graphIsReadOnly(editor) || (shape.props as any).nodeData?.focusBoundary)) return;
        return false;
      },
    );

    return () => {
      disposeAfterChange();
      disposeOperationComplete();
      disposeBeforeDelete();
    };
  }, [setNodes]);

  /**
   * Sync nodes → tldraw shapes whenever nodes change.
   */
  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    // Echo-guard: while a shapes→nodes sync is mid-flight, skip pushing back to
    // the editor — but ONLY after the first sync. The very first sync
    // (lastNodesHash === '') must run even if a spurious from-editor op fires at
    // mount, otherwise a graph present at mount time never renders (there are no
    // shapes yet, so there is nothing to echo). If we bail, RETRY once the echo
    // flag clears — else an app-originated change that landed inside the echo
    // window (e.g. group-as-module fires right after a shift-select) is dropped
    // forever, because [nodes] won't change again to re-trigger this effect.
    if (isSyncingFromEditor.current && lastNodesHash.current !== '') {
      const t = window.setTimeout(() => setResyncTick((v) => v + 1), 60);
      return () => window.clearTimeout(t);
    }

    // Geometry hash only; data is compared by object identity below. Serializing
    // every node's rich data (ports, metrics, evidence) on every sync — once per
    // drag frame — dominated the work for a large graph and bought nothing: a
    // card reacts to a data change exactly when the node's data object changes.
    const previousData = lastNodeDataRef.current;
    let dataChanged = nodes.length !== previousData.size;
    if (!dataChanged) {
      for (const n of nodes) if (previousData.get(n.id) !== n.data) { dataChanged = true; break; }
    }
    const hash = nodes.map((n) => `${n.id}:${n.position.x}:${n.position.y}:${n.parentNode}:${n.type}:${n.hidden}:${n.style?.width}:${n.style?.height}`).join('|');
    if (!dataChanged && hash === lastNodesHash.current) { synced.current?.(sourceNodes); return; }
    lastNodesHash.current = hash;
    for (const n of nodes) previousData.set(n.id, n.data);
    // Evidence can change without geometry (for example a new profile batch).
    // Publish the same current nodes used for cards, independently of edge caching.

    isSyncingToEditor.current = true;
    try {
      const existingShapes = new Map(
        editor.getCurrentPageShapes()
          .filter((s) => isModelShape(s))
          .map((s) => [s.id, s])
      );

      const nodeShapeIds = new Set<TLShapeId>();
      const shapesToCreate: any[] = [];
      const shapesToUpdate: any[] = [];
      // Shape ids whose node changed TYPE: deleted and rebuilt in the same
      // run, because tldraw will not change a shape's type in place.
      const shapesToRecreate: string[] = [];

      // A module child's position is stored relative to its module; shapes are
      // absolute. Resolve here so an expanded module's layers render INSIDE the
      // frame instead of scattering toward the page origin.
      const nodeById = new Map(nodes.map((node) => [node.id, node] as const));
      const absoluteCache = new Map<string, { x: number; y: number }>();
      const resolving = new Set<string>();
      const absolutePositionOf = (node: ModelNode) => {
        const cached = absoluteCache.get(node.id);
        if (cached) return cached;
        // Parent cycles are invalid, but a malformed imported projection must
        // not recurse forever and freeze the browser. Keep the node's own
        // position as the finite fallback; the projection diagnostic remains
        // responsible for explaining the bad boundary to the user.
        if (resolving.has(node.id)) return node.position;
        resolving.add(node.id);
        let result = node.position;
        if (node.parentNode) {
          const parent = nodeById.get(node.parentNode as string);
          if (parent) {
            const origin = absolutePositionOf(parent);
            result = { x: origin.x + node.position.x, y: origin.y + node.position.y };
          }
        }
        resolving.delete(node.id);
        absoluteCache.set(node.id, result);
        return result;
      };

      for (const node of nodes) {
        if (node.hidden) continue;

        const shapeId = nodeIdToShapeId(node.id);
        nodeShapeIds.add(shapeId);
        const existing = existingShapes.get(shapeId);

        const tldrawType = rfTypeToTldrawType(node.type || 'layerNode');
        const isModule = node.type === 'moduleNode';

        const pos = absolutePositionOf(node);
        // A module's footprint changes every time it expands or collapses (the
        // reconciler rewrites style.width/height), so w/h belong in the UPDATE
        // partial too — set only at create time, an expanded module keeps its
        // collapsed frame and its children render outside it.
        const displaySize = presentation.sizes.get(node.id) ?? nodeCardSize(node);
        const moduleSize = isModule ? displaySize : null;

        // A tldraw shape's TYPE is immutable: `updateShape` keeps the original
        // ShapeUtil, so a node whose type flips goes on rendering as whatever it
        // was first created as. That happens routinely once a whole project is on
        // the canvas — an opaque `custom_code` card becomes a `moduleNode` frame
        // the moment it gains children — and the card would never turn into a
        // frame until the user collapsed its parent and reopened it. Recreate.
        const typeChanged = !!existing && (existing as any).type !== tldrawType;
        if (typeChanged) {
          shapesToRecreate.push(shapeId);
        }

        if (existing && !typeChanged) {
          // One partial per shape: tldraw derives every partial in a batch from
          // the original store record. A separate props update for this ID would
          // restore the old position. Keep geometry and data atomic, including
          // pure module resizes and card heights changed by measured shape rows.
          //
          // Only push a partial when something actually moved or changed. A
          // whole-graph rewrite per sync is what made selection and drag slow:
          // updating every shape's props re-runs every card.
          const current = existing as any;
          const desiredH = displaySize.h;
          const moved = Math.abs(current.x - pos.x) > 0.01 || Math.abs(current.y - pos.y) > 0.01;
          const resized = isModule
            ? Math.abs((current.props?.w ?? 0) - moduleSize!.w) > 0.5 || Math.abs((current.props?.h ?? 0) - moduleSize!.h) > 0.5
            : Math.abs((current.props?.h ?? 0) - desiredH) > 0.5 || Math.abs((current.props?.w ?? 0) - displaySize.w) > 0.5;
          const dataChanged = current.props?.nodeData !== node.data;
          if (moved || resized || dataChanged) {
            shapesToUpdate.push({
              id: shapeId,
              type: tldrawType as any,
              x: pos.x,
              y: pos.y,
              props: {
                nodeData: node.data,
                ...(moduleSize || {}),
                ...(isModule ? {} : { h: desiredH, w: displaySize.w }),
              },
            } as any);
          }
        } else {
          // Create new shape
          shapesToCreate.push({
            id: shapeId,
            type: tldrawType,
            x: pos.x,
            y: pos.y,
            props: {
              w: displaySize.w,
              // Match the card's REAL rendered height, so the selection box,
              // resize handles and hit area cover the whole card rather than
              // just its top 120px.
              h: displaySize.h,
              nodeData: node.data,
            },
          });
        }
      }

      // Find shapes to delete (nodes that were removed)
      const shapesToDelete = Array.from(existingShapes.keys())
        .filter((id) => !nodeShapeIds.has(id));

      withProjectionWriteAccess(editor, () => {
        editor.run(() => {
          if (shapesToDelete.length > 0) {
            editor.deleteShapes(shapesToDelete);
          }
          // Before the creates, so a retyped node's old shape is gone by the time
          // its replacement is made under the same id.
          if (shapesToRecreate.length > 0) {
            editor.deleteShapes(shapesToRecreate as any);
          }
          if (shapesToCreate.length > 0) {
            editor.createShapes(shapesToCreate);
          }
          if (shapesToUpdate.length > 0) {
            editor.updateShapes(shapesToUpdate);
          }
        }, { history: 'ignore' });
      });
      synced.current?.(sourceNodes);
    } finally {
      setTimeout(() => {
        isSyncingToEditor.current = false;
      }, 0);
    }
  }, [nodes, sourceNodes, presentation.sizes, editorReady, resyncTick]);

  /**
   * Sync edges → tldraw model-edge shapes whenever edges or node positions change.
   */
  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    // See the nodes effect: allow the first sync through, and RETRY (don't drop)
    // when a from-editor op is mid-flight so app-originated edge changes aren't
    // permanently swallowed.
    if (isSyncingFromEditor.current && lastEdgesHash.current !== '') {
      const t = window.setTimeout(() => setResyncTick((v) => v + 1), 60);
      return () => window.clearTimeout(t);
    }

    const visibleEdges = edges.filter((edge) => !edge.hidden);
    const renderableEdges = visibleEdges.filter((edge) => (
      editor.getShape(nodeIdToShapeId(edge.source)) &&
      editor.getShape(nodeIdToShapeId(edge.target))
    ));

    // The height belongs in this hash, not just the position. An edge is anchored
    // at `shape.y + shape.h / 2`, and a card's h GROWS when shape inference adds
    // its In/Out rows — so a hash of positions alone left every edge attached
    // where the card's middle used to be, ~8px above the ports it is drawn
    // between. Invisible while nothing ever inferred shapes without being asked;
    // permanent once inference became automatic. Read the h off the shape rather
    // than recomputing it: it is the same number the endpoint maths below uses,
    // and it is already fresh — the nodes effect is declared first, so its
    // updateShapes call has applied by the time this runs. Include width and
    // absolute position too: resizing or reparenting a module can move anchors
    // without changing a node's relative position or height.
    const hash = `${direction}-` +
      renderableEdges.map((e) => `${e.id}:${e.source}:${e.target}:${e.sourceHandle ?? e.data?.sourceHandle}:${e.targetHandle ?? e.data?.targetHandle}:${e.hidden}:${e.data?.diffStatus}:${e.data?.role}:${JSON.stringify(e.data?.originalEdgeIds)}:${e.data?._relationship}`).join('|') +
      '-' +
      nodes.map((n) => {
        const shape = editor.getShape(nodeIdToShapeId(n.id)) as any;
        return `${n.id}:${n.parentNode ?? n.data.parentId}:${shape?.x}:${shape?.y}:${shape?.props?.w}:${shape?.props?.h}:${(n.data.captureDisplayPorts ?? n.data.capturePorts ?? []).map(p => p.id).join(',')}`;
      }).join(',');
    if (hash === lastEdgesHash.current) return;
    lastEdgesHash.current = hash;
    updateFocusConnections(editor, renderableEdges.map(edge => ({ id: edge.id, source: edge.source, target: edge.target,
      role: edge.data?.role, count: edge.data?.originalEdgeIds?.length ?? 1, originals: edge.data?.originalEdgeIds, relationship: edge.data?._relationship })), nodes);
    // Which side each card's edges leave and enter, published for the port dots.
    // Same sync point as focus; recomputed only on a real graph change.
    const router = createEdgeRouter(nodes, renderableEdges, direction);
    routerRef.current = router;
    portSides.set(router.sides);
    const boxFor = (id: string) => {
      const shape = editor.getShape(nodeIdToShapeId(id)) as any;
      return shape ? shapeBox(shape) : undefined;
    };

    isSyncingToEditor.current = true;
    try {
      // Diff instead of delete-all/recreate-all. The old path destroyed every
      // edge and issued one createShape per edge on every sync — including once
      // per drag frame — so a 500-edge graph churned its whole edge layer to move
      // two endpoints. Geometry and the named props are compared per edge; only
      // the edges that changed, appeared or vanished are written.
      const existingEdges = editor.getCurrentPageShapes()
        .filter((s) => s.type === EDGE_SHAPE_TYPE || s.type === 'arrow');
      const existingById = new Map(existingEdges.map((shape) => [shape.id as string, shape as any]));
      const desiredIds = new Set<string>();
      const desired: Array<{ id: TLShapeId; x: number; y: number; props: Record<string, unknown> }> = [];
      for (const edge of renderableEdges) {
        const edgeShapeId = edgeIdToShapeId(edge);
        const route = router.route(edge, boxFor);
        if (!route) continue;
        const geometry = routedShapeGeometry(route);
        desiredIds.add(edgeShapeId as string);
        desired.push({ id: edgeShapeId, x: geometry.x, y: geometry.y, props: {
          ...geometry.props,
          diffStatus: edge.data?.diffStatus,
          // The only prop that names the edge. Everything else here is pure
          // geometry, so a click on the curve could not report what it hit.
          edgeId: edge.id,
          sourceId: edge.source,
          targetId: edge.target,
          role: edge.data?.role,
          connectionCount: Math.max(1, edge.data?.originalEdgeIds?.length ?? 1),
          // What the project document says this edge means. The shape draws a
          // `builds` quietly and a `feeds` lit, so the pipeline reads out of
          // the call graph instead of through it.
          relationship: (edge.data as { _relationship?: string } | undefined)?._relationship,
        } });
      }

      const toCreate = desired.filter((edge) => !existingById.has(edge.id as string));
      const toUpdate: any[] = [];
      const numeric = ['w', 'h', 'startX', 'startY', 'endX', 'endY'] as const;
      for (const edge of desired) {
        const got = existingById.get(edge.id as string);
        if (!got) continue;
        const geometrySame = Math.abs(got.x - edge.x) < 0.01 && Math.abs(got.y - edge.y) < 0.01
          && numeric.every((key) => Math.abs((got.props?.[key] ?? 0) - (edge.props[key] as number)) < 0.01);
        const metaSame = got.props?.diffStatus === edge.props.diffStatus && got.props?.edgeId === edge.props.edgeId
          && got.props?.sourceId === edge.props.sourceId && got.props?.targetId === edge.props.targetId
          && got.props?.role === edge.props.role && got.props?.connectionCount === edge.props.connectionCount
          && got.props?.relationship === edge.props.relationship
          && got.props?.startSide === edge.props.startSide && got.props?.endSide === edge.props.endSide
          && got.props?.curveOffset === edge.props.curveOffset;
        if (!geometrySame || !metaSame) toUpdate.push({ id: edge.id, type: EDGE_SHAPE_TYPE as any, x: edge.x, y: edge.y, props: edge.props });
      }
      const toDelete = existingEdges.filter((shape) => !desiredIds.has(shape.id as string)).map((shape) => shape.id);

      withProjectionWriteAccess(editor, () => editor.run(() => {
        if (toDelete.length > 0) {
          for (const id of toDelete) {
            const shape = editor.getShape(id) as any;
            if (shape?.isLocked) editor.updateShape({ id, type: shape.type, isLocked: false } as any);
          }
          editor.deleteShapes(toDelete);
        }
        if (toCreate.length > 0) {
          editor.createShapes(toCreate.map((edge) => ({
            id: edge.id, type: EDGE_SHAPE_TYPE as any, x: edge.x, y: edge.y, isLocked: true, props: edge.props as any,
          })));
        }
        if (toUpdate.length > 0) editor.updateShapes(toUpdate);
      }, { history: 'ignore', ignoreShapeLock: true }));

      // Z-order only changes when the edge set does; a geometry update keeps the
      // ordering it already had.
      if (toCreate.length > 0 || toDelete.length > 0) {
        try {
          withProjectionWriteAccess(editor, () => editor.run(() => {
            // Edges are locked against user edits, including ordinary reordering.
            // Keep them below cards but above opaque expanded-module surfaces.
            editor.sendToBack(toCreate.map((edge) => edge.id));
            const byId = new Map(nodes.map(node => [node.id, node]));
            const depth = (node: ModelNode) => {
              const seen = new Set<string>();
              let parent = node.data.parentId || node.parentNode;
              while (parent && !seen.has(parent)) {
                seen.add(parent);
                const ancestor = byId.get(parent);
                parent = ancestor?.data.parentId || ancestor?.parentNode;
              }
              return seen.size;
            };
            nodes.filter(node => node.type === 'moduleNode' && node.data.expanded && !node.hidden)
              .sort((a, b) => depth(b) - depth(a))
              .forEach(node => editor.sendToBack([nodeIdToShapeId(node.id)]));
          }, { history: 'ignore', ignoreShapeLock: true }));
        } catch (e) {
          // Ordering is cosmetic; stale/deleted edge records should not break sync.
        }
      }
    } finally {
      setTimeout(() => {
        isSyncingToEditor.current = false;
      }, 0);
    }
  }, [edges, nodes, direction, editorReady, resyncTick]);

  // Keep the shape id -> node id map in step with the graph.
  useEffect(() => {
    const map = new Map<string, string>();
    nodes.forEach((node) => map.set(nodeIdToShapeId(node.id) as string, node.id));
    nodeIdForShapeRef.current = map;
  }, [nodes]);

  /**
   * Handle selection changes from tldraw.
   */
  const handleSelectionChange = useCallback((selectedIds: TLShapeId[]) => {
    const editor = editorRef.current;
    const nodeIds = selectedIds
      .filter((shapeId) => isModelShape(editor?.getShape(shapeId)))
      .map(shapeId => nodeIdForShapeRef.current.get(shapeId as string))
      .filter((nodeId): nodeId is string => nodeId !== undefined && !nodesRef.current.find(node => node.id === nodeId)?.data.focusBoundary);
    onSelectionChange?.(nodeIds);
  }, [onSelectionChange]);

  /**
   * Handle shape click from tldraw. Selection itself is handled by
   * handleSelectionChange; this only reports a real click (TldrawCanvas filters
   * out drags) on a model node, so the page can open its inspector without a
   * mere drag doing so. Non-node shapes (e.g. the chat card) are ignored.
   */
  const handleShapeClick = useCallback((shapeId: TLShapeId) => {
    if (isModelShape(editorRef.current?.getShape(shapeId))) onNodeClick?.();
  }, [onNodeClick]);

  /**
   * Handle a double-click on a shape: activation, not selection.
   *
   * The node id is resolved through the forward map rather than by un-sanitising
   * the shape id, because that sanitisation is lossy — `blocks.0.l1` and
   * `blocks_0_l1` become the same shape id under the retired lossy encoding — and a
   * node id that comes back wrong points the editor at the wrong line.
   */
  const handleShapeDoubleClick = useCallback((shapeId: TLShapeId) => {
    if (!isModelShape(editorRef.current?.getShape(shapeId))) return;
    const nodeId = nodeIdForShapeRef.current.get(shapeId as string);
    if (nodeId) onNodeActivate?.(nodeId);
  }, [onNodeActivate]);

  /**
   * Handle canvas click (deselect).
   */
  const handleCanvasClick = useCallback(() => {
    // Let tldraw handle the selection via handleSelectionChange
  }, []);

  /**
   * Programmatically fit the canvas to content.
   */
  const fitView = useCallback(() => {
    editorRef.current?.zoomToFit({ animation: { duration: cameraDuration() } });
  }, []);

  /**
   * Reveal one node: select its shape in tldraw AND pan to it. Setting React
   * data.isSelected alone did neither — the shape utils read selection from
   * editor.getSelectedShapeIds(), and nothing moved the camera — so quick-jump,
   * the shape-table links and the diagnostics jumps looked like no-ops on any
   * off-screen node. centerOnPoint (not zoomToBounds) pans without changing the
   * user's zoom, and cameraDuration already honours prefers-reduced-motion.
   */
  const focusPendingNode = useCallback(() => {
    const editor = editorRef.current;
    const nodeId = pendingFocusNodeRef.current;
    if (!editor || !nodeId) return;
    if (!nodesRef.current.some((node) => node.id === nodeId)) {
      pendingFocusNodeRef.current = null;
      return;
    }
    const shapeId = nodeIdToShapeId(nodeId);
    if (!editor.getShape(shapeId)) return;
    pendingFocusNodeRef.current = null;
    editor.select(shapeId);
    const bounds = editor.getShapePageBounds(shapeId);
    if (bounds) editor.centerOnPoint(bounds.center, { animation: { duration: cameraDuration(300) } });
  }, []);

  // A jump can expand a collapsed ancestor. Select only after the node sync
  // above has created its child shape, including an echo-guard retry.
  useEffect(() => {
    focusPendingNode();
  }, [nodes, editorReady, resyncTick, focusPendingNode]);

  const focusNode = useCallback((nodeId: string) => {
    pendingFocusNodeRef.current = nodeId;
    focusPendingNode();
  }, [focusPendingNode]);

  return {
    editorRef,
    editorReady,
    handleEditorReady,
    handleSelectionChange,
    handleShapeClick,
    handleShapeDoubleClick,
    handleCanvasClick,
    fitView,
    focusNode,
  };
}
