/**
 * TldrawCanvas — drop-in replacement for the ReactFlow canvas.
 *
 * This component renders a tldraw editor with only our custom ML node shapes.
 * All default tldraw drawing tools and UI are hidden. The canvas matches our
 * existing dark theme via CSS custom properties.
 *
 * Usage:
 *   <TldrawCanvas
 *     onEditorReady={(editor) => storeEditorRef(editor)}
 *     onSelectionChange={(ids) => handleSelectionChange(ids)}
 *   />
 */
import React, { useCallback, useEffect, useMemo, useRef } from 'react';
import {
  Tldraw,
  Editor,
  useEditor,
  TLEventInfo,
  TLShapeId,
  track,
} from 'tldraw';
import 'tldraw/tldraw.css';
import './tldraw-theme.css';
import './graphAppearance.css';
import './graphConnections.css';

import { LayerShapeUtil } from './LayerShapeUtil';
import { InputShapeUtil } from './InputShapeUtil';
import { OutputShapeUtil } from './OutputShapeUtil';
import { ModuleShapeUtil } from './ModuleShapeUtil';
import { ModelEdgeShapeUtil } from './ModelEdgeShapeUtil';
import { CanvasMinimap } from './CanvasMinimap';
import { ConnectionLayer } from './ConnectionLayer';
import { ConnectionGraph, ConnectionGraphContext, cancelConnection } from './connectionDrag';
import { GraphEdgeLike } from './connectivity';
import { EDGE_HIT_PX, findEdgeAtPoint, findEdgeWithMidpoint, selectedEdgeId } from './edgeSelection';
import { doubleClickedShapeId } from './doubleClickTarget';
import { EdgeInsertLayer } from './EdgeInsertLayer';
import { LocalBrowserContext } from '../ide/localBrowserContext';
import { NodeParametersPopover } from './NodeParametersPopover';
import { GraphFocusOverlay, GraphNavigationContext } from './GraphFocusOverlay';
import { setTraceDirection } from './graphFocus';
import { cameraDuration } from './reducedMotion';
import { registerGraphAccess } from './graphAccess';
import { getCanvasTheme, onCanvasThemeChange } from '../styles/canvasTheme';
import { isModelShape } from './canvasShapePolicy';
import { closeNodeParameters, openNodeParameters, parameterTarget } from './nodeParameters';
import {
  EdgeInsertChannel,
  EdgeInsertContext,
  cancelEdgeInsert,
  hoveredEdge as hoveredEdgeAtom,
} from './edgeInsert';

/* ── Props ───────────────────────────────────────────────── */

export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface TldrawCanvasProps {
  /** Move the overview clear of the canvas Ask panel. */
  sidePanelOpen?: boolean;
  /** Keep navigation/selection available while tldraw refuses shape edits. */
  readOnly?: boolean;
  /** Standalone viewers supply local assets instead of the default CDN. */
  assetUrls?: React.ComponentProps<typeof Tldraw>['assetUrls'];
  /** Surface navigation placed above search in the shared screen-space stack. */
  navigation?: React.ReactNode;
  /** Called once when the tldraw editor is fully mounted. */
  /** May return a teardown; it is handed to tldraw's onMount so StrictMode's
   *  remount unregisters whatever the callback registered. */
  onEditorReady?: (editor: Editor) => (() => void) | void;
  /** Called whenever the user selection changes. */
  onSelectionChange?: (selectedIds: TLShapeId[]) => void;
  onToolChange?: (toolId: string) => void;
  /** Called when a shape is clicked. */
  onShapeClick?: (shapeId: TLShapeId) => void;
  /** Called when a shape is double-clicked. */
  onShapeDoubleClick?: (shapeId: TLShapeId) => void;
  /** Called when the canvas background is clicked (deselect). */
  onCanvasClick?: () => void;
  /** External palette drag/drop handlers. */
  onExternalDrop?: (event: React.DragEvent<HTMLDivElement>) => void;
  onExternalDragOver?: (event: React.DragEvent<HTMLDivElement>) => void;
  /** Live logical edges — hidden ones included, synthetic frame edges excluded —
   *  so the connect gesture can reject duplicates and cycles. */
  connectEdges?: readonly GraphEdgeLike[];
  /** Commit a connection the gesture already validated. */
  onConnect?: (source: string, target: string) => void;
  /** Create a layer and splice it into an existing edge, replacing that edge. */
  onInsertOnEdge?: (edgeId: string, layerTypeId: string) => void;
  /** Optional children to render inside the tldraw canvas context. */
  children?: React.ReactNode;
}

/* ── Custom shape utilities ──────────────────────────────── */

const customShapeUtils = [
  LayerShapeUtil,
  InputShapeUtil,
  OutputShapeUtil,
  ModuleShapeUtil,
  ModelEdgeShapeUtil,
];

/* ── Hide ALL default tldraw UI ──────────────────────────── */

const hiddenComponents = {
  MainMenu: null,
  StylePanel: null,
  NavigationPanel: null,
  PageMenu: null,
  HelpMenu: null,
  ActionsMenu: null,
  ContextMenu: null,
  QuickActions: null,
  MenuPanel: null,
  TopPanel: null,
  SharePanel: null,
  DebugPanel: null,
  DebugMenu: null,
  HelperButtons: null,
  Minimap: null,
  ZoomMenu: null,
  KeyboardShortcutsDialog: null,
  RichTextToolbar: null,
  // tldraw's own Select/Hand toolbar paints bottom-centre, on top of our
  // .ui2-builder-toolbar, and its Select duplicates ours. Hand stays reachable
  // via `h` and space-drag.
  Toolbar: null,
};

/* ── Selection watcher (runs inside <Tldraw> tree) ───────── */

const ToolWatcher = track(function ToolWatcher({ onChange }: { onChange?: (toolId: string) => void }) {
  const editor = useEditor();
  const tool = editor.getCurrentToolId();
  useEffect(() => { onChange?.(tool); }, [onChange, tool]);
  return null;
});

const SelectionWatcher = track(function SelectionWatcher({
  onSelectionChange,
  onShapeClick,
  onShapeDoubleClick,
  onCanvasClick,
  canEditParameters,
}: Pick<TldrawCanvasProps, 'onSelectionChange' | 'onShapeClick' | 'onShapeDoubleClick' | 'onCanvasClick'> & {
  /** False on the read-only page, where source is not the canvas's to change. */
  canEditParameters: boolean;
}) {
  const editor = useEditor();

  // Watch selection changes reactively
  const selectedIds = editor.getSelectedShapeIds();

  useEffect(() => {
    onSelectionChange?.(selectedIds);
    if (selectedIds.length) selectedEdgeId.set(null);
  }, [selectedIds, onSelectionChange]);

  useEffect(() => { setTraceDirection(editor, 'local'); }, [editor, selectedIds]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest('input, textarea, select, button, summary, [role="region"][tabindex], [contenteditable="true"]') || event.ctrlKey || event.metaKey || event.altKey) return;
      if (event.key === 'Escape') {
        setTraceDirection(editor, 'local'); selectedEdgeId.set(null); closeNodeParameters();
      } else if (event.key === 'Enter' && editor.getSelectedShapeIds().length === 1) {
        event.preventDefault(); event.stopPropagation();
        onShapeDoubleClick?.(editor.getSelectedShapeIds()[0]);
      } else if (event.key === '[' || event.key === ']') {
        const shapes = editor.getCurrentPageShapes().filter(isModelShape).sort((a, b) => a.x - b.x || a.y - b.y);
        if (!shapes.length) return;
        const current = shapes.findIndex(shape => shape.id === editor.getSelectedShapeIds()[0]);
        const index = current < 0 ? 0 : (current + (event.key === ']' ? 1 : -1) + shapes.length) % shapes.length;
        event.preventDefault(); event.stopPropagation();
        editor.select(shapes[index].id);
        const bounds = editor.getShapePageBounds(shapes[index].id);
        if (bounds) editor.centerOnPoint(bounds.center, { animation: { duration: cameraDuration(180) } });
      }
    };
    const container = editor.getContainer();
    container.addEventListener('keydown', onKey, true);
    return () => container.removeEventListener('keydown', onKey, true);
  }, [editor, onShapeDoubleClick]);

  // The parameter editor belongs to one card. Selecting anything else — another
  // node, several, or nothing — takes it away, so it never outlives the node it
  // is aimed at or sits over a selection it does not describe.
  useEffect(() => {
    const target = parameterTarget.get();
    if (target && (selectedIds.length !== 1 || selectedIds[0] !== target)) closeNodeParameters();
  }, [selectedIds]);

  // Report a REAL click on a shape — not the pointer-down that starts a drag.
  // tldraw's SelectTool selects on pointer-DOWN (PointingShape.onEnter), before
  // it can know whether the gesture becomes a drag, so selection alone cannot
  // mean "clicked". The tool state can: PointingShape leaves for 'translating'
  // as soon as the drag threshold is crossed, so a pointer_up still in
  // 'select.pointing_shape' is a press that never moved. 'before-event' runs
  // ahead of the state chart's own pointer_up transition.
  useEffect(() => {
    const handleBeforeEvent = (info: TLEventInfo) => {
      if (info.name !== 'pointer_up' || editor.getPath() !== 'select.pointing_shape') return;
      const selected = editor.getSelectedShapeIds();
      if (selected.length !== 1) return;
      onShapeClick?.(selected[0]);
      // Parameters open from THIS gesture rather than from the selection, which is
      // what the docked Inspector got wrong: tldraw selects on pointer-down, so a
      // selection-driven panel opened on every drag. See nodeParameters.ts.
      if (canEditParameters && isModelShape(editor.getShape(selected[0]))) {
        openNodeParameters(selected[0]);
      }
    };
    editor.on('before-event', handleBeforeEvent);
    return () => {
      editor.off('before-event', handleBeforeEvent);
    };
  }, [editor, onShapeClick, canEditParameters]);

  // Report a finished double-click on a shape. tldraw's click detector reports it
  // on the canvas without naming the shape; doubleClickedShapeId resolves it the
  // way the select tool does.
  useEffect(() => {
    const handleEvent = (info: TLEventInfo) => {
      const shapeId = doubleClickedShapeId(editor, info);
      if (shapeId) onShapeDoubleClick?.(shapeId);
    };
    editor.on('event', handleEvent);
    return () => {
      editor.off('event', handleEvent);
    };
  }, [editor, onShapeDoubleClick]);

  // Click an edge to select it, react-flow style. An edge cannot be selected by
  // tldraw (locked, pointerEvents:'none', empty indicator path), so clicking one
  // arrives as a CANVAS click that happens to land near a curve — hence the same
  // 'pointer_up while still pointing_canvas' test the shape-click detector uses.
  // 'before-event' fires before inputs are updated, so the point comes off the
  // event rather than editor.inputs.
  useEffect(() => {
    let hoveredEdge: string | null = null;

    const hitTest = (info: TLEventInfo) => {
      const point = (info as any).point;
      if (!point) return null;
      return findEdgeAtPoint(
        editor,
        editor.screenToPage(point),
        EDGE_HIT_PX / editor.getZoomLevel(),
      );
    };

    const handleBeforeEvent = (info: TLEventInfo) => {
      if (info.name === 'pointer_move') {
        // Bail before touching the cursor unless the editor is idle, or this
        // fights every drag, brush and tool that owns the cursor meanwhile.
        if (editor.getPath() !== 'select.idle') return;
        const point = (info as any).point;
        // The midpoint comes along so the insert affordance has somewhere to sit;
        // it is the curve's t=0.5 point, not an approximation.
        const found = point
          ? findEdgeWithMidpoint(
              editor,
              editor.screenToPage(point),
              EDGE_HIT_PX / editor.getZoomLevel(),
            )
          : null;
        const hit = found?.edgeId ?? null;
        if (hit === hoveredEdge) return;
        hoveredEdge = hit;
        // Written through an atom, never a re-render of the shape layer: a
        // re-render on hover detaches the element under the cursor and the `+`
        // becomes unclickable.
        hoveredEdgeAtom.set(found);
        // 'pointer' is in TL_CURSOR_TYPES. 'crosshair' and 'not-allowed' are
        // NOT — setCursor throws on those.
        editor.setCursor({ type: hit ? 'pointer' : 'default' });
        return;
      }
      if (info.name !== 'pointer_up' || editor.getPath() !== 'select.pointing_canvas') return;
      selectedEdgeId.set(hitTest(info));
    };

    editor.on('before-event', handleBeforeEvent);
    return () => {
      editor.off('before-event', handleBeforeEvent);
    };
  }, [editor]);

  return null; // This component is invisible — it only watches state
});

/* ── Main canvas component ───────────────────────────────── */

export const CanvasAssetContext = React.createContext<React.ComponentProps<typeof Tldraw>['assetUrls']>(undefined);

/**
 * tldraw takes ONE component per slot, and both of the canvas's screen-space
 * affordances live in this one: the edge-insert picker and the on-node parameter
 * editor. Declared at module scope, not inline — a new element type on every
 * render would remount whichever popover is open on every camera tick.
 */
const CanvasOverlays: React.FC = () => (
  <>
    <EdgeInsertLayer />
    <NodeParametersPopover />
  </>
);

/** Decorative page-space dots follow the camera without enabling grid snapping. */
const CanvasBackground = track(function CanvasBackground() {
  const { x, y, z } = useEditor().getCamera();
  let spacing = 20 * z;
  // Keep page-aligned dots at overview zoom by showing every second grid point.
  while (spacing < 12) spacing *= 2;
  const offset = (20 * z - spacing) / 2;
  return <div className="tl-background"><div className="canvas-dot-grid" aria-hidden="true" style={{
    backgroundSize: `${spacing}px ${spacing}px`, backgroundPosition: `${x * z + offset}px ${y * z + offset}px`,
  }} /></div>;
});

export const TldrawCanvas: React.FC<TldrawCanvasProps> = ({
  sidePanelOpen = false,
  readOnly = false,
  assetUrls,
  navigation,
  onEditorReady,
  onSelectionChange,
  onToolChange,
  onShapeClick,
  onShapeDoubleClick,
  onCanvasClick,
  onExternalDrop,
  onExternalDragOver,
  connectEdges,
  onConnect,
  onInsertOnEdge,
  children,
}) => {
  const local = React.useContext(LocalBrowserContext);
  const bundledAssets = React.useContext(CanvasAssetContext);
  // A REF, not a context value: ports read the graph only at pointerdown, so
  // passing it by value would re-render every port on every graph edit.
  const connectionGraph = useRef<ConnectionGraph>({ edges: [], commit: () => {} });
  connectionGraph.current = {
    edges: connectEdges || [],
    commit: readOnly ? () => {} : onConnect || (() => {}),
  };

  // A gesture must not outlive the canvas — its listeners are on window.
  useEffect(() => cancelConnection, []);
  useEffect(() => cancelEdgeInsert, []);
  useEffect(() => closeNodeParameters, []);

  const edgeInsert = useRef<EdgeInsertChannel>({ insert: () => {} });
  edgeInsert.current = { insert: readOnly ? () => {} : onInsertOnEdge || (() => {}) };

  // TWO different slots, deliberately.
  //
  // OnTheCanvas mounts inside .tl-html-layer and BEFORE ShapesLayer, so it
  // inherits the camera transform (page coordinates, free) but is painted
  // UNDERNEATH every card. That is right for the connection ghost, which should
  // pass behind the nodes it connects.
  //
  // It is wrong for an interactive popover: the insert picker used to live here and
  // vanished behind whatever card it overlapped. InFrontOfTheCanvas renders after
  // the shapes and outside the transform, so that layer works in SCREEN space and
  // projects its own points — and its wrapper calls editor.markEventAsHandled on
  // pointer down/up, so a click inside the picker cannot reach the canvas.
  const components = useMemo(
    () => ({
      ...hiddenComponents,
      Background: CanvasBackground,
      OnTheCanvas: ConnectionLayer,
      InFrontOfTheCanvas: CanvasOverlays,
    }),
    []
  );

  const readOnlyRef = useRef(readOnly);
  readOnlyRef.current = readOnly;

  const handleMount = useCallback(
    (editor: Editor) => {
      registerGraphAccess(editor, readOnlyRef);
      editor.user.updateUserPreferences({ colorScheme: getCanvasTheme() });
      const stopThemeUpdates = onCanvasThemeChange(theme => editor.user.updateUserPreferences({ colorScheme: theme }));
      // Deliberately false, never bound to the readOnly prop. Read-only on this page
      // means the SOURCE cannot change; canvas layout is presentation state, and the
      // read-only contract
      // itself keeps layout/fit available. tldraw's isReadonly is all-or-nothing, so
      // it took node dragging with it — the thing a reader does most on a graph they
      // are trying to understand, and the one useTldrawBridge already syncs back
      // (its operation-complete handler writes dragged positions into the nodes).
      // The one destructive gesture isReadonly uniquely covered is deletion, and
      // nothing else gated it: blockDrawingShortcuts below blocks tool letters, not
      // Delete. useTldrawBridge vetoes user deletes instead, where it can tell a
      // user gesture from its own programmatic cleanup.
      editor.updateInstanceState({ isReadonly: false }, { history: 'ignore' });

      // tldraw's skip-to-main-content blurs its own button and then focuses the
      // container on a 100ms timer — but only when a shape exists, so an empty
      // canvas leaves focus on BODY, and anything racing the timer can steal it
      // back. Land the promised focus ourselves once the timer has fired.
      const container = editor.getContainer();
      const ownerDoc = container.ownerDocument;
      const onSkipToMainContent = (event: MouseEvent) => {
        const target = event.target as HTMLElement | null;
        if (!target?.closest('.tl-skip-to-main-content')) return;
        window.setTimeout(() => {
          if (ownerDoc.activeElement !== container) container.focus();
        }, 180);
      };
      ownerDoc.addEventListener('click', onSkipToMainContent, true);

      // Return the callback's teardown so tldraw runs it on unmount. StrictMode
      // mounts → unmounts → remounts, so dropping this leaves everything
      // onEditorReady registered attached twice, and every side-effect handler
      // then fires twice per change.
      const teardown = onEditorReady?.(editor);
      return () => {
        stopThemeUpdates();
        ownerDoc.removeEventListener('click', onSkipToMainContent, true);
        if (typeof teardown === 'function') teardown();
      };
    },
    // No `readOnly`: the mount no longer branches on it. Leaving it here would
    // remount the editor whenever the flag changed, for nothing.
    [onEditorReady]
  );

  const blockDrawingShortcuts = useCallback((event: React.KeyboardEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement | null;
    const isTyping =
      target?.tagName === 'INPUT' ||
      target?.tagName === 'TEXTAREA' ||
      target?.isContentEditable;

    if (isTyping || event.metaKey || event.ctrlKey || event.altKey) return;

    const drawingShortcutKeys = new Set([
      'a', 'd', 'e', 'f', 'g', 'l', 'n', 'o', 'p', 'r', 't', 'v', 'x',
    ]);

    if (drawingShortcutKeys.has(event.key.toLowerCase())) {
      event.preventDefault();
    }
  }, []);

  const uiOverrides = useMemo<import('tldraw').TLUiOverrides>(() => ({
    tools(_editor, tools) {
      return {
        select: tools.select,
        hand: tools.hand,
      };
    }
  }), []);

  return (
    <div
      className="tldraw-canvas-wrapper"
      onKeyDownCapture={blockDrawingShortcuts}
      onDropCapture={readOnly ? undefined : onExternalDrop}
      onDragOverCapture={readOnly ? undefined : onExternalDragOver}
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
      }}
    >
      <ConnectionGraphContext.Provider value={connectionGraph}>
        <EdgeInsertContext.Provider value={edgeInsert}>
        <GraphNavigationContext.Provider value={navigation}>
        <Tldraw
          // The "Get a license for production" watermark is tldraw's unlicensed
          // notice, and it reads to a user as "this product is built on an unpaid
          // component". Buying the licence is a human decision, so this is the
          // plumbing for it and nothing more: with no key set, tldraw behaves
          // exactly as before and the watermark stays. Drop the purchased key into
          // REACT_APP_TLDRAW_LICENSE_KEY (canvas/.env.local locally, the build
          // environment in CI) and it goes away legitimately.
          //
          // The watermark must NOT be hidden any other way. tldraw's licence
          // forbids obscuring it, and BuilderPageV2's diagnostics drawer reserves
          // `pr-28` precisely so the bar's right-most controls stay clickable
          // underneath it rather than being covered up.
          licenseKey={process.env.REACT_APP_TLDRAW_LICENSE_KEY}
          assetUrls={assetUrls ?? bundledAssets}
          shapeUtils={customShapeUtils}
          components={components as any}
          overrides={uiOverrides}
          // Double-clicking makes tldraw create a `text` shape, which the bridge then
          // deletes as a stray — that create/delete cycle exceeds the store update depth.
          options={{ createTextOnCanvasDoubleClick: false, maxShapesPerPage: 100_000 }}
          onMount={handleMount}
        >
          <SelectionWatcher
            canEditParameters={!readOnly || !!local?.sourceEdits}
            onSelectionChange={onSelectionChange}
            onShapeClick={onShapeClick}
            onShapeDoubleClick={onShapeDoubleClick}
            onCanvasClick={onCanvasClick}
          />
          <ToolWatcher onChange={onToolChange} />
          <div className="canvas-navigation-overlays">
            <GraphFocusOverlay />
            <CanvasMinimap sidePanelOpen={sidePanelOpen} />
          </div>
          {children}
        </Tldraw>
        </GraphNavigationContext.Provider>
        </EdgeInsertContext.Provider>
      </ConnectionGraphContext.Provider>
    </div>
  );
};

export default TldrawCanvas;
