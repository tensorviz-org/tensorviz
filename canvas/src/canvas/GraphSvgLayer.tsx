import React from 'react';
import type { Editor } from 'tldraw';
import { graphToSvg, graphSvgBounds } from '../services/graphSvg';
import { ModelEdge, ModelNode } from '../types';
import { useCanvasTheme } from '../styles/canvasTheme';
import type { LayoutDirection } from './edgeAnchors';
import { NodeExplanationContext } from './NodeExplanationContext';

const PADDING = 24;

/** Minimum overlap as a fraction of the smaller of graph and viewport. */
const MIN_VISIBLE_FRACTION = 0.4;

/**
 * Clamp a camera so the graph cannot be panned off into empty space. The overview
 * has no tldraw shapes of its own, so without this the shared camera scrolls
 * forever. Overlap is capped by the viewport so large graphs remain pannable.
 */
export function clampOverviewCamera(
  camera: { x: number; y: number; z: number },
  viewport: { width: number; height: number },
  box: { x: number; y: number; w: number; h: number },
  visibleFraction = MIN_VISIBLE_FRACTION,
): { x: number; y: number; z: number } {
  const z = camera.z || 1;
  const visibleW = Math.min(box.w, viewport.width / z) * visibleFraction;
  const visibleH = Math.min(box.h, viewport.height / z) * visibleFraction;
  return {
    ...camera,
    // Keep an overlap at either edge without excluding the graph's endpoints.
    x: Math.min(Math.max(camera.x, visibleW - (box.x + box.w)), viewport.width / z - box.x - visibleW),
    y: Math.min(Math.max(camera.y, visibleH - (box.y + box.h)), viewport.height / z - box.y - visibleH),
  };
}

/**
 * The whole reconciled graph as ONE <img> backed by an SVG blob.
 *
 * This is the memory answer for a large expanded graph: tldraw keeps every card
 * mounted (its culling is CSS-only), but here the diagram is a single element.
 *
 * Gestures are NOT reimplemented. The layer takes no pointer events, so wheel,
 * trackpad, pinch, space-drag, middle-mouse and the hand tool all reach the
 * (shape-less but still mounted) tldraw canvas underneath, exactly as on the
 * interactive canvas. The SVG is transformed to follow tldraw's own camera, and
 * clicks are mapped back through the same camera for selection. Panning is
 * clamped so the graph cannot be scrolled entirely out of view.
 */
export default function GraphSvgLayer({ nodes, edges, direction, selectedIds, onSelect, onActivate, editor }: {
  nodes: ModelNode[];
  edges: ModelEdge[];
  direction?: LayoutDirection;
  selectedIds?: Set<string>;
  onSelect(id: string | null): void;
  onActivate?(id: string): void;
  editor: Editor | null;
}) {
  const explanations = React.useContext(NodeExplanationContext);
  const explanationOptions = React.useMemo(() => ({ explanationStates: explanations?.annotations ?? explanations?.states, annotationsVisible: explanations?.annotationsVisible }),
    [explanations?.annotations, explanations?.states, explanations?.annotationsVisible]);
  const bounds = React.useMemo(() => graphSvgBounds(nodes, explanationOptions), [nodes, explanationOptions]);
  const changedNodes = React.useMemo(() => new Map(nodes.filter(node => node.data.diffStatus).map(node => [node.id, node.data])), [nodes]);
  const theme = useCanvasTheme();
  const svg = React.useMemo(() => graphToSvg(nodes, edges, { theme, direction, ...explanationOptions }), [nodes, edges, theme, direction, explanationOptions]);
  const url = React.useMemo(() => (typeof URL.createObjectURL === 'function'
    ? URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }))
    : `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`), [svg]);
  React.useEffect(() => () => { if (url.startsWith('blob:')) URL.revokeObjectURL(url); }, [url]);

  const minX = bounds.length ? Math.min(...bounds.map((box) => box.x)) - PADDING : 0;
  const minY = bounds.length ? Math.min(...bounds.map((box) => box.y)) - PADDING : 0;
  const width = bounds.length ? Math.max(...bounds.map((box) => box.x + box.w)) - minX + PADDING : 1;
  const height = bounds.length ? Math.max(...bounds.map((box) => box.y + box.h)) - minY + PADDING : 1;
  const originalZoomSteps = React.useMemo(() => editor?.getCameraOptions().zoomSteps, [editor]);
  React.useEffect(() => () => {
    if (editor && originalZoomSteps) editor.setCameraOptions({ zoomSteps: originalZoomSteps });
  }, [editor, originalZoomSteps]);

  const containerRef = React.useRef<HTMLDivElement>(null);
  const boundsRef = React.useRef(bounds);
  boundsRef.current = bounds;
  const [camera, setCamera] = React.useState(() => editor?.getCamera() ?? { x: 0, y: 0, z: 1 });
  /** screenBounds − layer origin, which is normally zero but measured so a
   *  shifted/portaled layer still lines up with tldraw's viewport. */
  const [origin, setOrigin] = React.useState({ x: 0, y: 0 });

  const measure = React.useCallback(() => {
    const element = containerRef.current;
    if (!element || !editor) return;
    const rect = element.getBoundingClientRect();
    const screen = editor.getViewportScreenBounds();
    setOrigin({ x: screen.x - rect.x, y: screen.y - rect.y });
  }, [editor]);

  // Follow tldraw's camera. Camera lives in the session store, so this does not
  // wake on document edits. Out-of-range pans are pushed back through setCamera,
  // which re-enters this listener and settles on the clamped value.
  React.useEffect(() => {
    if (!editor || !originalZoomSteps) return;
    const box = { x: minX, y: minY, w: width, h: height };
    const update = () => {
      const next = editor.getCamera();
      const clamped = clampOverviewCamera(next, editor.getViewportScreenBounds(), box);
      if (Math.abs(clamped.x - next.x) > 0.5 || Math.abs(clamped.y - next.y) > 0.5) {
        editor.setCamera(clamped);
      }
      // Initial bounds and detail changes can clamp before the listener exists.
      // Always publish the actual corrected camera to the SVG transform too.
      setCamera(editor.getCamera());
    };
    const resize = () => {
      const viewport = editor.getViewportScreenBounds();
      // The default 5% minimum cannot fit tall captures. Add a smaller step
      // only for this SVG, keeping the editor's regular zoom steps on exit.
      const minimum = Math.min(originalZoomSteps[0],
        Math.max(1, viewport.width - 48) / width, Math.max(1, viewport.height - 48) / height);
      editor.setCameraOptions({ zoomSteps: minimum < originalZoomSteps[0]
        ? [minimum, ...originalZoomSteps] : originalZoomSteps });
      measure(); update();
    };
    resize();
    const unsubscribe = editor.store.listen(update, { scope: 'session' });
    editor.on('resize', resize);
    return () => { unsubscribe(); editor.off('resize', resize); };
  }, [editor, measure, minX, minY, width, height, originalZoomSteps]);

  // Selection: the click lands on tldraw, so map the point through the same
  // camera. A drag (pan/select) is not a click.
  React.useEffect(() => {
    if (!editor) return;
    let down: { x: number; y: number } | null = null;
    const onDown = (event: PointerEvent) => { if (event.button === 0) down = { x: event.clientX, y: event.clientY }; };
    const onUp = (event: PointerEvent) => {
      const start = down;
      down = null;
      if (!start || event.button !== 0) return;
      if (Math.hypot(event.clientX - start.x, event.clientY - start.y) > 4) return;
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect || event.clientX < rect.left || event.clientX > rect.right
        || event.clientY < rect.top || event.clientY > rect.bottom) return;
      const point = editor.screenToPage({ x: event.clientX, y: event.clientY });
      const hit = [...boundsRef.current].reverse()
        .find((box) => point.x >= box.x && point.x <= box.x + box.w
          && point.y >= box.y && point.y <= box.y + box.h);
      onSelect(hit ? hit.id : null);
    };
    const activate = (event: MouseEvent) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect || event.clientX < rect.left || event.clientX > rect.right
        || event.clientY < rect.top || event.clientY > rect.bottom) return;
      const point = editor.screenToPage({ x: event.clientX, y: event.clientY });
      const hit = [...boundsRef.current].reverse().find(box => point.x >= box.x && point.x <= box.x + box.w
        && point.y >= box.y && point.y <= box.y + box.h);
      if (hit) onActivate?.(hit.id);
    };
    window.addEventListener('pointerdown', onDown, true);
    window.addEventListener('pointerup', onUp, true);
    window.addEventListener('dblclick', activate, true);
    return () => {
      window.removeEventListener('pointerdown', onDown, true);
      window.removeEventListener('pointerup', onUp, true);
      window.removeEventListener('dblclick', activate, true);
    };
  }, [editor, onSelect, onActivate]);

  const z = camera.z || 1;
  const transform = `translate(${(minX + camera.x) * z + origin.x}px, ${(minY + camera.y) * z + origin.y}px) scale(${z})`;

  return <div ref={containerRef} data-testid="graph-svg-layer"
    style={{ position: 'absolute', inset: 0, overflow: 'hidden', background: 'var(--tv-canvas)', zIndex: 15,
      pointerEvents: 'none' }}>
    <div data-testid="graph-svg-content" style={{ position: 'absolute', top: 0, left: 0, width, height,
      transform, transformOrigin: '0 0' }}>
      <img src={url} width={width} height={height} draggable={false}
        style={{ display: 'block', pointerEvents: 'none', userSelect: 'none' }} alt="Expanded graph overview" />
      {bounds.filter(box => changedNodes.has(box.id)).map(box => {
        const data = changedNodes.get(box.id)!;
        const color = data.error || data.moduleDiagnosticStatus === 'error' || data.moduleStatus === 'error' ? 'var(--tv-error)'
          : data.diffStatus === 'added' ? 'var(--tv-added)' : data.diffStatus === 'removed' ? 'var(--tv-removed)'
            : data.diffStatus === 'rewired' ? 'var(--tv-inputs-changed)' : 'var(--tv-modified)';
        return <div key={`diff:${box.id}`} data-diff-node-id={box.id}
          style={{ position: 'absolute', left: box.x - minX, top: box.y - minY, width: box.w, height: box.h,
            outline: `${2 / z}px ${data.diffStatus === 'removed' ? 'dashed' : 'solid'} ${color}`,
            borderRadius: 6, pointerEvents: 'none' }} />;
      })}
      {bounds.filter((box) => selectedIds?.has(box.id)).map((box) =>
        <div key={box.id} className="graph-svg-selection" style={{ position: 'absolute', left: box.x - minX, top: box.y - minY,
          width: box.w, height: box.h, boxSizing: 'border-box', outline: `${1.5 / z}px solid var(--tv-selection)`, outlineOffset: `${3.5 / z}px`,
          borderRadius: 6, pointerEvents: 'none' }} />)}
    </div>
  </div>;
}
