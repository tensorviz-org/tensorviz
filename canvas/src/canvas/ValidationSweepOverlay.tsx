import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Editor } from 'tldraw';
import { ModelNode, ModelEdge } from '../types';
import { nodeIdToShapeId } from './canvasShapeId';
import { topologicalOrder } from '../services/topologicalOrder';
import { prefersReducedMotion } from './reducedMotion';

interface ValidationSweepOverlayProps {
  editor: Editor | null;
  nodes: ModelNode[];
  edges: ModelEdge[];
  /** Bump this to (re)play the sweep — e.g. a counter incremented after each verify. */
  runId: number;
  /** Source projections also contain nodes the recorded run never reached. */
  measuredOnly?: boolean;
}

interface Rect { x: number; y: number; w: number; h: number; }
interface Step { id: string; ok: boolean; }

// Budget: the wave must reach the last of 100 nodes within ~900ms. Larger graphs
// ripple a touch faster; small graphs get a slower, more legible per-node beat.
const SWEEP_BUDGET_MS = 900;
const MAX_STEP_MS = 70;
const MIN_STEP_MS = 4;
const HALO_MS = 480; // must match the ui2-sweep-pop keyframe duration

function stepDelay(count: number): number {
  if (count <= 1) return MAX_STEP_MS;
  return Math.max(MIN_STEP_MS, Math.min(MAX_STEP_MS, Math.floor(SWEEP_BUDGET_MS / count)));
}

/**
 * Shader-free canvas overlay. On each new runId it walks the graph in
 * topological order and flashes a green (ok) / red (error) halo over each node
 * as the "wave" reaches it — the shape engine walking the graph, made visible.
 * The persistent error state lives on node.data.error (painted by LayerShapeUtil);
 * this is the transient pulse on top. No-op under prefers-reduced-motion.
 */
export const ValidationSweepOverlay: React.FC<ValidationSweepOverlayProps> = ({
  editor, nodes, edges, runId, measuredOnly = false,
}) => {
  const [order, setOrder] = useState<Step[] | null>(null);
  const [litCount, setLitCount] = useState(0);
  const [rects, setRects] = useState<Record<string, Rect>>({});
  const timers = useRef<number[]>([]);

  const clearTimers = () => {
    timers.current.forEach((t) => window.clearTimeout(t));
    timers.current = [];
  };

  // Start a sweep when runId advances.
  useEffect(() => {
    if (!editor || runId <= 0 || nodes.length === 0 || prefersReducedMotion()) return;

    const ids = nodes.map((n) => n.id);
    const ordered = topologicalOrder(ids, edges.map((e) => ({ source: e.source, target: e.target })));
    const okById = new Map(nodes.map((n) => [n.id, !n.data?.error] as const));
    const eligible = new Set(nodes.filter(node => !measuredOnly || node.data?.error
      || node.data?.parameters?._shape_provenance === 'observed').map(node => node.id));
    const steps: Step[] = ordered.filter(id => eligible.has(id)).map((id) => ({ id, ok: okById.get(id) !== false }));

    clearTimers();
    setOrder(steps);
    setLitCount(0);

    const step = stepDelay(steps.length);
    steps.forEach((_, i) => {
      timers.current.push(window.setTimeout(() => setLitCount(i + 1), i * step));
    });
    // Tear down once the last halo has finished fading.
    timers.current.push(window.setTimeout(() => {
      setOrder(null);
      setLitCount(0);
    }, steps.length * step + HALO_MS + 80));

    return clearTimers;
    // Intentionally keyed on runId (+ editor): a new verify replays; node/edge
    // closures are read fresh because runId changes on every verify.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, editor]);

  // Track each node's screen rect; recompute on camera/pan/zoom while playing.
  const recompute = useCallback(() => {
    if (!editor || !order) return;
    const next: Record<string, Rect> = {};
    for (const { id } of order) {
      const shape = editor.getShape(nodeIdToShapeId(id)) as { x: number; y: number; props?: { w?: number; h?: number } } | undefined;
      if (!shape) continue;
      const w = shape.props?.w ?? 200;
      const h = shape.props?.h ?? 120;
      const tl = editor.pageToViewport({ x: shape.x, y: shape.y });
      const br = editor.pageToViewport({ x: shape.x + w, y: shape.y + h });
      next[id] = { x: tl.x, y: tl.y, w: br.x - tl.x, h: br.y - tl.y };
    }
    setRects(next);
  }, [editor, order]);

  useEffect(() => {
    if (!editor || !order) return;
    recompute();
    const offPage = editor.sideEffects.registerAfterChangeHandler('instance_page_state', recompute);
    const offCamera = editor.sideEffects.registerAfterChangeHandler('camera', recompute);
    return () => { offPage(); offCamera(); };
  }, [editor, order, recompute]);

  if (!order) return null;

  return (
    <div className="ui2-sweep-overlay" aria-hidden>
      {order.slice(0, litCount).map(({ id, ok }) => {
        const r = rects[id];
        if (!r) return null;
        return (
          <div
            key={id}
            className={`ui2-sweep-halo ${ok ? 'ok' : 'err'}`}
            style={{ left: r.x, top: r.y, width: r.w, height: r.h }}
          />
        );
      })}
    </div>
  );
};
