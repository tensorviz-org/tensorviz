import React, { useMemo } from 'react';
import { ModelEdge, ModelGraph, ModelNode } from '../../types';
import { graphNeedsLayout, layoutTopLevel } from '../../services/canvasLayout';
import { layerCardHeight, ioCardHeight } from '../../canvas/cardMetrics';
import { categoryColor } from '../../data/categoryColor';
import { anchorPoint, boxCenter, sourceSide, targetSide } from '../../canvas/edgeAnchors';

/**
 * A zoomed-out render of the graph as it is actually laid out on the canvas.
 *
 * The positions are the real ones: `view.node_positions` rides on every project
 * in the list response and reaches here as `node.position` via
 * graphDocumentV2ToModelGraph. So a card shows the user THEIR model — an F5-TTS
 * import and a three-layer template look nothing alike, which is the point.
 *
 * Two things make that work at 340x112:
 *
 * 1. LEVEL OF DETAIL. Real graphs are extreme ribbons — a measured F5-TTS import
 *    is 39490x620 canvas units, 262 columns by 5 rows. Uniform zoom-to-fit gives
 *    each node ~1.3px, so drawing rounded rects would produce mush. Below a
 *    legibility threshold the nodes become ticks, which reads honestly as "a long
 *    dense model" instead of pretending to be a diagram.
 *
 * 2. AN ASPECT CLAMP. Fitting 64:1 content into a 3:1 box uniformly leaves a
 *    5px-tall hairline floating in an empty card. Past a limit the vertical scale
 *    is allowed to run ahead of the horizontal one, capped, so the row structure
 *    stays visible. That is a deliberate, bounded distortion of a thumbnail — not
 *    of anything anyone measures.
 *
 * A graph with no saved positions (never opened on the canvas) has nothing to
 * zoom out FROM, so it falls back to laying itself out by topology.
 */

export type GraphThumbStatus = 'verified' | 'error' | 'unvalidated';

type GraphThumbProps = {
  graph?: Pick<ModelGraph, 'nodes' | 'edges'> | null;
  /** Marks the failing node in red, matching the canvas's error state. */
  status?: GraphThumbStatus;
  /** Node id that failed validation. Falls back to the last node when unknown. */
  errorNodeId?: string;
  height?: number;
  className?: string;
};

const VIEW_W = 340;
const VIEW_H = 112;
const PAD = 8;
/**
 * Extra headroom at the top. The status pill sits over the thumbnail's top-left
 * corner; measured against the rendered card it occupies view-box y 5..26, so a
 * graph that fills the full height runs under it. 28 clears it with a margin.
 * Reserving the band unconditionally is cheaper than teaching this component
 * about the pill, and it costs a full-height graph about a quarter of its scale.
 */
const PAD_TOP = 28;

const EDGE_STROKE = 'rgba(196,196,202,0.42)';
const ERROR_STROKE = 'rgba(255,123,114,0.85)';

/**
 * Card widths match the shared Canvas. The HEIGHTS come from
 * canvas/cardMetrics, which is the module the canvas itself measures with — it is
 * deliberately tldraw-free (services/canvasLayout already imports it), so a
 * thumbnail can share it without dragging the editor into a page that has no
 * canvas on it.
 */
const NODE_W = 200;
const MODULE_W = 240;
const MODULE_H = 160;

/** Below this rendered width a rounded rect is indistinguishable from a smear. */
const TICK_BELOW = 2.6;
/**
 * Only content flatter than this fraction of the box gets the aspect clamp.
 *
 * Set low on purpose. An ordinary graph — a five-layer chain is 1400x120 canvas
 * units — is genuinely a wide flat thing, and stretching it to fill a 3:1 card
 * would render 200x120 cards as squares, i.e. lie about the shape of the nodes.
 * Ordinary graphs therefore scale uniformly and simply do not fill the card
 * vertically, which is the truth. The clamp exists only for the extreme ribbons
 * (a measured import is 39490x620, 64:1) where uniform fit leaves a hairline.
 */
const MIN_VERTICAL_FILL = 0.22;
/** How far the vertical scale may run ahead of the horizontal one. */
const ASPECT_CAP = 6;
/** Element budget. A 290-node graph is fine; a pathological one must not hang. */
const MAX_NODES = 900;
const MAX_EDGES = 1200;

type Rect = { id: string; x: number; y: number; w: number; h: number; color: string };

function nodeSize(node: ModelNode): { w: number; h: number } {
  const data = node.data;
  const kind = String(data?.layerType?.id || data?.type || node.type || '').toLowerCase();
  if (data?.isGroup || kind === 'module' || kind === 'group' || kind === 'modulenode') {
    return { w: MODULE_W, h: MODULE_H };
  }
  if (kind === 'input' || kind === 'output') return { w: NODE_W, h: ioCardHeight(data) };
  // The real rendered height: a card grows with its parameter and shape rows, so
  // a Conv2d with six params is genuinely taller than a ReLU.
  return { w: NODE_W, h: layerCardHeight(data ?? {}) };
}

/**
 * The accent the canvas paints this node with. Colour is what makes a zoomed-out
 * graph readable — a conv stack is teal, activations are yellow, norms are mauve —
 * so a thumbnail of grey outlines throws away the one channel that survives at
 * this scale. An errored node takes the error hue, exactly as the card does.
 */
function nodeColor(node: ModelNode, failing: boolean): string {
  if (failing || node.data?.error) return 'var(--error)';
  const data = node.data;
  const kind = String(data?.layerType?.id || data?.type || node.type || '').toLowerCase();
  if (data?.isGroup || kind === 'module' || kind === 'group' || kind === 'modulenode') {
    return categoryColor('module');
  }
  if (kind === 'input' || kind === 'output') return categoryColor('io');
  return categoryColor(data?.layerType?.category);
}

/**
 * Where the nodes go: the saved canvas positions, or the layout the canvas would
 * assign if you opened the project right now.
 *
 * `graphNeedsLayout` and `layoutTopLevel` are the canvas's OWN load-time logic —
 * the same dagre pass, the same "these positions are degenerate" heuristic (two
 * overlapping nodes, or two or more parked at the origin). Reusing them is the
 * whole point: the tile then shows what opening the project shows, instead of a
 * second opinion about how the graph should look.
 *
 * That matters more than it sounds. A partial layout is common — measured on a
 * real import, 23 of its 40 nodes had no saved position while 17 did — and an
 * earlier cut of this drew those 23 stacked into one bright smear in the corner.
 */
function placeNodes(
  nodes: ModelNode[],
  edges: ModelEdge[],
): { rects: Rect[]; source: 'canvas' | 'dagre' } {
  // The canvas hides the non-canonical copies of a repeated stack, so a x12
  // encoder draws as one block here too.
  const visible = nodes.filter((n) => !n.data?.repeatHidden);
  if (!visible.length) return { rects: [], source: 'canvas' };

  let positions: Map<string, { x: number; y: number }> | null = null;
  let source: 'canvas' | 'dagre' = 'canvas';
  if (graphNeedsLayout(visible)) {
    try {
      positions = layoutTopLevel(visible, edges);
      source = 'dagre';
    } catch {
      // dagre throwing on a malformed graph must degrade to the saved positions,
      // not blank the card.
      positions = null;
    }
  }

  const rects: Rect[] = [];
  for (const node of visible) {
    const p = positions?.get(node.id) ?? node.position;
    if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
    const { w, h } = nodeSize(node);
    rects.push({ id: node.id, x: p.x, y: p.y, w, h, color: nodeColor(node, false) });
  }
  return { rects, source };
}

export type ThumbPlacement = {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  /** The canvas's accent for this node's layer category. */
  color: string;
};
export type ThumbLayout = {
  placed: ThumbPlacement[];
  segments: string[];
  /** True once nodes are too small to draw as cards; they render as ticks. */
  ticks: boolean;
  /** 'canvas' = the saved positions; 'dagre' = the layout the canvas would assign. */
  source: 'canvas' | 'dagre';
  truncated: boolean;
};

/**
 * Exported so the scaling can be asserted against real graph geometry — the
 * extreme-ribbon and ordinary-chain regimes behave differently and both matter.
 */
export function computeThumbLayout(
  nodes: ModelNode[],
  edges: ModelEdge[],
): ThumbLayout | null {
  if (!nodes.length) return null;

  const drawable = nodes.filter((n) => !n.data?.repeatHidden).length;
  const { rects: allRects, source } = placeNodes(nodes, edges);
  if (!allRects.length) return null;

  const rects = allRects.length > MAX_NODES ? allRects.slice(0, MAX_NODES) : allRects;
  // Reported rather than silent, so a caller can say so if it wants to.
  const truncated = rects.length < drawable;

  // Content bounds in canvas units.
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const r of rects) {
    minX = Math.min(minX, r.x);
    minY = Math.min(minY, r.y);
    maxX = Math.max(maxX, r.x + r.w);
    maxY = Math.max(maxY, r.y + r.h);
  }
  const spanX = Math.max(1, maxX - minX);
  const spanY = Math.max(1, maxY - minY);

  const boxW = VIEW_W - PAD * 2;
  const boxH = VIEW_H - PAD_TOP - PAD;
  const fit = Math.min(boxW / spanX, boxH / spanY);
  const sx = fit;
  let sy = fit;
  // A 64:1 ribbon fitted uniformly is a hairline in an empty card. Let the
  // vertical scale run ahead, capped, so its row structure stays visible.
  if (spanY * fit < boxH * MIN_VERTICAL_FILL) {
    sy = Math.min(fit * ASPECT_CAP, (boxH * MIN_VERTICAL_FILL) / spanY, boxH / spanY);
  }

  const offX = PAD + (boxW - spanX * sx) / 2;
  const offY = PAD_TOP + (boxH - spanY * sy) / 2;

  const placed = rects.map((r) => ({
    id: r.id,
    x: offX + (r.x - minX) * sx,
    y: offY + (r.y - minY) * sy,
    w: Math.max(0.8, r.w * sx),
    h: Math.max(0.8, r.h * sy),
    color: r.color,
  }));
  const byId = new Map(placed.map((p) => [p.id, p]));

  // Level of detail: real cards while they are legible, ticks once they are not.
  const ticks = NODE_W * sx < TICK_BELOW;

  const segments: string[] = [];
  let edgeCount = 0;
  for (const { source, target } of edges) {
    if (edgeCount >= MAX_EDGES) break;
    const a = byId.get(source);
    const b = byId.get(target);
    if (!a || !b) continue;
    // Float the endpoints to the sides the live canvas would use, rather than
    // always drawing source-right to target-left.
    const start = anchorPoint(a, sourceSide(a, boxCenter(b)));
    const end = anchorPoint(b, targetSide(b, boxCenter(a)));
    // A coincident pair has no run to draw.
    if (Math.abs(end.x - start.x) < 0.5 && Math.abs(end.y - start.y) < 0.5) continue;
    segments.push(`M${start.x.toFixed(1)} ${start.y.toFixed(1)}L${end.x.toFixed(1)} ${end.y.toFixed(1)}`);
    edgeCount += 1;
  }

  return { placed, segments, ticks, source, truncated };
}

export default function GraphThumb({
  graph,
  status = 'unvalidated',
  errorNodeId,
  height = 132,
  className,
}: GraphThumbProps) {
  const view = useMemo(() => {
    const nodes = graph?.nodes ?? [];
    const edges = (graph?.edges ?? []) as ModelEdge[];
    const layout = computeThumbLayout(nodes, edges);
    if (!layout) return null;
    const byId = new Set(layout.placed.map((p) => p.id));
    const failing =
      status === 'error'
        ? (errorNodeId && byId.has(errorNodeId) ? errorNodeId : layout.placed[layout.placed.length - 1]?.id)
        : undefined;
    return { ...layout, failing };
  }, [graph, status, errorNodeId]);

  return (
    <div className={`ui2-graph-thumb ${className ?? ''}`} style={{ height }}>
      <div className="ui2-graph-thumb-grid" aria-hidden="true" />
      <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} preserveAspectRatio="xMidYMid meet" aria-hidden="true">
        {!view ? (
          // An empty graph gets the dashed stub the design uses, not a blank box
          // that reads as a failed render.
          <rect
            x={VIEW_W / 2 - 55}
            y={VIEW_H / 2 - 20}
            width={110}
            height={40}
            rx={5}
            fill="none"
            stroke="rgba(255,255,255,0.20)"
            strokeWidth={1.2}
            strokeDasharray="4 4"
          />
        ) : (
          <>
            {view.segments.length > 0 && (
              <g
                stroke={EDGE_STROKE}
                strokeWidth={view.ticks ? 0.5 : 1.1}
                fill="none"
                shapeRendering={view.ticks ? 'optimizeSpeed' : undefined}
              >
                <path d={view.segments.join('')} />
              </g>
            )}
            {view.ticks
              ? // Dense mode: solid ticks in the category hue. Colour survives at
                // this scale where an outline does not, so a conv stack still
                // reads as a conv stack.
                view.placed
                  .filter((p) => p.id !== view.failing)
                  .map((p) => (
                    <rect
                      key={p.id}
                      x={p.x}
                      y={p.y}
                      width={Math.max(1, p.w)}
                      height={p.h}
                      rx={0.5}
                      fill={p.color}
                      fillOpacity={0.62}
                      stroke="none"
                      shapeRendering="optimizeSpeed"
                    />
                  ))
              : // Card mode: the canvas's own treatment — a tinted fill under an
                // accent border, at the same 11%/36% mix the node card uses.
                view.placed
                  .filter((p) => p.id !== view.failing)
                  .map((p) => (
                    <rect
                      key={p.id}
                      x={p.x}
                      y={p.y}
                      width={p.w}
                      height={p.h}
                      rx={Math.min(4, p.w / 4)}
                      fill={p.color}
                      fillOpacity={0.16}
                      stroke={p.color}
                      strokeOpacity={0.62}
                      strokeWidth={1.1}
                    />
                  ))}
            {view.failing &&
              view.placed
                .filter((p) => p.id === view.failing)
                .map((p) => (
                  <g key={p.id} fill="none" stroke={ERROR_STROKE} strokeWidth={1.4}>
                    <rect x={p.x} y={p.y} width={p.w} height={p.h} rx={Math.min(4, p.w / 4)} />
                    {/* The cross only reads once the box is big enough to hold it. */}
                    {p.w > 12 && p.h > 10 && (
                      <path
                        d={
                          `M${(p.x + p.w / 2 - 8).toFixed(1)} ${(p.y + p.h / 2 - 6).toFixed(1)}` +
                          `l16 12M${(p.x + p.w / 2 + 8).toFixed(1)} ${(p.y + p.h / 2 - 6).toFixed(1)}l-16 12`
                        }
                      />
                    )}
                  </g>
                ))}
          </>
        )}
      </svg>
    </div>
  );
}
