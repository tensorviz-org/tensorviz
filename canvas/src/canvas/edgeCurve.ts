import type { AnchorBox, AnchorSide } from './edgeAnchors';

/**
 * Edge curve geometry, shared by the renderer (ModelEdgeShapeUtil) and the
 * layout that sizes the edge shape's box (useTldrawBridge). Both MUST agree: the
 * renderer draws in the shape's local box, so a curve the box does not contain
 * is silently clipped by the SVG viewport.
 *
 * The bug this replaces: control points were unconditionally HORIZONTAL
 * (c1 = start + offset on x, c2 = end - offset on x). For a forward edge that is
 * the right S-curve, but when a node is dragged so its target sits to the LEFT
 * of its source, c1 pushes further right while c2 pushes further left — the
 * curve doubles back over itself. And because the box was padded a fixed 28px
 * from the endpoints only, the resulting bulge was clipped.
 */

export interface CurvePoints {
  c1x: number;
  c1y: number;
  c2x: number;
  c2y: number;
}

/**
 * Control points chosen on the DOMINANT axis.
 *
 * - Vertical-dominant (|dy| > |dx|): a vertical S. This is the case for the
 *   internals of an expanded module, which dagre lays out rankdir 'TB' — those
 *   edges are ~215px backward on x and ~126px down on y, and forcing a
 *   horizontal S on them is what produced lateral loops inside a module frame.
 * - Horizontal-dominant: the original horizontal S, which is correct for the
 *   left-to-right chain the top-level layout produces.
 *
 * The offset is proportional to the span on that axis and clamped, so a short
 * hop gets a gentle curve and a long one does not balloon.
 */
export function edgeControlPoints(startX: number, startY: number, endX: number, endY: number, offsetAcross = 0, startSide?: AnchorSide, endSide?: AnchorSide): CurvePoints {
  const dx = endX - startX;
  const dy = endY - startY;
  if (startSide && endSide) {
    if (startSide === 'right' && endSide === 'right' && offsetAcross > 0) {
      return { c1x: startX + offsetAcross, c1y: startY, c2x: endX + offsetAcross, c2y: endY };
    }
    if (startSide === 'bottom' && endSide === 'bottom' && offsetAcross > 0) {
      return { c1x: startX, c1y: startY + offsetAcross, c2x: endX, c2y: endY + offsetAcross };
    }
    const mixedAxes = ['top', 'bottom'].includes(startSide) !== ['top', 'bottom'].includes(endSide);
    const control = (x: number, y: number, side: AnchorSide) => {
      const vertical = side === 'top' || side === 'bottom';
      let pull = vertical ? Math.max(24, Math.min(120, Math.abs(dy) * .5))
        : Math.max(40, Math.min(160, Math.abs(dx) * .5));
      if (mixedAxes) pull = Math.min(pull, vertical ? Math.abs(dy) : Math.abs(dx));
      return { x: x + (side === 'right' ? pull : side === 'left' ? -pull : offsetAcross),
        y: y + (side === 'bottom' ? pull : side === 'top' ? -pull : offsetAcross) };
    };
    const a = control(startX, startY, startSide), b = control(endX, endY, endSide);
    return { c1x: a.x, c1y: a.y, c2x: b.x, c2y: b.y };
  }

  if (Math.abs(dy) > Math.abs(dx)) {
    const offset = Math.max(24, Math.min(120, Math.abs(dy) * 0.5));
    const dir = dy >= 0 ? 1 : -1;
    return { c1x: startX + offsetAcross, c1y: startY + offset * dir, c2x: endX + offsetAcross, c2y: endY - offset * dir };
  }

  const offset = Math.max(40, Math.min(160, Math.abs(dx) * 0.5));
  const dir = dx >= 0 ? 1 : -1;
  return { c1x: startX + offset * dir, c1y: startY + offsetAcross, c2x: endX - offset * dir, c2y: endY + offsetAcross };
}

/** Cubic Bezier value on one axis at t. */
function cubicAt(t: number, p0: number, p1: number, p2: number, p3: number): number {
  const mt = 1 - t;
  return mt * mt * mt * p0 + 3 * mt * mt * t * p1 + 3 * mt * t * t * p2 + t * t * t * p3;
}

/**
 * Exact extrema of one axis of a cubic, by solving the derivative's roots —
 * not a sampled approximation, so the box is tight AND never clips. The
 * derivative of a cubic Bezier is a quadratic: at^2 + bt + c.
 */
function cubicExtrema(p0: number, p1: number, p2: number, p3: number): { min: number; max: number } {
  let min = Math.min(p0, p3);
  let max = Math.max(p0, p3);

  const a = 3 * (-p0 + 3 * p1 - 3 * p2 + p3);
  const b = 6 * (p0 - 2 * p1 + p2);
  const c = 3 * (p1 - p0);

  const consider = (t: number) => {
    if (t <= 0 || t >= 1) return;
    const v = cubicAt(t, p0, p1, p2, p3);
    if (v < min) min = v;
    if (v > max) max = v;
  };

  if (Math.abs(a) < 1e-9) {
    // Degenerates to linear: bt + c = 0
    if (Math.abs(b) > 1e-9) consider(-c / b);
  } else {
    const disc = b * b - 4 * a * c;
    if (disc >= 0) {
      const root = Math.sqrt(disc);
      consider((-b + root) / (2 * a));
      consider((-b - root) / (2 * a));
    }
  }

  return { min, max };
}

/**
 * The page-space box that exactly contains the curve between two anchor points,
 * plus `pad` for the stroke, its glow and the arrow head.
 */
export function edgeCurveBounds(
  startX: number,
  startY: number,
  endX: number,
  endY: number,
  pad: number,
  offsetAcross = 0,
  startSide?: AnchorSide,
  endSide?: AnchorSide,
): { minX: number; minY: number; maxX: number; maxY: number } {
  const { c1x, c1y, c2x, c2y } = edgeControlPoints(startX, startY, endX, endY, offsetAcross, startSide, endSide);
  const xs = cubicExtrema(startX, c1x, c2x, endX);
  const ys = cubicExtrema(startY, c1y, c2y, endY);
  return {
    minX: xs.min - pad,
    minY: ys.min - pad,
    maxX: xs.max + pad,
    maxY: ys.max + pad,
  };
}

/** Shortest distance from a point to a line segment. */
function distanceToSegment(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): number {
  const dx = bx - ax;
  const dy = by - ay;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared < 1e-12) return Math.sqrt((px - ax) ** 2 + (py - ay) ** 2);
  let t = ((px - ax) * dx + (py - ay) * dy) / lengthSquared;
  t = Math.max(0, Math.min(1, t));
  return Math.sqrt((px - (ax + t * dx)) ** 2 + (py - (ay + t * dy)) ** 2);
}

/**
 * Shortest distance from a point to the drawn STROKE, by walking the same cubic
 * as a polyline.
 *
 * Hit-testing an edge cannot go through its bounding box: a long S-curve's box
 * is mostly empty, so a box test would claim clicks that land nowhere near the
 * line. And the edge shape cannot answer for itself — ModelEdgeShapeUtil renders
 * with pointerEvents:'none' and returns an empty indicator path on purpose, so
 * edges never steal marquee-select, drag-to-pan or click-empty-to-deselect.
 *
 * Coordinates are the shape's LOCAL box, matching props.startX/endX.
 */
export function distanceToEdgeCurve(
  px: number,
  py: number,
  startX: number,
  startY: number,
  endX: number,
  endY: number,
  samples = 48,
  offsetAcross = 0,
  startSide?: AnchorSide,
  endSide?: AnchorSide,
): number {
  const { c1x, c1y, c2x, c2y } = edgeControlPoints(startX, startY, endX, endY, offsetAcross, startSide, endSide);
  let best = Infinity;
  let prevX = startX;
  let prevY = startY;
  for (let i = 1; i <= samples; i += 1) {
    const t = i / samples;
    const x = cubicAt(t, startX, c1x, c2x, endX);
    const y = cubicAt(t, startY, c1y, c2y, endY);
    const d = distanceToSegment(px, py, prevX, prevY, x, y);
    if (d < best) best = d;
    prevX = x;
    prevY = y;
  }
  return best;
}

/** Separate a dragged backward connector without changing its chosen sides. */
export function edgeBow(
  sx: number, sy: number, ex: number, ey: number,
  startSide: AnchorSide, endSide: AnchorSide,
): number {
  const vertical = startSide === 'bottom' || startSide === 'top';
  const forward = startSide === 'bottom' ? ey - sy : startSide === 'top' ? sy - ey
    : startSide === 'right' ? ex - sx : sx - ex;
  if (forward >= 0) return 0;
  const across = vertical ? ex - sx : ey - sy;
  return (across < 0 ? -1 : 1) * Math.max(60, Math.abs(across) * .5);
}

/** Right-side skip lane. Both controls point out of the card; the usual equal-
 * width column peaks 16px beyond its right edge, inside the frame padding. */
export function rightLanePull(sx: number, sy: number, ex: number, ey: number, obstacles: readonly AnchorBox[]): number {
  const candidates = obstacles.filter(box => box.y > Math.min(sy, ey) && box.y + box.h < Math.max(sy, ey));
  let pull = (Math.max(sx, ex, ...candidates.map(box => box.x + box.w)) + 16 - Math.min(sx, ex)) / .75;
  const timeAtY = (y: number) => {
    let lo = 0, hi = 1;
    for (let i = 0; i < 32; i++) {
      const t = (lo + hi) / 2;
      if ((cubicAt(t, sy, sy, ey, ey) < y) === (ey > sy)) lo = t; else hi = t;
    }
    return (lo + hi) / 2;
  };
  for (const box of candidates) {
    const a = timeAtY(box.y), b = timeAtY(box.y + box.h);
    const baseline = Math.min(cubicAt(a, sx, sx, ex, ex), cubicAt(b, sx, sx, ex, ex));
    const factor = Math.min(3 * a * (1 - a), 3 * b * (1 - b));
    pull = Math.max(pull, (box.x + box.w + 12 - baseline) / factor);
  }
  return pull;
}

/** The same lane below a left-to-right row: `rightLanePull` with x and y swapped. */
export function bottomLanePull(sx: number, sy: number, ex: number, ey: number, obstacles: readonly AnchorBox[]): number {
  return rightLanePull(sy, sx, ey, ex, obstacles.map(box => ({ x: box.y, y: box.x, w: box.h, h: box.w })));
}

export function edgeCurveMidpoint(sx: number, sy: number, ex: number, ey: number,
  bow = 0, startSide?: AnchorSide, endSide?: AnchorSide): { x: number; y: number } {
  const c = edgeControlPoints(sx, sy, ex, ey, bow, startSide, endSide);
  return { x: cubicAt(.5, sx, c.c1x, c.c2x, ex), y: cubicAt(.5, sy, c.c1y, c.c2y, ey) };
}
