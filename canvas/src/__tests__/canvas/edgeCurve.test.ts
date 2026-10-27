/**
 * Edge curve geometry. The bug this pins: control points used to be
 * unconditionally horizontal, so an edge whose target sat LEFT of its source
 * doubled back over itself, and the shape box was padded from the endpoints only
 * so the bulge was clipped by its own SVG viewport.
 *
 * These tests assert the two invariants the renderer and the box-sizing code
 * must both satisfy, for every relative position a dragged node can produce.
 */
import { distanceToEdgeCurve, edgeControlPoints, edgeCurveBounds } from '../../canvas/edgeCurve';

const cubicAt = (t: number, p0: number, p1: number, p2: number, p3: number) => {
  const mt = 1 - t;
  return mt * mt * mt * p0 + 3 * mt * mt * t * p1 + 3 * mt * t * t * p2 + t * t * t * p3;
};

/** Every relative position a user can drag a node into. */
const CASES: Array<[string, number, number]> = [
  ['forward, level (the normal chain)', 400, 0],
  ['forward, short hop', 90, 40],
  ['forward and steeply down', 60, 300],
  ['BACKWARD, level — the reported bug', -400, 30],
  ['BACKWARD and up', -400, -300],
  ['straight down (expanded-module internals are rankdir TB)', 0, 260],
  ['near-vertical, slightly backward', -10, -260],
  ['barely backward, level', -40, 0],
  ['coincident', 0, 0],
];

describe('edge curve geometry', () => {
  it('keeps a removed review connection outside an inserted tile, with matching bounds and hit testing', () => {
    const c = edgeControlPoints(0, 0, 0, 300, 220);
    const x = cubicAt(.5, 0, c.c1x, c.c2x, 0);
    const y = cubicAt(.5, 0, c.c1y, c.c2y, 300);
    expect(x).toBeGreaterThan(140);
    expect(edgeCurveBounds(0, 0, 0, 300, 8, 220).maxX).toBeGreaterThan(x);
    expect(distanceToEdgeCurve(x, y, 0, 0, 0, 300, 48, 220)).toBeLessThan(1);
    expect(distanceToEdgeCurve(0, 150, 0, 0, 0, 300, 48, 220)).toBeGreaterThan(20);
  });

  it.each(CASES)('%s: the curve never doubles back past either endpoint', (_label, dx, dy) => {
    const sx = 0;
    const sy = 0;
    const ex = dx;
    const ey = dy;
    const { c1x, c1y, c2x, c2y } = edgeControlPoints(sx, sy, ex, ey);

    // "Doubling back" is the control point overshooting the far endpoint on the
    // axis the curve travels along — that is what produced the S that folded
    // over itself and read as a detached edge.
    if (Math.abs(dy) > Math.abs(dx)) {
      // Vertical-dominant: control points stay on the source/target x, and each
      // must sit BETWEEN the endpoints on y.
      expect(c1x).toBe(sx);
      expect(c2x).toBe(ex);
      const lo = Math.min(sy, ey);
      const hi = Math.max(sy, ey);
      expect(c1y).toBeGreaterThanOrEqual(lo - 1e-9);
      expect(c1y).toBeLessThanOrEqual(hi + 1e-9);
      expect(c2y).toBeGreaterThanOrEqual(lo - 1e-9);
      expect(c2y).toBeLessThanOrEqual(hi + 1e-9);
    } else if (dx !== 0) {
      // Horizontal-dominant: the first control point leans TOWARD the target and
      // the second leans back toward the source — never the reverse.
      const towardTarget = dx > 0 ? c1x >= sx : c1x <= sx;
      const towardSource = dx > 0 ? c2x <= ex : c2x >= ex;
      expect(towardTarget).toBe(true);
      expect(towardSource).toBe(true);
    }
  });

  it.each(CASES)('%s: the reported bounds contain the whole curve', (_label, dx, dy) => {
    const sx = 0;
    const sy = 0;
    const ex = dx;
    const ey = dy;
    const pad = 28;
    const { minX, minY, maxX, maxY } = edgeCurveBounds(sx, sy, ex, ey, pad);
    const { c1x, c1y, c2x, c2y } = edgeControlPoints(sx, sy, ex, ey);

    for (let i = 0; i <= 200; i += 1) {
      const t = i / 200;
      const x = cubicAt(t, sx, c1x, c2x, ex);
      const y = cubicAt(t, sy, c1y, c2y, ey);
      expect(x).toBeGreaterThanOrEqual(minX);
      expect(x).toBeLessThanOrEqual(maxX);
      expect(y).toBeGreaterThanOrEqual(minY);
      expect(y).toBeLessThanOrEqual(maxY);
    }
  });

  it('bounds are tight, not a blanket over-pad', () => {
    // A level forward edge bows on neither axis, so the box should be exactly
    // the endpoints plus the padding — proving the extrema are solved rather
    // than guessed at with a fixed margin.
    const { minX, minY, maxX, maxY } = edgeCurveBounds(0, 0, 400, 0, 28);
    expect(minX).toBeCloseTo(-28);
    expect(maxX).toBeCloseTo(428);
    expect(minY).toBeCloseTo(-28);
    expect(maxY).toBeCloseTo(28);
  });

  it('a vertical edge curves on y, not sideways into a lateral loop', () => {
    // Expanded-module internals are laid out rankdir TB. Forcing the horizontal
    // S on them is what produced lateral loops inside the module frame.
    const { minX, maxX } = edgeCurveBounds(0, 0, 0, 260, 0);
    expect(maxX - minX).toBeCloseTo(0);
  });
});

describe('clicking an edge hit-tests the stroke, not the box', () => {
  // A diagonal edge: its bounding box is mostly empty space, which is exactly why
  // the click test cannot be a box test.
  const EDGE = { sx: 0, sy: 0, ex: 400, ey: 300 };
  // Screen slop from edgeSelection.EDGE_HIT_PX, at zoom 1.
  const TOLERANCE = 10;

  it('a point deep inside the padded box but far from the stroke is NOT a hit', () => {
    // Top-right corner of the box. The curve is nowhere near it — it is down at
    // y ~= 300 by the time x reaches 390.
    const point = { x: 390, y: 10 };
    const { minX, minY, maxX, maxY } = edgeCurveBounds(EDGE.sx, EDGE.sy, EDGE.ex, EDGE.ey, 28);

    // The box really does contain it, so a box-based hit test WOULD claim this
    // click. That is the regression this test exists to catch.
    expect(point.x).toBeGreaterThan(minX);
    expect(point.x).toBeLessThan(maxX);
    expect(point.y).toBeGreaterThan(minY);
    expect(point.y).toBeLessThan(maxY);

    const distance = distanceToEdgeCurve(point.x, point.y, EDGE.sx, EDGE.sy, EDGE.ex, EDGE.ey);
    expect(distance).toBeGreaterThan(TOLERANCE);
  });

  it('a point on the stroke IS a hit', () => {
    // Midpoint of a symmetric cubic sits at the average of its four control
    // points, which for this edge is exactly the chord's midpoint.
    const distance = distanceToEdgeCurve(200, 150, EDGE.sx, EDGE.sy, EDGE.ex, EDGE.ey);
    expect(distance).toBeLessThan(1);
  });

  it('registers every point along the curve, not just near its endpoints', () => {
    // Walk the same cubic the renderer draws and demand each sample be a hit. A
    // distance that only measured the endpoints — or that sampled too coarsely —
    // still passes the two cases above but fails here.
    const [sx, sy, ex, ey] = [0, 0, -400, 30];
    const { c1x, c1y, c2x, c2y } = edgeControlPoints(sx, sy, ex, ey);
    for (let i = 1; i < 10; i += 1) {
      const t = i / 10;
      const x = cubicAt(t, sx, c1x, c2x, ex);
      const y = cubicAt(t, sy, c1y, c2y, ey);
      expect(distanceToEdgeCurve(x, y, sx, sy, ex, ey)).toBeLessThan(1);
    }
  });
});
