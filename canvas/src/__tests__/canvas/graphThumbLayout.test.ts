import { computeThumbLayout } from '../../components/ui2/GraphThumb';
import { layerCardHeight, IO_CARD_HEIGHT } from '../../canvas/cardMetrics';
import { ModelNode } from '../../types';

/**
 * The project card shows a zoomed-out render of the graph's REAL canvas layout.
 *
 * Two regimes have to work, and they pull in opposite directions:
 *
 *  - An ordinary chain (a 5-layer template is ~1080x120 canvas units) must scale
 *    UNIFORMLY. Stretching it to fill a 3:1 card would draw 200x120 cards as
 *    squares — a lie about the shape of the nodes.
 *  - A real import is an extreme ribbon. The f5_tts document measured in this
 *    repo is 39490x620 over 262 columns and 5 rows, i.e. 64:1. Uniform fit gives
 *    each node 1.6px, so it has to drop to ticks and get a bounded vertical lift,
 *    or the card is an empty box with a hairline in it.
 *
 * These assert the observable consequences of that, not the constants.
 */

const VIEW_W = 340;
const VIEW_H = 112;
const PAD = 8;
/** Headroom reserved for the status pill over the thumbnail's top-left corner. */
const PAD_TOP = 28;
const BOX_W = VIEW_W - PAD * 2;
const BOX_H = VIEW_H - PAD_TOP - PAD;
/** Content is centred in the band below the pill, not in the whole view box. */
const BAND_CENTRE_Y = PAD_TOP + BOX_H / 2;

function layer(id: string, x: number, y: number): ModelNode {
  return {
    id,
    position: { x, y },
    // `category` is what the registry resolves for a real node (see
    // layerTypeForSemanticNode); omitting it here would test a shape the app
    // never produces.
    data: { layerType: { id: 'conv2d', name: 'Conv2d', category: 'conv' } as any, parameters: {} },
  } as unknown as ModelNode;
}

function io(id: string, x: number, y: number, kind: 'input' | 'output'): ModelNode {
  return {
    id,
    position: { x, y },
    data: {
      layerType: { id: kind, name: kind === 'input' ? 'Input' : 'Output', category: 'io' } as any,
      parameters: {},
    },
  } as unknown as ModelNode;
}

function chain(n: number, spacing = 220): { nodes: ModelNode[]; edges: Array<{ source: string; target: string }> } {
  const nodes = Array.from({ length: n }, (_, i) => layer(`n${i}`, i * spacing, 60));
  const edges = nodes.slice(1).map((_, i) => ({ source: `n${i}`, target: `n${i + 1}` }));
  return { nodes, edges };
}

function contentBox(placed: Array<{ x: number; y: number; w: number; h: number }>) {
  const x = Math.min(...placed.map((p) => p.x));
  const y = Math.min(...placed.map((p) => p.y));
  const right = Math.max(...placed.map((p) => p.x + p.w));
  const bottom = Math.max(...placed.map((p) => p.y + p.h));
  return { x, y, w: right - x, h: bottom - y };
}

describe('real canvas positions', () => {
  it('lays out from the saved positions rather than re-deriving them', () => {
    // Two nodes placed on the SAME column but different rows. A topology layout
    // would put them in sequence; the real layout stacks them.
    const nodes = [layer('a', 0, 0), layer('b', 400, 0), layer('c', 400, 300)];
    const layout = computeThumbLayout(nodes, [
      { source: 'a', target: 'b' },
      { source: 'a', target: 'c' },
    ])!;
    expect(layout.source).toBe('canvas');
    const [a, b, c] = layout.placed;
    expect(b.x).toBeCloseTo(c.x, 5); // same column, as saved
    expect(c.y).toBeGreaterThan(b.y); // stacked, as saved
    expect(a.x).toBeLessThan(b.x);
  });

  it("runs the canvas's own dagre pass when nothing has ever been placed", () => {
    // A graph never opened on the canvas has all-zero positions. Rather than
    // inventing a second layout algorithm, the thumbnail runs the same pass the
    // canvas runs on load, so the tile matches what opening it will show.
    const nodes = [layer('a', 0, 0), layer('b', 0, 0), layer('c', 0, 0)];
    const layout = computeThumbLayout(nodes, [
      { source: 'a', target: 'b' },
      { source: 'b', target: 'c' },
    ])!;
    expect(layout.source).toBe('dagre');
    // The layout must still put a 3-chain in left-to-right order.
    const xs = layout.placed.map((p) => p.x);
    expect(xs[0]).toBeLessThan(xs[1]);
    expect(xs[1]).toBeLessThan(xs[2]);
  });

  it('re-lays-out when only SOME nodes were ever placed', () => {
    // Measured on a real import: 17 of 40 nodes had saved positions and 23 did
    // not, so the unplaced ones defaulted to the origin and drew as one bright
    // stack in the corner. The canvas never puts two cards at one coordinate, so
    // a shared position proves the layout is partial.
    const nodes = [
      layer('placed1', 400, 60),
      layer('placed2', 700, 60),
      layer('unplaced1', 0, 0),
      layer('unplaced2', 0, 0),
    ];
    const layout = computeThumbLayout(nodes, [])!;
    expect(layout.source).toBe('dagre');
    // Nothing may share a position afterwards.
    const positions = layout.placed.map((p) => `${p.x.toFixed(2)},${p.y.toFixed(2)}`);
    expect(new Set(positions).size).toBe(positions.length);
  });

  it('honours the real card sizes, so an IO marker is shorter than a layer', () => {
    // Both are 200 wide, but the heights come from the canvas's own cardMetrics —
    // the expectation is derived from that module rather than hardcoded, so this
    // tracks the card geometry instead of drifting from it.
    const nodes = [io('in', 0, 0, 'input'), layer('l', 300, 0)];
    const layout = computeThumbLayout(nodes, [{ source: 'in', target: 'l' }])!;
    const [inp, lay] = layout.placed;
    expect(inp.w).toBeCloseTo(lay.w, 5);
    expect(inp.h).toBeLessThan(lay.h);
    expect(lay.h / inp.h).toBeCloseTo(layerCardHeight({}) / IO_CARD_HEIGHT, 2);
  });

  it('gives each node the accent its layer category has on the canvas', () => {
    // Colour is the channel that survives a zoom-out; grey outlines throw it away.
    const nodes = [io('in', 0, 0, 'input'), layer('conv', 300, 0)];
    const layout = computeThumbLayout(nodes, [])!;
    expect(layout.placed[0].color).toBe('var(--syntax-type)');
    expect(layout.placed[1].color).toBe('var(--cat-conv)');
  });

  it('skips the hidden copies of a repeated stack, as the canvas does', () => {
    const nodes = [
      layer('a', 0, 0),
      { ...layer('b', 300, 0), data: { ...layer('b', 300, 0).data, repeatHidden: true } } as ModelNode,
    ];
    const layout = computeThumbLayout(nodes, [])!;
    expect(layout.placed.map((p) => p.id)).toEqual(['a']);
  });
});

describe('an ordinary chain', () => {
  const { nodes, edges } = chain(5);
  const layout = computeThumbLayout(nodes, edges)!;

  it('scales uniformly, so node proportions are not a lie', () => {
    // Every node is 200x120 on the canvas; the ratio must survive.
    for (const p of layout.placed) {
      expect(p.w / p.h).toBeCloseTo(200 / 120, 1);
    }
  });

  it('fills the width and centres what is left over below the pill band', () => {
    const box = contentBox(layout.placed);
    expect(box.w).toBeCloseTo(BOX_W, 0);
    expect(box.y + box.h / 2).toBeCloseTo(BAND_CENTRE_Y, 0);
    // Nothing may intrude on the reserved headroom.
    expect(box.y).toBeGreaterThanOrEqual(PAD_TOP - 0.01);
  });

  it('draws cards, not ticks', () => {
    expect(layout.ticks).toBe(false);
  });

  it('draws one edge per connection', () => {
    expect(layout.segments).toHaveLength(4);
  });

  it('never overflows the view box', () => {
    for (const p of layout.placed) {
      expect(p.x).toBeGreaterThanOrEqual(0);
      expect(p.y).toBeGreaterThanOrEqual(0);
      expect(p.x + p.w).toBeLessThanOrEqual(VIEW_W);
      expect(p.y + p.h).toBeLessThanOrEqual(VIEW_H);
    }
  });
});

describe('an extreme ribbon (the measured f5_tts geometry)', () => {
  // 262 columns across 39490 units, 5 rows across 620 — 64:1.
  const nodes: ModelNode[] = [];
  const edges: Array<{ source: string; target: string }> = [];
  for (let col = 0; col < 262; col += 1) {
    const row = col % 5;
    nodes.push(layer(`c${col}`, 40 + col * 150, 60 + row * 155));
    if (col > 0) edges.push({ source: `c${col - 1}`, target: `c${col}` });
  }
  const layout = computeThumbLayout(nodes, edges)!;

  it('drops to ticks once a card would be narrower than its own stroke', () => {
    expect(layout.ticks).toBe(true);
    // Every node is sub-pixel wide at this zoom — that is why ticks exist.
    expect(Math.max(...layout.placed.map((p) => p.w))).toBeLessThan(2.6);
  });

  it('lifts the content off a hairline instead of leaving the card empty', () => {
    // Uniform fit on this geometry gives spanY * (BOX_W / spanX) ≈ 6 units of
    // height inside a 96-unit box — a 6% hairline. The clamp has to beat that.
    const spanX = 261 * 150 + 200;
    const spanY = 4 * 155 + 120;
    const uniformHeight = spanY * (BOX_W / spanX);
    expect(uniformHeight).toBeLessThan(BOX_H * 0.1);

    const box = contentBox(layout.placed);
    // Materially taller than the uniform hairline, and a usable share of the box.
    expect(box.h).toBeGreaterThan(uniformHeight * 2);
    expect(box.h).toBeGreaterThan(BOX_H * 0.2);
  });

  it('keeps the lift bounded, so rows are not stretched arbitrarily', () => {
    const box = contentBox(layout.placed);
    // The clamp targets a fraction of the box; it must not fill it outright,
    // which would distort a 64:1 graph into a 3:1 one.
    expect(box.h).toBeLessThan(BOX_H * 0.6);
  });

  it('still fills the width and stays inside the view box', () => {
    const box = contentBox(layout.placed);
    expect(box.w).toBeCloseTo(BOX_W, 0);
    for (const p of layout.placed) {
      expect(p.x + p.w).toBeLessThanOrEqual(VIEW_W + 0.01);
      expect(p.y + p.h).toBeLessThanOrEqual(VIEW_H + 0.01);
    }
  });

  it('renders every node — a 262-column import is not silently truncated', () => {
    expect(layout.placed).toHaveLength(262);
    expect(layout.truncated).toBe(false);
  });
});

describe('degenerate input', () => {
  it('returns null for an empty graph so the caller can draw the stub', () => {
    expect(computeThumbLayout([], [])).toBeNull();
  });

  it('survives a single node', () => {
    const layout = computeThumbLayout([layer('only', 10, 10)], [])!;
    expect(layout.placed).toHaveLength(1);
    const p = layout.placed[0];
    expect(p.x + p.w / 2).toBeCloseTo(VIEW_W / 2, 0);
    expect(p.y + p.h / 2).toBeCloseTo(BAND_CENTRE_Y, 0);
  });

  it('ignores edges pointing at nodes that are not in the graph', () => {
    const layout = computeThumbLayout([layer('a', 0, 0), layer('b', 300, 0)], [
      { source: 'a', target: 'ghost' },
      { source: 'a', target: 'b' },
    ])!;
    expect(layout.segments).toHaveLength(1);
  });

  it('does not spin on a cycle when it has to derive the layout', () => {
    // All-zero positions force the topology path; a cycle must not hang it.
    const nodes = [layer('a', 0, 0), layer('b', 0, 0)];
    const layout = computeThumbLayout(nodes, [
      { source: 'a', target: 'b' },
      { source: 'b', target: 'a' },
    ])!;
    expect(layout.placed).toHaveLength(2);
  });

  it('caps a pathological node count rather than emitting 10k rects', () => {
    const many = Array.from({ length: 1200 }, (_, i) => layer(`n${i}`, i * 150, 60));
    const layout = computeThumbLayout(many, [])!;
    expect(layout.truncated).toBe(true);
    expect(layout.placed.length).toBeLessThanOrEqual(900);
  });
});
