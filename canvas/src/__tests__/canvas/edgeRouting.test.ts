import { edgeBow, edgeControlPoints, edgeCurveBounds, distanceToEdgeCurve, edgeCurveMidpoint, rightLanePull } from '../../canvas/edgeCurve';
import type { AnchorSide } from '../../canvas/edgeAnchors';
import { createEdgeRouter, routedShapeGeometry, skipEdgeIds } from '../../canvas/edgeRouting';
import { graphToSvg } from '../../services/graphSvg';
import { nodeCardSize } from '../../canvas/cardMetrics';
import type { ModelEdge, ModelNode } from '../../types';

const cubic = (t: number, a: number, b: number, c: number, d: number) =>
  (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * d;
const sides: AnchorSide[] = ['left', 'right', 'top', 'bottom'];

it('classifies shortcuts across a full-model chain without searching later layers', () => {
  const ids = Array.from({ length: 13_200 }, (_, index) => `node-${index}`);
  const chain = ids.slice(1).map((target, index) => ({ id: `normal-${index}`, source: ids[index], target }));
  const shortcuts = ids.slice(2).map((target, index) => ({ id: `skip-${index}`, source: ids[index], target }));
  expect(skipEdgeIds([...chain, ...shortcuts], () => true)).toEqual(new Set(shortcuts.map(edge => edge.id)));
});

it('ordinary routes read only endpoint boxes instead of scanning all siblings', () => {
  const nodes = ['a', 'b', 'c'].map(id => ({ id, type: 'layerNode', parentNode: 'root',
    position: { x: 0, y: 0 }, data: {} })) as ModelNode[];
  const edges = [{ id: 'a-b', source: 'a', target: 'b' }] as ModelEdge[];
  const boxes = jest.fn(() => ({ x: 0, y: 0, w: 200, h: 100 }));
  expect(createEdgeRouter(nodes, edges).route(edges[0], boxes)).toBeDefined();
  expect(boxes.mock.calls).toEqual([['a'], ['b']]);
});

it('meets each selected port squarely and shares geometry with bounds and picking', () => {
  for (const startSide of sides) for (const endSide of sides) {
    const c = edgeControlPoints(0, 0, 400, 300, 0, startSide, endSide);
    const normal = (side: AnchorSide) => side === 'left' ? [-1, 0] : side === 'right' ? [1, 0] : side === 'top' ? [0, -1] : [0, 1];
    const a = normal(startSide), b = normal(endSide);
    expect(c.c1x * a[1] - c.c1y * a[0]).toBeCloseTo(0);
    expect((c.c2x - 400) * b[1] - (c.c2y - 300) * b[0]).toBeCloseTo(0);
    expect(c.c1x * a[0] + c.c1y * a[1]).toBeGreaterThan(0);
    for (const bow of [0, 220]) {
      const points = edgeControlPoints(0, 0, 400, 300, bow, startSide, endSide);
      const bounds = edgeCurveBounds(0, 0, 400, 300, 28, bow, startSide, endSide);
      for (let i = 0; i <= 40; i++) {
        const t = i / 40, x = cubic(t, 0, points.c1x, points.c2x, 400), y = cubic(t, 0, points.c1y, points.c2y, 300);
        expect(x).toBeGreaterThanOrEqual(bounds.minX); expect(x).toBeLessThanOrEqual(bounds.maxX);
        expect(y).toBeGreaterThanOrEqual(bounds.minY); expect(y).toBeLessThanOrEqual(bounds.maxY);
        expect(distanceToEdgeCurve(x, y, 0, 0, 400, 300, 48, bow, startSide, endSide)).toBeLessThan(1);
      }
    }
  }
});

it('keeps a skip in a compact right-side lane clear of the stacked cards', () => {
  const obstacles = [{ x: 0, y: 148, w: 200, h: 100 }, { x: 0, y: 296, w: 200, h: 100 }];
  const pull = rightLanePull(200, 50, 200, 494, obstacles);
  const c = edgeControlPoints(200, 50, 200, 494, pull, 'right', 'right');
  const bounds = edgeCurveBounds(200, 50, 200, 494, 0, pull, 'right', 'right');
  expect(bounds.maxX).toBeLessThanOrEqual(224);
  for (let i = 0; i <= 500; i++) {
    const t = i / 500, x = cubic(t, 200, c.c1x, c.c2x, 200), y = cubic(t, 50, c.c1y, c.c2y, 494);
    for (const box of obstacles) expect(x > box.x && x < box.x + box.w && y > box.y && y < box.y + box.h).toBe(false);
  }
});

it('keeps short mixed-axis entry and exit hooks monotone without sagging', () => {
  for (const [sx, sy, ex, ey, from, to] of [[0, 0, 90, 12, 'right', 'top'], [0, 0, 90, 24, 'bottom', 'left']] as const) {
    expect(edgeBow(sx, sy, ex, ey, from, to)).toBe(0);
    const bounds = edgeCurveBounds(sx, sy, ex, ey, 0, 0, from, to);
    expect(bounds).toEqual({ minX: sx, minY: sy, maxX: ex, maxY: ey });
  }
});

it('uses one side per shared tensor and keeps skip ports stable when dragged', () => {
  const nodes = ['a', 'b', 'c'].map((id, i) => ({ id, parentNode: id === 'a' ? 'root' : 'nested', type: 'layerNode',
    position: { x: 0, y: i * 148 }, data: { capturePorts: [
      { id: `${id}:in`, direction: 'in' }, { id: `${id}:out`, direction: 'out' },
    ] } })) as ModelNode[];
  const edges = [['a', 'b'], ['b', 'c'], ['a', 'c']].map(([source, target]) => ({
    id: `${source}-${target}`, source, target, sourceHandle: `${source}:out`, targetHandle: `${target}:in`,
  })) as ModelEdge[];
  const router = createEdgeRouter(nodes, edges);
  const boxes = new Map(nodes.map(node => [node.id, { ...node.position, w: 200, h: 100 }]));
  const normal = router.route(edges[0], id => boxes.get(id))!, skip = router.route(edges[2], id => boxes.get(id))!;
  expect(skip.startSide).toBe('right'); expect(skip.endSide).toBe('right');
  expect(normal.start).toEqual(skip.start);
  expect(router.sides.get('c')!.ports!['c:in']).toBe('right');
  boxes.get('c')!.y = -300;
  expect(router.route(edges[2], id => boxes.get(id))!.endSide).toBe('right');
});

it('separates a backwards connector and places the insert point on its bowed stroke', () => {
  const bow = edgeBow(200, 0, 0, 0, 'right', 'left');
  const mid = edgeCurveMidpoint(200, 0, 0, 0, bow, 'right', 'left');
  expect(mid.y).toBeGreaterThanOrEqual(24);
  expect(distanceToEdgeCurve(mid.x, mid.y, 200, 0, 0, 0, 48, bow, 'right', 'left')).toBeLessThan(1);
});

it('keeps removed review connections out of active skip paths and port selection', () => {
  const nodes = ['gelu', 'norm', 'head'].map(id => ({ id, type: 'layerNode', parentNode: 'root',
    position: { x: 0, y: 0 }, data: {} })) as ModelNode[];
  const chain = [
    { id: 'gelu-norm', source: 'gelu', target: 'norm' },
    { id: 'norm-head', source: 'norm', target: 'head' },
  ] as ModelEdge[];
  const ghost = { id: 'review:removed', source: 'gelu', target: 'head', data: { diffStatus: 'removed' } } as ModelEdge;
  const router = createEdgeRouter(nodes, [...chain, ghost]);
  expect(router.sides).toEqual(createEdgeRouter(nodes, chain).sides);
  expect(skipEdgeIds([...chain, ghost], () => true).size).toBe(0);
  // A removed segment must not supply an alternate path for a live edge either.
  expect(skipEdgeIds([{ ...chain[0], data: { diffStatus: 'removed' } }, chain[1],
    { ...ghost, data: {} }], () => true).size).toBe(0);
  expect(router.route(ghost, () => ({ x: 0, y: 0, w: 200, h: 100 }))?.curveOffset).toBe(220);
});

it('uses identical captured port slots in the live route and SVG, and reads live boxes', () => {
  const port = (id: string, direction: string) => ({ id, nodeId: 'b', direction, argumentKind: 'value', paths: [] });
  const nodes = [
    { id: 'a', type: 'layerNode', position: { x: 0, y: 0 }, data: { layerType: { id: 'linear', name: 'Linear' }, parameters: {} } },
    { id: 'b', type: 'layerNode', parentNode: 'root', position: { x: 400, y: 150 },
      data: { layerType: { id: 'linear', name: 'Linear' }, parameters: {}, capturePorts: [port('in-1', 'in'), port('in-2', 'in')] } },
  ] as unknown as ModelNode[];
  const edge = { id: 'e', source: 'a', target: 'b', targetHandle: 'in-2' } as ModelEdge;
  const boxes = new Map(nodes.map(node => [node.id, { ...node.position, ...nodeCardSize(node) }]));
  const router = createEdgeRouter(nodes), route = router.route(edge, id => boxes.get(id))!;
  expect(route.startSide).toBe('right'); expect(route.endSide).toBe('top');
  expect(route.end.x).toBeCloseTo(400 + 200 * 2 / 3); expect(route.end.y).toBe(150);
  const svg = graphToSvg(nodes, [edge]);
  const c = edgeControlPoints(route.start.x, route.start.y, route.end.x, route.end.y, route.curveOffset, route.startSide, route.endSide);
  expect(svg).toContain(`M ${route.start.x} ${route.start.y} C ${c.c1x} ${c.c1y} ${c.c2x} ${c.c2y} ${route.end.x} ${route.end.y}`);
  const before = JSON.stringify([nodes, edge]);
  boxes.get('a')!.x = 800;
  const moved = router.route(edge, id => boxes.get(id))!;
  expect(moved.start.x).toBe(1000); expect(moved.curveOffset).not.toBe(0);
  expect(routedShapeGeometry(moved).props.startSide).toBe('right');
  expect(JSON.stringify([nodes, edge])).toBe(before);
});


it('does not invent semantic boundary dots for an expanded captured frame in SVG', () => {
  const node = { id: 'root', type: 'moduleNode', position: { x: 0, y: 0 }, style: { width: 340, height: 400 },
    data: { layerType: { id: 'module', name: 'Model' }, parameters: {}, expanded: true,
      capturePorts: [{ id: 'in', direction: 'in' }, { id: 'out', direction: 'out' }], captureDisplayPorts: [],
      inputPorts: ['in'], outputPorts: ['out'] } } as ModelNode;
  const image = new DOMParser().parseFromString(graphToSvg([node], []), 'image/svg+xml');
  expect(image.querySelectorAll('circle[opacity="0.75"]')).toHaveLength(0);
});
