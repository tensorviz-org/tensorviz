import { reconcileModuleView, graphNeedsLayout, layoutTopLevel, dagreLayout } from '../../services/canvasLayout';
import { ModelNode, ModelEdge } from '../../types';
import { layerCardHeight, layerCardWidth, nodeCardSize } from '../../canvas/cardMetrics';
import { createEdgeRouter } from '../../canvas/edgeRouting';
import { edgeControlPoints } from '../../canvas/edgeCurve';
import { graphNodeOrigin } from '../../services/graphOverview';
import { endpointOnSide } from '../../canvas/capturePorts';
import { graphToSvg } from '../../services/graphSvg';

/**
 * These guard the template-render fix: a flat graph with a module + children
 * must collapse correctly (children hidden, internal edges hidden, module
 * sized) and expand correctly (children parented + visible, module grown).
 * They fail if reconcile stops hiding collapsed children or drops the repeat
 * badge — the exact regressions that made templates look broken.
 */

const moduleNode = (id: string, expanded: boolean, extra: any = {}): ModelNode =>
  ({
    id,
    type: 'moduleNode',
    position: { x: 0, y: 0 },
    data: {
      layerType: { id: 'module', name: 'Module' },
      parameters: { name: id },
      isGroup: true,
      expanded,
      moduleInstanceId: id,
      ...extra,
    },
  } as unknown as ModelNode);

const layer = (id: string, parentId?: string): ModelNode =>
  ({
    id,
    type: 'layerNode',
    position: { x: 0, y: 0 },
    data: { layerType: { id: 'conv2d', name: 'Conv2d' }, parameters: {}, parentId },
  } as unknown as ModelNode);

const edge = (id: string, source: string, target: string): ModelEdge =>
  ({ id, source, target, data: {} } as unknown as ModelEdge);

it.each(['LR', 'TB'] as const)('lays out a full-model dependency chain without recursion or lost nodes: %s', rankdir => {
  const ids = Array.from({ length: 13_200 }, (_, index) => `node-${index}`);
  const sizes = new Map(ids.map(id => [id, { w: 240, h: 120 }]));
  const edges = ids.slice(1).map((target, index) => ({ source: ids[index], target }));
  const positions = dagreLayout(ids, sizes, edges, { rankdir });
  expect(positions.size).toBe(ids.length);
  const axis = rankdir === 'LR' ? 'x' : 'y', size = rankdir === 'LR' ? 240 : 120;
  expect(edges.every(edge => positions.get(edge.target)![axis] > positions.get(edge.source)![axis] + size)).toBe(true);
  expect(dagreLayout(ids, sizes, edges, { rankdir })).toEqual(positions);
});

it('retains large cyclic graphs in finite, non-overlapping positions', () => {
  const ids = Array.from({ length: 1100 }, (_, index) => `node-${index}`);
  const edges = ids.map((source, index) => ({ source, target: ids[(index + 1) % ids.length] }));
  const positions = dagreLayout(ids, new Map(), edges);
  expect(positions.size).toBe(ids.length);
  expect(new Set(Array.from(positions.values()).map(pos => `${pos.x}:${pos.y}`)).size).toBe(ids.length);
  expect(Array.from(positions.values()).every(pos => Number.isFinite(pos.x) && Number.isFinite(pos.y))).toBe(true);
});

const buildGraph = (expanded: boolean) => {
  const nodes: ModelNode[] = [
    layer('input'),
    moduleNode('m1', expanded, { repeatCount: 3, repeatBlockLabel: 'Bottleneck' }),
    layer('m1__a', 'm1'),
    layer('m1__b', 'm1'),
    layer('output'),
  ];
  const edges: ModelEdge[] = [
    edge('e_in', 'input', 'm1'),
    edge('e_out', 'm1', 'output'),
    edge('e_internal', 'm1__a', 'm1__b'),
  ];
  return { nodes, edges };
};

describe('reconcileModuleView — collapsed module', () => {
  const { nodes, edges } = buildGraph(false);
  const result = reconcileModuleView(nodes, edges, { relayout: true });
  const byId = new Map(result.nodes.map((n) => [n.id, n]));

  it('hides the children of a collapsed module', () => {
    expect(byId.get('m1__a')!.hidden).toBe(true);
    expect(byId.get('m1__b')!.hidden).toBe(true);
  });

  it('keeps the module and surrounding nodes visible', () => {
    expect(byId.get('m1')!.hidden).toBeFalsy();
    expect(byId.get('input')!.hidden).toBeFalsy();
    expect(byId.get('output')!.hidden).toBeFalsy();
  });

  it('hides edges internal to the collapsed module but keeps boundary edges', () => {
    const internal = result.edges.find((e) => e.id === 'e_internal')!;
    const boundaryIn = result.edges.find((e) => e.id === 'e_in')!;
    expect(internal.hidden).toBe(true);
    expect(boundaryIn.hidden).toBe(false);
  });

  it('gives the collapsed module the footprint of its visible content', () => {
    const m = byId.get('m1')! as any;
    expect(m.style.width).toBe(250);
    // This number is the module's shape box and the spacing its neighbours get, so it
    // must hold everything the card draws: a 56px header, 2px border, 22px body
    // padding and the 27px pill row (layer count, repeat, status). The old fixed 240
    // reserved a mini-map and sublayer chips the card no longer draws, leaving an
    // empty lower section; the rendered card is checked against this box in QA.
    expect(m.style.height).toBe(107);
  });

  it('preserves the repeat-badge metadata', () => {
    expect(byId.get('m1')!.data.repeatCount).toBe(3);
    expect(byId.get('m1')!.data.repeatBlockLabel).toBe('Bottleneck');
  });
});

describe('reconcileModuleView — expanded module', () => {
  const { nodes, edges } = buildGraph(true);
  const result = reconcileModuleView(nodes, edges, { relayout: true });
  const byId = new Map(result.nodes.map((n) => [n.id, n]));

  it('shows children and parents them to the module', () => {
    const a = byId.get('m1__a')!;
    expect(a.hidden).toBe(false);
    expect(a.parentNode).toBe('m1');
    expect(a.extent).toBe('parent');
  });

  it('shows the internal edge', () => {
    expect(result.edges.find((e) => e.id === 'e_internal')!.hidden).toBe(false);
  });

  it('grows the module beyond its collapsed footprint', () => {
    const m = byId.get('m1')! as any;
    expect(m.style.width).toBeGreaterThanOrEqual(250);
    expect(m.style.height).toBeGreaterThanOrEqual(140);
  });
});

describe('reconcileModuleView — edges belong to their lowest expanded module', () => {
  // WHY: internal edges are bucketed to the one expanded module that owns both
  // endpoints so a fully-expanded graph does not rescan every edge per frame. A
  // mis-bucket would drop or duplicate an internal edge, so this pins the mapping
  // for sibling modules with both an internal and a cross-module edge.
  const build = () => {
    const nodes: ModelNode[] = [
      moduleNode('m1', true), layer('m1__a', 'm1'), layer('m1__b', 'm1'),
      moduleNode('m2', true), layer('m2__a', 'm2'), layer('m2__b', 'm2'),
    ];
    const edges: ModelEdge[] = [
      edge('e_m1_internal', 'm1__a', 'm1__b'),
      edge('e_m2_internal', 'm2__a', 'm2__b'),
      edge('e_cross', 'm1__b', 'm2__a'),
    ];
    return { nodes, edges };
  };

  const result = reconcileModuleView(build().nodes, build().edges, { relayout: true });
  const byId = new Map(result.nodes.map((n) => [n.id, n]));

  it('keeps each internal edge visible and on its own endpoints', () => {
    const internal = result.edges.find((e) => e.id === 'e_m1_internal')!;
    expect(internal.hidden).toBe(false);
    expect(internal.source).toBe('m1__a');
    expect(internal.target).toBe('m1__b');
    expect(internal.data?.synthetic).toBeFalsy();
  });

  it('does not fold a cross-module edge into a synthetic wrapper edge', () => {
    const cross = result.edges.find((e) => e.id === 'e_cross')!;
    expect(cross.hidden).toBe(false);
    expect(cross.source).toBe('m1__b');
    expect(cross.target).toBe('m2__a');
  });

  it('parents every child to its own module', () => {
    expect(byId.get('m1__a')!.parentNode).toBe('m1');
    expect(byId.get('m2__a')!.parentNode).toBe('m2');
  });
});

describe('layout helpers', () => {
  it('hides only duplicate expanded wrapper connections and aligns the external chain', () => {
    const { nodes, edges } = buildGraph(true);
    nodes.push(layer('head'), layer('result'));
    const original = [...edges, edge('child-in', 'input', 'm1__a'), edge('child-out', 'm1__b', 'output'),
      edge('head', 'output', 'head'), edge('result', 'head', 'result')];
    const before = JSON.stringify([nodes, original]);
    const result = reconcileModuleView(nodes, original, { relayout: true });
    const byId = new Map(result.nodes.map(n => [n.id, n]));
    expect(result.edges.filter(e => e.hidden).map(e => e.id)).toEqual(['e_in', 'e_out']);
    expect(result.edges.map(({ hidden, ...e }) => e)).toEqual(original);
    const first = byId.get('m1__a')!, last = byId.get('m1__b')!, frame = byId.get('m1')!;
    const mid = (id: string) => byId.get(id)!.position.y + nodeCardSize(byId.get(id)!).h / 2;
    expect(mid('input')).toBeCloseTo(frame.position.y + first.position.y - 12);
    expect(mid('output')).toBeCloseTo(frame.position.y + last.position.y + nodeCardSize(last).h + 24);
    expect(mid('head')).toBeCloseTo(mid('output'));
    expect(mid('result')).toBeCloseTo(mid('head'));
    expect(reconcileModuleView(result.nodes, result.edges, { preserveChildPositions: true }).nodes.map(n => n.position))
      .toEqual(result.nodes.map(n => n.position));
    expect(JSON.stringify([nodes, original])).toBe(before);
    const closed = reconcileModuleView(result.nodes.map(n => n.id === 'm1'
      ? { ...n, data: { ...n.data, expanded: false } } : n), result.edges);
    expect(closed.edges.find(e => e.id === 'e_in')?.hidden).toBe(false);
    expect(closed.edges.find(e => e.id === 'e_out')?.hidden).toBe(false);
    const wrapperOnly = reconcileModuleView(nodes, edges, { relayout: true });
    expect(wrapperOnly.edges.filter(e => e.hidden)).toHaveLength(0);
  });

  it('stops external chain alignment before a merge from another source', () => {
    const nodes = [moduleNode('root', true), layer('first', 'root'), layer('last', 'root'),
      layer('flatten'), layer('merge'), layer('other'), layer('output')];
    const edges = [edge('inside', 'first', 'last'), edge('exit', 'last', 'flatten'),
      edge('to-merge', 'flatten', 'merge'), edge('other-merge', 'other', 'merge'), edge('end', 'merge', 'output')];
    const result = reconcileModuleView(nodes, edges, { relayout: true });
    const byId = new Map(result.nodes.map(n => [n.id, n]));
    const mid = (id: string) => byId.get(id)!.position.y + nodeCardSize(byId.get(id)!).h / 2;
    expect(mid('flatten')).not.toBeCloseTo(mid('merge'));
    expect(mid('merge')).toBeCloseTo(mid('output'));
    const top = result.nodes.filter(n => !n.parentNode);
    for (const a of top) for (const b of top) if (a.id !== b.id) {
      const sa = nodeCardSize(a), sb = nodeCardSize(b);
      expect(a.position.x < b.position.x + sb.w && a.position.x + sa.w > b.position.x
        && a.position.y < b.position.y + sb.h && a.position.y + sa.h > b.position.y).toBe(false);
    }
  });

  it('preserves the current graph layout when removed review edges are overlaid', () => {
    const nodes = [moduleNode('root', true), layer('gelu', 'root'), layer('norm', 'root'), layer('head', 'root')];
    const chain = [edge('gelu-norm', 'gelu', 'norm'), edge('norm-head', 'norm', 'head')];
    const ghost = { ...edge('review:removed', 'gelu', 'head'), data: { diffStatus: 'removed' as const } };
    const current = reconcileModuleView(nodes, chain, { relayout: true });
    const review = reconcileModuleView(nodes, [...chain, ghost], { relayout: true });
    expect(review.nodes).toEqual(current.nodes);
    expect(review.edges.find(e => e.id === ghost.id)).toEqual({ ...ghost, hidden: false });
    const top = nodes.filter(n => n.id !== 'root').map(n => ({ ...n, data: { ...n.data, parentId: undefined } }));
    expect(layoutTopLevel(top, [...chain, ghost])).toEqual(layoutTopLevel(top, chain));
  });

  it('aligns external cards with the real entry and exit of an expanded module', () => {
    const { nodes } = buildGraph(true);
    const edges = [edge('in', 'input', 'm1__a'), edge('inside', 'm1__a', 'm1__b'), edge('out', 'm1__b', 'output')];
    const result = reconcileModuleView(nodes, edges, { relayout: true });
    const byId = new Map(result.nodes.map(n => [n.id, n]));
    const module = byId.get('m1')!, input = byId.get('input')!, output = byId.get('output')!;
    const first = byId.get('m1__a')!, last = byId.get('m1__b')!;
    expect(input.position.y + layerCardHeight(input.data) / 2).toBeCloseTo(module.position.y + first.position.y - 12);
    expect(output.position.y + layerCardHeight(output.data) / 2).toBeCloseTo(module.position.y + last.position.y + layerCardHeight(last.data) + 24);
    const manual = layoutTopLevel(result.nodes, result.edges);
    for (const node of result.nodes.filter(n => !n.parentNode)) expect(manual.get(node.id)).toEqual(node.position);
    const restored = reconcileModuleView(result.nodes, result.edges, { relayout: false, preserveChildPositions: true });
    expect(restored.nodes.map(n => n.position)).toEqual(result.nodes.map(n => n.position));
    expect(restored.edges).toEqual(result.edges);
  });

  it('reserves the rendered height of tall sibling cards in both layout entry points', () => {
    const nodes = [layer('input'), layer('left'), layer('right'), layer('output')];
    for (const node of nodes) {
      node.data.parameters = { in_channels: 32, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, groups: 1 };
      node.data.inputShape = [1, 32, 64, 64];
      node.data.outputShape = [1, 64, 64, 64];
    }
    const edges = [edge('il', 'input', 'left'), edge('ir', 'input', 'right'),
      edge('lo', 'left', 'output'), edge('ro', 'right', 'output')];
    const reconciled = reconcileModuleView(nodes, edges, { relayout: true });
    const manual = layoutTopLevel(nodes, edges);
    const siblings = reconciled.nodes.filter(n => n.id === 'left' || n.id === 'right')
      .sort((a, b) => a.position.y - b.position.y);
    expect(siblings[1].position.y - siblings[0].position.y - layerCardHeight(siblings[0].data)).toBeGreaterThanOrEqual(48);
    for (const node of reconciled.nodes) expect(manual.get(node.id)).toEqual(node.position);
    const input = reconciled.nodes.find(n => n.id === 'input')!;
    expect(siblings[0].position.x - input.position.x - layerCardWidth(input.data)).toBeGreaterThanOrEqual(90);
  });

  it('graphNeedsLayout flags a freshly-flattened (all-origin) graph', () => {
    const { nodes } = buildGraph(false);
    expect(graphNeedsLayout(nodes)).toBe(true);
  });

  it('relayout spreads top-level nodes to distinct positions', () => {
    const { nodes, edges } = buildGraph(false);
    const result = reconcileModuleView(nodes, edges, { relayout: true });
    const top = result.nodes.filter((n) => !n.parentNode);
    const coords = new Set(top.map((n) => `${Math.round(n.position.x)},${Math.round(n.position.y)}`));
    expect(coords.size).toBe(top.length); // no two top-level nodes overlap
  });

  it('layoutTopLevel positions every top-level node and skips children', () => {
    const { nodes, edges } = buildGraph(false);
    const pos = layoutTopLevel(nodes, edges);
    expect(pos.has('m1')).toBe(true);
    expect(pos.has('input')).toBe(true);
    expect(pos.has('m1__a')).toBe(false); // children are not repositioned here
  });
});


it('reserves a clear nested header lane for a crossing skip without moving saved children', () => {
  const nodes = [moduleNode('root', true), moduleNode('stem', false, { parentId: 'root' }),
    moduleNode('nested', true, { parentId: 'root' }),
    moduleNode('a', false, { parentId: 'nested' }), moduleNode('b', false, { parentId: 'nested' }),
    layer('add', 'nested')];
  const chain = [edge('stem-a', 'stem', 'a'), edge('a-b', 'a', 'b'), edge('b-add', 'b', 'add')];
  const edges = [...chain, edge('skip', 'stem', 'add')];
  const plain = reconcileModuleView(nodes, chain, { relayout: true });
  const laid = reconcileModuleView(nodes, edges, { relayout: true });
  const byId = new Map(laid.nodes.map(node => [node.id, node]));
  const boxFor = (id: string) => ({ ...graphNodeOrigin(laid.nodes, id)!, ...nodeCardSize(byId.get(id)!) });
  const frame = boxFor('nested'), stem = boxFor('stem'), inner = boxFor('a');
  expect(frame.w).toBe(nodeCardSize(plain.nodes.find(node => node.id === 'nested')!).w + 96);
  expect(stem.x + stem.w / 2).toBeCloseTo(inner.x + inner.w / 2);
  const route = createEdgeRouter(laid.nodes, laid.edges).route(laid.edges.find(edge => edge.id === 'skip')!, boxFor)!;
  const c = edgeControlPoints(route.start.x, route.start.y, route.end.x, route.end.y,
    route.curveOffset, route.startSide, route.endSide);
  const at = (t: number, a: number, b: number, c: number, d: number) =>
    (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * d;
  let headerSamples = 0;
  for (let i = 0; i <= 1000; i++) {
    const x = at(i / 1000, route.start.x, c.c1x, c.c2x, route.end.x);
    const y = at(i / 1000, route.start.y, c.c1y, c.c2y, route.end.y);
    expect(x).toBeLessThan(frame.x + frame.w - 16);
    if (y >= frame.y && y <= inner.y) {
      headerSamples++;
      expect(x).toBeLessThan(frame.x + frame.w - 44 - 8);
    }
  }
  expect(headerSamples).toBeGreaterThan(0);
  const restored = reconcileModuleView(laid.nodes, laid.edges, { preserveChildPositions: true });
  for (const id of ['stem', 'a', 'b', 'add']) {
    expect(restored.nodes.find(node => node.id === id)!.position).toEqual(byId.get(id)!.position);
  }
  const collapsed = nodes.map(node => node.id === 'nested' ? { ...node, data: { ...node.data, expanded: false } } : node);
  const withSkip = reconcileModuleView(collapsed, edges, { relayout: true });
  const withoutSkip = reconcileModuleView(collapsed, chain, { relayout: true });
  expect(withSkip.nodes.map(node => [node.id, node.position, node.style]))
    .toEqual(withoutSkip.nodes.map(node => [node.id, node.position, node.style]));
});

describe('layout direction', () => {
  // input → root{ a → b → nested{ x → y } → c } → output, fully expanded.
  const nestedGraph = () => ({
    nodes: [layer('input'), moduleNode('root', true), layer('a', 'root'), layer('b', 'root'),
      moduleNode('nested', true, { parentId: 'root' }), layer('x', 'nested'), layer('y', 'nested'),
      layer('c', 'root'), layer('output')],
    edges: [edge('in', 'input', 'a'), edge('ab', 'a', 'b'), edge('bx', 'b', 'x'), edge('xy', 'x', 'y'),
      edge('yc', 'y', 'c'), edge('out', 'c', 'output')],
  });
  const boxes = (nodes: ModelNode[]) => (id: string) => ({ ...graphNodeOrigin(nodes, id)!, ...nodeCardSize(nodes.find(n => n.id === id)!) });
  const flow = (nodes: ModelNode[], ids: string[], axis: 'x' | 'y') => {
    const box = boxes(nodes);
    for (let i = 1; i < ids.length; i++) {
      const previous = box(ids[i - 1]), next = box(ids[i]);
      // Each step advances past the previous card on the flow axis.
      expect(next[axis]).toBeGreaterThanOrEqual(previous[axis] + (axis === 'x' ? previous.w : previous.h));
    }
  };

  it.each(['horizontal', 'vertical'] as const)('runs the top level and every expanded frame %s, with matching ports', direction => {
    const { nodes, edges } = nestedGraph();
    const laid = reconcileModuleView(nodes, edges, { relayout: true, direction });
    const axis = direction === 'horizontal' ? 'x' : 'y';
    flow(laid.nodes, ['input', 'root', 'output'], axis);
    flow(laid.nodes, ['a', 'b', 'nested', 'c'], axis);
    flow(laid.nodes, ['x', 'y'], axis);
    const root = boxes(laid.nodes)('root');
    // A chain of four laid along one axis is longer on that axis.
    expect(direction === 'horizontal' ? root.w > root.h : root.h > root.w).toBe(true);
    const router = createEdgeRouter(laid.nodes, laid.edges, direction);
    for (const route of laid.edges.filter(e => !e.hidden).map(e => router.route(e, boxes(laid.nodes))!)) {
      expect([route.startSide, route.endSide]).toEqual(direction === 'horizontal' ? ['right', 'left'] : ['bottom', 'top']);
    }
  });

  it('keeps the legacy row with stacked module contents when no direction is given', () => {
    const { nodes, edges } = nestedGraph();
    const laid = reconcileModuleView(nodes, edges, { relayout: true });
    flow(laid.nodes, ['input', 'root', 'output'], 'x');
    flow(laid.nodes, ['a', 'b', 'nested', 'c'], 'y');
    const sides = createEdgeRouter(laid.nodes, laid.edges).sides;
    expect(sides.get('input')).toMatchObject({ in: 'left', out: 'right' });
    expect(sides.get('b')).toMatchObject({ in: 'top', out: 'bottom' });
  });

  it.each(['horizontal', 'vertical'] as const)('lines boundary cards up straight with the real entry and exit (%s)', direction => {
    const { nodes, edges } = nestedGraph();
    const laid = reconcileModuleView(nodes, edges, { relayout: true, direction });
    const box = boxes(laid.nodes);
    const across = direction === 'horizontal' ? 'y' : 'x';
    const [inSide, outSide] = direction === 'horizontal' ? ['left', 'right'] as const : ['top', 'bottom'] as const;
    const port = (id: string, side: typeof inSide | typeof outSide) =>
      endpointOnSide(laid.nodes.find(n => n.id === id)!.data, undefined, box(id), side)[across];
    expect(port('input', outSide)).toBeCloseTo(port('a', inSide));
    expect(port('output', inSide)).toBeCloseTo(port('c', outSide));
    // The manual Auto layout entry point agrees with reconcile.
    const manual = layoutTopLevel(laid.nodes, laid.edges, direction);
    for (const node of laid.nodes.filter(n => !n.parentNode)) expect(manual.get(node.id)).toEqual(node.position);
  });

  it('changes only presentation: semantic data and edges are identical in every direction', () => {
    const { nodes, edges } = nestedGraph();
    const before = JSON.stringify([nodes, edges]);
    const results = [undefined, 'horizontal', 'vertical'].map(direction =>
      reconcileModuleView(nodes, edges, { relayout: true, direction: direction as 'horizontal' | 'vertical' | undefined }));
    for (const result of results.slice(1)) {
      expect(result.nodes.map(n => n.data)).toEqual(results[0].nodes.map(n => n.data));
      expect(result.edges).toEqual(results[0].edges);
    }
    expect(JSON.stringify([nodes, edges])).toBe(before);
  });

  it('restores saved horizontal positions and frame sizes without drift', () => {
    const { nodes, edges } = nestedGraph();
    const laid = reconcileModuleView(nodes, edges, { relayout: true, direction: 'horizontal' });
    const restored = reconcileModuleView(laid.nodes, laid.edges, { preserveChildPositions: true, direction: 'horizontal' });
    expect(restored.nodes.map(n => [n.id, n.position, n.style])).toEqual(laid.nodes.map(n => [n.id, n.position, n.style]));
  });

  it('routes a skip inside a horizontal frame through a lane below the row, clear of the skipped card', () => {
    const nodes = [moduleNode('root', true), layer('a', 'root'), layer('b', 'root'), layer('c', 'root')];
    const edges = [edge('ab', 'a', 'b'), edge('bc', 'b', 'c'), edge('skip', 'a', 'c')];
    const laid = reconcileModuleView(nodes, edges, { relayout: true, direction: 'horizontal' });
    const box = boxes(laid.nodes);
    const router = createEdgeRouter(laid.nodes, laid.edges, 'horizontal');
    const skip = laid.edges.find(e => e.id === 'skip')!;
    const route = router.route(skip, box)!;
    expect([route.startSide, route.endSide]).toEqual(['bottom', 'bottom']);
    const c = edgeControlPoints(route.start.x, route.start.y, route.end.x, route.end.y, route.curveOffset, route.startSide, route.endSide);
    const at = (t: number, p0: number, p1: number, p2: number, p3: number) =>
      (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3;
    const skipped = box('b'), frame = box('root');
    for (let i = 0; i <= 500; i++) {
      const x = at(i / 500, route.start.x, c.c1x, c.c2x, route.end.x), y = at(i / 500, route.start.y, c.c1y, c.c2y, route.end.y);
      expect(x > skipped.x && x < skipped.x + skipped.w && y > skipped.y && y < skipped.y + skipped.h).toBe(false);
      expect(y).toBeLessThanOrEqual(frame.y + frame.h);
    }
    // SVG overview/export draws the same lane as the live canvas.
    expect(graphToSvg(laid.nodes, laid.edges, { direction: 'horizontal' }))
      .toContain(`M ${route.start.x} ${route.start.y} C ${c.c1x} ${c.c1y} ${c.c2x} ${c.c2y} ${route.end.x} ${route.end.y}`);
    // Without named tensors a card has one input side, so the chain shares the
    // lane's entry. Dragging a card above its neighbours flips no side.
    const dragged = (id: string) => id === 'c' ? { ...box('c'), y: box('c').y - 600 } : box(id);
    expect(router.route(skip, dragged)).toMatchObject({ startSide: 'bottom', endSide: 'bottom' });
    expect(router.route(laid.edges.find(e => e.id === 'bc')!, dragged)).toMatchObject({ startSide: 'right', endSide: 'bottom' });
  });
});
