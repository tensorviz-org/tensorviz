import type { ModelEdge, ModelGraph, ModelNode } from '../../types';
import { canEnterModule, focusScopeForTargets, focusLayoutKey, layoutModuleFocus, matchingFocusSibling,
  moduleFocusPath, placeModuleFocus, projectModuleFocus } from '../../services/moduleFocus';
import { nodeCardSize } from '../../canvas/cardMetrics';
import { createEdgeRouter } from '../../canvas/edgeRouting';
import { edgeControlPoints } from '../../canvas/edgeCurve';
import { graphDocumentV2ToModelGraph, modelGraphToDocumentV2 } from '../../services/graphDocumentAdapter';
import captureDocument from '../../test/fixtures/canvas/capture-document.json';
import { deriveOperationGroups, revealOperationGroup } from '../../services/operationGroups';

const node = (id: string, parentId?: string, module = false): ModelNode => ({ id,
  type: module ? 'moduleNode' : 'layerNode', position: { x: 17, y: 23 },
  data: { layerType: { id: module ? 'module' : 'linear', name: module ? 'Module' : 'Linear' } as any,
    parameters: { name: id.split('.').pop(), _module_path: id }, parentId,
    ...(module ? { expanded: false, moduleInstanceId: id } : {}) },
});
const wire = (source: string, target: string, id = `${source}>${target}`): ModelEdge => ({ id, source, target });
const nested = (): ModelGraph => ({ nodes: [node('input'), node('root', undefined, true),
  node('root.a', 'root', true), node('root.a.attn', 'root.a', true), node('root.a.attn.op', 'root.a.attn'),
  node('root.a.ff', 'root.a', true), node('root.a.ff.op', 'root.a.ff'),
  node('root.b', 'root', true), node('root.b.attn', 'root.b', true), node('root.b.attn.op', 'root.b.attn'),
  node('root.b.ff', 'root.b', true), node('root.b.ff.op', 'root.b.ff'), node('output')],
  edges: [wire('input', 'root.a.attn.op'), wire('root.a.attn.op', 'root.a.ff.op'),
    wire('root.a.ff.op', 'root.b.attn.op'), wire('root.b.attn.op', 'root.b.ff.op'), wire('root.b.ff.op', 'output')],
});

it('shows only immediate children and true crossing edges, leaving the logical graph untouched', () => {
  const graph = nested(), before = JSON.stringify(graph);
  const focused = projectModuleFocus(graph.nodes, graph.edges, 'root.a')!;
  expect(focused.nodes.filter(n => !n.data.focusBoundary).map(n => n.id)).toEqual(['root.a.attn', 'root.a.ff']);
  expect(focused.nodes.filter(n => n.data.focusBoundary)).toHaveLength(2);
  expect(focused.nodes.every(n => !n.hidden && !n.parentNode && !n.data.parentId && !n.data.expanded)).toBe(true);
  expect(focused.edges.flatMap(e => e.data?.originalEdgeIds)).toEqual(expect.arrayContaining([
    'input>root.a.attn.op', 'root.a.attn.op>root.a.ff.op', 'root.a.ff.op>root.b.attn.op']));
  expect(focused.edges).toHaveLength(3);
  expect(JSON.stringify(graph)).toBe(before);
});

it('never invents internal boundary wiring when a static wrapper has no recorded child connection', () => {
  const graph = { nodes: [node('in'), node('m', undefined, true), node('a', 'm'), node('b', 'm')],
    edges: [wire('in', 'm'), wire('a', 'b')] };
  const focused = projectModuleFocus(graph.nodes, graph.edges, 'm')!;
  expect(focused.edges).toHaveLength(1);
  expect(focused.edges[0].source).toBe('a');
  expect(focused.nodes.some(n => n.data.focusBoundary)).toBe(false);
});

it('preserves captured uses, ordered ports and semantic export through repeated focus visits', () => {
  const graph = graphDocumentV2ToModelGraph(captureDocument as any);
  const root = graph.nodes.find(n => n.type === 'moduleNode')!;
  const before = modelGraphToDocumentV2(graph, { name: 'capture', revision: 1 }).semantic;
  for (let i = 0; i < 3; i++) {
    const focus = projectModuleFocus(graph.nodes, graph.edges, root.id)!;
    expect(focus).not.toBeNull();
    const byId = new Map(focus.nodes.map(n => [n.id, n]));
    for (const edge of focus.edges) {
      expect(edge.data?.captureEdge).toBeDefined();
      const original = graph.edges.find(e => e.id === edge.data!.originalEdgeIds![0])!;
      expect(edge.data!.captureEdge).toEqual(original.data!.captureEdge);
      for (const [id, handle] of [[edge.source, edge.sourceHandle], [edge.target, edge.targetHandle]]) {
        const n = byId.get(id!)!;
        expect((n.data.captureDisplayPorts ?? n.data.capturePorts)?.some(p => p.id === handle)).toBe(true);
      }
    }
  }
  expect(modelGraphToDocumentV2(graph, { name: 'capture', revision: 1 }).semantic).toEqual(before);
});

it('refuses missing, depth-limited, empty and cyclic scopes', () => {
  const graph = nested();
  graph.nodes.push(node('empty', undefined, true), node('cycle', 'cycle', true), node('cycle.op', 'cycle'));
  const limited = graph.nodes.find(n => n.id === 'root.b')!;
  limited.data.expandable = false;
  for (const id of ['missing', 'empty', 'cycle', 'root.b']) {
    expect(canEnterModule(graph.nodes, id)).toBe(false);
    expect(projectModuleFocus(graph.nodes, graph.edges, id)).toBeNull();
  }
  expect(moduleFocusPath(graph.nodes, 'root.a.attn').map(n => n.id)).toEqual(['root', 'root.a', 'root.a.attn']);
});

it('never turns hidden repeat or pass-through children into external boundaries', () => {
  const a = node('a', 'root', true), repeated = node('repeat', 'root', true), dropout = node('dropout', 'root', true);
  repeated.data.repeatHidden = true;
  dropout.data.capturePassThrough = { valueId: 'v', inputPortId: 'drop:in', sourceNodeId: 'a.op', sourcePortId: 'a:out' };
  const nodes = [node('root', undefined, true), a, node('a.op', 'a'), repeated, node('repeat.op', 'repeat'), dropout];
  const edges = [wire('a.op', 'repeat.op'), wire('a.op', 'dropout')];
  const hidden = projectModuleFocus(nodes, edges, 'root')!;
  expect(hidden.nodes.map(n => n.id)).toEqual(['a']);
  expect(hidden.edges).toHaveLength(0);
  expect(canEnterModule(nodes, 'repeat')).toBe(false);
  expect(canEnterModule([nodes[0], dropout], 'root')).toBe(false);
  dropout.data.passThroughShown = true;
  const shown = projectModuleFocus(nodes, edges, 'root')!;
  expect(shown.nodes.map(n => n.id)).toEqual(['a', 'dropout']);
  const annotation = shown.edges.find(e => e.id === 'passthrough:dropout')!;
  expect(annotation.data?.originalEdgeIds).toEqual([]);
  expect(annotation.data?._relationship).toBe('unchanged input');
});

it('deduplicates input fan-out while retaining distinct output uses and direction', () => {
  const nodes = [node('outer'), node('root', undefined, true), node('a', 'root'), node('b', 'root')];
  const edges = [
    { ...wire('outer', 'a', 'in1'), sourceHandle: 'value' },
    { ...wire('outer', 'b', 'in2'), sourceHandle: 'value' },
    { ...wire('a', 'outer', 'out1'), targetHandle: 'arg0' },
    { ...wire('b', 'outer', 'out2'), targetHandle: 'arg1' },
  ];
  const focus = projectModuleFocus(nodes, edges, 'root')!;
  expect(focus.nodes.filter(n => n.data.focusBoundary?.direction === 'in')).toHaveLength(1);
  expect(focus.nodes.filter(n => n.data.focusBoundary?.direction === 'out')).toHaveLength(2);
  expect(focus.edges).toHaveLength(4);
  expect(focus.edges[0].source).toBe(focus.edges[1].source);
});

it('reserves layout space for a removed review crossing without making it an active skip path', () => {
  const nodes = [node('outside'), node('root', undefined, true), node('a', 'root')];
  const edge = { ...wire('outside', 'a', 'review:removed'), data: { diffStatus: 'removed' as const } };
  const focus = projectModuleFocus(nodes, [edge], 'root')!;
  expect(focus.edges[0].data?.diffStatus).toBe('removed');
  const placed = placeModuleFocus(focus, layoutModuleFocus(focus, { width: 1000, height: 500 }, 'LR'));
  const external = placed.nodes.find(n => n.data.focusBoundary)!;
  expect(external.position.x).toBeLessThan(placed.nodes.find(n => n.id === 'a')!.position.x);
});

it('requires grouped targets to be explicitly revealed before selecting their parent scope', () => {
  const captured = (id: string, name: string, input: string, output: string): ModelNode => ({ ...node(id, 'root'),
    data: { ...node(id, 'root').data, parameters: { _shape_provenance: 'observed' }, shapeEvidence: 'measured',
      layerType: { id: 'captured_operation', name } as any,
      captureRef: { version: 1, captureId: 'run', sourceId: 'src', semanticId: id, kind: 'operation' },
      capturePorts: (['in', 'out'] as const).map(direction => ({ id: `${id}:${direction}`, nodeId: id, direction,
        valueId: direction === 'in' ? input : output, memberPortId: null, argumentKind: 'value',
        paths: [direction === 'in' ? ['arguments', 'args', 0] : ['output']], shape: [2, 8], dtype: 'torch.float32', device: 'cpu' })) } });
  const root = captured('root', 'Model', 'x', 'y'); root.type = 'moduleNode'; root.data.parentId = undefined;
  root.data.captureRef!.kind = 'invocation'; root.data.captureDetailLevel = 'layers';
  const nodes = [root, captured('add', 'aten.add.Tensor', 'x', 'a'), captured('permute', 'aten.permute.default', 'a', 'b'),
    captured('view', 'aten.view.default', 'b', 'c'), captured('bmm', 'aten.bmm.default', 'c', 'y')];
  const edges = [['add', 'permute', 'a'], ['permute', 'view', 'b'], ['view', 'bmm', 'c']].map(([source, target, valueId]) => ({
    ...wire(source, target), sourceHandle: `${source}:out`, targetHandle: `${target}:in`,
    data: { captureEdge: { id: `${source}>${target}`, kind: 'argument' as const, valueId, sourcePortId: `${source}:out`,
      targetPortId: `${target}:in`, displaySourcePortId: null, displayTargetPortId: null } } }));
  expect(focusScopeForTargets(nodes, edges, ['permute'])).toBeNull();
  expect(focusScopeForTargets(revealOperationGroup(deriveOperationGroups(nodes, edges), 'permute'), edges, ['permute'])).toBe('root');
});

it('retains a matching descendant path and stops at absent or ambiguous descendants', () => {
  const graph = nested();
  expect(matchingFocusSibling(graph.nodes, 'root.a.attn', 'root.a', 'root.b')).toBe('root.b.attn');
  graph.nodes = graph.nodes.filter(n => !n.id.startsWith('root.b.attn'));
  expect(matchingFocusSibling(graph.nodes, 'root.a.attn', 'root.a', 'root.b')).toBe('root.b');
  const twin = node('twin', 'root.b', true); twin.data.parameters._module_path = 'root.b.attn';
  graph.nodes.push(node('root.b.attn', 'root.b', true), node('op', 'root.b.attn'), twin, node('twin.op', 'twin'));
  expect(matchingFocusSibling(graph.nodes, 'root.a.attn', 'root.a', 'root.b')).toBe('root.b');
});

const chain = (count: number): ModelGraph => ({ nodes: Array.from({ length: count }, (_, i) => node(String(i))),
  edges: Array.from({ length: count - 1 }, (_, i) => wire(String(i), String(i + 1))) });

it('chooses by measured graph fit, supports explicit direction and keeps disconnected nodes', () => {
  const graph = chain(5);
  const wide = layoutModuleFocus(graph, { width: 1600, height: 600 });
  const tall = layoutModuleFocus(graph, { width: 390, height: 900 });
  expect(wide.direction).toBe('LR'); expect(tall.direction).toBe('TB');
  expect(layoutModuleFocus(graph, { width: 390, height: 900 }, 'LR').direction).toBe('LR');
  graph.nodes.push(node('disconnected'));
  const layout = layoutModuleFocus(graph, { width: 1600, height: 900 });
  expect(layout.positions.size).toBe(6);
  const placed = placeModuleFocus(graph, layout);
  for (const a of placed.nodes) for (const b of placed.nodes) if (a.id !== b.id) {
    const sa = nodeCardSize(a), sb = nodeCardSize(b);
    expect(a.position.x + sa.w <= b.position.x || b.position.x + sb.w <= a.position.x
      || a.position.y + sa.h <= b.position.y || b.position.y + sb.h <= a.position.y).toBe(true);
  }
});

it('ignores selection in layout identity and lays out a 240-node scope without dropping content', () => {
  const graph = chain(240), key = focusLayoutKey(graph);
  graph.nodes[0] = { ...graph.nodes[0], selected: true, data: { ...graph.nodes[0].data, isSelected: true } };
  expect(focusLayoutKey(graph)).toBe(key);
  const layout = layoutModuleFocus(graph, { width: 1440, height: 800 });
  expect(layout.positions.size).toBe(240);
  expect(Number.isFinite(layout.bounds.w) && Number.isFinite(layout.bounds.h)).toBe(true);
});

it.each(['LR', 'TB'] as const)('uses stable %s ports and a skip lane outside the intervening card', direction => {
  const graph = chain(3); graph.edges.push(wire('0', '2', 'skip'));
  const placed = placeModuleFocus(graph, layoutModuleFocus(graph, { width: 1200, height: 800 }, direction));
  const router = createEdgeRouter(placed.nodes, placed.edges);
  const boxes = new Map(placed.nodes.map(n => [n.id, { ...n.position, ...nodeCardSize(n) }]));
  const route = router.route(placed.edges[2], id => boxes.get(id))!;
  const side = direction === 'LR' ? 'bottom' : 'right';
  expect(route.startSide).toBe(side); expect(route.endSide).toBe(side);
  const points = edgeControlPoints(route.start.x, route.start.y, route.end.x, route.end.y, route.curveOffset, side, side);
  const middle = boxes.get('1')!;
  if (direction === 'LR') expect(Math.min(points.c1y, points.c2y)).toBeGreaterThan(middle.y + middle.h);
  else expect(Math.min(points.c1x, points.c2x)).toBeGreaterThan(middle.x + middle.w);
  const cubic = (t: number, a: number, b: number, c: number, d: number) => (1 - t) ** 3 * a
    + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * d;
  let samples = 0;
  for (let step = 1; step < 100; step++) {
    const t = step / 100;
    const x = cubic(t, route.start.x, points.c1x, points.c2x, route.end.x);
    const y = cubic(t, route.start.y, points.c1y, points.c2y, route.end.y);
    if (direction === 'LR' && x >= middle.x && x <= middle.x + middle.w) {
      expect(y).toBeGreaterThanOrEqual(middle.y + middle.h + 11.99); samples++;
    } else if (direction === 'TB' && y >= middle.y && y <= middle.y + middle.h) {
      expect(x).toBeGreaterThanOrEqual(middle.x + middle.w + 11.99); samples++;
    }
  }
  expect(samples).toBeGreaterThan(0);
  const ordinary = router.route(placed.edges[0], id => boxes.get(id))!;
  expect(ordinary.endSide).toBe(direction === 'LR' ? 'left' : 'top');
});
