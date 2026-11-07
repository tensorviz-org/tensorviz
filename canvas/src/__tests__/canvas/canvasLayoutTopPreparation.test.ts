import { reconcileModuleView } from '../../services/canvasLayout';
import { projectModuleFocus } from '../../services/moduleFocus';
import type { ModelEdge, ModelNode } from '../../types';

const mockSizes: string[] = [];
const mockDagre = { calls: 0 };
jest.mock('../../canvas/cardMetrics', () => {
  const actual = jest.requireActual('../../canvas/cardMetrics');
  return { ...actual, nodeCardSize: (node: ModelNode) => { mockSizes.push(node.id); return actual.nodeCardSize(node); } };
});
jest.mock('dagre', () => {
  const actual = jest.requireActual('dagre');
  return { ...actual, layout: (...args: any[]) => { mockDagre.calls += 1; return actual.layout(...args); } };
});

const node = (id: string, parentId?: string, expanded?: boolean): ModelNode => ({
  id, type: expanded === undefined ? 'layerNode' : 'moduleNode', position: { x: 37, y: 91 },
  data: { parentId, layerType: { id: expanded === undefined ? 'linear' : 'module', name: id } as any,
    parameters: { name: id }, ...(expanded === undefined ? {} : { expanded, moduleInstanceId: id }) },
});
const chain = (count: number) => ({
  nodes: Array.from({ length: count }, (_, i) => ({ ...node(`n${i}`), position: { x: i * 251 - 400, y: i % 3 * 73 } })),
  edges: Array.from({ length: count - 1 }, (_, i) => ({ id: `e${i}`, source: `n${i}`, target: `n${i + 1}` })) as ModelEdge[],
});

beforeEach(() => { mockSizes.length = 0; mockDagre.calls = 0; });

test.each([undefined, false])('saved top-level positions skip unused size preparation with relayout=%s', relayout => {
  const graph = chain(64), before = JSON.stringify(graph);
  const result = reconcileModuleView(graph.nodes, graph.edges, { relayout });
  expect(result.nodes.map(n => [n.id, n.position])).toEqual(graph.nodes.map(n => [n.id, n.position]));
  expect(result.edges.map(e => [e.id, e.source, e.target, e.hidden])).toEqual(graph.edges.map(e => [e.id, e.source, e.target, false]));
  expect(JSON.stringify(graph)).toBe(before);
  expect(mockDagre.calls).toBe(0);
  expect(mockSizes).toEqual([]);
});

test.each(['horizontal', 'vertical'] as const)('explicit %s relayout still prepares and places every top-level card', direction => {
  const graph = chain(4), before = JSON.stringify(graph);
  const result = reconcileModuleView(graph.nodes, graph.edges, { relayout: true, direction });
  expect(mockDagre.calls).toBeGreaterThan(0);
  expect(mockSizes).toEqual(expect.arrayContaining(graph.nodes.map(n => n.id)));
  expect(result.nodes.map(n => n.position)).not.toEqual(graph.nodes.map(n => n.position));
  const axis = direction === 'horizontal' ? 'x' : 'y';
  for (let i = 1; i < result.nodes.length; i++) expect(result.nodes[i].position[axis]).toBeGreaterThan(result.nodes[i - 1].position[axis]);
  expect(JSON.stringify(graph)).toBe(before);
});

test.each([{}, { relayout: false }, { preserveChildPositions: true }, { skipChildLayout: true }])(
  'non-relayout options %o do not classify unused top-level layout connections', options => {
    const graph = chain(12);
    let reads = 0;
    graph.edges.forEach(edge => { edge.data = { get _relationship() { reads += 1; return 'feeds'; } }; });
    const result = reconcileModuleView(graph.nodes, graph.edges, options);
    expect(reads).toBe(0);
    expect(result.nodes.map(n => n.position)).toEqual(graph.nodes.map(n => n.position));
    expect(result.edges.map(e => e.id)).toEqual(graph.edges.map(e => e.id));
  },
);

test('collapsed contents still fold display connections and keep canonical positions and edges', () => {
  const nodes = [node('outside'), node('root', undefined, false), node('a', 'root'), node('b', 'root')];
  const edges = [{ id: 'in', source: 'outside', target: 'a' }, { id: 'ab', source: 'a', target: 'b' }];
  const before = JSON.stringify({ nodes, edges });
  const result = reconcileModuleView(nodes, edges, { preserveChildPositions: true });
  expect(result.nodes.find(n => n.id === 'outside')!.position).toEqual(nodes[0].position);
  expect(result.nodes.find(n => n.id === 'root')!.position).toEqual(nodes[1].position);
  for (const id of ['a', 'b']) expect(result.nodes.find(n => n.id === id)).toMatchObject({ parentNode: 'root', hidden: true, position: { x: 37, y: 91 } });
  expect(result.edges.find(e => e.id === 'in')).toMatchObject({ source: 'outside', target: 'a', hidden: true });
  expect(result.edges.find(e => e.data?.synthetic)).toMatchObject({ source: 'outside', target: 'root', data: { originalEdgeIds: ['in'] } });
  expect(JSON.stringify({ nodes, edges })).toBe(before);
  expect(mockSizes).not.toContain('outside');
});

test('module-focus projection keeps boundary wires without preparing unused overview card sizes', () => {
  const nodes = [node('outside'), node('root', undefined, false), node('a', 'root'), node('b', 'root')];
  const edges = [{ id: 'in', source: 'outside', target: 'a' }, { id: 'ab', source: 'a', target: 'b' }];
  const before = JSON.stringify({ nodes, edges });
  const result = projectModuleFocus(nodes, edges, 'root')!;
  expect(result.nodes.filter(n => !n.data.focusBoundary).map(n => n.id)).toEqual(['a', 'b']);
  expect(result.nodes.filter(n => n.data.focusBoundary)).toHaveLength(1);
  expect(result.edges.flatMap(e => e.data?.originalEdgeIds)).toEqual(['in', 'ab']);
  expect(JSON.stringify({ nodes, edges })).toBe(before);
  expect(mockDagre.calls).toBe(0);
  expect(mockSizes).not.toContain('outside');
});

test('empty saved graph keeps the empty result', () => {
  expect(reconcileModuleView([], [])).toEqual({ nodes: [], edges: [] });
  expect(mockSizes).toEqual([]);
  expect(mockDagre.calls).toBe(0);
});

test.each(['horizontal', 'vertical'] as const)('explicit %s relayout keeps the flow story separate from scaffolding', direction => {
  const nodes = ['main', 'data', 'model', 'trainer'].map(id => node(id));
  const edge = (source: string, target: string, relationship: string): ModelEdge => ({
    id: `${source}-${target}`, source, target, data: { _relationship: relationship },
  });
  const edges = [edge('data', 'model', 'feeds'), edge('model', 'trainer', 'trains'),
    edge('main', 'data', 'builds'), edge('main', 'model', 'builds'), edge('main', 'trainer', 'builds')];
  const options = { relayout: true, direction };
  const plain = reconcileModuleView(nodes, edges, options);
  const removed = { ...edge('main', 'trainer', 'feeds'), id: 'removed-main-trainer-feed',
    data: { _relationship: 'feeds', diffStatus: 'removed' as const } };
  const result = reconcileModuleView(nodes, [...edges, removed], options);
  expect(result.nodes.map(n => [n.id, n.position])).toEqual(plain.nodes.map(n => [n.id, n.position]));
  const byId = new Map(result.nodes.map(n => [n.id, n.position]));
  const along = direction === 'horizontal' ? 'x' : 'y', across = direction === 'horizontal' ? 'y' : 'x';
  expect(byId.get('data')![along]).toBeLessThan(byId.get('model')![along]);
  expect(byId.get('model')![along]).toBeLessThan(byId.get('trainer')![along]);
  expect(byId.get('data')![across]).toBe(byId.get('model')![across]);
  expect(byId.get('model')![across]).toBe(byId.get('trainer')![across]);
  expect(byId.get('main')![across]).toBeGreaterThan(byId.get('trainer')![across]);
  expect(result.edges.find(e => e.id === removed.id)?.data?.diffStatus).toBe('removed');
});
