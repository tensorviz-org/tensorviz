import { reconcileModuleView } from '../../services/canvasLayout';
import { projectModuleFocus } from '../../services/moduleFocus';
import type { ModelEdge, ModelNode } from '../../types';

const mockSkips = { calls: 0 };
jest.mock('../../canvas/edgeRouting', () => {
  const actual = jest.requireActual('../../canvas/edgeRouting');
  return { ...actual, skipEdgeIds: (...args: Parameters<typeof actual.skipEdgeIds>) => {
    mockSkips.calls += 1;
    return actual.skipEdgeIds(...args);
  } };
});

const node = (id: string, parentId?: string, expanded?: boolean): ModelNode => ({
  id, type: expanded === undefined ? 'layerNode' : 'moduleNode', position: { x: 37, y: 91 },
  data: { parentId, layerType: { id: expanded === undefined ? 'linear' : 'module', name: id } as any,
    parameters: { name: id }, ...(expanded === undefined ? {} : { expanded, moduleInstanceId: id }) },
});
const edge = (source: string, target: string): ModelEdge => ({ id: `${source}-${target}`, source, target });
const nested = (outer = false, inner = false) => ({
  nodes: [node('outside'), node('root', undefined, outer), node('nested', 'root', inner),
    node('a', 'nested'), node('b', 'nested')],
  edges: [edge('outside', 'a'), edge('a', 'b')],
});
beforeEach(() => { mockSkips.calls = 0; });

test.each([false, true])('flat graphs skip child preparation even when relayout=%s', relayout => {
  const nodes = [node('a'), node('b'), node('c')], edges = [edge('a', 'b'), edge('b', 'c'), edge('a', 'c')];
  const before = JSON.stringify({ nodes, edges });
  const result = reconcileModuleView(nodes, edges, { relayout });
  expect(mockSkips.calls).toBe(0);
  expect(result.nodes.map(n => n.id)).toEqual(['a', 'b', 'c']);
  expect(result.edges.map(e => e.id)).toEqual(edges.map(e => e.id));
  expect(result.edges.every(e => !e.hidden)).toBe(true);
  expect(JSON.stringify({ nodes, edges })).toBe(before);
});

test.each([false, true])('collapsed ancestors skip preparation with descendant expanded=%s', inner => {
  const graph = nested(false, inner), before = JSON.stringify(graph);
  const result = reconcileModuleView(graph.nodes, graph.edges);
  expect(mockSkips.calls).toBe(0);
  for (const id of ['nested', 'a', 'b']) expect(result.nodes.find(n => n.id === id)?.hidden).toBe(true);
  expect(result.nodes.find(n => n.id === 'root')).toMatchObject({ position: { x: 37, y: 91 } });
  expect(result.nodes.find(n => n.id === 'root')?.hidden).not.toBe(true);
  expect(result.edges.find(e => e.data?.synthetic)).toMatchObject({ source: 'outside', target: 'root',
    data: { originalEdgeIds: ['outside-a'] } });
  expect(JSON.stringify(graph)).toBe(before);
});

test.each([{}, { relayout: true }, { preserveChildPositions: true }])('skipChildLayout omits preparation with options %o', options => {
  const graph = nested(true, true), before = JSON.stringify(graph);
  const result = reconcileModuleView(graph.nodes, graph.edges, { ...options, skipChildLayout: true });
  expect(mockSkips.calls).toBe(0);
  for (const id of ['a', 'b']) expect(result.nodes.find(n => n.id === id)).toMatchObject({ hidden: false, parentNode: 'nested' });
  expect(JSON.stringify(graph)).toBe(before);
});

test.each([{}, { relayout: true }, { preserveChildPositions: true }])('multiple visible expanded frames share one preparation with options %o', options => {
  const graph = nested(true, true), before = JSON.stringify(graph);
  const result = reconcileModuleView(graph.nodes, graph.edges, options);
  expect(mockSkips.calls).toBe(1);
  for (const id of ['root', 'nested']) {
    const frame = result.nodes.find(n => n.id === id)!;
    expect(Number(frame.style?.width)).toBeGreaterThan(0);
    expect(Number(frame.style?.height)).toBeGreaterThan(0);
  }
  expect(JSON.stringify(graph)).toBe(before);
});

test('module focus retains boundary wires without child-layout preparation', () => {
  const graph = nested(), before = JSON.stringify(graph);
  const result = projectModuleFocus(graph.nodes, graph.edges, 'nested')!;
  expect(mockSkips.calls).toBe(0);
  expect(result.nodes.filter(n => !n.data.focusBoundary).map(n => n.id)).toEqual(['a', 'b']);
  expect(result.nodes.filter(n => n.data.focusBoundary)).toHaveLength(1);
  expect(result.edges.flatMap(e => e.data?.originalEdgeIds)).toEqual(['outside-a', 'a-b']);
  expect(JSON.stringify(graph)).toBe(before);
});

test('preparation is local to the current reconciliation', () => {
  const graph = nested(true, true);
  reconcileModuleView(graph.nodes, graph.edges);
  expect(mockSkips.calls).toBe(1);
  reconcileModuleView(graph.nodes, graph.edges);
  expect(mockSkips.calls).toBe(2);
  const collapsed = nested();
  reconcileModuleView(collapsed.nodes, collapsed.edges);
  expect(mockSkips.calls).toBe(2);
});

test('empty graph needs no child preparation', () => {
  expect(reconcileModuleView([], [])).toEqual({ nodes: [], edges: [] });
  expect(mockSkips.calls).toBe(0);
});

test('collapsed modules still aggregate descendant diagnostics without preparation', () => {
  const graph = nested();
  graph.nodes.find(n => n.id === 'a')!.data.error = 'Recorded failure';
  const result = reconcileModuleView(graph.nodes, graph.edges);
  expect(mockSkips.calls).toBe(0);
  for (const id of ['root', 'nested']) expect(result.nodes.find(n => n.id === id)?.data).toMatchObject({
    moduleDiagnosticCount: 1, moduleDiagnosticStatus: 'error',
    moduleDiagnostics: [expect.objectContaining({ nodeId: 'a', message: 'Recorded failure' })],
  });
});

test('saved expanded children still reserve crossing-skip header clearance', () => {
  const nodes = [node('root', undefined, true), node('stem', 'root', false), node('nested', 'root', true),
    node('a', 'nested', false), node('b', 'nested', false), node('add', 'nested')];
  const chain = [edge('stem', 'a'), edge('a', 'b'), edge('b', 'add')];
  const plain = reconcileModuleView(nodes, chain, { preserveChildPositions: true });
  const result = reconcileModuleView(nodes, [...chain, edge('stem', 'add')], { preserveChildPositions: true });
  expect(mockSkips.calls).toBe(2);
  const width = (graph: typeof result) => Number(graph.nodes.find(n => n.id === 'nested')!.style!.width);
  // Saved children keep their x positions; only the trailing 48px reserve grows.
  expect(width(result)).toBe(width(plain) + 48);
  for (const id of ['a', 'b', 'add']) expect(result.nodes.find(n => n.id === id)?.position).toEqual({ x: 37, y: 91 });
});
