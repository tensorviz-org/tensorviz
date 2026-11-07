import { mergeProjection } from '../../ide/applyProjection';
import type { ModelGraph, ModelNode } from '../../types';

const mockLayout = { calls: 0 };
jest.mock('../../services/canvasLayout', () => {
  const actual = jest.requireActual('../../services/canvasLayout');
  return { ...actual, layoutTopLevel: (...args: Parameters<typeof actual.layoutTopLevel>) => {
    mockLayout.calls += 1;
    return actual.layoutTopLevel(...args);
  } };
});
const node = (id: string, position: ModelNode['position'] = { x: 17, y: 29 }): ModelNode => ({
  id, type: 'layerNode', position,
  data: { layerType: { id: 'linear', name: 'Linear' } as any, parameters: {} },
});
const graph = (nodes: ModelNode[]): ModelGraph => ({ nodes, edges: [] });
beforeEach(() => { mockLayout.calls = 0; });

test('saved-view additions classify positions with linear ID reads', () => {
  const count = 128;
  let reads = 0;
  const incoming = graph(Array.from({ length: count }, (_, i) => {
    const value = node(`n${i}`);
    Object.defineProperty(value, 'id', { get() { reads += 1; return `n${i}`; }, enumerable: true });
    return value;
  }));
  const result = mergeProjection(incoming, [], { viewRestored: true });
  expect(reads).toBeLessThanOrEqual(6 * count);
  expect(mockLayout.calls).toBe(0);
  result.nodes.forEach((value, i) => expect(value).toBe(incoming.nodes[i]));
  expect(result.added).toEqual(Array.from({ length: count }, (_, i) => `n${i}`));
  expect(result.edges).toBe(incoming.edges);
});

test('a placed first duplicate keeps a later unplaced duplicate untouched', () => {
  const incoming = graph([node('same'), node('same', { x: 0, y: 0 })]);
  const result = mergeProjection(incoming, [], { viewRestored: true });
  expect(mockLayout.calls).toBe(0);
  expect(result.added).toEqual(['same', 'same']);
  expect(result.nodes[0]).toBe(incoming.nodes[0]);
  expect(result.nodes[1]).toBe(incoming.nodes[1]);
});

test('an unplaced first duplicate lays out every occurrence of that ID', () => {
  const incoming = graph([node('same', { x: 0, y: 0 }), node('same')]);
  const result = mergeProjection(incoming, [], { viewRestored: true });
  expect(mockLayout.calls).toBe(1);
  expect(result.added).toEqual(['same', 'same']);
  expect(result.nodes[0].position).not.toEqual({ x: 0, y: 0 });
  expect(result.nodes[1].position).toEqual(result.nodes[0].position);
});

test.each([undefined, null, { x: 0, y: 0 }])('new position %j still needs placement', position => {
  const value = { ...node('new'), position } as ModelNode;
  const result = mergeProjection(graph([value]), [], { viewRestored: true });
  expect(mockLayout.calls).toBe(1);
  expect(result.nodes[0].position).not.toEqual(position);
  expect(result.nodes[0].position).not.toEqual({ x: 0, y: 0 });
});

test.each([{ x: 0, y: 7 }, { x: -2, y: 0 }])('a single zero axis remains placed: %j', position => {
  const value = node('new', position);
  const result = mergeProjection(graph([value]), [], { viewRestored: true });
  expect(mockLayout.calls).toBe(0);
  expect(result.nodes[0]).toBe(value);
});

test('last existing duplicate still supplies canvas state', () => {
  const first = { ...node('old', { x: 10, y: 20 }), selected: true };
  const last = { ...node('old', { x: 90, y: 70 }), selected: false, hidden: true };
  const result = mergeProjection(graph([node('old'), node('old')]), [first, last]);
  expect(mockLayout.calls).toBe(0);
  expect(result.added).toEqual([]);
  result.nodes.forEach(value => {
    expect(value.position).toBe(last.position);
    expect(value.selected).toBeUndefined();
    expect(value.hidden).toBe(true);
  });
});

test('addition, removal and node order retain duplicate IDs', () => {
  const result = mergeProjection(graph([node('new'), node('old'), node('new'), node('other')]),
    [node('gone'), node('old'), node('gone')]);
  expect(result.added).toEqual(['new', 'new', 'other']);
  expect(result.removed).toEqual(['gone', 'gone']);
  expect(result.nodes.map(n => n.id)).toEqual(['new', 'old', 'new', 'other']);
  expect(mockLayout.calls).toBe(0);
});

test('partial restore only places new unplaced cards', () => {
  const incoming = graph([node('old'), node('saved'), node('new', { x: 0, y: 0 })]);
  const existing = node('old', { x: 230, y: 450 });
  const result = mergeProjection(incoming, [existing], { viewRestored: true, direction: 'vertical' });
  expect(mockLayout.calls).toBe(1);
  expect(result.nodes[0].position).toBe(existing.position);
  expect(result.nodes[1]).toBe(incoming.nodes[1]);
  expect(result.nodes[2].position).not.toEqual({ x: 0, y: 0 });
});

test('initial opening retains its whole-graph layout branch', () => {
  const result = mergeProjection(graph([node('a'), node('b')]), []);
  expect(mockLayout.calls).toBe(1);
  expect(result.added).toEqual(['a', 'b']);
  expect(result.nodes[0].position).not.toEqual({ x: 17, y: 29 });
});

test('empty projection retains repeated removals without layout', () => {
  const result = mergeProjection(graph([]), [node('old'), node('old')]);
  expect(result).toEqual({ nodes: [], edges: [], added: [], removed: ['old', 'old'] });
  expect(mockLayout.calls).toBe(0);
});
