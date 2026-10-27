import { findQuickJumpNodes } from '../../components/ui2/quickJump';

test('a renamed operation remains reachable through its exact dispatcher target', () => {
  const nodes = [{ id: 'fx_1', title: 'Embedding lookup', searchText: 'aten.embedding.default' }];
  expect(findQuickJumpNodes(nodes, 'embedding lookup')).toEqual(nodes);
  expect(findQuickJumpNodes(nodes, 'aten.embedding')).toEqual(nodes);
});

test('an exact parent identity precedes its many children before the result limit', () => {
  const children = Array.from({ length: 30 }, (_, i) => ({ id: `trunk1.block${i}`, title: 'Module' }));
  const parent = { id: 'trunk1', title: 'Module' };
  const nodes = [...children, parent];
  const result = findQuickJumpNodes(nodes, ' TRUNK1 ');
  expect(result[0]).toBe(parent);
  expect(result).toHaveLength(20);
  expect(result.slice(1)).toEqual(children.slice(0, 19));
  expect(nodes).toEqual([...children, parent]);
});

test('type searches retain spatial ordering and empty queries retain the existing limit', () => {
  const nodes = Array.from({ length: 25 }, (_, i) => ({ id: `fc${i}`, title: 'Linear' }));
  expect(findQuickJumpNodes(nodes, 'linear')).toEqual(nodes.slice(0, 20));
  expect(findQuickJumpNodes(nodes, '')).toEqual(nodes.slice(0, 20));
  expect(findQuickJumpNodes(nodes, 'missing')).toEqual([]);
});

test('local Canvas can count and page every repeated layer without changing the builder limit', () => {
  const nodes = Array.from({ length: 96 }, (_, i) => ({ id: `transformer.h.${i}.q_proj`, title: 'Linear' }));
  expect(findQuickJumpNodes(nodes, 'linear', nodes.length)).toEqual(nodes);
  expect(findQuickJumpNodes(nodes, '', nodes.length)).toHaveLength(96);
  expect(findQuickJumpNodes(nodes, 'linear')).toHaveLength(20);
  expect(findQuickJumpNodes(nodes, 'transformer.h.90.q_proj', nodes.length)[0]).toBe(nodes[90]);
});

test('a captured module remains searchable by its human-readable path as well as its opaque identity', () => {
  const nodes = [{ id: 'invocation:abc123', title: 'Linear', path: 'transformer.h.17.q_proj' }];
  expect(findQuickJumpNodes(nodes, 'h.17.q_proj', nodes.length)).toEqual(nodes);
  expect(findQuickJumpNodes(nodes, 'invocation:abc123', nodes.length)).toEqual(nodes);
});
