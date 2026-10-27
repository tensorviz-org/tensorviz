import { topologicalOrder } from '../../services/topologicalOrder';

// The sweep paints nodes in the order the shape engine resolves them, so the
// ordering MUST place every source strictly before its targets. These tests
// encode that contract (they fail if a source ever trails its target).
function assertRespectsDependencies(order: string[], edges: { source: string; target: string }[]) {
  const pos = new Map(order.map((id, i) => [id, i] as const));
  for (const { source, target } of edges) {
    expect(pos.get(source)!).toBeLessThan(pos.get(target)!);
  }
}

describe('topologicalOrder', () => {
  it('orders a simple chain inputs-first', () => {
    const edges = [
      { source: 'a', target: 'b' },
      { source: 'b', target: 'c' },
    ];
    expect(topologicalOrder(['c', 'b', 'a'], edges)).toEqual(['a', 'b', 'c']);
  });

  it('places every source before its targets in a diamond', () => {
    const edges = [
      { source: 'in', target: 'l' },
      { source: 'in', target: 'r' },
      { source: 'l', target: 'out' },
      { source: 'r', target: 'out' },
    ];
    const order = topologicalOrder(['out', 'r', 'l', 'in'], edges);
    expect(order[0]).toBe('in');
    expect(order[order.length - 1]).toBe('out');
    assertRespectsDependencies(order, edges);
  });

  it('returns every node exactly once', () => {
    const ids = ['a', 'b', 'c', 'd'];
    const order = topologicalOrder(ids, [
      { source: 'a', target: 'b' },
      { source: 'a', target: 'c' },
      { source: 'b', target: 'd' },
      { source: 'c', target: 'd' },
    ]);
    expect([...order].sort()).toEqual([...ids].sort());
  });

  it('still covers every node when the graph has a cycle (no infinite loop)', () => {
    const ids = ['a', 'b', 'c'];
    const order = topologicalOrder(ids, [
      { source: 'a', target: 'b' },
      { source: 'b', target: 'c' },
      { source: 'c', target: 'a' }, // cycle
    ]);
    expect([...order].sort()).toEqual(['a', 'b', 'c']);
    expect(order.length).toBe(3);
  });

  it('ignores edges referencing unknown nodes', () => {
    const order = topologicalOrder(['a', 'b'], [
      { source: 'a', target: 'b' },
      { source: 'ghost', target: 'b' },
      { source: 'a', target: 'ghost' },
    ]);
    expect(order).toEqual(['a', 'b']);
  });

  it('handles isolated nodes (no edges)', () => {
    expect(topologicalOrder(['x', 'y', 'z'], [])).toEqual(['x', 'y', 'z']);
  });
});
