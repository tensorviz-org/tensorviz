import { indexConnections, seedEdgeDirections, traceConnections } from '../../canvas/neighborhood';

const edges = [
  { id: 'ab', source: 'a', target: 'b' }, { id: 'bc', source: 'b', target: 'c' },
  { id: 'cd', source: 'c', target: 'd' }, { id: 'ac', source: 'a', target: 'c', role: 'residual' },
  { id: 'xy', source: 'x', target: 'y' },
];
it('defaults to direct neighbors without flooding an entire connected model', () => {
  const focus = traceConnections(indexConnections(edges), ['b'], 'local');
  expect(Array.from(focus.edgeIds)).toEqual(['ab', 'bc']);
  expect(focus.nodeIds.has('d')).toBe(false);
});
it('traces only the requested direction, including skips', () => {
  expect(Array.from(traceConnections(indexConnections(edges), ['c'], 'upstream').edgeIds).sort()).toEqual(['ab', 'ac', 'bc']);
  expect(Array.from(traceConnections(indexConnections(edges), ['c'], 'downstream').edgeIds)).toEqual(['cd']);
});
it('splits the selection\'s own edges by direction and leaves the rest neutral', () => {
  const index = indexConnections(edges);
  const { incoming, outgoing } = seedEdgeDirections(index, ['c'], index.edges.keys());
  expect(Array.from(incoming).sort()).toEqual(['ac', 'bc']);
  expect(Array.from(outgoing)).toEqual(['cd']);
  expect(incoming.has('ab')).toBe(false);
});
it('leaves an edge between two selected seeds neutral, since it is both', () => {
  const { incoming, outgoing } = seedEdgeDirections(indexConnections(edges), ['a', 'c'], edges.map(edge => edge.id));
  expect(incoming.has('ac')).toBe(false);
  expect(outgoing.has('ac')).toBe(false);
});
it('terminates cycles, preserves parallel edge identities and reports bounded work', () => {
  const graph = indexConnections([...edges, { id: 'parallel', source: 'a', target: 'b' }, { id: 'cycle', source: 'd', target: 'a' }]);
  const complete = traceConnections(graph, ['a'], 'downstream');
  expect(complete.edgeIds.size).toBe(6);
  expect(complete.edgeIds.has('xy')).toBe(false);
  const bounded = traceConnections(graph, ['a'], 'downstream', 2);
  expect(bounded.edgeIds.size).toBe(2);
  expect(bounded.truncated).toBe(true);
});
