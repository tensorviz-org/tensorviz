import { createGraphFile, parseGraphFile } from '../../services/graphFile';
import { nestedGraph } from '../../test/fixtures/graphFileFixture';

/** Count bucket registrations instead of elapsed time: copying the existing
 * sibling array on every append used to register a new bucket once per child. */
function importWideModule(width: number) {
  const graph = nestedGraph();
  const template = graph.nodes.find(node => node.id === 'fc')!;
  const ids = Array.from({ length: width }, (_, index) => `child-${index}`);
  graph.nodes.push(...ids.map(id => ({ ...template, id })));
  const contents = JSON.stringify(createGraphFile(graph, { name: 'Wide', status: 'static' }));
  const originalSet = Map.prototype.set;
  let registrations = 0;
  const spy = jest.spyOn(Map.prototype, 'set').mockImplementation(function (this: Map<unknown, unknown>, key, value) {
    if (key === 'inner' && Array.isArray(value)) registrations += 1;
    return originalSet.call(this, key, value);
  });
  try {
    return { file: parseGraphFile(contents), registrations, ids };
  } finally {
    spy.mockRestore();
  }
}

it.each([4, 400])('registers one child bucket for a module with %s siblings', width => {
  const { file, registrations, ids } = importWideModule(width);
  const byId = new Map(file.document.semantic.nodes.map(node => [node.node_id, node]));
  expect(registrations).toBe(1);
  expect(byId.get('inner')!.metadata.moduleMembers).toEqual(['fc', ...ids]);
  expect(byId.get('inner')!.metadata.moduleLayerCount).toBe(width + 1);
  expect(byId.get('block')!.metadata.moduleMembers).toEqual(['inner']);
  expect(byId.get('block')!.metadata.moduleLayerCount).toBe(width + 1);
});

it('reconstructs independent member lists on each import without trusting supplied members', () => {
  const source = createGraphFile(nestedGraph(), { name: 'Nested', status: 'static' });
  source.document.semantic.nodes.find(node => node.node_id === 'inner')!.metadata.moduleMembers = ['untrusted'];
  const contents = JSON.stringify(source);
  const first = parseGraphFile(contents);
  const second = parseGraphFile(contents);
  const members = (file: typeof first) => file.document.semantic.nodes.find(node => node.node_id === 'inner')!.metadata.moduleMembers as string[];
  expect(members(first)).toEqual(['fc']);
  members(first).push('local-edit');
  expect(members(second)).toEqual(['fc']);
  expect(source.document.semantic.nodes.find(node => node.node_id === 'inner')!.metadata.moduleMembers).toEqual(['untrusted']);
});
