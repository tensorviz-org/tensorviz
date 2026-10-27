import { createGraphFile, parseGraphFile, graphFileName, GRAPH_FILE_MAX_BYTES } from '../../services/graphFile';
import { graphDocumentV2ToModelGraph } from '../../services/graphDocumentAdapter';
import { nestedGraph } from '../../test/fixtures/graphFileFixture';

const snapshot = () => createGraphFile(nestedGraph(), { name: 'Net', status: 'static' });
const mutate = (fn: (file: any) => void) => { const file = snapshot(); fn(file); return JSON.stringify(file); };

it('round trips complete collapsed contents, connections, parameters and layout', () => {
  const file = parseGraphFile(JSON.stringify(snapshot()));
  const graph = graphDocumentV2ToModelGraph(file.document);
  expect(graph.nodes.map(n => n.id)).toEqual(['input', 'block', 'inner', 'fc', 'output']);
  expect(graph.nodes[3]).toMatchObject({ position: { x: 100, y: 80 }, data: { parentId: 'inner', parameters: { in_features: 8, out_features: 4 } } });
  expect(graph.nodes[1].data.expanded).toBe(false);
  expect(graph.edges.map(e => [e.source, e.target])).toEqual([['input', 'fc'], ['fc', 'output']]);
  expect(file.status).toBe('static');
});

it('retains full-model node counts and still refuses oversized node lists', () => {
  const graph = nestedGraph();
  const template = graph.nodes[3];
  graph.nodes = Array.from({ length: 13_200 }, (_, index) => ({ ...template, id: `op-${index}`,
    data: { ...template.data, parentId: undefined, parameters: { index } } }));
  graph.edges = [];
  const file = createGraphFile(graph, { name: 'Full model', status: 'static' });
  const reopened = parseGraphFile(JSON.stringify(file));
  expect(reopened.document.semantic.nodes.map(node => node.node_id)).toEqual(graph.nodes.map(node => node.id));
  expect(reopened.document.semantic.nodes[13_199].params.index).toBe(13_199);
  file.document.semantic.nodes = Array(20_001).fill(file.document.semantic.nodes[0]);
  expect(() => parseGraphFile(JSON.stringify(file))).toThrow(/too many/);
});

it.each(['static', 'traced', 'stale'] as const)('preserves %s provenance and measured shapes', status => {
  const graph = nestedGraph(); graph.nodes[3].data.inputShape = [1, 8]; graph.nodes[3].data.outputShape = [1, 4];
  const file = parseGraphFile(JSON.stringify(createGraphFile(graph, { name: 'Net', status })));
  expect(file.status).toBe(status);
  expect(graphDocumentV2ToModelGraph(file.document).nodes[3].data.outputShape).toEqual([1, 4]);
});

it('does not export source, absolute paths, arbitrary metadata, session or provider fields', () => {
  const graph = nestedGraph();
  graph.metadata = { source: 'SECRET_SOURCE', sessionId: 'SECRET_SESSION', token: 'SECRET_TOKEN' };
  Object.assign(graph.nodes[3].data, { source: 'SECRET_SOURCE', error: 'SECRET_ERROR', onToggleExpand: () => undefined });
  Object.assign(graph.nodes[3].data.parameters, { _defined_in: '/Users/private/net.py', api_key: 'SECRET_KEY', nested: { path: '/Users/private/weights.pt', token: 'SECRET_TOKEN', value: 8 } });
  const text = JSON.stringify(createGraphFile(graph, { name: 'Net', status: 'static' }));
  expect(text).not.toMatch(/SECRET|\/Users\/private|_defined_in|api_key/);
  expect(parseGraphFile(text).document.semantic.nodes[3].params.nested).toEqual({ value: 8 });
});

it('filters display-only edges and retains expansion state', () => {
  const graph = nestedGraph(); graph.nodes[1].data.expanded = true;
  graph.edges.push({ id: 'display', source: 'input', target: 'block', data: { synthetic: true } });
  const file = createGraphFile(graph, { name: 'Net', status: 'static' });
  expect(file.document.semantic.edges).toHaveLength(2);
  expect(file.document.view.collapsed_modules).toEqual(['inner']);
});

it.each([
  ['invalid JSON', '{', /valid JSON/],
  ['wrong format', JSON.stringify({ format: 'other', version: 1 }), /tensorViz graph/],
  ['unsupported version', JSON.stringify({ format: 'tensorviz.graph', version: 9 }), /version/],
])('rejects %s with a useful error', (_, text, error) => expect(() => parseGraphFile(text)).toThrow(error));

it.each([
  ['duplicate nodes', (f: any) => f.document.semantic.nodes.push(f.document.semantic.nodes[0]), /Duplicate/],
  ['dangling edge', (f: any) => f.document.semantic.edges[0].target_node_id = 'missing', /missing node/],
  ['missing position', (f: any) => delete f.document.view.node_positions.fc, /position/],
  ['invalid coordinate', (f: any) => f.document.view.node_positions.fc.x = 'bad', /coordinate/],
  ['unknown parent', (f: any) => f.document.semantic.nodes[3].metadata.parentId = 'missing', /parent/],
  ['parent cycle', (f: any) => f.document.semantic.nodes[1].metadata.parentId = 'inner', /cycle/],
  ['non-module parent', (f: any) => f.document.semantic.nodes[3].metadata.parentId = 'input', /parent/],
  ['invalid status', (f: any) => f.status = 'verified', /status/],
  ['invalid params', (f: any) => f.document.semantic.nodes[0].params = [], /parameters/],
])('rejects %s before rendering', (_, fn, error) => expect(() => parseGraphFile(mutate(fn))).toThrow(error));

it('rejects prototype keys and oversized files', () => {
  expect(() => parseGraphFile('{"__proto__":{"polluted":true}}')).toThrow(/Unsafe/);
  expect(() => parseGraphFile(' '.repeat(GRAPH_FILE_MAX_BYTES + 1))).toThrow(/96 MiB/);
  expect(({} as any).polluted).toBeUndefined();
});

it('accepts files beyond the former 64 MiB bound needed by full decode graphs', () => {
  const file = snapshot();
  const contents = JSON.stringify(file) + ' '.repeat(64 * 1024 * 1024);
  expect(parseGraphFile(contents)).toEqual(file);
});

it('does not trust imported metadata or embedded migration payloads', () => {
  const file = parseGraphFile(mutate(f => { f.document.semantic.metadata = { migration: { legacyPayload: { nodes: [], edges: [] } } }; f.document.semantic.nodes[3].metadata.source = 'PRIVATE'; }));
  expect(graphDocumentV2ToModelGraph(file.document).nodes).toHaveLength(5);
  expect(JSON.stringify(file)).not.toContain('PRIVATE');
});

it('provides a safe portable filename', () => {
  expect(graphFileName('My Model')).toBe('My-Model.tensorviz.json');
  expect(graphFileName('../../')).toBe('model.tensorviz.json');
});

it.each([null, ''])('exports live top-level nodes with parent=%s and neutral optional metadata', parentId => {
  const graph = nestedGraph();
  Object.assign(graph.nodes[0].data, { parentId, repeatCount: 0, repeatBlockLabel: '', repeatHidden: null, inputShape: null, outputShape: null });
  const file = createGraphFile(graph, { name: 'Net', status: 'static' });
  expect(file.document.semantic.nodes[0].metadata).not.toHaveProperty('parentId');
  expect(file.document.semantic.nodes[0].metadata).not.toHaveProperty('repeatCount');
  expect(file.document.semantic.nodes[3].metadata.parentId).toBe('inner');
});

it('preserves the imported class label on a collapsed module', () => {
  const graph = nestedGraph(); graph.nodes[1].data.moduleTemplateId = 'models.blocks::ResidualBlock';
  const file = parseGraphFile(JSON.stringify(createGraphFile(graph, { name: 'Net', status: 'static' })));
  expect(graphDocumentV2ToModelGraph(file.document).nodes[1].data.moduleTemplateId).toBe('models.blocks::ResidualBlock');
});

// Orientation belongs to the portable view, never to model semantics.
it.each(['horizontal', 'vertical'] as const)('round trips %s layout without exporting arbitrary layout metadata', direction => {
  const file = createGraphFile(nestedGraph(), { name: 'Net', status: 'static', direction });
  file.document.view.layout.privatePath = '/Users/private/model.py';
  const restored = parseGraphFile(JSON.stringify(file));
  expect(restored.document.view.layout).toEqual({ direction });
  expect(restored.document.semantic).toEqual(snapshot().document.semantic);
});

it('rejects an invalid portable layout direction', () => {
  expect(() => parseGraphFile(mutate(file => { file.document.view.layout.direction = 'diagonal'; }))).toThrow(/layout direction/);
});
