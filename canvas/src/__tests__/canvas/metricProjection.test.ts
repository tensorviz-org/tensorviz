import fixtures from '../../test/fixtures/metricProjections.json';
import { applyProjection } from '../../ide/projection';
import { overlayScale } from '../../canvas/graphOverlay';
import { measuredOutputTensor, tensorShapes } from '../../canvas/graphSemantics';
import { createGraphFile, parseGraphFile } from '../../services/graphFile';
import { graphDocumentV2ToModelGraph } from '../../services/graphDocumentAdapter';
import { ProjectionPayload } from '../../ide/protocol.generated';

jest.mock('../../ide/bridge', () => ({ postToHost: jest.fn() }));

const payload = (name: keyof typeof fixtures, greyed = false): ProjectionPayload => ({
  document: fixtures[name], graphId: 'GoodNet', file: 'goodnet.py', cls: 'GoodNet',
  sourceDigest: 'same-source', traced: true, greyed, viewRestored: true,
} as unknown as ProjectionPayload);

beforeEach(() => { window.__TENSORVIZ_LOCAL_BROWSER__ = true; });
afterEach(() => { delete window.__TENSORVIZ_LOCAL_BROWSER__; });

it('replaces batch evidence from the actual backend projection and preserves exact small/zero counts', () => {
  const old = applyProjection(payload('batch2'), [])!;
  const current = applyProjection(payload('batch3'), old.nodes)!;
  const scale = overlayScale(current.nodes, 'activation');
  expect([scale.min, scale.max]).toEqual([24, 48]);
  expect(current.nodes.find(n => n.id === 'fc1')!.data.outputShape).toEqual([3, 4]);
  expect(current.nodes.find(n => n.id === 'fc2')!.data.outputShape).toEqual([3, 2]);
  expect(Array.from(overlayScale(current.nodes, 'params').values.values()).map(v => v.value)).toEqual([20, 10, 0]);
  expect(current.nodes.map(n => n.position)).toEqual(old.nodes.map(n => n.position));
});

it.each([['batch3', 24, 48], ['double', 48, 96], ['empty', 0, 0]] as const)(
  'preserves %s byte/dtype evidence through export/import and excludes private source fields', (name, min, max) => {
    const graph = applyProjection(payload(name), [])!;
    graph.nodes[1].data.parameters._source_file = '/private/SECRET.py';
    (graph.nodes[1].data.outputTensor as any).source = '/private/SECRET.py';
    const file = parseGraphFile(JSON.stringify(createGraphFile(graph, { name: 'GoodNet', status: 'traced' })));
    const restored = graphDocumentV2ToModelGraph(file.document);
    expect(JSON.stringify(file)).not.toContain('SECRET');
    expect(overlayScale(restored.nodes, 'activation')).toMatchObject({ min, max, unmeasured: 1 });
    expect(overlayScale(restored.nodes, 'params').values.get('relu')!.value).toBe(0);
    graph.nodes.forEach((node, i) => {
      expect(measuredOutputTensor(restored.nodes[i].data)).toEqual(measuredOutputTensor(node.data));
    });
  });

it('uses one observed-shape rule for cards, bytes and portable export even if presentation shapes lag', () => {
  const graph = applyProjection(payload('batch3'), [])!;
  const fc = graph.nodes.find(n => n.id === 'fc1')!;
  fc.data.outputShape = [2, 4];
  expect(tensorShapes(fc.data).outputShape).toEqual([3, 4]);
  expect(overlayScale([fc], 'activation').max).toBe(48);
  const restored = graphDocumentV2ToModelGraph(createGraphFile(graph, { name: 'GoodNet', status: 'traced' }).document);
  expect(restored.nodes.find(n => n.id === 'fc1')!.data.outputShape).toEqual([3, 4]);
});

it('does not carry old bytes through missing/incomplete output, stale refresh or static replacement', () => {
  const old = applyProjection(payload('batch2'), [])!;
  const incomplete = JSON.parse(JSON.stringify(payload('batch3')));
  delete incomplete.document.view.layout.node_data.fc1.outputTensor;
  const next = applyProjection(incomplete, old.nodes)!;
  expect(overlayScale(next.nodes, 'activation').values.has('fc1')).toBe(false);
  const stale = applyProjection(payload('batch3', true), next.nodes)!;
  expect(overlayScale(stale.nodes, 'activation').values.size).toBe(0);
  expect(overlayScale(stale.nodes, 'params').values.size).toBe(0);
  const portable = createGraphFile(next, { name: 'GoodNet', status: 'stale' });
  const restored = graphDocumentV2Model(portable);
  expect(overlayScale(restored.nodes, 'params').values.size).toBe(0);
  expect(overlayScale(restored.nodes, 'activation').values.size).toBe(0);
  const staticPayload = JSON.parse(JSON.stringify(payload('batch3')));
  staticPayload.traced = false;
  staticPayload.document.semantic.nodes.forEach((n: any) => { n.params = {}; });
  staticPayload.document.view.layout.node_data = {};
  const stat = applyProjection(staticPayload, stale.nodes)!;
  expect(overlayScale(stat.nodes, 'activation').values.size).toBe(0);
  expect(overlayScale(stat.nodes, 'params').values.size).toBe(0);
});

function graphDocumentV2Model(file: ReturnType<typeof createGraphFile>) {
  return graphDocumentV2ToModelGraph(parseGraphFile(JSON.stringify(file)).document);
}

it.each([1, 2, 3])('reads legacy version %s without guessing bytes or recovering rounded counts', version => {
  const graph = applyProjection(payload('batch3'), [])!;
  const file: any = createGraphFile(graph, { name: 'GoodNet', status: 'traced' });
  file.version = version;
  delete file.collaboration;
  file.document.semantic.nodes.forEach((n: any) => {
    delete n.metadata.outputTensor;
    delete n.metadata.parameterCount;
    n.params._params_M = 0;
  });
  const restored = graphDocumentV2Model(file);
  expect(restored.nodes.find(n => n.id === 'fc1')!.data.outputShape).toEqual([3, 4]);
  expect(overlayScale(restored.nodes, 'activation').values.size).toBe(0);
  expect(overlayScale(restored.nodes, 'params').values.size).toBe(0);
});

it.each([
  { parameterCount: -1 }, { parameterCount: 1.5 }, { parameterCount: Number.MAX_SAFE_INTEGER + 1 },
  { outputTensor: { shape: [3, 4], dtype: '/private/model.py', logicalBytes: 48 } },
  { outputTensor: { shape: [3, 4], dtype: 'float32' } },
  { outputTensor: { shape: [3.5, 4], dtype: 'float32', logicalBytes: 48 } },
  { outputTensor: { shape: '3x4', dtype: 'float32', logicalBytes: 48 } },
])('refuses malformed metric metadata at the portable boundary: %j', invalid => {
  const file = createGraphFile(applyProjection(payload('batch3'), [])!, { name: 'GoodNet', status: 'traced' });
  Object.assign(file.document.semantic.nodes[1].metadata, invalid);
  expect(() => parseGraphFile(JSON.stringify(file))).toThrow(/Invalid/);
});
