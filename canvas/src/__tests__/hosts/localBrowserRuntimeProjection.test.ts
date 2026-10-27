import { applyProjection } from '../../ide/projection';
import { ProjectionPayload } from '../../ide/protocol.generated';
import { modelGraphToDocumentV2 } from '../../services/graphDocumentAdapter';
import { ModelNode } from '../../types';
import { moduleDefinitionNodeId } from '../../canvas/moduleSourceTarget';
import { GraphDocumentV2 } from '../../types/graphDocument';
import { nodeCardSize } from '../../canvas/cardMetrics';
import { graphNodeOrigin } from '../../services/graphOverview';
import { projectionForCanvas } from '../../ide/localBrowserPresentation';
import { reconcileModuleView } from '../../services/canvasLayout';
import { buildInferablePayload } from '../../services/shapeInferencePayload';
import { createGraphFile, parseGraphFile } from '../../services/graphFile';
import { initialGraphFileView } from '../../viewer/graphFileView';
import captureFixture from '../../test/fixtures/canvas/capture-document.json';

jest.mock('../../ide/bridge', () => ({ postToHost: jest.fn() }));

const node = (id: string, parameters: Record<string, unknown>): ModelNode => ({
  id, type: 'layerNode', position: { x: 100, y: 100 },
  data: { layerType: { id: 'linear', name: 'Linear', parameters: [] }, parameters },
} as unknown as ModelNode);

function projection(nodes: ModelNode[], extra: Partial<ProjectionPayload> = {}): ProjectionPayload {
  return { graphId: 'g', cls: 'Net', file: 'net.py', sourceDigest: 'digest', revision: 2,
    traced: true, greyed: false, viewRestored: true, spanmap: {},
    document: modelGraphToDocumentV2({ nodes, edges: [] }, { name: 'Net', revision: 2 }), ...extra };
}

beforeEach(() => { window.__TENSORVIZ_LOCAL_BROWSER__ = true; });
afterEach(() => { window.__TENSORVIZ_LOCAL_BROWSER__ = false; });

it('retains every canonical failed call and capture metadata when applying a partial graph', () => {
  const document = JSON.parse(JSON.stringify(captureFixture)) as GraphDocumentV2;
  const failed = document.semantic.nodes.filter(n => n.metadata.captureRef &&
    ['invocation', 'operation'].includes((n.metadata.captureRef as { kind: string }).kind));
  failed.forEach(n => { n.metadata.error = 'Captured failure'; n.params._shape_provenance = 'failed'; });
  const result = applyProjection(projection([], { document, traceError: {
    type: 'RuntimeError', message: 'Mapped failure', nodeId: failed[failed.length - 1].node_id,
  } }), [])!;
  expect(result.nodes.filter(n => n.data.error).map(n => n.id)).toEqual(failed.map(n => n.node_id));
  expect(result.metadata).toEqual(document.semantic.metadata);
  const stale = applyProjection(projection([], { document, greyed: true }), result.nodes)!;
  expect(stale.nodes.every(n => n.data.shapeEvidence === 'stale')).toBe(true);
  expect(stale.nodes.filter(n => n.data.error)).toHaveLength(0); // Historical failures cannot mark the edited source as broken.
});

it('canonical source coverage survives the real projection adapter with scalar state and unknown call shapes', () => {
  const projected = applyProjection(projection([
    node('deep.fc', { _shape_provenance: 'observed', _in_shape: '2x4', _out_shape: '2x4',
      _capture_coverage: { status: 'observed', invocationIds: ['call:2'] } }),
    node('unused', { _shape_provenance: 'not_observed', _capture_coverage: { status: 'not_observed', invocationIds: [] } }),
    node('parameter', { _shape_provenance: 'state', _out_shape: [],
      _capture_coverage: { status: 'state', stateValueIds: ['v0'], invocationIds: [] } }),
  ]), [])!;
  expect(projected.nodes[0].data.outputShape).toEqual([2, 4]);
  expect(projected.nodes[1].data.outputShape).toBeUndefined();
  expect(projected.nodes[1].data.shapeEvidence).toBe('unknown');
  expect(projected.nodes[2].data.outputShape).toEqual([]);
  expect(projected.nodes[2].data.shapeEvidence).toBe('measured');
  expect(projected.nodes[0].data.parameters._capture_coverage.invocationIds).toEqual(['call:2']);
});

it.each([false, true])('keeps static ModuleList semantics, opacity and wires through projection and export (workspace=%s)', workspace => {
  const layerType = workspace ? { id: 'custom_code', name: 'custom_code', parameters: [] }
    : { id: 'modulelist', name: 'ModuleList', parameters: [] };
  const parameters = { _shape_provenance: 'static', _kind: 'class', _raw_call: 'nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])',
    _modules: [{ op: 'Linear', params: { in_features: 8, out_features: 8 } }, { op: 'Linear', params: { in_features: 8, out_features: 8 } }],
    ...(workspace ? { _class: 'ModuleList', _opaque: true } : {}) };
  // The Python ingest encoder's split metadata/view format, before any frontend save.
  const document = { schemaVersion: 2, semantic: { graph_id: 'g', name: 'Net', revision: 0, inputs: [], modules: [], operations: [], metadata: {},
    nodes: [
      { node_id: 'input', kind: 'input', op: 'Input', params: { shape: '2,8' }, in_ports: [], out_ports: ['y'], metadata: {} },
      { node_id: 'blocks', kind: 'module', op: layerType.name, params: parameters, in_ports: ['x'], out_ports: ['y'], metadata: { layerType } },
      { node_id: 'python@Net#0', kind: 'layer', op: 'Python', params: { _raw_call: 'block(x)', _shape_provenance: 'static' }, in_ports: ['x'], out_ports: ['y'], metadata: {} },
      { node_id: 'output', kind: 'output', op: 'Output', params: {}, in_ports: ['x'], out_ports: [], metadata: {} },
    ], edges: [
      { edge_id: 'in-loop', source_node_id: 'input', target_node_id: 'python@Net#0', source_port: 'y', target_port: 'x', metadata: {} },
      { edge_id: 'loop-out', source_node_id: 'python@Net#0', target_node_id: 'output', source_port: 'y', target_port: 'x', metadata: {} },
    ] }, view: { graph_id: 'g', node_positions: {}, collapsed_modules: ['blocks'], selection: [], layout: {
      node_canvas: { blocks: { type: 'moduleNode' } }, node_data: { blocks: { isGroup: true, moduleInstanceId: 'blocks', moduleTemplateId: 'ModuleList',
        moduleMembers: [], moduleLayerCount: 0, expanded: false, moduleType: 'container' } },
    } } } as GraphDocumentV2;
  const payload = projection([], { document, traced: false });
  const previous = { ...node('blocks', {}), type: 'moduleNode', data: { ...node('blocks', {}).data, expanded: true } };
  const applied = applyProjection(payload, [previous])!;
  const graph = reconcileModuleView(applied.nodes, applied.edges);
  const block = graph.nodes.find(n => n.id === 'blocks')!;
  expect(block.data).toMatchObject({ parameters, moduleTemplateId: 'ModuleList', moduleMembers: [], moduleLayerCount: 0, expanded: false, expandable: false });
  expect(block.data.layerType).toMatchObject(layerType);
  expect(graph.nodes.map(n => n.id)).toEqual(document.semantic.nodes.map(n => n.node_id));
  expect(graph.edges.map(e => [e.source, e.target])).toEqual([['input', 'python@Net#0'], ['python@Net#0', 'output']]);
  const saved = modelGraphToDocumentV2(graph, { name: 'Net', revision: 1 });
  expect(saved.semantic.nodes.find(n => n.node_id === 'blocks')).toMatchObject({ kind: 'module', op: layerType.name, params: parameters });
  const inferable = buildInferablePayload(graph.nodes, graph.edges);
  expect(inferable.inferableNodes.map(n => n.id)).toEqual(['input', 'python@Net#0', 'output']);
  expect(inferable.inferableEdges).toEqual(graph.edges);
  const portable = parseGraphFile(JSON.stringify(createGraphFile(graph, { name: 'Net', status: 'static' })));
  const viewed = initialGraphFileView(portable);
  // Portable privacy strips internal/source parameters, but keeps the declared
  // container identity and never turns its missing internals into children.
  expect(viewed.nodes.find(n => n.id === 'blocks')!.data).toMatchObject({ parameters: {}, moduleTemplateId: 'ModuleList', expanded: false });
  expect(viewed.nodes.map(n => n.id).sort()).toEqual(graph.nodes.map(n => n.id).sort());
  expect(viewed.edges.map(e => [e.source, e.target])).toEqual(graph.edges.map(e => [e.source, e.target]));
});

it('retains F7 dotted module identity, definition navigation and actual boundary/member edges', () => {
  const wrapper = { ...node('block', {}),
    type: 'moduleNode', data: { ...node('block', {}).data,
      layerType: { id: 'custom_code', name: 'custom_code', parameters: [] },
      parameters: { _class: 'ScaleBlock', _import_path: 'blocks', _defined_in: 'blocks.py',
        _source_file: 'dotted_model.py', _raw_call: 'blocks.ScaleBlock()' },
      isGroup: true, expanded: false, moduleType: 'custom', moduleInstanceId: 'block',
      moduleTemplateId: 'blocks::ScaleBlock', moduleMembers: ['block.proj'], moduleLayerCount: 1,
      moduleEntryNodeId: 'block.proj', moduleExitNodeId: 'block.proj' } } as unknown as ModelNode;
  const child = node('block.proj', { _source_file: 'blocks.py', _raw_call: 'nn.Linear(4, 4)' });
  child.data.parentId = 'block';
  const nodes = [node('input', {}), wrapper, child, node('head', {}), node('output', {})];
  const pairs = [['input', 'block'], ['block', 'head'], ['input', 'block.proj'], ['block.proj', 'head'], ['head', 'output']];
  const payload = projection(nodes, { traced: false, document: modelGraphToDocumentV2({ nodes,
    edges: pairs.map(([source, target]) => ({ id: `${source}-${target}`, source, target })) }, { name: 'DottedNet', revision: 2 }) });
  const result = applyProjection(payload, [])!;
  const block = result.nodes.find(n => n.id === 'block')!;
  expect(block.type).toBe('moduleNode');
  expect(block.data.parameters).toEqual(wrapper.data.parameters);
  expect(block.data.moduleMembers).toEqual(['block.proj']);
  expect(result.nodes.find(n => n.id === 'block.proj')!.data.parentId).toBe('block');
  expect(moduleDefinitionNodeId(block.data.moduleMembers, block.id, 'shape:block')).toBe('block.proj');
  expect(result.edges.map(e => [e.source, e.target])).toEqual(pairs);
});

it('puts recorded shapes in the original node fields and keeps canvas position/selection', () => {
  const previous = { ...node('fc', {}), selected: true, position: { x: 800, y: 400 } };
  const result = applyProjection(projection([node('fc', {
    _shape_provenance: 'observed', _in_shape: '1x4', _out_shape: '1x8',
  })]), [previous])!;
  expect(result.nodes[0].data).toMatchObject({ inputShape: [1, 4], outputShape: [1, 8] });
  expect(result.nodes[0].position).toEqual(previous.position);
  expect(result.nodes[0].selected).toBe(true);
});

it('shows a mapped failure and partial input without inventing outputs for unreached or declared nodes', () => {
  const result = applyProjection(projection([
    node('bad', { _shape_provenance: 'unreached', _in_shape: '1x4' }),
    node('later', { _shape_provenance: 'declared', _out_shape: '1x8' }),
  ], { traceError: { nodeId: 'bad', message: 'Forward failed here.' } }), [])!;
  expect(result.nodes[0].data).toMatchObject({ inputShape: [1, 4], error: 'Forward failed here.' });
  expect(result.nodes[0].data.outputShape).toBeUndefined();
  expect(result.nodes[1].data.outputShape).toBeUndefined();
  expect(result.nodes[1].data.error).toBeUndefined();
});

it('clears old runtime fields on a new static projection and never guesses an error node', () => {
  const previous = { ...node('fc', {}), data: { ...node('fc', {}).data,
    inputShape: [1, 4], outputShape: [1, 8], error: 'old failure' } };
  const fresh = applyProjection(projection([node('fc', {})], { traced: false }), [previous])!;
  expect(fresh.nodes[0].data.inputShape).toBeUndefined();
  expect(fresh.nodes[0].data.outputShape).toBeUndefined();
  expect(fresh.nodes[0].data.error).toBeUndefined();
  const unmapped = applyProjection(projection([node('fc', {})], {
    traceError: { nodeId: 'missing', message: 'Forward failed.' },
  }), [])!;
  expect(unmapped.nodes[0].data.error).toBeUndefined();
});

it('keeps unknown/multiple tensor dimensions unknown and accepts measured zero dimensions', () => {
  const result = applyProjection(projection([
    node('unknown', { _shape_provenance: 'observed', _in_shape: '1x?', _out_shape: [[1, 2], [1, 3]] }),
    node('zero', { _shape_provenance: 'observed', _out_shape: '0x8' }),
  ]), [])!;
  expect(result.nodes[0].data.inputShape).toBeUndefined();
  expect(result.nodes[0].data.outputShape).toBeUndefined();
  expect(result.nodes[1].data.outputShape).toEqual([0, 8]);
});


it('places shared node ids afresh when changing models, then preserves that model on refresh', () => {
  const first = projection([node('shared-input', {}), node('shared-output', {})], { graphId: 'layout:first', viewRestored: false });
  const opened = applyProjection(first, [])!;
  const arranged = opened.nodes.map(n => ({ ...n, position: { x: 9000, y: 8000 } }));
  const refreshed = applyProjection(first, arranged)!;
  expect(refreshed.nodes.map(n => n.position)).toEqual(arranged.map(n => n.position));
  const second = { ...first, graphId: 'layout:second', file: 'other.py', cls: 'Other' };
  const switched = applyProjection(second, arranged)!;
  expect(switched.nodes.every(n => n.position.x < 9000 && n.position.y < 8000)).toBe(true);
  const custom = switched.nodes.map(n => ({ ...n, position: { x: 650, y: 775 } }));
  expect(applyProjection(second, custom)!.nodes.map(n => n.position)).toEqual(custom.map(n => n.position));
});


it.each([false, true])('settles first-capture badge geometry while preserving manual moves (moved=%s)', moved => {
  const document = JSON.parse(JSON.stringify(captureFixture)) as GraphDocumentV2;
  // A collapsed captured call gains a stale badge during opening. A plain
  // operation tile would not exercise the module-height transition.
  const call = document.semantic.nodes.find(n => n.kind === 'layer')!;
  call.kind = 'module';
  call.metadata.captureRef = { ...(call.metadata.captureRef as object), kind: 'invocation' };
  document.view.collapsed_modules.push(call.node_id);
  const spanmap = Object.fromEntries(document.semantic.nodes.map(n => [n.node_id, { file: 'net.py', start: [1, 0], end: [2, 0] }]));
  const payload = projectionForCanvas(projection([], { graphId: `layout:settle:${moved}`, document, spanmap, viewRestored: false }));
  const first = applyProjection({ ...payload, greyed: true }, [])!;
  const rendered = reconcileModuleView(first.nodes, first.edges);
  const before = moved ? rendered.nodes.map(n => ({ ...n, position: { x: n.position.x + 10, y: n.position.y + 20 } })) : rendered.nodes;
  const settled = applyProjection(payload, before)!;
  if (moved) {
    expect(settled.nodes.map(n => n.position)).toEqual(before.map(n => n.position));
    return;
  }
  const actual = reconcileModuleView(settled.nodes, settled.edges);
  const output = actual.nodes.find(n => n.type === 'outputNode')!;
  const exit = actual.edges.find(e => !e.hidden && e.target === output.id)!;
  const source = actual.nodes.find(n => n.id === exit.source)!;
  expect(output.position.y + nodeCardSize(output).h / 2 - graphNodeOrigin(actual.nodes, source.id)!.y - nodeCardSize(source).h).toBeCloseTo(24);
});
