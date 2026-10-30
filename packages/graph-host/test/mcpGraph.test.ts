import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { McpGraphService, type McpModel, type JsonObject } from '@tensorviz/graph-host/mcpGraph';
import type { ProjectionPayload } from '@tensorviz/contracts/protocol';

suite('MCP graph inspection', () => {
  let model: McpModel | undefined;
  let client: Client, server: McpServer, graph: McpGraphService;
  const target = { workspaceFolder: '/work', file: 'net.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
  setup(async () => {
    model = { target: { ...target }, current: true, projection: { traced: false, spanmap: { fc: { file: 'net.py', start: [4, 0], end: [4, 32] } }, document: { semantic: {
      nodes: [{ node_id: 'input', kind: 'input', op: 'Input', params: {} },
        { node_id: 'fc', kind: 'layer', op: 'Linear', params: { out_features: 8, _shape_provenance: 'observed', _source_code: 'private' }, metadata: { canvas: { data: { outputShape: [1, 8] } } } }],
      edges: [{ edge_id: 'e1', source_node_id: 'input', target_node_id: 'fc', source_port: 'y', target_port: 'x' }],
    } } } as unknown as ProjectionPayload };
    graph = new McpGraphService('host-one', () => model);
    server = new McpServer({ name: 'fixture', version: '1' }); graph.register(server);
    client = new Client({ name: 'generic-reader', version: '1' });
    const [a, b] = InMemoryTransport.createLinkedPair();
    await server.connect(a); await client.connect(b);
  });
  teardown(async () => { await client.close(); await server.close(); });
  async function call(name: string, args: JsonObject = {}) {
    const reply = await client.callTool({ name, arguments: args });
    return { error: reply.isError, data: reply.structuredContent as any };
  }
  const ref = () => graph.reference(model!);
  test('attached inspection matches MCP facts and rejects stale or unknown selections', async () => {
    const reference = ref();
    for (const [offset, limit] of [[0, 50], [0, 1], [1, 1]]) {
      const direct = graph.inspectNodes(reference, ['fc'], offset, limit);
      const viaTool = await call('inspect_nodes', { ...reference, nodeIds: ['fc'], edgeOffset: offset, limit });
      assert.deepEqual(direct, viaTool.data);
    }
    const inspected = graph.inspectNodes(reference, ['fc']);
    assert.deepEqual((inspected.nodes as any[])[0].parameters, { out_features: 8 });
    assert.deepEqual((inspected.nodes as any[])[0].shapes, { input: null, output: [1, 8], provenance: 'observed' });
    assert.throws(() => graph.inspectNodes(reference, ['missing']), /distinct node ids/);
    model!.current = false;
    assert.throws(() => graph.inspectNodes(reference, ['fc']), /revision changed/);
    model!.current = true; model!.target.documentVersion++;
    assert.throws(() => graph.inspectNodes(reference, ['fc']), /revision changed/);
  });
  test('counts first pages and real outcomes without sending graph contents', async () => {
    await call('get_graph', { ...ref(), mode: 'graph', limit: 1 });
    await call('get_graph', { ...ref(), mode: 'graph', nodeOffset: 1, limit: 1 });
    await call('inspect_nodes', { ...ref(), nodeIds: ['fc'] });
    const old = ref(); model!.target.documentVersion++;
    assert.equal((await call('get_graph', old)).data.error.code, 'STALE_TARGET');
  });

  test('canonical Python document exposes the same invocation identities and typed argument uses', async () => {
    const document = JSON.parse(readFileSync(resolve(__dirname,
      '../../../canvas/src/test/fixtures/canvas/capture-document.json'), 'utf8'));
    model!.projection.document = document;
    model!.projection.traced = true;
    const add = document.semantic.nodes.find((node: any) => node.metadata.captureRef.semanticId === 'add2');
    const result = await call('inspect_nodes', { ...ref(), nodeIds: [add.node_id] });
    assert.equal(result.error, undefined);
    const node = result.data.nodes[0];
    assert.deepEqual(node.captureRef, add.metadata.captureRef);
    assert.deepEqual(node.typedPorts, add.metadata.capturePorts);
    assert.deepEqual(node.metrics, { parameterCount: null, outputTensor: { shape: [2, 4], dtype: 'float32', logicalBytes: 32 } });
    assert.equal(node.parentId, add.metadata.parentId);
    const operands = result.data.edges.filter((edge: any) => edge.targetNodeId === add.node_id);
    assert.equal(operands.length, 2);
    assert.equal(new Set(operands.map((edge: any) => edge.targetPort)).size, 2);
    assert.equal(new Set(operands.map((edge: any) => edge.captureEdge.valueId)).size, 1);
    assert.deepEqual((await call('get_graph', { ...ref() })).data.capture, document.semantic.metadata.capture);
    const root = document.semantic.nodes.find((item: any) => item.metadata.captureRef.kind === 'invocation');
    assert.equal((await call('inspect_nodes', { ...ref(), nodeIds: [root.node_id] })).data.nodes[0].metrics.parameterCount, 0);
    model!.projection.greyed = true;
    assert.deepEqual((await call('inspect_nodes', { ...ref(), nodeIds: [root.node_id] })).data.nodes[0].metrics,
      { parameterCount: null, outputTensor: null });
  });

  test('reads exact Python-converter ownership and operand ports through the MCP transport', async () => {
    model!.projection.document = JSON.parse(readFileSync(resolve(__dirname,
      '../../../canvas/src/test/fixtures/canvas/ported-document.json'), 'utf8'));
    const result = await call('inspect_nodes', { ...ref(), nodeIds: ['add', 'factory'] });
    assert.equal(result.error, undefined);
    assert.equal(result.data.nodes[0].parentId, 'block');
    assert.deepEqual(result.data.nodes[0].inputPorts, ['args:0', 'args:1', 'alpha']);
    assert.deepEqual(result.data.nodes[1].inputPorts, []);
    assert.deepEqual(result.data.edges.slice(0, 2).map((edge: any) => [edge.sourcePort, edge.targetPort]),
      [['h', 'args:0'], ['h', 'args:1']]);
  });

  test('keeps canonical coverage distinct from cached shapes and legacy canvas ownership', async () => {
    model!.projection.traced = true;
    const node = (model!.projection.document as any).semantic.nodes[1];
    node.metadata.parentId = null;
    node.metadata.canvas.parentId = 'stale-parent';
    node.params._in_shape = '1x4';
    node.params._out_shape = '4';
    for (const provenance of ['unreached', 'not_observed', 'capture_limited', 'static_only', 'unresolved', 'failed', 'unknown', 'state']) {
      node.params._shape_provenance = provenance;
      const result = await call('inspect_nodes', { ...ref(), nodeIds: ['fc'] });
      assert.equal(result.data.nodes[0].parentId, null);
      assert.equal(result.data.nodes[0].shapes.provenance, provenance);
      assert.deepEqual(result.data.nodes[0].shapes.input, provenance === 'state' ? [1, 4] : null);
      assert.deepEqual(result.data.nodes[0].shapes.output, provenance === 'state' ? [4] : null);
      const page = await call('get_graph', { ...ref(), mode: 'graph', nodeIds: ['fc'] });
      assert.deepEqual(page.data.nodes[0].shapes, result.data.nodes[0].shapes);
    }
  });

  test('discovery does not need a model, backend or execution', async () => {
    model = undefined;
    assert.deepEqual((await call('list_models')).data.models, []);
    assert.equal((await call('get_graph', { modelId: 'missing', revision: 'a'.repeat(64) })).data.error.code, 'NO_MODEL');
  });
  test('returns paginated topology and measured shapes without internal source fields', async () => {
    const page = await call('get_graph', { ...ref(), mode: 'graph', limit: 1 });
    assert.equal(page.data.nextNodeOffset, 1);
    assert.equal(page.data.edges[0].targetNodeId, 'fc');
    const detail = await call('inspect_nodes', { ...ref(), nodeIds: ['fc'] });
    assert.equal(detail.data.nodes[0].source.file, 'net.py');
    assert.deepEqual(detail.data.nodes[0].parameters, { out_features: 8 });
    assert.deepEqual(detail.data.nodes[0].shapes, { input: null, output: [1, 8], provenance: 'observed' });
  });
  test('reads real trace shape fields and never substitutes stale canvas estimates', async () => {
    model!.projection.traced = true;
    const semantic = model!.projection.document as any;
    const node = semantic.semantic.nodes[1];
    node.params._in_shape = '1x10';
    node.params._out_shape = '1x8';
    node.metadata.canvas.data.outputShape = [99, 99];
    let result = await call('inspect_nodes', { ...ref(), nodeIds: ['fc'] });
    assert.deepEqual(result.data.nodes[0].shapes, { input: [1, 10], output: [1, 8], provenance: 'observed' });
    node.params._shape_provenance = 'unreached';
    result = await call('inspect_nodes', { ...ref(), nodeIds: ['fc'] });
    assert.equal(result.data.nodes[0].shapes.input, null, 'An unreached node must not expose a cached input as capture evidence');
    assert.equal(result.data.nodes[0].shapes.output, null);
    node.params._shape_provenance = 'declared';
    result = await call('inspect_nodes', { ...ref(), nodeIds: ['fc'] });
    assert.deepEqual(result.data.nodes[0].shapes, { input: [1, 10], output: [1, 8], provenance: 'declared' });
    node.params._out_shape = 'unknown';
    result = await call('inspect_nodes', { ...ref(), nodeIds: ['fc'] });
    assert.equal(result.data.nodes[0].shapes.output, null);
  });
  test('rejects a changed source revision and a foreign window before returning nodes', async () => {
    const old = ref(); model!.target.documentVersion++;
    assert.equal((await call('get_graph', old)).data.error.code, 'STALE_TARGET');
    assert.equal((await call('get_graph', { ...ref(), modelId: 'other-window' })).data.error.code, 'STALE_TARGET');
    model!.current = false;
    assert.equal((await call('get_graph', ref())).data.error.code, 'STALE_TARGET');
  });
  test('unknown or duplicate node identities fail rather than narrowing silently', async () => {
    for (const nodeIds of [['fc', 'missing'], ['fc', 'fc']]) {
      assert.equal((await call('inspect_nodes', { ...ref(), nodeIds })).data.error.code, 'UNKNOWN_NODE');
    }
  });
  test('unknown shapes stay unknown, including in a graph that has some trace evidence', async () => {
    model!.projection.traced = true;
    const result = await call('inspect_nodes', { ...ref(), nodeIds: ['input'] });
    assert.deepEqual(result.data.nodes[0].shapes, { input: null, output: null, provenance: 'unknown' });
  });
});
