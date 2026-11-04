import * as assert from 'node:assert/strict';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { McpGraphService, object, type McpModel } from '@tensorviz/graph-host/mcpGraph';
import { agentGraphContext } from '../../canvasAgentContext';
import { captureTools } from '../../mcpHost';
import { registerMcpCaptureTools } from '@tensorviz/graph-host/mcpCaptures';
import { CaptureInspector } from '@tensorviz/graph-host/captureInspection';
import { AttemptLedger } from '@tensorviz/graph-host/attemptLedger';

suite('Canvas graph context', () => {
  const target = { workspaceFolder: '/work', file: 'net.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
  const fixture = () => ({ target: { ...target }, current: true, projection: { traced: false, spanmap: {}, document: { semantic: {
    nodes: [{ node_id: 'input', kind: 'input', op: 'Input', params: {} },
      { node_id: 'fc', kind: 'layer', op: 'Linear', params: { out_features: 8 }, metadata: { outputShape: [1, 8] } }],
    edges: [{ edge_id: 'e1', source_node_id: 'input', target_node_id: 'fc', source_port: 'y', target_port: 'x' }],
  } } } } as unknown as McpModel);

  test('supplies the same small graph, shapes and provenance as the existing MCP tool', async () => {
    const model = fixture(), graph = new McpGraphService('context', () => model);
    const server = new McpServer({ name: 'fixture', version: '1' }); graph.register(server);
    const client = new Client({ name: 'reader', version: '1' });
    const [a, b] = InMemoryTransport.createLinkedPair();
    try {
      await server.connect(a); await client.connect(b);
      const ref = graph.reference(model), attached = agentGraphContext(graph, ref);
      const result = await client.callTool({ name: 'get_graph', arguments: { ...ref, mode: 'graph', limit: 200 } });
      assert.equal((attached.nodes as any[])[1].shapes.output[1], 8);
      assert.deepEqual(attached, { ...object(result.structuredContent), contextLimited: false });
      model.target.documentVersion++;
      assert.throws(() => agentGraphContext(graph, ref), /revision changed/);
    } finally { await client.close(); await server.close(); }
  });

  test('window routing metadata does not enter the strict compact-context query', async () => {
    const model = fixture(), graph = new McpGraphService('context', () => model);
    const tools = captureTools(server => registerMcpCaptureTools(server, graph, new CaptureInspector(new AttemptLedger())));
    const result = await tools.get('get_model_context')!.handler({ ...graph.reference(model), hostId: 'context' }, {});
    assert.equal(result.isError, undefined);
    assert.equal(result.structuredContent?.reason, 'no_latest_capture');
    assert.equal(result.structuredContent?.available, false);
  });

  test('compact-context preload uses 8 KiB with exact continuation and preserves the normal 64 KiB default', () => {
    const model = fixture(), graph = new McpGraphService('context', () => model);
    (model.projection.document as any).semantic.nodes = Array.from({ length: 100 }, (_, i) =>
      ({ node_id: `node-${i}`, op: 'Linear', params: { description: '界'.repeat(80) } }));
    const ref = graph.reference(model), small = agentGraphContext(graph, ref, 8 * 1024), normal = agentGraphContext(graph, ref);
    assert.ok(Buffer.byteLength(JSON.stringify(small)) <= 8 * 1024);
    assert.ok((small.nodes as any[]).length > 0 && (small.nodes as any[]).length < (normal.nodes as any[]).length);
    assert.equal(small.nextNodeOffset, (small.nodes as any[]).length);
    assert.deepEqual(small.nodes, (normal.nodes as any[]).slice(0, small.nextNodeOffset as number));
    assert.equal(normal.nextNodeOffset, null);
  });

  test('retains whole records and exact continuation offsets within the byte budget', () => {
    const model = fixture(), graph = new McpGraphService('context', () => model);
    const semantic = (model.projection.document as any).semantic;
    semantic.nodes = Array.from({ length: 205 }, (_, i) => ({ node_id: `node-${i}`, op: 'Linear', params: {} }));
    let attached = agentGraphContext(graph, graph.reference(model));
    assert.equal(attached.contextLimited, true);
    assert.equal(attached.nextNodeOffset, (attached.nodes as any[]).length);
    assert.ok(Number(attached.nextNodeOffset) > 0 && Number(attached.nextNodeOffset) <= 200);
    semantic.nodes[1].params.description = '界'.repeat(32_000);
    attached = agentGraphContext(graph, graph.reference(model));
    assert.ok(Buffer.byteLength(JSON.stringify(attached)) <= 64 * 1024);
    assert.equal((attached.nodes as any[]).length, 1);
    assert.equal(attached.nextNodeOffset, 1, 'an oversized record is omitted, never silently skipped');
    semantic.metadata = { capture: { note: '界'.repeat(32_000) } };
    attached = agentGraphContext(graph, graph.reference(model));
    assert.equal(attached.contextLimited, true); assert.equal(attached.nextNodeOffset, 0);
    assert.match(String(attached.omitted), /metadata/);
    assert.ok(Buffer.byteLength(JSON.stringify(attached)) <= 64 * 1024);
  });
});
