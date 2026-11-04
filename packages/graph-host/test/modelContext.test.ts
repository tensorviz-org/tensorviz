import * as assert from 'node:assert/strict';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { AttemptLedger } from '@tensorviz/graph-host/attemptLedger';
import { CaptureInspector } from '@tensorviz/graph-host/captureInspection';
import { McpGraphService, type McpModel } from '@tensorviz/graph-host/mcpGraph';
import { registerMcpCaptureTools } from '@tensorviz/graph-host/mcpCaptures';
import { buildModelContext, getModelContext, MODEL_CONTEXT_GUIDANCE } from '@tensorviz/graph-host/modelContext';
import { canonicalFixture } from '../../../tests/fixtures/semantic.cjs';

suite('compact canonical model context', () => {
  test('preserves aliases, repeated operand order, modes, zero/unknown counts and missing output', () => {
    const { capture } = canonicalFixture();
    const root = capture.invocations[0];
    capture.modules[0].paths = ['', 'tied'];
    root.arguments = { args: [{ kind: 'tuple', items: [{ kind: 'value', id: 'h' }, { kind: 'value', id: 'h' }] }],
      kwargs: { nested: { kind: 'dict', entries: [{ key: { kind: 'str', value: 'x' }, value: { kind: 'value', id: 'b' } }] } } };
    const second: typeof root = { ...structuredClone(root), id: 'second', sequence: 4, parameterCount: null, output: null,
      parameterCountUnavailable: 'item_limit', training: true, gradEnabled: true, status: 'interrupted' };
    second.arguments = { args: [{ kind: 'opaque', reason: 'depth_limit', typeName: null }], kwargs: {} };
    capture.invocations.push(second);
    capture.modules.push({ id: 'unobserved', definitionId: capture.modules[0].definitionId, paths: ['unused'] });
    capture.status = 'partial'; capture.coverage.omissions = ['depth_limit'];
    const packet = buildModelContext(capture, { maxBytes: 10_000 }) as any;
    assert.equal(packet.status, 'partial'); assert.deepEqual(packet.coverage.omissions, ['depth_limit']);
    const row = packet.modules.find((m: any) => m.id === 'module:0');
    assert.deepEqual(row.paths, ['', 'tied']); assert.equal(row.calls.length, 2);
    assert.deepEqual(row.calls[0].inputs.tensors.map((t: any) => t.id), ['h', 'h', 'b']);
    assert.deepEqual(row.calls[0].inputs.tensors.map((t: any) => t.shape), [[2, 4], [2, 4], [2, 4]]);
    assert.equal(row.calls[0].parameterCount, 0); assert.equal(row.calls[0].parameterCountUnavailable, null);
    assert.equal(row.calls[1].parameterCount, null); assert.equal(row.calls[1].parameterCountUnavailable, 'item_limit');
    assert.equal(row.calls[1].training, true); assert.equal(row.calls[1].gradEnabled, true);
    assert.deepEqual(row.calls[1].inputs.unavailable, ['depth_limit']);
    assert.deepEqual(row.calls[1].outputs, { tensors: [], unavailable: ['not_observed'] });
    assert.deepEqual(packet.modules.find((m: any) => m.id === 'unobserved').calls, []);
    delete root.parameterCount; capture.values[0].shape = null;
    const unknown = (buildModelContext(capture, { focus: 'call:0' }) as any).modules[0].calls[0];
    assert.equal(unknown.parameterCountUnavailable, 'not_observed');
    assert.deepEqual(unknown.inputs.unavailable, ['tensor_shape_unavailable', 'tensor_shape_unavailable']);
  });

  test('bounds UTF-8 JSON, omits whole call lists and retrieves exact focus without substring matches', () => {
    const { capture } = canonicalFixture(), call = capture.invocations[0];
    capture.modules.push({ id: 'shared', definitionId: capture.modules[0].definitionId, paths: ['encoder.shared', '界'.repeat(70)] });
    for (let i = 1; i <= 24; i++) capture.invocations.push({ ...structuredClone(call), id: `shared:${i}`, moduleId: 'shared', sequence: i });
    const small = buildModelContext(capture) as any;
    assert.ok(Buffer.byteLength(JSON.stringify(small)) <= 4000);
    assert.equal(small.modules[0].id, 'module:0', 'root has priority over later calls');
    assert.equal(small.modules.some((m: any) => m.id === 'shared'), false);
    assert.deepEqual(small.omitted, { modules: 1, calls: 24 });
    for (const focus of ['shared', 'encoder.shared', 'shared:24', 'model.Model']) {
      const focused = buildModelContext(capture, { focus, maxBytes: 20_000 }) as any;
      assert.equal(focused.modules.find((m: any) => m.id === 'shared').calls.length, 24);
      assert.equal(focused.omitted.calls, 0);
    }
    assert.equal((buildModelContext(capture, { focus: 'encoder' }) as any).matchedModules, 0);
    assert.throws(() => buildModelContext(capture, { maxBytes: 100 }), /byte budget/);
    assert.throws(() => buildModelContext(capture, { maxBytes: 100_001 }), /byte budget/);
  });

  test('real MCP transport shares the projection, enforces revision/source and never borrows older evidence', async () => {
    const { capture, plan } = canonicalFixture(), ledger = new AttemptLedger(), inspector = new CaptureInspector(ledger);
    const target = { workspaceFolder: '/ws', file: 'model.py', cls: 'Model', documentVersion: 1,
      sourceDigest: 'a'.repeat(64), backendRevision: 1 };
    const model = { target, current: true, projection: { traced: true, spanmap: {},
      document: { semantic: { nodes: [], edges: [] } } } } as unknown as McpModel;
    const handle = ledger.begin(capture.attemptId, target, 'manual', null);
    handle.resolved({ plan, applicabilityDigest: capture.planDigest });
    handle.captured({ ok: true, exitCode: 0, envelope: { protocol_version: 1,
      result: { diagnostics: { semantic_capture: capture } }, spans: {}, error: null, worker: {} } });
    handle.finish('passed');
    let refreshes = 0, staleOnRefresh = false;
    const graph = new McpGraphService('context', () => model, undefined, undefined, async () => {
      refreshes++; if (staleOnRefresh) model.current = false;
    }), server = new McpServer({ name: 'fixture', version: '1' });
    registerMcpCaptureTools(server, graph, inspector);
    const client = new Client({ name: 'reader', version: '1' }), [a, b] = InMemoryTransport.createLinkedPair();
    try {
      await server.connect(a); await client.connect(b);
      const tool = (await client.listTools()).tools.find(t => t.name === 'get_model_context')!;
      assert.equal(tool.annotations?.readOnlyHint, true); assert.ok(tool.description?.includes(MODEL_CONTEXT_GUIDANCE));
      const read = async (args = graph.reference(model)) => (await client.callTool({ name: 'get_model_context', arguments: args })).structuredContent as any;
      const ref = graph.reference(model), packet = await read();
      assert.equal(packet.available, true);
      assert.equal(refreshes, 1, 'current compact facts await the host freshness guard');
      assert.deepEqual(packet, getModelContext(graph, inspector, ref));
      assert.ok(Buffer.byteLength(JSON.stringify(packet)) <= 4000);
      model.target.documentVersion++;
      assert.equal((await read(ref)).error.code, 'STALE_TARGET');
      assert.equal((await read()).available, true, 'unchanged source content still matches the capture');
      model.target.sourceDigest = 'b'.repeat(64);
      assert.equal((await read()).reason, 'source_mismatch');
      model.target.sourceDigest = 'a'.repeat(64);
      const newer = ledger.begin('newer-without-capture', target, 'manual', null);
      assert.equal((await read()).reason, 'no_latest_capture');
      newer.finish('refused'); assert.equal((await read()).reason, 'no_latest_capture');
      staleOnRefresh = true;
      assert.equal(model.current, true);
      assert.equal((await read()).error.code, 'STALE_TARGET', 'a tool must not outrun the standalone freshness watcher');
    } finally { await client.close(); await server.close(); }
  });
});
