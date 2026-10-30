import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { sourceContextPage, sourceCaptureContext, sourceContextQuerySchema, type SourceContextResult } from '@tensorviz/graph-host/sourceContext';
import { registerMcpSourceContextTool } from '@tensorviz/graph-host/mcpSourceContext';
import { McpGraphService, type McpModel } from '@tensorviz/graph-host/mcpGraph';
import { AttemptLedger } from '@tensorviz/graph-host/attemptLedger';
import { CaptureInspector } from '@tensorviz/graph-host/captureInspection';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import type { ProjectionPayload } from '@tensorviz/contracts/protocol';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { canonicalFixture } from '../../../tests/fixtures/semantic.cjs';

const oracle = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../../tests/fixtures/contracts/source-context-oracle.json'), 'utf8'));
const page = () => sourceContextPage(oracle.queries[0].page, oracle.sources, oracle.queries[0].query);
const capture = () => structuredClone(oracle.capture) as CaptureSnapshot;

suite('source/configuration context', () => {
  test('actual Python pages pass the host gate and exact source links reach real ATen operations', () => {
    const observed = capture();
    for (const { query, page: value } of oracle.queries) {
      const result = sourceContextPage(value, oracle.sources, query);
      const context = sourceCaptureContext(result, observed);
      assert.equal(context.appliesToCurrentRuntime, 'not_established');
      if (query.symbol === 'forward') assert.ok(context.links.some(link => link.ref.kind === 'definition'));
      else {
        const operations = context.links.flatMap(link => link.ref.kind === 'operation' ? observed.operations.filter(item => item.id === link.ref.id) : []);
        assert.ok(operations.some(item => item.target === 'aten.view.default'));
        assert.ok(context.unmatchedRecords > 0, 'source-only imports and configuration are not relabelled as runtime observations');
      }
    }
  });

  for (const change of ['digest', 'file', 'span', 'scope', 'tokens', 'quality', 'snapshot']) test(`capture correspondence rejects ${change} mismatches`, () => {
    const observed = capture();
    for (const binding of observed.sourceBindings) {
      if (change === 'quality') binding.quality = 'ambiguous';
      else if (change === 'snapshot') binding.snapshotId = 'other';
      else for (const anchor of binding.anchors) {
        if (change === 'digest') anchor.digest = 'f'.repeat(64);
        if (change === 'file') anchor.file = 'elsewhere.py';
        if (change === 'span') anchor.startLine++;
        if (change === 'scope') anchor.scope = 'Other.forward';
        if (change === 'tokens') anchor.tokenDigest = 'f'.repeat(64);
      }
    }
    assert.deepEqual(sourceCaptureContext(page(), observed).links, []);
  });

  for (const change of ['index', 'inventory', 'excerpt', 'range', 'extra', 'paging', 'oversize']) test(`response validation refuses ${change} corruption`, () => {
    const value = structuredClone(oracle.queries[0].page);
    const record = value.items.find((item: any) => item.anchor);
    if (change === 'index') value.indexId = 'f'.repeat(64);
    if (change === 'inventory') value.files[0].file = 'secret.py';
    if (change === 'excerpt') record.excerpt = 'invented source';
    if (change === 'range') record.anchor.endLine = 9999;
    if (change === 'extra') value.execute = true;
    if (change === 'paging') value.nextOffset = 0;
    if (change === 'oversize') record.excerpt = 'x'.repeat(65536);
    assert.throws(() => sourceContextPage(value, oracle.sources, oracle.queries[0].query), /did not match/);
  });

  test('bounded correspondence reports omitted exact matches without expanding output indefinitely', () => {
    const observed = capture(), original = observed.operations.find(item => item.target === 'aten.view.default')!;
    observed.operations.push(...Array.from({ length: 200 }, (_, i) => ({ ...original, id: `extra:${i}` })));
    const context = sourceCaptureContext(page(), observed);
    assert.ok(context.omittedMatches > 0); assert.ok(context.links.length <= 64);
    assert.ok(Buffer.byteLength(JSON.stringify(context)) < 16 * 1024);
  });

  test('historical source matches preserve partial and traced-program evidence labels', () => {
    const observed = capture(); observed.status = 'partial'; observed.coverage.scope = 'traced_program';
    const context = sourceCaptureContext(page(), observed);
    assert.equal(context.captureStatus, 'partial'); assert.equal(context.coverageScope, 'traced_program');
    assert.equal(context.appliesToCurrentRuntime, 'not_established');
    assert.deepEqual(context.coverageInspection.path, ['coverage']);
  });

  test('anchor continuation requires the exact source index and cannot silently mix selectors', () => {
    const anchor = page().items.find(item => 'id' in item)!;
    assert.ok('id' in anchor);
    for (const query of [{ file: 'model.py' }, { file: 'model.py', anchorIds: [anchor.id] },
      { file: 'model.py', symbol: 'WIDTH', anchorIds: [anchor.id], indexId: page().indexId }]) {
      assert.equal(sourceContextQuerySchema.safeParse(query).success, false);
    }
  });

  test('actual MCP accepts a current source query and checks revision and capture retention after async work', async () => {
    const observed = capture(), ledger = new AttemptLedger(), inspector = new CaptureInspector(ledger);
    const target = { workspaceFolder: '/ws', file: 'model.py', cls: 'Model', documentVersion: 1,
      sourceDigest: oracle.sources.find((item: any) => item.file === 'model.py').digest, backendRevision: 1 };
    let model: McpModel = { target, current: true, projection: { document: { semantic: { nodes: [], edges: [] } } } as unknown as ProjectionPayload };
    const graph = new McpGraphService('source-context-test', () => model), ref = graph.reference(model);
    const attempt = ledger.begin(observed.attemptId, target, 'manual', null);
    const { plan } = canonicalFixture(observed.attemptId); plan.source = structuredClone(observed.source);
    plan.captureLimits = structuredClone(observed.limits);
    assert.ok(observed.planDigest);
    attempt.resolved({ plan, applicabilityDigest: observed.planDigest });
    attempt.captured({ ok: true, exitCode: 0, envelope: { protocol_version: 1,
      result: { diagnostics: { semantic_capture: observed } }, spans: {}, error: null, worker: {} } }); attempt.finish('passed');
    assert.equal(inspector.list(target).captures[0]?.format, 'semantic', 'the actual oracle must pass production retention');
    let mutation = () => {}, reads = 0;
    const server = new McpServer({ name: 'source-test', version: '1' });
    registerMcpSourceContextTool(server, graph, inspector, async () => {
      reads++; mutation();
      return { ...page(), collection: { availableFiles: 2, omittedFiles: 0, byteLimit: 1048576, fileLimit: 16 } } satisfies SourceContextResult;
    });
    const client = new Client({ name: 'source-test', version: '1' }), [a, b] = InMemoryTransport.createLinkedPair();
    await Promise.all([server.connect(a), client.connect(b)]);
    const args = { ...ref, ...oracle.queries[0].query, captureId: observed.id, sourceId: observed.source.id };
    try {
      const tools = await client.listTools();
      assert.equal(tools.tools.find(item => item.name === 'query_source_context')?.annotations?.readOnlyHint, true);
      const good = await client.callTool({ name: 'query_source_context', arguments: args });
      assert.ok(!good.isError, JSON.stringify(good));
      assert.ok((good.structuredContent as any).captureContext.links.length);
      for (const fields of [{ sourceId: undefined }, { revision: 'f'.repeat(64) }]) {
        const before = reads;
        assert.ok((await client.callTool({ name: 'query_source_context', arguments: { ...args, ...fields } })).isError);
        assert.equal(reads, before);
      }
      mutation = () => { model = { ...model, current: false }; };
      const stale = await client.callTool({ name: 'query_source_context', arguments: args });
      assert.equal((stale.structuredContent as any).error.code, 'STALE_TARGET');
      model = { ...model, current: true };
      mutation = () => { ledger.begin('other', { ...target, cls: 'Other' }, 'manual', null).finish('refused'); };
      const evicted = await client.callTool({ name: 'query_source_context', arguments: args });
      assert.equal((evicted.structuredContent as any).error.code, 'UNKNOWN_CAPTURE');
    } finally { await client.close(); await server.close(); }
  });
});
