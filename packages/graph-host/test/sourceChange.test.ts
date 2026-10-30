import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { sourceContextPage, sourceCaptureContext, sourceContextQuerySchema } from '@tensorviz/graph-host/sourceContext';
import { registerMcpSourceContextTool } from '@tensorviz/graph-host/mcpSourceContext';
import { McpGraphService, type McpModel } from '@tensorviz/graph-host/mcpGraph';
import { AttemptLedger } from '@tensorviz/graph-host/attemptLedger';
import { CaptureInspector } from '@tensorviz/graph-host/captureInspection';
import type { ProjectionPayload } from '@tensorviz/contracts/protocol';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { canonicalFixture } from '../../../tests/fixtures/semantic.cjs';

const oracle = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../../tests/fixtures/contracts/source-change-oracle.json'), 'utf8'));
const first = oracle.queries[0];
suite('conditional source configuration changes', () => {
  test('actual parser effects retain exact constructor correspondence without turning arithmetic into observed values', () => {
    for (const { query, page } of oracle.queries) {
      const result = sourceContextPage(page, oracle.sources, query);
      assert.equal(result.changeAnalysis?.evidence, 'conditional_source_arithmetic');
      const context = sourceCaptureContext(result, oracle.capture);
      const modules = context.links.filter(link => link.ref.kind === 'module').map(link => link.ref.id);
      assert.ok(oracle.capture.modules.filter((item: any) => item.paths[0]).every((item: any) => modules.includes(item.id)));
      assert.equal(context.appliesToCurrentRuntime, 'not_established');
    }
  });
  test('exact source changes link to constructor arguments and independently observed candidate parameter shapes', () => {
    for (const { query, page } of oracle.queries) {
      const result = sourceContextPage(page, oracle.sources, query);
      const impact = (sourceCaptureContext(result, oracle.capture) as any).configurationImpact;
      assert.equal(impact.evidence, 'conditional_constructor_state_dimensions');
      assert.equal(impact.effects.length, 3);
      const actual = oracle.independentRuns.find((run: any) => run.width === query.assumeValue);
      for (const effect of impact.effects) {
        assert.equal(effect.status, 'derived');
        assert.ok(effect.arguments.every((argument: any) => argument.sourceBefore === argument.observedBefore));
        for (const state of effect.stateShapes) {
          assert.equal(state.status, 'derived');
          assert.ok(state.valueIds.length);
          const name = state.names.find((name: string) => Object.hasOwn(actual.parameterShapes, name));
          assert.deepEqual(state.after, actual.parameterShapes[name]);
        }
      }
      const shared = impact.effects.find((effect: any) => effect.stateShapes.some((state: any) => state.names.includes('tied.weight')));
      const weight = shared.stateShapes.find((state: any) => state.names.includes('tied.weight'));
      assert.ok(weight.names.includes('up.weight'));
      assert.equal(weight.objectIds.length, 1, 'two registrations of the same module retain one parameter object');
      assert.equal(impact.forwardCompatibility, 'not_established');
      assert.equal(impact.checkpointCompatibility, 'not_established');
    }
  });

  for (const [defect, reason] of [
    ['class', 'constructor_identity_unverified'], ['initializer', 'initializer_arguments_unavailable'],
    ['mismatch', 'observed_source_value_mismatch'], ['unknown', 'source_argument_unknown'],
    ['expansion', 'expanded_or_omitted_arguments'], ['omitted', 'expanded_or_omitted_arguments'],
    ['unsupported-class', 'unsupported_module'], ['unsupported-argument', 'unsupported_initializer_argument'],
    ['negative', 'invalid_or_unavailable_dimension'],
  ]) test(`constructor prediction refuses ${defect} evidence`, () => {
    const capture = structuredClone(oracle.capture), page = sourceContextPage(first.page, oracle.sources, first.query);
    const module = capture.modules.find((item: any) => item.paths.includes('up'));
    const context = sourceCaptureContext(page, capture);
    const link = context.links.find(item => item.ref.id === module.id)!;
    const record = page.items.find(item => 'id' in item && item.id === link.anchorId) as any;
    if (defect === 'class') module.constructorClassVerified = false;
    if (defect === 'initializer') module.initializerArguments = null;
    if (defect === 'mismatch') module.initializerArguments.positional[1].value.value = '99';
    if (defect === 'unknown') record.configurationChange.arguments[0] = { path: ['args', 1], status: 'unknown', before: null, after: null, reason: 'unresolved_binding' };
    if (defect === 'expansion') record.configurationChange.hasExpansion = true;
    if (defect === 'omitted') record.configurationChange.omittedArguments = 1;
    if (defect === 'unsupported-class') capture.definitions.find((item: any) => item.id === module.definitionId).qualifiedName = 'CustomLinear';
    if (defect === 'unsupported-argument') {
      record.configurationChange.arguments[0].path = ['kwargs', 'device'];
      module.initializerArguments.positional.find((item: any) => item.name === 'device').value = { kind: 'int', value: '12' };
    }
    if (defect === 'negative') record.configurationChange.arguments[0].after = -1;
    const effect = sourceCaptureContext(page, capture).configurationImpact!.effects.find(item => item.moduleId === module.id)!;
    assert.equal(effect.status, 'unknown'); assert.equal(effect.reason, reason);
    assert.deepEqual(effect.stateShapes, []);
  });

  for (const [defect, reason] of [['missing', 'state_not_observed'], ['shape', 'recorded_state_shape_mismatch'],
    ['object', 'state_identity_ambiguous']]) test(`state prediction preserves ${defect} uncertainty`, () => {
    const capture = structuredClone(oracle.capture), page = sourceContextPage(first.page, oracle.sources, first.query);
    const module = capture.modules.find((item: any) => item.paths.includes('up'));
    const weight = capture.values.find((item: any) => item.state.some((state: any) => state.name === 'up.weight'));
    if (defect === 'missing') capture.values = capture.values.filter((item: any) => !item.state.some((state: any) => ['up.weight', 'tied.weight'].includes(state.name)));
    if (defect === 'shape') weight.shape = [99, 8];
    if (defect === 'object') capture.values.push({ ...weight, id: 'value:other', objectId: 'object:other' });
    const effect = sourceCaptureContext(page, capture).configurationImpact!.effects.find(item => item.moduleId === module.id)!;
    const state = effect.stateShapes.find(item => item.parameter === 'weight')!;
    assert.equal(state.status, 'unknown'); assert.equal(state.reason, reason); assert.equal(state.after, null);
  });

  test('changed source identity and traced-program evidence never acquire constructor predictions', () => {
    for (const defect of ['source', 'backend']) {
      const capture = structuredClone(oracle.capture), page = sourceContextPage(first.page, oracle.sources, first.query);
      if (defect === 'source') for (const file of capture.source.files) file.digest = 'f'.repeat(64);
      else capture.backend = 'jaxpr';
      const impact = sourceCaptureContext(page, capture).configurationImpact!;
      assert.equal(impact.effects.length, 0); assert.ok(impact.unlinkedCalls > 0);
    }
  });

  test('large constructor inventories are bounded and returned predictions cannot mutate evidence', () => {
    const capture = structuredClone(oracle.capture), page = sourceContextPage(first.page, oracle.sources, first.query);
    const module = capture.modules.find((item: any) => item.paths.includes('up'));
    capture.modules.push(...Array.from({ length: 80 }, (_, i) => ({ ...module, id: `module:extra${i}` })));
    const bounded = sourceCaptureContext(page, capture), original = JSON.stringify(bounded);
    assert.ok(bounded.configurationImpact!.omittedEffects > 0); assert.ok(bounded.omittedMatches > 0);
    assert.ok(bounded.configurationImpact!.effects.length <= 16);
    assert.ok(Buffer.byteLength(JSON.stringify(bounded.configurationImpact)) <= 12 * 1024);
    bounded.configurationImpact!.effects[0].stateShapes[0].after![0] = 999;
    assert.equal(JSON.stringify(sourceCaptureContext(page, capture)), original);
  });

  for (const defect of ['missing', 'unrequested', 'changed-assumption', 'observed-label', 'known-null', 'unknown-value', 'wrong-kind', 'wrong-definition', 'oversized-arguments']) {
    test(`host rejects ${defect} change evidence`, () => {
      const page = structuredClone(first.page), query = { ...first.query };
      const definition = page.items.find((item: any) => item.id === page.changeAnalysis.definitionId);
      const call = page.items.find((item: any) => item.kind === 'call');
      if (defect === 'missing') delete page.changeAnalysis;
      if (defect === 'unrequested') delete query.assumeValue;
      if (defect === 'changed-assumption') page.changeAnalysis.after++;
      if (defect === 'observed-label') page.changeAnalysis.evidence = 'observed';
      if (defect === 'known-null') definition.configurationChange.after = null;
      if (defect === 'unknown-value') definition.configurationChange.status = 'unknown';
      if (defect === 'wrong-kind') definition.configurationChange = call.configurationChange;
      if (defect === 'wrong-definition') definition.name = 'OTHER';
      if (defect === 'oversized-arguments') call.configurationChange.arguments = Array(17).fill(call.configurationChange.arguments[0]);
      assert.throws(() => sourceContextPage(page, oracle.sources, query), /did not match/);
    });
  }
  test('assumptions require a downstream symbol selector and bounded integer', () => {
    for (const fields of [{ direction: 'both' }, { assumeValue: 1.5 }, { assumeValue: Number.MAX_SAFE_INTEGER + 1 },
      { symbol: undefined, anchorIds: [first.page.changeAnalysis.definitionId], indexId: first.page.indexId }]) {
      assert.equal(sourceContextQuerySchema.safeParse({ ...first.query, ...fields }).success, false);
    }
  });
  test('production MCP exposes the assumption and preserves revision checks without execution or edit hooks', async () => {
    let current = true, reads = 0;
    const model = { target: { workspaceFolder: '/ws', file: 'model.py', cls: 'Model', documentVersion: 1,
      sourceDigest: oracle.sources.find((item: any) => item.file === 'model.py').digest, backendRevision: 1 },
      get current() { return current; }, projection: { document: { semantic: { nodes: [], edges: [] } } } as unknown as ProjectionPayload } satisfies McpModel;
    const graph = new McpGraphService('source-change', () => model), ledger = new AttemptLedger(), inspector = new CaptureInspector(ledger);
    const observed = structuredClone(oracle.capture), attempt = ledger.begin(observed.attemptId, model.target, 'manual', null);
    const { plan } = canonicalFixture(observed.attemptId); plan.source = structuredClone(observed.source); plan.captureLimits = structuredClone(observed.limits);
    attempt.resolved({ plan, applicabilityDigest: observed.planDigest });
    attempt.captured({ ok: true, exitCode: 0, envelope: { protocol_version: 1,
      result: { diagnostics: { semantic_capture: observed } }, spans: {}, error: null, worker: {} } }); attempt.finish('passed');
    assert.equal(inspector.list(model.target).captures[0]?.format, 'semantic');
    const server = new McpServer({ name: 'source-change', version: '1' });
    registerMcpSourceContextTool(server, graph, inspector, async (_ref, query) => {
      reads++; assert.equal(query.assumeValue, 16);
      return { ...sourceContextPage(first.page, oracle.sources, query), collection: { availableFiles: 2, omittedFiles: 0, byteLimit: 1048576, fileLimit: 16 } };
    });
    const client = new Client({ name: 'source-change', version: '1' }), [a, b] = InMemoryTransport.createLinkedPair();
    await Promise.all([server.connect(a), client.connect(b)]);
    try {
      const args = { ...graph.reference(model), ...first.query, captureId: observed.id, sourceId: observed.source.id };
      const tools = await client.listTools();
      assert.ok(tools.tools[0].inputSchema.properties?.assumeValue);
      const good = await client.callTool({ name: 'query_source_context', arguments: args });
      assert.ok(!good.isError, JSON.stringify(good));
      assert.equal((good.structuredContent as any).changeAnalysis.after, 16);
      assert.equal((good.structuredContent as any).captureContext.configurationImpact.effects.length, 3);
      const operations = (good.structuredContent as any).captureContext.operationImpact;
      assert.equal(operations.forwardCompatibility, 'not_established');
      assert.ok(operations.effects.some((effect: any) => effect.target === 'aten.view.default'
        && effect.status === 'derived' && effect.localCheck.status === 'incompatible'
        && effect.dimensions[0].basis === 'captured_tensor_dimension'));
      const combined = (good.structuredContent as any).captureContext.computationImpact;
      assert.deepEqual(combined.summary, { compatible: 7, incompatible: 1, unknown: 2 });
      assert.equal(combined.outputs[0].shape, null);
      assert.equal(combined.forwardCompatibility, 'not_established');
      current = false;
      assert.ok((await client.callTool({ name: 'query_source_context', arguments: args })).isError);
      assert.equal(reads, 1);
    } finally { await client.close(); await server.close(); }
  });
});
