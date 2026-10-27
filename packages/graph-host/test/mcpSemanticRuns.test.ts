import * as assert from 'node:assert/strict';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { McpGraphService, type McpModel } from '@tensorviz/graph-host/mcpGraph';
import { McpValidationRuns, registerMcpValidationTools } from '@tensorviz/graph-host/mcpValidation';
import { CaptureInspector, inspectCapture } from '@tensorviz/graph-host/captureInspection';
import { inspectSemanticEvidence } from '@tensorviz/graph-host/mcpSemanticEvidence';
import { ValidationSummaryStore } from '@tensorviz/graph-host/validationSummary';
import type { ProjectionPayload } from '@tensorviz/contracts/protocol';
import type { CaptureSnapshot } from '@tensorviz/graph-host/semanticCapture';
import { canonicalFixture } from '../../../tests/fixtures/semantic.cjs';

suite('MCP canonical run evidence', () => {
  let model: McpModel, graph: McpGraphService, runs: McpValidationRuns, server: McpServer, client: Client;
  let summaries: ValidationSummaryStore, produced: CaptureSnapshot;
  let transform: (capture: CaptureSnapshot) => void, invocations: number;
  setup(async () => {
    invocations = 0; transform = () => {};
    summaries = new ValidationSummaryStore(() => {});
    model = { current: true, target: { workspaceFolder: '/ws', file: 'model.py', cls: 'Model',
      documentVersion: 1, backendRevision: 1, sourceDigest: 'a'.repeat(64) },
      projection: { traced: false, revision: 1, spanmap: {}, document: { semantic: { nodes: [], edges: [] } } } as unknown as ProjectionPayload };
    graph = new McpGraphService('host', () => model);
    runs = new McpValidationRuns(() => model, async (_target, _signal, _inputs, _progress, measured, captured, _execution, _job, attempt) => {
      invocations++;
      const { capture, plan } = canonicalFixture(attempt!.snapshot().id);
      capture.id = `capture:${capture.attemptId}`;
      attempt!.resolved({ plan, applicabilityDigest: capture.planDigest });
      transform(capture); produced = capture;
      const outcome = { ok: true as const, exitCode: 0, envelope: { protocol_version: 1, spans: {}, error: null, worker: {},
        result: { diagnostics: { semantic_capture: capture, run_evidence: { operations: [{ operationId: 'lossy-fallback' }],
          operationCount: 999, coverage: [{ modulePath: 'incorrect', reached: false }], moduleCalls: [], truncated: [] } } } } };
      attempt!.captured(outcome); captured(outcome);
      model.projection = { ...model.projection, traced: true, document: { semantic: { metadata: { capture: { id: capture.id } },
        nodes: [{ node_id: 'root', op: 'model.Model', kind: 'module', params: {}, metadata: { captureRef: {
          version: 1, captureId: capture.id, sourceId: capture.source.id, kind: 'invocation', semanticId: 'call:0' } } }], edges: [] } } };
      measured(model.target, model.projection);
      return { status: 'passed', detail: 'Observed test forward.' };
    }, () => {}, summaries);
    server = new McpServer({ name: 'canonical-test', version: '1' });
    registerMcpValidationTools(server, graph, runs, 'owner');
    client = new Client({ name: 'reader', version: '1' });
    const [a, b] = InMemoryTransport.createLinkedPair(); await server.connect(a); await client.connect(b);
  });
  teardown(async () => { runs.clear(); await client.close(); await server.close(); });
  const call = async (name: string, args: object = {}) =>
    (await client.callTool({ name, arguments: { ...args } })).structuredContent as any;
  async function start() {
    const run = await call('validate_model', { ...graph.reference(model), inputs: [{ shape: [2, 4], dtype: 'float32', fill: 'ones' }] });
    await new Promise<void>(resolve => setImmediate(resolve));
    return String(run.runId);
  }

  test('uses the browser reader and keeps a run after slot eviction, retargeting and source edits', async () => {
    const runId = await start(), capture = structuredClone(produced);
    const selector = { entity: { kind: 'operation' as const, id: 'add2' }, path: ['arguments', 'args'], offset: 0, limit: 50 };
    const expected = inspectCapture(capture, { ...selector, captureId: capture.id, sourceId: capture.source.id });
    const initial = await call('inspect_run', { runId, section: 'capture', ...selector });
    assert.deepEqual(initial.page, expected);
    assert.equal(initial.captureUnavailable, null);
    assert.equal(initial.format, 'semantic-v1');
    const browser = new CaptureInspector(summaries.ledger);
    assert.deepEqual(initial.page, browser.inspect(model.target, { ...selector, captureId: capture.id, sourceId: capture.source.id }));
    const summary = await call('inspect_run', { runId });
    assert.equal(summary.operationCount, 2);
    assert.equal(summary.notObservedModulePathCount, 0);
    assert.ok(!JSON.stringify(summary).includes('lossy-fallback'));
    produced.operations[1].arguments.args = [];
    await start();
    assert.equal(summaries.ledger.capture(capture.id), undefined, 'live complete slot has evicted the previous capture');
    model.current = false; model.target.documentVersion++; model.target.cls = 'Other';
    assert.deepEqual((await call('inspect_run', { runId, section: 'capture', ...selector })).page, expected);
    const direct = runs.inspect('owner', runId, 'capture', 0, 50, selector) as any;
    direct.page.items.length = 0;
    assert.deepEqual((await call('inspect_run', { runId, section: 'capture', ...selector })).page, expected);
    assert.throws(() => runs.inspect('stranger', runId, 'capture'), /unavailable/);
    assert.equal(invocations, 2, 'reading retained evidence never executes a new worker');
  });

  test('retains shared registration identity and exact paths through run graph and module pages', async () => {
    transform = capture => { capture.modules[0].paths = ['', 'alias']; };
    const runId = await start();
    const graphPage = await call('inspect_run', { runId, section: 'graph' });
    assert.deepEqual(graphPage.nodes.items[0].modulePaths, ['', 'alias']);
    assert.equal(graphPage.nodes.items[0].modulePath, null, 'shared aliases never pick an arbitrary source identity');
    assert.equal(graphPage.nodes.items[0].moduleId, produced.modules[0].id);
    assert.equal(graphPage.snapshot.capture.id, produced.id);
    const calls = await call('inspect_run', { runId, section: 'modules' });
    assert.deepEqual(calls.calls.path, ['invocations']);
    assert.equal(calls.coverage.total, produced.coverage.modules.length);
    const ops = await call('inspect_run', { runId, section: 'operations', limit: 1 });
    assert.equal(ops.operations.total, 2); assert.equal(ops.operations.nextOffset, 1);
  });

  test('a malformed or foreign canonical capture cannot fall back to lossy operation claims', async () => {
    transform = capture => { capture.attemptId = 'other'; };
    const runId = await start();
    const summary = await call('inspect_run', { runId });
    assert.equal(summary.format, 'unavailable'); assert.equal(summary.captureUnavailable, 'invalid_capture');
    assert.equal((await call('inspect_run', { runId, section: 'operations' })).operations, undefined);
    assert.equal((await call('inspect_run', { runId, section: 'capture' })).error.code, 'CAPTURE_UNAVAILABLE');
  });

  test('compares canonical literals through the real MCP tool and preserves inspectable references', async () => {
    const baselineRunId = await start();
    transform = capture => { capture.operations[0].arguments.args[1] = { kind: 'int', value: '2' }; };
    const candidateRunId = await start();
    const summary = await call('compare_runs', { baselineRunId, candidateRunId });
    assert.equal(summary.format, 'semantic-v1'); assert.equal(summary.equivalentCapturedComputation, false);
    assert.equal(summary.changedFieldCount, 1); assert.ok(!JSON.stringify(summary).includes('lossy-fallback'));
    const changes = await call('compare_runs', { baselineRunId, candidateRunId, section: 'nodes' });
    const argument = changes.changes.items.find((item: any) => item.kind === 'operation' && item.field === 'arguments');
    assert.equal(argument.baseline.value.args[1].value, '1'); assert.equal(argument.candidate.value.args[1].value, '2');
    const original = await call('inspect_run', { ...argument.baselineInspect, path: ['arguments', 'args', 1, 'value'] });
    assert.equal(original.page.node.value, '1');
    const roles = await call('compare_runs', { baselineRunId, candidateRunId, section: 'correspondence', limit: 2 });
    assert.equal(roles.pairs.items.length, 2); assert.equal(roles.pairs.nextOffset, 2);
    assert.equal(invocations, 2, 'comparison and inspection must not execute another model');
  });

  test('a rejected canonical run cannot reenter the legacy layer comparer', async () => {
    const baselineRunId = await start();
    transform = capture => { capture.attemptId = 'foreign-attempt'; };
    const candidateRunId = await start();
    const summary = await call('compare_runs', { baselineRunId, candidateRunId });
    assert.equal(summary.format, 'unavailable'); assert.equal(summary.semanticComparisonAvailable, false);
    assert.equal(summary.equivalentCapturedComputation, null); assert.equal(summary.nodeChangeCount, undefined);
  });

  test('compares a large argument exactly while keeping MCP pages bounded and the original tree accessible', async () => {
    transform = capture => { capture.operations[0].arguments.kwargs.notes = {
      kind: 'list', items: Array.from({ length: 200 }, () => ({ kind: 'str', value: 'a'.repeat(3000) })) }; };
    const baselineRunId = await start();
    transform = capture => { capture.operations[0].arguments.kwargs.notes = {
      kind: 'list', items: Array.from({ length: 200 }, (_, index) => ({ kind: 'str', value: (index === 199 ? 'b' : 'a').repeat(3000) })) }; };
    const candidateRunId = await start();
    const changes = await call('compare_runs', { baselineRunId, candidateRunId, section: 'nodes' });
    assert.equal(changes.equivalentCapturedComputation, false); assert.ok(Buffer.byteLength(JSON.stringify(changes)) < 64 * 1024);
    const argument = changes.changes.items.find((item: any) => item.kind === 'operation' && item.field === 'arguments');
    assert.equal(argument.baseline.inline, false); assert.ok(argument.baseline.serializedBytes > 600000);
    const exact = await call('inspect_run', { ...argument.candidateInspect, path: ['arguments', 'kwargs', 'notes', 'items', 199, 'value'] });
    assert.equal(exact.page.node.value, 'b'.repeat(3000)); assert.equal(invocations, 2);
  });

  test('oversized single records remain navigable and invalid selectors fail explicitly', async () => {
    transform = capture => { capture.operations[0].arguments.kwargs.notes = {
      kind: 'list', items: Array.from({ length: 200 }, () => ({ kind: 'str', value: 'a'.repeat(3000) })) }; };
    const runId = await start();
    const page = await call('inspect_run', { runId, section: 'capture', entity: { kind: 'operation', id: 'add1' }, path: ['arguments', 'kwargs', 'notes', 'items'], limit: 2 });
    assert.equal(page.page.total, 200); assert.equal(page.page.nextOffset, 2);
    assert.ok(Buffer.byteLength(JSON.stringify(page.page)) < 64 * 1024);
    const value = await call('inspect_run', { runId, section: 'capture', entity: { kind: 'operation', id: 'add1' }, path: ['arguments', 'kwargs', 'notes', 'items', 199, 'value'] });
    assert.equal(value.page.node.value, 'a'.repeat(3000));
    assert.equal((await call('inspect_run', { runId, section: 'capture', path: ['__proto__'] })).error.code, 'INVALID_PATH');
    assert.equal((await call('inspect_run', { runId, section: 'operations', path: ['operations'] })).error.code, 'INVALID_QUERY');
  });
});

suite('canonical failure producer paths', () => {
  test('follows actual values through nested repeated inputs and keeps caught failures distinct', () => {
    const { capture } = canonicalFixture();
    capture.status = 'partial'; capture.output = null;
    capture.invocations[0] = { ...capture.invocations[0], status: 'failed', output: null, diagnosticId: 'failure' };
    capture.operations[1] = { ...capture.operations[1], status: 'failed', output: null, diagnosticId: 'failure' };
    capture.operations[1].arguments.args = [{ kind: 'list', items: capture.operations[1].arguments.args }];
    capture.values.pop();
    capture.diagnostics = [{ id: 'failure', phase: 'forward', code: 'invalid_shape', field: ['args', 0], names: [],
      sourceBindingId: null, remedy: 'inspect_failure' }];
    const failure = inspectSemanticEvidence(capture, 'failure', 0, 50, { path: [] }) as any;
    assert.deepEqual(failure.matchingFailureOperations.items.map((op: any) => op.id), ['add2']);
    assert.deepEqual(failure.upstream.items.map((op: any) => op.id), ['add1']);
    assert.equal(failure.failedInvocations.items[0].id, 'call:0');
    capture.invocations[0].diagnosticId = 'later-assertion';
    const assertion = inspectSemanticEvidence(capture, 'failure', 0, 50, { path: [] }) as any;
    assert.equal(assertion.matchingFailureOperations.count, 0);
    assert.equal(assertion.upstream.count, 0);
    assert.equal(assertion.failedOperations.count, 1);
  });
});
