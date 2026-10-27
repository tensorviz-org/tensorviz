import * as assert from 'node:assert/strict';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { McpGraphService, type McpModel } from '@tensorviz/graph-host/mcpGraph';
import { McpValidationRuns, registerMcpValidationTools } from '@tensorviz/graph-host/mcpValidation';
import type { LocalBrowserHostPublish } from '@tensorviz/graph-host/localBrowserHostTransport';
import type { LocalBrowserValidationOutcome, ProjectionPayload } from '@tensorviz/contracts/protocol';
import type { TensorInputs } from '@tensorviz/graph-host/tensorInputs';
import type { TraceOutcome } from '@tensorviz/graph-host/traceWorkerCore';
import type { ExecutionSpec } from '@tensorviz/graph-host/executionSpec';
import { ValidationSummaryStore } from '@tensorviz/graph-host/validationSummary';

suite('MCP validation jobs', () => {
  const inputs: TensorInputs = [{ shape: [1, 8], dtype: 'float32', fill: 'ones' }];
  let model: McpModel, graph: McpGraphService, runs: McpValidationRuns, server: McpServer, client: Client;
  let finish: (value: LocalBrowserValidationOutcome) => void;
  let signal: AbortSignal, received: TensorInputs | undefined;
  let count: number, messages: LocalBrowserHostPublish[];
  let summaries: ValidationSummaryStore;
  let capture: (outcome: TraceOutcome) => void, receivedSetup: ExecutionSpec;
  const tick = () => new Promise<void>(resolve => setImmediate(resolve));
  setup(async () => {
    count = 0; messages = [];
    summaries = new ValidationSummaryStore(summary => messages.push({ type: 'validationState', ...summary }));
    model = { target: { workspaceFolder: '/ws', file: 'net.py', cls: 'Net', documentVersion: 1,
      sourceDigest: 'a'.repeat(64), backendRevision: 1 }, current: true,
      projection: { revision: 1, traced: false, spanmap: {}, document: { semantic: { nodes: [], edges: [] } } } as unknown as ProjectionPayload };
    graph = new McpGraphService('host', () => model);
    runs = new McpValidationRuns(() => model, (_target, abort, tensors, progress, measured, captured, execution, job, attempt) => {
      capture = outcome => { attempt?.captured(outcome); captured(outcome); }; receivedSetup = execution;
      count++; signal = abort; received = tensors; progress('tracing');
      return new Promise(resolve => { finish = value => {
        if ((value.status === 'passed' || value.status === 'failed') && model.projection.traced) measured(model.target, model.projection);
        resolve(value);
      }; });
    }, message => messages.push(message), summaries);
    server = new McpServer({ name: 'fixture', version: '1' });
    registerMcpValidationTools(server, graph, runs, 'client-one');
    client = new Client({ name: 'any-compatible-agent', version: '1' });
    const [a, b] = InMemoryTransport.createLinkedPair(); await server.connect(a); await client.connect(b);
  });
  teardown(async () => { runs.clear(); if (finish) finish({ status: 'cancelled', detail: 'closed' }); await client.close(); await server.close(); });
  async function call(name: string, args: Record<string, unknown>) {
    const value = await client.callTool({ name, arguments: args });
    return { error: value.isError, data: value.structuredContent as any };
  }
  const ref = () => graph.reference(model);
  const start = () => call('validate_model', { ...ref(), inputs });
  test('polling and paging do not start another run', async () => {
    const run = (await start()).data;
    await call('get_run', { runId: run.runId }); await call('get_run', { runId: run.runId });
    assert.equal(count, 1);
    finish({ status: 'failed', detail: 'private-source-and-exception' }); await tick();
    assert.equal((await call('inspect_run', { runId: run.runId, section: 'graph', offset: 1 })).error, undefined);
    assert.equal((await call('get_run', { runId: run.runId })).data.status, 'failed');
    assert.equal(count, 1);
  });
  function measured() { model = { ...model, target: { ...model.target, backendRevision: 2 },
    projection: { ...model.projection, traced: true, revision: 2 } }; }

  test('publishes standard SDK tool schemas for explicit execution and retained inspection', async () => {
    const listed = await client.listTools();
    const validate = listed.tools.find(tool => tool.name === 'validate_model');
    assert.ok(validate?.inputSchema.properties?.execution);
    assert.ok(listed.tools.some(tool => tool.name === 'inspect_run'));
    assert.equal(listed.tools.some(tool => tool.name === 'profile_model' || tool.name === 'capture_training_step'), false);
    const inspect = listed.tools.find(tool => tool.name === 'inspect_run');
    assert.doesNotMatch(JSON.stringify(inspect?.inputSchema), /profile_modules|profile_operations|benchmark/);
  });

  test('returns immediately, reports progress, and retains results after revision changes', async () => {
    const original = ref(), run = (await start()).data;
    assert.ok(run.runId); assert.equal(count, 1); assert.deepEqual(received, inputs);
    assert.equal((await call('get_run', { runId: run.runId })).data.phase, 'tracing');
    assert.equal((await start()).data.error.code, 'BUSY');
    measured(); finish({ status: 'passed', detail: 'Passed for this input.' }); await tick();
    const result = (await call('get_run', { runId: run.runId })).data;
    assert.equal(result.status, 'passed'); assert.equal(result.revision, original.revision);
    assert.equal(result.attempt.id, run.runId);
    assert.equal(result.attempt.origin, 'mcp');
    assert.equal(result.attempt.status, 'succeeded');
    assert.match(result.attempt.requestDigest, /^[a-f0-9]{64}$/);
    assert.equal(result.result.revision, ref().revision);
    assert.deepEqual(messages.at(-1), { type: 'validationState', runId: run.runId, target: model.target,
      status: 'passed', detail: 'Passed for this input.', projectionRevision: 2,
      evidence: { origin: 'mcp', attemptStatus: 'succeeded', sourceId: null, planDigest: null,
        lastCompleteCaptureId: null, lastPartialCaptureId: null, displayedCapture: null, setup: null } });
    assert.equal(summaries.snapshot(model.target)!.runId, run.runId);
    assert.equal(summaries.snapshot(model.target)!.projectionRevision, 2);
    model.target.documentVersion++;
    assert.equal((await call('get_run', { runId: run.runId })).data.status, 'passed');
  });
  test('run inspection distinguishes a timeout with no capture from an ordinary failed forward', async () => {
    const run = (await start()).data;
    capture({ ok: false, timedOut: true, reason: 'stopped', detail: 'private worker text' });
    finish({ status: 'failed', detail: 'Execution timed out.' }); await tick();
    const result = (await call('inspect_run', { runId: run.runId })).data;
    assert.equal(result.attempt.status, 'timed_out');
    assert.deepEqual(result.attempt.captureIds, []);
    assert.equal(result.attempt.planDigest, null);
    assert.ok(!JSON.stringify(result).includes('private worker text'));
  });
  test('a repeated validation of identical shapes can still certify that run', async () => {
    measured();
    const before = ref(), run = (await start()).data;
    finish({ status: 'passed', detail: 'Same shapes measured again.' }); await tick();
    const result = (await call('get_run', { runId: run.runId })).data;
    assert.equal(result.status, 'passed'); assert.equal(result.result.revision, before.revision);
  });
  test('refuses stale targets before starting and rejects expressions', async () => {
    const old = ref(); model.target.documentVersion++;
    assert.equal((await call('validate_model', { ...old, inputs })).data.error.code, 'STALE_TARGET');
    assert.ok((await call('validate_model', { ...ref(), inputs: 'torch.randn(8)' })).error);
    assert.equal(count, 0);
  });
  test('only the owning session can poll or cancel; cancellation wins over a late success', async () => {
    const run = (await start()).data;
    assert.throws(() => runs.get('someone-else', run.runId), /unavailable/);
    assert.throws(() => runs.cancel('someone-else', run.runId), /unavailable/);
    assert.equal(signal.aborted, false);
    assert.equal((await call('cancel_run', { runId: run.runId })).data.status, 'cancelling');
    assert.equal(signal.aborted, true); measured(); finish({ status: 'passed', detail: 'late' }); await tick();
    const result = (await call('get_run', { runId: run.runId })).data;
    assert.equal(result.status, 'cancelled'); assert.equal(result.result, null);
  });
  test('a failed forward retains its partial projection; refusal never borrows old shapes', async () => {
    const first = (await start()).data; measured(); finish({ status: 'failed', detail: 'Partial forward.' }); await tick();
    assert.equal((await call('get_run', { runId: first.runId })).data.result.revision, ref().revision);
    const second = (await start()).data; finish({ status: 'refused', detail: 'Save dependencies.' }); await tick();
    assert.equal((await call('get_run', { runId: second.runId })).data.result, null);
    assert.ok(!('projectionRevision' in messages.at(-1)!));
    assert.equal(summaries.snapshot(model.target)!.runId, second.runId);
    assert.equal(summaries.snapshot(model.target)!.projectionRevision, undefined);
  });
  test('changed source cannot receive a passed verdict or sweep', async () => {
    const run = (await start()).data; measured(); model.current = false;
    finish({ status: 'passed', detail: 'late' }); await tick();
    assert.equal((await call('get_run', { runId: run.runId })).data.status, 'refused');
    assert.ok(!('projectionRevision' in messages.at(-1)!));
  });
  test('session disposal cancels execution and removes its results', async () => {
    const run = (await start()).data; runs.cancelOwner('client-one');
    assert.equal(signal.aborted, true); finish({ status: 'cancelled', detail: 'stopped' }); await tick();
    assert.equal((await call('get_run', { runId: run.runId })).data.error.code, 'UNKNOWN_RUN');
  });

  test('retains immutable failure paths, shapes, setup and graph after live edits', async () => {
    const run = (await start()).data;
    const evidence = { operationCount: 2, moduleCallCount: 2, truncated: [],
      operations: [{ operationId: 'op-0', index: 0, operation: 'aten.avg_pool2d', modulePath: 'down',
        inputs: [{ shape: [1, 1, 56, 56], inputPath: 'args[0]' }], outputs: [{ shape: [1, 1, 28, 28] }], status: 'completed' }],
      failure: { operationId: 'op-1', operation: 'aten.add.Tensor', source: { file: '/ws/net.py', line: 12 },
        inputs: [{ shape: [1, 1, 28, 28], producerOperationId: 'op-0' }, { shape: [1, 1, 56, 56], inputPath: 'args[0]' }], outputs: [] },
      coverage: [{ modulePath: 'down', reached: true }, { modulePath: 'later', reached: false }], moduleCalls: [] };
    capture({ ok: false, reason: 'private-exception-text', detail: 'private-stderr', exitCode: 4,
      envelope: { protocol_version: 1, worker: { torch: '2.7.1' }, spans: {},
        error: { type: 'RuntimeError', message: 'private-exception-text', frames: [{ file: '/ws/net.py', line: 12, name: 'forward' }] },
        result: { diagnostics: { run_evidence: evidence, execution: { mode: 'eval', environment: { device: 'cpu' }, state: { digest: 'state-a' } } } } } });
    measured();
    model.projection.document = { semantic: { nodes: [{ node_id: 'down', op: 'AvgPool2d', params: {
      _module_path: 'down', _shape_provenance: 'observed', _out_shape: '1x1x28x28' } }], edges: [] } };
    finish({ status: 'failed', detail: 'Check tensorViz Output in VS Code.' }); await tick();
    evidence.failure.inputs[0].shape[2] = 99;
    model.target.documentVersion++;
    model.projection.document = { semantic: { nodes: [], edges: [] } };
    const failure = (await call('inspect_run', { runId: run.runId, section: 'failure' })).data;
    assert.equal(failure.immutable, true); assert.equal(failure.status, 'failed');
    assert.deepEqual(failure.failure.inputs.map((input: any) => input.shape), [[1, 1, 28, 28], [1, 1, 56, 56]]);
    assert.equal(failure.upstream.items[0].modulePath, 'down');
    assert.equal(failure.error.frames[0].line, 12);
    assert.ok(!JSON.stringify(failure).includes('private-'));
    const frozen = (await call('inspect_run', { runId: run.runId, section: 'graph' })).data;
    assert.deepEqual(frozen.nodes.items[0].shapes.output, [1, 1, 28, 28]);
    assert.equal(frozen.nodes.items[0].modulePath, 'down');
    const setup = (await call('inspect_run', { runId: run.runId, section: 'setup' })).data;
    assert.equal(setup.setup.state.digest, 'state-a');
    const modules = (await call('inspect_run', { runId: run.runId, section: 'modules', limit: 1 })).data;
    assert.equal(modules.coverage.items[0].modulePath, 'down'); assert.equal(modules.coverage.nextOffset, 1);
    assert.throws(() => runs.inspect('other-session', run.runId), /unavailable/);
    const direct = runs.inspect('client-one', run.runId, 'failure') as any;
    direct.failure.inputs[0].shape[2] = 500;
    assert.equal((runs.inspect('client-one', run.runId, 'failure') as any).failure.inputs[0].shape[2], 28);
    capture({ ok: true, exitCode: 0, envelope: { protocol_version: 1, result: null, error: null, spans: {}, worker: {} } });
    assert.equal((runs.inspect('client-one', run.runId, 'failure') as any).failure.inputs[0].shape[2], 28);
  });

  test('setup paging keeps large module, checkpoint and import inventories readable', async () => {
    const run = (await start()).data;
    const names = Array.from({ length: 4000 }, (_, i) => `layer_${i}_${'x'.repeat(150)}`);
    capture({ ok: true, exitCode: 0, envelope: { protocol_version: 1, worker: {}, spans: {}, error: null,
      result: { diagnostics: { execution: { moduleModes: Object.fromEntries(names.map(name => [name, false])),
        resolvedPlan: { id: 'plan', source: { id: 'source', files: names.map(path => ({ path, digest: 'abc', role: 'source' })) } },
        importedSources: names.map(file => ({ file, digest: 'abc' })), setup: { checkpoint: { missingKeys: names, unexpectedKeys: names } } } } } } });
    finish({ status: 'passed', detail: 'passed' }); await tick();
    const result = await call('inspect_run', { runId: run.runId, section: 'setup', limit: 2 });
    assert.ok(!result.error);
    assert.equal(result.data.moduleModes.items.length, 2);
    assert.equal(result.data.moduleModes.nextOffset, 2);
    assert.equal(result.data.checkpointKeys.count, 8000);
    assert.equal(result.data.importedSources.count, 4000);
    assert.equal(result.data.planFiles.count, 4000);
    assert.equal(result.data.planFiles.items.length, 2);
    assert.equal(result.data.setup.resolvedPlan.source.files, undefined);
    assert.equal(result.data.setup.moduleModes, undefined);
    assert.equal(result.data.setup.setup.checkpoint.missingKeys, undefined);
    assert.ok(JSON.stringify(result.data).length < 10000);
  });

  test('accepts explicit fixture/device setup while rejecting ambiguous input contracts', async () => {
    const execution = { inputs: { fixture: 'fixtures/batch.pt' }, device: 'cuda:0',
      checkpoint: { file: 'weights.pt' }, constructorArgs: { width: 32 } };
    assert.equal((await call('validate_model', { ...ref(), inputs, execution })).data.error.code, 'INVALID_SETUP');
    assert.equal((await call('validate_model', { ...ref() })).data.error.code, 'INVALID_SETUP');
    const result = (await call('validate_model', { ...ref(), execution })).data;
    assert.ok(result.runId); assert.equal(received, undefined);
    assert.equal(receivedSetup.device, 'cuda:0'); assert.equal(receivedSetup.checkpoint?.strict, true);
    finish({ status: 'refused', detail: 'Unavailable device.' }); await tick();
    assert.equal((await call('inspect_run', { runId: result.runId })).data.evidenceAvailable, false);
  });

  test('worker setup failures are inspectable without borrowing an older projection', async () => {
    const run = (await start()).data;
    capture({ ok: false, reason: 'private setup text', detail: '', exitCode: 3,
      envelope: { protocol_version: 1, result: null, spans: {}, worker: { torch: '2.7.1' },
        error: { type: 'ModuleNotFoundError', message: 'private setup text', frames: [] } } });
    finish({ status: 'refused', detail: 'Construction failed.' }); await tick();
    const inspected = (await call('inspect_run', { runId: run.runId, section: 'failure' })).data;
    assert.equal(inspected.error.type, 'ModuleNotFoundError'); assert.equal(inspected.error.stage, 'setup');
    assert.equal(inspected.result, null); assert.equal(inspected.failure, null);
  });

  test('a second worker callback cannot replace evidence while the run is still active', async () => {
    const run = (await start()).data;
    const outcome = (name: string): TraceOutcome => ({ ok: true, exitCode: 0,
      envelope: { protocol_version: 1, result: { diagnostics: { run_evidence: { operations: [{ operationId: name }] } } },
        spans: {}, error: null, worker: {} } });
    capture(outcome('first'));
    capture(outcome('late-replacement'));
    const inspected = await call('inspect_run', { runId: run.runId, section: 'operations' });
    assert.equal(inspected.data.items[0].operationId, 'first');
    finish({ status: 'failed', detail: 'Stopped.' }); await tick();
  });


  test('evicts old evidence by bytes as well as record count', async function () {
    this.timeout(15_000);
    const payload = 'x'.repeat(10 * 1024 * 1024);
    let first = '', last = '';
    for (let index = 0; index < 8; index++) {
      const run = (await start()).data; first ||= run.runId; last = run.runId;
      capture({ ok: true, exitCode: 0, envelope: { protocol_version: 1, spans: {}, error: null, worker: {},
        result: { diagnostics: { execution: { retainedMetadata: payload } } } } });
      finish({ status: 'failed', detail: 'Retained test record.' }); await tick();
    }
    assert.equal((await call('get_run', { runId: first })).data.error.code, 'UNKNOWN_RUN');
    assert.equal((await call('get_run', { runId: last })).data.evidenceAvailable, true);
  });

  test('oversized evidence is disclosed while the original graph remains readable', async function () {
    this.timeout(10_000);
    const run = (await start()).data;
    capture({ ok: true, exitCode: 0, envelope: { protocol_version: 1, spans: {}, error: null, worker: {},
      result: { diagnostics: { execution: { retainedMetadata: 'x'.repeat(16 * 1024 * 1024) } } } } });
    finish({ status: 'failed', detail: 'Result exceeded retention budget.' }); await tick();
    const inspected = (await call('inspect_run', { runId: run.runId, section: 'graph' })).data;
    assert.equal(inspected.evidenceAvailable, false);
    assert.match(inspected.evidenceUnavailable, /16 MiB/);
    assert.equal(inspected.snapshot.traced, false); assert.deepEqual(inspected.nodes.items, []);
  });

  test('retention is bounded while the most recent result stays observable', async () => {
    let first = '', last = '';
    for (let i = 0; i < 33; i++) {
      const run = (await start()).data; first ||= run.runId; last = run.runId;
      finish({ status: 'refused', detail: 'No execution.' }); await tick();
    }
    assert.equal((await call('get_run', { runId: first })).data.error.code, 'UNKNOWN_RUN');
    assert.equal((await call('get_run', { runId: last })).data.status, 'refused');
  });
  test('compare_runs enforces completion, ownership, expiry and read-only retained paging', async () => {
    const first = (await start()).data;
    assert.equal((await call('compare_runs', { baselineRunId: first.runId, candidateRunId: first.runId })).data.error.code, 'RUN_IN_PROGRESS');
    finish({ status: 'failed', detail: 'First retained run.' }); await tick();
    const second = (await start()).data;
    finish({ status: 'failed', detail: 'Second retained run.' }); await tick();
    const comparisons = await call('compare_runs', { baselineRunId: first.runId, candidateRunId: second.runId, section: 'nodes', limit: 1 });
    assert.equal(comparisons.error, undefined); assert.equal(comparisons.data.changes.count, 0);
    assert.equal(count, 2, 'inspection never starts another worker');
    assert.throws(() => runs.compare('other', first.runId, second.runId), /unavailable/);
    assert.ok((await call('compare_runs', { baselineRunId: first.runId, candidateRunId: second.runId, atol: -1 })).error);
    const now = Date.now;
    try {
      Date.now = () => now() + 31 * 60_000;
      assert.throws(() => runs.compare('client-one', first.runId, second.runId), /unavailable/);
    } finally { Date.now = now; }
  });

});
