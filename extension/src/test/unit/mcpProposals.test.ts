import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { GraphMcpServer } from '../../mcpServer';
import { McpGraphService, type McpModel } from '@tensorviz/graph-host/mcpGraph';
import { McpProposals, registerMcpProposalTools } from '../../mcpProposals';
import type { SourceProposalOutcome, SourceProposalRequest, SourceProposalReview } from '../../localBrowserSourceEdits';
import type { ProjectionPayload } from '@tensorviz/contracts/protocol';
import { CandidateHistory } from '../../candidateHistory';
import { CandidateReview } from '../../candidateReview';
import type { CandidateRunResult } from '@tensorviz/graph-host/candidateExecution';
import type { CandidateExecution } from '../../executionPlan';
import { canonicalFixture, appliedFixture } from '../../../../tests/fixtures/semantic.cjs';
import { GraphToolError } from '@tensorviz/graph-host/mcpGraph';

suite('standard HTTP MCP source proposals', () => {
  let endpoint: GraphMcpServer, graph: McpGraphService, proposals: McpProposals, model: McpModel;
  let clients: Array<{ client: Client; transport: StreamableHTTPClientTransport }>;
  let received: SourceProposalRequest[], signal: AbortSignal;
  let finish: ((result: SourceProposalOutcome) => void) | undefined;
  let report: ((review: SourceProposalReview) => void) | undefined;
  let history: CandidateHistory, candidate: CandidateReview | undefined, executions: number;
  let execute: (signal: AbortSignal) => Promise<CandidateRunResult>;
  let appliedReader: (requestId: string) => Promise<ReturnType<typeof appliedFixture>>;
  let owners: string[];
  function candidateResult(): CandidateRunResult {
    const { capture, plan } = canonicalFixture('candidate:mcp');
    return { evidence: { execution: { id: plan.id, plan } as CandidateExecution, capture, baseCurrent: true, candidateCurrent: true },
      outcome: { ok: true, exitCode: 0, executionCurrent: true, recheckExecution: async () => true,
        envelope: { protocol_version: 1, result: {}, error: null, worker: {}, spans: {} } } };
  }
  const edit = { kind: 'set_param', nodeId: 'fc', param: 'out_features', value: 16 };
  const tick = () => new Promise<void>(resolve => setImmediate(resolve));
  setup(async () => {
    clients = []; received = []; finish = undefined; history = new CandidateHistory(); candidate = undefined;
    executions = 0; owners = []; execute = async () => candidateResult();
    appliedReader = async requestId => { assert.equal(requestId, received.at(-1)?.requestId); return appliedFixture(); };
    model = { target: { workspaceFolder: '/ws', file: 'net.py', cls: 'Net', documentVersion: 1,
      sourceDigest: 'a'.repeat(64), backendRevision: 1 }, current: true, projection: {
        document: { semantic: { nodes: [{ node_id: 'fc' }, { node_id: 'relu' }], edges: [] } },
      } as unknown as ProjectionPayload };
    proposals = new McpProposals({ propose: async (request, abort, review) => {
      report = review;
      received.push(request); signal = abort!;
      if (signal.aborted) return { status: 'cancelled', detail: 'Stopped.' };
      const metadata = { title: 'Source diff — also affects 2 other projected instances', file: 'net.py', beforeDigest: 'a'.repeat(64), afterDigest: 'b'.repeat(64) };
      candidate = ['set_literal', 'replace_activation', 'reconnect_argument', 'insert_unary_op', 'delete_unary_op', 'set_constructor_literal', 'set_configuration_literal'].includes(request.edit.kind) ? new CandidateReview(canonicalFixture().capture, signal, () => model.current,
        async abort => { executions++; return execute(abort); }, { history, requestId: request.requestId }) : undefined;
      candidate?.onDidChange(value => review?.({ ...metadata, candidate: value }));
      review?.({ ...metadata, ...(candidate ? { candidate: candidate.snapshot() } : {}) });
      const outcome = await new Promise<SourceProposalOutcome>(resolve => { finish = resolve; signal.addEventListener('abort', () => resolve({ status: 'cancelled', detail: 'Stopped.' }), { once: true }); });
      await candidate?.finish(); return outcome;
    }, runCandidate: requestId => {
      if (requestId !== received.at(-1)?.requestId || !candidate?.available) throw new GraphToolError('CANDIDATE_UNAVAILABLE', 'No replay recipe.');
      void candidate.run(); return history.get(requestId, candidate.snapshot().runId!);
    }, getCandidate: (requestId, runId) => history.get(requestId, runId),
    listCandidates: requestId => history.list(requestId),
    inspectCandidate: (requestId, runId, query) => history.inspect(requestId, runId, query),
    cancelCandidate: (requestId, runId) => {
      const retained = history.get(requestId, runId);
      if (requestId === received.at(-1)?.requestId && candidate?.snapshot().runId === runId) candidate.cancel();
      return retained;
    }, forgetCandidates: requestId => history.forget(requestId), inspectAppliedSource: requestId => appliedReader(requestId) });
    endpoint = new GraphMcpServer((server, owner) => {
      owners.push(owner);
      registerMcpProposalTools(server, graph, proposals, owner);
      return () => proposals.cancelOwner(owner);
    });
    graph = new McpGraphService(endpoint.hostId, () => model);
    await endpoint.start();
    await connect('any-compatible-agent');
  });
  teardown(async () => {
    proposals.clear(); finish?.({ status: 'cancelled', detail: 'Stopped.' });
    await Promise.allSettled(clients.map(({ client }) => client.close())); await endpoint.close();
  });
  async function connect(name: string) {
    const connection = endpoint.connection!;
    const client = new Client({ name, version: '1' });
    const transport = new StreamableHTTPClientTransport(new URL(connection.url));
    await client.connect(transport); const pair = { client, transport }; clients.push(pair); return pair;
  }
  async function call(name: string, args: Record<string, unknown>, client = clients[0].client) {
    const result = await client.callTool({ name, arguments: args }); return { error: result.isError, data: result.structuredContent as any };
  }
  const ref = () => graph.reference(model);
  const start = (value: unknown = edit) => call('propose_edit', { ...ref(), edit: value });
  const startCaptured = () => {
    const { intent } = JSON.parse(readFileSync(join(__dirname, '../../../../tests/fixtures/contracts/captured-source-edit.json'), 'utf8'));
    return start({ ...intent, nodeId: 'fc' });
  };

  test('applied-source inspection is read-only, owner-scoped and independent of candidate execution', async () => {
    const proposalId = (await startCaptured()).data.proposalId;
    assert.equal((await call('inspect_applied_proposal', { proposalId })).data.error.code, 'APPLIED_SOURCE_UNAVAILABLE');
    finish!({ status: 'applied', detail: 'Applied.' }); await tick();
    const before = (await call('get_proposal', { proposalId })).data;
    const result = await call('inspect_applied_proposal', { proposalId });
    assert.equal(result.error, undefined); assert.deepEqual(result.data.applied, appliedFixture()); assert.equal(executions, 0);
    assert.deepEqual((await call('get_proposal', { proposalId })).data, before);
    const other = await connect('other-owner');
    assert.equal((await call('inspect_applied_proposal', { proposalId }, other.client)).data.error.code, 'UNKNOWN_PROPOSAL');
    assert.deepEqual((await clients[0].client.listTools()).tools.find(tool => tool.name === 'inspect_applied_proposal')!.annotations,
      { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false });
  });

  test('revoking an MCP owner during source inspection prevents a late evidence reply', async () => {
    const proposalId = (await startCaptured()).data.proposalId;
    finish!({ status: 'applied', detail: 'Applied.' }); await tick();
    let release!: (value: ReturnType<typeof appliedFixture>) => void;
    appliedReader = async () => new Promise(resolve => { release = resolve; });
    const pending = proposals.inspectAppliedSource(owners[0], proposalId);
    proposals.cancelOwner(owners[0]); release(appliedFixture());
    await assert.rejects(pending, /unavailable/); assert.equal(executions, 0);
  });

  test('advertises standard schemas, returns before native review completes, and never advertises automatic apply', async () => {
    const tools = (await clients[0].client.listTools()).tools;
    assert.deepEqual(tools.map(tool => tool.name), ['propose_edit', 'get_proposal', 'inspect_applied_proposal', 'run_proposal_candidate',
      'cancel_proposal_candidate', 'list_proposal_candidates', 'inspect_proposal_candidate']);
    assert.deepEqual(tools.find(tool => tool.name === 'run_proposal_candidate')!.annotations,
      { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: true });
    assert.ok(JSON.stringify(tools[0].inputSchema).includes('group_layers'));
    const proposal = (await start()).data;
    assert.ok(proposal.proposalId); assert.deepEqual(received[0].edit, edit);
    const state = (await call('get_proposal', { proposalId: proposal.proposalId })).data;
    assert.equal(state.status, 'review'); assert.match(state.review.title, /2 other projected instances/);
    assert.ok(!('before' in state.review));
    finish!({ status: 'applied', detail: 'Applied to the unsaved buffer; use Undo to revert.' }); await tick();
    model.target.documentVersion++;
    const result = (await call('get_proposal', { proposalId: proposal.proposalId })).data;
    assert.equal(result.status, 'applied'); assert.equal(result.target.documentVersion, 1);
  });
  test('isolates review state between clients and cancels on session termination', async () => {
    const proposal = (await start()).data;
    const other = await connect('another-agent');
    assert.equal((await call('get_proposal', { proposalId: proposal.proposalId }, other.client)).data.error.code, 'UNKNOWN_PROPOSAL');
    assert.equal((await start()).data.error.code, 'BUSY');
    await clients[0].transport.terminateSession(); await tick();
    assert.equal(signal.aborted, true);
  });
  test('retains the native candidate summary separately from the source Apply outcome', async () => {
    const { intent } = JSON.parse(readFileSync(join(__dirname, '../../../../tests/fixtures/contracts/captured-source-edit.json'), 'utf8'));
    const proposal = (await start({ ...intent, nodeId: 'fc' })).data;
    const snapshot: SourceProposalReview = { title: 'Native reviewed candidate', file: 'net.py',
      beforeDigest: 'a'.repeat(64), afterDigest: 'b'.repeat(64), candidate: {
        staticCheck: 'passed', runtime: 'failed', freshness: 'current', capture: 'partial',
        inputs: 'same', state: 'unavailable', cleanup: 'complete', semanticPostconditions: 'not_checked', diagnostics: ['invalid_shape'],
      } };
    report!(snapshot);
    const state = (await call('get_proposal', { proposalId: proposal.proposalId })).data;
    assert.equal(state.status, 'review'); assert.deepEqual(state.review.candidate, snapshot.candidate);
    finish!({ status: 'applied', detail: 'Source applied explicitly; copied candidate failed.' }); await tick();
    const completed = (await call('get_proposal', { proposalId: proposal.proposalId })).data;
    assert.equal(completed.status, 'applied'); assert.equal(completed.review.candidate.runtime, 'failed');
  });
  test('rejects stale and unknown node targets before reaching source services', async () => {
    const old = ref(); model.target.documentVersion++;
    assert.equal((await call('propose_edit', { ...old, edit })).data.error.code, 'STALE_TARGET');
    assert.equal((await start({ ...edit, nodeId: 'elsewhere' })).data.error.code, 'UNKNOWN_NODE');
    assert.equal(received.length, 0);
  });
  test('rejects arbitrary code, paths, automatic apply and invalid literal bounds', async () => {
    for (const value of [{ kind: 'chat', prompt: 'rewrite everything', nodeIds: [] }, { ...edit, path: '../secret.py' },
      { ...edit, apply: true }, { ...edit, value: { expression: 'eval(1)' } }, { ...edit, value: 'x'.repeat(1025) },
      { kind: 'group_layers', name: 'block', nodeIds: ['fc', 'fc'] }]) assert.ok((await start(value)).error);
    assert.equal(received.length, 0);
  });
  test('supports all four deterministic intents while retaining refusal details', async () => {
    for (const value of [edit, { kind: 'remove_layer', nodeId: 'fc' },
      { kind: 'add_layer', afterNodeId: 'fc', nodeId: 'drop', layerType: 'Dropout', params: { p: 0.1 } },
      { kind: 'group_layers', nodeIds: ['fc', 'relu'], name: 'block' }]) {
      const proposal = (await start(value)).data;
      assert.deepEqual(received.at(-1)!.edit, value);
      finish!({ status: 'refused', detail: 'A shared expression cannot be replaced with a literal.' }); await tick();
      assert.match((await call('get_proposal', { proposalId: proposal.proposalId })).data.detail, /shared expression/);
    }
  });

  test('explicit execution returns immediately and retry IDs never reexecute after closure or eviction', async () => {
    let release!: (value: CandidateRunResult) => void;
    execute = () => new Promise(resolve => { release = resolve; });
    const { proposalId } = (await startCaptured()).data, requestId = randomUUID();
    assert.equal(executions, 0);
    const first = (await call('run_proposal_candidate', { proposalId, requestId })).data.candidate;
    assert.equal(first.summary.runtime, 'running'); assert.equal(executions, 1);
    assert.equal((await call('run_proposal_candidate', { proposalId, requestId })).data.candidate.runId, first.runId);
    assert.equal((await call('run_proposal_candidate', { proposalId, requestId: randomUUID() })).data.candidate.runId, first.runId);
    release(candidateResult()); await tick();
    finish!({ status: 'applied', detail: 'Applied explicitly.' }); await tick(); model.current = false;
    const retried = (await call('run_proposal_candidate', { proposalId, requestId })).data.candidate;
    assert.equal(retried.runId, first.runId); assert.equal(retried.review, 'historical'); assert.equal(executions, 1);
    assert.equal((await call('run_proposal_candidate', { proposalId, requestId: randomUUID() })).data.error.code, 'CANDIDATE_UNAVAILABLE');
    history.clear();
    assert.equal((await call('run_proposal_candidate', { proposalId, requestId })).data.error.code, 'UNKNOWN_CANDIDATE');
    assert.equal(executions, 1);
  });

  test('retains native-run failure pages and independent static/runtime observations after Discard', async () => {
    execute = async () => {
      const value = candidateResult(); value.outcome = { ok: false, reason: 'failure', detail: '', executionCurrent: true };
      value.evidence!.capture!.status = 'partial';
      value.evidence!.postconditions = { version: 1, executionId: value.evidence!.execution.id, captureId: value.evidence!.capture!.id,
        scope: 'observed_expression', recipe: 'binary_scalar_v1', status: 'unavailable', reason: 'capture_incomplete',
        bindingId: null, observedOperations: 0 };
      return value;
    };
    const { proposalId } = (await startCaptured()).data;
    await candidate!.run(); const runId = candidate!.snapshot().runId!;
    finish!({ status: 'cancelled', detail: 'Discarded.' }); await tick(); model.current = false;
    const list = (await call('list_proposal_candidates', { proposalId })).data.candidates;
    assert.equal(list.length, 1); assert.equal(list[0].runId, runId); assert.equal(list[0].summary.runtime, 'failed');
    assert.equal(list[0].summary.staticCheck, 'passed'); assert.equal(list[0].summary.capture, 'partial');
    assert.equal(list[0].summary.semanticPostconditions, 'unavailable');
    assert.equal(list[0].summary.postconditions.reason, 'capture_incomplete');
    const info = (await call('inspect_proposal_candidate', { proposalId, runId })).data;
    assert.equal(info.candidate.review, 'historical');
    assert.deepEqual(info.candidate.summary.postconditions, list[0].summary.postconditions);
    const query = { ...info.candidate.capture, path: ['operations'], offset: 0, limit: 1 };
    const first = (await call('inspect_proposal_candidate', { proposalId, runId, query })).data.page;
    assert.equal(first.items.length, 1); assert.equal(first.nextOffset, 1);
    const second = (await call('inspect_proposal_candidate', { proposalId, runId, query: { ...query, offset: first.nextOffset } })).data.page;
    assert.notEqual(first.items[0].key, second.items[0].key);
    assert.equal((await call('inspect_proposal_candidate', { proposalId, runId, query: { ...query, sourceId: 'wrong' } })).data.error.code, 'SOURCE_MISMATCH');
  });

  for (const [fixture, recipe] of [['captured-activation-edit', 'functional_activation_v1'],
    ['captured-module-activation-edit', 'module_activation_v1'], ['captured-reconnection-edit', 'straight_line_tensor_v1'],
    ['captured-unary-import', 'straight_line_unary_insert_v1'], ['captured-unary-delete', 'straight_line_unary_delete_v1'],
    ['captured-constructor-edit', 'constructor_scalar_v1'], ['captured-configuration-edit', 'configuration_scalar_v1']] as const) {
    test(`retains scoped ${recipe} evidence through standard agent controls`, async () => {
      execute = async () => {
        const result = candidateResult();
        result.evidence!.postconditions = { version: 1, executionId: result.evidence!.execution.id,
          captureId: result.evidence!.capture!.id, recipe,
          scope: recipe === 'configuration_scalar_v1' ? 'observed_configuration_arguments'
            : recipe === 'constructor_scalar_v1' ? 'observed_initializer_entries' : 'observed_expression', status: 'verified',
          reason: recipe === 'straight_line_tensor_v1' ? 'observed_requested_argument'
            : recipe === 'straight_line_unary_insert_v1' ? 'observed_requested_insertion'
              : recipe === 'straight_line_unary_delete_v1' ? 'observed_requested_deletion'
                : recipe === 'configuration_scalar_v1' ? 'observed_requested_configuration'
                  : recipe === 'constructor_scalar_v1' ? 'observed_requested_constructor' : 'observed_requested_activation',
          bindingId: 'binding:expression', observedOperations: ['constructor_scalar_v1', 'configuration_scalar_v1'].includes(recipe) ? 0 : 2,
          observedInitializers: ['constructor_scalar_v1', 'configuration_scalar_v1'].includes(recipe) ? 2 : 0,
          configurationUses: recipe === 'configuration_scalar_v1' ? [{ bindingId: 'binding:expression', sourceArgumentPath: ['args', 0] }] : [] };
        return result;
      };
      const { intent } = JSON.parse(readFileSync(join(__dirname, `../../../../tests/fixtures/contracts/${fixture}.json`), 'utf8'));
      const { proposalId } = (await start({ ...intent, nodeId: 'fc' })).data;
      const started = await call('run_proposal_candidate', { proposalId, requestId: randomUUID() });
      const runId = started.data.candidate.runId; await tick();
      const state = (await call('get_proposal', { proposalId })).data;
      assert.equal(state.review.candidate.semanticPostconditions, 'verified');
      assert.equal(state.review.candidate.postconditions.recipe, recipe);
      finish!({ status: 'cancelled', detail: 'Discarded explicitly.' }); await tick();
      const historical = (await call('inspect_proposal_candidate', { proposalId, runId })).data.candidate;
      assert.equal(historical.review, 'historical'); assert.deepEqual(historical.summary.postconditions, state.review.candidate.postconditions);
      assert.equal(executions, 1);
    });
  }

  test('all candidate controls resolve session ownership before reaching source services', async () => {
    const { proposalId } = (await startCaptured()).data, other = await connect('candidate-intruder');
    const run = (await call('run_proposal_candidate', { proposalId, requestId: randomUUID() })).data.candidate;
    for (const [name, args] of [
      ['run_proposal_candidate', { proposalId, requestId: randomUUID() }], ['list_proposal_candidates', { proposalId }],
      ['cancel_proposal_candidate', { proposalId, runId: run.runId }], ['inspect_proposal_candidate', { proposalId, runId: run.runId }],
    ] as const) assert.equal((await call(name, args, other.client)).data.error.code, 'UNKNOWN_PROPOSAL');
    assert.equal(executions, 1);
    assert.equal((await call('list_proposal_candidates', { proposalId })).data.candidates.length, 1);
  });

  test('revoking a session cancels execution and cannot resurrect evidence on late completion', async () => {
    let release!: (value: CandidateRunResult) => void, runningSignal: AbortSignal | undefined;
    execute = abort => { runningSignal = abort; return new Promise(resolve => { release = resolve; }); };
    const { proposalId } = (await startCaptured()).data;
    await call('run_proposal_candidate', { proposalId, requestId: randomUUID() });
    await clients[0].transport.terminateSession(); await tick();
    assert.equal(runningSignal!.aborted, true); assert.deepEqual(history.list(`mcp-${proposalId}`), []);
    release(candidateResult()); await tick(); assert.deepEqual(history.list(`mcp-${proposalId}`), []);
  });

  test('cancel targets only the requested run and never decides the source review', async () => {
    const { proposalId } = (await startCaptured()).data;
    const first = (await call('run_proposal_candidate', { proposalId, requestId: randomUUID() })).data.candidate;
    await tick(); let release!: (value: CandidateRunResult) => void, runningSignal: AbortSignal | undefined;
    execute = abort => { runningSignal = abort; return new Promise(resolve => { release = resolve; }); };
    const second = (await call('run_proposal_candidate', { proposalId, requestId: randomUUID() })).data.candidate;
    assert.notEqual(second.runId, first.runId);
    await call('cancel_proposal_candidate', { proposalId, runId: first.runId }); assert.equal(runningSignal!.aborted, false);
    await call('cancel_proposal_candidate', { proposalId, runId: second.runId }); assert.equal(runningSignal!.aborted, true);
    release(candidateResult()); await tick();
    assert.equal((await call('inspect_proposal_candidate', { proposalId, runId: second.runId })).data.candidate.summary.runtime, 'cancelled');
    assert.equal((await call('get_proposal', { proposalId })).data.status, 'review');
  });

  test('static-only proposals and malformed execution/inspection requests cannot run Python', async () => {
    const { proposalId } = (await start()).data;
    assert.equal((await call('run_proposal_candidate', { proposalId, requestId: randomUUID() })).data.error.code, 'CANDIDATE_UNAVAILABLE');
    assert.ok((await call('run_proposal_candidate', { proposalId, requestId: 'not-a-uuid' })).error);
    assert.ok((await call('inspect_proposal_candidate', { proposalId, runId: randomUUID(), query: { captureId: 'x', sourceId: 'y', path: [-1] } })).error);
    assert.equal(executions, 0);
  });

  test('request-ID bounds refuse new execution while preserving idempotent retries', async () => {
    const { proposalId } = (await startCaptured()).data, requests = Array.from({ length: 128 }, () => randomUUID());
    let release!: (value: CandidateRunResult) => void;
    execute = () => new Promise(resolve => { release = resolve; });
    for (const requestId of requests) assert.ok(!(await call('run_proposal_candidate', { proposalId, requestId })).error);
    const limit = await call('run_proposal_candidate', { proposalId, requestId: randomUUID() });
    assert.equal(limit.data.error.code, 'REQUEST_LIMIT');
    assert.ok(!(await call('run_proposal_candidate', { proposalId, requestId: requests[0] })).error);
    assert.equal(executions, 1);
    release(candidateResult()); await tick();
  });

  test('proposal expiry forgets owned evidence without allowing a retry to execute', async () => {
    const { proposalId } = (await startCaptured()).data, requestId = randomUUID();
    await call('run_proposal_candidate', { proposalId, requestId });
    finish!({ status: 'cancelled', detail: 'Discarded.' }); await tick();
    assert.equal(history.list(`mcp-${proposalId}`).length, 1);
    const before = Date.now;
    try {
      const expired = before() + 31 * 60_000; Date.now = () => expired;
      assert.equal((await call('run_proposal_candidate', { proposalId, requestId })).data.error.code, 'UNKNOWN_PROPOSAL');
      assert.deepEqual(history.list(`mcp-${proposalId}`), []); assert.equal(executions, 1);
    } finally { Date.now = before; }
  });

  test('proposal retention evicts its candidate payload as well as its summary', async () => {
    const first = (await startCaptured()).data.proposalId;
    await call('run_proposal_candidate', { proposalId: first, requestId: randomUUID() });
    finish!({ status: 'cancelled', detail: 'Discarded.' }); await tick();
    for (let n = 0; n < 32; n++) {
      // Exercise the retention bound directly; socket routing is covered above.
      proposals.start(owners[0], graph, ref(), edit as any); await tick();
      finish!({ status: 'cancelled', detail: 'Discarded.' }); await tick();
    }
    assert.deepEqual(history.list(`mcp-${first}`), []);
    assert.equal((await call('get_proposal', { proposalId: first })).data.error.code, 'UNKNOWN_PROPOSAL');
  });
  for (const fixture of ['captured-unary-insert', 'captured-unary-import', 'captured-unary-delete', 'captured-reconnection-edit', 'captured-source-edit', 'captured-activation-edit', 'captured-module-activation-edit',
    'captured-activation-import', 'captured-module-activation-import', 'captured-constructor-edit', 'captured-constructor-keyword', 'captured-custom-constructor', 'captured-configuration-edit', 'captured-configuration-module', 'captured-activation-options', 'captured-module-activation-options']) {
    test(`round-trips the real ${fixture} intent through standard HTTP without applying`, async () => {
      const { intent } = JSON.parse(readFileSync(join(__dirname, `../../../../tests/fixtures/contracts/${fixture}.json`), 'utf8'));
      const request = { ...intent, nodeId: 'fc' };
      const started = await start(request);
      assert.ok(!started.error);
      assert.deepEqual(received[0].edit, request);
      const state = (await call('get_proposal', { proposalId: started.data.proposalId })).data;
      assert.equal(state.status, 'review');
      finish!({ status: 'cancelled', detail: 'Discarded in VS Code.' }); await tick();
      assert.equal((await call('get_proposal', { proposalId: started.data.proposalId })).data.status, 'cancelled');
    });
  }
});
