import * as assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { CanvasAgentRecording } from '../../canvasAgentRecording';
import { CanvasAgentRuntime } from '../../canvasAgentRuntime';
import { LocalBrowserGraph } from '../../localBrowserGraph';
import type { AgentRunEvent, AgentRunInput } from '../../canvasAgent';

suite('local agent test recording', () => {
  let root: string;
  setup(async () => { root = await mkdtemp(path.join(tmpdir(), 'tensorviz-recording-test-')); });
  teardown(async () => { await rm(root, { recursive: true, force: true }); });
  const entries = async (recording: CanvasAgentRecording) => (await readFile(recording.file, 'utf8')).trim().split('\n').map(line => JSON.parse(line));

  test('preserves exact text in ordered private local records and closes explicitly', async () => {
    const directory = path.join(root, 'session');
    const recording = new CanvasAgentRecording(directory, { version: 'fixture' }, () => {});
    const text = 'α\n\nnext paragraph';
    recording.record({ kind: 'agent_event', turnId: 'turn', event: { kind: 'text', text } });
    recording.stop('requested'); await recording.closed;
    recording.record({ kind: 'after_stop' });
    const rows = await entries(recording);
    assert.deepEqual(rows.map(row => row.kind), ['recording_start', 'agent_event', 'recording_end']);
    assert.equal(rows[1].event.text, text);
    assert.deepEqual(rows.map(row => row.sequence), [1, 2, 3]);
    assert.ok(rows.every((row, index) => Number.isFinite(row.elapsedMs) && (!index || row.elapsedMs >= rows[index - 1].elapsedMs)));
    assert.equal(rows[2].reason, 'requested');
    if (process.platform !== 'win32') {
      assert.equal((await stat(recording.file)).mode & 0o777, 0o600);
      assert.equal((await stat(directory)).mode & 0o777, 0o700);
    }
  });

  test('oversized events retain a digest and explicit omission instead of a partial answer', async () => {
    const recording = new CanvasAgentRecording(root, {}, () => {}, { eventBytes: 512, totalBytes: 8192, queueBytes: 8192 });
    recording.record({ kind: 'tool_result', turnId: 'turn', callId: 'call', result: 'x'.repeat(1000) });
    recording.stop(); await recording.closed;
    const row = (await entries(recording))[1];
    assert.equal(row.kind, 'omitted_event'); assert.equal(row.originalKind, 'tool_result');
    assert.equal(row.turnId, 'turn'); assert.equal(row.callId, 'call');
    assert.equal(row.reason, 'event_byte_limit'); assert.match(row.sha256, /^[a-f0-9]{64}$/);
    assert.ok(row.bytes > 1000); assert.equal(row.result, undefined);
  });

  for (const budget of ['total', 'queue']) test(`${budget} limit ends recording visibly without affecting the caller`, async () => {
    const notices: string[] = [];
    const recording = new CanvasAgentRecording(root, {}, text => notices.push(text),
      { eventBytes: 2048, totalBytes: budget === 'total' ? 4096 : 65536, queueBytes: budget === 'queue' ? 2048 : 65536 });
    for (let i = 0; i < 20; i++) recording.record({ kind: 'test', value: 'x'.repeat(1000) });
    await recording.closed;
    const rows = await entries(recording);
    assert.equal(rows.at(-1).reason, budget === 'total' ? 'total_byte_limit' : 'write_queue_limit');
    assert.deepEqual(rows.map(row => row.sequence), rows.map((_, index) => index + 1));
    assert.equal(notices.length, 1);
    if (budget === 'total') assert.ok((await stat(recording.file)).size <= 4096);
  });

  test('serialization and write failures do not escape into the model workflow', async () => {
    const recording = new CanvasAgentRecording(root, {}, () => { throw new Error('broken notification'); });
    const circular: Record<string, unknown> = { kind: 'invalid' }; circular.self = circular;
    assert.doesNotThrow(() => recording.record(circular)); await recording.closed;
    assert.equal((await entries(recording)).at(-1).reason, 'serialization_failure');
    const notices: string[] = [];
    const failed = new CanvasAgentRecording(root, {}, text => notices.push(text));
    (failed as any).stream.destroy(new Error('fixture disk failure')); await failed.closed;
    assert.match(notices[0], /recording failed.*incomplete/);
    assert.doesNotThrow(() => failed.record({ kind: 'ignored' }));
  });

  test('STOP closes an idle recorder and a stopped directory cannot restart accidentally', async () => {
    const recording = new CanvasAgentRecording(root, {}, () => {});
    await writeFile(path.join(root, 'STOP'), 'requested\n');
    await recording.closed;
    assert.equal((await entries(recording)).at(-1).reason, 'requested');
    assert.throws(() => new CanvasAgentRecording(root, {}, () => {}), /session was stopped/);
    assert.throws(() => new CanvasAgentRecording('relative', {}, () => {}), /absolute/);
  });

  for (const pipeline of ['native', 'acp'] as const) for (const provider of ['codex', 'claude'] as const) {
    test(`${provider}/${pipeline} records paired real MCP outcomes without changing response or tool results`, async () => {
      const observed: any[] = [], visible: AgentRunEvent[] = [];
      const runtime = new CanvasAgentRuntime({ pipeline: () => pipeline, executable: async () => 'fixture', release() {},
        observe: event => { observed.push(structuredClone(event)); if (event.kind === 'tool_result') (event as any).result.content = []; },
        tools: () => new Map([
          ['read_fixture', { config: { inputSchema: {}, annotations: { readOnlyHint: true } }, handler: async () => ({ content: [{ type: 'text' as const, text: 'actual evidence' }] }) }],
          ['refused_fixture', { config: { inputSchema: {} }, handler: async () => ({ isError: true, content: [{ type: 'text' as const, text: 'refused' }] }) }],
          ['failed_fixture', { config: { inputSchema: {} }, handler: async () => { throw new Error('fixture tool failure'); } }],
        ]) });
      const fixture = runtime as any; let url = '';
      const answer = async (emit: (event: AgentRunEvent) => void) => {
        const client = new Client({ name: 'recording-test', version: '1' });
        try {
          await client.connect(new StreamableHTTPClientTransport(new URL(url), { requestInit: { headers: { Authorization: `Bearer ${fixture.token}` } } }));
          const result = await client.callTool({ name: 'read_fixture', arguments: {} });
          assert.deepEqual(result.content, [{ type: 'text', text: 'actual evidence' }]);
          await client.callTool({ name: 'refused_fixture', arguments: {} });
          await client.callTool({ name: 'failed_fixture', arguments: {} });
          emit({ kind: 'text', text: 'A\n\nB' });
        } finally { await client.close(); }
      };
      fixture.startCodex = fixture.startClaude = async (_executable: string, _input: unknown, endpoint: string) => {
        url = endpoint; return { process: { completed: new Promise(() => {}) }, sent: false };
      };
      fixture.turn = async (_session: unknown, _input: unknown, _text: string, emit: (event: AgentRunEvent) => void) => answer(emit);
      if (pipeline === 'acp') fixture.getAcp = () => fixture.acp ??= {
        controller: new AbortController(), sent: false, prepared: false,
        ready: fixture.connect(provider).then((endpoint: { url: string }) => {
          url = endpoint.url;
          return { prompt: async (_text: string, emit: (event: AgentRunEvent) => void) => answer(emit), dispose: async () => {} };
        }),
      };
      try {
        await runtime.run({ provider, cwd: root, context: 'Context', conversationId: 'conversation',
          messages: [{ id: 'request', role: 'user', text: 'Inspect this' }], settings: { effort: 'medium' } },
        event => visible.push(event), new AbortController().signal);
        assert.deepEqual(visible.filter(event => event.kind === 'text'), [{ kind: 'text', text: 'A\n\nB' }]);
        const waiting = `Waiting for ${provider === 'claude' ? 'Claude' : 'Codex'}…`;
        assert.equal(visible.filter(event => event.kind === 'activity' && event.text === waiting).length, 3,
          'success, refusal and failure all end their tool activity');
        const start = observed.find(event => event.kind === 'turn_start');
        assert.equal(start.pipeline, pipeline); assert.equal(start.input.provider, provider);
        const calls = observed.filter(event => event.kind === 'tool_start');
        assert.equal(calls.length, 3);
        for (const call of calls) {
          assert.equal(call.turnId, start.turnId);
          const ends = observed.filter(event => ['tool_result', 'tool_error'].includes(event.kind) && event.callId === call.callId);
          assert.equal(ends.length, 1); assert.equal(ends[0].turnId, start.turnId);
        }
        assert.equal(observed.find(event => event.kind === 'tool_result' && event.name === 'refused_fixture').result.isError, true);
        assert.equal(observed.find(event => event.kind === 'tool_error').error, 'fixture tool failure');
        assert.equal(observed.at(-1).outcome, 'completed');
        assert.ok(!JSON.stringify(observed).includes(fixture.token), 'private transport credential must not enter recording');
      } finally { runtime.clear(); }
    });
  }

  test('observer failures preserve provider errors and cancellation stays distinguishable', async () => {
    const observed: any[] = [], controller = new AbortController();
    const runtime = new CanvasAgentRuntime({ tools: () => new Map(), release() {}, executable: async () => '',
      observe: event => { observed.push(event); throw new Error('observer failure'); } });
    const input: AgentRunInput = { provider: 'codex', cwd: root, context: '', messages: [] };
    const failure = new Error('original provider error');
    (runtime as any).runTurn = async () => { throw failure; };
    await assert.rejects(runtime.run(input, () => {}, controller.signal), error => error === failure);
    assert.equal(observed.at(-1).outcome, 'error');
    controller.abort();
    await assert.rejects(runtime.run(input, () => {}, controller.signal), error => error === failure);
    assert.equal(observed.at(-1).outcome, 'cancelled');
  });

  test('a late tool result keeps its original turn identity after cancellation and a new turn', async () => {
    const observed: any[] = [], old = new AbortController();
    const runtime = new CanvasAgentRuntime({ tools: () => new Map(), release() {}, executable: async () => '', observe: event => observed.push(event) });
    const fixture = runtime as any;
    fixture.toolTurn = { traceId: 'old', signal: old.signal };
    let finish!: (value: unknown) => void;
    const pending = fixture.observedTool('read_fixture', {}, () => new Promise(resolve => { finish = resolve; }));
    old.abort(); fixture.toolTurn = { traceId: 'new', signal: new AbortController().signal };
    finish({ content: [] }); await pending;
    assert.deepEqual(observed.map(event => event.turnId), ['old', 'old']);
    assert.equal(observed[1].activeTurn, false); assert.equal(observed[1].cancelled, true);
  });

  test('host observation retains terminal displayed text and readiness without copying every partial answer', () => {
    const graph = Object.create(LocalBrowserGraph.prototype) as any, observed: any[] = [];
    graph.options = { observeAgent: (event: unknown) => observed.push(event) };
    const state = { sessionId: 'chat', revision: 1, provider: 'codex', status: 'running', requestId: 'request', activity: '',
      messages: [{ id: 'request', role: 'user', text: 'Question' }, { id: 'answer', role: 'assistant', text: 'Reply' }],
      agents: { codex: { status: 'checking', detail: 'Preparing the agent connection…', models: [{ id: 'large-catalog' }] } } };
    graph.publish({ type: 'agentState', viewId: 'view', state });
    assert.deepEqual(observed[0].state.messages, [{ id: 'request', role: 'user' }, { id: 'answer', role: 'assistant' }]);
    assert.equal(observed[0].state.agents.codex.status, 'checking');
    assert.equal(observed[0].state.agents.codex.models, undefined);
    graph.publish({ type: 'agentState', viewId: 'view', state: { ...state, status: 'idle' } });
    assert.equal(observed[1].state.messages[1].text, 'Reply');
    graph.options.observeAgent = () => { throw new Error('observer failed'); };
    assert.doesNotThrow(() => graph.publish({ type: 'agentState', viewId: 'view', state }));
  });
});
