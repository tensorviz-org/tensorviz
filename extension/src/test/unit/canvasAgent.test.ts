import * as assert from 'node:assert/strict';
import { CanvasAgent, type AgentRunEvent, type AgentRunInput } from '../../canvasAgent';
import type { CanvasAgentAction, CanvasAgentState, GraphOpeningState, LocalBrowserAgentRequest } from '@tensorviz/contracts/protocol';

const target = { workspaceFolder: '/project', file: 'model.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
const request = (action: CanvasAgentAction, viewId = 'view'): LocalBrowserAgentRequest => ({ type: 'agentRequest', protocolVersion: 1, workspaceSessionId: 'workspace', messageId: 'message', viewId, ...action });
const send = (id = 'one'): CanvasAgentAction => ({ action: 'send', requestId: id, target, nodeIds: ['activation'], provider: 'codex', prompt: 'Replace ReLU with GELU' });
const tick = () => new Promise<void>(resolve => setImmediate(resolve));
const opening = (): GraphOpeningState => ({ id: 'opening', target, phase: 'needs_setup', hasCapture: false, detail: 'Missing input.' });

suite('Canvas agent session', () => {
  let agent: CanvasAgent, states: Array<{ viewId: string; state: CanvasAgentState }>, runs: AgentRunInput[], cleared: number;
  let emit: (event: AgentRunEvent) => void, finish: () => void, signal: AbortSignal;
  const latest = () => states.at(-1)!.state;
  setup(() => {
    states = []; runs = []; cleared = 0;
    agent = new CanvasAgent({ prepare: async () => ({ cwd: '/project', context: 'pinned context' }),
      run: async (input, callback, abort) => { runs.push(input); emit = callback; signal = abort; await new Promise<void>(resolve => { finish = resolve; }); },
      publish: (viewId, state) => states.push({ viewId, state }), clear: () => { cleared++; } });
  });
  teardown(() => { agent.dispose(); finish?.(); });

  test('minimizing/disconnecting does not cancel; reconnect restores completed history', async () => {
    agent.handle(request(send())); await tick();
    agent.disconnect('view');
    emit({ kind: 'text', text: 'Review the proposed change in VS Code.' }); finish(); await tick();
    assert.equal(signal.aborted, false);
    agent.handle(request({ action: 'sync' }, 'reconnected'));
    assert.equal(latest().status, 'idle'); assert.equal(latest().messages[1].text, 'Review the proposed change in VS Code.');
    assert.deepEqual(latest().context?.nodeIds, ['activation']);
    assert.equal(states.at(-1)!.viewId, 'reconnected');
  });

  test('a duplicate send cannot start twice and Stop rejects late output', async () => {
    agent.handle(request(send())); agent.handle(request(send())); await tick();
    assert.equal(runs.length, 1);
    agent.handle(request({ action: 'stop', requestId: 'old-request' })); assert.equal(signal.aborted, false);
    agent.handle(request({ action: 'stop', requestId: 'one' })); assert.equal(signal.aborted, true); assert.equal(cleared, 1);
    emit({ kind: 'text', text: 'late answer' }); finish(); await tick();
    assert.equal(latest().status, 'stopped'); assert.equal(latest().messages[1].text, '');
    agent.handle(request(send())); await tick(); assert.equal(runs.length, 1);
  });

  test('a new model cannot silently inherit the prior conversation', async () => {
    agent.handle(request(send())); await tick(); finish(); await tick();
    agent.handle(request({ ...send('two'), action: 'send', target: { ...target, cls: 'Other' }, requestId: 'two', nodeIds: [], provider: 'codex', prompt: 'Explain' })); await tick();
    assert.equal(runs.length, 1); assert.match(latest().activity, /another model/);
    agent.handle(request({ action: 'clear' }));
    assert.equal(latest().messages.length, 0); assert.equal(cleared, 1);
  });

  test('streaming coalesces updates without losing the final reply', async () => {
    agent.handle(request(send())); await tick();
    const before = states.length;
    for (let i = 0; i < 100; i++) emit({ kind: 'text', text: 'x' });
    assert.equal(states.length, before, 'token bursts must not publish 100 whole histories to the graph');
    finish(); await tick();
    assert.equal(latest().messages[1].text, 'x'.repeat(100));
    assert.equal(latest().status, 'idle');
  });

  test('a preparation refusal makes no provider request', async () => {
    agent.dispose();
    agent = new CanvasAgent({ prepare: async () => { throw new Error('Trust the workspace first.'); },
      run: async input => { runs.push(input); }, publish: (viewId, state) => states.push({ viewId, state }), clear() {} });
    agent.handle(request(send())); await tick();
    assert.equal(runs.length, 0); assert.equal(latest().status, 'error'); assert.match(latest().activity, /Trust/);
    assert.equal(latest().messages.length, 0, 'refused requests must leave the draft available in the view');
  });

  test('Use Agent auto-selects the sole ready provider, sends the opening task once and verifies capture', async () => {
    agent.dispose();
    let captured = false;
    agent = new CanvasAgent({ prepare: async (_target, nodes, draft, _signal, intent) => {
      assert.equal(intent, 'open_graph'); assert.deepEqual(nodes, []); assert.equal(draft, undefined);
      return { cwd: target.workspaceFolder, context: 'pending setup' };
    }, inspect: async provider => ({ status: provider === 'claude' ? 'ready' : 'notInstalled', detail: 'Install Codex.', models: [] }),
    run: async input => { runs.push(input); await new Promise<void>(resolve => { finish = resolve; }); },
    openingComplete: () => captured, publish: (viewId, state) => states.push({ viewId, state }), clear() {} });
    agent.handle(request({ action: 'sync' }));
    agent.openGraph(opening()); agent.openGraph(opening()); await tick();
    assert.equal(runs.length, 1); assert.equal(runs[0].provider, 'claude'); assert.equal(runs[0].intent, 'open_graph');
    assert.match(runs[0].messages[0].text, /Net in model.py/); assert.match(runs[0].messages[0].text, /Save a complete setup; saving runs it once to capture the graph, so do not start other runs/);
    finish(); await tick();
    assert.equal(latest().status, 'idle', 'an agent question is a conversation, not a provider error');
    assert.match(latest().activity, /Setup needs your input/);
    assert.equal(latest().agents?.claude.status, 'ready', 'a setup blocker is not a broken provider connection');
    agent.handle(request({ action: 'send', requestId: 'answer-setup', prompt: 'Use batch size 2', provider: 'claude', target, nodeIds: [] }));
    await tick(); assert.equal(runs[1].intent, 'open_graph', 'a reply to a setup question keeps setup context without a captured graph');
    captured = true; finish(); await tick();
    assert.equal(latest().status, 'idle');
  });

  test('Use Agent keeps saved provider/model/effort and Stop cancels the automatically started run', async () => {
    agent.handle(request({ action: 'configure', provider: 'claude', settings: { model: 'chosen-model', effort: 'high' }, requestId: 'settings' }));
    agent.openGraph(opening()); await tick();
    assert.equal(runs[0].provider, 'claude'); assert.deepEqual(runs[0].settings, { model: 'chosen-model', effort: 'high' });
    agent.handle(request({ action: 'stop', requestId: latest().requestId! }));
    assert.ok(signal.aborted); finish(); await tick(); assert.equal(latest().status, 'stopped');
  });

  test('Use Agent starts a fresh conversation when the selected model changes', async () => {
    agent.handle(request(send())); await tick(); finish(); await tick();
    agent.openGraph({ ...opening(), target: { ...target, cls: 'Other' } }); await tick();
    assert.equal(runs.length, 2); assert.equal(runs[1].messages.length, 1);
    assert.match(runs[1].messages[0].text, /Other/); assert.equal(cleared, 1);
  });

  test('source changes during readiness checks cannot launch a stale automatic opening', async () => {
    agent.dispose(); let preparations = 0;
    agent = new CanvasAgent({ prepare: async () => {
      if (++preparations > 1) throw new Error('Source changed.');
      return { cwd: '/project', context: '' };
    }, inspect: async () => ({ status: 'ready', detail: '', models: [] }),
    run: async input => { runs.push(input); }, publish: (viewId, state) => states.push({ viewId, state }), clear() {} });
    agent.handle(request({ action: 'sync' })); agent.openGraph(opening()); await tick();
    assert.equal(runs.length, 0); assert.equal(latest().status, 'error'); assert.match(latest().activity, /Source changed/);
  });

  test('Stop also cancels a capture started by the automatic setup task', async () => {
    agent.dispose(); const cancelled: unknown[] = [];
    agent = new CanvasAgent({ prepare: async () => ({ cwd: '/project', context: '' }),
      run: async () => { await new Promise<void>(resolve => { finish = resolve; }); },
      cancelOpening: target => cancelled.push(target), publish: (viewId, state) => states.push({ viewId, state }), clear() {} });
    agent.handle(request({ action: 'sync' })); agent.openGraph(opening()); await tick();
    agent.handle(request({ action: 'stop', requestId: latest().requestId! }));
    assert.deepEqual(cancelled, [target]); finish(); await tick(); assert.equal(latest().status, 'stopped');
  });
});
