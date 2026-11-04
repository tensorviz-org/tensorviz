import * as assert from 'node:assert/strict';
import { CanvasAgent, type AgentRunInput, type AgentRunEvent } from '../../canvasAgent';
import type { CanvasAgentAction, CanvasAgentPermissionRequest, CanvasAgentPermissionPresentation, CanvasAgentState, GraphOpeningState, LocalBrowserAgentRequest } from '@tensorviz/contracts/protocol';

const target = { workspaceFolder: '/project', file: 'model.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
const request = (action: CanvasAgentAction): LocalBrowserAgentRequest => ({ type: 'agentRequest', protocolVersion: 1, workspaceSessionId: 'workspace', messageId: 'message', viewId: 'view', ...action });
const tick = () => new Promise<void>(resolve => setImmediate(resolve));
const environment = (ready = false): GraphOpeningState => ({ id: 'opening', target, phase: 'needs_setup', hasCapture: false, detail: ready ? 'Set model inputs.' : 'Choose Python.',
  environment: { reason: ready ? 'ready' : 'selection_required', detail: '', candidates: [] } });
const permission: CanvasAgentPermissionRequest = { operation: 'python.environment.create', scope: 'isolated_environment', effects: ['create'], facts: { environment: 'net', model: 'model.py' } };
const presentation: CanvasAgentPermissionPresentation = { title: 'Prepare Python for Net?', summary: 'Net needs its model libraries to record tensor shapes.', allowLabel: 'Prepare Python', denyLabel: 'Skip setup' };

suite('Canvas agent host-driven setup continuation', () => {
  let agent: CanvasAgent, opening: GraphOpeningState | undefined, states: CanvasAgentState[], runs: AgentRunInput[], finishes: Array<() => void>;
  let emitters: Array<(event: AgentRunEvent) => void>;
  let preparations: number, cancelled: number, signals: AbortSignal[], waits: Array<(value: GraphOpeningState | undefined) => void>;
  const latest = () => states.at(-1)!;
  const update = (value: GraphOpeningState | undefined) => { opening = value; waits.splice(0).forEach(resolve => resolve(value)); };
  const start = async () => { agent.handle(request({ action: 'sync' })); agent.openGraph(opening!); await tick(); };
  const finish = async () => { finishes.shift()!(); await tick(); };
  setup(() => {
    opening = environment(); states = []; runs = []; finishes = []; waits = []; preparations = 0; cancelled = 0; signals = []; emitters = [];
    agent = new CanvasAgent({
      preferences: { provider: 'claude', selections: { claude: { model: 'saved-model', effort: 'high' } } },
      prepare: async () => ({ cwd: '/project', context: `fresh host context ${++preparations}` }),
      openingProgress: async (_target, signal, waitMs) => {
        signal.throwIfAborted();
        if (!waitMs || !opening || !['resolving', 'saving', 'capturing'].includes(opening.phase)) return opening;
        return new Promise<GraphOpeningState | undefined>((resolve, reject) => {
          const abort = () => reject(signal.reason);
          signal.addEventListener('abort', abort, { once: true });
          waits.push(value => { signal.removeEventListener('abort', abort); resolve(value); });
        });
      },
      run: async (input, emit, signal) => { runs.push(input); emitters.push(emit); signals.push(signal); await new Promise<void>(resolve => finishes.push(resolve)); },
      cancelOpening: () => { cancelled++; }, publish: (_view, state) => states.push(state), clear() {},
    });
  });
  teardown(async () => { agent.dispose(); finishes.splice(0).forEach(resolve => resolve()); waits.splice(0).forEach(resolve => resolve(undefined)); await tick(); });

  test('Python completion continues through inputs and capture in the same turn and runtime conversation', async () => {
    await start(); const requestId = latest().requestId, sessionId = latest().sessionId;
    opening = { ...opening!, phase: 'resolving' }; await finish();
    assert.equal(latest().status, 'running'); assert.equal(runs.length, 1);
    update(environment(true)); await tick();
    assert.equal(runs.length, 2); assert.equal(latest().requestId, requestId); assert.equal(latest().sessionId, sessionId);
    assert.equal(runs[1].conversationId, runs[0].conversationId); assert.equal(signals[1], signals[0]);
    assert.equal(runs[1].provider, 'claude'); assert.deepEqual(runs[1].settings, { model: 'saved-model', effort: 'high' });
    assert.notEqual(runs[1].context, runs[0].context); assert.equal(preparations, 3);
    assert.match(runs[1].messages.at(-1)!.text, /continu|HOST|Host/);
    assert.equal(latest().messages.filter(turn => turn.role === 'user').length, 1, 'host continuation must not appear as another user request');
    opening = { ...environment(true), phase: 'capturing' }; await finish();
    assert.equal(latest().status, 'running');
    update({ ...environment(true), phase: 'ready', hasCapture: true }); await tick();
    assert.equal(latest().status, 'idle'); assert.equal(latest().activity, ''); assert.equal(runs.length, 2, 'capture completion must not trigger another model invocation');
  });

  test('finished provider callbacks cannot overwrite host waiting or the next continuation', async () => {
    await start();
    opening = { ...opening!, phase: 'resolving' }; await finish();
    const waiting = structuredClone(latest());
    emitters[0]({ kind: 'activity', text: 'Stale first-run activity' });
    assert.deepEqual(latest(), waiting);
    update(environment(true)); await tick();
    assert.equal(runs.length, 2);
    emitters[1]({ kind: 'activity', text: 'Preparing current inputs' });
    const currentState = structuredClone(latest()), count = states.length;
    emitters[0]({ kind: 'activity', text: 'Stale first-run activity' });
    emitters[0]({ kind: 'text', text: 'Stale first-run answer' });
    emitters[0]({ kind: 'opened_graph', text: '', target: { ...target, cls: 'OtherModel' } });
    assert.equal(states.length, count);
    assert.deepEqual(latest(), currentState);
    opening = { ...environment(true), phase: 'ready', hasCapture: true }; await finish();
    assert.equal(latest().status, 'idle');
  });

  test('late operation plans wake host polling so the same provider can present each permission', async () => {
    await start(); const approve = agent.permissionRequester(target), sessionId = latest().sessionId, requestId = latest().requestId;
    opening = { ...opening!, phase: 'resolving' }; await finish();
    assert.equal(waits.length, 1, 'the provider has ended while host setup is still running');
    const creation = approve(permission, signals[0]); await tick();
    assert.equal(runs.length, 2, 'new facts wake the host wait without waiting for its timeout');
    assert.equal(latest().status, 'running'); assert.equal(latest().permissions!.at(-1)!.presentation, undefined);
    assert.equal(runs[1].conversationId, sessionId); assert.equal(latest().requestId, requestId); assert.equal(signals[1], signals[0]);
    assert.match(runs[1].messages.at(-1)!.text, /python.environment.create/);
    let id = latest().permissions!.at(-1)!.id; agent.presentPermission(id, presentation);
    agent.handle(request({ action: 'permission', permissionId: id, decision: 'allow' })); assert.equal(await creation, true);
    await finish();
    const installation = approve({ ...permission, operation: 'python.packages.install', effects: ['install'], facts: { packages: [{ name: 'torch', version: '2.9.0' }] } }, signals[0]);
    await tick(); assert.equal(runs.length, 3);
    assert.match(runs[2].messages.at(-1)!.text, /2.9.0/);
    id = latest().permissions!.at(-1)!.id; agent.presentPermission(id, { ...presentation, title: 'Install Net libraries?', allowLabel: 'Install libraries' });
    agent.handle(request({ action: 'permission', permissionId: id, decision: 'allow' })); assert.equal(await installation, true);
    update(environment(true)); await finish(); assert.equal(runs.length, 4, 'verified Python readiness resumes model inputs after presentation segments');
    assert.equal(latest().messages.filter(message => message.role === 'user').length, 1, 'all host resumptions belong to the original visible task');
    opening = { ...environment(true), phase: 'ready', hasCapture: true }; await finish();
    assert.ok(latest().permissions!.every(receipt => receipt.status === 'approved' && receipt.presentation)); assert.equal(latest().status, 'idle');
  });

  test('a setup started from an ordinary chat message owns a late first permission and continues to inputs', async () => {
    agent.handle(request({ action: 'send', requestId: 'chat-turn', provider: 'claude', target, nodeIds: [], prompt: 'Set up this model.' })); await tick();
    const approve = agent.permissionRequester(target), sessionId = latest().sessionId;
    opening = { ...opening!, phase: 'resolving' }; await finish();
    assert.equal(latest().status, 'running', 'a captured asynchronous operation keeps its originating turn active before its first permission');
    const creation = approve(permission, signals[0]); await tick(); assert.equal(runs.length, 2);
    assert.equal(latest().requestId, 'chat-turn'); assert.equal(runs[1].conversationId, sessionId); assert.equal(signals[1], signals[0]);
    const id = latest().permissions!.at(-1)!.id; agent.presentPermission(id, presentation);
    agent.handle(request({ action: 'permission', permissionId: id, decision: 'allow' })); assert.equal(await creation, true);
    update(environment(true)); await finish(); assert.equal(runs.length, 3);
    assert.match(runs[2].messages.at(-1)!.text, /workflow.nextStep/);
    opening = { ...environment(true), phase: 'ready', hasCapture: true }; await finish(); assert.equal(latest().status, 'idle');
  });

  test('a declined setup operation ends truthfully without retrying or falling back to native review', async () => {
    await start(); const approve = agent.permissionRequester(target);
    opening = { ...opening!, phase: 'resolving' }; await finish();
    const decision = approve(permission, signals[0]);
    agent.presentPermission(latest().permissions!.at(-1)!.id, presentation);
    agent.handle(request({ action: 'permission', permissionId: latest().permissions!.at(-1)!.id, decision: 'deny' })); assert.equal(await decision, false);
    update(environment()); await tick();
    assert.equal(runs.length, 1); assert.equal(latest().status, 'idle');
    assert.equal(await approve(permission, signals[0]), false, 'finished turn cannot inherit a manual approval fallback');
  });

  test('an unchanged input stage leaves the agent question available for the user', async () => {
    opening = environment(true); await start(); await finish();
    assert.equal(runs.length, 1); assert.equal(latest().status, 'idle'); assert.match(latest().activity, /needs your input/);
  });

  for (const phase of ['failed', 'cancelled', 'awaiting_execution', 'source_only'] as const) {
    test(`${phase} is a terminal setup blocker without automatic retries`, async () => {
      await start(); opening = { ...environment(true), phase, detail: 'Host requires user action.' }; await finish();
      assert.equal(runs.length, 1); assert.equal(latest().status, phase === 'failed' ? 'error' : 'idle');
      if (phase === 'awaiting_execution') assert.match(latest().activity, /execution approval/);
    });
  }

  test('Stop while host setup is running aborts polling and cannot start a continuation', async () => {
    await start(); opening = { ...opening!, phase: 'resolving' }; await finish();
    agent.handle(request({ action: 'stop', requestId: latest().requestId! })); await tick();
    assert.equal(cancelled, 1); assert.equal(signals[0].aborted, true); assert.equal(latest().status, 'stopped');
    update(environment(true)); await tick(); assert.equal(runs.length, 1);
  });

  for (const change of ['opening', 'source', 'missing'] as const) {
    test(`a changed ${change} invalidates the bound setup before continuing`, async () => {
      await start(); opening = { ...opening!, phase: 'resolving' }; await finish();
      update(change === 'missing' ? undefined : { ...environment(true), ...(change === 'opening' ? { id: 'replacement' } : { target: { ...target, sourceDigest: 'b'.repeat(64) } }) });
      await tick(); assert.equal(runs.length, 1); assert.equal(latest().status, 'idle'); assert.match(latest().activity, /setup or source changed/);
    });
  }
});
