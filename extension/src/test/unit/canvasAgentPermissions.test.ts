import * as assert from 'node:assert/strict';
import { CanvasAgent, type AgentRunEvent } from '../../canvasAgent';
import { CANVAS_AGENT_PERMISSION_LIMIT, isCanvasAgentPermission, isCanvasAgentPermissionMode, isCanvasAgentPermissionPresentation,
  isCanvasAgentPermissionRequest, isCanvasAgentState, isLocalBrowserMessage,
  type CanvasAgentAction, type CanvasAgentPermissionMode, type CanvasAgentPermissionPresentation,
  type CanvasAgentPermissionRequest, type CanvasAgentState, type LocalBrowserAgentRequest } from '@tensorviz/contracts/protocol';

const target = { workspaceFolder: '/project', file: 'model.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
const permission: CanvasAgentPermissionRequest = { operation: 'python.environment.create', scope: 'isolated_environment', effects: ['create'],
  facts: { project: '/project', model: 'model.py', environment: { name: 'net', existing: false } } };
const presentation: CanvasAgentPermissionPresentation = { title: 'Prepare Python for Net?', summary: 'Net needs its own Python libraries to record tensor shapes.',
  allowLabel: 'Prepare Python', denyLabel: 'Skip for now' };
const message = (action: CanvasAgentAction, viewId = 'view'): LocalBrowserAgentRequest => ({ type: 'agentRequest', protocolVersion: 1,
  workspaceSessionId: 'workspace', messageId: 'message', viewId, ...action });
const send = (requestId = 'turn'): CanvasAgentAction => ({ action: 'send', requestId, provider: 'codex', target, nodeIds: [], prompt: 'Set up this project.' });
const tick = () => new Promise<void>(resolve => setImmediate(resolve));

suite('Canvas agent dynamic permissions', () => {
  let agent: CanvasAgent, states: CanvasAgentState[], finishes: Array<() => void>, emit: (event: AgentRunEvent) => void;
  const latest = () => states.at(-1)!;
  const last = () => latest().permissions!.at(-1)!;
  const decide = (permissionId: string, decision: 'allow' | 'deny', viewId = 'view') => agent.handle(message({ action: 'permission', permissionId, decision }, viewId));
  const present = (id = last().id) => agent.presentPermission(id, presentation);
  const mode = (value: CanvasAgentPermissionMode, viewId = 'view') => agent.handle(message({ action: 'permission_mode', mode: value }, viewId));
  const start = async () => { agent.handle(message(send())); await tick(); };
  setup(() => {
    states = []; finishes = [];
    agent = new CanvasAgent({ prepare: async () => ({ cwd: '/project', context: 'Current model and source.' }),
      run: async (_input, callback) => { emit = callback; await new Promise<void>(resolve => finishes.push(resolve)); },
      publish: (_view, state) => { assert.ok(isCanvasAgentState(state), 'published permissions must satisfy the wire contract'); states.push(state); }, clear() {} });
  });
  teardown(async () => { agent.dispose(); finishes.forEach(finish => finish()); await tick(); });

  for (const value of ['ask', 'auto_setup', 'full_setup'] as const) {
    for (const effect of ['create', 'install', 'replace', 'execute_code'] as const) {
      for (const scope of ['isolated_environment', 'existing_environment', 'machine'] as const) {
        test(`${value}: ${effect} in ${scope} follows effects policy only after agent presentation`, async () => {
          await start(); mode(value);
          const result = agent.requestPermission(target, { ...permission, operation: 'another.tool.operation', effects: [effect], scope }, new AbortController().signal);
          let settled = false; void result.then(() => { settled = true; }); await tick();
          assert.equal(settled, false, 'even an automatic policy waits for a valid contextual presentation');
          assert.equal(last().status, 'pending'); assert.equal(last().presentation, undefined);
          present();
          const automatic = value === 'full_setup' && scope !== 'machine' || (value === 'auto_setup' && scope === 'isolated_environment' && ['create', 'install'].includes(effect));
          if (automatic) {
            assert.equal(await result, true); assert.equal(last().approval, 'policy'); assert.equal(last().mode, value);
          } else {
            assert.equal(last().status, 'pending'); decide(last().id, 'allow');
            assert.equal(await result, true); assert.equal(last().approval, 'user'); assert.equal(last().mode, undefined);
          }
          assert.deepEqual(last().presentation, presentation); assert.equal(last().operation, 'another.tool.operation');
        });
      }
    }
  }

  test('the default is ask, and manual work never inherits a later chat policy', async () => {
    agent.handle(message({ action: 'sync' })); assert.equal(latest().permissionMode, 'ask');
    const manual = agent.permissionRequester(target); mode('full_setup'); await start();
    assert.equal(await manual(permission, new AbortController().signal), undefined);
    assert.equal(latest().permissions, undefined);
  });

  test('Allow and policy escalation cannot authorize a request before its presentation', async () => {
    await start(); const result = agent.requestPermission(target, permission, new AbortController().signal);
    decide(last().id, 'allow'); mode('full_setup');
    assert.equal(last().status, 'pending'); assert.equal(last().presentation, undefined);
    present(); assert.equal(await result, true); assert.equal(last().mode, 'full_setup');
  });

  test('escalating policy applies to presented requests while downgrading affects the next decision', async () => {
    await start(); const first = agent.requestPermission(target, permission, new AbortController().signal); present();
    mode('auto_setup'); assert.equal(await first, true); assert.equal(last().mode, 'auto_setup');
    const build = agent.requestPermission(target, { ...permission, effects: ['install', 'execute_code'] }, new AbortController().signal); present();
    mode('ask'); mode('auto_setup'); assert.equal(last().status, 'pending');
    mode('full_setup'); assert.equal(await build, true); assert.equal(last().mode, 'full_setup');
    const next = agent.requestPermission(target, permission, new AbortController().signal); mode('ask'); present();
    assert.equal(last().status, 'pending'); decide(last().id, 'deny'); assert.equal(await next, false);
    assert.equal(latest().permissions![1].status, 'approved');
  });

  test('only connected views can change policy, including during an active turn', async () => {
    await start(); mode('full_setup', 'unknown'); mode('full_setup', 'unknown'); assert.equal(latest().permissionMode, 'ask');
    agent.handle(message({ action: 'sync' }, 'unknown')); mode('auto_setup', 'unknown'); assert.equal(latest().permissionMode, 'auto_setup');
    agent.disconnect('unknown'); mode('full_setup', 'unknown'); assert.equal(latest().permissionMode, 'auto_setup');
  });

  test('the host operation and tool reads are snapshots, unaffected by caller mutations', async () => {
    await start(); const original = structuredClone(permission);
    const result = agent.requestPermission(target, original, new AbortController().signal);
    original.effects.push('execute_code'); (original.facts.environment as Record<string, unknown>).name = 'changed';
    const read = agent.pendingPermission()!; read.effects.push('replace'); read.facts.project = '/another';
    assert.deepEqual(agent.pendingPermission()!.effects, ['create']); assert.deepEqual(last().facts, permission.facts);
    present(); decide(last().id, 'allow'); assert.equal(await result, true);
    assert.deepEqual(last().facts, permission.facts); assert.deepEqual(last().effects, ['create']);
  });

  test('agent presentation cannot mutate an operation, be rewritten, or be replayed onto another receipt', async () => {
    await start(); const first = agent.requestPermission(target, permission, new AbortController().signal), id = last().id;
    assert.throws(() => agent.presentPermission(id, { ...presentation, effects: ['create'] } as unknown as CanvasAgentPermissionPresentation));
    assert.equal(last().presentation, undefined);
    const generated = { ...presentation }; agent.presentPermission(id, generated); generated.title = 'Changed after submission';
    assert.deepEqual(last().presentation, presentation);
    assert.throws(() => agent.presentPermission(id, { ...presentation, title: 'Replace all Python?' }));
    assert.deepEqual(last().presentation, presentation); decide(id, 'allow'); assert.equal(await first, true);
    const second = agent.requestPermission(target, permission, new AbortController().signal);
    assert.throws(() => agent.presentPermission(id, presentation)); assert.equal(last().presentation, undefined);
    present(); decide(last().id, 'deny'); assert.equal(await second, false);
  });

  test('only a valid generated presentation makes a pending request reviewable', async () => {
    await start(); const result = agent.requestPermission(target, permission, new AbortController().signal);
    for (const invalid of [{ ...presentation, title: '' }, { ...presentation, summary: 'x'.repeat(513) },
      { ...presentation, allowLabel: 'Approve\nanything' }, { ...presentation, denyLabel: ' ' }, { ...presentation, title: ' padded ' }]) {
      assert.throws(() => agent.presentPermission(last().id, invalid)); assert.equal(last().presentation, undefined);
    }
    present(); assert.deepEqual(last().presentation, presentation);
    decide(last().id, 'deny'); assert.equal(await result, false);
  });

  test('each decision belongs to one current turn, exact receipt and synced view', async () => {
    await start(); const result = agent.requestPermission(target, permission, new AbortController().signal); present();
    const receipt = last(); assert.equal(receipt.requestId, 'turn');
    emit({ kind: 'activity', text: 'Waiting for Codex…' }); assert.equal(latest().activity, 'Waiting for your approval…');
    decide('other', 'allow'); decide(receipt.id, 'allow', 'unknown'); decide(receipt.id, 'allow', 'unknown'); assert.equal(last().status, 'pending');
    agent.handle(message({ action: 'sync' }, 'unknown')); decide(receipt.id, 'allow', 'unknown'); assert.equal(await result, true);
    decide(receipt.id, 'deny'); assert.equal(last().status, 'approved');
    assert.equal(agent.pendingPermission(), undefined); assert.throws(() => present(receipt.id));
  });

  test('active requests for another model or source never open a native fallback', async () => {
    await start(); mode('full_setup');
    for (const changed of [{ ...target, cls: 'Other' }, { ...target, file: 'other.py' }, { ...target, sourceDigest: 'b'.repeat(64) }]) {
      await assert.rejects(agent.requestPermission(changed, permission, new AbortController().signal), /model or source changed/);
    }
    assert.equal(latest().permissions, undefined);
  });

  test('only one operation can wait; cancelling needs no generated presentation', async () => {
    await start(); const result = agent.requestPermission(target, permission, new AbortController().signal);
    await assert.rejects(agent.requestPermission(target, permission, new AbortController().signal), /Another/);
    assert.equal(latest().permissions!.length, 1); decide(last().id, 'deny'); assert.equal(await result, false);
  });

  test('operation abort cancels either presentation phase without ending the conversation', async () => {
    await start();
    for (const described of [false, true]) {
      const controller = new AbortController(), result = agent.requestPermission(target, permission, controller.signal);
      if (described) present(); const id = last().id; controller.abort();
      assert.equal(await result, false); assert.equal(last().status, 'cancelled'); assert.equal(latest().status, 'running');
      decide(id, 'allow'); assert.throws(() => present(id)); assert.equal(last().status, 'cancelled');
    }
  });

  test('Stop cancels pending work and late agent presentation or UI clicks cannot restart it', async () => {
    await start(); const result = agent.requestPermission(target, permission, new AbortController().signal), id = last().id;
    agent.handle(message({ action: 'stop', requestId: 'turn' })); assert.equal(await result, false);
    decide(id, 'allow'); assert.throws(() => present(id)); assert.equal(agent.pendingPermission(), undefined);
    assert.equal(latest().status, 'stopped'); assert.equal(last().status, 'cancelled');
  });

  test('Clear resets Ask and stale operation, presentation and user decisions cannot inherit a new full-access turn', async () => {
    await start(); mode('full_setup'); const requester = agent.permissionRequester(target), sessionId = latest().sessionId;
    const result = requester(permission, new AbortController().signal), id = last().id;
    agent.handle(message({ action: 'clear' })); assert.equal(await result, false); assert.equal(latest().permissionMode, 'ask');
    agent.handle(message(send('second'))); await tick(); mode('full_setup'); assert.notEqual(latest().sessionId, sessionId);
    assert.equal(await requester(permission, new AbortController().signal), false); assert.throws(() => present(id));
    const current = agent.permissionRequester(target)(permission, new AbortController().signal);
    decide(id, 'allow'); assert.equal(last().requestId, 'second'); assert.equal(last().status, 'pending');
    present(); assert.equal(await current, true);
  });

  test('changing model or source cancels the request and invalidates agent presentation', async () => {
    await start(); const requester = agent.permissionRequester(target), result = requester(permission, new AbortController().signal), id = last().id;
    emit({ kind: 'opened_graph', text: '', target: { ...target, sourceDigest: 'b'.repeat(64) } });
    assert.equal(await result, false); assert.equal(last().status, 'cancelled'); assert.throws(() => present(id));
    await assert.rejects(requester(permission, new AbortController().signal), /model or source changed/);
  });

  test('an early provider reply resumes the same turn to present the actual pending operation', async () => {
    await start(); const result = agent.requestPermission(target, permission, new AbortController().signal), id = last().id, sessionId = latest().sessionId;
    finishes.shift()!(); await tick();
    assert.equal(latest().status, 'running'); assert.equal(latest().sessionId, sessionId); assert.equal(last().id, id);
    assert.equal(finishes.length, 1, 'the provider receives another segment for its missing presentation');
    present(); decide(id, 'allow'); assert.equal(await result, true);
    finishes.shift()!(); await tick(); assert.equal(latest().status, 'idle');
  });

  test('a provider that repeatedly ends without presentation is bounded and cancels the operation', async () => {
    await start(); const result = agent.requestPermission(target, permission, new AbortController().signal);
    for (let attempt = 0; attempt < 4 && finishes.length; attempt++) { finishes.shift()!(); await tick(); }
    assert.equal(await result, false); assert.equal(last().status, 'cancelled');
    assert.notEqual(latest().status, 'running'); assert.match(latest().activity, /permission|approval|explain|present/i);
  });

  test('presented decisions survive an early provider reply until the user decides', async () => {
    await start(); const result = agent.requestPermission(target, permission, new AbortController().signal); present();
    finishes.shift()!(); await tick(); assert.equal(latest().status, 'running'); assert.equal(finishes.length, 0);
    decide(last().id, 'deny'); assert.equal(await result, false); await tick(); assert.equal(latest().status, 'idle');
  });

  test('disposal cancels unpresented requests and an operation captured after turn completion cannot fall back to native review', async () => {
    await start(); const requester = agent.permissionRequester(target); finishes.shift()!(); await tick();
    assert.equal(await requester(permission, new AbortController().signal), false); assert.equal(latest().permissions, undefined);
    agent.handle(message(send('second'))); await tick(); const result = agent.requestPermission(target, permission, new AbortController().signal);
    agent.dispose(); assert.equal(await result, false);
  });

  test('later stages and later turns cannot inherit permission from an earlier agent operation', async () => {
    await start(); const requester = agent.permissionRequester(target);
    const first = requester(permission, new AbortController().signal); present(); decide(last().id, 'allow'); assert.equal(await first, true);
    finishes.shift()!(); await tick();
    assert.equal(await requester({ ...permission, operation: 'python.packages.install', effects: ['install'] }, new AbortController().signal), false);
    agent.handle(message(send('second'))); await tick();
    assert.equal(await requester(permission, new AbortController().signal), false); assert.equal(latest().permissions!.length, 1);
    const next = agent.permissionRequester(target)(permission, new AbortController().signal); present(); decide(last().id, 'deny'); assert.equal(await next, false);
  });

  test('Stop between operations denies later requests and a mismatched requester is rejected immediately', async () => {
    await start(); assert.throws(() => agent.permissionRequester({ ...target, cls: 'Other' }), /model or source changed/);
    const requester = agent.permissionRequester(target), first = requester(permission, new AbortController().signal);
    present(); decide(last().id, 'allow'); assert.equal(await first, true); agent.handle(message({ action: 'stop', requestId: 'turn' }));
    assert.equal(await requester({ ...permission, effects: ['execute_code'] }, new AbortController().signal), false);
  });

  test('manual and cancelled operations retain their original fallback contract', async () => {
    assert.equal(await agent.requestPermission(target, permission, new AbortController().signal), undefined);
    const cancelled = new AbortController(); cancelled.abort(); assert.equal(await agent.requestPermission(target, permission, cancelled.signal), false);
    assert.equal(states.length, 0);
  });

  test('permission history is bounded and every receipt retains its agent-authored presentation', async () => {
    await start();
    for (let i = 0; i < CANVAS_AGENT_PERMISSION_LIMIT + 5; i++) {
      const result = agent.requestPermission(target, permission, new AbortController().signal); present(); decide(last().id, 'allow'); assert.equal(await result, true);
    }
    assert.equal(latest().permissions!.length, CANVAS_AGENT_PERMISSION_LIMIT);
    assert.ok(latest().permissions!.every(item => item.status === 'approved' && item.presentation?.title === presentation.title));
  });

  test('malformed or oversized host facts never reach an approval control', async () => {
    await start();
    for (const invalid of [{ ...permission, effects: ['shell'] }, { ...permission, facts: { text: 'x'.repeat(12001) } }, { ...permission, title: 'Host-written title' }]) {
      await assert.rejects(agent.requestPermission(target, invalid as unknown as CanvasAgentPermissionRequest, new AbortController().signal), /incomplete|large/);
    }
    assert.equal(latest().permissions, undefined);
  });
});

suite('Canvas agent dynamic permission wire contract', () => {
  test('operations accept new identifiers and structured facts without new presentation templates', () => {
    assert.equal(isCanvasAgentPermissionRequest(permission), true);
    assert.equal(isCanvasAgentPermissionRequest({ ...permission, operation: 'custom.future.operation', effects: ['create', 'install'] }), true);
    const cyclic: Record<string, unknown> = {}; cyclic.self = cyclic;
    const tooDeep = Array.from({ length: 12 }).reduce<unknown>(value => ({ child: value }), 'leaf');
    for (const invalid of [{ ...permission, operation: '' }, { ...permission, operation: 'x'.repeat(129) },
      { ...permission, scope: undefined }, { ...permission, scope: 'workspace' }, { ...permission, scope: ['isolated_environment'] },
      { ...permission, effects: [] }, { ...permission, effects: ['create', 'create'] }, { ...permission, effects: ['shell'] },
      { ...permission, effects: 'create' }, { ...permission, facts: [] }, { ...permission, facts: { oversized: 'x'.repeat(12001) } },
      { ...permission, facts: cyclic }, { ...permission, facts: tooDeep }, { ...permission, facts: { bad: undefined } },
      { ...permission, facts: { bad: NaN } }, { ...permission, facts: Object.fromEntries(Array.from({ length: 65 }, (_, i) => [`key${i}`, i])) },
      { ...permission, title: 'Host copy' }, { ...permission, allowLabel: 'Allow' }]) {
      assert.equal(isCanvasAgentPermissionRequest(invalid), false);
    }
  });

  test('agent copy has exact bounded plain-text fields with no executable or policy fields', () => {
    assert.equal(isCanvasAgentPermissionPresentation(presentation), true);
    for (const invalid of [{ ...presentation, title: '' }, { ...presentation, title: 'x'.repeat(161) }, { ...presentation, summary: 'x'.repeat(513) },
      { ...presentation, allowLabel: 'x'.repeat(65) }, { ...presentation, denyLabel: '' }, { ...presentation, denyLabel: 'x'.repeat(65) },
      { ...presentation, title: ' padded ' }, { ...presentation, summary: 'Line one\nLine two' }, { ...presentation, allowLabel: ' ' },
      { ...presentation, operation: 'different.operation' }, { ...presentation, approved: true }, { ...presentation, mode: 'full_setup' }]) {
      assert.equal(isCanvasAgentPermissionPresentation(invalid), false);
    }
  });

  test('approved receipts require generated presentation and preserve exact policy scope and effects', () => {
    const receipt = { ...permission, id: 'permission', requestId: 'turn', status: 'pending' };
    assert.equal(isCanvasAgentPermission(receipt), true);
    assert.equal(isCanvasAgentPermission({ ...receipt, presentation }), true);
    assert.equal(isCanvasAgentPermission({ ...receipt, status: 'approved' }), false);
    assert.equal(isCanvasAgentPermission({ ...receipt, status: 'cancelled' }), true);
    const approved = { ...receipt, status: 'approved', presentation };
    assert.equal(isCanvasAgentPermission({ ...approved, approval: 'user' }), true);
    assert.equal(isCanvasAgentPermission({ ...approved, approval: 'policy', mode: 'auto_setup' }), true);
    assert.equal(isCanvasAgentPermission({ ...approved, effects: ['execute_code'], scope: 'existing_environment', approval: 'policy', mode: 'full_setup' }), true);
    for (const invalid of [{ ...approved, approval: 'policy' }, { ...approved, approval: 'policy', mode: 'ask' },
      { ...approved, approval: 'policy', mode: ['full_setup'] }, { ...approved, approval: 'user', mode: 'full_setup' },
      { ...approved, approval: 'policy', mode: 'auto_setup', scope: 'existing_environment' },
      { ...approved, approval: 'policy', mode: 'auto_setup', effects: ['install', 'execute_code'] },
      ...['pending', 'denied', 'cancelled'].map(status => ({ ...approved, status, approval: 'user' })),
      { ...receipt, decision: 'allow' }, { ...receipt, status: 'always_allow' }]) assert.equal(isCanvasAgentPermission(invalid), false);
  });

  test('mode and decision messages cannot forge generated presentation or operation receipts', () => {
    for (const mode of ['ask', 'auto_setup', 'full_setup']) {
      assert.equal(isCanvasAgentPermissionMode(mode), true);
      const action = message({ action: 'permission_mode', mode } as CanvasAgentAction);
      assert.equal(isLocalBrowserMessage(action), true); assert.equal(isLocalBrowserMessage({ ...action, approval: 'policy' }), false);
      assert.equal(isLocalBrowserMessage({ ...action, permissions: [] }), false);
    }
    for (const mode of ['always_allow', ['ask'], null, { mode: 'ask' }]) {
      assert.equal(isCanvasAgentPermissionMode(mode), false); assert.equal(isLocalBrowserMessage({ ...message({ action: 'permission_mode', mode: 'ask' }), mode }), false);
    }
    const decision = message({ action: 'permission', permissionId: 'permission', decision: 'allow' });
    assert.equal(isLocalBrowserMessage(decision), true); assert.equal(isLocalBrowserMessage({ ...decision, decision: 'deny' }), true);
    for (const invalid of [{ ...decision, requestId: 'other-turn' }, { ...decision, command: 'install' }, { ...decision, presentation },
      { ...decision, decision: 'always_allow' }, { ...decision, permissionId: '' }, { ...decision, permissionId: 'x'.repeat(129) }]) assert.equal(isLocalBrowserMessage(invalid), false);
  });

  test('state rejects duplicate, oversized, unbound or unknown permission payloads', () => {
    const receipt = { ...permission, id: 'permission', requestId: 'turn', status: 'pending' as const };
    const state: CanvasAgentState = { sessionId: 'session', revision: 0, provider: 'codex', status: 'running', messages: [], activity: '',
      requestId: 'turn', context: { target, nodeIds: [] }, permissions: [receipt] };
    assert.equal(isCanvasAgentState(state), true); assert.equal(isCanvasAgentState({ ...state, permissionMode: 'full_setup' }), true);
    for (const invalid of [{ ...state, permissionMode: 'always_allow' }, { ...state, permissionMode: ['ask'] },
      { ...state, permissions: Array.from({ length: 21 }, (_, i) => ({ ...receipt, id: `id-${i}`, status: 'cancelled' })) },
      { ...state, permissions: [receipt, receipt] }, { ...state, permissions: [receipt, { ...receipt, id: 'second' }] },
      { ...state, requestId: 'another-turn' }, { ...state, status: 'idle' }, { ...state, context: undefined },
      { ...state, permissions: [{ ...receipt, command: 'install' }] }, { ...state, arbitraryCommand: 'run' }]) assert.equal(isCanvasAgentState(invalid), false);
  });
});
