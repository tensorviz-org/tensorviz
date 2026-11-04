import * as assert from 'node:assert/strict';
import { CanvasAgent, type AgentRunEvent, type AgentRunInput } from '../../canvasAgent';
import type { CanvasAgentAction, CanvasAgentPermissionRequest, CanvasAgentState, GraphOpeningState, LocalBrowserAgentRequest, ModelImportDiagnostic } from '@tensorviz/contracts/protocol';

const target = { workspaceFolder: '/project', file: 'model.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
const request = (action: CanvasAgentAction): LocalBrowserAgentRequest => ({ type: 'agentRequest', protocolVersion: 1, workspaceSessionId: 'workspace', messageId: 'message', viewId: 'view', ...action });
const tick = () => new Promise<void>(resolve => setImmediate(resolve));
const environment = (): GraphOpeningState => ({ id: 'opening', target, phase: 'needs_setup', hasCapture: false, detail: 'Choose Python.',
  environment: { reason: 'selection_required', detail: '', candidates: [] } });
const importMessage = "cannot import name 'ALL_ATTENTION_FUNCTIONS' from 'transformers.modeling_utils'";
const diagnostic: ModelImportDiagnostic = {
  exceptionType: 'ImportError', message: importMessage,
  traceback: `Traceback (most recent call last):\n  File "/project/model.py", line 4, in <module>\nImportError: ${importMessage}`,
  causes: [{ exceptionType: 'ImportError', message: importMessage, module: 'transformers.modeling_utils' }],
  runtime: { platform: 'darwin', architecture: 'arm64', python: '3.12.13', torch: '2.6.0', cuda: null, cudaAvailable: false, mpsAvailable: true }, truncated: false,
};
const failed = (): GraphOpeningState => ({ ...environment(), phase: 'failed', detail: `Model import failed: ${importMessage}`,
  failure: { kind: 'environment', code: 'MODEL_IMPORT_FAILED', importFailure: true, diagnostic } });
const commandPermission: CanvasAgentPermissionRequest = { operation: 'project.command.run', scope: 'machine', effects: ['execute_code'],
  facts: { command: ['python', '-c', 'import platform; print(platform.machine())'], cwd: '/project' } };

suite('Canvas agent command and failure investigation lifecycle', () => {
  let agent: CanvasAgent, opening: GraphOpeningState, states: CanvasAgentState[], runs: AgentRunInput[], signals: AbortSignal[];
  let emitters: Array<(event: AgentRunEvent) => void>, finishes: Array<() => void>, workWaits: Array<() => void>;
  let busy: boolean, unreadJobs: string[], preparations: number, completedSignals: AbortSignal[];
  const latest = () => states.at(-1)!;
  const finish = async () => { assert.ok(finishes.length, 'a provider segment must be active'); finishes.shift()!(); await tick(); };
  const updateWork = async (running: boolean, unread: string[] = []) => { busy = running; unreadJobs = unread; workWaits.splice(0).forEach(resolve => resolve()); await tick(); };
  const send = async (requestId = 'chat-turn') => {
    agent.handle(request({ action: 'send', requestId, provider: 'claude', target, nodeIds: [], prompt: 'Investigate this model setup.' })); await tick();
  };
  const startSetup = async () => { agent.openGraph(opening); await tick(); };
  setup(() => {
    opening = environment(); states = []; runs = []; signals = []; emitters = []; finishes = []; workWaits = [];
    busy = false; unreadJobs = []; preparations = 0; completedSignals = [];
    agent = new CanvasAgent({
      preferences: { provider: 'claude', selections: { claude: { model: 'saved-model', effort: 'high' } } },
      prepare: async () => ({ cwd: '/project', context: `fresh host evidence ${++preparations}` }),
      openingProgress: async (_target, signal) => { signal.throwIfAborted(); return opening; },
      projectWork: {
        running: signal => { assert.equal(signal, signals.at(-1)); return busy; },
        unread: signal => { assert.equal(signal, signals.at(-1)); return [...unreadJobs]; },
        wait: async (signal, waitMs) => {
          assert.equal(waitMs, 10000); signal.throwIfAborted();
          await new Promise<void>((resolve, reject) => {
            const abort = () => reject(signal.reason);
            signal.addEventListener('abort', abort, { once: true });
            workWaits.push(() => { signal.removeEventListener('abort', abort); resolve(); });
          });
        },
      },
      run: async (input, emit, signal) => { runs.push(input); emitters.push(emit); signals.push(signal); await new Promise<void>(resolve => finishes.push(resolve)); },
      finishTurn: signal => completedSignals.push(signal), publish: (_view, state) => states.push(state), clear() {},
    });
    agent.handle(request({ action: 'sync' }));
  });
  teardown(async () => { agent.dispose(); finishes.splice(0).forEach(resolve => resolve()); workWaits.splice(0).forEach(resolve => resolve()); await tick(); });

  test('a command outlives its provider segment and its terminal result resumes the same visible request', async () => {
    await send(); const requestId = latest().requestId, sessionId = latest().sessionId;
    busy = true; emitters[0]({ kind: 'text', text: 'I will inspect the host runtime.' }); await finish();
    assert.equal(latest().status, 'running'); assert.equal(workWaits.length, 1); assert.equal(runs.length, 1); assert.deepEqual(completedSignals, []);
    await updateWork(false, ['command-1']);
    assert.equal(runs.length, 2); assert.equal(latest().requestId, requestId); assert.equal(latest().sessionId, sessionId);
    assert.equal(runs[1].conversationId, runs[0].conversationId); assert.equal(signals[1], signals[0]);
    assert.notEqual(runs[1].context, runs[0].context);
    assert.match(runs[1].messages.at(-1)!.text, /get_project_command/);
    assert.match(runs[1].messages.at(-1)!.text, /command-1/); assert.match(runs[1].messages.at(-1)!.text, /Do not rerun/);
    assert.equal(latest().messages.filter(message => message.role === 'user').length, 1);
    assert.doesNotMatch(latest().messages.map(message => message.text).join('\n'), /Host command continuation/);
    unreadJobs = []; await finish();
    assert.equal(latest().status, 'idle'); assert.equal(runs.length, 2); assert.deepEqual(completedSignals, [signals[0]]);
  });

  test('an ignored command result gets two inspection attempts and then stops without starting another command', async () => {
    await send(); unreadJobs = ['command-ignored']; await finish(); await finish();
    assert.equal(runs.length, 3);
    for (const run of runs.slice(1)) {
      assert.match(run.messages.at(-1)!.text, /get_project_command/); assert.match(run.messages.at(-1)!.text, /command-ignored/);
      assert.match(run.messages.at(-1)!.text, /Do not rerun/);
    }
    await finish(); assert.equal(runs.length, 3); assert.equal(latest().status, 'error');
    assert.match(latest().activity, /command finished.*did not inspect/i);
    assert.equal(latest().messages.filter(message => message.role === 'user').length, 1);
  });

  test('command approval presentation and result inspection remain in the originating turn', async () => {
    await send(); const operation = agent.captureOperation(), requestId = latest().requestId;
    busy = true; const approved = operation.approve(commandPermission, operation.signal); await finish();
    assert.equal(runs.length, 2); assert.match(runs[1].messages.at(-1)!.text, /project.command.run/);
    const id = latest().permissions!.at(-1)!.id;
    agent.presentPermission(id, { title: 'Inspect this machine?', summary: 'Read the Python host architecture to check model compatibility.', allowLabel: 'Run inspection', denyLabel: 'Skip inspection' });
    assert.equal(latest().permissions!.at(-1)!.status, 'pending');
    agent.handle(request({ action: 'permission', permissionId: id, decision: 'allow' })); assert.equal(await approved, true);
    await finish(); assert.equal(latest().status, 'running'); assert.equal(runs.length, 2);
    await updateWork(false, ['command-reviewed']); assert.equal(runs.length, 3);
    assert.match(runs[2].messages.at(-1)!.text, /get_project_command/); assert.equal(latest().requestId, requestId);
    assert.ok(signals.every(signal => signal === operation.signal));
    unreadJobs = []; await finish(); assert.equal(latest().status, 'idle');
    assert.equal(latest().permissions!.at(-1)!.approval, 'user');
  });

  test('Stop aborts an in-flight command wait and cannot resume its completed result', async () => {
    await send(); busy = true; await finish();
    agent.handle(request({ action: 'stop', requestId: latest().requestId! })); await tick();
    assert.equal(signals[0].aborted, true); assert.equal(latest().status, 'stopped');
    await updateWork(false, ['command-after-stop']); assert.equal(runs.length, 1); assert.equal(latest().status, 'stopped');
  });

  test('a newly failed background import gets one evidence-based diagnosis continuation', async () => {
    await startSetup(); const requestId = latest().requestId;
    opening = failed(); await finish(); assert.equal(runs.length, 2);
    const prompt = runs[1].messages.at(-1)!.text;
    assert.match(prompt, /Host setup diagnosis continuation/); assert.match(prompt, /ALL_ATTENTION_FUNCTIONS/);
    assert.match(prompt, /darwin/); assert.match(prompt, /arm64/); assert.match(prompt, /"cudaAvailable":false/);
    assert.match(prompt, /Do not repeat/); assert.equal(latest().requestId, requestId); assert.equal(runs[1].intent, 'open_graph');
    assert.equal(signals[1], signals[0]); assert.equal(latest().messages.filter(message => message.role === 'user').length, 1);
    await finish(); assert.equal(runs.length, 2); assert.equal(latest().status, 'error'); assert.match(latest().activity, /ALL_ATTENTION_FUNCTIONS/);
  });

  test('an actual setup-tool evidence event suppresses a redundant diagnosis for that failure', async () => {
    await startSetup(); opening = failed(); emitters[0]({ kind: 'setup_evidence', opening });
    emitters[0]({ kind: 'text', text: `The import is blocked by: ${importMessage}.` }); await finish();
    assert.equal(runs.length, 1); assert.equal(latest().status, 'error'); assert.match(latest().activity, /ALL_ATTENTION_FUNCTIONS/);
  });

  test('evidence for another source cannot suppress the current import diagnosis', async () => {
    await startSetup(); opening = failed();
    emitters[0]({ kind: 'setup_evidence', opening: { ...opening, target: { ...target, sourceDigest: 'b'.repeat(64) } } });
    await finish(); assert.equal(runs.length, 2); assert.match(runs[1].messages.at(-1)!.text, /ALL_ATTENTION_FUNCTIONS/);
    await finish(); assert.equal(latest().status, 'error');
  });

  test('a user reply after failed setup keeps the model setup intent in the same conversation', async () => {
    await startSetup(); opening = failed(); emitters[0]({ kind: 'setup_evidence', opening }); await finish();
    const conversationId = latest().sessionId; await send('investigate-follow-up');
    assert.equal(runs.length, 2); assert.equal(runs[1].intent, 'open_graph'); assert.equal(runs[1].conversationId, conversationId);
    emitters[1]({ kind: 'setup_evidence', opening }); await finish(); assert.equal(latest().status, 'error');
    assert.equal(latest().messages.filter(message => message.role === 'user').length, 2);
  });

  test('captured tool authority expires on Stop and cannot borrow a fresh signal or later turn', async () => {
    assert.throws(() => agent.captureOperation(), /No Canvas agent task is active/);
    await send(); const operation = agent.captureOperation(); assert.equal(operation.current(), true); assert.deepEqual(operation.target, target);
    assert.notEqual(operation.target, target); assert.equal(operation.signal, signals[0]);
    agent.handle(request({ action: 'stop', requestId: latest().requestId! })); await tick();
    assert.equal(operation.current(), false); assert.equal(await operation.approve(commandPermission, new AbortController().signal), false);
    await finish(); await send('second-turn'); assert.equal(agent.captureOperation().current(), true);
    assert.equal(await operation.approve(commandPermission, new AbortController().signal), false);
    assert.equal(latest().permissions?.length ?? 0, 0, 'a stale operation cannot create approval in a new turn');
    await finish();
  });

  test('captured tool authority rejects cancelled callers and a changed model identity', async () => {
    await send(); const operation = agent.captureOperation(), cancelled = new AbortController(); cancelled.abort();
    assert.equal(await operation.approve(commandPermission, cancelled.signal), false); assert.equal(operation.current(), true);
    emitters[0]({ kind: 'opened_graph', text: '', target: { ...target, sourceDigest: 'b'.repeat(64) } });
    assert.equal(operation.current(), false); assert.equal(await operation.approve(commandPermission, new AbortController().signal), false);
    assert.equal(latest().permissions?.length ?? 0, 0); await finish();
  });
});
