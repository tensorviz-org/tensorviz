import * as assert from 'node:assert/strict';
import { McpOpenGraph, registerMcpOpenGraph, type OpenGraphArgs } from '@tensorviz/graph-host/mcpOpenGraph';
import { captureTools } from '../../mcpHost';
import { GraphToolError, type JsonObject } from '@tensorviz/graph-host/mcpGraph';
import type { GraphOpeningState } from '@tensorviz/contracts/protocol';

suite('MCP unified opening', () => {
  const tick = () => new Promise(resolve => setImmediate(resolve));
  const next = (result: JsonObject) => result.nextAction as { kind: string; arguments: OpenGraphArgs; requiredArguments: string[] };
  const blocker = { reason: 'Required fixture is unavailable.', evidence: 'model.py: forward reads a supplied fixture; no fixture is present in the workspace.', nextStep: 'Provide the fixture path.' };
  function rig(options: { missing?: boolean; installError?: GraphToolError; environment?: boolean } = {}) {
    let state: GraphOpeningState | undefined;
    let actions = 0, opens = 0, installs = 0, focuses = 0;
    const service = new McpOpenGraph({
      open: async () => {
        opens++;
        state = { id: `opening-${opens}`, target: { workspaceFolder: '/w', file: 'model.py', cls: 'Net',
          sourceDigest: 'a'.repeat(64), documentVersion: 1, backendRevision: 0 },
          phase: 'needs_setup', hasCapture: false, detail: 'Choose inputs.',
          discovery: { model: 'model:Net', sourceDigest: 'a'.repeat(64), init: [], inputs: [], examples: [],
            factories: [], configs: [], parents: [], constraints: [], truncated: false,
            resolution: { status: 'partial', draft: { profile: {} }, unresolved: ['inputs.x'], evidence: [] } } };
        return state;
      },
      state: () => state,
      action: async (_before, action) => {
        actions++;
        if (action.kind === 'submit') state = { ...state!, phase: 'ready', hasCapture: true };
        if (action.kind === 'cancel') state = { ...state!, phase: 'cancelled' };
        return true;
      },
      model: () => ({ models: [] }),
      dependencies: async () => ({ missingModules: installs || options.missing === false ? [] : [{ module: 'einops', requirement: 'einops>=0.8' }] }),
      install: async (_state, packages) => { assert.deepEqual(packages, ['einops>=0.8']); installs++; if (options.installError) throw options.installError; },
      focusFailure: async () => { focuses++; return { focused: true }; },
      environment: async () => options.environment ?? true,
    });
    return { service, get state() { return state; }, set state(value) { state = value; },
      get actions() { return actions; }, get opens() { return opens; }, get installs() { return installs; }, get focuses() { return focuses; } };
  }
  test('is registered without a graph and starts once, then polls without re-execution', async () => {
    const r = rig(), tools = captureTools(sink => registerMcpOpenGraph(sink, r.service, 'one'));
    assert.ok(tools.has('open_graph'));
    const started = r.service.invoke('one', {});
    assert.equal(started.status, 'opening'); assert.equal(started.validated, false);
    await tick();
    const result = r.service.invoke('one', { requestId: started.requestId as string });
    assert.equal(result.status, 'needs_setup'); assert.deepEqual(result.unresolved, ['inputs.x']);
    assert.ok(result.inputContract); assert.ok(result.draft); assert.equal(r.opens, 1);
  });

  test('execution review is a user choice rather than a setup failure or endless polling', async () => {
    const r = rig(), started = r.service.invoke('one', {}); await tick();
    r.state = { ...r.state!, phase: 'awaiting_execution', execution: { id: 'review', kind: 'capture', phase: 'review',
      device: 'cuda:0', configuredDevice: 'cuda:0', interpreter: '/python', detail: 'Review', inputs: '{}', mode: 'eval', gradients: false,
      stages: [{ label: 'Capture', calls: 1 }] } };
    const result = r.service.invoke('one', { requestId: started.requestId as string });
    assert.equal(result.status, 'awaiting_execution'); assert.equal(result.validated, false);
    assert.equal(result.continuationRequired, false); assert.equal(next(result).kind, 'user_execution');
    assert.equal(r.actions, 0, 'an agent cannot approve the review by polling');
  });

  test('typed environment recovery gives the agent candidate IDs and exact native repair while retaining the opening', async () => {
    const r = rig(), started = r.service.invoke('one', {}); await tick();
    const environment = { reason: 'selection_required' as const, detail: 'Choose project Python.', candidates: [{ id: 'project',
      interpreter: '/w/.venv/bin/python', source: 'Model folder', status: 'libraries_present' as const,
      python: '3.12.13', torch: '2.6.0+cu124', fingerprint: 'c'.repeat(64) }] };
    r.state = { ...r.state!, environment };
    const requestId = started.requestId as string;
    const choice = r.service.invoke('one', { requestId });
    assert.equal(next(choice).kind, 'select_environment'); assert.deepEqual(next(choice).requiredArguments, []);
    assert.equal(next(choice).arguments.candidateId, 'project', 'the only environment in the model folder is offered, not selected');
    assert.equal(r.actions, 0);
    assert.match(choice.helpRequest as string, /2.6.0\+cu124|environmentRecovery/);
    assert.throws(() => r.service.invoke('one', { ...next(choice).arguments, candidateId: '/caller/python' }), /listed available candidateId/);
    r.service.invoke('one', { ...next(choice).arguments, candidateId: 'project' }); await tick();
    assert.equal(r.opens, 1); assert.equal(r.actions, 1); assert.equal(r.state!.id, 'opening-1');
    r.state = { ...r.state!, environment: { ...environment, reason: 'missing_libraries', selectedId: 'project', requirements: ['einops>=0.8'] } };
    const repair = r.service.invoke('one', { requestId });
    assert.equal(next(repair).kind, 'review_dependencies'); assert.deepEqual(next(repair).arguments.packages, ['einops>=0.8']);
    r.service.invoke('one', next(repair).arguments); await tick();
    assert.equal(r.opens, 1); assert.equal(r.actions, 2); assert.equal(r.installs, 0, 'the opening controller owns inline native review and resume');
  });

  test('MCP exposes the native installed-version recovery only for reported project pins', async () => {
    const r = rig({ missing: false }), tools = captureTools(sink => registerMcpOpenGraph(sink, r.service, 'one'));
    assert.ok(tools.get('open_graph')!.config.inputSchema.action.safeParse('use_installed_versions').success);
    const started = r.service.invoke('one', {}); await tick();
    const identity = { requestId: started.requestId as string, openingId: r.state!.id, sourceDigest: r.state!.target.sourceDigest };
    r.state = { ...r.state!, environment: { reason: 'incompatible', detail: 'torch 2.15.0 (needs ==2.0.0)',
      candidates: [{ id: 'project', interpreter: '/w/.venv/bin/python', source: 'Project', status: 'libraries_present', selected: true }],
      selectedId: 'project', declaredPins: ['torch==2.0.0'] } };
    const recovery = r.service.invoke('one', { requestId: identity.requestId });
    assert.equal(next(recovery).kind, 'use_installed_versions');
    assert.match(recovery.next as string, /without installing or downgrading/);
    r.service.invoke('one', next(recovery).arguments); await tick();
    assert.equal(r.actions, 1); assert.equal(r.installs, 0); assert.equal(r.opens, 1);
    r.state = { ...r.state!, environment: { ...r.state!.environment!, declaredPins: undefined } };
    assert.throws(() => r.service.invoke('one', { ...identity, action: 'use_installed_versions' }), /project pin mismatches/);
    assert.equal(r.actions, 1);
    assert.throws(() => r.service.invoke('one', { ...identity, sourceDigest: 'b'.repeat(64), action: 'use_installed_versions' }), /source changed/);
  });

  test('permission recovery retains the requested model and waits for a user decision', async () => {
    const r = rig();
    let allowed = false, decisions = 0, opens = 0;
    let accept!: () => void;
    const target = { workspaceRoot: '/w', file: 'model.py', cls: 'Net', interpreter: '/env/python' };
    const service = new McpOpenGraph({
      open: async args => {
        assert.deepEqual(Object.fromEntries(Object.entries(args).filter(([, value]) => value !== undefined)), target); opens++;
        if (!allowed) throw new GraphToolError('APPROVAL_DECLINED', 'Declined.');
        r.service.invoke('fixture', {}); await tick();
        return r.state!;
      }, state: () => r.state, action: async () => true, model: () => ({}),
      permission: async () => { decisions++; await new Promise<void>(resolve => { accept = resolve; }); allowed = true; },
    });
    const started = service.invoke('one', target); await tick();
    const requestId = started.requestId as string;
    assert.equal(next(service.invoke('one', { requestId })).kind, 'permission_required');
    assert.equal(opens, 1); assert.equal(decisions, 0);
    service.invoke('one', { requestId, action: 'request_permission' });
    await tick(); assert.equal(opens, 1); assert.equal(decisions, 1);
    accept(); await tick(); await tick();
    assert.equal(service.invoke('one', { requestId }).status, 'needs_setup');
    assert.equal(opens, 2);
  });

  test('the execution environment can be inspected before opening without selecting or executing a model', async () => {
    const signal = new AbortController().signal;
    const environment = { source: 'managed', interpreter: '/tensorviz/venv/bin/python', python: '3.11.9' };
    const service = new McpOpenGraph({ open: async () => { throw new Error('must not open'); }, state: () => undefined,
      action: async () => { throw new Error('must not change setup'); }, model: () => ({}),
      inspectEnvironment: async (file, actualSignal) => {
        assert.equal(file, '/w/model.py'); assert.equal(actualSignal, signal);
        return { ok: true, environment };
      } });
    const tools = captureTools(sink => registerMcpOpenGraph(sink, service, 'one'));
    const tool = tools.get('get_execution_environment')!;
    assert.equal(tool.config.annotations.readOnlyHint, true);
    const reply = await tool.handler({ file: '/w/model.py' }, { signal });
    assert.deepEqual(reply.structuredContent, { ok: true, environment });
  });

  test('source-change interruption returns an exact refresh continuation; user cancellation does not', async () => {
    const r = rig(), request = r.service.invoke('one', {}); await tick();
    r.state = { ...r.state!, phase: 'cancelled', cancellationReason: 'source_changed',
      target: { ...r.state!.target, sourceDigest: 'b'.repeat(64), documentVersion: 2 } };
    let result = r.service.invoke('one', { requestId: request.requestId as string });
    assert.equal(result.status, 'source_changed'); assert.equal(result.continuationRequired, true);
    assert.equal(result.validated, false);
    assert.equal(next(result).kind, 'restart');
    assert.deepEqual(next(result).arguments, { file: '/w/model.py', cls: 'Net' });
    assert.equal(r.opens, 1, 'reading interruption evidence must not itself start capture');
    result = r.service.invoke('one', { requestId: request.requestId as string, openingId: r.state.id,
      sourceDigest: r.state.target.sourceDigest, action: 'report_blocker', blocker });
    assert.equal(result.status, 'blocked'); assert.equal(result.continuationRequired, false);
    r.state = { ...r.state!, cancellationReason: undefined };
    result = r.service.invoke('one', { requestId: request.requestId as string });
    assert.equal(result.status, 'cancelled'); assert.equal(result.continuationRequired, false);
    assert.equal(next(result).kind, 'stopped');
  });

  test('the source-change continuation opens a new request instead of reusing stale recovery state', async () => {
    const r = rig(), request = r.service.invoke('one', { file: '/w/model.py', cls: 'Net' }); await tick();
    r.state = { ...r.state!, phase: 'cancelled', cancellationReason: 'source_changed' };
    const interrupted = r.service.invoke('one', { requestId: request.requestId as string });
    const refreshed = r.service.invoke('one', next(interrupted).arguments); await tick();
    assert.notEqual(refreshed.requestId, request.requestId);
    assert.equal(r.opens, 2);
    r.service.invoke('one', { requestId: refreshed.requestId as string, openingId: r.state!.id,
      sourceDigest: r.state!.target.sourceDigest, draft: { profile: {} } }); await tick();
    assert.equal(r.service.invoke('one', { requestId: refreshed.requestId as string }).validated, true);
  });

  test('opening reports the capture environment even when dependency setup fails', async () => {
    const r = rig(); r.service.invoke('seed', {}); await tick();
    const environment = { interpreter: '/tensorviz/venv/bin/python', source: 'managed' };
    const service = new McpOpenGraph({ open: async () => r.state!, state: () => r.state,
      action: async () => true, model: () => ({}), dependencies: async () => ({ environment, missingModules: [] }),
      install: async () => { throw new GraphToolError('DEPENDENCY_CONFLICT', 'Conflicting requirements.'); } });
    const request = service.invoke('one', {}); await tick();
    service.invoke('one', { requestId: request.requestId as string, openingId: r.state!.id,
      sourceDigest: r.state!.target.sourceDigest, action: 'install_dependencies', packages: ['fairscale'] });
    await tick();
    const result = service.invoke('one', { requestId: request.requestId as string });
    assert.deepEqual(result.environment, environment); assert.equal(next(result).kind, 'inspect_failure');
    assert.notEqual(next(result).arguments.action, 'environment');
  });
  test('browser environment recovery cannot report the old interpreter inventory', async () => {
    const r = rig(); r.service.invoke('seed', {}); await tick();
    const service = new McpOpenGraph({ open: async () => r.state!, state: () => r.state,
      action: async () => true, model: () => ({}), dependencies: async () => ({
        environment: { executable: '/project/.venv/bin/python', python: '3.12', torch: '2.6', pip: null },
        missingModules: [{ module: 'old_dependency' }] }) });
    const request = service.invoke('one', {}); await tick();
    r.state = { ...r.state!, environment: { reason: 'ready', detail: 'Recovered', selectedId: 'new', candidates: [
      { id: 'new', interpreter: '/project/.venv/bin/python', source: 'Selected', status: 'libraries_present',
        python: '3.11', torch: '2.7', prefix: '/project/.venv' }] } };
    const result = service.invoke('one', { requestId: request.requestId as string });
    assert.equal((result.environment as JsonObject).python, '3.11');
    assert.equal((result.environment as JsonObject).torch, '2.7');
    assert.equal((result.environment as JsonObject).executable, '/project/.venv/bin/python');
    assert.equal(result.dependencies, undefined, 'old missing-library facts must not survive a changed environment');
  });
  for (const changed of [false, true]) test(`dependency inspection ${changed ? 'expires' : 'remains'} when Python versions stay the same`, async () => {
    const r = rig(); r.service.invoke('seed', {}); await tick();
    r.state = { ...r.state!, environment: { reason: 'missing_libraries', detail: 'Missing import', selectedId: 'project', candidates: [
      { id: 'project', interpreter: '/project/.venv/bin/python', source: 'Selected', status: 'libraries_present',
        python: '3.12', torch: '2.6', prefix: '/project/.venv', fingerprint: 'a'.repeat(64) }] } };
    const dependencies = { environment: { executable: '/project/.venv/bin/python', python: '3.12', torch: '2.6', pip: null },
      missingModules: [{ module: 'pkg_resources' }] };
    const service = new McpOpenGraph({ open: async () => r.state!, state: () => r.state,
      action: async () => true, model: () => ({}), dependencies: async () => dependencies });
    const request = service.invoke('one', {}); await tick();
    const environment = r.state.environment!;
    r.state = { ...r.state, phase: 'ready', hasCapture: true, environment: { ...environment, reason: 'ready', candidates: [
      { ...environment.candidates[0], fingerprint: changed ? 'b'.repeat(64) : 'a'.repeat(64) }] } };
    const result = service.invoke('one', { requestId: request.requestId as string });
    assert.equal(result.validated, true);
    assert.equal((result.environment as JsonObject).executable, dependencies.environment.executable);
    assert.deepEqual(result.dependencies, changed ? undefined : dependencies,
      'same-path package replacement must invalidate old missing imports; unchanged inventory stays usable');
  });
  test('submits inputs through the shared opening and requires a measured capture', async () => {
    const r = rig(), started = r.service.invoke('one', {}); await tick();
    const args = { requestId: started.requestId as string, openingId: r.state!.id, sourceDigest: r.state!.target.sourceDigest };
    r.service.invoke('one', { ...args, draft: { profile: {} } }); await tick();
    assert.equal(r.service.invoke('one', { requestId: args.requestId }).validated, true);
    r.state = { ...r.state!, phase: 'failed', hasCapture: true };
    assert.equal(r.service.invoke('one', { requestId: args.requestId }).validated, false);
    r.state = { ...r.state!, phase: 'ready', hasCapture: false };
    assert.equal(r.service.invoke('one', { requestId: args.requestId }).validated, false);
  });
  test('a foreign connection or changed source cannot submit or cancel the request', async () => {
    const r = rig(), started = r.service.invoke('one', {}); await tick();
    const args = { requestId: started.requestId as string, openingId: r.state!.id, sourceDigest: 'b'.repeat(64), draft: { profile: {} } };
    assert.throws(() => r.service.invoke('two', args), /another connection/);
    assert.throws(() => r.service.invoke('one', args), /source changed/);
    assert.equal(r.actions, 0);
    r.service.cancelOwner('two'); assert.equal(r.actions, 0);
    r.service.cancelOwner('one'); assert.equal(r.actions, 1);
  });
  test('retargeting never reports the replacement graph as this request result', async () => {
    const r = rig(), started = r.service.invoke('one', {}); await tick();
    r.state = { ...r.state!, id: 'another-opening', phase: 'ready', hasCapture: true };
    const result = r.service.invoke('one', { requestId: started.requestId as string });
    assert.equal(result.status, 'superseded'); assert.equal(result.validated, false);
    assert.equal(result.opening, undefined);
    r.service.cancelOwner('one'); assert.equal(r.actions, 0);
  });
  test('another start cannot interrupt a capture already in progress', async () => {
    const r = rig(); r.service.invoke('one', {}); await tick();
    r.state = { ...r.state!, phase: 'capturing' };
    assert.throws(() => r.service.invoke('two', {}), /in progress/);
    assert.equal(r.opens, 1);
  });
  test('dependency recovery uses the approved installer then retries, but never during capture', async () => {
    const r = rig(), started = r.service.invoke('one', {}); await tick();
    const args = { requestId: started.requestId as string, openingId: r.state!.id,
      sourceDigest: r.state!.target.sourceDigest, action: 'install_dependencies' as const, packages: ['einops>=0.8'] };
    r.state = { ...r.state!, phase: 'capturing' };
    assert.throws(() => r.service.invoke('one', args), /capture to finish/); assert.equal(r.installs, 0);
    r.state = { ...r.state!, phase: 'failed' };
    r.service.invoke('one', args); await tick();
    assert.equal(r.installs, 1); assert.equal(r.actions, 1);
    assert.deepEqual(r.service.invoke('one', { requestId: args.requestId }).dependencies, { missingModules: [] });
  });
  test('a captured failure focuses once and reports the renderer receipt separately from validation', async () => {
    const r = rig(), started = r.service.invoke('one', {}); await tick();
    const args = { requestId: started.requestId as string };
    r.state = { ...r.state!, phase: 'failed', hasCapture: true };
    assert.deepEqual(r.service.invoke('one', args).failureFocus, { status: 'pending' });
    await tick();
    assert.deepEqual(r.service.invoke('one', args).failureFocus, { focused: true });
    assert.equal(r.service.invoke('one', args).validated, false); assert.equal(r.focuses, 1);
  });

  test('MCP exposes the same failed-opening investigation context without retrying', async () => {
    const r = rig(), started = r.service.invoke('one', {}); await tick();
    r.state = { ...r.state!, phase: 'failed', failure: { kind: 'execution' }, detail: 'Forward raised an exception.' };
    const result = r.service.invoke('one', { requestId: started.requestId as string });
    assert.match(result.helpRequest as string, /MCP directly/);
    assert.ok((result.helpRequest as string).includes(r.state.id));
    assert.ok((result.helpRequest as string).includes(r.state.target.sourceDigest));
    assert.match(result.helpRequest as string, /Forward raised an exception/);
    assert.equal(result.validated, false); assert.equal(r.opens, 1); assert.equal(r.actions, 0);
    r.state = { ...r.state, phase: 'ready', hasCapture: true };
    assert.equal(r.service.invoke('one', { requestId: started.requestId as string }).helpRequest, undefined);
  });
  test('disconnect cancels the new opening while a retry is still capturing', async () => {
    let finish!: () => void;
    const r = rig(); r.service.invoke('seed', {}); await tick();
    let state = r.state!, cancelledId: string | undefined;
    const service = new McpOpenGraph({ open: async () => state, state: () => state, model: () => ({}),
      action: async (_before, action) => {
        if (action.kind === 'retry') {
          state = { ...state, id: 'retry-opening', phase: 'capturing' };
          await new Promise<void>(resolve => { finish = resolve; });
        } else if (action.kind === 'cancel') { cancelledId = state.id; state = { ...state, phase: 'cancelled' }; }
        return true;
      } });
    const request = service.invoke('one', {}); await tick();
    service.invoke('one', { requestId: request.requestId as string, openingId: state.id,
      sourceDigest: state.target.sourceDigest, action: 'retry' });
    service.cancelOwner('one');
    assert.equal(cancelledId, 'retry-opening'); finish(); await tick();
  });
  test('opening errors without a setup tell the agent to correct and restart', async () => {
    const service = new McpOpenGraph({ open: async () => { throw new GraphToolError('CLASS_REQUIRED', 'Choose Net or Other.'); },
      state: () => undefined, model: () => ({}), action: async () => false });
    const request = service.invoke('one', {}); await tick();
    const result = service.invoke('one', { requestId: request.requestId as string });
    assert.match(String(result.next), /start a new open_graph/); assert.equal(result.opening, undefined);
  });
  test('selecting an environment resumes the same request with exactly one new opening', async () => {
    const r = rig(), request = r.service.invoke('one', {}); await tick();
    r.service.invoke('one', { requestId: request.requestId as string, openingId: r.state!.id,
      sourceDigest: r.state!.target.sourceDigest, action: 'environment' });
    await tick();
    const result = r.service.invoke('one', { requestId: request.requestId as string });
    assert.equal(result.status, 'needs_setup'); assert.equal(r.opens, 2); assert.equal(r.actions, 0);
    assert.equal((result.opening as GraphOpeningState).id, 'opening-2');
  });

  test('returned actions carry one request through dependencies, setup and capture', async () => {
    const r = rig();
    let result = r.service.invoke('one', {});
    assert.equal(result.continuationRequired, true); assert.equal(next(result).kind, 'poll');
    await tick();
    result = r.service.invoke('one', next(result).arguments);
    assert.equal(next(result).kind, 'review_dependencies');
    assert.deepEqual(next(result).requiredArguments, ['packages']);
    assert.equal(r.installs, 0, 'Static imports are evidence, never automatic installation consent');
    result = r.service.invoke('one', { ...next(result).arguments, packages: ['einops>=0.8'] });
    assert.equal(result.continuationRequired, true);
    await tick();
    result = r.service.invoke('one', next(result).arguments);
    assert.equal(next(result).kind, 'complete_setup');
    assert.deepEqual(next(result).requiredArguments, ['draft']);
    assert.equal(result.validated, false); assert.equal(result.continuationRequired, true);
    result = r.service.invoke('one', { ...next(result).arguments, draft: { profile: {} } });
    await tick();
    result = r.service.invoke('one', next(result).arguments);
    assert.equal(result.validated, true); assert.equal(result.continuationRequired, false);
    assert.equal(next(result).kind, 'done'); assert.equal(r.opens, 1); assert.equal(r.installs, 1);
  });

  test('partial captures and source-only graphs remain unfinished, including a missing capture in ready', async () => {
    const r = rig({ missing: false }), request = r.service.invoke('one', {}); await tick();
    for (const [phase, hasCapture, action] of [['failed', true, 'inspect_failure'], ['source_only', false, 'retry'],
      ['ready', false, 'inspect_failure']] as const) {
      r.state = { ...r.state!, phase, hasCapture };
      const result = r.service.invoke('one', { requestId: request.requestId as string });
      assert.equal(result.validated, false); assert.equal(result.continuationRequired, true);
      assert.equal(next(result).kind, action);
    }
  });

  test('setup corrections return current requirements instead of superseded discovery omissions', async () => {
    const r = rig({ missing: false }), request = r.service.invoke('one', {}); await tick();
    r.state = { ...r.state!, requirements: ['inputs.x: dtype must be int64'] };
    const result = r.service.invoke('one', { requestId: request.requestId as string });
    assert.deepEqual(result.unresolved, r.state.requirements);
    assert.equal(next(result).kind, 'complete_setup');
  });

  test('a concrete blocker stops continuation without reporting success and can resume after resolution', async () => {
    const r = rig({ missing: false }), request = r.service.invoke('one', {}); await tick();
    const identity = { requestId: request.requestId as string, openingId: r.state!.id, sourceDigest: r.state!.target.sourceDigest };
    let result = r.service.invoke('one', { ...identity, action: 'report_blocker', blocker });
    assert.equal(result.status, 'blocked'); assert.equal(result.continuationRequired, false);
    assert.equal(result.validated, false); assert.equal((result.blocker as JsonObject).source, 'agent');
    assert.equal(r.actions, 0, 'A reported blocker must not execute or mutate the graph');
    result = r.service.invoke('one', { requestId: identity.requestId });
    assert.equal(result.status, 'blocked', 'Polling retains the reason the agent stopped');
    r.service.invoke('one', { ...identity, draft: { profile: {} } }); await tick();
    result = r.service.invoke('one', { requestId: identity.requestId });
    assert.equal(result.validated, true); assert.equal(result.blocker, undefined);
  });

  test('blockers need evidence, ownership and the current source, and cannot interrupt capture', async () => {
    const r = rig(), request = r.service.invoke('one', {});
    assert.throws(() => r.service.invoke('one', { requestId: request.requestId as string, action: 'report_blocker', blocker }), /still running/);
    await tick();
    const args = { requestId: request.requestId as string, openingId: r.state!.id, sourceDigest: r.state!.target.sourceDigest, action: 'report_blocker' as const, blocker };
    assert.throws(() => r.service.invoke('two', args), /another connection/);
    assert.throws(() => r.service.invoke('one', { ...args, sourceDigest: 'b'.repeat(64) }), /source changed/);
    assert.throws(() => r.service.invoke('one', { ...args, blocker: { ...blocker, evidence: '  ' } }), /observed evidence/);
    assert.throws(() => r.service.invoke('one', { ...args, blocker: undefined }), /observed evidence/);
    assert.throws(() => r.service.invoke('one', { ...args, action: 'retry' }), /needs action/);
    for (const phase of ['capturing', 'ready'] as const) {
      r.state = { ...r.state!, phase, hasCapture: phase === 'ready' };
      assert.throws(() => r.service.invoke('one', args), /incomplete opening/);
    }
    assert.equal(r.actions, 0);
  });

  test('a blocker for old source cannot mask changed or independently completed work', async () => {
    const r = rig(), request = r.service.invoke('one', {}); await tick();
    r.service.invoke('one', { requestId: request.requestId as string, openingId: r.state!.id,
      sourceDigest: r.state!.target.sourceDigest, action: 'report_blocker', blocker });
    r.state = { ...r.state!, target: { ...r.state!.target, sourceDigest: 'b'.repeat(64) } };
    let result = r.service.invoke('one', { requestId: request.requestId as string });
    assert.equal(result.blocker, undefined); assert.equal(result.continuationRequired, true);
    r.state = { ...r.state!, target: { ...r.state!.target, sourceDigest: 'a'.repeat(64) }, phase: 'ready', hasCapture: true };
    result = r.service.invoke('one', { requestId: request.requestId as string });
    assert.equal(result.blocker, undefined); assert.equal(result.validated, true);
  });

  test('declined dependency approval is a concrete stop, not an instruction to prompt again', async () => {
    const r = rig({ installError: new GraphToolError('INSTALL_DECLINED', 'The user declined the native plan.') });
    const request = r.service.invoke('one', {}); await tick();
    let result = r.service.invoke('one', { requestId: request.requestId as string });
    r.service.invoke('one', { ...next(result).arguments, packages: ['einops>=0.8'] }); await tick();
    result = r.service.invoke('one', { requestId: request.requestId as string });
    assert.equal(result.status, 'blocked'); assert.equal(result.continuationRequired, false);
    assert.equal(result.validated, false); assert.equal((result.blocker as JsonObject).source, 'host');
    assert.equal(next(result).arguments, undefined); assert.equal(r.installs, 1);
    assert.equal(r.actions, 0, 'Declined installation must not retry the model');
  });

  test('a failed install plan asks for diagnosis before another installation attempt', async () => {
    const r = rig({ installError: new GraphToolError('DEPENDENCY_CONFLICT', 'Existing package constraints conflict.') });
    const request = r.service.invoke('one', {}); await tick();
    const result = r.service.invoke('one', { requestId: request.requestId as string });
    r.service.invoke('one', { ...next(result).arguments, packages: ['einops>=0.8'] }); await tick();
    const failed = r.service.invoke('one', { requestId: request.requestId as string });
    assert.equal(failed.continuationRequired, true); assert.equal(next(failed).kind, 'inspect_failure');
    assert.equal(next(failed).arguments.action, undefined); assert.equal(r.installs, 1);
  });

  test('an unfinished environment choice stops without prompting again and resumes after selection', async () => {
    const options = { environment: false }, r = rig(options), request = r.service.invoke('one', {}); await tick();
    const args = { requestId: request.requestId as string, openingId: r.state!.id,
      sourceDigest: r.state!.target.sourceDigest, action: 'environment' as const };
    r.service.invoke('one', args); await tick();
    let result = r.service.invoke('one', { requestId: args.requestId });
    assert.equal(result.status, 'blocked'); assert.equal(result.continuationRequired, false);
    assert.equal((result.blocker as JsonObject).source, 'host'); assert.equal(next(result).arguments, undefined);
    assert.equal(r.opens, 1);
    options.environment = true;
    r.service.invoke('one', args); await tick();
    result = r.service.invoke('one', { requestId: args.requestId });
    assert.equal(result.blocker, undefined); assert.equal(result.continuationRequired, true);
    assert.equal(r.opens, 2);
  });

  test('errors before model selection support target correction or an explicit blocker', async () => {
    const service = new McpOpenGraph({ open: async () => { throw new GraphToolError('CLASS_REQUIRED', 'Choose Net or Other.'); },
      state: () => undefined, model: () => ({}), action: async () => false });
    const request = service.invoke('one', { file: '/w/model.py' }); await tick();
    let result = service.invoke('one', { requestId: request.requestId as string });
    assert.equal(result.continuationRequired, true); assert.equal(next(result).kind, 'restart');
    assert.equal(next(result).arguments.file, '/w/model.py');
    result = service.invoke('one', { requestId: request.requestId as string, action: 'report_blocker', blocker: {
      reason: 'Two models match the request.', evidence: 'model.py defines Net and Other; neither is referenced by an entry point.', nextStep: 'Choose Net or Other.' } });
    assert.equal(result.status, 'blocked'); assert.equal(result.validated, false);
  });
});
