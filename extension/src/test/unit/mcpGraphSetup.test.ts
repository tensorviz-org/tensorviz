import * as assert from 'node:assert/strict';
import { registerMcpGraphSetup } from '@tensorviz/graph-host/mcpGraphSetup';
import { GraphOpeningController, type GraphOpeningDependencies } from '@tensorviz/graph-host/graphOpening';
import type { ProfileLookup } from '@tensorviz/graph-host/profileValidation';
import type { GraphSetupHost } from '../../graphSetupHost';
import type { CanvasAgentPermission, GraphEnvironmentRecovery, GraphSetupAction, GraphSetupDiscovery, GraphSetupDraft, LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';

suite('MCP graph setup', () => {
  const target: LocalBrowserTargetIdentity = { workspaceFolder: '/workspace', file: 'model.py', cls: 'Net',
    documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 0 };
  const discovery: GraphSetupDiscovery = { model: 'model:Net', sourceDigest: target.sourceDigest,
    init: [], inputs: [], examples: [], factories: [], configs: [], parents: [], constraints: [], truncated: false,
    resolution: { status: 'partial', draft: { profile: { kind: 'model', inputs: { args: [], kwargs: {} } } },
      evidence: [], unresolved: ['inputs.x'] } };
  async function rig(restoredDraft?: GraphSetupDraft) {
    let saves = 0, reads = 0;
    let discoveredDraft: GraphSetupDraft | undefined;
    const deps: GraphOpeningDependencies = { current: () => true, lookup: async () => ({ ok: false, kind: 'none' }),
      discover: async () => ({ discovery }), save: async () => { saves++; return { requirements: ['inputs.x: Required.'] }; },
      capture: async () => { throw new Error('Discovery must never execute.'); }, recover: async () => undefined, parent: async () => {}, publish: () => {} };
    const opening = new GraphOpeningController(deps); await opening.open(target);
    const handlers: Record<string, (args: any, extra?: { signal?: AbortSignal }) => Promise<any>> = {};
    const configs: Record<string, any> = {};
    const setup = { discover: async (_target: unknown, _signal: unknown, saved?: GraphSetupDraft) => {
      reads++; discoveredDraft = saved; return { discovery, draft: discovery.resolution!.draft, restoredDraft };
    } };
    registerMcpGraphSetup({ registerTool: (name: string, schema: unknown, handler: any) => { handlers[name] = handler; configs[name] = schema; } } as never,
      opening, setup as unknown as GraphSetupHost);
    return { opening, handlers, configs, setup, get saves() { return saves; }, get reads() { return reads; }, get discoveredDraft() { return discoveredDraft; } };
  }
  test('reads source choices through the shared adapter without saving or running', async () => {
    const r = await rig(); const result = await r.handlers.get_graph_setup({});
    assert.equal(result.structuredContent.discovery.model, 'model:Net');
    assert.equal(r.reads, 1); assert.equal(r.saves, 0);
  });
  test('packages unresolved inputs and a fillable draft for an LLM without saving or running', async () => {
    const r = await rig(); const data = (await r.handlers.suggest_graph_inputs({})).structuredContent;
    assert.equal(data.model.file, 'model.py');
    assert.equal(data.model.cls, 'Net');
    assert.deepEqual(data.resolution.unresolved, ['inputs.x']);
    assert.equal(data.draft.profile.kind, 'model');
    assert.match(data.guidance, /Net\.forward/);
    assert.equal(r.reads, 1); assert.equal(r.saves, 0);
  });
  test('publishes the tensor grammar so a client never reads tensorViz source', async () => {
    const r = await rig(); const contract = (await r.handlers.suggest_graph_inputs({})).structuredContent.inputContract;
    assert.deepEqual(contract.dtypes, ['bool', 'uint8', 'int8', 'int16', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'float64']);
    assert.deepEqual(Object.keys(contract.tensor.generator).sort(), ['full', 'normal', 'randint', 'uniform', 'values']);
    assert.deepEqual(contract.tensor.generator.randint.fields, ['low', 'high', 'seed']);
    assert.equal(contract.tensor.fields.includes('generator'), true);
  });
  const savedDraft: GraphSetupDraft = { profileName: 'my_model', profile: { kind: 'model', model: 'model:Net',
    init: { args: [], kwargs: { width: 12 } }, inputs: { args: [{ kind: 'tensor', shape: [3, 12], dtype: 'float64',
      requires_grad: false, generator: { kind: 'full', value: 2 } }], kwargs: {} } } };
  for (const tool of ['get_graph_setup', 'suggest_graph_inputs']) {
    test(`${tool} rejects a late read after agent help updates the same opening draft`, async () => {
      const r = await rig();
      let release!: () => void;
      const wait = new Promise<void>(resolve => { release = resolve; });
      const discover = r.setup.discover;
      r.setup.discover = async (...args) => { await wait; return discover(...args); };
      const read = r.handlers[tool]({});
      await r.opening.action(r.opening.state!.id, 'help', target, { kind: 'agent_help', draft: savedDraft });
      release();
      assert.equal((await read).isError, true);
      assert.deepEqual(r.opening.state?.draft, savedDraft);
      assert.equal(r.saves, 0);
    });
    test(`${tool} exposes the pending form draft shared through agent help without saving`, async () => {
      const r = await rig();
      await r.opening.action(r.opening.state!.id, 'help', target, { kind: 'agent_help', draft: savedDraft });
      assert.deepEqual((await r.handlers[tool]({})).structuredContent.draft, savedDraft);
      assert.equal(r.saves, 0);
    });
    test(`${tool} retains the selected draft when source discovery proposes different inputs`, async () => {
      const r = await rig();
      await r.opening.action(r.opening.state!.id, 'configure', target, { kind: 'submit', draft: savedDraft });
      const saves = r.saves, data = (await r.handlers[tool]({})).structuredContent;
      assert.deepEqual(data.draft, savedDraft);
      assert.deepEqual(r.discoveredDraft, savedDraft, 'discovery must receive the selected profile for helper restoration');
      assert.equal(r.saves, saves, 'reading a suggestion must not save or execute it');
    });
    test(`${tool} restores saved configuration fields but preserves pending configuration edits`, async () => {
      const restored = { ...savedDraft, configObject: { index: 0, parameter: 'config', values: { width: 12 } } };
      const r = await rig(restored);
      await r.opening.action(r.opening.state!.id, 'configure', target, { kind: 'submit', draft: savedDraft });
      assert.deepEqual((await r.handlers[tool]({})).structuredContent.draft, restored);
      const pending = { ...restored, configObject: { ...restored.configObject, values: { width: 16 } } };
      await r.opening.action(r.opening.state!.id, 'edit-config', target, { kind: 'submit', draft: pending });
      assert.deepEqual((await r.handlers[tool]({})).structuredContent.draft, pending);
    });
  }
  test('stale setup cannot write files', async () => {
    const r = await rig(); const result = await r.handlers.configure_graph_setup({ openingId: 'old', sourceDigest: target.sourceDigest, draft: { profile: {} } });
    assert.equal(result.isError, true); assert.equal(r.saves, 0);
  });
  test('explicit configuration enters the same opening and returns grouped requirements', async () => {
    const r = await rig(), id = r.opening.state!.id;
    await r.handlers.configure_graph_setup({ openingId: id, sourceDigest: target.sourceDigest, draft: { profile: {} } });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(r.saves, 1); assert.equal(r.opening.state!.id, id);
    assert.deepEqual(r.opening.state!.requirements, ['inputs.x: Required.']);
  });

  async function environmentRig(overrides: Partial<GraphEnvironmentRecovery> = {}, supportsRecovery = true, pendingPermission?: () => CanvasAgentPermission | undefined) {
    let environment: GraphEnvironmentRecovery = { reason: 'selection_required', detail: 'Choose Python.', canCreate: true,
      candidates: [{ id: 'project', interpreter: '/workspace/.venv/bin/python', source: 'Project environment', status: 'libraries_present',
        modelChecked: true }, { id: 'unavailable', interpreter: '/missing/bin/python', source: 'Previously used', status: 'unavailable' }], ...overrides };
    const actions: GraphSetupAction[] = [];
    let release!: (value: boolean | GraphEnvironmentRecovery) => void;
    let freshDigest = target.sourceDigest, current = true;
    const dependencies: GraphOpeningDependencies = { current: () => current, environment: async () => environment,
      lookup: async () => ({ ok: false, kind: 'none' }), discover: async () => ({ discovery }),
      save: async () => assert.fail('Environment setup must not save model inputs.'),
      capture: async () => assert.fail('Environment setup must not invent a model invocation.'),
      recover: async () => undefined, parent: async () => {}, publish: () => {},
      ...(supportsRecovery ? { recoverEnvironment: async (_target: unknown, _environment: unknown, action: GraphSetupAction) => {
        actions.push(action);
        const result = await new Promise<boolean | GraphEnvironmentRecovery>(resolve => { release = resolve; });
        if (typeof result === 'object') environment = result;
        return result;
      } } : {}) };
    const opening = new GraphOpeningController(dependencies); await opening.open(target);
    const handlers: Record<string, (args: any, extra?: { signal?: AbortSignal }) => Promise<any>> = {};
    const setup = { discover: async () => ({ discovery: { ...discovery, sourceDigest: freshDigest } }) };
    registerMcpGraphSetup({ registerTool: (name: string, _schema: unknown, handler: any) => { handlers[name] = handler; } } as never, opening, setup, pendingPermission);
    const invoke = (args: Record<string, unknown>) => handlers.configure_graph_environment({ openingId: opening.state!.id,
      sourceDigest: target.sourceDigest, ...args });
    return { opening, handlers, setup, actions, invoke, release: (value: boolean | GraphEnvironmentRecovery) => release(value),
      staleSource: () => { freshDigest = 'b'.repeat(64); }, changedSource: () => { current = false; } };
  }

  test('busy setup reads remain immediate by default and accept only bounded integer long-poll delays', async () => {
    const r = await environmentRig(); await r.invoke({ action: 'create_environment', name: 'Net-env' });
    assert.equal((await r.handlers.get_graph_setup({})).structuredContent.opening.phase, 'resolving');
    assert.equal(r.actions.length, 1);
    r.release(false);
    const read = await rig(), delay = read.configs.get_graph_setup.inputSchema.waitMs;
    for (const valid of [undefined, 0, 1, 10000]) assert.equal(delay.safeParse(valid).success, true);
    for (const invalid of [-1, 1.5, 10001, '10000']) assert.equal(delay.safeParse(invalid).success, false);
  });

  test('Canvas setup reads expose an actual unpresented operation immediately during resolving', async () => {
    const permission: CanvasAgentPermission = { operation: 'python.packages.install', scope: 'isolated_environment', effects: ['install'],
      facts: { python: '/storage/net/bin/python', packages: [{ name: 'torch', version: '2.9.0' }] }, id: 'permission', requestId: 'turn', status: 'pending' };
    const r = await environmentRig({}, true, () => structuredClone(permission));
    const configured = await r.invoke({ action: 'create_environment', name: 'Net-env' });
    assert.deepEqual(configured.structuredContent.permission, permission);
    const controller = new AbortController(); let settled = false;
    const read = r.handlers.get_graph_setup({ waitMs: 10000 }, { signal: controller.signal }); void read.then(() => { settled = true; });
    await new Promise(resolve => setImmediate(resolve)); const immediate = settled; controller.abort();
    const result = await read; r.release(false);
    assert.equal(immediate, true, 'the provider must receive missing presentation facts before waiting for a user decision');
    assert.equal(result.isError, undefined); assert.equal(result.structuredContent.opening.phase, 'resolving');
    assert.deepEqual(result.structuredContent.permission, permission); assert.equal(r.actions.length, 1);
  });

  test('a presented Canvas permission continues normal long polling while the user reviews it', async () => {
    const permission: CanvasAgentPermission = { operation: 'python.environment.create', scope: 'isolated_environment', effects: ['create'],
      facts: { name: 'Net-env' }, id: 'permission', requestId: 'turn', status: 'pending',
      presentation: { title: 'Prepare Python for Net?', summary: 'Net needs its own model libraries.', allowLabel: 'Prepare Python', denyLabel: 'Skip setup' } };
    const r = await environmentRig({}, true, () => permission); await r.invoke({ action: 'create_environment', name: 'Net-env' });
    const controller = new AbortController(); let settled = false;
    const read = r.handlers.get_graph_setup({ waitMs: 10000 }, { signal: controller.signal }); void read.then(() => { settled = true; });
    await new Promise(resolve => setImmediate(resolve)); const waited = !settled; controller.abort();
    assert.equal((await read).isError, true); r.release(false);
    assert.equal(waited, true, 'already-presented requests must not cause rapid polling');
  });

  test('external setup registration preserves its payload while Canvas reports no pending request explicitly', async () => {
    const external = await environmentRig();
    const configured = await external.invoke({ action: 'create_environment', name: 'Net-env' });
    assert.equal(Object.hasOwn(configured.structuredContent, 'permission'), false);
    assert.equal(Object.hasOwn((await external.handlers.get_graph_setup({})).structuredContent, 'permission'), false);
    assert.equal(Object.hasOwn((await external.handlers.suggest_graph_inputs({})).structuredContent, 'permission'), false);
    external.release(false);
    const canvas = await environmentRig({}, true, () => undefined); await canvas.invoke({ action: 'create_environment', name: 'Net-env' });
    assert.equal((await canvas.handlers.get_graph_setup({})).structuredContent.permission, null); canvas.release(false);
  });

  test('long-poll returns the changed opening before its timeout without starting another operation', async () => {
    const r = await environmentRig(); await r.invoke({ action: 'create_environment', name: 'Net-env' });
    const previous = r.opening.state;
    let settled = false;
    const read = r.handlers.get_graph_setup({ waitMs: 1000 }); void read.then(() => { settled = true; });
    await new Promise(resolve => setTimeout(resolve, 20)); assert.equal(settled, false);
    r.release(false);
    const result = await read;
    assert.equal(result.isError, undefined);
    assert.notEqual(r.opening.state, previous);
    assert.equal(result.structuredContent.opening.phase, 'needs_setup');
    assert.equal(r.actions.length, 1, 'a progress read must not repeat environment creation');
  });

  test('long-poll times out with unchanged progress while host approval is still pending', async () => {
    const r = await environmentRig(); await r.invoke({ action: 'create_environment', name: 'Net-env' });
    const previous = r.opening.state, start = Date.now();
    const result = await r.handlers.get_graph_setup({ waitMs: 40 });
    assert.ok(Date.now() - start >= 30, 'the read should wait instead of rapidly polling the host');
    assert.equal(result.structuredContent.opening, previous);
    assert.equal(result.structuredContent.opening.phase, 'resolving');
    assert.equal(r.actions.length, 1); r.release(false);
  });

  test('cancelling a pending progress read ends the wait without cancelling or repeating the operation', async () => {
    const r = await environmentRig(); await r.invoke({ action: 'create_environment', name: 'Net-env' });
    const previous = r.opening.state, controller = new AbortController();
    const read = r.handlers.get_graph_setup({ waitMs: 10000 }, { signal: controller.signal });
    controller.abort();
    assert.equal((await read).isError, true);
    assert.equal(r.opening.state, previous);
    assert.equal(r.actions.length, 1); r.release(false);
  });

  test('an already cancelled setup read never performs source discovery or a setup action', async () => {
    const r = await rig(), controller = new AbortController(); controller.abort();
    assert.equal((await r.handlers.get_graph_setup({}, { signal: controller.signal })).isError, true);
    assert.equal(r.reads, 0); assert.equal(r.saves, 0);
  });

  test('setup discovery receives the MCP cancellation signal and cannot publish a late result', async () => {
    const r = await rig(), controller = new AbortController(), discover = r.setup.discover;
    let discoveredSignal: AbortSignal | undefined;
    r.setup.discover = async (...args) => {
      discoveredSignal = args[1] as AbortSignal;
      await new Promise<void>((_resolve, reject) => discoveredSignal!.addEventListener('abort', () => reject(new Error('Read cancelled.')), { once: true }));
      return discover(...args);
    };
    const read = r.handlers.get_graph_setup({}, { signal: controller.signal });
    assert.equal(discoveredSignal, controller.signal);
    controller.abort(); assert.equal((await read).isError, true);
    assert.equal(r.saves, 0);
  });

  test('the agent selects only the host-listed candidate and continues the same opening without saving inputs', async () => {
    const r = await environmentRig(), id = r.opening.state!.id;
    const result = await r.invoke({ action: 'select_environment', candidateId: 'project' });
    assert.equal(result.structuredContent.requested, true);
    assert.equal(result.structuredContent.opening.phase, 'resolving');
    assert.deepEqual(result.structuredContent.workflow.nextStep, { kind: 'wait', tool: 'get_graph_setup', waitMs: 10000 });
    assert.deepEqual(r.actions, [{ kind: 'select_environment', candidateId: 'project' }]);
    r.release({ ...r.opening.state!.environment!, reason: 'ready', selectedId: 'project' });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(r.opening.state!.id, id);
    assert.equal(r.opening.state!.phase, 'needs_setup');
    assert.equal(r.opening.state!.environment!.reason, 'ready');
    const ready = await r.handlers.get_graph_setup({});
    assert.equal(ready.structuredContent.workflow.stages.python, 'ready');
    assert.deepEqual(ready.structuredContent.workflow.nextStep, { kind: 'inputs', tool: 'suggest_graph_inputs' });
  });

  test('the agent requests one host-owned named environment and returns while native setup is pending', async () => {
    const r = await environmentRig();
    const result = await r.invoke({ action: 'create_environment', name: 'Net-env' });
    assert.equal(result.structuredContent.requested, true);
    assert.match(result.structuredContent.nextStep, /requested does not mean approved, installed or captured/);
    assert.deepEqual(r.actions, [{ kind: 'create_environment', name: 'Net-env' }]);
    const replay = await r.invoke({ action: 'create_environment', name: 'Net-env' });
    assert.equal(replay.isError, true);
    assert.equal(r.actions.length, 1, 'a polling agent cannot create a second environment during native setup');
    r.release(false);
    await new Promise(resolve => setImmediate(resolve));
  });

  test('dependency setup delegates declared repair to native review instead of granting installation approval', async () => {
    const r = await environmentRig({ reason: 'missing_libraries', selectedId: 'project', requirements: ['transformers==4.57.6'] });
    const result = await r.invoke({ action: 'install_dependencies' });
    assert.equal(result.structuredContent.requested, true);
    assert.deepEqual(r.actions, [{ kind: 'install_dependencies' }]);
    assert.equal(r.opening.state!.phase, 'resolving');
    r.release(false);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(r.opening.state!.phase, 'needs_setup', 'declining approval cannot produce a ready capture');
  });

  test('input setup cannot claim a save while host Python recovery is still pending', async () => {
    const r = await environmentRig();
    const before = r.opening.state;
    const result = await r.handlers.configure_graph_setup({ openingId: before!.id,
      sourceDigest: target.sourceDigest, draft: savedDraft });
    assert.equal(result.isError, true); assert.match(JSON.stringify(result), /Python setup is not ready/);
    assert.equal(r.opening.state, before); assert.deepEqual(r.actions, []);
  });

  test('a host-declared compatibility copy uses its existing reviewed dependency action', async () => {
    const r = await environmentRig({ reason: 'incompatible', selectedId: 'project', compatibilityPackages: ['numpy<2'] });
    assert.equal((await r.invoke({ action: 'install_dependencies' })).isError, undefined);
    assert.deepEqual(r.actions, [{ kind: 'install_dependencies' }]);
    r.release(false);
  });

  for (const args of [
    { action: 'select_environment', candidateId: '/arbitrary/python' },
    { action: 'select_environment', candidateId: 'unavailable' },
    { action: 'select_environment' },
    { action: 'select_environment', candidateId: 'project', name: 'unexpected' },
    { action: 'create_environment', name: '../project' },
    { action: 'create_environment', name: 'CON' },
    { action: 'create_environment', name: 'x'.repeat(49) },
    { action: 'create_environment', name: 'valid', candidateId: 'project' },
    { action: 'install_dependencies' },
  ]) test(`invalid environment action cannot reach native setup: ${JSON.stringify(args)}`, async () => {
    const r = await environmentRig();
    assert.equal((await r.invoke(args)).isError, true);
    assert.deepEqual(r.actions, []);
  });

  test('unsupported host creation and an install without a selected available Python are refused', async () => {
    const r = await environmentRig({ canCreate: undefined });
    assert.equal((await r.invoke({ action: 'create_environment', name: 'Net-env' })).isError, true);
    const unselected = await environmentRig({ reason: 'missing_libraries', requirements: ['numpy'] });
    assert.equal((await unselected.invoke({ action: 'install_dependencies' })).isError, true);
    assert.deepEqual(r.actions, []); assert.deepEqual(unselected.actions, []);
  });

  test('a controller refusal is reported rather than claiming that environment work started', async () => {
    const r = await environmentRig({}, false);
    const result = await r.invoke({ action: 'select_environment', candidateId: 'project' });
    assert.equal(result.isError, true);
    assert.deepEqual(r.actions, []);
  });

  test('stale opening IDs and freshly changed source cannot start setup', async () => {
    const r = await environmentRig();
    assert.equal((await r.invoke({ action: 'create_environment', name: 'Net-env', openingId: 'old' })).isError, true);
    r.staleSource();
    assert.equal((await r.invoke({ action: 'create_environment', name: 'Net-env' })).isError, true);
    assert.deepEqual(r.actions, []);
  });

  test('the controller still rejects a source change after discovery and stopped openings stay stopped', async () => {
    const r = await environmentRig(); r.changedSource();
    assert.equal((await r.invoke({ action: 'select_environment', candidateId: 'project' })).isError, true);
    assert.equal(r.opening.state!.phase, 'cancelled');
    assert.equal((await r.invoke({ action: 'create_environment', name: 'Net-env' })).isError, true);
    assert.deepEqual(r.actions, []);
  });

  test('a late environment source check cannot act on a replaced opening', async () => {
    const r = await environmentRig();
    let release!: () => void;
    r.setup.discover = async () => { await new Promise<void>(resolve => { release = resolve; }); return { discovery }; };
    const pending = r.invoke({ action: 'create_environment', name: 'Net-env' });
    await r.opening.open({ ...target, cls: 'Other' });
    release();
    assert.equal((await pending).isError, true);
    assert.deepEqual(r.actions, []);
  });

  test('failed or declined dependency work cannot be repeated through another agent request', async () => {
    const r = await environmentRig({ reason: 'missing_libraries', selectedId: 'project', requirements: ['numpy'] });
    Object.assign(r.opening.state!, { phase: 'failed', detail: 'Installation was declined.', failure: { kind: 'environment', code: 'INSTALL_DECLINED' } });
    assert.equal((await r.invoke({ action: 'install_dependencies' })).isError, true);
    assert.equal((await r.invoke({ action: 'create_environment', name: 'Another-env' })).isError, true);
    assert.equal((await r.invoke({ action: 'select_environment', candidateId: 'project' })).isError, true);
    assert.deepEqual(r.actions, []);
  });

  test('agent setup continues selection, reviewed dependencies, inputs and one capture on the same opening', async () => {
    let saved = false, captures = 0;
    let environment: GraphEnvironmentRecovery = { reason: 'selection_required', detail: 'Choose Python.', canCreate: true,
      candidates: [{ id: 'project', interpreter: '/workspace/.venv/bin/python', source: 'Project environment', status: 'libraries_present' }] };
    const actions: GraphSetupAction[] = [];
    const resolved = { ok: true, profile: 'model', run: {}, guard: { spec: { profile: savedDraft.profile } } } as unknown as ProfileLookup;
    const opening = new GraphOpeningController({ current: () => true, environment: async () => environment,
      recoverEnvironment: async (_target, _environment, action) => {
        actions.push(action);
        environment = action.kind === 'select_environment'
          ? { ...environment, selectedId: 'project', reason: 'missing_libraries', requirements: ['transformers==4.57.6'], detail: 'Review model libraries.' }
          : { ...environment, reason: 'ready', requirements: undefined, detail: 'Python is ready.' };
        return environment;
      }, lookup: async () => saved ? resolved : { ok: false, kind: 'none' }, discover: async () => ({ discovery }),
      save: async (_target, draft) => { assert.deepEqual(draft, savedDraft); saved = true; return {}; },
      capture: async captureTarget => { captures++; return { target: captureTarget, hasCapture: true,
        outcome: { status: 'passed', detail: 'Captured the selected inputs.' } }; },
      recover: async () => undefined, parent: async () => {}, publish: () => {} });
    await opening.open(target);
    const id = opening.state!.id;
    const handlers: Record<string, (args: any, extra?: { signal?: AbortSignal }) => Promise<any>> = {};
    registerMcpGraphSetup({ registerTool: (name: string, _schema: unknown, handler: any) => { handlers[name] = handler; } } as never,
      opening, { discover: async () => ({ discovery }) });
    const identity = { openingId: id, sourceDigest: target.sourceDigest };
    assert.equal((await handlers.configure_graph_environment({ ...identity, action: 'select_environment', candidateId: 'project' })).isError, undefined);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(opening.state!.environment!.reason, 'missing_libraries');
    assert.equal((await handlers.configure_graph_environment({ ...identity, action: 'install_dependencies' })).isError, undefined);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(opening.state!.environment!.reason, 'ready');
    assert.equal(opening.state!.phase, 'needs_setup');
    const setup = await handlers.get_graph_setup({});
    assert.equal(setup.structuredContent.opening.id, id);
    assert.deepEqual(setup.structuredContent.workflow.stages, { python: 'ready', inputs: 'needed', capture: 'not_started' });
    assert.equal(setup.structuredContent.workflow.nextStep.kind, 'inputs');
    const inputs = await handlers.suggest_graph_inputs({});
    assert.equal(inputs.structuredContent.workflow.nextStep.kind, 'inputs');
    assert.equal((await handlers.configure_graph_setup({ ...identity, draft: savedDraft })).isError, undefined);
    await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(actions, [{ kind: 'select_environment', candidateId: 'project' }, { kind: 'install_dependencies' }]);
    assert.equal(opening.state!.id, id);
    assert.equal(opening.state!.phase, 'ready');
    assert.equal(opening.state!.hasCapture, true);
    assert.equal(captures, 1, 'the same opening should capture exactly once after setup');
    const completed = await handlers.get_graph_setup({});
    assert.equal(completed.structuredContent.workflow.nextStep.kind, 'complete');
    assert.deepEqual(completed.structuredContent.workflow.stages, { python: 'ready', inputs: 'saved', capture: 'ready' });
  });
});
