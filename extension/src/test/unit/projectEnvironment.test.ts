import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { ENVIRONMENT_KEY, ENVIRONMENTS_KEY, savedEnvironment, rememberEnvironment, EnvironmentChoice, EnvironmentRunner, environmentExecutable,
  discoverEnvironments, resolveCaptureEnvironment, selectProjectEnvironment, selectedExecutionInterpreter, selectedExecutionEnvironment, inspectExecutionEnvironment,
  environmentScope, savedEnvironmentLabel } from '../../projectEnvironment';
import { ManagedRuntime } from '../../runtime';
import { graphMcpHelpRequest } from '@tensorviz/contracts/graphHelpRequest';
import { reset, state, setInfoResponder, infoMessages } from '../vscodeStub';

suite('project environment selection and consent', () => {
  let directory: string;
  let context: vscode.ExtensionContext;
  let memory: Map<string, unknown>;
  let machine: Map<string, unknown>;
  let calls: string[];
  const choice: EnvironmentChoice = { kind: 'project', interpreter: '/project/.venv/bin/python' };
  let runner: EnvironmentRunner;
  let processEnvironment: { virtual?: string; conda?: string };
  setup(() => {
    reset(); infoMessages.length = 0;
    state.workspaceFolders = [{ uri: vscode.Uri.file('/project') }];
    processEnvironment = { virtual: process.env.VIRTUAL_ENV, conda: process.env.CONDA_PREFIX };
    delete process.env.VIRTUAL_ENV; delete process.env.CONDA_PREFIX;
    directory = fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-env-unit-'));
    memory = new Map(); machine = new Map(); calls = [];
    context = folderContext(memory);
    runner = async (_python, _backend, args) => {
      calls.push(args[0]);
      return args[0] === 'inspect-capture' ? { ok: true, python: '3.12.10', torch: '2.5.1+cu121', pip: true, missing: [] }
        : { ok: true, torch: '2.5.1+cu121', cuda: '12.1' };
    };
  });
  teardown(() => {
    fs.rmSync(directory, { recursive: true, force: true });
    if (processEnvironment.virtual === undefined) delete process.env.VIRTUAL_ENV; else process.env.VIRTUAL_ENV = processEnvironment.virtual;
    if (processEnvironment.conda === undefined) delete process.env.CONDA_PREFIX; else process.env.CONDA_PREFIX = processEnvironment.conda;
  });

  const memento = (store: Map<string, unknown>) => ({
    get: (key: string) => store.get(key), update: async (key: string, value: unknown) => { store.set(key, value); },
  });
  /** One folder's context. `machine` is shared by every folder, as globalState is on a real host. */
  function folderContext(folder: Map<string, unknown>): vscode.ExtensionContext {
    return { globalStorageUri: vscode.Uri.file(directory), workspaceState: memento(folder), globalState: memento(machine),
    } as unknown as vscode.ExtensionContext;
  }
  const readyRuntime = (interpreter: string) => ({
    check: async () => ({ state: 'ready', interpreter }),
    provision: () => { throw new Error('must not recreate a ready environment'); },
  } as unknown as ManagedRuntime);

  test('nested project choices never replace or inherit workspace choices, including legacy choices', async () => {
    state.workspaceFolders = [{ uri: vscode.Uri.file(directory) }];
    const project = path.join(directory, 'dinov2');
    fs.mkdirSync(project); fs.writeFileSync(path.join(project, 'pyproject.toml'), '');
    const model = vscode.Uri.file(path.join(project, 'model.py'));
    const unrelated = vscode.Uri.file(path.join(directory, 'agent_test.py'));
    const workspaceChoice = { kind: 'project' as const, interpreter: '/workspace/python' };
    memory.set(ENVIRONMENT_KEY, workspaceChoice);
    assert.deepEqual(savedEnvironment(context, unrelated), workspaceChoice);
    assert.equal(savedEnvironment(context, model), undefined);
    await rememberEnvironment(context, choice, model);
    assert.equal(environmentScope(model), project);
    assert.deepEqual(savedEnvironment(context, model), choice);
    assert.deepEqual(savedEnvironment(context, unrelated), workspaceChoice);
    assert.equal(await selectedExecutionInterpreter(context, model, async () => undefined), choice.interpreter);
    assert.equal(await selectedExecutionInterpreter(context, unrelated, async () => undefined), workspaceChoice.interpreter);
    assert.equal(savedEnvironmentLabel(model), `Saved for project: ${project}`);
    const candidates = await discoverEnvironments(context, '/backend', { resource: model, signal: new AbortController().signal, runner });
    assert.equal(candidates.find(candidate => candidate.selected)?.source, savedEnvironmentLabel(model));
    state.workspaceFolders = [{ uri: vscode.Uri.file(project) }];
    assert.deepEqual(savedEnvironment(context, model), choice, 'opening the nested folder preserves the same project key');
  });
  test('saved environment labels keep the project visible within the candidate wire limit', () => {
    const project = `/${'parent/'.repeat(50)}model_project`;
    state.workspaceFolders = [{ uri: vscode.Uri.file(project) }];
    const label = savedEnvironmentLabel(vscode.Uri.file(path.join(project, 'model.py')));
    assert.ok(label.length <= 256);
    assert.ok(label.endsWith('/model_project'));
  });

  test('Browse saves a chosen Python before any worker check or dependency review', async () => {
    memory.set(ENVIRONMENT_KEY, { kind: 'managed' });
    const result = await selectProjectEnvironment(context, '/backend', { picker: async () => choice,
      runner: async () => { throw new Error('Browse must not probe, plan or install the chosen Python'); } });
    assert.equal(result.ok, true); assert.deepEqual(savedEnvironment(context), choice);
    assert.equal(state.warningMessages.length, 0); assert.equal(infoMessages.length, 0);
    assert.ok(result.ok && /Python selected/.test(result.banner));
    assert.deepEqual(calls, []);
  });
  test('Browse accepts an entered venv path even with missing capture libraries', async () => {
    memory.set(ENVIRONMENT_KEY, choice);
    const executable = path.join(directory, 'new', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
    fs.mkdirSync(path.dirname(executable), { recursive: true }); fs.writeFileSync(executable, '');
    state.quickPickAnswers = ['Enter Python or environment path…']; state.inputBoxAnswers = [path.dirname(path.dirname(executable))];
    const result = await selectProjectEnvironment(context, '/backend', { runner: async (_python, _backend, args) => {
      calls.push(args[0]); assert.equal(args[0], 'inspect-capture', 'the picker may read inventory only');
      return { ok: true, python: '3.12.2', missing: ['torch', 'pydantic'] };
    } });
    assert.equal(result.ok, true); assert.deepEqual(savedEnvironment(context), { kind: 'project', interpreter: executable, explicit: true });
    assert.deepEqual(calls, ['inspect-capture']);
    assert.equal(state.warningMessages.length, 0); assert.equal(infoMessages.length, 0);
    assert.deepEqual(fs.readdirSync(directory), ['new'], 'selection creates no installation plan');
  });
  test('Browse can select an unprepared managed fallback without provisioning it', async () => {
    state.quickPickAnswers = ['Use a separate tensorViz environment'];
    const result = await selectProjectEnvironment(context, '/backend', { runner, runtime: {
      check: async () => ({ state: 'missing' }), provision: () => { throw new Error('Browse must not download a runtime'); },
    } as unknown as ManagedRuntime });
    assert.equal(result.ok, true); assert.deepEqual(savedEnvironment(context), { kind: 'managed', explicit: true });
    assert.equal(state.warningMessages.length, 0); assert.equal(infoMessages.length, 0); assert.deepEqual(calls, []);
  });
  test('cancelling Browse after picking preserves the previous selection', async () => {
    memory.set(ENVIRONMENT_KEY, { kind: 'managed' }); let cancelled = false;
    const result = await selectProjectEnvironment(context, '/backend', { isCancelled: () => cancelled,
      picker: async () => { cancelled = true; return choice; } });
    assert.ok(!result.ok && result.cancelled); assert.deepEqual(savedEnvironment(context), { kind: 'managed' });
    assert.deepEqual(calls, []);
  });
  test('Browse selects only the model-owning folder and refuses untrusted workspaces', async () => {
    const resource = vscode.Uri.file('/other/model.py'); state.workspaceFolders!.push({ uri: vscode.Uri.file('/other') });
    memory.set(ENVIRONMENTS_KEY, { '/project': { kind: 'managed' } });
    await selectProjectEnvironment(context, '/backend', { resource, picker: async () => choice });
    assert.deepEqual(savedEnvironment(context, resource), choice); assert.deepEqual(savedEnvironment(context), { kind: 'managed' });
    state.isTrusted = false;
    const result = await selectProjectEnvironment(context, '/backend', { picker: async () => { throw new Error('must not show picker'); } });
    assert.equal(result.ok, false); assert.deepEqual(calls, []);
  });

  test('execution honors a saved managed choice even when a project Python exists', async () => {
    memory.set(ENVIRONMENT_KEY, { kind: 'managed' });
    let projectLookups = 0;
    state.extensions.set('ms-python.python', { activate: async () => {
      projectLookups += 1;
      return { environments: { getActiveEnvironmentPath: () => ({ path: '/project/python' }),
        resolveEnvironment: async () => ({ executable: { uri: vscode.Uri.file('/project/python') } }) } };
    } });
    assert.equal(await selectedExecutionInterpreter(context, vscode.Uri.file('/project/model.py'),
      async () => '/managed/python'), '/managed/python');
    assert.equal(await selectedExecutionInterpreter(context, vscode.Uri.file('/project/model.py'),
      async () => undefined), undefined, 'an unavailable selected runtime must not switch environments');
    assert.equal(projectLookups, 0);
    assert.equal(infoMessages.length, 0);
  });

  test('execution preserves a selected project path without probing a fallback', async () => {
    memory.set(ENVIRONMENT_KEY, choice);
    machine.set(ENVIRONMENT_KEY, { kind: 'managed' });
    assert.equal(await selectedExecutionInterpreter(context, vscode.Uri.file('/project/model.py'),
      async () => { throw new Error('must not check fallback'); }), choice.interpreter);
  });

  test('environment inspection reports the same managed or project choice capture resolves', async () => {
    const resource = vscode.Uri.file('/project/model.py'), managed = async () => '/managed/python';
    memory.set(ENVIRONMENT_KEY, { kind: 'managed' });
    assert.deepEqual(await selectedExecutionEnvironment(context, resource, managed), { interpreter: '/managed/python', source: 'managed' });
    memory.set(ENVIRONMENT_KEY, choice);
    assert.deepEqual(await selectedExecutionEnvironment(context, resource, managed), { interpreter: choice.interpreter, source: 'project' });
    assert.equal(await selectedExecutionInterpreter(context, resource, managed), choice.interpreter);
    assert.equal(infoMessages.length, 0, 'Inspection must not ask the user to choose again');
  });

  test('inspection distinguishes an available managed fallback from an explicitly selected one', async () => {
    const resource = vscode.Uri.file('/project/model.py');
    const options = { resource, signal: new AbortController().signal, readyManaged: async () => '/managed/python', runner };
    const unresolved = await inspectExecutionEnvironment(context, '/backend', options);
    assert.equal(unresolved.resolution, 'selection_required'); assert.equal(unresolved.ok, false);
    assert.equal(unresolved.environment, undefined, 'an available fallback is not yet the capture/install target');
    assert.equal(unresolved.candidates?.[0].managed, true);
    assert.deepEqual(unresolved.nextAction, { tool: 'open_graph', arguments: { file: resource.fsPath } });
    assert.deepEqual(calls, ['inspect-capture']); assert.equal(memory.size, 0);
    memory.set(ENVIRONMENT_KEY, { kind: 'managed' }); calls.length = 0;
    const selected = await inspectExecutionEnvironment(context, '/backend', options);
    assert.equal(selected.resolution, 'selected'); assert.equal(selected.environment?.interpreter, '/managed/python');
    assert.deepEqual(calls, ['inspect-runtime']); assert.equal(infoMessages.length, 0);
  });

  test('inspection preserves broken explicit choices and never probes another environment', async () => {
    memory.set(ENVIRONMENT_KEY, choice);
    const inspected: string[] = [];
    const result = await inspectExecutionEnvironment(context, '/backend', { resource: vscode.Uri.file('/project/model.py'),
      signal: new AbortController().signal, readyManaged: async () => { throw new Error('must not substitute managed'); },
      runner: async python => { inspected.push(python); return { ok: false, message: 'missing interpreter' }; } });
    assert.equal(result.ok, false); assert.equal(result.resolution, 'selected');
    assert.equal(result.environment?.interpreter, choice.interpreter); assert.deepEqual(inspected, [choice.interpreter]);
    memory.clear(); state.configuration.set('tensorviz.pythonPath', path.join(directory, 'missing-python'));
    const configured = await inspectExecutionEnvironment(context, '/backend', { resource: vscode.Uri.file('/project/model.py'),
      signal: new AbortController().signal, readyManaged: async () => { throw new Error('must not substitute managed'); }, runner });
    assert.equal(configured.ok, false); assert.equal(configured.resolution, 'selected');
    assert.equal(configured.environment?.source, 'setting'); assert.deepEqual(calls, []);
  });

  test('inspection uses the owning folder choice and refuses untrusted or cancelled reads before probing', async () => {
    const resource = vscode.Uri.file('/other/model.py'); state.workspaceFolders!.push({ uri: vscode.Uri.file('/other') });
    memory.set(ENVIRONMENTS_KEY, { '/project': { kind: 'managed' }, '/other': choice });
    const controller = new AbortController();
    const options = { resource, signal: controller.signal, readyManaged: async () => { throw new Error('wrong folder'); }, runner };
    const selected = await inspectExecutionEnvironment(context, '/backend', options);
    assert.equal(selected.environment?.interpreter, choice.interpreter); calls.length = 0;
    state.isTrusted = false;
    assert.equal((await inspectExecutionEnvironment(context, '/backend', options)).ok, false);
    controller.abort();
    await assert.rejects(inspectExecutionEnvironment(context, '/backend', options), /abort/i);
    assert.deepEqual(calls, []);
  });

  test('explicit unavailable Python never becomes an automatic managed run', async () => {
    state.configuration.set('tensorviz.pythonPath', path.join(directory, 'missing-python'));
    assert.equal(await selectedExecutionInterpreter(context, vscode.Uri.file('/project/model.py'),
      async () => { throw new Error('must not fall back past an explicit setting'); }), undefined);
  });

  test('unselected execution uses the configured Python before any managed lookup', async () => {
    const executable = path.join(directory, 'python');
    fs.writeFileSync(executable, '');
    state.configuration.set('tensorviz.pythonPath', executable);
    assert.equal(await selectedExecutionInterpreter(context, vscode.Uri.file('/project/model.py'),
      async () => { throw new Error('must not check fallback'); }), executable);
  });

  for (const suggestion of ['python-extension', 'managed']) test(`unselected execution and agent help do not promote a ${suggestion} suggestion`, async () => {
    const resource = vscode.Uri.file('/project/model.py');
    let managedChecks = 0;
    if (suggestion === 'python-extension') state.extensions.set('ms-python.python', { activate: async () => ({ environments: {
      getActiveEnvironmentPath: (actual: vscode.Uri) => { assert.equal(actual, resource); return { path: '/python-api' }; },
      resolveEnvironment: async () => ({ executable: { uri: vscode.Uri.file('/python-api/bin/python') } }),
    } }) });
    const readyManaged = async () => { managedChecks++; return '/managed/python'; };
    const environment = await selectedExecutionEnvironment(context, resource, readyManaged);
    assert.deepEqual(environment, { source: 'selection_required' }, 'inspection must not report a suggestion as the capture target');
    assert.equal(await selectedExecutionInterpreter(context, resource, readyManaged), undefined);
    const help = graphMcpHelpRequest({ interpreter: environment.interpreter, environmentSource: environment.source,
      opening: { id: 'unselected', target: { workspaceFolder: '/project', file: 'model.py', cls: 'Net',
        documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 0 },
      phase: 'needs_setup', hasCapture: false, detail: 'Choose Python.',
      environment: { reason: 'selection_required', candidates: [], detail: 'Choose Python.' } } });
    assert.match(help, /selection_required/);
    assert.doesNotMatch(help, /reportedEnvironment|selectedEnvironment|\/python-api|\/managed\/python/);
    assert.equal(managedChecks, 0, 'unselected reads must not check or provision the managed runtime');
    assert.equal(infoMessages.length, 0, 'execution and inspection must not open a chooser');
    assert.equal(savedEnvironment(context, resource), undefined);
  });

  test('compatible selected environment never checks/downloads the managed runtime', async () => {
    const runtime = { check: () => { throw new Error('must not access managed runtime'); } } as unknown as ManagedRuntime;
    const result = await resolveCaptureEnvironment(context, '/backend', { runner, runtime, picker: async () => choice });
    assert.ok(result.ok);
    assert.equal(result.interpreter, choice.interpreter);
    assert.match(result.banner, /Torch 2.5.1\+cu121, CUDA 12.1/);
    assert.deepEqual(calls, ['inspect-capture', 'probe-capture']);
    assert.deepEqual(savedEnvironment(context), choice);
    assert.equal(await selectedExecutionInterpreter(context, vscode.Uri.file('/project/model.py'), async () => undefined), choice.interpreter);
  });

  test('remembered environment is rechecked without asking again', async () => {
    memory.set(ENVIRONMENT_KEY, choice);
    assert.ok((await resolveCaptureEnvironment(context, '/backend', { runner,
      picker: async () => { throw new Error('unexpected picker'); } })).ok);
    assert.deepEqual(calls, ['inspect-capture', 'probe-capture']);
  });

  test('remembered managed environment is reused without a picker or fresh install', async () => {
    memory.set(ENVIRONMENT_KEY, { kind: 'managed' });
    const interpreter = '/tensorviz/runtime/venv/bin/python';
    const runtime = {
      check: async () => ({ state: 'ready', interpreter }),
      provision: () => { throw new Error('must not recreate a ready environment'); },
    } as unknown as ManagedRuntime;
    const result = await resolveCaptureEnvironment(context, '/backend', { runtime,
      picker: async () => { throw new Error('must reuse the saved choice'); }, runner });
    assert.ok(result.ok);
    assert.equal(result.interpreter, interpreter);
    assert.deepEqual(calls, []);
    assert.equal(infoMessages.length, 0);
    assert.deepEqual(savedEnvironment(context), { kind: 'managed' });
  });

  test('a global managed default cannot hide a new folder’s project environment', async () => {
    machine.set(ENVIRONMENT_KEY, { kind: 'managed' });
    let asked = 0;
    const result = await resolveCaptureEnvironment(context, '/backend', { runner,
      picker: async () => { asked += 1; return choice; } });
    assert.ok(result.ok);
    assert.equal(result.interpreter, choice.interpreter);
    assert.equal(asked, 1);
  });

  test('a folder that chose its own interpreter keeps it over the managed default', async () => {
    machine.set(ENVIRONMENT_KEY, { kind: 'managed' });
    memory.set(ENVIRONMENT_KEY, choice);
    const runtime = { check: () => { throw new Error('the folder chose a project interpreter'); } } as unknown as ManagedRuntime;
    const result = await resolveCaptureEnvironment(context, '/backend', { runner, runtime,
      picker: async () => { throw new Error('unexpected picker'); } });
    assert.ok(result.ok);
    assert.equal(result.interpreter, choice.interpreter);
    assert.equal(await selectedExecutionInterpreter(context, vscode.Uri.file('/project/model.py'), async () => undefined), choice.interpreter);
  });

  test('choosing the managed runtime remembers it only for this folder', async () => {
    const interpreter = '/tensorviz/runtime/venv/bin/python';
    const result = await resolveCaptureEnvironment(context, '/backend', { runtime: readyRuntime(interpreter), runner,
      picker: async () => ({ kind: 'managed' }) });
    assert.ok(result.ok);
    assert.deepEqual(savedEnvironment(context), { kind: 'managed' });
    assert.equal(machine.get(ENVIRONMENT_KEY), undefined);
  });

  test('a project interpreter is remembered for this folder only', async () => {
    // Usually a venv inside this repository: another folder must be asked, not silently traced
    // with this project's Torch.
    assert.ok((await resolveCaptureEnvironment(context, '/backend', { runner, picker: async () => choice })).ok);
    assert.deepEqual(savedEnvironment(context), choice);
    assert.equal(machine.get(ENVIRONMENT_KEY), undefined);
    let asked = 0;
    const other = await resolveCaptureEnvironment(folderContext(new Map()), '/backend', { runner,
      picker: async () => { asked += 1; return undefined; } });
    assert.ok(!other.ok && 'cancelled' in other);
    assert.equal(asked, 1);
  });

  test('Select environment asks again despite the machine default and keeps that default', async () => {
    machine.set(ENVIRONMENT_KEY, { kind: 'managed' });
    let asked = 0;
    const result = await resolveCaptureEnvironment(context, '/backend', { forcePicker: true, runner,
      picker: async () => { asked += 1; return choice; } });
    assert.ok(result.ok);
    assert.equal(asked, 1);
    assert.deepEqual(savedEnvironment(context), choice);
    assert.deepEqual(machine.get(ENVIRONMENT_KEY), { kind: 'managed' });
  });

  test('Select environment replaces this folder choice without creating a global default', async () => {
    memory.set(ENVIRONMENT_KEY, choice);
    let asked = 0;
    const interpreter = '/tensorviz/runtime/venv/bin/python';
    const result = await resolveCaptureEnvironment(context, '/backend', { forcePicker: true, runtime: readyRuntime(interpreter), runner,
      picker: async () => { asked += 1; return { kind: 'managed' }; } });
    assert.ok(result.ok);
    assert.equal(asked, 1);
    assert.deepEqual(savedEnvironment(context), { kind: 'managed' });
    assert.equal(machine.get(ENVIRONMENT_KEY), undefined);
  });

  test('dismissing selection runs and installs nothing', async () => {
    const result = await resolveCaptureEnvironment(context, '/backend', { runner, picker: async () => undefined });
    assert.ok(!result.ok && 'cancelled' in result);
    assert.deepEqual(calls, []);
    assert.equal(memory.size, 0);
  });

  test('untrusted workspace never executes a selected Python', async () => {
    state.isTrusted = false;
    const result = await resolveCaptureEnvironment(context, '/backend', { runner, picker: async () => choice });
    assert.ok(!result.ok);
    assert.deepEqual(calls, []);
  });

  function missingRunner(): EnvironmentRunner {
    let installed = false;
    return async (_python, _backend, args) => {
      calls.push(args[0]);
      if (args[0] === 'inspect-capture') return { ok: true, python: '3.12', torch: '2.5.1+cu121', pip: true, missing: ['pydantic'] };
      if (args[0] === 'probe-capture') return installed ? { ok: true, torch: '2.5.1+cu121', cuda: '12.1' }
        : { ok: false, message: 'Missing Python module: pydantic.' };
      if (args[0] === 'plan-capture') return { ok: true, digest: 'approved-plan', packages: [{ name: 'pydantic', version: '2.13.0' }] };
      assert.equal(args[0], 'install');
      assert.equal(args[2], 'approved-plan');
      installed = true;
      return { ok: true };
    };
  }

  test('review and install are separate; dismissing the install leaves packages untouched', async () => {
    state.warningAnswers = ['Review missing libraries'];
    const result = await resolveCaptureEnvironment(context, '/backend', { runner: missingRunner(), picker: async () => choice });
    assert.ok(!result.ok && 'cancelled' in result);
    assert.deepEqual(calls, ['inspect-capture', 'probe-capture', 'plan-capture']);
    assert.equal(memory.size, 0);
    assert.deepEqual(fs.readdirSync(directory), []);
  });

  test('explicit consent installs exactly the plan, reprobes, then remembers the environment', async () => {
    state.warningAnswers = ['Review missing libraries'];
    setInfoResponder((_message, actions) => actions.includes('Install missing libraries') ? 'Install missing libraries' : undefined);
    const result = await resolveCaptureEnvironment(context, '/backend', { runner: missingRunner(), picker: async () => choice });
    assert.ok(result.ok);
    assert.deepEqual(calls, ['inspect-capture', 'probe-capture', 'plan-capture', 'install', 'probe-capture']);
    assert.deepEqual(savedEnvironment(context), choice);
    assert.deepEqual(fs.readdirSync(directory), []);
  });

  test('conflict plan installs nothing and preserves the previous selection', async () => {
    memory.set(ENVIRONMENT_KEY, { kind: 'managed' });
    state.warningAnswers = ['Review missing libraries'];
    const delegate = missingRunner();
    const result = await resolveCaptureEnvironment(context, '/backend', { forcePicker: true, picker: async () => choice,
      runner: async (...args) => args[2][0] === 'plan-capture'
        ? { ok: false, message: 'Installed Torch would need to change.' } : delegate(...args) });
    assert.ok(!result.ok);
    assert.ok(!calls.includes('install'));
    assert.deepEqual(savedEnvironment(context), { kind: 'managed' });
  });

  test('a startup import alone cannot bless missing feature dependencies', async () => {
    const result = await resolveCaptureEnvironment(context, '/backend', { picker: async () => choice,
      runner: async (_python, _backend, args) => args[0] === 'inspect-capture'
        ? { ok: true, pip: true, missing: ['pydantic'] } : { ok: true, torch: '2.5.1' } });
    assert.ok(!result.ok);
    assert.equal(memory.size, 0);
    assert.equal(state.warningMessages.length, 1);
  });

  test('incompatible installed versions are reported without offering an upgrade', async () => {
    const result = await resolveCaptureEnvironment(context, '/backend', { picker: async () => choice,
      runner: async (_python, _backend, args) => {
        assert.ok(args[0] === 'inspect-capture' || args[0] === 'probe-capture');
        return { ok: true, pip: true, incompatible: ['pydantic 1.10 (needs >=2.12,<3)'] };
      } });
    assert.ok(!result.ok);
    assert.equal(memory.size, 0);
    assert.match(state.warningMessages[0], /incompatible/);
  });

  test('cancelling during a probe cannot remember or install the environment', async () => {
    let cancelled = false;
    const result = await resolveCaptureEnvironment(context, '/backend', { picker: async () => choice,
      isCancelled: () => cancelled, runner: async (...args) => { cancelled = true; return runner(...args); } });
    assert.ok(!result.ok && 'cancelled' in result);
    assert.ok(!calls.includes('install'));
    assert.equal(memory.size, 0);
  });

  test('Conda environments discovered by the Python extension can be selected directly', async () => {
    const executable = path.join(directory, process.platform === 'win32' ? 'python.exe' : 'bin/python');
    fs.mkdirSync(path.dirname(executable), { recursive: true }); fs.writeFileSync(executable, '');
    state.workspaceFolders = [{ uri: { fsPath: '/project' } }];
    state.extensions.set('ms-python.python', { activate: async () => ({ environments: {
      getActiveEnvironmentPath: () => ({ path: '', id: '' }), resolveEnvironment: async () => undefined,
      known: [{ path: directory }],
    } }) });
    state.quickPickAnswers = [executable];
    const result = await resolveCaptureEnvironment(context, '/backend', { runner });
    assert.ok(result.ok);
    assert.equal(result.interpreter, executable);
  });

  test('a pipless capture environment uses the service installer without requiring server libraries', async () => {
    state.warningAnswers.push('Review missing libraries'); setInfoResponder(() => 'Install missing libraries');
    const original = missingRunner();
    const result = await resolveCaptureEnvironment(context, '/backend', { managerPython: '/tensorviz/service/python',
      picker: async () => choice, runner: async (python, backend, args, ...rest) => {
        assert.equal(python, '/tensorviz/service/python');
        assert.deepEqual(args.slice(0, 2), ['--target-python', choice.interpreter]);
        const answer = await original(choice.interpreter, backend, args.slice(2), ...rest);
        return args[2] === 'inspect-capture' ? { ...answer, pip: false } : answer;
      } });
    assert.ok(result.ok);
    assert.deepEqual(calls, ['inspect-capture', 'probe-capture', 'plan-capture', 'install', 'probe-capture']);
    assert.ok(!infoMessages.join('\n').includes('Add pip'));
  });


  test('a legacy workspace choice migrates only to the first root and new selections remain isolated', async () => {
    state.workspaceFolders = [{ uri: vscode.Uri.file('/project') }, { uri: vscode.Uri.file('/other') }];
    memory.set(ENVIRONMENT_KEY, choice);
    const second = vscode.Uri.file('/other/model.py');
    assert.equal(savedEnvironment(context, second), undefined);
    const otherChoice: EnvironmentChoice = { kind: 'project', interpreter: '/other/.venv/bin/python' };
    await rememberEnvironment(context, otherChoice, second);
    assert.equal(memory.get(ENVIRONMENT_KEY), undefined);
    assert.deepEqual(savedEnvironment(context, vscode.Uri.file('/project/model.py')), choice);
    assert.deepEqual(savedEnvironment(context, second), otherChoice);
    assert.deepEqual(savedEnvironment(context, vscode.Uri.file('/other')), otherChoice,
      'profile resolution at the folder URI must use the same choice as model execution');
    assert.equal(savedEnvironment(context, vscode.Uri.file('/outside/model.py')), undefined);
    assert.deepEqual(Object.keys(memory.get(ENVIRONMENTS_KEY) as object), ['/project', '/other']);
  });

  test('a second-root opening rechecks and remembers its own interpreter', async () => {
    state.workspaceFolders = [{ uri: vscode.Uri.file('/project') }, { uri: vscode.Uri.file('/other') }];
    memory.set(ENVIRONMENT_KEY, choice);
    const resource = vscode.Uri.file('/other/model.py');
    const otherChoice: EnvironmentChoice = { kind: 'project', interpreter: '/other/python' };
    const result = await resolveCaptureEnvironment(context, '/backend', { resource, runner,
      picker: async () => otherChoice });
    assert.ok(result.ok);
    assert.equal(result.interpreter, otherChoice.interpreter);
    assert.equal(await selectedExecutionInterpreter(context, resource, async () => undefined), otherChoice.interpreter);
    assert.equal(await selectedExecutionInterpreter(context, vscode.Uri.file('/project/model.py'), async () => undefined), choice.interpreter);
  });

  test('the picker asks the Python extension about the actual model resource', async () => {
    const executable = path.join(directory, 'python'); fs.writeFileSync(executable, '');
    const resource = vscode.Uri.file('/project/nested/model.py');
    state.extensions.set('ms-python.python', { activate: async () => ({ environments: {
      getActiveEnvironmentPath: (actual: vscode.Uri) => { assert.equal(actual, resource); return { path: directory }; },
      resolveEnvironment: async () => ({ executable: { uri: vscode.Uri.file(executable) } }),
    } }) });
    state.quickPickAnswers = [executable];
    const result = await resolveCaptureEnvironment(context, '/backend', { resource, runner });
    assert.ok(result.ok); assert.equal(result.interpreter, executable);
  });

  test('environment folders resolve to their platform executable', () => {
    const executable = path.join(directory, process.platform === 'win32' ? 'python.exe' : 'bin/python');
    fs.mkdirSync(path.dirname(executable), { recursive: true }); fs.writeFileSync(executable, '');
    assert.equal(environmentExecutable(directory), executable);
    assert.equal(environmentExecutable(executable), executable);
    assert.equal(environmentExecutable('python'), undefined);
  });

  test('discovery retains a missing saved choice and orders the owning folder before known alternatives', async () => {
    const root = path.join(directory, 'owner'), otherRoot = path.join(directory, 'other');
    const makePython = (folder: string) => {
      const executable = path.join(folder, process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
      fs.mkdirSync(path.dirname(executable), { recursive: true }); fs.writeFileSync(executable, ''); return executable;
    };
    const own = makePython(path.join(root, '.venv')), other = makePython(path.join(otherRoot, '.venv'));
    const known = makePython(path.join(directory, 'known'));
    const unavailable = path.join(directory, 'deleted', 'bin', 'python');
    state.workspaceFolders = [{ uri: vscode.Uri.file(root) }, { uri: vscode.Uri.file(otherRoot) }];
    const resource = vscode.Uri.file(path.join(root, 'nested', 'model.py'));
    memory.set(ENVIRONMENTS_KEY, { [root]: { kind: 'project', interpreter: unavailable } });
    state.extensions.set('ms-python.python', { activate: async () => ({ environments: {
      getActiveEnvironmentPath: (actual: vscode.Uri) => { assert.equal(actual, resource); return { path: root }; },
      resolveEnvironment: async () => ({ executable: { uri: vscode.Uri.file(own) } }),
      known: [{ path: path.dirname(path.dirname(known)) }],
    } }) });
    const probes: string[] = [];
    const rows = await discoverEnvironments(context, '/backend', { resource, managerPython: '/service/python',
      signal: new AbortController().signal, readyManaged: async () => '/managed/python',
      runner: async (python, backend, args) => {
        assert.equal(python, '/service/python'); assert.equal(backend, '/backend');
        assert.deepEqual(args, ['--target-python', args[1], 'inspect-capture']);
        probes.push(args[1]); return args[1] === unavailable ? { ok: false } : { ok: true, python: '3.12.13', torch: '2.6.0+cu124', missing: [] };
      } });
    assert.deepEqual(rows.map(row => row.interpreter), [unavailable, own, other, known, '/managed/python']);
    assert.equal(rows[0].selected, true); assert.equal(rows[0].status, 'unavailable');
    assert.equal(rows[1].pythonSelected, true); assert.equal(rows[1].status, 'libraries_present');
    assert.equal(rows[1].torch, '2.6.0+cu124');
    assert.equal(rows[4].managed, true); assert.equal(rows[4].selected, undefined);
    assert.ok(probes.length <= 16); assert.deepEqual(savedEnvironment(context, resource), { kind: 'project', interpreter: unavailable });
  });

  test('untrusted discovery never starts metadata Python or provisions a runtime', async () => {
    state.isTrusted = false;
    await assert.rejects(discoverEnvironments(context, '/backend', { signal: new AbortController().signal,
      runner: async () => { throw new Error('must not execute'); },
      readyManaged: async () => { throw new Error('must not access a runtime'); } }), /Trust/);
  });

  test('a nested model project offers its venv before the outer workspace environment', async () => {
    const project = path.join(directory, 'workspace', 'nested');
    const executable = (root: string) => path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
    const nested = executable(project), outer = executable(path.dirname(project));
    for (const file of [nested, outer]) { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, ''); }
    state.workspaceFolders = [{ uri: vscode.Uri.file(path.dirname(project)) }];
    const rows = await discoverEnvironments(context, '/backend', { resource: vscode.Uri.file(path.join(project, 'model.py')),
      workspaceRoot: project, signal: new AbortController().signal, runner });
    assert.deepEqual(rows.map(row => row.interpreter), [nested, outer]);
    assert.equal(rows[0].source, 'Model’s project folder'); assert.equal(rows[1].source, 'Owning workspace folder');
  });

  test('an explicitly selected managed environment remains first and suggestions do not replace it', async () => {
    memory.set(ENVIRONMENT_KEY, { kind: 'managed' });
    state.extensions.set('ms-python.python', { activate: async () => ({ environments: {
      getActiveEnvironmentPath: () => ({ path: '/project/python' }), resolveEnvironment: async () => undefined,
    } }) });
    const rows = await discoverEnvironments(context, '/backend', { signal: new AbortController().signal, runner,
      readyManaged: async () => '/managed/python' });
    assert.equal(rows[0].interpreter, '/managed/python'); assert.equal(rows[0].selected, true); assert.equal(rows[0].managed, true);
    assert.equal(rows[1].pythonSelected, true); assert.equal(rows[1].selected, undefined);
  });

  test('cancelled metadata collection does not display an environment picker', async () => {
    let cancelled = false;
    state.extensions.set('ms-python.python', { activate: async () => ({ environments: {
      getActiveEnvironmentPath: () => ({ path: choice.interpreter }), resolveEnvironment: async () => undefined,
    } }) });
    state.quickPickAnswers = ['Enter Python or environment path…'];
    const result = await resolveCaptureEnvironment(context, '/backend', { isCancelled: () => cancelled,
      runner: async () => { cancelled = true; return { ok: false }; } });
    assert.ok(!result.ok && 'cancelled' in result);
    assert.equal(state.quickPickAnswers.length, 1, 'the cancelled picker must not consume user interaction');
    assert.equal(memory.size, 0);
  });
});
