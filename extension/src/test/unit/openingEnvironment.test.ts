import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { checkOpeningEnvironment, recoverOpeningEnvironment, type OpeningEnvironmentOptions } from '../../openingEnvironment';
import { automaticProjectCandidate, ENVIRONMENTS_KEY, discoverEnvironments, environmentCost, rankEnvironmentCandidates, readyForModel, inspectExecutionEnvironment } from '../../projectEnvironment';
import type { GraphEnvironmentCandidate, ModelImportDiagnostic } from '@tensorviz/contracts/protocol';
import { reset, setInfoResponder, state } from '../vscodeStub';
import type { EnvironmentRunner } from '@tensorviz/graph-host/projectEnvironmentCore';
import { GraphOpeningError } from '@tensorviz/graph-host/graphOpening';
import { GraphToolError } from '@tensorviz/graph-host/mcpGraph';

const importMessage = "cannot import name 'ALL_ATTENTION_FUNCTIONS' from 'transformers.modeling_utils'";
const importDiagnostic: ModelImportDiagnostic = {
  exceptionType: 'ImportError', message: importMessage,
  traceback: `Traceback (most recent call last):\n  File "/project/model.py", line 4, in <module>\nImportError: ${importMessage}`,
  causes: [{ exceptionType: 'ImportError', message: importMessage, module: 'transformers.modeling_utils' }],
  runtime: { platform: 'darwin', architecture: 'arm64', python: '3.12.13', torch: '2.6.0', cuda: null,
    cudaAvailable: false, mpsAvailable: true }, truncated: false,
};

suite('native opening environment recovery', () => {
  let root: string, store: Map<string, unknown>, options: OpeningEnvironmentOptions, calls: string[][];
  let previous: { virtual?: string; conda?: string };
  setup(() => {
    reset(); previous = { virtual: process.env.VIRTUAL_ENV, conda: process.env.CONDA_PREFIX };
    delete process.env.VIRTUAL_ENV; delete process.env.CONDA_PREFIX;
    root = fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-opening-env-')); fs.mkdirSync(path.join(root, '.venv', 'bin'), { recursive: true });
    fs.writeFileSync(path.join(root, '.venv', 'bin', 'python'), 'fixture');
    store = new Map(); calls = []; state.workspaceFolders = [{ uri: vscode.Uri.file(root) }];
    const runner: EnvironmentRunner = async (python, _backend, args) => {
      assert.equal(python, '/service/python'); assert.equal(args[0], '--target-python'); calls.push(args);
      return args[2] === 'inspect-capture' || args[2] === 'inspect-candidate' ? { ok: true, python: '3.12.13', torch: '2.6.0', fingerprint: 'a'.repeat(64),
        environment: { executable: args[1], prefix: path.dirname(path.dirname(args[1])), python: '3.12.13', platform: 'darwin', architecture: 'arm64', torch: '2.6.0', pip: null },
        ...(args[2] === 'inspect-candidate' ? { model: { missingModules: [], incompatible: [] } } : {}) }
        : { ok: true };
    };
    options = { context: { workspaceState: { get: (key: string) => store.get(key), update: async (key: string, value: unknown) => { store.set(key, value); } },
      globalState: {}, globalStorageUri: vscode.Uri.file(root) } as unknown as vscode.ExtensionContext,
      target: { workspaceFolder: root, file: 'model.py', cls: 'Net', documentVersion: 1, sourceDigest: 'b'.repeat(64), backendRevision: 0 },
      backendDir: '/backend', managerPython: '/service/python', signal: new AbortController().signal,
      readyManaged: async () => undefined, workerCache: new Set(), runner };
  });
  teardown(() => {
    fs.rmSync(root, { recursive: true, force: true });
    if (previous.virtual === undefined) delete process.env.VIRTUAL_ENV; else process.env.VIRTUAL_ENV = previous.virtual;
    if (previous.conda === undefined) delete process.env.CONDA_PREFIX; else process.env.CONDA_PREFIX = previous.conda;
  });
  const choose = () => store.set(ENVIRONMENTS_KEY, { [root]: { kind: 'project', interpreter: path.join(root, '.venv', 'bin', 'python') } });
  const host = () => ({ current: async () => true, dependencies: async () => { throw new Error('selection must not install'); }, changed: () => {} });
  // Read-only probes: the static model check reads source and metadata, never imports the model or worker.
  const metadataOnly = ['inspect-capture', 'inspect-candidate'];

  test('automatic opening choices ignore other roots, global and ambiguous alternatives', () => {
    const row = (interpreter: string, source = 'Model’s project folder', flags = {}): GraphEnvironmentCandidate =>
      ({ id: interpreter, interpreter, source, status: 'libraries_present', ...flags });
    const local = row('/project/.venv/python');
    assert.equal(automaticProjectCandidate([local, row('/other/python', 'Other workspace folder: other')]), local);
    assert.equal(automaticProjectCandidate([row('/global/python', 'Known to the Python extension')]), undefined);
    assert.equal(automaticProjectCandidate([local, row('/project/venv/python')]), undefined);
    assert.equal(automaticProjectCandidate([row('/tool/python', 'tensorViz-managed fallback', { managed: true })]), undefined);
    const saved = row('/removed/python', 'Saved for this folder', { selected: true, status: 'unavailable' });
    const selected = row('/selected/python', 'Selected in VS Code', { pythonSelected: true });
    assert.equal(automaticProjectCandidate([saved, selected]), saved);
    assert.equal(automaticProjectCandidate([local, selected]), selected);
  });

  test('background discovery does not run a worker/model import or save a choice', async () => {
    const candidates = await discoverEnvironments(options.context, options.backendDir, { ...options, resource: vscode.Uri.file(root) });
    assert.equal(candidates.length, 1);
    assert.deepEqual(calls.map(args => args[2]), ['inspect-capture']); assert.equal(store.get(ENVIRONMENTS_KEY), undefined);
  });
  test('Create works without saved Python, selects the checked new venv for this folder and clears obsolete execution', async () => {
    const blocked = await checkOpeningEnvironment(options, { refresh: true });
    assert.equal(blocked.canCreate, true); let changed = 0; const base = options.runner!;
    options.runner = async (python, backend, args, ...rest) => {
      if (args[0] === 'create-environment') { fs.mkdirSync(args[1], { recursive: true }); return { ok: true }; }
      if (args[2] === 'inspect-model-setup') return { ok: true, sourcePackages: [], manifestDigest: 'declared' };
      if (args[2] === 'plan-model-environment') return { ok: true, digest: 'reviewed', packages: [{ name: 'torch', version: '2.10.0' }] };
      if (args[2] === 'install' || args[2] === 'probe-model') return { ok: true };
      return base(python, backend, args, ...rest);
    };
    setInfoResponder(() => 'Install and continue');
    const recovered = await recoverOpeningEnvironment(options, blocked, { kind: 'create_environment', name: 'my-model' }, { ...host(), changed: () => changed++ });
    assert.equal(typeof recovered === 'object' && recovered.reason, 'ready'); assert.equal(changed, 1);
    const remembered = (store.get(ENVIRONMENTS_KEY) as Record<string, { interpreter: string }>)[root];
    assert.ok(remembered.interpreter.startsWith(path.join(fs.realpathSync(root), 'model-environments') + path.sep));
    assert.notEqual(remembered.interpreter, '/service/python');
    assert.equal(path.basename(path.dirname(path.dirname(remembered.interpreter))), 'my-model');
  });
  test('agent environment consent is forwarded before creation and refusal cannot change the saved Python', async () => {
    const blocked = await checkOpeningEnvironment(options, { refresh: true });
    calls.length = 0;
    const requests: string[] = [];
    options.approve = async (request, signal) => {
      requests.push(request.operation); assert.equal(signal, options.signal);
      assert.deepEqual(request.effects, ['create']); assert.equal(request.scope, 'isolated_environment');
      assert.equal(request.facts.name, 'agent-model');
      assert.ok(!fs.existsSync(path.join(root, 'model-environments')));
      assert.deepEqual(calls, []); return false;
    };
    await assert.rejects(recoverOpeningEnvironment(options, blocked, { kind: 'create_environment', name: 'agent-model' }, host()), /not approved/);
    assert.deepEqual(requests, ['python.environment.create']); assert.equal(store.get(ENVIRONMENTS_KEY), undefined);
    assert.ok(!fs.existsSync(path.join(root, 'model-environments')));
    assert.ok(calls.every(args => ['inspect-capture', 'inspect-candidate', 'inspect-model'].includes(args[2])));
  });
  test('failed environment creation reaches graph opening with the exact import diagnostic and original error code', async () => {
    const blocked = await checkOpeningEnvironment(options, { refresh: true });
    const base = options.runner!;
    options.runner = async (python, backend, args, ...rest) => {
      if (args[0] === 'create-environment') { fs.mkdirSync(args[1], { recursive: true }); return { ok: true }; }
      if (args[2] === 'inspect-model-setup') return { ok: true, sourcePackages: [], manifestDigest: 'declared' };
      if (args[2] === 'plan-model-environment') return { ok: true, digest: 'reviewed', packages: [{ name: 'torch', version: '2.10.0' }] };
      if (args[2] === 'install') return { ok: true };
      if (args[2] === 'probe-model') return { ok: false, code: 'MODEL_IMPORT_FAILED',
        message: `Model import failed: ImportError: ${importMessage}`, diagnostic: importDiagnostic };
      return base(python, backend, args, ...rest);
    };
    let changed = 0;
    setInfoResponder(() => 'Install and continue');
    await assert.rejects(recoverOpeningEnvironment(options, blocked, { kind: 'create_environment', name: 'failed-model' },
      { ...host(), changed: () => changed++ }), error => {
      assert.ok(error instanceof GraphOpeningError);
      assert.equal(error.failure?.kind, 'environment');
      assert.equal(error.failure?.code, 'MODEL_IMPORT_FAILED');
      assert.equal(error.failure?.importFailure, true);
      assert.deepEqual(error.failure?.diagnostic, importDiagnostic);
      assert.match(error.message, /ALL_ATTENTION_FUNCTIONS/);
      assert.equal(error.environment?.reason, 'selection_required');
      return true;
    });
    assert.equal(changed, 0, 'a failed import cannot select the incomplete environment');
    assert.equal(store.get(ENVIRONMENTS_KEY), undefined);
    assert.deepEqual(fs.readdirSync(path.join(root, 'model-environments')), []);
  });
  test('dependency recovery preserves observed import evidence without guessing a GPU cause', async () => {
    choose(); const recovery = await checkOpeningEnvironment(options);
    await assert.rejects(recoverOpeningEnvironment(options, { ...recovery, reason: 'missing_libraries', requirements: ['transformers'] },
      { kind: 'install_dependencies' }, { ...host(), dependencies: async () => {
        throw new GraphToolError('MODEL_IMPORT_FAILED', `Model import failed: ImportError: ${importMessage}`, importDiagnostic);
      } }), error => {
      assert.ok(error instanceof GraphOpeningError);
      assert.deepEqual(error.failure, { kind: 'environment', code: 'MODEL_IMPORT_FAILED', importFailure: true, diagnostic: importDiagnostic });
      assert.equal(error.environment?.selectedId, recovery.selectedId);
      assert.match(error.message, /cannot import name/);
      assert.doesNotMatch(error.message, /requires CUDA|GPU required/);
      return true;
    });
  });
  test('explicit opening uses one compatible project Python and still verifies model and worker readiness', async () => {
    assert.equal((await checkOpeningEnvironment(options)).reason, 'ready');
    assert.deepEqual(store.get(ENVIRONMENTS_KEY), { [root]: { kind: 'project', interpreter: path.join(root, '.venv', 'bin', 'python') } });
    assert.deepEqual(calls.map(args => args[2]), ['inspect-candidate', 'inspect-capture', 'inspect-model', 'probe-capture']);
  });
  test('first inspection lists the local venv without selecting it; opening then establishes the inspected target', async () => {
    const resource = vscode.Uri.file(path.join(root, 'model.py')), expected = path.join(root, '.venv/bin/python');
    const base = options.runner!;
    const runner: EnvironmentRunner = async (python, backend, args, ...rest) => args[0] === 'inspect-runtime'
      ? { ok: true, environment: { executable: python, prefix: path.dirname(path.dirname(python)), python: '3.12.13',
        platform: 'darwin', architecture: 'arm64', torch: '2.6.0', pip: null } }
      : base(python, backend, args, ...rest);
    const inspect = () => inspectExecutionEnvironment(options.context, options.backendDir, { ...options, resource, runner });
    const before = await inspect();
    assert.equal(before.resolution, 'selection_required'); assert.equal(before.environment, undefined);
    assert.equal(before.candidates?.[0].interpreter, expected);
    assert.equal(store.get(ENVIRONMENTS_KEY), undefined, 'metadata reads must not silently choose Python');
    assert.deepEqual(calls.map(args => args[2]), ['inspect-capture']);
    assert.equal((await checkOpeningEnvironment(options)).reason, 'ready');
    const after = await inspect();
    assert.equal(after.resolution, 'selected'); assert.equal(after.environment?.interpreter, expected);
    assert.equal(after.environment?.source, 'project');
  });
  test('failed creation reports an environment failure without selecting or treating it as a successful repair', async () => {
    const blocked = await checkOpeningEnvironment(options, { refresh: true });
    options.runner = async () => { throw new GraphToolError('PACKAGE_UNAVAILABLE', 'helper==1 is unavailable for this Python.'); };
    await assert.rejects(recoverOpeningEnvironment(options, blocked, { kind: 'create_environment' }, host()),
      error => error instanceof GraphOpeningError && error.failure?.kind === 'environment' && /helper==1/.test(error.message) && !!error.environment);
    assert.equal(store.get(ENVIRONMENTS_KEY), undefined);
  });
  test('the model check starts while the capture check is still running, with the same result', async () => {
    choose();
    const sequential = await checkOpeningEnvironment(options);
    const sequentialCalls = calls.map(args => args[2]);
    calls.length = 0;
    options.workerCache.clear();
    const base = options.runner!;
    let release!: () => void;
    const gate = new Promise<void>(resolve => { release = resolve; });
    const order: string[] = [];
    options.runner = async (python, backend, args, ...rest) => {
      order.push(`start ${args[2]}`);
      if (args[2] === 'inspect-capture') await gate;
      const result = await base(python, backend, args, ...rest);
      order.push(`end ${args[2]}`);
      return result;
    };
    const pending = checkOpeningEnvironment(options);
    await new Promise(resolve => setTimeout(resolve, 20));
    assert.deepEqual(order, ['start inspect-capture', 'start inspect-model', 'end inspect-model'],
      'the model check ran before the capture check finished');
    release();
    const parallel = await pending;
    assert.deepEqual(parallel, sequential);
    assert.deepEqual(calls.map(args => args[2]).sort(), sequentialCalls.sort(), 'the same checks ran, each once');
  });
  test('a broken saved path cannot be replaced by automatic project resolution', async () => {
    store.set(ENVIRONMENTS_KEY, { [root]: { kind: 'project', interpreter: '/removed/python' } });
    const base = options.runner!;
    options.runner = async (python, backend, args, ...rest) => args[1] === '/removed/python'
      ? { ok: false, message: 'missing' } : base(python, backend, args, ...rest);
    const recovery = await checkOpeningEnvironment(options);
    assert.equal(recovery.reason, 'unavailable');
    assert.deepEqual(store.get(ENVIRONMENTS_KEY), { [root]: { kind: 'project', interpreter: '/removed/python' } });
    assert.ok(calls.every(args => metadataOnly.includes(args[2])));
  });
  test('capture library conflicts remain checked at model opening before running the worker', async () => {
    choose();
    const base = options.runner!;
    options.runner = async (python, backend, args, ...rest) => {
      assert.notEqual(args[2], 'probe-capture', 'an incompatible model environment must not run its worker');
      const result = await base(python, backend, args, ...rest);
      return args[2] === 'inspect-capture' ? { ...result, incompatible: ['pydantic 2.10.6 (needs <3,>=2.12)'] } : result;
    };
    const result = await checkOpeningEnvironment(options);
    assert.equal(result.reason, 'incompatible');
    assert.match(JSON.stringify(result), /pydantic 2\.10\.6/);
    assert.ok(calls.some(args => args[2] === 'inspect-capture'));
  });
  test('two local environments require a choice and never run a model or worker', async () => {
    fs.mkdirSync(path.join(root, 'venv', 'bin'), { recursive: true }); fs.writeFileSync(path.join(root, 'venv', 'bin', 'python'), 'fixture');
    options.runner = async (_python, _backend, args) => {
      calls.push(args); return { ok: true, python: '3.12.14', torch: '2.7.1' };
    };
    assert.equal((await checkOpeningEnvironment(options)).reason, 'selection_required');
    assert.equal(store.get(ENVIRONMENTS_KEY), undefined);
    assert.ok(calls.every(args => metadataOnly.includes(args[2])));
  });
  test('ranking puts environments needing the fewest changes first and never recommends unusable ones', () => {
    const row = (id: string, flags: Partial<GraphEnvironmentCandidate> = {}): GraphEnvironmentCandidate => ({ id, interpreter: `/${id}/python`,
      source: 'Known to the Python extension', status: 'libraries_present', torch: '2.7.1', modelChecked: true, ...flags });
    const ranked = rankEnvironmentCandidates([
      row('old-python', { status: 'incompatible', incompatible: ['Python 3.9.6 (needs 3.10 or newer)'] }),
      row('no-torch', { torch: null, status: 'missing_libraries', missing: ['torch'] }),
      row('pins', { modelIncompatible: ['transformers 4.57.6 (needs >=5)'] }),
      row('two-missing', { modelMissing: ['einops', 'timm'] }),
      row('managed-ready', { managed: true, source: 'tensorViz-managed fallback' }),
      row('ready'),
      row('gone', { status: 'unavailable', torch: undefined, modelChecked: undefined }),
      row('saved-ready', { selected: true, source: 'Saved for this folder' }),
    ]);
    assert.deepEqual(ranked.map(item => item.id), ['saved-ready', 'ready', 'managed-ready', 'two-missing', 'pins', 'old-python', 'no-torch', 'gone']);
    assert.deepEqual(['ready', 'two-missing', 'pins', 'no-torch', 'old-python'].map(id => environmentCost(ranked.find(item => item.id === id)!)), [0, 2, 3, undefined, undefined]);
    assert.equal(readyForModel(row('unchecked', { modelChecked: undefined })), false, 'an unchecked model is never assumed ready');
    assert.equal(environmentCost(row('overlap', { status: 'missing_libraries', missing: ['numpy'], modelMissing: ['numpy>=2', 'einops'] })), 2,
      'a library both checks name installs once');
  });

  // `.venv` and `venv` in the model's folder; `gaps` says what each still lacks for the model.
  const dotVenv = () => path.join(root, '.venv', 'bin', 'python'), venv = () => path.join(root, 'venv', 'bin', 'python');
  const transformers = { missing: [{ module: 'transformers', requirement: 'transformers>=4.48' }] };
  const twoEnvironments = (gaps: Record<string, { missing?: Array<{ module: string; requirement: string | null }>; incompatible?: string[] }>) => {
    fs.mkdirSync(path.join(root, 'venv', 'bin'), { recursive: true }); fs.writeFileSync(venv(), 'fixture');
    const base = options.runner!;
    options.runner = async (python, backend, args, ...rest) => {
      const result = await base(python, backend, args, ...rest), gap = gaps[args[1]];
      if (gap && args[2] === 'inspect-candidate') return { ...result, model: { missingModules: gap.missing ?? [], incompatible: gap.incompatible ?? [] } };
      if (gap && args[2] === 'inspect-model') return { ...result, missingModules: gap.missing ?? [], incompatible: gap.incompatible ?? [] };
      return result;
    };
  };

  test('an opening uses the environment that can already run the model, without asking', async () => {
    twoEnvironments({ [dotVenv()]: transformers });
    assert.equal((await checkOpeningEnvironment(options)).reason, 'ready');
    assert.deepEqual(store.get(ENVIRONMENTS_KEY), { [root]: { kind: 'project', interpreter: venv() } }, 'saved as tensorViz’s own choice');
    assert.ok(calls.every(args => metadataOnly.includes(args[2]) || args[1] === venv()), 'only the adopted environment is fully checked');
    assert.ok(calls.some(args => args[2] === 'probe-capture' && args[1] === venv()));
  });

  test('a choice tensorViz saved gives way once to a ready environment; the user’s own choice stays', async () => {
    twoEnvironments({ [dotVenv()]: transformers });
    store.set(ENVIRONMENTS_KEY, { [root]: { kind: 'project', interpreter: dotVenv() } });
    assert.equal((await checkOpeningEnvironment(options)).reason, 'ready');
    assert.deepEqual(store.get(ENVIRONMENTS_KEY), { [root]: { kind: 'project', interpreter: venv() } });

    store.set(ENVIRONMENTS_KEY, { [root]: { kind: 'project', interpreter: dotVenv(), explicit: true } });
    const kept = await checkOpeningEnvironment(options);
    assert.equal(kept.reason, 'missing_libraries');
    assert.deepEqual(store.get(ENVIRONMENTS_KEY), { [root]: { kind: 'project', interpreter: dotVenv(), explicit: true } });
    assert.equal(kept.recommendedId, undefined, 'the user’s choice stays preselected');
    assert.deepEqual(kept.candidates.map(item => [item.interpreter, item.modelChecked, item.modelMissing]),
      [[venv(), true, undefined], [dotVenv(), true, ['transformers>=4.48']]], 'the ready alternative is ranked first and shown as ready');
  });

  test('an observed import failure keeps the saved environment instead of switching', async () => {
    twoEnvironments({ [dotVenv()]: { incompatible: ['transformers 4.57.6 (needs >=5)'] } });
    store.set(ENVIRONMENTS_KEY, { [root]: { kind: 'project', interpreter: dotVenv() } });
    const failed = await checkOpeningEnvironment(options, { importFailure: true });
    assert.equal(failed.reason, 'incompatible');
    assert.deepEqual(store.get(ENVIRONMENTS_KEY), { [root]: { kind: 'project', interpreter: dotVenv() } });
    assert.equal(failed.recommendedId, failed.candidates.find(item => item.interpreter === venv())?.id);
  });

  test('with nothing ready, the opening recommends the environment needing the fewest changes', async () => {
    twoEnvironments({ [dotVenv()]: transformers, [venv()]: { incompatible: ['transformers 4.57.6 (needs >=5)'] } });
    const recovery = await checkOpeningEnvironment(options);
    assert.equal(recovery.reason, 'selection_required');
    assert.equal(store.get(ENVIRONMENTS_KEY), undefined, 'two local environments still need a choice');
    assert.deepEqual(recovery.candidates.map(item => item.interpreter), [dotVenv(), venv()], 'one addition ranks above one version change');
    assert.equal(recovery.recommendedId, recovery.candidates[0].id);
    assert.match(recovery.detail, /fewest changes/);
  });

  test('a multi-root opening saves only the model owner, never the first folder', async () => {
    const other = fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-other-root-'));
    try {
      state.workspaceFolders = [{ uri: vscode.Uri.file(other) }, { uri: vscode.Uri.file(root) }];
      assert.equal((await checkOpeningEnvironment(options)).reason, 'ready');
      assert.deepEqual(Object.keys(store.get(ENVIRONMENTS_KEY) as object), [root]);
    } finally { fs.rmSync(other, { recursive: true, force: true }); }
  });
  test('saved project selection probes capture rather than server libraries and caches only worker success', async () => {
    choose();
    assert.equal((await checkOpeningEnvironment(options)).reason, 'ready');
    assert.equal((await checkOpeningEnvironment(options)).reason, 'ready');
    assert.deepEqual(calls.map(args => args[2]), ['inspect-capture', 'inspect-model', 'probe-capture', 'inspect-capture', 'inspect-model']);
    assert.ok(calls.every(args => args[1] === path.join(root, '.venv', 'bin', 'python')));
  });
  test('selection rechecks candidate fingerprint and remembers the exact owning folder without resetting the opening', async () => {
    const recovery = await checkOpeningEnvironment(options, { refresh: true }); let changed = 0;
    const selected = await recoverOpeningEnvironment(options, recovery, { kind: 'select_environment', candidateId: recovery.candidates[0].id },
      { ...host(), changed: () => { changed++; } });
    assert.equal(typeof selected === 'object' && selected.reason, 'ready');
    assert.deepEqual(store.get(ENVIRONMENTS_KEY), { [root]: { kind: 'project', interpreter: path.join(root, '.venv', 'bin', 'python'), explicit: true } });
    assert.equal(changed, 1);
    assert.deepEqual(calls.map(args => args[2]), ['inspect-candidate', 'inspect-capture', 'inspect-capture', 'inspect-model', 'probe-capture']);
  });

  test('a selected empty environment offers one native review for capture and source libraries before any worker import', async () => {
    choose(); options.runner = async (_python, _backend, args) => {
      calls.push(args);
      return args[2] === 'inspect-capture' ? { ok: true, python: '3.12.13', missing: ['torch', 'pydantic>=2.12,<3'],
        environment: { executable: args[1], prefix: path.join(root, '.venv'), python: '3.12.13',
          platform: 'darwin', architecture: 'arm64', torch: null, pip: null } }
        : { ok: true, missingModules: [{ module: 'diffusers', requirement: 'diffusers==0.35.2' },
          { module: 'timm', requirement: 'timm' }] };
    };
    const recovery = await checkOpeningEnvironment(options);
    assert.equal(recovery.reason, 'missing_libraries');
    assert.deepEqual(recovery.requirements, ['torch', 'pydantic>=2.12,<3', 'diffusers==0.35.2', 'timm']);
    assert.ok(calls.every(args => [...metadataOnly, 'inspect-model'].includes(args[2])));
    assert.match(recovery.detail, /Review the exact PyPI wheel versions/);
  });
  test('changed candidate fingerprints and stale model consent cannot select or install', async () => {
    const recovery = await checkOpeningEnvironment(options, { refresh: true }); recovery.candidates[0].fingerprint = 'c'.repeat(64);
    const refused = await recoverOpeningEnvironment(options, recovery, { kind: 'select_environment', candidateId: recovery.candidates[0].id }, host());
    assert.equal(typeof refused === 'object' && refused.reason, 'selection_required'); assert.equal(store.get(ENVIRONMENTS_KEY), undefined);
    await assert.rejects(recoverOpeningEnvironment(options, recovery, { kind: 'select_environment', candidateId: recovery.candidates[0].id },
      { ...host(), current: async () => false }), /workspace or model changed/);
  });
  test('trying installed versions accepts only the offered pins for this interpreter, until it is selected again', async () => {
    choose(); const base = options.runner!;
    options.runner = async (python, backend, args, ...rest) => args[2] === 'inspect-model'
      ? { ok: true, incompatible: ['torch 2.7.1 (needs ==2.0.0)'], incompatibleRequirements: ['torch==2.0.0'] } : base(python, backend, args, ...rest);
    const blocked = await checkOpeningEnvironment(options);
    assert.equal(blocked.reason, 'incompatible'); assert.deepEqual(blocked.declaredPins, ['torch==2.0.0']);
    const tried = await recoverOpeningEnvironment(options, blocked, { kind: 'use_installed_versions' }, host());
    assert.equal(typeof tried === 'object' && tried.reason, 'ready');
    assert.equal((await checkOpeningEnvironment(options)).reason, 'ready', 'a later refresh keeps the choice');
    assert.equal(store.get(ENVIRONMENTS_KEY) && Object.keys(store.get(ENVIRONMENTS_KEY) as object).length, 1, 'no pin is written to settings');
    // Choosing the environment again starts a fresh check against the project pins.
    const selected = await recoverOpeningEnvironment(options, blocked, { kind: 'select_environment', candidateId: blocked.selectedId! }, host());
    assert.equal(typeof selected === 'object' && selected.reason, 'incompatible');
    assert.equal((await checkOpeningEnvironment(options)).reason, 'incompatible');
  });
  test('the saved project environment replaces its pinned versions after both answers; the managed one never does', async () => {
    choose(); const base = options.runner!, interpreter = path.join(root, '.venv', 'bin', 'python'); let torch = '2.7.1';
    options.runner = async (python, backend, args, ...rest) => {
      if (args[2] === 'inspect-model') return torch === '2.0.0' ? { ok: true }
        : { ok: true, incompatible: [`torch ${torch} (needs ==2.0.0)`], incompatibleRequirements: ['torch==2.0.0'] };
      if (args[2] === 'plan-replacement') {
        calls.push(args); assert.deepEqual(JSON.parse(args[4]), ['torch==2.0.0']); assert.equal(args[5], root);
        return { ok: true, digest: 'reviewed', replacesTorch: true, packages: [{ name: 'torch', version: '2.0.0', previous: '2.7.1' }] };
      }
      if (args[2] === 'install') { calls.push(args); torch = '2.0.0'; return { ok: true }; }
      return base(python, backend, args, ...rest);
    };
    const blocked = await checkOpeningEnvironment(options);
    assert.equal(blocked.replaceable, true);
    const progress = { run: vscode.window.withProgress, location: vscode.ProgressLocation };
    (vscode as any).ProgressLocation = { Notification: 15 };
    (vscode.window as any).withProgress = async (_options: unknown, task: Function) => task({},
      { isCancellationRequested: false, onCancellationRequested: () => ({ dispose() {} }) });
    try {
      const dependencies = async () => ({ python: interpreter, managerPython: '/service/python', backendDir: '/backend', storage: root,
        signal: options.signal, current: async () => true, runner: options.runner });
      state.warningAnswers = ['Replace versions', 'Replace PyTorch'];
      const replaced = await recoverOpeningEnvironment(options, blocked, { kind: 'replace_versions' }, { ...host(), dependencies });
      assert.equal(typeof replaced === 'object' && replaced.reason, 'ready');
      assert.deepEqual(state.warningMessages, ['Replace library versions in this environment?', 'Replace PyTorch in this environment?']);
      assert.deepEqual(calls.filter(args => ['plan-replacement', 'install'].includes(args[2])).map(args => args[2]), ['plan-replacement', 'install']);
    } finally { (vscode.window as any).withProgress = progress.run; (vscode as any).ProgressLocation = progress.location; }
    // The managed runtime is shared by every model: never offered, and refused if asked.
    torch = '2.7.1'; store.set(ENVIRONMENTS_KEY, { [root]: { kind: 'managed' } }); options.readyManaged = async () => interpreter;
    const managed = await checkOpeningEnvironment(options);
    assert.equal(managed.reason, 'incompatible'); assert.deepEqual(managed.declaredPins, ['torch==2.0.0']); assert.equal(managed.replaceable, undefined);
    assert.equal(await recoverOpeningEnvironment(options, { ...managed, replaceable: true }, { kind: 'replace_versions' }, host()), false);
  });
  test('refresh is metadata only even when a saved environment was previously ready', async () => {
    choose(); const recovery = await checkOpeningEnvironment(options); calls.length = 0;
    const refreshed = await recoverOpeningEnvironment(options, recovery, { kind: 'refresh_environments' }, host());
    assert.equal(typeof refreshed === 'object' && refreshed.reason, 'selection_required');
    assert.deepEqual(calls.map(args => args[2]), ['inspect-candidate']);
  });
  test('unexpected repair errors retain diagnostics and recheck instead of claiming capture or retrying', async () => {
    choose(); const recovery = await checkOpeningEnvironment(options), diagnostics: string[] = [];
    options.diagnostic = message => diagnostics.push(message);
    await assert.rejects(recoverOpeningEnvironment(options, { ...recovery, reason: 'missing_libraries', requirements: ['scipy'] },
      { kind: 'install_dependencies' }, { ...host(), dependencies: async () => { throw new Error('native review unavailable'); } }),
      error => error instanceof GraphOpeningError && error.failure?.code === 'DEPENDENCY_SETUP_FAILED'
        && error.environment?.reason === 'selection_required' && /Library setup failed/.test(error.message));
    assert.match(diagnostics[0], /native review unavailable/);
    assert.equal(diagnostics.length, 1);
  });
  test('library failures preserve the named package and structured recovery code', async () => {
    choose(); const recovery = await checkOpeningEnvironment(options);
    await assert.rejects(recoverOpeningEnvironment(options, { ...recovery, reason: 'missing_libraries', requirements: ['triton>=3.5.0'] },
      { kind: 'install_dependencies' }, { ...host(), dependencies: async () => { throw new GraphToolError('PACKAGE_UNAVAILABLE', 'triton>=3.5.0: No distribution for this platform.'); } }),
      error => error instanceof GraphOpeningError && error.failure?.code === 'PACKAGE_UNAVAILABLE'
        && /triton>=3.5.0/.test(error.message) && error.environment?.selectedId === recovery.selectedId);
  });
  for (const librariesAvailable of [false, true]) test(`post-install verification gates continuation (libraries available=${librariesAvailable})`, async () => {
    choose(); let installed = false, changed = 0; const stages: string[] = [], base = options.runner!;
    options.runner = async (python, backend, args, ...rest) => {
      if (args[2] === 'inspect-model') return { ok: true, missingModules: installed && librariesAvailable ? [] : [{ module: 'scipy', requirement: 'scipy' }] };
      if (args[2] === 'plan-packages') return { ok: true, digest: 'reviewed', packages: [{ name: 'scipy', version: '1.16.0' }] };
      if (args[2] === 'install') { installed = true; return { ok: true }; }
      return base(python, backend, args, ...rest);
    };
    const recovery = await checkOpeningEnvironment(options);
    const progress = { run: vscode.window.withProgress, location: vscode.ProgressLocation };
    (vscode as any).ProgressLocation = { Notification: 15 };
    (vscode.window as any).withProgress = async (_options: unknown, task: Function) => task({},
      { isCancellationRequested: false, onCancellationRequested: () => ({ dispose() {} }) });
    setInfoResponder(() => 'Install dependencies');
    try {
      const repair = recoverOpeningEnvironment(options, recovery, { kind: 'install_dependencies' }, { ...host(), changed: () => changed++, progress: detail => stages.push(detail),
        dependencies: async () => ({ python: path.join(root, '.venv', 'bin', 'python'), managerPython: '/service/python', backendDir: '/backend', storage: root,
          signal: options.signal, current: async () => true, runner: options.runner }) });
      if (librariesAvailable) assert.equal((await repair as any).reason, 'ready');
      else await assert.rejects(repair, error => error instanceof GraphOpeningError && error.failure?.code === 'LIBRARY_CHECK_FAILED' && error.environment?.reason === 'missing_libraries');
      assert.equal(changed, 1); assert.equal(installed, true);
      assert.equal(stages.at(-1), 'Checking capture tools and required model libraries…');
      assert.ok(stages.some(detail => detail === 'Downloading and installing the reviewed libraries…'));
    } finally { (vscode.window as any).withProgress = progress.run; (vscode as any).ProgressLocation = progress.location; }
  });
});
