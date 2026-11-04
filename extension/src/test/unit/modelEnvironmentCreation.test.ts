import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { createModelEnvironment, type ModelEnvironmentCreationOptions } from '../../modelEnvironmentCreation';
import { CanvasAgent } from '../../canvasAgent';
import { GraphToolError } from '@tensorviz/graph-host/mcpGraph';
import type { CanvasAgentState, LocalBrowserAgentRequest, LocalBrowserTargetIdentity, ModelImportDiagnostic } from '@tensorviz/contracts/protocol';
import { infoMessageDetails, infoMessages, reset, setInfoResponder, state } from '../vscodeStub';

const importMessage = "cannot import name 'ALL_ATTENTION_FUNCTIONS' from 'transformers.modeling_utils'";
const importDiagnostic: ModelImportDiagnostic = {
  exceptionType: 'ImportError', message: importMessage,
  traceback: `Traceback (most recent call last):\n  File "/project/model.py", line 4, in <module>\nImportError: ${importMessage}`,
  causes: [{ exceptionType: 'ImportError', message: importMessage, module: 'transformers.modeling_utils' }],
  runtime: { platform: 'darwin', architecture: 'arm64', python: '3.12.13', torch: '2.6.0', cuda: null,
    cudaAvailable: false, mpsAvailable: true }, truncated: false,
};

suite('native model environment creation', () => {
  let storage: string, calls: string[], progress: string[], selected: string | undefined, options: ModelEnvironmentCreationOptions;
  setup(() => {
    reset(); infoMessages.length = 0; storage = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-new-model-')));
    calls = []; progress = []; selected = undefined;
    options = { managerPython: '/tool/python', backendDir: '/backend', storage, workspace: '/model', file: '/model/model.py',
      signal: new AbortController().signal, current: async () => true, progress: detail => progress.push(detail),
      selectEnvironment: async python => { selected = python; },
      runner: async (manager, _backend, args) => {
        assert.equal(manager, '/tool/python');
        const action = args[0] === '--target-python' ? args[2] : args[0]; calls.push(action);
        if (action === 'create-environment') {
          assert.ok(args[1].startsWith(storage + path.sep)); fs.mkdirSync(args[1], { recursive: true }); return { ok: true };
        }
        assert.ok(args[1].startsWith(storage + path.sep), 'never install into service or existing project Python');
        if (action === 'inspect-model-setup') return { ok: true, manifestDigest: 'declared', sourcePackages: ['helper @ git+https://example.org/helper.git@' + 'a'.repeat(40)] };
        if (action === 'plan-model-environment-source') { assert.equal(args[6], 'declared'); return { ok: true, digest: 'reviewed', packages: [{ name: 'torch', version: '2.10.0' }, { name: 'helper', version: '1' }] }; }
        if (action === 'install') { assert.equal(selected, undefined); assert.equal(args[4], 'reviewed'); }
        return { ok: true };
      } };
    state.warningAnswers = ['Build libraries']; setInfoResponder(() => 'Install and continue');
  });
  teardown(() => fs.rmSync(storage, { recursive: true, force: true }));
  test('creates an isolated venv, reviews source and exact versions, checks then selects it', async () => {
    await createModelEnvironment(options);
    assert.deepEqual(calls, ['create-environment', 'inspect-model-setup', 'plan-model-environment-source', 'install', 'probe-capture', 'probe-model']);
    assert.ok(selected?.startsWith(path.join(storage, 'model-environments') + path.sep));
    assert.ok(progress.some(text => text.includes('Downloading and installing')));
    assert.match(infoMessageDetails[0], /torch 2\.10\.0/);
    assert.match(state.warningMessages[0], /Build model libraries/);
    assert.ok(fs.existsSync(path.dirname(path.dirname(selected!))));
  });
  test('agent reviews creation before writes, then source builds and exact package versions in chat', async () => {
    const requests: string[] = [];
    options.approve = async (request, signal) => {
      assert.equal(signal, options.signal); requests.push(request.operation);
      assert.equal(request.scope, 'isolated_environment');
      assert.deepEqual(Object.keys(request).sort(), ['effects', 'facts', 'operation', 'scope']);
      if (request.operation === 'python.environment.create') {
        assert.deepEqual(calls, []); assert.deepEqual(fs.readdirSync(storage), []);
        assert.deepEqual(request.effects, ['create']);
        assert.equal(request.facts.model, options.file); assert.equal(request.facts.workspace, options.workspace);
        assert.equal(request.facts.destinationRoot, path.join(storage, 'model-environments'));
      }
      if (request.operation === 'python.packages.build') {
        assert.deepEqual(request.effects, ['execute_code']);
        assert.deepEqual(request.facts.access, { filesystem: true, network: true });
        assert.deepEqual(request.facts.sourcePackages, ['helper @ git+https://example.org/helper.git@' + 'a'.repeat(40)]);
      }
      if (request.operation === 'python.packages.install') {
        assert.deepEqual(request.effects, ['install']); assert.equal(request.facts.planDigest, 'reviewed');
        assert.deepEqual(request.facts.packages, [{ name: 'torch', version: '2.10.0' }, { name: 'helper', version: '1' }]);
      }
      return true;
    };
    await createModelEnvironment(options);
    assert.deepEqual(requests, ['python.environment.create', 'python.packages.build', 'python.packages.install']);
    assert.deepEqual(state.warningMessages, []); assert.deepEqual(infoMessages, []);
    assert.ok(selected); assert.ok(progress.some(detail => detail.includes('approval in chat')));
  });
  test('declining initial agent consent creates no directory, resolves nothing and never falls back to native review', async () => {
    options.approve = async () => false;
    await assert.rejects(createModelEnvironment(options), /Creating a model environment was not approved/);
    assert.deepEqual(calls, []); assert.deepEqual(fs.readdirSync(storage), []);
    assert.deepEqual(state.warningMessages, []); assert.deepEqual(infoMessages, []);
  });
  test('declining source or package consent in chat removes the incomplete environment without native prompts', async () => {
    for (const declined of ['python.packages.build', 'python.packages.install']) {
      calls.length = 0;
      options.approve = async request => request.operation !== declined;
      await assert.rejects(createModelEnvironment(options), /not approved/);
      assert.ok(!calls.includes('install')); assert.equal(selected, undefined);
      assert.deepEqual(fs.readdirSync(path.join(storage, 'model-environments')), []);
      assert.deepEqual(state.warningMessages, []); assert.deepEqual(infoMessages, []);
    }
  });
  test('stopping while initial agent permission is pending prevents all environment work', async () => {
    const controller = new AbortController(); options.signal = controller.signal;
    options.approve = async () => { controller.abort(); return true; };
    await assert.rejects(createModelEnvironment(options), /abort/i);
    assert.deepEqual(calls, []); assert.deepEqual(fs.readdirSync(storage), []);
  });
  test('stopping during package approval cleans up before installation or model execution', async () => {
    const controller = new AbortController(); options.signal = controller.signal;
    options.approve = async request => { if (request.operation === 'python.packages.install') controller.abort(); return true; };
    await assert.rejects(createModelEnvironment(options), /abort/i);
    assert.ok(!calls.includes('install')); assert.ok(!calls.includes('probe-model')); assert.equal(selected, undefined);
    assert.deepEqual(fs.readdirSync(path.join(storage, 'model-environments')), []);
  });
  test('a callback without an active agent retains the existing native reviews', async () => {
    options.approve = async () => undefined;
    await createModelEnvironment(options);
    assert.equal(state.warningMessages.length, 1); assert.equal(infoMessages.length, 1); assert.ok(selected);
  });
  test('a creation approved in chat cannot fall back to native review after its agent turn finishes', async () => {
    const target: LocalBrowserTargetIdentity = { workspaceFolder: '/model', file: 'model.py', cls: 'Net',
      documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
    let latest: CanvasAgentState | undefined, finish!: () => void;
    const agent = new CanvasAgent({ prepare: async () => ({ cwd: '/model', context: 'The model being set up.' }),
      run: async () => { await new Promise<void>(resolve => { finish = resolve; }); },
      publish: (_view, snapshot) => { latest = snapshot; }, clear() {} });
    const request = (action: Record<string, unknown>) => ({ type: 'agentRequest', protocolVersion: 1,
      workspaceSessionId: 'workspace', messageId: 'message', viewId: 'view', ...action } as LocalBrowserAgentRequest);
    let releaseCreate!: () => void, created!: () => void;
    const createStarted = new Promise<void>(resolve => { created = resolve; });
    const createGate = new Promise<void>(resolve => { releaseCreate = resolve; });
    const original = options.runner!;
    options.runner = async (...args) => {
      const result = await original(...args);
      if (args[2][0] === 'create-environment') { created(); await createGate; }
      return result;
    };
    try {
      agent.handle(request({ action: 'send', requestId: 'setup-turn', provider: 'codex', target, nodeIds: [], prompt: 'Set up this model.' }));
      await new Promise<void>(resolve => setImmediate(resolve));
      options.approve = agent.permissionRequester(target);
      const pending = createModelEnvironment(options);
      await new Promise<void>(resolve => setImmediate(resolve));
      const id = latest!.permissions![0].id;
      agent.presentPermission(id, { title: 'Prepare model Python?', summary: 'Create a separate Python for this model.',
        allowLabel: 'Create model Python', denyLabel: 'Skip creation' });
      agent.handle(request({ action: 'permission', permissionId: id, decision: 'allow' }));
      await createStarted;
      finish(); await new Promise<void>(resolve => setImmediate(resolve));
      releaseCreate();
      await assert.rejects(pending, /Source builds were not approved/);
      assert.deepEqual(calls, ['create-environment', 'inspect-model-setup']);
      assert.deepEqual(state.warningMessages, []); assert.deepEqual(infoMessages, []); assert.equal(selected, undefined);
      assert.deepEqual(fs.readdirSync(path.join(storage, 'model-environments')), []);
    } finally { agent.dispose(); finish?.(); releaseCreate(); }
  });
  test('the chosen name is the actual venv directory and appears in progress and installation review', async () => {
    options.name = 'kimi-k3'; const run = options.runner!;
    options.runner = async (...args) => {
      if (args[2][2] === 'plan-model-environment-source') assert.equal(args[2][7], 'kimi-k3');
      return run(...args);
    };
    await createModelEnvironment(options);
    assert.equal(path.basename(path.dirname(path.dirname(selected!))), 'kimi-k3');
    assert.ok(progress.some(text => text.includes('“kimi-k3”')));
    assert.match(infoMessages.at(-1)!, /Install 2 libraries in “kimi-k3”/);
  });
  for (const name of ['', '../existing', 'CON', 'a'.repeat(49)]) test(`invalid name ${JSON.stringify(name)} cannot create or modify an environment`, async () => {
    options.name = name;
    await assert.rejects(createModelEnvironment(options), /environment name/);
    assert.deepEqual(calls, []); assert.deepEqual(fs.readdirSync(storage), []);
  });
  test('declining a source build deletes only the new venv and never selects or installs', async () => {
    state.warningAnswers = [undefined];
    await assert.rejects(createModelEnvironment(options), /Source builds were not approved/);
    assert.deepEqual(calls, ['create-environment', 'inspect-model-setup']); assert.equal(selected, undefined);
    assert.deepEqual(fs.readdirSync(path.join(storage, 'model-environments')), []);
  });
  test('an unsuccessful model import deletes the new environment without selecting it', async () => {
    const run = options.runner!;
    options.runner = async (...args) => args[2][2] === 'probe-model' ? { ok: false, code: 'IMPORT_FAILED', message: 'Model import failed.' } : run(...args);
    await assert.rejects(createModelEnvironment(options), /Model import failed/); assert.equal(selected, undefined);
    assert.deepEqual(fs.readdirSync(path.join(storage, 'model-environments')), []);
  });
  test('an import failure retains its exact diagnostic after the unsuccessful environment is removed', async () => {
    const run = options.runner!;
    options.runner = async (...args) => args[2][2] === 'probe-model'
      ? { ok: false, code: 'MODEL_IMPORT_FAILED', message: `Model import failed: ImportError: ${importMessage}`, diagnostic: importDiagnostic }
      : run(...args);
    await assert.rejects(createModelEnvironment(options), error => {
      assert.ok(error instanceof GraphToolError);
      assert.equal(error.code, 'MODEL_IMPORT_FAILED');
      assert.match(error.message, /ALL_ATTENTION_FUNCTIONS/);
      assert.deepEqual(error.diagnostic, importDiagnostic);
      return true;
    });
    assert.equal(selected, undefined);
    assert.deepEqual(fs.readdirSync(path.join(storage, 'model-environments')), []);
    assert.deepEqual(calls, ['create-environment', 'inspect-model-setup', 'plan-model-environment-source', 'install', 'probe-capture']);
  });
  test('an unavailable distribution names the requirement and discards the unselected venv', async () => {
    const run = options.runner!;
    options.runner = async (...args) => args[2][2] === 'plan-model-environment-source'
      ? { ok: false, code: 'PACKAGE_UNAVAILABLE', requirement: 'helper==1', message: 'No distribution satisfies these requirements for this Python/platform.' } : run(...args);
    await assert.rejects(createModelEnvironment(options), /helper==1: No distribution/);
    assert.equal(selected, undefined); assert.ok(!calls.includes('install'));
    assert.deepEqual(fs.readdirSync(path.join(storage, 'model-environments')), []);
  });
  test('Mamba-like unavailable packages stop before a futile source-build prompt', async () => {
    const run = options.runner!;
    options.runner = async (...args) => args[2][2] === 'inspect-model-setup' ? { ok: true, manifestDigest: 'declared', sourcePackages: [] }
      : args[2][2] === 'plan-model-environment' ? { ok: false, code: 'NO_COMPATIBLE_WHEEL', requirement: 'triton>=3.5.0', message: 'No compatible wheel.' }
        : args[2][2] === 'check-package-availability' ? { ok: false, code: 'PACKAGE_UNAVAILABLE', requirement: 'triton>=3.5.0', message: 'No distribution is available for this Python/platform.' } : run(...args);
    await assert.rejects(createModelEnvironment(options), /triton>=3.5.0: No distribution/);
    assert.deepEqual(state.warningMessages, []); assert.deepEqual(infoMessageDetails, []);
    assert.equal(selected, undefined); assert.ok(!calls.includes('install'));
    assert.deepEqual(fs.readdirSync(path.join(storage, 'model-environments')), []);
  });
  test('an available source-only package still requires approval and a reviewed installation plan', async () => {
    const run = options.runner!;
    options.runner = async (...args) => args[2][2] === 'inspect-model-setup' ? { ok: true, manifestDigest: 'declared', sourcePackages: [] }
      : args[2][2] === 'plan-model-environment' ? { ok: false, code: 'NO_COMPATIBLE_WHEEL', requirement: 'helper==1', message: 'No compatible wheel.' }
        : run(...args);
    await createModelEnvironment(options);
    assert.ok(calls.includes('check-package-availability'));
    assert.match(state.warningMessages[0], /Build model libraries/);
    assert.match(infoMessageDetails[0], /helper 1/); assert.ok(selected);
  });
  test('cancellation during install prevents worker probes and discards the incomplete environment', async () => {
    const abort = new AbortController(); options.signal = abort.signal; const run = options.runner!;
    options.runner = async (...args) => { const result = await run(...args); if (args[2][2] === 'install') abort.abort(); return result; };
    await assert.rejects(createModelEnvironment(options), /abort/i); assert.equal(selected, undefined);
    assert.ok(!calls.includes('probe-model')); assert.deepEqual(fs.readdirSync(path.join(storage, 'model-environments')), []);
  });
});
