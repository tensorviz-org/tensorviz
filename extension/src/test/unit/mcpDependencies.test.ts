import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { ensureModelDependencies, installMcpDependencies, replaceModelVersions, type DependencyInstallOptions } from '../../mcpDependencies';
import { GraphToolError } from '@tensorviz/graph-host/mcpGraph';
import { reset, state, setInfoResponder, infoMessages } from '../vscodeStub';

suite('MCP model dependency approval', () => {
  let storage: string, calls: string[], options: DependencyInstallOptions;
  setup(async () => {
    reset(); infoMessages.length = 0; setInfoResponder(undefined); storage = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-dependency-test-')); calls = [];
    options = { storage, python: '/project/python', backendDir: '/backend', packages: ['einops>=0.8'],
      signal: new AbortController().signal, current: async () => true,
      runner: async (_python, _backend, args) => {
        calls.push(args[0]);
        if (args[0] === 'plan-packages') {
          assert.deepEqual(JSON.parse(args[2]), ['einops>=0.8']);
          return { ok: true, digest: 'reviewed', packages: [{ name: 'einops', version: '0.8.1' }] };
        }
        assert.equal(args[2], 'reviewed'); return { ok: true };
      } };
  });
  teardown(async () => { await fs.rm(storage, { recursive: true, force: true }); reset(); });
  function sourceOnly() {
    options.runner = async (python, _backend, args) => {
      assert.equal(python, options.python, 'Planning, building and installing must target the capture interpreter');
      calls.push(args[0]);
      if (args[0] === 'plan-packages') return { ok: false, code: 'NO_COMPATIBLE_WHEEL', message: 'No wheel is available.' };
      if (args[0] === 'plan-packages-source') return { ok: true, digest: 'built-and-reviewed',
        packages: [{ name: 'einops', version: '0.8.1' }], builtFromSource: ['einops'] };
      assert.equal(args[2], 'built-and-reviewed'); return { ok: true };
    };
  }
  test('a tool request alone cannot authorize installation', async () => {
    await assert.rejects(installMcpDependencies(options), /not approved/);
    assert.deepEqual(calls, ['plan-packages']); assert.deepEqual(await fs.readdir(storage), []);
  });
  test('consent installs exactly the reviewed plan and removes temporary files', async () => {
    setInfoResponder(() => 'Install dependencies');
    await installMcpDependencies(options);
    assert.deepEqual(calls, ['plan-packages', 'install']); assert.deepEqual(await fs.readdir(storage), []);
  });
  test('agent approval reviews the frozen package versions in chat without a native modal', async () => {
    const stages: string[] = []; options.progress = detail => stages.push(detail);
    options.approve = async (request, signal) => {
      assert.equal(request.operation, 'python.packages.install'); assert.equal(signal, options.signal);
      assert.equal(request.scope, 'existing_environment');
      assert.deepEqual(Object.keys(request).sort(), ['effects', 'facts', 'operation', 'scope']);
      assert.deepEqual(request.effects, ['install']); assert.equal(request.facts.planDigest, 'reviewed');
      assert.deepEqual(request.facts.packages, [{ name: 'einops', version: '0.8.1' }]);
      assert.equal(request.facts.python, '/project/python'); assert.equal(request.facts.preservesTorch, true);
      return true;
    };
    await installMcpDependencies(options);
    assert.deepEqual(calls, ['plan-packages', 'install']); assert.deepEqual(infoMessages, []);
    assert.ok(stages.includes('Waiting for approval in chat…'));
    assert.ok(stages.every(detail => !detail.includes('VS Code')));
  });
  test('denied agent approval never installs or falls back to a native modal', async () => {
    setInfoResponder(() => 'Install dependencies'); options.approve = async () => false;
    await assert.rejects(installMcpDependencies(options), /not approved/);
    assert.deepEqual(calls, ['plan-packages']); assert.deepEqual(infoMessages, []);
  });
  test('version and PyTorch replacement reviews both declare changes to an existing environment', async () => {
    const operations: string[] = [];
    options.approve = async request => {
      assert.equal(request.scope, 'existing_environment'); operations.push(request.operation);
      assert.deepEqual(request.effects, ['install', 'replace']); assert.equal(request.facts.planDigest, 'replacement');
      assert.deepEqual(request.facts.packages, [{ name: 'torch', previous: '2.6.0', version: '2.7.1' }]);
      assert.equal(request.facts.affectsSharedEnvironment, true); assert.equal(request.facts.backupAvailable, false);
      return true;
    };
    options.runner = async (_python, _backend, args) => {
      calls.push(args[0]);
      if (args[0] === 'plan-replacement') return { ok: true, digest: 'replacement', replacesTorch: true,
        packages: [{ name: 'torch', version: '2.7.1', previous: '2.6.0' }] };
      assert.equal(args[0], 'install'); assert.equal(args[2], 'replacement'); return { ok: true };
    };
    await replaceModelVersions({ ...options, packages: ['torch==2.7.1'], project: '/project' });
    assert.deepEqual(operations, ['python.packages.replace', 'python.packages.replace_torch']);
    assert.deepEqual(calls, ['plan-replacement', 'install']); assert.deepEqual(state.warningMessages, []); assert.deepEqual(infoMessages, []);
  });
  test('agent source-build and install decisions are separate and never use native modals', async () => {
    sourceOnly(); const requests: string[] = [];
    options.approve = async request => {
      requests.push(request.operation); assert.equal(request.scope, 'existing_environment');
      if (request.operation === 'python.packages.build') {
        assert.deepEqual(request.effects, ['execute_code']); assert.deepEqual(request.facts.access, { filesystem: true, network: true });
        assert.deepEqual(request.facts.requestedPackages, ['einops>=0.8']);
      } else {
        assert.equal(request.facts.planDigest, 'built-and-reviewed'); assert.deepEqual(request.facts.builtFromSource, ['einops']);
      }
      return request.operation !== 'python.packages.install';
    };
    await assert.rejects(installMcpDependencies(options), /not approved/);
    assert.deepEqual(requests, ['python.packages.build', 'python.packages.install']);
    assert.deepEqual(calls, ['plan-packages', 'plan-packages-source']);
    assert.deepEqual(state.warningMessages, []); assert.deepEqual(infoMessages, []);
  });
  test('denying a source build in chat cannot use native approval or run package build code', async () => {
    sourceOnly(); options.approve = async request => { assert.equal(request.operation, 'python.packages.build'); return false; };
    state.warningAnswers.push('Build from source');
    await assert.rejects(installMcpDependencies(options), /not approved/);
    assert.deepEqual(calls, ['plan-packages']); assert.deepEqual(state.warningMessages, []);
  });
  test('a stale active-agent permission cannot fall back to native approval', async () => {
    options.approve = async () => { throw new Error('The active agent target changed.'); };
    setInfoResponder(() => 'Install dependencies');
    await assert.rejects(installMcpDependencies(options), /active agent target changed/);
    assert.deepEqual(calls, ['plan-packages']); assert.deepEqual(infoMessages, []);
  });
  test('stopping during chat permission never starts an install', async () => {
    const controller = new AbortController(); options.signal = controller.signal;
    options.approve = async () => { controller.abort(); return true; };
    await assert.rejects(installMcpDependencies(options), /abort/i);
    assert.deepEqual(calls, ['plan-packages']); assert.deepEqual(infoMessages, []);
  });
  test('an inactive agent callback preserves native package review', async () => {
    options.approve = async () => undefined; setInfoResponder(() => 'Install dependencies');
    await installMcpDependencies(options);
    assert.deepEqual(calls, ['plan-packages', 'install']); assert.equal(infoMessages.length, 1);
  });
  test('installation reports real stages and never declares success before opening verification', async () => {
    const stages: string[] = []; options.progress = detail => stages.push(detail);
    setInfoResponder(() => 'Install dependencies'); await installMcpDependencies(options);
    assert.deepEqual(stages, ['Resolving and downloading compatible model libraries…', 'Waiting for package approval in VS Code…', 'Downloading and installing the reviewed libraries…']);
  });
  test('a named package unavailable on this platform stops before source-build approval', async () => {
    options.runner = async (_python, _backend, args) => {
      calls.push(args[0]);
      return args[0] === 'plan-packages' ? { ok: false, code: 'NO_COMPATIBLE_WHEEL', requirement: 'triton>=3.5.0' }
        : { ok: false, code: 'PACKAGE_UNAVAILABLE', requirement: 'triton>=3.5.0', message: 'No distribution for this Python/platform.' };
    };
    await assert.rejects(installMcpDependencies(options), error => error instanceof GraphToolError
      && error.code === 'PACKAGE_UNAVAILABLE' && /triton>=3.5.0/.test(error.message));
    assert.deepEqual(calls, ['plan-packages', 'check-package-availability']);
    assert.equal(state.warningMessages.length, 0); assert.deepEqual(await fs.readdir(storage), []);
  });
  test('availability of a source package still requires source-build and frozen-install approval', async () => {
    sourceOnly(); const original = options.runner!;
    options.runner = async (python, backend, args, ...rest) => {
      if (args[0] === 'check-package-availability') { calls.push(args[0]); return { ok: true }; }
      const result = await original(python, backend, args, ...rest);
      return args[0] === 'plan-packages' ? { ...result, requirement: 'einops>=0.8' } : result;
    };
    state.warningAnswers.push('Build from source'); setInfoResponder(() => 'Install dependencies');
    await installMcpDependencies(options);
    assert.deepEqual(calls, ['plan-packages', 'check-package-availability', 'plan-packages-source', 'install']);
  });
  test('tool-owned pip targets the project Python for planning and approved installation', async () => {
    options.managerPython = '/tensorviz/tool/python';
    const original = options.runner!;
    options.runner = async (python, backend, args, ...rest) => {
      assert.equal(python, options.managerPython);
      assert.deepEqual(args.slice(0, 2), ['--target-python', options.python]);
      return original(options.python, backend, args.slice(2), ...rest);
    };
    setInfoResponder(() => 'Install dependencies');
    await installMcpDependencies(options);
    assert.deepEqual(calls, ['plan-packages', 'install']);
  });
  test('untrusted workspaces never resolve or install packages', async () => {
    state.isTrusted = false;
    await assert.rejects(installMcpDependencies(options), /Trust/); assert.deepEqual(calls, []);
  });
  test('changed source or environment after review prevents installation', async () => {
    let current = true; options.current = async () => current;
    setInfoResponder(() => { current = false; return 'Install dependencies'; });
    await assert.rejects(installMcpDependencies(options), /changed after review/);
    assert.deepEqual(calls, ['plan-packages']);
  });
  test('disconnect during review prevents installation even if approved', async () => {
    const controller = new AbortController(); options.signal = controller.signal;
    setInfoResponder(() => { controller.abort(); return 'Install dependencies'; });
    await assert.rejects(installMcpDependencies(options), /not approved/);
    assert.deepEqual(calls, ['plan-packages']);
  });
  test('source build approval is required before any package build code runs', async () => {
    sourceOnly();
    await assert.rejects(installMcpDependencies(options), (error: unknown) => error instanceof GraphToolError && error.code === 'SOURCE_BUILD_DECLINED');
    assert.deepEqual(calls, ['plan-packages']); assert.equal(state.warningMessages.length, 1);
    assert.deepEqual(await fs.readdir(storage), []);
  });
  test('a reviewed source build still needs approval of the final wheel installation', async () => {
    sourceOnly(); state.warningAnswers.push('Build from source');
    await assert.rejects(installMcpDependencies(options), /installation was not approved/);
    assert.deepEqual(calls, ['plan-packages', 'plan-packages-source']);
  });
  test('source builds and installs use the same environment with the resulting frozen plan', async () => {
    sourceOnly(); state.warningAnswers.push('Build from source'); setInfoResponder(() => 'Install dependencies');
    await installMcpDependencies(options);
    assert.deepEqual(calls, ['plan-packages', 'plan-packages-source', 'install']);
    assert.deepEqual(await fs.readdir(storage), []);
  });
  test('pipless source resolution and final installation retain the same explicit target', async () => {
    sourceOnly(); options.managerPython = '/tensorviz/tool/python';
    const original = options.runner!;
    options.runner = async (python, backend, args, ...rest) => {
      assert.equal(python, options.managerPython);
      assert.deepEqual(args.slice(0, 2), ['--target-python', options.python]);
      return original(options.python, backend, args.slice(2), ...rest);
    };
    state.warningAnswers.push('Build from source'); setInfoResponder(() => 'Install dependencies');
    await installMcpDependencies(options);
    assert.deepEqual(calls, ['plan-packages', 'plan-packages-source', 'install']);
  });
  test('a changed target after source approval prevents the build', async () => {
    sourceOnly(); state.warningAnswers.push('Build from source');
    let checks = 0; options.current = async () => ++checks < 3;
    await assert.rejects(installMcpDependencies(options), /changed after source-build review/);
    assert.deepEqual(calls, ['plan-packages']);
  });
  test('a resolver conflict stays in the capture environment and never falls back to source builds', async () => {
    options.runner = async () => ({ ok: false, code: 'DEPENDENCY_CONFLICT', message: 'Pinned requirements conflict.' });
    await assert.rejects(installMcpDependencies(options), /Pinned requirements conflict/);
    assert.equal(state.warningMessages.length, 0);
  });
});

suite('Canvas model dependency installation', () => {
  let storage: string, calls: string[], installed: boolean;
  let options: Parameters<typeof ensureModelDependencies>[0];
  const vscode = require('../vscodeStub');
  let previousProgress: unknown, previousLocation: unknown;
  setup(async () => {
    reset(); infoMessages.length = 0; calls = []; installed = false;
    storage = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-canvas-dependencies-'));
    previousProgress = vscode.window.withProgress; previousLocation = vscode.ProgressLocation;
    vscode.ProgressLocation = { Notification: 15 };
    vscode.window.withProgress = async (_options: unknown, action: Function) => action({},
      { isCancellationRequested: false, onCancellationRequested: () => ({ dispose() {} }) });
    options = { storage, python: '/project/python', backendDir: '/backend', workspace: '/workspace', file: '/workspace/model.py',
      signal: new AbortController().signal, current: async () => true,
      runner: async (python, _backend, args) => {
        assert.equal(python, options.python, 'inspection, planning and installation must use the capture environment');
        calls.push(args[0]);
        if (args[0] === 'inspect-model') {
          assert.deepEqual(args.slice(1), [options.workspace, options.file, JSON.stringify(options.missingModules ?? [])]);
          return { ok: true, missingModules: installed ? [] : [{ module: 'transformers', requirement: 'transformers<5' },
            { module: 'optional_cuda', requirement: 'optional-cuda', optional: true }] };
        }
        if (args[0] === 'plan-packages') {
          assert.deepEqual(JSON.parse(args[2]), ['transformers<5']);
          return { ok: true, digest: 'frozen', packages: [{ name: 'transformers', version: '4.57.6' }] };
        }
        assert.deepEqual([args[0], args[2]], ['install', 'frozen']);
        installed = true; return { ok: true };
      } };
  });
  teardown(async () => {
    vscode.window.withProgress = previousProgress; vscode.ProgressLocation = previousLocation;
    await fs.rm(storage, { recursive: true, force: true }); reset();
  });
  for (const python of ['/project/python', '/tensorviz/runtime/venv/bin/python']) test(`offers required libraries and verifies installation in ${python}`, async () => {
    options.python = python;
    setInfoResponder(() => 'Install dependencies');
    assert.equal(await ensureModelDependencies(options), true);
    assert.deepEqual(calls, ['inspect-model', 'plan-packages', 'install', 'inspect-model']);
    assert.equal(await ensureModelDependencies(options), false, 'reopening a ready environment must not reinstall');
    assert.equal(infoMessages.length, 1);
  });
  test('declining review stops before installation and leaves a retryable error', async () => {
    await assert.rejects(ensureModelDependencies(options), /not approved/);
    assert.deepEqual(calls, ['inspect-model', 'plan-packages']);
  });
  test('inspection and installation share the target even when its pip lives elsewhere', async () => {
    options.managerPython = '/tensorviz/tool/python';
    const original = options.runner!;
    options.runner = async (python, backend, args, ...rest) => {
      assert.equal(python, options.managerPython);
      assert.deepEqual(args.slice(0, 2), ['--target-python', options.python]);
      return original(options.python, backend, args.slice(2), ...rest);
    };
    setInfoResponder(() => 'Install dependencies');
    assert.equal(await ensureModelDependencies(options), true);
    assert.deepEqual(calls, ['inspect-model', 'plan-packages', 'install', 'inspect-model']);
  });
  test('unknown import names are reported without guessing a package', async () => {
    options.runner = async () => ({ ok: true, missingModules: [{ module: 'private_lib', requirement: null }] });
    await assert.rejects(ensureModelDependencies(options), /correct package names.*Refresh/);
    assert.equal(infoMessages.length, 0);
  });
  test('optional dependencies do not block or trigger installation', async () => {
    options.runner = async () => ({ ok: true, missingModules: [{ module: 'flash_attn', requirement: null, optional: true }] });
    assert.equal(await ensureModelDependencies(options), false);
    assert.equal(infoMessages.length, 0);
  });
  test('a runtime missing-module fact is forwarded to the scanner', async () => {
    options.missingModules = ['transformers']; setInfoResponder(() => 'Install dependencies');
    assert.equal(await ensureModelDependencies(options), true);
  });
  test('a source or environment change during review prevents installation', async () => {
    let current = true; options.current = async () => current;
    setInfoResponder(() => { current = false; return 'Install dependencies'; });
    await assert.rejects(ensureModelDependencies(options), /changed after review/);
    assert.deepEqual(calls, ['inspect-model', 'plan-packages']);
  });
  test('an import still missing after pip completes is a failure, never a retry loop', async () => {
    const runner = options.runner!;
    options.runner = async (...args) => { const result = await runner(...args); installed = false; return result; };
    setInfoResponder(() => 'Install dependencies');
    await assert.rejects(ensureModelDependencies(options), /still cannot find/);
    assert.equal(calls.filter(call => call === 'install').length, 1);
  });
  test('an installed library triggers compatibility repair only after an import failure', async () => {
    let candidate: string | undefined, selected: string | undefined;
    options.selectEnvironment = async python => { selected = python; };
    options.runner = async (python, _backend, args) => {
      calls.push(args[0]);
      if (args[0] === 'inspect-model') return { ok: true, compatibilityPackages: ['transformers==4.57.6'] };
      if (args[0] === 'clone-compatibility') {
        assert.equal(python, options.python);
        candidate = path.join(args[1], 'bin', 'python'); return { ok: true, interpreter: candidate };
      }
      assert.equal(python, candidate);
      if (args[0] === 'plan-compatibility') {
        assert.equal(args[3], options.workspace);
        return { ok: true, digest: 'compatibility', packages: [{ name: 'transformers', version: '4.57.6' }] };
      }
      return { ok: true };
    };
    assert.equal(await ensureModelDependencies(options), false, 'a working newer library must be left alone');
    assert.deepEqual(calls, ['inspect-model']);
    options.importFailure = true; setInfoResponder(() => 'Install compatible versions');
    assert.equal(await ensureModelDependencies(options), true);
    assert.equal(selected, candidate);
    assert.deepEqual(calls.slice(1), ['inspect-model', 'clone-compatibility', 'plan-compatibility', 'install', 'probe-model']);
    assert.equal(infoMessages.length, 1);
  });
});
