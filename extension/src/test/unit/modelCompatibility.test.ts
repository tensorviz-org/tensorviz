import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { repairModelCompatibility, type ModelCompatibilityOptions } from '../../modelCompatibility';
import { GraphToolError } from '@tensorviz/graph-host/mcpGraph';
import type { ModelImportDiagnostic } from '@tensorviz/contracts/protocol';
import { reset, setInfoResponder, state, infoMessages } from '../vscodeStub';

const importMessage = "cannot import name 'ALL_ATTENTION_FUNCTIONS' from 'transformers.modeling_utils'";
const importDiagnostic: ModelImportDiagnostic = {
  exceptionType: 'ImportError', message: importMessage,
  traceback: `Traceback (most recent call last):\n  File "/project/model.py", line 4, in <module>\nImportError: ${importMessage}`,
  causes: [{ exceptionType: 'ImportError', message: importMessage, module: 'transformers.modeling_utils' }],
  runtime: { platform: 'darwin', architecture: 'arm64', python: '3.12.13', torch: '2.6.0', cuda: null,
    cudaAvailable: false, mpsAvailable: true }, truncated: false,
};

suite('model library compatibility repair', () => {
  let storage: string, calls: string[], selected: string | undefined, candidate: string;
  let options: ModelCompatibilityOptions;
  setup(async () => {
    reset(); infoMessages.length = 0; setInfoResponder(undefined); calls = []; selected = undefined;
    storage = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-compatibility-test-'));
    options = { storage, python: '/original/python', backendDir: '/backend', workspace: '/project', file: '/project/model.py',
      signal: new AbortController().signal, current: async () => true,
      selectEnvironment: async python => { selected = python; },
      runner: async (python, _backend, args) => {
        calls.push(args[0]);
        if (args[0] === 'clone-compatibility') {
          assert.equal(python, options.python);
          assert.deepEqual(JSON.parse(args[2]), ['transformers==4.57.6']);
          candidate = path.join(args[1], 'bin', 'python');
          return { ok: true, interpreter: candidate };
        }
        assert.equal(python, candidate, 'never plan replacements, install or test imports in the original environment');
        if (args[0] === 'plan-compatibility') return { ok: true, digest: 'frozen', packages: [
          { name: 'transformers', version: '4.57.6' }, { name: 'tokenizers', version: '0.22.2' }] };
        if (args[0] === 'install') assert.equal(args[2], 'frozen');
        if (args[0] === 'probe-model') assert.deepEqual(args.slice(1), ['/project', '/project/model.py']);
        return { ok: true };
      } };
  });
  teardown(async () => { await fs.rm(storage, { recursive: true, force: true }); reset(); });
  test('only a reviewed, verified candidate becomes the workspace environment and survives temporary recovery', async () => {
    setInfoResponder(() => 'Install compatible versions');
    assert.equal(await repairModelCompatibility(options, ['transformers==4.57.6']), true);
    assert.equal(selected, candidate);
    assert.deepEqual(calls, ['clone-compatibility', 'plan-compatibility', 'install', 'probe-model']);
    const root = path.resolve(candidate, '../../..');
    assert.ok((await fs.stat(root)).isDirectory());
    await assert.rejects(fs.stat(path.join(root, 'tensorviz-owner.json')), { code: 'ENOENT' });
  });
  test('declining review removes the candidate without installing or selecting it', async () => {
    await assert.rejects(repairModelCompatibility(options, ['transformers==4.57.6']), /not approved/);
    assert.deepEqual(calls, ['clone-compatibility', 'plan-compatibility']);
    assert.equal(selected, undefined);
    assert.deepEqual(await fs.readdir(path.join(storage, 'model-environments')), []);
  });
  test('agent consents before copying, then reviews exact versions without a native modal', async () => {
    const requests: string[] = [];
    options.approve = async request => {
      requests.push(request.operation);
      assert.equal(request.scope, 'isolated_environment');
      assert.deepEqual(Object.keys(request).sort(), ['effects', 'facts', 'operation', 'scope']);
      if (request.operation === 'python.environment.copy') {
        assert.deepEqual(calls, []); assert.deepEqual(await fs.readdir(storage), []);
        assert.deepEqual(request.effects, ['create']); assert.equal(request.facts.sourcePython, options.python);
        assert.deepEqual(request.facts.requestedPackages, ['transformers==4.57.6']);
      } else {
        assert.deepEqual(request.effects, ['install']); assert.equal(request.facts.planDigest, 'frozen');
        assert.equal(request.facts.python, candidate); assert.equal(request.facts.sourcePython, options.python);
        assert.deepEqual(request.facts.packages, [{ name: 'transformers', version: '4.57.6' }, { name: 'tokenizers', version: '0.22.2' }]);
      }
      return true;
    };
    assert.equal(await repairModelCompatibility(options, ['transformers==4.57.6']), true);
    assert.deepEqual(requests, ['python.environment.copy', 'python.packages.install']);
    assert.deepEqual(infoMessages, []); assert.equal(selected, candidate);
  });
  test('declining initial chat permission creates no copy and never falls back', async () => {
    options.approve = async () => false; setInfoResponder(() => 'Install compatible versions');
    await assert.rejects(repairModelCompatibility(options, ['transformers==4.57.6']), /not approved/);
    assert.deepEqual(calls, []); assert.deepEqual(await fs.readdir(storage), []); assert.deepEqual(infoMessages, []);
  });
  test('denying or stopping package permission in chat removes the copy without installing', async () => {
    for (const stop of [false, true]) {
      calls.length = 0;
      const controller = new AbortController(); options.signal = controller.signal;
      options.approve = async request => {
        if (request.operation === 'python.environment.copy') return true;
        if (stop) controller.abort();
        return stop;
      };
      await assert.rejects(repairModelCompatibility(options, ['transformers==4.57.6']), /not approved|changed/);
      assert.deepEqual(calls, ['clone-compatibility', 'plan-compatibility']); assert.equal(selected, undefined);
      assert.deepEqual(await fs.readdir(path.join(storage, 'model-environments')), []); assert.deepEqual(infoMessages, []);
    }
  });
  test('a failed import discards the candidate and retains the original selection', async () => {
    const run = options.runner!;
    options.runner = async (...args) => args[2][0] === 'probe-model'
      ? { ok: false, message: 'Model still fails to import.' } : run(...args);
    setInfoResponder(() => 'Install compatible versions');
    await assert.rejects(repairModelCompatibility(options, ['transformers==4.57.6']), /still fails/);
    assert.equal(selected, undefined);
    assert.deepEqual(await fs.readdir(path.join(storage, 'model-environments')), []);
  });
  test('a failing compatible copy preserves the import cause and observed host after cleanup', async () => {
    const run = options.runner!;
    options.runner = async (...args) => args[2][0] === 'probe-model'
      ? { ok: false, code: 'MODEL_IMPORT_FAILED', message: `Model import failed: ImportError: ${importMessage}`, diagnostic: importDiagnostic }
      : run(...args);
    setInfoResponder(() => 'Install compatible versions');
    await assert.rejects(repairModelCompatibility(options, ['transformers==4.57.6']), error => {
      assert.ok(error instanceof GraphToolError);
      assert.equal(error.code, 'MODEL_IMPORT_FAILED');
      assert.match(error.message, /ALL_ATTENTION_FUNCTIONS/);
      assert.deepEqual(error.diagnostic, importDiagnostic);
      return true;
    });
    assert.equal(selected, undefined);
    assert.equal(options.python, '/original/python');
    assert.deepEqual(await fs.readdir(path.join(storage, 'model-environments')), []);
  });
  test('changes during review prevent installation and promotion', async () => {
    let current = true; options.current = async () => current;
    setInfoResponder(() => { current = false; return 'Install compatible versions'; });
    await assert.rejects(repairModelCompatibility(options, ['transformers==4.57.6']), /changed/);
    assert.deepEqual(calls, ['clone-compatibility', 'plan-compatibility']);
    assert.equal(selected, undefined);
  });
  test('untrusted workspaces do not clone or run model imports', async () => {
    state.isTrusted = false;
    await assert.rejects(repairModelCompatibility(options, ['transformers==4.57.6']), /changed/);
    assert.deepEqual(calls, []);
  });
});
