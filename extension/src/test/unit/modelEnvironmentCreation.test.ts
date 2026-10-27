import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { createModelEnvironment, type ModelEnvironmentCreationOptions } from '../../modelEnvironmentCreation';
import { infoMessageDetails, infoMessages, reset, setInfoResponder, state } from '../vscodeStub';

suite('native model environment creation', () => {
  let storage: string, calls: string[], progress: string[], selected: string | undefined, options: ModelEnvironmentCreationOptions;
  setup(() => {
    reset(); storage = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-new-model-')));
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
