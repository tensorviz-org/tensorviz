import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { replaceDependencies, replacementSummary, type DependencyInstallOptions } from '../dist/dependencyInstallCore';
import type { HelperResult } from '../dist/projectEnvironmentCore';

const torchPlan: HelperResult = { ok: true, digest: 'reviewed', replacesTorch: true, packages: [
  { name: 'torch', version: '2.0.0', previous: '2.7.1' }, { name: 'torchvision', version: '0.15.1', previous: '0.22.1' },
  { name: 'sympy', version: '1.13.1' }] };

suite('in-place version replacement', () => {
  let storage: string, calls: string[][], options: DependencyInstallOptions & { project: string }, plan: HelperResult;
  setup(async () => {
    storage = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-replacement-')); calls = []; plan = torchPlan;
    options = { storage, python: '/project/.venv/bin/python', managerPython: '/tool/python', backendDir: '/backend',
      packages: ['torch==2.0.0'], project: '/project', signal: new AbortController().signal, current: async () => true,
      runner: async (python, _backend, args) => {
        assert.equal(python, '/tool/python'); assert.deepEqual(args.slice(0, 2), ['--target-python', '/project/.venv/bin/python']);
        calls.push(args.slice(2));
        return args[2] === 'plan-replacement' ? plan : { ok: true };
      } };
  });
  teardown(async () => { await fs.rm(storage, { recursive: true, force: true }); });
  const host = (answers: { review?: boolean; torch?: boolean }, asked: string[] = []) => ({
    trusted: () => true,
    reviewReplacement: async () => { asked.push('review'); return !!answers.review; },
    confirmTorch: async () => { asked.push('torch'); return !!answers.torch; },
  });

  test('lists each change as old → new, and additions separately', () => {
    assert.deepEqual(replacementSummary(torchPlan), ['torch 2.7.1 → 2.0.0', 'torchvision 0.22.1 → 0.15.1', '+ sympy 1.13.1']);
  });

  test('a Torch/CUDA change needs a second confirmation; only both answers install the reviewed digest', async () => {
    const asked: string[] = [];
    await replaceDependencies(options, host({ review: true, torch: true }, asked));
    assert.deepEqual(asked, ['review', 'torch']);
    assert.equal(calls[0][0], 'plan-replacement'); assert.deepEqual(JSON.parse(calls[0][2]), ['torch==2.0.0']); assert.equal(calls[0][3], '/project');
    assert.deepEqual(calls[1].slice(0, 1), ['install']); assert.equal(calls[1][2], 'reviewed');
    assert.deepEqual(await fs.readdir(storage), [], 'the plan directory is removed');
  });

  test('declining either question changes nothing', async () => {
    await assert.rejects(replaceDependencies(options, host({ review: false })), /not approved\. Nothing was changed/);
    await assert.rejects(replaceDependencies(options, host({ review: true, torch: false })), /Replacing PyTorch was not approved/);
    assert.ok(calls.every(args => args[0] === 'plan-replacement'));
  });

  test('a plan without Torch changes asks once', async () => {
    plan = { ok: true, digest: 'reviewed', packages: [{ name: 'numpy', version: '1.26.4', previous: '2.1.0' }] };
    const asked: string[] = [];
    await replaceDependencies(options, host({ review: true }, asked));
    assert.deepEqual(asked, ['review']); assert.deepEqual(calls.map(args => args[0]), ['plan-replacement', 'install']);
  });

  test('a missing build is reported in plain words before any question', async () => {
    plan = { ok: false, code: 'NO_MATCHING_BUILD', message: 'torch 2.0.0 has no build for Python 3.12 on macOS arm64. Choose an environment with a supported Python, or try the installed versions.' };
    const asked: string[] = [];
    await assert.rejects(replaceDependencies(options, host({ review: true, torch: true }, asked)), /torch 2\.0\.0 has no build for Python 3\.12/);
    assert.deepEqual(asked, []); assert.deepEqual(calls.map(args => args[0]), ['plan-replacement']);
  });

  test('a changed model or environment after review never installs', async () => {
    let checks = 0;
    options.current = async () => ++checks < 3;
    await assert.rejects(replaceDependencies(options, host({ review: true, torch: true })), /changed after review/);
    assert.deepEqual(calls.map(args => args[0]), ['plan-replacement']);
  });
});
