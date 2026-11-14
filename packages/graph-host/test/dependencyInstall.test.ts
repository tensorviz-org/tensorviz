import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { installDependencies, type DependencyInstallOptions } from '../dist/dependencyInstallCore';
import type { HelperResult } from '../dist/projectEnvironmentCore';

suite('dependency installation source-build review', () => {
  let storage: string, calls: string[][], plan: HelperResult, options: DependencyInstallOptions;
  setup(async () => {
    storage = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-install-')); calls = [];
    options = { storage, python: '/capture/bin/python', backendDir: '/backend', packages: ['torch==2.14.1', 'numpy'],
      signal: new AbortController().signal, current: async () => true,
      runner: async (_python, _backend, args) => { calls.push(args); return args[0] === 'plan-packages' ? plan : { ok: true }; } };
  });
  teardown(async () => { await fs.rm(storage, { recursive: true, force: true }); });
  const host = (asked: string[]) => ({ trusted: () => true,
    reviewSource: async () => { asked.push('source'); return true; },
    reviewInstall: async () => { asked.push('install'); return true; } });

  test('an unavailable PyTorch build is reported without asking to build it from source', async () => {
    // The helper classifies a missing PyTorch wheel this way because no PyTorch sdist exists.
    plan = { ok: false, code: 'PACKAGE_UNAVAILABLE', requirement: 'torch==2.14.1',
      message: 'The package index has no PyTorch build matching this requirement for macOS x86_64 with Python 3.12.13.' };
    const asked: string[] = [];
    await assert.rejects(installDependencies(options, host(asked)),
      (error: Error & { code?: string }) => error.code === 'PACKAGE_UNAVAILABLE' && /^torch==2\.14\.1: .*macOS x86_64/.test(error.message));
    assert.deepEqual(asked, []);
    assert.deepEqual(calls.map(args => args[0]), ['plan-packages']);
  });

  test('a package with no compatible wheel still asks before a source build', async () => {
    plan = { ok: false, code: 'NO_COMPATIBLE_WHEEL', requirement: 'fairscale' };
    const asked: string[] = [];
    await assert.rejects(installDependencies({ ...options, packages: ['fairscale'] }, host(asked)));
    assert.deepEqual(asked, ['source']);
    assert.deepEqual(calls.map(args => args[0]), ['plan-packages', 'check-package-availability', 'plan-packages-source']);
  });
});
