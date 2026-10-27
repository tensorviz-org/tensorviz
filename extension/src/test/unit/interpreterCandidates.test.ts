import * as assert from 'node:assert/strict';
import * as path from 'node:path';

import { interpreterCandidates, missingModuleFrom, repoRootFor } from '../../preflight';
import { reset, state } from '../vscodeStub';

const REPO = path.join(path.sep, 'repos', 'canvas');

const VENV =
  process.platform === 'win32'
    ? path.join(REPO, 'venv', 'Scripts', 'python.exe')
    : path.join(REPO, 'venv', 'bin', 'python');

/** Install a fake ms-python whose environments API answers with the given paths. */
function fakePythonExtension(active: string, resolvedExecutable?: string): void {
  state.extensions.set('ms-python.python', {
    async activate() {
      return {
        environments: {
          getActiveEnvironmentPath: () => ({ id: 'fake', path: active }),
          resolveEnvironment: async () =>
            resolvedExecutable ? { executable: { uri: { fsPath: resolvedExecutable } } } : undefined,
        },
      };
    },
  });
}

/** Service candidates must not borrow the model's Python selection. A dependency-poor
 *  project environment must still be able to open the service and show setup recovery. */
suite('preflight.interpreterCandidates', () => {
  let savedBackendPython: string | undefined;

  setup(() => {
    reset();
    savedBackendPython = process.env.BACKEND_PYTHON;
    delete process.env.BACKEND_PYTHON;
  });

  teardown(() => {
    if (savedBackendPython === undefined) {
      delete process.env.BACKEND_PYTHON;
    } else {
      process.env.BACKEND_PYTHON = savedBackendPython;
    }
  });

  test('service precedence excludes ms-python: BACKEND_PYTHON, repo venv, python3, python', async () => {
    fakePythonExtension('/envs/viz', '/envs/viz/bin/python');
    process.env.BACKEND_PYTHON = '/opt/backend/python';

    assert.deepEqual(await interpreterCandidates(REPO), [
      '/opt/backend/python',
      VENV,
      'python3',
      'python',
    ]);
  });

  test('a resolved project environment cannot replace the service runtime', async () => {
    fakePythonExtension('/envs/viz', '/envs/viz/bin/python');
    assert.deepEqual(await interpreterCandidates(REPO, '/managed/bin/python'), [
      '/managed/bin/python', VENV, 'python3', 'python',
    ]);
  });

  test('an unresolved project environment is not a service fallback', async () => {
    fakePythonExtension('/envs/viz/bin/python', undefined);
    assert.deepEqual(await interpreterCandidates(REPO), [VENV, 'python3', 'python']);
  });

  test('drops sources that are absent rather than emitting an empty candidate', async () => {
    // No ms-python installed and no BACKEND_PYTHON: spawning '' would fail with a message
    // about the empty string rather than about the missing interpreter.
    assert.deepEqual(await interpreterCandidates(REPO), [VENV, 'python3', 'python']);
  });

  test('an ms-python that throws does not take the whole list down', async () => {
    state.extensions.set('ms-python.python', {
      async activate() {
        throw new Error('activation failed');
      },
    });
    assert.deepEqual(await interpreterCandidates(REPO), [VENV, 'python3', 'python']);
  });

  test('tensorviz.pythonPath is excluded from the service runtime candidates', async () => {
    reset();
    state.configuration.set('tensorviz.pythonPath', '/opt/explicit/bin/python');

    const candidates = await interpreterCandidates('/repo', '/managed/bin/python');

    assert.equal(candidates[0], '/managed/bin/python');
    assert.ok(!candidates.includes('/opt/explicit/bin/python'),
      'a capture setting must not select or repair the service runtime');
  });

  test('the venv beside the CHECKOUT is tried, not only the one beside the backend dir', async () => {
    // P0 changed what callers pass: `resolveBackendDir` hands over a BACKEND directory
    // (`<repo>/backend`, or `<extension>/backend` from the .vsix) where the repository
    // root used to be handed over. docs/local-setup.md tells people to build the venv at the
    // checkout root, so probing only the argument silently stopped trying the one venv the
    // documentation asks for — and the failure message then told them to build another one
    // somewhere that would not be tried either.
    reset();
    const backendDir = path.join(REPO, 'backend');
    const rootVenv =
      process.platform === 'win32'
        ? path.join(REPO, 'venv', 'Scripts', 'python.exe')
        : path.join(REPO, 'venv', 'bin', 'python');
    const backendVenv =
      process.platform === 'win32'
        ? path.join(backendDir, 'venv', 'Scripts', 'python.exe')
        : path.join(backendDir, 'venv', 'bin', 'python');

    const candidates = await interpreterCandidates(backendDir);

    assert.ok(candidates.includes(rootVenv), `the checkout-root venv ${rootVenv} was never tried`);
    assert.ok(candidates.includes(backendVenv), 'the backend-dir venv is still tried');
    assert.ok(
      candidates.indexOf(backendVenv) < candidates.indexOf(rootVenv),
      'the nearer venv should be preferred to the one a directory up',
    );
  });

  test('a candidate is never offered twice', async () => {
    // The two venv roots collapse to one path when a backend directory is its own parent.
    reset();
    const candidates = await interpreterCandidates(path.sep);
    assert.equal(new Set(candidates).size, candidates.length, candidates.join(', '));
  });

  test('an unset tensorviz.pythonPath changes nothing', async () => {
    // The other half: adding a setting must not reorder or pad the list for the
    // people who never touch it, or the managed runtime stops being first for the
    // fork, which is the promise it exists to keep.
    reset();

    const candidates = await interpreterCandidates('/repo', '/managed/bin/python');

    assert.equal(candidates[0], '/managed/bin/python');
    assert.ok(!candidates.includes(''), 'an empty setting leaked into the candidates');
  });
});

/** ide-plan §15 item 5: the first-run failure told everyone to install torch, whichever of
 *  torch, uvicorn and fastapi was actually missing, and pointed at a venv path that P0's
 *  backend-directory change had made wrong. Both are pure functions, so both are pinned here. */
suite('preflight.missingModuleFrom / repoRootFor', () => {
  test('the module Python named is the module we name', () => {
    assert.equal(missingModuleFrom("ModuleNotFoundError: No module named 'fastapi'"), 'fastapi');
    assert.equal(missingModuleFrom("No module named 'torch'"), 'torch');
    // A submodule failure is reported by its top-level package: telling someone to install
    // `torch.distributed` would send them looking for a package that does not exist.
    assert.equal(missingModuleFrom("No module named 'torch.distributed'"), 'torch');
  });

  test('a failure that is not a missing module names nothing rather than guessing', () => {
    assert.equal(missingModuleFrom('Segmentation fault'), undefined);
    assert.equal(missingModuleFrom(''), undefined);
  });

  test('advice points at the checkout root, not at backend/', () => {
    assert.equal(repoRootFor(path.join(REPO, 'backend')), REPO);
    // A packaged copy is also named backend, so its parent is the extension root.
    const staged = path.join(path.sep, 'ext', 'backend');
    assert.equal(repoRootFor(staged), path.dirname(staged));
  });
});
