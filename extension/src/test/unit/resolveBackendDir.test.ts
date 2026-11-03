import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

import type * as vscode from 'vscode';

import { REPO_PATH_MEMENTO, resolveBackendDir } from '../../preflight';
import { RUNTIME_STAMP } from '../../runtime';
import { reset, state } from '../vscodeStub';

/**
 * Rewritten from `resolveRepoPath.test.ts` for task.md P0.
 *
 * The old function demanded a canvas CHECKOUT — both `backend/server_v2.py` and
 * `canvas/package.json` — before the extension would start anything. That made "clone
 * our repository" a prerequisite for opening your own model, which is the single largest
 * piece of ceremony Phase 2 exists to delete. The React build has shipped inside the vsix
 * since Stage A, so the frontend marker was checking for something the extension no longer
 * needs at all.
 *
 * What replaces it resolves a BACKEND DIRECTORY, and the two sources that come first — the
 * app bundle and the vsix — depend on nothing the user has to do. `tensorviz.repoPath`
 * survives underneath them as a developer override (D21): the Electron suites and the owner
 * run from a checkout, and no user is ever told the setting exists.
 */

const tempDirs: string[] = [];

function tempDir(prefix: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

/** A directory holding `server_v2.py`, i.e. one that validates as a backend. */
function makeBackend(parent: string, name = 'backend'): string {
  const dir = path.join(parent, name);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'server_v2.py'), '');
  return dir;
}

/** A TensorViz checkout, as `tensorviz.repoPath` and a workspace folder are read. */
function makeCheckout(complete = true): string {
  const dir = tempDir('tensorviz-repo-');
  if (complete) {
    makeBackend(dir, 'backend');
  } else {
    fs.mkdirSync(path.join(dir, 'backend'), { recursive: true });
  }
  return dir;
}

/** An extension directory laid out the way `ide/build.sh` stages a fork bundle.
 *
 * `bundledLayout` walks two levels up from the extension, so the built-in has to sit at
 * `<app>/extensions/tensorviz` for `<app>/runtime` and `<app>/backend` to be found. */
function makeBundle(options: { stamp: boolean; backend: boolean }): string {
  const app = tempDir('tensorviz-app-');
  const extensionPath = path.join(app, 'extensions', 'tensorviz');
  fs.mkdirSync(extensionPath, { recursive: true });
  fs.mkdirSync(path.join(app, 'runtime'), { recursive: true });
  if (options.stamp) {
    fs.writeFileSync(path.join(app, 'runtime', RUNTIME_STAMP), '{}');
  }
  if (options.backend) {
    makeBackend(app, 'backend');
  }
  return extensionPath;
}

interface FakeContext {
  extensionPath: string;
  globalState: {
    get<T>(key: string): T | undefined;
    update(key: string, value: unknown): Thenable<void>;
  };
  updates: { key: string; value: unknown }[];
}

function fakeContext(extensionPath: string, remembered?: string): FakeContext {
  const store = new Map<string, unknown>();
  if (remembered !== undefined) {
    store.set(REPO_PATH_MEMENTO, remembered);
  }
  const ctx: FakeContext = {
    extensionPath,
    globalState: {
      get<T>(key: string): T | undefined {
        return store.get(key) as T | undefined;
      },
      async update(key: string, value: unknown): Promise<void> {
        ctx.updates.push({ key, value });
        if (value === undefined) {
          store.delete(key);
        } else {
          store.set(key, value);
        }
      },
    },
    updates: [],
  };
  return ctx;
}

function resolve(ctx: FakeContext) {
  return resolveBackendDir(ctx as unknown as vscode.ExtensionContext);
}

suite('preflight.resolveBackendDir', () => {
  setup(reset);

  suiteTeardown(() => {
    for (const dir of tempDirs) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  test('a directory holding server_v2.py alone is accepted', () => {
    // The single assertion that says the checkout requirement is gone. There is no
    // canvas/package.json anywhere near this directory and it still resolves.
    const extensionPath = tempDir('tensorviz-ext-');
    makeBackend(extensionPath);

    const result = resolve(fakeContext(extensionPath));
    assert.equal(result.ok, true);
    if (result.ok) {
      assert.equal(result.backendDir, path.join(extensionPath, 'backend'));
    }
  });

  test('the bundle wins over the vsix, which wins over repoPath', () => {
    // The order is the contract. A user running the fork must get the backend that was
    // built and verified with their app, not one left behind by a developer setting.
    const extensionPath = makeBundle({ stamp: true, backend: true });
    const app = path.resolve(extensionPath, '..', '..');
    makeBackend(extensionPath);
    const checkout = makeCheckout();
    state.configuration.set('tensorviz.repoPath', checkout);

    const bundleFirst = resolve(fakeContext(extensionPath));
    assert.equal(bundleFirst.ok, true);
    if (bundleFirst.ok) {
      assert.equal(bundleFirst.backendDir, path.join(app, 'backend'));
      assert.match(bundleFirst.source, /bundled/);
    }

    // Remove only the stamp: two directories at the right relative path are not a
    // tensorViz IDE bundle, so this must fall to the vsix rather than accept them.
    fs.rmSync(path.join(app, 'runtime', RUNTIME_STAMP));
    const vsixSecond = resolve(fakeContext(extensionPath));
    assert.equal(vsixSecond.ok, true);
    if (vsixSecond.ok) {
      assert.equal(vsixSecond.backendDir, path.join(extensionPath, 'backend'));
      assert.match(vsixSecond.source, /inside the extension/);
    }

    // And with no backend in the vsix either, the developer override is what is left.
    fs.rmSync(path.join(extensionPath, 'backend'), { recursive: true });
    const settingThird = resolve(fakeContext(extensionPath));
    assert.equal(settingThird.ok, true);
    if (settingThird.ok) {
      assert.equal(settingThird.backendDir, path.join(checkout, 'backend'));
      assert.match(settingThird.source, /setting/);
    }
  });

  test('a configured path that does not validate HARD FAILS, never falls through', () => {
    // Unchanged in spirit from the old test, and still the point of the function.
    // Starting a backend against a different tree is the worst outcome available:
    // everything comes up fine and is merely wrong.
    const extensionPath = tempDir('tensorviz-ext-');
    const broken = makeCheckout(false);
    const usable = makeCheckout();
    state.configuration.set('tensorviz.repoPath', broken);
    state.workspaceFolders = [{ uri: { fsPath: usable } }];

    const result = resolve(fakeContext(extensionPath, usable));
    assert.equal(result.ok, false);
    if (!result.ok) {
      // No picker either: the user typed a path, so the fix is that path, not a dialog.
      assert.equal(result.offerPicker, false);
      assert.match(result.detail, /server_v2\.py/);
      assert.equal(result.detail.includes(usable), false, 'must not suggest a different tree');
    }
  });

  test('an open checkout still works with no setting at all', () => {
    // The plain "open the TensorViz repo in VS Code" flow, which must not need a
    // setting to keep working.
    const extensionPath = tempDir('tensorviz-ext-');
    const folder = makeCheckout();
    state.workspaceFolders = [{ uri: { fsPath: makeCheckout(false) } }, { uri: { fsPath: folder } }];

    const result = resolve(fakeContext(extensionPath));
    assert.equal(result.ok, true);
    if (result.ok) {
      assert.equal(result.backendDir, path.join(folder, 'backend'));
      assert.match(result.source, /workspace folder/);
    }
  });

  test('a stale remembered folder is deleted, not just skipped', () => {
    // globalState has no user-facing editor, so a remembered-but-wrong path that survived
    // would be both wrong and unfixable by the user.
    const extensionPath = tempDir('tensorviz-ext-');
    const stale = makeCheckout(false);
    const ctx = fakeContext(extensionPath, stale);

    const result = resolve(ctx);
    assert.equal(result.ok, false);
    assert.deepEqual(ctx.updates, [{ key: REPO_PATH_MEMENTO, value: undefined }]);
    assert.equal(ctx.globalState.get(REPO_PATH_MEMENTO), undefined);
  });

  test('with nothing to go on it offers the picker and says what it looked at', () => {
    const extensionPath = tempDir('tensorviz-ext-');
    const result = resolve(fakeContext(extensionPath));
    assert.equal(result.ok, false);
    if (!result.ok) {
      assert.equal(result.offerPicker, true);
      assert.match(result.detail, /the backend bundled with this app/);
      assert.match(result.detail, /the backend inside this extension/);
      assert.match(result.detail, /tensorviz\.repoPath/);
      assert.match(result.detail, /server_v2\.py/);
    }
  });
});
