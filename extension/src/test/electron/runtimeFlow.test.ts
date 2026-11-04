/** Step 5 Electron legs: the managed-runtime first-install, refusal, and reuse flows in a
 *  real extension host.
 *
 * Why this suite exists separately from the unit lifecycle suite: consent runs through real
 *  vscode.window notifications, preflight's resume of the command that asked is the product
 *  contract, and "does the second run ask again" is only honestly answerable against the
 *  actual checkEnvironment flow — a stubbed answer says nothing about what a user sees.
 *
 * Deterministic and offline:
 *  - the runtime archive is fabricated in-test (scripted interpreter tar, same builder as the
 *    unit suites) and streamed through TENSORVIZ_TEST_RUNTIME_ARCHIVE;
 *  - the manifest the host reads is a copy re-pinned to that archive
 *    (TENSORVIZ_TEST_RUNTIME_MANIFEST), so the shipped pin never moves and — critically for
 *    leg 3 — the fabricated provision is never stale;
 *  - TENSORVIZ_TEST_PRESEED_MANAGED is turned OFF for these legs: the preseed exists so that
 *    every OTHER Electron suite runs against a ready runtime instead of deadlocking on the
 *    consent modal; this suite is precisely about what happens when nobody preseeded.
 */

import * as assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

import * as vscode from 'vscode';

import { archive, gz } from '../../../../tests/fixtures/tar.cjs';
import { SCRIPTED_PYTHON } from '../unit/stockRuntimeFixture';

const EXTENSION_ID = 'tensorviz.tensorviz-vscode';
/** The checkout, four directories up from out/test/electron. */
const REPO = path.resolve(__dirname, '..', '..', '..', '..');
const CONSENT_SUBSTRING = 'Install tensorViz’s runtime';

interface TestApi {
  backendOrigin(): string | undefined;
  globalStorageDir(): string | undefined;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

suite('runtimeFlow', () => {
  let temp: string;
  let runtimeRoot: string;
  let api: TestApi;
  /** The capture spy answers the first-install consent from this variable; every other
   *  notification is recorded and dismissed. */
  let answerConsent: 'Install' | 'Cancel' | 'dismiss';
  /** Every notification text, in order. */
  const shown: string[] = [];
  let consentCount: number;
  const restore: Array<() => void | Thenable<void>> = [];

  function captureNotifications(): void {
    const surface = vscode.window as unknown as Record<string, unknown>;
    const originalQuickPick = vscode.window.showQuickPick;
    surface.showQuickPick = async (...args: unknown[]) => {
      const options = args[1] as vscode.QuickPickOptions | undefined;
      if (options?.title === 'Choose the Python environment for tensorViz') {
        const items = await (args[0] as Thenable<ReadonlyArray<{ choice?: { kind: string } }>>);
        const managed = items.find((item) => item.choice?.kind === 'managed');
        assert.ok(managed, 'the separate managed environment must remain an explicit choice');
        return managed;
      }
      return Reflect.apply(originalQuickPick, vscode.window, args);
    };
    restore.push(() => { surface.showQuickPick = originalQuickPick; });
    for (const name of [
      'showInformationMessage',
      'showWarningMessage',
      'showErrorMessage',
    ] as const) {
      const surface = vscode.window as unknown as Record<string, unknown>;
      const original = surface[name];
      const spy = (message: unknown): Thenable<string | undefined> => {
        const text = String(message);
        shown.push(text);
        if (text.includes(CONSENT_SUBSTRING)) {
          consentCount += 1;
          // modal prompts resolve to the picked BUTTON; dismissal is undefined, which the
          // product must treat as decline (silence never means consent).
          return Promise.resolve(answerConsent === 'dismiss' ? undefined : answerConsent);
        }
        return Promise.resolve(undefined);
      };
      surface[name] = spy;
      restore.push(() => {
        surface[name] = original;
      });
    }
  }

  suiteSetup(async function () {
    this.timeout(120_000);
    if (process.platform === 'win32') {
      // The scripted interpreter is a POSIX shell script; the lifecycle equivalent of these
      // legs is the host-independent unit suite, and the real-runtime Electron proof is POSIX
      // in this repository today (the archive carries an sh interpreter).
      this.skip();
    }
    const python = process.env.TENSORVIZ_TEST_PYTHON;
    assert.ok(python, 'not run: TENSORVIZ_TEST_PYTHON unset');
    for (const key of [
      'BACKEND_PYTHON', 'TENSORVIZ_TEST_PRESEED_MANAGED',
      'TENSORVIZ_TEST_RUNTIME_ARCHIVE', 'TENSORVIZ_TEST_RUNTIME_MANIFEST',
    ]) {
      const value = process.env[key];
      restore.push(() => {
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      });
    }
    process.env.BACKEND_PYTHON = python;
    // Disable preseed before activation, not after replacing a runtime another
    // suite's live backend may still own. The runner gives this suite first use.
    process.env.TENSORVIZ_TEST_PRESEED_MANAGED = '0';

    temp = fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-runtime-flow-'));

    const extension = vscode.extensions.getExtension(EXTENSION_ID);
    assert.ok(extension, `${EXTENSION_ID} was not loaded by the extension host`);
    api = ((await extension.activate()) as { __test: TestApi }).__test;
    const globalStorage = api.globalStorageDir();
    assert.ok(globalStorage, 'no globalStorage directory from the test api');
    runtimeRoot = path.join(globalStorage, 'runtime');
    assert.equal(api.backendOrigin(), undefined, 'runtimeFlow must run before a backend owns the runtime');
    assert.ok(!fs.existsSync(runtimeRoot), 'runtimeFlow requires a fresh test profile, not an existing runtime');

    // The backend against which provisioning's verify probe imports. resolveBackendDir needs
    // the checkout; localBrowserGraph's suite does the same.
    const settings = vscode.workspace.getConfiguration('tensorviz');
    const repoPath = settings.inspect('repoPath')?.globalValue;
    restore.push(() => settings.update('repoPath', repoPath, vscode.ConfigurationTarget.Global));
    await settings.update('repoPath', REPO, vscode.ConfigurationTarget.Global);

    captureNotifications();

    // The offline runtime the install legs provision: scripted interpreter archive plus the
    // shipped manifest re-pinned to it (build BEFORE any command runs so the seams are
    // complete when the first consent is answered).
    const tarball = gz(
      archive([
        { name: 'python/', type: '5', mode: 0o755 },
        { name: 'python/bin', type: '5', mode: 0o755 },
        // Keep archive/venv operations offline, but run the real tool service and
        // environment helpers now required by Check Environment.
        { name: 'python/bin/python3', mode: 0o755, data: Buffer.from(
          SCRIPTED_PYTHON.slice(0, SCRIPTED_PYTHON.lastIndexOf('exit 0')) + 'exec "$TENSORVIZ_TEST_PYTHON" "$@"\n',
        ) },
      ]),
    );
    const assetPath = path.join(temp, 'asset.tar.gz');
    fs.writeFileSync(assetPath, tarball);
    const manifest = JSON.parse(
      fs.readFileSync(path.join(REPO, 'extension', 'runtime-manifest.json'), 'utf8'),
    ) as { platforms: Record<string, { sha256: string }> };
    const key = `${process.platform}-${process.arch}`;
    assert.ok(manifest.platforms[key], `no manifest platform entry for ${key}`);
    manifest.platforms[key] = {
      ...manifest.platforms[key],
      sha256: createHash('sha256').update(tarball).digest('hex'),
    };
    const manifestPath = path.join(temp, 'runtime-manifest.json');
    fs.writeFileSync(manifestPath, JSON.stringify(manifest));
    process.env.TENSORVIZ_TEST_RUNTIME_ARCHIVE = assetPath;
    process.env.TENSORVIZ_TEST_RUNTIME_MANIFEST = manifestPath;
  });

  suiteTeardown(async () => {
    // Restore every override even after a partial setup or failed assertion, and
    // report cleanup failures rather than masking them behind the first error.
    const errors: unknown[] = [];
    // The runner removes this host's profile after VS Code stops its backend.
    // Do not replace an interpreter that still owns the service process.
    for (const undo of restore.reverse()) {
      try { await undo(); } catch (error) { errors.push(error); }
    }
    try {
      if (temp) fs.rmSync(temp, { recursive: true, force: true });
    } catch (error) { errors.push(error); }
    if (errors.length) throw new AggregateError(errors, 'runtimeFlow fixture cleanup failed');
  });

  test('declining the managed fallback installs nothing and does not select a different environment', async function () {
    this.timeout(240_000);
    answerConsent = 'dismiss';
    consentCount = 0;

    await vscode.commands.executeCommand('tensorviz.checkEnvironment');

    assert.equal(consentCount, 1, 'the fresh host asked for first-install consent exactly once');
    assert.ok(
      !fs.existsSync(runtimeRoot),
      'declining must not create the runtime root — nothing downloads, nothing installs',
    );
    const banner = shown
      .filter((text) => text.startsWith('python OK:'))
      .pop();
    assert.equal(banner, undefined, 'declining must not silently start another environment');
  });

  test('accepting first-install provisions the verified runtime and resumes the command', async function () {
    this.timeout(240_000);
    answerConsent = 'Install';
    consentCount = 0;

    await vscode.commands.executeCommand('tensorviz.checkEnvironment');

    assert.equal(consentCount, 1);
    assert.ok(
      fs.existsSync(path.join(runtimeRoot, 'venv')) &&
        fs.existsSync(path.join(runtimeRoot, '.tensorviz-provisioned.json')),
      'a complete install leaves the venv and the provision stamp',
    );
    const banner = shown.filter((text) => text.startsWith('python OK:')).pop();
    assert.ok(banner, `the command resumed after provisioning and re-probed; saw ${shown.join(' | ')}`);
    assert.ok(
      banner.includes('tensorViz managed runtime'),
      'the resumed command ran on the runtime that was just installed',
    );
    assert.ok(
      !banner.includes(process.env.BACKEND_PYTHON as string),
      'the env fallback must not win once the managed runtime exists',
    );
  });

  test('the second use reuses the verified runtime without re-asking', async function () {
    this.timeout(240_000);
    answerConsent = 'dismiss';
    consentCount = 0;
    shown.length = 0;

    await vscode.commands.executeCommand('tensorviz.checkEnvironment');

    assert.equal(
      consentCount,
      0,
      'a ready, previously provisioned runtime must not raise first-install consent',
    );
    const banner = shown.filter((text) => text.startsWith('python OK:')).pop();
    assert.ok(banner?.includes('tensorViz managed runtime'), 'the reused runtime answered');
  });

  test('Repair Runtime rebuilds into staging and reports the swap', async function () {
    this.timeout(240_000);
    shown.length = 0;

    await vscode.commands.executeCommand('tensorviz.repairRuntime');

    assert.ok(
      shown.includes('The tensorViz runtime is ready.'),
      `repair must confirm it completed; saw: ${shown.join(' | ')}`,
    );
    assert.ok(fs.existsSync(path.join(runtimeRoot, 'venv')), 'the repaired venv is in service');
    const leftovers = fs
      .readdirSync(runtimeRoot)
      .filter((entry) => /^(staging|old)-/.test(entry));
    assert.deepEqual(leftovers, [], 'a completed repair leaves no staging behind');
  });
});
