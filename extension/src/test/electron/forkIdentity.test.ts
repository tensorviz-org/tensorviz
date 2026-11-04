import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';

import * as vscode from 'vscode';

const EXTENSION_ID = 'tensorviz.tensorviz-vscode';

/** The identity a bundle must have to be ours. Every value here is the `tensorviz` side of an
 *  entry in `ide/expected-product.json`; `ide/scripts/verify-product.py` asserts the same values
 *  by reading product.json off disk, and this suite asserts what the *running* application then
 *  reports for them, which is a different question. A build can carry a correct product.json and
 *  still launch as "Code - OSS" if a patch stopped applying to the file the workbench reads. */
const APP_NAME = 'tensorViz IDE';
const URI_SCHEME = 'tensorviz';

/** Where gulp stages our built-in inside the app directory, from `builtinExtension` in
 *  ide/expected-product.json. `vscode.env.appRoot` is that directory as the running app sees it. */
const BUILTIN_MANIFEST = path.join('extensions', 'tensorviz', 'package.json');

/** Does the fork we ship actually *behave* as the fork we ship?
 *
 *  These tests only mean anything inside a built bundle, so they run only when
 *  `TENSORVIZ_IDE_BINARY` points `.vscode-test.mjs` at one (`ide/scripts/smoke.sh
 *  --print-binary <bundle>` prints the value). Against a downloaded stock VS Code every
 *  assertion here would be false, and asserting them anyway would turn "no fork was built" into
 *  a red suite that says nothing about our code.
 *
 *  The skip is therefore deliberate and it is LOUD: it prints why it skipped. That is the one
 *  difference from the backend suites, which fail rather than skip when `TENSORVIZ_TEST_PYTHON`
 *  is unset (task.md M2's test-infrastructure rule) — an interpreter is something a job can
 *  always be given, a fork bundle is an hour-long build that only some jobs do. What the rule
 *  forbids is a suite that skips in silence and reads as proof, so this one says so out loud. */
suite('forkIdentity', () => {
  suiteSetup(function () {
    if (!process.env.TENSORVIZ_IDE_BINARY) {
      console.log(
        'forkIdentity: SKIPPED — TENSORVIZ_IDE_BINARY is unset, so this run is a downloaded ' +
          'stock VS Code and there is no fork identity to assert. Build a bundle and run: ' +
          'TENSORVIZ_IDE_BINARY="$(ide/scripts/smoke.sh --print-binary ide/VSCode-darwin-arm64)" ' +
          'npm run test:electron',
      );
      this.skip();
    }
  });

  test('the running application calls itself tensorViz IDE', () => {
    assert.equal(
      vscode.env.appName,
      APP_NAME,
      `the app under test reports appName ${JSON.stringify(vscode.env.appName)}. Either ` +
        `TENSORVIZ_IDE_BINARY points at something that is not our bundle, or the nameLong/nameShort ` +
        `patch stopped applying and the build is Code - OSS wearing our icon.`,
    );
  });

  test('the running application owns the tensorviz url protocol', () => {
    // urlProtocol is what `tensorviz://` links resolve to and what the OS registers. Sharing
    // `code-oss` with an upstream build is exactly the collision expected-product.json's
    // "must differ from codeOss" rule exists to prevent, and only a launched app can say
    // which scheme it actually took.
    assert.equal(
      vscode.env.uriScheme,
      URI_SCHEME,
      `the app under test registers the ${JSON.stringify(vscode.env.uriScheme)} uri scheme, not ` +
        `${JSON.stringify(URI_SCHEME)}.`,
    );
  });

  test('the bundle carries tensorViz as a built-in and the extension host activates it', async function () {
    this.timeout(60_000);

    assert.ok(vscode.env.appRoot, 'vscode.env.appRoot is empty, so there is no app directory to look in');
    const manifestPath = path.join(vscode.env.appRoot, BUILTIN_MANIFEST);
    assert.ok(
      fs.existsSync(manifestPath),
      `no built-in manifest at ${manifestPath}. The bundle booted, so the fork built — but our ` +
        `extension was not staged into it, which is the whole point of forking.`,
    );
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    assert.equal(`${manifest.publisher}.${manifest.name}`, EXTENSION_ID);

    // The copy the host loads is NOT necessarily that one: the test runner passes
    // --extensionDevelopmentPath, and a development extension shadows a built-in with the same
    // id. So this asserts the two facts that are ours to assert here — the bundle ships the
    // built-in, and the extension host in this bundle can load and activate that id — and
    // prints which copy won, because a reader of a passing log should not have to guess.
    // `ide/scripts/smoke.sh` step 2 is what proves the built-in copy alone is scannable, with
    // no development path in the picture.
    const extension = vscode.extensions.getExtension(EXTENSION_ID);
    assert.ok(extension, `${EXTENSION_ID} was not loaded by the fork's extension host`);
    await extension.activate();
    assert.equal(extension.isActive, true);
    console.log(`forkIdentity: the active copy of ${EXTENSION_ID} is ${extension.extensionPath}`);
  });

  test('VS Code usage reporting is off, as product.json says it is', () => {
    // `enableTelemetry: false` in product.json is what makes this false: with it, the extension
    // host reports usage collection unsupported no matter what the user's settings say.
    assert.equal(
      vscode.env.isTelemetryEnabled,
      false,
      'usage reporting is enabled inside the fork, so enableTelemetry:false is no longer in the built product.json.',
    );
  });
});
