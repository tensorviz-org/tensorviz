import * as assert from 'node:assert/strict';

import * as vscode from 'vscode';
import type { TestApi } from '../../extension';

const EXTENSION_ID = 'tensorviz.tensorviz-vscode';

/** The one thing only a real extension host can answer: does this manifest, compiled and
 *  loaded by VS Code itself, actually produce the command the palette offers?
 *
 *  A wrong `main`, a contributes/activate mismatch, or a top-level import that throws all
 *  package cleanly and then answer "command not found" at the moment a user tries it — and the
 *  unit run, which never loads the real `vscode` module, cannot see any of them. */
suite('activation', () => {
  test('activating the extension registers tensorviz.openCanvas', async function () {
    this.timeout(60_000);

    const extension = vscode.extensions.getExtension(EXTENSION_ID);
    assert.ok(extension, `${EXTENSION_ID} was not loaded by the extension host`);

    await extension.activate();
    assert.equal(extension.isActive, true);

    const commands = await vscode.commands.getCommands(true);
    assert.ok(
      commands.includes('tensorviz.openCanvas'),
      'tensorviz.openCanvas is contributed by package.json but was never registered',
    );
  });

  test('Get Started reveals the panel without a welcome editor or runtime setup', async function () {
    this.timeout(30_000);
    const extension = vscode.extensions.getExtension(EXTENSION_ID)!;
    const api = ((await extension.activate()) as { __test: TestApi }).__test;
    const tabs = vscode.window.tabGroups.all.flatMap(group => group.tabs);
    assert.equal(extension.packageJSON.contributes.walkthroughs, undefined);
    assert.equal(api.backendOrigin(), undefined);

    await vscode.commands.executeCommand('tensorviz.getStarted');
    const deadline = Date.now() + 20_000;
    while (!api.sidebarState()?.homeVisible && Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    assert.equal(api.sidebarState()?.homeVisible, true, 'the existing Canvas home should be visible');
    assert.deepEqual(vscode.window.tabGroups.all.flatMap(group => group.tabs), tabs,
      'Get Started should not open an editor tab');
    assert.equal(api.backendOrigin(), undefined, 'revealing the panel should not set up the runtime');
  });
});
