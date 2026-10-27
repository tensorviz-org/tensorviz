import * as assert from 'node:assert/strict';

import * as vscode from 'vscode';

const EXTENSION_ID = 'tensorviz.tensorviz';

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
});
