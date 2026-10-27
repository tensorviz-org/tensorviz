import * as assert from 'node:assert/strict';
import * as vscode from 'vscode';
import { pickWorkspaceModel } from '../../workspaceModelPicker';
import { reset, state } from '../vscodeStub';

suite('workspace model picker', () => {
  const folders = [{ uri: vscode.Uri.file('/first'), name: 'first', index: 0 },
    { uri: vscode.Uri.file('/second'), name: 'second', index: 1 }];
  const model = { name: 'Net', file: 'models/net.py', line: 7, bases: ['nn.Module'], ctor_required: 0 };
  setup(() => reset());

  test('same-named models carry their owning folder and file through explicit selection', async () => {
    const original = vscode.window.showQuickPick;
    const roots: string[] = [];
    vscode.window.showQuickPick = (async (items: any[], options: any) => {
      assert.equal(options.matchOnDescription, true);
      assert.deepEqual(items.slice(0, 2).map(item => item.description), ['first/models/net.py', 'second/models/net.py']);
      return items[1];
    }) as never;
    try {
      assert.deepEqual(await pickWorkspaceModel(folders, async root => { roots.push(root); return [model]; }),
        { uri: vscode.Uri.joinPath(folders[1].uri, model.file).toString(), cls: 'Net' });
      assert.deepEqual(roots, ['/first', '/second']);
    } finally { vscode.window.showQuickPick = original; }
  });

  test('Escape does not automatically open even the only discovered model', async () => {
    state.quickPickAnswers.push(undefined);
    assert.equal(await pickWorkspaceModel(folders.slice(0, 1), async () => [model]), undefined);
  });

  test('browsing an undiscovered model returns only its file for the existing class chooser', async () => {
    state.quickPickAnswers.push('Choose a Python file…');
    state.openDialogResult = [vscode.Uri.file('/first/custom.py')];
    assert.deepEqual(await pickWorkspaceModel(folders.slice(0, 1), async () => []), { uri: vscode.Uri.file('/first/custom.py').toString() });
  });

  test('an empty window can browse without claiming a successful workspace scan', async () => {
    state.openDialogResult = [vscode.Uri.file('/outside/net.py')];
    assert.deepEqual(await pickWorkspaceModel([], async () => { throw new Error('must not scan'); }),
      { uri: vscode.Uri.file('/outside/net.py').toString() });
    state.openDialogResult = undefined;
    assert.equal(await pickWorkspaceModel([], async () => []), undefined);
  });

  test('a failed scan is not presented as no models found', async () => {
    await assert.rejects(pickWorkspaceModel(folders, async () => { throw new Error('scan unavailable'); }), /scan unavailable/);
  });
});
