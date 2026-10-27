import * as assert from 'node:assert/strict';
import type * as VSCode from 'vscode';
import { isHostFile, matchesHostFile } from '../../workspaceFiles';
import { vscodeSourceIO } from '../../workspaceSources';
import { vscodeProjectSourceIO } from '../../projectSources';
import { WorkspaceClassIndex } from '../../workspaceClassIndex';
import { OpenAsGraphLenses } from '../../openAsGraphLenses';
import { shouldStartBackend } from '../../activation';
import { unsupportedRemote } from '../../remoteSupport';
import { Uri, workspace, state, reset } from '../vscodeStub';

// VS Code transforms the owning SSH authority to file: before workspace APIs.
// Raw strings and desktop-local/virtual resources must never alias that document.
const file = '/remote/project/model.py';
const uri = (scheme: string, authority = '') => ({ ...Uri.file(file), scheme, authority }) as VSCode.Uri;

suite('remote workspace document identity', () => {
  const restore: Array<() => void> = [];
  function patch(object: object, key: string, value: unknown) {
    const old = Object.getOwnPropertyDescriptor(object, key);
    Object.defineProperty(object, key, { configurable: true, value });
    restore.push(() => old ? Object.defineProperty(object, key, old) : Reflect.deleteProperty(object, key));
  }
  setup(() => reset());
  teardown(() => { for (const undo of restore.splice(0).reverse()) undo(); });

  test('only local desktop and SSH are admitted', () => {
    assert.equal(unsupportedRemote(undefined), undefined);
    assert.equal(unsupportedRemote('ssh-remote'), undefined);
    for (const kind of ['wsl', 'dev-container', 'codespaces', 'tunnel', 'unknown']) {
      assert.match(unsupportedRemote(kind)!, /not supported/);
    }
  });

  test('host files activate and discover with file selectors; raw foreign URIs do not', async () => {
    const trigger = { languageId: 'python', text: 'class Net(nn.Module):\n    pass',
      trusted: true, backendRunning: false, startFailed: false };
    state.workspaceFolders = [{ uri: Uri.file('/remote/project') }];
    let scans = 0;
    const index = new WorkspaceClassIndex(async root => {
      scans++; assert.equal(root, '/remote/project');
      return [{ name: 'Net', file: 'model.py', line: 1, bases: ['nn.Module'], ctor_required: 0 }];
    });
    try {
      for (const scheme of ['vscode-local', 'vscode-remote', 'untitled', 'git']) {
        const foreign = uri(scheme, 'ssh-remote+other');
        assert.equal(isHostFile(foreign), false);
        assert.equal(matchesHostFile(foreign, file), false);
        assert.equal(shouldStartBackend({ ...trigger, scheme }), false);
        await index.refresh(foreign);
      }
      assert.equal(scans, 0);
      assert.equal(shouldStartBackend({ ...trigger, scheme: 'file' }), true);
      await index.refresh(uri('file'));
      const lenses = new OpenAsGraphLenses(index).provideCodeLenses({ uri: uri('file'), languageId: 'python' } as VSCode.TextDocument);
      assert.equal(lenses.length, 1);
      assert.equal((lenses[0].command!.arguments![0] as { uri: string }).uri, uri('file').toString());
    } finally { index.dispose(); }
  });

  test('source snapshots read the remote dirty buffer despite same-path foreign documents', () => {
    const foreign = { uri: uri('vscode-local'), isDirty: true, version: 99, getText: () => 'wrong desktop bytes' };
    const remote = { uri: uri('file'), isDirty: true, version: 2, getText: () => 'remote unsaved bytes' };
    patch(workspace, 'textDocuments', [foreign, { ...foreign, uri: uri('vscode-remote', 'ssh-remote+other') }, remote]);
    for (const io of [vscodeSourceIO('/remote/project'), vscodeProjectSourceIO('/remote/project')]) {
      assert.equal(io.openDocument('model.py'), remote);
      assert.equal(io.openDocument('model.py')!.getText(), 'remote unsaved bytes');
    }
  });

  test('enumeration cannot import a foreign authority by stripping its path', async () => {
    patch(workspace, 'findFiles', async () => [uri('file'), uri('vscode-local'), uri('vscode-remote', 'ssh-remote+other')]);
    const vscode = require('../vscodeStub');
    patch(vscode, 'RelativePattern', class { constructor(public base: unknown, public pattern: string) {} });
    assert.deepEqual(await vscodeSourceIO('/remote/project').list(), ['model.py']);
    assert.deepEqual(await vscodeProjectSourceIO('/remote/project').list(), ['model.py']);
  });
});
