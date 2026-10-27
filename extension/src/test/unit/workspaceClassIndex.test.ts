import * as assert from 'node:assert/strict';
import * as path from 'node:path';

import { WorkspaceClassIndex, WorkspaceClass } from '../../workspaceClassIndex';
import { OpenAsGraphLenses } from '../../openAsGraphLenses';
import { Uri, reset, state } from '../vscodeStub';

const ROOT = path.join(path.sep, 'ws');

function scanned(...classes: Array<Partial<WorkspaceClass>>): WorkspaceClass[] {
  return classes.map((entry) => ({
    name: 'Net',
    file: 'models/resid.py',
    line: 10,
    bases: ['nn.Module'],
    ctor_required: 0,
    ...entry,
  }));
}

suite('WorkspaceClassIndex', () => {
  setup(() => {
    reset();
    state.workspaceFolders = [{ uri: { fsPath: ROOT } }];
  });

  test('asks the backend once per root, not once per keystroke', async () => {
    let calls = 0;
    const lenses = new WorkspaceClassIndex(async () => {
      calls += 1;
      return scanned({});
    });
    await lenses.refresh(Uri.file(path.join(ROOT, 'models', 'resid.py')) as never);

    for (let i = 0; i < 5; i += 1) {
      lenses.classesFor(ROOT);
    }

    assert.equal(calls, 1, 'reading the index never re-scans');
  });

  test('keeps the classes it has when a scan fails', async () => {
    // The normal case, not an exceptional one: the backend is not running until
    // the canvas is opened, and every save asks again. A failed scan that wiped
    // the index would empty the Models view on the next save.
    let fail = false;
    const lenses = new WorkspaceClassIndex(async () => {
      if (fail) throw new Error('backend not running');
      return scanned({ name: 'Net', line: 14 });
    });
    await lenses.refresh(Uri.file(path.join(ROOT, 'models', 'resid.py')) as never);

    fail = true;
    await lenses.refresh(Uri.file(path.join(ROOT, 'models', 'resid.py')) as never);

    assert.equal(lenses.classesFor(ROOT)?.length, 1);
  });

  test('falls back to the file’s own directory when no folder is open', () => {
    state.workspaceFolders = undefined;
    assert.equal(
      WorkspaceClassIndex.rootFor(Uri.file(path.join(ROOT, 'models', 'resid.py')) as never),
      path.join(ROOT, 'models'),
    );
  });

  test('a scan that could not run leaves the index unset, not empty', async () => {
    // The difference between "this folder has no models" and "nobody has looked"
    // is the difference between a true sentence and a false one. The Models view
    // reads exactly this: an unset index means "Start tensorViz to scan this
    // folder", an empty one means "No nn.Module classes here yet".
    //
    // The bug this pins: the production source answered `[]` while the backend was
    // not running, so a user staring at a file containing `class Net(nn.Module)`
    // was told their folder had none, and the honest message with its Open Canvas
    // button could never appear.
    const lenses = new WorkspaceClassIndex(async () => {
      throw new Error('the backend is not running');
    });
    const uri = Uri.file('/work/models/resid.py') as unknown as import('vscode').Uri;

    await lenses.refresh(uri);

    assert.equal(
      lenses.classesFor(WorkspaceClassIndex.rootFor(uri)),
      undefined,
      'a failed scan recorded itself as a completed scan of nothing',
    );
    lenses.dispose();
  });

  test('a scan that ran and found nothing IS recorded as empty', async () => {
    // The other half. Without this the fix above could be "never record anything",
    // which would make the honest message permanent and equally wrong.
    const lenses = new WorkspaceClassIndex(async () => []);
    const uri = Uri.file('/work/models/resid.py') as unknown as import('vscode').Uri;

    await lenses.refresh(uri);

    assert.deepEqual(lenses.classesFor(WorkspaceClassIndex.rootFor(uri)), []);
    lenses.dispose();
  });

  test('backend readiness makes an already-open imported subclass discoverable without an edit', async () => {
    let ready = false;
    const index = new WorkspaceClassIndex(async () => {
      if (!ready) throw new Error('backend not running');
      return scanned({ name: 'Model', file: 'model.py', line: 2, bases: ['ImportedBase'] });
    });
    const provider = new OpenAsGraphLenses(index);
    let redraws = 0;
    provider.onDidChangeCodeLenses(() => redraws++);
    const document = { uri: Uri.file(path.join(ROOT, 'model.py')), languageId: 'python' } as import('vscode').TextDocument;
    assert.deepEqual(provider.provideCodeLenses(document), []);
    await index.refresh(document.uri);
    assert.equal(redraws, 0, 'a cold scan must not publish a false empty result');
    ready = true;
    await index.refresh(document.uri);
    assert.equal(redraws, 1, 'readiness must notify VS Code to ask for lenses again');
    const actual = provider.provideCodeLenses(document);
    assert.equal(actual.length, 1);
    assert.equal(actual[0].command?.arguments?.[0].cls, 'Model');
    provider.dispose(); index.dispose();
  });
});
