import * as assert from 'node:assert/strict';
import * as path from 'node:path';

import { OpenAsGraphLenses } from '../../openAsGraphLenses';
import { WorkspaceClass, WorkspaceClassIndex } from '../../workspaceClassIndex';
import { CodeLens, Uri, reset, state } from '../vscodeStub';

const ROOT = path.join(path.sep, 'ws');

function pythonDocument(fsPath: string): { languageId: string; uri: { fsPath: string; scheme: string } } {
  return { languageId: 'python', uri: { fsPath, scheme: 'file' } };
}

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

/** The real index, so these cover the seam the provider actually runs against. */
async function indexed(...classes: Array<Partial<WorkspaceClass>>): Promise<WorkspaceClassIndex> {
  const index = new WorkspaceClassIndex(async () => scanned(...classes));
  await index.refresh(Uri.file(path.join(ROOT, 'models', 'resid.py')) as never);
  return index;
}

const residPy = () => pythonDocument(path.join(ROOT, 'models', 'resid.py')) as never;

suite('OpenAsGraphLenses', () => {
  setup(() => {
    reset();
    state.workspaceFolders = [{ uri: { fsPath: ROOT } }];
  });

  test('puts one lens on each class this file declares, at the class line', async () => {
    const index = await indexed(
      { name: 'Net', line: 14 },
      { name: 'Block', line: 30 },
      { name: 'Other', file: 'models/gate.py' },
    );
    const produced = new OpenAsGraphLenses(index).provideCodeLenses(residPy()) as unknown as CodeLens[];

    assert.equal(produced.length, 2, 'a class in another file must not get a lens here');
    // 1-based from the backend, 0-based in the editor.
    assert.deepEqual(produced.map((lens) => lens.range.startLine), [13, 29]);
    assert.deepEqual(produced.map((lens) => lens.command?.command), [
      'tensorviz.openAsGraph',
      'tensorviz.openAsGraph',
    ]);
    // Naming the class is load-bearing: openAsGraph without `cls` opens the whole
    // project, so a lens that dropped it would look per-class and act project-wide.
    assert.deepEqual(
      produced.map((lens) => (lens.command?.arguments?.[0] as { cls: string }).cls),
      ['Net', 'Block'],
    );
  });

  test('ignores anything that is not a Python file on disk', async () => {
    const index = await indexed({ name: 'Net', line: 14 });
    const lenses = new OpenAsGraphLenses(index);

    const notebook = { languageId: 'json', uri: { fsPath: path.join(ROOT, 'a.ipynb'), scheme: 'file' } };
    const untitled = { languageId: 'python', uri: { fsPath: path.join(ROOT, 'x.py'), scheme: 'untitled' } };
    assert.deepEqual(lenses.provideCodeLenses(notebook as never), []);
    assert.deepEqual(lenses.provideCodeLenses(untitled as never), []);
  });

  test('renders nothing on a cold index rather than blocking on a scan', () => {
    // provideCodeLenses runs on every keystroke-idle. It must never await the
    // backend, which throws outright while the backend is still starting.
    let calls = 0;
    const index = new WorkspaceClassIndex(async () => {
      calls += 1;
      return scanned({ name: 'Net', line: 14 });
    });
    const produced = new OpenAsGraphLenses(index).provideCodeLenses(residPy());
    assert.deepEqual(produced, [], 'a cold index has no answer yet, and must not invent one');
    assert.equal(calls, 1, 'it should have asked for a scan in the background');
  });

  /**
   * The regression that got the lens deleted, and which no test has ever covered:
   * on a project canvas the class is already drawn, so offering to *open* it offers
   * something that has already happened.
   */
  test('withholds the offer for a class the canvas is already drawing', async () => {
    const index = await indexed({ name: 'Net', line: 14 }, { name: 'Block', line: 30 });
    const drawn = new Set(['models/resid.py::Net']);
    const lenses = new OpenAsGraphLenses(
      index,
      (root, file, cls) => root === ROOT && drawn.has(`${file}::${cls}`),
    );

    const produced = lenses.provideCodeLenses(residPy()) as unknown as CodeLens[];
    assert.deepEqual(
      produced.map((lens) => (lens.command?.arguments?.[0] as { cls: string }).cls),
      ['Block'],
      'Net is on screen; Block is not, and still deserves the offer',
    );
  });

  test('offers every lens again once nothing is drawn', async () => {
    const index = await indexed({ name: 'Net', line: 14 });
    let canvasOpen = true;
    const lenses = new OpenAsGraphLenses(index, () => canvasOpen);
    assert.equal((lenses.provideCodeLenses(residPy()) as unknown as CodeLens[]).length, 0);

    // Closing the canvas makes the withheld offer meaningful again. Without this the
    // lens would stay hidden for the rest of the window, because the project panel
    // outlives the canvas that was drawing from it.
    canvasOpen = false;
    assert.equal((lenses.provideCodeLenses(residPy()) as unknown as CodeLens[]).length, 1);
  });

  test('asks VS Code to redraw when the drawn set moves', async () => {
    const index = await indexed({ name: 'Net', line: 14 });
    const lenses = new OpenAsGraphLenses(index);
    let redraws = 0;
    lenses.onDidChangeCodeLenses(() => {
      redraws += 1;
    });

    // Opening a canvas changes no CLASS, so the class index never fires and VS Code
    // would keep showing the answer from the last save.
    lenses.refreshDrawn();
    assert.equal(redraws, 1);
  });

  test('redraws when the class index rescans', async () => {
    const index = await indexed({ name: 'Net', line: 14 });
    const lenses = new OpenAsGraphLenses(index);
    let redraws = 0;
    lenses.onDidChangeCodeLenses(() => {
      redraws += 1;
    });
    await index.refresh(Uri.file(path.join(ROOT, 'models', 'resid.py')) as never);
    assert.ok(redraws >= 1, 'a new scan can add or remove a class, so the lenses must redraw');
  });
});
