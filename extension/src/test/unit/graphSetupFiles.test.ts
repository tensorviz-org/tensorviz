import * as assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm, symlink, link } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { readSetupFile, saveSetupFiles, openSetupHelper, type PreparedGraphSetup } from '../../graphSetupFiles';

const vscode = require('../vscodeStub');
suite('graph setup file persistence', () => {
  let root: string, restore: Array<() => void>, applies: number;
  const documents = new Map<string, any>();
  function patch(object: any, key: string, value: unknown) {
    const descriptor = Object.getOwnPropertyDescriptor(object, key);
    Object.defineProperty(object, key, { configurable: true, writable: true, value });
    restore.push(() => descriptor ? Object.defineProperty(object, key, descriptor) : delete object[key]);
  }
  async function document(uri: any) {
    const file = uri.fsPath;
    if (!documents.has(file)) {
      const doc = { uri, version: 1, isDirty: false, text: await readFile(file, 'utf8'),
        getText() { return this.text; }, positionAt(offset: number) { return { offset }; },
        async save() { await writeFile(file, this.text); this.isDirty = false; this.version++; return true; } };
      documents.set(file, doc);
    }
    return documents.get(file);
  }
  setup(async () => {
    vscode.reset(); restore = []; applies = 0; documents.clear();
    root = await mkdtemp(path.join(tmpdir(), 'tensorviz-opening-files-'));
    patch(vscode.workspace, 'textDocuments', []);
    Object.defineProperty(vscode.workspace, 'textDocuments', { configurable: true, get: () => [...documents.values()] });
    patch(vscode.workspace, 'openTextDocument', document);
    patch(vscode.window, 'showTextDocument', async (doc: any) => ({ document: doc }));
    patch(vscode, 'WorkspaceEdit', class {
      entries: any[] = [];
      createFile(uri: any) { this.entries.push({ kind: 'create', uri }); }
      insert(uri: any, _position: any, text: string) { this.entries.push({ kind: 'text', uri, text }); }
      replace(uri: any, _range: any, text: string) { this.entries.push({ kind: 'text', uri, text }); }
    });
    patch(vscode.workspace, 'applyEdit', async (edit: any) => {
      applies++;
      for (const entry of edit.entries) {
        if (entry.kind === 'create') {
          await mkdir(path.dirname(entry.uri.fsPath), { recursive: true });
          await writeFile(entry.uri.fsPath, '', { flag: 'wx' });
        } else {
          const doc = await document(entry.uri);
          doc.text = entry.text; doc.isDirty = true; doc.version++;
        }
      }
      return true;
    });
  });
  teardown(async () => { for (const undo of restore.reverse()) undo(); await rm(root, { recursive: true, force: true }); });
  const prepared = (): PreparedGraphSetup => ({ profile: 'net', config: 'version = 1\n', factory: null });
  test('a desktop-local or foreign remote buffer cannot shadow saved SSH setup', async () => {
    const file = path.join(root, 'tensorviz.toml');
    await writeFile(file, 'version = 1\n');
    for (const scheme of ['vscode-local', 'vscode-remote']) {
      documents.set('foreign', { uri: { ...vscode.Uri.file(file), scheme, authority: 'ssh-remote+other' },
        isDirty: true, getText: () => '# another machine' });
      assert.equal((await readSetupFile(root)).text, 'version = 1\n');
    }
    const doc = await document(vscode.Uri.file(file));
    doc.isDirty = true;
    await assert.rejects(readSetupFile(root), /Save or discard/);
  });
  test('factory scaffold stays in the editor for review and never replaces an existing helper', async () => {
    const file = '.tensorviz/setup_Net_0123456789ab.py';
    await openSetupHelper(root, file, '# complete this factory\n', () => true, new AbortController().signal);
    const doc = documents.get(path.join(root, file));
    assert.ok(doc.isDirty);
    assert.equal((await readSetupFile(root)).exists, false, 'scaffolding does not invent inputs or create a profile');
    doc.text = '# my implementation'; await doc.save();
    await openSetupHelper(root, file, '# complete this factory\n', () => true, new AbortController().signal);
    assert.equal(doc.text, '# my implementation');
    assert.equal(applies, 1);
  });
  test('saves config and generated factory in one host edit', async () => {
    const before = await readSetupFile(root);
    const output = { ...prepared(), factory: { file: '.tensorviz/setup_Net_0123456789ab.py', source: 'def make_model():\n    return None\n' } };
    await saveSetupFiles(root, before, output, () => true, new AbortController().signal);
    assert.equal(applies, 1);
    assert.equal(await readFile(path.join(root, 'tensorviz.toml'), 'utf8'), output.config);
    assert.equal(await readFile(path.join(root, output.factory.file), 'utf8'), output.factory.source);
    assert.ok([...documents.values()].every(doc => !doc.isDirty));
  });
  test('a config created while the form was open is preserved', async () => {
    const before = await readSetupFile(root);
    await writeFile(path.join(root, 'tensorviz.toml'), '# user work\n');
    await assert.rejects(saveSetupFiles(root, before, prepared(), () => true, new AbortController().signal), /changed/);
    assert.equal(applies, 0);
    assert.equal(await readFile(path.join(root, 'tensorviz.toml'), 'utf8'), '# user work\n');
  });
  test('unsaved config changes cannot be overwritten', async () => {
    const file = path.join(root, 'tensorviz.toml');
    await writeFile(file, 'version = 1\n');
    const before = await readSetupFile(root);
    const doc = await document(vscode.Uri.file(file));
    doc.text = '# unsaved'; doc.isDirty = true;
    await assert.rejects(saveSetupFiles(root, before, prepared(), () => true, new AbortController().signal), /Save or discard/);
    assert.equal(applies, 0);
    assert.equal(doc.text, '# unsaved');
  });
  for (const kind of ['symlink', 'hardlink'] as const) test(`${kind} config is never rewritten`, async () => {
    const original = path.join(root, 'original.toml');
    await writeFile(original, '# original');
    await (kind === 'symlink' ? symlink : link)(original, path.join(root, 'tensorviz.toml'));
    await assert.rejects(readSetupFile(root), /links/);
    assert.equal(await readFile(original, 'utf8'), '# original');
  });
  test('a linked helper directory is refused before any edit', async () => {
    const before = await readSetupFile(root);
    await mkdir(path.join(root, 'elsewhere'));
    await symlink(path.join(root, 'elsewhere'), path.join(root, '.tensorviz'));
    await assert.rejects(saveSetupFiles(root, before, { ...prepared(), factory: {
      file: '.tensorviz/setup_Net_0123456789ab.py', source: 'x = 1' } }, () => true, new AbortController().signal), /links/);
    assert.equal(applies, 0);
  });
  test('cancellation before write leaves no files', async () => {
    const before = await readSetupFile(root), controller = new AbortController();
    controller.abort();
    await assert.rejects(saveSetupFiles(root, before, prepared(), () => true, controller.signal), /cancelled/);
    assert.equal(applies, 0);
    assert.equal((await readSetupFile(root)).exists, false);
  });
});
