import * as assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { reviewEditorDiff, reviewFileChanges } from '../../editorGraphReview';
import { reviewGit } from '@tensorviz/graph-host/gitReviewSources';
import { state, resetEditors, editorState } from '../vscodeStub';
const vscode = require('../vscodeStub');

suite('Graph diff from the active editor', () => {
  let root: string, openDocument: any;
  const uri = (scheme: string, file: string, ref?: string) => ({ scheme, fsPath: file,
    query: ref === undefined ? '' : JSON.stringify({ path: file, ref }),
    toString() { return `${scheme}:${file}?${this.query}`; } });
  setup(async () => {
    resetEditors(); state.workspaceFolders = undefined;
    root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-editor-review-')));
    await reviewGit(root, ['init', '-q']);
    await fs.writeFile(path.join(root, 'model.py'), 'base\n'); await fs.writeFile(path.join(root, 'helper.py'), 'helper base\n');
    await reviewGit(root, ['add', '.']); await reviewGit(root, ['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '-qm', 'base']);
    openDocument = vscode.workspace.openTextDocument;
  });
  teardown(async () => { vscode.workspace.openTextDocument = openDocument; resetEditors(); state.workspaceFolders = undefined; await fs.rm(root, { recursive: true, force: true }); });
  const noGitHub = async (): Promise<any> => { throw new Error('A local diff must not use GitHub.'); };
  test('SCM actions compare their selected Git sides, include unsaved text, and write no view or config files', async () => {
    const file = path.join(root, 'model.py');
    await fs.writeFile(file, 'staged\n'); await reviewGit(root, ['add', 'model.py']); await fs.writeFile(file, 'working\n');
    editorState.documents.set(file, 'unsaved\n');
    const status = await reviewGit(root, ['status', '--porcelain', '--untracked-files=all']);
    for (const [kind, expectedBefore, expectedAfter] of [
      ['working', 'staged\n', 'unsaved\n'], ['staged', 'base\n', 'staged\n'], ['head', 'base\n', 'unsaved\n'],
    ] as const) {
      const session = await reviewFileChanges(uri('file', file) as any, kind);
      assert.equal(await session.before.read('model.py'), expectedBefore);
      assert.equal(await session.after.read('model.py'), expectedAfter);
      assert.equal(session.preferredFile, 'model.py');
      assert.equal(await (await session.refresh()).after.read('model.py'), expectedAfter);
    }
    assert.equal(await reviewGit(root, ['status', '--porcelain', '--untracked-files=all']), status);
    assert.deepEqual((await fs.readdir(root)).sort(), ['.git', 'helper.py', 'model.py']);
  });
  test('opens a local diff without requiring a workspace or choosing a model, using the displayed unsaved text', async () => {
    const file = path.join(root, 'model.py');
    const input = { original: uri('git', file, 'HEAD'), modified: uri('file', file) } as any;
    editorState.documents.set(file, 'unsaved\n');
    vscode.workspace.openTextDocument = async (item: any) => ({ getText: () => item.scheme === 'git' ? 'base\n' : 'unsaved\n' });
    const status = await reviewGit(root, ['status', '--porcelain']);
    const session = await reviewEditorDiff(input, noGitHub, '/unused');
    assert.equal(await session.before.read('model.py'), 'base\n');
    assert.equal(await session.after.read('model.py'), 'unsaved\n');
    assert.equal(await session.before.read('helper.py'), 'helper base\n');
    assert.deepEqual(session.locate!(input.modified), { file: 'model.py', side: 'after' });
    assert.equal(await reviewGit(root, ['status', '--porcelain']), status);
  });
  test('uses the index for staged diff documents instead of substituting current working files', async () => {
    const file = path.join(root, 'model.py');
    await fs.writeFile(file, 'staged\n'); await reviewGit(root, ['add', 'model.py']); await fs.writeFile(file, 'working\n');
    const input = { original: uri('git', file, 'HEAD'), modified: uri('git', file, '') } as any;
    vscode.workspace.openTextDocument = async (item: any) => ({ getText: () => JSON.parse(item.query).ref === 'HEAD' ? 'base\n' : 'staged\n' });
    const session = await reviewEditorDiff(input, noGitHub, '/unused');
    assert.equal(await session.after.read('model.py'), 'staged\n');
  });
  test('refuses a stale code diff rather than comparing different source behind it', async () => {
    const file = path.join(root, 'model.py');
    const input = { original: uri('git', file, 'HEAD'), modified: uri('file', file) } as any;
    vscode.workspace.openTextDocument = async () => ({ getText: () => 'stale content\n' });
    await assert.rejects(reviewEditorDiff(input, noGitHub, '/unused'), /code diff changed/);
  });
  test('unknown diff providers disclose missing dependency revisions', async () => {
    const input = { original: uri('custom', '/virtual/model.py'), modified: uri('custom2', '/virtual/model.py') } as any;
    vscode.workspace.openTextDocument = async (item: any) => ({ getText: () => item.scheme === 'custom' ? 'before\n' : 'after\n' });
    const session = await reviewEditorDiff(input, noGitHub, '/unused');
    assert.equal(session.before.truncated, true); assert.equal(session.after.truncated, true);
    assert.deepEqual([...session.after.files.keys()], ['model.py']);
  });
  test('empty-tree added-file sentinels retain the real base repository and its dependencies', async () => {
    const file = path.join(root, 'new.py');
    await fs.writeFile(file, 'new model'); await reviewGit(root, ['add', 'new.py']);
    const empty = createHash('sha1').update('tree 0\0').digest('hex');
    const input = { original: uri('git', file, empty), modified: uri('git', file, '') } as any;
    vscode.workspace.openTextDocument = async (item: any) => ({ getText: () => JSON.parse(item.query).ref === empty ? '' : 'new model' });
    const session = await reviewEditorDiff(input, noGitHub, '/unused');
    assert.equal(session.before.files.has('new.py'), false);
    assert.equal(await session.before.read('helper.py'), 'helper base\n');
    assert.equal(await session.after.read('new.py'), 'new model');
  });
  test('empty-tree deletion keeps other files and does not resurrect a deleted clean buffer', async () => {
    const file = path.join(root, 'model.py');
    await fs.rm(file); editorState.documents.set(file, 'base\n');
    const empty = createHash('sha1').update('tree 0\0').digest('hex');
    const input = { original: uri('git', file, '~'), modified: uri('git', file, empty) } as any;
    vscode.workspace.openTextDocument = async (item: any) => ({ getText: () => JSON.parse(item.query).ref === empty ? '' : 'base\n' });
    const session = await reviewEditorDiff(input, noGitHub, '/unused');
    assert.equal(session.after.files.has('model.py'), false);
    assert.equal(await session.after.read('helper.py'), 'helper base\n');
    assert.equal(await session.before.read('model.py'), 'base\n');
  });

});
