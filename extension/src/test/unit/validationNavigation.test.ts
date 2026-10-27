import * as assert from 'node:assert/strict';
import { mkdtemp, realpath, writeFile, symlink, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { createHash } from 'node:crypto';
import { revealExecutionNode } from '../../validationNavigation';
import { openStubEditor, resetEditors } from '../vscodeStub';

suite('execution source navigation', () => {
  let root: string;
  const body = 'class Model:\n    def forward(self, x):\n        return x\n';
  setup(async () => { root = await realpath(await mkdtemp(path.join(tmpdir(), 'tensorviz-navigation-'))); resetEditors(); });
  teardown(async () => { resetEditors(); await rm(root, { recursive: true, force: true }); });
  const evidence = () => ({ root, source: { file: 'model.py', line: 3, column: 8 },
    file: { path: 'model.py', digest: createHash('sha256').update(body).digest('hex'), role: 'source' as const, documentVersion: null } });
  test('opens only the recorded saved file and exact source position', async () => {
    await writeFile(path.join(root, 'model.py'), body);
    const editor = openStubEditor(path.join(root, 'model.py'), body);
    assert.equal((await revealExecutionNode(evidence)).ok, true);
    assert.equal(editor.selection.active.line, 2); assert.equal(editor.selection.active.character, 8);
  });
  test('rejects changed disk, changed buffers, missing identities and stale async evidence', async () => {
    await writeFile(path.join(root, 'model.py'), body + '# edit');
    openStubEditor(path.join(root, 'model.py'), body);
    assert.equal((await revealExecutionNode(evidence)).ok, false);
    await writeFile(path.join(root, 'model.py'), body);
    openStubEditor(path.join(root, 'model.py'), body + '# unsaved');
    assert.equal((await revealExecutionNode(evidence)).ok, false);
    assert.equal((await revealExecutionNode(() => undefined)).ok, false);
    let checks = 0;
    assert.equal((await revealExecutionNode(() => ++checks === 1 ? evidence() : undefined)).ok, false);
  });
  test('refuses traversal and symlinks outside the workspace', async () => {
    const outside = path.join(root, 'outside.py'); await writeFile(outside, body);
    const nested = await mkdtemp(path.join(root, 'workspace-'));
    await symlink(outside, path.join(nested, 'model.py'));
    assert.equal((await revealExecutionNode(() => ({ ...evidence(), root: nested }))).ok, false);
    assert.equal((await revealExecutionNode(() => ({ ...evidence(), source: { file: '../outside.py', line: 1, column: 0 } }))).ok, false);
  });
});
