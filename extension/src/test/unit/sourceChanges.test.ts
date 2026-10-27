import * as assert from 'node:assert/strict';
import { SourceWriter, sha256, type CodeArtifact } from '../../sourceChanges';
import { Uri, editorState, reset, resetEditors } from '../vscodeStub';

suite('provider-independent source writes', () => {
  const before = 'width = 8\n', after = 'width = 16\n';
  const artifact: CodeArtifact = {
    path: 'net.py', before, after, before_digest: sha256(before), after_digest: sha256(after),
    edits: [{ op_id: 'width', path: 'net.py', before_digest: sha256(before), range: { start: [1, 8], end: [1, 9] }, after: '16' }],
  };
  setup(() => { reset(); resetEditors(); editorState.documents.set('/work/net.py', before); });
  const writer = () => new SourceWriter(Uri.file('/work') as unknown as import('vscode').Uri, { appendLine() {} });

  test('writes the reviewed change in one workspace edit without a backend', async () => {
    assert.equal((await writer().write([artifact], () => true)).outcome, 'applied');
    assert.equal(editorState.appliedEdits.length, 1);
    assert.equal(editorState.appliedEdits[0].replacements[0].newText, '16');
  });

  test('preserves changes made by the user after a review', async () => {
    editorState.documents.set('/work/net.py', 'width = 32\n');
    assert.equal((await writer().write([artifact])).outcome, 'failed');
    assert.equal(editorState.appliedEdits.length, 0);
  });

  test('a matching buffer is insufficient when the source context is stale', async () => {
    assert.equal((await writer().write([artifact], () => false)).outcome, 'failed');
    assert.equal(editorState.appliedEdits.length, 0);
  });
});
