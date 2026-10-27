import * as assert from 'node:assert/strict';
import { GraphFilePrompt } from '../../graphFilePrompt';
import { WorkspaceClassIndex } from '../../workspaceClassIndex';
import {
  activeEditorChanges, commands, editorState, fireConfigurationChange, infoMessages,
  reset, resetEditors, resetWebviews, setInfoResponder, state, Uri, webviewState,
} from '../vscodeStub';

const flush = () => new Promise<void>(resolve => setImmediate(resolve));

function focus(file: string, languageId = 'python', scheme = 'file', isDirty = false): void {
  const editor = { document: { uri: { ...Uri.file(file), scheme }, languageId, isDirty } };
  editorState.visibleTextEditors.splice(0, editorState.visibleTextEditors.length, editor as never);
  activeEditorChanges.fire(editor as never);
}

suite('Graph file discovery prompt', () => {
  let index: WorkspaceClassIndex;
  let prompt: GraphFilePrompt | undefined;

  setup(() => {
    reset(); resetEditors(); resetWebviews();
    state.workspaceFolders = [{ uri: { fsPath: '/ws' } }];
    index = new WorkspaceClassIndex(async () => [
      { name: 'Net', file: 'model.py', line: 1, bases: ['nn.Module'], ctor_required: 0 },
      { name: 'Block', file: 'model.py', line: 5, bases: ['nn.Module'], ctor_required: 0 },
      { name: 'Other', file: 'other.py', line: 1, bases: ['nn.Module'], ctor_required: 0 },
    ]);
    commands.registerCommand('tensorviz.openAsGraph', () => undefined);
  });

  teardown(() => { prompt?.dispose(); prompt = undefined; index.dispose(); });

  test('offers a restored model after discovery finishes, without opening it on dismissal', async () => {
    focus('/ws/model.py');
    prompt = new GraphFilePrompt(index);
    assert.equal(infoMessages.length, 0, 'a cold index must not invent an offer');
    await index.refresh(Uri.file('/ws/model.py') as never);
    await flush();
    assert.equal(infoMessages.length, 1);
    assert.match(infoMessages[0], /model\.py/);
    assert.equal(webviewState.commands.length, 0, 'discovery is not permission to capture');
  });

  test('opening from a notification keeps its file when editor focus changes', async () => {
    await index.refresh(Uri.file('/ws/model.py') as never);
    focus('/ws/model.py');
    setInfoResponder(() => { focus('/ws/notes.py'); return 'Open as Graph'; });
    prompt = new GraphFilePrompt(index);
    await flush();
    assert.deepEqual(webviewState.commands, [
      { id: 'tensorviz.openAsGraph', args: [{ uri: '/ws/model.py' }] },
    ], 'the shared command must choose among the offered file\'s models');
  });

  test('tab switching and repeated scans do not repeat an offer, but another model file gets one', async () => {
    await index.refresh(Uri.file('/ws/model.py') as never);
    focus('/ws/model.py');
    prompt = new GraphFilePrompt(index);
    focus('/ws/notes.py'); focus('/ws/model.py');
    await index.refresh(Uri.file('/ws/model.py') as never);
    assert.equal(infoMessages.length, 1);
    focus('/ws/other.py');
    assert.equal(infoMessages.length, 2);
  });

  test('ignores ordinary files, virtual files, unsaved changes and untrusted workspaces', async () => {
    await index.refresh(Uri.file('/ws/model.py') as never);
    prompt = new GraphFilePrompt(index);
    focus('/ws/notes.py');
    focus('/ws/model.py', 'plaintext');
    focus('/ws/model.py', 'python', 'vscode-local');
    focus('/ws/model.py', 'python', 'file', true);
    state.isTrusted = false;
    focus('/ws/model.py');
    assert.equal(infoMessages.length, 0);
    state.isTrusted = true;
    await index.refresh(Uri.file('/ws/model.py') as never);
    assert.equal(infoMessages.length, 1);
  });

  test('a late scan only offers the active file, not the model the user already left', async () => {
    focus('/ws/model.py');
    prompt = new GraphFilePrompt(index);
    focus('/ws/notes.py');
    await index.refresh(Uri.file('/ws/model.py') as never);
    assert.equal(infoMessages.length, 0);
    focus('/ws/model.py');
    assert.equal(infoMessages.length, 1);
  });

  test('settings suppress prompts and take effect without reloading', async () => {
    state.configuration.set('tensorviz.showGraphPrompt', false);
    await index.refresh(Uri.file('/ws/model.py') as never);
    focus('/ws/model.py');
    prompt = new GraphFilePrompt(index);
    assert.equal(infoMessages.length, 0);
    fireConfigurationChange('tensorviz.showGraphPrompt', true);
    assert.equal(infoMessages.length, 1);
    fireConfigurationChange('tensorviz.showGraphPrompt', false);
    focus('/ws/other.py');
    assert.equal(infoMessages.length, 1);
  });

  test('disable action persists the preference and does not open a graph', async () => {
    setInfoResponder(() => 'Disable graph prompts');
    await index.refresh(Uri.file('/ws/model.py') as never);
    focus('/ws/model.py');
    prompt = new GraphFilePrompt(index);
    await flush();
    assert.equal(state.configuration.get('tensorviz.showGraphPrompt'), false);
    assert.equal(webviewState.commands.length, 0);
    focus('/ws/other.py');
    assert.equal(infoMessages.length, 1);
  });

  test('disposing the extension prevents a pending notification from opening a graph', async () => {
    setInfoResponder(() => 'Open as Graph');
    await index.refresh(Uri.file('/ws/model.py') as never);
    focus('/ws/model.py');
    prompt = new GraphFilePrompt(index);
    prompt.dispose();
    await flush();
    assert.equal(webviewState.commands.length, 0);
    focus('/ws/other.py');
    assert.equal(infoMessages.length, 1);
  });
});
