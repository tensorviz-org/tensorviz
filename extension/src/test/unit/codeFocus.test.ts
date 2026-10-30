import * as assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { CodeFocus } from '../../codeFocus';
import { DEFAULT_CODE_FOCUS, type CodeFocusOptions, type LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import { commands, documentChanges, editorState, fireConfigurationChange, openStubEditor, Range, reset, resetEditors, state, window } from '../vscodeStub';

const file = '/workspace/model.py';
const source = ['import torch', '', 'class Model(nn.Module):', '    def __init__(self):',
  '        self.scale = nn.Parameter(torch.ones(8))', '', '    def forward(self, x):',
  '        return x * self.scale', '', '    def reset(self):', '        self.scale.fill_(1)', '',
  'def other():', '    pass', ''].join('\n');
const digest = createHash('sha256').update(source).digest('hex');
const target: LocalBrowserTargetIdentity = { workspaceFolder: '/workspace', file: 'model.py', cls: 'Model',
  sourceDigest: digest, documentVersion: 1, backendRevision: 0 };
const selection = (nodeIds = ['scale'], viewId = 'canvas') => ({ viewId, target, nodeIds, edgeIds: [] as string[] });
const parameter = { declaration: [[5, 5]], uses: [[8, 8], [11, 11]], declarationFunctions: [[4, 5]],
  useFunctions: [[7, 8], [10, 11]], limitation: 'Direct references only.' };
const span = { file: 'model.py', file_digest: digest, focus: {
  class: [[3, 11]], function: [[4, 5]], node: [[5, 5]], parameter } };

suite('codeFocus', () => {
  let focus: CodeFocus;
  let current: any;
  let messages: any[];
  setup(() => {
    reset(); resetEditors();
    state.configuration.set('tensorviz.codeFocus', { ...DEFAULT_CODE_FOCUS, enabled: true });
    current = { projection: { spanmap: { scale: span,
      op: { ...span, focus: { class: [[3, 11]], function: [[7, 8]], node: [[8, 8]] } } } },
      dependencies: new Map([['model.py', digest]]) };
    messages = [];
    openStubEditor(file, source);
    focus = new CodeFocus(() => current, message => messages.push(message));
  });
  teardown(() => focus.dispose());
  // Which files a selection opened in an editor, in order.
  const opened = () => {
    // The stub attaches showTextDocument at runtime, outside its declared type.
    const stub = window as typeof window & { showTextDocument(document: { uri: { fsPath: string } }): Promise<unknown> };
    const shown: string[] = [];
    const show = stub.showTextDocument;
    stub.showTextDocument = async document => { shown.push(document.uri.fsPath); return show(document); };
    teardown(() => { stub.showTextDocument = show; });
    return shown;
  };
  const ranges = (kind = 'dim', path = file) => {
    const call = editorState.decorationCalls.filter(c => c.fsPath === path
      && (kind === 'dim' ? c.type.options.opacity !== undefined : c.type.options.isWholeLine)).at(-1);
    return (call?.ranges as Range[] ?? []).map(r => [r.start.line + 1, r.end.line + 1]);
  };
  const options = (changes: Partial<CodeFocusOptions>) => ({ ...DEFAULT_CODE_FOCUS, enabled: true, ...changes });

  test('parameter Both preserves declaration and every direct use, without editing or moving the caret', async () => {
    const editor = editorState.visibleTextEditors[0];
    editor.selection = { active: { line: 13, character: 4 } };
    await focus.select(selection());
    assert.deepEqual(ranges(), [[1, 4], [6, 7], [9, 10], [12, 15]]);
    assert.deepEqual(ranges('highlight'), [[5, 5], [8, 8], [11, 11]]);
    assert.deepEqual(editor.selection.active, { line: 13, character: 4 });
    assert.equal(messages.at(-1).parameter, true);
    assert.equal(editorState.appliedEdits.length, 0);
  });

  test('class scope keeps the complete owning class; Function keeps enclosing methods for selected references', async () => {
    await focus.configure(selection(), options({ scope: 'class' }));
    assert.deepEqual(ranges(), [[1, 2], [12, 15]]);
    await focus.configure(selection(), options({ scope: 'function', references: 'uses' }));
    assert.deepEqual(ranges(), [[1, 6], [9, 9], [12, 15]]);
    assert.deepEqual(ranges('highlight'), [[8, 8], [11, 11]]);
    await focus.configure(selection(), options({ references: 'declaration', opacity: 40 }));
    assert.deepEqual(ranges(), [[1, 4], [6, 15]]);
    assert.equal(editorState.decorationTypes.at(-1)?.options.opacity, '0.4');
    await new Promise(resolve => setTimeout(resolve, 0));
    assert.deepEqual(state.configuration.get('tensorviz.codeFocus'), options({ references: 'declaration', opacity: 40 }));
  });

  test('multi-selection preserves the union and switching selection clears prior marks', async () => {
    await focus.configure(selection(['scale', 'op']), options({ references: 'declaration' }));
    assert.deepEqual(ranges('highlight'), [[5, 5], [8, 8]]);
    await focus.select(selection(['op']), false);
    assert.deepEqual(ranges('highlight'), [[8, 8]]);
    assert.deepEqual(ranges(), [[1, 7], [9, 15]]);
  });

  test('unmapped or mixed node/edge selection never leaves misleading partial focus', async () => {
    await focus.select(selection(), false);
    await focus.select(selection(['scale', 'missing']), false);
    assert.deepEqual(ranges(), []);
    await focus.select({ ...selection(), edgeIds: ['edge'] }, false);
    assert.deepEqual(ranges(), []);
    await focus.select(selection([]), false);
    assert.deepEqual(ranges('highlight'), []);
  });

  test('options wait for the current projection instead of erasing the previous selection', async () => {
    await focus.select(selection(), false);
    const kept = ranges();
    assert.ok(kept.length);
    current = undefined;
    await focus.configure(selection(['op']), options({ scope: 'class', references: 'uses' }));
    assert.deepEqual(ranges(), kept, 'an in-flight projection does not wipe the previous focus');
    current = { projection: { spanmap: { scale: span,
      op: { ...span, focus: { class: [[3, 11]], function: [[7, 8]], node: [[8, 8]] } } } },
      dependencies: new Map([['model.py', digest]]) };
    await focus.retry();
    assert.deepEqual(ranges(), [[1, 2], [12, 15]], 'class scope applies to the selection passed in, not the previous node');
  });

  test('reference options on a non-parameter say so instead of looking unchanged', async () => {
    await focus.configure(selection(['op']), options({ references: 'uses' }));
    assert.match(messages.at(-1).detail, /not a parameter/);
    assert.equal(messages.at(-1).parameter, false);
    assert.deepEqual(ranges('highlight'), [[8, 8]]);
  });

  test('unsaved edits and stale projections remove decorations immediately', async () => {
    await focus.select(selection(), false);
    const editor = openStubEditor(file, source + '# edit');
    documentChanges.fire({ document: editor.document });
    assert.deepEqual(ranges(), []);
    await focus.select(selection(), false);
    assert.deepEqual(ranges(), []);
    assert.match(messages.at(-1).detail, /changed/);
    openStubEditor(file, source);
    current = undefined;
    await focus.select(selection(), false);
    assert.deepEqual(ranges(), []);
  });

  test('hidden/disconnected views clear only the owning selection and cancelling async selection cannot repaint', async () => {
    await focus.select(selection(), false);
    focus.clear('other-view');
    assert.ok(ranges().length);
    focus.disconnectBrowsers('canvas');
    assert.ok(ranges().length);
    focus.clear('canvas');
    assert.deepEqual(ranges(), []);
    const pending = focus.select(selection(), false);
    focus.clear('canvas');
    await pending;
    assert.deepEqual(ranges(), []);
    await focus.select(selection(['op'], 'browser'), false);
    focus.disconnectBrowsers('canvas');
    assert.deepEqual(ranges(), []);
  });

  test('invalid spans, unmatched dependencies and missing parameter uses leave code readable', async () => {
    for (const invalid of [
      { ...span, file: '../outside.py' }, { ...span, file_digest: 'wrong' },
      { ...span, focus: { ...span.focus, parameter: null } },
    ]) {
      current.projection.spanmap.scale = invalid;
      await focus.configure(selection(), options({ references: 'uses', scope: invalid.focus.parameter ? 'class' : 'node' }));
      assert.deepEqual(ranges(), []);
    }
    current.projection.spanmap.scale = { ...span, focus: { ...span.focus, class: [[0, 1000]] } };
    await focus.configure(selection(), options({ scope: 'class' }));
    assert.deepEqual(ranges(), []);
    current.projection.spanmap.scale = { ...span, focus: { ...span.focus, parameter: { ...parameter, uses: [] } } };
    await focus.configure(selection(), options({ references: 'uses' }));
    assert.deepEqual(ranges(), []);
  });

  test('Code focus off never opens or dims code, and clears marks left while it was on', async () => {
    const shown = opened();
    await focus.select({ ...selection(), follow: true });
    assert.equal(shown.length, 1);
    assert.ok(ranges().length);
    await focus.select({ ...selection(['op']), follow: false });
    assert.deepEqual(ranges(), []);
    assert.deepEqual(ranges('highlight'), []);
    assert.equal(shown.length, 1, 'a selection with Code focus off must not open its source');
    assert.match(messages.at(-1).detail, /Code focus is off/);
    await focus.configure({ ...selection(['op']), follow: false }, options({ scope: 'class' }));
    assert.deepEqual(ranges(), []);
    assert.equal(shown.length, 1, 'changing focus settings with Code focus off must not open code');
  });

  test('Code focus on opens the selected source even with dimming off, without dimming it', async () => {
    state.configuration.set('tensorviz.codeFocus', { ...DEFAULT_CODE_FOCUS });
    focus.dispose();
    focus = new CodeFocus(() => current, message => messages.push(message));
    const shown = opened();
    await focus.select({ ...selection(['op']), follow: true });
    assert.deepEqual(shown, [file]);
    assert.deepEqual(ranges(), []);
    assert.match(messages.at(-1).detail, /Dimming is off/);
    await focus.select({ ...selection(['op']), follow: true });
    assert.equal(shown.length, 1, 'reselecting the same node does not reopen it');
  });

  test('turning Code focus on with a node selected opens its source', async () => {
    const shown = opened();
    await focus.select({ ...selection(['op']), follow: false });
    assert.deepEqual(shown, []);
    await focus.select({ ...selection(['op']), follow: true });
    assert.deepEqual(shown, [file]);
    assert.deepEqual(ranges('highlight'), [[8, 8]]);
  });

  test('showing source first makes room beside a Canvas that fills the window, and only then', async () => {
    const shown = opened();
    let fail = false;
    const registration = commands.registerCommand('workbench.action.restoreAuxiliaryBar', () => {
      shown.push('restore');
      if (fail) throw new Error('command unavailable');
    });
    try {
      await focus.select({ ...selection(['op']), follow: true });
      assert.deepEqual(shown, ['restore', file]);
      await focus.select({ ...selection(['op']), follow: true });
      await focus.select({ ...selection(), follow: false });
      assert.deepEqual(shown, ['restore', file], 'no source shown, so the layout is left alone');
      fail = true;
      await focus.select({ ...selection(), follow: true });
      assert.deepEqual(shown, ['restore', file, 'restore', file], 'a failed restore still shows the source');
    } finally { registration.dispose(); }
  });

  test('a view that sends no Code focus keeps the setting: Focus code off opens nothing', async () => {
    state.configuration.set('tensorviz.codeFocus', { ...DEFAULT_CODE_FOCUS });
    focus.dispose();
    focus = new CodeFocus(() => current, message => messages.push(message));
    const shown = opened();
    await focus.select(selection(['op']));
    assert.deepEqual(shown, []);
    assert.match(messages.at(-1).detail, /Focus code is off/);
  });

  test('turning focus off from settings clears marks and disposing releases decorations', async () => {
    await focus.select(selection(), false);
    fireConfigurationChange('tensorviz.codeFocus', { ...DEFAULT_CODE_FOCUS });
    assert.deepEqual(ranges(), []);
    focus.dispose();
    assert.ok(editorState.decorationTypes.every(type => type.disposed));
  });
});
