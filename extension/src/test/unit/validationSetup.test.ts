import * as assert from 'node:assert/strict';
import { ValidationSetup } from '../../validationSetup';
import { CONFIG_STARTER, modelConfigStarter } from '../../validationConfig';

// These tests exercise native buffer/review ordering, not disk writes or a normal profile.
const vscode = require('../vscodeStub');
suite('native validation configuration workflow', () => {
  let restore: Array<() => void>, workflow: ValidationSetup;
  let current: boolean, exists: boolean, text: string, choice: string, reads: number;
  let edits: any[], shown: any[], diagnostics: any[];
  let review: () => void;
  let expectedStarter: string;
  const target = { root: '/workspace', session: 's', current: () => current };
  function patch(object: any, key: string, value: any) {
    const descriptor = Object.getOwnPropertyDescriptor(object, key);
    Object.defineProperty(object, key, { configurable: true, writable: true, value });
    restore.push(() => descriptor ? Object.defineProperty(object, key, descriptor) : delete object[key]);
  }
  function uri(scheme = 'file', file = '/workspace/tensorviz.toml'): any {
    return { scheme, fsPath: file, toString: () => `${scheme}:${file}`, with: (value: any) => uri(value.scheme, file) };
  }
  setup(() => {
    restore = []; current = true; exists = false; text = ''; choice = 'Apply'; reads = 0;
    edits = []; shown = []; diagnostics = []; review = () => {}; expectedStarter = CONFIG_STARTER;
    vscode.reset();
    class Missing extends Error { code = 'FileNotFound'; }
    patch(vscode, 'FileSystemError', Missing);
    patch(vscode.Uri, 'file', () => uri());
    patch(vscode.Uri, 'from', (value: any) => uri(value.scheme, value.path));
    patch(vscode.workspace, 'fs', { stat: async () => { if (!exists) throw new Missing(); return {}; } });
    patch(vscode.workspace, 'openTextDocument', async (value: any) => {
      reads++;
      if (value.scheme === 'file' && !exists) throw new Missing();
      return { uri: value, getText: () => text, positionAt: (offset: number) => new vscode.Position(0, offset) };
    });
    patch(vscode.workspace, 'registerTextDocumentContentProvider', (_scheme: string, provider: any) => {
      assert.equal(provider.provideTextDocumentContent(uri(_scheme, '/after/tensorviz.toml')), expectedStarter);
      return { dispose() {} };
    });
    patch(vscode.workspace, 'applyEdit', async (edit: any) => { edits.push(edit); return true; });
    patch(vscode.WorkspaceEdit.prototype, 'insert', function (this: any, value: any, _position: any, content: string) {
      this.target = value; this.content = content;
    });
    patch(vscode, 'commands', { executeCommand: async (command: string) => {
      assert.equal(command, 'vscode.diff'); review();
    } });
    patch(vscode.window, 'tabGroups', { all: [], close: async () => {} });
    patch(vscode.window, 'showInformationMessage', async () => choice);
    patch(vscode.window, 'showTextDocument', async (...args: any[]) => { shown.push(args); });
    patch(vscode.languages, 'createDiagnosticCollection', () => ({
      set: (_uri: any, entries: any[]) => { diagnostics = entries; }, delete() {}, dispose() {},
    }));
    patch(vscode, 'Range', class { constructor(public start: any, public end: any) {} });
    patch(vscode, 'Diagnostic', class { constructor(public range: any, public message: string) {} });
    patch(vscode, 'DiagnosticSeverity', { Error: 0 });
    workflow = new ValidationSetup();
  });
  teardown(() => { workflow?.dispose(); for (const undo of restore.reverse()) undo(); });
  test('Apply reviews first, writes one unsaved buffer edit, and never saves', async () => {
    let reviewed = false;
    review = () => { assert.equal(edits.length, 0); reviewed = true; };
    await workflow.open(target);
    assert.equal(reviewed, true); assert.equal(edits.length, 1);
    assert.equal(edits[0].target.scheme, 'untitled');
    assert.equal(edits[0].target.fsPath, '/workspace/tensorviz.toml');
    assert.equal(edits[0].content, CONFIG_STARTER); assert.equal(exists, false);
    assert.equal(shown.length, 1);
  });
  test('graph setup can create a target-specific starter without saving or replacing existing content', async () => {
    expectedStarter = modelConfigStarter('whisper.model:Whisper');
    await workflow.open(target, [], expectedStarter);
    assert.equal(edits[0].content, expectedStarter);
    assert.equal(exists, false);
    exists = true; text = '[profiles.user]\nmodel = "user:Net"';
    await workflow.open(target, ['profiles', 'user'], expectedStarter);
    assert.equal(edits.length, 1, 'opening an existing file cannot write the starter');
  });
  for (const mode of ['discard', 'retarget', 'trust', 'appeared', 'dirty'] as const) {
    test(`${mode} during review cannot apply the starter`, async () => {
      review = () => {
        if (mode === 'discard') choice = 'Discard';
        if (mode === 'retarget') current = false;
        if (mode === 'trust') vscode.state.isTrusted = false;
        if (mode === 'appeared') exists = true;
        if (mode === 'dirty') text = 'user work';
      };
      await workflow.open(target);
      assert.equal(edits.length, 0);
    });
  }
  test('existing config opens at the selected field with no generated edit', async () => {
    exists = true; text = '[profiles.run]\ndevice = "bad"';
    await workflow.open(target, ['profiles', 'run', 'device']);
    assert.equal(edits.length, 0); assert.equal(shown.length, 1);
    assert.equal(shown[0][1].selection.start.character, text.indexOf('device'));
  });
  test('execution input Edit selects the exact shape in the selected quoted profile without edits', async () => {
    exists = true; text = '[profiles.other.inputs]\nargs = [{ kind = "tensor", shape = [9] }]\n'
      + '[profiles."run.v2".inputs]\nargs = [{ kind = "tensor", shape = [1, 8], dtype = "float32" }]';
    await workflow.reveal(target, ['profiles', 'run.v2', 'inputs', 'args', 0, 'shape']);
    assert.equal(edits.length, 0); assert.equal(shown.length, 1);
    assert.equal(shown[0][1].selection.start.character, text.lastIndexOf('shape'));
  });
  test('navigation refuses missing files and stale targets rather than creating configuration', async () => {
    await assert.rejects(workflow.reveal(target, ['profiles', 'run', 'mode']));
    assert.equal(edits.length, 0); assert.equal(shown.length, 0);
    current = false; const before = reads;
    await workflow.reveal(target, ['profiles', 'run', 'mode']);
    assert.equal(reads, before);
  });
  test('diagnostics preserve full field and remedy and selection opens that field', async () => {
    exists = true; text = '[profiles.run]\ndevice = "bad"';
    patch(vscode.window, 'showQuickPick', async (items: any[]) => items[0]);
    await workflow.show(target, [{ code: 'invalid', phase: 'static', profile: 'run',
      field: ['profiles', 'run', 'device'], reason: 'Invalid device.', remedy: 'Use cpu, mps or cuda:N.' }]);
    assert.match(diagnostics[0].message, /profiles.run.device: Invalid device. Use cpu/);
    assert.equal(shown[0][1].selection.start.character, text.indexOf('device'));
  });
  test('retarget before setup does not read or open another workspace', async () => {
    current = false; await workflow.open(target); assert.equal(reads, 0);
  });
});
