import * as assert from 'node:assert/strict';
import * as vscode from 'vscode';
import { registerCanvasCommands } from '../../graphSurface';
import { SidebarCanvas, SIDEBAR_CANVAS_VIEW, CANVAS_VISIBLE } from '../../sidebarCanvas';
import { LocalBrowserGraph } from '../../localBrowserGraph';
import { commands, editorState, openStubEditor, showStubEditor, reset, resetEditors, resetWebviews, state, webviewState } from '../vscodeStub';

suite('Canvas command routing', () => {
  let canvas: SidebarCanvas;
  let registrations: vscode.Disposable[];
  let opens: unknown[], projects: unknown[], browsers: number;
  let prepares: number;
  setup(() => {
    reset(); resetEditors(); resetWebviews(); opens = []; projects = []; browsers = 0; prepares = 0;
    canvas = new SidebarCanvas({
      prepare: async () => { prepares++; return { appUrl: 'http://127.0.0.1:9000/', appOrigin: 'http://127.0.0.1:9000', port: 9000, workspaceDir: '/ws' }; },
      receive() {}, replay() {}, channel: { appendLine() {} } as unknown as vscode.OutputChannel,
    });
    registrations = [canvas.register(), ...registerCanvasCommands(canvas, {
      open: async (...args) => { opens.push(args); },
      openProject: async (...args) => { projects.push(args); },
      openSelectedInBrowser: async () => { browsers++; },
    } as Pick<LocalBrowserGraph, 'open' | 'openProject' | 'openSelectedInBrowser'>)];
  });
  teardown(() => { canvas.dispose(); registrations.forEach(r => r.dispose()); resetWebviews(); resetEditors(); reset(); });

  test('Get Started reveals the same starting screen without setup or a welcome editor', async () => {
    await commands.executeCommand('tensorviz.getStarted');
    const view = webviewState.views[0];
    assert.match(view.webview.html, /Explore your model/);
    assert.equal(canvas.visible, true);
    await commands.executeCommand('tensorviz.hideCanvas');
    await commands.executeCommand('tensorviz.getStarted');
    assert.equal(canvas.visible, true);
    assert.equal(webviewState.views.length, 1);
    assert.equal(prepares, 0, 'getting started is not consent to set up a runtime');
    assert.deepEqual(opens, []); assert.deepEqual(projects, []);
    assert.ok(webviewState.commands.every(c => c.id !== 'workbench.action.openWalkthrough'));
  });

  test('CodeLens/Models arguments select the exact class on the singleton regardless of legacy settings', async () => {
    openStubEditor('/ws/incidental.py', 'class Other: pass');
    const target = { uri: vscode.Uri.file('/ws/net.py').toString(), cls: 'Second' };
    for (const surface of ['browser', 'editor', 'sidebar', 'invalid']) {
      state.configuration.set('tensorviz.graphSurface', surface);
      await commands.executeCommand('tensorviz.openAsGraph', target);
    }
    assert.deepEqual(opens, Array(4).fill([target, 'canvas']));
    assert.equal(webviewState.views.length, 1);
    assert.equal(browsers, 0);
  });

  test('sidebar home freezes the editor file for opening and retry', async () => {
    const targets: unknown[] = [];
    registrations.push(vscode.commands.registerCommand('tensorviz.openAsGraph', async target => {
      targets.push(target);
      openStubEditor('/ws/unrelated.py', '');
      throw new Error('Discovery failed');
    }));
    await canvas.reveal();
    openStubEditor('/ws/hf_models.py', '');
    const view = webviewState.views[0];
    view.receive({ type: 'homeAction', action: 'open' });
    await new Promise<void>(resolve => setImmediate(resolve));
    view.receive({ type: 'homeAction', action: 'retry' });
    await new Promise<void>(resolve => setImmediate(resolve));
    assert.deepEqual(targets, [{ uri: '/ws/hf_models.py' }, { uri: '/ws/hf_models.py' }]);
  });

  test('Show/Open/Toggle without a target retain the model or useful empty Canvas', async () => {
    openStubEditor('/ws/incidental.py', 'class Other: pass');
    await commands.executeCommand('tensorviz.openCanvas');
    const view = webviewState.views[0], writes = view.htmlWrites;
    await commands.executeCommand('tensorviz.toggleCanvas');
    assert.equal(canvas.requestedVisible, false);
    assert.equal(webviewState.contexts.get(CANVAS_VISIBLE), false, 'native hide updates actual visibility');
    view.setVisible(false);
    assert.equal(webviewState.contexts.get(CANVAS_VISIBLE), false);
    await commands.executeCommand('tensorviz.toggleCanvas');
    await commands.executeCommand('tensorviz.showSidebarCanvas');
    assert.equal(view.htmlWrites, writes);
    assert.equal(webviewState.views.length, 1);
    assert.deepEqual(opens, []); assert.deepEqual(projects, []);
    assert.equal(browsers, 0);
    assert.equal(canvas.visible, true);
    assert.ok(webviewState.commands.filter(c => !c.id.startsWith('tensorviz.') && c.id !== 'setContext')
      .every(c => [ `${SIDEBAR_CANVAS_VIEW}.focus`, `${SIDEBAR_CANVAS_VIEW}.removeView` ].includes(c.id)));
  });

  test('browser actions delegate to the retained owner and never pass the incidental editor or title argument', async () => {
    openStubEditor('/other/model.py', 'class Other: pass');
    await commands.executeCommand('tensorviz.openCanvasInBrowser', { uri: '/other/model.py', cls: 'Other' });
    await commands.executeCommand('tensorviz.openProjectInBrowser', { folder: '/other' });
    assert.equal(browsers, 2); assert.deepEqual(opens, []); assert.deepEqual(projects, []);
    assert.equal(webviewState.views.length, 0, 'opening the browser does not reveal or create another Canvas');
  });

  test('project command preserves its explicit folder and reveal failures do not select a target', async () => {
    await commands.executeCommand('tensorviz.openProject', { folder: '/chosen' });
    assert.deepEqual(projects, [[{ folder: '/chosen' }, 'canvas']]);
    canvas.dispose();
    await commands.executeCommand('tensorviz.openAsGraph', { uri: '/ws/net.py', cls: 'Net' });
    assert.deepEqual(opens, []);
  });

  test('palette targets are captured before revealing Canvas moves editor focus', async () => {
    state.workspaceFolders = [{ uri: { fsPath: '/first' } }, { uri: { fsPath: '/chosen' } }];
    registrations.push(...registerCanvasCommands({
      visible: false, hide: async () => {}, reveal: async () => {},
      show: async () => { showStubEditor(openStubEditor('/first/incidental.py', '').document); return true; },
    }, {
      open: async (...args) => { opens.push(args); },
      openProject: async (...args) => { projects.push(args); },
      openSelectedInBrowser: async () => {},
    }));
    showStubEditor(openStubEditor('/chosen/net.py', 'class Net: pass').document);
    await commands.executeCommand('tensorviz.openAsGraph');
    assert.deepEqual(projects, [[{ folder: '/chosen' }, 'canvas']], 'unnamed class keeps project-opening behavior');
    showStubEditor(openStubEditor('/chosen/net.py', 'class Net: pass').document);
    await commands.executeCommand('tensorviz.openAsGraph', { cls: 'Net' });
    assert.deepEqual(opens, [[{ uri: '/chosen/net.py', cls: 'Net' }, 'canvas']]);
  });

  test('Canvas is revealed before model discovery; the clicked file stays frozen and cancellation never opens the project', async () => {
    let selected = true;
    let reveals = 0;
    const resolved: unknown[] = [];
    registrations.push(...registerCanvasCommands({
      visible: false, hide: async () => {}, reveal: async () => {},
      show: async () => { reveals++; showStubEditor(openStubEditor('/ws/other.py', '').document); return true; },
    }, {
      open: async (...args) => { opens.push(args); },
      openProject: async (...args) => { projects.push(args); },
      openSelectedInBrowser: async () => {},
    }, async target => {
      assert.ok(reveals > resolved.length, 'progress must be visible while class discovery runs');
      resolved.push(target);
      return selected ? { uri: target.uri!, cls: 'Net' } : undefined;
    }));
    showStubEditor(openStubEditor('/ws/model.py', 'class Net: pass').document);
    await commands.executeCommand('tensorviz.openAsGraph');
    assert.deepEqual(resolved, [{ uri: '/ws/model.py' }]);
    assert.deepEqual(opens, [[{ uri: '/ws/model.py', cls: 'Net' }, 'canvas']]);
    assert.equal(reveals, 1, 'discovery and opening share one reveal');
    selected = false;
    await commands.executeCommand('tensorviz.openAsGraph');
    assert.equal(opens.length, 1);
    assert.deepEqual(projects, []);
  });
});


suite('Graph command selected tab', () => {
  let registrations: vscode.Disposable[], restore: (() => void)[];
  let resolved: ({ uri?: string; cls?: string })[], opens: unknown[];
  let activeInput: unknown;
  let resolve: (target: { uri?: string; cls?: string }) => Promise<{ uri: string; cls: string } | undefined>;
  function patch(object: object, key: string, value: unknown): void {
    const previous = Object.getOwnPropertyDescriptor(object, key);
    Object.defineProperty(object, key, { value, configurable: true, writable: true });
    restore.push(() => previous ? Object.defineProperty(object, key, previous) : Reflect.deleteProperty(object, key));
  }
  setup(() => {
    reset(); resetEditors(); restore = []; resolved = []; opens = []; activeInput = undefined;
    patch(vscode, 'TabInputText', class { constructor(public uri: vscode.Uri) {} });
    patch(vscode.window, 'tabGroups', { all: [], activeTabGroup: { get activeTab() { return activeInput === undefined ? undefined : { input: activeInput }; } } });
    resolve = async target => target.uri?.endsWith('.py') ? { uri: target.uri, cls: 'Net' } : undefined;
    registrations = registerCanvasCommands({ visible: false, hide: async () => {}, reveal: async () => {}, show: async () => true }, {
      open: async (...args) => { opens.push(args); },
      openProject: async () => assert.fail('model command must not fall through to a project'),
      openSelectedInBrowser: async () => {},
    }, async target => { resolved.push(target); return resolve(target); });
  });
  teardown(() => { registrations.forEach(r => r.dispose()); restore.reverse().forEach(undo => undo()); resetEditors(); reset(); });

  test('a selected model tab survives temporary loss of text-editor focus', async () => {
    activeInput = new vscode.TabInputText(vscode.Uri.file('/ws/new_model.py'));
    assert.equal(vscode.window.activeTextEditor, undefined);
    await commands.executeCommand('tensorviz.openAsGraph');
    assert.deepEqual(resolved, [{ uri: '/ws/new_model.py' }]);
    assert.deepEqual(opens, [[{ uri: '/ws/new_model.py', cls: 'Net' }, 'canvas']]);
  });

  test('webview focus retains the visible Python editor before considering unrelated tabs', async () => {
    openStubEditor('/ws/hf_models.py', 'class Net: pass');
    patch(vscode.window, 'activeTextEditor', undefined);
    activeInput = new vscode.TabInputText(vscode.Uri.file('/ws/unrelated.py'));
    await commands.executeCommand('tensorviz.openAsGraph');
    assert.deepEqual(resolved, [{ uri: '/ws/hf_models.py' }]);
    assert.deepEqual(opens, [[{ uri: '/ws/hf_models.py', cls: 'Net' }, 'canvas']]);
  });

  test('a Python tab in another group survives webview focus', async () => {
    (vscode.window.tabGroups.all as unknown as object[]).push({
      activeTab: { input: new vscode.TabInputText(vscode.Uri.file('/ws/model.py')) },
    });
    activeInput = { webview: true };
    await commands.executeCommand('tensorviz.openAsGraph');
    assert.deepEqual(resolved, [{ uri: '/ws/model.py' }]);
  });

  test('a non-Python active editor does not hide a visible Python file', async () => {
    openStubEditor('/ws/model.pyw', '');
    openStubEditor('/ws/README.md', '');
    await commands.executeCommand('tensorviz.openAsGraph');
    assert.deepEqual(resolved, [{ uri: '/ws/model.pyw' }]);
  });

  test('without a usable Python file the resolver can offer the workspace picker', async () => {
    openStubEditor('/ws/README.md', '');
    activeInput = new vscode.TabInputText(vscode.Uri.parse('untitled:/ws/model.py'));
    resolve = async target => {
      assert.equal(target.uri, undefined);
      return { uri: '/ws/picked.py', cls: 'Picked' };
    };
    await commands.executeCommand('tensorviz.openAsGraph');
    assert.deepEqual(opens, [[{ uri: '/ws/picked.py', cls: 'Picked' }, 'canvas']]);
  });

  test('the explicit selection wins over both editor and tab', async () => {
    openStubEditor('/ws/editor.py', '');
    activeInput = new vscode.TabInputText(vscode.Uri.file('/ws/tab.py'));
    await commands.executeCommand('tensorviz.openAsGraph', { uri: '/ws/chosen.py', cls: 'Chosen' });
    assert.deepEqual(resolved, [{ uri: '/ws/chosen.py', cls: 'Chosen' }]);
  });

  test('the focused text editor wins when the tab snapshot has not caught up', async () => {
    openStubEditor('/ws/new.py', '');
    activeInput = new vscode.TabInputText(vscode.Uri.file('/ws/old.py'));
    await commands.executeCommand('tensorviz.openAsGraph');
    assert.equal(resolved[0].uri, '/ws/new.py');
  });

  test('selection stays on the requested file while discovery moves editor focus', async () => {
    activeInput = new vscode.TabInputText(vscode.Uri.file('/ws/first.py'));
    resolve = async target => {
      activeInput = new vscode.TabInputText(vscode.Uri.file('/ws/second.py'));
      return { uri: target.uri!, cls: 'Net' };
    };
    await commands.executeCommand('tensorviz.openAsGraph');
    assert.deepEqual(opens, [[{ uri: '/ws/first.py', cls: 'Net' }, 'canvas']]);
  });

  for (const kind of ['no_tab', 'custom_tab', 'non_python', 'non_host']) {
    test(`${kind} cannot reopen an unrelated model`, async () => {
      openStubEditor('/ws/old.py', '');
      editorState.visibleTextEditors.length = 0;
      activeInput = kind === 'no_tab' ? undefined : kind === 'custom_tab'
        ? { uri: vscode.Uri.file('/ws/old.py') } : new vscode.TabInputText(kind === 'non_host'
          ? vscode.Uri.parse('vscode-local:/ws/old.py') : vscode.Uri.file('/ws/README.md'));
      await commands.executeCommand('tensorviz.openAsGraph');
      assert.deepEqual(resolved, [{ uri: undefined }]);
      assert.equal(opens.length, 0);
    });
  }
});
