import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import * as vm from 'node:vm';
import { CanvasShell, CanvasBridge, shellHtml, errorHtml } from '../../shell';
import { StubWebviewView, setInfoResponder, setSaveDialogChoice, webviewState } from '../vscodeStub';
import * as stub from '../vscodeStub';

suite('shared canvas shell', () => {
  const target = { appUrl: 'http://127.0.0.1:9000/?token=secret', appOrigin: 'http://127.0.0.1:9000', port: 9000, workspaceDir: '/ws' };
  test('explicit copy writes exact selected text, acknowledges success/failure and rejects oversized requests', async () => {
    const view = new StubWebviewView(), writes: string[] = [], logs: string[] = [];
    Object.assign(stub, { env: { clipboard: { writeText: async (text: string) => { writes.push(text); if (text === 'fail') throw new Error('private failure'); } } } });
    const bridge = new CanvasBridge(view.webview as never, target, { appendLine: (line: string) => logs.push(line) } as never);
    try {
      await bridge.handleForTest({ type: 'clipboard', requestId: 'copy', text: 'Selected\nreply' });
      await bridge.handleForTest({ type: 'clipboard', requestId: 'failed', text: 'fail' });
      await bridge.handleForTest({ type: 'clipboard', requestId: 'large', text: 'x'.repeat(64001) });
      assert.deepEqual(writes, ['Selected\nreply', 'fail']);
      assert.deepEqual(view.messages, [{ type: 'clipboardWritten', requestId: 'copy' }, { type: 'clipboardWritten', requestId: 'failed', error: true }]);
      assert.deepEqual(logs, []);
    } finally { bridge.dispose(); delete (stub as { env?: unknown }).env; }
  });
  test('explicit paste reads the native clipboard, bounds the reply and never logs its text', async () => {
    const view = new StubWebviewView();
    const logs: string[] = [];
    let reads = 0;
    Object.assign(stub, { env: { clipboard: { readText: async () => { reads++; return 'private clipboard'.repeat(2000); } } } });
    const bridge = new CanvasBridge(view.webview as never, target, { appendLine: (line: string) => logs.push(line) } as never);
    try {
      assert.equal(reads, 0, 'mount does not read clipboard');
      for (const requestId of [undefined, '', 'x'.repeat(129)]) await bridge.handleForTest({ type: 'readClipboard', requestId } as never);
      assert.equal(reads, 0, 'invalid requests do not read clipboard');
      await bridge.handleForTest({ type: 'readClipboard', requestId: 'paste-1' });
      assert.equal(reads, 1);
      assert.deepEqual(view.messages, [{ type: 'clipboardText', requestId: 'paste-1', text: 'private clipboard'.repeat(2000).slice(0, 16000) }]);
      assert.deepEqual(logs, []);
      assert.ok(!JSON.stringify(bridge.received).includes('private clipboard'));
    } finally { bridge.dispose(); delete (stub as { env?: unknown }).env; }
  });
  test('native clipboard failures produce a correlated failure without exposing error details', async () => {
    const view = new StubWebviewView();
    Object.assign(stub, { env: { clipboard: { readText: async () => { throw new Error('private host error'); } } } });
    const bridge = new CanvasBridge(view.webview as never, target, { appendLine() {} } as never);
    try {
      await bridge.handleForTest({ type: 'readClipboard', requestId: 'paste-2' });
      assert.deepEqual(view.messages, [{ type: 'clipboardText', requestId: 'paste-2', error: true }]);
    } finally { bridge.dispose(); delete (stub as { env?: unknown }).env; }
  });
  test('the palette request invokes only the fixed native command without arguments', async () => {
    const view = new StubWebviewView();
    const bridge = new CanvasBridge(view.webview as never, target, { appendLine: () => {} } as never);
    const before = webviewState.commands.length;
    webviewState.handlers.set('workbench.action.showCommands', () => {});
    try {
      view.receive({ type: 'showCommandPalette', command: 'workbench.action.closeWindow', args: ['ignored'] });
      await new Promise(resolve => setImmediate(resolve));
      assert.deepEqual(webviewState.commands.slice(before), [{ id: 'workbench.action.showCommands', args: [] }]);
    } finally {
      webviewState.handlers.delete('workbench.action.showCommands');
      bridge.dispose();
    }
  });
  test('the iframe bridge accepts requests only from its exact frame and origin', () => {
    const toHost: unknown[] = [], toFrame: unknown[] = [];
    const frame = { postMessage: (message: unknown) => toFrame.push(message) };
    let receive!: (event: unknown) => void;
    const html = shellHtml(target);
    vm.runInNewContext(html.match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/)![1], {
      acquireVsCodeApi: () => ({ postMessage: (message: unknown) => toHost.push(message) }),
      document: { getElementById: () => ({ contentWindow: frame }) },
      MutationObserver: class { observe() {} },
      window: { addEventListener: (_type: string, listener: typeof receive) => { receive = listener; } },
    });
    const request = { type: 'canvasConnect', protocolVersion: 1 };
    receive({ source: frame, origin: 'https://other.example', data: request });
    receive({ source: {}, origin: target.appOrigin, data: request });
    assert.deepEqual(toHost, []);
    receive({ source: frame, origin: target.appOrigin, data: request });
    assert.deepEqual(toHost, [request]);
    assert.match(html, /frame-src http:\/\/127\.0\.0\.1:9000;/);
    assert.deepEqual(CanvasShell.options(9000), CanvasBridge.options(9000), 'editor-only comparison retains the same webview options');
  });

  test('the empty Canvas opens only the native model picker, never a webview target or command', async () => {
    const view = new StubWebviewView();
    const bridge = new CanvasBridge(view.webview as never, target, { appendLine() {} } as never);
    const before = webviewState.commands.length;
    webviewState.handlers.set('tensorviz.openAsGraph', () => {});
    try {
      view.receive({ type: 'openAsGraph', command: 'workbench.action.closeWindow', uri: 'file:///foreign.py', args: ['ignored'] });
      await new Promise(resolve => setImmediate(resolve));
      assert.deepEqual(webviewState.commands.slice(before), [{ id: 'tensorviz.openAsGraph', args: [] }]);
    } finally {
      webviewState.handlers.delete('tensorviz.openAsGraph');
      bridge.dispose();
    }
  });

  test('the shell supplies the IDE theme at connection and forwards later theme changes', () => {
    const messages: { message: any; origin: string }[] = [];
    const frame = { postMessage: (message: unknown, origin: string) => messages.push({ message, origin }) };
    let receive!: (event: unknown) => void;
    let changed!: () => void;
    let kind = 'vscode-light';
    const body = { getAttribute: () => kind };
    vm.runInNewContext(shellHtml(target).match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/)![1], {
      acquireVsCodeApi: () => ({ postMessage() {} }),
      document: { body, getElementById: () => ({ contentWindow: frame }) },
      MutationObserver: class {
        constructor(callback: () => void) { changed = callback; }
        observe(element: unknown, options: unknown) {
          assert.equal(element, body);
          assert.equal(JSON.stringify(options), JSON.stringify({ attributes: true, attributeFilter: ['data-vscode-theme-kind'] }));
        }
      },
      window: { addEventListener: (_type: string, listener: typeof receive) => { receive = listener; } },
    });
    receive({ source: frame, origin: target.appOrigin, data: { type: 'ready' } });
    assert.equal(messages[0].message.type, 'hostReady');
    assert.equal(messages[0].message.theme, 'light');
    assert.equal(messages[0].origin, target.appOrigin);
    messages.length = 0;
    receive({ data: { type: 'canvasConnected', sessionId: 's', viewId: 'v' } });
    assert.equal(messages[0].message.theme, 'light');
    assert.equal(messages[0].origin, target.appOrigin);
    kind = 'vscode-dark'; changed();
    assert.equal(messages[1].message.type, 'canvasTheme');
    assert.equal(messages[1].message.theme, 'dark');
    kind = 'vscode-high-contrast-light'; changed();
    assert.equal(messages[2].message.theme, 'light');
  });

  test('consumed native traffic is not retained or logged; disposing removes only the bridge', async () => {
    const view = new StubWebviewView();
    const logs: string[] = [];
    const bridge = new CanvasBridge(view.webview as never, target, { appendLine: (line: string) => logs.push(line) } as never);
    let requests = 0;
    bridge.onMessage(() => { requests++; return true; });
    view.receive({ type: 'canvasPresentation', state: { privateData: 'not a log' } });
    assert.equal(requests, 1);
    assert.deepEqual(bridge.received, []);
    assert.deepEqual(logs, []);
    bridge.dispose();
    view.receive({ type: 'canvasConnect' });
    assert.equal(requests, 1);
    assert.equal(view.disposed, false, 'the bridge does not own a view or a controller');
    assert.match(errorHtml('<offline>', '&detail', 'tensorViz: Show Sidebar Canvas'), /&lt;offline&gt;/);
  });

  test('SVG export asks where to save, then offers Open and Reveal', async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-export-'));
    const view = new StubWebviewView();
    const bridge = new CanvasBridge(view.webview as never, { ...target, workspaceDir: dir }, { appendLine() {} } as never);
    const svg = Buffer.from('<svg><text>Net</text></svg>').toString('base64');
    try {
      setSaveDialogChoice(null);
      await bridge.handleForTest({ type: 'saveFile', file: { name: 'graph.svg', bytes: svg } });
      assert.equal(fs.existsSync(path.join(dir, 'tensorviz-exports', 'graph.svg')), false);
      setSaveDialogChoice(path.join(dir, 'chosen.svg'));
      setInfoResponder(() => 'Reveal');
      webviewState.handlers.set('revealFileInOS', () => {});
      await bridge.handleForTest({ type: 'saveFile', file: { name: 'graph.svg', bytes: svg } });
      assert.equal(fs.readFileSync(path.join(dir, 'chosen.svg'), 'utf8'), '<svg><text>Net</text></svg>');
      assert.match(stub.infoMessages.at(-1) ?? '', /Saved chosen\.svg/);
      assert.equal(webviewState.commands.at(-1)?.id, 'revealFileInOS');
      assert.equal((webviewState.commands.at(-1)?.args[0] as { fsPath: string }).fsPath, path.join(dir, 'chosen.svg'));
    } finally {
      bridge.dispose();
      setSaveDialogChoice(undefined);
      setInfoResponder(undefined);
      webviewState.handlers.delete('revealFileInOS');
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});
