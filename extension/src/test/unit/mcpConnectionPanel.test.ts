import * as assert from 'node:assert/strict';
import { runInNewContext } from 'node:vm';
import { closeMcpConnection, showMcpConnection, updateMcpConnectionStatus } from '../../mcpConnectionPanel';
import type { GraphOpeningState, LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import { graphMcpHelpRequest } from '@tensorviz/contracts/graphHelpRequest';

const vscode = require('../vscodeStub');
const target: LocalBrowserTargetIdentity = { workspaceFolder: '/workspace', file: 'whisper/model.py', cls: 'Whisper',
  sourceDigest: 'a'.repeat(64), documentVersion: 1, backendRevision: 0 };

suite('MCP setup guidance', () => {
  let restore: Array<() => void>, panel: any, receive: (message: unknown) => Promise<void>;
  let copied: string[], commands: string[], replies: any[], commandArguments: unknown[][];
  const context = { subscriptions: [] } as any;
  function patch(object: any, key: string, value: any) {
    const descriptor = Object.getOwnPropertyDescriptor(object, key);
    Object.defineProperty(object, key, { configurable: true, writable: true, value });
    restore.push(() => descriptor ? Object.defineProperty(object, key, descriptor) : delete object[key]);
  }
  setup(() => {
    restore = []; copied = []; commands = []; replies = []; commandArguments = [];
    let dispose = () => {};
    panel = { webview: { html: '', postMessage: async (message: any) => replies.push(message),
      onDidReceiveMessage: (handler: typeof receive) => { receive = handler; return { dispose() {} }; } },
    reveal() {}, dispose: () => dispose(), onDidDispose: (handler: () => void) => { dispose = handler; } };
    patch(vscode.window, 'createWebviewPanel', () => panel);
    patch(vscode, 'ViewColumn', { Active: -1 });
    patch(vscode, 'env', { clipboard: { writeText: async (text: string) => { copied.push(text); } } });
    patch(vscode.commands, 'executeCommand', async (name: string, ...args: unknown[]) => { commands.push(name); commandArguments.push(args); });
  });
  teardown(() => { closeMcpConnection(); restore.reverse().forEach(undo => undo()); });

  test('SSH guidance distinguishes agent locations and forwarding is explicit', async () => {
    showMcpConnection('http://127.0.0.1:8849/mcp', context, target, 'ssh');
    assert.match(panel.webview.html, /Use it for agents running on that server/);
    assert.match(panel.webview.html, /Forward MCP to This Desktop/);
    assert.doesNotMatch(panel.webview.html, /id="setup-agent"/);
    await receive({ type: 'setup_agent' });
    assert.deepEqual(commands, []);
    await receive({ type: 'forward' });
    assert.deepEqual(commands, ['tensorviz.forwardMcp']);
    showMcpConnection('http://127.0.0.1:53128/mcp', context, target, 'desktop');
    assert.match(panel.webview.html, /Model files, Python and tools still run on the SSH host/);
    await receive({ type: 'copy' });
    assert.equal(copied[0], 'http://127.0.0.1:53128/mcp');
    await receive({ type: 'forward' });
    assert.equal(commands.length, 1, 'stale or hidden forwarding controls do nothing');
    await receive({ type: 'setup_agent' });
    assert.equal(commands.length, 1, 'remote guidance cannot install into desktop agent settings');
  });

  test('local MCP setup enters the approved connection flow without a capture', async () => {
    showMcpConnection('http://127.0.0.1:8849/mcp', context);
    assert.match(panel.webview.html, /id="setup-agent"/);
    assert.match(panel.webview.html, /do not add the URL again/);
    assert.match(panel.webview.html, /If you use the plugin, do not add this connection/);
    assert.doesNotMatch(panel.webview.html, /Install plugin|MCP only/);
    await receive({ type: 'setup_agent' });
    assert.deepEqual(commands, ['tensorviz.connectMcp']);
  });

  test('configured connections skip manual setup and follow actual connection and disconnect receipts', async () => {
    const url = 'http://127.0.0.1:8849/mcp';
    updateMcpConnectionStatus({ state: 'Ready', url, sessions: 0 });
    showMcpConnection(url, context, undefined, 'local', 'Codex');
    assert.match(panel.webview.html, /VS Code connection configured for Codex/);
    assert.match(panel.webview.html, /do not need to add another server or copy a URL/);
    assert.doesNotMatch(panel.webview.html, /plugin includes|passed the connection check/);
    assert.doesNotMatch(panel.webview.html, /<h2>Connect your agent<\/h2>|Creating this URL|Set up coding agent/);
    assert.match(panel.webview.html, /<details><summary>Connection details for other agents/);
    await receive({ type: 'ready' });
    assert.equal(replies.at(-1).title, 'Waiting for your agent');
    assert.equal(replies.at(-1).repair, false);
    assert.deepEqual(commands, []);
    updateMcpConnectionStatus({ state: 'Connected', url, sessions: 1 });
    assert.equal(replies.at(-1).title, 'An agent is connected', 'a receipt does not identify which agent or chat loaded the tools');
    updateMcpConnectionStatus({ state: 'Off', sessions: 0 });
    assert.equal(replies.at(-1).title, 'Graph tools are unavailable');
    assert.equal(replies.at(-1).repair, true);
    updateMcpConnectionStatus({ state: 'Connected', url: 'http://127.0.0.1:8850/mcp', sessions: 1 });
    assert.equal(replies.at(-1).title, 'The MCP address changed');
    await receive({ type: 'setup_agent' });
    assert.deepEqual(commands, ['tensorviz.connectMcp']);
  });

  test('the rendered connection screen updates its status without replacing the model prompt', () => {
    showMcpConnection('http://127.0.0.1:8849/mcp', context, target, 'local', 'Claude Code');
    const elements = new Map<string, any>();
    let onMessage = (_event: any) => {};
    const sent: unknown[] = [];
    runInNewContext(panel.webview.html.match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/)[1], {
      acquireVsCodeApi: () => ({ postMessage: (value: unknown) => sent.push(value) }),
      window: { addEventListener: (_name: string, handler: typeof onMessage) => { onMessage = handler; } },
      document: { getElementById: (id: string) => {
        if (!elements.has(id)) elements.set(id, { textContent: '', hidden: true, addEventListener() {} });
        return elements.get(id);
      } },
    });
    assert.equal((sent[0] as any).type, 'ready');
    const prompt = elements.get('prompt').textContent;
    onMessage({ data: { type: 'connection', title: 'Waiting for your agent', detail: 'Start a new chat.', repair: false } });
    assert.equal(elements.get('connection-title').textContent, 'Waiting for your agent');
    assert.equal(elements.get('setup-agent').hidden, true);
    onMessage({ data: { type: 'connection', title: 'Graph tools are unavailable', detail: 'Reconnect.', repair: true } });
    assert.equal(elements.get('setup-agent').hidden, false);
    assert.equal(elements.get('prompt').textContent, prompt);
  });

  test('defaults to any model without starting a capture and offers the selected target separately', async () => {
    showMcpConnection('http://127.0.0.1:8766/mcp', context, target);
    assert.match(panel.webview.html, /Creating this URL does not connect an agent/);
    assert.match(panel.webview.html, /an agent is optional/);
    assert.match(panel.webview.html, /<details>\s*<summary>Use the model open in Canvas<\/summary>/);
    assert.doesNotMatch(panel.webview.html, /<details open/);
    assert.deepEqual(commands, []);
    await receive({ type: 'copy' });
    await receive({ type: 'copy_prompt' });
    assert.equal(copied[0], 'http://127.0.0.1:8766/mcp');
    assert.match(copied[1], /Identify the intended model file and class, then use open_graph/);
    assert.doesNotMatch(copied[1], /Whisper|\/workspace|whisper\/model/);
    await receive({ type: 'copy_target_prompt' });
    assert.match(copied[2], /Whisper in whisper\/model.py in workspace \/workspace/);
    assert.match(copied[2], /Use open_graph with file "\/workspace\/whisper\/model.py" and cls "Whisper"/);
    assert.match(copied[2], /verify the returned target/);
    assert.equal(replies.at(-1).text, 'Setup request copied.');
    await receive({ type: 'manual' });
    assert.deepEqual(commands, ['tensorviz.openCanvas'], 'return must only reveal the retained Canvas');
  });

  test('reopening refreshes the suggested target rather than copying the earlier class', async () => {
    showMcpConnection('http://127.0.0.1:8766/mcp', context, target);
    showMcpConnection('http://127.0.0.1:8767/mcp', context, { ...target, cls: 'MultiHeadAttention' });
    await receive({ type: 'copy_target_prompt' });
    assert.match(copied[0], /configure MultiHeadAttention/);
    await receive({ type: 'copy' });
    assert.equal(copied[1], 'http://127.0.0.1:8767/mcp');
  });

  test('file names are inert text in the webview and never break the script', () => {
    const file = 'models/</script><script>bad()</script>.py';
    showMcpConnection('http://127.0.0.1:8766/mcp', context, { ...target, file });
    assert.equal((panel.webview.html.match(/<script /g) ?? []).length, 1);
    assert.equal((panel.webview.html.match(/<\/script>/g) ?? []).length, 1);
    const elements = new Map<string, any>();
    runInNewContext(panel.webview.html.match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/)[1], {
      acquireVsCodeApi: () => ({ postMessage() {} }), window: { addEventListener() {} },
      document: { getElementById: (id: string) => {
        if (!elements.has(id)) elements.set(id, { textContent: '', addEventListener() {} });
        return elements.get(id);
      } },
    });
    assert.ok(elements.get('target').textContent.includes(file));
  });

  test('missing selection keeps generic guidance and rejects selected-model and unknown actions', async () => {
    showMcpConnection('http://127.0.0.1:8766/mcp', context);
    assert.match(panel.webview.html, /Tell your agent which model you want to open/);
    assert.doesNotMatch(panel.webview.html, /<details>/);
    await receive({ type: 'copy_prompt' });
    assert.match(copied.pop()!, /Identify the intended model file and class, then use open_graph/);
    await receive({ type: 'copy_target_prompt' });
    await receive({ type: 'copy_help_prompt' });
    await receive({ type: 'configure_graph_setup' });
    assert.deepEqual(commands, []); assert.deepEqual(copied, []);
  });

  test('failure help freezes the model, opening and environment without sending a request or retrying', async () => {
    const opening: GraphOpeningState = { id: 'failed-opening', target: { ...target }, phase: 'failed', hasCapture: false,
      failure: { kind: 'environment', missingModules: ['torchmetrics', 'cv2'] }, detail: 'Model import failed.' };
    updateMcpConnectionStatus({ state: 'Ready', url: 'http://127.0.0.1:8766/mcp', sessions: 0 });
    showMcpConnection('http://127.0.0.1:8766/mcp', context, opening.target, 'local', undefined,
      { opening, interpreter: '/project/.venv/bin/python', environmentSource: 'project' });
    assert.match(panel.webview.html, /Investigate this failure/);
    assert.match(panel.webview.html, /No request is sent automatically/);
    assert.deepEqual(commands, []); assert.deepEqual(copied, []);
    await receive({ type: 'ready' });
    assert.equal(replies.at(-1).title, 'Waiting for your agent');
    updateMcpConnectionStatus({ state: 'Connected', url: 'http://127.0.0.1:8766/mcp', sessions: 1 });
    assert.equal(replies.at(-1).title, 'An agent is connected');
    opening.target.cls = 'DifferentModel'; opening.detail = 'A later failure.';
    await receive({ type: 'copy_help_prompt' });
    assert.match(copied[0], /MCP directly/);
    assert.match(copied[0], /failed-opening/);
    assert.match(copied[0], /Whisper/);
    assert.match(copied[0], /\/project\/\.venv\/bin\/python/);
    assert.match(copied[0], /torchmetrics/);
    assert.doesNotMatch(copied[0], /DifferentModel|A later failure/);
    assert.equal(replies.at(-1).text, 'Investigation request copied.');
    await receive({ type: 'setup_agent' });
    assert.deepEqual(commands, ['tensorviz.connectMcp']);
    assert.equal((commandArguments[0][0] as GraphOpeningState).target.cls, 'Whisper');
    assert.equal(commandArguments[0][1], true, 'explicit setup must enter agent configuration, not reopen the help guide');
    showMcpConnection('http://127.0.0.1:8766/mcp', context, target);
    await receive({ type: 'copy_help_prompt' });
    assert.equal(copied.length, 1, 'a normal guide must not copy a historical failure');
  });

  test('paused environment help does not label an unexecuted model as a failed forward', async () => {
    const opening: GraphOpeningState = { id: 'environment-opening', target, phase: 'needs_setup', hasCapture: false,
      detail: 'Choose Python.', environment: { reason: 'selection_required', candidates: [], detail: 'Choose Python.' } };
    showMcpConnection('http://127.0.0.1:8766/mcp', context, target, 'local', undefined, { opening });
    assert.match(panel.webview.html, /Investigate this environment/);
    assert.doesNotMatch(panel.webview.html, /Investigate this failure|opening that failed/);
    await receive({ type: 'copy_help_prompt' });
    assert.match(copied[0], /environment-opening/);
    assert.match(copied[0], /selection_required/);
    assert.doesNotMatch(copied[0], /TensorViz failure/);
  });
  test('input setup help carries pending values and continues the exact opening before starting another', async () => {
    const opening: GraphOpeningState = { id: 'pending-inputs', target, phase: 'needs_setup', hasCapture: false,
      detail: 'Complete inputs.', draft: { profileName: 'saved_model', profile: { kind: 'model',
        inputs: { args: [], kwargs: { x: { kind: 'tensor', shape: [3, 12], dtype: 'float64', requires_grad: false,
          generator: { kind: 'full', value: 2 } } } } } } };
    showMcpConnection('http://127.0.0.1:8766/mcp', context, target, 'local', undefined, { opening });
    assert.match(panel.webview.html, /Complete this input setup/);
    assert.doesNotMatch(panel.webview.html, /Investigate this failure/);
    await receive({ type: 'copy_help_prompt' });
    assert.match(copied[0], /suggest_graph_inputs/);
    assert.match(copied[0], /configure_graph_setup/);
    assert.match(copied[0], /pending-inputs/);
    assert.match(copied[0], /saved_model/);
    assert.match(copied[0], /float64/);
    assert.match(copied[0], /pending draft/);
    assert.deepEqual(commands, []);
  });

  test('failure details are bounded and script-like model text stays inert', () => {
    const opening: GraphOpeningState = { id: 'failed-opening', target: { ...target, file: '</script><script>bad()</script>.py' },
      phase: 'failed', hasCapture: false, failure: { kind: 'execution' }, detail: 'x'.repeat(4000) + 'PRIVATE_TAIL' };
    const request = graphMcpHelpRequest({ opening });
    assert.doesNotMatch(request, /PRIVATE_TAIL/);
    assert.match(request, /get_execution_environment/);
    assert.doesNotMatch(request, /reportedEnvironment/, 'an unknown interpreter must not be guessed');
    showMcpConnection('http://127.0.0.1:8766/mcp', context, opening.target, 'local', undefined, { opening });
    assert.equal((panel.webview.html.match(/<script /g) ?? []).length, 1);
    assert.equal((panel.webview.html.match(/<\/script>/g) ?? []).length, 1);
  });

  test('an unconfigured failure guide keeps setup available while reporting actual connection status', () => {
    const opening: GraphOpeningState = { id: 'failure', target, phase: 'failed', hasCapture: false, detail: 'Capture stopped.' };
    showMcpConnection('http://127.0.0.1:8766/mcp', context, target, 'local', undefined, { opening });
    const elements = new Map<string, any>(); let onMessage = (_event: any) => {};
    runInNewContext(panel.webview.html.match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/)[1], {
      acquireVsCodeApi: () => ({ postMessage() {} }),
      window: { addEventListener: (_name: string, handler: typeof onMessage) => { onMessage = handler; } },
      document: { getElementById: (id: string) => {
        if (!elements.has(id)) elements.set(id, { textContent: '', hidden: false, addEventListener() {} });
        return elements.get(id);
      } },
    });
    onMessage({ data: { type: 'connection', title: 'Waiting for your agent', detail: 'Set up or reconnect.', repair: false } });
    assert.equal(elements.get('setup-agent').hidden, false);
    assert.equal(elements.get('connection-title').textContent, 'Waiting for your agent');
    assert.match(elements.get('help-prompt').textContent, /"openingId": "failure"/);
  });
});
