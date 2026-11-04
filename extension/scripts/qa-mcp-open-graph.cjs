/** Installed-host MCP smoke test. qa-host.mjs supplies a disposable profile/workspace. */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const net = require('node:net');
const vscode = require('vscode');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StreamableHTTPClientTransport } = require('@modelcontextprotocol/sdk/client/streamableHttp.js');

const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(read, test, timeout = 120000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { const value = await read(); if (test(value)) return value; await pause(300); }
  throw new Error('Timed out waiting for MCP opening state');
}
exports.run = async () => {
  const root = process.env.TV_QA_ROOT;
  assert.ok(root && root.startsWith('/private/tmp/tensorviz-mcp-'), 'Use a task-owned disposable QA root');
  const fixture = path.join(root, 'fixture');
  assert.equal(vscode.workspace.workspaceFolders[0].uri.fsPath, fixture);
  const extension = vscode.extensions.getExtension('tensorviz.tensorviz-vscode');
  assert.deepEqual(await fs.readdir(path.join(extension.extensionPath, 'out')), ['extension.js'],
    'Exercise the installed bundle without loose compiled modules masking missing imports');
  const api = (await extension.activate()).__test;
  assert.equal(api.localBrowserTarget(), undefined, 'MCP must work before any graph is open');
  api.seedManagedRuntimeForTests();
  const socket = net.createServer();
  await new Promise(resolve => socket.listen(0, '127.0.0.1', resolve));
  const port = socket.address().port;
  await new Promise(resolve => socket.close(resolve));
  // Test-owned opt-in. Production setup and config approval are tested separately.
  await fs.mkdir(api.globalStorageDir(), { recursive: true });
  await fs.writeFile(path.join(api.globalStorageDir(), 'mcp-connection.json'), JSON.stringify({ enabled: true, port }));
  const connection = await until(() => api.mcpConnection(), value => !!value, 10000);
  const client = new Client({ name: 'tensorviz-installed-qa', version: '1' });
  const transport = new StreamableHTTPClientTransport(new URL(connection.url));
  await client.connect(transport);
  async function call(name, args = {}) {
    const result = await client.callTool({ name, arguments: args });
    assert.ok(!result.isError, JSON.stringify(result.structuredContent));
    return result.structuredContent;
  }
  async function settled(request) {
    assert.equal(request.continuationRequired, true);
    assert.equal(request.nextAction.kind, 'poll');
    assert.equal(request.nextAction.arguments.hostId, request.hostId);
    return until(() => call(request.nextAction.tool, request.nextAction.arguments),
      value => !['opening', 'saving', 'resolving', 'capturing'].includes(value.status));
  }
  const tensor = width => ({ kind: 'tensor', shape: [2, width], dtype: 'float32', requires_grad: false,
    generator: { kind: 'normal', mean: 0, std: 1, seed: 0 } });
  try {
    const tools = await client.listTools();
    assert.ok(tools.tools.some(tool => tool.name === 'open_graph'));
    assert.ok(tools.tools.some(tool => tool.name === 'get_execution_environment'));
    const environment = await call('get_execution_environment', { file: path.join(fixture, 'models/net.py') });
    assert.equal(environment.ok, true, JSON.stringify(environment));
    assert.equal(environment.environment.source, 'managed');
    assert.ok(environment.environment.interpreter.startsWith(api.globalStorageDir() + path.sep),
      'Report the disposable host runtime, not the QA driver or agent shell Python');
    assert.ok(environment.environment.executable); assert.ok(environment.environment.prefix);
    assert.ok(environment.environment.python); assert.ok(environment.environment.torch);
    assert.ok(environment.environment.pip);
    assert.equal((await call('list_models')).models.length, 0);
    const first = await settled(await call('open_graph', { file: path.join(fixture, 'models/net.py'), cls: 'Net' }));
    assert.equal(first.status, 'ready', JSON.stringify(first)); assert.equal(first.validated, true);
    assert.equal(first.continuationRequired, false); assert.equal(first.nextAction.kind, 'done');
    assert.equal(first.models[0].traced, true);
    for (const key of ['interpreter', 'executable', 'prefix', 'python', 'torch', 'pip']) {
      assert.equal(first.environment[key], environment.environment[key], `Capture environment changed: ${key}`);
    }
    const file = path.join(fixture, 'needs.py');
    // Required integer dimensions can now be synthesized; an opaque label still requires setup.
    await fs.writeFile(file, 'import torch.nn as nn\nclass Needs(nn.Module):\n    def __init__(self, label):\n        super().__init__()\n        self.label = label\n        self.fc = nn.Linear(8, 4)\n    def forward(self, x):\n        return self.fc(x)\n');
    let result = await settled(await call('open_graph', { file, cls: 'Needs' }));
    assert.equal(result.status, 'needs_setup', JSON.stringify(result));
    assert.equal(result.validated, false); assert.ok(result.inputContract);
    assert.equal(result.continuationRequired, true); assert.equal(result.nextAction.kind, 'complete_setup');
    assert.deepEqual(result.nextAction.requiredArguments, ['draft']);
    const continuation = result.nextAction.arguments;
    assert.equal(continuation.hostId, result.hostId);
    const blocked = await call('open_graph', { ...continuation, action: 'report_blocker', blocker: {
      reason: 'This fixture leaves the constructor label unspecified.',
      evidence: 'needs.py defines __init__(self, label) without a default or source call.',
      nextStep: 'Choose the label for this invocation.',
    } });
    assert.equal(blocked.status, 'blocked'); assert.equal(blocked.validated, false);
    assert.equal(blocked.continuationRequired, false); assert.equal(blocked.blocker.source, 'agent');
    const draft = { profile: { kind: 'model', model: 'needs:Needs', mode: 'eval', device: 'cpu', grad_enabled: false,
      init: { args: [], kwargs: { label: 'qa' } }, inputs: { args: [tensor(8)], kwargs: {} } } };
    result = await settled(await call('open_graph', { ...continuation, draft }));
    assert.equal(result.status, 'ready', JSON.stringify(result)); assert.equal(result.validated, true);
    assert.equal(result.continuationRequired, false); assert.equal(result.blocker, undefined);
    assert.equal(result.models[0].traced, true); assert.equal(api.sidebarState().requested, true);
    // Reopen setup through the existing controller; the unified tool can repair a failed recipe.
    const config = path.join(fixture, 'tensorviz.toml');
    const saved = await fs.readFile(config, 'utf8');
    assert.ok(saved.includes('needs:Needs'));
    // Alter the actual forward to expose a real failing captured operation.
    const document = await vscode.workspace.openTextDocument(vscode.Uri.file(file));
    const edit = new vscode.WorkspaceEdit();
    edit.replace(document.uri, new vscode.Range(document.positionAt(0), document.positionAt(document.getText().length)),
      document.getText().replace('return self.fc(x)', 'return self.fc(x[:, :7])'));
    assert.equal(await vscode.workspace.applyEdit(edit), true);
    assert.equal(await document.save(), true);
    result = await settled(await call('open_graph', { file, cls: 'Needs' }));
    assert.equal(result.status, 'failed', JSON.stringify(result)); assert.equal(result.validated, false);
    assert.equal(result.continuationRequired, true); assert.equal(result.nextAction.kind, 'inspect_failure');
    if (result.failureFocus?.status === 'pending') result = await until(() => call('open_graph', {
      requestId: result.requestId, hostId: result.hostId }), value => value.failureFocus?.status !== 'pending', 10000);
    assert.equal(result.failureFocus?.focused, true, JSON.stringify(result.failureFocus));
    const foreign = new Client({ name: 'other-agent', version: '1' });
    const foreignTransport = new StreamableHTTPClientTransport(new URL(connection.url));
    await foreign.connect(foreignTransport);
    const denied = await foreign.callTool({ name: 'open_graph', arguments: { requestId: result.requestId, hostId: result.hostId } });
    assert.equal(denied.isError, true); assert.equal(denied.structuredContent.error.code, 'UNKNOWN_REQUEST');
    await foreignTransport.terminateSession(); await foreign.close();
    const shapeFailure = result;
    const missingFile = path.join(fixture, 'missing.py');
    await fs.writeFile(missingFile, 'import tensorviz_missing_dependency_for_qa\nimport torch.nn as nn\n# tensorviz: input=2,8\nclass Missing(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(8, 4)\n    def forward(self, x):\n        return self.fc(x)\n');
    result = await settled(await call('open_graph', { file: missingFile, cls: 'Missing' }));
    assert.equal(result.validated, false);
    assert.ok(result.dependencies.missingModules.some(item => item.module === 'tensorviz_missing_dependency_for_qa'));
    const refused = await settled(await call('open_graph', { requestId: result.requestId, hostId: result.hostId,
      openingId: result.opening.id, sourceDigest: result.opening.target.sourceDigest,
      action: 'install_dependencies', packages: ['../not-a-package'] }));
    assert.equal(refused.error.code, 'DEPENDENCY_CONFLICT');
    assert.equal(refused.continuationRequired, true); assert.equal(refused.nextAction.kind, 'inspect_failure');
    await fs.writeFile(path.join(root, 'evidence/mcp-open-graph.json'), JSON.stringify({ tools: tools.tools.length,
      noGraphDiscovery: true, automaticCapture: first.status, submittedCapture: 'ready', shapeFailure: shapeFailure.status,
      partialCapture: shapeFailure.opening.hasCapture, failureFocused: shapeFailure.failureFocus.focused,
      missingDependencyDetected: true, unsafePackageRefused: true,
      sessionIsolation: true, sidebar: api.sidebarState().requested, continuation: true,
      blockerAndResume: true, environmentDiscovery: true, bundleOnly: true }, null, 2));
    console.log('Installed MCP: exact environment discovery, continuation through setup/capture, blocker/resume, and session isolation passed.');
  } finally {
    await transport.terminateSession().catch(() => {}); await client.close();
    await vscode.commands.executeCommand('tensorviz.disconnectMcp');
  }
};
