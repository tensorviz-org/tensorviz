const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { PluginPermissions } = require('../../../plugins/tensorviz/runtime/permissions.cjs');
const { ElicitRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const { execFileSync } = require('node:child_process');

test('fresh managed Python captures without an existing interpreter and isolates two checkouts', { timeout: 600000 }, async t => {
  assert.equal(`${process.platform}-${process.arch}`, 'darwin-arm64', 'This acceptance target currently claims only macOS Apple Silicon.');
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-managed-acceptance-')));
  const plugin = path.join(root, 'plugin'), state = path.join(root, 'state');
  const first = path.join(root, 'first'), second = path.join(root, 'second'), declined = path.join(root, 'declined');
  let client;
  try {
    await fs.cp(path.resolve(__dirname, '../../../plugins/tensorviz'), plugin, { recursive: true });
    for (const project of [first, second, declined]) {
      await fs.mkdir(project);
      await fs.writeFile(path.join(project, 'model.py'), 'from torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, 3)\n def forward(self, x): return self.fc(x)\n');
    }
    const approvals = [];
    client = new Client({ name: 'managed-runtime-acceptance', version: '1' }, { capabilities: { elicitation: { form: {} } } });
    client.setRequestHandler(ElicitRequestSchema, async request => {
      const message = request.params.message;
      approvals.push(message);
      assert.ok(message.includes(root));
      assert.ok(!message.includes('globalStorage'));
      assert.match(message, /build\/install required dependencies/);
      return { action: 'accept', content: { confirm: !message.includes(`Workspace: ${declined}\n`) } };
    });
    await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(plugin, 'runtime/host.cjs')], cwd: root,
      env: { PATH: '', TENSORVIZ_DATA_DIR: state }, stderr: 'inherit' }));
    const call = async (name, args) => {
      const result = await client.callTool({ name, arguments: args });
      assert.ok(!result.isError, JSON.stringify(result)); return result.structuredContent;
    };
    const poll = async reply => {
      const deadline = Date.now() + 240000;
      let previous;
      while (!['ready', 'failed', 'blocked', 'needs_setup', 'cancelled'].includes(reply.status) && Date.now() < deadline) {
        if (reply.runtimeProgress && reply.runtimeProgress !== previous) {
          previous = reply.runtimeProgress; t.diagnostic(previous);
        }
        await new Promise(resolve => setTimeout(resolve, 200));
        reply = await call('open_graph', { requestId: reply.requestId });
      }
      return reply;
    };
    const open = project => call('open_graph', { workspaceRoot: project, file: 'model.py', cls: 'Tiny', managedRuntime: true });
    const refused = await poll(await open(declined));
    assert.equal(refused.error.code, 'APPROVAL_DECLINED', JSON.stringify(refused));
    assert.equal(await new PluginPermissions(state).status(), 'denied');
    await assert.rejects(fs.stat(path.join(state, 'managed')), { code: 'ENOENT' });
    // The user then consents at installation; provisioning must not ask again.
    await new PluginPermissions(state).save('granted');
    const result = await poll(await open(first));
    assert.equal(result.status, 'ready', JSON.stringify(result));
    const interpreter = result.environment.executable;
    assert.ok(interpreter.startsWith(state));
    assert.equal(result.environment.torch, '2.14.1');
    const inventory = JSON.parse(execFileSync(interpreter, ['-I', '-c', 'import json,importlib.metadata as m;print(json.dumps(sorted(d.metadata["Name"].lower() for d in m.distributions())))'], { encoding: 'utf8' }));
    assert.ok(!inventory.includes('fastapi') && !inventory.includes('uvicorn') && !inventory.includes('libcst'), 'managed capture contains no tool-server libraries');
    const reused = await poll(await open(first));
    assert.equal(reused.status, 'ready', JSON.stringify(reused));
    assert.equal(reused.environment.executable, interpreter);
    assert.equal(approvals.length, 1);
    const other = await poll(await open(second));
    assert.equal(other.status, 'ready', JSON.stringify(other));
    assert.notEqual(other.environment.executable, interpreter);
    assert.equal((await fs.readdir(path.join(state, 'downloads'))).length, 1, 'verified Python download is shared');
    assert.equal((await fs.readdir(path.join(state, 'python'))).length, 1, 'immutable base Python is shared');
    assert.equal((await fs.readdir(path.join(state, 'service'))).length, 1, 'both checkouts share one separate tool service');
    assert.equal(approvals.length, 1, 'Installed permission covers both managed environments without new prompts');
  } finally { await client?.close(); await fs.rm(root, { recursive: true, force: true }); }
});
