const { readyService } = require('../fixtures/service.cjs');
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { ElicitRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const { PluginPermissions } = require('../../../plugins/tensorviz/runtime/permissions.cjs');
const python = process.env.TENSORVIZ_TEST_PYTHON;
assert.ok(python && path.isAbsolute(python), 'Set the explicit test interpreter.');
const delay = () => new Promise(resolve => setTimeout(resolve, 100));

test('copied plugin retains explicit runs, compares captures, rejects stale approval and cancels its worker', { timeout: 180000 }, async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-runs-')));
  const project = path.join(root, 'project'), plugin = path.join(root, 'plugin');
  const marker = path.join(project, 'started');
  let client;
  try {
    await fs.mkdir(project);
    await fs.cp(path.resolve(__dirname, '../../../plugins/tensorviz'), plugin, { recursive: true });
    await readyService(plugin, path.join(root, 'state'));
    const source = `import time\nfrom pathlib import Path\nfrom torch import nn\nclass Tiny(nn.Module):\n def __init__(self, pause=0):\n  super().__init__()\n  self.pause = pause\n  self.fc = nn.Linear(4, 3)\n def forward(self, x):\n  if self.pause:\n   Path(${JSON.stringify(marker)}).write_text('running')\n   time.sleep(self.pause)\n  return self.fc(x).relu()\n`;
    await fs.writeFile(path.join(project, 'model.py'), source);
    const recipe = 'version = 1\n[profiles.tiny]\nkind = "model"\nmodel = "model:Tiny"\nmode = "eval"\ndevice = "cpu"\ngrad_enabled = false\ninit = { args = [], kwargs = {} }\ninputs = { args = [{ kind = "tensor", shape = [2, 4], dtype = "float32", requires_grad = false, generator = { kind = "full", value = 1 } }], kwargs = {} }\n';
    await fs.writeFile(path.join(project, 'tensorviz.toml'), recipe);
    const permissions = new PluginPermissions(path.join(root, 'state'));
    let editDuringApproval = false;
    client = new Client({ name: 'retained-runs-test', version: '1' }, { capabilities: { elicitation: { form: {} } } });
    client.setRequestHandler(ElicitRequestSchema, async request => {
      assert.match(request.params.message, /without asking again/);
      if (editDuringApproval) {
        editDuringApproval = false;
        await fs.writeFile(path.join(project, 'model.py'), source.replace('.relu()', '.sigmoid()'));
      }
      return { action: 'accept', content: { confirm: true } };
    });
    await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(plugin, 'runtime/host.cjs')], cwd: root,
      env: { PATH: '', TENSORVIZ_DATA_DIR: path.join(root, 'state') }, stderr: 'inherit' }));
    const call = async (name, args) => {
      const result = await client.callTool({ name, arguments: args });
      assert.ok(!result.isError, JSON.stringify(result)); return result.structuredContent;
    };
    const poll = async (name, initial, key) => {
      let value = initial;
      for (let count = 0; count < 500 && !['ready', 'passed', 'failed', 'blocked', 'cancelled', 'refused', 'source_changed'].includes(value.status); count++) {
        await delay(); value = await call(name, { [key]: initial[key] });
      }
      return value;
    };
    const open = () => call('open_graph', { workspaceRoot: project, file: 'model.py', cls: 'Tiny', interpreter: python });
    assert.equal((await poll('open_graph', await open(), 'requestId')).status, 'ready');
    const reference = async () => {
      const model = (await call('list_models', {})).models[0];
      return { modelId: model.modelId, revision: model.revision };
    };
    const execution = batch => ({ inputs: { kwargs: { x: { shape: [batch, 4], dtype: 'float32', fill: 'ones' } } } });
    const validate = async setup => call('validate_model', { ...await reference(), execution: setup });
    const first = await poll('get_run', await validate(execution(2)), 'runId');
    assert.equal(first.status, 'passed', JSON.stringify(first));
    assert.equal(first.attempt.id, first.runId);
    assert.ok(first.capture.id);
    const before = await call('inspect_run', { runId: first.runId, section: 'graph', limit: 200 });
    assert.equal(before.format, 'semantic-v1');
    assert.ok(before.nodes.items.some(node => node.shapes.output?.join(',') === '2,3'));
    const second = await poll('get_run', await validate(execution(3)), 'runId');
    assert.equal(second.status, 'passed', JSON.stringify(second));
    const comparison = await call('compare_runs', { baselineRunId: first.runId, candidateRunId: second.runId });
    assert.equal(comparison.semanticComparisonAvailable, true);
    assert.equal(comparison.format, 'semantic-v1');
    assert.deepEqual(await call('inspect_run', { runId: first.runId, section: 'graph', limit: 200 }), before);
    const failed = await poll('get_run', await validate({ inputs: { args: [{ shape: [2, 5], dtype: 'float32', fill: 'ones' }] } }), 'runId');
    assert.equal(failed.status, 'failed');
    assert.ok(failed.capture.id, 'partial failure evidence is retained');
    const slow = await validate({ ...execution(2), constructorArgs: { pause: 120 } });
    for (let count = 0; count < 200; count++) {
      if (await fs.stat(marker).catch(() => undefined)) break;
      await delay();
    }
    assert.ok(await fs.stat(marker).catch(() => undefined), JSON.stringify(await call('get_run', { runId: slow.runId })));
    const competing = await poll('open_graph', await open(), 'requestId');
    assert.equal(competing.error.code, 'BUSY');
    await call('cancel_run', { runId: slow.runId });
    assert.equal((await poll('get_run', slow, 'runId')).status, 'cancelled');
    await permissions.save('denied');
    const declined = await poll('get_run', await validate(execution(4)), 'runId');
    assert.equal(declined.status, 'refused');
    assert.equal(declined.capture, null);
    assert.match(declined.detail, /declined/);
    // Simulate resetting consent; a saved edit during the new decision must
    // still invalidate the frozen invocation, even when plugin permission is granted.
    await fs.rm(permissions.file);
    editDuringApproval = true;
    const stale = await poll('get_run', await validate(execution(5)), 'runId');
    assert.equal(stale.status, 'refused');
    assert.equal(stale.capture, null, 'a saved edit during approval must prevent execution');
    assert.deepEqual(await call('inspect_run', { runId: first.runId, section: 'graph', limit: 200 }), before);
    assert.equal(await fs.readFile(path.join(project, 'tensorviz.toml'), 'utf8'), recipe);
  } finally {
    await client?.close();
    await fs.rm(root, { recursive: true, force: true });
  }
});
