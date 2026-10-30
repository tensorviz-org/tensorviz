const { readyService } = require('../fixtures/service.cjs');
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { ElicitRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const python = process.env.TENSORVIZ_TEST_PYTHON;
assert.ok(python && path.isAbsolute(python), 'Set the explicit test interpreter.');

test('copied plugin reviews saved Git source without capture and does not fabricate browser display', { timeout: 60000 }, async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-review-')));
  const project = path.join(root, 'checkout'), plugin = path.join(root, 'plugin');
  let client;
  try {
    await fs.mkdir(project);
    await fs.cp(path.resolve(__dirname, '../../../plugins/tensorviz'), plugin, { recursive: true });
    await readyService(plugin, path.join(root, 'state'));
    const marker = path.join(project, 'imported');
    const source = activation => `from pathlib import Path\nPath(${JSON.stringify(marker)}).touch()\nraise RuntimeError('Neither revision may be imported')\nfrom torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, 3)\n  self.activation = nn.${activation}()\n def forward(self,x):\n  return self.activation(self.fc(x))\n`;
    await fs.writeFile(path.join(project, 'model.py'), source('ReLU'));
    const git = args => execFileSync('git', ['-C', project, ...args], { stdio: 'pipe' });
    git(['init', '-q']); git(['add', 'model.py']);
    git(['-c', 'user.name=TensorViz Test', '-c', 'user.email=test@localhost', '-c', 'commit.gpgsign=false', 'commit', '-qm', 'base']);
    await fs.writeFile(path.join(project, 'model.py'), source('Sigmoid'));
    let prompts = 0;
    client = new Client({ name: 'source-review-test', version: '1' }, { capabilities: { elicitation: { form: {} } } });
    client.setRequestHandler(ElicitRequestSchema, async request => {
      assert.match(request.params.message, /inspect environment/);
      prompts++; return { action: 'accept', content: { confirm: true } };
    });
    await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(plugin, 'runtime/host.cjs')], cwd: root,
      env: { PATH: process.env.PATH, TENSORVIZ_DATA_DIR: path.join(root, 'state') }, stderr: 'inherit' }));
    const call = async (name, args) => {
      const result = await client.callTool({ name, arguments: args });
      assert.ok(!result.isError, JSON.stringify(result)); return result.structuredContent;
    };
    let environment = await call('get_execution_environment', { workspaceRoot: project, interpreter: python });
    for (let count = 0; count < 200 && environment.status === 'checking'; count++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      environment = await call('get_execution_environment', { requestId: environment.requestId });
    }
    assert.equal(environment.status, 'ready');
    const result = await call('open_graph_diff', { workspaceRoot: project, file: 'model.py', cls: 'Tiny' });
    assert.equal(result.displayed, false);
    assert.equal(result.status, 'awaiting_browser');
    assert.equal(result.comparison.executed, false);
    assert.equal(result.models.length, 1);
    assert.equal(result.models[0].label, 'Tiny');
    assert.deepEqual(result.changedFiles, ['model.py']);
    assert.equal(result.nextAction.kind, 'open_browser');
    assert.ok(result.url.startsWith('http://127.0.0.1:'));
    const waiting = await call('open_graph_diff', { reviewId: result.reviewId });
    assert.equal(waiting.displayed, false);
    assert.equal(waiting.url, undefined, 'polling never creates another browser handoff');
    assert.equal(waiting.nextAction.kind, 'poll');
    assert.deepEqual((await call('list_models', {})).models, [], 'source review does not claim a live capture');
    assert.equal(await fs.stat(marker).catch(() => null), null);
    assert.equal(await fs.stat(path.join(project, 'tensorviz.toml')).catch(() => null), null);
    assert.equal(prompts, 1, 'neither revision executes or requests capture permission');
    await fs.writeFile(path.join(project, 'model.py'), source('Tanh'));
    const stale = await call('open_graph_diff', { reviewId: result.reviewId });
    assert.equal(stale.status, 'stale');
    assert.equal(stale.displayed, false);
    const bad = await client.callTool({ name: 'open_graph_diff', arguments: { workspaceRoot: project, base: 'no-such-revision' } });
    assert.equal(bad.structuredContent.error.code, 'GIT_COMPARISON_UNAVAILABLE');
  } finally { await client?.close(); await fs.rm(root, { recursive: true, force: true }); }
});
