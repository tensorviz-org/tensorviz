const { copyBuiltPlugin } = require('../fixtures/plugin.cjs');
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { execFileSync } = require('node:child_process');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { ElicitRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const python = process.env.TENSORVIZ_TEST_PYTHON;
assert.ok(python && path.isAbsolute(python), 'Set the explicit test interpreter.');
const delay = () => new Promise(resolve => setTimeout(resolve, 100));

test('two simultaneous copied-plugin hosts isolate related Git worktrees, recipes, handoffs and cancellation', { timeout: 300000 }, async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-host-isolation-')));
  const first = path.join(root, 'first'), second = path.join(root, 'second'), plugin = path.join(root, 'plugin');
  const clients = [];
  const recipe = 'version = 1\n[profiles.tiny]\nkind = "model"\nmodel = "model:Tiny"\nmode = "eval"\ndevice = "cpu"\ngrad_enabled = false\ninit = { args = [], kwargs = {} }\ninputs = { args = [{ kind = "tensor", shape = [2, 4], dtype = "float32", requires_grad = false, generator = { kind = "full", value = 1 } }], kwargs = {} }\n';
  const otherRecipe = recipe.replace('shape = [2, 4]', 'shape = [7, 4]');
  const source = `import time\nfrom pathlib import Path\nfrom torch import nn\nclass Tiny(nn.Module):\n def __init__(self, pause=0):\n  super().__init__()\n  self.pause = pause\n  self.fc = nn.Linear(4, 3)\n def forward(self, x):\n  if self.pause:\n   Path(__file__).with_suffix('.running').write_text('running')\n   time.sleep(self.pause)\n  return self.fc(x)\n`;
  const git = args => execFileSync('git', ['-C', first, ...args], { stdio: 'pipe' });
  const start = async project => {
    const client = new Client({ name: 'worktree-isolation-test', version: '1' }, { capabilities: { elicitation: { form: {} } } });
    clients.push(client);
    let approvals = 0;
    client.setRequestHandler(ElicitRequestSchema, async request => {
      assert.ok(request.params.message.includes(project), 'An approval must name only the intended checkout');
      assert.ok(!request.params.message.includes(project === first ? second : first));
      assert.ok(request.params.message.includes(python));
      approvals++;
      return { action: 'accept', content: { confirm: true } };
    });
    await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(plugin, 'runtime/host.cjs')],
      cwd: root, env: { PATH: '', TENSORVIZ_DATA_DIR: path.join(root, 'shared-state') }, stderr: 'inherit' }));
    const call = async (name, args = {}) => {
      const reply = await client.callTool({ name, arguments: args });
      assert.ok(!reply.isError, JSON.stringify(reply)); return reply.structuredContent;
    };
    const reject = async (name, args, code) => {
      const reply = await client.callTool({ name, arguments: args });
      assert.equal(reply.isError, true); assert.equal(reply.structuredContent.error.code, code);
    };
    return { project, call, reject, approvals: () => approvals };
  };
  const poll = async (host, tool, initial, key) => {
    let result = initial;
    for (let n = 0; n < 2400 && !['ready', 'passed', 'failed', 'blocked', 'needs_setup', 'cancelled', 'refused', 'source_changed'].includes(result.status); n++) {
      await delay(); result = await host.call(tool, { [key]: initial[key] });
    }
    return result;
  };
  try {
    await fs.mkdir(first);
    await fs.writeFile(path.join(first, 'model.py'), source);
    await fs.writeFile(path.join(first, 'tensorviz.toml'), recipe);
    git(['init', '--quiet']);
    git(['add', 'model.py', 'tensorviz.toml']);
    git(['-c', 'user.name=TensorViz Test', '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '--quiet', '-m', 'synthetic fixture']);
    git(['worktree', 'add', '--detach', second, 'HEAD']);
    await fs.writeFile(path.join(second, 'model.py'), source.replace('nn.Linear(4, 3)', 'nn.Linear(4, 5)'));
    await fs.writeFile(path.join(second, 'tensorviz.toml'), otherRecipe);
    await copyBuiltPlugin(plugin);
    const a = await start(first), b = await start(second);
    const opened = await Promise.all([a, b].map(async host => poll(host, 'open_graph', await host.call('open_graph', {
      workspaceRoot: host.project, file: 'model.py', cls: 'Tiny', interpreter: python,
    }), 'requestId')));
    assert.ok(opened.every(result => result.status === 'ready'), JSON.stringify(opened));
    const refs = opened.map(result => ({ modelId: result.models[0].modelId, revision: result.models[0].revision }));
    assert.notEqual(refs[0].modelId, refs[1].modelId);
    for (const [index, host] of [a, b].entries()) {
      const graph = await host.call('get_graph', { ...refs[index], mode: 'graph', limit: 100 });
      assert.equal(graph.target.workspaceFolder, host.project);
      assert.ok(graph.nodes.some(node => node.shapes.output?.join(',') === (index ? '7,5' : '2,3')));
      assert.ok(host.approvals() <= 1, 'A host must not duplicate the plugin consent prompt');
      await host.reject('open_graph', { requestId: opened[1 - index].requestId }, 'UNKNOWN_REQUEST');
      await host.reject('get_graph', refs[1 - index], 'STALE_TARGET');
    }
    assert.equal(a.approvals() + b.approvals(), 1, 'Concurrent hosts share consent while retaining separate project identities');
    const handoffA = await a.call('open_graph_view', refs[0]);
    const handoffB = await b.call('open_graph_view', refs[1]);
    assert.notEqual(new URL(handoffA.url).origin, new URL(handoffB.url).origin);
    await b.reject('open_graph_view', { ...refs[1], handoffId: handoffA.handoffId }, 'UNKNOWN_HANDOFF');
    const running = await a.call('validate_model', { ...refs[0], execution: {
      inputs: { args: [{ shape: [2, 4], dtype: 'float32', fill: 'ones' }] }, constructorArgs: { pause: 120 },
    } });
    for (let n = 0; n < 300 && !await fs.stat(path.join(first, 'model.running')).catch(() => null); n++) await delay();
    assert.ok(await fs.stat(path.join(first, 'model.running')).catch(() => null), 'The first host must have an active worker');
    await b.reject('cancel_run', { runId: running.runId }, 'UNKNOWN_RUN');
    assert.equal((await a.call('get_run', { runId: running.runId })).status, 'running', 'Another host must not cancel this worker');
    assert.ok((await b.call('get_graph', { ...refs[1], mode: 'graph', limit: 100 })).nodes.some(node => node.shapes.output?.join(',') === '7,5'));
    await a.call('cancel_run', { runId: running.runId });
    assert.equal((await poll(a, 'get_run', running, 'runId')).status, 'cancelled');
    for (const directory of [first, second]) assert.equal(await fs.readFile(path.join(directory, 'tensorviz.toml'), 'utf8'), directory === first ? recipe : otherRecipe);
    assert.equal(await fs.readFile(path.join(first, 'model.py'), 'utf8'), source);
    assert.equal(await fs.readFile(path.join(second, 'model.py'), 'utf8'), source.replace('nn.Linear(4, 3)', 'nn.Linear(4, 5)'));
  } finally {
    for (const client of clients) await client.close();
    await fs.rm(root, { recursive: true, force: true });
  }
});
