// A disposable MCP client for manual/Computer browser acceptance. No browser automation.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const readline = require('node:readline');
const { execFileSync } = require('node:child_process');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { ElicitRequestSchema } = require('@modelcontextprotocol/sdk/types.js');

(async () => {
  const python = process.env.TENSORVIZ_TEST_PYTHON;
  assert.ok(python && path.isAbsolute(python), 'Set an explicit test interpreter.');
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-browser-acceptance-')));
  const project = path.join(root, 'project'), plugin = path.join(root, 'plugin');
  const setup = process.argv.includes('--setup');
  const review = process.argv.includes('--review');
  const capturedReview = process.argv.includes('--captured-review');
  const helper = process.argv.includes('--helper');
  await fs.mkdir(project);
  await fs.cp(path.resolve(__dirname, '../../../plugins/tensorviz'), plugin, { recursive: true });
  const source = activation => `from torch import nn\n${helper ? 'from helper import WIDTH\n' : ''}class Tiny(nn.Module):\n def __init__(self${setup ? ', variant: str' : ''}):\n  super().__init__()\n  self.fc = nn.Linear(4,${helper ? 'WIDTH' : '3'})\n  self.activation = nn.${activation}()\n def forward(self,x):\n  return self.activation(self.fc(x))\n`;
  if (helper) await fs.writeFile(path.join(project, 'helper.py'), 'WIDTH = 3\n');
  await fs.writeFile(path.join(project, 'model.py'), source('ReLU'));
  if (review || capturedReview) {
    const git = args => execFileSync('git', ['-C', project, ...args], { stdio: 'pipe' });
    git(['init', '-q']); git(['add', 'model.py']);
    git(['-c', 'user.name=TensorViz Test', '-c', 'user.email=test@localhost', '-c', 'commit.gpgsign=false', 'commit', '-qm', 'base']);
  }
  if (!setup) await fs.writeFile(path.join(project, 'tensorviz.toml'), `version = 1\n[profiles.tiny]\nkind = "model"\nmodel = "model:Tiny"\nmode = "eval"\ngrad_enabled = false\ndevice = "cpu"\ninit = { args = [], kwargs = {} }\ninputs = { args = [{kind = "tensor", shape = [2, 4], dtype = "float32", requires_grad = false, generator = {kind = "full", value = 1.0}}], kwargs = {} }\n`);
  const client = new Client({ name: 'synthetic-browser-acceptance', version: '1' }, { capabilities: { elicitation: { form: {} } } });
  client.setRequestHandler(ElicitRequestSchema, async request => {
    assert.ok(request.params.message.includes(project), 'Only the disposable test workspace is authorized.');
    assert.ok(request.params.message.includes(python));
    return { action: 'accept', content: { confirm: true } };
  });
  const transport = new StdioClientTransport({ command: process.execPath, args: [path.join(plugin, 'runtime/host.cjs')], cwd: root,
    env: { PATH: review || capturedReview ? process.env.PATH : '', TENSORVIZ_DATA_DIR: path.join(root, 'state') }, stderr: 'inherit' });
  await client.connect(transport);
  const call = async (name, args) => {
    const reply = await client.callTool({ name, arguments: args });
    assert.ok(!reply.isError, JSON.stringify(reply.structuredContent));
    return reply.structuredContent;
  };
  let reference, handoff, openingId;
  async function capture(allowSetup = false) {
    let reply = await call('open_graph', { workspaceRoot: project, file: 'model.py', cls: 'Tiny', interpreter: python });
    openingId = reply.requestId;
    for (let count = 0; !['ready', 'blocked', 'failed', 'cancelled', 'needs_setup'].includes(reply.status) && count < 300; count++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      reply = await call('open_graph', { requestId: reply.requestId });
    }
    assert.equal(reply.status, allowSetup ? 'needs_setup' : 'ready', JSON.stringify(reply));
    reference = { modelId: reply.models[0].modelId, revision: reply.models[0].revision };
    return reply;
  }
  if (review) {
    let environment = await call('get_execution_environment', { workspaceRoot: project, interpreter: python });
    while (environment.status === 'checking') {
      await new Promise(resolve => setTimeout(resolve, 100));
      environment = await call('get_execution_environment', { requestId: environment.requestId });
    }
    assert.equal(environment.status, 'ready');
  } else await capture(setup);
  if (review) {
    await fs.writeFile(path.join(project, 'model.py'), source('Sigmoid'));
    handoff = await call('open_graph_diff', { workspaceRoot: project, file: 'model.py', cls: 'Tiny' });
  } else handoff = await call('open_graph_view', reference);
  console.log(JSON.stringify({ event: 'ready', url: handoff.url, root, reference, handoffId: handoff.handoffId, reviewId: handoff.reviewId }));
  const input = readline.createInterface({ input: process.stdin });
  try {
    for await (const line of input) {
      try {
        const command = JSON.parse(line);
        let result;
        if (command.action === 'stop') break;
        if (command.action === 'status') {
          const progress = await call('open_graph', { requestId: openingId });
          reference = { modelId: progress.models[0].modelId, revision: progress.models[0].revision };
          result = { status: progress.status, opening: progress.opening, reference };
        }
        else if (command.action === 'display') result = handoff.reviewId ? await call('open_graph_diff', { reviewId: handoff.reviewId })
          : await call('open_graph_view', { ...reference, handoffId: handoff.handoffId });
        else if (command.action === 'change') { await fs.writeFile(path.join(project, 'model.py'), source('Tanh')); result = { changed: true }; }
        else if (command.action === 'change-helper' && helper) { await fs.writeFile(path.join(project, 'helper.py'), 'WIDTH = 2\n'); result = { changed: true }; }
        else if (command.action === 'review') { handoff = await call('open_graph_diff', { workspaceRoot: project, file: 'model.py', cls: 'Tiny' }); result = handoff; }
        else if (command.action === 'view') { handoff = await call('open_graph_view', reference); result = handoff; }
        else if (command.action === 'graph') {
          const graph = await call('get_graph', { ...reference, mode: 'graph', limit: 200 });
          result = { ...reference, nodes: graph.nodes.map(node => ({ nodeId: node.nodeId, display: node.display, name: node.name, shapes: node.shapes })) };
        }
        else if (command.action === 'focus') result = await call('focus_view', { ...reference, viewId: command.viewId, nodeIds: [command.nodeId] });
        else if (command.action === 'selection') result = await call('get_selection', { ...reference, viewId: command.viewId });
        else if (command.action === 'refresh') {
          await fs.writeFile(path.join(project, 'model.py'), source('Sigmoid'));
          const captured = await capture(); result = { status: captured.status, reference };
        }
        else throw new Error('Unknown acceptance command.');
        console.log(JSON.stringify({ event: command.action, result }));
      } catch (error) { console.log(JSON.stringify({ event: 'error', message: error.message })); }
    }
  } finally { input.close(); process.stdin.pause(); await client.close(); await fs.rm(root, { recursive: true, force: true }); }
})().catch(error => { console.error(error.message); process.exitCode = 1; });
