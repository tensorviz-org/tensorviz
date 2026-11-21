const { test } = require('node:test');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { mkdtemp, mkdir, cp, writeFile, rm, realpath } = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { ElicitRequestSchema } = require('@modelcontextprotocol/sdk/types.js');

const python = process.env.TENSORVIZ_TEST_PYTHON;
assert.ok(python && path.isAbsolute(python), 'Set TENSORVIZ_TEST_PYTHON to a compatible Python 3.11+ environment.');
const recipe = `version = 1\n[profiles.tiny]\nkind = "model"\nmodel = "model:Tiny"\nmode = "eval"\ngrad_enabled = false\ndevice = "cpu"\ninit = { args = [], kwargs = {} }\ninputs = { args = [{kind = "tensor", shape = [2, 4], dtype = "float32", requires_grad = false, generator = {kind = "full", value = 1.0}}], kwargs = {} }\n`;
const model = 'from torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, 3)\n def forward(self, x):\n  return self.fc(x)\n';

async function poll(client, initial) {
  let result = initial;
  for (let count = 0; count < 2400 && !['ready', 'failed', 'blocked', 'cancelled', 'needs_setup'].includes(result.status); count++) {
    await new Promise(resolve => setTimeout(resolve, 100));
    result = (await client.callTool({ name: 'open_graph', arguments: { requestId: result.requestId } })).structuredContent;
  }
  return result;
}

test('a project pin that differs from the installed Torch can run with the installed version', { timeout: 300000 }, async () => {
  const installed = execFileSync(python, ['-c', 'import torch; print(torch.__version__.split("+")[0])'], { encoding: 'utf8' }).trim();
  const pin = installed === '2.0.0' ? 'torch==2.0.1' : 'torch==2.0.0';
  const root = await realpath(await mkdtemp(path.join(os.tmpdir(), 'tensorviz-installed-versions-')));
  const project = path.join(root, 'project'), plugin = path.join(root, 'installed plugin');
  await mkdir(project);
  await cp(process.env.TENSORVIZ_TEST_RUNTIME ?? path.resolve(__dirname, '../../../plugins/tensorviz'), plugin, { recursive: true });
  await writeFile(path.join(project, 'model.py'), model);
  await writeFile(path.join(project, 'tensorviz.toml'), recipe);
  await writeFile(path.join(project, 'requirements.txt'), `${pin}\n`);
  const transport = new StdioClientTransport({ command: process.env.TENSORVIZ_TEST_RUNTIME ? path.join(plugin, 'tensorviz-host') : process.execPath,
    args: process.env.TENSORVIZ_TEST_RUNTIME ? [] : [path.join(plugin, 'runtime/host.cjs')], cwd: root,
    env: { PATH: '', TENSORVIZ_DATA_DIR: path.join(root, 'state') }, stderr: 'pipe' });
  let stderr = '';
  transport.stderr?.on('data', value => { stderr += value; });
  const client = new Client({ name: 'approved-integration-test', version: '1' }, { capabilities: { elicitation: { form: {} } } });
  client.setRequestHandler(ElicitRequestSchema, async () => ({ action: 'accept', content: { confirm: true } }));
  const call = async (name, args) => {
    const result = await client.callTool({ name, arguments: args });
    assert.ok(!result.isError, JSON.stringify(result.structuredContent));
    return result.structuredContent;
  };
  try {
    await client.connect(transport);
    const target = { workspaceRoot: project, file: 'model.py', cls: 'Tiny', profile: 'tiny', interpreter: python };
    const pinned = await poll(client, await call('open_graph', target));
    assert.equal(pinned.status, 'needs_setup', JSON.stringify(pinned));
    assert.equal(pinned.opening.environment.reason, 'incompatible');
    assert.deepEqual(pinned.opening.environment.declaredPins, [pin]);
    assert.equal(pinned.nextAction.kind, 'use_installed_versions');
    assert.equal(pinned.nextAction.arguments.action, 'use_installed_versions');

    const accepted = await poll(client, await call('open_graph', pinned.nextAction.arguments));
    assert.equal(accepted.status, 'ready', JSON.stringify(accepted));
    assert.equal(accepted.validated, true);

    const reopened = await poll(client, await call('open_graph', target));
    assert.equal(reopened.status, 'ready', 'accepted pins last for this workspace and interpreter during the session');
    assert.equal(stderr, '');
  } finally {
    await client.close();
    await rm(root, { recursive: true, force: true });
  }
});
