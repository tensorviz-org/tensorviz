const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { promisify } = require('node:util');
const execFile = promisify(require('node:child_process').execFile);
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { ElicitRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const base = process.env.TENSORVIZ_TEST_PYTHON;
assert.ok(base && path.isAbsolute(base), 'Set the explicit read-only test interpreter. Installs use a disposable environment.');
const python = async (executable, ...args) => (await execFile(executable, ['-I', '-B', ...args])).stdout.trim();

test('approved repair before the service exists resumes the same opening and preserves installed Torch', { timeout: 180000 }, async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-install-integration-')));
  const project = path.join(root, 'project'), plugin = path.join(root, 'plugin'), environment = path.join(root, 'environment');
  const interpreter = path.join(environment, process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  let client;
  try {
    await fs.mkdir(project);
    await fs.cp(path.resolve(__dirname, '../../../plugins/tensorviz'), plugin, { recursive: true });
    await python(base, '-m', 'venv', '--without-pip', environment);
    const baseSite = await python(base, '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))');
    const site = await python(interpreter, '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))');
    // Expose the known test libraries read-only; all pip writes target the disposable prefix.
    await fs.writeFile(path.join(site, 'test-libraries.pth'), `${baseSite}\n`);
    const facts = 'import json,sys,importlib.metadata as m,importlib.util; print(json.dumps({"prefix":sys.prefix,"torch":m.version("torch"),"humanize":importlib.util.find_spec("humanize") is not None}))';
    const before = JSON.parse(await python(interpreter, '-c', facts));
    assert.equal(before.prefix, environment); assert.equal(before.humanize, false);
    // Simulate an absent service requirement as well as a required model import.
    await fs.appendFile(path.join(plugin, 'backend/requirements-project.txt'), '\nhumanize==4.12.3\n');
    await fs.writeFile(path.join(project, 'requirements.txt'), 'humanize==4.12.3\n');
    await fs.writeFile(path.join(project, 'model.py'), 'import humanize\nfrom torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, 3)\n def forward(self, x): return self.fc(x)\n');
    const prompts = [];
    client = new Client({ name: 'dependency-acceptance', version: '1' }, { capabilities: { elicitation: { form: {} } } });
    client.setRequestHandler(ElicitRequestSchema, async request => {
      const message = request.params.message;
      assert.ok(message.includes(interpreter)); prompts.push(message);
      assert.match(message, /build\/install required dependencies/);
      assert.equal(JSON.parse(await python(interpreter, '-c', facts)).humanize, false);
      return { action: 'accept', content: { confirm: true } };
    });
    await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(plugin, 'runtime/host.cjs')], cwd: root,
      env: { PATH: '', TENSORVIZ_DATA_DIR: path.join(root, 'state') }, stderr: 'inherit' }));
    const call = async args => {
      const result = await client.callTool({ name: 'open_graph', arguments: args });
      assert.ok(!result.isError, JSON.stringify(result)); return result.structuredContent;
    };
    const poll = async reply => {
      const deadline = Date.now() + 120000;
      while (!['ready', 'failed', 'blocked', 'needs_setup', 'cancelled'].includes(reply.status) && Date.now() < deadline) {
        await new Promise(resolve => setTimeout(resolve, 100));
        reply = await call({ requestId: reply.requestId });
      }
      return reply;
    };
    const missing = await poll(await call({ workspaceRoot: project, file: 'model.py', cls: 'Tiny', interpreter }));
    assert.equal(missing.error.code, 'ENVIRONMENT_REQUIREMENTS', JSON.stringify(missing));
    assert.deepEqual(missing.environment.missing, ['humanize==4.12.3']);
    const repaired = await poll(await call({ ...missing.nextAction.arguments, requestId: missing.requestId }));
    assert.equal(repaired.requestId, missing.requestId);
    assert.equal(repaired.status, 'ready', JSON.stringify(repaired));
    assert.equal(prompts.length, 1, 'Dependency repair and capture reuse the original plugin consent');
    const after = JSON.parse(await python(interpreter, '-c', facts));
    assert.equal(after.humanize, true); assert.equal(after.torch, before.torch);
    assert.equal(JSON.parse(await python(base, '-c', facts)).humanize, false, 'the original environment is unchanged');
    assert.match(await python(interpreter, '-c', 'import humanize; print(humanize.__file__)'), new RegExp(environment));
  } finally { await client?.close(); await fs.rm(root, { recursive: true, force: true }); }
});
