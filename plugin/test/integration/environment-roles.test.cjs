const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { constants } = require('node:fs');
const { readyService } = require('../fixtures/service.cjs');
const { execFileSync } = require('node:child_process');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { ElicitRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const { chromium } = require('../../../canvas/node_modules/@playwright/test');
const { gunzipSync } = require('node:zlib');

const python = process.env.TENSORVIZ_TEST_CAPTURE_PYTHON, other = process.env.TENSORVIZ_TEST_OTHER_PYTHON;
const manager = process.env.TENSORVIZ_TEST_SERVICE_PYTHON;
const missingBase = process.env.TENSORVIZ_TEST_MISSING_PYTHON;
assert.ok([python, other, manager].every(value => value && path.isAbsolute(value)), 'Set two capture interpreters and a prepared tool-service interpreter.');
const inventory = executable => JSON.parse(execFileSync(executable, ['-I', '-c', 'import json,importlib.metadata as m;print(json.dumps({d.metadata["Name"].lower():d.version for d in m.distributions()}))'], { encoding: 'utf8' }));

test('installed plugin captures in pipless project Python and keeps its service across environment switches', { timeout: 120000 }, async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-plugin-roles-')));
  const project = path.join(root, 'project'), plugin = path.join(root, 'plugin'), storage = path.join(root, 'state');
  const before = [inventory(python), inventory(other)];
  assert.ok(before.every(items => !items.pip && !items.fastapi && !items.uvicorn && !items.libcst), 'capture fixtures must lack project installers and tool-server libraries');
  let client;
  try {
    await fs.mkdir(project);
    await fs.cp(path.resolve(__dirname, '../../../plugins/tensorviz'), plugin, { recursive: true });
    await readyService(plugin, storage);
    await fs.writeFile(path.join(project, 'model.py'), 'from torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, 3)\n def forward(self, x): return self.fc(x)\n');
    await fs.writeFile(path.join(project, 'tensorviz.toml'), 'version=1\n[profiles.tiny]\nkind="model"\nmodel="model:Tiny"\nmode="eval"\ngrad_enabled=false\ndevice="cpu"\ninit={args=[],kwargs={}}\ninputs={args=[{kind="tensor",shape=[2,4],dtype="float32",requires_grad=false,generator={kind="full",value=1.0}}],kwargs={}}\n');
    let prompts = 0;
    client = new Client({ name: 'plugin-role-acceptance', version: '1' }, { capabilities: { elicitation: { form: {} } } });
    client.setRequestHandler(ElicitRequestSchema, async () => { prompts++; return { action: 'accept', content: { confirm: true } }; });
    await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(plugin, 'runtime/host.cjs')], cwd: root,
      env: { PATH: '', TENSORVIZ_DATA_DIR: storage }, stderr: 'inherit' }));
    const call = async (name, args) => {
      const reply = await client.callTool({ name, arguments: args });
      assert.ok(!reply.isError, JSON.stringify(reply)); return reply.structuredContent;
    };
    const open = async interpreter => {
      let reply = await call('open_graph', { workspaceRoot: project, file: 'model.py', cls: 'Tiny', interpreter });
      for (let count = 0; count < 400 && ['opening', 'resolving', 'capturing'].includes(reply.status); count++) {
        await new Promise(resolve => setTimeout(resolve, 100)); reply = await call('open_graph', { requestId: reply.requestId });
      }
      assert.equal(reply.status, 'ready', JSON.stringify(reply));
      assert.equal(reply.validated, true);
      return reply;
    };
    const first = await open(python);
    const discovered = await call('list_models', { workspaceRoot: project });
    assert.equal(discovered.discovery.status, 'ready', 'successful opening preserves the inspected-environment receipt used by static discovery and source review');
    const view = await call('open_graph_view', { modelId: first.models[0].modelId, revision: first.models[0].revision });
    const second = await open(other);
    const again = await call('open_graph_view', { modelId: second.models[0].modelId, revision: second.models[0].revision });
    assert.equal(new URL(again.url).origin, new URL(view.url).origin, 'changing capture Python must not restart the tool service');
    assert.equal(prompts, 1, 'the existing plugin permission owns inspection and execution');
    assert.deepEqual(inventory(python), before[0]); assert.deepEqual(inventory(other), before[1]);
  } finally { await client?.close(); await fs.rm(root, { recursive: true, force: true }); }
});

test('installed plugin Canvas selects, repairs and resumes the retained model, with MCP help for a real failure', { timeout: 180000 }, async () => {
  assert.ok(missingBase && path.isAbsolute(missingBase), 'Set a fresh pipless Torch/Pydantic fixture missing torchmetrics, OpenCV and SciPy.');
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-plugin-recovery-')));
  const project = path.join(root, 'project'), plugin = path.join(root, 'plugin'), storage = path.join(root, 'state');
  // Each run owns its mutable environment; rerunning the gate starts missing again.
  const prefix = path.join(root, 'capture');
  await fs.cp(path.dirname(path.dirname(missingBase)), prefix, { recursive: true, verbatimSymlinks: true, mode: constants.COPYFILE_FICLONE });
  const missing = path.join(prefix, 'bin', 'python');
  const before = inventory(missing);
  assert.ok(!before.pip && !before.scipy && !before.torchmetrics && !before['opencv-python']);
  let client, browser;
  try {
    await fs.mkdir(path.join(project, 'pkg'), { recursive: true });
    await fs.symlink(path.dirname(path.dirname(missing)), path.join(project, '.venv'));
    await fs.cp(path.resolve(__dirname, '../../../plugins/tensorviz'), plugin, { recursive: true });
    await readyService(plugin, storage);
    const marker = path.join(project, 'initializer-ran');
    await fs.writeFile(path.join(project, 'pkg/__init__.py'), `import torchmetrics, cv2, scipy\nfrom pathlib import Path\nPath(${JSON.stringify(marker)}).touch()\nINITIALIZED=True\n`);
    await fs.writeFile(path.join(project, 'pkg/model.py'), 'from torch import nn\nfrom pkg import INITIALIZED\nclass RecoveryNet(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc=nn.Linear(4,3)\n def forward(self,x):\n  assert INITIALIZED\n  return self.fc(x)\n');
    const recipe = (profile, model) => `\n[profiles.${profile}]\nkind="model"\nmodel="${model}"\nmode="eval"\ngrad_enabled=false\ndevice="cpu"\ninit={args=[],kwargs={}}\ninputs={args=[{kind="tensor",shape=[2,4],dtype="float32",requires_grad=false,generator={kind="full",value=1.0}}],kwargs={}}\n`;
    await fs.writeFile(path.join(project, 'tensorviz.toml'), 'version=1\n' + recipe('recovery', 'pkg.model:RecoveryNet'));
    let prompts = 0;
    client = new Client({ name: 'plugin-recovery-acceptance', version: '1' }, { capabilities: { elicitation: { form: {} } } });
    client.setRequestHandler(ElicitRequestSchema, async () => { prompts++; return { action: 'accept', content: { confirm: true } }; });
    await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(plugin, 'runtime/host.cjs')], cwd: root,
      env: { PATH: '', TENSORVIZ_DATA_DIR: storage }, stderr: 'inherit' }));
    const call = async (name, args) => {
      const reply = await client.callTool({ name, arguments: args });
      assert.ok(!reply.isError, JSON.stringify(reply)); return reply.structuredContent;
    };
    const poll = async reply => {
      for (let count = 0; count < 1200 && ['opening', 'resolving', 'saving', 'capturing'].includes(reply.status); count++) {
        await new Promise(resolve => setTimeout(resolve, 100)); reply = await call('open_graph', { requestId: reply.requestId });
      }
      return reply;
    };
    const panel = await client.callTool({ name: 'open_graph_panel', arguments: {} });
    const initial = await poll(await call('open_graph', { workspaceRoot: project, file: 'pkg/model.py', cls: 'RecoveryNet' }));
    assert.equal(initial.status, 'needs_setup', JSON.stringify(initial));
    assert.equal(initial.opening.environment.reason, 'selection_required');
    assert.equal(initial.validated, false);
    const exchange = await client.callTool({ name: 'graph_panel_exchange', arguments: { panelId: panel._meta.panelId, after: 0 } });
    assert.ok(!exchange.isError, JSON.stringify(exchange));
    let snapshot;
    for (let n = 0; n < 100 && !snapshot; n++) {
      const packet = (await client.callTool({ name: 'graph_panel_exchange', arguments: { panelId: panel._meta.panelId, after: 0 } }))._meta.panel;
      snapshot = packet.events.map(event => { const value = JSON.parse(event.wire); return value.encoding ? JSON.parse(gunzipSync(Buffer.from(value.data, 'base64'))) : value; })
        .find(message => message.type === 'snapshot' && message.state.opening);
      if (!snapshot) await new Promise(resolve => setTimeout(resolve, 50));
    }
    assert.equal(snapshot.state.opening.id, initial.opening.id, 'Canvas receives recovery before capture');
    const view = await call('open_graph_view', { modelId: initial.models[0].modelId, revision: initial.models[0].revision });
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const errors = new Set(); page.on('pageerror', error => errors.add(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.add(message.text()); });
    await page.goto(view.url);
    await page.getByRole('heading', { name: 'Choose Python for this model', exact: true }).waitFor();
    const candidate = initial.opening.environment.candidates.find(item => item.interpreter.startsWith(project + '/.venv/'));
    assert.ok(candidate);
    const rejected = await client.callTool({ name: 'open_graph', arguments: { requestId: initial.requestId, openingId: initial.opening.id,
      sourceDigest: initial.opening.target.sourceDigest, action: 'select_environment', candidateId: 'not-listed' } });
    assert.equal(rejected.structuredContent.error.code, 'INVALID_REQUEST');
    await page.locator(`input[value="${candidate.id}"]`).check();
    await page.getByRole('button', { name: /^Use / }).click();
    await page.getByRole('heading', { name: 'Add the missing libraries', exact: true }).waitFor();
    assert.equal(await fs.stat(marker).catch(() => null), null, 'metadata and library discovery must not import the model');
    assert.deepEqual(inventory(missing), before);
    const selected = await call('open_graph', { requestId: initial.requestId });
    assert.equal(selected.opening.id, initial.opening.id);
    assert.deepEqual(selected.opening.environment.requirements.sort(), ['opencv-python', 'scipy', 'torchmetrics']);
    const artifacts = process.env.TENSORVIZ_TEST_ARTIFACTS;
    if (artifacts) {
      assert.ok(path.isAbsolute(artifacts)); await fs.mkdir(artifacts, { recursive: true });
      await page.screenshot({ path: path.join(artifacts, 'plugin-missing-desktop.png') });
      await page.setViewportSize({ width: 390, height: 844 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      await page.screenshot({ path: path.join(artifacts, 'plugin-missing-narrow.png') });
      await page.setViewportSize({ width: 1280, height: 900 });
    }
    await page.getByRole('button', { name: 'Install missing libraries', exact: true }).click();
    await page.getByRole('heading', { name: 'Preparing your environment', exact: true }).waitFor();
    await page.getByRole('button', { name: 'Stop', exact: true }).waitFor();
    await page.getByText('Traced — shapes measured', { exact: true }).waitFor({ timeout: 120000 });
    const ready = await poll(await call('open_graph', { requestId: initial.requestId }));
    assert.equal(ready.status, 'ready', JSON.stringify(ready));
    assert.equal(ready.opening.id, initial.opening.id);
    assert.equal(ready.validated, true);
    assert.ok(await fs.stat(marker));
    const after = inventory(missing);
    for (const [name, version] of Object.entries(before)) assert.equal(after[name], version);
    assert.ok(after.scipy && after.torchmetrics && after['opencv-python']);
    assert.ok(!after.pip && !after.fastapi && !after.uvicorn && !after.libcst);
    assert.deepEqual(inventory(missingBase), before, 'repair must mutate only this run’s disposable copy');
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    assert.deepEqual([...errors], []);
    if (artifacts) await page.screenshot({ path: path.join(artifacts, 'plugin-captured-narrow.png') });
    await fs.writeFile(path.join(project, 'pkg/broken.py'), 'from torch import nn\nclass BrokenNet(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc=nn.Linear(4,3)\n def forward(self,x):\n  y=self.fc(x)\n  raise ValueError("plugin acceptance failure")\n');
    await fs.appendFile(path.join(project, 'tensorviz.toml'), recipe('broken', 'pkg.broken:BrokenNet'));
    const failed = await poll(await call('open_graph', { workspaceRoot: project, file: 'pkg/broken.py', cls: 'BrokenNet' }));
    assert.equal(failed.status, 'failed', JSON.stringify(failed));
    assert.equal(failed.opening.failure.kind, 'execution');
    assert.equal(failed.opening.hasCapture, true);
    assert.match(JSON.stringify(failed.helpRequest), /MCP directly/);
    assert.match(JSON.stringify(failed.helpRequest), /BrokenNet/);
    assert.equal(prompts, 1, 'repair and capture keep the existing plugin permission');
    if (artifacts) await fs.writeFile(path.join(artifacts, 'plugin-acceptance.json'), JSON.stringify({ openingId: initial.opening.id,
      ready: { status: ready.status, validated: ready.validated, openingId: ready.opening.id }, before, after,
      failure: failed.opening, helpRequest: failed.helpRequest, browserErrors: [...errors], prompts }, null, 2));
  } finally { await browser?.close(); await client?.close(); await fs.rm(root, { recursive: true, force: true }); }
});

test('declined plugin permission requires explicit recovery, and a missing choice stays visible until listed selection', { timeout: 120000 }, async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-plugin-unavailable-')));
  const project = path.join(root, 'project'), plugin = path.join(root, 'plugin'), storage = path.join(root, 'state');
  let client;
  try {
    await fs.mkdir(project); await fs.cp(path.resolve(__dirname, '../../../plugins/tensorviz'), plugin, { recursive: true });
    await readyService(plugin, storage);
    await fs.symlink(path.dirname(path.dirname(python)), path.join(project, '.venv'));
    const marker = path.join(project, 'imported');
    await fs.writeFile(path.join(project, 'model.py'), `from torch import nn\nfrom pathlib import Path\nPath(${JSON.stringify(marker)}).touch()\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc=nn.Linear(4,3)\n def forward(self,x): return self.fc(x)\n`);
    await fs.writeFile(path.join(project, 'tensorviz.toml'), 'version=1\n[profiles.tiny]\nkind="model"\nmodel="model:Tiny"\nmode="eval"\ngrad_enabled=false\ndevice="cpu"\ninit={args=[],kwargs={}}\ninputs={args=[{kind="tensor",shape=[2,4],dtype="float32",requires_grad=false,generator={kind="full",value=1.0}}],kwargs={}}\n');
    let prompts = 0;
    client = new Client({ name: 'plugin-unavailable-acceptance', version: '1' }, { capabilities: { elicitation: { form: {} } } });
    client.setRequestHandler(ElicitRequestSchema, async () => ({ action: 'accept', content: { confirm: ++prompts > 1 } }));
    await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(plugin, 'runtime/host.cjs')], cwd: root,
      env: { PATH: '', TENSORVIZ_DATA_DIR: storage }, stderr: 'inherit' }));
    const call = async args => {
      const result = await client.callTool({ name: 'open_graph', arguments: args });
      assert.ok(!result.isError, JSON.stringify(result)); return result.structuredContent;
    };
    const poll = async reply => {
      for (let count = 0; count < 400 && ['opening', 'resolving', 'saving', 'capturing'].includes(reply.status); count++) {
        await new Promise(resolve => setTimeout(resolve, 100)); reply = await call({ requestId: reply.requestId });
      }
      return reply;
    };
    const declined = await poll(await call({ workspaceRoot: project, file: 'model.py', cls: 'Tiny', interpreter: path.join(project, 'removed/bin/python') }));
    assert.equal(declined.status, 'blocked'); assert.equal(declined.error.code, 'APPROVAL_DECLINED');
    assert.equal(declined.nextAction.arguments.action, 'request_permission');
    assert.equal(await fs.stat(marker).catch(() => null), null);
    const choice = await poll(await call({ requestId: declined.requestId, action: 'request_permission' }));
    assert.equal(choice.status, 'needs_setup', JSON.stringify(choice));
    assert.equal(choice.opening.environment.reason, 'unavailable');
    const saved = choice.opening.environment.candidates.find(candidate => candidate.selected);
    assert.equal(saved.interpreter, path.join(project, 'removed/bin/python')); assert.equal(saved.status, 'unavailable');
    const identity = { requestId: choice.requestId, openingId: choice.opening.id, sourceDigest: choice.opening.target.sourceDigest };
    const refreshed = await poll(await call({ ...identity, action: 'refresh_environments' }));
    assert.equal(refreshed.opening.id, choice.opening.id);
    assert.equal(await fs.stat(marker).catch(() => null), null, 'refreshing must not import the model');
    const candidate = refreshed.opening.environment.candidates.find(item => item.interpreter.startsWith(project + '/.venv/'));
    const ready = await poll(await call({ ...identity, action: 'select_environment', candidateId: candidate.id }));
    assert.equal(ready.status, 'ready', JSON.stringify(ready)); assert.equal(ready.validated, true);
    assert.equal(ready.opening.id, choice.opening.id); assert.ok(await fs.stat(marker));
    assert.equal(prompts, 2, 'only explicit permission recovery reopens a declined prompt');
  } finally { await client?.close(); await fs.rm(root, { recursive: true, force: true }); }
});
