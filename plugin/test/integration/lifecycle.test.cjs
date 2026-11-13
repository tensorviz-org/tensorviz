const { copyBuiltPlugin } = require('../fixtures/plugin.cjs');
const { readyService } = require('../fixtures/service.cjs');
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
const python = process.env.TENSORVIZ_TEST_PYTHON;
assert.ok(python && path.isAbsolute(python), 'Set the explicit test interpreter.');
const delay = () => new Promise(resolve => setTimeout(resolve, 100));
async function alive(pid) {
  if (process.platform === 'win32') { try { process.kill(pid, 0); return true; } catch { return false; } }
  try { return !/^[Z\s]*$/.test((await execFile('ps', ['-o', 'stat=', '-p', String(pid)])).stdout); }
  catch { return false; }
}

for (const scenario of ['cancel', 'launcher-crash', 'source-change', 'dependency-change']) {
  test(`copied-plugin worker lifecycle: ${scenario}`, { timeout: 60000 }, async () => {
    const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-lifecycle-')));
    const project = path.join(root, 'project'), plugin = path.join(root, 'plugin');
    const marker = path.join(project, 'worker.json'), release = path.join(project, 'release');
    let client, transport, pids;
    try {
      await fs.mkdir(project);
      await copyBuiltPlugin(plugin);
    await readyService(plugin, path.join(root, 'state'));
      const sourceChange = scenario === 'source-change' || scenario === 'dependency-change';
      const source = `import os, sys, json, subprocess, time\nfrom torch import nn\nfrom helper import WIDTH\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, WIDTH)\n def forward(self, x):\n  child = ${sourceChange ? 'None' : "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])"}\n  with open(${JSON.stringify(marker)}, 'w') as out: json.dump({'worker': os.getpid(), 'child': child.pid if child else None}, out)\n  while not os.path.exists(${JSON.stringify(release)}): time.sleep(0.05)\n  return self.fc(x)\n`;
      await fs.writeFile(path.join(project, 'model.py'), source);
      await fs.writeFile(path.join(project, 'helper.py'), 'WIDTH = 3\n');
      await fs.writeFile(path.join(project, 'tensorviz.toml'), 'version = 1\n[profiles.tiny]\nkind = "model"\nmodel = "model:Tiny"\nmode = "eval"\ndevice = "cpu"\ngrad_enabled = false\ninit = { args = [], kwargs = {} }\ninputs = { args = [{ kind = "tensor", shape = [2, 4], dtype = "float32", requires_grad = false, generator = { kind = "full", value = 1 } }], kwargs = {} }\n');
      client = new Client({ name: 'lifecycle-test', version: '1' }, { capabilities: { elicitation: { form: {} } } });
      client.setRequestHandler(ElicitRequestSchema, async () => ({ action: 'accept', content: { confirm: true } }));
      transport = new StdioClientTransport({ command: process.execPath, args: [path.join(plugin, 'runtime/host.cjs')], cwd: root,
        env: { PATH: '', TENSORVIZ_DATA_DIR: path.join(root, 'state') }, stderr: 'inherit' });
      await client.connect(transport);
      const call = async args => {
        const result = await client.callTool({ name: 'open_graph', arguments: args });
        assert.ok(!result.isError, JSON.stringify(result)); return result.structuredContent;
      };
      const opening = await call({ workspaceRoot: project, file: 'model.py', cls: 'Tiny', interpreter: python });
      for (let count = 0; count < 250 && !pids; count++) {
        try { pids = JSON.parse(await fs.readFile(marker, 'utf8')); } catch { await delay(); }
        if (!pids && count % 10 === 0) {
          const progress = await call({ requestId: opening.requestId });
          assert.ok(!['failed', 'blocked', 'needs_setup', 'cancelled'].includes(progress.status), JSON.stringify(progress));
        }
      }
      assert.ok(pids?.worker, JSON.stringify(await call({ requestId: opening.requestId })));
      if (scenario === 'launcher-crash') process.kill(transport.pid, 'SIGKILL');
      else if (scenario === 'cancel') {
        const cancelled = await call({ requestId: opening.requestId, action: 'cancel' });
        assert.equal(cancelled.status, 'cancelled');
      } else {
        if (scenario === 'dependency-change') await fs.writeFile(path.join(project, 'helper.py'), 'WIDTH = 2\n');
        else await fs.writeFile(path.join(project, 'model.py'), source.replace('nn.Linear(4, WIDTH)', 'nn.Linear(4, 2)'));
        await fs.writeFile(release, 'finish the obsolete forward');
        let result;
        for (let count = 0; count < 150; count++) {
          result = await call({ requestId: opening.requestId });
          if (!['opening', 'resolving', 'capturing'].includes(result.status)) break;
          await delay();
        }
        assert.equal(result.status, 'source_changed', JSON.stringify(result));
        assert.equal(result.validated, false);
        assert.equal(result.nextAction.kind, 'restart');
        const listed = await client.callTool({ name: 'list_models', arguments: {} });
        assert.ok(listed.structuredContent.models.every(model => !model.current && !model.traced),
          'only a stale source preview may remain; obsolete capture must not become current');
        if (scenario === 'dependency-change') assert.equal(await fs.readFile(path.join(project, 'model.py'), 'utf8'), source,
          'a dependency-only edit must invalidate execution even when the selected model file is unchanged');
        let refreshed = await call({ workspaceRoot: project, file: 'model.py', cls: 'Tiny', interpreter: python });
        for (let count = 0; count < 150 && ['opening', 'resolving', 'capturing'].includes(refreshed.status); count++) {
          await delay(); refreshed = await call({ requestId: refreshed.requestId });
        }
        assert.equal(refreshed.status, 'ready', JSON.stringify(refreshed));
        assert.ok(refreshed.models[0].current && refreshed.models[0].traced, 'Only a fresh capture may restore current evidence');
        const graph = await client.callTool({ name: 'get_graph', arguments: {
          modelId: refreshed.models[0].modelId, revision: refreshed.models[0].revision, mode: 'graph', limit: 100,
        } });
        assert.ok(!graph.isError, JSON.stringify(graph));
        assert.ok(graph.structuredContent.nodes.some(node => node.shapes.output?.join(',') === '2,2'),
          'Reopening must capture the changed dependency instead of reusing its old output width');
      }
      for (const pid of [pids.worker, pids.child].filter(Boolean)) {
        for (let count = 0; count < 70 && await alive(pid); count++) await delay();
        assert.equal(await alive(pid), false, `owned process ${pid} survived ${scenario}`);
      }
    } finally {
      await client?.close();
      if (pids?.worker && process.platform !== 'win32') { try { process.kill(-pids.worker, 'SIGKILL'); } catch {} }
      await fs.rm(root, { recursive: true, force: true });
    }
  });
}
