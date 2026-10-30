const { readyService } = require('../fixtures/service.cjs');
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { ElicitRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const python = process.env.TENSORVIZ_TEST_PYTHON;
assert.ok(python && path.isAbsolute(python), 'Set the explicit test interpreter.');

test('workspace discovery is static, bounded and independent of the active graph', { timeout: 60000 }, async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-discovery-')));
  const project = path.join(root, 'project'), other = path.join(root, 'other'), plugin = path.join(root, 'plugin');
  let client;
  try {
    await fs.mkdir(project); await fs.mkdir(other);
    await fs.cp(path.resolve(__dirname, '../../../plugins/tensorviz'), plugin, { recursive: true });
    await readyService(plugin, path.join(root, 'state'));
    const marker = path.join(project, 'imported');
    await fs.writeFile(path.join(project, 'base.py'), `from pathlib import Path\nPath(${JSON.stringify(marker)}).touch()\nfrom torch import nn\nclass Base(nn.Module):\n pass\n`);
    await fs.writeFile(path.join(project, 'model.py'), 'from base import Base\nclass Model(Base):\n pass\n');
    await fs.writeFile(path.join(project, 'broken.py'), 'class Broken(\n');
    await fs.mkdir(path.join(project, '.venv'));
    await fs.writeFile(path.join(project, '.venv/ignored.py'), 'from torch import nn\nclass Wrong(nn.Module):\n pass\n');
    let prompts = 0;
    client = new Client({ name: 'workspace-discovery-test', version: '1' }, { capabilities: { elicitation: { form: {} } } });
    client.setRequestHandler(ElicitRequestSchema, async () => { prompts++; return { action: 'accept', content: { confirm: true } }; });
    await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(plugin, 'runtime/host.cjs')], cwd: other,
      env: { PATH: '', TENSORVIZ_DATA_DIR: path.join(root, 'state') }, stderr: 'inherit' }));
    const call = async (name, args) => {
      const result = await client.callTool({ name, arguments: args });
      assert.ok(!result.isError, JSON.stringify(result)); return result.structuredContent;
    };
    const unprepared = await call('list_models', { workspaceRoot: project });
    assert.equal(unprepared.discovery.status, 'environment_required');
    assert.equal(prompts, 0, 'discovery cannot grant environment execution permission');
    let environment = await call('get_execution_environment', { workspaceRoot: project, interpreter: python });
    for (let count = 0; count < 200 && environment.status === 'checking'; count++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      environment = await call('get_execution_environment', { requestId: environment.requestId });
    }
    assert.equal(environment.status, 'ready');
    assert.equal(environment.environment.compatible, true);
    const discovered = await call('list_models', { workspaceRoot: project, limit: 1 });
    assert.deepEqual(discovered.models, [], 'static discovery does not open or select a graph');
    assert.equal(discovered.discovery.executed, false);
    assert.equal(discovered.discovery.count, 2);
    assert.equal(discovered.discovery.models[0].cls, 'Base');
    assert.equal(discovered.discovery.nextOffset, 1);
    assert.ok(discovered.discovery.notes.some(note => note.file === 'broken.py'));
    const next = await call('list_models', { workspaceRoot: project, offset: 1, limit: 1 });
    assert.equal(next.discovery.models[0].cls, 'Model');
    assert.equal(next.discovery.models[0].nextAction.arguments.workspaceRoot, project);
    assert.equal(next.discovery.models[0].nextAction.arguments.interpreter, python);
    assert.equal(await fs.stat(marker).catch(() => null), null, 'source analysis must never import a discovered file');
    assert.equal((await call('list_models', { workspaceRoot: other })).discovery.count, 0);
    assert.equal(prompts, 1);
    await Promise.all(Array.from({ length: 256 }, (_, index) => fs.writeFile(path.join(project, `filler${index}.py`), '# bounded inventory\n')));
    assert.equal((await call('list_models', { workspaceRoot: project })).discovery.truncated, true);
    assert.equal(await fs.stat(path.join(project, 'tensorviz.toml')).catch(() => null), null);
  } finally { await client?.close(); await fs.rm(root, { recursive: true, force: true }); }
});
