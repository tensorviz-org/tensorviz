// Official Codex registration changes in a disposable profile; no live settings.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { spawnSync } = require('node:child_process');
const { build } = require('esbuild');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { ElicitRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const { planInstall, applyInstall, recoverInstall } = require('../../../plugins/tensorviz/scripts/install.cjs');
const { withCodexConfig } = require('../../../plugins/tensorviz/scripts/codex-config.cjs');
const { configureCodexPermissions } = require('../../../plugins/tensorviz/scripts/codex-permissions.cjs');

test('Codex migrates explicit hosts and updates/rolls back without replacing active version assets', { timeout: 180000 }, async () => {
  const python = process.env.TENSORVIZ_TEST_PYTHON;
  assert.ok(python && path.isAbsolute(python), 'Set TENSORVIZ_TEST_PYTHON to a compatible interpreter.');
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-migration-')));
  const profile = path.join(root, 'profile'), project = path.join(root, 'project');
  const env = { ...process.env, CODEX_HOME: profile, TENSORVIZ_DATA_DIR: path.join(root, 'state') };
  const clients = [];
  const run = async args => {
    const result = spawnSync(process.env.TENSORVIZ_TEST_CODEX || 'codex', args, {
      env, cwd: project, encoding: 'utf8', timeout: 30000, detached: true, stdio: ['ignore', 'pipe', 'pipe'],
    });
    try { process.kill(-result.pid, 'SIGTERM'); } catch {}
    assert.equal(result.status, 0, result.stderr || result.error?.message);
    return result.stdout;
  };
  const listing = async () => JSON.parse(await run(['mcp', 'list', '--json']));
  const currentPath = async () => (await listing()).find(entry => entry.transport?.env?.PLUGIN_ROOT)?.transport.env.PLUGIN_ROOT;
  const withConfig = action => withCodexConfig(action, { command: process.env.TENSORVIZ_TEST_CODEX || 'codex', env, cwd: project });
  const apply = options => applyInstall({ ...options, cwd: project, dataDirectory: env.TENSORVIZ_DATA_DIR }, run,
    () => configureCodexPermissions({ command: process.env.TENSORVIZ_TEST_CODEX || 'codex', env, cwd: project }), withConfig);
  const catalog = async (directory, name) => {
    await fs.mkdir(path.join(directory, '.agents/plugins'), { recursive: true });
    await fs.writeFile(path.join(directory, '.agents/plugins/marketplace.json'), JSON.stringify({ name, plugins: [{
      name: 'tensorviz', source: { source: 'local', path: './plugins/tensorviz' },
      policy: { installation: 'AVAILABLE', authentication: 'ON_INSTALL' }, category: 'Productivity',
    }] }));
  };
  const start = async installed => {
    const client = new Client({ name: 'installed-version-acceptance', version: '1' }, { capabilities: { elicitation: { form: {} } } });
    clients.push(client);
    client.setRequestHandler(ElicitRequestSchema, async request => {
      assert.ok(request.params.message.includes(project) && request.params.message.includes(python));
      assert.match(request.params.message, /inspect environment|capture/);
      return { action: 'accept', content: { confirm: true } };
    });
    await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(installed, 'runtime/host.cjs')],
      cwd: project, env, stderr: 'pipe' }));
    const call = async (name, args = {}) => {
      const response = await client.callTool({ name, arguments: args });
      assert.ok(!response.isError, JSON.stringify(response));
      return response.structuredContent;
    };
    return { client, call };
  };
  const capture = async (session, interpreter) => {
    let result = await session.call('open_graph', { workspaceRoot: project, file: 'model.py', cls: 'Tiny', ...(interpreter ? { interpreter } : {}) });
    for (let n = 0; n < 450 && !['ready', 'failed', 'blocked', 'needs_setup', 'cancelled'].includes(result.status); n++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      result = await session.call('open_graph', { requestId: result.requestId });
    }
    assert.equal(result.status, 'ready', JSON.stringify(result));
    return { modelId: result.models[0].modelId, revision: result.models[0].revision };
  };
  try {
    await fs.mkdir(profile); await fs.mkdir(project);
    const model = 'from torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, 3)\n def forward(self, x):\n  return self.fc(x)\n';
    const recipe = 'version = 1\n# retain this comment\n[profiles.tiny]\nkind = "model"\nmodel = "model:Tiny"\nmode = "eval"\ngrad_enabled = false\ndevice = "cpu"\ninit = { args = [], kwargs = {} }\ninputs = { args = [{kind = "tensor", shape = [2, 4], dtype = "float32", requires_grad = false, generator = {kind = "full", value = 1.0}}], kwargs = {} }\n';
    await fs.writeFile(path.join(project, 'model.py'), model);
    await fs.writeFile(path.join(project, 'tensorviz.toml'), recipe);
    const interpreterBefore = await fs.stat(python);
    await fs.writeFile(path.join(profile, 'config.toml'), '# unrelated preference\nmodel = "test-preserved-model"\n');
    await run(['mcp', 'add', 'unrelated', '--url', 'http://127.0.0.1:2/mcp']);
    const companionRoot = path.join(root, 'companion'), original = path.join(root, 'release A'), updated = path.join(root, 'release B');
    // Reproduce the retired companion registration without shipping a second plugin.
    await catalog(companionRoot, 'tensorviz');
    await fs.mkdir(path.join(companionRoot, 'plugins/tensorviz/.codex-plugin'), { recursive: true });
    await fs.writeFile(path.join(companionRoot, 'plugins/tensorviz/.codex-plugin/plugin.json'), JSON.stringify({
      name: 'tensorviz', version: '0.1.0', description: 'Legacy VS Code companion fixture', mcpServers: './.mcp.json',
    }));
    await fs.writeFile(path.join(companionRoot, 'plugins/tensorviz/.mcp.json'), JSON.stringify({ mcpServers: { tensorviz: { type: 'http', url: 'http://127.0.0.1:1/mcp' } } }));
    await run(['plugin', 'marketplace', 'add', companionRoot]);
    await run(['plugin', 'add', 'tensorviz@tensorviz']);
    await catalog(original, 'tensorviz-standalone');
    await fs.cp(path.resolve(__dirname, '../../../plugins/tensorviz'), path.join(original, 'plugins/tensorviz'), { recursive: true });
    await run(['mcp', 'add', 'old-graph', '--url', 'http://127.0.0.1:1/mcp']);
    await run(['mcp', 'add', 'tensorviz', '--url', 'http://127.0.0.1:1/mcp']);
    const before = await fs.readFile(path.join(profile, 'config.toml'), 'utf8');
    const blocked = await applyInstall({ catalog: original, host: 'standalone', sessionsStopped: true }, run);
    assert.equal(blocked.status, 'blocked');
    assert.deepEqual(blocked.manual.map(entry => entry.name), ['old-graph', 'tensorviz'], 'Aliases and exact-name conflicts must both be identified');
    assert.equal(await fs.readFile(path.join(profile, 'config.toml'), 'utf8'), before);
    assert.match((await planInstall({ catalog: original }, run)).report.blockers.join(' '), /Choose --host/);
    assert.equal((await apply({ catalog: original, host: 'standalone' })).status, 'blocked');
    const migrated = await apply({ catalog: original, host: 'standalone', replaceConnections: true });
    assert.equal(migrated.status, 'installed');
    assert.ok(await currentPath(), 'Disabled manual tensorviz must not hide the plugin server');
    const manualState = () => run(['--disable', 'plugins', 'mcp', 'list', '--json']).then(JSON.parse);
    assert.ok((await manualState()).filter(entry => /tensorviz|old-graph/.test(entry.name)).every(entry => entry.enabled === false));
    assert.equal((await recoverInstall(migrated.recovery, run, withConfig)).status, 'restored');
    assert.equal(await fs.readFile(path.join(profile, 'config.toml'), 'utf8').then(text => text.includes('mcp_servers.old-graph')), true);
    assert.ok((await manualState()).every(entry => entry.enabled));
    assert.equal((await apply({ catalog: original, host: 'standalone', replaceConnections: true })).status, 'installed');
    assert.equal((await apply({ catalog: original, host: 'existing', connection: 'tensorviz-previous' })).status, 'installed');
    assert.equal((await manualState()).find(entry => entry.name === 'tensorviz-previous').enabled, true);
    assert.equal((await apply({ catalog: original, host: 'standalone', replaceConnections: true })).status, 'installed');
    assert.equal((await apply({ catalog: original, host: 'companion' })).status, 'installed');
    assert.equal((await apply({ catalog: original, host: 'standalone' })).status, 'installed');
    const firstPath = await currentPath(), first = await start(firstPath);
    const firstRef = await capture(first, python);
    const oldAssets = await fs.readFile(path.join(firstPath, 'browser/index.html'));
    const originalVersion = (await first.call('list_models')).host.version;
    const panelUri = `ui://tensorviz/graph-panel-${originalVersion}.html`;
    const firstPanel = await first.client.callTool({ name: 'open_graph_panel', arguments: {} });
    const firstExchange = await first.client.callTool({ name: 'graph_panel_exchange', arguments: { panelId: firstPanel._meta.panelId, after: 0 } });
    assert.ok(firstExchange._meta.panel.session.viewId);
    const oldPanel = (await first.client.readResource({ uri: panelUri })).contents[0].text;
    // Repair a same-version install after another client adds a colliding manual entry.
    await run(['mcp', 'add', 'tensorviz', '--url', 'http://127.0.0.1:1/mcp']);
    assert.equal((await apply({ catalog: original, host: 'standalone', replaceConnections: true })).status, 'installed');
    assert.ok(await currentPath());
    // A synthetic evaluation build also verifies the explicit 0.1.7 -> 0.1.0 reset.
    const nextVersion = '0.1.7-evaluation-test';
    await fs.cp(original, updated, { recursive: true });
    for (const file of ['plugin.json', '.codex-plugin/plugin.json', '.claude-plugin/plugin.json']) {
      const target = path.join(updated, 'plugins/tensorviz', file), data = JSON.parse(await fs.readFile(target, 'utf8'));
      data.version = nextVersion; await fs.writeFile(target, JSON.stringify(data));
    }
    await build({ absWorkingDir: path.resolve(__dirname, '../..'), entryPoints: ['src/main.ts'],
      outfile: path.join(updated, 'plugins/tensorviz/runtime/host.cjs'), bundle: true, platform: 'node', target: 'node22', format: 'cjs',
      define: { TENSORVIZ_VERSION: JSON.stringify(nextVersion) } });
    const upgrade = await apply({ catalog: updated, host: 'standalone', replaceCatalog: true });
    assert.equal(upgrade.status, 'installed');
    assert.ok(upgrade.rollbackCommand.includes(original));
    const secondPath = await currentPath();
    assert.notEqual(firstPath, secondPath);
    assert.equal(await fs.stat(firstPath).catch(() => null), null, 'Codex replaces its previous cache; the running host must have retained its own resources');
    assert.equal((await first.call('list_models')).host.version, originalVersion);
    assert.ok((await first.call('get_graph', { ...firstRef, mode: 'graph', limit: 100 })).nodes.length);
    const oldView = await first.call('open_graph_view', firstRef);
    const oldPage = await fetch(oldView.url);
    assert.equal(oldPage.status, 200);
    assert.equal(await oldPage.text(), oldAssets.toString(), 'Existing browser assets must still be served after a client update');
    assert.equal((await first.client.readResource({ uri: panelUri })).contents[0].text, oldPanel, 'The native panel resource survives cache replacement');
    const preservedPanel = await first.client.callTool({ name: 'graph_panel_exchange', arguments: { panelId: firstPanel._meta.panelId, after: 0 } });
    assert.equal(preservedPanel._meta.panel.session.viewId, firstExchange._meta.panel.session.viewId, 'An active native panel retains its broker view');
    await capture(first); // The old backend must still be able to launch capture workers.
    const second = await start(secondPath);
    assert.equal((await second.call('list_models')).host.version, nextVersion);
    const nextPanelUri = `ui://tensorviz/graph-panel-${nextVersion}.html`;
    assert.equal((await second.client.readResource({ uri: nextPanelUri })).contents[0].text, oldPanel);
    const secondRef = await capture(second); // Remembered interpreter survives plugin replacement.
    assert.equal((await apply({ catalog: original, host: 'standalone', replaceCatalog: true })).status, 'installed');
    assert.equal(await currentPath(), firstPath);
    assert.ok((await second.call('get_graph', { ...secondRef, mode: 'graph', limit: 100 })).nodes.length);
    assert.equal((await second.client.readResource({ uri: nextPanelUri })).contents[0].text, oldPanel, 'Rollback preserves resources in an already running newer host');
    await capture(second);
    const rollback = await start(firstPath);
    assert.equal((await rollback.call('list_models')).host.version, originalVersion);
    await capture(rollback);
    // Same-version changes must be refused before modifying cached files.
    await fs.appendFile(path.join(original, 'plugins/tensorviz/scripts/launch.cjs'), '\n// changed bytes\n');
    assert.match((await applyInstall({ catalog: original, host: 'standalone' }, run)).blockers.join(' '), /new version/);
    for (const client of clients) await client.close();
    clients.length = 0;
    await run(['plugin', 'remove', 'tensorviz@tensorviz-standalone']);
    assert.equal(await fs.readFile(path.join(project, 'model.py'), 'utf8'), model);
    assert.equal(await fs.readFile(path.join(project, 'tensorviz.toml'), 'utf8'), recipe);
    assert.equal((await fs.stat(python)).mtimeMs, interpreterBefore.mtimeMs);
    assert.deepEqual((await manualState()).map(entry => entry.name), ['old-graph', 'tensorviz-previous', 'tensorviz-previous-2', 'unrelated']);
    assert.ok((await manualState()).filter(entry => entry.name !== 'unrelated').every(entry => entry.enabled === false));
    assert.match(await fs.readFile(path.join(profile, 'config.toml'), 'utf8'), /# unrelated preference\nmodel = "test-preserved-model"/);
    console.log('Migration, both host choices, update, rollback and preserved active captures passed through real Codex caches.');
  } finally {
    for (const client of clients) await client.close();
    await fs.rm(root, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
  }
});
