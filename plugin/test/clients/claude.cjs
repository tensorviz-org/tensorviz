// Claude's documented stream-json control bridge; no model turn or provider key.
const toolNames = require('../tool-names.cjs');
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { spawn, spawnSync } = require('node:child_process');
const { createInterface } = require('node:readline');
const { checkClientRecovery } = require('../fixtures/client-recovery.cjs');

test('Claude installs the standalone plugin and captures through its native MCP bridge', { timeout: 180000 }, async () => {
  const python = process.env.TENSORVIZ_TEST_PYTHON;
  assert.ok(python && path.isAbsolute(python), 'Set TENSORVIZ_TEST_PYTHON to a compatible interpreter.');
  const cli = process.env.TENSORVIZ_TEST_CLAUDE || 'claude';
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-claude-')));
  const marketplace = path.join(root, 'marketplace with spaces'), project = path.join(root, 'project');
  const env = { ...process.env, CLAUDE_CONFIG_DIR: path.join(root, 'profile'), TENSORVIZ_DATA_DIR: path.join(root, 'state'),
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: '1', ANTHROPIC_API_KEY: 'test-no-provider-calls', CLAUDE_CODE_OAUTH_TOKEN: '' };
  const pending = new Map();
  let child, lines, stderr = '', nextId = 0, approvals = 0;
  const stop = pid => { if (pid) { try { process.kill(-pid, 'SIGTERM'); } catch {} } };
  const cliRun = args => {
    const result = spawnSync(cli, ['--bare', ...args], { env, cwd: project, detached: true, encoding: 'utf8', timeout: 30000 });
    stop(result.pid);
    assert.equal(result.status, 0, result.stderr || result.error?.message);
    return result.stdout;
  };
  try {
    await fs.mkdir(project); await fs.mkdir(env.CLAUDE_CONFIG_DIR);
    const model = 'from torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, 3)\n def forward(self, x):\n  return self.fc(x)\n';
    const recipe = 'version = 1\n[profiles.tiny]\nkind = "model"\nmodel = "model:Tiny"\nmode = "eval"\ngrad_enabled = false\ndevice = "cpu"\ninit = { args = [], kwargs = {} }\ninputs = { args = [{kind = "tensor", shape = [2, 4], dtype = "float32", requires_grad = false, generator = {kind = "full", value = 1.0}}], kwargs = {} }\n';
    await fs.writeFile(path.join(project, 'model.py'), model);
    await fs.writeFile(path.join(project, 'tensorviz.toml'), recipe);
    await fs.mkdir(path.join(marketplace, '.claude-plugin'), { recursive: true });
    await fs.writeFile(path.join(marketplace, '.claude-plugin/marketplace.json'), JSON.stringify({ name: 'tensorviz-standalone',
      owner: { name: 'TensorViz' }, plugins: [{ name: 'tensorviz', source: './plugins/tensorviz' }] }));
    await fs.cp(process.env.TENSORVIZ_TEST_PLUGIN ?? path.resolve(__dirname, '../../../plugins/tensorviz'), path.join(marketplace, 'plugins/tensorviz'), { recursive: true });
    const mcpFile = path.join(marketplace, 'plugins/tensorviz/.mcp.json');
    const mcp = JSON.parse(await fs.readFile(mcpFile, 'utf8'));
    mcp.mcpServers.tensorviz.env = { ...mcp.mcpServers.tensorviz.env, TENSORVIZ_DATA_DIR: env.TENSORVIZ_DATA_DIR };
    if (process.env.TENSORVIZ_TEST_DOWNLOAD_CA) mcp.mcpServers.tensorviz.env = { ...mcp.mcpServers.tensorviz.env,
      TENSORVIZ_RUNTIME_CACHE_DIR: path.join(root, 'runtime-cache'), CURL_CA_BUNDLE: process.env.TENSORVIZ_TEST_DOWNLOAD_CA };
    await fs.writeFile(mcpFile, JSON.stringify(mcp));
    cliRun(['plugin', 'marketplace', 'add', marketplace]);
    cliRun(['plugin', 'install', 'tensorviz@tensorviz-standalone', '--scope', 'user']);
    const installed = JSON.parse(cliRun(['plugin', 'list', '--json']));
    assert.equal(installed.length, 1);
    assert.equal(installed[0].id, 'tensorviz@tensorviz-standalone');
    assert.equal(installed[0].enabled, true);
    assert.ok(installed[0].installPath);
    console.log('Installed with', cliRun(['--version']).trim());
    // Only this disposable user's settings are loaded. The placeholder API key
    // prevents use of account credentials; no user/model message is submitted.
    child = spawn(cli, ['-p', '--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose',
      '--no-session-persistence', '--setting-sources', 'user'], { env, cwd: project, detached: true, stdio: ['pipe', 'pipe', 'pipe'] });
    child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-4000); });
    lines = createInterface({ input: child.stdout });
    lines.on('line', line => {
      const message = JSON.parse(line);
      if (message.type === 'control_response' && pending.has(message.response.request_id)) {
        const request = pending.get(message.response.request_id);
        clearTimeout(request.timer); pending.delete(message.response.request_id);
        message.response.subtype === 'error' ? request.reject(new Error(message.response.error)) : request.resolve(message.response.response);
      } else if (message.type === 'control_request') {
        const request = message.request;
        if (request.subtype === 'elicitation' && request.mode === 'form' && request.message.includes(project)
          && request.message.includes(python) && /inspect environment|capture/.test(request.message)) {
          approvals++;
          child.stdin.write(JSON.stringify({ type: 'control_response', response: { subtype: 'success', request_id: message.request_id,
            response: { action: 'accept', content: { confirm: true } } } }) + '\n');
        } else child.stdin.write(JSON.stringify({ type: 'control_response', response: { subtype: 'error', request_id: message.request_id,
          error: 'Only the exact synthetic test environment and capture are authorized.' } }) + '\n');
      }
    });
    const request = data => new Promise((resolve, reject) => {
      const id = String(++nextId), timer = setTimeout(() => { pending.delete(id); reject(new Error(`${data.subtype} timed out: ${stderr}`)); }, 30000);
      pending.set(id, { resolve, reject, timer });
      child.stdin.write(JSON.stringify({ type: 'control_request', request_id: id, request: data }) + '\n');
    });
    const initialized = await request({ subtype: 'initialize' });
    assert.match(JSON.stringify(initialized.commands), /tensorviz:model-workflow/);
    // Initialization can finish while the plugin connection is still pending.
    let connection;
    const connectionDeadline = Date.now() + 30000;
    do {
      connection = (await request({ subtype: 'mcp_status' })).mcpServers.find(server => server.source === 'plugin');
      if (connection?.status === 'connected') break;
      await new Promise(resolve => setTimeout(resolve, 100));
    } while (Date.now() < connectionDeadline);
    assert.equal(connection?.status, 'connected', JSON.stringify(connection));
    const call = async (tool, args = {}) => {
      // Exercise Claude's native MCP control bridge without a model turn.
      const result = await request({ subtype: 'mcp_call', tool: `mcp__plugin_tensorviz_tensorviz__${tool}`, arguments: args });
      assert.ok(!result.isError, JSON.stringify(result));
      return result.structuredContent;
    };
    const discovery = await call('list_models');
    assert.equal(discovery.host.kind, 'standalone');
    const status = await request({ subtype: 'mcp_status' });
    assert.equal(status.mcpServers.length, 1);
    assert.equal(status.mcpServers[0].source, 'plugin');
    assert.equal(status.mcpServers[0].tools.length, toolNames.length);
    for (const [tool, query] of [
      ['find_computation', { text: 'Linear' }],
      ['query_computation', { anchors: [{ kind: 'value', id: 'missing' }] }],
      ['analyze_computation_change', { changes: [{ valueId: 'missing', shape: [2, 4] }] }],
    ]) {
      await assert.rejects(request({ subtype: 'mcp_call', tool: `mcp__plugin_tensorviz_tensorviz__${tool}`,
        arguments: { modelId: 'unopened', captureId: 'missing', sourceId: 'missing', ...query } }),
        error => JSON.parse(error.message).error.code === 'NO_MODEL', 'analysis must not open or run a model');
    }
    console.log(`Loaded ${toolNames.length} plugin-owned tools (including 4 app-only Canvas tools), checked all three computation tools and loaded model-workflow through Claude.`);
    assert.equal(await fs.stat(env.TENSORVIZ_DATA_DIR).catch(() => null), null);
    let capture = await call('open_graph', { workspaceRoot: project, file: 'model.py', cls: 'Tiny', interpreter: python });
    for (let n = 0; n < 450 && !['ready', 'failed', 'blocked', 'needs_setup', 'cancelled'].includes(capture.status); n++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      capture = await call('open_graph', { requestId: capture.requestId });
    }
    assert.equal(capture.status, 'ready', JSON.stringify(capture));
    assert.equal(approvals, 1);
    const permission = JSON.parse(await fs.readFile(path.join(env.TENSORVIZ_DATA_DIR, 'permissions/standalone.json'), 'utf8'));
    assert.equal(permission.decision, 'granted', 'Consent must be saved inside this test fixture');
    const graph = await call('get_graph', { modelId: capture.models[0].modelId, revision: capture.models[0].revision, mode: 'graph', limit: 100 });
    assert.ok(graph.nodes.some(node => JSON.stringify(node.shapes).includes('3')));
    const context = await call('get_model_context', { modelId: capture.models[0].modelId, revision: capture.models[0].revision });
    assert.equal(context.available, true, JSON.stringify(context));
    assert.ok(Buffer.byteLength(JSON.stringify(context)) <= 4000);
    const fc = context.modules.find(module => module.paths.includes('fc'));
    assert.equal(fc.calls[0].parameterCount, 15);
    assert.deepEqual(fc.calls[0].inputs.tensors[0].shape, [2, 4]);
    assert.deepEqual(fc.calls[0].outputs.tensors[0].shape, [2, 3]);
    console.log('Captured Tiny and inspected measured shapes through Claude with one saved plugin permission.');
    const pendingPanels = new Set();
    for (let count = 0; count < 16; count++) {
      const reply = await request({ subtype: 'mcp_call', tool: 'mcp__plugin_tensorviz_tensorviz__open_graph_panel', arguments: {} });
      assert.ok(!reply.isError, JSON.stringify(reply));
      assert.equal(reply.structuredContent.displayed, false);
      assert.ok(reply._meta.panelId);
      assert.ok(!pendingPanels.has(reply._meta.panelId), 'every opener keeps a distinct frame identity, including at capacity');
      assert.equal(reply.structuredContent.status, 'awaiting_view');
      assert.equal(reply.structuredContent.fallbackTool, 'open_graph_view');
      pendingPanels.add(reply._meta.panelId);
    }
    console.log('Sixteen unopened graph requests retained distinct identities without exhausting the panel budget through the Claude bridge.');
    await checkClientRecovery({ call, project, python, model, recipe, initial: capture });
    child.stdin.end();
    await new Promise(resolve => child.once('exit', resolve));
    stop(child.pid); child = undefined;
    cliRun(['plugin', 'uninstall', 'tensorviz@tensorviz-standalone', '--scope', 'user']);
    assert.equal(await fs.readFile(path.join(project, 'model.py'), 'utf8'), model);
    assert.equal(await fs.readFile(path.join(project, 'tensorviz.toml'), 'utf8'), recipe);
  } finally {
    for (const request of pending.values()) clearTimeout(request.timer);
    lines?.close(); stop(child?.pid);
    await fs.rm(root, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
  }
});
