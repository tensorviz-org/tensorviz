// Real CLI installation and app-server MCP bridge, isolated from live client settings.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { spawn, spawnSync } = require('node:child_process');
const { createInterface } = require('node:readline');

const agentTest = process.env.TENSORVIZ_TEST_CODEX_AGENT === '1';
const installationConsent = process.env.TENSORVIZ_TEST_INSTALL_PERMISSIONS === '1';

test(agentTest ? 'Codex selects the installed skill and captures during an ordinary model task'
  : 'Codex installs the copied plugin, captures through native elicitation, and uninstalls cleanly', { timeout: agentTest ? 300000 : 120000 }, async () => {
  const python = process.env.TENSORVIZ_TEST_PYTHON;
  assert.ok(python && path.isAbsolute(python), 'Set TENSORVIZ_TEST_PYTHON to an existing compatible interpreter.');
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-codex-client-')));
  const marketplace = path.join(root, 'marketplace with spaces'), project = path.join(root, 'project');
  const clientState = path.join(root, 'codex-profile');
  const env = { ...process.env, CODEX_HOME: clientState, TENSORVIZ_DATA_DIR: path.join(root, 'tensorviz-state'), RUST_LOG: 'codex_rmcp_client=info,codex_mcp=info' };
  const cli = process.env.TENSORVIZ_TEST_CODEX || 'codex';
  const pending = new Map();
  let child, lines, nextId = 0, stderr = '', approvals = 0, toolApprovals = 0, completedTurn, serverName;
  const completedItems = new Map();
  const activeCalls = new Map(), openingRequests = new Set();
  const redact = value => String(value).replace(/https?:\/\/[^\s"'<>]+/g, '[URL omitted]')
    .replace(/eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/g, '[token omitted]');
  const stopGroup = pid => { if (pid) { try { process.kill(-pid, 'SIGTERM'); } catch {} } };
  const cliRun = args => {
    const result = spawnSync(cli, args, { env, cwd: project, detached: true, encoding: 'utf8', timeout: 30000, stdio: ['ignore', 'pipe', 'pipe'] });
    stopGroup(result.pid);
    assert.equal(result.status, 0, result.stderr || result.error?.message);
    return result.stdout;
  };
  try {
    await fs.mkdir(project, { recursive: true });
    const modelSource = 'from torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, 3)\n def forward(self, x):\n  return self.fc(x)\n';
    await fs.writeFile(path.join(project, 'model.py'), modelSource);
    const recipe = 'version = 1\n[profiles.tiny]\nkind = "model"\nmodel = "model:Tiny"\nmode = "eval"\ngrad_enabled = false\ndevice = "cpu"\ninit = { args = [], kwargs = {} }\ninputs = { args = [{kind = "tensor", shape = [2, 4], dtype = "float32", requires_grad = false, generator = {kind = "full", value = 1.0}}], kwargs = {} }\n';
    await fs.writeFile(path.join(project, 'tensorviz.toml'), recipe);
    await fs.mkdir(clientState);
    if (agentTest) {
      const authFile = process.env.TENSORVIZ_TEST_AGENT_AUTH_FILE;
      assert.ok(authFile && path.isAbsolute(authFile), 'This opt-in model test requires an explicitly authorized TENSORVIZ_TEST_AGENT_AUTH_FILE.');
      await fs.chmod(root, 0o700);
      await fs.chmod(clientState, 0o700);
      await fs.copyFile(authFile, path.join(clientState, 'auth.json'));
      await fs.chmod(path.join(clientState, 'auth.json'), 0o600);
      await fs.writeFile(path.join(clientState, 'config.toml'), 'cli_auth_credentials_store = "file"\n[features]\napps = false\n');
      await fs.writeFile(path.join(project, 'python-path.txt'), python + '\n');
      delete env.OPENAI_API_KEY;
      delete env.CODEX_API_KEY;
    }
    await fs.mkdir(path.join(marketplace, '.agents/plugins'), { recursive: true });
    const pluginSource = process.env.TENSORVIZ_TEST_PLUGIN ?? path.resolve(__dirname, '../../../plugins/tensorviz');
    assert.ok(path.isAbsolute(pluginSource));
    await fs.cp(pluginSource, path.join(marketplace, 'plugins/tensorviz'), { recursive: true });
    // Codex filters inherited MCP environment variables. Pin only the copied
    // test manifest to this fixture so live saved consent cannot affect it.
    const mcpFile = path.join(marketplace, 'plugins/tensorviz/mcp.json');
    const mcp = JSON.parse(await fs.readFile(mcpFile, 'utf8'));
    mcp.mcpServers.tensorviz.env = { ...mcp.mcpServers.tensorviz.env, TENSORVIZ_DATA_DIR: env.TENSORVIZ_DATA_DIR };
    if (process.env.TENSORVIZ_TEST_DOWNLOAD_CA) mcp.mcpServers.tensorviz.env = { ...mcp.mcpServers.tensorviz.env,
      HOME: path.join(root, 'runtime-home'), CURL_CA_BUNDLE: process.env.TENSORVIZ_TEST_DOWNLOAD_CA };
    await fs.writeFile(mcpFile, JSON.stringify(mcp));
    await fs.writeFile(path.join(marketplace, '.agents/plugins/marketplace.json'), JSON.stringify({ name: 'tensorviz-standalone', plugins: [{
      name: 'tensorviz', source: { source: 'local', path: './plugins/tensorviz' },
      policy: { installation: 'AVAILABLE', authentication: 'ON_INSTALL' }, category: 'Productivity',
    }] }));
    const thin = await fs.stat(path.join(pluginSource, 'runtime-lock.json')).catch(() => null);
    let report;
    if (thin) {
      assert.equal(installationConsent, false, 'Native client installation leaves TensorViz consent to first use');
      cliRun(['plugin', 'marketplace', 'add', marketplace, '--json']);
      const installed = JSON.parse(cliRun(['plugin', 'add', 'tensorviz@tensorviz-standalone', '--json']));
      report = { status: 'installed', installed: [installed] };
    } else {
      const { applyInstall } = require(path.join(marketplace, 'plugins/tensorviz/scripts/install.cjs'));
      const { configureCodexPermissions } = require(path.join(marketplace, 'plugins/tensorviz/scripts/codex-permissions.cjs'));
      report = await applyInstall({ catalog: marketplace, host: 'standalone', acceptPermissions: installationConsent,
        dataDirectory: env.TENSORVIZ_DATA_DIR }, cliRun, () => configureCodexPermissions({ command: cli, env }));
    }
    assert.equal(report.status, 'installed', JSON.stringify(report));
    const servers = JSON.parse(cliRun(['mcp', 'list', '--json']));
    assert.equal(servers.length, 1, JSON.stringify({ report, plugins: JSON.parse(cliRun(['plugin', 'list', '--marketplace', 'tensorviz-standalone', '--json'])) }));
    const installedEnv = servers[0].transport.env;
    assert.equal(installedEnv.TENSORVIZ_DATA_DIR, env.TENSORVIZ_DATA_DIR, 'MCP must use the disposable permission and runtime directory');
    const installedPath = installedEnv.PLUGIN_ROOT;
    const installation = { ...report.installed[0], installedPath };
    assert.ok((await fs.stat(path.join(installation.installedPath, thin ? 'runtime-lock.json' : 'runtime/host.cjs'))).size > 0);
    console.log('Installed with', cliRun(['--version']).trim());
    child = spawn(cli, ['app-server', '--stdio'], { env, cwd: project, detached: true, stdio: ['pipe', 'pipe', 'pipe'] });
    child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-4000); });
    lines = createInterface({ input: child.stdout });
    lines.on('line', line => {
      const message = JSON.parse(line);
      if (message.id !== undefined && pending.has(message.id)) {
        const { resolve, reject, timer } = pending.get(message.id);
        clearTimeout(timer); pending.delete(message.id);
        message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result);
      } else if (message.id !== undefined && message.method === 'mcpServer/elicitation/request'
        && message.params.mode === 'form' && message.params.message.includes(project)
        && message.params.message.includes(python) && message.params.message.startsWith('Allow TensorViz to work with local models without asking again?')) {
        approvals++;
        child.stdin.write(JSON.stringify({ id: message.id, result: { action: 'accept', content: { confirm: true } } }) + '\n');
      } else if (agentTest && message.id !== undefined && message.method === 'mcpServer/elicitation/request'
        && message.params.serverName === serverName && message.params.mode === 'form'
        && message.params.message === `Allow the ${serverName} MCP server to run tool "open_graph"?`
        && !message.params.requestedSchema?.required?.length
        && [...activeCalls.values()].some(item => item.server === serverName && item.tool === 'open_graph'
          && (openingRequests.has(item.arguments?.requestId)
            || (item.arguments?.workspaceRoot === project && item.arguments?.cls === 'Tiny'
              && ['model.py', path.join(project, 'model.py')].includes(item.arguments?.file)
              && item.arguments?.interpreter === python)))) {
        toolApprovals++;
        child.stdin.write(JSON.stringify({ id: message.id, result: { action: 'accept', content: {} } }) + '\n');
      } else if (message.id !== undefined) {
        if (agentTest) console.log('Unexpected client request:', message.method,
          redact(message.params?.message ?? message.params?.reason ?? '').slice(0,1200),
          JSON.stringify({ mode: message.params?.mode, schema: message.params?.requestedSchema }));
        child.stdin.write(JSON.stringify({ id: message.id, error: { code: -32601, message: 'No unexpected operations are authorized in this installation test.' } }) + '\n');
      } else if (agentTest && message.method === 'item/started' && message.params.item.type === 'mcpToolCall') {
        activeCalls.set(message.params.item.id, message.params.item);
      } else if (agentTest && message.method === 'item/completed') {
        completedItems.set(message.params.item.id, message.params.item);
        const item = message.params.item;
        activeCalls.delete(item.id);
        if (item.type === 'mcpToolCall' && item.tool === 'open_graph' && item.result?.structuredContent?.requestId)
          openingRequests.add(item.result.structuredContent.requestId);
        if (item.type === 'mcpToolCall') console.log('Agent tool:', item.tool, item.status,
          item.error ? redact(JSON.stringify(item.error)).slice(0, 1500) : '');
      } else if (agentTest && message.method === 'turn/completed') {
        completedTurn = message.params.turn;
      }
    });
    const request = (method, params) => new Promise((resolve, reject) => {
      const id = ++nextId;
      const timer = setTimeout(() => { pending.delete(id); reject(new Error(`${method} timed out${agentTest ? '' : ': ' + stderr}`)); }, 30000);
      pending.set(id, { resolve, reject, timer });
      child.stdin.write(JSON.stringify({ id, method, params }) + '\n');
    });
    await request('initialize', { clientInfo: { name: 'tensorviz-acceptance', version: '1' }, capabilities: { experimentalApi: true } });
    child.stdin.write(JSON.stringify({ method: 'initialized' }) + '\n');
    const started = await request('thread/start', { cwd: project, ephemeral: true, approvalPolicy: 'on-request', sandbox: 'workspace-write',
      ...(agentTest ? { developerInstructions: 'Work only with the synthetic project, its listed interpreter, and installed plugin skills/tools. Do not read credentials or unrelated user files, install packages, edit project files, or start other agents. No browser UI is attached to this test client.' } : {}) });
    const threadId = started.thread.id;
    const status = await request('mcpServerStatus/list', { threadId });
    assert.equal(status.data.length, 1, 'An isolated install must start only one plugin-owned server');
    const server = status.data.find(item => Object.values(item.tools).some(tool => tool.name === 'open_graph'));
    assert.ok(server, `Installed TensorViz tools unavailable: ${JSON.stringify(status)}\n${stderr}`);
    assert.ok(Object.values(server.tools).some(tool => tool.name === 'open_graph_diff'));
    serverName = server.name;
    const call = async (tool, args) => {
      const result = await request('mcpServer/tool/call', { threadId, server: server.name, tool, arguments: args });
      assert.ok(!result.isError, JSON.stringify(result)); return result.structuredContent;
    };
    const discovery = await call('list_models', {});
    assert.equal(discovery.host.kind, 'standalone');
    assert.equal(discovery.host.version, installation.version);
    if (installationConsent) assert.deepEqual(await fs.readdir(env.TENSORVIZ_DATA_DIR), ['permissions'], 'Installation consent must not provision Python or inspect projects');
    else assert.equal(await fs.stat(env.TENSORVIZ_DATA_DIR).catch(() => null), null, 'installation/discovery must not provision or select an environment');
    const skills = await request('skills/list', { cwds: [project], forceReload: true });
    assert.ok(skills.data.flatMap(entry => entry.skills).some(skill => skill.enabled
      && skill.pluginId === installation.pluginId && skill.path.startsWith(installation.installedPath + path.sep)
      && skill.name.endsWith('model-workflow')), 'The installed skill must come from this copied plugin');
    console.log(`Loaded ${Object.keys(server.tools).length} tools and model-workflow through the installed plugin.`);
    let capture;
    if (agentTest) {
      const prompt = 'Explain the tensor flow through Tiny in model.py and verify the input and output shapes using the saved example. The available Python interpreter is recorded in python-path.txt. Do not edit the project or install packages.';
      assert.ok(!/tensorviz/i.test(prompt));
      console.log('Agent model:', started.model, 'Prompt:', prompt);
      await request('turn/start', { threadId, input: [{ type: 'text', text: prompt, text_elements: [] }] });
      const deadline = Date.now() + 240000;
      while (!completedTurn && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 200));
      assert.ok(completedTurn, 'The model turn did not finish within four minutes.');
      assert.equal(completedTurn.status, 'completed', 'The model turn failed or was interrupted.');
      for (const item of completedTurn.items) completedItems.set(item.id, item);
      const items = [...completedItems.values()];
      const skillPath = path.join(installation.installedPath, 'skills/model-workflow/SKILL.md');
      const skillRead = items.some(item => item.type === 'commandExecution' && item.exitCode === 0 && item.command.includes(skillPath));
      const calls = items.filter(item => item.type === 'mcpToolCall' && item.server === server.name);
      capture = calls.filter(item => item.tool === 'open_graph' && item.status === 'completed')
        .map(item => item.result?.structuredContent).find(result => result?.status === 'ready');
      console.log('Behavioral evidence:', JSON.stringify({ skillRead, captureReady: Boolean(capture), toolApprovals, tools: calls.map(item => item.tool) }));
      const answer = items.filter(item => item.type === 'agentMessage').map(item => item.text).join('\n');
      console.log('Agent report:', redact(answer).slice(-4000));
      assert.ok(skillRead, 'The agent must actually read the installed skill; manifest discovery alone does not pass.');
      if (installationConsent) assert.equal(toolApprovals, 0, 'Installation consent covers the TensorViz server policy');
      else assert.ok(toolApprovals > 0, 'Codex tool-call approval must be handled separately from TensorViz plugin permission.');
      assert.ok(capture, 'The agent itself must finish a measured capture; direct harness calls do not pass.');
      assert.ok(calls.some(item => item.tool === 'get_graph' && item.status === 'completed'), 'The agent must inspect the captured graph.');
      assert.match(answer, /\b2\b[\s,×x*]+4\b/);
      assert.match(answer, /\b2\b[\s,×x*]+3\b/);
      console.log('Agent selected the installed skill, completed capture, inspected evidence and reported [2, 4] -> [2, 3].');
    } else {
      capture = await call('open_graph', { workspaceRoot: project, file: 'model.py', cls: 'Tiny', interpreter: python });
      for (let count = 0; count < 300 && !['ready', 'failed', 'blocked', 'cancelled', 'needs_setup'].includes(capture.status); count++) {
        await new Promise(resolve => setTimeout(resolve, 100));
        capture = await call('open_graph', { requestId: capture.requestId });
      }
    }
    assert.equal(capture.status, 'ready', JSON.stringify(capture));
    assert.equal(approvals, installationConsent ? 0 : 1, 'Codex uses installation consent or asks for plugin permission once');
    const permission = JSON.parse(await fs.readFile(path.join(env.TENSORVIZ_DATA_DIR, 'permissions/standalone.json'), 'utf8'));
    assert.equal(permission.decision, 'granted', 'Consent must be saved inside this test fixture');
    const ref = { modelId: capture.models[0].modelId, revision: capture.models[0].revision };
    const graph = await call('get_graph', { ...ref, mode: 'graph', limit: 100 });
    assert.ok(graph.nodes.some(node => JSON.stringify(node.shapes).includes('3')), 'Captured Linear output shape must remain inspectable through Codex');
    assert.equal(await fs.readFile(path.join(project, 'tensorviz.toml'), 'utf8'), recipe);
    assert.equal(await fs.readFile(path.join(project, 'model.py'), 'utf8'), modelSource);
    console.log(`Captured Tiny and inspected measured shapes through Codex with ${approvals} TensorViz permission prompts.`);
    const exited = new Promise(resolve => child.once('exit', resolve));
    child.stdin.end();
    stopGroup(child.pid);
    await exited;
    child = undefined;
    cliRun(['plugin', 'remove', 'tensorviz@tensorviz-standalone']);
    assert.equal(await fs.readFile(path.join(project, 'tensorviz.toml'), 'utf8'), recipe, 'Uninstall must preserve project setup');
  } finally {
    for (const item of pending.values()) clearTimeout(item.timer);
    lines?.close(); stopGroup(child?.pid);
    await fs.rm(root, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
  }
});
