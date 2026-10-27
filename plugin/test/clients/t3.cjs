// Real T3 server -> Codex plugin -> compiled TensorViz runtime, without a browser.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const https = require('node:https');
const net = require('node:net');
const { randomUUID, createHash } = require('node:crypto');
const { spawn, execFileSync } = require('node:child_process');
const WebSocket = require('ws');
const esbuild = require('esbuild');

const repository = path.resolve(__dirname, '../../..');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function bounded(label, promise, timeout = 20000) {
  let timer;
  try { return await Promise.race([promise, new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`Timed out: ${label}`)), timeout);
  })]); } finally { clearTimeout(timer); }
}
async function eventually(label, read, timeout = 20000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) { const result = await read(); if (result) return result; await delay(100); }
  throw new Error(`Timed out: ${label}`);
}
function required(name) {
  const value = process.env[name];
  assert.ok(value && path.isAbsolute(value), `Set ${name} to an absolute local path.`);
  return value;
}
async function port() {
  const server = net.createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const result = server.address().port;
  await new Promise(resolve => server.close(resolve));
  return result;
}
async function connectRpc(url, token) {
  const socket = new WebSocket(url, { headers: { Authorization: `Bearer ${token}` }, handshakeTimeout: 10000 });
  const pending = new Map();
  let sequence = 0;
  socket.on('message', bytes => {
    const frame = JSON.parse(bytes);
    const request = pending.get(frame.requestId);
    if (!request || frame._tag !== 'Exit') return;
    pending.delete(frame.requestId); clearTimeout(request.timer);
    frame.exit._tag === 'Success' ? request.resolve(frame.exit.value)
      : request.reject(new Error(JSON.stringify(frame.exit.cause)));
  });
  await new Promise((resolve, reject) => { socket.once('open', resolve); socket.once('error', reject); });
  return {
    call(tag, payload) {
      return new Promise((resolve, reject) => {
        const id = String(++sequence);
        const timer = setTimeout(() => { pending.delete(id); reject(new Error(`${tag} timed out`)); }, 60000);
        pending.set(id, { resolve, reject, timer });
        socket.send(JSON.stringify({ _tag: 'Request', id, tag, payload, headers: [] }));
      });
    },
    close() {
      for (const request of pending.values()) { clearTimeout(request.timer); request.reject(new Error('Test connection closed')); }
      pending.clear(); socket.close();
    },
  };
}

test('T3 installs the thin plugin and preserves capture, app context and remounted panels', { timeout: 900000 }, async () => {
  const binary = required('TENSORVIZ_TEST_T3');
  const upstream = required('TENSORVIZ_TEST_T3_SOURCE');
  const effectModules = required('TENSORVIZ_TEST_T3_MODULES');
  const python = required('TENSORVIZ_TEST_PYTHON');
  const auth = required('TENSORVIZ_TEST_AGENT_AUTH_FILE');
  const permission = required('TENSORVIZ_TEST_PERMISSION_FILE');
  assert.equal(JSON.parse(await fs.readFile(permission, 'utf8')).decision, 'granted', 'Reuse an existing owner-authorized permission only');
  const revision = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: upstream, encoding: 'utf8' }).trim();
  assert.equal(revision, '511d20c6f20c2ed6ff787aba068d7d673e2949b9', 'Review changed T3 contracts before updating the pinned test');
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-t3-e2e-')));
  await fs.chmod(root, 0o700);
  const base = path.join(root, 't3'), profile = path.join(root, 'codex'), project = path.join(root, 'project');
  const data = path.join(root, 'state'), runtimeHome = path.join(root, 'runtime-home');
  const version = (await import('../../scripts/release-version.mjs')).pluginVersion();
  let downloads, rpc, child, panel;
  let downloadCount = 0;
  const cli = (command, args, options = {}) => execFileSync(command, args,
    { encoding: 'utf8', timeout: 30000, ...options });
  try {
    for (const directory of [path.join(base, 'userdata'), profile, project, path.join(data, 'permissions')])
      await fs.mkdir(directory, { recursive: true });
    await fs.chmod(path.join(data, 'permissions'), 0o700);
    await fs.copyFile(auth, path.join(profile, 'auth.json')); await fs.chmod(path.join(profile, 'auth.json'), 0o600);
    await fs.copyFile(permission, path.join(data, 'permissions/standalone.json'));
    await fs.chmod(path.join(data, 'permissions/standalone.json'), 0o600);
    const baseUrl = process.env.TENSORVIZ_TEST_OPENAI_BASE_URL;
    await fs.writeFile(path.join(profile, 'config.toml'), 'cli_auth_credentials_store = "file"\n'
      + (baseUrl ? `openai_base_url = ${JSON.stringify(baseUrl)}\n` : '') + '[features]\napps = true\n');
    await fs.writeFile(path.join(base, 'userdata/settings.json'), JSON.stringify({ providerInstances: { codex: {
      driver: 'codex', enabled: true, config: { binaryPath: process.env.TENSORVIZ_TEST_CODEX || 'codex', homePath: profile,
        shadowHomePath: '', launchArgs: '', customModels: [] },
    } } }));
    const model = activation => `from torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, 3)\n  self.activation = nn.${activation}()\n def forward(self, x):\n  return self.activation(self.fc(x))\n`;
    const recipe = 'version = 1\n[profiles.tiny]\nkind = "model"\nmodel = "model:Tiny"\nmode = "eval"\ngrad_enabled = false\ndevice = "cpu"\ninit = {args = [], kwargs = {}}\ninputs = {args = [{kind = "tensor", shape = [2, 4], dtype = "float32", requires_grad = false, generator = {kind = "full", value = 1.0}}], kwargs = {}}\n';
    await fs.writeFile(path.join(project, 'model.py'), model('ReLU'));
    await fs.writeFile(path.join(project, 'tensorviz.toml'), recipe);
    cli('git', ['init', '-q', '-b', 'main', project]);
    cli('git', ['add', '.'], { cwd: project });
    cli('git', ['-c', 'user.name=TensorViz test', '-c', 'user.email=test@localhost', 'commit', '-qm', 'fixture'], { cwd: project });

    const archive = await fs.readFile(path.join(repository, 'plugin/dist', `tensorviz-runtime-${version}-darwin-arm64.tar.gz`));
    cli('openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1', '-subj', '/CN=localhost',
      '-addext', 'subjectAltName=IP:127.0.0.1', '-keyout', path.join(root, 'test.key'), '-out', path.join(root, 'test.pem')], { stdio: 'ignore' });
    downloads = https.createServer({ key: await fs.readFile(path.join(root, 'test.key')), cert: await fs.readFile(path.join(root, 'test.pem')) }, (_req, res) => {
      downloadCount++; res.writeHead(200, { 'Content-Length': archive.length }); res.end(archive);
    });
    await new Promise(resolve => downloads.listen(0, '127.0.0.1', resolve));
    const { exportPublic } = await import('../../scripts/export-public.mjs');
    const marketplace = path.join(root, 'marketplace');
    await exportPublic(marketplace, { runtimeLock: { formatVersion: 1, version, platform: 'darwin-arm64', pythonABI: 'cpython-312-darwin',
      sha256: createHash('sha256').update(archive).digest('hex'), bytes: archive.length,
      url: `https://127.0.0.1:${downloads.address().port}/runtime.tar.gz` } });
    const mcpFile = path.join(marketplace, 'plugins/tensorviz/mcp.json');
    const mcp = JSON.parse(await fs.readFile(mcpFile, 'utf8'));
    mcp.mcpServers.tensorviz.env = { TENSORVIZ_DATA_DIR: data, HOME: runtimeHome, CURL_CA_BUNDLE: path.join(root, 'test.pem') };
    await fs.writeFile(mcpFile, JSON.stringify(mcp));
    const codex = process.env.TENSORVIZ_TEST_CODEX || 'codex';
    const env = { ...process.env, CODEX_HOME: profile };
    delete env.NODE_TEST_CONTEXT;
    cli(codex, ['plugin', 'marketplace', 'add', marketplace, '--json'], { env });
    cli(codex, ['plugin', 'add', 'tensorviz@tensorviz-standalone', '--json'], { env });
    const installed = JSON.parse(cli(codex, ['mcp', 'list', '--json'], { env }));
    assert.equal(installed.length, 1, 'T3 must load one plugin-owned TensorViz server');
    assert.equal(await fs.stat(path.join(installed[0].transport.env.PLUGIN_ROOT, 'runtime/host.cjs')).catch(() => null), null);
    console.log('Installed the source-free public plugin through Codex.');

    const serverPort = await port();
    child = spawn(binary, ['--base-dir', base, '--host', '127.0.0.1', '--port', String(serverPort), '--no-browser'],
      { env, cwd: root, detached: true, stdio: 'ignore' });
    await eventually('T3 server readiness', async () => {
      if (child.exitCode !== null) throw new Error('T3 server exited during startup');
      return fetch(`http://127.0.0.1:${serverPort}/api/health`, { signal: AbortSignal.timeout(3000) })
        .then(response => { void response.body?.cancel(); return response.status < 500; }).catch(() => false);
    });
    console.log('T3 headless server is ready.');
    const token = cli(binary, ['auth', 'session', 'issue', '--base-dir', base, '--ttl', '10m', '--token-only']).trim();
    rpc = await connectRpc(`ws://127.0.0.1:${serverPort}/ws?orchestrationProtocol=2`, token);
    console.log('Authenticated T3 RPC connection established.');
    const configuration = await rpc.call('server.getConfig', {});
    assert.match(configuration.environment.serverVersion, /^0\.0\.46-nightly/);
    const projectId = randomUUID(), threadId = randomUUID();
    const command = payload => rpc.call('orchestration.dispatchCommand', { commandId: randomUUID(), threadId, ...payload });
    await rpc.call('projects.mutate', { type: 'project.create', commandId: randomUUID(), projectId, title: 'TensorViz automated acceptance', workspaceRoot: project });
    await command({ type: 'thread.create', createdBy: 'user', creationSource: 'web', projectId, title: 'TensorViz acceptance',
      modelSelection: { instanceId: 'codex', model: process.env.TENSORVIZ_TEST_T3_MODEL || 'gpt-6-astra' },
      runtimeMode: 'full-access', interactionMode: 'default', branch: 'main', worktreePath: null });
    await command({ type: 'message.dispatch', createdBy: 'user', creationSource: 'web', messageId: randomUUID(), attachments: [],
      dispatchMode: { type: 'start_immediately' },
      text: `Validate the installed TensorViz model-workflow skill on this disposable project. Read its skill, open model.py class Tiny using saved profile tiny and interpreter ${python}, wait for capture completion, then call open_graph_panel once. The plugin may download its compiled runtime and provision its private service environment using the existing owner-granted permission; continue polling through that normal setup. There is no renderer yet, so report capture separately from display. Do not edit project files, manually install packages or modify the supplied interpreter, change permissions, use browsers or other agents, or access other projects.` });
    const projection = await eventually('T3 agent opens the Model graph MCP App', async () => {
      const value = await rpc.call('orchestration.getThreadProjection', { threadId });
      const requests = value.runtimeRequests.filter(request => request.status === 'pending');
      assert.equal(requests.length, 0, 'Unexpected approval: this fixture reuses explicit saved TensorViz permission');
      assert.ok(!value.runs.some(run => run.status === 'failed'), 'T3 provider turn failed');
      return value.runs.length && value.runs.every(run => run.status === 'completed')
        && value.turnItems.some(item => item.toolName === 'tensorviz.open_graph_panel' && item.status === 'completed') ? value : false;
    }, 600000); // A clean service environment downloads the CPU PyTorch wheel.
    const itemId = projection.turnItems.find(item => item.toolName === 'tensorviz.open_graph_panel').id;
    const { item } = await rpc.call('orchestration.getTurnItem', { threadId, itemId });
    const appRef = item.output.t3McpApp, opener = item.output.result;
    assert.ok(appRef?.attachmentId, 'T3 must recognize and snapshot the actual plugin resource');
    const html = await fs.readFile(path.join(base, 'userdata/attachments', appRef.attachmentId + '.html'), 'utf8');
    assert.match(html, /Content-Security-Policy/); assert.match(html, /connect-src 'none'/);
    assert.ok(Buffer.byteLength(html) <= 5 * 1024 * 1024);
    assert.equal(downloadCount, 1, 'The installed thin plugin must download its compiled runtime once');
    const scope = { threadId, itemId };
    const tool = async (name, args = {}) => {
      const result = await rpc.call('mcpApps.callTool', { ...scope, name, arguments: args });
      assert.ok(!result.isError, JSON.stringify(result.structuredContent)); return result.structuredContent;
    };
    const initial = (await tool('list_models')).models[0];
    assert.ok(initial?.current, 'The real agent must open a current model');
    assert.equal((await tool('list_captures', { modelId: initial.modelId })).latestAttempt?.status, 'succeeded',
      'Successful capture must be distinguished from an unexecuted source projection');
    console.log(`T3 ${configuration.environment.serverVersion} captured Tiny through the compiled plugin and stored its app resource.`);

    // Execute T3's actual shared bridge and the actual MCP Apps SDK/Canvas transport.
    // Only the postMessage delivery mechanism is an in-process wire; no browser is used.
    const build = async (entry, filename, options = {}) => {
      const outfile = path.join(root, filename);
      await esbuild.build({ entryPoints: [entry], outfile, platform: 'node', format: 'cjs', bundle: true, logLevel: 'silent', ...options });
      return require(outfile);
    };
    const { makeMcpAppHost, McpAppHostRefusal } = await build(path.join(upstream, 'packages/client-runtime/src/mcpApps/host.ts'), 't3-host.cjs', {
      alias: { '@t3tools/shared/mcpApp': path.join(upstream, 'packages/shared/src/mcpApp.ts') }, nodePaths: [effectModules],
    });
    const { createPanelTransport } = await build(path.join(repository, 'canvas/src/plugin/panelTransport.ts'), 'panel-transport.cjs');
    const { App } = await import('@modelcontextprotocol/ext-apps');
    let approvalCount = 0, allowAction = true, contextCount = 0;
    async function mount() {
      const app = new App({ name: 'TensorViz acceptance', version }, {}, { autoResize: false });
      const events = [], notices = [];
      let host, onmessage;
      const transport = { start: async () => {}, send: async message => host.receive(message), close: async () => transport.onclose?.(),
        set onmessage(value) { onmessage = value; }, get onmessage() { return onmessage; } };
      host = makeMcpAppHost({ app: appRef, hostVersion: configuration.environment.serverVersion,
        post: message => queueMicrotask(() => onmessage?.(message)),
        hostContext: () => ({ theme: 'dark', styles: { variables: {} }, displayMode: 'inline',
          availableDisplayModes: ['inline', 'fullscreen'], containerDimensions: { width: 1100, maxHeight: 2000 }, platform: 'web' }),
        callTool: async input => {
          const info = await rpc.call('mcpApps.toolInfo', { ...scope, name: input.name });
          assert.ok(info.callable, 'T3 must allow the app to call this tool');
          if (!info.readOnly) { approvalCount++; if (!allowAction) throw new McpAppHostRefusal('Declined by the user.'); }
          return rpc.call('mcpApps.callTool', { ...scope, ...input });
        },
        readResource: input => rpc.call('mcpApps.readResource', { ...scope, ...input }),
        updateModelContext: async input => { await rpc.call('mcpApps.updateModelContext', { ...scope, conversationThreadId: threadId, ...input }); contextCount++; },
        requestDisplayMode: async mode => mode,
        openLink: async () => assert.fail('No external links'), sendMessage: async () => assert.fail('No extra agent turns'),
        downloadFile: async () => assert.fail('No downloads from the app'), onRequestTeardown() {}, onSizeChanged() {},
      });
      const received = new Promise(resolve => { app.ontoolresult = resolve; });
      host.setToolCall({ arguments: {}, result: opener });
      await bounded('MCP App initialization', app.connect(transport));
      assert.equal((await bounded('original tool result', received))._meta.panelId, opener._meta.panelId);
      assert.equal((await app.requestDisplayMode({ mode: 'fullscreen' })).mode, 'fullscreen');
      const view = createPanelTransport(app, opener._meta.panelId, () => {}, value => notices.push(value));
      app.onteardown = async () => { await view.detach(); return {}; };
      panel = { app, host, view, events, notices, async unmount() { await host.teardown(); await app.close(); } };
      view.onMessage(message => events.push(message));
      await bounded('Canvas transport connection', view.connect());
      await eventually('Canvas receives its snapshot', () => events.find(event => event.type === 'snapshot' && event.state.projection));
      return panel;
    }
    panel = await mount();
    const snapshot = panel.events.find(event => event.type === 'snapshot');
    const models = await tool('list_models'), reference = models.models[0];
    const captured = await tool('list_captures', { modelId: reference.modelId });
    const graph = await tool('get_graph', { modelId: reference.modelId, revision: reference.revision, mode: 'graph', limit: 100 });
    assert.ok(graph.nodes.some(node => node.shapes.output?.join(',') === '2,3'));
    const viewId = panel.view.session.viewId, target = snapshot.state.activeTarget;
    panel.view.send({ type: 'viewSelection', viewId, target, nodeIds: [graph.nodes[0].nodeId], edgeIds: [] });
    await eventually('selection and context reach T3', async () => contextCount && (await tool('list_models')).views.some(view => view.viewId === viewId && view.current));
    assert.deepEqual((await tool('get_selection', { modelId: reference.modelId, revision: reference.revision, viewId })).nodeIds, [graph.nodes[0].nodeId]);
    assert.equal(approvalCount, 0, 'Polling and selection must not trigger execution approvals');
    console.log('T3 bridge initialization, fullscreen negotiation, current-view receipt and selection context passed.');

    const capture = captured.captures.find(value => value.id === captured.displayedCaptureId);
    const inspectId = randomUUID();
    panel.view.send({ type: 'captureInspectRequest', target, requestId: inspectId,
      query: { captureId: capture.id, sourceId: capture.sourceId, path: [], offset: 0, limit: 20 } });
    const detail = await eventually('captured details reach the app', () => panel.events.find(event => event.type === 'captureInspectResult' && event.requestId === inspectId));
    assert.equal(detail.page.captureId, capture.id); assert.ok(detail.page.items.length);
    const focus = tool('focus_view', { modelId: reference.modelId, revision: reference.revision, viewId, nodeIds: [graph.nodes[0].nodeId] });
    const focusRequest = await eventually('agent focus reaches the app', () => panel.events.find(event => event.type === 'focusView'));
    panel.view.send({ type: 'viewFocusResult', viewId, target, requestId: focusRequest.requestId, ok: true, detail: 'Protocol test acknowledged focus.' });
    assert.equal((await focus).focused, true);
    assert.equal(approvalCount, 0, 'Inspection and focus acknowledgements are read-only');
    await assert.rejects(rpc.call('mcpApps.callTool', { threadId, itemId: randomUUID(), name: 'list_models', arguments: {} }), /not-an-app/);

    const contextMessageId = randomUUID();
    await command({ type: 'message.dispatch', createdBy: 'user', creationSource: 'web', messageId: contextMessageId, attachments: [],
      dispatchMode: { type: 'start_immediately' },
      text: 'From the Model graph app context delivered with this turn, reply with the exact selected viewId and nodeIds. Do not call tools or read files.' });
    const continued = await eventually('T3 forwards app selection into the next agent turn', async () => {
      const value = await rpc.call('orchestration.getThreadProjection', { threadId });
      return value.runs.length === 2 && value.runs.every(run => run.status === 'completed') ? value : false;
    }, 120000);
    assert.ok(continued.messages.some(message => message.role === 'assistant' && message.text.includes(viewId)
      && message.text.includes(graph.nodes[0].nodeId)), 'The real agent must receive the exact app selection');
    console.log('Captured-detail inspection, agent focus, app ownership and next-turn selection context passed.');

    const opening = () => panel.events.filter(event => event.type === 'graphOpening').at(-1)?.state
      ?? panel.events.find(event => event.type === 'snapshot').state.opening;
    const refresh = () => panel.view.send({ type: 'graphSetupRequest', target: opening().target,
      openingId: opening().id, requestId: randomUUID(), action: { kind: 'retry' } });
    allowAction = false; refresh();
    await eventually('T3 declined action remains visible', () => panel.notices.includes('Action cancelled. The current graph is still available.'));
    assert.equal(panel.view.status.phase, 'connected');
    assert.deepEqual(await tool('list_captures', { modelId: reference.modelId }), captured);
    assert.equal(approvalCount, 1);
    allowAction = true;
    await fs.writeFile(path.join(project, 'model.py'), model('Sigmoid'));
    await eventually('source change invalidates current evidence', async () => !(await tool('list_models')).models[0].current);
    const stale = await rpc.call('mcpApps.callTool', { ...scope, name: 'get_graph', arguments: { modelId: reference.modelId, revision: reference.revision } });
    assert.equal(stale.structuredContent.error.code, 'STALE_TARGET');
    refresh();
    const refreshed = await eventually('approved refresh captures changed source', async () => {
      const value = await tool('list_models');
      return value.models[0].current && value.models[0].revision !== reference.revision ? value.models[0] : false;
    }, 60000);
    const currentGraph = await tool('get_graph', { modelId: refreshed.modelId, revision: refreshed.revision, mode: 'graph', limit: 100 });
    assert.match(JSON.stringify(currentGraph), /sigmoid/i);
    assert.equal(approvalCount, 2); assert.equal(panel.view.session.viewId, viewId);
    await eventually('updated setup state reaches the frame', () => opening().phase === 'ready' && opening().target.sourceDigest !== target.sourceDigest);
    console.log('Declined refresh preserved the graph; approved refresh captured edited source in the same frame.');

    await fs.writeFile(path.join(project, 'model.py'), model('Sigmoid').replace('nn.Linear(4, 3)', 'nn.Linear(5, 3)'));
    refresh();
    await eventually('shape failure retains useful capture evidence', async () => {
      const state = await tool('list_captures', { modelId: reference.modelId });
      return state.latestAttempt.status === 'failed' && state.lastPartialCaptureId;
    }, 60000);
    assert.equal(panel.view.status.phase, 'connected');
    await eventually('failure recovery controls reach the frame', () => opening().phase === 'failed');
    await fs.writeFile(path.join(project, 'model.py'), model('ReLU'));
    refresh();
    await eventually('repair captures original model and inputs', async () => {
      const state = await tool('list_captures', { modelId: reference.modelId });
      return state.latestAttempt.status === 'succeeded' && state.current;
    }, 60000);
    assert.equal(approvalCount, 4);
    console.log('A real failing forward retained partial evidence; repairing source recovered without changing inputs.');

    const validate = async batch => {
      const current = (await tool('list_models')).models[0];
      const run = await tool('validate_model', { modelId: current.modelId, revision: current.revision,
        execution: { inputs: { args: [{ shape: [batch, 4], dtype: 'float32', fill: 'ones' }] } } });
      const finished = await eventually('explicit validation completes without disconnecting the app', async () => {
        const value = await tool('get_run', { runId: run.runId }); return value.finishedAt ? value : false;
      }, 60000);
      assert.equal(finished.status, 'passed', JSON.stringify(finished));
      assert.equal(finished.attempt.status, 'succeeded');
      return finished;
    };
    const firstRun = await validate(3), secondRun = await validate(2);
    const evidence = await tool('inspect_run', { runId: firstRun.runId, section: 'graph', limit: 100 });
    assert.ok(evidence.nodes.items.some(node => node.shapes.output?.join(',') === '3,3'));
    assert.equal((await tool('compare_runs', { baselineRunId: firstRun.runId, candidateRunId: secondRun.runId })).semanticComparisonAvailable, true);
    assert.equal(panel.view.status.phase, 'connected');
    console.log('Explicit validation, immutable run inspection and comparison remained connected through T3.');

    const retained = await tool('list_captures', { modelId: reference.modelId });
    await panel.unmount(); panel = await mount();
    assert.notEqual(panel.view.session.viewId, viewId);
    assert.deepEqual(await tool('list_captures', { modelId: reference.modelId }), retained, 'Remount must retain evidence without recapture');
    const restoredView = panel.view.session.viewId;
    const restoredTarget = panel.events.find(event => event.type === 'snapshot').state.activeTarget;
    panel.view.send({ type: 'viewSelection', viewId: restoredView, target: restoredTarget, nodeIds: [], edgeIds: [] });
    await eventually('remounted frame has a current receipt', async () => {
      const value = await tool('list_models');
      return !value.views.some(view => view.viewId === viewId) && value.views.some(view => view.viewId === restoredView && view.current);
    });
    assert.equal(approvalCount, 4, 'teardown and remount require no execution approval');
    console.log('T3 teardown/remount retained the capture and established a new view.');
    assert.equal(await fs.readFile(path.join(project, 'tensorviz.toml'), 'utf8'), recipe);
    assert.equal(await fs.readFile(path.join(project, 'model.py'), 'utf8'), model('ReLU'));
    // End only this fixture's actual provider process. Detaching a thread alone
    // can leave T3's shared Codex session alive, which legitimately keeps its app usable.
    const processes = cli('ps', ['-axo', 'pid=,ppid=,command=']).trim().split('\n').map(line => {
      const [, pid, parent, command] = line.match(/^\s*(\d+)\s+(\d+)\s+(.+)$/);
      return { pid: Number(pid), parent: Number(parent), command };
    });
    const owned = new Set([child.pid]);
    for (let previous = -1; previous !== owned.size;) {
      previous = owned.size;
      for (const process of processes) if (owned.has(process.parent)) owned.add(process.pid);
    }
    const providers = processes.filter(process => owned.has(process.pid) && /\bapp-server(?:\s|$)/.test(process.command));
    assert.equal(providers.length, 1, 'Only this disposable T3 server may own the provider being stopped');
    process.kill(providers[0].pid, 'SIGTERM');
    await eventually('stopped provider session ends the panel', () => panel.view.status.phase === 'ended');
    await assert.rejects(rpc.call('mcpApps.callTool', { ...scope, name: 'list_models', arguments: {} }), /session-stopped|request-failed/);
    assert.equal((await rpc.call('orchestration.getThreadProjection', { threadId })).runs.length, 2, 'App polling cannot restart an agent turn');
    await panel.unmount(); panel = undefined;
    cli(codex, ['plugin', 'remove', 'tensorviz@tensorviz-standalone'], { env });
    assert.equal(JSON.parse(cli(codex, ['mcp', 'list', '--json'], { env })).length, 0);
    console.log('Stopped sessions fail visibly without starting a turn; uninstall preserved the model and saved inputs.');
  } finally {
    await panel?.unmount().catch(() => {});
    rpc?.close(); downloads?.closeAllConnections(); downloads?.close();
    if (child?.pid) { try { process.kill(-child.pid, 'SIGTERM'); } catch {} }
    if (process.env.TENSORVIZ_TEST_KEEP_T3 === '1') console.log(`T3 test files retained at ${root}`);
    else await fs.rm(root, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
  }
});
