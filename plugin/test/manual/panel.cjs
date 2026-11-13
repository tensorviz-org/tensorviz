// Real MCP Apps bridge + disposable model for Computer/browser acceptance.
// This harness is not evidence of native Codex rendering. No browser automation.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { randomUUID } = require('node:crypto');
const { createServer } = require('node:http');
const readline = require('node:readline');
const { build } = require('esbuild');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { ElicitRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const { readyService } = require('../fixtures/service.cjs');

(async () => {
  const python = process.env.TENSORVIZ_TEST_PYTHON;
  assert.ok(python && path.isAbsolute(python), 'Set an explicit compatible test interpreter.');
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-panel-acceptance-')));
  const project = path.join(root, 'project');
  const native = process.env.TENSORVIZ_TEST_RUNTIME;
  const plugin = native ?? process.env.TENSORVIZ_TEST_PLUGIN ?? path.resolve(__dirname, '../../../plugins/tensorviz');
  const token = randomUUID(), base = `/${token}`;
  await fs.mkdir(project);
  const source = activation => `from torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, 3)\n  self.activation = nn.${activation}()\n def forward(self, x):\n  return self.activation(self.fc(x))\n`;
  await fs.writeFile(path.join(project, 'model.py'), source('ReLU'));
  await fs.writeFile(path.join(project, 'tensorviz.toml'), `version = 1\n[profiles.tiny]\nkind = "model"\nmodel = "model:Tiny"\nmode = "eval"\ndevice = "cpu"\ngrad_enabled = false\ninit = { args = [], kwargs = {} }\ninputs = { args = [{ kind = "tensor", shape = [2, 4], dtype = "float32", requires_grad = false, generator = { kind = "full", value = 1 } }], kwargs = {} }\n`);
  const client = new Client({ name: 'synthetic-panel-acceptance', version: '1' }, { capabilities: { elicitation: { form: {} } } });
  client.setRequestHandler(ElicitRequestSchema, async request => {
    assert.ok(request.params.message.includes(project) && request.params.message.includes(python));
    return { action: 'accept', content: { confirm: true } };
  });
  if (process.env.TENSORVIZ_TEST_SERVICE_PYTHON) await readyService(plugin, path.join(root, 'state'));
  await client.connect(new StdioClientTransport({ command: native ? path.join(plugin, 'tensorviz-host') : process.execPath,
    args: native ? [] : [path.join(plugin, 'runtime/host.cjs')], cwd: root,
    env: { PATH: '', TENSORVIZ_DATA_DIR: path.join(root, 'state') }, stderr: 'inherit' }));
  const call = async (name, args = {}) => {
    const result = await client.callTool({ name, arguments: args });
    assert.ok(!result.isError, JSON.stringify(result)); return result.structuredContent;
  };
  let ref, context;
  async function capture() {
    let result = await call('open_graph', { workspaceRoot: project, file: 'model.py', cls: 'Tiny', interpreter: python });
    for (let n = 0; n < 400 && !['ready', 'failed', 'blocked', 'cancelled', 'needs_setup'].includes(result.status); n++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      result = await call('open_graph', { requestId: result.requestId });
    }
    assert.equal(result.status, 'ready', JSON.stringify(result));
    ref = { modelId: result.models[0].modelId, revision: result.models[0].revision };
    return ref;
  }
  const opener = (await client.listTools()).tools.find(tool => tool.name === 'open_graph_panel');
  const resource = (await client.readResource({ uri: opener._meta.ui.resourceUri })).contents[0];
  assert.equal(resource.mimeType, 'text/html;profile=mcp-app');
  const script = await build({ stdin: { contents: `
    import { AppBridge, PostMessageTransport } from '@modelcontextprotocol/ext-apps/app-bridge';
    const frame = document.querySelector('iframe');
    const api = (route, args = {}) => fetch('${base}/' + route, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(args) }).then(r => r.json());
    const bridge = new AppBridge(null, { name: 'TensorViz acceptance harness', version: '1' }, { serverTools: {}, updateModelContext: {} },
      { hostContext: { theme: 'dark', displayMode: 'fullscreen', availableDisplayModes: ['fullscreen'] } });
    bridge.oncalltool = args => api('tool', args);
    bridge.onupdatemodelcontext = async args => {
      const value = args.structuredContent?.tensorviz;
      document.querySelector('#context').textContent = JSON.stringify(value && { viewId: value.viewId, nodeIds: value.nodeIds, edgeIds: value.edgeIds, tooLarge: value.tooLarge });
      await api('context', args); return {};
    };
    bridge.oninitialized = async () => { await bridge.sendToolInput({ arguments: {} }); await bridge.sendToolResult(await api('open')); };
    bridge.connect(new PostMessageTransport(frame.contentWindow, frame.contentWindow)).then(() => { frame.src = '${base}/panel'; });
  `, resolveDir: path.resolve(__dirname, '../..') }, bundle: true, platform: 'browser', format: 'iife', write: false });
  let origin;
  const server = createServer(async (req, res) => {
    try {
      if (req.headers.host !== new URL(origin).host || !req.url.startsWith(`${base}/`)) { res.writeHead(403).end(); return; }
      if (req.method === 'GET') {
        if (req.url === `${base}/`) { res.setHeader('Content-Type', 'text/html'); res.end(`<!doctype html><meta charset="utf-8"><title>TensorViz MCP Apps acceptance</title><style>body{margin:0;background:#10141c;color:#cbd5e1;font:13px system-ui}header{padding:8px}iframe{width:100%;height:calc(100vh - 76px);border:0}#context{max-height:30px;overflow:auto}</style><header>MCP Apps bridge harness · Native Codex acceptance remains pending<div id="context">No selection context yet</div></header><iframe title="Model graph" sandbox="allow-scripts allow-downloads"></iframe><script src="${base}/bridge.js"></script>`); return; }
        if (req.url === `${base}/bridge.js`) { res.setHeader('Content-Type', 'text/javascript'); res.end(script.outputFiles[0].text); return; }
        if (req.url === `${base}/panel`) {
          res.setHeader('Content-Type', resource.mimeType);
          res.setHeader('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src data:; worker-src blob:");
          res.end(resource.text); return;
        }
        res.writeHead(404).end(); return;
      }
      if (req.method !== 'POST' || req.headers.origin !== origin) { res.writeHead(403).end(); return; }
      let raw = '';
      for await (const chunk of req) { raw += chunk; if (raw.length > 150000) throw new Error('Request too large'); }
      const args = JSON.parse(raw), route = req.url.slice(base.length + 1);
      let result;
      if (route === 'open') result = await client.callTool({ name: 'open_graph_panel', arguments: {} });
      else if (route === 'context') { context = args; result = {}; }
      else if (route === 'tool' && ['graph_panel_read', 'graph_panel_exchange', 'detach_graph_panel', 'close_graph_panel'].includes(args.name)) result = await client.callTool(args);
      else { res.writeHead(403).end(); return; }
      res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(result));
    } catch (error) { res.writeHead(500).end(JSON.stringify({ error: error.message })); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  origin = `http://127.0.0.1:${server.address().port}`;
  const handoffFile = process.env.TENSORVIZ_TEST_HANDOFF_FILE;
  if (handoffFile) {
    assert.ok(path.isAbsolute(handoffFile), 'Use an absolute task-owned handoff file.');
    await fs.writeFile(handoffFile, JSON.stringify({ url: `${origin}${base}/` }), { mode: 0o600 });
    console.log(JSON.stringify({ event: 'ready', handoff: 'written privately', scope: 'synthetic MCP Apps bridge' }));
  } else console.log(JSON.stringify({ url: `${origin}${base}/`, root, event: 'ready' }));
  const input = readline.createInterface({ input: process.stdin });
  try {
    for await (const line of input) {
      try {
        const command = JSON.parse(line);
        if (command.action === 'stop') break;
        let result;
        if (command.action === 'capture') result = await capture();
        else if (command.action === 'change') { await fs.writeFile(path.join(project, 'model.py'), source('Sigmoid')); result = await call('list_models'); }
        else if (command.action === 'refresh') result = await capture();
        else if (command.action === 'status') result = { ...(await call('list_models')), context };
        else if (command.action === 'graph') result = await call('get_graph', { ...ref, mode: 'graph', limit: 100 });
        else if (command.action === 'focus') result = await call('focus_view', { ...ref, viewId: command.viewId, nodeIds: [command.nodeId] });
        else if (command.action === 'selection') result = await call('get_selection', { ...ref, viewId: command.viewId });
        else throw new Error('Unknown command');
        console.log(JSON.stringify({ event: command.action, result }).replaceAll(root, '$FIXTURE_ROOT'));
      } catch (error) { console.log(JSON.stringify({ event: 'error', message: error.message })); }
    }
  } finally { input.close(); process.stdin.pause(); server.close(); server.closeAllConnections(); await client.close(); await fs.rm(root, { recursive: true, force: true }); }
})().catch(error => { console.error(error); process.exitCode = 1; });
