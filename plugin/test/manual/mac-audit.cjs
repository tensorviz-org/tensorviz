// Reproducible native-runtime checks. Synthetic elicitation is limited to this
// disposable fixture; this is not a signed-in desktop or model-provider test.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { ElicitRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const { readyService } = require('../fixtures/service.cjs');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const source = (width = 3, body = 'return self.activation(self.fc(x))', activation = 'ReLU') => `import time\nfrom torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, ${width})\n  nn.init.constant_(self.fc.weight, 0.25)\n  nn.init.constant_(self.fc.bias, 0.5)\n  self.activation = nn.${activation}()\n def forward(self, x):\n  ${body}\n`;
const recipe = width => `version=1\n[profiles.tiny]\nkind="model"\nmodel="model:Tiny"\nmode="eval"\ngrad_enabled=false\ndevice="cpu"\ninit={args=[],kwargs={}}\ninputs={args=[{kind="tensor",shape=[2,${width}],dtype="float32",requires_grad=false,generator={kind="full",value=1.0}}],kwargs={}}\n`;
(async () => {
  const runtime = process.env.TENSORVIZ_TEST_RUNTIME, python = process.env.TENSORVIZ_TEST_PYTHON;
  const parent = process.env.TENSORVIZ_AUDIT_ROOT;
  assert.ok([runtime, python, parent].every(p => p && path.isAbsolute(p)), 'Set absolute runtime, Python and task-owned TENSORVIZ_AUDIT_ROOT paths.');
  await fs.mkdir(parent, { recursive: true });
  const root = await fs.mkdtemp(path.join(parent, 'native-audit-'));
  const project = path.join(root, 'project'), state = path.join(root, 'state');
  await fs.mkdir(project);
  if (process.env.TENSORVIZ_TEST_SERVICE_PYTHON) await readyService(runtime, state);
  const report = { classification: 'native MCP automation; synthetic fixture consent; no signed-in GUI', checks: [] };
  let client;
  const write = async (text = source(), inputWidth = 4) => {
    await fs.writeFile(path.join(project, 'model.py'), text);
    await fs.writeFile(path.join(project, 'tensorviz.toml'), recipe(inputWidth));
  };
  const connect = async () => {
    client = new Client({ name: 'tensorviz-native-mac-audit', version: '1' }, { capabilities: { elicitation: { form: {} } } });
    client.setRequestHandler(ElicitRequestSchema, async request => {
      assert.ok(request.params.message.includes(project), 'never accept permission for another workspace');
      return { action: 'accept', content: { confirm: true } };
    });
    await client.connect(new StdioClientTransport({ command: path.join(runtime, 'tensorviz-host'),
      env: { PATH: process.env.PATH, TENSORVIZ_DATA_DIR: state, PIP_CACHE_DIR: path.join(root, 'pip-cache') }, stderr: 'pipe' }));
  };
  const raw = (name, args = {}) => client.callTool({ name, arguments: args }, undefined, { timeout: 120000 });
  const call = async (name, args = {}) => {
    const r = await raw(name, args);
    if (r.isError) report.lastToolError = { tool: name, code: r.structuredContent?.error?.code };
    assert.ok(!r.isError, `unexpected ${name} error`); return r.structuredContent;
  };
  const open = (extra = {}) => call('open_graph', { workspaceRoot: project, file: 'model.py', cls: 'Tiny', interpreter: python, waitMs: 0, ...extra });
  const poll = async initial => {
    let r = initial;
    for (let n = 0; n < 300 && !['ready', 'failed', 'blocked', 'cancelled', 'needs_setup', 'source_changed'].includes(r.status); n++) {
      await delay(100); r = await call('open_graph', { requestId: r.requestId, waitMs: 1000 });
    }
    return r;
  };
  const record = (name, r) => { const value = { name, status: r.status, validated: r.validated, error: r.error?.code, nextAction: r.nextAction?.kind };
    report.checks.push(value); console.log(JSON.stringify(value)); };
  try {
    await write(); await connect(); report.runtime = client.getServerVersion();
    const initial = await poll(await open()); assert.equal(initial.status, 'ready'); record('first-capture', initial);
    const old = { modelId: initial.models[0].modelId, revision: initial.models[0].revision };
    const before = await call('get_model_context', old), fc = before.modules.find(m => m.paths.includes('fc'));
    assert.deepEqual(fc.calls[0].inputs.tensors[0].shape, [2, 4]);
    assert.deepEqual(fc.calls[0].outputs.tensors[0].shape, [2, 3]); assert.equal(fc.calls[0].parameterCount, 15);
    let firstPanel, panelErrors = 0;
    for (let n = 0; n < 16; n++) { const r = await raw('open_graph_panel'); if (r.isError) panelErrors++; else firstPanel ??= r._meta.panelId; }
    report.unmountedPanels = { requests: 16, errors: panelErrors };
    await write(source(2, undefined, 'Sigmoid'));
    const changed = await poll(await open()); assert.equal(changed.status, 'ready'); record('changed-capture', changed);
    assert.notEqual(changed.models[0].revision, old.revision);
    const after = await call('get_model_context', { modelId: changed.models[0].modelId, revision: changed.models[0].revision });
    const fcAfter = after.modules.find(m => m.paths.includes('fc'));
    assert.deepEqual(fcAfter.calls[0].outputs.tensors[0].shape, [2, 2]); assert.equal(fcAfter.calls[0].parameterCount, 10);
    report.shapes = { before: [2, 3], after: [2, 2], input: [2, 4] };
    await write(source(), 5);
    const mismatch = await poll(await open()); record('shape-mismatch', mismatch); assert.notEqual(mismatch.status, 'ready');
    assert.ok(mismatch.error || mismatch.nextAction, 'failure needs actionable recovery');
    await write(source(3, "raise RuntimeError('synthetic model failure')"));
    const exception = await poll(await open()); record('model-exception', exception); assert.notEqual(exception.status, 'ready');
    assert.ok(exception.error || exception.nextAction);
    await write(); const recovered = await poll(await open()); assert.equal(recovered.status, 'ready'); record('recovery', recovered);
    await write(source(3, 'time.sleep(20); return self.fc(x)'));
    const pending = await open();
    const repeatedRaw = await raw('open_graph', { workspaceRoot: project, file: 'model.py', cls: 'Tiny', interpreter: python, waitMs: 0 });
    report.repeatedCapture = { isError: repeatedRaw.isError === true, code: repeatedRaw.structuredContent?.error?.code,
      sameRequest: repeatedRaw.structuredContent?.requestId === pending.requestId };
    assert.ok(report.repeatedCapture.sameRequest || report.repeatedCapture.code === 'OPENING_BUSY');
    const cancelled = await call('open_graph', { requestId: pending.requestId, action: 'cancel' });
    assert.equal(cancelled.status, 'cancelled'); record('cancel', cancelled);
    await write(); await client.close(); await connect();
    const restarted = await poll(await open()); assert.equal(restarted.status, 'ready'); record('restart-reconnect', restarted);
    const stale = await raw('open_graph', { requestId: pending.requestId });
    report.staleRequest = { isError: stale.isError === true, code: stale.structuredContent?.error?.code };
    assert.equal(report.staleRequest.code, 'UNKNOWN_REQUEST');
    const wrongPython = process.env.TENSORVIZ_AUDIT_INCOMPATIBLE_PYTHON;
    if (wrongPython) { const r = await poll(await open({ interpreter: wrongPython })); record('incompatible-python', r); assert.notEqual(r.status, 'ready'); }
    // A required constructor value must be requested instead of invented.
    await fs.rm(path.join(project, 'tensorviz.toml'));
    await fs.writeFile(path.join(project, 'model.py'), source().replace('def __init__(self):', 'def __init__(self, width, variant: str):').replace('nn.Linear(4, 3)', 'nn.Linear(width, 3)'));
    const missing = await poll(await open()); record('missing-constructor-input', missing); assert.equal(missing.status, 'needs_setup');
    report.passed = true;
  } catch (error) { report.passed = false; report.failure = error.code || error.name; throw error; }
  finally { await client?.close(); await fs.writeFile(path.join(root, 'report.json'), JSON.stringify(report, null, 2) + '\n'); console.log(JSON.stringify({ report: 'report.json inside task-owned native-audit directory', passed: report.passed })); }
})().catch(error => { console.error(error.name + ': ' + (error.code || 'audit assertion failed; inspect the task-owned report')); process.exitCode = 1; });
