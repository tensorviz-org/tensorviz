const { test } = require('node:test');
const assert = require('node:assert/strict');
const { mkdtemp, mkdir, cp, writeFile, readFile, rm, realpath } = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { gunzipSync } = require('node:zlib');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { ElicitRequestSchema } = require('@modelcontextprotocol/sdk/types.js');

const python = process.env.TENSORVIZ_TEST_PYTHON;
assert.ok(python && path.isAbsolute(python), 'Set TENSORVIZ_TEST_PYTHON to a compatible Python 3.11+ environment.');
const recipe = `version = 1\n[profiles.tiny]\nkind = "model"\nmodel = "model:Tiny"\nmode = "eval"\ngrad_enabled = false\ndevice = "cpu"\ninit = { args = [], kwargs = {} }\ninputs = { args = [{kind = "tensor", shape = [2, 4], dtype = "float32", requires_grad = false, generator = {kind = "full", value = 1.0}}], kwargs = {} }\n`;
const model = activation => `from torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, 3)\n  self.activation = nn.${activation}()\n def forward(self, x):\n  return self.activation(self.fc(x))\n`;

async function poll(client, initial) {
  let result = initial;
  for (let count = 0; count < 450 && !['ready', 'failed', 'blocked', 'cancelled', 'needs_setup'].includes(result.status); count++) {
    await new Promise(resolve => setTimeout(resolve, 100));
    result = (await client.callTool({ name: 'open_graph', arguments: { requestId: result.requestId } })).structuredContent;
  }
  return result;
}

test('copied plugin captures and connects the Canvas bridge with exact selection, focus and refresh receipts', { timeout: 120000 }, async () => {
  const root = await realpath(await mkdtemp(path.join(os.tmpdir(), 'tensorviz-installed-capture-')));
  const project = path.join(root, 'project with spaces'), plugin = path.join(root, 'installed plugin');
  await mkdir(project);
  await cp(process.env.TENSORVIZ_TEST_RUNTIME ?? path.resolve(__dirname, '../../../plugins/tensorviz'), plugin, { recursive: true });
  await writeFile(path.join(project, 'model.py'), model('ReLU'));
  await writeFile(path.join(project, 'tensorviz.toml'), recipe);
  const transport = new StdioClientTransport({ command: process.env.TENSORVIZ_TEST_RUNTIME ? path.join(plugin, 'tensorviz-host') : process.execPath,
    args: process.env.TENSORVIZ_TEST_RUNTIME ? [] : [path.join(plugin, 'runtime/host.cjs')], cwd: root,
    env: { PATH: '', TENSORVIZ_DATA_DIR: path.join(root, 'state') }, stderr: 'pipe' });
  let stderr = '', prompts = [];
  transport.stderr?.on('data', value => { stderr += value; });
  const client = new Client({ name: 'approved-integration-test', version: '1' }, { capabilities: { elicitation: { form: {} } } });
  client.setRequestHandler(ElicitRequestSchema, async request => {
    prompts.push(request.params.message);
    assert.ok(request.params.message.includes(project));
    assert.ok(request.params.message.includes(python));
    return { action: 'accept', content: { confirm: true } };
  });
  const call = async (name, args) => {
    const result = await client.callTool({ name, arguments: args });
    assert.ok(!result.isError, JSON.stringify(result.structuredContent));
    return result.structuredContent;
  };
  try {
    await client.connect(transport);
    const opener = await client.callTool({ name: 'open_graph_panel', arguments: {} });
    const panelId = opener._meta.panelId;
    let after = 0, session;
    const exchange = async (messages = []) => {
      const name = messages.some(message => message.type === 'graphSetupRequest') ? 'graph_panel_exchange' : 'graph_panel_read';
      const reply = await client.callTool({ name, arguments: { panelId, after, messages } });
      assert.ok(!reply.isError, JSON.stringify(reply));
      assert.deepEqual(reply.content, []);
      assert.equal(reply.structuredContent, undefined, 'Canvas messages are UI-only');
      const packet = reply._meta.panel;
      session = packet.session;
      return { ...packet, events: packet.events.map(event => {
        assert.equal(event.sequence, ++after);
        const encoded = JSON.parse(event.wire);
        return encoded.encoding ? JSON.parse(gunzipSync(Buffer.from(encoded.data, 'base64'))) : encoded;
      }) };
    };
    const untilMessage = async predicate => {
      for (let n = 0; n < 100; n++) {
        const found = (await exchange()).events.find(predicate);
        if (found) return found;
        await new Promise(resolve => setTimeout(resolve, 30));
      }
      assert.fail('Canvas bridge did not receive the expected graph update');
    };
    assert.equal((await exchange()).waitingForModel, true);
    assert.equal(prompts.length, 0);
    const first = await poll(client, await call('open_graph', { workspaceRoot: project, file: 'model.py', cls: 'Tiny', interpreter: python }));
    assert.equal(first.status, 'ready', JSON.stringify(first));
    assert.equal(first.validated, true);
    const reference = first.models[0];
    const graph = await call('get_graph', { modelId: reference.modelId, revision: reference.revision, mode: 'graph', limit: 200 });
    assert.ok(graph.nodes.some(node => node.shapes.output?.join(',') === '2,3'));
    assert.equal(prompts.length, 1, 'initial inspection and capture share one plugin permission');
    const snapshot = await untilMessage(message => message.type === 'snapshot' && message.state.projection);
    const target = snapshot.state.activeTarget, viewId = session.viewId;
    const ref = { modelId: reference.modelId, revision: reference.revision };
    const beforeReceipt = await call('list_models', {});
    assert.equal(beforeReceipt.views.find(view => view.viewId === viewId).current, false, 'a socket is not a display receipt');
    const selected = { type: 'viewSelection', viewId: 'forged-view', target, nodeIds: [graph.nodes[0].nodeId], edgeIds: [] };
    const invalidBatch = await client.callTool({ name: 'graph_panel_exchange', arguments: { panelId, after,
      messages: [selected, { type: 'sourceEditRequest' }] } });
    assert.equal(invalidBatch.structuredContent.error.code, 'INVALID_PANEL_REQUEST');
    const noPartialWrite = await client.callTool({ name: 'get_selection', arguments: { ...ref, viewId } });
    assert.equal(noPartialWrite.structuredContent.error.code, 'SELECTION_UNAVAILABLE');
    await exchange([selected]);
    let selection;
    for (let n = 0; n < 100; n++) {
      selection = await client.callTool({ name: 'get_selection', arguments: { ...ref, viewId } });
      if (!selection.isError) break;
      await new Promise(resolve => setTimeout(resolve, 30));
    }
    assert.deepEqual(selection.structuredContent.nodeIds, selected.nodeIds, 'the broker supplies the view identity');
    const focus = client.callTool({ name: 'focus_view', arguments: { ...ref, viewId, nodeIds: selected.nodeIds } });
    const request = await untilMessage(message => message.type === 'focusView');
    await exchange([{ type: 'viewFocusResult', viewId, target, requestId: request.requestId, ok: true, detail: 'Test renderer applied focus.' }]);
    assert.equal((await focus).structuredContent.focused, true);
    await writeFile(path.join(project, 'model.py'), model('Sigmoid'));
    const stale = await call('list_models', {});
    assert.equal(stale.models[0].current, false);
    assert.equal(stale.views.find(view => view.viewId === viewId).current, false);
    const refused = await client.callTool({ name: 'get_graph', arguments: { modelId: reference.modelId, revision: reference.revision } });
    assert.equal(refused.structuredContent.error.code, 'STALE_TARGET');
    await exchange([{ type: 'graphSetupRequest', target: first.opening.target, openingId: first.opening.id,
      requestId: require('node:crypto').randomUUID(), action: { kind: 'retry' } }]);
    let refreshed;
    for (let n = 0; n < 400; n++) {
      refreshed = await call('list_models', {});
      if (refreshed.models[0].current && refreshed.models[0].revision !== reference.revision) break;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    assert.equal(refreshed.models[0].current, true, 'the panel Refresh action must reach the existing capture workflow');
    assert.equal(prompts.length, 1, 'source edits and recapture reuse the saved permission');
    assert.equal(await readFile(path.join(project, 'tensorviz.toml'), 'utf8'), recipe);
    assert.notEqual(refreshed.models[0].revision, reference.revision);
    const current = await call('get_graph', { modelId: refreshed.models[0].modelId, revision: refreshed.models[0].revision, mode: 'graph', limit: 200 });
    assert.match(JSON.stringify(current), /sigmoid/i);
    assert.ok(current.nodes.some(node => node.source), 'current source correspondence survives recapture');
    const updated = await untilMessage(message => message.type === 'projection' && message.target.sourceDigest !== target.sourceDigest);
    assert.equal(session.viewId, viewId, 'recapture reuses the same panel');
    await exchange([{ type: 'viewSelection', viewId, target: updated.target, nodeIds: [], edgeIds: [] }]);
    await exchange(); // Acknowledge the old renderer's updates before T3 reloads it.
    const retained = await call('list_captures', { modelId: refreshed.models[0].modelId });
    const refusedRestart = await client.callTool({ name: 'graph_panel_exchange', arguments: { panelId, after: 0,
      messages: [{ type: 'graphSetupRequest', target: first.opening.target, openingId: first.opening.id,
        requestId: require('node:crypto').randomUUID(), action: { kind: 'retry' } }] } });
    assert.equal(refusedRestart.structuredContent.error.code, 'INVALID_CURSOR', 'reconnection cannot replay a stale action');
    after = 0;
    const restored = await untilMessage(message => message.type === 'snapshot' && message.state.projection);
    assert.notEqual(session.viewId, viewId, 'a restored iframe needs its own fresh display receipt');
    assert.deepEqual(restored.state.activeTarget, updated.target);
    assert.deepEqual(await call('list_captures', { modelId: refreshed.models[0].modelId }), retained,
      'restoring a panel must reuse captured evidence without running the model');
    assert.equal((await call('list_models', {})).views.find(view => view.viewId === session.viewId).current, false);
    await exchange([{ type: 'viewSelection', target: restored.state.activeTarget, nodeIds: [], edgeIds: [] }]);
    await client.callTool({ name: 'close_graph_panel', arguments: { panelId } });
    const closed = await client.callTool({ name: 'graph_panel_exchange', arguments: { panelId, after } });
    assert.equal(closed.structuredContent.error.code, 'PANEL_CLOSED');
    await writeFile(path.join(project, 'model.py'), model('Sigmoid').replace('nn.Linear(4, 3)', 'nn.Linear(5, 3)'));
    const failure = await poll(client, await call('open_graph', { workspaceRoot: project, file: 'model.py', cls: 'Tiny' }));
    assert.equal(failure.status, 'failed', JSON.stringify(failure));
    assert.equal(failure.validated, false);
    assert.equal(failure.opening.hasCapture, true, 'a failing forward retains the observed graph');
    assert.match(failure.opening.detail, /shapes|multipl/i);
    const captures = await call('list_captures', { modelId: failure.models[0].modelId });
    assert.equal(captures.latestAttempt.status, 'failed');
    await writeFile(path.join(project, 'model.py'), model('Sigmoid'));
    const repaired = await poll(client, await call('open_graph', { workspaceRoot: project, file: 'model.py', cls: 'Tiny' }));
    assert.equal(repaired.status, 'ready', JSON.stringify(repaired));
    assert.equal(repaired.opening.failure, undefined);
    assert.equal(await readFile(path.join(project, 'tensorviz.toml'), 'utf8'), recipe, 'repair never changes inputs to hide a model error');
    assert.equal(stderr, '');
  } finally {
    await client.close();
    await rm(root, { recursive: true, force: true });
  }
});
