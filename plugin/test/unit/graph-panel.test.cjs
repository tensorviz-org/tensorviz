const { test } = require('node:test');
const assert = require('node:assert/strict');
const { randomUUID } = require('node:crypto');
const { McpServer } = require('@modelcontextprotocol/sdk/server/mcp.js');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { InMemoryTransport } = require('@modelcontextprotocol/sdk/inMemory.js');
const { GraphPanels, registerGraphPanel } = require('../../out/host/graphPanel.cjs');

test('empty panels have bounded lifetimes and cannot access another connection', async () => {
  const cancelled = [];
  const host = { graph: { list: () => ({ models: [] }) }, graphView: { cancelOwner: id => cancelled.push(id) } };
  const panels = new GraphPanels(host), other = new GraphPanels(host);
  try {
    const id = panels.open(), signal = new AbortController().signal;
    assert.equal((await panels.exchange(id, 0, [], signal)).waitingForModel, true);
    await assert.rejects(other.exchange(id, 0, [], signal), { code: 'PANEL_CLOSED' });
    await assert.rejects(panels.exchange(id, 1, [], signal), { code: 'INVALID_CURSOR' });
    for (let i = 1; i < 8; i++) panels.open();
    assert.throws(() => panels.open(), { code: 'PANEL_LIMIT' });
    panels.close(id);
    assert.ok(cancelled.includes(`panel:${id}`));
    await assert.rejects(panels.exchange(id, 0, [], signal), { code: 'PANEL_CLOSED' });
    assert.doesNotThrow(() => panels.open());
  } finally { panels.dispose(); other.dispose(); }
});

test('Canvas transport bypasses model output truncation without leaking data into content', async () => {
  const server = new McpServer({ name: 'panel-test', version: '1' });
  const panels = await registerGraphPanel(server, { graphView: { cancelOwner() {} } }, '/unused', 'test');
  const client = new Client({ name: 'panel-test', version: '1' });
  const [a, b] = InMemoryTransport.createLinkedPair();
  try {
    await server.connect(a); await client.connect(b);
    const packet = { events: [{ sequence: 1, wire: 'x'.repeat(600 * 1024) }] };
    panels.exchange = async () => packet;
    const result = await client.callTool({ name: 'graph_panel_exchange', arguments: { panelId: randomUUID(), after: 0 } });
    assert.ok(!result.isError);
    assert.deepEqual(result.content, []);
    assert.equal(result.structuredContent, undefined);
    assert.deepEqual(result._meta.panel, packet);
  } finally { panels.dispose(); await client.close(); await server.close(); }
});

test('a remounted panel survives stale frame cleanup and cancels an unfinished connection', async () => {
  let finishOpening;
  const host = { graph: { list: () => ({ models: [] }) }, graphView: { cancelOwner() {} },
    openingView: signal => new Promise(resolve => {
      finishOpening = resolve;
      signal.addEventListener('abort', () => resolve(undefined), { once: true });
    }) };
  const panels = new GraphPanels(host), signal = new AbortController().signal;
  try {
    const id = panels.open();
    const pending = panels.exchange(id, 0, [], signal, 'frame-1');
    await panels.disconnect(id, 'frame-1');
    await pending;
    host.openingView = undefined;
    assert.equal((await panels.exchange(id, 0, [], signal, 'frame-2')).waitingForModel, true);
    await panels.disconnect(id, 'frame-1');
    panels.close(id, 'frame-1');
    await assert.rejects(panels.exchange(id, 1, [], signal, 'frame-1'), { code: 'PANEL_REPLACED' });
    await assert.rejects(panels.exchange(id, 0, [{ type: 'graphSetupRequest' }], signal, 'frame-1'), { code: 'PANEL_REPLACED' });
    assert.equal((await panels.exchange(id, 0, [], signal, 'frame-2')).waitingForModel, true);
    panels.close(id, 'frame-2');
    await assert.rejects(panels.exchange(id, 0, [], signal, 'frame-2'), { code: 'PANEL_CLOSED' });
    finishOpening(undefined);
  } finally { panels.dispose(); }
});

test('read-only panel transport rejects a mixed action batch before forwarding anything', async () => {
  const server = new McpServer({ name: 'panel-test', version: '1' });
  const panels = await registerGraphPanel(server, { graphView: { cancelOwner() {} } }, '/unused', 'test');
  const client = new Client({ name: 'panel-test', version: '1' });
  const [a, b] = InMemoryTransport.createLinkedPair();
  const forwarded = [];
  panels.exchange = async (_id, _after, messages) => { forwarded.push(messages); return { events: [] }; };
  const args = { panelId: randomUUID(), after: 0 };
  try {
    await server.connect(a); await client.connect(b);
    for (const type of ['graphSetupRequest', 'graphReviewRequest', 'graphReviewApplied', 'unknown']) {
      const reply = await client.callTool({ name: 'graph_panel_read', arguments: { ...args,
        messages: [{ type: 'viewSelection' }, { type }] } });
      assert.equal(reply.isError, true);
      assert.equal(reply.structuredContent.error.code, 'INVALID_PANEL_REQUEST');
    }
    assert.deepEqual(forwarded, [], 'a read-only tool must never dispatch even part of a batch containing an action');
    for (const messages of [[], ...['viewSelection', 'viewFocusResult', 'captureInspectRequest', 'captureInspectCancel'].map(type => [{ type }])]) {
      const reply = await client.callTool({ name: 'graph_panel_read', arguments: { ...args, messages } });
      assert.ok(!reply.isError);
      assert.deepEqual(reply.content, []);
      assert.equal(reply.structuredContent, undefined);
      assert.deepEqual(reply._meta.panel, { events: [] });
      assert.deepEqual(forwarded.at(-1), messages);
    }
  } finally { panels.dispose(); await client.close(); await server.close(); }
});
