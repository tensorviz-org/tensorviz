const { test } = require('node:test');
const assert = require('node:assert/strict');
const { randomUUID } = require('node:crypto');
const { McpServer } = require('@modelcontextprotocol/sdk/server/mcp.js');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { InMemoryTransport } = require('@modelcontextprotocol/sdk/inMemory.js');
const { GraphPanels, registerGraphPanel } = require('../../out/host/graphPanel.cjs');

test('a panel remounts after five idle minutes and expires after an idle hour', async t => {
  t.mock.timers.enable({ apis: ['Date', 'setInterval'], now: 1_000_000 });
  const host = { graph: { list: () => ({ models: [] }) }, graphView: { cancelOwner() {} } };
  const panels = new GraphPanels(host), signal = new AbortController().signal;
  try {
    const id = panels.open();
    await panels.exchange(id, 0, [], signal, 'original-frame');
    await panels.disconnect(id, 'original-frame');
    t.mock.timers.tick(6 * 60_000);
    assert.equal((await panels.exchange(id, 0, [], signal, 'returned-frame')).waitingForModel, true);
    await panels.disconnect(id, 'returned-frame');
    t.mock.timers.tick(59 * 60_000);
    assert.ok(panels.panels.has(id), 'navigation retains the opener for an hour');
    t.mock.timers.tick(75_000);
    await assert.rejects(panels.exchange(id, 0, [], signal, 'late-frame'), { code: 'PANEL_CLOSED' });
  } finally { panels.dispose(); }
});

test('panel traffic renews the idle hour, and explicit closure releases it immediately', async t => {
  t.mock.timers.enable({ apis: ['Date', 'setInterval'], now: 1_000_000 });
  const host = { graph: { list: () => ({ models: [] }) }, graphView: { cancelOwner() {} } };
  const panels = new GraphPanels(host), signal = new AbortController().signal;
  try {
    const id = panels.open();
    for (let i = 0; i < 3; i++) {
      await panels.exchange(id, 0, [], signal, 'frame');
      t.mock.timers.tick(59 * 60_000);
    }
    assert.equal((await panels.exchange(id, 0, [], signal, 'frame')).waitingForModel, true);
    panels.close(id, 'frame');
    await assert.rejects(panels.exchange(id, 0, [], signal, 'frame'), { code: 'PANEL_CLOSED' });
  } finally { panels.dispose(); }
});

test('empty panels have bounded lifetimes and cannot access another connection', async () => {
  const cancelled = [];
  const host = { graph: { list: () => ({ models: [] }) }, graphView: { cancelOwner: id => cancelled.push(id) } };
  const panels = new GraphPanels(host), other = new GraphPanels(host);
  try {
    const id = panels.open(), signal = new AbortController().signal;
    assert.equal((await panels.exchange(id, 0, [], signal)).waitingForModel, true);
    await assert.rejects(other.exchange(id, 0, [], signal), { code: 'PANEL_CLOSED' });
    await assert.rejects(panels.exchange(id, 1, [], signal), { code: 'INVALID_CURSOR' });
    for (let i = 1; i < 8; i++) await panels.exchange(panels.open(), 0, [], signal);
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


for (const [name, capabilities] of [
  ['text-only client', {}],
  ['MCP Apps client', { extensions: { 'io.modelcontextprotocol/ui': { mimeTypes: ['text/html;profile=mcp-app'] } } }],
]) test(`${name}: unmounted retries stay bounded and late bridges keep separate identities`, async () => {
  const server = new McpServer({ name: 'panel-test', version: '1' });
  const host = { graph: { list: () => ({ models: [] }) }, graphView: { cancelOwner() {} } };
  const panels = await registerGraphPanel(server, host, '/unused', 'test');
  const client = new Client({ name: 'panel-test', version: '1' }, { capabilities });
  const [a, b] = InMemoryTransport.createLinkedPair();
  try {
    await server.connect(a); await client.connect(b);
    const tools = (await client.listTools()).tools;
    assert.equal(tools.find(tool => tool.name === 'open_graph_panel')._meta.ui.resourceUri,
      'ui://tensorviz/graph-panel-test.html');
    for (const name of ['graph_panel_read', 'graph_panel_exchange', 'detach_graph_panel', 'close_graph_panel']) {
      assert.deepEqual(tools.find(tool => tool.name === name)._meta.ui.visibility, ['app']);
    }
    const ids = new Set();
    for (let attempt = 0; attempt < 20; attempt++) {
      const result = await client.callTool({ name: 'open_graph_panel', arguments: {} });
      assert.ok(!result.isError);
      assert.deepEqual(result.structuredContent, { displayed: false, status: 'awaiting_view', fallbackTool: 'open_graph_view' });
      assert.match(result.content[0].text, /display is not confirmed/);
      assert.match(result.content[0].text, /open_graph_view/);
      assert.match(result._meta.panelId, /^[a-f0-9-]{36}$/);
      ids.add(result._meta.panelId);
    }
    assert.equal(ids.size, 20, 'each tool result keeps its own app identity');
    const allIds = [...ids], id = allIds.at(-1), previousId = allIds.at(-2);
    await assert.rejects(panels.exchange(allIds[0], 0, [], new AbortController().signal), { code: 'PANEL_CLOSED' });
    // T3 can attach a separate bridge after initialize and after the opener.
    const attached = await client.callTool({ name: 'graph_panel_read', arguments: { panelId: id, clientId: 'late-frame', after: 0 } });
    assert.ok(!attached.isError);
    assert.equal(attached._meta.panel.waitingForModel, true);
    const next = await client.callTool({ name: 'open_graph_panel', arguments: {} });
    assert.notEqual(next._meta.panelId, id, 'an attached panel is never reused as a pending opener');
    await client.callTool({ name: 'detach_graph_panel', arguments: { panelId: id, clientId: 'late-frame' } });
    const retry = await client.callTool({ name: 'open_graph_panel', arguments: {} });
    assert.notEqual(retry._meta.panelId, next._meta.panelId);
    const previous = await client.callTool({ name: 'graph_panel_read', arguments: { panelId: previousId, clientId: 'second-late-frame', after: 0 } });
    assert.ok(!previous.isError, 'another delayed frame has its own retained opener');
    const remounted = await client.callTool({ name: 'graph_panel_read', arguments: { panelId: id, clientId: 'remounted-frame', after: 0 } });
    assert.ok(!remounted.isError);
    assert.equal(remounted._meta.panel.waitingForModel, true);
  } finally { panels.dispose(); await client.close(); await server.close(); }
});

test('pending opener cleanup and invalid requests do not exhaust or steal panel slots', async () => {
  const host = { graph: { list: () => ({ models: [] }) }, graphView: { cancelOwner() {} } };
  const panels = new GraphPanels(host), signal = new AbortController().signal;
  try {
    const pending = panels.open();
    await assert.rejects(panels.exchange(pending, 1, [], signal), { code: 'INVALID_CURSOR' });
    assert.notEqual(panels.open(), pending);
    panels.close(pending);
    const replacement = panels.open();
    assert.notEqual(replacement, pending);
    await assert.rejects(panels.exchange(pending, 0, [], signal), { code: 'PANEL_CLOSED' });
    // A legacy frame without clientId also counts as attached.
    await panels.exchange(replacement, 0, [], signal);
    assert.notEqual(panels.open(), replacement);
  } finally { panels.dispose(); }
});

test('only the latest eight unmounted openers survive, and late frames attach independently', async () => {
  const cancelled = [];
  const host = { graph: { list: () => ({ models: [] }) }, graphView: { cancelOwner: id => cancelled.push(id) } };
  const panels = new GraphPanels(host), signal = new AbortController().signal;
  try {
    const ids = Array.from({ length: 20 }, () => panels.open());
    assert.equal(new Set(ids).size, 20);
    for (const id of ids.slice(0, -8)) await assert.rejects(panels.exchange(id, 0, [], signal), { code: 'PANEL_CLOSED' });
    const kept = ids.slice(-8);
    for (const [index, id] of kept.entries()) {
      assert.equal((await panels.exchange(id, 0, [], signal, `frame-${index}`)).waitingForModel, true);
    }
    const cancellations = cancelled.length;
    for (const [index, id] of kept.entries()) await panels.exchange(id, 0, [], signal, `frame-${index}`);
    assert.equal(cancelled.length, cancellations, 'independent frames never replace one another');
    await panels.disconnect(kept[0], 'frame-0');
    assert.throws(() => panels.open(), { code: 'PANEL_LIMIT' }, 'detached frames retain their slot for remount');
    assert.equal((await panels.exchange(kept[0], 0, [], signal, 'remounted-frame')).waitingForModel, true);
  } finally { panels.dispose(); }
});

test('an attachment in flight is protected from retry eviction', async () => {
  let finishOpening;
  const host = { graph: { list: () => ({ models: [] }) }, graphView: { cancelOwner() {} },
    openingView: () => new Promise(resolve => { finishOpening = resolve; }) };
  const panels = new GraphPanels(host), signal = new AbortController().signal;
  try {
    const id = panels.open();
    const connection = panels.exchange(id, 0, [], signal, 'slow-frame');
    for (let attempt = 0; attempt < 20; attempt++) assert.notEqual(panels.open(), id);
    finishOpening(undefined);
    assert.equal((await connection).waitingForModel, true);
    host.openingView = undefined;
    assert.equal((await panels.exchange(id, 0, [], signal, 'slow-frame')).waitingForModel, true);
  } finally { panels.dispose(); }
});

test('retired panel clients cannot reclaim ownership or mutate queued state', async () => {
  const cancelled = [];
  const host = { graph: { list: () => ({ models: [] }) }, graphView: { cancelOwner: id => cancelled.push(id) } };
  const panels = new GraphPanels(host), signal = new AbortController().signal;
  try {
    const id = panels.open();
    await panels.exchange(id, 0, [], signal, 'old');
    await panels.exchange(id, 0, [], signal, 'current');
    const panel = panels.panels.get(id);
    panel.events.push({ sequence: 1, wire: 'queued' });
    panel.sequence = 1; panel.bytes = 6; panel.touched = 123;
    const controller = panel.controller, cancellations = cancelled.length;
    await assert.rejects(panels.exchange(id, 0, [], signal, 'old'), { code: 'PANEL_REPLACED' });
    assert.equal(panel.clientId, 'current');
    assert.equal(panel.controller, controller);
    assert.equal(cancelled.length, cancellations);
    assert.equal(panel.touched, 123);
    assert.deepEqual(panel.events, [{ sequence: 1, wire: 'queued' }]);
    assert.equal(panel.bytes, 6);
    assert.equal(panel.sequence, 1);
    await panels.exchange(id, 0, [], signal, 'current');
  } finally { panels.dispose(); }
});

test('panel retirement is bounded and fails closed without replacing its current owner', async () => {
  const host = { graph: { list: () => ({ models: [] }) }, graphView: { cancelOwner() {} } };
  const panels = new GraphPanels(host), signal = new AbortController().signal;
  try {
    const id = panels.open();
    for (let i = 0; i <= 64; i++) await panels.exchange(id, 0, [], signal, `frame-${i}`);
    const panel = panels.panels.get(id), controller = panel.controller;
    assert.equal(panel.retiredClientIds.size, 64);
    for (let i = 65; i < 80; i++) {
      await assert.rejects(panels.exchange(id, 0, [], signal, `frame-${i}`), { code: 'PANEL_REPLACED' });
    }
    assert.equal(panel.retiredClientIds.size, 64);
    assert.equal(panel.clientId, 'frame-64');
    assert.equal(panel.controller, controller);
    await panels.exchange(id, 0, [], signal, 'frame-64');
    await panels.exchange(id, 0, [], signal); // Legacy clients remain supported.
  } finally { panels.dispose(); }
});
