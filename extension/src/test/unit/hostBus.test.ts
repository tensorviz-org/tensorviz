import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { BusError, HOST_BUS_PROTOCOL, HostBusServer, HostNode, HostRegistry, callHost, type HostAnnouncement } from '../../hostBus';
import { type JsonObject } from '@tensorviz/graph-host/mcpGraph';
import { HostGraphHub, LocalGraphProvider, type HostTool, type HostToolset } from '../../mcpHost';

const tempDir = () => fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-bus-'));
const modelId = (host: string) => `${host}:${'a'.repeat(20)}`;

function tool(result: JsonObject): HostTool {
  return { config: { description: 'fixture', inputSchema: { modelId: z.string().optional() } },
    handler: async () => ({ content: [{ type: 'text', text: JSON.stringify(result) }], structuredContent: result }) };
}

suite('host bus registry and transport', () => {
  let dir: string;
  setup(() => { dir = tempDir(); });
  teardown(() => { fs.rmSync(dir, { recursive: true, force: true }); });

  test('announces, lists and withdraws a live record', async () => {
    const registry = new HostRegistry(dir);
    const record: HostAnnouncement = { protocol: HOST_BUS_PROTOCOL, hostId: 'host-a', pid: process.pid, workspaceFolder: '/w', port: 1234, token: 't', startedAt: Date.now() };
    await registry.announce(record);
    assert.deepEqual((await registry.list()).map(item => item.hostId), ['host-a']);
    await registry.withdraw('host-a');
    assert.deepEqual(await registry.list(), []);
  });

  test('drops a record whose process is gone', async () => {
    const registry = new HostRegistry(dir);
    await registry.announce({ protocol: HOST_BUS_PROTOCOL, hostId: 'dead', pid: 2147483647, workspaceFolder: null, port: 1, token: 't', startedAt: Date.now() });
    assert.deepEqual(await registry.list(), []);
    assert.equal(fs.readdirSync(dir).length, 0);
  });
  test('heartbeats never temporarily withdraw a live owner', async () => {
    const registry = new HostRegistry(dir);
    const record: HostAnnouncement = { protocol: HOST_BUS_PROTOCOL, hostId: 'live', pid: process.pid,
      workspaceFolder: '/workspace', port: 1234, token: 't', startedAt: Date.now() };
    await registry.announce(record);
    await Promise.all([
      (async () => { for (let i = 0; i < 40; i++) await registry.announce(record); })(),
      (async () => { for (let i = 0; i < 80; i++) assert.equal((await registry.list()).length, 1); })(),
    ]);
    assert.deepEqual(fs.readdirSync(dir), ['live.json']);
  });

  test('answers a token-authenticated call and refuses a wrong token', async () => {
    const server = new HostBusServer(async (method, params) => ({ method, echo: params }));
    const port = await server.start();
    try {
      assert.deepEqual(await callHost({ port, token: server.token }, 'describe', { x: 1 }), { method: 'describe', echo: { x: 1 } });
      await assert.rejects(callHost({ port, token: 'wrong' }, 'describe', {}), (error: unknown) => error instanceof BusError && error.code === 'UNAUTHORIZED');
    } finally { await server.close(); }
  });

  test('propagates a structured tool error across the wire', async () => {
    const server = new HostBusServer(async () => { throw new BusError('NO_MODEL', 'nothing here'); });
    const port = await server.start();
    try {
      await assert.rejects(callHost({ port, token: server.token }, 'call', {}), (error: unknown) => error instanceof BusError && error.code === 'NO_MODEL');
    } finally { await server.close(); }
  });

  test('times out when a peer accepts but never answers', async () => {
    const hanging = new HostBusServer(() => new Promise<never>(() => {}));
    const port = await hanging.start();
    try {
      await assert.rejects(callHost({ port, token: hanging.token }, 'describe', {}, 50), (error: unknown) => error instanceof BusError && error.code === 'BUS_TIMEOUT');
    } finally { await hanging.close(); }
  });

  test('HostNode start/stop publishes and removes its own record', async () => {
    const node = new HostNode({ hostId: 'node-a', workspaceFolder: '/w', registryDir: dir, handler: async () => ({ ok: true }) });
    await node.start();
    assert.equal((await node.registry.list()).length, 1);
    await node.stop();
    assert.equal((await node.registry.list()).length, 0);
  });
});

suite('single MCP endpoint over the host bus', () => {
  let dir: string;
  let peer: HostNode;
  let client: Client;
  let server: McpServer;
  let hub: HostGraphHub;
  let released: string[];
  let calledOwners: string[];
  const hubHost = 'hub-host';
  const peerHost = 'peer-host';

  setup(async () => {
    dir = tempDir();
    released = []; calledOwners = [];
    const peerTools: HostToolset = new Map([
      ['get_graph', tool({ remote: true })],
      ['validate_model', tool({ remote: true })],
      ['open_graph', tool({ requestId: 'remote-request', nextAction: { kind: 'poll', tool: 'open_graph', arguments: { requestId: 'remote-request' } } })],
      ['get_graph_setup', tool({ remote: true })],
      ['get_execution_environment', tool({ interpreter: '/peer/venv/bin/python' })],
      ['open_graph_diff', { ...tool({ status: 'displayed', reviewId: 'peer-review' }), config: { inputSchema: { file: z.string().optional() } } }],
      ['focus_view', tool({ remote: true })],
    ]);
    const provider = new LocalGraphProvider(owner => { calledOwners.push(owner); return peerTools; }, () => ({
      hostId: peerHost, workspaceFolders: ['/peer', '/second'], views: [{ viewId: '5', current: true }],
      models: [{ modelId: modelId(peerHost), revision: 'a'.repeat(64), current: true }],
    }), owner => released.push(owner));
    peer = new HostNode({
      hostId: peerHost, workspaceFolder: '/peer', registryDir: dir,
      handler: async (method, params, signal) => {
        if (method === 'describe') return provider.list();
        const { name, args, owner } = params as any;
        if (method === 'release') { provider.release(owner); return {}; }
        return provider.call(name, args, signal, owner);
      },
    });
    await peer.start();

    const localTools: HostToolset = new Map([
      ['list_models', tool({ local: true })],
      ['get_graph', tool({ local: true })],
      ['validate_model', tool({ local: true })],
      ['open_graph', { ...tool({ local: true, nextAction: { kind: 'poll', tool: 'open_graph', arguments: { requestId: 'local-request' } } }), config: { inputSchema: { file: z.string().optional(), requestId: z.string().optional() } } }],
      ['get_graph_setup', tool({ local: true })],
      ['get_execution_environment', { ...tool({ interpreter: '/hub/venv/bin/python' }), config: { inputSchema: { file: z.string().optional() } } }],
      ['open_graph_diff', { ...tool({ status: 'displayed', reviewId: 'local-review' }), config: { inputSchema: { file: z.string().optional() } } }],
      ['focus_view', tool({ local: true })],
    ]);
    hub = new HostGraphHub({
      hostId: hubHost, owner: 'hub-host/session-one', workspaceFolder: '/hub', registry: new HostRegistry(dir), tools: localTools,
      listLocal: () => ({ hostId: hubHost, views: [{ viewId: '1', current: true }], models: [{ modelId: modelId(hubHost), revision: 'a'.repeat(64), target: { workspaceFolder: '/hub', file: 'h.py', cls: 'H', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 }, current: true, traced: true, truncated: false }], detail: '' }),
      remote: new Set(localTools.keys()),
    });
    server = new McpServer({ name: 'hub', version: '1' });
    hub.register(server);
    client = new Client({ name: 'agent', version: '1' });
    const [a, b] = InMemoryTransport.createLinkedPair();
    await server.connect(a); await client.connect(b);
  });

  teardown(async () => {
    await client.close(); await server.close(); await hub.dispose(); await peer.stop();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  async function call(name: string, args: JsonObject = {}) {
    const reply = await client.callTool({ name, arguments: args });
    return { error: reply.isError, data: reply.structuredContent as any };
  }

  test('list_models aggregates every window and tags ownership', async () => {
    const result = await call('list_models');
    const models = result.data.models as any[];
    assert.deepEqual(models.map(model => model.hostId).sort(), [hubHost, peerHost]);
    assert.deepEqual(result.data.views.map((view: any) => view.hostId).sort(), [hubHost, peerHost]);
    assert.deepEqual(result.data.hosts.map((host: any) => host.reachable), [true, true]);
  });

  test('a read-only tool routes to the owning window', async () => {
    const local = await call('get_graph', { modelId: modelId(hubHost), revision: 'a'.repeat(64) });
    assert.equal((local.data as any).local, true);
    const remote = await call('get_graph', { modelId: modelId(peerHost), revision: 'a'.repeat(64) });
    assert.equal((remote.data as any).remote, true);
  });

  test('execution routes to the owning window with session identity', async () => {
    const result = await call('validate_model', { modelId: modelId(peerHost) });
    assert.equal(result.error, undefined);
    assert.equal(result.data.remote, true);
    assert.deepEqual(calledOwners, ['hub-host/session-one']);
  });

  test('opening routes by any owning workspace and subsequent setup/focus stays there', async () => {
    const opened = await call('open_graph', { file: '/second/model.py' });
    assert.equal(opened.data.hostId, peerHost);
    assert.equal(opened.data.requestId, 'remote-request');
    assert.equal((await call('get_graph_setup')).data.remote, true);
    assert.equal((await call('focus_view', { modelId: modelId(peerHost) })).data.remote, true);
    await hub.dispose();
    assert.deepEqual(released, ['hub-host/session-one']);
  });

  test('ambiguous empty windows require hostId; explicit host cannot contradict a model', async () => {
    assert.equal((await call('open_graph')).data.error.code, 'HOST_REQUIRED');
    assert.equal((await call('open_graph', { hostId: peerHost })).data.requestId, 'remote-request');
    assert.equal((await call('get_graph', { hostId: hubHost, modelId: modelId(peerHost) })).data.error.code, 'HOST_MISMATCH');
  });

  test('continuation arguments keep their owning window when another call changes the selected host', async () => {
    const remote = await call('open_graph', { file: '/second/model.py' });
    assert.deepEqual(remote.data.nextAction.arguments, { requestId: 'remote-request', hostId: peerHost });
    const local = await call('open_graph', { hostId: hubHost });
    assert.deepEqual(local.data.nextAction.arguments, { requestId: 'local-request', hostId: hubHost });
    const resumed = await call('open_graph', remote.data.nextAction.arguments);
    assert.equal(resumed.data.hostId, peerHost); assert.equal(resumed.data.requestId, 'remote-request');
  });

  test('environment inspection routes by file before opening and never uses the hub interpreter for a peer', async () => {
    const remote = await call('get_execution_environment', { file: '/second/model.py' });
    assert.equal(remote.data.hostId, peerHost); assert.equal(remote.data.interpreter, '/peer/venv/bin/python');
    const local = await call('get_execution_environment', { hostId: hubHost });
    assert.equal(local.data.interpreter, '/hub/venv/bin/python');
  });

  test('graph diff routes by workspace without a live model and refuses ambiguous windows', async () => {
    const remote = await call('open_graph_diff', { file: '/second/model.py' });
    assert.equal(remote.data.hostId, peerHost); assert.equal(remote.data.reviewId, 'peer-review');
    assert.equal((await call('open_graph_diff')).data.error.code, 'HOST_REQUIRED');
    const local = await call('open_graph_diff', { hostId: hubHost });
    assert.equal(local.data.reviewId, 'local-review');
  });

  test('a file outside the only open workspace is a plain error', async () => {
    await peer.stop();
    const opened = await call('open_graph', { file: '/hub/../secret.py' });
    assert.equal(opened.error, true);
    assert.equal(opened.data.error.code, 'OUTSIDE_WORKSPACE');
    assert.equal(opened.data.approval, undefined);
    assert.notEqual(opened.data.nextAction?.kind, 'approval_required');
  });

  test('an unknown remote host reports NO_MODEL', async () => {
    const result = await call('get_graph', { modelId: modelId('ghost-host'), revision: 'a'.repeat(64) });
    assert.equal(result.error, true);
    assert.equal((result.data as any).error.code, 'NO_MODEL');
  });
});

suite('remote session lifetime', () => {
  test('disconnect and dead endpoint hosts revoke only their own work', async () => {
    const released: string[] = [];
    const provider = new LocalGraphProvider(owner => new Map([['run', tool({ owner })]]), () => ({}), owner => released.push(owner));
    const signal = new AbortController().signal;
    assert.deepEqual(await provider.call('run', {}, signal, 'a/one'), { owner: 'a/one' });
    assert.deepEqual(await provider.call('run', {}, signal, 'b/two'), { owner: 'b/two' });
    provider.reap(new Set(['b']));
    assert.deepEqual(released, ['a/one']);
    provider.release('b/two');
    assert.deepEqual(released, ['a/one', 'b/two']);
  });
});
