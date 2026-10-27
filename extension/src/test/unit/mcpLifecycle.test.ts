import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import * as net from 'node:net';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { McpLifecycle } from '../../mcpLifecycle';
import { GraphMcpServer } from '../../mcpServer';
import { HostNode, HostRegistry } from '../../hostBus';

suite('persistent MCP connection', () => {
  let dir: string;
  const windows: Array<{ lifecycle: McpLifecycle; node: HostNode }> = [];
  setup(async () => { dir = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-mcp-lifecycle-')); });
  teardown(async () => {
    for (const window of windows.splice(0)) { await window.lifecycle.dispose(); await window.node.stop(); }
    await fs.rm(dir, { recursive: true, force: true });
  });
  async function window(id: string, beforeStart?: () => Promise<void>) {
    let endpoint: GraphMcpServer | undefined;
    const local = () => endpoint?.connection ? { ...endpoint.connection, sessions: endpoint.sessionCount } : undefined;
    const lifecycle = new McpLifecycle({ directory: dir, hostId: id, registry: new HostRegistry(path.join(dir, 'hosts')),
      start: async port => {
        await beforeStart?.();
        const server = new GraphMcpServer(() => {}, id);
        try { const result = await server.start(port); endpoint = server; return result; }
        catch (error) { await server.close(); throw error; }
      },
      stop: async () => { const old = endpoint; endpoint = undefined; await old?.close(); }, local, publish: () => {},
    });
    const node = new HostNode({ hostId: id, workspaceFolder: null, registryDir: path.join(dir, 'hosts'),
      handler: async method => {
        if (method === 'sync') { await lifecycle.synchronize(); return {}; }
        return { models: [], mcp: local() };
      } });
    await node.start();
    const value = { lifecycle, node, local };
    windows.push(value);
    return value;
  }
  async function availablePort(): Promise<number> {
    const server = net.createServer();
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
    const port = (server.address() as net.AddressInfo).port;
    await new Promise<void>(resolve => server.close(() => resolve()));
    return port;
  }
  test('stays off until consent; reconnects without a project or graph', async () => {
    const first = await window('one'); await first.lifecycle.refresh();
    assert.equal(first.local(), undefined); assert.equal(first.lifecycle.status.state, 'Off');
    await first.lifecycle.setEnabled(true);
    const url = first.lifecycle.status.url!;
    assert.ok(Number(new URL(url).port) > 0);
    assert.equal(JSON.parse(await fs.readFile(path.join(dir, 'mcp-connection.json'), 'utf8')).port, 0,
      'new connections must select a free port by default');
    await first.lifecycle.dispose(); await first.node.stop();
    const next = await window('next'); await next.lifecycle.refresh();
    assert.equal(next.lifecycle.status.state, 'Ready'); assert.ok(next.local());
    assert.equal(next.lifecycle.status.url, url);
  });
  test('one owner, shared connected state, same URL after owner closes, global disconnect', async () => {
    const first = await window('one'), second = await window('two');
    await first.lifecycle.setEnabled(true, await availablePort());
    assert.ok(first.local()); assert.equal(second.local(), undefined);
    const url = first.lifecycle.status.url!;
    const client = new Client({ name: 'test-agent', version: '1' });
    const transport = new StreamableHTTPClientTransport(new URL(url));
    await client.connect(transport);
    await second.lifecycle.refresh();
    assert.equal(second.lifecycle.status.state, 'Connected');
    await client.close();
    await first.lifecycle.dispose(); await first.node.stop();
    await second.lifecycle.refresh();
    assert.equal(second.lifecycle.status.url, url); assert.ok(second.local());
    await second.lifecycle.setEnabled(false);
    assert.equal(second.local(), undefined); assert.equal(second.lifecycle.status.state, 'Off');
    const third = await window('three'); await third.lifecycle.refresh(); assert.equal(third.local(), undefined);
  });
  test('automatic selection persists a real port and shares it across simultaneous windows and takeover', async () => {
    const first = await window('one'), second = await window('two');
    await Promise.all([first.lifecycle.setEnabled(true, 0), second.lifecycle.setEnabled(true, 0)]);
    await Promise.all([first.lifecycle.refresh(), second.lifecycle.refresh()]);
    const url = first.lifecycle.status.url!;
    assert.ok(Number(new URL(url).port) > 0);
    assert.equal(second.lifecycle.status.url, url);
    assert.equal([first.local(), second.local()].filter(Boolean).length, 1, 'only one window owns the shared endpoint');
    const owner = first.local() ? first : second, next = first.local() ? second : first;
    await owner.lifecycle.dispose(); await owner.node.stop();
    await next.lifecycle.refresh();
    assert.equal(next.lifecycle.status.url, url, 'takeover should retain the agent URL');
    assert.ok(next.local());
    await next.lifecycle.setEnabled(false);
    await next.lifecycle.setEnabled(true, 0);
    assert.equal(next.lifecycle.status.url, url, 'reconnecting must reuse the allocated port when available');
  });
  test('an occupied automatic port is replaced without connecting to or stopping the foreign listener', async () => {
    const first = await window('one');
    await first.lifecycle.setEnabled(true, 0);
    const previous = first.lifecycle.status.url!;
    await first.lifecycle.dispose(); await first.node.stop();
    const listener = net.createServer(() => { assert.fail('TensorViz must not probe an unrelated listener'); });
    await new Promise<void>(resolve => listener.listen(Number(new URL(previous).port), '127.0.0.1', resolve));
    try {
      const next = await window('next'); await next.lifecycle.refresh();
      assert.equal(next.lifecycle.status.state, 'Ready');
      assert.notEqual(next.lifecycle.status.url, previous);
      assert.equal(listener.listening, true);
      const peer = await window('peer'); await peer.lifecycle.refresh();
      assert.equal(peer.lifecycle.status.url, next.lifecycle.status.url);
      assert.equal(peer.local(), undefined);
    } finally { await new Promise<void>(resolve => listener.close(() => resolve())); }
  });
  test('a foreign listener never masquerades as a TensorViz connection', async () => {
    const listener = net.createServer();
    await new Promise<void>(resolve => listener.listen(0, '127.0.0.1', resolve));
    try {
      const first = await window('one');
      await first.lifecycle.setEnabled(true, (listener.address() as net.AddressInfo).port);
      assert.equal(first.lifecycle.status.state, 'Off'); assert.equal(first.lifecycle.status.url, undefined);
      assert.match(first.lifecycle.status.error!, /another application/);
    } finally { await new Promise<void>(resolve => listener.close(() => resolve())); }
  });
  test('disconnect from a non-owner stops the owner and prevents reopening', async () => {
    const first = await window('one'), second = await window('two');
    await first.lifecycle.setEnabled(true, await availablePort());
    await second.lifecycle.setEnabled(false);
    assert.equal(first.local(), undefined); assert.equal(first.lifecycle.status.state, 'Off');
    await first.lifecycle.refresh(); assert.equal(first.local(), undefined);
  });
  test('invalid settings fail closed and out-of-range ports are refused', async () => {
    const first = await window('one');
    for (const port of [-1, 65536, 1.5]) await assert.rejects(first.lifecycle.setEnabled(true, port), /MCP port/);
    await first.lifecycle.setEnabled(true, await availablePort());
    await fs.writeFile(path.join(dir, 'mcp-connection.json'), '{invalid');
    await first.lifecycle.refresh(); assert.equal(first.local(), undefined);
    assert.equal(first.lifecycle.status.state, 'Off');
  });
  test('disconnect waits for an old consent tick before confirming the owner stopped', async () => {
    let release!: () => void, started = false;
    const first = await window('one', async () => {
      started = true; await new Promise<void>(resolve => { release = resolve; });
    });
    const second = await window('two');
    const enabling = first.lifecycle.setEnabled(true, await availablePort());
    while (!started) await new Promise(resolve => setImmediate(resolve));
    const disabling = second.lifecycle.setEnabled(false);
    release();
    await Promise.all([enabling, disabling]);
    assert.equal(first.local(), undefined); assert.equal(first.lifecycle.status.state, 'Off');
  });
});
