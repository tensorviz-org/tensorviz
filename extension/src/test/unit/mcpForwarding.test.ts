import * as assert from 'node:assert/strict';
import * as net from 'node:net';
import { request } from 'node:http';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { GraphMcpServer } from '../../mcpServer';
import { McpForwarding } from '../../mcpForwarding';

suite('MCP desktop SSH endpoint', () => {
  const endpoints: GraphMcpServer[] = [];
  const tunnels: net.Server[] = [];
  const sockets = new Set<net.Socket>();
  const clients: Client[] = [];
  let forwarding: McpForwarding;
  function create() {
    const endpoint = new GraphMcpServer(server => {
      server.registerTool('read_fixture', { inputSchema: {} }, async () => ({ content: [{ type: 'text', text: 'remote graph' }] }));
    });
    endpoints.push(endpoint); return endpoint;
  }
  async function tunnel(origin: string): Promise<string> {
    const port = Number(new URL(origin).port);
    const server = net.createServer(downstream => {
      const upstream = net.connect(port, '127.0.0.1');
      for (const socket of [downstream, upstream]) {
        sockets.add(socket); socket.on('close', () => sockets.delete(socket));
        socket.on('error', () => { downstream.destroy(); upstream.destroy(); });
      }
      downstream.pipe(upstream).pipe(downstream);
    });
    tunnels.push(server);
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
    return `http://127.0.0.1:${(server.address() as net.AddressInfo).port}`;
  }
  async function connect(url: string) {
    const client = new Client({ name: 'desktop-test', version: '1' }); clients.push(client);
    const transport = new StreamableHTTPClientTransport(new URL(url));
    await client.connect(transport); return { client, transport };
  }
  function status(url: string, headers: Record<string, string> = {}): Promise<number> {
    return new Promise((resolve, reject) => {
      const req = request(url, { headers }, res => { res.resume(); resolve(res.statusCode!); });
      req.on('error', reject); req.end();
    });
  }
  setup(() => { forwarding = new McpForwarding(create, tunnel); });
  teardown(async () => {
    await Promise.allSettled(clients.splice(0).map(client => client.close()));
    await forwarding.close();
    await Promise.allSettled(endpoints.splice(0).map(endpoint => endpoint.close()));
    for (const socket of sockets) socket.destroy();
    await Promise.all(tunnels.splice(0).map(server => new Promise<void>(resolve => server.close(() => resolve()))));
  });

  test('a real TCP forward with a different port accepts an SDK client and exact Host/Origin pairs', async () => {
    const url = await forwarding.open();
    const internal = endpoints[0].connection!.url;
    assert.notEqual(url, internal);
    const { client } = await connect(url);
    assert.deepEqual((await client.callTool({ name: 'read_fixture', arguments: {} })).content,
      [{ type: 'text', text: 'remote graph' }]);
    assert.equal(forwarding.sessionCount, 1);
    const externalOrigin = new URL(url).origin, internalOrigin = new URL(internal).origin;
    assert.equal(await status(url, { Origin: externalOrigin }), 400, 'known origin reaches protocol checks');
    assert.equal(await status(internal, { Origin: internalOrigin }), 400, 'remote agents still work');
    assert.equal(await status(url, { Origin: internalOrigin }), 403, 'never mix internal Origin with external Host');
    assert.equal(await status(internal, { Origin: externalOrigin }), 403);
    assert.equal(await status(url, { Host: '127.0.0.1:1' }), 403);
    assert.equal(await status(url, { Origin: 'https://evil.example' }), 403);
    assert.throws(() => endpoints[0].allowForwardedOrigin('http://127.0.0.1:2'), /before connecting/);
  });

  test('re-forwarding revokes old sessions; another window never inherits the desktop authority', async () => {
    const old = await forwarding.open();
    const { transport } = await connect(old);
    const id = transport.sessionId!;
    const next = await forwarding.open();
    assert.notEqual(next, old);
    assert.equal(await status(next, { 'Mcp-Session-Id': id }), 404);
    const remote = create(); const own = await remote.start();
    assert.equal(await status(own.url, { Host: new URL(next).host }), 403);
    assert.equal((await (await connect(own.url)).client.listTools()).tools.length, 1);
    await forwarding.close();
    assert.equal(forwarding.url, undefined);
    assert.equal(forwarding.sessionCount, 0);
    assert.equal((await (await connect(own.url)).client.listTools()).tools.length, 1, 'revoking desktop access leaves the separate remote endpoint live');
  });

  test('disconnect during tunnel resolution cannot resurrect an endpoint', async () => {
    let release!: (origin: string) => void;
    let resolving!: () => void;
    const reached = new Promise<void>(resolve => { resolving = resolve; });
    forwarding = new McpForwarding(create, async () => {
      resolving(); return new Promise(resolve => { release = resolve; });
    });
    const opening = forwarding.open(); await reached;
    await forwarding.close(); release('http://127.0.0.1:53128');
    await assert.rejects(opening, /cancelled/);
    assert.equal(endpoints[0].connection, undefined);
    assert.equal(forwarding.url, undefined);
  });

  test('unsupported public forwarding closes its listener without publishing a URL', async () => {
    forwarding = new McpForwarding(create, async () => 'https://public.example');
    await assert.rejects(forwarding.open(), /private loopback/);
    assert.equal(endpoints[0].connection, undefined);
    assert.equal(forwarding.url, undefined);
  });
});
