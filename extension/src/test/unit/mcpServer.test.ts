import * as assert from 'node:assert/strict';
import { request } from 'node:http';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { GraphMcpServer, type McpConnection } from '../../mcpServer';

suite('standard MCP HTTP transport', () => {
  let server: GraphMcpServer;
  let connection: McpConnection;
  const clients: Client[] = [];
  let disposed = 0;
  setup(async () => {
    disposed = 0;
    server = new GraphMcpServer(mcp => {
      mcp.registerTool('read_fixture', { description: 'Read a fixture', inputSchema: {}, annotations: { readOnlyHint: true } }, async () => ({ content: [{ type: 'text', text: 'graph facts' }] }));
      return () => { disposed++; };
    });
    connection = await server.start();
  });
  teardown(async () => { await Promise.allSettled(clients.splice(0).map(client => client.close())); await server.close(); });
  async function connect(name: string) {
    const client = new Client({ name, version: '1.0' });
    const transport = new StreamableHTTPClientTransport(new URL(connection.url));
    clients.push(client);
    await client.connect(transport);
    return { client, transport };
  }

  test('an arbitrary standard client initializes, lists and invokes tools', async () => {
    const { client } = await connect('any-compatible-agent');
    assert.equal(client.getServerVersion()?.name, 'tensorviz');
    assert.deepEqual((await client.listTools()).tools.map(tool => tool.name), ['read_fixture']);
    const result = await client.callTool({ name: 'read_fixture', arguments: {} });
    assert.deepEqual(result.content, [{ type: 'text', text: 'graph facts' }]);
  });

  test('checks origin, host, request size and unknown session before dispatch', async () => {
    assert.equal((await fetch(connection.url, { headers: { Origin: 'https://untrusted.example' } })).status, 403);
    const wrongHost = await new Promise<number>(resolve => {
      const req = request(connection.url, { headers: { Host: 'elsewhere.example' } }, res => { res.resume(); resolve(res.statusCode!); });
      req.end();
    });
    assert.equal(wrongHost, 403);
    assert.equal((await fetch(connection.url)).status, 400);
    assert.equal((await fetch(connection.url, { headers: { 'Mcp-Session-Id': 'missing' } })).status, 404);
    assert.equal((await fetch(connection.url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: 'x'.repeat(65537) })).status, 413);
  });

  test('legacy companion probes do not count as agent connections', async () => {
    const probe = await connect('tensorviz-connection-check');
    await probe.client.listTools();
    assert.equal(server.sessionCount, 0);
    await connect('actual-agent');
    assert.equal(server.sessionCount, 1);
    await probe.transport.terminateSession();
    assert.equal(server.sessionCount, 1);
  });

  test('sessions can terminate and a different client can reconnect', async () => {
    const first = await connect('client-one');
    await first.transport.terminateSession();
    assert.equal(disposed, 1);
    const second = await connect('client-two');
    assert.equal((await second.client.listTools()).tools.length, 1);
  });

  test('window sessions cannot cross endpoints', async () => {
    const other = new GraphMcpServer(() => {});
    const otherConnection = await other.start();
    try {
      assert.notEqual(server.hostId, other.hostId);
      const { transport } = await connect('first-window');
      assert.equal((await fetch(otherConnection.url, { headers: { 'Mcp-Session-Id': transport.sessionId! } })).status, 404);
    } finally { await other.close(); }
  });

  test('a port collision fails without replacing the other endpoint', async () => {
    const other = new GraphMcpServer(() => {});
    try { await assert.rejects(other.start(Number(new URL(connection.url).port)), /EADDRINUSE/); }
    finally { await other.close(); }
    assert.equal((await (await connect('still-live')).client.listTools()).tools.length, 1);
  });

  test('a private Canvas endpoint requires its own bearer token even for an existing session', async () => {
    const privateServer = new GraphMcpServer(mcp => {
      mcp.registerTool('read_canvas', { inputSchema: {} }, async () => ({ content: [{ type: 'text', text: 'canvas' }] }));
    }, undefined, 'canvas-test-secret');
    const { url } = await privateServer.start();
    const client = new Client({ name: 'canvas-agent-test', version: '1' });
    const transport = new StreamableHTTPClientTransport(new URL(url), { requestInit: { headers: { Authorization: 'Bearer canvas-test-secret' } } });
    try {
      assert.equal((await fetch(url)).status, 401);
      assert.equal((await fetch(url, { headers: { Authorization: 'Bearer another-conversation' } })).status, 401);
      await client.connect(transport);
      assert.equal((await client.listTools()).tools[0].name, 'read_canvas');
      assert.equal((await fetch(url, { headers: { 'Mcp-Session-Id': transport.sessionId! } })).status, 401);
    } finally { await client.close(); await privateServer.close(); }
  });

  test('closing during startup cannot leave an orphaned listener', async () => {
    const endpoint = new GraphMcpServer(() => {});
    const startup = endpoint.start();
    const stopped = endpoint.close();
    await assert.rejects(startup, /closed during startup/);
    await stopped;
    assert.equal(endpoint.connection, undefined);
    await endpoint.close();
  });

  test('closing revokes sessions and releases the listening port', async () => {
    const { transport } = await connect('closing-client');
    await server.close();
    assert.equal(disposed, 1);
    const replacement = new GraphMcpServer(() => {});
    try {
      const fresh = await replacement.start(Number(new URL(connection.url).port));
      assert.equal((await fetch(fresh.url, { headers: { 'Mcp-Session-Id': transport.sessionId! } })).status, 404);
    } finally { await replacement.close(); }
  });
});
