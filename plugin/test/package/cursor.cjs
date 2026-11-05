const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { execFileSync } = require('node:child_process');
const { createHash } = require('node:crypto');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');

test('Cursor package uses its supported path variable and shares the verified engine bytes', { timeout: 30000 }, async () => {
  const { verifyRelease } = await import('../../scripts/package.mjs');
  const version = require('../../package.json').version;
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz cursor package '));
  const client = new Client({ name: 'cursor-package-contract', version: '1' });
  try {
    for (const name of ['tensorviz', 'tensorviz-cursor']) {
      const archive = path.resolve(__dirname, `../../dist/${name}-${version}.tar.gz`);
      const hash = createHash('sha256').update(await fs.readFile(archive)).digest('hex');
      assert.equal((await fs.readFile(`${archive}.sha256`, 'utf8')).split(' ')[0], hash);
      execFileSync('tar', ['-xzf', archive, '-C', root]);
    }
    const base = await verifyRelease(path.join(root, `tensorviz-${version}`));
    const directory = path.join(root, `tensorviz-cursor-${version}`);
    const cursor = await verifyRelease(directory);
    assert.equal(cursor.target, 'cursor');
    assert.deepEqual(cursor.compatibility.verified, [], 'Do not advertise real-client support from a package-only check');
    for (const [name, digest] of Object.entries(base.files)) {
      if (/^plugins\/tensorviz\/(runtime|backend|browser|skills)\//.test(name)) assert.equal(cursor.files[name], digest, `Engine bytes drifted: ${name}`);
    }
    const plugin = path.join(directory, 'plugins/tensorviz');
    const config = JSON.parse(await fs.readFile(path.join(plugin, 'mcp.json'), 'utf8')).mcpServers.tensorviz;
    assert.deepEqual(config.args, ['${CURSOR_PLUGIN_ROOT}/scripts/launch.cjs']);
    const data = path.join(root, 'state');
    await client.connect(new StdioClientTransport({ command: config.command,
      args: config.args.map(arg => arg.replaceAll('${CURSOR_PLUGIN_ROOT}', plugin)), cwd: root,
      env: { PATH: [path.dirname(process.execPath), '/usr/bin', '/bin'].join(path.delimiter), TENSORVIZ_DATA_DIR: data }, stderr: 'pipe' }));
    const tools = (await client.listTools()).tools;
    assert.ok(tools.some(tool => tool.name === 'open_graph'));
    assert.ok(tools.some(tool => tool.name === 'open_graph_view'));
    const discovery = await client.callTool({ name: 'list_models', arguments: {} });
    assert.equal(discovery.structuredContent.host.version, version);
    assert.equal(await fs.stat(data).catch(() => null), null, 'Plugin discovery must not provision Python or create consent');
    await fs.writeFile(path.join(plugin, 'mcp.json'), JSON.stringify({ mcpServers: { tensorviz: { ...config, args: ['${PLUGIN_ROOT}/scripts/launch.cjs'] } } }));
    await assert.rejects(verifyRelease(directory), /integrity mismatch/);
    // A self-consistent inventory must still reject the incompatible client path.
    cursor.files['plugins/tensorviz/mcp.json'] = createHash('sha256').update(await fs.readFile(path.join(plugin, 'mcp.json'))).digest('hex');
    await fs.writeFile(path.join(directory, 'release-manifest.json'), JSON.stringify(cursor));
    await assert.rejects(verifyRelease(directory), /Invalid Cursor plugin path contract/);
    cursor.target = 'unknown';
    await fs.writeFile(path.join(directory, 'release-manifest.json'), JSON.stringify(cursor));
    await assert.rejects(verifyRelease(directory), /Unknown release target/);
  } finally { await client.close(); await fs.rm(root, { recursive: true, force: true }); }
});
