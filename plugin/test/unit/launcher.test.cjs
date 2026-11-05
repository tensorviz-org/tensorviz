const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const vm = require('node:vm');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');

test('release launcher diagnoses missing/old Node and incomplete installs before starting MCP', async () => {
  const scripts = path.resolve(__dirname, '../../../plugins/tensorviz/scripts');
  const missing = spawnSync('/bin/sh', [path.join(scripts, 'launch.sh')], { env: { PATH: '' }, encoding: 'utf8' });
  assert.equal(missing.status, 127);
  assert.equal(missing.stdout, '');
  assert.match(missing.stderr, /Node.js 22 or newer/);
  let diagnostic = '';
  const processStub = { versions: { node: '20.0.0' }, stderr: { write: message => { diagnostic += message; } } };
  vm.runInNewContext(await fs.readFile(path.join(scripts, 'launch.cjs'), 'utf8'), { process: processStub,
    require() { throw new Error('An unsupported runtime must not load the host'); } });
  assert.equal(processStub.exitCode, 1);
  assert.match(diagnostic, /Node.js 22 or newer/);
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-launcher-'));
  try {
    await fs.cp(scripts, path.join(root, 'scripts'), { recursive: true });
    const incomplete = spawnSync(process.execPath, [path.join(root, 'scripts/launch.cjs')], { encoding: 'utf8' });
    assert.equal(incomplete.status, 1);
    assert.equal(incomplete.stdout, '');
    assert.match(incomplete.stderr, /complete built plugin release/);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});

test('installer shell entry explains missing Node and incomplete packages before loading code', async () => {
  const scripts = path.resolve(__dirname, '../../../plugins/tensorviz/scripts');
  const missing = spawnSync('/bin/sh', [path.join(scripts, 'install.sh')], { env: { PATH: '' }, encoding: 'utf8' });
  assert.equal(missing.status, 127);
  assert.match(missing.stderr, /Install it from https:\/\/nodejs.org/);
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-installer-'));
  try {
    await fs.cp(scripts, path.join(root, 'scripts'), { recursive: true });
    const result = spawnSync('/bin/sh', [path.join(root, 'scripts/install.sh')], { env: { ...process.env, PATH: path.dirname(process.execPath) }, encoding: 'utf8' });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /incomplete tensorViz package/);
    assert.ok(!result.stderr.includes('MODULE_NOT_FOUND'));
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});

test('manifest Node launcher discovers tools from a copied plugin whose path contains spaces', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-launcher-'));
  const plugin = path.join(root, 'plugin with spaces');
  const client = new Client({ name: 'launcher-test', version: '1' });
  try {
    await fs.cp(path.resolve(__dirname, '../../../plugins/tensorviz'), plugin, { recursive: true });
    const { pluginVersion } = await import('../../scripts/release-version.mjs');
    const metadata = JSON.parse(await fs.readFile(path.join(plugin, 'plugin.json'), 'utf8'));
    await fs.writeFile(path.join(plugin, 'plugin.json'), JSON.stringify({ ...metadata, version: pluginVersion() }));
    const manifest = JSON.parse(await fs.readFile(path.join(plugin, '.mcp.json'), 'utf8')).mcpServers.tensorviz;
    assert.equal(manifest.command, 'node');
    assert.deepEqual(manifest.args, ['${CLAUDE_PLUGIN_ROOT}/scripts/launch.cjs']);
    const portable = JSON.parse(await fs.readFile(path.join(plugin, 'mcp.json'), 'utf8')).mcpServers.tensorviz;
    assert.equal(portable.command, 'node');
    assert.deepEqual(portable.args, ['${PLUGIN_ROOT}/scripts/launch.cjs']);
    await client.connect(new StdioClientTransport({ command: manifest.command,
      args: manifest.args.map(arg => arg.replaceAll('${CLAUDE_PLUGIN_ROOT}', plugin)), cwd: root,
      env: { PATH: path.dirname(process.execPath) }, stderr: 'inherit' }));
    assert.ok((await client.listTools()).tools.some(tool => tool.name === 'open_graph'));
  } finally { await client.close(); await fs.rm(root, { recursive: true, force: true }); }
});
