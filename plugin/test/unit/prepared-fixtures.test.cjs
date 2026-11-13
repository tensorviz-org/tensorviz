const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { ManagedRuntime } = require('../../out/host/runtime.cjs');
const { readyService } = require('../fixtures/service.cjs');
const { copyBuiltPlugin } = require('../fixtures/plugin.cjs');

test('prepared service uses the host cache location, markers and CPU pin without provisioning', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-prepared-fixture-'));
  const previous = process.env.TENSORVIZ_TEST_SERVICE_PYTHON;
  try {
    const plugin = path.join(root, 'plugin'), storage = path.join(root, 'state');
    await fs.mkdir(path.join(plugin, 'backend'), { recursive: true });
    await fs.copyFile(path.resolve(__dirname, '../../../plugins/tensorviz/runtime-manifest.json'), path.join(plugin, 'runtime-manifest.json'));
    await fs.writeFile(path.join(plugin, 'backend/requirements-standalone.txt'), '# fixture\ntorch==2.14.1\ntomli==2.0.1; python_version < "3.11"\n');
    const runtime = new ManagedRuntime(path.join(plugin, 'backend'), storage, { approve: async () => assert.fail('must reuse prepared service') });
    if (!runtime.availability().available) { t.skip('Managed runtime is unavailable on this platform.'); return; }
    const prefix = path.join(root, 'prepared');
    const manager = path.join(prefix, ...(process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python']));
    await fs.mkdir(path.dirname(manager), { recursive: true });
    await fs.writeFile(manager, 'this fixture interpreter must not execute');
    process.env.TENSORVIZ_TEST_SERVICE_PYTHON = manager;
    await readyService(plugin, storage);
    const location = await runtime.location(root, 'service');
    assert.ok(location.packages.includes(`torch==2.14.1${['linux', 'win32'].includes(process.platform) ? '+cpu' : ''}`));
    assert.ok(!location.packages.some(value => value.includes(';')));
    assert.equal(await fs.readFile(location.ready, 'utf8'), location.owner);
    assert.equal(await runtime.provisionService(root, new AbortController().signal), location.interpreter);
  } finally {
    if (previous === undefined) delete process.env.TENSORVIZ_TEST_SERVICE_PYTHON;
    else process.env.TENSORVIZ_TEST_SERVICE_PYTHON = previous;
    await fs.rm(root, { recursive: true, force: true });
  }
});

test('integration plugin copy stamps all manifests with built host version without editing source', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-built-fixture-'));
  const source = path.resolve(__dirname, '../../../plugins/tensorviz');
  const names = ['plugin.json', '.codex-plugin/plugin.json', '.claude-plugin/plugin.json'];
  const before = await Promise.all(names.map(name => fs.readFile(path.join(source, name), 'utf8')));
  try {
    await copyBuiltPlugin(path.join(root, 'plugin'));
    const { version } = JSON.parse(await fs.readFile(path.resolve(__dirname, '../../out/host-version.json'), 'utf8'));
    for (const name of names) assert.equal(JSON.parse(await fs.readFile(path.join(root, 'plugin', name), 'utf8')).version, version);
    assert.deepEqual(await Promise.all(names.map(name => fs.readFile(path.join(source, name), 'utf8'))), before);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});
