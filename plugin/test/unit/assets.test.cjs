const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { pinAssets } = require('../../out/host/assets.cjs');

test('session resources survive deletion of the installed plugin and are removed on disposal', async () => {
  const plugin = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-asset-test-'));
  let session;
  try {
    await fs.writeFile(path.join(plugin, 'plugin.json'), JSON.stringify({ version: '1.0.0' }));
    for (const dir of ['backend', 'browser']) await fs.mkdir(path.join(plugin, dir));
    await fs.writeFile(path.join(plugin, 'backend/server.py'), 'print("fixture")');
    await fs.writeFile(path.join(plugin, 'browser/index.html'), '<p>old version</p>');
    await fs.writeFile(path.join(plugin, 'runtime-manifest.json'), '{}');
    session = await pinAssets(plugin, '1.0.0');
    assert.equal((await fs.stat(session.directory)).mode & 0o777, 0o700);
    await fs.rm(plugin, { recursive: true });
    assert.equal(await fs.readFile(path.join(session.directory, 'browser/index.html'), 'utf8'), '<p>old version</p>');
    assert.equal(await fs.readFile(path.join(session.directory, 'backend/server.py'), 'utf8'), 'print("fixture")');
    session.dispose(); session.dispose();
    assert.equal(await fs.stat(session.directory).catch(() => null), null);
  } finally { session?.dispose(); await fs.rm(plugin, { recursive: true, force: true }); }
});

test('version mismatches and symbolic links cannot become session resources', async () => {
  const plugin = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-asset-test-'));
  try {
    await fs.writeFile(path.join(plugin, 'plugin.json'), JSON.stringify({ version: '1.0.0' }));
    await assert.rejects(pinAssets(plugin, '2.0.0'), /No model was run/);
    await fs.symlink(plugin, path.join(plugin, 'backend'));
    await assert.rejects(pinAssets(plugin, '1.0.0'), /No model was run/);
  } finally { await fs.rm(plugin, { recursive: true, force: true }); }
});
