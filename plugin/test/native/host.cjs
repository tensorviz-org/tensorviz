const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

test('packaged host has no readable original source and can revoke permission without starting a model', async () => {
  const { nativePlatform } = await import('../../scripts/native-platforms.mjs');
  const binary = path.resolve(__dirname, '../../out/native-runtime', nativePlatform().host);
  const state = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-native-permission-'));
  try {
    const env = { ...(process.env.SystemRoot ? { SystemRoot: process.env.SystemRoot } : {}), PATH: '', TENSORVIZ_DATA_DIR: state };
    const info = spawnSync(binary, ['--verify-native-package'], { env, encoding: 'utf8' });
    assert.equal(info.status, 0, info.stderr);
    assert.equal(JSON.parse(info.stdout).sourceIncluded, false);
    assert.deepEqual(await fs.readdir(state), [], 'Inspecting the binary must not create consent or runtime state');
    const revoked = spawnSync(binary, ['--revoke-permission'], { env, encoding: 'utf8' });
    assert.equal(revoked.status, 0, revoked.stderr);
    assert.equal(revoked.stdout, '');
    assert.equal(JSON.parse(await fs.readFile(path.join(state, 'permissions/standalone.json'))).decision, 'denied');
    assert.deepEqual(await fs.readdir(state), ['permissions']);
  } finally { await fs.rm(state, { recursive: true, force: true }); }
});
