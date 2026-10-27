const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const pluginLicense = require('../../../canvas/scripts/plugin-license.cjs');

test('plugin builds reuse only the configured local tldraw key without importing other settings', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-license-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const file = path.join(root, '.env.local');
  await fs.writeFile(file, 'REACT_APP_TLDRAW_LICENSE_KEY="fixture-license"\nREACT_APP_TEST_PRIVATE=fixture-private\n');
  const env = {};
  const before = { ...process.env };
  assert.equal(pluginLicense(env, file), 'fixture-license');
  assert.deepEqual(env, {});
  assert.deepEqual({ ...process.env }, before, 'Reading the license must not load unrelated local configuration');
  assert.equal(pluginLicense({ REACT_APP_TLDRAW_LICENSE_KEY: 'frontend-key' }, file), 'frontend-key');
  assert.equal(pluginLicense({ TENSORVIZ_BROWSER_LICENSE_KEY: 'override', REACT_APP_TLDRAW_LICENSE_KEY: 'frontend-key' }, file), 'override');
  assert.equal(pluginLicense({ TENSORVIZ_BROWSER_LICENSE_KEY: '' }, file), '', 'An explicit empty override must allow an unlicensed evaluation build');
  assert.equal(pluginLicense({}, path.join(root, 'absent')), '');
  assert.throws(() => pluginLicense({}, root), 'Unreadable license configuration must not silently build without it');
});
