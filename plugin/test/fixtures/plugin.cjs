const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');

// Match source-development packaging without modifying tracked source manifests.
async function copyBuiltPlugin(destination) {
  const { version } = JSON.parse(await fs.readFile(path.resolve(__dirname, '../../out/host-version.json'), 'utf8'));
  assert.equal(typeof version, 'string', 'Build the plugin host before integration tests.');
  assert.ok(version.length > 0, 'The built host must have a version.');
  await fs.cp(path.resolve(__dirname, '../../../plugins/tensorviz'), destination, { recursive: true });
  const { stampVersion } = await import('../../scripts/package.mjs');
  await stampVersion(destination, version);
}

module.exports = { copyBuiltPlugin };
