const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { createHash } = require('node:crypto');

// Behavioral suites use a real, prepared service. isolation/open-graph retain
// cold provisioning coverage with empty storage and PATH, including two owners.
async function readyService(plugin, storage) {
  const manager = process.env.TENSORVIZ_TEST_SERVICE_PYTHON;
  assert.ok(manager && path.isAbsolute(manager), 'Set a prepared tool-service interpreter.');
  const digest = value => createHash('sha256').update(value).digest('hex');
  const manifest = JSON.parse(await fs.readFile(path.join(plugin, 'runtime-manifest.json'), 'utf8'));
  const sha256 = manifest.platforms[`${process.platform}-${process.arch}`].sha256;
  const packages = (await fs.readFile(path.join(plugin, 'backend/requirements-standalone.txt'), 'utf8')).split(/\r?\n/)
    .map(line => line.trim()).filter(line => line && !line.startsWith('#'));
  const directory = path.join(storage, 'service', digest(JSON.stringify([sha256, packages])).slice(0, 24));
  const owner = JSON.stringify({ role: 'service', runtime: sha256, requirements: digest(JSON.stringify(packages)) });
  await fs.mkdir(directory, { recursive: true });
  await fs.symlink(path.dirname(path.dirname(manager)), path.join(directory, 'venv'), 'dir');
  await fs.writeFile(path.join(directory, 'owner.json'), owner);
  await fs.writeFile(path.join(directory, 'ready.json'), owner);
}

module.exports = { readyService };
