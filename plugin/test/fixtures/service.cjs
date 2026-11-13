const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { ManagedRuntime } = require('../../out/host/runtime.cjs');

// Behavioral suites use a real, prepared service. isolation/open-graph retain
// cold provisioning coverage with empty storage and PATH, including two owners.
async function readyService(plugin, storage) {
  const manager = process.env.TENSORVIZ_TEST_SERVICE_PYTHON;
  assert.ok(manager && path.isAbsolute(manager), 'Set a prepared tool-service interpreter.');
  // Use the production location calculation: Python markers and platform-specific
  // CPU requirements must produce the same cache key as the real host.
  const runtime = new ManagedRuntime(path.join(plugin, 'backend'), storage, {
    approve: async () => { throw new Error('The prepared-service fixture must not provision.'); },
  });
  const { directory, prefix, owner, record, ready } = await runtime.location(storage, 'service');
  await fs.mkdir(directory, { recursive: true });
  await fs.symlink(path.dirname(path.dirname(manager)), prefix, 'dir');
  await fs.writeFile(record, owner);
  await fs.writeFile(ready, owner);
}

module.exports = { readyService };
