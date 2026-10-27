const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

test('the legacy source packager cannot be invoked accidentally', () => {
  const result = spawnSync(process.execPath, [path.resolve(__dirname, '../../scripts/package.mjs')], { encoding: 'utf8' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Source packaging needs --allow-source-development/);
  assert.match(result.stderr, /Release packaging needs --release/);
});

test('native runtime packaging rejects source, sourcemaps, symbolic links and incomplete output', async () => {
  const { auditNativeRuntime } = await import('../../scripts/package-native.mjs');
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-native-audit-'));
  try {
    for (const name of ['worker.py', 'host.cjs', 'browser.js.map', 'module.c', 'compiled.pyc']) {
      await fs.writeFile(path.join(root, name), 'fixture');
      await assert.rejects(auditNativeRuntime(root), /Core source or unsafe file/);
      await fs.rm(path.join(root, name));
    }
    await fs.symlink(os.tmpdir(), path.join(root, 'linked'));
    await assert.rejects(auditNativeRuntime(root), /link or special file/);
    await fs.rm(path.join(root, 'linked'));
    await assert.rejects(auditNativeRuntime(root), /incomplete/);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});
