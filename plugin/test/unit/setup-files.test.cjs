const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { readSetup, saveSetup } = require('../../out/host/setupFiles.cjs');
const { withResourceLock } = require('../../out/host/locks.cjs');

test('setup saves refuse concurrent edits and link aliases, and serialize actual destinations', async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-setup-write-')));
  const signal = new AbortController().signal;
  try {
    const config = path.join(root, 'tensorviz.toml');
    const original = '# Keep my notes\nversion = 1\n';
    await fs.writeFile(config, original);
    const before = await readSetup(root);
    const prepared = { profile: 'tiny', config: original + '[profiles.tiny]\n', factory: null };
    await fs.writeFile(config, original + '# Another writer\n');
    await assert.rejects(saveSetup(root, before, prepared, signal, async () => true), { code: 'SETUP_CHANGED' });
    assert.match(await fs.readFile(config, 'utf8'), /Another writer/);
    const current = await readSetup(root);
    await withResourceLock(config, signal, async () => {
      await assert.rejects(saveSetup(root, current, prepared, signal, async () => true), { code: 'RESOURCE_BUSY' });
    });
    await saveSetup(root, current, prepared, signal, async () => true);
    assert.equal(await fs.readFile(config, 'utf8'), prepared.config);
    const external = path.join(root, 'other.toml');
    await fs.rename(config, external); await fs.symlink(external, config);
    await assert.rejects(readSetup(root), { code: 'INVALID_DESTINATION' });
    await fs.unlink(config); await fs.link(external, config);
    await assert.rejects(readSetup(root), { code: 'INVALID_DESTINATION' });
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});

test('generated helpers never replace edits and cancelled setup changes no config', async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-helper-')));
  try {
    const before = await readSetup(root);
    const factory = { file: '.tensorviz/setup_Tiny_abcdef123456.py', source: 'def make_model(): pass\n' };
    const prepared = { profile: 'tiny', config: 'version = 1\n', factory };
    const cancelled = new AbortController(); cancelled.abort();
    await assert.rejects(saveSetup(root, before, prepared, cancelled.signal, async () => true));
    assert.equal((await readSetup(root)).identity, null);
    await fs.mkdir(path.join(root, '.tensorviz'));
    await fs.writeFile(path.join(root, factory.file), '# user edited\n');
    await assert.rejects(saveSetup(root, before, prepared, new AbortController().signal, async () => true), { code: 'HELPER_CHANGED' });
    assert.equal((await readSetup(root)).identity, null);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});
