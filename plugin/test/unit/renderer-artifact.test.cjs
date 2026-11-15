const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');

async function fixture(t) {
  const api = await import('../../scripts/renderer-artifact.mjs');
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-shared-renderer-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const browser = path.join(root, 'source'), directory = path.join(root, 'artifact');
  await fs.mkdir(browser);
  await fs.writeFile(path.join(browser, 'index.html'), 'browser');
  await fs.writeFile(path.join(browser, 'panel.html'), 'panel');
  await fs.writeFile(path.join(browser, 'main.js'), 'fixture-renderer-key');
  const notices = path.join(root, 'notices.md');
  await fs.writeFile(notices, 'Original renderer dependency terms');
  const expected = { version: '1.2.3', sourceRevision: 'a'.repeat(40), key: 'fixture-renderer-key' };
  await api.writeRendererArtifact(browser, notices, directory, expected);
  return { ...api, root, directory, expected };
}
test('one shared renderer installs exact browser/panel bytes and retains notices', async t => {
  const f = await fixture(t), destination = path.join(f.root, 'installed');
  await f.installRendererArtifact(f.directory, destination, f.expected);
  assert.equal(await fs.readFile(path.join(destination, 'panel.html'), 'utf8'), 'panel');
  assert.equal(await fs.readFile(path.join(f.directory, 'DEPENDENCY_NOTICES.md'), 'utf8'), 'Original renderer dependency terms');
});
test('native notices include host and renderer terms without Canvas dependencies or foreign build paths', async t => {
  const f = await fixture(t), repository = path.join(f.root, 'native-checkout');
  for (const name of ['plugin/out', 'plugin/licenses', 'plugin/node_modules/host-dependency']) {
    await fs.mkdir(path.join(repository, name), { recursive: true });
  }
  await fs.writeFile(path.join(repository, 'plugin/out/host-inputs.json'), JSON.stringify({ inputs: { 'node_modules/host-dependency/index.js': {} } }));
  await fs.writeFile(path.join(repository, 'plugin/licenses/sources.json'), '[]');
  await fs.writeFile(path.join(repository, 'plugin/node_modules/host-dependency/package.json'), JSON.stringify({ name: 'host-dependency', version: '1.0.0' }));
  await fs.writeFile(path.join(repository, 'plugin/node_modules/host-dependency/LICENSE'), 'Original host dependency terms');
  const output = path.join(repository, 'DEPENDENCY_NOTICES.md');
  await f.writeNativeDependencyNotices(repository, output, { artifact: f.directory, ...f.expected });
  const text = await fs.readFile(output, 'utf8');
  assert.match(text, /Original host dependency terms/);
  assert.match(text, /Original renderer dependency terms/);
  assert.ok(!text.includes(f.root));
  await assert.rejects(fs.stat(path.join(repository, 'canvas/node_modules')), { code: 'ENOENT' });
});
for (const patch of [{ version: '1.2.4' }, { sourceRevision: 'b'.repeat(40) }, { sourceRevision: 'main' }, { key: 'different-key' }]) {
  test(`rejects a shared renderer with mismatched ${Object.keys(patch)[0]}`, async t => {
    const f = await fixture(t);
    await assert.rejects(f.verifyRendererArtifact(f.directory, { ...f.expected, ...patch }), /provenance|production license/);
  });
}
for (const mutation of ['changed', 'missing', 'extra', 'link']) test(`rejects ${mutation} shared renderer content before replacing an installed browser`, async t => {
  const f = await fixture(t), destination = path.join(f.root, 'installed');
  await fs.mkdir(destination); await fs.writeFile(path.join(destination, 'keep.txt'), 'previous browser');
  if (mutation === 'changed') await fs.writeFile(path.join(f.directory, 'browser/panel.html'), 'tampered');
  if (mutation === 'missing') await fs.rm(path.join(f.directory, 'DEPENDENCY_NOTICES.md'));
  if (mutation === 'extra') await fs.writeFile(path.join(f.directory, 'unexpected.txt'), 'untracked');
  if (mutation === 'link') await fs.symlink(path.join(f.directory, 'browser/main.js'), path.join(f.directory, 'browser/link.js'));
  await assert.rejects(f.installRendererArtifact(f.directory, destination, f.expected), /inventory|checksums|link or special/);
  assert.equal(await fs.readFile(path.join(destination, 'keep.txt'), 'utf8'), 'previous browser');
});
