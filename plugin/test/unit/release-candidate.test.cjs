const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { execFileSync } = require('node:child_process');

async function candidateFixture(t, { dirty = false, urlTag = 'plugin-v' } = {}) {
  const { publicFiles } = await import('../../scripts/export-public.mjs');
  const { sha256, createReleaseCandidate, verifyReleaseCandidate } = await import('../../scripts/release-candidate.mjs');
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-candidate-audit-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const version = '1.2.3', sourceRevision = 'a'.repeat(40), archiveName = `tensorviz-runtime-${version}-linux-x64.tar.gz`;
  const native = path.join(root, 'runtime'); await fs.mkdir(native);
  await fs.writeFile(path.join(native, 'tensorviz-host'), 'synthetic runtime');
  const manifest = { version, sourceRevision, platform: 'linux-x64', pythonABI: 'cpython-312-x86_64-linux-gnu', coreSourceIncluded: false,
    rendererDistributionCoverage: 'owner-confirmed', workingTreeChanged: dirty, files: { 'tensorviz-host': sha256('synthetic runtime') } };
  if (dirty === 'missing') delete manifest.workingTreeChanged;
  await fs.writeFile(path.join(native, 'release-manifest.json'), JSON.stringify(manifest));
  const archive = path.join(root, archiveName);
  execFileSync('tar', ['-czf', archive, '-C', native, 'tensorviz-host', 'release-manifest.json']);
  const bytes = await fs.readFile(archive);
  const runtime = { formatVersion: 1, version, platform: manifest.platform, pythonABI: manifest.pythonABI, bytes: bytes.length, sha256: sha256(bytes),
    url: `https://github.com/tensorviz-org/tensorviz/releases/download/${urlTag}${version}/${archiveName}` };
  await fs.writeFile(`${archive}.json`, JSON.stringify(runtime));
  const integration = path.join(root, 'integration'), files = {};
  for (const file of publicFiles) {
    const content = file.endsWith('/runtime-lock.json') ? JSON.stringify(runtime) : `audited fixture: ${file}`;
    await fs.mkdir(path.dirname(path.join(integration, file)), { recursive: true });
    await fs.writeFile(path.join(integration, file), content); files[file] = sha256(content);
  }
  await fs.writeFile(path.join(integration, 'PUBLICATION.json'), JSON.stringify({ version, sourceRevision, runtime, files }));
  const directory = await createReleaseCandidate(integration, [archive]);
  return { directory, root, sha256, verifyReleaseCandidate, catalog: path.join(directory, `tensorviz-catalog-${version}.zip`), publicFiles };
}

test('catalog ZIP contains exact audited integration including local client catalogs and locks', async t => {
  const f = await candidateFixture(t); await f.verifyReleaseCandidate(f.directory, { publishing: true });
  const files = execFileSync('unzip', ['-Z1', f.catalog], { encoding: 'utf8' }).trim().split('\n');
  assert.deepEqual(files.sort(), [...f.publicFiles, 'PUBLICATION.json'].sort());
  assert.ok(files.includes('.claude-plugin/marketplace.json'));
  assert.ok(files.includes('.agents/plugins/marketplace.json'));
  assert.ok(files.includes('plugins/tensorviz/runtime-lock.json'));
});
for (const dirty of [true, 'missing']) test(`publishing rejects dirty or missing clean provenance (${dirty})`, async t => {
  const f = await candidateFixture(t, { dirty });
  await assert.rejects(f.verifyReleaseCandidate(f.directory, { publishing: true }), /clean reviewed source/);
});
test('publishing rejects legacy extension tag namespace', async t => {
  const f = await candidateFixture(t, { urlTag: 'v' });
  await assert.rejects(f.verifyReleaseCandidate(f.directory, { publishing: true }), /exact distribution release URL/);
});
for (const mutation of ['extra', 'changed']) test(`catalog audit rejects ${mutation} ZIP content even if outer checksum is rewritten`, async t => {
  const f = await candidateFixture(t);
  const file = mutation === 'extra' ? 'unexpected.txt' : 'README.md';
  await fs.writeFile(path.join(f.root, file), 'tampered');
  execFileSync('zip', ['-q', f.catalog, file], { cwd: f.root });
  const releasePath = path.join(f.directory, 'release.json');
  const release = JSON.parse(await fs.readFile(releasePath));
  release.files[path.basename(f.catalog)] = f.sha256(await fs.readFile(f.catalog));
  await fs.writeFile(releasePath, JSON.stringify(release));
  await assert.rejects(f.verifyReleaseCandidate(f.directory), mutation === 'extra' ? /catalog ZIP differs/ : /catalog ZIP checksum mismatch/);
});

test('release build derives source repository plugin-v URLs and refuses unsafe configuration', async () => {
  const { releaseBuildConfig } = await import('../../scripts/build-release.mjs');
  const env = { TENSORVIZ_REVIEWED_SOURCE_SHA: 'a'.repeat(40), TENSORVIZ_COMPILER_PYTHON: '/venv/bin/python',
    TLDRAW_DISTRIBUTION_CONFIRMED: 'true', TLDRAW_LICENSE_KEY: 'fixture-not-a-real-license' };
  assert.equal(releaseBuildConfig(env, '1.2.3', 'linux-x64').url, 'https://github.com/tensorviz-org/tensorviz/releases/download/plugin-v1.2.3/tensorviz-runtime-1.2.3-linux-x64.tar.gz');
  for (const patch of [{ TENSORVIZ_REVIEWED_SOURCE_SHA: 'main' }, { TENSORVIZ_COMPILER_PYTHON: 'python' },
    { TLDRAW_DISTRIBUTION_CONFIRMED: 'false' }, { TLDRAW_LICENSE_KEY: '' }, { TENSORVIZ_RELEASE_VERSION: '9.0.0' }]) {
    assert.throws(() => releaseBuildConfig({ ...env, ...patch }, '1.2.3'));
  }
});
