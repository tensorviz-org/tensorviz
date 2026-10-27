const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

test('release versions advance for each run, retain their identity on retry and reject invalid inputs', async () => {
  const { releaseVersion } = await import('../../scripts/release-version.mjs');
  assert.equal(releaseVersion('0.3.0', '12'), '0.3.12');
  assert.equal(releaseVersion('0.3.0', '12'), '0.3.12');
  assert.equal(releaseVersion('0.3.0', '13'), '0.3.13');
  for (const value of ['0', '-1', '1\n2', '1.5', '9007199254740992']) assert.throws(() => releaseVersion('0.3.0', value));
  assert.throws(() => releaseVersion('0.3.0-dev', '1'));
});

test('previous catalogs retain exact source exclusion and checksums during documentation migration', async () => {
  const { exportPublic } = await import('../../scripts/export-public.mjs');
  const { pluginVersion } = await import('../../scripts/release-version.mjs');
  const { auditIntegration } = await import('../../scripts/release-candidate.mjs');
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-previous-catalog-'));
  const output = path.join(root, 'integration');
  try {
    await exportPublic(output, { runtimeLock: { formatVersion: 1, version: pluginVersion(),
      platform: 'darwin-arm64', pythonABI: 'cpython-312-darwin', bytes: 100,
      sha256: 'a'.repeat(64), url: 'https://example.invalid/runtime.tar.gz' } });
    const publicationFile = path.join(output, 'PUBLICATION.json');
    const publication = JSON.parse(await fs.readFile(publicationFile, 'utf8'));
    for (const file of ['plugins/tensorviz/docs/installation.md', 'cursor/plugins/tensorviz/docs/installation.md', 'docs/installation.md']) {
      await fs.rm(path.join(output, file));
      delete publication.files[file];
      await fs.writeFile(publicationFile, JSON.stringify(publication));
      if (Object.keys(publication.files).length === 40) continue;
      await assert.rejects(auditIntegration(output), /allowlist/);
      await auditIntegration(output, { allowPrevious: true });
      await fs.writeFile(path.join(output, 'core.py'), 'private source');
      await assert.rejects(auditIntegration(output, { allowPrevious: true }), /allowlist/);
      await fs.rm(path.join(output, 'core.py'));
    }
    await fs.appendFile(path.join(output, 'README.md'), 'unexpected edit');
    await assert.rejects(auditIntegration(output, { allowPrevious: true }), /checksum mismatch/);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});
