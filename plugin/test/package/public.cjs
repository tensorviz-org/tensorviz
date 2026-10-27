const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { createHash } = require('node:crypto');

test('public integration export contains no core source, build tree or runtime payload', async () => {
  const { exportPublic } = await import('../../scripts/export-public.mjs');
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-public-check-'));
  const output = path.join(temporary, 'integration');
  const version = (await import('../../scripts/release-version.mjs')).pluginVersion();
  const lock = { formatVersion: 1, version, platform: 'darwin-arm64', pythonABI: 'cpython-312-darwin',
    url: 'https://example.invalid/runtime.tar.gz', sha256: 'a'.repeat(64), bytes: 100 };
  try {
    await assert.rejects(exportPublic(output, { sourceOnly: true }), /retired/);
    await assert.rejects(exportPublic(output, { runtimeLock: { ...lock, url: 'http://example.invalid/runtime' } }), /HTTPS/);
    await exportPublic(output, { runtimeLock: lock });
    const publication = JSON.parse(await fs.readFile(path.join(output, 'PUBLICATION.json'), 'utf8'));
    assert.equal(publication.status, 'compiled-integration');
    for (const [file, expected] of Object.entries(publication.files)) {
      assert.ok(!/(?:^|\/)(?:source|src|backend|backend|runtime|node_modules|\.git|\.tensorviz)(?:\/|$)/.test(file), file);
      assert.ok(!/\.(?:py|pyc|so|ts|tsx|map|cjs|tar\.gz)$/.test(file), file);
      const actual = createHash('sha256').update(await fs.readFile(path.join(output, file))).digest('hex');
      assert.equal(actual, expected, file);
    }
    assert.equal(Object.keys(publication.files).length, 41, 'Any additional public file needs an explicit privacy review');
    for (const prefix of ['', 'plugins/tensorviz/', 'cursor/plugins/tensorviz/']) {
      const readme = await fs.readFile(path.join(output, prefix, 'README.md'), 'utf8');
      const guide = await fs.readFile(path.join(output, prefix, 'docs/installation.md'), 'utf8');
      assert.ok(readme.includes(`tensorviz-${version}.zip`), 'Download links must follow the packaged version');
      assert.ok(readme.includes('## Install in T3 Code'));
      assert.ok(guide.includes('## T3 Code and large graphs'));
      assert.ok(!readme.includes('{{VERSION}}') && !guide.includes('{{VERSION}}'));
    }
    assert.ok(publication.files['plugins/tensorviz/README.md'], 'Submitted plugin folders must carry their own usage and privacy docs');
    assert.ok(publication.files['plugins/tensorviz/scripts/launch.sh']);
    assert.ok(publication.files['.agents/plugins/marketplace.json']);
    for (const file of ['.codex-plugin/plugin.json', 'plugin.json']) {
      const manifest = JSON.parse(await fs.readFile(path.join(output, 'plugins/tensorviz', file)));
      const ui = manifest.interface ?? manifest.extensions['com.openai'].interface;
      assert.equal(ui.category, 'Developer Tools');
      for (const field of ['websiteURL', 'supportURL', 'privacyPolicyURL', 'termsOfServiceURL']) {
        assert.equal(new URL(ui[field]).protocol, 'https:', `Directory reviewers need an accessible ${field}`);
      }
      assert.ok(Array.isArray(ui.defaultPrompt));
    }
    const codex = JSON.parse(await fs.readFile(path.join(output, 'plugins/tensorviz/mcp.json')));
    assert.equal(codex.mcpServers.tensorviz.type, 'stdio', 'Codex requires the portable transport discriminator');
    assert.equal(codex.mcpServers.tensorviz.command, 'sh', 'Preserve the portable command accepted by the real Codex client');
    const cursor = JSON.parse(await fs.readFile(path.join(output, 'cursor/plugins/tensorviz/mcp.json')));
    assert.equal(cursor.mcpServers.tensorviz.args[0], '${CURSOR_PLUGIN_ROOT}/scripts/launch.sh');
    await assert.rejects(exportPublic(output, { runtimeLock: lock }), /EEXIST/);
  } finally { await fs.rm(temporary, { recursive: true, force: true }); }
});
