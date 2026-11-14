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
      const build = await fs.readFile(path.join(output, prefix, 'BUILD.md'), 'utf8');
      const guide = await fs.readFile(path.join(output, prefix, 'docs/installation.md'), 'utf8');
      assert.ok(readme.includes(`tensorviz-${version}.zip`), 'Download links must follow the packaged version');
      assert.ok(readme.includes('## Install in T3 Code'));
      assert.ok(guide.includes('## T3 Code and large graphs'));
      for (const [name, text] of [['README.md', readme], ['BUILD.md', build], ['docs/installation.md', guide]]) {
        const template = await fs.readFile(path.resolve(__dirname, '../../packaging/public', name), 'utf8');
        assert.equal(text, template.replaceAll('{{VERSION}}', version), `${prefix}${name} must preserve the complete guide`);
        assert.ok(!text.includes('{{VERSION}}'), `${prefix}${name} must substitute the release version`);
      }
      assert.match(readme, /\*\*29\s+tools\*\*: 25 agent tools and four tools reserved for the Model graph app/);
      assert.match(readme, /macOS ARM64\/x64; Linux x64\s+glibc 2\.31\+ or ARM64 glibc 2\.35\+; Windows x64\/ARM64/);
      assert.match(readme, /installed 0\.3\.104 remains Darwin-only/);
      assert.ok(build.includes(`The ${version} host registers **29 MCP tools**`));
      assert.match(build, /expose \*\*25\*\* to the agent/);
      const inventory = build.split('\n').filter(line => line.startsWith('|'));
      const names = lines => [...lines.join('\n').matchAll(/`([a-z_]+)`/g)].map(match => match[1]).sort();
      assert.deepEqual(names(inventory), require('../tool-names.cjs').slice().sort(), 'Document every registered tool exactly once');
      assert.deepEqual(names(inventory.filter(line => line.startsWith('| Model graph app only |'))),
        ['close_graph_panel', 'detach_graph_panel', 'graph_panel_exchange', 'graph_panel_read'], 'Keep app-only transports separate from agent tools');
      assert.ok(build.includes(`The ${version} production pipeline assembles six independently validated native`));
      for (const platform of ['darwin-arm64', 'darwin-x64', 'linux-x64', 'linux-arm64', 'win32-x64', 'win32-arm64']) assert.ok(build.includes('`' + platform + '`'));
      assert.match(build, /CPython 3\.12 ABI must match/);
      assert.match(build, /glibc >=2\.31/);
      assert.match(build, /Windows Claude Code\/Cursor and musl are outside/);
      assert.match(guide, /macOS ARM64\/x64, Linux x64 \(glibc >=2\.31\), Linux ARM64/);
      assert.match(guide, /Version 0\.3\.104 was macOS-only/);
      assert.match(guide, /## 0\.3\.104 source and asset audit — November 7, 2026/);
      assert.match(guide, /installed and enabled\n0\.3\.104/, 'The historical installation audit must retain its audited version');
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

