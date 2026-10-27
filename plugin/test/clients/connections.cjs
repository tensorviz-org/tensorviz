const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { withCodexConfig } = require('../../../plugins/tensorviz/scripts/codex-config.cjs');
const { planConnections, writeConnections } = require('../../../plugins/tensorviz/scripts/codex-connections.cjs');

test('Codex host switching preserves credentials, comments and unrelated changes and restores the original choice', { timeout: 45000 }, async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-connections-')));
  const file = path.join(root, 'config.toml');
  const original = '# preserve my choices\n[mcp_servers.tensorviz]\nurl = "http://127.0.0.1:8849/mcp"\nhttp_headers = { Authorization = "synthetic-secret" }\n[mcp_servers.other]\nurl = "http://127.0.0.1:2/mcp"\n[plugins."tensorviz@tensorviz"]\nenabled = true\n';
  try {
    await fs.writeFile(file, original);
    await withCodexConfig(async request => {
      const plan = await planConnections(request, [
        { section: 'mcp_servers', name: 'tensorviz', enabled: false },
        { section: 'plugins', name: 'tensorviz@tensorviz', enabled: false },
      ], root);
      assert.ok(!JSON.stringify(plan).includes('synthetic-secret'));
      await writeConnections(request, plan);
      let config = await request('config/read', { cwd: root, includeLayers: true });
      assert.equal(config.config.mcp_servers.tensorviz.enabled, false);
      assert.equal(config.config.plugins['tensorviz@tensorviz'].enabled, false);
      assert.equal(config.config.mcp_servers.other.enabled, true);
      assert.match(await fs.readFile(file, 'utf8'), /Authorization = "synthetic-secret"/);
      await fs.appendFile(file, '\n# new unrelated preference\n[features]\napps = false\n');
      await writeConnections(request, plan, true);
      assert.equal(await fs.readFile(file, 'utf8'), original + '\n# new unrelated preference\n[features]\napps = false\n');
      await writeConnections(request, plan);
      await fs.writeFile(file, (await fs.readFile(file, 'utf8')).replace('8849', '8850'));
      await assert.rejects(writeConnections(request, plan, true), /newer settings were preserved/);
      config = await request('config/read', { cwd: root, includeLayers: true });
      assert.equal(config.config.mcp_servers.tensorviz.enabled, false);
      assert.equal(config.config.mcp_servers.tensorviz.url, 'http://127.0.0.1:8850/mcp');
    }, { command: process.env.TENSORVIZ_TEST_CODEX || 'codex', env: { ...process.env, CODEX_HOME: root }, cwd: root });
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});

test('an exact-name collision is preserved under a disabled name and restored without exposing credentials', { timeout: 45000 }, async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-connection-archive-')));
  const file = path.join(root, 'config.toml');
  try {
    await fs.writeFile(file, '# preserve unrelated text\nmodel = "test-preserved-model"\n[mcp_servers.tensorviz]\nurl = "http://127.0.0.1:8849/mcp"\nhttp_headers = { Authorization = "synthetic-secret" }\n[mcp_servers.other]\nurl = "http://127.0.0.1:2/mcp"\n');
    await withCodexConfig(async request => {
      const plan = await planConnections(request, [{ section: 'mcp_servers', name: 'tensorviz', enabled: false, archiveAs: 'tensorviz-previous' }], root);
      assert.ok(!JSON.stringify(plan).includes('synthetic-secret'));
      await writeConnections(request, plan);
      const archived = (await request('config/read', { cwd: root, includeLayers: true })).config.mcp_servers;
      assert.equal(archived.tensorviz, undefined, 'A disabled exact-name entry would still shadow the plugin');
      assert.equal(archived['tensorviz-previous'].enabled, false);
      assert.equal(archived['tensorviz-previous'].http_headers.Authorization, 'synthetic-secret');
      await writeConnections(request, plan, true);
      const restored = (await request('config/read', { cwd: root, includeLayers: true })).config.mcp_servers;
      assert.equal(restored.tensorviz.enabled, true);
      assert.equal(restored.tensorviz.http_headers.Authorization, 'synthetic-secret');
      assert.equal(restored['tensorviz-previous'], undefined);
      assert.match(await fs.readFile(file, 'utf8'), /# preserve unrelated text/);
      assert.match(await fs.readFile(file, 'utf8'), /model = "test-preserved-model"/);
      assert.equal(restored.other.url, 'http://127.0.0.1:2/mcp');
    }, { command: process.env.TENSORVIZ_TEST_CODEX || 'codex', env: { ...process.env, CODEX_HOME: root }, cwd: root });
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});
