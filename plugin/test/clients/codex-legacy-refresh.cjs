// Opt-in installed-client regression. Supply a complete official public catalog
// clone; never use a user's CODEX_HOME. This does not run a provider/agent turn.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { execFileSync, spawnSync } = require('node:child_process');

test('Codex explicitly replaces the macOS-only 0.3.104 cache after catalog refresh', {
  skip: !process.env.TENSORVIZ_LEGACY_PUBLIC_CATALOG,
}, async () => {
  const source = process.env.TENSORVIZ_LEGACY_PUBLIC_CATALOG;
  assert.ok(path.isAbsolute(source));
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-legacy-refresh-'));
  try {
    const catalog = path.join(temp, 'catalog'), home = path.join(temp, 'home');
    const codexHome = path.join(home, '.codex');
    await fs.mkdir(codexHome, { recursive: true });
    const git = args => execFileSync('git', args, { encoding: 'utf8' }).trim();
    git(['clone', '--no-hardlinks', source, catalog]);
    const latest = git(['-C', catalog, 'rev-parse', 'HEAD']);
    git(['-C', catalog, 'checkout', '--detach', 'v0.3.104']);
    assert.equal(git(['-C', catalog, 'rev-parse', 'HEAD']), 'e3396fba3bf69af75cbf041882e7ccfaecc9c0b9');
    const cli = process.env.TENSORVIZ_TEST_CODEX || 'codex';
    const env = { PATH: process.env.PATH, HOME: home, CODEX_HOME: codexHome };
    const run = args => execFileSync(cli, args, { env, cwd: temp, encoding: 'utf8', timeout: 30000 });
    await fs.writeFile(path.join(codexHome, 'config.toml'), '# preserve user settings\nmodel = "preserved"\n');
    run(['plugin', 'marketplace', 'add', catalog]);
    const old = JSON.parse(run(['plugin', 'add', 'tensorviz@tensorviz-standalone', '--json']));
    assert.equal(old.version, '0.3.104');
    if (process.platform === 'linux') {
      const failed = spawnSync('sh', [path.join(old.installedPath, 'scripts/launch.sh')], { env, encoding: 'utf8' });
      assert.equal(failed.status, 1);
      assert.match(failed.stderr, /supports macOS Apple Silicon/);
    }
    git(['-C', catalog, 'checkout', '--detach', latest]);
    const advertised = JSON.parse(await fs.readFile(path.join(catalog, 'plugins/tensorviz/plugin.json'), 'utf8'));
    const servers = JSON.parse(run(['mcp', 'list', '--json']));
    assert.ok(servers.find(s => s.name === 'tensorviz').transport.args.some(a => a.includes('/0.3.104/')),
      'catalog changes alone must not be mistaken for installed-cache replacement');
    const updated = JSON.parse(run(['plugin', 'add', 'tensorviz@tensorviz-standalone', '--json']));
    assert.equal(updated.version, advertised.version);
    assert.notEqual(updated.installedPath, old.installedPath);
    const actual = JSON.parse(await fs.readFile(path.join(updated.installedPath, 'runtime-lock.json'), 'utf8'));
    assert.equal(actual.version, advertised.version);
    assert.equal(actual.runtimes['linux-x64'].platform, 'linux-x64');
    const current = JSON.parse(run(['mcp', 'list', '--json'])).find(s => s.name === 'tensorviz');
    assert.ok(current.transport.args.includes(path.join(updated.installedPath, 'scripts/launch.sh')));
    assert.match(await fs.readFile(path.join(codexHome, 'config.toml'), 'utf8'), /model = "preserved"/);
  } finally { await fs.rm(temp, { recursive: true, force: true }); }
});
