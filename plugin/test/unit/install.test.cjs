const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { spawnSync } = require('node:child_process');
const { planInstall, applyInstall, guideInstall, recoverInstall } = require('../../../plugins/tensorviz/scripts/install.cjs');
const { PluginPermissions } = require('../../../plugins/tensorviz/runtime/permissions.cjs');

async function fixture(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-install-plan-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const catalog = path.join(root, 'release'), plugin = path.join(catalog, 'plugins/tensorviz');
  for (const dir of ['.agents/plugins', 'plugins/tensorviz/.codex-plugin', 'plugins/tensorviz/.claude-plugin']) await fs.mkdir(path.join(catalog, dir), { recursive: true });
  await fs.writeFile(path.join(catalog, '.agents/plugins/marketplace.json'), JSON.stringify({ name: 'tensorviz-standalone', plugins: [{ name: 'tensorviz', source: { source: 'local', path: './plugins/tensorviz' } }] }));
  for (const file of ['plugin.json', '.codex-plugin/plugin.json', '.claude-plugin/plugin.json']) await fs.writeFile(path.join(plugin, file), JSON.stringify({ name: 'tensorviz', version: '0.1.0' }));
  for (const file of ['scripts/launch.cjs', 'runtime/host.cjs', 'runtime/permissions.cjs', 'backend/standalone_server.py', 'browser/index.html']) {
    await fs.mkdir(path.dirname(path.join(plugin, file)), { recursive: true });
    await fs.writeFile(path.join(plugin, file), 'fixture');
  }
  const state = { plugins: [], catalogs: [{ name: 'openai-curated' }], manual: [] }, calls = [];
  const run = async args => {
    calls.push(args);
    if (args[0] === '--disable') {
      assert.deepEqual(args, ['--disable', 'plugins', 'mcp', 'list', '--json']);
      return JSON.stringify(state.manual);
    }
    if (args[0] === 'mcp') return JSON.stringify([{ name: 'tensorviz', enabled: true, transport: { env: { PLUGIN_ROOT: plugin } } }]);
    if (args[1] === 'list') {
      assert.equal(args[2], '--marketplace', 'Discovery must not query the slow unscoped remote directory');
      return JSON.stringify({ installed: state.plugins.filter(entry => entry.pluginId.endsWith(`@${args[3]}`)) });
    }
    if (args[1] === 'marketplace' && args[2] === 'list') return JSON.stringify({ marketplaces: state.catalogs });
    if (args[1] === 'marketplace' && args[2] === 'add') state.catalogs.push({ name: 'tensorviz-standalone', marketplaceSource: { sourceType: 'local', source: args[3] } });
    else if (args[1] === 'marketplace' && args[2] === 'remove') state.catalogs = state.catalogs.filter(entry => entry.name !== args[3]);
    else if (args[1] === 'remove') state.plugins = state.plugins.filter(entry => entry.pluginId !== args[2]);
    else if (args[1] === 'add') {
      const entry = { pluginId: args[2], name: 'tensorviz', enabled: true, version: '0.1.0' };
      state.plugins = state.plugins.filter(item => item.pluginId !== args[2]).concat(entry);
      return JSON.stringify({ ...entry, installedPath: plugin });
    } else throw new Error(`Unexpected test command: ${args}`);
    return '';
  };
  const withConfig = async action => action(async (method, params) => {
    const raw = { mcp_servers: Object.fromEntries(state.manual.map(entry => [entry.name, { ...entry.transport, ...(entry.enabled !== undefined ? { enabled: entry.enabled } : {}) }])),
      plugins: Object.fromEntries(state.plugins.map(entry => [entry.pluginId, { enabled: entry.enabled }])) };
    if (method === 'config/read') return { config: raw, layers: [{ name: { type: 'user', file: path.join(root, 'config.toml') }, version: 'test-version', config: raw }] };
    assert.equal(method, 'config/batchWrite');
    for (const edit of params.edits) {
      const [, section, quoted, enabledField] = edit.keyPath.match(/^(mcp_servers|plugins)\.(".*")(\.enabled)?$/);
      const name = JSON.parse(quoted);
      if (!enabledField) {
        assert.equal(section, 'mcp_servers');
        state.manual = state.manual.filter(entry => entry.name !== name);
        if (edit.value) {
          const { enabled, ...transport } = edit.value;
          state.manual.push({ name, enabled: enabled ?? true, ...(Object.keys(transport).length ? { transport } : {}) });
        }
        continue;
      }
      const entry = section === 'plugins' ? state.plugins.find(item => item.pluginId === name) : state.manual.find(item => item.name === name);
      if (edit.value === null) delete entry.enabled;
      else entry.enabled = edit.value;
    }
    return { status: 'ok' };
  });
  const options = { catalog, dataDirectory: path.join(root, 'state') };
  const apply = extra => applyInstall({ ...options, ...extra }, run, undefined, withConfig);
  return { root, catalog, plugin, state, calls, run, withConfig, options, apply };
}
const companion = { name: 'tensorviz', pluginId: 'tensorviz@tensorviz', enabled: true, version: '0.0.2' };

test('preflight performs no mutations, and apply requires an explicit host choice', async t => {
  const f = await fixture(t);
  assert.equal((await planInstall({ catalog: f.catalog }, f.run)).report.status, 'ready');
  assert.equal(f.calls.length, 3);
  await assert.rejects(applyInstall({ catalog: f.catalog }, f.run), /explicit --host/);
  const result = await applyInstall({ catalog: f.catalog, host: 'plugin' }, f.run);
  assert.equal(result.status, 'installed');
  assert.equal(result.host, 'plugin');
  assert.deepEqual(f.state.plugins.map(entry => entry.pluginId), ['tensorviz@tensorviz-standalone']);
});

test('installation ignores nested JSON object key order while preserving all manual values', async t => {
  const f = await fixture(t);
  f.state.manual.push({ name: 'unrelated', enabled: true, transport: { type: 'stdio', command: 'node', args: ['server.cjs', '--flag'], env: { FIRST: 'one', SECOND: 'two' } } });
  const reorder = value => Array.isArray(value) ? value.map(reorder)
    : value && typeof value === 'object' ? Object.fromEntries(Object.entries(value).reverse().map(([key, item]) => [key, reorder(item)])) : value;
  let reads = 0;
  const run = async args => {
    const result = await f.run(args);
    return args[0] === '--disable' && ++reads > 1 ? JSON.stringify(reorder(JSON.parse(result))) : result;
  };
  assert.equal((await applyInstall({ ...f.options, host: 'standalone' }, run)).status, 'installed');
  assert.deepEqual(f.state.manual[0].transport.args, ['server.cjs', '--flag']);
});

test('changed manual values and argument order still stop installation before writes', async t => {
  for (const change of [entry => { entry.transport.env.FIRST = 'changed'; }, entry => { entry.transport.args.reverse(); }]) {
    const f = await fixture(t);
    f.state.manual.push({ name: 'unrelated', enabled: true, transport: { command: 'node', args: ['one', 'two'], env: { FIRST: 'original' } } });
    let reads = 0;
    const run = async args => {
      if (args[0] === '--disable' && ++reads === 2) change(f.state.manual[0]);
      return f.run(args);
    };
    await assert.rejects(applyInstall({ ...f.options, host: 'standalone' }, run), /changed during preflight/);
    assert.ok(!f.calls.some(args => args.includes('add')));
  }
});

test('every registered catalog is inspected and an unreadable catalog blocks installation', async t => {
  const f = await fixture(t);
  f.state.catalogs.push({ name: 'custom' });
  f.state.plugins.push({ name: 'tensorviz', pluginId: 'tensorviz@custom', enabled: true });
  assert.equal((await f.apply({ host: 'standalone' })).status, 'blocked');
  assert.deepEqual(f.calls.filter(args => args[1] === 'list').map(args => args[3]), ['openai-curated', 'custom']);
  await assert.rejects(applyInstall({ ...f.options, host: 'standalone' }, args =>
    args[1] === 'list' && args[3] === 'custom' ? '{}' : f.run(args)), /Unsupported Codex listing/);
  assert.ok(!f.calls.some(args => args.includes('add')));
});

test('explicit installation consent is persisted only after a verified successful install', async t => {
  const f = await fixture(t);
  const dataDirectory = path.join(f.root, 'state');
  const permissions = new PluginPermissions(dataDirectory);
  const options = { catalog: f.catalog, host: 'standalone', acceptPermissions: true, dataDirectory };
  const plan = await planInstall(options, f.run);
  assert.match(plan.report.permissions.summary, /without asking again/);
  assert.equal(await permissions.status(), 'missing', 'Preflight must not accept permissions');
  f.state.manual.push({ name: 'tensorviz', enabled: true });
  assert.equal((await applyInstall(options, f.run)).status, 'blocked');
  assert.equal(await permissions.status(), 'missing', 'Blocked installation cannot create consent');
  f.state.manual = [];
  let configured = 0;
  assert.equal((await applyInstall(options, f.run, async () => { configured++; return { status: 'ok' }; })).status, 'installed');
  assert.equal(configured, 1);
  assert.equal(await permissions.status(), 'granted');
});

test('a failed client permission write does not claim installation consent', async t => {
  const f = await fixture(t);
  const dataDirectory = path.join(f.root, 'state');
  await assert.rejects(applyInstall({ catalog: f.catalog, host: 'standalone', acceptPermissions: true, dataDirectory },
    f.run, async () => { throw new Error('Client policy write failed'); }), /Client policy write failed/);
  assert.equal(await new PluginPermissions(dataDirectory).status(), 'missing');
  assert.equal(f.state.plugins[0].pluginId, 'tensorviz@tensorviz-standalone', 'Installation remains visible and recoverable');
});

test('manual credentials remain private and enabled manual TensorViz entries block migration', async t => {
  const f = await fixture(t);
  f.state.manual.push({ name: 'tensorviz', enabled: true, transport: { type: 'streamable_http', url: 'https://private.example', http_headers: { Authorization: 'private-test-secret' } } });
  const original = JSON.stringify(f.state);
  const result = await applyInstall({ catalog: f.catalog, host: 'standalone' }, f.run);
  assert.equal(result.status, 'blocked');
  assert.match(result.blockers.join(' '), /manual/);
  assert.ok(!JSON.stringify(result).includes('private-test-secret'));
  assert.equal(JSON.stringify(f.state), original);
});

test('companion switching preserves both plugin caches and unrelated manual settings', async t => {
  const f = await fixture(t);
  f.state.catalogs.push({ name: 'tensorviz' });
  f.state.plugins.push({ ...companion });
  f.state.manual.push({ name: 'another-server', enabled: true }, { name: 'tensorviz-old', enabled: false });
  const original = JSON.stringify(f.state.manual);
  const defaultPlan = await planInstall({ catalog: f.catalog }, f.run);
  assert.match(defaultPlan.report.blockers.join(' '), /Choose --host/);
  assert.equal((await f.apply({ host: 'standalone' })).status, 'installed');
  assert.equal(JSON.stringify(f.state.manual), original);
  assert.deepEqual(f.state.plugins.filter(entry => entry.enabled).map(entry => entry.pluginId), ['tensorviz@tensorviz-standalone']);
  assert.equal((await f.apply({ host: 'companion' })).status, 'installed');
  assert.deepEqual(f.state.plugins.filter(entry => entry.enabled).map(entry => entry.pluginId), ['tensorviz@tensorviz']);
  assert.equal(f.state.plugins.length, 2);
  assert.ok(!f.calls.some(args => args[1] === 'remove'), 'Changing hosts must not delete assets used by active chats');
});

test('manual migration is reversible and preserves credentials and pre-disabled entries', async t => {
  const f = await fixture(t);
  f.state.manual.push({ name: 'tensorviz', enabled: true, transport: { url: 'http://localhost:8849/mcp', http_headers: { Authorization: 'synthetic-secret' } } }, { name: 'tensorviz-other', enabled: false });
  const original = JSON.stringify([...f.state.manual].sort((a, b) => a.name.localeCompare(b.name)));
  const installed = await f.apply({ host: 'standalone', replaceConnections: true });
  assert.equal(installed.status, 'installed');
  assert.ok(f.state.manual.every(entry => entry.enabled === false));
  assert.ok(!JSON.stringify(installed).includes('synthetic-secret'));
  assert.ok(!(await fs.readFile(installed.recovery, 'utf8')).includes('synthetic-secret'));
  assert.equal((await recoverInstall(installed.recovery, f.run, f.withConfig)).status, 'restored');
  assert.equal(JSON.stringify([...f.state.manual].sort((a, b) => a.name.localeCompare(b.name))), original);
  assert.equal(f.state.plugins[0].enabled, false);
});

test('a failed install restores the existing connection automatically', async t => {
  const f = await fixture(t);
  f.state.manual.push({ name: 'tensorviz', enabled: true });
  const failingRun = args => {
    if (args[0] === 'plugin' && args[1] === 'add') throw new Error('Synthetic install failure');
    return f.run(args);
  };
  await assert.rejects(applyInstall({ ...f.options, host: 'standalone', replaceConnections: true }, failingRun, undefined, f.withConfig), /previous connections were restored/);
  assert.equal(f.state.manual[0].enabled, true);
});

test('guided setup combines explicit host selection and consent and cancellation changes nothing', async t => {
  const f = await fixture(t);
  f.state.manual.push({ name: 'tensorviz', enabled: true });
  const original = JSON.stringify(f.state);
  let answers = ['1', 'yes'];
  const output = [], questions = [];
  const ask = async question => { questions.push(question); return answers.shift(); };
  const options = await guideInstall(f.options, f.run, ask, line => output.push(line));
  assert.equal(options.host, 'plugin');
  assert.equal(options.replaceConnections, true);
  assert.equal(options.acceptPermissions, true);
  assert.equal(questions.length, 2, 'Host selection and one final consent, not separate operation approvals');
  assert.match(output.join('\n'), /Disable and preserve tensorviz/);
  assert.equal(JSON.stringify(f.state), original, 'Review is not application');
  answers = ['1', 'no'];
  assert.equal(await guideInstall(f.options, f.run, ask, () => {}), null);
  answers = ['2', 'yes'];
  const existing = await guideInstall(f.options, f.run, ask, () => {});
  assert.equal(existing.host, 'existing');
  assert.equal(existing.connection, 'tensorviz');
  assert.equal(existing.acceptPermissions, undefined);
  assert.equal(JSON.stringify(f.state), original);
});

test('guided updates reuse consent and do not ask to choose an already-disabled connection', async t => {
  const f = await fixture(t);
  f.state.catalogs.push({ name: 'tensorviz-standalone', marketplaceSource: { sourceType: 'local', source: f.catalog } });
  f.state.plugins.push({ name: 'tensorviz', pluginId: 'tensorviz@tensorviz-standalone', enabled: true, version: '0.1.0' });
  f.state.manual.push({ name: 'tensorviz-previous', enabled: false });
  await new PluginPermissions(f.options.dataDirectory).save('granted');
  const questions = [], output = [];
  const options = await guideInstall(f.options, f.run, async question => { questions.push(question); return 'yes'; }, text => output.push(text));
  assert.equal(options.host, 'plugin');
  assert.equal(options.acceptPermissions, undefined);
  assert.equal(questions.length, 1);
  assert.match(questions[0], /Apply these installation changes/);
  assert.ok(!output.join('\n').includes('Allow TensorViz to work with local models'));
});

test('unknown plugin ownership and changed client state cannot be silently migrated', async t => {
  const f = await fixture(t);
  f.state.catalogs.push({ name: 'custom' });
  f.state.plugins.push({ name: 'tensorviz', pluginId: 'tensorviz@custom' });
  assert.equal((await applyInstall({ catalog: f.catalog, host: 'standalone' }, f.run)).status, 'blocked');
  f.state.plugins = [];
  let inspections = 0;
  const racingRun = async args => {
    if (args[0] === '--disable' && ++inspections === 2) f.state.manual.push({ name: 'tensorviz', enabled: true });
    return f.run(args);
  };
  await assert.rejects(applyInstall({ catalog: f.catalog, host: 'standalone' }, racingRun), /changed during preflight/);
  assert.ok(!f.calls.some(args => args[1] === 'add'));
});

test('catalog replacement is explicit and does not uninstall the active standalone plugin', async t => {
  const f = await fixture(t);
  f.state.catalogs.push({ name: 'tensorviz-standalone', marketplaceSource: { sourceType: 'local', source: path.dirname(f.catalog) } });
  f.state.plugins.push({ name: 'tensorviz', pluginId: 'tensorviz@tensorviz-standalone', version: '0.0.9', enabled: true });
  const options = { catalog: f.catalog, host: 'standalone' };
  assert.equal((await applyInstall(options, f.run)).status, 'blocked');
  assert.equal((await applyInstall({ ...options, replaceCatalog: true }, f.run)).status, 'installed');
  assert.ok(!f.calls.some(args => args[1] === 'remove'), 'Updates must not issue a plugin uninstall');
});

test('unparseable CLI output cannot expose configuration in the diagnostic', async t => {
  const f = await fixture(t);
  await assert.rejects(planInstall({ catalog: f.catalog }, async () => 'private-credential'), error =>
    /unsupported listing/.test(error.message) && !error.message.includes('private-credential'));
});

test('incomplete release reports how to repair the package before modifying Codex', async t => {
  const f = await fixture(t);
  await fs.rm(path.join(f.plugin, 'runtime/host.cjs'));
  await assert.rejects(f.apply({ host: 'standalone' }), /release is incomplete.*Extract the complete archive/);
  assert.equal(f.calls.length, 0);
});

test('a missing Codex CLI produces an actionable installation message', async t => {
  const f = await fixture(t);
  const result = spawnSync(process.execPath, [path.resolve(__dirname, '../../../plugins/tensorviz/scripts/install.cjs'), '--catalog', f.catalog], { env: { ...process.env, PATH: '' }, encoding: 'utf8' });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Codex CLI is missing.*npm install -g @openai\/codex/);
  assert.ok(!result.stderr.includes('ENOENT'));
});
