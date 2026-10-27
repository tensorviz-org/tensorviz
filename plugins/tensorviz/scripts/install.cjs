#!/usr/bin/env node
// Explicit Codex installation helper. Never invoked by MCP startup or the skill.
const fs = require('node:fs/promises');
const path = require('node:path');
const { execFile } = require('node:child_process');
const { promisify, isDeepStrictEqual } = require('node:util');
const { createHash, randomUUID } = require('node:crypto');
const { PluginPermissions, permissionSummary } = require('../runtime/permissions.cjs');
const { configureCodexPermissions } = require('./codex-permissions.cjs');
const { withCodexConfig } = require('./codex-config.cjs');
const { planConnections, writeConnections } = require('./codex-connections.cjs');
const installPermissionSummary = `${permissionSummary}\n\nThis installer also sets Codex's default tool approval to Always allow for tensorviz@tensorviz-standalone's tensorviz server. Existing per-tool overrides, disabled tools and organization policies still apply. Other plugins and global sandbox/approval settings are unchanged.`;

const standaloneId = 'tensorviz@tensorviz-standalone';
const companionId = 'tensorviz@tensorviz';
// Keep the installed catalog and permission identity stable across the display rename.
const canonicalHost = host => host === 'standalone' ? 'plugin' : host;
const catalogName = 'tensorviz-standalone';
const execute = promisify(execFile);
const json = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const shellQuote = value => `'${value.replaceAll("'", "'\\''")}'`;
// Auth availability is computed by Codex and changes when a server is disabled.
const manualSettings = entries => entries.map(({ auth_status, ...entry }) => entry).sort((a, b) => a.name.localeCompare(b.name));
const listing = text => {
  try { return JSON.parse(text); }
  catch { throw new Error('Codex returned an unsupported listing. Update the official CLI before installation.'); }
};

function codexCommand() {
  return async args => {
    try {
      const result = await execute('codex', args, { timeout: 30000, maxBuffer: 2 * 1024 * 1024 });
      return result.stdout;
    } catch (error) {
      // Client output may contain credentials. Never echo it or rewrite settings ourselves.
      if (error.code === 'ENOENT') throw new Error('Codex CLI is missing. Install it with npm install -g @openai/codex, then run this installer again.');
      throw new Error('Codex could not complete the command. Check its official CLI and plugin settings, then retry.');
    }
  };
}

async function inventory(root) {
  const files = {};
  async function visit(relative) {
    for (const entry of await fs.readdir(path.join(root, relative), { withFileTypes: true })) {
      if (entry.name === '.gitignore') continue;
      const file = path.join(relative, entry.name);
      if (entry.isDirectory()) await visit(file);
      else if (entry.isFile()) files[file] = createHash('sha256').update(await fs.readFile(path.join(root, file))).digest('hex');
      else throw new Error('Plugin files must be regular files and directories.');
    }
  }
  await visit('');
  return JSON.stringify(Object.entries(files).sort(([a], [b]) => a.localeCompare(b)));
}

async function inspect(run) {
  const catalogs = listing(await run(['plugin', 'marketplace', 'list', '--json']));
  if (!Array.isArray(catalogs.marketplaces) || catalogs.marketplaces.some(entry => typeof entry?.name !== 'string' || !entry.name)) {
    throw new Error('Unsupported Codex listing format. Update the official CLI before installation.');
  }
  // Unscoped listing also queries the remote directory and can time out. Inspect
  // every registered catalog so other tensorViz owners still block migration.
  const plugins = [];
  for (const catalog of catalogs.marketplaces) {
    const result = listing(await run(['plugin', 'list', '--marketplace', catalog.name, '--json']));
    if (!Array.isArray(result.installed)) throw new Error('Unsupported Codex listing format. Update the official CLI before installation.');
    plugins.push(...result.installed);
  }
  // This invocation-only feature flag excludes ALL plugin overlays, revealing manual
  // entries even when an installed plugin uses the same MCP name. No settings write.
  const manual = listing(await run(['--disable', 'plugins', 'mcp', 'list', '--json']));
  if (!Array.isArray(manual)) {
    throw new Error('Unsupported Codex listing format. Update the official CLI before installation.');
  }
  return { plugins, catalogs: catalogs.marketplaces, manual };
}

async function planInstall({ catalog, host, connection, cwd = process.cwd(), replaceConnections = false, replaceCatalog = false, acceptPermissions = false }, run, withConfig = withCodexConfig) {
  host = canonicalHost(host);
  if (host !== undefined && !['plugin', 'existing', 'companion'].includes(host)) throw new Error('Choose --host plugin or --host existing.');
  catalog = await fs.realpath(catalog).catch(() => { throw new Error('The selected release folder is missing. Extract the complete tensorViz archive into a persistent folder and retry.'); });
  const descriptor = await json(path.join(catalog, '.agents/plugins/marketplace.json')).catch(() => { throw new Error('This folder is not a packaged tensorViz release. Extract the complete release archive, or build it with npm run package --prefix plugin.'); });
  if (descriptor.name !== catalogName || descriptor.plugins?.length !== 1
    || descriptor.plugins[0].name !== 'tensorviz' || descriptor.plugins[0].source?.source !== 'local'
    || descriptor.plugins[0].source.path !== './plugins/tensorviz') throw new Error('Expected the packaged tensorViz plugin catalog.');
  const pluginRoot = path.join(catalog, 'plugins/tensorviz');
  const manifest = await json(path.join(pluginRoot, 'plugin.json'));
  if (manifest.name !== 'tensorviz' || typeof manifest.version !== 'string') throw new Error('Invalid plugin manifest.');
  for (const kind of ['.codex-plugin', '.claude-plugin']) {
    if ((await json(path.join(pluginRoot, kind, 'plugin.json'))).version !== manifest.version) throw new Error('Plugin manifest versions differ.');
  }
  for (const file of ['scripts/launch.sh', 'runtime/host.cjs', 'runtime/permissions.cjs', 'backend/standalone_server.py', 'browser/index.html']) {
    if (!(await fs.stat(path.join(pluginRoot, file)).catch(() => null))?.isFile()) throw new Error(`The release is incomplete (${file} is missing). Extract the complete archive again; do not install a source-only checkout.`);
  }
  const state = await inspect(run);
  const standalone = state.plugins.find(entry => entry.pluginId === standaloneId);
  const companion = state.plugins.find(entry => entry.pluginId === companionId);
  const unknown = state.plugins.filter(entry => entry.name === 'tensorviz' && ![standaloneId, companionId].includes(entry.pluginId));
  const previous = state.catalogs.filter(entry => entry.name === catalogName);
  const blockers = [], actions = [];
  if (unknown.length) blockers.push('Another tensorViz plugin is installed under a different catalog. Resolve it in Codex before choosing a host.');
  // Recognize endpoint aliases to the installed companion as well as named entries.
  let companionUrl;
  if (companion?.source?.source === 'local' && typeof companion.source.path === 'string') {
    try { companionUrl = (await json(path.join(companion.source.path, '.mcp.json'))).mcpServers?.tensorviz?.url; }
    catch { blockers.push('The companion connection metadata is unavailable. Repair or remove that plugin in Codex first.'); }
  }
  const manual = state.manual.filter(entry => /tensorviz/i.test(entry.name ?? '')
    || entry.transport?.args?.some(arg => typeof arg === 'string' && /[/\\]tensorviz[/\\]/i.test(arg))
    || companionUrl && entry.transport?.url === companionUrl);
  const enabledManual = manual.filter(entry => entry.enabled !== false);
  const choices = [...(companion ? [{ name: companionId, kind: 'plugin', enabled: companion.enabled }] : []),
    ...manual.map(entry => ({ name: entry.name, kind: 'manual', enabled: entry.enabled !== false }))];
  if (!host && choices.some(entry => entry.enabled)) blockers.push('Choose --host plugin to use the independent plugin, or --host existing to keep a current connection.');
  if (host === 'companion') connection = companionId;
  if (host === 'existing' || host === 'companion') {
    connection ??= choices.length === 1 ? choices[0].name : undefined;
    const selected = choices.find(entry => entry.name === connection);
    if (!selected) blockers.push('Select an existing connection with --connection NAME from the listed choices.');
    if (standalone?.enabled) actions.push({ kind: 'connection', section: 'plugins', name: standaloneId, enabled: false });
    for (const choice of choices) {
      if (selected && choice.enabled !== (choice.name === selected.name)) actions.push({ kind: 'connection', section: choice.kind === 'manual' ? 'mcp_servers' : 'plugins', name: choice.name, enabled: choice.name === selected.name });
    }
  } else {
    if (enabledManual.length && !replaceConnections) blockers.push('The manual tensorViz connections listed below are active. Use the guided installer or --replace-connections to preserve and disable them during installation.');
    // Even a disabled manual entry named tensorviz shadows plugin tools in Codex.
    for (const entry of manual.filter(entry => entry.enabled !== false || entry.name === 'tensorviz')) {
      let archiveAs;
      if (entry.name === 'tensorviz') {
        archiveAs = 'tensorviz-previous';
        for (let index = 2; state.manual.some(item => item.name === archiveAs); index++) archiveAs = `tensorviz-previous-${index}`;
        if (!replaceConnections && entry.enabled === false) blockers.push('The disabled manual tensorviz entry shadows plugin tools. Use guided setup or --replace-connections to preserve it under another name.');
      }
      actions.push({ kind: 'connection', section: 'mcp_servers', name: entry.name, enabled: false, ...(archiveAs ? { archiveAs } : {}) });
    }
    if (companion?.enabled) actions.push({ kind: 'connection', section: 'plugins', name: companionId, enabled: false });
    if (previous.length > 1 || previous[0] && previous[0].marketplaceSource?.sourceType !== 'local') {
      blockers.push('The plugin catalog is not a unique local release. Resolve that catalog in Codex before installation.');
    } else if (previous[0]) {
      const current = await fs.realpath(previous[0].marketplaceSource.source).catch(() => null);
      if (current !== catalog) {
        if (!replaceCatalog) blockers.push('Updating or rolling back changes the plugin catalog source. Review previousCatalog and pass --replace-catalog to select this release.');
        actions.push({ kind: 'remove_catalog', name: catalogName });
        actions.push({ kind: 'add_catalog', path: catalog });
      }
    } else actions.push({ kind: 'add_catalog', path: catalog });
    actions.push({ kind: 'install_plugin', pluginId: standaloneId, version: manifest.version });
    if (standalone?.version === manifest.version && standalone.enabled === true) {
      const servers = listing(await run(['mcp', 'list', '--json']));
      let installedRoot = servers.find(entry => entry.name === 'tensorviz' && entry.enabled && entry.transport?.env?.PLUGIN_ROOT)?.transport.env.PLUGIN_ROOT;
      if (!installedRoot && replaceConnections && manual.some(entry => entry.name === 'tensorviz')) {
        const skills = await withConfig(request => request('skills/list', { cwds: [cwd], forceReload: true }));
        const skill = skills.data?.flatMap(entry => entry.skills).find(entry => entry.pluginId === standaloneId && entry.path?.endsWith('/skills/model-workflow/SKILL.md'));
        if (skill) installedRoot = path.resolve(path.dirname(skill.path), '../..');
      }
      if (!installedRoot || await inventory(installedRoot) !== await inventory(pluginRoot)) {
        blockers.push('This version is already installed with different or unverified files. Build a new version; do not replace files used by existing sessions.');
      }
    }
  }
  const report = { status: blockers.length ? 'blocked' : 'ready', host: host ?? 'plugin', version: manifest.version,
    catalog, previousCatalog: previous[0]?.marketplaceSource?.sourceType === 'local' ? previous[0].marketplaceSource.source : null,
    installed: [companion, standalone].filter(Boolean).map(entry => ({ pluginId: entry.pluginId, version: entry.version, enabled: entry.enabled })),
    manual: manual.map(entry => ({ name: entry.name, enabled: entry.enabled })), choices, connection, blockers, actions,
    ...(!['existing', 'companion'].includes(host) ? { permissions: { summary: installPermissionSummary, acceptAtInstallation: acceptPermissions } } : {}) };
  return { report, state, pluginRoot };
}

async function applyInstall(options, run, configure = configureCodexPermissions, withConfig = withCodexConfig) {
  options = { ...options, host: canonicalHost(options.host) };
  if (!options.host) throw new Error('Applying installation requires an explicit --host choice, or an interactive terminal for guided setup.');
  const plan = await planInstall(options, run, withConfig);
  if (plan.report.blockers.length) return plan.report;
  if (!isDeepStrictEqual(await inspect(run), plan.state)) throw new Error('Codex registrations changed during preflight. Rerun before applying.');
  const changes = plan.report.actions.filter(action => action.kind === 'connection').map(({ kind, ...change }) => change);
  const connectionPlan = changes.length ? await withConfig(request => planConnections(request, changes, options.cwd)) : null;
  const rollbackCommand = plan.report.previousCatalog && plan.report.previousCatalog !== plan.report.catalog
    ? `node ${shellQuote(__filename)} --catalog ${shellQuote(plan.report.previousCatalog)} --host plugin --replace-catalog --apply` : undefined;
  let recovery;
  if (connectionPlan) {
    const directory = path.join(new PluginPermissions(options.dataDirectory).directory, 'installations');
    await fs.mkdir(directory, { recursive: true, mode: 0o700 });
    const stat = await fs.lstat(directory);
    if (!stat.isDirectory() || stat.isSymbolicLink() || (stat.mode & 0o077) || process.getuid && stat.uid !== process.getuid()) throw new Error('The installation recovery directory must be private and owned by you.');
    recovery = path.join(directory, `${randomUUID()}.json`);
    await fs.writeFile(recovery, JSON.stringify({ version: 1, connectionPlan, standaloneWasEnabled: plan.state.plugins.some(entry => entry.pluginId === standaloneId && entry.enabled) }), { mode: 0o600, flag: 'wx' });
  }
  try {
    if (connectionPlan) await withConfig(request => writeConnections(request, connectionPlan));
    for (const action of plan.report.actions) {
      if (action.kind === 'remove_catalog') await run(['plugin', 'marketplace', 'remove', action.name]);
      if (action.kind === 'add_catalog') await run(['plugin', 'marketplace', 'add', action.path, '--json']);
      if (action.kind === 'install_plugin') {
        const result = listing(await run(['plugin', 'add', action.pluginId, '--json']));
        if (result.pluginId !== standaloneId || result.version !== plan.report.version || !result.installedPath
          || await inventory(result.installedPath) !== await inventory(plan.pluginRoot)) {
          throw new Error('Codex did not install the exact requested release. Inspect plugin settings and rerun preflight before opening a new chat.');
        }
      }
    }
    const after = await inspect(run);
    const desired = options.host === 'plugin' ? standaloneId : plan.report.connection;
    const expectedManual = plan.state.manual.map(entry => {
      const changed = changes.find(change => change.section === 'mcp_servers' && change.name === entry.name);
      return changed ? { ...entry, name: changed.archiveAs ?? entry.name, enabled: changed.enabled } : entry;
    });
    const manualNames = new Set([...plan.report.manual.map(entry => entry.name), ...changes.map(change => change.archiveAs).filter(Boolean)]);
    const active = [...after.plugins.filter(entry => [standaloneId, companionId].includes(entry.pluginId) && entry.enabled).map(entry => entry.pluginId),
      ...after.manual.filter(entry => manualNames.has(entry.name) && entry.enabled !== false).map(entry => entry.name)];
    const manualMatches = isDeepStrictEqual(manualSettings(after.manual), manualSettings(expectedManual));
    if (active.length !== 1 || active[0] !== desired || !manualMatches) {
      throw new Error(`Installation verification failed. Expected active host ${desired}; found ${active.join(', ') || 'none'}. Manual connection settings ${manualMatches ? 'match' : 'differ from'} the reviewed changes.`);
    }
    if (options.host === 'plugin' && options.acceptPermissions) {
      plan.report.permissions.clientPolicy = await configure();
      await new PluginPermissions(options.dataDirectory).save('granted');
      plan.report.permissions.status = 'granted';
    }
    return { ...plan.report, status: 'installed',
      installed: after.plugins.filter(entry => [standaloneId, companionId].includes(entry.pluginId))
        .map(entry => ({ pluginId: entry.pluginId, version: entry.version, enabled: entry.enabled })),
      ...(recovery ? { recovery, recoveryCommand: `node ${shellQuote(__filename)} --recover ${shellQuote(recovery)}` } : {}),
      ...(rollbackCommand ? { rollbackCommand } : {}),
      nextAction: 'Start a new chat to load the selected host. Existing chats keep their current connection. Keep previous release directories for rollback.' };
  } catch (error) {
    if (recovery) {
      try { await recoverInstall(recovery, run, withConfig); }
      catch { throw new Error(`Installation did not finish. Some settings could not be restored because they changed or Codex is unavailable. Run: node ${shellQuote(__filename)} --recover ${shellQuote(recovery)}`); }
      throw new Error(`Installation did not finish; the previous connections were restored. ${error.message} Retry the same installer command.`);
    }
    if (rollbackCommand) throw new Error(`${error.message} Retry the same installer command, or return to the previous release: ${rollbackCommand}`);
    throw error;
  }
}

async function recoverInstall(file, run, withConfig = withCodexConfig) {
  const stat = await fs.lstat(file);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size > 65536 || (stat.mode & 0o077) || process.getuid && stat.uid !== process.getuid()) throw new Error('Recovery requires the private receipt written by this installer.');
  const record = await json(file);
  if (record.version !== 1 || !record.connectionPlan?.entries?.length) throw new Error('Invalid installation recovery receipt.');
  await withConfig(request => writeConnections(request, record.connectionPlan, true, true));
  const state = await inspect(run);
  const standalone = state.plugins.find(entry => entry.pluginId === standaloneId);
  if (standalone && standalone.enabled !== record.standaloneWasEnabled) {
    await withConfig(async request => {
      const plan = await planConnections(request, [{ section: 'plugins', name: standaloneId, enabled: record.standaloneWasEnabled }], record.connectionPlan.cwd);
      await writeConnections(request, plan);
    });
  }
  await withConfig(request => writeConnections(request, record.connectionPlan, true));
  return { status: 'restored', nextAction: 'The previous connection choices are restored. Start a new chat, or rerun the installer.' };
}

async function guideInstall(options, run, ask, print = console.log) {
  const selected = { ...options, host: canonicalHost(options.host) };
  const preview = (await planInstall(selected, run)).report;
  print(`tensorViz ${preview.version}\nInstalled: ${preview.installed.map(entry => `${entry.pluginId} ${entry.version}${entry.enabled ? '' : ' (disabled)'}`).join(', ') || 'none'}`);
  if (!selected.host && preview.choices.some(entry => entry.enabled)) {
    print('1. Use the tensorViz plugin (works without VS Code)');
    preview.choices.forEach((entry, index) => print(`${index + 2}. Keep ${entry.name}${entry.enabled ? '' : ' (currently disabled)'}`));
    const answer = (await ask('Choose a host [1], or 0 to cancel: ')).trim() || '1';
    const choice = Number(answer);
    if (choice === 0) return null;
    if (!/^\d+$/.test(answer) || choice < 1 || choice > preview.choices.length + 1) throw new Error('Choose one of the listed host numbers; no settings changed.');
    selected.host = choice === 1 ? 'plugin' : 'existing';
    if (choice > 1) selected.connection = preview.choices[choice - 2].name;
  }
  selected.host ??= 'plugin';
  selected.replaceConnections = true;
  selected.replaceCatalog = true;
  const plan = (await planInstall(selected, run)).report;
  if (plan.blockers.length) throw new Error(plan.blockers.join('\n'));
  print('\nInstallation changes:');
  for (const action of plan.actions) {
    if (action.kind === 'connection') print(`- ${action.enabled ? 'Enable' : 'Disable and preserve'} ${action.name}${action.archiveAs ? ` as ${action.archiveAs} so it cannot shadow the plugin` : ''}.`);
    if (action.kind === 'install_plugin') print(`- Install tensorViz ${action.version}; preserve saved model inputs, environments and consent.`);
    if (action.kind === 'add_catalog') print(`- Use release ${action.path}.`);
  }
  if (plan.previousCatalog && plan.previousCatalog !== plan.catalog && selected.host === 'plugin') print(`Previous release for rollback: ${plan.previousCatalog}`);
  const needsPermission = selected.host === 'plugin' && await new PluginPermissions(selected.dataDirectory).status() !== 'granted';
  if (needsPermission) print(`\n${installPermissionSummary}`);
  const answer = await ask(needsPermission ? '\nInstall and allow tensorViz? [y/N] ' : '\nApply these installation changes? [y/N] ');
  if (!/^(y|yes)$/i.test(answer.trim())) return null;
  if (needsPermission) selected.acceptPermissions = true;
  return selected;
}

async function main() {
  let options = { catalog: path.resolve(__dirname, '../../..') };
  let apply = false, recovery, guided = false;
  for (let index = 2; index < process.argv.length; index++) {
    const arg = process.argv[index];
    if (arg === '--apply') apply = true;
    else if (arg === '--sessions-stopped') options.sessionsStopped = true;
    else if (arg === '--replace-catalog') options.replaceCatalog = true;
    else if (arg === '--replace-connections') options.replaceConnections = true;
    else if (arg === '--accept-permissions') options.acceptPermissions = true;
    else if (arg === '--host' && process.argv[index + 1]) options.host = process.argv[++index];
    else if (arg === '--catalog' && process.argv[index + 1]) options.catalog = process.argv[++index];
    else if (arg === '--connection' && process.argv[index + 1]) options.connection = process.argv[++index];
    else if (arg === '--recover' && process.argv[index + 1]) recovery = path.resolve(process.argv[++index]);
    else throw new Error('Usage: node install.cjs [--catalog RELEASE] [--host plugin|existing] [--connection NAME] [--replace-catalog] [--replace-connections] [--accept-permissions] [--apply] | --recover RECEIPT');
  }
  const run = codexCommand();
  if (recovery) { console.log(JSON.stringify(await recoverInstall(recovery, run), null, 2)); return; }
  if (apply && !options.acceptPermissions && process.stdin.isTTY && process.stdout.isTTY) {
    guided = true;
    const readline = require('node:readline/promises').createInterface({ input: process.stdin, output: process.stdout });
    try { options = await guideInstall(options, run, question => readline.question(question)); }
    finally { readline.close(); }
    if (!options) { console.log('Installation cancelled. No settings changed.'); return; }
  }
  const report = apply ? await applyInstall(options, run) : (await planInstall(options, run)).report;
  if (guided && report.status === 'installed') {
    console.log(`\ntensorViz setup complete. Selected host: ${report.host === 'plugin' ? 'tensorViz plugin ' + report.version : report.connection}.\n${report.nextAction}`);
    if (report.recoveryCommand) console.log(`Restore the previous connection choice:\n${report.recoveryCommand}`);
    if (report.rollbackCommand) console.log(`Return to the previous release:\n${report.rollbackCommand}`);
  } else console.log(JSON.stringify(report, null, 2));
  if (report.status === 'blocked') process.exitCode = 2;
}

module.exports = { planInstall, applyInstall, guideInstall, recoverInstall };
if (require.main === module) main().catch(error => { console.error(error.message); process.exitCode = 1; });
