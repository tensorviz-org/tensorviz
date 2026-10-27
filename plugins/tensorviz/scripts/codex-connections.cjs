const { createHash } = require('node:crypto');

const fingerprint = entry => {
  const { enabled, ...rest } = entry;
  return createHash('sha256').update(JSON.stringify(rest)).digest('hex');
};
const keyPath = change => `${change.section}.${JSON.stringify(change.name)}.enabled`;
const tablePath = (section, name) => `${section}.${JSON.stringify(name)}`;
const userLayer = config => config.layers?.find(layer => layer.name.type === 'user' && !layer.name.profile);

// Plans contain names and hashes, never connection URLs, headers or credentials.
async function planConnections(request, changes, cwd = process.cwd()) {
  const config = await request('config/read', { cwd, includeLayers: true });
  const user = userLayer(config);
  if (!user) throw new Error('Codex did not expose its user configuration. Update the official CLI and retry.');
  const entries = changes.map(change => {
    if (!['mcp_servers', 'plugins'].includes(change.section) || typeof change.name !== 'string' || typeof change.enabled !== 'boolean') throw new Error('Invalid connection change.');
    const other = config.layers.find(layer => !layer.disabledReason && layer !== user && layer.config?.[change.section]?.[change.name]);
    if (other) throw new Error(`Connection ${change.name} also has project or managed settings. Change its enabled setting at ${other.name.dotCodexFolder ?? other.name.file ?? 'the organization policy source'}, then retry. Codex only supports automatic writes to the user configuration.`);
    const original = user.config?.[change.section]?.[change.name];
    if (!original) throw new Error(`Connection ${change.name} has changed or is not owned by the user configuration. Rerun installation.`);
    if (change.archiveAs && (change.section !== 'mcp_servers' || change.enabled !== false || config.layers.some(layer => layer.config?.mcp_servers?.[change.archiveAs]))) throw new Error('The preserved-connection name is no longer available. Rerun installation.');
    return { ...change, before: original.enabled ?? null, fingerprint: fingerprint(original) };
  });
  return { filePath: user.name.file, cwd, entries };
}

async function writeConnections(request, plan, restore = false, checkOnly = false) {
  const config = await request('config/read', { cwd: plan.cwd, includeLayers: true });
  const user = userLayer(config);
  if (user?.name.file !== plan.filePath) throw new Error('The Codex profile changed. Rerun the installer in the original profile.');
  const edits = [];
  for (const change of plan.entries) {
    const entry = user.config?.[change.section]?.[change.name];
    if (change.archiveAs) {
      const archived = user.config?.mcp_servers?.[change.archiveAs];
      const source = restore ? archived : entry, destination = restore ? entry : archived;
      const expectedEnabled = restore ? false : change.before;
      const desiredEnabled = restore ? change.before : false;
      if (!source && destination && fingerprint(destination) === change.fingerprint && (destination.enabled ?? null) === desiredEnabled) continue;
      if (!source || destination || fingerprint(source) !== change.fingerprint || (source.enabled ?? null) !== expectedEnabled) throw new Error(`Connection ${change.name} changed after planning. Its newer settings were preserved; review it in Codex before retrying.`);
      const value = { ...source };
      if (desiredEnabled === null) delete value.enabled;
      else value.enabled = desiredEnabled;
      edits.push({ keyPath: tablePath('mcp_servers', restore ? change.name : change.archiveAs), value, mergeStrategy: 'replace' },
        { keyPath: tablePath('mcp_servers', restore ? change.archiveAs : change.name), value: null, mergeStrategy: 'replace' });
      continue;
    }
    const target = restore ? change.before : change.enabled;
    const expected = restore ? change.enabled : change.before;
    if (!entry || fingerprint(entry) !== change.fingerprint) throw new Error(`Connection ${change.name} changed after planning. Its newer settings were preserved; review it in Codex before retrying.`);
    const current = entry.enabled ?? null;
    if (current === target) continue; // An interrupted attempt may already have applied this edit.
    if (current !== expected) throw new Error(`Connection ${change.name} was changed elsewhere. Its newer choice was preserved.`);
    edits.push({ keyPath: keyPath(change), value: target, mergeStrategy: 'replace' });
  }
  if (checkOnly) return;
  if (edits.length) {
    const result = await request('config/batchWrite', { filePath: plan.filePath, expectedVersion: user.version, edits, reloadUserConfig: false });
    if (result.status !== 'ok') throw new Error('A client or organization policy overrides the connection choice. Existing settings are retained for recovery.');
  }
  const after = await request('config/read', { cwd: plan.cwd, includeLayers: true });
  for (const change of plan.entries) {
    const intended = (restore ? change.before : change.enabled) !== false;
    const name = change.archiveAs && !restore ? change.archiveAs : change.name;
    if ((after.config?.[change.section]?.[name]?.enabled !== false) !== intended) throw new Error(`Codex did not activate the selected host for ${change.name}. Restart Codex or review its overriding policy.`);
    if (change.archiveAs && after.config?.mcp_servers?.[restore ? change.archiveAs : change.name]) throw new Error('The previous manual connection still shadows the selected host. Restore the prior choice and review Codex settings.');
  }
}

module.exports = { planConnections, writeConnections };
