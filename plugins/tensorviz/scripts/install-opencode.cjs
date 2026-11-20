#!/usr/bin/env node
'use strict';
// Registers a TensorViz plugin's MCP server and model-workflow skill with opencode.
//
//   node install-opencode.cjs (--project <dir> | --global) [--plugin-root <dir>] [--remove] [--write]
//
// The server uses the launch command the plugin declares in its own .mcp.json, so a built source
// plugin runs `node scripts/launch.cjs` and a published catalog plugin runs `scripts/launch.sh`.
// --plugin-root defaults to the plugin that contains this script. Without --write nothing
// changes; the planned actions are printed. --remove deletes only entries and links that point at
// a TensorViz launcher or skill, from any plugin path.

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { isDeepStrictEqual } = require('node:util');

const SERVER_NAME = 'tensorviz';
const SKILL_LINK_NAME = 'tensorviz-model-workflow';
const SCHEMA_URL = 'https://opencode.ai/config.json';
const ROOT_VARIABLE = /\$\{(?:CLAUDE_PLUGIN_ROOT|PLUGIN_ROOT|CURSOR_PLUGIN_ROOT)\}/g;
const LAUNCHER = /[\\/]scripts[\\/](?:launch\.cjs|launch\.sh|tensorviz-bootstrap\.exe)$/;
const SKILL_TARGET = /[\\/]skills[\\/]model-workflow[\\/]?$/;
const USAGE = 'Usage: node install-opencode.cjs (--project <dir> | --global) [--plugin-root <dir>] [--remove] [--write]';

class InstallError extends Error {}

const isObject = value => typeof value === 'object' && value !== null && !Array.isArray(value);
const isOurEntry = entry => isObject(entry) && entry.type === 'local' && Array.isArray(entry.command)
  && entry.command.some(part => typeof part === 'string' && LAUNCHER.test(part));
const isOurSkillLink = target => typeof target === 'string' && SKILL_TARGET.test(target);
// Junctions need no Developer Mode or administrator rights on Windows.
const linkType = platform => (platform === 'win32' ? 'junction' : 'dir');

function parseArgs(argv) {
  const options = { write: false, remove: false, scopes: [], project: null, pluginRoot: null };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--write') options.write = true;
    else if (arg === '--remove') options.remove = true;
    else if (arg === '--global') options.scopes.push('global');
    else if (arg === '--project' || arg === '--plugin-root') {
      const value = argv[++index];
      if (value === undefined || value.startsWith('--')) throw new InstallError(`${arg} needs a directory.\n${USAGE}`);
      if (arg === '--project') { options.scopes.push('project'); options.project = value; } else options.pluginRoot = value;
    } else throw new InstallError(`Unknown option: ${arg}\n${USAGE}`);
  }
  if (options.scopes.length !== 1) throw new InstallError(USAGE);
  return options;
}

function scopePaths(options, env) {
  if (options.scopes[0] === 'global') {
    // opencode follows XDG_CONFIG_HOME for its global directory, as `opencode debug paths` shows.
    const dir = path.join(env.XDG_CONFIG_HOME || path.join(os.homedir(), '.config'), 'opencode');
    return { configs: ['opencode.json', 'opencode.jsonc'].map(name => path.join(dir, name)), skillsDir: path.join(dir, 'skills') };
  }
  const dir = path.resolve(options.project);
  if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) throw new InstallError(`Project directory not found: ${dir}`);
  const configs = ['opencode.json', 'opencode.jsonc', '.opencode/opencode.json', '.opencode/opencode.jsonc'];
  return { configs: configs.map(name => path.join(dir, name)), skillsDir: path.join(dir, '.opencode', 'skills') };
}

function readJson(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch (error) { throw new InstallError(`Could not read ${file}: ${error.message}`); }
}

// Mirrors the plugin's own MCP declaration with its root variable resolved.
function resolveLaunch(pluginRoot, probe) {
  const root = path.resolve(pluginRoot);
  const file = ['.mcp.json', 'mcp.json'].map(name => path.join(root, name)).find(candidate => fs.existsSync(candidate));
  const declaration = file && readJson(file)?.mcpServers?.[SERVER_NAME];
  if (!isObject(declaration)) throw new InstallError(`${root} has no TensorViz MCP declaration (.mcp.json or mcp.json). Pass --plugin-root <TensorViz plugin directory>.`);
  if (declaration.cwd !== undefined) throw new InstallError(`${file} launches relative to a working directory, which this installer does not translate. Windows catalog plugins are not supported.`);
  if (typeof declaration.command !== 'string' || (declaration.args !== undefined && !Array.isArray(declaration.args))) {
    throw new InstallError(`${file} does not declare a local launch command.`);
  }
  const expand = value => {
    const expanded = String(value).replace(ROOT_VARIABLE, () => root);
    if (/\$\{[^}]*\}/.test(expanded)) throw new InstallError(`${file} uses a variable this installer cannot resolve: ${value}`);
    return expanded;
  };
  const command = [declaration.command, ...(declaration.args ?? [])].map(expand);
  const launcher = command.find(part => LAUNCHER.test(part));
  if (!launcher || !fs.existsSync(launcher)) throw new InstallError(`${file} does not start a TensorViz launcher that exists in ${root}.`);
  if (path.isAbsolute(command[0]) && !fs.existsSync(command[0])) throw new InstallError(`Launch command not found: ${command[0]}`);
  const notes = [];
  if (path.basename(launcher) === 'launch.cjs' && !fs.existsSync(path.join(root, 'runtime', 'host.cjs'))) {
    throw new InstallError(`${root} has no built runtime. Build it with \`npm run build --prefix plugin\`, or pass --plugin-root <clone>/distribution/plugin-catalog/plugins/tensorviz to use the published plugin.`);
  }
  if (path.basename(launcher) === 'launch.sh' && fs.readFileSync(launcher, 'utf8').includes('--prepare-runtime')) {
    notes.push(`Before the first connection, and after updating this plugin, fetch and verify its runtime: sh ${JSON.stringify(launcher)} --prepare-runtime (opencode gives a local server 30 seconds to start, which a first download can exceed).`);
  }
  if (command[0] === 'node') {
    const major = probe('node');
    if (major === null) throw new InstallError('opencode starts this plugin with `node` from its PATH, but no `node` was found on this PATH.');
    if (major < 22) throw new InstallError(`opencode starts this plugin with \`node\` from its PATH, which is Node.js ${major} here. TensorViz needs Node.js 22 or newer.`);
    notes.push('opencode must find Node.js 22 or newer on its own PATH; a GUI launch can see a different PATH than this shell.');
  }
  const entry = { type: 'local', command };
  if (isObject(declaration.env)) entry.environment = Object.fromEntries(Object.entries(declaration.env).map(([key, value]) => [key, expand(value)]));
  const skill = path.join(root, 'skills', 'model-workflow');
  return { root, entry, notes, skillSource: fs.existsSync(path.join(skill, 'SKILL.md')) ? skill : null };
}

function majorVersion(command) {
  const result = spawnSync(command, ['--version'], { encoding: 'utf8', timeout: 10000 });
  const match = !result.error && result.status === 0 && /(\d+)\.\d+\.\d+/.exec(result.stdout);
  return match ? Number(match[1]) : null;
}

// Edits the one config file in this scope; several files, or comments, need a manual edit.
function selectConfig(configs) {
  const existing = configs.filter(file => fs.existsSync(file));
  if (existing.length === 0) return { file: configs[0], text: null, config: null };
  if (existing.length > 1) return { blocked: `opencode merges ${existing.join(' and ')} here, so none of them was edited.` };
  const text = fs.readFileSync(existing[0], 'utf8');
  let config;
  try { config = JSON.parse(text); } catch { return { blocked: `${existing[0]} is not plain JSON (it may contain comments), so it was not edited.` }; }
  if (!isObject(config)) throw new InstallError(`${existing[0]} must contain a JSON object.`);
  return { file: existing[0], text, config };
}

// opencode 2 nests servers under mcp.servers; 1.x keeps them directly under mcp.
function configLayout(config, probe) {
  if (isObject(config?.mcp?.servers)) return 'v2';
  if (isObject(config?.mcp) && Object.keys(config.mcp).length > 0) return 'v1';
  return (probe('opencode') ?? 1) >= 2 ? 'v2' : 'v1';
}

function snippet(layout, entry) {
  const servers = { [SERVER_NAME]: entry };
  return JSON.stringify({ mcp: layout === 'v2' ? { servers } : servers }, null, 2);
}

function serialize(config, originalText) {
  const indent = (originalText && /^([ \t]+)\S/m.exec(originalText)?.[1]) || '  ';
  return `${JSON.stringify(config, null, indent)}\n`;
}

function planConfig(selected, launch, remove, probe) {
  const notes = [];
  if (selected.blocked) {
    notes.push(remove
      ? `Manual step: ${selected.blocked} Remove the "${SERVER_NAME}" MCP server by hand.`
      : `Manual step: ${selected.blocked} Add or update this entry by hand:\n${snippet(configLayout(null, probe), launch.entry)}`);
    return { notes, nextText: null };
  }
  const next = selected.config ? structuredClone(selected.config) : { $schema: SCHEMA_URL };
  if (remove) {
    let changed = false;
    for (const [holder, key] of [[next.mcp, `mcp.${SERVER_NAME}`], [next.mcp?.servers, `mcp.servers.${SERVER_NAME}`]]) {
      if (!isObject(holder) || holder[SERVER_NAME] === undefined) continue;
      if (isOurEntry(holder[SERVER_NAME])) { delete holder[SERVER_NAME]; changed = true; notes.push(`remove ${key} from ${selected.file}`); } else notes.push(`leave ${key} in ${selected.file} (it does not run a TensorViz launcher)`);
    }
    return { notes, nextText: changed ? serialize(next, selected.text) : null };
  }
  const layout = configLayout(next, probe);
  if (!isObject(next.mcp)) next.mcp = {};
  if (layout === 'v2' && !isObject(next.mcp.servers)) next.mcp.servers = {};
  const holder = layout === 'v2' ? next.mcp.servers : next.mcp;
  const key = layout === 'v2' ? `mcp.servers.${SERVER_NAME}` : `mcp.${SERVER_NAME}`;
  const existing = holder[SERVER_NAME];
  if (existing !== undefined && !isOurEntry(existing)) throw new InstallError(`${key} in ${selected.file} does not run a TensorViz launcher. Remove it or edit it by hand.`);
  if (isDeepStrictEqual(existing, launch.entry)) return { notes: [`${selected.file} already has the current ${key} entry`], nextText: null };
  holder[SERVER_NAME] = launch.entry;
  notes.push(`${existing ? 'update' : 'add'} ${key} in ${selected.file}: ${launch.entry.command.join(' ')}`);
  return { notes, nextText: serialize(next, selected.text) };
}

function planLink(skillsDir, launch, remove) {
  const link = { path: path.join(skillsDir, SKILL_LINK_NAME), action: null };
  let stat = null;
  try { stat = fs.lstatSync(link.path); } catch { stat = null; }
  const current = stat?.isSymbolicLink() ? fs.readlinkSync(link.path) : null;
  if (remove) {
    if (isOurSkillLink(current)) return { ...link, action: 'remove', note: `remove skill link ${link.path}` };
    return { ...link, note: stat ? `leave ${link.path} in place (not a TensorViz skill link)` : null };
  }
  if (!launch.skillSource) return { ...link, note: `skip the skill: ${launch.root} has no skills/model-workflow` };
  const target = launch.skillSource;
  if (!stat) return { ...link, action: 'create', target, note: `link ${link.path} -> ${target}` };
  if (current === target) return { ...link, note: `skill link ${link.path} already present` };
  if (isOurSkillLink(current)) return { ...link, action: 'replace', target, previous: current, note: `relink ${link.path}: ${current} -> ${target}` };
  throw new InstallError(`${link.path} already exists and is not a TensorViz skill link. Move it aside and rerun.`);
}

function removeLink(file, platform) {
  try { fs.unlinkSync(file); } catch (error) {
    if (platform !== 'win32' || !['EPERM', 'EISDIR'].includes(error.code)) throw error;
    fs.rmdirSync(file); // removes a junction without touching its target
  }
}

function applyLink(link, platform) {
  if (link.action === 'remove') return removeLink(link.path, platform);
  fs.mkdirSync(path.dirname(link.path), { recursive: true });
  if (link.action === 'replace') removeLink(link.path, platform);
  try { fs.symlinkSync(link.target, link.path, linkType(platform)); } catch (error) {
    if (link.action === 'replace') { try { fs.symlinkSync(link.previous, link.path, linkType(platform)); } catch { /* reported below */ } }
    throw error;
  }
}

function freeBackupPath(file) {
  for (let index = 0; ; index++) {
    const candidate = index === 0 ? `${file}.bak` : `${file}.bak.${index}`;
    if (!fs.existsSync(candidate)) return candidate;
  }
}

// Writes the config first, then the link; a failed link restores the config exactly.
function apply(file, originalText, nextText, link, platform) {
  let backup = null;
  if (nextText !== null) {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    if (originalText !== null) { backup = freeBackupPath(file); fs.copyFileSync(file, backup); }
    fs.writeFileSync(file, nextText);
  }
  if (!link.action) return backup;
  try { applyLink(link, platform); } catch (error) {
    if (nextText !== null) {
      if (originalText === null) fs.rmSync(file, { force: true }); else fs.writeFileSync(file, originalText);
      if (backup) fs.rmSync(backup, { force: true });
    }
    throw new InstallError(`Could not ${link.action} the skill link ${link.path}: ${error.message}. Nothing was changed.`);
  }
  return backup;
}

function main(argv, { env = process.env, platform = process.platform, probe = majorVersion, out = process.stdout } = {}) {
  const options = parseArgs(argv);
  const scope = scopePaths(options, env);
  const launch = options.remove ? null : resolveLaunch(options.pluginRoot ?? path.resolve(__dirname, '..'), probe);
  const selected = selectConfig(scope.configs);
  const config = planConfig(selected, launch, options.remove, probe);
  const link = planLink(scope.skillsDir, launch, options.remove);
  const steps = config.notes.filter(note => !note.startsWith('Manual step:'));
  if (link.note) steps.push(link.note);
  if (steps.length === 0) steps.push('nothing to change');
  out.write(`${options.write ? 'Applying' : 'Dry run (pass --write to apply)'}:\n`);
  for (const step of steps) out.write(`  - ${step}\n`);
  for (const note of [...(launch?.notes ?? []), ...config.notes.filter(item => item.startsWith('Manual step:'))]) out.write(`${note}\n`);
  if (!options.write) return;
  const backup = apply(selected.file, selected.text ?? null, config.nextText, link, platform);
  if (backup) out.write(`Backed up the previous config to ${backup}\n`);
  out.write('Done. Restart opencode, then check with: opencode mcp list\n');
}

if (require.main === module) {
  try { main(process.argv.slice(2)); } catch (error) {
    process.stderr.write(`${error instanceof InstallError ? '' : 'TensorViz opencode setup failed: '}${error.message}\n`);
    process.exitCode = 1;
  }
}

module.exports = { main, linkType, isOurEntry, isOurSkillLink, configLayout, serialize, InstallError };
