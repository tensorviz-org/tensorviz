#!/usr/bin/env node
'use strict';
// Registers the TensorViz MCP server and model-workflow skill with opencode.
//
//   node scripts/install-opencode.cjs --project <dir> [--write]
//   node scripts/install-opencode.cjs --global [--write]
//   node scripts/install-opencode.cjs --project <dir> --remove [--write]
//
// Without --write nothing is changed; the planned actions are printed.
// Only entries this script created are touched on --remove.

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const SERVER_NAME = 'tensorviz';
const SKILL_LINK_NAME = 'tensorviz-model-workflow';
const SCHEMA_URL = 'https://opencode.ai/config.json';

const pluginRoot = path.resolve(__dirname, '..');
const launcher = path.join(pluginRoot, 'scripts', 'launch.cjs');
const runtimeEntry = path.join(pluginRoot, 'runtime', 'host.cjs');
const skillSource = path.join(pluginRoot, 'skills', 'model-workflow');

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

function usage() {
  fail('Usage: node scripts/install-opencode.cjs (--project <dir> | --global) [--remove] [--write]');
}

const args = process.argv.slice(2);
const write = args.includes('--write');
const remove = args.includes('--remove');
const isGlobal = args.includes('--global');
const projectAt = args.indexOf('--project');
if (isGlobal === (projectAt >= 0)) usage();

let scopeDir;
if (isGlobal) {
  scopeDir = path.join(os.homedir(), '.config', 'opencode');
} else {
  if (projectAt + 1 >= args.length || args[projectAt + 1].startsWith('--')) usage();
  scopeDir = path.resolve(args[projectAt + 1]);
  if (!fs.existsSync(scopeDir) || !fs.statSync(scopeDir).isDirectory()) fail(`Project directory not found: ${scopeDir}`);
}

const configPath = path.join(scopeDir, 'opencode.json');
const skillsDir = isGlobal ? path.join(scopeDir, 'skills') : path.join(scopeDir, '.opencode', 'skills');
const skillLink = path.join(skillsDir, SKILL_LINK_NAME);

const plannedNotes = [];

if (!remove) {
  if (Number(process.versions.node.split('.')[0]) < 22) fail('TensorViz requires Node.js 22 or newer.');
  if (!fs.existsSync(runtimeEntry)) fail('TensorViz runtime is missing. Install the complete built plugin release.');
}

// A bare `node` matches the plugin's own .mcp.json. Pinning this script's Node binary would
// break when that install is removed; opencode must find Node 22+ on its PATH.
const serverEntry = {
  type: 'local',
  command: ['node', launcher],
  enabled: true,
};

function isOurEntry(entry) {
  return entry && Array.isArray(entry.command) && entry.command.includes(launcher);
}

// ---- opencode.json -------------------------------------------------------
const configExists = fs.existsSync(configPath);
let config = null;
if (configExists) {
  try {
    config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  } catch {
    fail(`${configPath} is not plain JSON (it may contain comments). Add the entry by hand:\n${JSON.stringify({ mcp: { [SERVER_NAME]: serverEntry } }, null, 2)}`);
  }
  if (typeof config !== 'object' || config === null || Array.isArray(config)) fail(`${configPath} must contain a JSON object.`);
}

let configChanged = false;
let backupNeeded = false;
const nextConfig = config ? structuredClone(config) : { $schema: SCHEMA_URL };
nextConfig.mcp = nextConfig.mcp && typeof nextConfig.mcp === 'object' ? nextConfig.mcp : {};

if (remove) {
  if (isOurEntry(nextConfig.mcp[SERVER_NAME])) {
    delete nextConfig.mcp[SERVER_NAME];
    if (Object.keys(nextConfig.mcp).length === 0) delete nextConfig.mcp;
    configChanged = true;
    plannedNotes.push(`remove mcp.${SERVER_NAME} from ${configPath}`);
  } else if (nextConfig.mcp[SERVER_NAME]) {
    plannedNotes.push(`leave mcp.${SERVER_NAME} in ${configPath} (not created by this script)`);
  }
} else {
  const existing = nextConfig.mcp[SERVER_NAME];
  if (existing && !isOurEntry(existing)) fail(`mcp.${SERVER_NAME} already exists in ${configPath} and was not created by this script. Remove it or edit it by hand.`);
  if (JSON.stringify(existing) !== JSON.stringify(serverEntry)) {
    nextConfig.mcp[SERVER_NAME] = serverEntry;
    configChanged = true;
    backupNeeded = configExists;
    plannedNotes.push(`${configExists ? 'update' : 'create'} ${configPath}: mcp.${SERVER_NAME} -> ${serverEntry.command.join(' ')}`);
  } else {
    plannedNotes.push(`${configPath} already has the current mcp.${SERVER_NAME} entry`);
  }
}

// ---- skill symlink -------------------------------------------------------
let linkAction = null;
let linkStat = null;
try { linkStat = fs.lstatSync(skillLink); } catch { linkStat = null; }
if (remove) {
  if (linkStat && linkStat.isSymbolicLink() && fs.readlinkSync(skillLink) === skillSource) {
    linkAction = 'remove';
    plannedNotes.push(`remove skill link ${skillLink}`);
  } else if (linkStat) {
    plannedNotes.push(`leave ${skillLink} in place (not a link created by this script)`);
  }
} else if (!linkStat) {
  linkAction = 'create';
  plannedNotes.push(`link ${skillLink} -> ${skillSource}`);
} else if (linkStat.isSymbolicLink() && fs.readlinkSync(skillLink) === skillSource) {
  plannedNotes.push(`skill link ${skillLink} already present`);
} else {
  fail(`${skillLink} already exists and is not the TensorViz skill link. Move it aside and rerun.`);
}

// ---- report and apply ----------------------------------------------------
process.stdout.write(`${write ? 'Applying' : 'Dry run (pass --write to apply)'}:\n`);
for (const note of plannedNotes) process.stdout.write(`  - ${note}\n`);

if (!write) process.exit(0);

if (configChanged) {
  if (backupNeeded) fs.copyFileSync(configPath, `${configPath}.bak`);
  // --remove only edits the file; it never deletes opencode.json, which may hold other settings.
  fs.writeFileSync(configPath, `${JSON.stringify(nextConfig, null, 2)}\n`);
}

if (linkAction === 'create') {
  fs.mkdirSync(skillsDir, { recursive: true });
  fs.symlinkSync(skillSource, skillLink, 'dir');
} else if (linkAction === 'remove') {
  fs.unlinkSync(skillLink);
}

process.stdout.write('Done. Restart opencode, then check with: opencode mcp list\n');
