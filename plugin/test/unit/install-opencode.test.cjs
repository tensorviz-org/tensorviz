const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const repo = path.resolve(__dirname, '../../..');
const script = path.join(repo, 'plugins/tensorviz/scripts/install-opencode.cjs');
const catalog = path.join(repo, 'distribution/plugin-catalog/plugins/tensorviz');
const { linkType } = require(script);

const write = (file, text) => { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, text); };
const writeJson = (dir, name, value) => write(path.join(dir, name), JSON.stringify(value));
const readJson = (dir, name) => JSON.parse(fs.readFileSync(path.join(dir, name), 'utf8'));
const skillLink = project => path.join(project, '.opencode/skills/tensorviz-model-workflow');
const mkdir = (dir, name) => { const created = path.join(dir, name); fs.mkdirSync(created, { recursive: true }); return created; };

function sandbox(fn) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-opencode-'));
  try { fn(dir); } finally { fs.rmSync(dir, { recursive: true, force: true }); }
}

// A built source plugin with the real source launch declaration and a stub runtime.
function sourcePlugin(dir, name = 'source') {
  const root = path.join(dir, name, 'plugins/tensorviz');
  for (const file of ['scripts/launch.cjs', 'runtime/host.cjs', 'skills/model-workflow/SKILL.md']) write(path.join(root, file), '');
  fs.copyFileSync(path.join(repo, 'plugins/tensorviz/.mcp.json'), path.join(root, '.mcp.json'));
  return root;
}

// PATH holds only the fakes a test asks for: `node` (real or a version stub) and `opencode`.
function run(dir, args, { node = 'real', opencode = null, env = {} } = {}) {
  const bin = fs.mkdtempSync(path.join(dir, 'bin-'));
  if (node === 'real') fs.symlinkSync(process.execPath, path.join(bin, 'node'));
  else if (node) write(path.join(bin, 'node'), `#!/bin/sh\necho ${node}\n`);
  if (opencode) write(path.join(bin, 'opencode'), `#!/bin/sh\necho ${opencode}\n`);
  for (const name of fs.readdirSync(bin)) fs.chmodSync(path.join(bin, name), 0o755);
  return spawnSync(process.execPath, [script, ...args], { encoding: 'utf8', env: { HOME: path.join(dir, 'home'), PATH: bin, ...env } });
}

test('dry run prints the plan and changes nothing', () => sandbox(dir => {
  const project = mkdir(dir, 'project');
  const result = run(dir, ['--project', project, '--plugin-root', sourcePlugin(dir)]);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Dry run/);
  assert.deepEqual(fs.readdirSync(project), []);
}));

test('source plugin: registers node launch.cjs and the skill, idempotently', () => sandbox(dir => {
  const project = mkdir(dir, 'project');
  const root = sourcePlugin(dir);
  const args = ['--project', project, '--plugin-root', root, '--write'];
  const first = run(dir, args);
  assert.equal(first.status, 0, first.stderr);
  assert.match(first.stdout, /Node\.js 22 or newer on its own PATH/);
  assert.deepEqual(readJson(project, 'opencode.json').mcp.tensorviz, { type: 'local', command: ['node', path.join(root, 'scripts/launch.cjs')] });
  assert.equal(fs.readlinkSync(skillLink(project)), path.join(root, 'skills/model-workflow'));
  const second = run(dir, args);
  assert.equal(second.status, 0, second.stderr);
  assert.match(second.stdout, /already has the current mcp\.tensorviz entry/);
  assert.match(second.stdout, /already present/);
  assert.equal(fs.existsSync(path.join(project, 'opencode.json.bak')), false);
}));

test('published catalog plugin: registers /bin/sh launch.sh without needing node', () => sandbox(dir => {
  const project = mkdir(dir, 'project');
  const result = run(dir, ['--project', project, '--plugin-root', catalog, '--write'], { node: null });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(readJson(project, 'opencode.json').mcp.tensorviz.command, ['/bin/sh', path.join(catalog, 'scripts/launch.sh')]);
  assert.equal(fs.readlinkSync(skillLink(project)), path.join(catalog, 'skills/model-workflow'));
  assert.doesNotMatch(result.stdout, /Node\.js/);
}));

test('an unbuilt source plugin, or no Node 22 on PATH, is refused before any change', () => sandbox(dir => {
  const project = mkdir(dir, 'project');
  const root = sourcePlugin(dir);
  const args = ['--project', project, '--plugin-root', root, '--write'];
  fs.rmSync(path.join(root, 'runtime'), { recursive: true });
  const unbuilt = run(dir, args);
  assert.equal(unbuilt.status, 1);
  assert.match(unbuilt.stderr, /no built runtime.*--plugin-root/s);
  write(path.join(root, 'runtime/host.cjs'), '');
  const old = run(dir, args, { node: 'v20.11.0' });
  assert.equal(old.status, 1);
  assert.match(old.stderr, /Node\.js 20 here/);
  const missing = run(dir, args, { node: null });
  assert.equal(missing.status, 1);
  assert.match(missing.stderr, /no `node` was found/);
  assert.deepEqual(fs.readdirSync(project), []);
}));

test('a new release path replaces the old entry and link; --remove works after the release is gone', () => sandbox(dir => {
  const project = mkdir(dir, 'project');
  const oldRoot = sourcePlugin(dir, 'tensorviz-1.0.2');
  const newRoot = sourcePlugin(dir, 'tensorviz-1.0.3');
  assert.equal(run(dir, ['--project', project, '--plugin-root', oldRoot, '--write']).status, 0);
  const upgrade = run(dir, ['--project', project, '--plugin-root', newRoot, '--write']);
  assert.equal(upgrade.status, 0, upgrade.stderr);
  assert.match(upgrade.stdout, /update mcp\.tensorviz/);
  assert.match(upgrade.stdout, /relink/);
  assert.deepEqual(readJson(project, 'opencode.json').mcp.tensorviz.command, ['node', path.join(newRoot, 'scripts/launch.cjs')]);
  assert.equal(fs.readlinkSync(skillLink(project)), path.join(newRoot, 'skills/model-workflow'));
  assert.equal(fs.existsSync(path.join(project, 'opencode.json.bak')), true);
  fs.rmSync(path.join(dir, 'tensorviz-1.0.3'), { recursive: true });
  const removed = run(dir, ['--project', project, '--remove', '--write'], { node: null });
  assert.equal(removed.status, 0, removed.stderr);
  assert.equal(readJson(project, 'opencode.json').mcp.tensorviz, undefined);
  assert.throws(() => fs.lstatSync(skillLink(project)), { code: 'ENOENT' });
}));

test('a tensorviz entry that is not a TensorViz launcher is never replaced or removed', () => sandbox(dir => {
  const project = mkdir(dir, 'project');
  const foreign = { type: 'remote', url: 'http://127.0.0.1:9999/mcp' };
  writeJson(project, 'opencode.json', { mcp: { tensorviz: foreign } });
  const install = run(dir, ['--project', project, '--plugin-root', sourcePlugin(dir), '--write']);
  assert.equal(install.status, 1);
  assert.match(install.stderr, /does not run a TensorViz launcher/);
  const remove = run(dir, ['--project', project, '--remove', '--write']);
  assert.equal(remove.status, 0, remove.stderr);
  assert.match(remove.stdout, /leave mcp\.tensorviz/);
  assert.deepEqual(readJson(project, 'opencode.json').mcp.tensorviz, foreign);
}));

test('opencode 2 configs, and fresh configs for opencode 2, get the entry under mcp.servers', () => sandbox(dir => {
  const root = sourcePlugin(dir);
  const command = ['node', path.join(root, 'scripts/launch.cjs')];
  const v2 = mkdir(dir, 'v2-config');
  writeJson(v2, 'opencode.json', { mcp: { servers: { other: { type: 'local', command: ['true'] } } } });
  assert.equal(run(dir, ['--project', v2, '--plugin-root', root, '--write'], { opencode: '1.15.13' }).status, 0);
  assert.equal(readJson(v2, 'opencode.json').mcp.tensorviz, undefined);
  assert.deepEqual(readJson(v2, 'opencode.json').mcp.servers.tensorviz.command, command);
  const fresh2 = mkdir(dir, 'fresh-2');
  assert.equal(run(dir, ['--project', fresh2, '--plugin-root', root, '--write'], { opencode: 'opencode 2.1.0' }).status, 0);
  assert.deepEqual(readJson(fresh2, 'opencode.json').mcp.servers.tensorviz.command, command);
  const fresh1 = mkdir(dir, 'fresh-1');
  assert.equal(run(dir, ['--project', fresh1, '--plugin-root', root, '--write'], { opencode: '1.15.13' }).status, 0);
  assert.deepEqual(readJson(fresh1, 'opencode.json').mcp.tensorviz.command, command);
  assert.equal(run(dir, ['--project', fresh2, '--remove', '--write']).status, 0);
  assert.equal(readJson(fresh2, 'opencode.json').mcp.servers.tensorviz, undefined);
}));

test('config with comments, or several config files, is left for a manual edit; the skill is still linked', () => sandbox(dir => {
  const root = sourcePlugin(dir);
  const commented = mkdir(dir, 'commented');
  const text = '{\n  // keep me\n  "model": "x"\n}\n';
  write(path.join(commented, 'opencode.jsonc'), text);
  const result = run(dir, ['--project', commented, '--plugin-root', root, '--write']);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Manual step: .*opencode\.jsonc is not plain JSON/);
  assert.match(result.stdout, /"tensorviz"/);
  assert.equal(fs.readFileSync(path.join(commented, 'opencode.jsonc'), 'utf8'), text);
  assert.equal(fs.existsSync(path.join(commented, 'opencode.json')), false);
  assert.equal(fs.lstatSync(skillLink(commented)).isSymbolicLink(), true);
  const several = mkdir(dir, 'several');
  writeJson(several, 'opencode.json', {});
  writeJson(several, '.opencode/opencode.json', {});
  const both = run(dir, ['--project', several, '--plugin-root', root]);
  assert.equal(both.status, 0, both.stderr);
  assert.match(both.stdout, /opencode merges .*opencode\.json and .*\.opencode[\\/]opencode\.json/);
}));

test('a single comment-free config is edited in place, keeping its indentation', () => sandbox(dir => {
  const project = mkdir(dir, 'project');
  write(path.join(project, 'opencode.jsonc'), '{\n\t"$schema": "https://opencode.ai/config.json"\n}\n');
  const result = run(dir, ['--project', project, '--plugin-root', sourcePlugin(dir), '--write']);
  assert.equal(result.status, 0, result.stderr);
  assert.match(fs.readFileSync(path.join(project, 'opencode.jsonc'), 'utf8'), /^\t"mcp": \{$/m);
  assert.equal(fs.existsSync(path.join(project, 'opencode.json')), false);
}));

test('--global writes under XDG_CONFIG_HOME when set, else ~/.config', () => sandbox(dir => {
  const root = sourcePlugin(dir);
  const xdg = path.join(dir, 'xdg');
  const custom = run(dir, ['--global', '--plugin-root', root, '--write'], { env: { XDG_CONFIG_HOME: xdg } });
  assert.equal(custom.status, 0, custom.stderr);
  assert.ok(readJson(path.join(xdg, 'opencode'), 'opencode.json').mcp.tensorviz);
  assert.equal(fs.lstatSync(path.join(xdg, 'opencode/skills/tensorviz-model-workflow')).isSymbolicLink(), true);
  assert.equal(fs.existsSync(path.join(dir, 'home')), false);
  const fallback = run(dir, ['--global', '--plugin-root', root, '--write']);
  assert.equal(fallback.status, 0, fallback.stderr);
  assert.ok(readJson(path.join(dir, 'home/.config/opencode'), 'opencode.json').mcp.tensorviz);
}));

test('--remove backs up the config and never overwrites an earlier backup', () => sandbox(dir => {
  const project = mkdir(dir, 'project');
  writeJson(project, 'opencode.json', { model: 'kept' });
  assert.equal(run(dir, ['--project', project, '--plugin-root', sourcePlugin(dir), '--write']).status, 0);
  const removed = run(dir, ['--project', project, '--remove', '--write']);
  assert.equal(removed.status, 0, removed.stderr);
  assert.deepEqual(readJson(project, 'opencode.json.bak'), { model: 'kept' });
  assert.ok(readJson(project, 'opencode.json.bak.1').mcp.tensorviz);
  assert.deepEqual(readJson(project, 'opencode.json'), { model: 'kept', mcp: {} });
  assert.throws(() => fs.lstatSync(skillLink(project)), { code: 'ENOENT' });
}));

test('a failed skill link restores the config and reports a clean error', () => sandbox(dir => {
  const project = mkdir(dir, 'project');
  const original = '{\n  "model": "kept"\n}\n';
  write(path.join(project, 'opencode.json'), original);
  write(path.join(project, '.opencode/skills'), 'a file where the skills directory belongs');
  const result = run(dir, ['--project', project, '--plugin-root', sourcePlugin(dir), '--write']);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /^Could not create the skill link .*Nothing was changed\.\n$/s);
  assert.doesNotMatch(result.stderr, /\n\s+at /);
  assert.equal(fs.readFileSync(path.join(project, 'opencode.json'), 'utf8'), original);
  assert.equal(fs.existsSync(path.join(project, 'opencode.json.bak')), false);
}));

test('Windows catalog plugins are refused; Windows skill links use junctions', () => sandbox(dir => {
  const project = mkdir(dir, 'project');
  const windows = path.join(repo, 'distribution/plugin-catalog/plugins/tensorviz-windows-x64');
  const result = run(dir, ['--project', project, '--plugin-root', windows]);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Windows catalog plugins are not supported/);
  assert.equal(linkType('win32'), 'junction');
  assert.equal(linkType('darwin'), 'dir');
  assert.equal(linkType('linux'), 'dir');
}));

test('options need exactly one scope and reject unknown flags', () => sandbox(dir => {
  for (const args of [[], ['--global', '--project', dir], ['--project'], ['--project', dir, '--wirte']]) {
    const result = run(dir, args);
    assert.equal(result.status, 1, args.join(' '));
    assert.match(result.stderr, /Usage:/);
  }
}));
