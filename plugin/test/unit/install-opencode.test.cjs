const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const script = path.resolve(__dirname, '../../../plugins/tensorviz/scripts/install-opencode.cjs');
const launcher = path.resolve(__dirname, '../../../plugins/tensorviz/scripts/launch.cjs');
const skillSource = path.resolve(__dirname, '../../../plugins/tensorviz/skills/model-workflow');

function run(args) {
  return spawnSync(process.execPath, [script, ...args], { encoding: 'utf8' });
}

function withProject(fn) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-opencode-'));
  try { fn(dir); } finally { fs.rmSync(dir, { recursive: true, force: true }); }
}

test('opencode installer dry-runs by default and writes nothing', () => withProject(dir => {
  const result = run(['--project', dir]);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Dry run/);
  assert.equal(fs.existsSync(path.join(dir, 'opencode.json')), false);
  assert.equal(fs.existsSync(path.join(dir, '.opencode')), false);
}));

test('opencode installer writes the MCP entry and skill link, idempotently', () => withProject(dir => {
  const first = run(['--project', dir, '--write']);
  assert.equal(first.status, 0, first.stderr);
  const config = JSON.parse(fs.readFileSync(path.join(dir, 'opencode.json'), 'utf8'));
  assert.deepEqual(config.mcp.tensorviz.command, ['node', launcher]);
  assert.equal(config.mcp.tensorviz.type, 'local');
  assert.equal(config.mcp.tensorviz.enabled, true);
  const link = path.join(dir, '.opencode', 'skills', 'tensorviz-model-workflow');
  assert.equal(fs.lstatSync(link).isSymbolicLink(), true);
  assert.equal(fs.readlinkSync(link), skillSource);

  const second = run(['--project', dir, '--write']);
  assert.equal(second.status, 0, second.stderr);
  assert.match(second.stdout, /already has the current/);
  assert.match(second.stdout, /already present/);
  assert.equal(fs.existsSync(path.join(dir, 'opencode.json.bak')), false);
}));

test('opencode installer keeps unrelated settings and backs up the file it changes', () => withProject(dir => {
  const configPath = path.join(dir, 'opencode.json');
  fs.writeFileSync(configPath, JSON.stringify({ model: 'anthropic/claude-sonnet-5-5', mcp: { other: { type: 'local', command: ['true'] } } }));
  const result = run(['--project', dir, '--write']);
  assert.equal(result.status, 0, result.stderr);
  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  assert.equal(config.model, 'anthropic/claude-sonnet-5-5');
  assert.deepEqual(config.mcp.other.command, ['true']);
  assert.equal(config.mcp.tensorviz.enabled, true);
  assert.equal(fs.existsSync(path.join(dir, 'opencode.json.bak')), true);
}));

test('opencode installer refuses to overwrite a tensorviz entry it did not create', () => withProject(dir => {
  const configPath = path.join(dir, 'opencode.json');
  const foreign = { type: 'remote', url: 'http://127.0.0.1:9999/mcp' };
  fs.writeFileSync(configPath, JSON.stringify({ mcp: { tensorviz: foreign } }));
  const result = run(['--project', dir, '--write']);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /not created by this script/);
  assert.deepEqual(JSON.parse(fs.readFileSync(configPath, 'utf8')).mcp.tensorviz, foreign);
}));

test('opencode installer explains rather than editing a JSONC file with comments', () => withProject(dir => {
  fs.writeFileSync(path.join(dir, 'opencode.json'), '{ // comment\n "model": "x" }\n');
  const result = run(['--project', dir, '--write']);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /not plain JSON/);
  assert.match(result.stderr, /"tensorviz"/);
}));

test('opencode installer --remove deletes only its own entry and link', () => withProject(dir => {
  assert.equal(run(['--project', dir, '--write']).status, 0);
  fs.writeFileSync(path.join(dir, 'opencode.json'), JSON.stringify({ model: 'kept', mcp: { tensorviz: JSON.parse(fs.readFileSync(path.join(dir, 'opencode.json'), 'utf8')).mcp.tensorviz, other: { type: 'local', command: ['true'] } } }));
  const result = run(['--project', dir, '--remove', '--write']);
  assert.equal(result.status, 0, result.stderr);
  const config = JSON.parse(fs.readFileSync(path.join(dir, 'opencode.json'), 'utf8'));
  assert.equal(config.model, 'kept');
  assert.equal(config.mcp.tensorviz, undefined);
  assert.deepEqual(config.mcp.other.command, ['true']);
  assert.equal(fs.existsSync(path.join(dir, '.opencode', 'skills', 'tensorviz-model-workflow')), false);
}));

test('opencode installer requires exactly one of --project or --global', () => {
  assert.notEqual(run([]).status, 0);
  assert.notEqual(run(['--project']).status, 0);
});
