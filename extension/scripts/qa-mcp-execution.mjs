// Standard HTTP MCP client against a disposable installed VSIX and the documented demo.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { setTimeout as delay } from 'node:timers/promises';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
const root = process.env.TV_QA_ROOT;
assert(root && path.isAbsolute(root), 'TV_QA_ROOT must be a disposable installed test directory');
const control = path.join(root, 'control');
async function poll(read, done, label, timeout = 60_000) {
  const deadline = Date.now() + timeout;
  do { const value = await read(); if (done(value)) return value; await delay(100); } while (Date.now() < deadline);
  throw new Error(`Timed out: ${label}`);
}
async function command(type, output, extra = {}) {
  const id = randomUUID();
  await writeFile(path.join(control, 'command.json'), JSON.stringify({ type, id, ...extra }), { mode: 0o600 });
  return poll(async () => { try { return JSON.parse(await readFile(path.join(control, output), 'utf8')); } catch { return null; } }, value => value?.id === id, type);
}
const client = new Client({ name: 'installed-execution-client', version: '1' });
async function call(name, args = {}) {
  const result = await client.callTool({ name, arguments: args });
  assert.ok(!result.isError, JSON.stringify(result.structuredContent));
  return result.structuredContent;
}
const execution = { inputs: { fixture: 'diag/fixtures/batch.pt' }, checkpoint: { file: 'diag/fixtures/baseline.pt' }, device: 'cpu', mode: 'eval', seed: 0, deterministic: true };
async function run(name = 'validate_model', overrides = {}) {
  const model = (await call('list_models')).models[0];
  const started = await call(name, { modelId: model.modelId, revision: model.revision, execution, ...overrides });
  return poll(() => call('get_run', { runId: started.runId }), value => value.finishedAt !== null, name);
}
async function select(cls) {
  const source = await readFile(path.join(root, 'fixture/diag/demo_models.py'), 'utf8');
  const line = source.split('\n').findIndex(line => line.startsWith(`class ${cls}(`)) + 1;
  assert(line > 0);
  await command('select-model', 'selected-model.json', { file: 'diag/demo_models.py', line });
  await poll(() => call('list_models'), value => value.models[0]?.current && value.models[0].target.cls === cls, 'selected model');
}
try {
  await command('revert-clean', 'clean-state.json');
  const connection = await command('connect-mcp', 'mcp.json');
  await client.connect(new StreamableHTTPClientTransport(new URL(connection.url), { requestInit: { headers: connection.headers } }));
  await select('TinyNet');
  const checkpointBefore = await readFile(path.join(root, 'fixture/diag/fixtures/baseline.pt'));
  const baseline = await run(); assert.equal(baseline.status, 'passed', baseline.detail);
  const repeat = await run(); assert.equal(repeat.status, 'passed', repeat.detail);
  const same = await call('compare_runs', { baselineRunId: baseline.runId, candidateRunId: repeat.runId });
  assert.equal(same.outputs.allClose, true);
  const changed = await run('validate_model', { execution: { ...execution, checkpoint: { file: 'diag/fixtures/changed.pt' } } });
  assert.equal(changed.status, 'passed', changed.detail);
  const difference = await call('compare_runs', { baselineRunId: baseline.runId, candidateRunId: changed.runId });
  assert.equal(difference.outputs.allClose, false); assert.equal(difference.firstObservedDivergence.modulePath, 'head');
  const setup = await call('inspect_run', { runId: baseline.runId, section: 'setup' });
  assert(setup.importedSources.items.some(row => row.file === 'diag/demo_models.py'));
  console.log('PASS: installed controlled comparison, checkpoint divergence, native projection and imported source evidence');
  await command('edit', 'edit-result.json', { file: 'diag/losses.py', before: 'import torch', after: 'import torch  # unsaved helper' });
  const dirty = await run(); assert.equal(dirty.status, 'refused'); assert.match(dirty.detail, /Save changed Python/);
  await command('undo', 'undo-result.json', { file: 'diag/losses.py' });
  await command('revert-clean', 'clean-state.json');
  assert.deepEqual(await readFile(path.join(root, 'fixture/diag/fixtures/baseline.pt')), checkpointBefore, 'validation must leave the checkpoint file unchanged');
  await select('BrokenResidual');
  const failed = await run('validate_model', { execution: { inputs: { args: [{ shape: [1, 1, 56, 56], dtype: 'float32', fill: 'ones' }] }, device: 'cpu' } });
  assert.equal(failed.status, 'failed');
  const failure = await call('inspect_run', { runId: failed.runId, section: 'failure' });
  assert.equal(failure.error.nodeId, null, 'an unreached tail must not be labelled as the failing layer');
  assert.deepEqual(failure.failure.inputs.map(row => row.shape), [[1, 1, 28, 28], [1, 1, 56, 56]]);
  await writeFile(path.join(root, 'evidence/execution-results.json'), JSON.stringify({ status: 'passed',
    checks: ['controlled comparison', 'changed checkpoint', 'imported sources', 'residual mismatch', 'immutable after retarget'] }, null, 2));
  console.log('PASS: installed residual mismatch and immutable inspection after retargeting');
} finally {
  await client.close().catch(() => {});
  await command('disconnect-mcp', 'mcp-disconnected.json');
}
