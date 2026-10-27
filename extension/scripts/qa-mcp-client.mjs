// Real SDK journey against qa-local-browser.cjs and a disposable installed VSIX.
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
const file = 'complex_models/network.py';
const baseline = await readFile(path.join(root, 'fixture', file), 'utf8');
assert(baseline.includes('self.branch_fc = nn.Linear(10, 10)'), 'Wrong QA fixture');
async function poll(read, done, label, timeout = 30_000) {
  const deadline = Date.now() + timeout;
  do {
    const value = await read();
    if (done(value)) return value;
    await delay(100);
  } while (Date.now() < deadline);
  throw new Error(`Timed out: ${label}`);
}
async function command(type, output, extra = {}) {
  const id = randomUUID();
  await writeFile(path.join(control, 'command.json'), JSON.stringify({type, id, ...extra}), {mode: 0o600});
  return poll(async () => {
    try { return JSON.parse(await readFile(path.join(control, output), 'utf8')); } catch { return null; }
  }, value => value?.id === id, type);
}
const inspect = () => command('inspect', 'source-state.json');
const connection = await command('connect-mcp', 'mcp.json');
const client = new Client({name: 'independent-graph-client', version: '1.0.0'});
const foreign = new Client({name: 'second-independent-client', version: '1.0.0'});
const connect = c => c.connect(new StreamableHTTPClientTransport(new URL(connection.url), {requestInit:{headers:connection.headers}}));
async function call(name, args = {}, c = client, error = false) {
  const result = await c.callTool({name, arguments:args});
  assert.equal(!!result.isError, error, `${name}: ${JSON.stringify(result.structuredContent)}`);
  return result.structuredContent ?? JSON.parse(result.content[0].text);
}
const discover = () => call('list_models');
const ref = model => ({modelId:model.modelId, revision:model.revision});
const runTerminal = run => run.finishedAt !== null;
try {
  assert.equal((await fetch(connection.url)).status, 401);
  await connect(client); await connect(foreign);
  assert.deepEqual((await client.listTools()).tools.map(tool => tool.name).sort(), [
    'cancel_run', 'compare_runs', 'focus_view', 'get_graph', 'get_proposal', 'get_run', 'get_selection',
    'inspect_nodes', 'inspect_run', 'list_models', 'propose_edit', 'validate_model',
  ]);
  let discovery = await discover();
  assert.equal(discovery.models[0].target.file, file);
  let model = ref(discovery.models[0]);
  assert.equal((await call('get_graph', model)).nodeCount, 102);
  let view = discovery.views.find(v => v.current);
  // A just-connected page may not have reported selection yet.
  if (!view) {
    discovery = await poll(discover, d => d.views.some(v => v.current), 'canvas selection');
    view = discovery.views.find(v => v.current);
  }
  await call('focus_view', {...model, viewId:view.viewId, nodeIds:['branch_fc']});
  const selection = await poll(() => call('get_selection', {...model, viewId:view.viewId}), s => s.nodeIds.includes('branch_fc'), 'focused selection');
  assert.deepEqual(selection.nodeIds, ['branch_fc']);
  const inputs = [{shape:[1,3,8,8],dtype:'float32',fill:'ones'}];
  const run = await call('validate_model', {...model,inputs});
  const hidden = await call('get_run', {runId:run.runId}, foreign, true);
  assert.equal(hidden.error.code, 'UNKNOWN_RUN');
  const validated = await poll(() => call('get_run',{runId:run.runId}),runTerminal,'measured validation',60_000);
  assert.equal(validated.status,'passed',validated.detail);
  assert(validated.result);
  model = validated.result;
  const nodes = await call('inspect_nodes',{...model,nodeIds:['branch_fc']});
  assert.equal(nodes.nodes[0].shapes.provenance,'observed');
  assert.deepEqual(nodes.nodes[0].shapes.output,[1,10]);
  const failed = await call('validate_model',{...model,inputs:[{...inputs[0],shape:[1,2,8,8]}]});
  const failure = await poll(() => call('get_run',{runId:failed.runId}),runTerminal,'failing validation',60_000);
  assert.equal(failure.status,'failed');
  model = ref((await discover()).models[0]);
  const cancelled = await call('validate_model',{...model,inputs});
  await call('cancel_run',{runId:cancelled.runId});
  assert.equal((await poll(() => call('get_run',{runId:cancelled.runId}),runTerminal,'cancelled validation')).status,'cancelled');
  console.log('PASS: SDK discovery, auth, graph inspection, view receipt, measured/failed/cancelled validation and session isolation');

  const edit = {kind:'set_param',nodeId:'branch_fc',param:'out_features',value:12};
  async function proposal() {
    model = ref((await discover()).models[0]);
    const p = await call('propose_edit',{...model,edit});
    await poll(() => call('get_proposal',{proposalId:p.proposalId}),p => p.status !== 'preparing','native review');
    const state = await inspect();
    assert.equal(state.reviews.length,1);
    assert.equal(state.files[file].text,baseline);
    assert.equal(state.reviews[0].before,baseline);
    assert.equal(state.reviews[0].after,baseline.replace('nn.Linear(10, 10)','nn.Linear(10, 12)'));
    return p;
  }
  let p = await proposal();
  await command('discard','decision-result.json');
  assert.equal((await poll(() => call('get_proposal',{proposalId:p.proposalId}),p => p.status !== 'review','discard')).status,'cancelled');
  assert.equal((await inspect()).files[file].text,baseline);
  p = await proposal();
  await command('apply','decision-result.json');
  assert.equal((await poll(() => call('get_proposal',{proposalId:p.proposalId}),p => p.status !== 'review','apply')).status,'applied');
  const state = await inspect();
  assert(state.files[file].dirty);
  assert.equal(state.files[file].text,baseline.replace('nn.Linear(10, 10)','nn.Linear(10, 12)'));
  assert.equal(await readFile(path.join(root,'fixture',file),'utf8'),baseline);
  assert.equal((await command('undo','undo-result.json',{file})).text,baseline);
  await poll(discover,d => d.models[0].current && d.models[0].revision !== model.revision,'reprojection after Undo');
  p = await proposal();
  await command('edit','edit-result.json',{file,before:'nn.Linear(10, 10)',after:'nn.Linear(10, 14)'});
  await command('apply','decision-result.json');
  const stale = await poll(() => call('get_proposal',{proposalId:p.proposalId}),p => p.status !== 'review','stale refusal');
  assert.equal(stale.status,'refused');
  assert.equal((await inspect()).files[file].text,baseline.replace('nn.Linear(10, 10)','nn.Linear(10, 14)'));
  assert.equal((await command('undo','undo-result.json',{file})).text,baseline);
  console.log('PASS: MCP proposals use complete native diffs, Discard, unsaved Apply, Undo, reprojection and stale-source refusal');
  await command('disconnect-mcp','mcp-disconnected.json');
  await assert.rejects(() => client.listTools());
  console.log('PASS: disconnect revokes the installed endpoint');
} catch (error) {
  console.error('Installed MCP journey failed:', error);
  throw error;
} finally {
  await client.close().catch(() => {}); await foreign.close().catch(() => {});
  try {
    let state = await inspect();
    if (state.reviews.length) await command('discard','decision-result.json');
    const text = state.files[file].text;
    if ([12, 14].some(value => text === baseline.replace('nn.Linear(10, 10)', `nn.Linear(10, ${value})`))) {
      await command('undo','undo-result.json',{file});
      state = await inspect();
    }
    assert.equal(state.files[file].text,baseline,'QA left modified source');
  } finally { await command('disconnect-mcp','mcp-disconnected.json'); }
}
