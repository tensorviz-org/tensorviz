const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs/promises');
const path = require('node:path');
const { tmpdir } = require('node:os');
const { summarize, readJournal, report, start } = require('../ml-agent-recording.cjs');

function fixture() {
  const at = new Date().toISOString();
  return [
    { kind: 'recording_start', format: 1, pid: process.pid, elapsedMs: 0, metadata: {} },
    { kind: 'agent_request', elapsedMs: 10, message: { action: 'send', requestId: 'request', prompt: 'Question' } },
    { kind: 'turn_start', turnId: 'turn', elapsedMs: 30, pipeline: 'acp', input: { provider: 'claude', conversationId: 'chat', settings: { effort: 'medium' }, messages: [{ id: 'request', role: 'user', text: 'Question' }] } },
    { kind: 'tool_start', turnId: 'turn', callId: 'call', name: 'inspect_capture', elapsedMs: 40 },
    { kind: 'tool_result', turnId: 'turn', callId: 'call', name: 'inspect_capture', elapsedMs: 45, result: { isError: true }, activeTurn: true, cancelled: false },
    { kind: 'agent_event', turnId: 'turn', elapsedMs: 50, event: { kind: 'text', text: 'Checking.\n\n' } },
    { kind: 'agent_event', turnId: 'turn', elapsedMs: 70, event: { kind: 'text', text: 'Answer' } },
    { kind: 'turn_end', turnId: 'turn', elapsedMs: 80, outcome: 'completed' },
    { kind: 'host_agent_state', elapsedMs: 90, state: { requestId: 'request', status: 'idle', messages: [{ role: 'assistant', text: 'Checking.\n\nAnswer' }] } },
    { kind: 'recording_end', elapsedMs: 100, reason: 'requested' },
  ].map((event, index) => ({ ...event, at, sequence: index + 1 }));
}

test('timings keep Send, runtime, first text and displayed completion distinct; usefulness stays ungraded', () => {
  const result = summarize(fixture());
  const turn = result.turns[0];
  assert.equal(turn.sendToFirstTextMs, 40); assert.equal(turn.runtimeToFirstTextMs, 20);
  assert.equal(turn.runtimeMs, 50); assert.equal(turn.sendToUiTerminalMs, 80);
  assert.equal(turn.firstUsefulTextMs, null); assert.equal(turn.tools[0].durationMs, 5);
  assert.equal(turn.tools[0].outcome, 'reported_error');
  assert.equal(turn.receivedText, 'Checking.\n\nAnswer'); assert.equal(turn.displayedReply, turn.receivedText);
  assert.equal(result.quality, 'ungraded'); assert.equal(result.recording, 'stopped'); assert.equal(result.journalComplete, true);
});

test('cancellation, missing terminals and omitted evidence are retained without inventing completion', () => {
  const rows = fixture().slice(0, 7);
  rows.push({ kind: 'turn_end', turnId: 'turn', elapsedMs: 80, outcome: 'cancelled' });
  rows.push({ kind: 'omitted_event', originalKind: 'host_agent_state', elapsedMs: 90 });
  const result = summarize(rows);
  assert.equal(result.turns[0].outcome, 'cancelled'); assert.equal(result.turns[0].sendToUiTerminalMs, null);
  assert.equal(result.turns[0].displayedReply, null); assert.equal(result.omissions, 1); assert.equal(result.journalComplete, false);
  assert.equal(summarize(fixture(), true).journalComplete, false);
});

test('pre-runtime failure and unassigned tool attempts cannot disappear from the report', () => {
  const rows = fixture();
  rows.splice(2, 0, { kind: 'agent_request', elapsedMs: 12, message: { action: 'send', requestId: 'failed', prompt: 'Missing target' } },
    { kind: 'host_agent_state', elapsedMs: 14, state: { requestId: 'failed', status: 'error', activity: 'Stale target', messages: [] } },
    { kind: 'tool_start', callId: 'early', name: 'inspect_capture', elapsedMs: 16 },
    { kind: 'tool_result', callId: 'early', elapsedMs: 18, result: { isError: true }, activeTurn: false, cancelled: false },
    { kind: 'tool_error', callId: 'missing-start', name: 'read_source', elapsedMs: 19 });
  const result = summarize(rows);
  assert.equal(result.requestsWithoutRuntime[0].requestId, 'failed');
  assert.equal(result.requestsWithoutRuntime[0].uiStatus, 'error');
  assert.equal(result.toolsOutsideTurns[0].outcome, 'reported_error');
  assert.equal(result.unmatchedToolResults[0].callId, 'missing-start');
  assert.equal(result.journalComplete, false);
});

test('launcher uses explicit isolated paths and records only a launch request, never assumed recording', async () => {
  const root = await fs.mkdtemp(path.join(tmpdir(), 'tensorviz-recording-launch-'));
  try {
    const executable = path.join(root, 'code-fixture');
    await fs.writeFile(executable, '#!/usr/bin/env node\nrequire("node:fs").writeFileSync(require("node:path").join(process.env.TENSORVIZ_AGENT_RECORDING_DIR,"arguments.json"),JSON.stringify({args:process.argv.slice(2),directory:process.env.TENSORVIZ_AGENT_RECORDING_DIR}));\n', { mode: 0o700 });
    const output = path.join(root, 'recording');
    const options = { output, code: executable, workspace: root, profile: path.join(root, 'profile'), extensions: path.join(root, 'extensions') };
    const receipt = await start(options);
    assert.equal(receipt.status, 'launch_requested_not_yet_recording');
    let captured;
    for (let attempt = 0; attempt < 100; attempt++) {
      try { captured = JSON.parse(await fs.readFile(path.join(output, 'arguments.json'), 'utf8')); break; }
      catch (error) { if (error.code !== 'ENOENT') throw error; await new Promise(resolve => setTimeout(resolve, 10)); }
    }
    assert.ok(captured, 'launched fixture must acknowledge its environment and arguments');
    assert.equal(captured.directory, output);
    assert.deepEqual(captured.args, ['--new-window', '--user-data-dir', options.profile, '--extensions-dir', options.extensions, root]);
    assert.equal((await report(output)).status, 'no_recording_header');
    await assert.rejects(start(options), /EEXIST/);
    await assert.rejects(start({ ...options, output: 'relative' }), /absolute/);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});

test('liveness needs a current heartbeat and live process; a start record alone is not perpetual proof', () => {
  const rows = fixture().slice(0, -1), time = Date.parse(rows[0].at);
  assert.equal(summarize(rows, false, time + 1000).recording, 'live');
  assert.equal(summarize(rows, false, time + 16000).recording, 'unconfirmed');
  rows[0].pid = 99999999;
  assert.equal(summarize(rows, false, time + 1000).recording, 'unconfirmed');
});

test('journal reader distinguishes an incomplete final write from altered ordering', async () => {
  const root = await fs.mkdtemp(path.join(tmpdir(), 'tensorviz-recording-report-'));
  try {
    assert.equal((await report(root)).status, 'no_recording_header');
    const file = path.join(root, 'fixture.jsonl');
    await fs.writeFile(file, fixture().map(event => JSON.stringify(event)).join('\n') + '\n{"partial');
    const result = await readJournal(file);
    assert.equal(result.partialLastLine, true); assert.equal(result.events.length, 10);
    const rows = fixture(); rows[2].sequence = 99;
    await fs.writeFile(file, rows.map(event => JSON.stringify(event)).join('\n') + '\n');
    await assert.rejects(readJournal(file), /ordering/);
    await fs.writeFile(file, '{"kind":"recording_end","sequence":1,"elapsedMs":0}\n');
    await assert.rejects(readJournal(file), /header/);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});
