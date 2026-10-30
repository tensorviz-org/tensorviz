#!/usr/bin/env node
'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const { spawn } = require('node:child_process');

function absolute(value, name) {
  if (!value || !path.isAbsolute(value)) throw new Error(`${name} must be an absolute path.`);
  return value;
}

async function start(options) {
  for (const name of ['output', 'workspace', 'code', 'profile', 'extensions']) absolute(options[name], name);
  if (!(await fs.stat(options.workspace)).isDirectory()) throw new Error('workspace must be a directory.');
  if (options.extension) absolute(options.extension, 'extension');
  await fs.mkdir(options.output, { mode: 0o700 });
  const args = ['--new-window', '--user-data-dir', options.profile, '--extensions-dir', options.extensions];
  if (options.extension) args.push('--extensionDevelopmentPath', options.extension);
  args.push(options.workspace);
  const child = spawn(options.code, args, { env: { ...process.env, TENSORVIZ_AGENT_RECORDING_DIR: options.output },
    detached: process.platform !== 'win32', stdio: 'ignore', shell: false });
  await new Promise((resolve, reject) => { child.once('spawn', resolve); child.once('error', reject); });
  child.unref();
  const launch = { format: 1, requestedAt: new Date().toISOString(), pid: child.pid, code: options.code, args,
    recordingDirectory: options.output, status: 'launch_requested_not_yet_recording' };
  await fs.writeFile(path.join(options.output, 'launch.json'), JSON.stringify(launch, null, 2) + '\n', { mode: 0o600, flag: 'wx' });
  return launch;
}

async function readJournal(file) {
  const stat = await fs.lstat(file);
  if (!stat.isFile() || stat.size > 65 * 1024 * 1024) throw new Error(`Unsupported recording file: ${file}`);
  const text = await fs.readFile(file, 'utf8');
  const lines = text.split('\n'), partialLastLine = lines.pop() !== '';
  const events = lines.map(line => JSON.parse(line));
  let previous = -1;
  for (let index = 0; index < events.length; index++) {
    const event = events[index];
    if (event.sequence !== index + 1 || !Number.isFinite(event.elapsedMs) || event.elapsedMs < previous) {
      throw new Error(`Invalid event ordering in ${file}`);
    }
    previous = event.elapsedMs;
  }
  if (events[0]?.kind !== 'recording_start' || events[0].format !== 1) throw new Error(`Missing recording header in ${file}`);
  return { events, partialLastLine };
}

function summarize(events, partialLastLine = false, now = Date.now()) {
  const turns = new Map(), requests = new Map(), calls = new Map();
  let omissions = 0;
  const unmatchedToolResults = [];
  const request = id => {
    if (!requests.has(id)) requests.set(id, {});
    return requests.get(id);
  };
  for (const event of events) {
    if (event.kind === 'omitted_event') { omissions++; continue; }
    if (event.kind === 'agent_request' && event.message.action === 'send') {
      Object.assign(request(event.message.requestId), { sendMs: event.elapsedMs, prompt: event.message.prompt });
    }
    if (event.kind === 'host_agent_state' && event.state.requestId) {
      const item = request(event.state.requestId);
      if (event.state.status === 'running' && item.sendMs === undefined) item.sendMs = event.elapsedMs;
      if (['idle', 'stopped', 'error'].includes(event.state.status) && item.sendMs !== undefined && item.terminalMs === undefined) {
        item.terminalMs = event.elapsedMs; item.uiStatus = event.state.status; item.activity = event.state.activity;
        item.displayedReply = event.state.messages?.filter(message => message.role === 'assistant').at(-1)?.text ?? null;
      }
    }
    if (event.kind === 'turn_start') {
      const user = event.input.messages.filter(message => message.role === 'user').at(-1);
      turns.set(event.turnId, { turnId: event.turnId, requestId: user?.id, provider: event.input.provider,
        pipeline: event.pipeline, requestedSettings: event.input.settings ?? {}, conversationId: event.input.conversationId,
        prompt: user?.text ?? '', startMs: event.elapsedMs, firstTextMs: null, endMs: null, outcome: 'unfinished', receivedText: '', tools: [] });
    }
    const turn = turns.get(event.turnId);
    if (turn && event.kind === 'agent_event' && event.event.kind === 'text') {
      if (event.event.text && turn.firstTextMs === null) turn.firstTextMs = event.elapsedMs;
      turn.receivedText += event.event.text;
    }
    if (turn && event.kind === 'turn_end') Object.assign(turn, { endMs: event.elapsedMs, outcome: event.outcome, error: event.error });
    if (event.kind === 'tool_start') {
      const call = { callId: event.callId, turnId: event.turnId ?? null, name: event.name, startMs: event.elapsedMs, endMs: null, outcome: 'unfinished' };
      calls.set(event.callId, call); if (turn) turn.tools.push(call);
    }
    if (event.kind === 'tool_result' || event.kind === 'tool_error') {
      const call = calls.get(event.callId);
      if (call) Object.assign(call, { endMs: event.elapsedMs, outcome: event.kind === 'tool_error' ? 'error' : event.result.isError ? 'reported_error' : 'returned',
        activeTurn: event.activeTurn, cancelled: event.cancelled });
      else unmatchedToolResults.push({ callId: event.callId, turnId: event.turnId ?? null, name: event.name, kind: event.kind });
    }
  }
  const end = events.at(-1), header = events[0];
  let pidExists = false;
  try { process.kill(header.pid, 0); pidExists = true; } catch { /* No live process confirmed. */ }
  const heartbeat = events.filter(event => ['recording_start', 'recording_heartbeat'].includes(event.kind)).at(-1);
  const freshHeartbeat = now - Date.parse(heartbeat.at) >= 0 && now - Date.parse(heartbeat.at) < 15_000;
  const difference = (a, b) => a == null || b == null ? null : a - b;
  return { format: 1, timingScope: 'local_host_observation', quality: 'ungraded', modelIdentity: 'requested_settings_only',
    recording: end.kind === 'recording_end' ? 'stopped' : pidExists && freshHeartbeat ? 'live' : 'unconfirmed',
    stopReason: end.kind === 'recording_end' ? end.reason : null, partialLastLine, omissions,
    journalComplete: !partialLastLine && !omissions && !unmatchedToolResults.length && end.kind === 'recording_end'
      && ['requested', 'extension_shutdown'].includes(end.reason),
    toolsOutsideTurns: [...calls.values()].filter(call => !turns.has(call.turnId)), unmatchedToolResults,
    metadata: header.metadata, eventCount: events.length,
    turns: [...turns.values()].map(turn => {
      const item = requests.get(turn.requestId) ?? {};
      return { ...turn, displayedReply: item.displayedReply ?? null, uiStatus: item.uiStatus ?? null,
        sendToFirstTextMs: difference(turn.firstTextMs, item.sendMs), runtimeToFirstTextMs: difference(turn.firstTextMs, turn.startMs),
        runtimeMs: difference(turn.endMs, turn.startMs), sendToUiTerminalMs: difference(item.terminalMs, item.sendMs),
        firstUsefulTextMs: null, tools: turn.tools.map(call => ({ ...call, durationMs: difference(call.endMs, call.startMs) })) };
    }),
    requestsWithoutRuntime: [...requests].filter(([id]) => ![...turns.values()].some(turn => turn.requestId === id))
      .map(([requestId, item]) => ({ requestId, ...item })),
  };
}

async function report(directory) {
  const files = (await fs.readdir(directory)).filter(name => name.endsWith('.jsonl')).sort();
  const recordings = [];
  for (const name of files) {
    const journal = await readJournal(path.join(directory, name));
    recordings.push({ file: name, ...summarize(journal.events, journal.partialLastLine) });
  }
  return { recordings, status: recordings.length ? 'observed' : 'no_recording_header' };
}

async function main(argv) {
  const [command, ...args] = argv, options = {};
  if (!['start', 'stop', 'status', 'report'].includes(command) || args.length % 2) throw new Error('Use start|stop|status|report with --output /absolute/directory. start also requires --workspace, --code, --profile and --extensions.');
  for (let index = 0; index < args.length; index += 2) {
    const key = args[index].replace(/^--/, '');
    if (!args[index].startsWith('--') || !['output', 'workspace', 'code', 'profile', 'extensions', 'extension'].includes(key) || options[key]) throw new Error('Unknown or repeated argument.');
    options[key] = args[index + 1];
  }
  absolute(options.output, 'output');
  if (command === 'start') return start(options);
  if (command === 'stop') {
    await fs.writeFile(path.join(options.output, 'STOP'), 'User requested recording stop.\n', { mode: 0o600 });
    return { status: 'stop_requested', directory: options.output };
  }
  const result = await report(options.output);
  if (command === 'report') await fs.writeFile(path.join(options.output, 'report.json'), JSON.stringify(result, null, 2) + '\n', { mode: 0o600 });
  return result;
}

module.exports = { readJournal, summarize, report, start };
if (require.main === module) main(process.argv.slice(2)).then(result => process.stdout.write(JSON.stringify(result, null, 2) + '\n'))
  .catch(error => { process.stderr.write(error.message + '\n'); process.exitCode = 1; });
