import * as assert from 'node:assert/strict';
import { mock } from 'node:test';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { mkdtemp, writeFile, readFile, rm, mkdir, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { CanvasAgentRuntime } from '../../canvasAgentRuntime';
import { CanvasAgent, type AgentRunInput } from '../../canvasAgent';
import type { CanvasAgentAction, CanvasAgentState } from '@tensorviz/contracts/protocol';

suite('Persistent Canvas agent connections', function () {
  this.timeout(8000);
  let directory: string, executable: string, runtime: CanvasAgentRuntime, released: string[];
  const requests = async (): Promise<any[]> => (await readFile(path.join(directory, 'requests.jsonl'), 'utf8').catch(() => '')).trim().split('\n').filter(Boolean).map(line => JSON.parse(line));
  const launches = async () => (await requests()).filter(message => message.args);
  const alive = (pid: number) => { try { process.kill(pid, 0); return true; } catch { return false; } };
  async function until(check: () => boolean | Promise<boolean>): Promise<void> {
    for (let i = 0; i < 200; i++) { if (await check()) return; await delay(10); }
    assert.fail('Timed out waiting for the agent lifecycle');
  }
  const input = (provider: 'codex' | 'claude'): AgentRunInput => ({ provider, cwd: directory, context: 'current graph',
    messages: [{ id: 'first', role: 'user', text: 'first question' }] });
  setup(async () => {
    directory = await mkdtemp(path.join(tmpdir(), 'canvas-connection-')); executable = path.join(directory, 'agent'); released = [];
    await writeFile(executable, `#!${process.execPath}
const fs=require('node:fs'),path=require('node:path');
const flag=name=>fs.existsSync(path.join(__dirname,name));
const log=m=>fs.appendFileSync(path.join(__dirname,'requests.jsonl'),JSON.stringify(m)+'\\n');
const reply=m=>console.log(JSON.stringify(m));let statuses=0;
log({args:process.argv.slice(2),pid:process.pid});
if(process.argv[2]==='auth'){reply({loggedIn:true});process.exit(0)}
require('node:readline').createInterface({input:process.stdin}).on('line',line=>{
const m=JSON.parse(line);log(m);if(flag('hold-start')&&(m.method==='initialize'||m.request?.subtype==='initialize'))return;
if(m.method==='turn/interrupt'||m.request?.subtype==='interrupt'){
 if(flag('hold-interrupt'))return;
 const fail=flag('fail-interrupt');
 if(m.method)reply({id:m.id,...(fail?{error:true}:{result:{}})});
 else reply({type:'control_response',response:{request_id:m.request_id,subtype:fail?'error':'success',response:{}}});
 if(!fail&&!flag('hold-interrupt-result')){if(m.method)reply({method:'turn/completed',params:{threadId:'thread-'+process.pid,turn:{status:'interrupted'}}});else reply({type:'result',is_error:true});}
 return;
}
if(m.type==='control_request'){
 const response=m.request.subtype==='mcp_status'?{mcpServers:[{name:'tensorviz_canvas',status:flag('fail-mcp')?'failed':flag('slow-mcp')&&statuses++<2?'pending':'connected'}]}:{models:[{value:'default',displayName:'Default'}]};
 reply({type:'control_response',response:{request_id:m.request_id,subtype:'success',response}});return;
}
let result={};
if(m.method==='config/read')result={config:{mcp_servers:{external:{enabled:true}},plugins:{external:{enabled:true}}}};
if(m.method==='account/read')result={account:{type:'chatgpt'}};
if(m.method==='model/list')result={data:[{model:'one',displayName:'One',isDefault:true,supportedReasoningEfforts:[]}],nextCursor:null};
if(m.method==='thread/start')result={thread:{id:'thread-'+process.pid}};
if(m.method==='turn/start')result={turn:{id:'turn-'+m.id}};
if(m.method==='mcpServerStatus/list')result={data:flag('fail-mcp')?[]:[{name:'tensorviz_canvas',tools:{get_graph:{}}}]};
if(m.id!==undefined)reply({id:m.id,result});
if(m.method==='turn/start'||m.type==='user'){
 if(m.type==='user'&&!flag('hold-init'))reply({type:'system',subtype:'init'});
 const preparing=(m.params?.input?.[0]?.text??m.message?.content??'').includes('Host context preparation, not a user task.');
 if(flag('crash-turn'))process.exit(1);if(flag('hold-turn')||(preparing&&flag('hold-preparation')))return;
 if(preparing&&flag('fail-preparation')){if(m.type==='user')reply({type:'result',is_error:true});else reply({method:'turn/completed',params:{threadId:'thread-'+process.pid,turn:{status:'failed'}}});return;}
 if(m.type==='user')reply({type:'result',is_error:flag('fail-turn'),result:'answer'});
 else {reply({method:'item/agentMessage/delta',params:{threadId:'thread-'+process.pid,itemId:'answer',delta:'answer'}});reply({method:'turn/completed',params:{threadId:'thread-'+process.pid,turn:{status:flag('fail-turn')?'failed':'completed'}}});}
}
});
`, { mode: 0o700 });
    runtime = new CanvasAgentRuntime({ executable: async () => executable, tools: () => new Map(), release: owner => released.push(owner) });
  });
  teardown(async () => {
    runtime.clear();
    await until(async () => (await launches()).every(({ pid }) => !alive(pid)));
    await rm(directory, { recursive: true, force: true });
  });

  for (const provider of ['codex', 'claude'] as const) {
    test(`${provider}: opening and two messages share startup, tools and conversation`, async () => {
      const turn = input(provider), signal = new AbortController().signal;
      await Promise.all([runtime.warm(turn, signal), runtime.warm(turn, signal)]);
      assert.equal((await launches()).length, 1);
      assert.equal((await requests()).filter(m => m.method === 'turn/start' || m.type === 'user').length, 0, 'warm must not consume a model turn');
      const output: string[] = [];
      await runtime.run(turn, event => output.push(event.text), signal);
      await runtime.run({ ...turn, context: 'new graph selection', messages: [...turn.messages,
        { id: 'answer', role: 'assistant', text: 'previous reply' }, { id: 'second', role: 'user', text: 'follow-up question' }] }, event => output.push(event.text), signal);
      assert.deepEqual(output, ['answer', 'answer']);
      const log = await requests();
      assert.equal((await launches()).length, 1, 'neither Send may spawn another agent');
      assert.equal(log.filter(m => m.method === 'initialize' || m.request?.subtype === 'initialize').length, 1);
      const turns = log.filter(m => m.method === 'turn/start' || m.type === 'user');
      const text = provider === 'codex' ? turns[1].params.input[0].text : turns[1].message.content;
      assert.match(text, /new graph selection/); assert.match(text, /follow-up question/);
      assert.doesNotMatch(text, /first question|previous reply|Conversation so far/, 'native history must not be replayed every turn');
      assert.equal(released.length, 0, 'completed replies still own their native proposals');
      if (provider === 'codex') {
        assert.equal(log.filter(m => m.method === 'thread/start').length, 1);
        assert.equal(turns[0].params.threadId, turns[1].params.threadId);
        const config = log.find(m => m.method === 'thread/start').params.config;
        assert.equal(config.mcp_servers.external.enabled, false); assert.equal(config.plugins.external.enabled, false);
      }
    });
    test(`${provider}: Send joins an in-flight warm-up`, async () => {
      const turn = input(provider), signal = new AbortController().signal;
      await Promise.all([runtime.warm(turn, signal), runtime.run(turn, () => {}, signal)]);
      assert.equal((await launches()).length, 1);
    });
    test(`${provider}: one quiet preparation primes history without occupying the visible conversation`, async () => {
      const turn = input(provider), signal = new AbortController().signal;
      const warm = { ...turn, preparation: { context: 'snapshot before typing' } };
      await Promise.all([runtime.warm(warm, signal), runtime.warm(warm, signal)]);
      await until(() => !(runtime as any).session?.preparing);
      await runtime.warm(warm, signal);
      assert.equal((await requests()).filter(m => m.method === 'turn/start' || m.type === 'user').length, 1);
      const output: string[] = [];
      await runtime.run({ ...turn, context: 'fresh context at Send' }, event => output.push(event.text), signal);
      assert.deepEqual(output, ['answer']);
      const turns = (await requests()).filter(m => m.method === 'turn/start' || m.type === 'user');
      assert.equal(turns.length, 2); assert.equal((await launches()).length, 1);
      const text = provider === 'codex' ? turns[1].params.input[0].text : turns[1].message.content;
      assert.match(text, /fresh context at Send/); assert.match(text, /first question/);
      assert.doesNotMatch(text, /snapshot before typing|Host context preparation/);
    });
    test(`${provider}: early Send interrupts preparation and retains the native conversation`, async () => {
      await writeFile(path.join(directory, 'hold-preparation'), '');
      const turn = input(provider), signal = new AbortController().signal;
      await runtime.warm({ ...turn, preparation: { context: 'old snapshot' } }, signal);
      await until(async () => (await requests()).some(m => m.method === 'turn/start' || m.type === 'user'));
      const first = (await launches())[0];
      await until(async () => !!(await (runtime as any).session.ready).interrupt);
      const output: string[] = [];
      await runtime.run(turn, event => output.push(event.text), signal);
      assert.ok(alive(first.pid));
      await runtime.run(turn, event => output.push(event.text), signal);
      assert.deepEqual(output, ['answer', 'answer']);
      assert.equal((await launches()).length, 1, 'interrupt must preserve the connection and cache key');
      assert.equal((await requests()).filter(m => m.method === 'turn/interrupt' || m.request?.subtype === 'interrupt').length, 1);
      assert.equal(released.length, 0, 'preparation cancellation must not discard native proposal ownership');
    });
    for (const failure of ['fail-interrupt', 'hold-interrupt', 'hold-interrupt-result']) {
      test(`${provider}: ${failure} falls back to a fresh connection without leaking old cleanup`, async () => {
        await writeFile(path.join(directory, 'hold-preparation'), '');
        await writeFile(path.join(directory, failure), '');
        const turn = input(provider), signal = new AbortController().signal;
        await runtime.warm({ ...turn, preparation: { context: 'old snapshot' } }, signal);
        await until(async () => !!(await (runtime as any).session.ready).interrupt);
        const first = (await launches())[0], output: string[] = [];
        await runtime.run(turn, event => output.push(event.text), signal);
        await until(() => !alive(first.pid));
        await runtime.run(turn, event => output.push(event.text), signal);
        assert.deepEqual(output, ['answer', 'answer']); assert.equal((await launches()).length, 2);
      });
    }
    test(`${provider}: Stop during preparation interruption does not launch the real turn`, async () => {
      await writeFile(path.join(directory, 'hold-preparation'), '');
      await writeFile(path.join(directory, 'hold-interrupt'), '');
      const turn = input(provider), controller = new AbortController();
      await runtime.warm({ ...turn, preparation: { context: 'old snapshot' } }, controller.signal);
      await until(async () => !!(await (runtime as any).session.ready).interrupt);
      const running = runtime.run(turn, () => {}, controller.signal);
      const stopped = assert.rejects(running, /abort|Stopped/i);
      await until(async () => (await requests()).some(m => m.method === 'turn/interrupt' || m.request?.subtype === 'interrupt'));
      controller.abort(); await stopped;
      assert.equal((await launches()).length, 1);
      assert.equal((await requests()).filter(m => m.method === 'turn/start' || m.type === 'user').length, 1);
    });
    test(`${provider}: failed preparation falls back to a normal real request`, async () => {
      await writeFile(path.join(directory, 'fail-preparation'), '');
      const turn = input(provider), signal = new AbortController().signal;
      await runtime.warm({ ...turn, preparation: { context: 'current graph' } }, signal);
      await until(() => !(runtime as any).session);
      const output: string[] = [];
      await runtime.run(turn, event => output.push(event.text), signal);
      assert.deepEqual(output, ['answer']); assert.equal((await launches()).length, 2);
    });
    test(`${provider}: preparation times out and Clear cancels it without another request`, async () => {
      await writeFile(path.join(directory, 'hold-preparation'), '');
      const original = global.setTimeout;
      const timer = mock.method(global, 'setTimeout', ((fn: (...args: any[]) => void, ms: number, ...args: any[]) => original(fn, ms === 15_000 ? 30 : ms, ...args)) as typeof setTimeout);
      const turn = input(provider), signal = new AbortController().signal;
      try {
        await runtime.warm({ ...turn, preparation: { context: 'current graph' } }, signal);
        await until(() => !(runtime as any).session);
      } finally { timer.mock.restore(); }
      await runtime.warm({ ...turn, preparation: { context: 'current graph' } }, signal);
      runtime.clear();
      await until(async () => (await launches()).every(({ pid }) => !alive(pid)));
      assert.equal((await launches()).length, 2);
    });
    test(`${provider}: preparation blocks MCP reads and writes; the real request can use them`, async () => {
      runtime.clear();
      let calls = 0;
      runtime = new CanvasAgentRuntime({ executable: async () => executable, release() {}, tools: () => new Map(
        ['inspect_graph', 'propose_change'].map(name => [name, { config: { inputSchema: {}, annotations: { readOnlyHint: name === 'inspect_graph' } },
          handler: async () => { calls++; return { content: [{ type: 'text' as const, text: 'allowed' }] }; } }])) });
      await writeFile(path.join(directory, 'hold-preparation'), '');
      const turn = input(provider), controller = new AbortController();
      const client = async () => {
        const { url } = await (runtime as any).connection;
        const connected = new Client({ name: 'preparation-test', version: '1' });
        await connected.connect(new StreamableHTTPClientTransport(new URL(url), { requestInit: { headers: { Authorization: `Bearer ${(runtime as any).token}` } } }));
        return connected;
      };
      await runtime.warm({ ...turn, preparation: { context: 'current graph' } }, controller.signal);
      const preparing = await client();
      try {
        for (const name of ['inspect_graph', 'propose_change']) assert.equal((await preparing.callTool({ name, arguments: {} })).isError, true);
        assert.equal(calls, 0);
      } finally { await preparing.close(); }
      await writeFile(path.join(directory, 'hold-turn'), '');
      const running = runtime.run(turn, () => {}, controller.signal);
      const stopped = assert.rejects(running, /Stopped|closed|ended/);
      await until(async () => (await requests()).filter(m => m.method === 'turn/start' || m.type === 'user').length === 2);
      const active = await client();
      try {
        assert.notEqual((await active.callTool({ name: 'propose_change', arguments: {} })).isError, true);
        assert.equal(calls, 1);
      } finally { await active.close(); controller.abort(); await stopped; }
    });
    test(`${provider}: cancellation during initialization releases process and temporary files`, async () => {
      await writeFile(path.join(directory, 'hold-start'), '');
      const controller = new AbortController();
      const warming = runtime.warm(input(provider), controller.signal);
      const rejected = assert.rejects(warming, /Stopped|closed|ended/);
      await until(async () => (await launches()).length === 1);
      const launch = (await launches())[0]; controller.abort(); await rejected;
      await until(() => !alive(launch.pid));
      if (provider === 'claude') {
        const config = launch.args[launch.args.indexOf('--mcp-config') + 1];
        await until(async () => access(config).then(() => false, () => true));
      }
      await rm(path.join(directory, 'hold-start'));
      await runtime.warm(input(provider), new AbortController().signal);
      assert.equal((await launches()).length, 2);
    });
    test(`${provider}: Stop rejects an active turn; a later turn gets a clean connection`, async () => {
      await writeFile(path.join(directory, 'hold-turn'), '');
      const controller = new AbortController();
      const running = runtime.run(input(provider), () => {}, controller.signal);
      const rejected = assert.rejects(running, /Stopped|closed|ended/);
      await until(async () => (await requests()).some(m => m.method === 'turn/start' || m.type === 'user'));
      controller.abort(); await rejected;
      await rm(path.join(directory, 'hold-turn'));
      await runtime.run(input(provider), () => {}, new AbortController().signal);
      assert.equal((await launches()).length, 2);
    });
    test(`${provider}: Clear releases the busy guard before the old turn finishes unwinding`, async () => {
      await writeFile(path.join(directory, 'hold-turn'), '');
      const signal = new AbortController().signal;
      const old = runtime.run(input(provider), () => {}, signal);
      const stopped = assert.rejects(old, /Stopped|closed|ended/);
      await until(async () => (await requests()).some(m => m.method === 'turn/start' || m.type === 'user'));
      runtime.clear(true);
      const next = runtime.run(input(provider), () => {}, signal);
      await rm(path.join(directory, 'hold-turn'));
      await stopped; await next;
      assert.equal((await launches()).length, 2);
    });
    test(`${provider}: failed replies and idle process exits are never cached as healthy`, async () => {
      await writeFile(path.join(directory, 'fail-turn'), '');
      await assert.rejects(runtime.run(input(provider), () => {}, new AbortController().signal), /could not complete/);
      await rm(path.join(directory, 'fail-turn'));
      await runtime.run(input(provider), () => {}, new AbortController().signal);
      const launch = (await launches()).at(-1); process.kill(launch.pid);
      await until(() => !(runtime as any).session);
      await runtime.run(input(provider), () => {}, new AbortController().signal);
      assert.equal((await launches()).length, 3);
    });
    for (const reset of ['idle', 'stop']) test(`${provider}: ${reset} followed by preparation still replays prior user history and evidence`, async () => {
      const turn = { ...input(provider), conversationId: 'chat', selectionEvidence: {
        modelId: 'model', revision: 'a'.repeat(64), nodes: [{ parameters: { calibration_code: 37 } }],
      } }, signal = new AbortController().signal;
      await runtime.run(turn, () => {}, signal);
      if (reset === 'stop') runtime.clear(true);
      else (runtime as any).idleTimer._onTimeout();
      await runtime.warm({ ...turn, preparation: { context: 'fresh graph' } }, signal);
      await until(() => !(runtime as any).session?.preparing);
      await runtime.run({ ...turn, selectionEvidence: undefined, messages: [...turn.messages,
        { id: 'prior', role: 'assistant', text: 'Prior visible response' }, { id: 'new', role: 'user', text: 'Recall the original code' }] }, () => {}, signal);
      const prompts = (await requests()).filter(m => m.method === 'turn/start' || m.type === 'user');
      const text = provider === 'codex' ? prompts.at(-1).params.input[0].text : prompts.at(-1).message.content;
      assert.match(text, /calibration_code.*37/); assert.match(text, /Prior visible response/);
      assert.equal((await launches()).length, 2);
    });
    test(`${provider}: settings, instructions, workspace and Clear replace the session`, async () => {
      const original = input(provider), signal = new AbortController().signal;
      await runtime.warm(original, signal);
      await runtime.warm({ ...original, settings: { model: 'two', effort: 'high' } }, signal);
      await runtime.warm({ ...original, intent: 'open_graph' }, signal);
      await mkdir(path.join(directory, 'other'));
      await runtime.warm({ ...original, cwd: path.join(directory, 'other') }, signal);
      assert.equal((await launches()).length, 4); assert.equal(released.length, 0);
      runtime.clear(); assert.equal(released.length, 1);
      await runtime.warm(original, signal); assert.equal((await launches()).length, 5);
    });
    test(`${provider}: tool startup failures send no model prompt and can be retried`, async () => {
      await writeFile(path.join(directory, 'fail-mcp'), '');
      await assert.rejects(runtime.warm(input(provider), new AbortController().signal), /Canvas/);
      assert.ok(!(await requests()).some(m => m.method === 'turn/start' || m.type === 'user'));
      await rm(path.join(directory, 'fail-mcp')); await runtime.warm(input(provider), new AbortController().signal);
      assert.equal((await launches()).length, 2);
    });
  }
  test('Claude preparation before init reconnects instead of trusting a premature interrupt acknowledgement', async () => {
    await writeFile(path.join(directory, 'hold-preparation'), '');
    await writeFile(path.join(directory, 'hold-init'), '');
    const turn = input('claude'), signal = new AbortController().signal;
    await runtime.warm({ ...turn, preparation: { context: 'old snapshot' } }, signal);
    await until(async () => (await requests()).some(m => m.type === 'user'));
    await runtime.run(turn, () => {}, signal);
    assert.equal((await launches()).length, 2);
    assert.ok(!(await requests()).some(m => m.request?.subtype === 'interrupt'));
  });
  test('Claude waits for asynchronous MCP discovery without starting inference', async () => {
    await writeFile(path.join(directory, 'slow-mcp'), '');
    await runtime.warm(input('claude'), new AbortController().signal);
    const log = await requests(); assert.equal(log.filter(m => m.request?.subtype === 'mcp_status').length, 3);
    assert.ok(!log.some(m => m.type === 'user'));
  });
  test('a cleared startup cannot resurrect after executable resolution completes', async () => {
    let resolve!: (value: string) => void, first = true;
    runtime = new CanvasAgentRuntime({ executable: () => first ? (first = false, new Promise(done => { resolve = done; })) : Promise.resolve(executable), tools: () => new Map(), release() {} });
    const old = runtime.warm(input('codex'), new AbortController().signal);
    const rejected = assert.rejects(old, /abort/i); runtime.clear();
    await runtime.warm(input('codex'), new AbortController().signal); resolve(executable); await rejected;
    await runtime.run(input('codex'), () => {}, new AbortController().signal);
    assert.equal((await launches()).length, 1);
  });
  test('a Canvas conversation sends one preparation through a timeout, reopening, a model change and Refresh', async () => {
    // Every replacement session starts unsent; only the conversation can know it already paid for preparation.
    await writeFile(path.join(directory, 'hold-preparation'), '');
    // The preparation expiry fires by hand: a real short timer can reset a slow agent before it reads the request.
    let expire!: () => void;
    const original = global.setTimeout;
    const timer = mock.method(global, 'setTimeout', ((fn: (...args: any[]) => void, ms: number, ...args: any[]) =>
      ms === 15_000 ? void (expire = () => fn(...args)) : original(fn, ms, ...args)) as typeof setTimeout);
    const preparations = async () => (await requests()).filter(m => m.method === 'turn/start' && m.params.input[0].text.includes('Host context preparation')).length;
    let state!: CanvasAgentState;
    const agent = new CanvasAgent({ prepare: async () => ({ cwd: directory, context: 'current graph' }),
      prepareWarm: async () => ({ cwd: directory, context: 'current graph' }),
      inspect: (provider, refresh, signal) => runtime.inspect(provider, directory, refresh, signal),
      warm: (provider, settings, signal, intent, prepared) => runtime.warm({ provider, settings, intent, cwd: directory,
        ...(prepared ? { preparation: { context: prepared.context } } : {}) }, signal),
      run: (turn, emit, signal) => runtime.run(turn, emit, signal), clear: () => runtime.clear(), publish: (_view, value) => { state = value; } });
    const act = async (action: CanvasAgentAction) => {
      agent.handle({ type: 'agentRequest', protocolVersion: 1, workspaceSessionId: 'workspace', messageId: 'message', viewId: 'view', ...action });
      await until(() => !(agent as any).discovery && !(agent as any).warming && !(agent as any).active);
    };
    try {
      await act({ action: 'discover' });
      await until(async () => await preparations() === 1);
      expire();
      await until(() => !(runtime as any).session);
      await act({ action: 'discover' });
      await act({ action: 'configure', provider: 'codex', settings: { model: 'one' } });
      await act({ action: 'discover', refresh: true });
      await act({ action: 'send', requestId: 'question', provider: 'codex', prompt: 'Explain', nodeIds: [],
        target: { workspaceFolder: directory, file: 'model.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 } });
      assert.equal(state.messages[1]?.text, 'answer');
      assert.equal(await preparations(), 1, 'reconnections take fresh context from Send instead of charging another preparation');
      await act({ action: 'clear' });
      await until(async () => await preparations() === 2);
    } finally { timer.mock.restore(); agent.dispose(); }
  });
  test('a healthy connection survives readiness-cache expiry; explicit Refresh replaces it', async () => {
    const signal = new AbortController().signal;
    await runtime.inspect('codex', directory, false, signal); await runtime.warm(input('codex'), signal);
    const now = Date.now;
    try { Date.now = () => now() + 60_000; await runtime.inspect('codex', directory, false, signal); }
    finally { Date.now = now; }
    assert.equal((await launches()).length, 2, 'expired catalogue cache must not re-probe a live connection');
    await runtime.inspect('codex', directory, true, signal); await runtime.warm(input('codex'), signal);
    assert.equal((await launches()).length, 4);
  });
});
