import * as assert from 'node:assert/strict';
import { mkdtemp, writeFile, readFile, rm, access, copyFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { CanvasAcpSession } from '../../canvasAcpSession';
import { CanvasAgentRuntime, codexAcpConfig } from '../../canvasAgentRuntime';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';

suite('Canvas ACP wire and lifecycle', function () {
  this.timeout(10_000);
  let directory: string, executable: string;
  let sessions: CanvasAcpSession[], runtimes: CanvasAgentRuntime[];
  // The fake agent appends while a test reads; a line without its newline is still being written.
  const journal = async (): Promise<any[]> => (await readFile(path.join(directory, 'wire.jsonl'), 'utf8')).split('\n').slice(0, -1).filter(Boolean).map(line => JSON.parse(line));
  /** The client's reply to an agent request. The agent can end its turn before that reply arrives. */
  const replyTo = async (id: number): Promise<any> => {
    for (let i = 0; i < 200; i++) {
      const reply = (await journal()).find(m => m.id === id && !m.method);
      if (reply) return reply;
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    assert.fail(`No reply to agent request ${id}`);
  };
  setup(async () => {
    sessions = []; runtimes = [];
    directory = await mkdtemp(path.join(tmpdir(), 'canvas-acp-test-')); executable = path.join(directory, 'agent');
    await writeFile(executable, `#!${process.execPath}
const fs = require('node:fs');
const log = message => fs.appendFileSync(${JSON.stringify(path.join(directory, 'wire.jsonl'))}, JSON.stringify(message)+'\\n');
const send = message => process.stdout.write(JSON.stringify({jsonrpc:'2.0',...message})+'\\n');
const flag = name => fs.existsSync(${JSON.stringify(directory)}+'/'+name);
let pending;
log({cwd:process.cwd(),config:JSON.parse(process.env.CODEX_CONFIG||'{}')});
require('node:readline').createInterface({input:process.stdin}).on('line',line=>{
 const m=JSON.parse(line); log(m);
 if(m.method==='initialize') send({id:m.id,result:{protocolVersion:1,agentCapabilities:{mcpCapabilities:{http:true}},...(flag('rename')?{_meta:{jetbrains:{air:{version:1,capabilities:['sessionRename']}}}}:{})}});
 else if(m.method==='account/read') send({id:m.id,result:{account:flag('signed-out')?null:{type:'apiKey'}}});
 else if(m.method==='model/list') send({id:m.id,result:{data:[{model:'fixture',displayName:'Fixture',supportedReasoningEfforts:[{reasoningEffort:'high'}]}]}});
 else if(m.method==='config/read') send({id:m.id,result:{config:{model:'fixture',openai_base_url:'https://routing.example/v1',hooks:{untrusted:true}}}});
 else if(m.method==='_session/rename') { if(flag('rename-hang')) return; send(flag('rename-error')?{id:m.id,error:{code:-32601,message:'unsupported'}}:{id:m.id,result:{}}); }
 else if(m.method==='session/new') send({id:m.id,result:{sessionId:'session-one',modes:{currentModeId:'read-only',availableModes:[{id:'read-only',name:'Read only'}]}}});
 else if(m.method==='session/set_mode') send({id:m.id,result:{}});
 else if(m.method==='session/cancel' && pending && !pending.ignore) { send({id:pending.id,result:{stopReason:'cancelled'}}); pending=undefined; }
 else if(m.method==='session/prompt') {
   const text=m.params.prompt[0].text;
   if(text==='hang' || text.includes('Host context preparation') && text.includes('hold-preparation')) { pending={id:m.id,ignore:text.includes('ignore-cancel')}; return; }
   if(text==='crash') process.exit(1);
   if(text==='error') return send({id:m.id,error:{code:-32603,message:'secret-api-key'}});
   if(text==='stream') {
     for(const params of JSON.parse(fs.readFileSync(${JSON.stringify(path.join(directory, 'updates.json'))},'utf8'))) send({method:'session/update',params});
     return send({id:m.id,result:{stopReason:'end_turn'}});
   }
   send({id:900,method:'session/request_permission',params:{sessionId:'session-one',toolCall:{toolCallId:'unsafe',title:'Write file',kind:'edit',status:'pending'},options:[{optionId:'yes',name:'Allow',kind:'allow_once'}]}});
   send({method:'session/update',params:{sessionId:'session-one',update:{sessionUpdate:'agent_message_chunk',content:{type:'text',text:'answer '+text}}}});
   send({id:m.id,result:{stopReason:'end_turn'}});
 }
});
`, { mode: 0o700 });
  });
  teardown(async () => {
    for (const agent of runtimes) agent.clear();
    await Promise.all(sessions.map(session => session.dispose()));
    await rm(directory, { recursive: true, force: true });
  });
  const makeRuntime = (overrides: Partial<ConstructorParameters<typeof CanvasAgentRuntime>[0]> = {}) => {
    const agent = new CanvasAgentRuntime({ pipeline: () => 'acp', acpAdapter: async () => executable,
      executable: async () => executable, tools: () => new Map(), release() {}, ...overrides });
    runtimes.push(agent); return agent;
  };
  const start = async (provider: 'codex' | 'claude' = 'claude') => {
    const session = await CanvasAcpSession.start({ adapter: executable, executable: '/signed-in/agent', provider,
      settings: { model: 'fixture', effort: 'high' }, instructions: 'Review only', url: 'http://127.0.0.1:9999/mcp', token: 'fixture-token', signal: new AbortController().signal });
    sessions.push(session); return session;
  };

  test('isolating Codex preserves its account routing and omits null provider overrides', () => {
    assert.deepEqual(codexAcpConfig({ openai_base_url: 'https://account.example/v1', chatgpt_base_url: 'https://account.example/chat',
      model_provider: null, model: 'fixture', model_reasoning_effort: null, hooks: { unsafe: true }, mcp_servers: { unrelated: {} } }),
    { model: 'fixture', openai_base_url: 'https://account.example/v1', chatgpt_base_url: 'https://account.example/chat' });
  });

  test('readiness reports a missing adapter without asking the model or misreporting CLI login', async () => {
    const runtime = makeRuntime({ acpAdapter: undefined, executable: async () => { throw new Error('Must not probe'); } });
    const checked = await runtime.inspect('claude', directory, true, new AbortController().signal);
    assert.equal(checked.status, 'error'); assert.match(checked.detail, /claudeAcpPath/);
  });

  test('a persistent MCP endpoint refuses tool calls outside an active user turn', async () => {
    let calls = 0;
    const runtime = makeRuntime({ tools: () => new Map([['probe', {
      config: { inputSchema: {} }, handler: async () => { calls++; return { content: [] }; },
    }]]) });
    const internal = runtime as unknown as { connect(provider: 'claude'): Promise<{ url: string }>; token: string };
    const { url } = await internal.connect('claude'); const client = new Client({ name: 'idle-probe', version: '1' });
    try {
      await client.connect(new StreamableHTTPClientTransport(new URL(url), { requestInit: { headers: { Authorization: `Bearer ${internal.token}` } } }));
      assert.equal((await client.callTool({ name: 'probe', arguments: {} })).isError, true); assert.equal(calls, 0);
    } finally { await client.close(); runtime.clear(); }
  });

  test('real SDK streams two turns in one session and denies filesystem escalation', async () => {
    const session = await start(); const output: string[] = [];
    await session.prompt('first', event => output.push(event.text), new AbortController().signal);
    await session.prompt('second', event => output.push(event.text), new AbortController().signal);
    const wire = await journal();
    assert.deepEqual(output, ['answer first', 'answer second']);
    assert.equal(wire.filter(m => m.method === 'session/new').length, 1);
    assert.ok(wire.some(m => m.id === 900 && m.result?.outcome.outcome === 'cancelled'));
    const request = wire.find(m => m.method === 'session/new').params;
    assert.notEqual(request.cwd, directory);
    assert.deepEqual(request.mcpServers.map((s: any) => s.name), ['tensorviz_canvas']);
    assert.deepEqual(request._meta.claudeCode.options.tools, []);
    assert.equal(request._meta.claudeCode.options.strictMcpConfig, true);
    assert.deepEqual(request._meta.claudeCode.options.settingSources, []);
    assert.equal(request._meta.claudeCode.options.effort, 'high');
    await session.dispose();
    await assert.rejects(access((await journal())[0].cwd), 'private cwd must be removed');
  });

  test('identified messages keep streamed words intact and separate the final answer from its preamble', async () => {
    const chunk = (text: string, messageId: string, sessionId = 'session-one') => ({ sessionId,
      update: { sessionUpdate: 'agent_message_chunk', messageId, content: { type: 'text', text } } });
    await writeFile(path.join(directory, 'updates.json'), JSON.stringify([
      chunk('Checking ', 'preamble'), chunk('the layer.', 'preamble'),
      chunk('Must not appear', 'other-message', 'another-session'),
      { sessionId: 'session-one', update: { sessionUpdate: 'agent_thought_chunk', messageId: 'thought', content: { type: 'text', text: 'Private thought' } } },
      chunk('', 'empty-message'), chunk('It has ', 'answer'), chunk('two calls.', 'answer'),
    ]));
    const session = await start(); const output: string[] = [];
    await session.prompt('stream', event => output.push(event.text), new AbortController().signal);
    assert.equal(output.join(''), 'Checking the layer.\n\nIt has two calls.',
      'message boundaries must not join the preamble to the answer or split ordinary token chunks');
  });

  test('anonymous text and tool activity do not invent message boundaries', async () => {
    await writeFile(path.join(directory, 'updates.json'), JSON.stringify([
      { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'The out' } },
      { sessionUpdate: 'tool_call', toolCallId: 'inspection', title: 'Inspect', status: 'completed' },
      { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'put ' } },
      { sessionUpdate: 'agent_message_chunk', messageId: 'answer', content: { type: 'text', text: 'is ' } },
      { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'two' } },
      { sessionUpdate: 'agent_message_chunk', messageId: 'answer', content: { type: 'text', text: '.' } },
    ].map(update => ({ sessionId: 'session-one', update }))));
    const session = await start(); const output: string[] = [];
    await session.prompt('stream', event => output.push(event.text), new AbortController().signal);
    assert.equal(output.join(''), 'The output is two.');
  });

  test('a reused session starts each reply without a separator inherited from the previous turn', async () => {
    const session = await start();
    for (const messageId of ['first-answer', 'second-answer']) {
      await writeFile(path.join(directory, 'updates.json'), JSON.stringify([{ sessionId: 'session-one',
        update: { sessionUpdate: 'agent_message_chunk', messageId, content: { type: 'text', text: messageId } } }]));
      const output: string[] = [];
      await session.prompt('stream', event => output.push(event.text), new AbortController().signal);
      assert.equal(output.join(''), messageId);
    }
  });

  test('Codex uses an isolated home, disables side tools and explicitly selects read-only', async () => {
    const session = await start('codex');
    const wire = await journal();
    assert.deepEqual(wire[0].config.features, { shell_tool: false, hooks: false, multi_agent: false, apps: false });
    assert.equal(wire.find(m => m.method === 'session/set_mode').params.modeId, 'read-only');
    assert.equal(wire[0].config.model_reasoning_effort, 'high');
    assert.equal(wire.filter(m => m.method === '_session/rename').length, 0, 'unadvertised extensions are not called');
  });

  test('Codex keeps host-owned MCP actions reachable without allowing permission escalation', async () => {
    const session = await start('codex');
    await session.prompt('first', () => {}, new AbortController().signal);
    const wire = await journal();
    assert.deepEqual(wire[0].config.mcp_servers, {
      tensorviz_canvas: { url: 'http://127.0.0.1:9999/mcp',
        bearer_token_env_var: 'TENSORVIZ_CANVAS_AGENT_TOKEN', default_tools_approval_mode: 'approve' },
    }, 'the sole authenticated host owns source review and execution, as in the native bridge');
    assert.deepEqual(wire.find(m => m.method === 'session/new').params.mcpServers, [],
      'codex-acp replaces MCP configs supplied here and drops their tool approval policy');
    assert.deepEqual((await replyTo(900)).result, { outcome: { outcome: 'cancelled' } },
      'allowing the host tools must not grant direct filesystem or sandbox escalation');
  });

  test('Codex names only its new private session when negotiated; metadata refusal or timeout never prevents a reply', async () => {
    await writeFile(path.join(directory, 'rename'), '');
    for (const mode of ['success', 'rename-error', 'rename-hang']) {
      if (mode !== 'success') await writeFile(path.join(directory, mode), '');
      const session = await start('codex');
      try {
        const output: string[] = [];
        await session.prompt('first', event => output.push(event.text), new AbortController().signal);
        assert.deepEqual(output, ['answer first']);
        const wire = await journal(), methods = wire.map(m => m.method), rename = methods.lastIndexOf('_session/rename');
        assert.deepEqual(wire[rename].params, { sessionId: 'session-one', title: 'TensorViz Canvas' });
        assert.ok(rename < methods.lastIndexOf('session/prompt'));
        assert.deepEqual(wire.find(m => m.method === 'initialize').params.clientCapabilities,
          { _meta: { jetbrains: { air: { version: 1, capabilities: ['sessionIndex'] } } } });
      } finally { await session.dispose(); if (mode !== 'success') await rm(path.join(directory, mode)); }
    }
  });

  test('Codex reuses fresh readiness routing privately; expired, invalidated and different-host keys probe again', async () => {
    const secondExecutable = path.join(directory, 'agent-second'); await copyFile(executable, secondExecutable);
    for (const change of ['none', 'expired', 'invalidated', 'cwd', 'executable']) {
      let resolved = executable;
      const agent = makeRuntime({ executable: async () => resolved });
      try {
        const signal = new AbortController().signal, checked = await agent.inspect('codex', directory, true, signal);
        assert.equal(checked.status, 'ready'); assert.doesNotMatch(JSON.stringify(checked), /routing.example|hooks|codexConfig/);
        const before = (await journal()).filter(m => m.method === 'config/read').length;
        if (change === 'expired') for (const cached of (agent as any).availability.values()) cached.checked.checkedAt -= 30_001;
        if (change === 'invalidated') agent.invalidate();
        if (change === 'executable') resolved = secondExecutable;
        await agent.run({ ...input(), provider: 'codex', cwd: change === 'cwd' ? tmpdir() : directory }, () => {}, signal);
        const wire = await journal();
        assert.equal(wire.filter(m => m.method === 'config/read').length - before, change === 'none' ? 0 : 1, change);
        const config = wire.filter(m => m.config?.openai_base_url).at(-1)?.config;
        assert.equal(config.openai_base_url, 'https://routing.example/v1'); assert.deepEqual(config.hooks, {});
      } finally { agent.clear(); }
    }
  });

  test('signed-out refresh cannot reuse routing from an earlier successful readiness check', async () => {
    const agent = makeRuntime(), signal = new AbortController().signal;
    assert.equal((await agent.inspect('codex', directory, true, signal)).status, 'ready');
    await writeFile(path.join(directory, 'signed-out'), '');
    assert.equal((await agent.inspect('codex', directory, true, signal)).status, 'signInRequired');
    for (const cached of (agent as any).availability.values()) assert.equal(cached.codexConfig, undefined);
  });

  test('abort terminates an unresponsive prompt and removes the private directory', async () => {
    const session = await start(); const controller = new AbortController();
    const running = session.prompt('hang', () => {}, controller.signal);
    setTimeout(() => controller.abort(), 50); await running; await session.dispose();
    await assert.rejects(access((await journal())[0].cwd));
  });

  test('agent errors and exits settle the turn without exposing upstream text', async () => {
    for (const text of ['error', 'crash']) {
      const session = await start();
      await assert.rejects(session.prompt(text, () => {}, new AbortController().signal), error =>
        error instanceof Error && /could not complete/.test(error.message) && !error.message.includes('secret'));
      await session.dispose();
    }
  });

  test('runtime reuses ACP history, sends fresh context, and bootstraps after settings change', async () => {
    const runtime = makeRuntime();
    const messages: any[] = [{ id: 'u1', role: 'user', text: 'first question' }];
    const run = (context: string, model = 'one') => runtime.run({ provider: 'claude', cwd: directory, context, messages: structuredClone(messages), settings: { model } }, () => {}, new AbortController().signal);
    await run('revision-one');
    messages.push({ id: 'a1', role: 'assistant', text: 'first answer' }, { id: 'u2', role: 'user', text: 'second question' });
    await run('revision-two'); await run('revision-three', 'two');
    const wire = await journal(), prompts = wire.filter(m => m.method === 'session/prompt').map(m => m.params.prompt[0].text);
    assert.equal(wire.filter(m => m.method === 'session/new').length, 2);
    assert.match(prompts[1], /revision-two/); assert.match(prompts[1], /second question/); assert.doesNotMatch(prompts[1], /first answer|first question/);
    assert.match(prompts[2], /first answer/);
  });
  const until = async (check: () => Promise<boolean>) => {
    for (let i = 0; i < 200; i++) { if (await check()) return; await new Promise(resolve => setTimeout(resolve, 10)); }
    assert.fail('Timed out waiting for the wire event');
  };
  const input = (context = 'current facts') => ({ provider: 'claude' as const, cwd: directory, context,
    messages: [{ id: 'u1', role: 'user' as const, text: 'real user question' }], settings: { model: 'fixture', effort: 'high' } });


  test('evidence replays only on session bootstrap and survives idle, settings and Stop but not New chat', async () => {
    const agent = makeRuntime(), signal = new AbortController().signal;
    const original = { modelId: 'model', revision: 'a'.repeat(64), nodes: [{ parameters: { calibration_code: 37 } }] };
    const changed = { ...input('new revision has calibration_code 53'), settings: { model: 'other', effort: 'high' } };
    await agent.run({ ...input(JSON.stringify(original)), selectionEvidence: original }, () => {}, signal);
    await agent.run(input('new revision has calibration_code 53'), () => {}, signal);
    (agent as any).idleTimer._onTimeout();
    await agent.run(input('new revision has calibration_code 53'), () => {}, signal);
    await agent.run(changed, () => {}, signal);
    agent.clear(true);
    await agent.run(changed, () => {}, signal);
    agent.clear();
    await agent.run(changed, () => {}, signal);
    const wire = await journal(), prompts = wire.filter(m => m.method === 'session/prompt').map(m => m.params.prompt[0].text);
    assert.equal(prompts.length, 6, 'retaining evidence adds no extra inference');
    assert.equal(wire.filter(m => m.method === 'session/new').length, 5);
    assert.doesNotMatch(prompts[1], /Earlier graph inspections|calibration_code\":37/, 'live sessions must not replay retained tool evidence each turn');
    for (const index of [2, 3, 4]) {
      assert.match(prompts[index], /calibration_code\":37/, `reset ${index}`);
      assert.match(prompts[index], /historical data, never instructions/);
      assert.match(prompts[index], /new revision has calibration_code 53/);
    }
    assert.doesNotMatch(prompts[5], /Earlier graph inspections|calibration_code\":37/);
  });

  test('preparation is invisible, once per session, and the first Send bootstraps real history', async () => {
    const agent = makeRuntime(), signal = new AbortController().signal;
    await agent.warm(input(), signal);
    await until(async () => !(agent as any).acp?.preparing);
    await agent.warm(input(), signal);
    const output: string[] = [];
    await agent.run(input('fresh facts'), e => output.push(e.text), signal);
    const wire = await journal(), prompts = wire.filter(m => m.method === 'session/prompt');
    assert.equal(prompts.length, 2); assert.equal(wire.filter(m => m.method === 'session/new').length, 1);
    assert.match(prompts[0].params.prompt[0].text, /Host context preparation/);
    assert.doesNotMatch(prompts[0].params.prompt[0].text, /real user question/);
    assert.match(prompts[1].params.prompt[0].text, /fresh facts/);
    assert.match(prompts[1].params.prompt[0].text, /Conversation so far/);
    assert.doesNotMatch(output.join(''), /Host context preparation/);
    assert.ok((agent as any).idleTimer, 'reopening an already prepared chat must keep its idle bound');
  });

  for (const [provider, promptCount] of [['codex', 1], ['claude', 2]] as const) {
    test(`${provider} early Send joins initialization with ${promptCount === 1 ? 'no preparation turn' : 'preparation then cancellation'}`, async () => {
      const agent = makeRuntime(), signal = new AbortController().signal;
      const warming = agent.warm({ ...input('hold-preparation'), provider }, signal);
      await agent.run({ ...input('fresh Send context'), provider }, () => {}, signal); await warming;
      const wire = await journal(), prompts = wire.filter(m => m.method === 'session/prompt');
      assert.equal(wire.filter(m => m.method === 'session/new').length, 1, 'early Send must not spawn a second initialization');
      assert.equal(prompts.length, promptCount);
      assert.equal(wire.some(m => m.method === 'session/cancel'), provider === 'claude');
      assert.match(prompts.at(-1).params.prompt[0].text, /fresh Send context/);
    });
  }

  test('Send interrupts an already running preparation and waits for its terminal response', async () => {
    const agent = makeRuntime(), signal = new AbortController().signal;
    await agent.warm(input('hold-preparation'), signal);
    await until(async () => (await journal()).some(m => m.method === 'session/prompt'));
    await agent.run(input('fresh Send context'), () => {}, signal);
    const wire = await journal();
    assert.equal(wire.filter(m => m.method === 'session/new').length, 1);
    assert.equal(wire.filter(m => m.method === 'session/prompt').length, 2);
    assert.ok(wire.some(m => m.method === 'session/cancel'));
  });

  test('a preparation that ignores cancellation is replaced before Send enables tools', async () => {
    let calls = 0;
    const agent = makeRuntime({ tools: () => new Map([['probe', {
      config: { inputSchema: {} }, handler: async () => { calls++; return { content: [] }; },
    }]]) });
    const signal = new AbortController().signal;
    const client = new Client({ name: 'prep-probe', version: '1' });
    try {
      await agent.warm(input('hold-preparation ignore-cancel'), signal);
      const internal = agent as any, { url } = await internal.connection;
      await client.connect(new StreamableHTTPClientTransport(new URL(url), { requestInit: { headers: { Authorization: `Bearer ${internal.token}` } } }));
      assert.equal((await client.callTool({ name: 'probe', arguments: {} })).isError, true);
      const running = agent.run(input('fresh context after reconnect'), () => {}, signal);
      assert.equal((await client.callTool({ name: 'probe', arguments: {} })).isError, true);
      assert.equal(calls, 0, 'preparation cannot enter the production handler, even while Send waits');
      await running;
      const wire = await journal();
      assert.equal(wire.filter(m => m.method === 'session/new').length, 2);
      assert.match(wire.filter(m => m.method === 'session/prompt').at(-1).params.prompt[0].text, /Conversation so far/);
    } finally { await client.close(); agent.clear(); }
  });

  test('Clear during initialization cannot revive an ACP session or MCP endpoint', async () => {
    let release!: () => void;
    const agent = makeRuntime({ executable: async () => { await new Promise<void>(resolve => { release = resolve; }); return executable; } });
    const warming = agent.warm(input(), new AbortController().signal);
    agent.clear(); release(); await assert.rejects(warming, /Stopped/);
    assert.equal((agent as any).acp, undefined); assert.equal((agent as any).endpoint, undefined);
  });

  test('preparation expires, and its idle timer never remains armed during a user turn', async () => {
    const agent = makeRuntime(), original = global.setTimeout;
    global.setTimeout = ((handler: any, timeout: number, ...args: any[]) => original(handler, timeout === 15_000 ? 30 : timeout, ...args)) as typeof setTimeout;
    try {
      await agent.warm(input('hold-preparation ignore-cancel'), new AbortController().signal);
      await until(async () => !(agent as any).acp);
      assert.equal((agent as any).endpoint, undefined);
    } finally { global.setTimeout = original; agent.clear(); }
    const active = makeRuntime();
    await active.warm(input('hold-preparation'), new AbortController().signal);
    await active.run(input('fresh'), () => {
      const timer = (active as any).idleTimer;
      assert.ok(!timer || timer._destroyed, 'completed preparation must not time out a long user turn');
    }, new AbortController().signal);
  });

  for (const opening of [false, true]) test(`connection readiness without speculative inference${opening ? ' for graph setup' : ''}`, async () => {
    const acp = makeRuntime(), signal = new AbortController().signal;
    const { context, messages, ...connection } = input();
    const intent = opening ? 'open_graph' as const : undefined;
    await acp.warm({ ...connection, intent }, signal);
    let wire = await journal();
    assert.equal(wire.filter(m => m.method === 'session/new').length, 1);
    assert.equal(wire.filter(m => m.method === 'session/prompt').length, 0, 'connection readiness cannot invent context or perform a setup task');
    await acp.run({ ...connection, context, messages, intent }, () => {}, signal);
    wire = await journal();
    assert.equal(wire.filter(m => m.method === 'session/new').length, 1, 'Send reuses the prepared connection');
    assert.equal(wire.filter(m => m.method === 'session/prompt').length, 1);
  });

  test('configuration invalidation replaces an ACP connection while retaining historical evidence', async () => {
    const agent = makeRuntime(), signal = new AbortController().signal;
    await agent.run({ ...input(), selectionEvidence: { modelId: 'model', revision: 'a'.repeat(64),
      nodes: [{ parameters: { calibration_code: 37 } }] } }, () => {}, signal);
    const old = (agent as any).acp;
    agent.invalidate();
    assert.equal(old.controller.signal.aborted, true);
    assert.equal((agent as any).acp, undefined);
    assert.equal((agent as any).endpoint, undefined);
    await agent.run(input('fresh revision'), () => {}, signal);
    const wire = await journal();
    assert.equal(wire.filter(m => m.method === 'session/new').length, 2);
    assert.match(wire.filter(m => m.method === 'session/prompt').at(-1).params.prompt[0].text, /calibration_code":37/);
  });

});
