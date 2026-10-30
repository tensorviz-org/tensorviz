import * as assert from 'node:assert/strict';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { modelRefSchema } from '@tensorviz/graph-host/mcpGraph';
import { mcpEditSchema } from '../../mcpProposals';
import { CanvasAgentRuntime } from '../../canvasAgentRuntime';
import type { AgentRunEvent, AgentRunInput } from '../../canvasAgent';

suite('Canvas agent provider compatibility', () => {
  test('only a current host opening result hands the conversation to its selected model', async () => {
    for (const outcome of ['ready', 'needs_setup', 'failed', 'cancelled', 'malformed', 'unrelated', 'refused', 'aborted']) {
      const target = { workspaceFolder: '/fixture', file: 'checkpoint_model.py', cls: 'CheckpointModel',
        documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 2 };
      const controller = new AbortController(), events: AgentRunEvent[] = [];
      const name = outcome === 'unrelated' ? 'read_model_source' : 'open_graph';
      const runtime = new CanvasAgentRuntime({ executable: async () => 'fixture', release() {}, tools: () => new Map([
        [name, { config: { inputSchema: {}, annotations: { readOnlyHint: false } }, handler: async () => {
          if (outcome === 'aborted') controller.abort();
          return { ...(outcome === 'refused' ? { isError: true } : {}), content: [], structuredContent: {
            opening: { id: 'opening', target: outcome === 'malformed' ? { ...target, sourceDigest: 'invalid' } : target,
              phase: ['needs_setup', 'failed', 'cancelled'].includes(outcome) ? outcome : 'ready',
              hasCapture: outcome === 'ready', detail: 'Host opening state' },
          } };
        } }],
      ]) });
      const fixture = runtime as any;
      let url = '';
      fixture.startCodex = async (_executable: string, _input: unknown, endpoint: string) => {
        url = endpoint; return { process: { completed: new Promise(() => {}) }, sent: false };
      };
      fixture.turn = async () => {
        const client = new Client({ name: 'opening-handoff-test', version: '1' });
        try {
          await client.connect(new StreamableHTTPClientTransport(new URL(url), { requestInit: { headers: { Authorization: `Bearer ${fixture.token}` } } }));
          await client.callTool({ name, arguments: {} });
        } finally { await client.close(); }
      };
      try {
        const running = runtime.run({ provider: 'codex', cwd: '/fixture', conversationId: 'chat', context: 'Original model', messages: [] },
          event => events.push(event), controller.signal);
        if (outcome === 'aborted') await assert.rejects(running); else await running;
        const handoffs = events.filter(event => event.kind === 'opened_graph');
        assert.deepEqual(handoffs, ['ready', 'needs_setup', 'failed'].includes(outcome)
          ? [{ kind: 'opened_graph', text: '', target }] : [], outcome);
      } finally { runtime.clear(); }
    }
  });

  test('host-attached evidence survives Stop and effort changes but never crosses a conversation, provider or workspace', async () => {
    for (const change of ['stop', 'effort', 'clear', 'conversation', 'provider', 'cwd']) {
      const runtime = new CanvasAgentRuntime({ executable: async () => 'fixture', release() {}, tools: () => new Map() });
      const inputs: string[] = [];
      const fixture = runtime as any;
      fixture.startCodex = fixture.startClaude = async () => ({ process: { completed: new Promise(() => {}) }, sent: false });
      fixture.turn = async (_session: unknown, _input: unknown, text: string) => { inputs.push(text); };
      const base: AgentRunInput = { provider: 'codex', cwd: '/fixture', conversationId: 'chat-one', context: 'Current graph', messages: [], settings: { effort: 'medium' } };
      try {
        await runtime.run({ ...base, selectionEvidence: { modelId: 'model', revision: 'a'.repeat(64), nodes: [{ parameters: { calibration_code: 37 } }] } }, () => {}, new AbortController().signal);
        if (change === 'stop') runtime.clear(true);
        if (change === 'clear') runtime.clear();
        const next: AgentRunInput = { ...base, ...(change === 'effort' ? { settings: { effort: 'high' } } : {}),
          ...(change === 'conversation' ? { conversationId: 'chat-two' } : {}), ...(change === 'provider' ? { provider: 'claude' as const } : {}),
          ...(change === 'cwd' ? { cwd: '/other' } : {}) };
        await runtime.run(next, () => {}, new AbortController().signal);
        assert.doesNotMatch(inputs[0], /Earlier graph inspections/, 'first turn must not duplicate host-attached facts');
        assert.equal(inputs[1].includes('calibration_code'), ['stop', 'effort'].includes(change), change);
      } finally { runtime.clear(); }
    }
  });

  test('successful read-only tool evidence survives reset; aborted and refused inspections do not enter memory', async () => {
    for (const outcome of ['success', 'aborted', 'refused']) {
      const controller = new AbortController(), inputs: string[] = [];
      const runtime = new CanvasAgentRuntime({ executable: async () => 'fixture', release() {}, tools: () => new Map([
        ['inspect_nodes', { config: { inputSchema: {}, annotations: { readOnlyHint: true } }, handler: async () => {
          if (outcome === 'aborted') controller.abort();
          return { ...(outcome === 'refused' ? { isError: true } : {}), content: [],
            structuredContent: { modelId: 'model', revision: 'a'.repeat(64), nodes: [{ parameters: { calibration_code: 37 } }] } };
        } }],
      ]) });
      const fixture = runtime as any;
      let url = '';
      fixture.startCodex = async (_executable: string, _input: unknown, endpoint: string) => {
        url = endpoint; return { process: { completed: new Promise(() => {}) }, sent: false };
      };
      fixture.turn = async (_session: unknown, _input: unknown, text: string) => {
        inputs.push(text); if (inputs.length > 1) return;
        const client = new Client({ name: 'evidence-test', version: '1' });
        try {
          await client.connect(new StreamableHTTPClientTransport(new URL(url), { requestInit: { headers: { Authorization: `Bearer ${fixture.token}` } } }));
          await client.callTool({ name: 'inspect_nodes', arguments: {} });
        } finally { await client.close(); }
      };
      const input: AgentRunInput = { provider: 'codex', cwd: '/fixture', conversationId: 'chat', context: 'Current graph', messages: [] };
      try {
        const running = runtime.run(input, () => {}, controller.signal);
        if (outcome === 'aborted') await assert.rejects(running); else await running;
        runtime.clear(true);
        await runtime.run(input, () => {}, new AbortController().signal);
        assert.equal(inputs[1].includes('calibration_code'), outcome === 'success', outcome);
      } finally { runtime.clear(); }
    }
  });

  test('follow-up turns can find an earlier native proposal without exposing its ID in the chat', async () => {
    const runtime = new CanvasAgentRuntime({ executable: async () => 'fixture', release() {}, tools: () => new Map([
      ['propose_edit', { config: { inputSchema: { ...modelRefSchema, edit: mcpEditSchema }, annotations: { readOnlyHint: false } },
        handler: async () => ({ content: [], structuredContent: { proposalId: 'native-review-1', status: 'preparing' } }) }],
    ]) });
    const inputs: string[] = [];
    const fixture = runtime as any;
    let url = '';
    fixture.startCodex = async (_executable: string, _input: unknown, endpoint: string) => {
      url = endpoint; return { process: { completed: new Promise(() => {}) }, sent: false };
    };
    fixture.turn = async (_session: unknown, _input: unknown, text: string) => {
      inputs.push(text);
      if (inputs.length !== 1) return;
      const client = new Client({ name: 'canvas-receipt-test', version: '1' });
      try {
        await client.connect(new StreamableHTTPClientTransport(new URL(url), { requestInit: { headers: { Authorization: `Bearer ${fixture.token}` } } }));
        await client.callTool({ name: 'propose_edit', arguments: { modelId: 'model', revision: 'a'.repeat(64), edit: { kind: 'remove_layer', nodeId: 'act' } } });
      } finally { await client.close(); }
    };
    const turn = () => runtime.run({ provider: 'codex', cwd: '/fixture', context: 'current model', messages: [] }, () => {}, new AbortController().signal);
    try {
      await turn(); await turn();
      assert.match(inputs[1], /native-review-1/);
      assert.match(inputs[1], /poll current status/);
      runtime.clear(); await turn();
      assert.doesNotMatch(inputs[2], /native-review-1/);
    } finally { runtime.clear(); }
  });

  test('completed turns retain one endpoint and native-review owner until Clear', async () => {
    const released: string[] = [], owners: string[] = [], urls: string[] = [];
    const runtime = new CanvasAgentRuntime({ executable: async () => 'fixture', release: owner => released.push(owner), tools: () => new Map() });
    const fixture = runtime as any;
    fixture.startCodex = async (_executable: string, _input: unknown, url: string) => {
      owners.push(fixture.owner); urls.push(url); return { process: { completed: new Promise(() => {}) }, sent: false };
    };
    fixture.turn = async () => {};
    try {
      for (let i = 0; i < 20; i++) {
        await runtime.run({ provider: 'codex', cwd: '/fixture', context: '', messages: [] }, () => {}, new AbortController().signal);
        assert.equal((await fetch(urls.at(-1)!)).status, 401, 'the retained endpoint still requires its private token');
      }
      assert.equal(new Set(owners).size, 1, 'follow-up turns must still own pending native reviews');
      assert.equal(urls.length, 1, 'follow-ups must not recreate the transport');
      assert.equal(released.length, 0);
    } finally { runtime.clear(); }
    assert.deepEqual(released, [owners[0]]);
    await assert.rejects(fetch(urls[0]));
  });

  test('the Codex-facing edit schema still enforces the original typed edit boundary', async () => {
    const accepted: unknown[] = [];
    const runtime = new CanvasAgentRuntime({ executable: async () => '', release() {}, tools: () => new Map([
      ['propose_edit', { config: { inputSchema: { ...modelRefSchema, edit: mcpEditSchema } }, handler: async args => {
        accepted.push(args); return { content: [{ type: 'text', text: 'Native review prepared.' }] };
      } }],
    ]) });
    const fixture = runtime as unknown as { connect(provider: 'codex'): Promise<{ url: string }>; token: string };
    const { url } = await fixture.connect('codex');
    const client = new Client({ name: 'canvas-schema-test', version: '1' });
    try {
      await client.connect(new StreamableHTTPClientTransport(new URL(url), { requestInit: { headers: { Authorization: `Bearer ${fixture.token}` } } }));
      const ref = { modelId: 'model', revision: 'a'.repeat(64) };
      const invalid = await client.callTool({ name: 'propose_edit', arguments: { ...ref, edit: { kind: 'write_file', path: '/other.py', code: 'anything' } } });
      assert.equal(invalid.isError, true); assert.equal(accepted.length, 0);
      const extra = await client.callTool({ name: 'propose_edit', arguments: { ...ref, edit: { kind: 'remove_layer', nodeId: 'act', path: '/other.py' } } });
      assert.equal(extra.isError, true); assert.equal(accepted.length, 0);
      const valid = await client.callTool({ name: 'propose_edit', arguments: { ...ref, edit: { kind: 'remove_layer', nodeId: 'act' } } });
      assert.notEqual(valid.isError, true); assert.equal(accepted.length, 1);
    } finally { await client.close(); runtime.clear(); }
  });
});
