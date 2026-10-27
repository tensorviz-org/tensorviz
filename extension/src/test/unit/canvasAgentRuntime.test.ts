import * as assert from 'node:assert/strict';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { modelRefSchema } from '@tensorviz/graph-host/mcpGraph';
import { mcpEditSchema } from '../../mcpProposals';
import { CanvasAgentRuntime } from '../../canvasAgentRuntime';

suite('Canvas agent provider compatibility', () => {
  test('follow-up turns can find an earlier native proposal without exposing its ID in the chat', async () => {
    const runtime = new CanvasAgentRuntime({ executable: async () => 'fixture', release() {}, tools: () => new Map([
      ['propose_edit', { config: { inputSchema: { ...modelRefSchema, edit: mcpEditSchema }, annotations: { readOnlyHint: false } },
        handler: async () => ({ content: [], structuredContent: { proposalId: 'native-review-1', status: 'preparing' } }) }],
    ]) });
    const inputs: string[] = [];
    const fixture = runtime as unknown as { token: string; codex(executable: string, input: { context: string }, url: string): Promise<void> };
    fixture.codex = async (_executable, input, url) => {
      inputs.push(input.context);
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

  test('completed turns release their listeners but retain native-review ownership until Clear', async () => {
    const released: string[] = [], owners: string[] = [], urls: string[] = [];
    const runtime = new CanvasAgentRuntime({ executable: async () => 'fixture', release: owner => released.push(owner), tools: () => new Map() });
    const fixture = runtime as unknown as { owner: string; codex(executable: string, input: unknown, url: string): Promise<void> };
    fixture.codex = async (_executable, _input, url) => { owners.push(fixture.owner); urls.push(url); };
    try {
      for (let i = 0; i < 20; i++) {
        await runtime.run({ provider: 'codex', cwd: '/fixture', context: '', messages: [] }, () => {}, new AbortController().signal);
        await assert.rejects(fetch(urls.at(-1)!));
      }
      assert.equal(new Set(owners).size, 1, 'follow-up turns must still own pending native reviews');
      assert.equal(released.length, 0);
    } finally { runtime.clear(); }
    assert.deepEqual(released, [owners[0]]);
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
