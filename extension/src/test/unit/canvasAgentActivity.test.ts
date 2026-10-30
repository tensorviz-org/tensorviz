import * as assert from 'node:assert/strict';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { z } from 'zod';
import { CanvasAgentRuntime } from '../../canvasAgentRuntime';
import type { AgentRunEvent } from '../../canvasAgent';

suite('Canvas agent tool activity', () => {
  for (const cancelled of [false, true]) test(`overlapping tools finish accurately (cancelled=${cancelled})`, async () => {
    const events: AgentRunEvent[] = [], completions = new Map<number, () => void>();
    const runtime = new CanvasAgentRuntime({ pipeline: () => 'acp', executable: async () => 'fixture', release() {},
      tools: () => new Map([['read_model_source', { config: { inputSchema: { id: z.number() } }, handler: async args => {
        await new Promise<void>(resolve => completions.set(args.id, resolve));
        return { content: [] };
      } }]]) });
    const fixture = runtime as any, controller = new AbortController();
    fixture.emit = (event: AgentRunEvent) => events.push(event);
    fixture.toolTurn = { signal: controller.signal, activeTools: 0 };
    const client = new Client({ name: 'activity-test', version: '1' });
    try {
      const { url } = await fixture.connect('claude');
      await client.connect(new StreamableHTTPClientTransport(new URL(url), { requestInit: { headers: { Authorization: `Bearer ${fixture.token}` } } }));
      const first = client.callTool({ name: 'read_model_source', arguments: { id: 1 } });
      const second = client.callTool({ name: 'read_model_source', arguments: { id: 2 } });
      for (let i = 0; completions.size < 2 && i < 100; i++) await new Promise(resolve => setTimeout(resolve, 10));
      assert.equal(completions.size, 2);
      assert.equal(events.at(-1)?.text, 'Working with your model…');
      completions.get(1)!(); await first;
      assert.equal(events.at(-1)?.text, 'Working with your model…', 'one tool still runs');
      if (cancelled) {
        controller.abort(); fixture.toolTurn = { signal: new AbortController().signal, activeTools: 0 };
      }
      const before = events.length;
      completions.get(2)!(); await second;
      if (cancelled) assert.equal(events.length, before, 'a late tool cannot overwrite the new turn');
      else assert.equal(events.at(-1)?.text, 'Waiting for Claude…');
    } finally { for (const finish of completions.values()) finish(); await client.close(); runtime.clear(); }
  });
});
