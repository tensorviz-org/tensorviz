import * as assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { z } from 'zod';
import { CanvasNodeExplanation, type CanvasNodeExplanationOptions } from '../../canvasNodeExplanation';
import type { CanvasAgentRoute, AgentRunInput } from '../../canvasAgent';
import type { HostToolset } from '../../mcpHost';
import type { NodeExplanationCitation, NodeExplanationState } from '@tensorviz/contracts/nodeExplanation';
import type { LocalBrowserNodeExplanationRequest, LocalBrowserViewSelectionMessage } from '@tensorviz/contracts/protocol';

const hash = (value: string) => createHash('sha256').update(value).digest('hex');
const until = async (condition: () => boolean) => {
  const deadline = Date.now() + 3000;
  while (!condition()) { if (Date.now() > deadline) throw new Error('Explanation fixture did not settle.'); await new Promise(resolve => setTimeout(resolve, 2)); }
};
const payload = (citationIds = ['evidence-1']) => ({ title: 'Score candidates', purpose: 'Compare candidate features with the question.',
  inputs: ['Candidate features', 'Question features'], outputs: ['Candidate scores'], steps: ['Batch candidates and queries', 'Apply the scoring head'], uncertainties: ['Tensor shapes have not been recorded.'], citationIds });

suite('Source-only node explanations', () => {
  const services: CanvasNodeExplanation[] = [];
  teardown(() => { for (const service of services.splice(0)) service.dispose(); });
  function rig(run?: (tools: HostToolset, input: AgentRunInput, signal: AbortSignal) => Promise<void>, extras: Partial<CanvasNodeExplanationOptions> = {}) {
    let source = Array.from({ length: 220 }, (_, index) => index === 149 ? 'def score(candidates, question): return head(candidates, question)' : `# source line ${index + 1}`).join('\n');
    let valid = true, ref = { modelId: 'model-one', revision: hash('revision-one') };
    let route: CanvasAgentRoute = { provider: 'codex', settings: { model: 'fixture-model', effort: 'medium' }, resolvedModel: 'fixture-model', ready: true, busy: false };
    const target = { workspaceFolder: '/fixture', file: 'model.py', cls: 'Net', sourceDigest: hash(source), documentVersion: 1, backendRevision: 0 };
    const selection: LocalBrowserViewSelectionMessage = { type: 'viewSelection', protocolVersion: 1, workspaceSessionId: 'session', messageId: 'selection', viewId: 'view-one', target, nodeIds: ['node-one'], edgeIds: [] };
    const states: NodeExplanationState[] = [], inputs: AgentRunInput[] = [], signals: AbortSignal[] = [], toolsets: HostToolset[] = [], reveals: NodeExplanationCitation[] = [];
    const queries: unknown[] = []; let clears = 0, readCount = 0;
    const base: HostToolset = new Map([
      ['request_project_command', { config: { inputSchema: {} }, handler: () => { throw new Error('Command authority leaked.'); } }],
      ['configure_graph_environment', { config: { inputSchema: {} }, handler: () => { throw new Error('Setup authority leaked.'); } }],
      ['query_source_context', { config: { inputSchema: { modelId: z.string(), revision: z.string(), file: z.string(), symbol: z.string(), assumeValue: z.number().optional() } },
        handler: async (args: unknown) => { queries.push(args); return { content: [], structuredContent: { items: [{ kind: 'definition', name: 'score', excerpt: source.split('\n')[149],
          anchor: { file: 'model.py', digest: hash(source), startLine: 150, endLine: 150 } }] } }; } }],
    ]);
    const options: CanvasNodeExplanationOptions = { route: () => structuredClone(route), reference: () => ({ ...ref }),
      current: (candidate, revision) => valid && candidate.viewId === selection.viewId && candidate.nodeIds.length === 1 && candidate.nodeIds[0] === selection.nodeIds[0] && (!revision || revision === ref.revision),
      prepare: async () => ({ ref: { ...ref }, context: { call: 'self.score(candidates, question)' }, source: { file: 'model.py', startLine: 30 } }),
      read: async (_target, file) => { readCount++; if (file !== 'model.py') throw new Error('Not a graph-recorded source.'); return { file, source, digest: hash(source) }; },
      tools: () => base,
      runtime: load => ({ run: async (input, _emit, signal) => { const tools = load('private-owner'); inputs.push(input); signals.push(signal); toolsets.push(tools);
        if (run) await run(tools, input, signal); else await call(tools, 'submit_node_explanation', payload()); }, clear: () => { clears++; } }),
      publish: (_view, _target, state) => states.push(state), reveal: async (_target, citation) => { reveals.push(citation); }, debounceMs: 0, ...extras };
    const service = new CanvasNodeExplanation(options); services.push(service);
    const request = (action: 'retry' | 'reveal', changes: Record<string, unknown> = {}) => {
      const state = states.at(-1)!;
      return service.request({ type: 'nodeExplanationRequest', protocolVersion: 1, workspaceSessionId: 'session', messageId: 'request', viewId: selection.viewId, target, nodeId: state.nodeId, requestId: state.requestId,
        action, ...(action === 'reveal' ? { citationId: 'evidence-1' } : {}), ...changes } as LocalBrowserNodeExplanationRequest);
    };
    return { service, selection, states, inputs, signals, toolsets, reveals, queries, request,
      start: () => service.select(selection), done: () => until(() => states.at(-1)?.status === 'ready' || states.at(-1)?.status === 'unavailable'),
      get clears() { return clears; }, get reads() { return readCount; }, get route() { return route; }, set route(value: CanvasAgentRoute) { route = value; },
      set valid(value: boolean) { valid = value; }, set revision(value: string) { ref = { ...ref, revision: value }; }, set source(value: string) { source = value; } };
  }
  async function call(tools: HostToolset, name: string, args: unknown) {
    const tool = tools.get(name)!;
    return tool.handler(z.object(tool.config.inputSchema).strict().parse(args), { signal: new AbortController().signal });
  }

  test('a settled selection produces typed evidence without borrowing the user conversation', async () => {
    const r = rig(); r.start(); await r.done();
    assert.deepEqual(r.states.map(state => state.status), ['queued', 'loading', 'ready']);
    assert.ok(r.states.every(state => state.revision === hash('revision-one')));
    assert.equal(r.inputs[0].intent, 'node_explanation'); assert.equal(r.inputs[0].messages.length, 1);
    assert.equal(r.inputs[0].cwd, '/fixture'); assert.deepEqual(r.inputs[0].settings, r.route.settings);
    assert.equal(r.states.at(-1)!.explanation!.citations[0].file, 'model.py'); assert.equal(r.clears, 1);
    assert.deepEqual([...r.toolsets[0].keys()].sort(), ['query_source_context', 'read_model_source', 'submit_node_explanation']);
  });

  test('unready or foreground-busy routes never start discovery or a background provider', async () => {
    for (const state of [{ ready: false, busy: false }, { ready: true, busy: true }]) {
      const r = rig(); r.route = { ...r.route, ...state }; r.start(); await r.done();
      assert.equal(r.inputs.length, 0); assert.equal(r.reads, 0); assert.equal(r.states.at(-1)!.status, 'unavailable');
    }
  });

  test('clear, multi-selection and disconnected views never start inference', async () => {
    const r = rig(undefined, { debounceMs: 20 });
    for (const selected of [{ nodeIds: [] }, { nodeIds: ['node-one', 'node-two'] }, { edgeIds: ['edge-one'] }, { tooLarge: true as const }]) {
      r.service.select({ ...r.selection, ...selected });
    }
    r.start(); r.service.disconnect(r.selection.viewId); await new Promise(resolve => setTimeout(resolve, 25));
    assert.equal(r.inputs.length, 0);
  });

  test('selection deduplicates active work and successful cache is reused after reselection', async () => {
    const r = rig(); r.start(); r.start(); await r.done();
    r.service.select({ ...r.selection, nodeIds: [] }); r.start(); await r.done();
    assert.equal(r.inputs.length, 1); assert.equal(r.states.at(-1)!.status, 'ready');
  });

  test('model or full graph revision changes cannot reuse a previous explanation', async () => {
    const r = rig(); r.start(); await r.done();
    r.revision = hash('revision-two'); r.start(); await r.done(); assert.equal(r.inputs.length, 2);
    assert.equal(r.states.at(-1)!.revision, hash('revision-two'));
    r.route = { ...r.route, resolvedModel: 'other-model', settings: { model: 'other-model' } }; r.service.refresh(); await r.done();
    assert.equal(r.inputs.length, 3);
  });

  test('a helper outside the initial callsite window can be read and cited exactly', async () => {
    const r = rig(async tools => {
      const source = await call(tools, 'read_model_source', { file: 'model.py', startLine: 150, maxLines: 1 });
      assert.equal(source.isError, undefined); assert.match((source.structuredContent as any).source, /def score/);
      const result = await call(tools, 'submit_node_explanation', payload([(source.structuredContent as any).citation.id])); assert.equal(result.isError, undefined);
    });
    r.start(); await r.done(); assert.equal(r.states.at(-1)!.explanation!.citations[0].startLine, 150);
  });

  test('query tools pin the current model and revision and cannot enable hypothetical edits', async () => {
    const r = rig(async tools => {
      for (const args of [{ modelId: 'other', revision: hash('revision-one') }, { modelId: 'model-one', revision: hash('other') }, { modelId: 'model-one', revision: hash('revision-one'), assumeValue: 4 }]) {
        assert.equal((await call(tools, 'query_source_context', { file: 'model.py', symbol: 'score', ...args })).isError, true);
      }
      await call(tools, 'submit_node_explanation', payload());
    });
    r.start(); await r.done(); assert.equal(r.queries.length, 0); assert.equal(r.states.at(-1)!.status, 'ready');
  });

  test('bounded source queries create host-owned citation IDs and smaller defaults', async () => {
    const r = rig(async tools => {
      const result = await call(tools, 'query_source_context', { modelId: 'model-one', revision: hash('revision-one'), file: 'model.py', symbol: 'score' });
      assert.equal(result.isError, undefined);
      const citation = (result.structuredContent as any).citations[0];
      assert.equal(citation.startLine, 150); await call(tools, 'submit_node_explanation', payload([citation.id]));
    });
    r.start(); await r.done(); assert.deepEqual(r.queries[0], { modelId: 'model-one', revision: hash('revision-one'), file: 'model.py', symbol: 'score', limit: 16, maxEntities: 32, depth: 2 });
  });

  test('read windows and shared read count enforce bounded tools', async () => {
    const r = rig(async tools => {
      await assert.rejects(call(tools, 'read_model_source', { file: 'model.py', maxLines: 121 }));
      for (let index = 0; index < 4; index++) assert.equal((await call(tools, 'read_model_source', { file: 'model.py', startLine: 150, maxLines: 1 })).isError, undefined);
      assert.equal((await call(tools, 'read_model_source', { file: 'model.py' })).isError, true);
      assert.equal((await call(tools, 'read_model_source', { file: '../outside.py' })).isError, true);
      await call(tools, 'submit_node_explanation', payload());
    }); r.start(); await r.done(); assert.equal(r.states.at(-1)!.status, 'ready');
  });

  test('unseen, duplicated or model-supplied citation authority cannot produce a result', async () => {
    for (const args of [payload(['invented']), payload(['evidence-1', 'evidence-1']), { ...payload(), citations: [{ file: '/private/key', startLine: 1, endLine: 2 }] }]) {
      const r = rig(async tools => {
        const result = await tools.get('submit_node_explanation')!.handler(args, {}); assert.equal(result.isError, true);
      }); r.start(); await r.done(); assert.equal(r.states.at(-1)!.status, 'unavailable');
    }
  });

  test('failed over-budget evidence cannot be cited through a guessed host ID', async () => {
    const source = Array.from({ length: 220 }, (_, index) => index === 149 ? 'def score(candidates, question): return head(candidates, question)' : `# source line ${index + 1}`).join('\n');
    const tools: HostToolset = new Map([['query_source_context', { config: { inputSchema: { modelId: z.string(), revision: z.string(), file: z.string(), symbol: z.string() } },
      handler: async () => ({ content: [], structuredContent: { items: [{ excerpt: source.split('\n')[149], anchor: { file: 'model.py', digest: hash(source), startLine: 150, endLine: 150 } }], padding: 'x'.repeat(48 * 1024) } }) }]]);
    const r = rig(async toolset => {
      const failed = await call(toolset, 'query_source_context', { modelId: 'model-one', revision: hash('revision-one'), file: 'model.py', symbol: 'score' });
      assert.equal(failed.isError, true); assert.equal((failed.structuredContent as any).error.code, 'SOURCE_LIMIT');
      assert.equal((await call(toolset, 'submit_node_explanation', payload(['evidence-2']))).isError, true);
      // The exhausted cumulative budget also makes this otherwise valid read fail.
      assert.equal((await call(toolset, 'read_model_source', { file: 'model.py', startLine: 160, maxLines: 1 })).isError, true);
      assert.equal((await call(toolset, 'submit_node_explanation', payload(['evidence-3']))).isError, true);
      assert.equal((await call(toolset, 'submit_node_explanation', payload())).isError, undefined);
    }, { tools: () => tools });
    r.start(); await r.done(); assert.equal(r.states.at(-1)!.status, 'ready');
    assert.deepEqual(r.states.at(-1)!.explanation!.citations.map(citation => citation.id), ['evidence-1']);
  });

  test('changed evidence cannot be submitted or used to navigate', async () => {
    const first = rig(async tools => { first.source = 'changed after evidence was read'; assert.equal((await call(tools, 'submit_node_explanation', payload())).isError, true); });
    first.start(); await first.done(); assert.equal(first.states.at(-1)!.status, 'unavailable');
    const second = rig(); second.start(); await second.done(); second.source = 'changed after publication'; await second.request('reveal'); assert.equal(second.reveals.length, 0);
  });

  test('source freshness and foreground activity abort an in-flight run and suppress late results', async () => {
    let release!: () => void;
    const r = rig(async (tools, _input, signal) => { await new Promise<void>(resolve => { release = resolve; signal.addEventListener('abort', () => resolve(), { once: true }); });
      assert.equal((await call(tools, 'submit_node_explanation', payload())).isError, true); });
    r.start(); await until(() => r.inputs.length === 1); r.route = { ...r.route, busy: true }; r.service.refresh();
    await until(() => r.clears === 1); assert.equal(r.signals[0].aborted, true); release();
    assert.equal(r.states.some(state => state.status === 'ready'), false);
    r.route = { ...r.route, busy: false }; r.valid = false; r.service.refresh();
    assert.equal(r.inputs.length, 1);
  });

  test('unstructured provider prose is not accepted as a structured result', async () => {
    const r = rig(async () => {}); r.start(); await r.done(); assert.equal(r.states.at(-1)!.status, 'unavailable');
  });

  test('the deadline aborts the isolated provider and leaves a retryable state', async () => {
    const original = global.setTimeout;
    global.setTimeout = ((handler: (...args: any[]) => void, delay?: number, ...args: any[]) => original(handler, delay === 45000 ? 8 : delay, ...args)) as typeof setTimeout;
    try {
      const r = rig(async (_tools, _input, signal) => { await new Promise<void>(resolve => signal.addEventListener('abort', () => resolve(), { once: true })); });
      r.start(); await until(() => r.clears === 1);
      assert.equal(r.signals[0].aborted, true); assert.equal(r.states.at(-1)!.status, 'unavailable');
      assert.match(r.states.at(-1)!.detail!, /timed out/);
      const id = r.states.at(-1)!.requestId; await r.request('retry'); await until(() => r.clears === 2);
      assert.notEqual(r.states.at(-1)!.requestId, id);
    } finally { global.setTimeout = original; }
  });

  test('oversized source previews fail closed before starting a provider', async () => {
    const r = rig(); r.source = 'x'.repeat(13 * 1024); r.start(); await r.done();
    assert.equal(r.inputs.length, 0); assert.equal(r.states.at(-1)!.status, 'unavailable');
  });

  test('only a current ready request can reveal evidence and retry produces a fresh request', async () => {
    const r = rig(); r.start(); await r.done(); const requestId = r.states.at(-1)!.requestId;
    await r.request('reveal', { requestId: 'other-request' }); await r.request('reveal', { citationId: 'unknown' });
    await r.request('reveal'); assert.equal(r.reveals.length, 1);
    await r.request('retry'); await r.done(); assert.notEqual(r.states.at(-1)!.requestId, requestId); assert.equal(r.inputs.length, 2);
  });

  test('a new selection waits for cancellation of the previous isolated run', async () => {
    let active = 0, maximum = 0, finish!: () => void;
    const r = rig(async (tools, _input, signal) => {
      maximum = Math.max(maximum, ++active);
      if (r.inputs.length === 1) await new Promise<void>(resolve => { finish = resolve; });
      if (!signal.aborted) await call(tools, 'submit_node_explanation', payload()); active--;
    });
    r.start(); await until(() => r.inputs.length === 1);
    r.service.select({ ...r.selection, nodeIds: [] }); r.start(); await new Promise(resolve => setTimeout(resolve, 10)); assert.equal(r.inputs.length, 1);
    finish(); await r.done(); assert.equal(maximum, 1); assert.equal(r.inputs.length, 2);
  });
});
