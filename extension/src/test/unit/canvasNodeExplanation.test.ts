import * as assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { z } from 'zod';
import { CanvasNodeExplanation, nodeSourceDocumentationSymbol, type CanvasNodeExplanationOptions } from '../../canvasNodeExplanation';
import type { CanvasAgentRoute, AgentRunInput } from '../../canvasAgent';
import type { HostToolset } from '../../mcpHost';
import type { NodeExplanationCitation, NodeExplanationState, NodeSourceDocumentation } from '@tensorviz/contracts/nodeExplanation';
import type { LocalBrowserNodeExplanationRequest, LocalBrowserViewSelectionMessage } from '@tensorviz/contracts/protocol';

const hash = (value: string) => createHash('sha256').update(value).digest('hex');
const until = async (condition: () => boolean) => {
  const deadline = Date.now() + 3000;
  while (!condition()) { if (Date.now() > deadline) throw new Error('Explanation fixture did not settle.'); await new Promise(resolve => setTimeout(resolve, 2)); }
};
const payload = (citationIds = ['evidence-1']) => ({ title: 'Score candidates', purpose: 'Compare candidate features with the question.',
  inputs: ['Candidate features', 'Question features'], outputs: ['Candidate scores'], steps: ['Batch candidates and queries', 'Apply the scoring head'], uncertainties: ['Tensor shapes have not been recorded.'], citationIds });
const fixtureSource = () => Array.from({ length: 220 }, (_, index) => index === 149 ? 'def score(candidates, question): return head(candidates, question)' : `# source line ${index + 1}`).join('\n');
const documentation = (): NodeSourceDocumentation => ({ status: 'resolved', qualifiedName: 'score', kind: 'function', signature: 'def score(candidates, question)', truncated: false,
  citation: { id: 'documentation-fixture', file: 'model.py', sourceDigest: hash(fixtureSource()), startLine: 150, endLine: 150 } });

suite('Source-only node explanations', () => {
  const services: CanvasNodeExplanation[] = [];
  teardown(() => { for (const service of services.splice(0)) service.dispose(); });
  function rig(run?: (tools: HostToolset, input: AgentRunInput, signal: AbortSignal) => Promise<void>, extras: Partial<CanvasNodeExplanationOptions> = {}) {
    let source = fixtureSource();
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

  test('model-enriched interpretations are reusable and revealable while foreground work is busy', async () => {
    const explanation = { ...payload(), citations: [{ id: 'model-evidence', file: 'model.py', sourceDigest: hash(fixtureSource()), startLine: 1, endLine: 5 }] };
    const { citationIds, ...cached } = explanation;
    const r = rig(undefined, { cached: () => cached }); r.route = { ...r.route, busy: true }; r.start(); await r.done();
    assert.equal(r.inputs.length, 0); assert.equal(r.states.at(-1)!.status, 'ready');
    assert.equal(r.states.at(-1)!.explanation!.title, 'Score candidates'); await r.request('reveal', { citationId: 'model-evidence' }); assert.equal(r.reveals.length, 1);
  });

  test('a completed model batch updates an already-selected node without borrowing chat or regenerating', async () => {
    const r = rig(); r.route = { ...r.route, busy: true }; r.start(); await r.done();
    const { citationIds, ...generated } = payload();
    const state: NodeExplanationState = { requestId: 'model-run', nodeId: 'node-one', revision: hash('revision-one'), status: 'ready', documentation: documentation(),
      explanation: { ...generated, citations: [{ id: 'batch-evidence', file: 'model.py', sourceDigest: hash(fixtureSource()), startLine: 1, endLine: 5 }] } };
    await r.service.accept(r.selection.viewId, r.selection.target, state);
    assert.equal(r.states.at(-1)!.status, 'ready'); assert.equal(r.inputs.length, 0);
    await r.request('reveal', { citationId: 'batch-evidence' }); assert.equal(r.reveals.length, 1);
    assert.deepEqual(r.states.at(-1)!.documentation, documentation()); await r.request('reveal', { citationId: 'documentation-fixture' }); assert.equal(r.reveals.length, 2);
  });

  test('literal source signatures load and reveal with no connected agent or while chat is busy', async () => {
    for (const availability of [{ ready: false, busy: false }, { ready: true, busy: true }]) {
      let calls = 0;
      const r = rig(undefined, { documentation: async () => { calls++; return documentation(); } });
      r.route = { ...r.route, ...availability }; r.start();
      await until(() => r.states.at(-1)?.documentation?.status === 'resolved');
      assert.equal(r.inputs.length, 0); assert.equal(calls, 1); assert.equal(r.states.at(-1)!.status, 'unavailable');
      assert.deepEqual(r.states.at(-1)!.documentation, documentation());
      await r.request('reveal', { citationId: 'documentation-fixture' });
      assert.equal(r.reveals.length, 1); assert.equal(r.reveals[0].startLine, 150);
    }
  });

  test('pending source documentation survives provider and availability changes', async () => {
    let finish!: (value: NodeSourceDocumentation) => void, signal!: AbortSignal, calls = 0;
    const r = rig(undefined, { documentation: async (_selection, sourceSignal) => { calls++; signal = sourceSignal; return new Promise(resolve => { finish = resolve; }); } });
    r.route = { ...r.route, ready: false }; r.start(); await until(() => calls === 1);
    r.route = { ...r.route, provider: 'claude', busy: true }; r.service.refresh();
    assert.equal(signal.aborted, false); finish(documentation());
    await until(() => r.states.at(-1)?.documentation?.status === 'resolved');
    assert.equal(calls, 1); assert.equal(r.inputs.length, 0);
    r.route = { ...r.route, busy: false }; r.service.refresh();
    assert.deepEqual(r.states.at(-1)!.documentation, documentation()); assert.equal(calls, 1);
  });

  test('source documentation has a source-revision cache independent of agent routing', async () => {
    let calls = 0;
    const r = rig(undefined, { documentation: async () => { calls++; return documentation(); } });
    r.route = { ...r.route, ready: false }; r.start(); await until(() => !!r.states.at(-1)?.documentation);
    r.service.select({ ...r.selection, nodeIds: [] }); r.start(); await until(() => !!r.states.at(-1)?.documentation);
    assert.equal(calls, 1);
    r.revision = hash('new-source-graph'); r.start(); await until(() => !!r.states.at(-1)?.documentation);
    assert.equal(calls, 2);
  });

  test('source documentation remains visible when generated interpretation fails or times out', async () => {
    const failed = rig(async () => { throw new Error('Provider unavailable'); }, { documentation: async () => documentation() });
    failed.start(); await failed.done(); assert.deepEqual(failed.states.at(-1)!.documentation, documentation());
    const original = global.setTimeout;
    global.setTimeout = ((handler: (...args: any[]) => void, delay?: number, ...args: any[]) => original(handler, delay === 45000 ? 8 : delay, ...args)) as typeof setTimeout;
    try {
      const timed = rig(async (_tools, _input, signal) => { await new Promise<void>(resolve => signal.addEventListener('abort', () => resolve(), { once: true })); }, { documentation: async () => documentation() });
      timed.start(); await until(() => timed.clears === 1); assert.deepEqual(timed.states.at(-1)!.documentation, documentation());
      await timed.request('reveal', { citationId: 'documentation-fixture' }); assert.equal(timed.reveals.length, 1);
    } finally { global.setTimeout = original; }
  });

  test('source cancellation suppresses late documentation and aborts its reader without a provider', async () => {
    let finish!: (value: NodeSourceDocumentation) => void, signal!: AbortSignal;
    const r = rig(undefined, { documentation: async (_selection, sourceSignal) => { signal = sourceSignal; return new Promise(resolve => { finish = resolve; }); } });
    r.route = { ...r.route, ready: false }; r.start(); await until(() => !!signal);
    r.service.select({ ...r.selection, nodeIds: [] }); assert.equal(signal.aborted, true); finish(documentation());
    await new Promise(resolve => setTimeout(resolve, 10)); assert.equal(r.states.some(state => state.documentation), false);
  });

  test('documentation source identity is rechecked before publication and reveal', async () => {
    const changed = rig(undefined, { documentation: async () => { changed.source = 'changed declaration'; return documentation(); } });
    changed.route = { ...changed.route, ready: false }; changed.start(); await new Promise(resolve => setTimeout(resolve, 10));
    assert.equal(changed.states.some(state => state.documentation), false);
    const saved = rig(undefined, { documentation: async () => documentation() }); saved.route = { ...saved.route, ready: false };
    saved.start(); await until(() => !!saved.states.at(-1)?.documentation); saved.source = 'changed after publication';
    await saved.request('reveal', { citationId: 'documentation-fixture' }); assert.equal(saved.reveals.length, 0);
  });

  test('finite unavailable documentation evidence stays distinct from generated explanation failure', async () => {
    const r = rig(undefined, { documentation: async () => ({ status: 'unavailable', reason: 'decorated' }) });
    r.route = { ...r.route, ready: false }; r.start(); await until(() => !!r.states.at(-1)?.documentation);
    assert.deepEqual(r.states.at(-1)!.documentation, { status: 'unavailable', reason: 'decorated' }); assert.equal(r.inputs.length, 0);
  });

  test('retry can recover a failed static parser request without connecting an agent', async () => {
    let calls = 0;
    const r = rig(undefined, { documentation: async () => { if (++calls === 1) throw new Error('Static parser unavailable'); return documentation(); } });
    r.route = { ...r.route, ready: false }; r.start(); await until(() => calls === 1); await new Promise(resolve => setImmediate(resolve));
    await r.request('retry'); await until(() => !!r.states.at(-1)?.documentation);
    assert.equal(calls, 2); assert.equal(r.inputs.length, 0); assert.deepEqual(r.states.at(-1)!.documentation, documentation());
  });

  test('source selectors preserve literal callees and leave dynamic resolution to the parser', () => {
    assert.equal(nodeSourceDocumentationSymbol({ _raw_call: 'return self._score_hidden(samples, candidates, queries, mask)' }), 'self._score_hidden');
    assert.equal(nodeSourceDocumentationSymbol({ _raw_call: 'result = await helper(value)' }), 'helper');
    assert.equal(nodeSourceDocumentationSymbol({ _raw_call: '结果 = 模型.评分(输入)' }), '模型.评分');
    assert.equal(nodeSourceDocumentationSymbol({ _class: '模块.自定义层' }), '模块.自定义层');
    assert.equal(nodeSourceDocumentationSymbol({ _source_result_call: { target: 'x'.repeat(257) } }), undefined);
    assert.equal(nodeSourceDocumentationSymbol({ _raw_call: 'factory().score(value)', _source_result_call: { name: 'score', target: 'self.score' } }), 'self.score');
    assert.equal(nodeSourceDocumentationSymbol({ _raw_call: 'factory().score(value)' }), undefined);
    assert.equal(nodeSourceDocumentationSymbol({ _source_result_call: { name: 'score' } }), 'score');
    assert.equal(nodeSourceDocumentationSymbol({ _class: 'layers.CustomLayer' }), 'layers.CustomLayer');
    assert.equal(nodeSourceDocumentationSymbol({ _raw_call: '(lambda x: x)(value)' }), undefined);
    assert.equal(nodeSourceDocumentationSymbol({ _raw_call: 'items[index](value)' }), undefined);
    assert.equal(nodeSourceDocumentationSymbol({ label: 'Score candidates', operation: 'Linear' }), undefined);
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

  test('optional literal documentation never blocks generation and has its own bounded deadline', async () => {
    let signal!: AbortSignal;
    const r = rig(undefined, { documentationTimeoutMs: 10, documentation: async (_selection, value) => { signal = value; return new Promise(() => {}); } });
    r.start(); await r.done(); assert.equal(r.inputs.length, 1); assert.equal(r.states.at(-1)!.status, 'ready');
    await until(() => signal.aborted); assert.equal(r.states.at(-1)!.status, 'ready');
  });

  test('the source explanation deadline covers hung prepare, source reads and providers', async () => {
    for (const stage of ['prepare', 'read', 'provider']) {
      const extras: Partial<CanvasNodeExplanationOptions> = { timeoutMs: 5 };
      if (stage === 'prepare') extras.prepare = async () => new Promise(() => {});
      if (stage === 'read') extras.read = async () => new Promise(() => {});
      const r = rig(stage === 'provider' ? async () => new Promise(() => {}) : undefined, extras);
      r.start(); await r.done(); assert.equal(r.states.at(-1)!.status, 'unavailable'); assert.match(r.states.at(-1)!.detail!, /timed out/);
      if (stage === 'provider') await until(() => r.clears === 1);
      await r.request('retry'); await until(() => r.states.at(-1)!.status === 'unavailable' && r.states.at(-1)!.requestId !== r.states[0].requestId);
    }
  });

  test('source or trust invalidation settles queued work instead of leaving its last pending state visible', async () => {
    const r = rig(async () => new Promise(() => {})); r.start(); await until(() => r.inputs.length === 1);
    r.valid = false; r.service.refresh(); assert.equal(r.states.at(-1)!.status, 'unavailable'); assert.equal(r.signals[0].aborted, true);
    const queued = rig(undefined, { timeoutMs: 5 }); queued.start(); queued.route = { ...queued.route, busy: true };
    await queued.done(); assert.equal(queued.states.at(-1)!.status, 'unavailable'); assert.equal(queued.inputs.length, 0);
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

  test('a new selection clears the previous runtime without waiting for a provider that ignores abort', async () => {
    let finish!: () => void, calls = 0;
    const r = rig(async (tools, _input, signal) => {
      if (++calls === 1) await new Promise<void>(resolve => { finish = resolve; });
      else assert.ok(r.clears >= 1, 'the cancelled runtime must be cleared before another one starts');
      if (!signal.aborted) await call(tools, 'submit_node_explanation', payload());
    });
    r.start(); await until(() => r.inputs.length === 1);
    r.service.select({ ...r.selection, nodeIds: [] }); r.start(); await r.done();
    assert.equal(r.inputs.length, 2); assert.equal(r.signals[0].aborted, true); assert.equal(r.states.at(-1)!.status, 'ready');
    finish(); await new Promise(resolve => setImmediate(resolve)); assert.equal(r.states.at(-1)!.status, 'ready');
  });
});
