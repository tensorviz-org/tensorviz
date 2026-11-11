import * as assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mock } from 'node:test';
import { z } from 'zod';
import { CanvasModelExplanation, modelExplanationNodeContext, modelExplanationScope, type CanvasModelExplanationOptions } from '../../canvasModelExplanation';
import type { CanvasAgentRoute, AgentRunInput } from '../../canvasAgent';
import type { HostToolset } from '../../mcpHost';
import type { LocalBrowserModelExplanationRequest, LocalBrowserViewSelectionMessage } from '@tensorviz/contracts/protocol';
import { McpGraphService, type McpModel } from '@tensorviz/graph-host/mcpGraph';
import type { ProjectionPayload } from '@tensorviz/contracts/protocol';
import type { NodeSourceDocumentation, ModelExplanationState, NodeExplanationState } from '@tensorviz/contracts/nodeExplanation';
const hash = (text: string) => createHash('sha256').update(text).digest('hex');
const until = async (check: () => boolean) => { const end = Date.now() + 3000; while (!check()) { if (Date.now() > end) throw new Error('Model explanation did not settle.'); await new Promise(resolve => setTimeout(resolve, 2)); } };
const result = (nodeId: string, citationIds: string[]) => ({ nodeId, title: 'Encode audio', purpose: 'Encode audio into a compact representation.', inputs: ['Audio samples'], outputs: ['Latent features'], steps: ['Apply the encoder'], uncertainties: ['The source does not establish why this architecture was chosen.'], concepts: [{ term: 'VAE', explanation: 'A variational autoencoder learns a probabilistic latent representation.' }], citationIds });
async function call(tools: HostToolset, name: string, args: unknown) { const tool = tools.get(name)!; return tool.handler(z.object(tool.config.inputSchema).strict().parse(args), {}); }
suite('Explicit model explanation', () => {
  const services: CanvasModelExplanation[] = [];
  teardown(() => { for (const service of services.splice(0)) service.dispose(); });
  function rig(run?: (tools: HostToolset, input: AgentRunInput, signal: AbortSignal) => Promise<void>, extras: Partial<CanvasModelExplanationOptions> = {}) {
    let source = Array.from({ length: 100 }, (_, index) => index === 9 ? 'def encode(audio): return encoder(audio)' : `# ${index}`).join('\n');
    let valid = true, revision = hash('revision');
    let route: CanvasAgentRoute = { provider: 'codex', resolvedModel: 'fixture', settings: { model: 'fixture' }, ready: true, busy: false };
    const target = { workspaceFolder: '/fixture', file: 'model.py', cls: 'Model', sourceDigest: hash(source), documentVersion: 1, backendRevision: 0 };
    const states: ModelExplanationState[] = [], nodes: NodeExplanationState[] = [], inputs: AgentRunInput[] = [], toolsets: HostToolset[] = [], signals: AbortSignal[] = [];
    let clears = 0, changed = 0;
    const service = new CanvasModelExplanation({ route: settings => settings === undefined ? route : { ...route, settings: { ...settings }, resolvedModel: settings.model ?? route.resolvedModel }, current: (_request, rev) => valid && (!rev || rev === revision), reference: () => ({ modelId: 'model', revision }),
      prepare: async (_request, nodeId) => ({ context: { nodeId, callable: 'AudioVAE', edges: [{ sourceNodeId: nodeId, targetNodeId: 'decoder' }] }, source: { file: 'model.py', startLine: 10 } }),
      read: async (_target, file) => { if (file !== 'model.py') throw new Error('Not graph-recorded'); return { file, source, digest: hash(source) }; },
      tools: () => new Map([['request_project_command', { config: { inputSchema: {} }, handler: async () => { throw new Error('Execution leaked'); } }],
        ['query_source_context', { config: { inputSchema: { modelId: z.string(), revision: z.string(), file: z.string(), symbol: z.string() } }, handler: async () => ({ content: [], structuredContent: { items: [{ excerpt: 'def encode(audio)', anchor: { file: 'model.py', digest: hash(source), startLine: 10, endLine: 10 } }] } }) }]]),
      runtime: load => ({ run: async (input, _emit, signal) => { const tools = load('isolated'); inputs.push(input); toolsets.push(tools); signals.push(signal);
        if (run) await run(tools, input, signal); else { const context = JSON.parse(input.context); await call(tools, 'submit_model_explanations', { nodes: context.nodes.map((node: any) => result(node.nodeId, [node.evidenceId])) }); } }, clear: () => { clears++; } }),
      publish: (_view, _target, state) => states.push(state), publishNode: (_view, _target, state) => nodes.push(state), changed: () => { changed++; }, ...extras });
    services.push(service);
    const request: LocalBrowserModelExplanationRequest = { type: 'modelExplanationRequest', protocolVersion: 1, workspaceSessionId: 's', messageId: 'm', viewId: 'view', target, action: 'start', requestId: 'explain-one', nodeIds: ['encoder', 'decoder'] };
    return { service, request, states, nodes, inputs, toolsets, signals, start: (changes: Partial<LocalBrowserModelExplanationRequest> = {}) => service.request({ ...request, ...changes }),
      done: () => until(() => !!states.length && states.at(-1)!.status !== 'loading'), get clears() { return clears; }, get changed() { return changed; }, get route() { return route; }, set route(value: CanvasAgentRoute) { route = value; },
      set valid(value: boolean) { valid = value; }, set revision(value: string) { revision = value; }, set source(value: string) { source = value; } };
  }
  test('one explicit request shares source across nodes, preserves topology context and submits bounded typed results', async () => {
    const r = rig(); assert.equal(r.inputs.length, 0); r.start(); await r.done();
    assert.equal(r.inputs.length, 1); assert.equal(r.inputs[0].intent, 'model_explanation'); assert.equal(r.inputs[0].messages.length, 1);
    const context = JSON.parse(r.inputs[0].context); assert.equal(context.sourceEvidence.length, 1); assert.equal(context.nodes.length, 2); assert.equal(context.nodes[0].facts.edges[0].targetNodeId, 'decoder');
    assert.deepEqual([...r.toolsets[0].keys()].sort(), ['query_source_context', 'read_model_source', 'submit_model_explanations']);
    assert.equal(r.states.at(-1)!.status, 'ready'); assert.equal(r.states.at(-1)!.completed, 2); assert.equal(r.nodes.length, 2); assert.equal(r.nodes[0].explanation!.concepts![0].term, 'VAE'); assert.equal(r.clears, 1);
  });
  test('per-pass model and effort are immutable, published and part of the explanation cache', async () => {
    const r = rig(); const settings = { model: 'quick-model', effort: 'low' };
    r.start({ settings }); settings.model = 'changed-by-caller'; settings.effort = 'high'; await r.done();
    assert.deepEqual(r.inputs[0].settings, { model: 'quick-model', effort: 'low' });
    assert.deepEqual(r.states.at(-1)!.execution, { provider: 'codex', model: 'quick-model', effort: 'low' });
    assert.deepEqual(r.route.settings, { model: 'fixture' });
    r.start({ requestId: 'same-choice', settings: { model: 'quick-model', effort: 'low' } }); await r.done();
    assert.equal(r.inputs.length, 1, 'the same explanation choice reuses its own cached nodes');
    r.start({ requestId: 'different-choice', settings: { model: 'quick-model', effort: 'high' } }); await r.done();
    assert.equal(r.inputs.length, 2); assert.equal(r.inputs[1].settings?.effort, 'high');
  });
  test('an explicit pass survives chat model changes and still stops if the provider changes', async () => {
    const r = rig(async (_tools, _input, signal) => new Promise(resolve => signal.addEventListener('abort', () => resolve(), { once: true })));
    r.start({ settings: { model: 'quick-model', effort: 'low' } }); await until(() => r.signals.length === 1);
    r.route = { ...r.route, settings: { model: 'new-chat-model', effort: 'high' }, resolvedModel: 'new-chat-model' }; r.service.refresh();
    assert.equal(r.signals[0].aborted, false); assert.equal(r.service.busy, true);
    r.route = { ...r.route, provider: 'claude' }; r.service.refresh(); await r.done();
    assert.equal(r.signals[0].aborted, true); assert.equal(r.states.at(-1)!.status, 'unavailable');
  });
  test('cold preparation receives the per-pass choice and publishes its resolved defaults', async () => {
    const choices: unknown[] = [];
    const r = rig(undefined, { route: settings => ({ ...r.route, resolvedModel: settings?.model ?? r.route.resolvedModel,
      settings: { ...settings, effort: settings?.effort ?? 'low' } }), prepareRoute: async (_signal, settings) => {
      choices.push(settings); r.route = { ...r.route, ready: true };
      return { ...r.route, resolvedModel: settings?.model, settings: { ...settings, effort: settings?.effort ?? 'low' } };
    } });
    r.route = { ...r.route, ready: false };
    r.start({ settings: { model: 'quick-model' } }); await r.done();
    assert.deepEqual(choices, [{ model: 'quick-model' }]);
    assert.deepEqual(r.states.at(-1)!.execution, { provider: 'codex', model: 'quick-model', effort: 'low' });
    assert.equal(r.inputs[0].settings?.model, 'quick-model'); assert.equal(r.states.at(-1)!.status, 'ready');
  });
  test('live MCP inspection fields retain callable, source, parameters, actual ports and named graph neighbors', () => {
    const model: McpModel = { current: true, target: { workspaceFolder: '/fixture', file: 'model.py', cls: 'Model', sourceDigest: hash('source'), documentVersion: 1, backendRevision: 0 },
      projection: { traced: false, spanmap: { encoder: { file: 'model.py', start: [10, 0], end: [10, 30] } }, document: { semantic: {
        nodes: [{ node_id: 'input', kind: 'input', op: 'Input', params: {} },
          { node_id: 'encoder', kind: 'layer', op: 'AudioVAE', params: { sample_rate: 24000, _class: 'audio.AudioVAE', _raw_call: 'self.audio_vae(audio)', _shape_provenance: 'static_only' }, in_ports: ['audio'], out_ports: ['latents'] },
          { node_id: 'decoder', kind: 'layer', op: 'Decoder', params: {} }],
        edges: [{ edge_id: 'audio', source_node_id: 'input', target_node_id: 'encoder', source_port: 'samples', target_port: 'audio' },
          { edge_id: 'latents', source_node_id: 'encoder', target_node_id: 'decoder', source_port: 'latents', target_port: 'features' }],
      } } } as unknown as ProjectionPayload };
    const graph = new McpGraphService('fixture-host', () => model), ref = graph.reference(model);
    const prepared = modelExplanationNodeContext(model, graph.inspectNodes(ref, ['encoder']), 'encoder');
    assert.deepEqual(prepared.source, { file: 'model.py', startLine: 10 });
    assert.equal(prepared.context.node.kind, 'layer'); assert.equal(prepared.context.node.operation, 'AudioVAE');
    assert.equal(prepared.context.node.callable, 'audio.AudioVAE'); assert.equal(prepared.context.node.call, 'self.audio_vae(audio)');
    assert.deepEqual(prepared.context.node.parameters, { sample_rate: 24000 }); assert.deepEqual(prepared.context.node.inputPorts, ['audio']);
    assert.deepEqual(prepared.context.node.outputPorts, ['latents']); assert.equal((prepared.context.node.source as any).file, 'model.py');
    assert.deepEqual(prepared.context.edges.map(edge => [edge.sourceNodeId, edge.targetNodeId, edge.sourcePort, edge.targetPort]), [['input', 'encoder', 'samples', 'audio'], ['encoder', 'decoder', 'latents', 'features']]);
    assert.deepEqual(prepared.context.neighbors.map(node => [node.nodeId, node.operation]), [['input', 'Input'], ['decoder', 'Decoder']]); assert.equal(prepared.context.connectionsLimited, false);
  });


  test('view results explain only the supplied snapshot, validate real edges and retain source-linked callouts', async () => {
    const scope = { parent: { file: 'model.py', cls: 'Model' }, nodes: [{ nodeId: 'encoder', kind: 'layer', operation: 'Encoder' }, { nodeId: 'decoder', kind: 'layer', operation: 'Decoder' }], connections: [{ sourceNodeId: 'encoder', targetNodeId: 'decoder' }] };
    const r = rig(async (tools, input) => {
      const context = JSON.parse(input.context);
      assert.deepEqual(context.view, scope); assert.deepEqual(context.requestedNodeIds, ['encoder', 'decoder']); assert.deepEqual(context.visibleNodeIds, ['encoder', 'decoder']);
      const evidenceId = context.nodes[0].evidenceId;
      const nodes = context.nodes.map((node: any) => result(node.nodeId, [node.evidenceId]));
      const view = { summary: 'The visible components turn audio samples into learned features and reconstruct audio.',
        connections: [{ sourceNodeId: 'encoder', targetNodeId: 'decoder', label: 'Learned audio features', detail: 'Features provide the representation consumed by the decoder.', citationIds: [evidenceId] }],
        callouts: [{ title: 'A compact audio representation', body: 'The encoder represents the audio with learned features that the decoder consumes.', nodeIds: ['encoder', 'decoder'], citationIds: [evidenceId] }] };
      for (const invalid of [
        { ...view, connections: [{ ...view.connections[0], sourceNodeId: 'decoder', targetNodeId: 'encoder' }] },
        { ...view, callouts: [{ ...view.callouts[0], nodeIds: ['hidden-child'] }] },
        { ...view, connections: [{ ...view.connections[0], citationIds: ['invented'] }] },
      ]) assert.equal((await call(tools, 'submit_model_explanations', { nodes, view: invalid })).isError, true);
      assert.equal((await call(tools, 'submit_model_explanations', { nodes })).isError, true);
      assert.equal((await call(tools, 'submit_model_explanations', { nodes, view })).isError, undefined);
    }, { scope: () => scope });
    r.start(); await r.done();
    const view = r.states.at(-1)!.view!; assert.equal(r.states.at(-1)!.status, 'ready');
    assert.deepEqual(view.nodeIds, ['encoder', 'decoder']); assert.equal(view.connections[0].label, 'Learned audio features');
    assert.equal(view.callouts[0].citations[0].file, 'model.py'); assert.equal(view.callouts[0].citations[0].sourceDigest, r.request.target.sourceDigest);
    assert.match(r.inputs[0].messages[0].text, /visible view/);
    r.start({ requestId: 'same-view' }); await until(() => r.states.at(-1)!.requestId === 'same-view' && r.states.at(-1)!.status === 'ready');
    assert.equal(r.inputs.length, 1); assert.deepEqual(r.states.at(-1)!.view, view);
  });

  test('a changed visible scope reuses existing node explanations and generates a new view story', async () => {
    const r = rig(async (tools, input) => {
      const context = JSON.parse(input.context);
      await call(tools, 'submit_model_explanations', { nodes: context.nodes.map((node: any) => result(node.nodeId, [node.evidenceId])),
        view: { summary: `Explain ${context.visibleNodeIds.length} visible nodes.`, connections: [], callouts: [] } });
    }, { scope: request => ({ parent: { file: 'model.py', cls: 'Model' }, nodes: request.nodeIds.map(nodeId => ({ nodeId, kind: 'layer', operation: nodeId })), connections: [] }) });
    r.start(); await r.done();
    r.start({ requestId: 'smaller-view', nodeIds: ['decoder'] });
    await until(() => r.states.at(-1)!.requestId === 'smaller-view' && r.states.at(-1)!.status === 'ready');
    assert.equal(r.inputs.length, 2); assert.deepEqual(r.states.at(-1)!.view!.nodeIds, ['decoder']);
    assert.equal(r.states.at(-1)!.completed, 1); assert.equal(JSON.parse(r.inputs[1].context).nodes.length, 1);
  });

  test('view summaries merge across bounded batches and added citation links never double-count ready nodes', async () => {
    const ids = Array.from({ length: 8 }, (_, index) => `n${index}`);
    const r = rig(async (tools, input) => {
      const context = JSON.parse(input.context), first = context.nodes[0];
      assert.deepEqual(context.batchNodeIds, context.nodes.map((node: any) => node.nodeId));
      assert.deepEqual(context.requestedNodeIds, context.batchNodeIds); assert.deepEqual(context.visibleNodeIds, ids);
      const schema = z.object(tools.get('submit_model_explanations')!.config.inputSchema);
      const otherBatch = ids.find(id => !context.batchNodeIds.includes(id))!;
      assert.equal(schema.safeParse({ nodes: [result(otherBatch, [first.evidenceId])] }).success, false);
      const nodes = context.nodes.map((node: any) => result(node.nodeId, [node.evidenceId]));
      if (!context.viewProgress.summary) {
        assert.equal((await call(tools, 'submit_model_explanations', { nodes, view: { connections: [], callouts: [] } })).isError, true);
        assert.equal((await call(tools, 'submit_model_explanations', { nodes, view: { summary: 'A visible story.', connections: [], callouts: [{ title: 'Unrelated future batch', body: 'This does not touch any currently requested batch node.', nodeIds: ['n7'], citationIds: [first.evidenceId] }] } })).isError, true);
      } else {
        assert.equal(context.viewProgress.summary, 'A single story for this visible view.');
        assert.ok(context.viewProgress.explainedNodeIds.length >= 3);
      }
      await call(tools, 'submit_model_explanations', { nodes,
        view: { ...(context.viewProgress.summary ? {} : { summary: 'A single story for this visible view.' }), connections: [], callouts: [{ title: first.nodeId, body: 'A source-backed relationship in the current view.', nodeIds: ['n0', first.nodeId].filter((value, index, all) => all.indexOf(value) === index), citationIds: [first.evidenceId] }] } });
    }, { scope: () => ({ parent: { file: 'model.py', cls: 'Model' }, nodes: ids.map(nodeId => ({ nodeId, kind: 'layer', operation: nodeId })), connections: [] }) });
    r.start({ nodeIds: ids }); await r.done();
    assert.equal(r.states.at(-1)!.status, 'ready'); assert.equal(r.states.at(-1)!.completed, 8);
    assert.deepEqual(r.inputs.map(input => JSON.parse(input.context).nodes.length), [3, 3, 2]);
    assert.equal(r.states.at(-1)!.view!.summary, 'A single story for this visible view.');
    assert.deepEqual(r.states.at(-1)!.view!.nodeIds, ids); assert.equal(r.states.at(-1)!.view!.callouts.length, 3);
    assert.ok(r.states.every(state => state.completed <= state.total));
  });

  test('boundary inputs and outputs use explicit neighboring source context without inventing a call span', () => {
    const model: McpModel = { current: true, target: { workspaceFolder: '/fixture', file: 'model.py', cls: 'Model', sourceDigest: hash('source'), documentVersion: 1, backendRevision: 0 },
      projection: { traced: false, spanmap: { decoder: { file: 'model.py', start: [20, 0], end: [20, 30] } }, document: { semantic: {
        nodes: [{ node_id: 'condition', kind: 'input', op: 'Input', params: {} }, { node_id: 'decoder', kind: 'layer', op: 'Decoder', params: {} }, { node_id: 'audio', kind: 'output', op: 'Output', params: {} }, { node_id: 'hidden', kind: 'layer', op: 'Hidden', params: {} }],
        edges: [{ edge_id: 'condition-edge', source_node_id: 'condition', target_node_id: 'decoder', source_port: 'index', target_port: 'condition' }, { edge_id: 'audio-edge', source_node_id: 'decoder', target_node_id: 'audio', source_port: 'samples', target_port: 'result' }, { edge_id: 'hidden-edge', source_node_id: 'decoder', target_node_id: 'hidden' }],
      } } } as unknown as ProjectionPayload };
    const graph = new McpGraphService('fixture-host', () => model), ref = graph.reference(model);
    for (const nodeId of ['condition', 'audio']) {
      const prepared = modelExplanationNodeContext(model, graph.inspectNodes(ref, [nodeId]), nodeId);
      assert.deepEqual(prepared.source, { file: 'model.py', startLine: 20 });
      assert.equal(prepared.context.node.source, null); assert.equal(prepared.context.node.sourceMapping, 'neighbor_or_entry_context');
    }
    assert.throws(() => modelExplanationNodeContext(model, graph.inspectNodes(ref, ['hidden']), 'hidden'), /No source mapping/);
    const scope = modelExplanationScope(model, ['condition', 'decoder', 'audio']);
    assert.deepEqual(scope.nodes.map(node => node.nodeId), ['condition', 'decoder', 'audio']); assert.equal(scope.connections.length, 2);
    assert.ok(scope.connections.every(edge => edge.sourceNodeId !== 'hidden' && edge.targetNodeId !== 'hidden'));
  });

  test('resolved signatures and authored docs arrive before inference once per implementation and can be cited', async () => {
    const source = Array.from({ length: 100 }, (_, index) => index === 9 ? 'def encode(audio): return encoder(audio)' : `# ${index}`).join('\n');
    const documentation: NodeSourceDocumentation = { status: 'resolved', qualifiedName: 'AudioVAE.encode', kind: 'method', signature: 'def encode(self, audio: Tensor) -> Tensor',
      docstring: 'Encode audio samples into latent features.', truncated: false, citation: { id: 'documentation-host', file: 'model.py', sourceDigest: hash(source), startLine: 10, endLine: 10 } };
    const r = rig(async (tools, input) => {
      const context = JSON.parse(input.context); assert.equal(context.sourceDocumentation.length, 1);
      assert.equal(context.sourceDocumentation[0].signature, documentation.signature); assert.equal(context.sourceDocumentation[0].docstring, documentation.docstring);
      assert.ok(context.nodes.every((node: any) => node.documentationEvidenceId === context.sourceDocumentation[0].citation.id));
      await call(tools, 'submit_model_explanations', { nodes: context.nodes.map((node: any) => result(node.nodeId, [node.documentationEvidenceId])) });
    }, { documentation: async () => documentation });
    r.start(); await r.done(); assert.equal(r.states.at(-1)!.status, 'ready');
    assert.equal(r.nodes[0].documentation?.status, 'resolved'); assert.equal(r.nodes[0].explanation!.citations[0].startLine, 10);
    const unavailable = rig(undefined, { documentation: async () => { throw new Error('Parser unavailable'); } }); unavailable.start(); await unavailable.done();
    assert.equal(unavailable.states.at(-1)!.status, 'ready'); assert.equal(JSON.parse(unavailable.inputs[0].context).sourceDocumentation.length, 0);
  });

  test('larger requests use batches of six and leave already completed nodes usable', async () => {
    const r = rig(); r.start({ nodeIds: Array.from({ length: 14 }, (_, i) => `node-${i}`) }); await r.done();
    assert.deepEqual(r.inputs.map(input => JSON.parse(input.context).nodes.length), [6, 6, 2]); assert.equal(r.nodes.length, 14); assert.equal(r.states.at(-1)!.completed, 14);
  });
  test('moving past an empty batch publishes its actual next range without claiming completed nodes', async () => {
    let calls = 0;
    const r = rig(async (tools, input) => {
      if (++calls === 1) return;
      const context = JSON.parse(input.context);
      await call(tools, 'submit_model_explanations', { nodes: context.nodes.map((node: any) => result(node.nodeId, [node.evidenceId])) });
    });
    r.start({ nodeIds: Array.from({ length: 8 }, (_, index) => `node-${index}`) }); await r.done();
    const advanced = r.states.find(state => state.detail === 'Reading components 7–8 of 8…');
    assert.equal(advanced?.status, 'loading'); assert.equal(advanced?.completed, 0); assert.equal(calls, 2);
    assert.equal(r.states.at(-1)!.status, 'unavailable'); assert.equal(r.states.at(-1)!.completed, 2);
  });
  test('cached same-revision results avoid more model calls and can be reused by node selection', async () => {
    const r = rig(); r.start(); await r.done(); r.start({ requestId: 'second' }); await until(() => r.states.at(-1)!.requestId === 'second' && r.states.at(-1)!.status === 'ready');
    assert.equal(r.inputs.length, 1);
    const selection = { target: r.request.target, nodeIds: ['encoder'] } as LocalBrowserViewSelectionMessage;
    assert.equal(r.service.cached(selection, { modelId: 'model', revision: hash('revision') }, r.route)?.title, 'Encode audio');
    assert.equal(r.service.cached(selection, { modelId: 'model', revision: hash('new') }, r.route), undefined);
    assert.equal(r.service.cached(selection, { modelId: 'model', revision: hash('revision') }, { ...r.route, provider: 'claude' }), undefined);
  });
  test('same-view readiness refusals return an unavailable receipt while foreign or stale targets stay private', async () => {
    const r = rig(undefined, { refusal: request => request.viewId === 'view' && request.target.cls === 'Model'
      ? { ref: { modelId: 'model', revision: hash('revision') }, detail: 'Trust this workspace in VS Code, then explain again.' } : undefined });
    r.valid = false; r.start(); assert.equal(r.states.at(-1)!.status, 'unavailable'); assert.match(r.states.at(-1)!.detail!, /Trust this workspace/); assert.equal(r.inputs.length, 0);
    const count = r.states.length; r.start({ requestId: 'foreign', viewId: 'foreign' }); r.start({ requestId: 'stale', target: { ...r.request.target, cls: 'OldModel' } }); assert.equal(r.states.length, count);
  });

  test('unready and foreground-busy agents return actionable state without running', async () => {
    for (const change of [{ ready: false }, { busy: true }]) { const r = rig(); r.route = { ...r.route, ...change }; r.start(); await r.done(); assert.equal(r.states.at(-1)!.status, 'unavailable'); assert.ok(r.states.at(-1)!.detail); assert.equal(r.inputs.length, 0); }
  });
  test('an explicit cold start checks the configured provider and continues without opening chat or another click', async () => {
    let checks = 0;
    const r = rig(undefined, { prepareRoute: async () => {
      checks++; await new Promise(resolve => setImmediate(resolve));
      r.route = { ...r.route, ready: true, resolvedModel: 'discovered-default', settings: { effort: 'medium' } }; return r.route;
    } });
    r.route = { ...r.route, ready: false, resolvedModel: undefined, settings: {} };
    r.start(); assert.equal(r.states.at(-1)!.status, 'loading'); await r.done();
    assert.equal(checks, 1); assert.equal(r.states.at(-1)!.status, 'ready'); assert.equal(r.nodes.length, 2);
    assert.equal(r.inputs.length, 1); assert.deepEqual(r.inputs[0].settings, { effort: 'medium' });
    assert.deepEqual(r.states.filter(state => state.status === 'loading' && state.completed === 0).map(state => state.detail), ['Connecting to agent…', 'Reading model components…']);
    assert.equal(r.states.at(-1)!.detail, undefined, 'ready does not retain a stale activity label');
  });
  test('unavailable provider checks preserve the specific account reason and never start inference', async () => {
    const r = rig(undefined, { prepareRoute: async () => { r.route = { ...r.route, ready: false, detail: 'Sign in to Codex before explaining source.' }; return r.route; } });
    r.route = { ...r.route, ready: false }; r.start(); await r.done();
    assert.equal(r.states.at(-1)!.status, 'unavailable'); assert.equal(r.states.at(-1)!.detail, 'Sign in to Codex before explaining source.');
    assert.equal(r.inputs.length, 0); assert.equal(r.service.busy, false);
  });
  test('cancel and stale provider checks settle without late generation', async () => {
    for (const mode of ['cancel', 'stale', 'busy']) {
      let finish!: (route: CanvasAgentRoute) => void, signal!: AbortSignal;
      const r = rig(undefined, { prepareRoute: received => { signal = received; return new Promise(resolve => { finish = resolve; }); } });
      r.route = { ...r.route, ready: false }; r.start(); await until(() => !!finish);
      if (mode === 'cancel') r.start({ action: 'cancel', nodeIds: [] });
      else if (mode === 'stale') r.valid = false;
      else r.route = { ...r.route, busy: true };
      finish(r.route); await r.done();
      assert.equal(r.states.at(-1)!.status, mode === 'cancel' ? 'cancelled' : 'unavailable');
      assert.equal(signal.aborted, true); assert.equal(r.service.busy, false); assert.equal(r.inputs.length, 0);
    }
  });
  test('configured-provider discovery has a bounded deadline even when its inspector ignores cancellation', async () => {
    const original = global.setTimeout;
    const timer = mock.method(global, 'setTimeout', ((fn: (...args: any[]) => void, ms: number, ...args: any[]) => original(fn, ms === 45_000 ? 5 : ms, ...args)) as typeof setTimeout);
    try {
      const r = rig(undefined, { prepareRoute: () => new Promise(() => {}) }); r.route = { ...r.route, ready: false };
      r.start(); await r.done(); assert.equal(r.states.at(-1)!.status, 'unavailable'); assert.match(r.states.at(-1)!.detail!, /did not finish connecting/);
      assert.equal(r.inputs.length, 0); assert.equal(r.service.busy, false);
    } finally { timer.mock.restore(); }
  });
  test('cancel requires the same view, target and request; aborts isolated runtime', async () => {
    const r = rig(async (_tools, _input, signal) => new Promise(resolve => signal.addEventListener('abort', () => resolve(), { once: true })));
    r.start(); await until(() => r.signals.length === 1);
    r.start({ action: 'cancel', nodeIds: [], requestId: 'other' }); assert.equal(r.signals[0].aborted, false);
    r.start({ action: 'cancel', nodeIds: [], viewId: 'other' }); assert.equal(r.signals[0].aborted, false);
    r.start({ action: 'cancel', nodeIds: [] }); await r.done(); assert.equal(r.signals[0].aborted, true); assert.equal(r.states.at(-1)!.status, 'cancelled'); await until(() => r.clears === 1);
  });
  test('foreground work, source invalidation and view disconnect cancel pending explanations', async () => {
    for (const mode of ['busy', 'source', 'disconnect']) {
      const r = rig(async (_tools, _input, signal) => new Promise(resolve => signal.addEventListener('abort', () => resolve(), { once: true }))); r.start(); await until(() => r.signals.length === 1);
      if (mode === 'busy') { r.route = { ...r.route, busy: true }; r.service.refresh(); } else if (mode === 'source') { r.valid = false; r.service.invalidate(); } else r.service.disconnect('view');
      await r.done(); assert.equal(r.signals[0].aborted, true); assert.equal(r.nodes.length, 0);
    }
  });
  test('invalid IDs, invented evidence and duplicate results are rejected while a valid partial result is retained', async () => {
    const r = rig(async (tools, input) => {
      const node = JSON.parse(input.context).nodes[0];
      const submit = tools.get('submit_model_explanations')!;
      for (const [entries, reason] of [
        [[result('invented', [node.evidenceId])], /outside this batch/],
        [[result(node.nodeId, ['invented'])], /evidence IDs not supplied/],
        [[result(node.nodeId, [node.evidenceId, node.evidenceId])], /repeats a citation/],
        [[result(node.nodeId, [node.evidenceId]), result(node.nodeId, [node.evidenceId])], /submitted twice/],
      ] as const) {
        // Exercise host authority as well as the narrower provider-facing schema.
        const rejected = await submit.handler({ nodes: entries }, {});
        assert.equal(rejected.isError, true); assert.match(JSON.stringify(rejected.structuredContent), reason);
      }
      assert.equal((await call(tools, 'submit_model_explanations', { nodes: [result(node.nodeId, [node.evidenceId])] })).isError, undefined);
    });
    r.start(); await r.done(); assert.equal(r.states.at(-1)!.status, 'unavailable'); assert.equal(r.states.at(-1)!.completed, 1); assert.equal(r.nodes.length, 1);
  });
  test('source changes after context assembly refuse results and never mark an incomplete run ready', async () => {
    const r = rig(async (tools, input) => { const node = JSON.parse(input.context).nodes[0]; r.source = 'changed'; assert.equal((await call(tools, 'submit_model_explanations', { nodes: [result(node.nodeId, [node.evidenceId])] })).isError, true); });
    r.start(); await r.done(); assert.equal(r.nodes.length, 0); assert.equal(r.states.at(-1)!.status, 'unavailable');
  });
  test('provider failure and source mapping gaps preserve successful earlier batches', async () => {
    let calls = 0; const r = rig(async (tools, input) => { if (++calls === 2) throw new Error('provider failed'); const context = JSON.parse(input.context); await call(tools, 'submit_model_explanations', { nodes: context.nodes.map((node: any) => result(node.nodeId, [node.evidenceId])) }); });
    r.start({ nodeIds: Array.from({ length: 8 }, (_, i) => `n${i}`) }); await r.done(); assert.equal(r.states.at(-1)!.status, 'unavailable'); assert.equal(r.states.at(-1)!.completed, 6); assert.equal(r.nodes.length, 6); assert.match(r.states.at(-1)!.detail!, /agent could not finish/);
    const gap = rig(undefined, { prepare: async () => { throw new Error('No source mapping'); } }); gap.start(); await gap.done(); assert.equal(gap.inputs.length, 0); assert.equal(gap.states.at(-1)!.status, 'unavailable'); assert.match(gap.states.at(-1)!.detail!, /source mapping/);
  });
  test('oversized context and unauthorized source reads cannot expand the context budget', async () => {
    const tooLarge = rig(undefined, { prepare: async () => ({ context: { large: 'x'.repeat(5000) }, source: { file: 'model.py', startLine: 10 } }) }); tooLarge.start(); await tooLarge.done(); assert.equal(tooLarge.inputs.length, 0); assert.match(tooLarge.states.at(-1)!.detail!, /context exceeded/);
    const r = rig(async (tools, input) => { assert.equal((await call(tools, 'read_model_source', { file: 'secret.py' })).isError, true); const context = JSON.parse(input.context); await call(tools, 'submit_model_explanations', { nodes: context.nodes.map((node: any) => result(node.nodeId, [node.evidenceId])) }); }); r.start(); await r.done(); assert.equal(r.states.at(-1)!.status, 'ready');
  });
  test('over-budget evidence cannot be cited by guessing its host ID', async () => {
    const source = Array.from({ length: 100 }, (_, index) => index === 9 ? 'def encode(audio): return encoder(audio)' : `# ${index}`).join('\n');
    const r = rig(async (tools, input) => {
      const response = await call(tools, 'query_source_context', { modelId: 'model', revision: hash('revision'), file: 'model.py', symbol: 'encode' });
      assert.equal(response.isError, true);
      const node = JSON.parse(input.context).nodes[0];
      assert.equal((await call(tools, 'submit_model_explanations', { nodes: [result(node.nodeId, ['evidence-2'])] })).isError, true);
      assert.equal((await call(tools, 'submit_model_explanations', { nodes: [result(node.nodeId, [node.evidenceId])] })).isError, undefined);
    }, { tools: () => new Map([['query_source_context', { config: { inputSchema: { modelId: z.string(), revision: z.string(), file: z.string(), symbol: z.string() } },
      handler: async () => ({ content: [], structuredContent: { padding: 'x'.repeat(50000), items: [{ excerpt: 'def encode(audio)', anchor: { file: 'model.py', digest: hash(source), startLine: 10, endLine: 10 } }] } }) }]]) });
    r.start(); await r.done(); assert.equal(r.nodes.length, 1);
  });

  test('stale state discovered at provider return cannot leave a loading run behind', async () => {
    const r = rig(async () => { r.valid = false; }); r.start(); await r.done();
    assert.equal(r.states.at(-1)!.status, 'unavailable'); assert.equal(r.nodes.length, 0); assert.equal(r.service.busy, false);
  });

  test('the full batch deadline bounds hung preparation, source reads and an uncooperative provider', async () => {
    for (const stage of ['prepare', 'documentation', 'read', 'provider']) {
      const extras: Partial<CanvasModelExplanationOptions> = { timeoutMs: 5 };
      if (stage === 'prepare') extras.prepare = async () => new Promise(() => {});
      if (stage === 'documentation') extras.documentation = async () => new Promise(() => {});
      if (stage === 'read') extras.read = async () => new Promise(() => {});
      const r = rig(stage === 'provider' ? async () => new Promise(() => {}) : undefined, extras);
      r.start(); await r.done(); assert.match(r.states.at(-1)!.detail!, /timed out/); assert.equal(r.service.busy, false);
      if (stage === 'provider') await until(() => r.clears === 1);
      r.start({ requestId: 'retry' }); await until(() => r.states.at(-1)!.requestId === 'retry' && r.states.at(-1)!.status === 'unavailable');
    }
  });

  test('missing and rejected structured submissions have distinct recovery explanations', async () => {
    const missing = rig(async () => {}); missing.start(); await missing.done(); assert.match(missing.states.at(-1)!.detail!, /no verified explanation/);
    const rejected = rig(async (tools, input) => { const node = JSON.parse(input.context).nodes[0]; await call(tools, 'submit_model_explanations', { nodes: [result(node.nodeId, ['invented'])] }); });
    rejected.start(); await rejected.done(); assert.match(rejected.states.at(-1)!.detail!, /failed source validation/); assert.ok(rejected.states.at(-1)!.detail!.length <= 320);
  });

  test('duplicate delivery is idempotent and stale requests never start', async () => {
    const r = rig(); r.start(); r.start(); await r.done(); assert.equal(r.inputs.length, 1);
    const stale = rig(); stale.valid = false; stale.start(); assert.equal(stale.states.length, 0);
  });
});
