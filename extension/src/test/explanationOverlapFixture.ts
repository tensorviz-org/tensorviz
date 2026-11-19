/** Offline actual service -> LocalBrowserGraph publication/selection-adoption trace.
 * Only editor/model/source/provider boundaries are fixtures; no receipt is invented.
 */
import * as assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { z } from 'zod';
import { LocalBrowserGraph } from '../localBrowserGraph';
import type { CanvasAgent, CanvasAgentRoute } from '../canvasAgent';
import type { CanvasModelExplanation } from '../canvasModelExplanation';
import type { CanvasNodeExplanation } from '../canvasNodeExplanation';
import type { LocalBrowserSourceEdits } from '../localBrowserSourceEdits';
import type { LocalBrowserHostPublish } from '@tensorviz/graph-host/localBrowserHostTransport';
import type { LocalBrowserModelExplanationRequest, LocalBrowserViewSelectionMessage } from '@tensorviz/contracts/protocol';
import type { McpModel } from '@tensorviz/graph-host/mcpGraph';

export type OverlapOrder = 'older-first' | 'newer-first' | 'cancel-newer';
const until = async (check: () => boolean) => {
  const end = Date.now() + 3000;
  while (!check()) { if (Date.now() > end) throw new Error('Explanation overlap fixture did not settle.'); await new Promise(resolve => setTimeout(resolve, 2)); }
};

export async function runExplanationOverlapFixture(requests: LocalBrowserModelExplanationRequest[], order: OverlapOrder, revision: string) {
  assert.equal(requests.length, 2);
  const [older, newer] = requests, target = older.target;
  const source = '# older evidence\n# shared model source\n# newer evidence\n';
  const ref = { modelId: 'overlap-model', revision };
  const messages: LocalBrowserHostPublish[] = [], entered: string[] = [], cleared: string[] = [];
  const gates: (() => void)[] = [], signals: AbortSignal[] = [];
  const model: McpModel = { target, current: true, projection: { graphId: 'overlap-graph', revision: 1, cls: target.cls,
    file: target.file, sourceDigest: target.sourceDigest, greyed: false, viewRestored: false, traced: false,
    spanmap: { a: { file: target.file, start: [2, 0], end: [2, 1] } },
    document: { semantic: { nodes: [{ node_id: 'a', kind: 'layer', op: 'Encoder', params: {} }], edges: [] } },
  } as McpModel['projection'] };
  const route = (settings: CanvasAgentRoute['settings'] = {}): CanvasAgentRoute => ({ provider: 'codex', ready: true, busy: false,
    settings, resolvedModel: settings.model ?? 'fixture-model' });
  const graph = new LocalBrowserGraph({
    channel: { appendLine() {} }, backend: { origin: () => undefined, launchToken: () => undefined },
    launchOrigin: () => undefined, ensureBackend: async () => true, remoteName: () => undefined, openExternal: async () => false,
    agent: { run: async () => { throw new Error('Chat must not run.'); }, clear() {}, reference: () => ref,
      selection: () => ({ nodes: [{ nodeId: 'a', kind: 'layer', operation: 'Encoder' }], edges: [], edgeCount: 0 }),
      explanation: { tools: () => new Map(), runtime: load => {
        let requestId = '';
        return { clear: () => { cleared.push(requestId); }, run: async (input, _emit, signal) => {
          const index = entered.length;
          requestId = requests[index].requestId; entered.push(requestId); signals.push(signal);
          await new Promise<void>(resolve => { gates.push(resolve); });
          const tools = load(input.conversationId!);
          const read = tools.get('read_model_source')!;
          const response = await read.handler({ file: target.file, startLine: index === 0 ? 1 : 3, maxLines: 1 }, {});
          if (signal.aborted) return;
          assert.equal(response.isError, undefined);
          const citation = (response.structuredContent as { citation: { id: string } }).citation;
          const tool = tools.get('submit_model_explanations')!;
          const submitted = await tool.handler(z.object(tool.config.inputSchema).strict().parse({
            nodes: [{ nodeId: 'a', title: requestId, purpose: requestId, inputs: [], outputs: [], steps: [], uncertainties: [], citationIds: [citation.id] }],
            view: { summary: requestId, connections: [], callouts: [] },
          }), {});
          assert.equal(submitted.isError, undefined);
        } };
      } },
    },
  });
  // Supply the current serializer/editor state without starting a backend or provider.
  Object.defineProperty(graph, 'mcpModel', { get: () => model });
  const internals = graph as unknown as { agent: CanvasAgent; modelExplanation: CanvasModelExplanation;
    nodeExplanation: CanvasNodeExplanation; sourceEdits: LocalBrowserSourceEdits };
  internals.agent.routing = route;
  internals.agent.prepareExplanation = async (_signal, settings) => route(settings);
  internals.sourceEdits.readModelSource = async (_target, file = target.file) => {
    assert.equal(file, target.file);
    return { file, source, documentVersion: target.documentVersion, digest: createHash('sha256').update(source).digest('hex') };
  };
  graph.attachCanvas(message => { messages.push(structuredClone(message)); return true; });
  graph.setCanvasView(older.viewId, true);
  const selection: LocalBrowserViewSelectionMessage = { ...older, type: 'viewSelection', nodeIds: ['a'], edgeIds: [] };
  const cached = () => internals.modelExplanation.cached(selection, ref, route(newer.settings));
  const snapshot = () => ({ cached: cached() ?? null,
    selection: messages.filter(message => message.type === 'nodeExplanation' && !requests.some(request => request.requestId === message.state.requestId)).at(-1) ?? null });
  try {
    graph.handleCanvasMessage(selection);
    for (let index = 0; index < requests.length; index++) {
      graph.handleCanvasMessage(requests[index]);
      await until(() => gates.length === index + 1);
    }
    assert.ok(signals.every(signal => !signal.aborted), 'Both actual runtimes enter before either is released.');
    assert.equal(new Set(entered).size, 2);
    // Select while both run so the real accept path can adopt an authoritative result.
    internals.nodeExplanation.select(selection);
    const complete = async (index: number) => {
      gates[index]();
      await until(() => cleared.includes(requests[index].requestId));
      await new Promise<void>(resolve => setImmediate(resolve));
    };
    let beforeOlder;
    if (order === 'cancel-newer') {
      graph.handleCanvasMessage({ ...newer, action: 'cancel' });
      await until(() => cleared.includes(newer.requestId));
      beforeOlder = snapshot();
      await complete(0); gates[1]();
    } else if (order === 'newer-first') {
      await complete(1); beforeOlder = snapshot(); await complete(0);
    } else {
      beforeOlder = snapshot(); await complete(0); await complete(1);
    }
    await until(() => !internals.modelExplanation.busy);
    return { messages, entered, cleared, aborted: signals.map(signal => signal.aborted), beforeOlder, after: snapshot() };
  } finally { graph.dispose(); }
}
