import * as assert from 'node:assert/strict';
import { registerMcpGraphSetup } from '@tensorviz/graph-host/mcpGraphSetup';
import { GraphOpeningController, type GraphOpeningDependencies } from '@tensorviz/graph-host/graphOpening';
import type { GraphSetupHost } from '../../graphSetupHost';
import type { GraphSetupDiscovery, LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';

suite('MCP graph setup', () => {
  const target: LocalBrowserTargetIdentity = { workspaceFolder: '/workspace', file: 'model.py', cls: 'Net',
    documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 0 };
  const discovery: GraphSetupDiscovery = { model: 'model:Net', sourceDigest: target.sourceDigest,
    init: [], inputs: [], examples: [], factories: [], configs: [], parents: [], constraints: [], truncated: false,
    resolution: { status: 'partial', draft: { profile: { kind: 'model', inputs: { args: [], kwargs: {} } } },
      evidence: [], unresolved: ['inputs.x'] } };
  async function rig() {
    let saves = 0, reads = 0;
    const deps: GraphOpeningDependencies = { current: () => true, lookup: async () => ({ ok: false, kind: 'none' }),
      discover: async () => ({ discovery }), save: async () => { saves++; return { requirements: ['inputs.x: Required.'] }; },
      capture: async () => { throw new Error('Discovery must never execute.'); }, recover: async () => undefined, parent: async () => {}, publish: () => {} };
    const opening = new GraphOpeningController(deps); await opening.open(target);
    const handlers: Record<string, (args: any) => Promise<any>> = {};
    registerMcpGraphSetup({ registerTool: (name: string, _schema: unknown, handler: any) => { handlers[name] = handler; } } as never,
      opening, { discover: async () => { reads++; return { discovery }; } } as unknown as GraphSetupHost);
    return { opening, handlers, get saves() { return saves; }, get reads() { return reads; } };
  }
  test('reads source choices through the shared adapter without saving or running', async () => {
    const r = await rig(); const result = await r.handlers.get_graph_setup({});
    assert.equal(result.structuredContent.discovery.model, 'model:Net');
    assert.equal(r.reads, 1); assert.equal(r.saves, 0);
  });
  test('packages unresolved inputs and a fillable draft for an LLM without saving or running', async () => {
    const r = await rig(); const data = (await r.handlers.suggest_graph_inputs({})).structuredContent;
    assert.equal(data.model.file, 'model.py');
    assert.equal(data.model.cls, 'Net');
    assert.deepEqual(data.resolution.unresolved, ['inputs.x']);
    assert.equal(data.draft.profile.kind, 'model');
    assert.match(data.guidance, /Net\.forward/);
    assert.equal(r.reads, 1); assert.equal(r.saves, 0);
  });
  test('publishes the tensor grammar so a client never reads tensorViz source', async () => {
    const r = await rig(); const contract = (await r.handlers.suggest_graph_inputs({})).structuredContent.inputContract;
    assert.deepEqual(contract.dtypes, ['bool', 'uint8', 'int8', 'int16', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'float64']);
    assert.deepEqual(Object.keys(contract.tensor.generator).sort(), ['full', 'normal', 'randint', 'uniform', 'values']);
    assert.deepEqual(contract.tensor.generator.randint.fields, ['low', 'high', 'seed']);
    assert.equal(contract.tensor.fields.includes('generator'), true);
  });
  test('stale setup cannot write files', async () => {
    const r = await rig(); const result = await r.handlers.configure_graph_setup({ openingId: 'old', sourceDigest: target.sourceDigest, draft: { profile: {} } });
    assert.equal(result.isError, true); assert.equal(r.saves, 0);
  });
  test('explicit configuration enters the same opening and returns grouped requirements', async () => {
    const r = await rig(), id = r.opening.state!.id;
    await r.handlers.configure_graph_setup({ openingId: id, sourceDigest: target.sourceDigest, draft: { profile: {} } });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(r.saves, 1); assert.equal(r.opening.state!.id, id);
    assert.deepEqual(r.opening.state!.requirements, ['inputs.x: Required.']);
  });
});
