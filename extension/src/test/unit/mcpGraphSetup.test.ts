import * as assert from 'node:assert/strict';
import { registerMcpGraphSetup } from '@tensorviz/graph-host/mcpGraphSetup';
import { GraphOpeningController, type GraphOpeningDependencies } from '@tensorviz/graph-host/graphOpening';
import type { GraphSetupHost } from '../../graphSetupHost';
import type { GraphSetupDiscovery, GraphSetupDraft, LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';

suite('MCP graph setup', () => {
  const target: LocalBrowserTargetIdentity = { workspaceFolder: '/workspace', file: 'model.py', cls: 'Net',
    documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 0 };
  const discovery: GraphSetupDiscovery = { model: 'model:Net', sourceDigest: target.sourceDigest,
    init: [], inputs: [], examples: [], factories: [], configs: [], parents: [], constraints: [], truncated: false,
    resolution: { status: 'partial', draft: { profile: { kind: 'model', inputs: { args: [], kwargs: {} } } },
      evidence: [], unresolved: ['inputs.x'] } };
  async function rig(restoredDraft?: GraphSetupDraft) {
    let saves = 0, reads = 0;
    let discoveredDraft: GraphSetupDraft | undefined;
    const deps: GraphOpeningDependencies = { current: () => true, lookup: async () => ({ ok: false, kind: 'none' }),
      discover: async () => ({ discovery }), save: async () => { saves++; return { requirements: ['inputs.x: Required.'] }; },
      capture: async () => { throw new Error('Discovery must never execute.'); }, recover: async () => undefined, parent: async () => {}, publish: () => {} };
    const opening = new GraphOpeningController(deps); await opening.open(target);
    const handlers: Record<string, (args: any) => Promise<any>> = {};
    const setup = { discover: async (_target: unknown, _signal: unknown, saved?: GraphSetupDraft) => {
      reads++; discoveredDraft = saved; return { discovery, draft: discovery.resolution!.draft, restoredDraft };
    } };
    registerMcpGraphSetup({ registerTool: (name: string, _schema: unknown, handler: any) => { handlers[name] = handler; } } as never,
      opening, setup as unknown as GraphSetupHost);
    return { opening, handlers, setup, get saves() { return saves; }, get reads() { return reads; }, get discoveredDraft() { return discoveredDraft; } };
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
  const savedDraft: GraphSetupDraft = { profileName: 'my_model', profile: { kind: 'model', model: 'model:Net',
    init: { args: [], kwargs: { width: 12 } }, inputs: { args: [{ kind: 'tensor', shape: [3, 12], dtype: 'float64',
      requires_grad: false, generator: { kind: 'full', value: 2 } }], kwargs: {} } } };
  for (const tool of ['get_graph_setup', 'suggest_graph_inputs']) {
    test(`${tool} rejects a late read after agent help updates the same opening draft`, async () => {
      const r = await rig();
      let release!: () => void;
      const wait = new Promise<void>(resolve => { release = resolve; });
      const discover = r.setup.discover;
      r.setup.discover = async (...args) => { await wait; return discover(...args); };
      const read = r.handlers[tool]({});
      await r.opening.action(r.opening.state!.id, 'help', target, { kind: 'agent_help', draft: savedDraft });
      release();
      assert.equal((await read).isError, true);
      assert.deepEqual(r.opening.state?.draft, savedDraft);
      assert.equal(r.saves, 0);
    });
    test(`${tool} exposes the pending form draft shared through agent help without saving`, async () => {
      const r = await rig();
      await r.opening.action(r.opening.state!.id, 'help', target, { kind: 'agent_help', draft: savedDraft });
      assert.deepEqual((await r.handlers[tool]({})).structuredContent.draft, savedDraft);
      assert.equal(r.saves, 0);
    });
    test(`${tool} retains the selected draft when source discovery proposes different inputs`, async () => {
      const r = await rig();
      await r.opening.action(r.opening.state!.id, 'configure', target, { kind: 'submit', draft: savedDraft });
      const saves = r.saves, data = (await r.handlers[tool]({})).structuredContent;
      assert.deepEqual(data.draft, savedDraft);
      assert.deepEqual(r.discoveredDraft, savedDraft, 'discovery must receive the selected profile for helper restoration');
      assert.equal(r.saves, saves, 'reading a suggestion must not save or execute it');
    });
    test(`${tool} restores saved configuration fields but preserves pending configuration edits`, async () => {
      const restored = { ...savedDraft, configObject: { index: 0, parameter: 'config', values: { width: 12 } } };
      const r = await rig(restored);
      await r.opening.action(r.opening.state!.id, 'configure', target, { kind: 'submit', draft: savedDraft });
      assert.deepEqual((await r.handlers[tool]({})).structuredContent.draft, restored);
      const pending = { ...restored, configObject: { ...restored.configObject, values: { width: 16 } } };
      await r.opening.action(r.opening.state!.id, 'edit-config', target, { kind: 'submit', draft: pending });
      assert.deepEqual((await r.handlers[tool]({})).structuredContent.draft, pending);
    });
  }
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
