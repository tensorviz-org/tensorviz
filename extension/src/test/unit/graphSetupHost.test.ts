import * as assert from 'node:assert/strict';
import { mkdtemp, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { GraphSetupHost } from '../../graphSetupHost';
import { sha256Text } from '../../workspaceSources';
import type { GraphSetupDiscovery, LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';

const vscode = require('../vscodeStub');
suite('workspace graph setup adapter', () => {
  let root: string, target: LocalBrowserTargetIdentity, doc: any, oldOpen: any, current: boolean;
  let calls: Array<{ url: string; body: any }>, response: (url: string, body: any) => unknown;
  let sources: Record<string, string>, host: GraphSetupHost;
  const discovery = (): GraphSetupDiscovery => ({ model: 'model:Net', sourceDigest: target.sourceDigest,
    init: [], inputs: [{ name: 'x', kind: 'keyword', required: true }],
    examples: [], factories: [], configs: [], parents: [], constraints: [], truncated: false });
  setup(async () => {
    vscode.reset(); current = true; calls = [];
    root = await mkdtemp(path.join(tmpdir(), 'tensorviz-setup-host-'));
    sources = { 'model.py': 'import torch\nclass Net(torch.nn.Module):\n def forward(self, x): return x\n' };
    target = { workspaceFolder: root, file: 'model.py', cls: 'Net', documentVersion: 1,
      sourceDigest: sha256Text(sources['model.py']), backendRevision: 1 };
    doc = { version: 1, isDirty: false, uri: vscode.Uri.file(path.join(root, 'model.py')), getText: () => sources['model.py'] };
    oldOpen = vscode.workspace.openTextDocument;
    vscode.workspace.openTextDocument = async () => doc;
    response = () => discovery();
    host = new GraphSetupHost({ origin: () => 'http://127.0.0.1:1234', token: () => 'local-token', current: () => current,
      sourceIO: () => ({ list: async () => Object.keys(sources), read: async file => sources[file],
        openDocument: file => file === 'model.py' ? doc : undefined }),
      fetchImpl: (async (url: string, options: RequestInit) => {
        const body = JSON.parse(options.body as string);
        calls.push({ url, body });
        assert.equal((options.headers as Record<string, string>)['X-tensorViz-Token'], 'local-token');
        return new Response(JSON.stringify(await response(url, body)), { status: 200 });
      }) as typeof fetch });
  });
  teardown(async () => { vscode.workspace.openTextDocument = oldOpen; await rm(root, { recursive: true, force: true }); });
  test('discovery uses live source with no trace, guessed input or filesystem path payload', async () => {
    const result = await host.discover(target, new AbortController().signal);
    assert.equal(result.discovery.model, 'model:Net');
    assert.equal(calls.length, 1);
    assert.equal(calls[0].body.sources[0].source, doc.getText());
    assert.equal(calls[0].body.trace_result, undefined);
    assert.equal(calls[0].body.root.measured_input_shape, undefined);
    assert.ok(!JSON.stringify(calls[0].body).includes(root));
  });
  test('source resolution provides the draft and validates provenance before use', async () => {
    const resolution = { status: 'complete', draft: { profile: { kind: 'model' } },
      evidence: [{ field: 'inputs.x', kind: 'example', detail: 'Synthetic values.' }], unresolved: [] };
    response = () => ({ ...discovery(), resolution });
    const result = await host.discover(target, new AbortController().signal);
    assert.deepEqual(result.draft, resolution.draft);
    response = () => ({ ...discovery(), resolution: { ...resolution, unresolved: ['Missing x'] } });
    await assert.rejects(host.discover(target, new AbortController().signal), /unavailable/);
    response = () => ({ ...discovery(), resolution: { ...resolution, evidence: [{ kind: 'guessed' }] } });
    await assert.rejects(host.discover(target, new AbortController().signal), /unavailable/);
  });
  test('saved config fields are restored only from the requested profile and a valid discovery', async () => {
    const saved = { profileName: 'chosen', profile: { model: 'model:Net', factory: { file: '.tensorviz/setup.py' } } };
    const restored = { profile: { model: 'model:Net', init: { args: [], kwargs: {} } },
      configObject: { index: 0, parameter: 'config', values: { width: 16 } } };
    response = () => ({ ...discovery(), configs: [{ file: 'model.py', cls: 'Config', parameter: 'config', fields: [], construction: 'call' }], restoredDraft: restored });
    const prepared = await host.discover(target, new AbortController().signal, saved);
    assert.deepEqual(calls[0].body.profile, saved.profile);
    assert.deepEqual(prepared.restoredDraft, { ...restored, profileName: 'chosen' });
    assert.equal('restoredDraft' in prepared.discovery, false, 'restoration is host preparation, not a new public form contract');
    response = () => ({ ...discovery(), restoredDraft: restored });
    await assert.rejects(host.discover(target, new AbortController().signal, saved), /could not be restored/);
    response = () => ({ ...discovery(), restoredDraft: null });
    assert.equal((await host.discover(target, new AbortController().signal, saved)).restoredDraft, undefined);
  });
  test('late discovery is discarded when the active source changes', async () => {
    response = () => { current = false; return discovery(); };
    await assert.rejects(host.discover(target, new AbortController().signal), /changed/);
  });
  test('a dependency edit while the form is open requires fresh discovery', async () => {
    await host.discover(target, new AbortController().signal);
    sources['helper.py'] = 'x = 1';
    const result = await host.save(target, { profile: {} }, new AbortController().signal);
    assert.match(result.requirements![0], /Workspace source changed/);
    assert.equal(calls.length, 1, 'a stale snapshot cannot prepare files');
  });
  test('missing fields are returned together without writing a config', async () => {
    await host.discover(target, new AbortController().signal);
    response = () => ({ ok: false, requirements: [
      { field: ['init', 'kwargs', 'width'], reason: 'Required.' }, { field: ['inputs', 'args', 0], reason: 'Required.' }] });
    const result = await host.save(target, { profile: { kind: 'model' } }, new AbortController().signal);
    assert.deepEqual(result.requirements, ['init.kwargs.width: Required.', 'inputs.args.0: Required.']);
    await assert.rejects(readFile(path.join(root, 'tensorviz.toml')), { code: 'ENOENT' });
  });
  test('a config created by someone else after discovery is preserved', async () => {
    await host.discover(target, new AbortController().signal);
    await writeFile(path.join(root, 'tensorviz.toml'), '# user settings\n');
    response = () => ({ ok: true, profile: 'net', config: 'version = 1\n', factory: null });
    await assert.rejects(host.save(target, { profile: {} }, new AbortController().signal), /changed/);
    assert.equal(await readFile(path.join(root, 'tensorviz.toml'), 'utf8'), '# user settings\n');
  });

  const runtimeDraft = { profile: { kind: 'model', model: 'model:Net', mode: 'eval', device: 'cpu',
    grad_enabled: false, seed: 0, init: { args: [], kwargs: {} }, inputs: { args: [], kwargs: {} } } };
  const tensor = { kind: 'tensor', shape: [1, 4], dtype: 'float32', requires_grad: false,
    generator: { kind: 'full', value: 0 } };
  const probe = () => ({ trusted: async () => true, settings: () => ({ onSave: false, tier: 'real' as const }),
    worker: { cancel() {}, run: async (request: any): Promise<any> => {
      assert.equal(request.prepareInputs, true); assert.equal(await request.beforeLaunch(), true);
      assert.deepEqual(request.execution.inputs.binding, { args: [], kwargs: {} });
      return { ok: true, executionCurrent: true, recheckExecution: async () => true,
        envelope: { result: { graphExample: { binding: { args: [], kwargs: { x: tensor } }, detail: 'Example input.' } } } };
    } } });
  const prepareRuntime = async () => {
    response = () => ({ ...discovery(), resolution: { status: 'partial', runtimeInputs: true,
      draft: runtimeDraft, evidence: [], unresolved: ['inputs.x: provide a value.'] } });
    await host.discover(target, new AbortController().signal);
  };
  test('input preparation exposes import compatibility failures without calling them missing libraries', async () => {
    await prepareRuntime();
    const trace = probe();
    trace.worker.run = async () => ({ ok: false, reason: 'private exception detail', envelope: { error: {
      diagnostic: { code: 'execution_failed', phase: 'import', names: ['ValueError'], field: ['model'] },
    } } });
    const result = await host.inferInputs(target, runtimeDraft, new AbortController().signal, trace);
    assert.equal(result.importFailure, true);
    assert.equal(result.missingModules, undefined);
    assert.match(result.detail, /compatibility/);
    assert.ok(!result.detail.includes('private'));
  });
  test('input preparation exposes missing library facts for native installation', async () => {
    await prepareRuntime();
    const trace = probe();
    trace.worker.run = async () => ({ ok: false, reason: 'missing dependency', envelope: { error: {
      diagnostic: { code: 'missing_dependency', phase: 'import', names: ['transformers'], field: ['model'] },
    } } });
    const result = await host.inferInputs(target, runtimeDraft, new AbortController().signal, trace);
    assert.deepEqual(result.missingModules, ['transformers']);
    assert.equal(result.draft, undefined);
  });
  for (const names of [['input_ids', 'attention_mask'], ['sample', 'timestep', 'encoder_hidden_states']]) {
    test(`unstructured inference failure names ${names.join(', ')} and preserves worker details`, async () => {
      response = () => ({ ...discovery(), inputs: names.map(name => ({ name, kind: 'keyword', required: false })),
        resolution: { status: 'partial', runtimeInputs: true, draft: runtimeDraft, evidence: [],
          unresolved: names.map(name => `inputs.${name}: provide a value.`) } });
      await host.discover(target, new AbortController().signal);
      const trace = probe();
      trace.worker.run = async () => ({ ok: false, reason: 'No example binding.', detail: 'Forward signature needs explicit tensors.',
        envelope: { error: { message: 'Could not infer tensor dimensions.' } } });
      const result = await host.inferInputs(target, runtimeDraft, new AbortController().signal, trace);
      assert.equal(result.draft, undefined);
      assert.match(result.detail, /Automatic input inference could not resolve the forward inputs/);
      for (const name of names) assert.ok(result.detail.includes(name));
      assert.match(result.detail, /setup form.*suggest_graph_inputs \/ configure_graph_setup/);
      assert.match(result.detail, /Could not infer tensor dimensions/);
      assert.match(result.detail, /Forward signature needs explicit tensors/);
    });
  }
  test('an inference failure with no worker details still gives the input recovery path', async () => {
    await prepareRuntime();
    const trace = probe(); trace.worker.run = async () => ({ ok: false });
    const result = await host.inferInputs(target, runtimeDraft, new AbortController().signal, trace);
    assert.match(result.detail, /forward inputs: x/);
    assert.match(result.detail, /configure_graph_setup/);
    assert.equal(result.missingModules, undefined);
  });
  test('unresolved inherited input names survive when no forward signature was discovered', async () => {
    response = () => ({ ...discovery(), inputs: null, resolution: { status: 'partial', runtimeInputs: true,
      draft: runtimeDraft, evidence: [], unresolved: ['inputs.encoder_hidden_states: provide a tensor.'] } });
    await host.discover(target, new AbortController().signal);
    const trace = probe(); trace.worker.run = async () => ({ ok: false });
    assert.match((await host.inferInputs(target, runtimeDraft, new AbortController().signal, trace)).detail,
      /forward inputs: encoder_hidden_states/);
  });
  test('long worker errors stay within the opening requirement budget and retain the recovery instructions', async () => {
    await prepareRuntime();
    const trace = probe(); trace.worker.run = async () => ({ ok: false, reason: 'worker error '.repeat(200) });
    const result = await host.inferInputs(target, runtimeDraft, new AbortController().signal, trace);
    assert.ok(result.detail.length <= 1024);
    assert.match(result.detail, /configure_graph_setup/);
    assert.match(result.detail, /worker error/);
  });
  test('runtime proposal changes only inputs and requires current private worker evidence', async () => {
    await prepareRuntime();
    const result = await host.inferInputs(target, runtimeDraft, new AbortController().signal, probe());
    assert.deepEqual(result.draft?.profile, { ...runtimeDraft.profile, inputs: { args: [], kwargs: { x: tensor } } });
    assert.equal(calls.length, 1, 'the preparation step does not save a profile or publish a capture');
  });
  for (const condition of ['untrusted', 'worker_stale', 'late_stale', 'dependency_changed', 'config_appeared']) {
    test(`runtime proposal is refused when ${condition}`, async () => {
      await prepareRuntime(); const trace = probe(), run = trace.worker.run;
      if (condition === 'untrusted') trace.trusted = async () => false;
      trace.worker.run = async request => {
        assert.notEqual(condition, 'untrusted');
        const result = await run(request);
        if (condition === 'worker_stale') result.executionCurrent = false;
        if (condition === 'late_stale') result.recheckExecution = async () => false;
        if (condition === 'dependency_changed') sources['dependency.py'] = 'x = 1';
        if (condition === 'config_appeared') await writeFile(path.join(root, 'tensorviz.toml'), 'version = 1');
        return result;
      };
      assert.equal((await host.inferInputs(target, runtimeDraft, new AbortController().signal, trace)).draft, undefined);
    });
  }
  test('only unresolved inherited construction authorizes config defaults and persists a typed recipe', async () => {
    response = () => ({ ...discovery(), init: null, resolution: { status: 'partial', runtimeInputs: true,
      draft: runtimeDraft, evidence: [], unresolved: ['inputs: unresolved'] } });
    await host.discover(target, new AbortController().signal);
    const trace = probe(), run = trace.worker.run;
    trace.worker.run = async request => {
      assert.equal(request.prepareConfigDefaults, true);
      const result = await run(request);
      result.envelope.result.graphExample.configDefaults = true;
      return result;
    };
    const result = await host.inferInputs(target, runtimeDraft, new AbortController().signal, trace);
    assert.equal(result.draft?.configDefaults, true);
    assert.deepEqual(result.draft?.profile.init, runtimeDraft.profile.init);
    response = () => ({ ok: false, requirements: [{ field: ['test'], reason: 'Stop before writing.' }] });
    await host.save(target, result.draft!, new AbortController().signal);
    assert.equal(calls.at(-1)!.body.config_defaults, true);
  });
  test('an input-only worker cannot invent a config-default recipe', async () => {
    await prepareRuntime(); const trace = probe(), run = trace.worker.run;
    trace.worker.run = async request => {
      assert.equal(request.prepareConfigDefaults, undefined);
      const result = await run(request);
      result.envelope.result.graphExample.configDefaults = true;
      return result;
    };
    assert.equal((await host.inferInputs(target, runtimeDraft, new AbortController().signal, trace)).draft, undefined);
  });

  test('a modified form draft cannot enter automatic runtime preparation', async () => {
    await prepareRuntime(); const trace = probe();
    trace.worker.run = async () => { assert.fail('Changed inputs must not run a probe.'); };
    const edited = { profile: { ...runtimeDraft.profile, inputs: { args: [], kwargs: { x: tensor } } } };
    assert.equal((await host.inferInputs(target, edited, new AbortController().signal, trace)).draft, undefined);
  });

});
