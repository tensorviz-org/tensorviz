import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { prepareCapturedSourceEdit } from '../../capturedSourceEdit';
import { LocalBrowserSourceEdits } from '../../localBrowserSourceEdits';
import { isLocalBrowserSourceEdit, type ProjectionPayload } from '@tensorviz/contracts/protocol';
import { mcpEditSchema } from '../../mcpProposals';
import { McpGraphService } from '@tensorviz/graph-host/mcpGraph';
import { sha256 } from '../../sourceChanges';
import { editorState, reset, resetEditors } from '../vscodeStub';
const vscode = require('../vscodeStub');

const fixtures: Record<string, string> = { 'unary-insert': 'captured-unary-insert', 'unary-import': 'captured-unary-import', 'unary-delete': 'captured-unary-delete', reconnection: 'captured-reconnection-edit', literal: 'captured-source-edit', activation: 'captured-activation-edit',
  module: 'captured-module-activation-edit', 'activation-import': 'captured-activation-import', 'module-import': 'captured-module-activation-import',
  constructor: 'captured-constructor-edit', 'constructor-keyword': 'captured-constructor-keyword', 'constructor-custom': 'captured-custom-constructor',
  'activation-options': 'captured-activation-options', 'module-options': 'captured-module-activation-options',
  configuration: 'captured-configuration-edit', 'configuration-module': 'captured-configuration-module' };
const fixture = (recipe = 'literal') => JSON.parse(readFileSync(join(__dirname, `../../../../tests/fixtures/contracts/${fixtures[recipe]}.json`), 'utf8'));

for (const recipe of Object.keys(fixtures)) suite(`captured ${recipe} native review`, () => {
  setup(() => { reset(); resetEditors(); });
  function harness(review?: ConstructorParameters<typeof LocalBrowserSourceEdits>[0]['review'],
    draft?: ConstructorParameters<typeof LocalBrowserSourceEdits>[0]['draft']) {
    const data = fixture(recipe);
    const target = { workspaceFolder: '/work', file: 'model.py', cls: 'Model', documentVersion: 1,
      sourceDigest: sha256(data.source), backendRevision: 1 };
    const node = data.document.semantic.nodes.find((n: any) => n.metadata.captureRef.semanticId === data.intent.target.semanticId);
    const edit = { ...data.intent, nodeId: node.node_id };
    const context = { projection: { document: data.document, spanmap: data.spans } as ProjectionPayload,
      dependencies: new Map([['model.py', sha256(data.source)]]) };
    editorState.documents.set('/work/model.py', data.source);
    let retained = true;
    const requests: any[] = [];
    const host = new LocalBrowserSourceEdits({ draft, origin: () => 'http://localhost:9000', token: () => 'HOST_ONLY',
      context: () => context, capture: (_target, captureId, sourceId) => {
        if (!retained || captureId !== data.capture.id || sourceId !== data.capture.source.id) throw new Error('evicted');
        return data.capture;
      }, publish() {}, channel: { appendLine() {} }, review: review ?? (async () => true),
      fetchImpl: async (url, init) => {
        requests.push({ url, body: JSON.parse(String(init!.body)), headers: init!.headers });
        return new Response(JSON.stringify(data.plan));
      },
    });
    return { data, node, edit, target, context, requests, host, evict: () => { retained = false; },
      propose: () => host.propose({ requestId: 'captured-edit', target, edit }) };
  }

  test('actual Python capability reaches one native diff and precise WorkspaceEdit', async () => {
    const h = harness(async (title, artifact) => {
      assert.match(title, recipe.startsWith('configuration') ? /configuration definition.*all references.*runtime not run/ : /all executions.*runtime not run/);
      if (recipe.endsWith('options')) assert.match(title, /inplace=True preserved/);
      assert.equal(artifact.before, h.data.source);
      assert.equal(artifact.after, h.data.after);
      assert.equal(editorState.appliedEdits.length, 0);
      return true;
    });
    const outcome = await h.propose();
    assert.equal(outcome.status, 'applied');
    assert.match(outcome.detail, /runtime validation has not run/);
    assert.equal(h.requests.length, 1);
    assert.equal(h.requests[0].url, 'http://localhost:9000/api/ide/capture/edit');
    assert.deepEqual(h.requests[0].body.intent, h.data.intent);
    assert.deepEqual(h.requests[0].body.capture, h.data.capture);
    assert.equal(h.requests[0].headers['X-tensorViz-Token'], 'HOST_ONLY');
    assert.equal(editorState.appliedEdits.length, 1);
    const changes = editorState.appliedEdits[0].replacements;
    assert.equal(changes.length, h.data.plan.edits.length);
    assert.deepEqual(changes.map((change: any) => change.newText), h.data.plan.edits.map((edit: any) => edit.newText));
    assert.ok(changes.every((change: any) => change.uri.fsPath === '/work/model.py'));
  });

  test('captured typed review previews the exact recipe artifact without importing config as source', async () => {
    let disposed = false, shown = false;
    const h = harness(async (_title, _artifact, _signal, _candidate, onShown) => {
      onShown?.(async () => {}, () => {}); assert.ok(shown); return false;
    }, (input, _signal, current) => {
      assert.equal(input.canRevise, false); assert.equal(input.artifacts[0].before, h.data.source); assert.equal(input.artifacts[0].after, h.data.after);
      assert.deepEqual(input.sources.entries.map(source => source.file), ['model.py']); assert.equal(current(), true);
      assert.equal(editorState.appliedEdits.length, 0);
      return { show() { shown = true; }, dispose() { disposed = true; } };
    });
    editorState.documents.set('/work/tensorviz.toml', '# retained recipe');
    h.data.capture.source.files.push({ path: 'tensorviz.toml', role: 'config', digest: sha256('# retained recipe'), documentVersion: 1 });
    try {
      assert.equal((await h.propose()).status, 'cancelled'); assert.ok(disposed);
      assert.equal(editorState.appliedEdits.length, 0);
    } finally { h.host.dispose(); }
  });

  test('no-op produces no review or write, and discard writes nothing', async () => {
    const h = harness(async () => { assert.fail('no-op must not open review'); });
    h.edit.value = h.edit.expected;
    h.data.plan.edits = []; h.data.plan.source_digest = h.target.sourceDigest;
    const result = await h.propose();
    assert.equal(result.status, 'refused');
    if (!recipe.startsWith('unary')) assert.match(result.detail, /unchanged/);
    assert.equal(editorState.appliedEdits.length, 0);
    assert.equal((await harness(async () => false).propose()).status, 'cancelled');
    assert.equal(editorState.appliedEdits.length, 0);
  });

  for (const kind of ['version', 'dependency', 'configuration', 'eviction', 'cancel']) {
    test(`a changed ${kind} cannot apply after native review`, async () => {
      let reviewed = false;
      const h = harness(async () => {
        reviewed = true;
        if (kind === 'version') editorState.versions.set('/work/model.py', 2);
        if (kind === 'dependency') editorState.documents.set('/work/helper.py', '# changed\n');
        if (kind === 'configuration') editorState.documents.set('/work/tensorviz.toml', '# changed\n');
        if (kind === 'eviction') h.evict();
        if (kind === 'cancel') h.host.cancel();
        return true;
      });
      // Dependencies imported at runtime need not be in the static projection.
      const file = kind === 'configuration' ? 'tensorviz.toml' : 'helper.py';
      editorState.documents.set('/work/' + file, '# dependency\n');
      h.data.capture.source.files.push({ path: file, role: kind === 'configuration' ? 'config' : 'source',
        digest: sha256('# dependency\n'), documentVersion: 1 });
      assert.notEqual((await h.propose()).status, 'applied');
      assert.equal(reviewed, true);
      assert.equal(editorState.appliedEdits.length, 0);
    });
  }

  test('forged capabilities, source changes and other selections refuse before backend work', async () => {
    for (const change of [
      (h: ReturnType<typeof harness>) => { h.edit.capabilityId = 'unknown'; },
      (h: ReturnType<typeof harness>) => { h.edit.nodeId = 'other'; },
      (h: ReturnType<typeof harness>) => { h.edit.expected = h.edit.kind !== 'replace_activation' ? { kind: 'int', value: '99' } : 'tanh'; },
      () => { editorState.documents.set('/work/model.py', '# stale\n'); },
    ]) {
      const h = harness(async () => { assert.fail('forged request reached review'); });
      change(h);
      assert.equal((await h.propose()).status, 'refused');
      assert.equal(h.requests.length, 0);
    }
  });

  test('MCP inspection and intent schema agree with the real Python producer', () => {
    const h = harness();
    const model = { target: h.target, projection: h.context.projection, current: true };
    const graph = new McpGraphService('test-host', () => model);
    const node = graph.node(h.node, h.context.projection);
    assert.deepEqual(node.sourceEditCapabilities, h.node.metadata.sourceEditCapabilities);
    assert.ok(node.sourceEditCapabilities.some((cap: any) => cap.id === h.data.capability.id));
    assert.ok(isLocalBrowserSourceEdit(h.edit)); assert.ok(mcpEditSchema.safeParse(h.edit).success);
    for (const patch of [{ value: { kind: 'int', value: '2\n' } }, { scope: 'one_call' }, { scope: recipe.startsWith('configuration') ? 'source_expression' : 'configuration_definition' }, { path: 'model.py' },
      { argumentPath: ['args', recipe === 'reconnection' || recipe === 'unary-insert' || recipe === 'unary-import' ? 2 : recipe === 'unary-delete' ? 1 : 0] }, { value: { kind: 'float', value: -0 } }]) {
      assert.equal(isLocalBrowserSourceEdit({ ...h.edit, ...patch }), false);
      assert.equal(mcpEditSchema.safeParse({ ...h.edit, ...patch }).success, false);
    }
  });

  if (recipe.startsWith('constructor') || recipe.startsWith('configuration')) test('a substituted constructor field is refused before planning', async () => {
    const h = harness(async () => { assert.fail('substituted field reached review'); });
    h.edit.sourceArgumentPath = ['args', 0];
    assert.equal((await h.propose()).status, 'refused');
    assert.equal(h.requests.length, 0);
    assert.equal(editorState.appliedEdits.length, 0);
  });

  if (recipe === 'reconnection' || recipe.startsWith('unary')) test('the other use of the same tensor and unavailable values refuse before planning', async () => {
    for (const change of [
      (h: ReturnType<typeof harness>) => { h.edit.argumentPath = ['args', 1 - h.edit.argumentPath[1]]; },
      (h: ReturnType<typeof harness>) => { h.edit.value = h.data.capture.operations.at(-1).output; },
    ]) {
      const h = harness(async () => { assert.fail('unproven reconnection reached review'); });
      change(h);
      assert.equal((await h.propose()).status, 'refused');
      assert.equal(h.requests.length, 0);
      assert.equal(editorState.appliedEdits.length, 0);
    }
  });

  for (const file of ['model.py', 'helper.py', 'tensorviz.toml']) {
    for (const change of ['changed', 'unchanged', 'closed-again']) {
      test(`review rechecks ${file} after its document closes (${change})`, async () => {
        // VS Code keeps a closed TextDocument readable but frozen; the ordinary
        // test stub reads current text and cannot reproduce this lifecycle gap.
        const original = vscode.workspace.openTextDocument;
        const held = new Map<string, { document: any; closed: boolean }>();
        let reviewed = false, reopened = false;
        vscode.workspace.openTextDocument = async (uri: any) => {
          const previous = held.get(uri.fsPath);
          if (reviewed && reopened && change === 'closed-again') held.get('/work/' + file)!.closed = true;
          if (previous && !previous.closed) return previous.document;
          const document = await original(uri);
          const text = document.getText(), version = document.version;
          const entry = { document, closed: false };
          document.getText = () => text;
          Object.defineProperty(document, 'version', { get: () => version });
          Object.defineProperty(document, 'isClosed', { get: () => entry.closed });
          held.set(uri.fsPath, entry);
          if (reviewed && uri.fsPath === '/work/' + file) reopened = true;
          return document;
        };
        try {
          const h = harness(async () => {
            reviewed = true;
            held.get('/work/' + file)!.closed = true;
            if (change === 'changed') editorState.documents.set('/work/' + file, '# changed on disk while closed\n');
            return true;
          });
          if (file !== 'model.py') {
            editorState.documents.set('/work/' + file, '# dependency\n');
            h.data.capture.source.files.push({ path: file, role: file === 'tensorviz.toml' ? 'config' : 'source',
              digest: sha256('# dependency\n'), documentVersion: 1 });
          }
          const result = await h.propose();
          assert.equal(reviewed, true);
          assert.equal(result.status === 'applied', change === 'unchanged');
          assert.equal(editorState.appliedEdits.length, change === 'unchanged' ? 1 : 0);
          if (change === 'unchanged') assert.equal(reopened, true);
        } finally { vscode.workspace.openTextDocument = original; }
      });
    }
  }
});

suite('captured proposal transport boundary', () => {
  for (const recipe of ['unary-insert', 'unary-import', 'unary-delete']) test(`${recipe} cannot change the selected operand, original input, callee or import`, async () => {
    for (const change of [
      (p: any) => { p.edits[0].range.start[1]--; },
      (p: any) => { p.edits[0].newText = 'danger()'; },
      (p: any) => { p.edits[0].newText = recipe === 'unary-delete' ? 'x' : 'F.relu(a)'; },
      (p: any) => { p.edits[0].newText = recipe === 'unary-delete' ? 'h.clone()' : 'F.gelu(x)'; },
      (p: any) => { p.edits.push(p.edits[0]); },
      (p: any) => { p.capability.argumentPath = ['args', 1 - p.capability.argumentPath[1]]; },
    ]) {
      const data = fixture(recipe); change(data.plan);
      await assert.rejects(prepareCapturedSourceEdit(data.capture, { 'model.py': data.source }, data.intent, data.capability,
        { origin: 'http://localhost:9000', token: 'HOST_ONLY', signal: new AbortController().signal,
          fetchImpl: async () => new Response(JSON.stringify(data.plan)) }));
    }
  });

  test('unary deletion refuses another available tensor before backend work', async () => {
    const data = fixture('unary-delete');
    data.intent.value = data.capture.inputs.args[0];
    await assert.rejects(prepareCapturedSourceEdit(data.capture, { 'model.py': data.source }, data.intent, data.capability,
      { origin: 'http://localhost:9000', token: 'HOST_ONLY', signal: new AbortController().signal,
        fetchImpl: async () => { assert.fail('different bypass input reached backend'); } }), /proven original input/);
  });

  test('unary insertion verifies its exact required fresh import', async () => {
    for (const change of [
      (p: any) => { p.edits.pop(); },
      (p: any) => { p.edits[1].newText = 'import other\n'; },
      (p: any) => { p.edits[1].range.start[0]++; p.edits[1].range.end[0]++; },
    ]) {
      const data = fixture('unary-import'); change(data.plan);
      await assert.rejects(prepareCapturedSourceEdit(data.capture, { 'model.py': data.source }, data.intent, data.capability,
        { origin: 'http://localhost:9000', token: 'HOST_ONLY', signal: new AbortController().signal,
          fetchImpl: async () => new Response(JSON.stringify(data.plan)) }));
    }
  });

  for (const recipe of ['unary-insert', 'unary-delete']) test(`${recipe} wire contracts reject code, options and invalid input identities`, () => {
    const data = fixture(recipe);
    const edit = { ...data.intent, nodeId: 'operation' };
    for (const patch of [
      { value: recipe === 'unary-insert' ? 'F.gelu' : { kind: 'value', id: '' } },
      { expected: { kind: 'int', value: '1' } }, { argumentPath: ['args', recipe === 'unary-insert' ? 2 : 1] },
      { argumentPath: ['kwargs', 'input'] }, { argumentPath: ['args', true] },
      { options: { inplace: true } }, { name: 'h' }, { importPlan: {} },
    ]) {
      assert.equal(isLocalBrowserSourceEdit({ ...edit, ...patch }), false);
      assert.equal(mcpEditSchema.safeParse({ ...edit, ...patch }).success, false);
    }
  });

  test('reconnection wire guards accept only an exact operand and bounded ValueRefs', () => {
    const data = fixture('reconnection');
    const edit = { ...data.intent, nodeId: 'add' };
    for (const port of [0, 1]) {
      assert.ok(isLocalBrowserSourceEdit({ ...edit, argumentPath: ['args', port] }));
      assert.ok(mcpEditSchema.safeParse({ ...edit, argumentPath: ['args', port] }).success);
    }
    for (const patch of [
      ...[true, 2, -1, 0.5, '1'].map(port => ({ argumentPath: ['args', port] })),
      { argumentPath: ['kwargs', 'input'] }, { argumentPath: ['args', 1, 0] },
      { value: { kind: 'value', id: '' } }, { value: { kind: 'value', id: 'x'.repeat(129) } },
      { value: { ...edit.value, name: 'h' } }, { name: 'h' }, { target: { ...edit.target, kind: 'invocation' } },
    ]) {
      assert.equal(isLocalBrowserSourceEdit({ ...edit, ...patch }), false);
      assert.equal(mcpEditSchema.safeParse({ ...edit, ...patch }).success, false);
    }
  });

  test('reconnection backend cannot replace the other repeated operand or substitute source code', async () => {
    for (const change of [
      (p: any) => { p.edits[0].range.start[1] -= 4; p.edits[0].range.end[1] -= 4; },
      (p: any) => { p.edits[0].newText = 'x'; }, // Available, but not the requested h.
      (p: any) => { p.edits[0].newText = 'h.clone()'; },
      (p: any) => { p.edits[0].newText = 'h\n'; },
      (p: any) => { p.edits.push(p.edits[0]); },
      (p: any) => { p.capability.choices[0].name = 'danger'; },
      (p: any) => { p.capability.argumentPath = ['args', 0]; },
    ]) {
      const data = fixture('reconnection'); change(data.plan);
      await assert.rejects(prepareCapturedSourceEdit(data.capture, { 'model.py': data.source }, data.intent, data.capability,
        { origin: 'http://localhost:9000', token: 'HOST_ONLY', signal: new AbortController().signal,
          fetchImpl: async () => new Response(JSON.stringify(data.plan)) }));
    }
  });

  for (const recipe of ['activation-options', 'module-options']) test(`${recipe} refuses an incompatible activation before contacting the backend`, async () => {
    const data = fixture(recipe);
    data.intent.value = 'gelu';
    await assert.rejects(prepareCapturedSourceEdit(data.capture, { 'model.py': data.source }, data.intent, data.capability,
      { origin: 'http://localhost:9000', token: 'HOST_ONLY', signal: new AbortController().signal,
        fetchImpl: async () => { assert.fail('incompatible activation reached backend'); } }), /preserve the authored options/);
  });

  for (const recipe of ['constructor', 'configuration']) test(`${recipe} path wire guards preserve Python identifiers and reject invalid positions`, () => {
    const data = fixture(recipe);
    const edit = { ...data.intent, nodeId: 'call' };
    for (const sourceArgumentPath of [['args', 0], ['args', 255], ['kwargs', 'p'], ['kwargs', '宽度'], ['kwargs', 'é'], ['kwargs', '𐐀'.repeat(128)]]) {
      assert.ok(isLocalBrowserSourceEdit({ ...edit, sourceArgumentPath }));
      assert.ok(mcpEditSchema.safeParse({ ...edit, sourceArgumentPath }).success);
    }
    for (const sourceArgumentPath of [['args', -1], ['args', 256], ['args', true], ['args', 0.5], ['kwargs', 'p\n'],
      ['kwargs', 'p.x'], ['kwargs', '😀'], ['kwargs', 'a'.repeat(129)], ['kwargs', ''], ['args', 1, 0]]) {
      assert.equal(isLocalBrowserSourceEdit({ ...edit, sourceArgumentPath }), false);
      assert.equal(mcpEditSchema.safeParse({ ...edit, sourceArgumentPath }).success, false);
    }
    assert.equal(isLocalBrowserSourceEdit({ ...edit, target: { ...edit.target, kind: 'operation' } }), false);
  });

  for (const recipe of ['constructor', 'constructor-keyword', 'constructor-custom', 'configuration', 'configuration-module']) test(`${recipe} cannot replace another argument or inject extra code`, async () => {
    for (const change of [
      (p: any) => { p.edits[0].range.start[1]--; },
      (p: any) => { p.edits[0].newText = 'danger()'; },
      (p: any) => { p.edits[0].newText = '999'; },
      (p: any) => { p.edits.push({ range: { start: [1, 0], end: [1, 0] }, newText: 'import other\n' }); },
      (p: any) => { p.capability.sourceArgumentPath = ['args', 0]; },
    ]) {
      const data = fixture(recipe); change(data.plan);
      await assert.rejects(prepareCapturedSourceEdit(data.capture, { 'model.py': data.source }, data.intent, data.capability,
        { origin: 'http://localhost:9000', token: 'HOST_ONLY', signal: new AbortController().signal,
          fetchImpl: async () => new Response(JSON.stringify(data.plan)) }));
    }
  });
  for (const recipe of ['configuration', 'configuration-module']) test(`${recipe} cannot substitute the name reference for its definition`, async () => {
    const data = fixture(recipe);
    const reference = data.capability.reference.anchors[0];
    data.plan.edits[0].range = { start: [reference.startLine, reference.startColumn], end: [reference.endLine, reference.endColumn] };
    await assert.rejects(prepareCapturedSourceEdit(data.capture, { 'model.py': data.source }, data.intent, data.capability,
      { origin: 'http://localhost:9000', token: 'HOST_ONLY', signal: new AbortController().signal,
        fetchImpl: async () => new Response(JSON.stringify(data.plan)) }), /outside its exact source binding/);
  });
  function prepare(fetchImpl: typeof fetch, signal = new AbortController().signal) {
    const data = fixture();
    return prepareCapturedSourceEdit(data.capture, { 'model.py': data.source }, data.intent, data.capability,
      { origin: 'http://localhost:9000', token: 'HOST_ONLY', signal, fetchImpl });
  }
  test('backend cannot widen the range, change the literal, scope, or exact base', async () => {
    for (const change of [
      (p: any) => { p.edits[0].range.start = [1, 0]; },
      (p: any) => { p.edits[0].newText = 'danger()'; },
      (p: any) => { p.edits[0].newText = '999'; },
      (p: any) => { p.capability.scope.observedOperations++; },
      (p: any) => { p.file = 'other.py'; },
      (p: any) => { p.validation = 'runtime_verified'; },
    ]) {
      const data = fixture(); change(data.plan);
      await assert.rejects(prepare(async () => new Response(JSON.stringify(data.plan))));
    }
  });
  test('oversized and malformed responses cannot leak source or create proposals', async () => {
    let cancelled = false;
    await assert.rejects(prepare(async () => new Response(new ReadableStream({
      start(controller) { controller.enqueue(new Uint8Array(65 * 1024)); }, cancel() { cancelled = true; },
    }))));
    assert.equal(cancelled, true);
    for (const status of [200, 400, 413, 500]) {
      await assert.rejects(prepare(async () => new Response('PRIVATE_SOURCE', { status })), error => {
        assert.ok(error instanceof Error && !error.message.includes('PRIVATE_SOURCE')); return true;
      });
    }
  });
  test('a floating-point request cannot become an integer token with the same numeric value', async () => {
    const data = fixture();
    data.intent.value = { kind: 'float', value: 2 };
    const run = () => prepareCapturedSourceEdit(data.capture, { 'model.py': data.source }, data.intent, data.capability,
      { origin: 'http://localhost:9000', token: 'HOST_ONLY', signal: new AbortController().signal,
        fetchImpl: async () => new Response(JSON.stringify(data.plan)) });
    data.plan.edits[0].newText = '2';
    await assert.rejects(run(), /requested numeric literal/);
    data.plan.edits[0].newText = '2.0';
    assert.equal((await run()).edits[0].newText, '2.0');
  });
  for (const recipe of ['activation', 'module']) test(`${recipe} plans cannot switch namespace, replace inputs, add imports or choose another activation`, async () => {
    for (const change of [
      (p: any) => { p.edits[0].newText = `${p.capability.namespace}.${recipe === 'module' ? 'SiLU' : 'silu'}`; },
      (p: any) => { p.edits[0].newText = 'other.gelu'; },
      (p: any) => { p.edits[0].newText = 'F.gelu(h + 2)'; },
      (p: any) => { p.edits.push({ range: { start: [1, 0], end: [1, 0] }, newText: 'import other\n' }); },
      (p: any) => { p.capability.namespace = 'other'; },
    ]) {
      const data = fixture(recipe); change(data.plan);
      await assert.rejects(prepareCapturedSourceEdit(data.capture, { 'model.py': data.source }, data.intent, data.capability,
        { origin: 'http://localhost:9000', token: 'HOST_ONLY', signal: new AbortController().signal,
          fetchImpl: async () => new Response(JSON.stringify(data.plan)) }));
    }
  });
  test('module constructors require the exact class spelling rather than a functional name', async () => {
    const data = fixture('module'); data.plan.edits[0].newText = 'nn.gelu';
    await assert.rejects(prepareCapturedSourceEdit(data.capture, { 'model.py': data.source }, data.intent, data.capability,
      { origin: 'http://localhost:9000', token: 'HOST_ONLY', signal: new AbortController().signal,
        fetchImpl: async () => new Response(JSON.stringify(data.plan)) }), /requested activation/);
  });
  for (const recipe of ['activation-import', 'module-import']) test(`${recipe} only permits its exact required import in the same atomic edit`, async () => {
    for (const change of [
      (p: any) => { p.edits.pop(); },
      (p: any) => { p.edits[1].range.start[0]++; p.edits[1].range.end[0]++; },
      (p: any) => { p.edits[1].range.end[1] = 2; },
      (p: any) => { p.edits[1].newText += 'run()\n'; },
      (p: any) => { p.edits[1].newText = p.edits[1].newText.replace('torch.nn', 'other'); },
      (p: any) => { p.capability.importPlan.line++; },
      (p: any) => { p.edits.push(p.edits[1]); },
    ]) {
      const data = fixture(recipe); change(data.plan);
      await assert.rejects(prepareCapturedSourceEdit(data.capture, { 'model.py': data.source }, data.intent, data.capability,
        { origin: 'http://localhost:9000', token: 'HOST_ONLY', signal: new AbortController().signal,
          fetchImpl: async () => new Response(JSON.stringify(data.plan)) }));
    }
  });
  test('cancellation returns even if the transport ignores AbortSignal; late result cannot apply', async () => {
    const controller = new AbortController();
    let finish!: (response: Response) => void;
    const pending = prepare(async () => new Promise(resolve => { finish = resolve; }), controller.signal);
    controller.abort();
    await assert.rejects(pending, /cancelled/);
    finish(new Response(JSON.stringify(fixture().plan)));
    await Promise.resolve();
  });
});
