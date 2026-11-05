import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { mock } from 'node:test';
import { workspace } from 'vscode';
import { LocalBrowserSourceEdits } from '../../localBrowserSourceEdits';
import { sha256 } from '../../sourceChanges';
import { LocalBrowserSourceEdit, LocalBrowserSourceEditRequestMessage, ProjectionPayload } from '@tensorviz/contracts/protocol';
import { editorState, reset, resetEditors, state } from '../vscodeStub';

const BEFORE = 'import torch.nn as nn\nclass Net(nn.Module):\n    def __init__(self):\n        self.fc = nn.Linear(8, 8)\n';
const AFTER = BEFORE.replace('8, 8', '8, 16');
const TARGET = { workspaceFolder: '/work', file: 'net.py', cls: 'Net', documentVersion: 1, sourceDigest: sha256(BEFORE), backendRevision: 1 };
const EDIT: LocalBrowserSourceEdit = { kind: 'set_param', nodeId: 'fc', param: 'out_features', value: 16 };

suite('local browser source review', () => {
  setup(() => { reset(); resetEditors(); });
  teardown(() => mock.restoreAll());

  function harness(review?: ConstructorParameters<typeof LocalBrowserSourceEdits>[0]['review'], _analytics?: unknown,
    draft?: ConstructorParameters<typeof LocalBrowserSourceEdits>[0]['draft'],
    accepted?: ConstructorParameters<typeof LocalBrowserSourceEdits>[0]['accepted']) {
    editorState.documents.set('/work/net.py', BEFORE);
    editorState.documents.set('/work/dep.py', '# dependency\n');
    if (accepted) {
      const apply = (workspace as any).applyEdit;
      mock.method(workspace as any, 'applyEdit', async (edit: any) => {
        const result = await apply(edit);
        if (result) {
          editorState.documents.set('/work/net.py', AFTER);
          editorState.versions.set('/work/net.py', 2);
        }
        return result;
      });
    }
    const requests: Array<{ url: string; body: any; headers: any }> = [];
    const replies: any[] = [];
    const logs: string[] = [];
    let done!: (value: any) => void;
    const result = new Promise<any>(resolve => { done = resolve; });
    let current = true;
    const context = {
      projection: { spanmap: { fc: { file: 'net.py', start: [4, 8], end: [4, 32] } } } as unknown as ProjectionPayload,
      dependencies: new Map([['net.py', sha256(BEFORE)], ['dep.py', sha256('# dependency\n')]]),
    };
    const host = new LocalBrowserSourceEdits({
      draft,
      accepted,
      origin: () => 'http://127.0.0.1:9901', token: () => 'private-launch-credential',
      context: target => current && JSON.stringify(target) === JSON.stringify(TARGET) ? context : undefined,
      publish: reply => { replies.push(reply); done(reply); }, channel: { appendLine(line) { logs.push(line); } },
      review: review ?? (async () => true),
      fetchImpl: (async (url, init) => {
        requests.push({ url: String(url), body: JSON.parse(String(init?.body)), headers: init?.headers });
        const payload = String(url).endsWith('/buffer-classes')
          ? { classes: [{ name: 'Net', line: 2, end_line: 4 }] }
          : { edits: [{ range: { start: [4, 0], end: [5, 0] }, newText: '        self.fc = nn.Linear(8, 16)\n' }], refused: [], spanmap: {}, source_digest: sha256(AFTER) };
        return { ok: true, status: 200, json: async () => payload, text: async () => JSON.stringify(payload) } as Response;
      }) as typeof fetch,
    });
    const message = (edit = EDIT): LocalBrowserSourceEditRequestMessage => ({
      type: 'sourceEditRequest', protocolVersion: 1, workspaceSessionId: 'session-1', messageId: 'm1', requestId: 'edit1', target: TARGET, edit,
    });
    return { host, result, requests, replies, logs, message, context, retarget: () => { current = false; } };
  }

  test('accepted typed source starts its follow-up once, after write and review cleanup', async () => {
    let calls = 0, draftClosed = false;
    const h = harness(undefined, undefined, () => ({ show() {}, dispose() { draftClosed = true; } }), async change => {
      calls++;
      assert.equal(h.host.reviewing, false);
      assert.equal(draftClosed, true);
      assert.equal(editorState.appliedEdits.length, 1);
      assert.equal(change.file, 'net.py'); assert.deepEqual(change.target, TARGET);
      assert.equal(change.current(), true);
      editorState.documents.set('/work/dep.py', '# changed dependency');
      assert.equal(change.current(), false, 'a later dependency change cannot execute unreviewed source');
      return 'Saved and opening the updated model.';
    });
    h.host.handle(h.message()); h.host.handle(h.message());
    assert.equal((await h.result).status, 'applied');
    assert.equal(calls, 1);
    assert.match(h.replies[0].detail, /Saved and opening/);
    h.host.dispose();
  });

  for (const decision of ['discard', 'stale', 'cancel'] as const) test(`automatic capture never follows ${decision}`, async () => {
    let calls = 0;
    const h = harness(async () => {
      if (decision === 'stale') h.retarget();
      if (decision === 'cancel') h.host.cancel();
      return decision !== 'discard';
    }, undefined, undefined, async () => { calls++; return 'Unexpected run'; });
    const outcome = await h.host.propose(h.message());
    assert.notEqual(outcome.status, 'applied'); assert.equal(calls, 0);
    h.host.dispose();
  });

  test('automatic capture failure preserves the confirmed source write', async () => {
    const h = harness(undefined, undefined, undefined, async () => { throw new Error('save failed'); });
    const result = await h.host.propose(h.message());
    assert.equal(result.status, 'applied');
    assert.equal(editorState.documents.get('/work/net.py'), AFTER);
    assert.match(result.detail, /automatic capture could not start/);
    h.host.dispose();
  });

  for (const decision of ['accept', 'discard', 'stale', 'cancel', 'error'] as const) test(`typed draft ${decision} uses frozen sources and stays open through a guarded write`, async () => {
    let shown = false, disposed = false;
    const h = harness(async (_title, artifact, _signal, _candidate, onShown) => {
      assert.equal(artifact.after, AFTER); assert.equal(editorState.appliedEdits.length, 0);
      if (decision === 'error') throw new Error('diff unavailable');
      return new Promise(resolve => onShown?.(async () => {}, resolve));
    }, undefined, (input, signal, current) => {
      assert.equal(input.id, 'edit1'); assert.equal(input.canRevise, false);
      assert.deepEqual(input.target, TARGET); assert.equal(input.artifacts[0].before, BEFORE); assert.equal(input.artifacts[0].after, AFTER);
      assert.deepEqual(input.sources.entries.map(source => source.source), [BEFORE, '# dependency\n']);
      assert.equal(current(), true); assert.equal(signal.aborted, false);
      return { show(_reveal, choose) {
        shown = true;
        if (decision === 'stale') h.retarget();
        if (decision === 'cancel') h.host.cancel();
        choose(decision !== 'discard');
      }, dispose() { disposed = true; assert.equal(editorState.appliedEdits.length, decision === 'accept' ? 1 : 0); } };
    });
    try {
      const result = await h.host.propose(h.message());
      assert.equal(shown, decision !== 'error');
      assert.equal(disposed, !['stale', 'cancel'].includes(decision));
      if (['stale', 'cancel'].includes(decision)) assert.match(state.warningMessages.join(' '), /review context changed/);
      assert.equal(result.status, decision === 'accept' ? 'applied' : ['discard', 'cancel'].includes(decision) ? 'cancelled' : 'refused');
      assert.equal(editorState.appliedEdits.length, decision === 'accept' ? 1 : 0);
    } finally { h.host.dispose(); }
  });

  for (const throws of [false, true]) test(`failed write keeps the draft and surfaces the reason; throws=${throws}`, async () => {
    let disposed = false, republished = false;
    const h = harness(undefined, undefined, () => ({ show() {}, notifyDecisionFailed() { republished = true; }, dispose() { disposed = true; } }));
    mock.method(workspace as any, 'applyEdit', async () => {
      assert.equal(disposed, false);
      if (throws) throw new Error('editor unavailable');
      return false;
    });
    const result = await h.host.propose(h.message());
    assert.equal(result.status, 'refused');
    assert.equal(disposed, false);
    assert.equal(republished, true);
    assert.match(state.warningMessages.join(' '), throws ? /did not confirm/ : /editor rejected/);
    assert.ok(h.logs.some(line => line.includes(result.detail)));
    h.host.dispose();
  });

  test('preview disposal cannot invalidate an accepted write', async () => {
    let disposed = false;
    const h = harness(undefined, undefined, () => ({ show() {}, dispose() {
      assert.equal(editorState.appliedEdits.length, 1);
      disposed = true;
      h.retarget();
    } }));
    const result = await h.host.propose(h.message());
    assert.equal(result.status, 'applied');
    assert.equal(disposed, true);
    h.host.dispose();
  });

  test('reviews exact source before one WorkspaceEdit and keeps credentials inside host requests', async () => {
    const h = harness(async (_title, artifact) => {
      assert.equal(editorState.appliedEdits.length, 0);
      assert.equal(editorState.documents.get('/work/net.py'), BEFORE);
      assert.equal(artifact.before, BEFORE);
      assert.equal(artifact.after, AFTER);
      return true;
    });
    h.host.handle(h.message());
    const result = await h.result;
    assert.equal(result.status, 'applied');
    assert.equal(editorState.appliedEdits.length, 1);
    assert.equal(h.requests[1].body.source, BEFORE);
    assert.equal(h.requests[1].body.ops[0].node_id, 'fc');
    assert.equal(h.requests[1].headers['X-tensorViz-Token'], 'private-launch-credential');
    assert.ok(!JSON.stringify(h.replies).includes('private-launch-credential'));
    assert.ok(!JSON.stringify(h.replies).includes(BEFORE));
  });

  test('the host API reports native review and outcome without inventing a browser reply', async () => {
    const h = harness(async () => false);
    const reviews: unknown[] = [];
    const result = await h.host.propose(h.message(), new AbortController().signal, review => reviews.push(review));
    assert.equal(result.status, 'cancelled'); assert.equal(h.replies.length, 0);
    assert.deepEqual(reviews, [{ title: 'tensorViz source change — net.py', file: 'net.py', beforeDigest: sha256(BEFORE), afterDigest: sha256(AFTER) }]);
    assert.equal(editorState.appliedEdits.length, 0);
  });

  test('host and browser share one review; an aborted host request cannot apply', async () => {
    let reviewStarted!: () => void, approve!: (value: boolean) => void;
    const ready = new Promise<void>(resolve => { reviewStarted = resolve; });
    const h = harness(async () => { reviewStarted(); return new Promise(resolve => { approve = resolve; }); });
    const controller = new AbortController();
    const pending = h.host.propose(h.message(), controller.signal);
    await ready;
    h.host.handle({ ...h.message(), requestId: 'other-browser' });
    assert.equal((await h.result).status, 'refused');
    const declaration = await h.host.proposeDeclaration({ type: 'projectDeclare', protocolVersion: 1,
      workspaceSessionId: 'session-1', messageId: 'project-edit', project: { workspaceFolder: '/work',
        projectRoot: '', experimentId: 'train', indexDigest: 'a'.repeat(64), indexRevision: 1 },
      declaration: 'experiment', args: ['train'] });
    assert.match(declaration.detail, /Another source review/);
    controller.abort(); approve(true);
    assert.notEqual((await pending).status, 'applied'); assert.equal(editorState.appliedEdits.length, 0);
  });

  test('discarding a review applies nothing', async () => {
    const h = harness(async () => false);
    h.host.handle(h.message());
    assert.equal((await h.result).status, 'cancelled');
    assert.equal(editorState.appliedEdits.length, 0);
  });

  test('the native review discloses other instances sharing the source declaration', async () => {
    const h = harness(async (title) => {
      assert.match(title, /also affects 1 other projected instance/);
      return false;
    });
    h.context.projection.spanmap['second.fc'] = h.context.projection.spanmap.fc;
    h.host.handle(h.message());
    assert.equal((await h.result).status, 'cancelled');
  });

  for (const staleReexport of [false, true]) {
    test(`F7 dotted member edits use the owning definition; stale reexport=${staleReexport}`, async () => {
      const fixture = path.resolve(__dirname, '../../../../backend/tests/fixtures/ide_dotted_repo');
      const root = fs.readFileSync(path.join(fixture, 'dotted_model.py'), 'utf8');
      const block = fs.readFileSync(path.join(fixture, 'blocks.py'), 'utf8');
      const after = block.replace('Linear(4, 4)', 'Linear(4, 8)');
      const reexport = 'from blocks import ScaleBlock\n';
      for (const [file, text] of [['dotted_model.py', root], ['blocks.py', block], ['api.py', reexport]]) {
        editorState.documents.set(`/work/${file}`, text);
      }
      const target = { ...TARGET, file: 'dotted_model.py', cls: 'DottedNet', sourceDigest: sha256(root) };
      const span = { file: 'blocks.py', start: [6, 8], end: [6, 35] };
      const requests: any[] = [];
      let reviewed = false;
      const host = new LocalBrowserSourceEdits({
        origin: () => 'http://127.0.0.1:9901', token: () => 'private', channel: { appendLine() {} }, publish() {},
        context: () => ({ dependencies: new Map([['dotted_model.py', sha256(root)], ['blocks.py', sha256(block)], ['api.py', sha256(reexport)]]),
          projection: { spanmap: { block: { file: 'dotted_model.py', start: [8, 8], end: [8, 40] },
            'block.proj': span, 'second.proj': span } } as unknown as ProjectionPayload }),
        fetchImpl: (async (url, init) => {
          const body = JSON.parse(String(init?.body)); requests.push(body);
          assert.equal(body.file, 'blocks.py'); assert.equal(body.source, block);
          const payload = String(url).endsWith('/buffer-classes')
            ? { classes: [{ name: 'ScaleBlock', line: 3, end_line: 9 }] }
            : { edits: [{ range: { start: [6, 0], end: [7, 0] }, newText: '        self.proj = nn.Linear(4, 8)\n' }],
              refused: [], spanmap: {}, source_digest: sha256(after) };
          return { ok: true, status: 200, json: async () => payload, text: async () => JSON.stringify(payload) } as Response;
        }) as typeof fetch,
        review: async (title, artifact) => {
          reviewed = true;
          assert.match(title, /blocks.py.*also affects 1 other projected instance/);
          assert.equal(artifact.before, block); assert.equal(artifact.after, after);
          if (staleReexport) editorState.documents.set('/work/api.py', '# changed binding\n');
          return true;
        },
      });
      const result = await host.propose({ requestId: 'f7', target,
        edit: { kind: 'set_param', nodeId: 'block.proj', param: 'out_features', value: 8 } });
      assert.equal(reviewed, true);
      assert.equal(requests[1].cls, 'ScaleBlock');
      assert.deepEqual(requests[1].spanmap, { proj: span });
      assert.equal(requests[1].ops[0].node_id, 'proj');
      assert.equal(result.status === 'applied', !staleReexport);
      assert.equal(editorState.documents.get('/work/dotted_model.py'), root);
      assert.equal(editorState.appliedEdits.length, staleReexport ? 0 : 1);
      if (!staleReexport) {
        // The VS Code stub records WorkspaceEdit requests without mutating text.
        const replacements = editorState.appliedEdits[0].replacements;
        assert.equal(replacements.length, 1);
        assert.equal(replacements[0].uri.fsPath, '/work/blocks.py');
        assert.equal(replacements[0].newText, '        self.proj = nn.Linear(4, 8)\n');
        assert.deepEqual([replacements[0].range.start.line, replacements[0].range.end.line], [5, 6]);
      }
    });
  }

  test('typed review applies after identical-byte document version drift', async () => {
    const h = harness(async () => {
      editorState.versions.set('/work/net.py', 7);
      return true;
    });
    assert.equal((await h.host.propose(h.message())).status, 'applied');
    assert.equal(editorState.appliedEdits.length, 1);
    h.host.dispose();
  });

  for (const change of ['dependency', 'retarget', 'cancel'] as const) {
    test(`refuses apply after ${change} changes during native review`, async () => {
      let reviewed = false;
      const h = harness(async () => {
        reviewed = true;
        if (change === 'dependency') editorState.documents.set('/work/dep.py', '# changed\n');
        if (change === 'retarget') h.retarget();
        if (change === 'cancel') h.host.cancel('edit1');
        return true;
      });
      h.host.handle(h.message());
      assert.notEqual((await h.result).status, 'applied');
      assert.equal(reviewed, true, 'the test must reach review before exercising the stale-source guard');
      assert.equal(editorState.appliedEdits.length, 0);
    });
  }

  test('a dirty dependency is refused before calling the backend', async () => {
    const h = harness();
    editorState.documents.set('/work/dep.py', '# unsaved\n');
    h.host.handle(h.message());
    assert.equal((await h.result).status, 'refused');
    assert.equal(h.requests.length, 0);
  });

  test('an unknown selection cannot choose an arbitrary file', async () => {
    const h = harness();
    h.host.handle(h.message({ ...EDIT, nodeId: '../elsewhere.py' }));
    assert.equal((await h.result).status, 'refused');
    assert.equal(h.requests.length, 0);
    assert.equal(editorState.appliedEdits.length, 0);
  });

  for (const semantic of [
    { metadata: { capture: { version: 1 } } },
    { nodes: [{ node_id: 'fc', metadata: { captureRef: { kind: 'invocation' } } }] },
  ]) {
    test(`captured navigation spans cannot enter legacy codemods: ${JSON.stringify(semantic)}`, async () => {
      const h = harness(async () => { assert.fail('capture must not open legacy review'); });
      h.context.projection.document = { semantic };
      for (const edit of [EDIT, { kind: 'remove_layer', nodeId: 'fc' },
        { kind: 'add_layer', afterNodeId: 'fc', nodeId: 'activation', layerType: 'ReLU', params: {} },
        { kind: 'group_layers', nodeIds: ['fc', 'act'], name: 'features' }] as LocalBrowserSourceEdit[]) {
        const result = await h.host.propose({ requestId: edit.kind, target: TARGET, edit });
        assert.equal(result.status, 'refused');
        assert.match(result.detail, /explicit source-edit capability/);
      }
      assert.equal(h.requests.length, 0);
      assert.equal(editorState.appliedEdits.length, 0);
    });
  }

  for (const edit of [EDIT, { kind: 'remove_layer', nodeId: 'fc' },
    { kind: 'add_layer', afterNodeId: 'fc', nodeId: 'drop', layerType: 'dropout', params: { p: 0.5 } }] as LocalBrowserSourceEdit[]) {
    test(`captured exact constructor can plan and review ${edit.kind}`, async () => {
      const h = harness();
      const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../../tests/fixtures/contracts/captured-constructor-edit.json'), 'utf8'));
      const capability = structuredClone(fixture.document.semantic.nodes.flatMap((node: any) => node.metadata.sourceEditCapabilities || [])
        .find((cap: any) => cap.command === 'set_constructor_literal'));
      capability.expression.anchors = [{ ...capability.expression.anchors[0], file: 'net.py', digest: sha256(BEFORE),
        startLine: 4, startColumn: 18, endLine: 4, endColumn: 33 }];
      h.context.projection.document = { semantic: { metadata: { capture: {} }, nodes: [
        { node_id: 'fc', metadata: { captureRef: capability.target, sourceEditCapabilities: [capability] } },
      ] } } as any;
      const result = await h.host.propose({ requestId: edit.kind, target: TARGET, edit });
      assert.equal(result.status, 'applied', result.detail);
      assert.equal(h.requests.length, 2);
      assert.equal(editorState.appliedEdits.length, 1);
      const operations = h.requests[1].body.operations;
      assert.ok(operations || h.requests[1].body.ops);
    });
  }

  test('untrusted workspaces cannot prepare source edits', async () => {
    const h = harness();
    state.isTrusted = false;
    h.host.handle(h.message());
    assert.equal((await h.result).status, 'refused');
    assert.equal(h.requests.length, 0);
  });
});
