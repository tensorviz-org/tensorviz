import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { McpGraphService, type McpModel } from '@tensorviz/graph-host/mcpGraph';
import type { ProjectionPayload } from '@tensorviz/contracts/protocol';
import { LocalBrowserSourceEdits, type AgentCodeEdit, type SourceEditsOptions } from '../../localBrowserSourceEdits';
import { McpProposals, registerCanvasCodeTools, registerMcpProposalTools } from '../../mcpProposals';
import { GraphMcpServer } from '../../mcpServer';
import { sha256 } from '../../sourceChanges';
import { editorState, reset, resetEditors, state } from '../vscodeStub';

const BEFORE = 'from torchvision.models.resnet import ResNet, Bottleneck\n\nclass ResNet152(ResNet):\n    def __init__(self):\n        super().__init__(block=Bottleneck, layers=[3, 8, 36, 3])\n';
const AFTER = 'import torch.nn as nn\n' + BEFORE + '        self.fc = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), self.fc)\n';
const tick = () => new Promise<void>(resolve => setImmediate(resolve));

suite('Canvas agent normal code proposals', () => {
  let root: string, host: LocalBrowserSourceEdits, target: McpModel['target'], model: McpModel;
  let review: NonNullable<SourceEditsOptions['review']>, syntaxOk: boolean, syntaxStatus: number, requests: any[];
  let draft: SourceEditsOptions['draft'];
  const edit = (): AgentCodeEdit => ({ kind: 'replace_model_source', beforeDigest: sha256(BEFORE), source: AFTER });
  const propose = (signal?: AbortSignal) => host.proposeCode({ requestId: 'canvas-code', target, edit: edit() }, signal);
  setup(() => {
    reset(); resetEditors();
    root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-code-test-')));
    fs.writeFileSync(path.join(root, 'vision_models.py'), BEFORE);
    editorState.documents.set(path.join(root, 'vision_models.py'), BEFORE);
    target = { workspaceFolder: root, file: 'vision_models.py', cls: 'ResNet152', documentVersion: 1,
      sourceDigest: sha256(BEFORE), backendRevision: 1 };
    // A real captured shape: fc has no typed capability and no local declaration span.
    model = { target, current: true, projection: { document: { semantic: { metadata: { capture: {} },
      nodes: [{ node_id: 'fc', metadata: { captureRef: { kind: 'invocation' } } }], edges: [] } }, spanmap: {} } as unknown as ProjectionPayload };
    review = async () => false; syntaxOk = true; syntaxStatus = 400; requests = []; draft = undefined;
    host = new LocalBrowserSourceEdits({ origin: () => 'http://127.0.0.1:9901', token: () => 'private',
      context: value => model.current && JSON.stringify(value) === JSON.stringify(target)
        ? { projection: model.projection, dependencies: new Map([['vision_models.py', target.sourceDigest]]) } : undefined,
      channel: { appendLine() {} }, publish() {}, review: (...args) => review(...args),
      draft: (...args) => draft?.(...args) ?? { show() {}, dispose() {} },
      fetchImpl: (async (_url, init) => {
        requests.push(JSON.parse(String(init?.body)));
        return { ok: syntaxOk, status: syntaxOk ? 200 : syntaxStatus, json: async () => syntaxOk
          ? { classes: [{ name: 'ResNet152' }] }
          : { detail: { error: 'syntax_error', message: 'invalid syntax', line: 7, column: 8 } } } as Response;
      }) as typeof fetch,
    });
  });
  teardown(() => { host.dispose(); fs.rmSync(root, { recursive: true, force: true }); });

  test('a human follow-up preserves the unaccepted source, checks syntax before replacement, and Accept writes both changes', async () => {
    const proposals = new McpProposals(host), graph = new McpGraphService('canvas-test', () => model);
    const artifacts: string[] = [];
    let decide!: (approved: boolean) => void;
    review = async (_title, artifact, signal) => {
      artifacts.push(artifact.after);
      return new Promise(resolve => {
        decide = resolve; signal.addEventListener('abort', () => resolve(false), { once: true });
      });
    };
    const ref = graph.reference(model);
    try {
      const first = await proposals.startCode('canvas-owner', graph, ref, edit()); await tick();
      const id = String(first.proposalId), requestId = `mcp-${id}`;
      await assert.rejects(proposals.startCode('canvas-owner', graph, ref, edit()), /Finish or discard/);
      assert.throws(() => proposals.beginRevision('wrong-id', new AbortController().signal), /no longer available/);
      const turn = new AbortController();
      const context = proposals.beginRevision(requestId, turn.signal);
      assert.equal(context.source, AFTER);
      const read = await proposals.readSource(graph, ref, 'canvas-owner');
      assert.equal(read.source, BEFORE); assert.equal(read.pendingDraft?.source, AFTER);
      assert.equal((await proposals.readSource(graph, ref, 'other-owner')).pendingDraft, undefined);
      const revised = { ...edit(), source: AFTER + '        self.activation = nn.Sigmoid()\n' };
      syntaxOk = false;
      await assert.rejects(proposals.startCode('canvas-owner', graph, ref, revised), /syntax error/);
      assert.equal(proposals.get('canvas-owner', id).status, 'review');
      assert.deepEqual(artifacts, [AFTER], 'invalid revisions leave the usable draft and native review open');
      assert.equal(editorState.appliedEdits.length, 0);
      syntaxOk = true;
      const second = await proposals.startCode('canvas-owner', graph, ref, revised); await tick();
      assert.equal(proposals.get('canvas-owner', id).status, 'cancelled');
      assert.deepEqual(artifacts, [AFTER, revised.source]);
      assert.equal(editorState.appliedEdits.length, 0, 'a revision never applies the prior draft');
      assert.equal(fs.readFileSync(path.join(root, 'vision_models.py'), 'utf8'), BEFORE);
      decide(true); await tick();
      assert.equal(proposals.get('canvas-owner', String(second.proposalId)).status, 'applied');
      assert.equal(editorState.appliedEdits[0].replacements[0].newText, revised.source);
    } finally { proposals.clear(); await tick(); }
  });

  test('a finished or cancelled agent turn cannot supersede an open draft later', async () => {
    const proposals = new McpProposals(host), graph = new McpGraphService('canvas-test', () => model);
    review = async (_title, _artifact, signal) => new Promise(resolve => signal.addEventListener('abort', () => resolve(false), { once: true }));
    const ref = graph.reference(model);
    try {
      const first = await proposals.startCode('canvas-owner', graph, ref, edit()); await tick();
      const turn = new AbortController(); proposals.beginRevision(`mcp-${first.proposalId}`, turn.signal);
      proposals.endRevision(turn.signal);
      await assert.rejects(proposals.startCode('canvas-owner', graph, ref, edit()), /Finish or discard/);
      proposals.beginRevision(`mcp-${first.proposalId}`, turn.signal); turn.abort();
      await assert.rejects(proposals.startCode('canvas-owner', graph, ref, edit()), /Finish or discard/);
      assert.equal(proposals.get('canvas-owner', String(first.proposalId)).status, 'review');
      assert.equal(editorState.appliedEdits.length, 0);
    } finally { proposals.clear(); await tick(); }
  });

  for (const approved of [true, false]) test(`a human ${approved ? 'Accept' : 'Discard'} wins over an in-flight revision before review cleanup finishes`, async () => {
    const proposals = new McpProposals(host), graph = new McpGraphService('canvas-test', () => model);
    let decided!: () => void, closeReview!: (approved: boolean) => void, checked!: () => void;
    review = async (_title, _artifact, _signal, _candidate, _shown, onDecision) => {
      decided = onDecision!;
      return new Promise(resolve => { closeReview = resolve; });
    };
    const ref = graph.reference(model);
    try {
      const first = await proposals.startCode('canvas-owner', graph, ref, edit()); await tick();
      proposals.beginRevision(`mcp-${first.proposalId}`, new AbortController().signal);
      host.checkCode = async () => new Promise(resolve => { checked = resolve; });
      const revision = proposals.startCode('canvas-owner', graph, ref, { ...edit(), source: AFTER + '# revised\n' });
      decided(); // Native review has decided, but its asynchronous cleanup has not returned.
      checked();
      await assert.rejects(revision, /accepted, discarded or cancelled/);
      assert.equal((await proposals.readSource(graph, ref, 'canvas-owner')).pendingDraft, undefined);
      assert.throws(() => proposals.beginRevision(`mcp-${first.proposalId}`, new AbortController().signal), /no longer available/);
      closeReview(approved); await tick();
      assert.equal(proposals.get('canvas-owner', String(first.proposalId)).status, approved ? 'applied' : 'cancelled');
      assert.equal(editorState.appliedEdits.length, approved ? 1 : 0);
      if (approved) assert.equal(editorState.appliedEdits[0].replacements[0].newText, AFTER);
    } finally { closeReview?.(false); proposals.clear(); await tick(); }
  });

  test('the draft uses the exact review artifact and closes before Apply, Discard, or a failed review', async () => {
    for (const decision of [true, false, 'error'] as const) {
      resetEditors(); editorState.documents.set(path.join(root, 'vision_models.py'), BEFORE);
      let shown = false, disposed = false;
      draft = (input, signal, current) => {
        assert.equal(input.artifact.after, AFTER); assert.equal(input.sources.entries[0].source, BEFORE);
        assert.equal(signal.aborted, false); assert.ok(current());
        return { show() { shown = true; }, dispose() { disposed = true; } };
      };
      review = async (_title, _artifact, _signal, _candidate, onShown) => {
        onShown?.(async () => {}, () => {}); assert.ok(shown); assert.equal(disposed, false);
        assert.equal(editorState.appliedEdits.length, 0);
        if (decision === 'error') throw new Error('diff closed unexpectedly');
        return decision;
      };
      const result = await propose();
      assert.ok(disposed);
      assert.equal(result.status, decision === true ? 'applied' : decision === false ? 'cancelled' : 'refused');
    }
  });

  test('inherited captured fc can propose its workspace subclass with no writes before Apply', async () => {
    const source = await host.readModelSource(target);
    assert.equal(source.source, BEFORE); assert.equal(source.digest, sha256(BEFORE));
    review = async (title, artifact) => {
      assert.match(title, /syntax checked, runtime not run/);
      assert.equal(artifact.path, 'vision_models.py'); assert.equal(artifact.before, BEFORE); assert.equal(artifact.after, AFTER);
      assert.equal(editorState.appliedEdits.length, 0);
      assert.equal(fs.readFileSync(path.join(root, 'vision_models.py'), 'utf8'), BEFORE);
      return true;
    };
    assert.equal((await propose()).status, 'applied');
    assert.equal(editorState.appliedEdits.length, 1);
    assert.equal(editorState.appliedEdits[0].replacements[0].newText, AFTER);
    assert.equal(fs.readFileSync(path.join(root, 'vision_models.py'), 'utf8'), BEFORE, 'Apply remains unsaved');
    assert.deepEqual(requests, [{ file: 'vision_models.py', source: AFTER }], 'only AST syntax inspection, never model execution');
  });

  test('Discard and syntax failure write nothing', async () => {
    assert.equal((await propose()).status, 'cancelled');
    syntaxOk = false; let shown = false; review = async () => { shown = true; return true; };
    const result = await propose();
    assert.equal(result.status, 'refused'); assert.match(result.detail, /line 7:9: invalid syntax/); assert.equal(shown, false);
    syntaxStatus = 500;
    const unavailable = await propose();
    assert.match(unavailable.detail, /service is unavailable/);
    assert.doesNotMatch(unavailable.detail, /Correct it|syntax error/);
    assert.equal(editorState.appliedEdits.length, 0);
  });

  for (const mutation of ['bytes', 'version', 'trust', 'target', 'cancel', 'symlink', 'replacement'] as const) {
    test(`review cannot apply after ${mutation} changes`, async () => {
      const controller = new AbortController();
      review = async () => {
        const file = path.join(root, 'vision_models.py');
        if (mutation === 'bytes') editorState.documents.set(file, BEFORE + '# concurrent edit\n');
        if (mutation === 'version') editorState.versions.set(file, 2);
        if (mutation === 'trust') state.isTrusted = false;
        if (mutation === 'target') model.current = false;
        if (mutation === 'cancel') controller.abort();
        if (mutation === 'symlink' || mutation === 'replacement') {
          fs.renameSync(file, path.join(root, 'original.py'));
          if (mutation === 'symlink') fs.symlinkSync(path.join(root, 'original.py'), file);
          else fs.writeFileSync(file, BEFORE);
        }
        return true;
      };
      assert.notEqual((await propose(controller.signal)).status, 'applied');
      assert.equal(editorState.appliedEdits.length, 0);
    });
  }

  test('stale digest, oversized source, path traversal and symlinks cannot open a review', async () => {
    review = async () => { assert.fail('unsafe request reached native review'); };
    assert.equal((await host.proposeCode({ requestId: 'x', target, edit: { ...edit(), beforeDigest: 'a'.repeat(64) } })).status, 'refused');
    assert.equal((await host.proposeCode({ requestId: 'x', target, edit: { ...edit(), source: 'x'.repeat(262145) } })).status, 'refused');
    target.file = '../outside.py'; await assert.rejects(host.readModelSource(target));
    target.file = 'vision_models.py';
    fs.renameSync(path.join(root, target.file), path.join(root, 'original.py'));
    fs.symlinkSync(path.join(root, 'original.py'), path.join(root, target.file));
    await assert.rejects(host.readModelSource(target), /without symbolic links/);
    assert.equal(editorState.appliedEdits.length, 0);
  });

  test('normal code and typed graph edits share one native review reservation', async () => {
    let shown!: () => void, finish!: (value: boolean) => void;
    const ready = new Promise<void>(resolve => { shown = resolve; });
    review = async () => { shown(); return new Promise(resolve => { finish = resolve; }); };
    const pending = propose(); await ready;
    const result = await host.propose({ requestId: 'typed', target, edit: { kind: 'remove_layer', nodeId: 'fc' } });
    assert.match(result.detail, /Another source review/);
    assert.equal((await propose()).status, 'refused');
    finish(false); await pending;
  });

  test('real MCP fallback survives provider turn completion and is owner-scoped and cancellable', async () => {
    const proposals = new McpProposals(host);
    const graph = new McpGraphService('canvas-test', () => model);
    const endpoint = new GraphMcpServer(server => {
      registerMcpProposalTools(server, graph, proposals, 'canvas-owner');
      registerCanvasCodeTools(server, graph, proposals, 'canvas-owner');
    });
    let abort!: AbortSignal, finish!: (approved: boolean) => void;
    review = async (_title, _artifact, signal) => { abort = signal; return new Promise(resolve => { finish = resolve; }); };
    const client = new Client({ name: 'canvas-code-test', version: '1' });
    try {
      const { url } = await endpoint.start();
      await client.connect(new StreamableHTTPClientTransport(new URL(url)));
      const ref = graph.reference(model);
      const read = await client.callTool({ name: 'read_model_source', arguments: ref });
      assert.equal((read.structuredContent as { source: string }).source, BEFORE);
      const call = await client.callTool({ name: 'propose_code_edit', arguments: { ...ref, beforeDigest: sha256(BEFORE), source: AFTER } });
      assert.ok(!call.isError);
      const id = String((call.structuredContent as { proposalId: string }).proposalId);
      assert.ok(!JSON.stringify(call).includes('nn.Sequential'), 'receipts do not duplicate source bodies');
      for (let i = 0; i < 50 && !abort; i++) await tick();
      assert.equal(proposals.get('canvas-owner', id).status, 'review');
      assert.throws(() => proposals.get('other-owner', id), /unavailable/);
      await client.close(); await endpoint.close();
      assert.equal(abort.aborted, false, 'provider exit must not discard the user’s pending review');
      proposals.cancelOwner('canvas-owner'); assert.equal(abort.aborted, true);
      finish(true); await tick();
      assert.equal(editorState.appliedEdits.length, 0);
      assert.throws(() => proposals.get('canvas-owner', id), /unavailable/);
    } finally { finish?.(false); proposals.clear(); await client.close(); await endpoint.close(); }
  });
});
