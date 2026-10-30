import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as fs from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { AppliedSourceReceipts } from '../../appliedSourceReceipts';
import { AppliedSourceInspector, readAppliedSource, appliedCaptureGuard, type AppliedCapture, type AppliedSourceObservation } from '../../appliedSourceInspection';
import { sha256 } from '../../sourceChanges';
import { isAppliedSourceObservation } from '@tensorviz/contracts/protocol';
import { editorState, reset, resetEditors } from '../vscodeStub';

const fixture = () => JSON.parse(readFileSync(join(__dirname, '../../../../tests/fixtures/contracts/source-reconciliation.json'), 'utf8'));
const target = { workspaceFolder: '/work', file: 'model.py', cls: 'Model', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
type Options = ConstructorParameters<typeof AppliedSourceInspector>[1];
const tick = () => new Promise<void>(resolve => setImmediate(resolve));

function harness() {
  const data = fixture(), { current, ...evidence } = data.request;
  const store = new AppliedSourceReceipts(undefined, () => 10), reservation = store.reserve('review', target, evidence)!;
  store.confirm('review', reservation);
  const state = { source: 'reviewed_saved' as AppliedSourceObservation['source'], editor: true, resources: true,
    trusted: true, token: 'HOST_ONLY', live: { capture: current, startedAt: 20 } as AppliedCapture | undefined,
    setup: true as boolean | undefined, reads: 0, calls: 0, checks: 0 };
  const options: Options = {
    origin: () => 'http://localhost:9000', token: () => state.token, trusted: () => state.trusted,
    live: () => state.live, readSource: async () => { state.reads++; return { source: state.source, current: () => state.editor }; },
    captureGuard: async () => () => state.resources,
    checkSetup: async () => { state.checks++; return state.setup; },
    fetchImpl: async (_url, init) => {
      state.calls++; assert.deepEqual(JSON.parse(init!.body as string), data.request);
      return new Response(JSON.stringify(data.result));
    },
  };
  const inspector = new AppliedSourceInspector(store, options);
  return { data, store, state, options, inspector, inspect: () => inspector.inspect('review') };
}

suite('applied-source live inspection', () => {
  test('checks exact reviewed bytes and real capture witnesses independently of candidate history', async () => {
    const h = harness(), result = await h.inspect();
    assert.equal(result.source, 'reviewed_saved'); assert.equal(result.captureFreshness, 'current');
    assert.equal(isAppliedSourceObservation(result), true);
    assert.deepEqual(result.reconciliation, h.data.result); assert.equal(result.unavailable, null);
    assert.equal(h.state.checks, 2); assert.equal(h.state.calls, 1);
    assert.ok(!JSON.stringify(result).includes(h.data.request.reviewedSource));
  });

  test('unconfirmed or forgotten source cannot request a reconciliation', async () => {
    const h = harness(); h.store.forget('review');
    await assert.rejects(h.inspect(), /confirmed Apply receipt/);
    assert.equal(h.state.reads, 0); assert.equal(h.state.calls, 0);
  });

  for (const source of ['reviewed_unsaved', 'reverted', 'changed', 'unavailable'] as const) {
    test(`${source} does not borrow a live capture or run a model`, async () => {
      const h = harness(); h.state.source = source;
      const result = await h.inspect();
      assert.equal(result.source, source); assert.equal(result.captureFreshness, 'stale');
      assert.equal(result.reconciliation, null); assert.notEqual(result.unavailable, null);
      assert.equal(isAppliedSourceObservation(result), true);
      assert.equal(h.state.calls, 0); assert.equal(h.state.checks, 0);
    });
  }

  test('save, restore and redo are observed anew without sticky success or automatic execution', async () => {
    const h = harness(); h.state.source = 'reviewed_unsaved';
    assert.equal((await h.inspect()).unavailable, 'source_not_saved');
    h.state.source = 'reviewed_saved'; assert.equal((await h.inspect()).reconciliation?.postconditions?.status, 'verified');
    h.state.source = 'reverted'; assert.equal((await h.inspect()).reconciliation, null);
    h.state.source = 'reviewed_saved'; assert.equal((await h.inspect()).captureFreshness, 'current');
    assert.equal(h.state.calls, 2);
  });

  for (const missing of ['none', 'old', 'base']) test(`${missing} later capture cannot certify an applied edit`, async () => {
    const h = harness();
    if (missing === 'none') h.state.live = undefined;
    else if (missing === 'old') h.state.live!.startedAt = 1;
    else h.state.live!.capture = h.data.request.base;
    assert.equal((await h.inspect()).unavailable, 'capture_unavailable'); assert.equal(h.state.calls, 0);
  });

  test('without retained setup, observed semantics remain distinct from current environment certification', async () => {
    const h = harness(); h.state.setup = undefined;
    const result = await h.inspect();
    assert.equal(result.captureFreshness, 'unavailable'); assert.equal(result.reconciliation?.postconditions?.status, 'verified');
  });

  for (const change of ['editor', 'resources', 'trusted', 'token', 'receipt', 'capture', 'setup']) {
    test(`${change} changing during the backend check cannot publish a current result`, async () => {
      const h = harness();
      h.options.fetchImpl = async () => {
        if (change === 'editor') h.state.editor = false;
        if (change === 'resources') h.state.resources = false;
        if (change === 'trusted') h.state.trusted = false;
        if (change === 'token') h.state.token = 'replacement';
        if (change === 'receipt') h.store.forget('review');
        if (change === 'capture') h.state.live = undefined;
        if (change === 'setup') h.state.setup = false;
        return new Response(JSON.stringify(h.data.result));
      };
      const result = await h.inspect();
      assert.equal(result.captureFreshness, 'stale'); assert.equal(result.reconciliation, null);
      assert.equal(result.unavailable, 'check_stale');
      if (change === 'editor') assert.equal(result.source, 'unavailable');
    });
  }

  test('changed setup before the request prevents sending private evidence', async () => {
    const h = harness(); h.state.setup = false;
    assert.equal((await h.inspect()).unavailable, 'check_stale'); assert.equal(h.state.calls, 0);
  });

  test('each asynchronous preparation is followed by a freshness check', async () => {
    for (const step of ['source', 'resources', 'setup']) {
      const h = harness();
      if (step === 'source') h.options.readSource = async () => { h.state.editor = false; return { source: 'reviewed_saved', current: () => false }; };
      if (step === 'resources') h.options.captureGuard = async () => { h.state.editor = false; return () => true; };
      if (step === 'setup') h.options.checkSetup = async () => { h.state.resources = false; return true; };
      assert.equal((await h.inspect()).unavailable, 'check_stale'); assert.equal(h.state.calls, 0);
    }
  });

  test('newer reads cancel stalled reads; late completion cannot replace newer evidence', async () => {
    const h = harness(); let release!: () => void, entered = false;
    h.options.readSource = async () => {
      entered = true; await new Promise<void>(resolve => { release = resolve; });
      return { source: 'reviewed_saved', current: () => true };
    };
    const older = h.inspect(); await tick(); assert.equal(entered, true);
    h.options.readSource = async () => ({ source: 'reverted', current: () => true });
    const newer = await h.inspect(); assert.equal(newer.source, 'reverted');
    const result = await older; assert.equal(result.unavailable, 'check_stale');
    release(); await tick(); assert.equal(h.state.calls, 0);
  });

  test('owner cancellation ends a stalled read even if an adapter ignores abort', async () => {
    const h = harness(); h.options.readSource = async () => new Promise(() => {});
    const pending = h.inspect(); h.store.forget('review'); h.inspector.cancel('review');
    assert.equal((await pending).unavailable, 'check_stale'); assert.equal(h.state.calls, 0);
  });

  test('caller cancellation stops only its own inspection and does not affect a newer read', async () => {
    const h = harness(), signal = new AbortController();
    h.options.readSource = async () => new Promise(() => {});
    const first = h.inspector.inspect('review', signal.signal);
    h.options.readSource = async () => ({ source: 'reverted', current: () => true });
    const second = h.inspect(); signal.abort();
    assert.equal((await first).unavailable, 'check_stale'); assert.equal((await second).source, 'reverted');
    const reads = h.state.reads;
    await assert.rejects(h.inspector.inspect('review', signal.signal), /cancelled/);
    assert.equal(h.state.reads, reads);
  });

  test('private failures and evicted bodies stay explicit without source or arbitrary errors', async () => {
    const h = harness(); h.options.captureGuard = async () => { throw new Error('PRIVATE_FILE_CONTENT'); };
    const result = await h.inspect(); assert.equal(result.unavailable, 'check_failed');
    assert.ok(!JSON.stringify(result).includes('PRIVATE_FILE_CONTENT'));
    const limited = new AppliedSourceReceipts({ records: 32, evidenceBytes: 1, totalBytes: 1, ttlMs: 1000 }, () => 10);
    const { current: _current, ...evidence } = h.data.request;
    const id = limited.reserve('review', target, evidence)!; limited.confirm('review', id);
    const missing = await new AppliedSourceInspector(limited, h.options).inspect('review');
    assert.equal(missing.unavailable, 'evidence_unavailable'); assert.equal(missing.receipt.unavailable, 'too_large');
  });
});

suite('applied-source editor and dependency guards', () => {
  let root: string;
  setup(async () => { reset(); resetEditors(); root = await fs.realpath(await fs.mkdtemp(join(tmpdir(), 'tensorviz-applied-source-'))); });
  teardown(async () => { resetEditors(); reset(); await fs.rm(root, { recursive: true, force: true }); });

  test('buffer-only Apply, Save, other edits and source restoration have distinct states', async () => {
    const file = join(root, 'model.py'), before = '# before\n', after = '# after\n';
    await fs.writeFile(file, before); editorState.documents.set(file, after); editorState.dirtyDocuments.add(file);
    const receipt = { appliedAt: 1, file: 'model.py', beforeDigest: sha256(before), afterDigest: sha256(after), unavailable: null };
    const inspect = () => readAppliedSource({ ...target, workspaceFolder: root }, receipt, new AbortController().signal);
    let observed = await inspect(); assert.equal(observed.source, 'reviewed_unsaved'); assert.equal(observed.current(), true);
    await fs.writeFile(file, after); editorState.dirtyDocuments.delete(file);
    assert.equal(observed.current(), false);
    observed = await inspect(); assert.equal(observed.source, 'reviewed_saved'); assert.equal(observed.current(), true);
    editorState.documents.set(file, '# other\n'); editorState.versions.set(file, 2);
    assert.equal(observed.current(), false); assert.equal((await inspect()).source, 'changed');
    editorState.documents.set(file, before); assert.equal((await inspect()).source, 'reverted');
    editorState.documents.set(file, after); await fs.writeFile(file, '# external change');
    await assert.rejects(inspect(), /tracked resource/);
  });

  test('binary/helper changes and unsaved dependencies invalidate the whole capture without opening binary text', async () => {
    const files = [{ path: 'helper.py', digest: sha256('# helper\n') }, { path: 'weights.bin', digest: sha256('binary') }];
    await fs.writeFile(join(root, 'helper.py'), '# helper\n'); await fs.writeFile(join(root, 'weights.bin'), 'binary');
    editorState.documents.set(join(root, 'helper.py'), '# helper\n');
    const live = { capture: { source: { files } }, startedAt: 2 } as AppliedCapture;
    const guard = await appliedCaptureGuard({ ...target, workspaceFolder: root }, live, new AbortController().signal);
    assert.equal(guard(), true); assert.equal(editorState.documents.has(join(root, 'weights.bin')), false);
    editorState.dirtyDocuments.add(join(root, 'helper.py')); assert.equal(guard(), false);
    await assert.rejects(appliedCaptureGuard({ ...target, workspaceFolder: root }, live, new AbortController().signal));
    editorState.dirtyDocuments.clear(); await fs.writeFile(join(root, 'weights.bin'), 'changed'); assert.equal(guard(), false);
  });

  test('a source set requires every reviewed file saved; mixed restoration never certifies the original set', async () => {
    const files = ['model.py', 'config.py'].map(file => ({ file, beforeDigest: sha256('# before\n'), afterDigest: sha256('# after\n') }));
    for (const file of files) { await fs.writeFile(join(root, file.file), '# after\n'); editorState.documents.set(join(root, file.file), '# after\n'); }
    const receipt = { ...files[0], files, appliedAt: 1, unavailable: null };
    const inspect = () => readAppliedSource({ ...target, workspaceFolder: root }, receipt, new AbortController().signal);
    let observed = await inspect(); assert.equal(observed.source, 'reviewed_saved'); assert.equal(observed.current(), true);
    editorState.dirtyDocuments.add(join(root, 'config.py')); assert.equal(observed.current(), false);
    assert.equal((await inspect()).source, 'reviewed_unsaved'); editorState.dirtyDocuments.clear();
    editorState.documents.set(join(root, 'config.py'), '# before\n'); assert.equal((await inspect()).source, 'changed');
    editorState.documents.set(join(root, 'model.py'), '# before\n'); assert.equal((await inspect()).source, 'reverted');
    for (const file of files) editorState.documents.set(join(root, file.file), '# after\n');
    observed = await inspect(); assert.equal(observed.source, 'reviewed_saved');
    await fs.writeFile(join(root, 'config.py'), '# other\n'); assert.equal(observed.current(), false);
    await assert.rejects(inspect(), /tracked resource/);
  });
});
