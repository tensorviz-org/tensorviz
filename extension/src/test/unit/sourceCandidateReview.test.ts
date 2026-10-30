import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as path from 'node:path';
import { LocalBrowserSourceEdits, type SourceProposalReview, type SourceEditsOptions } from '../../localBrowserSourceEdits';
import type { CandidateRunResult } from '@tensorviz/graph-host/candidateExecution';
import type { CandidateExecution, ResolvedExecutionPlan } from '../../executionPlan';
import { sha256 } from '../../sourceChanges';
import { reviewSource, reviewSources } from '../../sourceReview';
import type { ProjectionPayload } from '@tensorviz/contracts/protocol';
import { editorState, reset, resetEditors } from '../vscodeStub';
import { candidateCheckpointFixture } from '../candidateCheckpointFixture';
const vscode = require('../vscodeStub');

async function until(check: () => boolean) {
  for (let n = 0; n < 100 && !check(); n++) await new Promise(resolve => setImmediate(resolve));
  assert.ok(check(), 'expected the native review to reach this state');
}

suite('native captured candidate controls', () => {
  let restore: Array<() => void>, data: any, tabs: any[], opened: any[], diffs: any[];
  let commands: Map<string, () => any>, providers: Map<string, any>, contexts: Map<string, boolean>;
  let reviews: SourceProposalReview[], calls: any[], logs: string[], runner: NonNullable<SourceEditsOptions['runCandidate']>;
  let plan: ResolvedExecutionPlan | undefined;
  let batch: boolean, onDiff: (() => void) | undefined, closedTabs: ((event: { closed: any[] }) => void) | undefined;
  let draftShown: boolean, draftDisposed: boolean;
  const activate = (document: any) => {
    const editor = { document }; (editorState.visibleTextEditors as any[]).push(editor); return editor;
  };
  const target = { workspaceFolder: '/work', file: 'model.py', cls: 'Model', documentVersion: 1, sourceDigest: '', backendRevision: 1 };
  function patch(object: any, key: string, value: any) {
    const descriptor = Object.getOwnPropertyDescriptor(object, key);
    Object.defineProperty(object, key, { configurable: true, writable: true, value });
    restore.push(() => descriptor ? Object.defineProperty(object, key, descriptor) : delete object[key]);
  }
  function result(): CandidateRunResult {
    return { outcome: { ok: true, executionCurrent: true, recheckExecution: async () => true, exitCode: 0,
      envelope: { protocol_version: 1, result: {}, error: null, spans: {}, worker: {} } },
      evidence: { execution: { id: 'candidate:native' } as CandidateExecution,
        capture: structuredClone(data.capture), baseCurrent: true, candidateCurrent: true } };
  }
  const command = (name: string) => commands.get(`tensorviz.${name}`)!();
  async function open(onReview = (review: SourceProposalReview) => { reviews.push(review); }) {
    const node = data.document.semantic.nodes.find((item: any) => item.metadata.captureRef.semanticId === data.intent.target.semanticId);
    const host = new LocalBrowserSourceEdits({ origin: () => 'http://localhost:9000', token: () => 'HOST_ONLY',
      context: () => ({ projection: { document: data.document, spanmap: data.spans } as ProjectionPayload,
        dependencies: new Map([['model.py', target.sourceDigest]]) }),
      capture: () => data.capture, capturePlan: () => plan,
      draft: input => {
        assert.equal(input.canRevise, false); assert.equal(input.artifacts[0].after, data.after);
        return { show() { draftShown = true; }, dispose() { draftDisposed = true; } };
      },
      runCandidate: async (request, signal) => { calls.push(request); return runner(request, signal); },
      publish() {}, channel: { appendLine: line => { logs.push(line); } }, fetchImpl: async () => new Response(JSON.stringify(data.plan)),
    });
    const pending = host.propose({ requestId: 'native', target, edit: { ...data.intent, nodeId: node.node_id } },
      undefined, onReview);
    await until(() => diffs.length === 1);
    return { pending, host };
  }

  setup(() => {
    reset(); resetEditors(); restore = []; commands = new Map(); providers = new Map(); contexts = new Map();
    reviews = []; calls = []; logs = []; tabs = []; opened = []; diffs = []; batch = false; onDiff = undefined; closedTabs = undefined;
    draftShown = false; draftDisposed = false;
    data = JSON.parse(readFileSync(path.join(__dirname, '../../../../tests/fixtures/contracts/captured-source-edit.json'), 'utf8'));
    target.sourceDigest = sha256(data.source); editorState.documents.set('/work/model.py', data.source);
    plan = { id: data.capture.attemptId, source: structuredClone(data.capture.source), captureLimits: data.capture.limits } as ResolvedExecutionPlan;
    runner = async () => result();
    patch(vscode.Uri, 'from', (value: any) => ({ ...value, toString: () => `${value.scheme}:${value.path}` }));
    patch(vscode, 'TabInputTextDiff', class { constructor(public original: any, public modified: any) {} });
    patch(vscode, 'TabInputText', class { constructor(public uri: any) {} });
    patch(vscode.workspace, 'registerTextDocumentContentProvider', (scheme: string, provider: any) => {
      providers.set(scheme, provider); return { dispose: () => providers.delete(scheme) };
    });
    const originalOpen = vscode.workspace.openTextDocument;
    patch(vscode.workspace, 'openTextDocument', async (uri: any) => providers.has(uri.scheme)
      ? { uri, getText: () => providers.get(uri.scheme).provideTextDocumentContent(uri) } : originalOpen(uri));
    const originalApply = vscode.workspace.applyEdit;
    patch(vscode.workspace, 'applyEdit', async (edit: any) => {
      const applied = await originalApply(edit);
      if (applied) {
        editorState.documents.set('/work/model.py', data.after); editorState.versions.set('/work/model.py', 2);
        editorState.dirtyDocuments.add('/work/model.py');
      }
      return applied;
    });
    patch(vscode.window, 'showTextDocument', async (document: any, options: any) => {
      assert.equal(options.preview, false, 'a preview result would close the diff and discard its proposal');
      assert.equal(options.viewColumn, opened.length ? 2 : vscode.ViewColumn.Beside);
      opened.push(document.getText()); tabs.push({ input: new vscode.TabInputText(document.uri) });
      return Object.assign(activate(document), { viewColumn: 2 });
    });
    patch(vscode, 'commands', {
      registerCommand(name: string, action: () => any) {
        commands.set(name, action); return { dispose: () => commands.delete(name) };
      },
      async executeCommand(name: string, ...args: any[]) {
        if (name === 'setContext') { contexts.set(args[0], args[1]); return; }
        assert.equal(name, 'vscode.diff');
        const [before, after, title] = args, provider = providers.get(after.scheme);
        assert.equal(args[3].preview, !plan && !batch, 'candidate and multi-file reviews must stay pinned');
        diffs.push({ title, before: provider.provideTextDocumentContent(before), after: provider.provideTextDocumentContent(after) });
        activate({ uri: after }); tabs.push({ input: new vscode.TabInputTextDiff(before, after) });
        onDiff?.();
      },
    });
    patch(vscode.window, 'tabGroups', { get all() { return [{ tabs }]; },
      onDidChangeTabs: (listener: typeof closedTabs) => { closedTabs = listener; return { dispose() {} }; }, close: async () => { tabs = []; } });
  });
  teardown(() => { for (const undo of restore.reverse()) undo(); resetEditors(); reset(); });

  const batchArtifacts = () => [
    { path: 'model.py', before: data.source, after: data.after, before_digest: sha256(data.source), after_digest: sha256(data.after), edits: [] },
    { path: 'config.py', before: 'WIDTH = 12\n', after: 'WIDTH = 16\n', before_digest: sha256('WIDTH = 12\n'), after_digest: sha256('WIDTH = 16\n'), edits: [] },
  ];

  test('Inspect opens the actual failed-load metadata beside the unchanged source diff', async () => {
    runner = async () => {
      const value = result(); value.evidence = candidateCheckpointFixture();
      value.outcome = { ...value.outcome, ok: false, reason: 'load failed', detail: '', exitCode: 3 }; return value;
    };
    const { pending } = await open(); await command('runSourceCandidate');
    assert.match(opened[0], /Actual load: failed/); assert.match(opened[0], /Checkpoint digest: b{64}/);
    assert.match(opened[0], /"left.weight"/); assert.match(opened[0], /"right.weight"/);
    assert.match(opened[0], /Capture: unavailable/); assert.equal(tabs.length, 2);
    assert.equal(editorState.appliedEdits.length, 0);
    command('discardSourceProposal'); assert.equal((await pending).status, 'cancelled');
  });

  test('multi-file review pins every complete diff before allowing one Apply decision', async () => {
    batch = true; plan = undefined;
    let decisions = 0, shown = false;
    onDiff = () => {
      assert.notEqual(contexts.get('tensorviz.sourceReviewActive'), true, 'controls appear after the complete set opens');
      command('applySourceProposal'); assert.equal(decisions, 0, 'Apply must wait for all files');
    };
    const pending = reviewSources('Two-file architecture change', batchArtifacts(), new AbortController().signal,
      undefined, () => { shown = true; }, () => { decisions++; });
    await until(() => shown);
    assert.deepEqual(diffs.map(diff => [diff.before, diff.after]), batchArtifacts().map(item => [item.before, item.after]));
    assert.match(diffs[0].title, /model\.py \(1\/2\)/); assert.match(diffs[1].title, /config\.py \(2\/2\)/);
    assert.equal(tabs.length, 2); assert.equal(editorState.appliedEdits.length, 0);
    command('applySourceProposal'); assert.equal(await pending, true); assert.equal(decisions, 1);
    assert.equal(providers.size, 0); assert.equal(commands.size, 0); assert.equal(tabs.length, 0);
  });

  test('closing either diff discards the entire set and a partial open cannot be accepted', async () => {
    batch = true; plan = undefined;
    const pending = reviewSources('Batch', batchArtifacts(), new AbortController().signal);
    await until(() => diffs.length === 2);
    closedTabs!({ closed: [tabs[0]] });
    assert.equal(await pending, false);
    assert.equal(providers.size, 0); assert.equal(commands.size, 0);
    diffs = [];
    onDiff = () => { if (diffs.length === 1) closedTabs!({ closed: [tabs[0]] }); };
    assert.equal(await reviewSources('Batch', batchArtifacts(), new AbortController().signal), false);
    assert.equal(diffs.length, 1, 'discard during opening must stop the remaining diffs');
  });

  test('failed multi-file opening cleans every private tab and command', async () => {
    batch = true; plan = undefined;
    onDiff = () => { if (diffs.length === 2) throw new Error('test opening failure'); };
    await assert.rejects(reviewSources('Batch', batchArtifacts(), new AbortController().signal), /opening failure/);
    assert.equal(tabs.length, 0); assert.equal(providers.size, 0); assert.equal(commands.size, 0);
    assert.equal(contexts.get('tensorviz.sourceReviewActive'), false);
  });

  test('cancellation during the ready notification cannot strand a multi-file review', async () => {
    batch = true; plan = undefined;
    const controller = new AbortController();
    const approved = await reviewSources('Batch', batchArtifacts(), controller.signal, undefined, () => controller.abort());
    assert.equal(approved, false);
    assert.equal(tabs.length, 0); assert.equal(providers.size, 0); assert.equal(commands.size, 0);
  });

  for (const surface of ['native', 'canvas']) test(`${surface} acceptance claims the decision synchronously before asynchronous diff cleanup`, async () => {
    plan = undefined;
    let decisions = 0, choose!: (approved: boolean) => void, finished = false;
    const pending = reviewSource('Draft', { path: 'model.py', before: data.source, after: data.after,
      before_digest: sha256(data.source), after_digest: sha256(data.after), edits: [] }, new AbortController().signal,
    undefined, (_reveal, decide) => { choose = decide; }, () => { decisions++; });
    void pending.then(() => { finished = true; });
    await until(() => !!choose);
    if (surface === 'native') command('applySourceProposal'); else choose(true);
    assert.equal(decisions, 1);
    assert.equal(finished, false, 'ownership must change before native cleanup awaits');
    choose(false);
    assert.equal(decisions, 1, 'a later decision cannot replace the first');
    assert.equal(await pending, true);
  });

  test('review remains static until Run; candidate success never auto-applies', async () => {
    const { pending, host } = await open();
    assert.ok(draftShown); assert.equal(draftDisposed, false);
    assert.equal(host.appliedSource('native'), null);
    assert.equal(calls.length, 0); assert.equal(reviews[0].candidate?.runtime, 'not_run');
    assert.deepEqual([diffs[0].before, diffs[0].after], [data.source, data.after]);
    await command('runSourceCandidate');
    assert.equal(calls.length, 1); assert.equal(editorState.appliedEdits.length, 0);
    assert.deepEqual(calls[0].candidate.intent, data.intent);
    assert.equal(calls[0].candidate.reviewedSource, data.after);
    assert.deepEqual(calls[0].candidate.plan, plan);
    assert.match(opened[0], /Runtime: succeeded/); assert.match(opened[0], /Semantic postconditions: not checked/);
    assert.equal(reviews.at(-1)?.candidate?.runtime, 'succeeded');
    command('applySourceProposal'); assert.equal((await pending).status, 'applied');
    assert.ok(draftDisposed);
    assert.equal(reviews.at(-1)?.candidate?.freshness, 'stale', 'the copied candidate never certifies the applied buffer');
    assert.equal(editorState.appliedEdits.length, 1); assert.equal(commands.size, 0); assert.equal(providers.size, 0);
    assert.equal(contexts.get('tensorviz.sourceCandidateAvailable'), false);
    const receipt = host.appliedSource('native')!;
    assert.equal(receipt.file, 'model.py'); assert.equal(receipt.beforeDigest, sha256(data.source));
    assert.equal(receipt.afterDigest, sha256(data.after)); assert.equal(receipt.unavailable, null);
    assert.equal(host.appliedSource('another-request'), null);
    host.forgetCandidates('native'); assert.equal(host.appliedSource('native'), null);
  });

  test('an unresolved automatic recipe offers static review without a Run command', async () => {
    plan = undefined; const { pending, host } = await open();
    assert.equal(commands.has('tensorviz.runSourceCandidate'), false);
    command('applySourceProposal'); assert.match((await pending).detail, /runtime validation has not run/);
    assert.equal(calls.length, 0);
    assert.ok(host.appliedSource('native'), 'an Apply receipt does not require candidate execution');
    assert.equal((await host.inspectAppliedSource('native')).source, 'reviewed_unsaved');
    editorState.documents.set('/work/model.py', data.source); editorState.versions.set('/work/model.py', 3);
    assert.equal((await host.inspectAppliedSource('native')).source, 'reverted');
    assert.equal(calls.length, 0, 'receipt inspection cannot execute the model');
  });

  test('host controls share the native run and retain exact pages after Discard', async () => {
    runner = async () => {
      const value = result();
      value.evidence!.execution = { id: plan!.id, plan: plan! } as CandidateExecution;
      return value;
    };
    const { pending, host } = await open();
    assert.throws(() => host.runCandidate('another-request'), /no open native review/);
    const first = host.runCandidate('native');
    assert.equal(first.summary.runtime, 'running');
    assert.equal(host.runCandidate('native').runId, first.runId);
    await until(() => host.getCandidate('native', first.runId).finishedAt !== null);
    assert.equal(calls.length, 1);
    const retained = host.getCandidate('native', first.runId);
    assert.equal(retained.unavailable, null);
    assert.throws(() => host.cancelCandidate('another-request', first.runId), /no longer retained/);
    command('discardSourceProposal'); await pending;
    assert.equal(host.appliedSource('native'), null);
    assert.equal(host.getCandidate('native', first.runId).review, 'historical');
    const page = host.inspectCandidate('native', first.runId, { ...retained.capture!, path: ['operations'], offset: 0, limit: 50 });
    assert.equal(page.total, data.capture.operations.length);
    assert.throws(() => host.runCandidate('native'), /no open native review/);
    host.forgetCandidates('native'); assert.deepEqual(host.listCandidates('native'), []);
  });

  test('cancelling an older retained run cannot cancel a newer native run', async () => {
    const { pending, host } = await open();
    const first = host.runCandidate('native');
    await until(() => host.getCandidate('native', first.runId).finishedAt !== null);
    let release!: (value: CandidateRunResult) => void, signal: AbortSignal | undefined;
    runner = async (_request, abort) => { signal = abort; return new Promise(resolve => { release = resolve; }); };
    const second = host.runCandidate('native'); await until(() => !!signal);
    host.cancelCandidate('native', first.runId); assert.equal(signal!.aborted, false);
    host.cancelCandidate('native', second.runId); assert.equal(signal!.aborted, true);
    release(result()); await until(() => host.getCandidate('native', second.runId).finishedAt !== null);
    assert.equal(host.getCandidate('native', second.runId).summary.runtime, 'cancelled');
    command('discardSourceProposal'); await pending;
    host.dispose(); assert.deepEqual(host.listCandidates('native'), []);
  });

  test('run commands cannot execute from another editor', async () => {
    const { pending } = await open(); activate({ uri: { scheme: 'file' } });
    await command('runSourceCandidate'); assert.equal(calls.length, 0);
    command('discardSourceProposal'); assert.equal((await pending).status, 'cancelled');
  });

  test('a failed forward is shown separately and static Apply remains explicit', async () => {
    runner = async () => {
      const value = result(); value.outcome = { ...value.outcome, ok: false, reason: 'private failure', detail: '', exitCode: 4 };
      value.evidence!.capture!.status = 'partial'; return value;
    };
    const { pending } = await open(); await command('runSourceCandidate');
    assert.match(opened[0], /Runtime: failed/); assert.match(opened[0], /Capture: partial/);
    assert.doesNotMatch(opened[0], /private failure/);
    command('applySourceProposal'); const outcome = await pending;
    assert.equal(outcome.status, 'applied'); assert.match(outcome.detail, /Copied candidate runtime: failed/);
  });

  test('Apply receipt preserves the scoped semantic result without certifying the live buffer', async () => {
    runner = async () => {
      const value = result();
      value.evidence!.postconditions = { version: 1, executionId: 'candidate:native', captureId: data.capture.id,
        recipe: 'binary_scalar_v1', scope: 'observed_expression', status: 'verified', reason: 'observed_requested_literal',
        bindingId: 'binding:literal', observedOperations: 1 };
      return value;
    };
    const { pending } = await open(); await command('runSourceCandidate');
    assert.match(opened[0], /Requested literal observed in 1 captured occurrence/);
    command('applySourceProposal'); const outcome = await pending;
    assert.equal(outcome.status, 'applied');
    assert.match(outcome.detail, /Requested literal observed in 1 captured occurrence/);
    assert.match(outcome.detail, /Other executions are unverified/);
    assert.match(outcome.detail, /Save and Validate the live source/);
    assert.doesNotMatch(outcome.detail, /postconditions not checked/);
  });

  test('a changed original source still refuses Apply after a successful candidate', async () => {
    const { pending, host } = await open(); await command('runSourceCandidate');
    editorState.documents.set('/work/model.py', data.source + '# changed');
    await command('inspectSourceCandidate'); assert.match(opened.at(-1), /Freshness: stale/);
    command('applySourceProposal'); assert.equal((await pending).status, 'refused');
    assert.equal(editorState.appliedEdits.length, 0);
    assert.equal(host.appliedSource('native'), null);
  });

  for (const action of ['applySourceProposal', 'discardSourceProposal']) test(`${action} waits for cancellation cleanup and prevents late result tabs`, async () => {
    let release!: () => void, stopped = false;
    const cleanup = new Promise<void>(resolve => { release = resolve; });
    runner = async (_request, signal) => {
      signal.addEventListener('abort', () => { stopped = true; }); await cleanup; return result();
    };
    const { pending } = await open(), running = command('runSourceCandidate');
    await until(() => calls.length === 1); command(action);
    await until(() => stopped); assert.equal(editorState.appliedEdits.length, 0);
    release(); const outcome = await pending; await running;
    assert.equal(outcome.status, action === 'applySourceProposal' ? 'applied' : 'cancelled');
    assert.equal(opened.length, 0); assert.equal(reviews.at(-1)?.candidate?.runtime, 'cancelled');
  });

  test('Cancel candidate run keeps native review available and writes nothing', async () => {
    runner = async (_request, signal) => {
      await new Promise<void>(resolve => signal.addEventListener('abort', () => resolve())); return result();
    };
    const { pending } = await open(), running = command('runSourceCandidate');
    await until(() => calls.length === 1); command('cancelSourceCandidate'); await running;
    assert.match(opened[0], /Runtime: cancelled/); assert.equal(editorState.appliedEdits.length, 0);
    assert.equal(commands.has('tensorviz.applySourceProposal'), true);
    command('discardSourceProposal'); await pending;
  });

  test('cleanup failure is visible and cannot apply source', async () => {
    runner = async () => ({ outcome: { ok: false, reason: 'cleanup', detail: '' }, cleanupFailed: true });
    const { pending, host } = await open(); await command('runSourceCandidate');
    assert.match(opened[0], /Private temporary files may remain/);
    runner = async () => result();
    assert.throws(() => host.runCandidate('native'), /no open native review/);
    await command('runSourceCandidate');
    assert.equal(calls.length, 1, 'neither transport nor native reruns can erase failed cleanup');
    command('applySourceProposal'); const outcome = await pending;
    assert.equal(outcome.status, 'refused'); assert.match(outcome.detail, /cleanup failed/);
    assert.equal(editorState.appliedEdits.length, 0);
    assert.equal(host.appliedSource('native'), null);
  });

  test('Apply while a result document is opening cannot reopen a disposed review', async () => {
    const original = vscode.workspace.openTextDocument;
    let entered = false, release!: () => void;
    const gate = new Promise<void>(resolve => { release = resolve; });
    patch(vscode.workspace, 'openTextDocument', async (uri: any) => {
      if (uri.path !== '/candidate-validation.md') return original(uri);
      entered = true; await gate; return { uri, getText: () => 'late result' };
    });
    const { pending } = await open(), running = command('runSourceCandidate');
    await until(() => entered); command('applySourceProposal');
    assert.equal((await pending).status, 'applied'); release(); await running;
    assert.equal(opened.length, 0); assert.equal(providers.size, 0);
  });

  test('a final summary observer cannot turn a successful source write into a refusal', async () => {
    const { pending, host } = await open(review => {
      reviews.push(review);
      if (editorState.appliedEdits.length) throw new Error('observer failed after Apply');
    });
    command('applySourceProposal'); assert.equal((await pending).status, 'applied');
    assert.equal(editorState.appliedEdits.length, 1);
    assert.ok(logs.some(line => line.includes('Could not publish the final candidate summary')));
    assert.ok(host.appliedSource('native'));
  });

  for (const revoke of ['forget', 'dispose']) test(`late native Apply preserves the write outcome but cannot restore a ${revoke} receipt`, async () => {
    let entered = false, release!: () => void;
    const gate = new Promise<void>(resolve => { release = resolve; }), apply = vscode.workspace.applyEdit;
    patch(vscode.workspace, 'applyEdit', async (edit: any) => { entered = true; await gate; return apply(edit); });
    const { pending, host } = await open(); command('applySourceProposal');
    await until(() => entered);
    if (revoke === 'forget') host.forgetCandidates('native'); else host.dispose();
    release(); assert.equal((await pending).status, 'applied');
    assert.equal(editorState.appliedEdits.length, 1); assert.equal(host.appliedSource('native'), null);
  });

  for (const failure of ['false', 'throw']) test(`uncertain/rejected editor write (${failure}) cannot produce a confirmed receipt`, async () => {
    patch(vscode.workspace, 'applyEdit', async () => { if (failure === 'throw') throw new Error('unknown receipt'); return false; });
    const { pending, host } = await open(); command('applySourceProposal');
    assert.equal((await pending).status, 'refused'); assert.equal(host.appliedSource('native'), null);
  });
});
