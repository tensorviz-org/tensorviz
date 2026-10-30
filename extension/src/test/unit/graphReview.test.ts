import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { GraphReview } from '../../graphReview';
import { reviewGit } from '@tensorviz/graph-host/gitReviewSources';
import { gitSnapshot, ReviewSession, workingSnapshot } from '../../reviewSnapshots';
import { documentChanges, editorState, resetEditors, state } from '../vscodeStub';
import type { GraphReviewState, LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
const vscode = require('../vscodeStub');

suite('native graph review authority', () => {
  let root: string, review: GraphReview, target: LocalBrowserTargetIdentity;
  let publications: any[], diffs: any[], provider: any, restore: (() => void)[];
  const patch = (object: any, key: string, value: any) => {
    const descriptor = Object.getOwnPropertyDescriptor(object, key);
    Object.defineProperty(object, key, { value, configurable: true, writable: true });
    restore.push(() => descriptor ? Object.defineProperty(object, key, descriptor) : delete object[key]);
  };
  const latest = () => publications.filter(message => message.type === 'graphReviewState').at(-1)?.state as GraphReviewState | null;
  setup(async () => {
    restore = []; publications = []; diffs = []; resetEditors(); state.errorMessages = [];
    root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-review-host-')));
    await reviewGit(root, ['init', '-q']);
    await fs.writeFile(path.join(root, 'model.py'), 'width = 32\n');
    await reviewGit(root, ['add', '.']);
    await reviewGit(root, ['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '-qm', 'before']);
    editorState.documents.set(path.join(root, 'model.py'), 'width = 64\n');
    target = { workspaceFolder: root, file: 'model.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 };
    patch(vscode, 'RelativePattern', class { constructor(public base: any, public pattern: string) {} });
    patch(vscode, 'ProgressLocation', { Window: 10 });
    patch(vscode.Uri, 'from', (value: any) => ({ ...value, toString: () => `${value.scheme}://${value.authority}${value.path}` }));
    patch(vscode.workspace, 'registerTextDocumentContentProvider', (_scheme: string, value: any) => { provider = value; return { dispose() {} }; });
    for (const key of ['onDidCreateFiles', 'onDidDeleteFiles', 'onDidRenameFiles']) patch(vscode.workspace, key, () => ({ dispose() {} }));
    patch(vscode.workspace, 'findFiles', async () => [vscode.Uri.file(path.join(root, 'model.py'))]);
    patch(vscode, 'Disposable', { from: (...values: any[]) => ({ dispose: () => values.forEach(value => value.dispose()) }) });
    patch(vscode.window, 'withProgress', (_options: any, action: any) => action());
    patch(vscode.window, 'tabGroups', { all: [], close: async () => undefined });
    patch(vscode, 'TabInputTextDiff', class {});
    patch(vscode, 'commands', { executeCommand: async (name: string, left: any, right: any, title: string, options: any) => {
      assert.equal(name, 'vscode.diff');
      diffs.push({ left, right, title, options, before: provider.provideTextDocumentContent(left), after: provider.provideTextDocumentContent(right) });
    } });
    patch(globalThis, 'fetch', async (_url: string, options: any) => {
      const body = JSON.parse(options.body);
      assert.equal(body.trace_result, undefined, 'review must never execute a revision');
      assert.equal(body.review_only, true, 'review must not overwrite the backend live graph');
      return new Response(JSON.stringify({ cls: body.root.cls, graph_id: 'g', revision: 1, source_digest: body.root.source_digest,
        projection_digest: body.root.source_digest, traced: false, document: {}, dependencies: [], diagnostics: [], truncated: body.truncated,
        spanmap: { layer: { file: 'model.py', start: [1, 0], end: [1, 10], kind: 'decl' } } }));
    });
    review = new GraphReview({ target: () => target, endpoint: () => ({ origin: 'http://localhost', token: 'test' }), publish: message => publications.push(message) });
  });
  teardown(async () => { review?.dispose(); restore.reverse().forEach(fn => fn()); resetEditors(); await fs.rm(root, { recursive: true, force: true }); });
  test('pins both sides, scopes navigation to review ID and mapped node, and marks dirty edits stale', async () => {
    await review.open({ before: 'HEAD', after: 'WORKTREE' });
    assert.deepEqual(state.errorMessages, []);
    assert.equal(diffs.length, 1); assert.equal(diffs[0].before, 'width = 32\n'); assert.equal(diffs[0].after, 'width = 64\n');
    assert.equal(review.ownsDiff({ original: diffs[0].left, modified: diffs[0].right } as any), true);
    assert.equal(review.ownsDiff({ original: vscode.Uri.file('/foreign.py'), modified: diffs[0].right } as any), false);
    const id = latest()!.id;
    assert.deepEqual(latest()!.liveContext, { workspaceFolder: root, file: 'model.py', cls: 'Net' });
    const request = (reviewId: string, nodeId: string) => ({ type: 'graphReviewRequest', action: 'reveal', reviewId, side: 'before', nodeId } as any);
    await review.handle(request('stale-review', 'layer'));
    await review.handle(request(id, '../../secret'));
    assert.equal(diffs.length, 1, 'foreign and unknown identities cannot open a document');
    await review.handle(request(id, 'layer'));
    assert.equal(diffs.length, 2);
    editorState.documents.set(path.join(root, 'model.py'), 'width = 128\n');
    documentChanges.fire({ document: { uri: vscode.Uri.file(path.join(root, 'model.py')) } } as any);
    assert.equal(latest()!.stale, true);
    assert.equal(provider.provideTextDocumentContent(diffs[0].right), 'width = 64\n', 'review source stays immutable after edits');
    await review.close(); assert.equal(latest(), null); assert.equal(review.active, false);
  });
  test('a retarget while source analysis is running cannot publish a review for the old model', async () => {
    const fetch = globalThis.fetch;
    patch(globalThis, 'fetch', async (...args: any[]) => { target = { ...target, cls: 'Other' }; return (fetch as any)(...args); });
    await review.open({ before: 'HEAD', after: 'WORKTREE' });
    assert.deepEqual(state.errorMessages, []); assert.equal(review.active, false); assert.equal(diffs.length, 0);
    assert.equal(publications.some(message => message.state), false);
  });
  test('an edit through a directory alias marks the immutable review stale', async () => {
    const alias = `${root}-alias`;
    await fs.symlink(root, alias, 'dir');
    try {
      await review.open({ before: 'HEAD', after: 'WORKTREE' });
      assert.equal(latest()!.stale, false);
      documentChanges.fire({ document: { uri: vscode.Uri.file(path.join(alias, 'model.py')) } } as any);
      assert.equal(latest()!.stale, true);
      assert.equal(provider.provideTextDocumentContent(diffs[0].right), 'width = 64\n');
    } finally { await fs.unlink(alias); }
  });
  test('bad projection identity is refused and never opens a misleading code diff', async () => {
    patch(globalThis, 'fetch', async () => new Response(JSON.stringify({ cls: 'Other', source_digest: 'wrong', traced: false })));
    await review.open({ before: 'HEAD', after: 'WORKTREE' });
    assert.equal(diffs.length, 0); assert.match(state.errorMessages[0], /did not match/);
  });
  test('a snapshot review opens without a live model, switches models, and never accepts arbitrary model IDs', async () => {
    const fetch = globalThis.fetch;
    patch(globalThis, 'fetch', async (url: string, options: any) => {
      if (url.endsWith('/review/index')) return new Response(JSON.stringify({ models: ['Encoder', 'Decoder'].map((cls, i) => ({
        file: 'model.py', cls, line: i + 1, endLine: i + 1, fingerprint: cls, dependencies: ['model.py'] })), notes: [] }));
      return fetch(url, options);
    });
    const before = await gitSnapshot(root, 'HEAD'), after = await workingSnapshot(root);
    const session: ReviewSession = { title: 'Pull request', folder: root, before, after, refresh: async () => session };
    target = undefined as any;
    await review.openSession(session);
    assert.deepEqual(state.errorMessages, []);
    assert.equal(latest()!.models?.length, 2);
    assert.equal(latest()!.after.cls, 'Encoder');
    assert.equal(latest()!.liveContext, undefined, 'a PR without a local model cannot borrow another graph');
    assert.equal(diffs.length, 1);
    await review.handle({ type: 'graphReviewRequest', action: 'selectModel', reviewId: latest()!.id, modelId: '../../secret' } as any);
    assert.equal(diffs.length, 1);
    const second = latest()!.models![1].id;
    await review.handle({ type: 'graphReviewRequest', action: 'selectModel', reviewId: latest()!.id, modelId: second } as any);
    assert.equal(latest()!.after.cls, 'Decoder');
    documentChanges.fire({ document: { uri: vscode.Uri.file(path.join(root, 'model.py')) } } as any);
    assert.equal(latest()!.stale, true);
    await review.handle({ type: 'graphReviewRequest', action: 'selectModel', reviewId: latest()!.id, modelId: latest()!.models![0].id } as any);
    assert.equal(latest()!.stale, true, 'switching models must not bless stale working snapshots');
  });
  test('failed model analysis publishes incomplete coverage instead of retaining another model', async () => {
    const fetch = globalThis.fetch;
    patch(globalThis, 'fetch', async (url: string, options: any) => {
      if (url.endsWith('/review/index')) return new Response(JSON.stringify({ models: ['Good', 'Bad'].map(cls => ({
        file: 'model.py', cls, line: 1, endLine: 2, fingerprint: cls, dependencies: ['model.py'] })), notes: [] }));
      if (JSON.parse(options.body).root.cls === 'Bad') return new Response('', { status: 422 });
      return fetch(url, options);
    });
    const session: ReviewSession = { title: 'PR', folder: root, before: await gitSnapshot(root, 'HEAD'), after: await workingSnapshot(root), refresh: async () => session };
    await review.openSession(session);
    await review.handle({ type: 'graphReviewRequest', action: 'selectModel', reviewId: latest()!.id, modelId: latest()!.models![1].id } as any);
    assert.equal(latest()!.after.cls, 'Bad');
    assert.equal(latest()!.modelStatus, 'unresolved');
    assert.equal(latest()!.after.truncated, true);
    assert.ok(latest()!.notes?.some(note => note.includes('422')));
  });

  test('MCP class selection is exact and cancellation cannot publish a different model', async () => {
    const fetch = globalThis.fetch;
    patch(globalThis, 'fetch', async (url: string, options: any) => {
      if (url.endsWith('/review/index')) return new Response(JSON.stringify({ models: ['Encoder', 'Decoder'].map((cls, i) => ({
        file: 'model.py', cls, line: i + 1, endLine: i + 1, fingerprint: cls, dependencies: ['model.py'] })), notes: [] }));
      return fetch(url, options);
    });
    const session: ReviewSession = { title: 'Local changes', folder: root, before: await gitSnapshot(root, 'HEAD'), after: await workingSnapshot(root),
      preferredFile: 'model.py', preferredClass: 'Decoder', refresh: async () => session };
    const result = await review.openSession(session);
    assert.equal(result?.after.cls, 'Decoder');
    await assert.rejects(review.openSession({ ...session, preferredClass: 'Missing' }), { code: 'MODEL_NOT_DISCOVERED' });
    assert.equal(latest()?.id, result?.id, 'a missing requested class must not replace the selected model');
    const controller = new AbortController();
    const discover = globalThis.fetch;
    patch(globalThis, 'fetch', async (...args: any[]) => { controller.abort(); return (discover as any)(...args); });
    assert.equal(await review.openSession({ ...session, preferredClass: 'Encoder' }, undefined, controller.signal), undefined);
    assert.equal(latest()?.id, result?.id, 'cancellation during discovery must preserve the visible comparison');
  });
  const indexNet = () => {
    const fetch = globalThis.fetch;
    patch(globalThis, 'fetch', async (url: string, options: any) => {
      if (url.endsWith('/review/index')) return new Response(JSON.stringify({ models: [{
        file: 'model.py', cls: 'Net', line: 1, endLine: 1, fingerprint: 'Net', dependencies: ['model.py'] }], notes: [] }));
      return fetch(url, options);
    });
  };
  test('the live model of the root project, also through a directory alias, is the context of its Git review', async () => {
    indexNet();
    const session: ReviewSession = { title: 'Local changes', folder: root, before: await gitSnapshot(root, 'HEAD'), after: await workingSnapshot(root),
      preferredFile: 'model.py', preferredClass: 'Net', refresh: async () => session };
    await review.openSession(session);
    assert.deepEqual(latest()!.liveContext, { workspaceFolder: root, file: 'model.py', cls: 'Net' });
    const select = () => review.handle({ type: 'graphReviewRequest', action: 'selectModel', reviewId: latest()!.id, modelId: latest()!.selectedModel } as any);
    const alias = `${root}-alias`;
    await fs.symlink(root, alias, 'dir');
    try {
      target = { ...target, workspaceFolder: alias };
      await select();
      assert.deepEqual(latest()!.liveContext, { workspaceFolder: alias, file: 'model.py', cls: 'Net' });
    } finally { await fs.unlink(alias); }
    target = { ...target, workspaceFolder: path.join(root, 'model.py', 'not-a-directory') };
    await select();
    assert.deepEqual(state.errorMessages, [], 'an unresolvable live path only leaves the review without context');
    assert.equal(latest()!.liveContext, undefined);
  });
  test('a nested project traced from its own root is the context of its Git review, and the same class elsewhere is not', async () => {
    await fs.mkdir(path.join(root, 'my-proj'));
    await fs.writeFile(path.join(root, 'my-proj', 'model.py'), 'width = 32\n');
    await reviewGit(root, ['add', '.']);
    await reviewGit(root, ['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'nested']);
    editorState.documents.set(path.join(root, 'my-proj', 'model.py'), 'width = 64\n');
    indexNet();
    // modelProjectRoot stops at a folder that is not a Python identifier.
    target = { ...target, workspaceFolder: path.join(root, 'my-proj') };
    const session: ReviewSession = { title: 'Local changes', folder: root, before: await gitSnapshot(root, 'HEAD'), after: await workingSnapshot(root),
      preferredFile: 'my-proj/model.py', preferredClass: 'Net', refresh: async () => session };
    await review.openSession(session);
    assert.deepEqual(state.errorMessages, []);
    assert.equal(latest()!.after.file, 'my-proj/model.py');
    assert.deepEqual(latest()!.liveContext, { workspaceFolder: path.join(root, 'my-proj'), file: 'model.py', cls: 'Net' });
    const other = latest()!.models!.find(model => model.file === 'model.py')!;
    await review.handle({ type: 'graphReviewRequest', action: 'selectModel', reviewId: latest()!.id, modelId: other.id } as any);
    assert.equal(latest()!.after.file, 'model.py');
    assert.equal(latest()!.liveContext, undefined, 'the same class in another file is another model');
  });

});
