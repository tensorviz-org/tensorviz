import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import * as os from 'node:os';
import * as path from 'node:path';
import { reviewResourceGuard, ReviewResourceError } from '../../sourceReviewResources';
import { LocalBrowserSourceEdits } from '../../localBrowserSourceEdits';
import { sha256 } from '../../sourceChanges';
import { editorState, reset, resetEditors } from '../vscodeStub';
import type { ProjectionPayload } from '@tensorviz/contracts/protocol';
const vscode = require('../vscodeStub');
const bytes = Buffer.from([0, 255, 128, 1, 0, 254]);
const digest = createHash('sha256').update(bytes).digest('hex');

suite('tracked binary resource review', () => {
  let root: string, controller: AbortController;
  const resources = [{ path: 'data/fixture.pt', digest }];
  setup(async () => {
    reset(); resetEditors(); controller = new AbortController();
    root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-review-resources-')));
    await fs.mkdir(path.join(root, 'data'));
    await fs.writeFile(path.join(root, resources[0].path), bytes);
  });
  teardown(async () => { controller.abort(); resetEditors(); reset(); await fs.rm(root, { recursive: true, force: true }); });
  const guard = (dirty = () => false) => reviewResourceGuard(root, resources, controller.signal, dirty);

  test('checks opaque bytes without decoding and keeps the input snapshot immutable', async () => {
    const requested = structuredClone(resources);
    const pending = reviewResourceGuard(root, requested, controller.signal, () => false);
    requested[0].path = '../other'; requested[0].digest = '0'.repeat(64);
    const current = await pending;
    assert.equal(current(), true);
    assert.deepEqual(await fs.readFile(path.join(root, resources[0].path)), bytes);
    controller.abort(); assert.equal(current(), false);
  });

  for (const change of ['contents', 'same-bytes-replacement', 'directory-replacement', 'delete', 'dirty']) {
    test(`${change} invalidates the final synchronous writer guard`, async () => {
      let dirty = false;
      const current = await guard(() => dirty), file = path.join(root, resources[0].path);
      // Same-length writes can share a coarse timestamp with the snapshot. Wait so ctime moves.
      if (change === 'contents') await new Promise((resolve) => setTimeout(resolve, 10));
      if (change === 'contents') await fs.writeFile(file, Buffer.from([0, 255, 128, 1, 0, 253]));
      if (change === 'same-bytes-replacement') { await fs.rename(file, file + '.old'); await fs.writeFile(file, bytes); }
      if (change === 'directory-replacement') {
        await fs.rename(path.join(root, 'data'), path.join(root, 'old-data'));
        await fs.mkdir(path.join(root, 'data')); await fs.writeFile(file, bytes);
      }
      if (change === 'delete') await fs.unlink(file);
      if (change === 'dirty') dirty = true;
      assert.equal(current(), false);
    });
  }

  test('restoring bytes and mtime cannot hide a write during review', async () => {
    const current = await guard(), file = path.join(root, resources[0].path), before = await fs.stat(file);
    // Coarse filesystem timestamps can leave ctime unchanged when both writes land in the
    // same tick as the snapshot. A short gap makes the replacement visible to the guard.
    const pause = () => new Promise((resolve) => setTimeout(resolve, 10));
    await pause();
    await fs.writeFile(file, 'changed');
    await pause();
    await fs.writeFile(file, bytes);
    await fs.utimes(file, before.atime, before.mtime);
    assert.equal(current(), false, 'ctime changes even when bytes and mtime are restored');
  });

  for (const invalid of ['digest', 'traversal', 'duplicate', 'dirty', 'cancelled', 'oversized', 'leaf-symlink', 'directory-symlink']) {
    test(`${invalid} refuses before any source review`, async () => {
      const requested = structuredClone(resources), file = path.join(root, resources[0].path);
      if (invalid === 'digest') requested[0].digest = '0'.repeat(64);
      if (invalid === 'traversal') requested[0].path = '../fixture.pt';
      if (invalid === 'duplicate') requested.push(requested[0]);
      if (invalid === 'cancelled') controller.abort();
      if (invalid === 'oversized') await fs.truncate(file, 2 * 1024 * 1024 * 1024 + 1);
      if (invalid === 'leaf-symlink') { await fs.rename(file, file + '.real'); await fs.symlink(file + '.real', file); }
      if (invalid === 'directory-symlink') {
        await fs.rename(path.join(root, 'data'), path.join(root, 'real-data'));
        await fs.symlink(path.join(root, 'real-data'), path.join(root, 'data'), 'dir');
      }
      await assert.rejects(reviewResourceGuard(root, requested, controller.signal, () => invalid === 'dirty'), ReviewResourceError);
    });
  }

  test('a stale second resource cannot be hidden by a valid first one', async () => {
    await fs.writeFile(path.join(root, 'checkpoint.pt'), 'other');
    await assert.rejects(reviewResourceGuard(root, [...resources, { path: 'checkpoint.pt', digest }],
      controller.signal, () => false), ReviewResourceError);
  });

  async function proposal(change: 'none' | 'during-review' | 'during-writer' | 'dirty' | 'stale' | 'role') {
    const data = JSON.parse(await fs.readFile(path.join(__dirname, '../../../../tests/fixtures/contracts/captured-source-edit.json'), 'utf8'));
    const target = { workspaceFolder: root, file: 'model.py', cls: 'Model', documentVersion: 1,
      sourceDigest: sha256(data.source), backendRevision: 1 };
    const model = path.join(root, 'model.py'), resource = path.join(root, resources[0].path);
    const node = data.document.semantic.nodes.find((item: any) => item.metadata.captureRef.semanticId === data.intent.target.semanticId);
    const edit = { ...data.intent, nodeId: node.node_id };
    editorState.documents.set(model, data.source);
    data.capture.source.files.push({ ...resources[0], role: 'native', documentVersion: null });
    const dependencies = new Map([['model.py', target.sourceDigest]]);
    if (change === 'role') dependencies.set(resources[0].path, digest);
    if (change === 'dirty') { editorState.documents.set(resource, 'unsaved'); editorState.dirtyDocuments.add(resource); }
    if (change === 'stale') await fs.writeFile(resource, 'changed');
    const original = vscode.workspace.openTextDocument;
    let reviewed = false, requested = false;
    vscode.workspace.openTextDocument = async (uri: any) => {
      assert.notEqual(uri.fsPath, resource, 'binary resources must not become editor text documents');
      if (reviewed && change === 'during-writer') await fs.writeFile(resource, 'changed before the final writer guard');
      return original(uri);
    };
    const host = new LocalBrowserSourceEdits({
      origin: () => 'http://localhost:9000', token: () => 'HOST_ONLY',
      context: () => ({ projection: { document: data.document, spanmap: data.spans } as ProjectionPayload, dependencies }),
      capture: () => data.capture, publish() {}, channel: { appendLine() {} },
      review: async (_title, artifact) => {
        reviewed = true; assert.equal(artifact.after, data.after);
        if (change === 'during-review') await fs.writeFile(resource, 'changed while reviewing');
        return true;
      },
      fetchImpl: async () => { requested = true; return new Response(JSON.stringify(data.plan)); },
    });
    try {
      const outcome = await host.propose({ requestId: 'resource-review', target, edit });
      return { outcome, reviewed, requested };
    } finally { vscode.workspace.openTextDocument = original; }
  }

  test('a captured edit with a binary fixture reaches native review and one exact WorkspaceEdit', async () => {
    const result = await proposal('none');
    assert.equal(result.outcome.status, 'applied');
    assert.equal(result.reviewed, true);
    assert.equal(editorState.appliedEdits.length, 1);
    assert.deepEqual(await fs.readFile(path.join(root, resources[0].path)), bytes);
    assert.match(result.outcome.detail, /runtime validation has not run/);
  });

  for (const change of ['during-review', 'during-writer', 'dirty', 'stale', 'role'] as const) {
    test(`a resource ${change} cannot authorize an applied edit`, async () => {
      const result = await proposal(change);
      assert.equal(result.outcome.status, 'refused');
      assert.equal(result.reviewed, change.startsWith('during-'));
      assert.equal(result.requested, change.startsWith('during-'));
      assert.equal(editorState.appliedEdits.length, 0);
    });
  }
});
