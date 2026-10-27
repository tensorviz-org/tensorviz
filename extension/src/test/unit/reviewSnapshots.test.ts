import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as os from 'node:os';
import { reviewGit } from '@tensorviz/graph-host/gitReviewSources';
import { discoverReview, gitSnapshot, indexSnapshot, reviewCoverage, ReviewSession, snapshotProject, workingSnapshot } from '../../reviewSnapshots';
import { editorState, resetEditors } from '../vscodeStub';

suite('automatic review inventories', () => {
  let root: string;
  setup(async () => {
    resetEditors(); root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-review-snapshots-'));
    await reviewGit(root, ['init', '-q']);
    await fs.writeFile(path.join(root, 'model.py'), 'before\n');
    await fs.writeFile(path.join(root, 'helper.py'), 'shared\n');
    await reviewGit(root, ['add', '.']);
    await reviewGit(root, ['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '-qm', 'base']);
  });
  teardown(async () => { resetEditors(); await fs.rm(root, { recursive: true, force: true }); });
  test('working and index snapshots use compatible identities and never change the branch or dirty buffers', async () => {
    const before = await gitSnapshot(root, 'HEAD');
    assert.deepEqual((await workingSnapshot(root)).files, before.files);
    await fs.writeFile(path.join(root, 'model.py'), 'staged\n'); await reviewGit(root, ['add', 'model.py']);
    const index = await indexSnapshot(root);
    editorState.documents.set(path.join(root, 'model.py'), 'unsaved\n');
    editorState.documents.set(path.join(root, 'new.py'), 'new unsaved\n');
    editorState.dirtyDocuments.add(path.join(root, 'model.py'));
    editorState.dirtyDocuments.add(path.join(root, 'new.py'));
    const status = await reviewGit(root, ['status', '--porcelain']);
    const after = await workingSnapshot(root);
    assert.equal(await index.read('model.py'), 'staged\n');
    assert.equal(await after.read('model.py'), 'unsaved\n');
    assert.equal(await after.read('new.py'), 'new unsaved\n');
    assert.equal(after.files.get('helper.py'), before.files.get('helper.py'));
    assert.equal(await reviewGit(root, ['status', '--porcelain']), status);
    await fs.writeFile(path.join(root, 'model.py'), 'later\n'); await reviewGit(root, ['add', 'model.py']);
    assert.equal(await index.read('model.py'), 'staged\n', 'index snapshots read captured blob IDs, not the mutable index');
    assert.equal(await after.read('model.py'), 'unsaved\n');
  });
  test('project boundaries come from each snapshot including metadata absent on disk', () => {
    const files = new Map([['models/pyproject.toml', 'a'], ['models/nested/net.py', 'b']]);
    assert.equal(snapshotProject('models/nested/net.py', files), 'models');
    assert.equal(snapshotProject('other/net.py', files), '');
  });
  test('discovers unchanged model files affected by shared code and pairs only unique renames', async () => {
    const before = await gitSnapshot(root, 'HEAD');
    await fs.writeFile(path.join(root, 'helper.py'), 'changed shared\n');
    await fs.writeFile(path.join(root, 'model.py'), 'after\n');
    const after = await workingSnapshot(root);
    const fetch = globalThis.fetch;
    globalThis.fetch = async (_url: any, options: any) => {
      const changed = JSON.parse(options.body).sources.find((entry: any) => entry.file === 'model.py').source === 'after\n';
      return new Response(JSON.stringify({ models: [
        { file: 'model.py', cls: changed ? 'Renamed' : 'Net', line: 1, endLine: 2, fingerprint: 'unique', dependencies: ['model.py', 'helper.py'] },
        { file: 'helper.py', cls: 'SharedConsumer', line: 1, endLine: 2, fingerprint: 'other', dependencies: ['helper.py'] },
      ], notes: [
        { file: 'helper.py', detail: 'Shared helper limitation.' },
        { file: 'datasets/unused.py', detail: 'Unrelated class limitation.' },
        ...(changed ? [{ file: 'model.py', detail: 'New limitation.' }] : []),
      ] }));
    };
    const session: ReviewSession = { title: 'Diff', before, after, refresh: async () => session };
    try {
      const inventory = await discoverReview(session, { origin: 'http://localhost', token: 'test' });
      assert.equal(inventory.models.length, 2);
      assert.equal(inventory.models.find(model => model.label.includes('Renamed'))?.status, 'renamed');
      assert.deepEqual(inventory.models.find(model => model.label === 'SharedConsumer')?.affectedFiles, ['helper.py']);
      inventory.notes.push({ side: 'before', detail: 'Source budget reached.' });
      const coverage = reviewCoverage(inventory, inventory.models[0]);
      assert.deepEqual(coverage.notes, [
        'Before and after: helper.py: Shared helper limitation.',
        'After: model.py: New limitation.',
        'Before: Source budget reached.',
      ]);
      assert.deepEqual(coverage.repositoryNotes, ['Before and after: datasets/unused.py: Unrelated class limitation.']);
    } finally { globalThis.fetch = fetch; }
  });
  test('a syntax error in a dependency is unresolved instead of a whole-file deletion', async () => {
    const before = await gitSnapshot(root, 'HEAD');
    await fs.writeFile(path.join(root, 'model.py'), 'after\n');
    const after = await workingSnapshot(root);
    const fetch = globalThis.fetch;
    globalThis.fetch = async (_url: any, options: any) => {
      const changed = JSON.parse(options.body).sources.find((entry: any) => entry.file === 'model.py').source === 'after\n';
      return new Response(JSON.stringify(changed
        ? { models: [], notes: [{ file: 'helper.py', detail: 'could not parse helper.py:2' }] }
        : { models: [{ file: 'model.py', cls: 'Net', line: 1, endLine: 2, fingerprint: 'unique', dependencies: ['model.py', 'helper.py'] }], notes: [] }));
    };
    const session: ReviewSession = { title: 'Diff', before, after, refresh: async () => session };
    try {
      const inventory = await discoverReview(session, { origin: 'http://localhost', token: 'test' });
      assert.equal(inventory.models.length, 1);
      assert.equal(inventory.models[0].status, 'unresolved');
      assert.equal(inventory.models[0].label, 'Net');
      assert.match(inventory.notes.map(note => note.detail).join('\n'), /could not parse helper\.py:2/);
    } finally { globalThis.fetch = fetch; }
  });
});
