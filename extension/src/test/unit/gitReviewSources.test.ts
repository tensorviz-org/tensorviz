import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { changedReviewFiles, gitReviewSourceSet, resolveReviewRevision, reviewGit } from '@tensorviz/graph-host/gitReviewSources';
import { collectSourceSet } from '../../workspaceSources';

suite('Git review sources', () => {
  let root: string;
  setup(async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-git-review-'));
    await reviewGit(root, ['init', '-q']);
    await fs.mkdir(path.join(root, 'models'));
    await fs.writeFile(path.join(root, 'models/net.py'), 'width = 32\n');
    await fs.writeFile(path.join(root, 'models/helper.py'), 'version = 1\n');
    await reviewGit(root, ['add', '.']);
    await reviewGit(root, ['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '-qm', 'before']);
  });
  teardown(async () => { await fs.rm(root, { recursive: true, force: true }); });
  test('inherited repository variables cannot redirect the explicitly selected workspace', async () => {
    const keys = ['GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE', 'GIT_OBJECT_DIRECTORY', 'GIT_IMPLICIT_WORK_TREE'];
    const previous = keys.map(key => process.env[key]);
    try {
      for (const key of keys) process.env[key] = key === 'GIT_IMPLICIT_WORK_TREE' ? '0' : path.join(root, 'foreign-repository');
      const commit = await resolveReviewRevision(root, 'HEAD');
      const sources = await gitReviewSourceSet(root, 'models/net.py', commit);
      assert.equal(sources.entries[0].source, 'width = 32\n');
      assert.equal(await reviewGit(root, ['status', '--porcelain']), '', 'reading another workspace must not change its index');
    } finally {
      keys.forEach((key, index) => previous[index] === undefined ? delete process.env[key] : process.env[key] = previous[index]);
    }
  });
  test('reads the pinned complete revision, preserving the index and dirty files', async () => {
    const commit = await resolveReviewRevision(root, 'HEAD');
    await fs.writeFile(path.join(root, 'models/net.py'), 'width = 64\n');
    await fs.unlink(path.join(root, 'models/helper.py'));
    const status = await reviewGit(root, ['status', '--porcelain']);
    const sources = await gitReviewSourceSet(path.join(root, 'models'), 'net.py', commit);
    assert.equal(sources.entries[0].source, 'width = 32\n');
    assert.equal(sources.entries.find(entry => entry.file === 'helper.py')?.source, 'version = 1\n');
    assert.equal(sources.truncated, false);
    assert.equal(await reviewGit(root, ['status', '--porcelain']), status);
    assert.equal(await fs.readFile(path.join(root, 'models/net.py'), 'utf8'), 'width = 64\n');
  });
  test('the working snapshot uses dirty buffers and reports added/deleted files', async () => {
    const before = await gitReviewSourceSet(root, 'models/net.py', await resolveReviewRevision(root, 'HEAD'));
    const after = await collectSourceSet('models/net.py', { list: async () => ['models/net.py', 'new.py'], read: async () => 'new = True\n',
      openDocument: file => file === 'models/net.py' ? { getText: () => 'width = 128\n', version: 7 } : undefined });
    assert.equal(after.entries[0].source, 'width = 128\n');
    assert.deepEqual(changedReviewFiles(before, after), ['models/helper.py', 'models/net.py', 'new.py']);
  });
  test('bad revisions and missing model files fail rather than falling back to current files', async () => {
    await assert.rejects(resolveReviewRevision(root, '--help'), /not available locally/);
    await assert.rejects(resolveReviewRevision(root, 'does-not-exist'), /not available locally/);
    await assert.rejects(gitReviewSourceSet(root, 'missing.py', await resolveReviewRevision(root, 'HEAD')), /unavailable/);
  });
});
