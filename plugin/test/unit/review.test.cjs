const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { reviewGit } = require('@tensorviz/graph-host/gitReviewSources');
const { StandaloneReview } = require('../../out/host/review.cjs');

test('review receipts belong to their connected view and every terminal result has a useful continuation', async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-review-unit-')));
  const foreign = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-review-hook-'));
  const keys = ['GIT_DIR', 'GIT_COMMON_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_IMPLICIT_WORK_TREE'];
  const previous = keys.map(key => process.env[key]);
  const signal = new AbortController().signal;
  let views = ['chosen', 'other'], unavailable = false;
  const review = new StandaloneReview({ signal, busy: () => false, prepare: async () => {}, connect: async () => {},
    target: () => undefined, publish() {}, view: async () => { throw new Error('Existing view must be reused'); },
    views: { list: () => views.map(viewId => ({ viewId })) },
    post: async (route, body) => route.endsWith('/index') ? {
      models: [{ file: 'model.py', cls: 'Tiny', line: 1, endLine: 2, fingerprint: 'tiny', dependencies: ['model.py'] }], notes: [],
    } : unavailable ? Promise.reject(new Error('Unsupported source')) : {
      source_digest: body.root.source_digest, cls: body.root.cls, traced: false, spanmap: {},
      document: { semantic: { nodes: [{ id: 'activation' }] } },
    },
  });
  try {
    await reviewGit(foreign, ['init', '-q']);
    const foreignConfig = await fs.readFile(path.join(foreign, '.git/config'), 'utf8');
    for (const key of keys) delete process.env[key];
    process.env.GIT_DIR = path.join(foreign, '.git');
    process.env.GIT_IMPLICIT_WORK_TREE = '0';
    const git = args => reviewGit(root, args);
    await fs.writeFile(path.join(root, 'model.py'), 'class Tiny: pass\n');
    await git(['init', '-q']); await git(['add', 'model.py']);
    await git(['-c', 'user.name=Test', '-c', 'user.email=test@localhost', '-c', 'commit.gpgsign=false', 'commit', '-qm', 'base']);
    assert.equal(await fs.readFile(path.join(foreign, '.git/config'), 'utf8'), foreignConfig, 'fixture setup cannot reconfigure the hook repository');
    await assert.rejects(fs.stat(path.join(foreign, '.git/index')), { code: 'ENOENT' }, 'fixture setup cannot stage files in the hook repository');
    await fs.writeFile(path.join(root, 'model.py'), 'class Tiny: value = 1\n');
    const args = { workspaceRoot: root, viewId: 'chosen' };
    const first = review.invoke(args, signal);
    await assert.rejects(review.invoke(args, signal), error => error.code === 'BUSY', 'lock covers asynchronous workspace resolution');
    let result = await first;
    assert.equal(result.status, 'awaiting_browser');
    const receipt = (viewId, mode, reviewId = result.reviewId) => review.handle({ type: 'graphReviewApplied', viewId, mode, reviewId });
    const poll = () => review.invoke({ reviewId: result.reviewId }, signal);
    await receipt('other', 'diff');
    assert.equal((await poll()).displayed, false, 'a different browser cannot attest this view');
    await receipt('chosen', 'diff');
    await receipt('other', 'before');
    assert.equal((await poll()).displayed, true, 'another view cannot overwrite this view receipt');
    await receipt('chosen', 'after');
    assert.equal((await poll()).status, 'awaiting_diff');
    assert.equal((await poll()).nextAction.kind, 'show_diff');
    await receipt('chosen', 'diff');
    views = ['other'];
    await review.handle({ type: 'viewState', viewId: 'chosen', connected: false });
    views = ['chosen', 'other'];
    assert.equal((await poll()).displayed, false, 'reconnection needs a new render');
    await receipt('chosen', 'diff', 'obsolete-review');
    assert.equal((await poll()).displayed, false);
    await fs.writeFile(path.join(root, 'model.py'), 'class Tiny: value = 2\n');
    assert.equal((await poll()).status, 'stale');
    await receipt('chosen', 'diff');
    assert.equal((await poll()).displayed, false, 'a render cannot certify changed source');
    unavailable = true;
    result = await review.invoke(args, signal);
    assert.equal(result.status, 'no_graph');
    assert.equal(result.nextAction.kind, 'inspect_source');
    assert.equal(result.coverage.incomplete, true);
    await fs.writeFile(path.join(root, 'model.py'), 'class Tiny: pass\n');
    result = await review.invoke(args, signal);
    assert.equal(result.status, 'no_models');
    assert.equal(result.nextAction.kind, 'inspect_source');
    assert.equal(result.displayed, false);
    review.close();
    await assert.rejects(poll(), error => error.code === 'STALE_REVIEW');
  } finally {
    keys.forEach((key, index) => previous[index] === undefined ? delete process.env[key] : process.env[key] = previous[index]);
    review.close();
    await fs.rm(root, { recursive: true, force: true });
    await fs.rm(foreign, { recursive: true, force: true });
  }
});
