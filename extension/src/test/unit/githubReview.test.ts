import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as os from 'node:os';
import { createHash } from 'node:crypto';
import { choosePullRequest, githubRepository, githubSnapshot, loadPullRequest, parsePullRequestUrl, GitHubRead } from '../../githubReview';

suite('PR review source loading', () => {
  let cache: string;
  const base = 'a'.repeat(40), head = 'b'.repeat(40), ancestor = 'c'.repeat(40), next = 'd'.repeat(40);
  const source = 'from torch import nn\nclass Net(nn.Module): pass\n';
  const sha = createHash('sha1').update(`blob ${Buffer.byteLength(source)}\0${source}`).digest('hex');
  setup(async () => { cache = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-pr-cache-')); });
  teardown(async () => { await fs.rm(cache, { recursive: true, force: true }); });
  test('only GitHub PR identities are accepted; credentials and arbitrary URLs are refused', () => {
    assert.deepEqual(parsePullRequestUrl('https://github.com/team/repo/pull/123/files'), { owner: 'team', repo: 'repo', number: 123 });
    for (const url of ['http://github.com/a/b/pull/1', 'https://github.com.evil/a/b/pull/1', 'https://user:secret@github.com/a/b/pull/1',
      'file:///repo', 'https://github.com/a/b/issues/1', 'https://github.com/a/b/pull/-1']) assert.throws(() => parsePullRequestUrl(url));
    assert.equal(githubRepository('git@github.com:team/repo.git'), 'team/repo');
    assert.equal(githubRepository('https://github.com/team/repo.git'), 'team/repo');
    assert.equal(githubRepository('https://evil/github.com/team/repo'), undefined);
  });
  test('uses the merge base and fork head, verifies source blobs, and refreshes only on request', async () => {
    const calls: string[] = []; let current = head;
    const read: GitHubRead = async <T>(route: string): Promise<T> => {
      calls.push(route);
      if (route === '/repos/team/repo/pulls/12') return { number: 12, title: 'Change a model',
        base: { sha: base, repo: { full_name: 'team/repo' } }, head: { sha: current, repo: { full_name: 'author/fork' } } } as T;
      if (route.includes('/compare/')) return { merge_base_commit: { sha: ancestor } } as T;
      if (route.includes('/git/trees/')) return { tree: [{ path: 'model.py', type: 'blob', mode: '100644', sha }], truncated: false } as T;
      if (route.includes('/git/blobs/')) return { sha, encoding: 'base64', content: Buffer.from(source).toString('base64'), size: Buffer.byteLength(source) } as T;
      throw new Error(`Unexpected route: ${route}`);
    };
    const session = await loadPullRequest({ owner: 'team', repo: 'repo', number: 12 }, read, cache);
    assert.equal(session.before.commit, ancestor, 'base tip includes changes outside the PR');
    assert.ok(calls.includes(`/repos/author/fork/git/trees/${head}?recursive=1`));
    assert.equal(await session.after.read('model.py'), source);
    current = next;
    assert.equal(await session.updated!(), true);
    assert.equal(session.after.commit, head, 'update detection must not replace the reviewed revision');
    assert.equal((await session.refresh()).after.commit, next);
    const cached = await githubSnapshot('author/fork', head, read, cache);
    const count = calls.filter(route => route.includes('/git/blobs/')).length;
    assert.equal(await cached.read('model.py'), source);
    assert.equal(calls.filter(route => route.includes('/git/blobs/')).length, count, 'verified blobs can be reused');
  });
  test('rejects corrupted source even when the server repeats the expected SHA', async () => {
    const read: GitHubRead = async <T>(route: string): Promise<T> => (route.includes('/git/trees/')
      ? { tree: [{ path: 'model.py', type: 'blob', mode: '100644', sha }], truncated: false }
      : { sha, encoding: 'base64', content: Buffer.from('bad').toString('base64'), size: 3 }) as T;
    const snapshot = await githubSnapshot('team/repo', head, read, cache);
    await assert.rejects(snapshot.read('model.py'), /digest did not match/);
  });
  test('retains truncation and excludes symlinks, traversal and oversized source from loading', async () => {
    const read: GitHubRead = async <T>(): Promise<T> => ({ truncated: true, tree: [
      { path: '../secret.py', type: 'blob', mode: '100644', sha },
      { path: 'link.py', type: 'blob', mode: '120000', sha },
      { path: 'huge.py', type: 'blob', mode: '100644', sha, size: 2_000_000 },
    ] }) as T;
    const snapshot = await githubSnapshot('team/repo', head, read, cache);
    assert.equal(snapshot.truncated, true); assert.deepEqual([...snapshot.files.keys()], ['huge.py']);
    assert.ok(snapshot.unavailable!.has('huge.py')); await assert.rejects(snapshot.read('huge.py'), /size limit/);
  });
  test('a URL can be pasted and accepted while the repository PR list is still loading', async () => {
    const vscode = require('../vscodeStub');
    const original = vscode.window.createQuickPick;
    const events: Record<string, Function> = {};
    const picker: any = { value: '', items: [], selectedItems: [], show() {}, dispose() {},
      onDidHide: (fn: Function) => { events.hide = fn; return { dispose() {} }; },
      onDidChangeValue: (fn: Function) => { events.change = fn; return { dispose() {} }; },
      onDidAccept: (fn: Function) => { events.accept = fn; return { dispose() {} }; } };
    vscode.window.createQuickPick = () => picker;
    try {
      const result = choosePullRequest('team/repo', () => new Promise(() => {}));
      assert.equal(picker.busy, true);
      picker.value = 'https://github.com/another/models/pull/42'; events.change();
      picker.selectedItems = [picker.items[0]]; events.accept();
      assert.deepEqual(await result, { owner: 'another', repo: 'models', number: 42 });
    } finally { vscode.window.createQuickPick = original; }
  });

});
