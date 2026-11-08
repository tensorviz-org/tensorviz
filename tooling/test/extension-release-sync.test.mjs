import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync, utimesSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';
import { syncGithubRelease, shouldUpdateRollingRelease } from '../../extension/scripts/sync-github-release.mjs';

const source = 'a'.repeat(40), repository = 'owner/extension';
function fixture(t) {
  const root = mkdtempSync(join(tmpdir(), 'tensorviz-release-test-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(join(root, 'extension'));
  // The archived names vsce produces: CHANGELOG.md is stored lowercase.
  const files = ['extension/package.json', 'extension/changelog.md', 'extension/code.js'];
  writeFileSync(join(root, files[0]), JSON.stringify({ name: 'tensorviz-vscode', publisher: 'tensorviz', version: '0.1.8' }));
  writeFileSync(join(root, files[1]), '# Changelog\n\n## 0.1.8\n\n- Release feature.\n\n## 0.1.7\n\n- Older feature.\n');
  writeFileSync(join(root, files[2]), 'published code');
  const pack = name => {
    const path = join(root, name);
    execFileSync('zip', ['-q', path, ...files], { cwd: root });
    return path;
  };
  const vsix = pack('original.vsix');
  const state = { releases: [], latest: { tag_name: 'v0.1.6' }, writes: [], assets: new Map(), nextId: 1, failUpload: false, conflict: false, unavailable: false, tagExists: false, tagLookupDenied: false };
  const missing = () => { const error = new Error('Not found'); error.stderr = 'HTTP 404'; throw error; };
  const command = (tool, args, options) => {
    if (tool === 'unzip') return execFileSync(tool, args, options);
    if (tool === 'curl') {
      assert.match(args.at(-1), /publishers\/tensorviz\/vsextensions\/tensorviz-vscode\/0\.1\.8\/vspackage$/);
      if (state.unavailable) throw new Error('Marketplace unavailable');
      writeFileSync(args[args.indexOf('--output') + 1], readFileSync(vsix));
      return '';
    }
    assert.equal(tool, 'gh');
    const endpoint = args.find(arg => arg.startsWith('repos/'));
    if (args[0] === 'api' && args.includes('--method')) {
      const body = JSON.parse(readFileSync(args[args.indexOf('--input') + 1]));
      state.writes.push(args[args.indexOf('--method') + 1]);
      if (args.includes('POST')) {
        assert.equal(body.draft, true, 'An incomplete release must not become visible');
        assert.equal(body.target_commitish, source);
        assert.ok(!state.releases.some(item => item.tag_name === body.tag_name));
        const release = { ...body, id: 1, assets: [] };
        state.releases.push(release);
        return JSON.stringify(release);
      }
      const release = state.releases.find(item => endpoint.endsWith(`/${item.id}`));
      assert.equal(release.assets.length, 2, 'Both package and checksum must exist before promotion');
      Object.assign(release, body);
      if (body.draft === false) state.tagExists = true;
      if (body.make_latest === 'true') state.latest = release;
      return JSON.stringify(release);
    }
    if (endpoint?.includes('/releases?')) return JSON.stringify([[], state.releases]);
    if (endpoint?.endsWith('/git/ref/tags/v0.1.8')) {
      if (state.tagLookupDenied) { const error = new Error('Forbidden'); error.stderr = 'HTTP 403'; throw error; }
      return state.conflict || state.tagExists ? JSON.stringify({ ref: 'refs/tags/v0.1.8' }) : missing();
    }
    if (endpoint?.endsWith('/commits/v0.1.8')) {
      if (state.conflict || state.tagExists) return JSON.stringify({ sha: state.conflict ? 'b'.repeat(40) : source });
      // GitHub's commits endpoint returns 422, not 404, for an absent release tag.
      const error = new Error('No commit found'); error.stderr = 'No commit found for SHA: v0.1.8 (HTTP 422)'; throw error;
    }
    if (endpoint?.endsWith('/releases/latest')) return JSON.stringify(state.latest);
    if (endpoint?.includes('/releases/assets/')) return state.assets.get(Number(endpoint.split('/').at(-1)));
    if (args[0] === 'release' && args[1] === 'upload') {
      assert.ok(!args.includes('--clobber'), 'Released bytes must never be replaced');
      const release = state.releases.find(item => item.tag_name === args[2]);
      if (state.failUpload && release.assets.length === 1) { state.failUpload = false; throw new Error('Interrupted upload'); }
      const id = state.nextId++;
      state.assets.set(id, readFileSync(args[3]));
      release.assets.push({ id, name: basename(args[3]) });
      state.writes.push('upload');
      if (state.moveTagOnUpload) state.conflict = true;
      return '';
    }
    throw new Error(`Unexpected command: ${args.join(' ')}`);
  };
  const sync = (candidate = vsix) => syncGithubRelease({ repository, source, vsix: candidate, command });
  return { root, files, vsix, state, sync, pack };
}

test('publishes exact Marketplace bytes, resumes a partial upload, and leaves completed releases unchanged', t => {
  const { state, sync, vsix } = fixture(t);
  state.failUpload = true;
  assert.throws(sync, /Interrupted upload/);
  assert.equal(state.releases[0].draft, true);
  assert.equal(state.latest.tag_name, 'v0.1.6');
  assert.equal(state.releases[0].assets.length, 1);
  const result = sync();
  assert.equal(result.version, '0.1.8');
  assert.equal(state.latest.tag_name, 'v0.1.8');
  assert.equal(state.latest.prerelease, false);
  assert.deepEqual(state.assets.get(1), readFileSync(vsix));
  assert.equal(String(state.assets.get(2)), `${result.sha256}  tensorviz-vscode-0.1.8.vsix\n`);
  assert.match(state.latest.body, /Release feature/);
  assert.ok(!state.latest.body.includes('Older feature'));
  const writes = state.writes.length;
  sync();
  assert.equal(state.writes.length, writes, 'A retry must be read-only after successful publication');
});

test('recovers a missing release from an equivalent rebuild while serving the original Marketplace ZIP', t => {
  const { root, files, pack, sync, state, vsix } = fixture(t);
  for (const file of files) utimesSync(join(root, file), new Date('2020-01-01'), new Date('2020-01-01'));
  const rebuilt = pack('rebuilt.vsix');
  assert.notDeepEqual(readFileSync(rebuilt), readFileSync(vsix));
  sync(rebuilt);
  assert.deepEqual(state.assets.get(1), readFileSync(vsix));
});

test('duplicate publication never substitutes newer code under an existing Marketplace version', t => {
  const { root, files, pack, sync, state } = fixture(t);
  writeFileSync(join(root, files[2]), 'unpublished changes');
  const changed = pack('changed.vsix');
  assert.throws(() => sync(changed), /already published with different contents/);
  assert.equal(state.writes.length, 0);
  sync();
  const writes = state.writes.length;
  assert.throws(() => sync(changed), /already published with different contents/);
  assert.equal(state.writes.length, writes, 'Later pushes keep the numbered release tied to the published package');
});

test('a retry of an older release cannot replace a newer Latest release', t => {
  const { state, sync } = fixture(t);
  const newer = { tag_name: 'v0.1.10', draft: false, prerelease: false, id: 2, assets: [] };
  state.releases.push(newer);
  state.latest = newer;
  sync();
  assert.equal(state.latest.tag_name, 'v0.1.10');
  assert.equal(state.releases.find(item => item.tag_name === 'v0.1.8').draft, false);
});

test('an existing matching tag can be reused, but a denied tag lookup stops publication', t => {
  const { state, sync } = fixture(t);
  state.tagLookupDenied = true;
  assert.throws(sync, /Forbidden/);
  assert.equal(state.writes.length, 0);
  state.tagLookupDenied = false;
  state.tagExists = true;
  sync();
  assert.equal(state.latest.tag_name, 'v0.1.8');
});

test('download failures, conflicting tags and altered release assets fail without overwriting publication', t => {
  const { state, sync } = fixture(t);
  state.unavailable = true;
  assert.throws(sync, /Marketplace unavailable/);
  assert.equal(state.writes.length, 0);
  state.unavailable = false;
  state.conflict = true;
  assert.throws(sync, /already points to another commit/);
  assert.equal(state.writes.length, 0);
  state.conflict = false;
  sync();
  state.assets.set(1, Buffer.from('different package'));
  const writes = state.writes.length;
  assert.throws(sync, /differs from Marketplace/);
  assert.equal(state.writes.length, writes);
});

test('workflow sync follows successful Marketplace publication and precedes the rolling preview update', () => {
  const workflow = readFileSync(new URL('../../.github/workflows/release-vsix.yml', import.meta.url), 'utf8');
  const publish = workflow.indexOf('npx --no-install vsce publish');
  const sync = workflow.indexOf('run: node extension/scripts/sync-github-release.mjs');
  const preview = workflow.indexOf('- name: Update the rolling GitHub VSIX download');
  assert.ok(publish > 0 && sync > publish && preview > sync);
  assert.ok(!workflow.includes('continue-on-error: true'));
});


for (const target of [undefined, 'main', 'b'.repeat(40)]) test(`draft without tag requires exact source target (${target})`, t => {
  const { state, sync } = fixture(t);
  state.releases.push({ id: 1, tag_name: 'v0.1.8', draft: true, target_commitish: target, assets: [] });
  assert.throws(sync, /exact reviewed source commit/);
  assert.deepEqual(state.writes, []);
});
test('matching source tag permits a recovered draft whose target was a branch name', t => {
  const { state, sync } = fixture(t); state.tagExists = true;
  state.releases.push({ id: 1, tag_name: 'v0.1.8', draft: true, target_commitish: 'main', assets: [] });
  sync(); assert.equal(state.latest.tag_name, 'v0.1.8');
});
test('duplicate drafts fail before any publication', t => {
  const { state, sync } = fixture(t);
  for (const id of [1, 2]) state.releases.push({ id, tag_name: 'v0.1.8', draft: true, target_commitish: source, assets: [] });
  assert.throws(sync, /Multiple releases/); assert.deepEqual(state.writes, []);
});
test('published releases with missing assets or missing source tags cannot be repaired silently', t => {
  const { state, sync } = fixture(t); sync();
  const writes = state.writes.length, release = state.releases[0];
  release.assets.pop();
  assert.throws(sync, /Published release is missing/);
  assert.equal(state.writes.length, writes);
  state.tagExists = false;
  assert.throws(sync, /must be a draft targeting/);
  assert.equal(state.writes.length, writes);
});
test('corrupt draft checksum is rejected before uploading a missing VSIX', t => {
  const { state, sync } = fixture(t);
  state.releases.push({ id: 1, tag_name: 'v0.1.8', draft: true, target_commitish: source, assets: [{ id: 10, name: 'tensorviz-vscode-0.1.8.vsix.sha256' }] });
  state.assets.set(10, Buffer.from('corrupt checksum'));
  assert.throws(sync, /differs from Marketplace/); assert.deepEqual(state.writes, []);
});
test('duplicate assets fail before uploading anything', t => {
  const { state, sync } = fixture(t); sync();
  state.releases[0].assets.push({ ...state.releases[0].assets[0] });
  const writes = state.writes.length;
  assert.throws(sync, /Duplicate release asset/); assert.equal(state.writes.length, writes);
});
test('rolling latest guard considers all pages and ignores native, draft and prerelease versions', () => {
  const release = (tag_name, extra = {}) => ({ tag_name, draft: false, prerelease: false, ...extra });
  let pages = [[release('v0.1.8')], [release('plugin-v9.0.0'), release('latest'), release('v0.1.10', { draft: true }), release('v0.2.0', { prerelease: true })]];
  const command = (tool, args) => { assert.equal(tool, 'gh'); assert.ok(args.includes('--paginate')); return JSON.stringify(pages); };
  const check = version => shouldUpdateRollingRelease({ repository, version, command });
  assert.equal(check('0.1.8'), true);
  pages[1].push(release('v0.1.10'));
  assert.equal(check('0.1.8'), false); assert.equal(check('0.1.10'), true);
  assert.throws(() => check('0.1.9'), /published numbered release/);
  assert.throws(() => check('../bad'), /stable extension version/);
  assert.throws(() => shouldUpdateRollingRelease({ repository, version: '0.1.8', command: () => { throw new Error('Forbidden'); } }), /Forbidden/);
});

test('tag movement during draft upload prevents promotion', t => {
  const { state, sync } = fixture(t); state.moveTagOnUpload = true;
  assert.throws(sync, /already points to another commit/);
  assert.equal(state.releases[0].draft, true);
  assert.ok(!state.writes.includes('PATCH'));
});
