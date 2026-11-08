import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { syncOpenVsxRelease, syncOpenVsxFromResult } from '../../extension/scripts/sync-openvsx-release.mjs';

const version = '0.1.15', source = 'a'.repeat(40), repository = 'tensorviz-org/tensorviz';
const downloadUrl = `https://open-vsx.org/api/tensorviz/tensorviz-vscode/${version}/file/package.vsix`;
const info = { namespace: 'tensorviz', name: 'tensorviz-vscode', version, files: { download: downloadUrl } };

function fixture(t) {
  const root = mkdtempSync(join(tmpdir(), 'tensorviz-openvsx-test-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(join(root, 'extension'));
  const makePackage = (manifest = {}) => {
    writeFileSync(join(root, 'extension/package.json'), JSON.stringify({
      publisher: 'tensorviz', name: 'tensorviz-vscode', version, ...manifest,
    }));
    const file = join(root, 'original.vsix');
    rmSync(file, { force: true });
    execFileSync('zip', ['-q', file, 'extension/package.json'], { cwd: root });
    return readFileSync(file);
  };
  const bytes = makePackage();
  const state = { marketplace: bytes, github: bytes, openvsx: bytes, existing: null,
    uploaded: false, publishCalls: [], sourceLookups: [], sleeps: [], requests: [],
    visibilityDelay: 0, lookupStatus: 200, marketplaceStatus: 200, downloadStatus: 200,
    failUpload: false, neverVisible: false, draft: false, sourceVersion: version, tagSource: source };
  const command = (tool, args, options) => {
    if (tool === 'unzip') return execFileSync(tool, args, options);
    if (tool === 'gh') {
      assert.equal(args[0], 'api', 'GitHub access must remain read-only');
      const endpoint = args[1];
      if (endpoint.includes('/contents/')) {
        state.sourceLookups.push(endpoint);
        return JSON.stringify({ content: Buffer.from(JSON.stringify({ version: state.sourceVersion })).toString('base64') });
      }
      if (endpoint === `repos/${repository}/commits/v${version}`) return JSON.stringify({ sha: state.tagSource });
      if (endpoint === `repos/${repository}/releases/tags/v${version}`) {
        return JSON.stringify({ draft: state.draft, prerelease: false,
          assets: [{ id: 1, name: `tensorviz-vscode-${version}.vsix` }] });
      }
      assert.equal(endpoint, `repos/${repository}/releases/assets/1`);
      return state.github;
    }
    assert.equal(tool, 'npx', 'Never build a new package or modify a GitHub release');
    assert.deepEqual(args.slice(0, 3), ['--yes', 'ovsx@1.3.0', 'publish']);
    assert.deepEqual(args.slice(4), ['--registryUrl', 'https://open-vsx.org']);
    assert.equal(options.env.OVSX_PAT, 'test-token');
    assert.ok(!args.includes('test-token'), 'Credentials must not appear in command arguments');
    assert.deepEqual(readFileSync(args[3]), bytes, 'Publish the exact Marketplace ZIP');
    state.publishCalls.push(args);
    if (state.failUpload) throw new Error('Upload rejected');
    state.uploaded = true;
    return '';
  };
  const request = async url => {
    state.requests.push(url);
    if (url.startsWith('https://marketplace.visualstudio.com/')) {
      assert.ok(url.endsWith(`/vsextensions/tensorviz-vscode/${version}/vspackage`));
      return new Response(state.marketplace, { status: state.marketplaceStatus });
    }
    if (url === downloadUrl) return new Response(state.openvsx, { status: state.downloadStatus });
    assert.equal(url, `https://open-vsx.org/api/tensorviz/tensorviz-vscode/${version}`);
    if (state.lookupStatus !== 200) return new Response('', { status: state.lookupStatus });
    if (state.existing) return Response.json(state.existing);
    if (!state.uploaded || state.neverVisible || state.visibilityDelay-- > 0) return new Response('', { status: 404 });
    return Response.json(info);
  };
  const sync = (options = {}) => syncOpenVsxRelease({ version, repository, token: 'test-token',
    command, request, sleep: async ms => state.sleeps.push(ms), ...options });
  const fromResult = (result, requestedSource = result.source) => {
    const file = join(root, 'release-result.json');
    writeFileSync(file, JSON.stringify(result));
    return syncOpenVsxFromResult({ file, requestedSource, repository, token: 'test-token',
      command, request, sleep: async ms => state.sleeps.push(ms) });
  };
  return { state, bytes, sync, makePackage, fromResult };
}

test('publishes the exact released package and waits for its public checksum before succeeding', async t => {
  const { state, sync, bytes } = fixture(t);
  state.visibilityDelay = 2;
  const result = await sync();
  assert.equal(result.published, true);
  assert.equal(result.version, version);
  assert.equal(result.sha256, createHash('sha256').update(bytes).digest('hex'));
  assert.equal(state.publishCalls.length, 1);
  assert.deepEqual(state.sleeps, [10_000, 10_000]);
  assert.equal(state.requests.at(-1), downloadUrl);
});

test('an identical existing release is a read-only retry, even without a publishing token', async t => {
  const { state, sync } = fixture(t);
  state.existing = info;
  assert.equal((await sync({ token: '' })).published, false);
  assert.equal(state.publishCalls.length, 0);
  assert.deepEqual(state.sleeps, []);
});

test('registry indexing beyond one minute does not fail a successful upload or publish twice', async t => {
  const { state, sync } = fixture(t);
  state.visibilityDelay = 12;
  assert.equal((await sync()).published, true);
  assert.equal(state.publishCalls.length, 1);
  assert.equal(state.sleeps.reduce((sum, ms) => sum + ms, 0), 120_000);
  assert.equal(state.requests.at(-1), downloadUrl);
});

test('a completed run selects its own version from its exact commit rather than current main', async t => {
  const { state, sync } = fixture(t);
  state.existing = info;
  await sync({ version: undefined, source });
  assert.deepEqual(state.sourceLookups, [`repos/${repository}/contents/extension/package.json?ref=${source}`]);
  await sync();
  assert.equal(state.sourceLookups.length, 1, 'An explicit manual version needs no source lookup');
});

test('the originating attempt supplies its exact reviewed source', async t => {
  const { state, fromResult } = fixture(t);
  state.existing = info;
  const requestedSource = source;
  const result = await fromResult({ schema: 1, repository, publish: true, source, version });
  assert.equal(result.version, version);
  assert.deepEqual(state.sourceLookups, [`repos/${repository}/contents/extension/package.json?ref=${source}`]);
});

test('a skipped release hands off its reason without touching any registry or publishing tool', async t => {
  const { state, fromResult } = fixture(t);
  const result = await fromResult({ schema: 1, repository, publish: false, source, version });
  assert.deepEqual(result, { skipped: true, reason: 'The originating run was build-only.' });
  assert.deepEqual(state.requests, []); assert.deepEqual(state.publishCalls, []); assert.deepEqual(state.sourceLookups, []);
});

test('a handoff from another workflow or disagreeing prepared source fails before upload', async t => {
  const { state, fromResult } = fixture(t);
  const result = { schema: 1, repository, publish: true, source, version };
  await assert.rejects(fromResult(result, 'c'.repeat(40)), /trusted source run/);
  assert.deepEqual(state.requests, []);
  state.sourceVersion = '0.1.14';
  await assert.rejects(fromResult(result), /Prepared source and release version disagree/);
  assert.deepEqual(state.requests, []);
  state.sourceVersion = version; state.tagSource = 'd'.repeat(40);
  await assert.rejects(fromResult(result), /Numbered release does not identify the prepared source/);
  assert.deepEqual(state.publishCalls, []);
});

test('conflicting existing Open VSX bytes fail without trying to overwrite them', async t => {
  const { state, sync } = fixture(t);
  state.existing = info;
  state.openvsx = Buffer.from('different published package');
  await assert.rejects(sync(), /Open VSX bytes differ/);
  assert.equal(state.publishCalls.length, 0);
});

test('GitHub and Marketplace must agree before any publication', async t => {
  const { state, sync } = fixture(t);
  state.github = Buffer.from('different GitHub package');
  await assert.rejects(sync(), /GitHub and Marketplace packages differ/);
  assert.equal(state.publishCalls.length, 0);
  assert.equal(state.requests.length, 1, 'Do not proceed to Open VSX with conflicting release bytes');
});

test('missing credentials and unpublished GitHub releases stop before uploading', async t => {
  const { state, sync } = fixture(t);
  await assert.rejects(sync({ token: '' }), /Set the OVSX_PAT Actions secret/);
  state.draft = true;
  await assert.rejects(sync(), /published numbered GitHub VSIX is required/);
  assert.equal(state.publishCalls.length, 0);
});

test('invalid release selectors and mismatched package identities fail before publication', async t => {
  const { state, sync, makePackage, bytes } = fixture(t);
  await assert.rejects(sync({ version: '../latest' }), /stable extension version/);
  await assert.rejects(sync({ version: undefined, source: 'main' }), /full source commit/);
  assert.equal(state.requests.length, 0);
  state.marketplace = makePackage({ publisher: 'another-publisher' });
  await assert.rejects(sync(), /Marketplace package identity/);
  state.marketplace = bytes;
  state.existing = { ...info, version: '0.1.14' };
  await assert.rejects(sync(), /Open VSX package identity/);
  assert.equal(state.publishCalls.length, 0);
});

test('registry errors are not mistaken for an absent version', async t => {
  const { state, sync } = fixture(t);
  state.lookupStatus = 403;
  await assert.rejects(sync(), /Open VSX lookup failed: HTTP 403/);
  assert.equal(state.publishCalls.length, 0);
  state.marketplaceStatus = 404;
  await assert.rejects(sync(), /Package download failed: HTTP 404/);
  assert.equal(state.publishCalls.length, 0);
});

test('upload failures, unavailable downloads and differing uploaded bytes cannot report success', async t => {
  const { state, sync } = fixture(t);
  state.failUpload = true;
  await assert.rejects(sync(), /Upload rejected/);
  assert.deepEqual(state.sleeps, []);
  state.failUpload = false;
  state.existing = info;
  state.downloadStatus = 503;
  await assert.rejects(sync(), /Package download failed: HTTP 503/);
  state.downloadStatus = 200;
  state.existing = null;
  state.openvsx = Buffer.from('wrong uploaded bytes');
  await assert.rejects(sync(), /Open VSX bytes differ/);
});

test('unconfirmed publication has a bounded wait and reports that verification remains incomplete', async t => {
  const { state, sync } = fixture(t);
  state.neverVisible = true;
  await assert.rejects(sync(), /publication is not visible yet/);
  assert.equal(state.publishCalls.length, 1);
  assert.equal(state.sleeps.reduce((sum, ms) => sum + ms, 0), 300_000);
});

test('workflow gates automatic publication on trusted successful main releases and serializes retries', () => {
  const workflow = readFileSync(new URL('../../.github/workflows/release-openvsx.yml', import.meta.url), 'utf8');
  assert.match(workflow, /workflows: \[Release VSIX\]/);
  assert.match(workflow, /types: \[completed\]/);
  assert.match(workflow, /branches: \[main\]/);
  assert.match(workflow, /github\.event\.workflow_run\.conclusion == 'success'/);
  assert.match(workflow, /github\.event\.workflow_run\.head_repository\.full_name == github\.repository/);
  assert.match(workflow, /github\.event_name == 'workflow_dispatch' && github\.ref == 'refs\/heads\/main'/);
  assert.match(workflow, /ref: \$\{\{ github\.sha \}\}/);
  assert.match(workflow, /persist-credentials: false/);
  assert.match(workflow, /contents: read/);
  assert.match(workflow, /cancel-in-progress: false/);
  assert.match(workflow, /OVSX_PAT: \$\{\{ secrets\.OVSX_PAT \}\}/);
  assert.match(workflow, /actions: read/);
  assert.match(workflow, /uses: actions\/download-artifact@v4/);
  assert.match(workflow, /name: extension-release-\$\{\{ github\.event\.workflow_run\.id \}\}/);
  assert.match(workflow, /run-id: \$\{\{ github\.event\.workflow_run\.id \}\}/);
  assert.match(workflow, /RELEASE_REQUEST_SOURCE: \$\{\{ github\.event\.workflow_run\.head_sha \}\}/);
  assert.ok(!workflow.includes('RELEASE_SOURCE:'), 'The attempt artifact provides the verified package source');
  assert.match(workflow, /run: node extension\/scripts\/sync-openvsx-release\.mjs/);
  assert.ok(!workflow.includes('continue-on-error'));
});
