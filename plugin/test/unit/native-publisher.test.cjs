const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const { readFileSync, mkdirSync, writeFileSync } = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { createHash } = require('node:crypto');
const sha = 'a'.repeat(40), other = 'b'.repeat(40), tag = 'v1.2.3', exportSHA = 'c'.repeat(40), previousSHA = 'd'.repeat(40);
const digest = bytes => createHash('sha256').update(bytes).digest('hex');

// Isolate publisher orchestration from archive fixtures. The production verifier
// is not replaceable by CLI flags; candidate/platform/license audits have their
// own archive tests. No packages, network, GitHub token or git checkout required.
async function fixture(t, options = {}) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'native-publisher-test-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const previous = process.env.TLDRAW_DISTRIBUTION_CONFIRMED;
  process.env.GH_TOKEN = 'distribution-test'; process.env.TENSORVIZ_SOURCE_TOKEN = 'source-test';
  t.after(() => { delete process.env.GH_TOKEN; delete process.env.TENSORVIZ_SOURCE_TOKEN; });
  process.env.TLDRAW_DISTRIBUTION_CONFIRMED = options.license === false ? 'false' : 'true';
  t.after(() => { if (previous === undefined) delete process.env.TLDRAW_DISTRIBUTION_CONFIRMED; else process.env.TLDRAW_DISTRIBUTION_CONFIRMED = previous; });
  const script = path.resolve(__dirname, '../../scripts/publish-native-release.mjs');
  await fs.copyFile(script, path.join(root, 'publisher.mjs'));
  await fs.copyFile(path.resolve(__dirname, '../../scripts/version-alignment.mjs'), path.join(root, 'version-alignment.mjs'));
  await fs.copyFile(path.resolve(__dirname, '../../scripts/release-migration.mjs'), path.join(root, 'release-migration.mjs'));
  await fs.writeFile(path.join(root, 'release-version.mjs'), "export const distributionRepository = 'tensorviz-org/tensorviz-plugins';\n");
  await fs.writeFile(path.join(root, 'release-candidate.mjs'), `
    import { createHash } from 'node:crypto'; import { readFile } from 'node:fs/promises'; import path from 'node:path';
    export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
    export async function auditIntegration(directory) { return JSON.parse(await readFile(path.join(directory, 'PUBLICATION.json'))); }
    export async function verifyReleaseCandidate(directory, options) {
      if (!options.publishing) throw new Error('publishing audit omitted');
      const release = JSON.parse(await readFile(path.join(directory, 'release.json')));
      if (release.rejectAudit) throw new Error('candidate audit failed');
      return release;
    }
  `);
  const publication = { version: '1.2.3', sourceRevision: options.source ?? sha, files: {}, runtime: {} };
  await fs.mkdir(path.join(root, 'integration'));
  await fs.writeFile(path.join(root, 'integration/PUBLICATION.json'), JSON.stringify(publication));
  const payload = Buffer.from('audited plugin ZIP');
  const release = { version: '1.2.3', sourceRevision: options.source ?? sha, files: { 'tensorviz-1.2.3.zip': digest(payload) }, publication, rejectAudit: options.rejectAudit };
  const manifest = Buffer.from(JSON.stringify(release));
  await fs.writeFile(path.join(root, 'release.json'), manifest);
  await fs.writeFile(path.join(root, 'tensorviz-1.2.3.zip'), payload);
  const contents = { 'tensorviz-1.2.3.zip': payload, 'release.json': manifest };
  let ref = (options.existing || options.orphanTag) ? { object: { type: options.tagType ?? 'commit', sha: options.tagSHA ?? exportSHA } } : null;
  let hosted = options.existing ? { id: 7, tag_name: tag, target_commitish: options.target ?? exportSHA, draft: options.draft !== false } : null;
  const assets = options.assets ?? (options.existing ? Object.keys(contents).map((name, i) => ({ name, id: i + 1 })) : []);
  const calls = [], downloads = [];
  const notFound = () => { throw Object.assign(new Error('404'), { stderr: 'HTTP 404: Not Found' }); };
  let checkout, detached = false, committed = false;
  const command = (tool, args, settings) => {
    calls.push([tool, ...args]);
    assert.equal(settings.env.TENSORVIZ_SOURCE_TOKEN, undefined);
    if (tool === 'git') {
      assert.equal(settings.env.GH_TOKEN, 'distribution-test');
      if (args[0] === 'rev-parse') return options.head ?? sha;
      if (args[0] === 'show') return JSON.stringify({ version: options.version ?? '1.2.3' });
      const command = args[2];
      if (command === 'clone') {
        checkout = args.at(-1); mkdirSync(checkout);
        mkdirSync(path.join(checkout, '.git'));
        writeFileSync(path.join(checkout, 'PUBLICATION.json'), JSON.stringify({ ...publication, version: options.newer ? '2.0.0' : options.promoted ? '1.2.3' : '1.2.2' }));
        return '';
      }
      if (command === 'rev-parse') return detached || committed || options.promoted ? exportSHA : previousSHA;
      if (command === 'checkout') {
        detached = true;
        writeFileSync(path.join(checkout, 'PUBLICATION.json'), JSON.stringify(options.tagMismatch ? { ...publication, sourceRevision: other } : publication));
        return '';
      }
      if (args.includes('commit')) { committed = true; return ''; }
      if (command === 'push') {
        if (args.at(-1).includes('refs/tags')) ref = { object: { type: 'commit', sha: exportSHA } };
        if (args.at(-1).includes('refs/heads') && options.pushRace) throw new Error('non-fast-forward');
        return '';
      }
      if (['fetch', 'add'].includes(command)) return '';
      throw new Error(`Unexpected git operation: ${args}`);
    }
    assert.equal(tool, 'gh');
    if (args[0] === 'release') {
      if (args[1] === 'upload') {
        const name = path.basename(args[3]); assets.push({ name, id: assets.length + 1 }); return '';
      }
      if (args[1] === 'edit') { hosted.draft = false; return ''; }
    }
    assert.equal(args[0], 'api');
    assert.equal(args[2], 'github.com');
    if (args[3].includes('/tensorviz/compare/')) {
      assert.equal(settings.env.GH_TOKEN, 'source-test');
      return JSON.stringify({ status: options.status ?? 'ahead', merge_base_commit: { sha: options.mergeBase ?? sha } });
    }
    assert.equal(settings.env.GH_TOKEN, 'distribution-test');
    const repositoryEndpoint = 'repos/tensorviz-org/tensorviz-plugins';
    // GitHub rejects the repository metadata endpoint with a trailing slash.
    if (args[3] === `${repositoryEndpoint}/`) return notFound();
    const endpoint = args[3] === repositoryEndpoint ? '' : args[3].replace(`${repositoryEndpoint}/`, '');
    const input = args.includes('--input') ? JSON.parse(readFileSync(args[args.indexOf('--input') + 1], 'utf8')) : null;
    const json = JSON.stringify;
    if (endpoint === '') return json({ private: !!options.privateDistribution, default_branch: 'main' });
    if (endpoint.startsWith('compare/')) return json({ status: options.status ?? 'ahead', merge_base_commit: { sha: options.mergeBase ?? sha } });
    if (endpoint === `git/ref/tags/${tag}`) return ref ? json(ref) : notFound();
    if (endpoint === 'git/ref/heads/main') return json({ object: { sha: options.race ? other : options.promoted ? exportSHA : previousSHA } });
    if (endpoint === 'git/refs') { assert.deepEqual(input, { ref: `refs/tags/${tag}`, sha }); ref = { object: { type: 'commit', sha } }; return json(ref); }
    if (endpoint === `releases/tags/${tag}`) {
      if (options.authFailure) throw Object.assign(new Error('forbidden'), { stderr: 'HTTP 403: Forbidden' });
      return hosted && !hosted.draft ? json(hosted) : notFound();
    }
    if (endpoint === 'releases?per_page=100') return json([options.duplicate ? [hosted, hosted] : hosted ? [hosted] : []]);
    if (endpoint === 'releases') { assert.equal(input.target_commitish, exportSHA); assert.equal(input.tag_name, tag); hosted = { ...input, id: 7 }; return json(hosted); }
    if (endpoint === 'releases/7/assets?per_page=100') return json([assets]);
    if (endpoint.startsWith('releases/assets/')) {
      const asset = assets.find(a => a.id === Number(endpoint.split('/').at(-1)));
      return options.corrupt ? Buffer.from('different') : contents[asset.name];
    }
    throw new Error(`Unexpected API call: ${endpoint}`);
  };
  const download = async url => { downloads.push(url); return { ok: !options.downloadFailure, arrayBuffer: async () => contents[url.split('/').at(-1)] }; };
  const { publishRelease } = await import(pathToFileURL(path.join(root, 'publisher.mjs')));
  return { run: () => publishRelease(root, { command, download, reviewedSource: options.reviewed ?? sha }), calls, downloads };
}
const writes = calls => calls.filter(c => c.includes('POST') || c[1] === 'release');

test('uses the canonical repository metadata endpoint without a trailing slash', async t => {
  const f = await fixture(t); await f.run();
  const metadata = f.calls.filter(c => c[0] === 'gh' && c[1] === 'api' && /^repos\/tensorviz-org\/tensorviz-plugins\/?$/.test(c[4]));
  assert.equal(metadata.length, 1);
  assert.equal(metadata[0][4], 'repos/tensorviz-org/tensorviz-plugins');
});

test('publishes only public export and advances catalog after all downloads', async t => {
  const f = await fixture(t); const result = await f.run();
  assert.equal(result.commit, exportSHA); assert.match(result.release, /\/v1.2.3$/);
  assert.equal(f.calls.filter(c => c[1] === 'release' && c[2] === 'upload').length, 2);
  assert.equal(f.downloads.length, 2);
  assert.ok(f.calls.some(c => c.includes(`${exportSHA}:refs/heads/main`)));
  assert.ok(!f.calls.some(c => c.some(v => /clobber|force|tensorviz\.git/.test(v))));
});
for (const [name, options, error] of [
  ['does not mistake authorization errors for absent releases', { authFailure: true }, /forbidden/],
  ['requires owner license confirmation', { license: false }, /coverage/],
  ['requires exact reviewed SHA', { reviewed: 'main' }, /exact reviewed/],
  ['audits before any write', { rejectAudit: true }, /audit failed/],
  ['rejects different candidate source', { source: other }, /Candidate/],
  ['rejects different checkout', { head: other }, /Checkout/],
  ['rejects unreviewed version override', { version: '1.2.2' }, /checked-in/],
  ['rejects non-main source', { status: 'diverged' }, /main history/],
  ['checks main merge base', { mergeBase: other }, /main history/],
  ['rejects moved tag', { existing: true, tagSHA: other }, /Existing release tag/],
  ['rejects annotated tag indirection', { existing: true, tagType: 'tag' }, /Existing release tag/],
  ['rejects different release target', { existing: true, target: other }, /provenance/],
  ['rejects duplicate draft releases', { existing: true, duplicate: true }, /Multiple releases/],
  ['never overwrites different hosted bytes', { existing: true, corrupt: true }, /Hosted asset differs/],
  ['never adds missing published assets', { existing: true, draft: false, assets: [] }, /Published release is missing/],
  ['rejects unrecognized assets', { existing: true, assets: [{ name: 'extra', id: 1 }] }, /Unexpected or duplicate/],
  ['rejects duplicate asset names', { existing: true, assets: [{ name: 'release.json', id: 1 }, { name: 'release.json', id: 2 }] }, /Unexpected or duplicate/],
]) test(name, async t => { const f = await fixture(t, options); await assert.rejects(f.run, error); assert.deepEqual(writes(f.calls), []); });

test('recovers authenticated draft and does not replace existing assets', async t => {
  const f = await fixture(t, { existing: true }); await f.run();
  assert.ok(f.calls.some(c => c.some(v => v.includes('releases?per_page=100'))));
  assert.equal(writes(f.calls).length, 1); assert.equal(writes(f.calls)[0][2], 'edit');
});
test('published release retry is read-only and rechecks public downloads', async t => {
  const f = await fixture(t, { existing: true, draft: false, promoted: true }); await f.run();
  assert.deepEqual(writes(f.calls), []); assert.equal(f.downloads.length, 2);
});
test('public verification failures are actionable and never push source main', async t => {
  const f = await fixture(t, { downloadFailure: true }); await assert.rejects(f.run, /Public asset verification failed/);
  assert.ok(!f.calls.some(c => c.some(v => v.includes('refs/heads/main'))));
});

test('resumes a partial draft by uploading only missing assets', async t => {
  const f = await fixture(t, { existing: true, assets: [{ name: 'tensorviz-1.2.3.zip', id: 1 }] });
  await f.run();
  const uploads = f.calls.filter(c => c[1] === 'release' && c[2] === 'upload');
  assert.equal(uploads.length, 1); assert.equal(path.basename(uploads[0][4]), 'release.json');
});
test('recovers interruption after tag creation without moving the tag', async t => {
  const f = await fixture(t, { orphanTag: true }); await f.run();
  assert.equal(f.calls.filter(c => c.includes('push') && c.some(v => v.includes('refs/tags'))).length, 0);
  assert.equal(f.calls.filter(c => c.includes('POST') && c.some(v => v.endsWith('/releases'))).length, 1);
});

for (const [name, options, error] of [
  ['rejects wrong exported provenance', { existing: true, tagMismatch: true }, /exported bytes/],
  ['does not downgrade catalog', { newer: true }, /older release/],
  ['requires public distribution not public source', { privateDistribution: true }, /Distribution repository/],
  ['detects concurrent catalog change', { race: true }, /changed during/],
  ['non-fast-forward catalog promotion fails closed', { pushRace: true }, /non-fast-forward/],
]) test(name, async t => { const f = await fixture(t, options); await assert.rejects(f.run, error); });


