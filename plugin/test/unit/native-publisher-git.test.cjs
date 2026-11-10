const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const { readFileSync } = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { pathToFileURL } = require('node:url');
const { createHash } = require('node:crypto');

// Exercise real Git object transport, export commits/tags and idempotent catalog
// promotion. Only GitHub HTTP and candidate auditing are substituted here; the
// latter has separate real archive/closed-inventory tests.
test('real Git publishes only public export history and safely retries', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'native-publisher-git-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const saved = {};
  for (const [key, value] of Object.entries({ TLDRAW_DISTRIBUTION_CONFIRMED: 'true', TENSORVIZ_SOURCE_TOKEN: 'source-read-fixture', GH_TOKEN: 'distribution-write-fixture' })) {
    saved[key] = process.env[key]; process.env[key] = value;
  }
  t.after(() => { for (const [key, value] of Object.entries(saved)) { if (value === undefined) delete process.env[key]; else process.env[key] = value; } });
  const git = (cwd, ...args) => execFileSync('git', args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
  const init = async name => { const dir = path.join(root, name); await fs.mkdir(dir); git(dir, 'init', '-q', '-b', 'main'); git(dir, 'config', 'user.name', 'Fixture'); git(dir, 'config', 'user.email', 'fixture@example.invalid'); return dir; };
  const source = await init('private-source');
  await fs.writeFile(path.join(source, 'private-core.txt'), 'PRIVATE SOURCE MUST NEVER BE EXPORTED');
  git(source, 'add', '.'); git(source, 'commit', '-qm', 'private source');
  const sourceSha = git(source, 'rev-parse', 'HEAD');
  const seed = await init('public-seed');
  const old = { version: '1.2.2', sourceRevision: 'a'.repeat(40), files: {} };
  await fs.writeFile(path.join(seed, 'PUBLICATION.json'), JSON.stringify(old));
  await fs.writeFile(path.join(seed, 'README.md'), 'Previous public catalog');
  git(seed, 'add', '.'); git(seed, 'commit', '-qm', 'public export');
  const oldSha = git(seed, 'rev-parse', 'HEAD');
  const remote = path.join(root, 'public.git');
  git(root, 'clone', '--bare', '-q', seed, remote);
  const candidate = path.join(root, 'candidate');
  await fs.mkdir(path.join(candidate, 'integration'), { recursive: true });
  const publication = { version: '1.2.3', sourceRevision: sourceSha, files: { 'README.md': 'fixture' } };
  await fs.writeFile(path.join(candidate, 'integration', 'PUBLICATION.json'), JSON.stringify(publication));
  await fs.writeFile(path.join(candidate, 'integration', 'README.md'), 'New public catalog');
  const payload = Buffer.from('audited ZIP fixture');
  const digest = bytes => createHash('sha256').update(bytes).digest('hex');
  const release = { version: '1.2.3', sourceRevision: sourceSha, files: { 'tensorviz-1.2.3.zip': digest(payload) }, publication };
  await fs.writeFile(path.join(candidate, 'release.json'), JSON.stringify(release));
  await fs.writeFile(path.join(candidate, 'tensorviz-1.2.3.zip'), payload);
  const moduleDir = path.join(root, 'module'); await fs.mkdir(moduleDir);
  await fs.copyFile(path.resolve(__dirname, '../../scripts/publish-native-release.mjs'), path.join(moduleDir, 'publisher.mjs'));
  await fs.writeFile(path.join(moduleDir, 'release-version.mjs'), "export const distributionRepository = 'tensorviz-org/tensorviz-plugins';\n");
  await fs.writeFile(path.join(moduleDir, 'release-candidate.mjs'), `
    import { readFile } from 'node:fs/promises'; import path from 'node:path'; import { createHash } from 'node:crypto';
    export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
    export const auditIntegration = async dir => JSON.parse(await readFile(path.join(dir, 'PUBLICATION.json')));
    export const verifyReleaseCandidate = async dir => JSON.parse(await readFile(path.join(dir, 'release.json')));
  `);
  let hosted = null; const assets = [];
  const missing = () => { throw Object.assign(new Error('missing'), { stderr: 'HTTP 404: Not Found' }); };
  const command = (tool, args, options = {}) => {
    if (tool === 'git') {
      if (args[0] === 'rev-parse') return `${sourceSha}\n`;
      if (args[0] === 'show') return JSON.stringify({ version: '1.2.3' });
      assert.equal(options.env.GH_TOKEN, 'distribution-write-fixture');
      assert.equal(options.env.TENSORVIZ_SOURCE_TOKEN, undefined);
      const actual = args.map(arg => arg === 'https://github.com/tensorviz-org/tensorviz-plugins.git' ? remote : arg);
      return execFileSync('git', actual, { ...options, stdio: ['ignore', 'pipe', 'pipe'] });
    }
    assert.equal(tool, 'gh');
    if (args[0] === 'release') {
      if (args[1] === 'upload') { assets.push({ id: assets.length + 1, name: path.basename(args[3]), bytes: readFileSync(args[3]) }); return ''; }
      if (args[1] === 'edit') { hosted.draft = false; return ''; }
      throw new Error(`Unexpected release command: ${args}`);
    }
    const endpoint = args[3];
    if (endpoint.startsWith('repos/tensorviz-org/tensorviz/compare/')) {
      assert.equal(options.env.GH_TOKEN, 'source-read-fixture');
      return JSON.stringify({ status: 'identical', merge_base_commit: { sha: sourceSha } });
    }
    assert.equal(options.env.GH_TOKEN, 'distribution-write-fixture');
    const repositoryEndpoint = 'repos/tensorviz-org/tensorviz-plugins';
    if (endpoint === `${repositoryEndpoint}/`) return missing();
    const api = endpoint === repositoryEndpoint ? '' : endpoint.replace(`${repositoryEndpoint}/`, '');
    if (!api) return JSON.stringify({ private: false, default_branch: 'main' });
    if (api.startsWith('git/ref/')) {
      try { return JSON.stringify({ object: { type: 'commit', sha: git(remote, 'rev-parse', '--verify', `refs/${api.slice(8)}`) } }); } catch { return missing(); }
    }
    if (api === 'releases/tags/v1.2.3') return hosted && !hosted.draft ? JSON.stringify(hosted) : missing();
    if (api === 'releases?per_page=100') return JSON.stringify([hosted ? [hosted] : []]);
    if (api === 'releases') { hosted = { ...JSON.parse(readFileSync(args[args.indexOf('--input') + 1], 'utf8')), id: 1 }; return JSON.stringify(hosted); }
    if (api === 'releases/1/assets?per_page=100') return JSON.stringify([assets.map(({ id, name }) => ({ id, name }))]);
    if (api.startsWith('releases/assets/')) return assets.find(a => a.id === Number(api.split('/').pop())).bytes;
    throw new Error(`Unexpected API: ${api}`);
  };
  const download = async url => ({ ok: true, arrayBuffer: async () => assets.find(a => a.name === url.split('/').pop()).bytes });
  const { publishRelease } = await import(pathToFileURL(path.join(moduleDir, 'publisher.mjs')));
  const first = await publishRelease(candidate, { command, download, reviewedSource: sourceSha });
  assert.notEqual(first.commit, sourceSha);
  assert.equal(git(remote, 'rev-parse', 'refs/heads/main'), first.commit);
  assert.equal(git(remote, 'rev-parse', 'refs/tags/v1.2.3'), first.commit);
  assert.equal(git(remote, 'rev-parse', `${first.commit}^`), oldSha);
  assert.deepEqual(git(remote, 'ls-tree', '-r', '--name-only', first.commit).split('\n'), ['PUBLICATION.json', 'README.md']);
  assert.throws(() => git(remote, 'cat-file', '-e', sourceSha));
  assert.equal(git(source, 'rev-parse', 'HEAD'), sourceSha);
  const retried = await publishRelease(candidate, { command, download, reviewedSource: sourceSha });
  assert.equal(retried.commit, first.commit);
  assert.equal(assets.length, 2);
});
