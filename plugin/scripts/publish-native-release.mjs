import { execFileSync } from 'node:child_process';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import { auditIntegration, sha256, verifyReleaseCandidate } from './release-candidate.mjs';
import { distributionRepository } from './release-version.mjs';

const sourceRoot = fileURLToPath(new URL('../..', import.meta.url));
const sourceRepository = 'tensorviz-org/tensorviz';
export function newerVersion(left, right) {
  const a = left.split('.').map(Number), b = right.split('.').map(Number);
  for (let i = 0; i < 3; i++) if (a[i] !== b[i]) return a[i] > b[i];
  return false;
}

/** Publish only the audited export. Never push private source objects/history. */
export async function publishRelease(directory, {
  command = execFileSync, download = fetch,
  reviewedSource = process.env.TENSORVIZ_REVIEWED_SOURCE_SHA,
} = {}) {
  if (process.env.TLDRAW_DISTRIBUTION_CONFIRMED !== 'true') throw new Error('Renderer distribution coverage still needs owner confirmation.');
  if (!/^[a-f0-9]{40}$/.test(reviewedSource ?? '')) throw new Error('An exact reviewed source SHA is required.');
  if (!process.env.TENSORVIZ_SOURCE_TOKEN || !process.env.GH_TOKEN) throw new Error('Separate source-read and distribution-write tokens are required.');
  directory = path.resolve(directory);
  const release = await verifyReleaseCandidate(directory, { publishing: true });
  if (release.sourceRevision !== reviewedSource) throw new Error('Candidate does not match the reviewed source SHA.');
  const repo = distributionRepository, tag = `v${release.version}`;
  if (repo !== 'tensorviz-org/tensorviz-plugins') throw new Error('Unexpected public distribution repository.');
  const env = { ...process.env };
  delete env.TENSORVIZ_SOURCE_TOKEN;
  // Prevent hooks or inherited Git state redirecting the separate public checkout.
  for (const name of execFileSync('git', ['rev-parse', '--local-env-vars'], { encoding: 'utf8' }).trim().split('\n')) delete env[name];
  const run = (tool, args, options = {}) => command(tool, args, {
    encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, cwd: sourceRoot, env, ...options,
  });
  const api = (endpoint, args = []) => JSON.parse(run('gh', ['api', '--hostname', 'github.com', `repos/${repo}${endpoint ? `/${endpoint}` : ''}`, ...args]));
  const optional = callback => {
    try { return callback(); } catch (error) {
      if (/HTTP 404\b/.test(String(error.stderr))) return null;
      throw error;
    }
  };
  if (run('git', ['rev-parse', 'HEAD']).trim() !== reviewedSource) throw new Error('Checkout does not match reviewed source SHA.');
  const sourceVersion = JSON.parse(run('git', ['show', `${reviewedSource}:plugin/package.json`])).version;
  if (sourceVersion !== release.version) throw new Error('Release version must equal the reviewed, checked-in plugin version.');
  const comparison = JSON.parse(run('gh', ['api', '--hostname', 'github.com', `repos/${sourceRepository}/compare/${reviewedSource}...main`],
    { env: { ...env, GH_TOKEN: process.env.TENSORVIZ_SOURCE_TOKEN } }));
  if (!['ahead', 'identical'].includes(comparison.status) || comparison.merge_base_commit?.sha !== reviewedSource) {
    throw new Error('Reviewed source must belong to source main history.');
  }
  const info = api('');
  if (info.private || info.default_branch !== 'main') throw new Error('Distribution repository must be public with main as its default branch.');
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-publish-'));
  const checkout = path.join(temporary, 'integration');
  const git = args => run('git', ['-c', 'credential.helper=!gh auth git-credential', ...args], { cwd: checkout });
  try {
    run('git', ['-c', 'credential.helper=!gh auth git-credential', 'clone', '--quiet', '--depth', '1', '--branch', 'main', `https://github.com/${repo}.git`, checkout]);
    const initialMain = git(['rev-parse', 'HEAD']).trim();
    const auditCheckout = async name => {
      const copy = path.join(temporary, name);
      await fs.cp(checkout, copy, { recursive: true, filter: source => path.basename(source) !== '.git' });
      return auditIntegration(copy, { allowPrevious: true });
    };
    const previous = await auditCheckout('previous');
    if (newerVersion(previous.version, release.version)) throw new Error('Refusing to roll the public catalog back to an older release.');
    const existingTag = optional(() => api(`git/ref/tags/${tag}`));
    let commit;
    if (existingTag) {
      if (existingTag.object?.type !== 'commit') throw new Error('Existing release tag must identify an export commit.');
      git(['fetch', '--quiet', 'origin', `refs/tags/${tag}:refs/tags/${tag}`]);
      git(['checkout', '--quiet', '--detach', tag]);
      commit = git(['rev-parse', 'HEAD']).trim();
      if (commit !== existingTag.object.sha) throw new Error('Existing release tag moved during verification.');
      const publication = await auditCheckout('tagged');
      if (!isDeepStrictEqual(publication, release.publication)) throw new Error('Existing release tag provenance or exported bytes differ.');
    } else {
      if (previous.version === release.version) throw new Error('Current catalog version has no matching immutable tag.');
      for (const item of await fs.readdir(checkout)) if (item !== '.git') await fs.rm(path.join(checkout, item), { recursive: true, force: true });
      await fs.cp(path.join(directory, 'integration'), checkout, { recursive: true });
      git(['add', '--all']);
      git(['-c', 'user.name=TensorViz releases', '-c', 'user.email=41898282+github-actions[bot]@users.noreply.github.com', 'commit', '--quiet', '-m', `Release TensorViz ${release.version}`]);
      commit = git(['rev-parse', 'HEAD']).trim();
    }
    const findRelease = () => {
      const published = optional(() => api(`releases/tags/${tag}`));
      if (published) return published;
      const matches = api('releases?per_page=100', ['--paginate', '--slurp']).flat().filter(item => item.tag_name === tag);
      if (matches.length > 1) throw new Error(`Multiple releases exist for ${tag}.`);
      return matches[0] ?? null;
    };
    let hosted = findRelease();
    if (hosted && (!existingTag || hosted.tag_name !== tag || hosted.target_commitish !== commit)) throw new Error('Existing release provenance does not match the public export commit.');
    if (!existingTag) git(['push', '--quiet', 'origin', `HEAD:refs/tags/${tag}`]);
    if (!hosted) {
      const request = path.join(temporary, 'request.json');
      await fs.writeFile(request, JSON.stringify({ tag_name: tag, target_commitish: commit, draft: true, prerelease: true,
        name: `TensorViz plugin ${release.version}`,
        body: `TensorViz plugin ${release.version} preview with compiled platform runtimes and CPython 3.12.\n\nInstall from https://github.com/${repo}. Complete local catalog and standalone client ZIPs are also attached. Core source is excluded; browser JavaScript remains inspectable.\n\nSource repository: ${sourceRepository}\nSource revision: ${reviewedSource}\nPublic export commit: ${commit}\n`,
      }));
      hosted = api('releases', ['--method', 'POST', '--input', request]);
    }
    const files = { ...release.files, 'release.json': sha256(await fs.readFile(path.join(directory, 'release.json'))) };
    const assets = api(`releases/${hosted.id}/assets?per_page=100`, ['--paginate', '--slurp']).flat();
    const seen = new Set();
    for (const asset of assets) {
      if (!Object.hasOwn(files, asset.name) || seen.has(asset.name)) throw new Error(`Unexpected or duplicate hosted asset: ${asset.name}`);
      seen.add(asset.name);
      const bytes = run('gh', ['api', '--hostname', 'github.com', `repos/${repo}/releases/assets/${asset.id}`, '-H', 'Accept: application/octet-stream'],
        { encoding: 'buffer', maxBuffer: 256 * 1024 * 1024 });
      if (sha256(bytes) !== files[asset.name]) throw new Error(`Hosted asset differs: ${asset.name}. Published bytes are never overwritten.`);
    }
    for (const name of Object.keys(files)) {
      if (seen.has(name)) continue;
      if (!hosted.draft) throw new Error(`Published release is missing ${name}; refusing to modify it.`);
      run('gh', ['release', 'upload', tag, path.join(directory, name), '--repo', repo]);
    }
    const assertTag = () => {
      const ref = api(`git/ref/tags/${tag}`);
      if (ref.object?.type !== 'commit' || ref.object.sha !== commit) throw new Error('Existing release tag moved during publication.');
    };
    assertTag();
    if (hosted.draft) run('gh', ['release', 'edit', tag, '--draft=false', '--repo', repo]);
    for (const [name, digest] of Object.entries(files)) {
      const response = await download(`https://github.com/${repo}/releases/download/${tag}/${name}`);
      const bytes = response.ok ? Buffer.from(await response.arrayBuffer()) : null;
      if (!bytes || sha256(bytes) !== digest) throw new Error(`Public asset verification failed: ${name}. Catalog not advanced; retry with the original candidate.`);
    }
    assertTag();
    // Refuse any concurrent catalog change. Never force-push or update source main.
    const currentMain = api('git/ref/heads/main').object?.sha;
    if (currentMain !== initialMain) throw new Error('Public catalog changed during publication; recheck before retrying.');
    if (currentMain !== commit) git(['push', '--quiet', 'origin', `${commit}:refs/heads/main`]);
    return { version: release.version, repository: `https://github.com/${repo}`, release: `https://github.com/${repo}/releases/tag/${tag}`, commit, sourceRevision: reviewedSource };
  } finally { await fs.rm(temporary, { recursive: true, force: true }); }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3) throw new Error('Usage: node publish-native-release.mjs CANDIDATE_DIRECTORY');
  console.log(JSON.stringify(await publishRelease(process.argv[2])));
}
