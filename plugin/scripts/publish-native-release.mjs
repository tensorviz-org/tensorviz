import { execFileSync } from 'node:child_process';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { sha256, verifyReleaseCandidate } from './release-candidate.mjs';
import { distributionRepository } from './release-version.mjs';

const sourceRoot = fileURLToPath(new URL('../..', import.meta.url));

/** Release audited assets against a reviewed SOURCE commit. Never write a branch. */
export async function publishRelease(directory, {
  command = execFileSync, download = fetch,
  reviewedSource = process.env.TENSORVIZ_REVIEWED_SOURCE_SHA,
} = {}) {
  if (process.env.TLDRAW_DISTRIBUTION_CONFIRMED !== 'true') throw new Error('Renderer distribution coverage still needs owner confirmation.');
  if (!/^[a-f0-9]{40}$/.test(reviewedSource ?? '')) throw new Error('An exact reviewed source SHA is required.');
  directory = path.resolve(directory);
  // This audits the closed ZIP inventory, platform locks, every checksum, archive
  // contents, provenance, distribution URLs and renderer coverage before writes.
  const release = await verifyReleaseCandidate(directory, { publishing: true });
  if (release.sourceRevision !== reviewedSource) throw new Error('Candidate does not match the reviewed source SHA.');
  const repo = distributionRepository, tag = `plugin-v${release.version}`;
  if (repo !== 'tensorviz-org/tensorviz') throw new Error('Native releases must use the source repository.');
  const run = (tool, args, options = {}) => command(tool, args, {
    encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, cwd: sourceRoot, ...options,
  });
  const api = (endpoint, args = []) => JSON.parse(run('gh', ['api', '--hostname', 'github.com', `repos/${repo}/${endpoint}`, ...args]));
  const optional = callback => {
    try { return callback(); } catch (error) {
      if (/HTTP 404/.test(String(error.stderr))) return null;
      throw error;
    }
  };
  if (run('git', ['rev-parse', 'HEAD']).trim() !== reviewedSource) throw new Error('Checkout does not match reviewed source SHA.');
  const sourceVersion = JSON.parse(run('git', ['show', `${reviewedSource}:plugin/package.json`])).version;
  if (sourceVersion !== release.version) throw new Error('Release version must equal the reviewed, checked-in plugin version.');
  const info = api('');
  if (info.private || info.default_branch !== 'main') throw new Error('Source repository must be public with main as its default branch.');
  const comparison = api(`compare/${reviewedSource}...main`);
  if (!['ahead', 'identical'].includes(comparison.status) || comparison.merge_base_commit?.sha !== reviewedSource) {
    throw new Error('Reviewed source must belong to source main history.');
  }
  const findRelease = () => {
    const published = optional(() => api(`releases/tags/${tag}`));
    if (published) return published;
    // The by-tag endpoint may omit drafts. Authenticated pagination recovers a
    // partially uploaded draft without silently creating a duplicate release.
    const matches = api('releases?per_page=100', ['--paginate', '--slurp']).flat().filter(item => item.tag_name === tag);
    if (matches.length > 1) throw new Error(`Multiple releases exist for ${tag}.`);
    return matches[0] ?? null;
  };
  const assertTag = ref => {
    if (ref?.object?.type !== 'commit' || ref.object.sha !== reviewedSource) throw new Error('Existing release tag does not identify the exact reviewed source commit.');
  };
  const existingTag = optional(() => api(`git/ref/tags/${tag}`));
  if (existingTag) assertTag(existingTag);
  let hosted = findRelease();
  if (hosted && (!existingTag || hosted.tag_name !== tag || hosted.target_commitish !== reviewedSource)) {
    throw new Error('Existing release provenance does not match the reviewed source commit.');
  }
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-publish-'));
  try {
    const post = async (endpoint, body) => {
      const request = path.join(temporary, 'request.json');
      await fs.writeFile(request, JSON.stringify(body));
      return api(endpoint, ['--method', 'POST', '--input', request]);
    };
    if (!existingTag) assertTag(await post('git/refs', { ref: `refs/tags/${tag}`, sha: reviewedSource }));
    if (!hosted) {
      hosted = await post('releases', { tag_name: tag, target_commitish: reviewedSource, draft: true, prerelease: true,
        name: `TensorViz plugin ${release.version}`,
        body: `TensorViz plugin ${release.version} preview with compiled platform runtimes and CPython 3.12.\n\n` +
          `Download and extract tensorviz-catalog-${release.version}.zip to install the complete local Claude/Codex catalog. ` +
          `For a standalone client plugin directory use tensorviz-${release.version}.zip or tensorviz-cursor-${release.version}.zip. ` +
          `These packages include pinned runtime locks. The source repository main branch is not a generated plugin catalog.\n\n` +
          `The plugin downloads its verified compiled runtime automatically. VS Code and Node.js are not required.\n\n` +
          `Source revision: ${reviewedSource}\n`,
      });
    }
    // Bind recovery to the exact original candidate, including its provenance.
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
    // Fail closed if the tag was moved during upload. No force updates, deletes,
    // catalog commits or source branch pushes are ever issued by this publisher.
    assertTag(api(`git/ref/tags/${tag}`));
    if (hosted.draft) run('gh', ['release', 'edit', tag, '--draft=false', '--repo', repo]);
    for (const [name, digest] of Object.entries(files)) {
      const response = await download(`https://github.com/${repo}/releases/download/${tag}/${name}`);
      const bytes = response.ok ? Buffer.from(await response.arrayBuffer()) : null;
      if (!bytes || sha256(bytes) !== digest) throw new Error(`Public asset verification failed: ${name}. Retry with the original candidate; do not rebuild this version.`);
    }
    return { version: release.version, repository: `https://github.com/${repo}`, release: `https://github.com/${repo}/releases/tag/${tag}`, commit: reviewedSource };
  } finally { await fs.rm(temporary, { recursive: true, force: true }); }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3) throw new Error('Usage: node plugin/scripts/publish-native-release.mjs CANDIDATE_DIRECTORY');
  console.log(JSON.stringify(await publishRelease(process.argv[2])));
}
