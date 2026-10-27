#!/usr/bin/env node
// Publish only the Marketplace's immutable package as a numbered GitHub release.
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const stableVersion = value => /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(value);
const compareVersions = (a, b) => {
  const left = a.split('.').map(Number), right = b.split('.').map(Number);
  for (let i = 0; i < 3; i++) if (left[i] !== right[i]) return left[i] - right[i];
  return 0;
};

export function syncGithubRelease({ repository, source, vsix, command = execFileSync }) {
  if (!/^[\w.-]+\/[\w.-]+$/.test(repository) || !/^[a-f0-9]{40}$/.test(source)) {
    throw new Error('A GitHub repository and full source commit are required.');
  }
  const run = (tool, args) => command(tool, args, { maxBuffer: 100 * 1024 * 1024, stdio: ['pipe', 'pipe', 'pipe'] });
  const gh = args => run('gh', args);
  const api = endpoint => JSON.parse(gh(['api', `repos/${repository}/${endpoint}`]));
  const optional = endpoint => {
    try { return api(endpoint); } catch (error) {
      if (String(error.stderr).includes('HTTP 404')) return null;
      throw error;
    }
  };
  const unzip = (file, entry) => run('unzip', ['-p', file, entry]);
  const manifest = JSON.parse(unzip(vsix, 'extension/package.json'));
  const { version, name, publisher } = manifest;
  if (!stableVersion(version) || !/^[\w-]+$/.test(name) || !/^[\w-]+$/.test(publisher)) {
    throw new Error('Expected a stable Marketplace extension identity.');
  }
  const tag = `v${version}`, asset = `${name}-${version}.vsix`;
  const directory = mkdtempSync(join(tmpdir(), 'tensorviz-release-sync-'));
  try {
    const file = join(directory, asset);
    const marketplace = `https://marketplace.visualstudio.com/_apis/public/gallery/publishers/${publisher}/vsextensions/${name}/${version}/vspackage`;
    // The Gallery can take a few minutes to expose a newly published package.
    run('curl', ['--fail', '--silent', '--show-error', '--location', '--compressed',
      '--retry', '6', '--retry-delay', '10', '--retry-all-errors', '--max-time', '60', '--output', file, marketplace]);
    const bytes = readFileSync(file);
    const published = JSON.parse(unzip(file, 'extension/package.json'));
    if (['version', 'name', 'publisher'].some(key => published[key] !== manifest[key])) {
      throw new Error('Marketplace package identity does not match the requested release.');
    }
    const pages = JSON.parse(gh(['api', '--paginate', '--slurp', `repos/${repository}/releases?per_page=100`]));
    const releases = pages.flat();
    let release = releases.find(item => item.tag_name === tag);
    if (!release) {
      // Rebuilt ZIP timestamps may differ on retry; every file must still match.
      const contents = archive => {
        const entries = String(run('unzip', ['-Z1', archive])).trim().split('\n').filter(entry => !entry.endsWith('/')).sort();
        return entries.map(entry => `${entry}\0${sha256(unzip(archive, entry))}`).join('\n');
      };
      if (sha256(readFileSync(vsix)) !== sha256(bytes) && contents(vsix) !== contents(file)) {
        throw new Error('This version is already published with different contents. Use its original source/package or bump the version; refusing to tag this build.');
      }
      const tagRef = optional(`git/ref/tags/${tag}`);
      const existingTag = tagRef ? api(`commits/${tag}`) : null;
      if (existingTag && existingTag.sha !== source) throw new Error(`${tag} already points to another commit; refusing to move it.`);
      // vsce archives CHANGELOG.md as extension/changelog.md; read the name it stored.
      const entry = String(run('unzip', ['-Z1', file])).split('\n').find(name => name.toLowerCase() === 'extension/changelog.md');
      const changelog = entry ? String(unzip(file, entry)) : '';
      const changes = changelog.split(`## ${version}\n`)[1]?.split(/\n## /)[0]?.trim();
      if (!changes) throw new Error(`The published changelog has no entry for ${version}.`);
      const request = join(directory, 'release.json');
      writeFileSync(request, JSON.stringify({ tag_name: tag, target_commitish: source, name: `tensorViz ${version}`,
        draft: true, prerelease: false, body: `${changes}\n\nPublished on the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=${publisher}.${name}). This download is the exact Marketplace package.\n\nSource: ${source}\n\nSHA-256: \`${sha256(bytes)}\`\n` }));
      release = JSON.parse(gh(['api', '--method', 'POST', `repos/${repository}/releases`, '--input', request]));
    }
    const checksum = `${sha256(bytes)}  ${asset}\n`;
    writeFileSync(`${file}.sha256`, checksum);
    for (const path of [file, `${file}.sha256`]) {
      const existing = release.assets.find(item => item.name === basename(path));
      if (existing) {
        const hosted = gh(['api', `repos/${repository}/releases/assets/${existing.id}`, '-H', 'Accept: application/octet-stream']);
        if (sha256(hosted) !== sha256(readFileSync(path))) throw new Error(`Existing ${basename(path)} differs from Marketplace; refusing to replace it.`);
      } else {
        gh(['release', 'upload', tag, path, '--repo', repository]);
      }
    }
    const newer = releases.some(item => !item.draft && !item.prerelease && stableVersion(item.tag_name.slice(1))
      && item.tag_name.startsWith('v') && compareVersions(item.tag_name.slice(1), version) > 0);
    const latest = optional('releases/latest');
    if (release.draft || release.prerelease || (!newer && latest?.tag_name !== tag)) {
      const request = join(directory, 'promote.json');
      writeFileSync(request, JSON.stringify({ draft: false, prerelease: false, make_latest: newer ? 'false' : 'true' }));
      gh(['api', '--method', 'PATCH', `repos/${repository}/releases/${release.id}`, '--input', request]);
    }
    return { version, sha256: sha256(bytes), url: `https://github.com/${repository}/releases/tag/${tag}` };
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (!process.argv.includes('--publish')) {
    console.error('Maintainer-only. Pass --publish with GITHUB_REPOSITORY, GITHUB_SHA, and a GitHub token. Nothing was published.');
    process.exit(1);
  }
  if (!(process.env.GH_TOKEN || process.env.GITHUB_TOKEN || '').trim()) {
    console.error('Maintainer publication requires GH_TOKEN or GITHUB_TOKEN. Nothing was published.');
    process.exit(1);
  }
  const manifest = JSON.parse(readFileSync(new URL('../package.json', import.meta.url)));
  const result = syncGithubRelease({ repository: process.env.GITHUB_REPOSITORY, source: process.env.GITHUB_SHA,
    vsix: resolve(process.argv[2] || `extension/${manifest.name}-${manifest.version}.vsix`) });
  console.log(`GitHub and Marketplace match: ${result.version}, SHA-256 ${result.sha256}\n${result.url}`);
}
