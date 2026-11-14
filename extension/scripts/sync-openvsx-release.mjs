#!/usr/bin/env node
// Mirror the immutable Marketplace/GitHub package, never the rolling main build.
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { appendFileSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { setTimeout } from 'node:timers/promises';
import { readReleaseResult } from '../../tooling/release/extension-release.mjs';

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const registry = 'https://open-vsx.org';
const publisher = 'tensorviz', name = 'tensorviz-vscode';

export async function syncOpenVsxRelease({ version, source, repository, token = process.env.OVSX_PAT,
  command = execFileSync, request = fetch, sleep = setTimeout }) {
  if (!/^[\w.-]+\/[\w.-]+$/.test(repository ?? '')) throw new Error('A GitHub repository is required.');
  const run = (tool, args, options = {}) => command(tool, args, { maxBuffer: 100 * 1024 * 1024,
    timeout: 120_000, stdio: ['pipe', 'pipe', 'pipe'], ...options });
  const api = endpoint => JSON.parse(run('gh', ['api', `repos/${repository}/${endpoint}`]));
  if (source || !version) {
    if (!/^[a-f0-9]{40}$/.test(source ?? '')) throw new Error('A release version or full source commit is required.');
    const manifest = api(`contents/extension/package.json?ref=${source}`);
    const preparedVersion = JSON.parse(Buffer.from(manifest.content, 'base64')).version;
    if (version && version !== preparedVersion) throw new Error('Prepared source and release version disagree.');
    version = preparedVersion;
  }
  if (!/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(version)) throw new Error('Expected a stable extension version.');
  if (Number(version.split('.')[0]) >= 1) throw new Error('Public 1.x releases require the unified source publisher.');
  const get = url => request(url, { signal: AbortSignal.timeout(30_000) });
  const download = async url => {
    const response = await get(url);
    if (!response.ok) throw new Error(`Package download failed: HTTP ${response.status}`);
    return Buffer.from(await response.arrayBuffer());
  };
  const bytes = await download(`https://marketplace.visualstudio.com/_apis/public/gallery/publishers/${publisher}/vsextensions/${name}/${version}/vspackage`);
  const checksum = sha256(bytes), asset = `${name}-${version}.vsix`;
  const directory = mkdtempSync(join(tmpdir(), 'tensorviz-openvsx-'));
  try {
    const file = join(directory, asset);
    writeFileSync(file, bytes);
    const manifest = JSON.parse(run('unzip', ['-p', file, 'extension/package.json']));
    if (manifest.publisher !== publisher || manifest.name !== name || manifest.version !== version) {
      throw new Error('Marketplace package identity does not match the requested release.');
    }
    const release = api(`releases/tags/v${version}`);
    if (source && api(`commits/v${version}`).sha !== source) throw new Error('Numbered release does not identify the prepared source.');
    const hosted = release.assets.find(item => item.name === asset);
    if (release.draft || release.prerelease || !hosted) throw new Error('A published numbered GitHub VSIX is required.');
    const github = run('gh', ['api', `repos/${repository}/releases/assets/${hosted.id}`, '-H', 'Accept: application/octet-stream']);
    if (sha256(github) !== checksum) throw new Error('GitHub and Marketplace packages differ; refusing Open VSX publication.');

    const metadata = async () => {
      const response = await get(`${registry}/api/${publisher}/${name}/${version}`);
      if (response.status === 404) return null;
      if (!response.ok) throw new Error(`Open VSX lookup failed: HTTP ${response.status}`);
      const data = await response.json();
      if (data.namespace !== publisher || data.name !== name || data.version !== version || !data.files?.download) {
        throw new Error('Open VSX package identity or download URL is missing or incorrect.');
      }
      return data;
    };
    let existing = await metadata();
    const published = !existing;
    if (!existing) {
      if (!token) throw new Error('Set the OVSX_PAT Actions secret for the tensorviz publisher.');
      run('npx', ['--yes', 'ovsx@1.3.0', 'publish', file, '--registryUrl', registry], {
        stdio: 'inherit', env: { ...process.env, OVSX_PAT: token },
      });
      // Allow five minutes for registry scanning/indexing after a successful upload.
      for (let attempt = 0; attempt < 31 && !existing; attempt++) {
        if (attempt) await sleep(10_000);
        existing = await metadata();
      }
      if (!existing) throw new Error('Open VSX publication is not visible yet; rerun to verify it.');
    }
    if (sha256(await download(existing.files.download)) !== checksum) {
      throw new Error('Open VSX bytes differ from Marketplace/GitHub; existing versions must not be overwritten.');
    }
    return { version, sha256: checksum, published, url: `${registry}/extension/${publisher}/${name}/${version}` };
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

export async function syncOpenVsxFromResult({ file, requestedSource, ...options }) {
  const result = readReleaseResult(file, requestedSource);
  if (!result.publish) return { skipped: true, reason: 'The originating run was build-only.' };
  if (options.repository !== result.repository) throw new Error('Release handoff repository mismatch.');
  return syncOpenVsxRelease({ ...options, version: result.version, source: result.source });
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (!process.argv.includes('--publish')) {
    console.error('Maintainer-only. Pass --publish with OVSX_PAT, GITHUB_REPOSITORY, and a published GitHub release. Nothing was published.');
    process.exit(1);
  }
  const prepared = process.env.RELEASE_RESULT ? readReleaseResult(process.env.RELEASE_RESULT, process.env.RELEASE_REQUEST_SOURCE) : null;
  if (prepared && !prepared.publish) {
    console.log('The originating run was build-only; nothing to synchronize.');
    process.exit(0);
  }
  if (!(process.env.OVSX_PAT || '').trim() || !(process.env.GH_TOKEN || process.env.GITHUB_TOKEN || '').trim()) {
    console.error('Maintainer publication requires OVSX_PAT and GH_TOKEN or GITHUB_TOKEN. Nothing was published.');
    process.exit(1);
  }
  const result = await syncOpenVsxRelease({ version: prepared?.version || process.env.OPENVSX_VERSION,
    source: prepared?.source || process.env.RELEASE_SOURCE, repository: process.env.GITHUB_REPOSITORY });
  const summary = `${result.published ? 'Published' : 'Verified existing'} Open VSX ${result.version}. ` +
    `Marketplace, numbered GitHub and Open VSX packages match: SHA-256 ${result.sha256}\n${result.url}\n`;
  console.log(summary);
  if (process.env.GITHUB_STEP_SUMMARY) appendFileSync(process.env.GITHUB_STEP_SUMMARY, summary);
}

