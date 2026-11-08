import { execFileSync } from 'node:child_process';
import { appendFileSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const releaseRepository = 'tensorviz-org/tensorviz';
const stableVersion = value => /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(value ?? '');

export function readReleaseResult(file, requestedSource) {
  const result = JSON.parse(readFileSync(file, 'utf8'));
  if (result.schema !== 1 || result.repository !== releaseRepository || typeof result.publish !== 'boolean'
    || !/^[a-f0-9]{40}$/.test(result.source ?? '') || result.source !== requestedSource
    || !stableVersion(result.version)) throw new Error('Release handoff does not match this trusted source run.');
  return result;
}

// Release versions are reviewed with the code. Never commit, push, or allocate a
// version from a workflow run number: those operations bypass PR review or reset
// when workflows move repositories.
export function prepareExtensionRelease({ root = process.cwd(), source, repository, publish = false,
  command = execFileSync } = {}) {
  if (repository !== releaseRepository || !/^[a-f0-9]{40}$/.test(source ?? '')
    || typeof publish !== 'boolean') throw new Error('Expected the upstream repository and exact source commit.');
  const git = args => String(command('git', args, { cwd: root, encoding: 'utf8' })).trim();
  if (git(['rev-parse', 'HEAD']) !== source) throw new Error('Checkout differs from the requested source.');
  if (git(['status', '--porcelain', '--untracked-files=no'])) throw new Error('Release source must be clean.');
  const manifest = JSON.parse(readFileSync(join(root, 'extension/package.json'), 'utf8'));
  const lock = JSON.parse(readFileSync(join(root, 'extension/package-lock.json'), 'utf8'));
  if (!stableVersion(manifest.version) || manifest.name !== 'tensorviz-vscode' || manifest.publisher !== 'tensorviz'
    || lock.name !== manifest.name || lock.version !== manifest.version
    || lock.packages?.['']?.version !== manifest.version) throw new Error('Extension manifest and lockfile identity/version must match.');
  const changelog = readFileSync(join(root, 'extension/CHANGELOG.md'), 'utf8');
  if (!changelog.split(`## ${manifest.version}\n`)[1]?.split(/\n## /)[0]?.trim()) {
    throw new Error('Commit release notes and the intended version in a reviewed PR before releasing.');
  }
  return { schema: 1, repository, source, version: manifest.version, publish };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.env.GITHUB_REF !== 'refs/heads/main') throw new Error('Release workflows run only from main.');
  const result = prepareExtensionRelease({ source: process.env.GITHUB_SHA, repository: process.env.GITHUB_REPOSITORY,
    publish: process.env.RELEASE_PUBLISH === 'true' });
  writeFileSync(join(process.env.RUNNER_TEMP, 'tensorviz-release.json'), JSON.stringify(result, null, 2) + '\n');
  appendFileSync(process.env.GITHUB_OUTPUT, `version=${result.version}\nsource=${result.source}\npublish=${result.publish}\n`);
  appendFileSync(process.env.GITHUB_STEP_SUMMARY, `Verified reviewed source ${result.source}, extension ${result.version}; publish=${result.publish}. No source files or branches changed.\n`);
}
