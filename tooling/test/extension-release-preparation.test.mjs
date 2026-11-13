import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { prepareExtensionRelease, readReleaseResult, releaseRepository } from '../release/extension-release.mjs';

const source = 'a'.repeat(40);
function fixture(t) {
  const root = mkdtempSync(join(tmpdir(), 'reviewed-release-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(join(root, 'extension'));
  const manifest = { name: 'tensorviz-vscode', publisher: 'tensorviz', version: '0.1.41' };
  const lock = { name: manifest.name, version: manifest.version, packages: { '': { version: manifest.version } } };
  const write = (name, value) => writeFileSync(join(root, 'extension', name), typeof value === 'string' ? value : JSON.stringify(value));
  write('package.json', manifest); write('package-lock.json', lock); write('CHANGELOG.md', '## 0.1.41\n\nReviewed release notes.\n');
  const calls = [], state = { head: source, dirty: '' };
  const command = (tool, args) => {
    calls.push([tool, ...args]); assert.equal(tool, 'git');
    if (args[0] === 'rev-parse') return state.head;
    assert.deepEqual(args, ['status', '--porcelain', '--untracked-files=no']); return state.dirty;
  };
  const prepare = options => prepareExtensionRelease({ root, source, repository: releaseRepository, command, ...options });
  return { root, manifest, lock, write, prepare, calls, state };
}

test('uses the reviewed version and exact source without creating commits or pushing', t => {
  const f = fixture(t), before = readFileSync(join(f.root, 'extension/package.json'));
  assert.deepEqual(f.prepare({ publish: true }), { schema: 1, repository: releaseRepository, source, version: '0.1.41', publish: true });
  assert.equal(f.prepare().publish, false);
  assert.deepEqual(readFileSync(join(f.root, 'extension/package.json')), before);
  assert.ok(f.calls.every(call => !call.includes('push') && !call.includes('commit')));
});

test('rejects forks, mutable refs, source drift and dirty tracked files', t => {
  const f = fixture(t);
  for (const options of [{ source: 'main' }, { repository: 'outside/tensorviz' }, { publish: 'true' }]) assert.throws(() => f.prepare(options));
  f.state.head = 'b'.repeat(40); assert.throws(() => f.prepare(), /Checkout differs/);
  f.state.head = source; f.state.dirty = ' M extension/package.json'; assert.throws(() => f.prepare(), /clean/);
});

test('requires matching lockfile identity, stable version and reviewed release notes', t => {
  const f = fixture(t);
  f.write('package-lock.json', { ...f.lock, version: '0.1.40' }); assert.throws(() => f.prepare(), /must match/);
  f.write('package-lock.json', f.lock); f.write('CHANGELOG.md', '## Unreleased\nPending notes.\n'); assert.throws(() => f.prepare(), /reviewed PR/);
  f.write('package.json', { ...f.manifest, version: '0.1.41-beta' }); assert.throws(() => f.prepare(), /must match/);
});

test('handoff rejects stale runs, forged repositories and malformed payloads', t => {
  const f = fixture(t), file = join(f.root, 'release.json'), valid = f.prepare({ publish: true });
  writeFileSync(file, JSON.stringify(valid)); assert.deepEqual(readReleaseResult(file, source), valid);
  for (const patch of [{ schema: 0 }, { source: 'b'.repeat(40) }, { repository: 'fork/tensorviz' }, { version: '../latest' }, { publish: 'true' }]) {
    writeFileSync(file, JSON.stringify({ ...valid, ...patch })); assert.throws(() => readReleaseResult(file, source), /trusted source run/);
  }
});

test('release workflows cannot execute fork PR code with secrets or write source branches', () => {
  for (const name of ['release-vsix', 'release-openvsx', 'release-plugin']) {
    const text = readFileSync(new URL(`../../.github/workflows/${name}.yml`, import.meta.url), 'utf8');
    assert.match(text, /github.repository == 'tensorviz-org\/tensorviz'/);
    assert.match(text, /vars.RELEASES_ENABLED == 'true'/);
    assert.doesNotMatch(text, /^\s+environment:/m, 'repository-secret publishing does not require a deployment environment');
    assert.match(text, /persist-credentials: false/);
    assert.match(text, /ref: \$\{\{ github.sha \}\}/);
    assert.match(text, /cancel-in-progress: false/);
    assert.doesNotMatch(text, /pull_request_target|pull_request:|git push|POSTHOG|pytorch-viz|mainbackend/);
  }
});


test('initial cutover cannot publish merely by merging even if an unknown repository variable is true', () => {
  for (const name of ['release-vsix', 'release-plugin']) {
    const text = readFileSync(new URL(`../../.github/workflows/${name}.yml`, import.meta.url), 'utf8');
    assert.doesNotMatch(text, /^  push:/m);
    assert.match(text, /workflow_dispatch:/);
    assert.match(text, /default: false/);
    assert.match(text, /RELEASE_PUBLISH: \$\{\{ vars.RELEASES_ENABLED == 'true' && inputs.publish \}\}/);
  }
});


test('native publishing isolates cross-repository credential from build jobs', () => {
  const text = readFileSync(new URL('../../.github/workflows/release-plugin.yml', import.meta.url), 'utf8');
  const [before, publish] = text.split('\n  publish:\n');
  assert.doesNotMatch(before, /PLUGIN_PUBLISH_TOKEN|TENSORVIZ_SOURCE_TOKEN/);
  assert.match(publish, /contents: read/);
  assert.doesNotMatch(publish, /contents: write|npm (?:ci|install)/);
  assert.match(publish, /GH_TOKEN: \$\{\{ secrets.PLUGIN_PUBLISH_TOKEN \}\}/);
  assert.match(publish, /TENSORVIZ_SOURCE_TOKEN: \$\{\{ github.token \}\}/);
  assert.match(before, /tensorviz-org\/tensorviz-plugins\/releases\/download\/v/);
});


test('repository-secret publishing preserves manual main-only activation and build isolation', () => {
  for (const name of ['release-vsix', 'release-plugin']) {
    const text = readFileSync(new URL(`../../.github/workflows/${name}.yml`, import.meta.url), 'utf8');
    const [before, publish] = text.split('\n  publish:\n');
    assert.ok(publish, 'separate publishing job remains');
    assert.doesNotMatch(before, /^\s+environment:/m, 'configuration/build/assembly do not access environments');
    assert.match(before, /permissions:\n  contents: read/);
    assert.doesNotMatch(before, /contents: write|VSCE_PAT|OVSX_PAT|PLUGIN_PUBLISH_TOKEN/);
    assert.match(before, /github.ref == 'refs\/heads\/main'/);
    assert.match(before, /github.event_name == 'workflow_dispatch'/);
    assert.match(before, /RELEASE_PUBLISH: \$\{\{ vars.RELEASES_ENABLED == 'true' && inputs.publish \}\}/);
    assert.match(before, /secrets.TLDRAW_LICENSE_KEY/);
    assert.match(publish, /if: needs\.(?:build|configuration)\.outputs\.publish == 'true'/);
    assert.doesNotMatch(publish, /^\s+environment:/m);
  }
  const plugin = readFileSync(new URL('../../.github/workflows/release-plugin.yml', import.meta.url), 'utf8');
  assert.match(plugin, /TLDRAW_DISTRIBUTION_CONFIRMED: \$\{\{ vars.TLDRAW_DISTRIBUTION_CONFIRMED \}\}/);
  const openvsx = readFileSync(new URL('../../.github/workflows/release-openvsx.yml', import.meta.url), 'utf8');
  assert.match(openvsx, /vars.RELEASES_ENABLED == 'true'/);
  assert.doesNotMatch(openvsx, /^\s+environment:/m);
  assert.match(openvsx, /github.event_name == 'workflow_dispatch' && github.ref == 'refs\/heads\/main'/);
  assert.match(openvsx, /github.event.workflow_run.conclusion == 'success'/);
  assert.match(openvsx, /github.event.workflow_run.head_branch == 'main'/);
  assert.match(openvsx, /github.event.workflow_run.head_repository.full_name == github.repository/);
  const runbook = readFileSync(new URL('../../docs/releasing.md', import.meta.url), 'utf8');
  assert.match(runbook, /Keep `RELEASES_ENABLED` unset or `false` throughout this validation/);
});


test('publisher credentials remain step-scoped with least-privilege repository tokens', () => {
  for (const [name, secret] of [['release-vsix', 'VSCE_PAT'], ['release-openvsx', 'OVSX_PAT'], ['release-plugin', 'PLUGIN_PUBLISH_TOKEN']]) {
    const text = readFileSync(new URL(`../../.github/workflows/${name}.yml`, import.meta.url), 'utf8');
    const [before, publish] = text.split('\n  publish:\n');
    assert.ok(publish, 'publisher remains a separate job');
    assert.match(before, /permissions:\n  contents: read/);
    assert.doesNotMatch(before, /VSCE_PAT|OVSX_PAT|PLUGIN_PUBLISH_TOKEN/);
    assert.equal((publish.match(new RegExp(`secrets\\.${secret}`, 'g')) ?? []).length, 1);
    assert.match(publish, new RegExp(`^          (?:${secret}|GH_TOKEN): \\$\\{\\{ secrets\\.${secret} \\}\\}$`, 'm'));
    for (const other of ['VSCE_PAT', 'OVSX_PAT', 'PLUGIN_PUBLISH_TOKEN'].filter(value => value !== secret)) {
      assert.ok(!publish.includes(other), `${name} does not receive ${other}`);
    }
    if (name === 'release-vsix') {
      assert.match(publish, /permissions:\n      contents: write/);
      assert.doesNotMatch(before, /contents: write/);
    } else {
      assert.doesNotMatch(text, /contents: write/);
    }
  }
  const runbook = readFileSync(new URL('../../docs/releasing.md', import.meta.url), 'utf8');
  assert.match(runbook, /repository Actions secrets/);
  assert.match(runbook, /No deployment environment or Enterprise upgrade is required/);
  assert.match(runbook, /Org-only review enforcement is deferred until public launch/);
});


test('prepared cutover versions exceed audited legacy floors and keep manifests, locks and docs aligned', () => {
  const read = relative => readFileSync(new URL(`../../${relative}`, import.meta.url), 'utf8');
  const newer = (version, floor) => {
    assert.match(version, /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/);
    const left = version.split('.').map(Number), right = floor.split('.').map(Number);
    for (let i = 0; i < 3; i++) if (left[i] !== right[i]) return left[i] > right[i];
    return false;
  };
  // Unified candidate follows extension history; exact legacy plugin transition
  // is separately guarded by version-alignment policy and native receipts.
  assert.equal(JSON.parse(read('extension/package.json')).version, JSON.parse(read('plugin/package.json')).version);
  // The cutover runbook requires a fresh check before activating publication.
  for (const [product, floor] of [['extension', '0.1.42'], ['plugin', '0.1.42']]) {
    const manifest = JSON.parse(read(`${product}/package.json`));
    const lock = JSON.parse(read(`${product}/package-lock.json`));
    assert.ok(newer(manifest.version, floor), `${product} must exceed the audited release floor`);
    assert.equal(lock.version, manifest.version);
    assert.equal(lock.packages[''].version, manifest.version);
    for (const document of ['docs/local-setup.md', 'docs/reference.md', 'docs/releasing.md']) {
      assert.ok(read(document).includes(manifest.version), `${document} must show the prepared ${product} version`);
    }
    if (product === 'extension') assert.ok(read('extension/CHANGELOG.md').includes(`## ${manifest.version}\n`));
  }
});

