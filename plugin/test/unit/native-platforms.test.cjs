const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { spawn } = require('node:child_process');

const entry = (platform, abi) => ({ formatVersion: 1, version: '0.3.61', platform, pythonABI: abi,
  bytes: 100, sha256: 'a'.repeat(64), url: `https://example.invalid/${platform}.tar.gz` });
const darwin = entry('darwin-arm64', 'cpython-312-darwin');
const linux = entry('linux-x64', 'cpython-312-x86_64-linux-gnu');
const table = () => ({ formatVersion: 2, version: '0.3.61', runtimes: { 'darwin-arm64': darwin, 'linux-x64': linux } });

test('only native Darwin ARM64 and glibc Linux x64 targets map to exact compiler/pkg ABI contracts', async () => {
  const { nativePlatform } = await import('../../scripts/native-platforms.mjs');
  assert.equal(nativePlatform('linux', 'x64').pkg, 'node22.23.4-linux-x64');
  assert.equal(nativePlatform('linux', 'x64').abi, linux.pythonABI);
  assert.equal(nativePlatform('darwin', 'arm64').pkg, 'node22.23.4-macos-arm64');
  for (const [os, arch] of [['linux', 'arm64'], ['darwin', 'x64'], ['win32', 'x64']]) assert.throws(() => nativePlatform(os, arch), /Unsupported/);
});

test('native targets, workflow Node versions and exact upstream notices stay aligned', async () => {
  const { createHash } = require('node:crypto');
  const { nativePlatforms } = await import('../../scripts/native-platforms.mjs');
  const root = path.resolve(__dirname, '../../..');
  const versions = new Set();
  for (const target of Object.values(nativePlatforms)) {
    const match = /^node(\d+\.\d+\.\d+)-(?:macos|linux)-(?:arm64|x64)$/.exec(target.pkg);
    assert.ok(match, 'pkg must select an exact embedded Node version, not a floating major');
    versions.add(match[1]);
    const notice = await fs.readFile(path.join(root, 'plugin/licenses', `node-${match[1]}.txt`));
    // Official nodejs/node v22.23.4 LICENSE Git blob, including bundled notices.
    const blob = createHash('sha1').update(`blob ${notice.length}\0`).update(notice).digest('hex');
    assert.equal(blob, 'a640a1f4708257449c53645022c8762488d7261f');
  }
  assert.equal(versions.size, 1);
  const workflow = await fs.readFile(path.join(root, '.github/workflows/release-plugin.yml'), 'utf8');
  const setup = [...workflow.matchAll(/node-version:\s*'([^']+)'/g)].map(match => match[1]);
  assert.equal(setup.length, 4, 'configuration, build, assembly and publication all pin Node');
  assert.deepEqual(new Set(setup), versions);
});

test('old Darwin locks and per-platform tables retain immutable HTTPS/size/digest/version/ABI checks', async () => {
  const { validateRuntimeLock } = await import('../../scripts/export-public.mjs');
  for (const lock of [darwin, linux, table()]) assert.equal(validateRuntimeLock(lock), lock);
  for (const invalid of [
    { ...linux, pythonABI: darwin.pythonABI }, { ...linux, bytes: '100' }, { ...linux, bytes: 0 },
    { ...linux, bytes: 268435457 }, { ...linux, version: '../0.3.61' }, { ...linux, sha256: 'x'.repeat(64) },
    ...['toString', 'constructor', '__proto__'].map(platform => ({ ...linux, platform, pythonABI: undefined })),
    { ...linux, url: 'http://example.invalid/runtime' }, { ...linux, url: 'https://user:secret@example.invalid/runtime' },
    { ...linux, url: 'https://example.invalid/runtime#fragment' }, { ...linux, platform: 'linux-arm64' },
    { ...table(), version: '0.3.62' }, { ...table(), runtimes: {} },
    { ...table(), runtimes: { 'linux-x64': darwin } },
    { ...table(), runtimes: { 'win32-x64': { ...linux, platform: 'win32-x64' } } },
  ]) assert.throws(() => validateRuntimeLock(invalid));
});

for (const [platform, abi] of [['darwin-arm64', darwin.pythonABI], ['linux-x64', linux.pythonABI]]) {
  test(`${platform} native inventory requires its own backend and still excludes private source`, async () => {
    const { auditNativeRuntime } = await import('../../scripts/package-native.mjs');
    const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-platform-inventory-'));
    try {
      for (const name of ['tensorviz-host', 'plugin.json', 'runtime-manifest.json', 'backend/native-backend.json',
        `backend/_tensorviz_backend.${abi}.so`, 'backend/requirements-project.txt', 'backend/requirements-standalone.txt',
        'browser/index.html', 'browser/panel.html', 'DEPENDENCY_NOTICES.md', 'licenses/node.txt', 'licenses/pkg.txt',
        'licenses/nuitka.txt', 'licenses/tldraw-5.0.1.md']) {
        await fs.mkdir(path.dirname(path.join(root, name)), { recursive: true });
        await fs.writeFile(path.join(root, name), 'synthetic inventory fixture');
      }
      assert.ok((await auditNativeRuntime(root, platform))[`backend/_tensorviz_backend.${abi}.so`]);
      await fs.writeFile(path.join(root, 'backend/private.py'), 'private implementation');
      await assert.rejects(auditNativeRuntime(root, platform), /Core source/);
      await fs.rm(path.join(root, 'backend/private.py'));
      await fs.rm(path.join(root, `backend/_tensorviz_backend.${abi}.so`));
      await assert.rejects(auditNativeRuntime(root, platform), /incomplete/);
    } finally { await fs.rm(root, { recursive: true, force: true }); }
  });
}

test('assembly refuses duplicate platforms and mixed versions before export or packaging', async () => {
  const { assembleNativeRelease } = await import('../../scripts/assemble-native-release.mjs');
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-assembly-inputs-'));
  try {
    const a = path.join(root, 'darwin.tar.gz'), b = path.join(root, 'linux.tar.gz');
    await fs.writeFile(`${a}.json`, JSON.stringify(darwin));
    await assert.rejects(assembleNativeRelease(path.join(root, 'integration'), [a, a]), /versions\/platforms differ/);
    await fs.writeFile(`${b}.json`, JSON.stringify({ ...linux, version: '0.3.62' }));
    await assert.rejects(assembleNativeRelease(path.join(root, 'integration'), [a, b]), /versions\/platforms differ/);
    await assert.rejects(fs.stat(path.join(root, 'integration')), { code: 'ENOENT' });
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});

// Runs unchanged in normal build shells; sandbox spawnSync restrictions are
// reported as failures, never hidden by skipping provenance/package checks.
for (const { entries, relative } of [
  { entries: [darwin], relative: true }, { entries: [linux], relative: true },
  { entries: [darwin, linux], relative: true }, { entries: [darwin, linux], relative: false },
]) test(`${entries.map(entry => entry.platform).join('+')} assembly verifies ${relative ? 'relative CLI' : 'absolute API'} paths and rejects missing/tampered assets`, async () => {
  const { assembleNativeRelease } = await import('../../scripts/assemble-native-release.mjs');
  const { pluginVersion } = await import('../../scripts/release-version.mjs');
  const { sha256, createReleaseCandidate, verifyReleaseCandidate } = await import('../../scripts/release-candidate.mjs');
  const { runtimeArchive } = await import('../../scripts/native-platforms.mjs');
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-native candidate-'));
  const run = (cmd, args, options = {}) => new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'], ...options }); let output = '', error = '';
    child.stdout.on('data', bytes => output += bytes); child.stderr.on('data', bytes => error += bytes);
    child.once('error', reject); child.once('close', code => code === 0 ? resolve(output.trim()) : reject(new Error(error)));
  });
  try {
    const version = pluginVersion(), sourceRevision = await run('git', ['rev-parse', 'HEAD']);
    const assets = path.join(root, entries.length === 1 ? 'plugin/dist' : 'plugin/out/native-assets');
    await fs.mkdir(assets, { recursive: true });
    await fs.mkdir(path.join(root, 'plugin/out'), { recursive: true });
    const archives = [];
    for (const original of entries) {
      const runtime = path.join(root, original.platform); await fs.mkdir(runtime);
      await fs.writeFile(path.join(runtime, 'tensorviz-host'), 'synthetic native host');
      await fs.writeFile(path.join(runtime, 'release-manifest.json'), JSON.stringify({ version, sourceRevision,
        platform: original.platform, pythonABI: original.pythonABI, coreSourceIncluded: false,
        rendererDistributionCoverage: 'pending-owner-confirmation', files: { 'tensorviz-host': sha256(Buffer.from('synthetic native host')) } }));
      const archive = path.join(assets, runtimeArchive({ ...original, version })); archives.push(archive);
      await run('tar', ['-czf', archive, '-C', runtime, 'tensorviz-host', 'release-manifest.json']);
      const bytes = await fs.readFile(archive), lock = { ...original, version, bytes: bytes.length, sha256: sha256(bytes) };
      await fs.writeFile(`${archive}.json`, JSON.stringify(lock));
    }
    const integration = path.join(root, 'plugin/out/integration');
    const candidate = relative ? await run(process.execPath, [path.resolve(__dirname, '../../scripts/assemble-native-release.mjs'),
      path.relative(root, integration), ...archives.map(archive => path.relative(root, archive))], { cwd: root })
      : await assembleNativeRelease(integration, archives);
    assert.equal(candidate, path.join(assets, `candidate-${version}`));
    assert.equal((await verifyReleaseCandidate(candidate)).archives.length, entries.length);
    await assert.rejects(createReleaseCandidate(integration, archives.slice(1)), /platforms differ/);
    const archive = path.join(candidate, path.basename(archives.at(-1)));
    await fs.appendFile(archive, 'tamper');
    await assert.rejects(verifyReleaseCandidate(candidate), /checksum mismatch/);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});
