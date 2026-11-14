'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const zlib = require('node:zlib');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const launcher = require('../../integration/launch.cjs');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
function tar(entries) {
  const blocks = [];
  for (const [name, body = '', type = '0'] of entries) {
    const data = Buffer.from(body); const header = Buffer.alloc(512);
    header.write(name, 0, 100); header.write('0000700\0', 100); header.write('0000000\0', 108); header.write('0000000\0', 116);
    header.write(data.length.toString(8).padStart(11, '0') + '\0', 124); header.write('00000000000\0', 136);
    header.fill(32, 148, 156); header.write(type, 156); header.write('ustar\0', 257); header.write('00', 263);
    const sum = header.reduce((sum, value) => sum + value, 0);
    header.write(sum.toString(8).padStart(6, '0') + '\0 ', 148);
    blocks.push(header, data, Buffer.alloc((512 - data.length % 512) % 512));
  }
  blocks.push(Buffer.alloc(1024)); return zlib.gzipSync(Buffer.concat(blocks));
}
const entry = (platform = 'linux-x64', pythonABI = 'cpython-312-x86_64-linux-gnu', bytes = Buffer.from('fixture')) => ({
  formatVersion: 1, version: '0.1.44', platform, pythonABI, bytes: bytes.length, sha256: hash(bytes), url: 'https://example.test/runtime.tar.gz',
});

for (const [platform, arch, abi] of [
  ['linux', 'x64', 'cpython-312-x86_64-linux-gnu'], ['linux', 'arm64', 'cpython-312-aarch64-linux-gnu'],
  ['darwin', 'x64', 'cpython-312-darwin'], ['darwin', 'arm64', 'cpython-312-darwin'],
  ['win32', 'x64', 'cp312-win_amd64'], ['win32', 'arm64', 'cp312-win_arm64'],
]) test(`select only exact pinned ${platform}-${arch} runtime`, () => {
  const pin = entry(`${platform}-${arch}`, abi);
  const lock = { formatVersion: 2, version: pin.version, runtimes: { [pin.platform]: pin } };
  assert.equal(launcher.selectRuntime(lock, platform, arch), pin);
  assert.throws(() => launcher.selectRuntime({ ...lock, version: '0.1.45' }, platform, arch), /No valid/);
  assert.throws(() => launcher.selectRuntime({ ...lock, runtimes: {} }, platform, arch), /No valid/);
  assert.throws(() => launcher.selectRuntime({ ...pin, pythonABI: 'different' }, platform, arch), /No valid/);
});
test('reject unknown platform and malformed locks', () => {
  assert.throws(() => launcher.selectRuntime(entry(), 'linux', 'ia32'), /Unsupported/);
  for (const patch of [{ bytes: 0 }, { bytes: 2 ** 30 }, { bytes: 1.5 }, { sha256: 'A'.repeat(64) }, { url: 'http://example.test' }, { url: 'https://u:p@example.test' }, { url: 'https://example.test/#secret' }]) {
    assert.throws(() => launcher.selectRuntime({ ...entry(), ...patch }, 'linux', 'x64'));
  }
});
test('Linux ABI checks reject old glibc and musl, without applying to Windows/macOS', () => {
  for (const version of [undefined, '2.30', 'musl', '2.3x']) assert.throws(() => launcher.checkHost('linux', { header: { glibcVersionRuntime: version } }, 'x64'), /glibc/);
  for (const version of ['2.31', '2.35', '2.40']) launcher.checkHost('linux', { header: { glibcVersionRuntime: version } }, 'x64');
  assert.throws(() => launcher.checkHost('linux', { header: { glibcVersionRuntime: '2.31' } }, 'arm64'), /glibc/);
  launcher.checkHost('linux', { header: { glibcVersionRuntime: '2.35' } }, 'arm64');
  launcher.checkHost('win32', {}); launcher.checkHost('darwin', {});
});
test('download enforces redirects, HTTPS, digest and exact bounded size', async () => {
  const bytes = Buffer.from('fixture'); const pin = entry('linux-x64', 'cpython-312-x86_64-linux-gnu', bytes);
  let calls = 0;
  assert.deepEqual(await launcher.download(pin, async () => ++calls === 1
    ? new Response(null, { status: 302, headers: { location: 'https://cdn.example.test/runtime' } }) : new Response(bytes)), bytes);
  await assert.rejects(launcher.download(pin, async () => new Response(null, { status: 302, headers: { location: 'http://bad.example/' } })), /HTTPS/);
  await assert.rejects(launcher.download(pin, async () => new Response(Buffer.from('toolongfixture'))), /exceeded/);
  await assert.rejects(launcher.download(pin, async () => new Response(Buffer.from('changed'))), /checksum/);
  await assert.rejects(launcher.download(pin, async () => new Response(null, { status: 404 })), /HTTP 404/);
  await assert.rejects(launcher.download(pin, async () => new Response(null, { status: 302, headers: { location: '/again' } })), /redirects/);
});
for (const name of ['../escape', '/absolute', 'C:/drive', 'a\\b', 'name:stream', 'dir/../escape', 'dir//x', 'CON', 'a/NUL.txt', 'foo.', 'foo ']) {
  test(`archive rejects unsafe path ${name}`, () => assert.throws(() => launcher.archiveFiles(tar([[name, 'data']])), /Unsafe/));
}
for (const type of ['1', '2', '3', '4', '6', 'g', 'S']) test(`archive rejects special tar entry ${type}`, () => {
  assert.throws(() => launcher.archiveFiles(tar([['payload', '', type]])), /Unsupported/);
});
test('archive rejects duplicate/case collisions and file-as-parent', () => {
  assert.throws(() => launcher.archiveFiles(tar([['a', '1'], ['a', '2']])), /Duplicate/);
  assert.throws(() => launcher.archiveFiles(tar([['A', '1'], ['a', '2']])), /case-colliding/);
  assert.throws(() => launcher.archiveFiles(tar([['a', '1'], ['a/b', '2']])), /shadows/);
});
test('extract safe files and validate cache bytes before execution', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-portable-test-'));
  try {
    const bytes = tar([['tensorviz-host', 'fixture'], ['backend/data.json', '{}']]);
    await launcher.extract(bytes, root);
    assert.equal(await fs.readFile(path.join(root, 'backend/data.json'), 'utf8'), '{}');
    const file = path.join(root, 'cached.tar.gz'); await fs.writeFile(file, bytes);
    const pin = entry('linux-x64', 'cpython-312-x86_64-linux-gnu', bytes);
    assert.deepEqual(await launcher.readVerified(file, pin), bytes);
    await fs.writeFile(file, Buffer.alloc(bytes.length));
    await assert.rejects(launcher.readVerified(file, pin), /checksum/);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});
test('PAX and GNU long paths retain safe files and reject traversal', () => {
  const pax = value => { let n = Buffer.byteLength(value) + 3; while (n !== String(n).length + 1 + Buffer.byteLength(value) + 1) n = String(n).length + 1 + Buffer.byteLength(value) + 1; return `${n} ${value}\n`; };
  const longName = 'backend/' + 'a'.repeat(120) + '.json';
  assert.equal(launcher.archiveFiles(tar([['PaxHeaders', pax(`path=${longName}`), 'x'], ['short', '{}']]))[0].name, longName);
  assert.equal(launcher.archiveFiles(tar([['LongLink', longName + '\0', 'L'], ['short', '{}']]))[0].name, longName);
  assert.throws(() => launcher.archiveFiles(tar([['PaxHeaders', pax('path=../escape'), 'x'], ['short', '{}']])), /Unsafe/);
});
test('tar truncation and header corruption never extract', () => {
  const bytes = zlib.gunzipSync(tar([['safe', 'data']]));
  bytes[0] ^= 1;
  assert.throws(() => launcher.archiveFiles(zlib.gzipSync(bytes)), /checksum/);
  assert.throws(() => launcher.archiveFiles(zlib.gzipSync(Buffer.alloc(500))), /terminator/);
  assert.throws(() => launcher.archiveFiles(tar([['dir/', 'not empty', '5']])), /directory contains data/);
});
test('Windows launches the pinned .exe directly without cmd or sh', { skip: process.platform !== 'win32' }, async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz portable windows '));
  try {
    const scripts = path.join(root, 'plugin', 'scripts'), cache = path.join(root, 'cache');
    await fs.mkdir(scripts, { recursive: true }); await fs.mkdir(cache);
    await fs.copyFile(path.resolve(__dirname, '../../integration/launch.cjs'), path.join(scripts, 'launch.cjs'));
    // A real Windows executable, not an emulated process.platform unit test.
    const bytes = tar([['tensorviz-host.exe', await fs.readFile(process.execPath)]]);
    const pin = entry(`win32-${process.arch}`, process.arch === 'arm64' ? 'cp312-win_arm64' : 'cp312-win_amd64', bytes);
    await fs.writeFile(path.join(root, 'plugin/runtime-lock.json'), JSON.stringify(pin));
    await fs.writeFile(path.join(cache, pin.sha256 + '.tar.gz'), bytes);
    const receipt = path.join(root, 'receipt');
    const code = 'const fs=require("node:fs"); fs.writeFileSync(process.env.RECEIPT,process.env.TORCHINDUCTOR_CACHE_DIR); process.stdin.pipe(process.stdout);';
    const result = spawnSync(process.execPath, [path.join(scripts, 'launch.cjs'), '-e', code], {
      input: 'windows-stdio\n', encoding: 'utf8', timeout: 30000,
      env: { ...process.env, TENSORVIZ_RUNTIME_CACHE_DIR: cache, RECEIPT: receipt },
    });
    assert.equal(result.status, 0, result.stderr); assert.equal(result.stdout, 'windows-stdio\n');
    assert.equal(await fs.stat(path.dirname(await fs.readFile(receipt, 'utf8'))).catch(() => null), null);
  } finally { await fs.rm(root, { recursive: true, force: true, maxRetries: 4 }); }
});
test('real POSIX process preserves MCP stdin and cleans private extracted session', { skip: process.platform === 'win32' }, async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-portable-process-'));
  try {
    const scripts = path.join(root, 'plugin', 'scripts'); const cache = path.join(root, 'cache');
    await fs.mkdir(scripts, { recursive: true }); await fs.mkdir(cache, { mode: 0o700 });
    await fs.copyFile(path.resolve(__dirname, '../../integration/launch.cjs'), path.join(scripts, 'launch.cjs'));
    const bytes = tar([['tensorviz-host', '#!/bin/sh\nIFS= read -r request\nprintf "%s\\n" "$request"\nprintf "%s" "$TORCHINDUCTOR_CACHE_DIR" > "$RECEIPT"\n']]);
    const pin = entry(`${process.platform}-${process.arch}`, process.platform === 'darwin' ? 'cpython-312-darwin' : process.arch === 'arm64' ? 'cpython-312-aarch64-linux-gnu' : 'cpython-312-x86_64-linux-gnu', bytes);
    await fs.writeFile(path.join(root, 'plugin/runtime-lock.json'), JSON.stringify(pin));
    await fs.writeFile(path.join(cache, pin.sha256 + '.tar.gz'), bytes);
    const receipt = path.join(root, 'receipt'); const input = '{"jsonrpc":"2.0","id":1}\n';
    const result = spawnSync(process.execPath, [path.join(scripts, 'launch.cjs')], { input, encoding: 'utf8', env: { ...process.env, TENSORVIZ_RUNTIME_CACHE_DIR: cache, RECEIPT: receipt } });
    assert.equal(result.status, 0, result.stderr); assert.equal(result.stdout, input);
    const session = path.dirname(await fs.readFile(receipt, 'utf8'));
    assert.equal(await fs.stat(session).catch(() => null), null);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});

test('Windows native spawn errors identify the failing stage without inventing a policy diagnosis', () => {
  for (const code of ['UNKNOWN', 'EACCES', 'ENOENT']) {
    const cause = Object.assign(new Error('spawn ' + code), { code });
    const error = launcher.nativeLaunchError(cause, 'win32');
    assert.equal(error.cause, cause);
    assert.equal(error.code, code);
    assert.match(error.message, /host did not start; Python setup was not reached/);
    assert.match(error.message, /If Application Control blocked/);
    assert.match(error.message, /Do not disable security protections/);
    assert.equal(launcher.nativeLaunchError(cause, 'linux'), cause);
    assert.equal(launcher.nativeLaunchError(cause, 'darwin'), cause);
  }
  assert.match(launcher.nativeLaunchError(new Error('private path'), 'win32').message, /\(UNKNOWN\)/);
  assert.doesNotMatch(launcher.nativeLaunchError(new Error('private path'), 'win32').message, /private path/);
});
