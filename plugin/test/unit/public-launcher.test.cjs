const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { createHash } = require('node:crypto');

const run = (command, args, options, input = '') => new Promise((resolve, reject) => {
  const child = spawn(command, args, { ...options, timeout: 10000 });
  let stdout = '', stderr = '';
  child.stdout.on('data', bytes => stdout += bytes);
  child.stderr.on('data', bytes => stderr += bytes);
  child.once('error', reject);
  child.once('close', (status, signal) => resolve({ status, signal, stdout, stderr }));
  child.stdin.end(input);
});
const ssh = { SSH_CONNECTION: '192.0.2.1 12345 192.0.2.2 22', SSH_CLIENT: '192.0.2.1 12345 22',
  SSH_TTY: '/dev/pts/synthetic', VSCODE_REMOTE_NAME: 'ssh-remote' };

async function fixture(system, architecture) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-public-launcher-'));
  const home = path.join(root, 'home'), tmp = path.join(root, 'tmp'), bin = path.join(root, 'bin');
  const plugin = path.join(root, 'plugin with spaces'), scripts = path.join(plugin, 'scripts');
  for (const directory of [home, tmp, bin, scripts]) await fs.mkdir(directory, { recursive: true });
  const executable = (name, text) => fs.writeFile(path.join(bin, name), text, { mode: 0o755 });
  await executable('uname', `#!/bin/sh\ncase "$1" in -s) echo '${system}';; -m) echo '${architecture}';; *) exit 99;; esac\n`);
  // Fully injected downloader: it can only copy this local fixture, never reach a
  // network. Even a regression to an absolute curl path uses this stub in the copy.
  await executable('curl', '#!/bin/sh\noutput=""\nprevious=""\nfor arg do\n [ "$previous" != --output ] || output="$arg"\n previous="$arg"\ndone\n[ "$arg" = https://127.0.0.1:1/runtime.tar.gz ] || exit 99\nprintf "download\\n" >> "$DOWNLOAD_LOG"\ncp "$FIXTURE_ARCHIVE" "$output"\n');
  await executable('plutil', '#!/usr/bin/python3\nimport json,sys\nwith open(sys.argv[-1]) as f: value=json.load(f)\nfor part in sys.argv[2].split("."): value=value[part]\nprint(value)\n');
  const checksum = `#!${process.execPath}\nconst fs = require('node:fs'); const crypto = require('node:crypto');\nconst file = process.argv.at(-1); console.log(crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex') + '  ' + file);\n`;
  await executable('shasum', checksum);
  await executable('sha256sum', checksum);
  await executable('getconf', '#!/bin/sh\necho "glibc 2.35"\n');
  const source = await fs.readFile(path.resolve(__dirname, '../../integration/launch.sh'), 'utf8');
  await fs.writeFile(path.join(scripts, 'launch.sh'), source
    .replaceAll('/usr/bin/plutil', path.join(bin, 'plutil')).replaceAll('/usr/bin/shasum', path.join(bin, 'shasum'))
    .replaceAll('/usr/bin/getconf', path.join(bin, 'getconf'))
    .replaceAll('/usr/bin/curl', path.join(bin, 'curl')));
  const env = { HOME: home, TMPDIR: tmp, PATH: `${bin}:/usr/bin:/bin`,
    FIXTURE_ARCHIVE: path.join(root, 'runtime.tar.gz'), DOWNLOAD_LOG: path.join(root, 'downloads'), EXEC_LOG: path.join(root, 'executions') };
  return { root, home, tmp, plugin, env,
    launch: (flags, input) => run('/bin/sh', [path.join(scripts, 'launch.sh')], { cwd: root, env: { ...env, ...flags } }, input),
    close: () => fs.rm(root, { recursive: true, force: true }) };
}

for (const [system, architecture, flags = {}] of [
  ['Linux', 'aarch64'], ['Darwin', 'x86_64'], ['Windows_NT', 'AMD64'], ['FreeBSD', 'amd64'], ['Linux', 'aarch64', ssh],
]) {
  test(`unsupported ${system}-${architecture}${flags.SSH_CONNECTION ? ' with SSH flags' : ''} stops before metadata/cache/network`, async () => {
    const f = await fixture(system, architecture);
    try {
      const result = await f.launch(flags);
      assert.equal(result.status, 1); assert.equal(result.signal, null); assert.equal(result.stdout, '');
      assert.equal(result.stderr, `TensorViz: Unsupported MCP execution host: ${system}-${architecture}. Compiled targets are macOS Apple Silicon and glibc Linux x64. No runtime was downloaded.\n`);
      assert.deepEqual(await fs.readdir(f.home), []); assert.deepEqual(await fs.readdir(f.tmp), []);
      assert.equal(await fs.stat(f.env.DOWNLOAD_LOG).catch(() => null), null);
    } finally { await f.close(); }
  });
}

for (const [system, architecture, platform, abi] of [
  ['Linux', 'x86_64', 'linux-x64', 'cpython-312-x86_64-linux-gnu'],
  ['Darwin', 'arm64', 'darwin-arm64', 'cpython-312-darwin'],
]) {
  for (const formatVersion of [1, 2]) test(`${platform} lock v${formatVersion} passes stdin to its async host and verifies/caches only its runtime using a local downloader`, async () => {
    const f = await fixture(system, architecture);
    try {
      const payload = path.join(f.root, 'payload'); await fs.mkdir(payload);
      await fs.writeFile(path.join(payload, 'tensorviz-host'), '#!/bin/sh\nIFS= read -r request || exit 88\necho ran >> "$EXEC_LOG"\nprintf \"native-fixture:%s\\n\" "$request"\n', { mode: 0o755 });
      assert.equal((await run('/usr/bin/tar', ['-czf', f.env.FIXTURE_ARCHIVE, '-C', payload, 'tensorviz-host'], { env: f.env })).status, 0);
      const archive = await fs.readFile(f.env.FIXTURE_ARCHIVE);
      const entry = { formatVersion: 1, version: '0.3.61', platform, pythonABI: abi,
        bytes: archive.length, sha256: createHash('sha256').update(archive).digest('hex'), url: 'https://127.0.0.1:1/runtime.tar.gz' };
      const other = platform === 'linux-x64'
        ? { ...entry, platform: 'darwin-arm64', pythonABI: 'cpython-312-darwin', sha256: 'b'.repeat(64) }
        : { ...entry, platform: 'linux-x64', pythonABI: 'cpython-312-x86_64-linux-gnu', sha256: 'b'.repeat(64) };
      const lock = formatVersion === 1 ? entry : { formatVersion: 2, version: entry.version,
        runtimes: { [other.platform]: other, [platform]: entry } };
      const file = path.join(f.plugin, 'runtime-lock.json');
      await fs.writeFile(file, JSON.stringify(lock));
      const request = '{"jsonrpc":"2.0","id":1,"method":"initialize"}';
      for (let count = 0; count < 2; count++) {
        const result = await f.launch(ssh, `${request}\n`);
        assert.equal(result.status, 0, result.stderr);
        assert.equal(result.stdout, `native-fixture:${request}\n`, 'Async host must receive the original MCP stdin, not EOF');
        assert.deepEqual(await fs.readdir(f.tmp), [], 'Session extraction must be cleaned');
      }
      assert.equal(await fs.readFile(f.env.DOWNLOAD_LOG, 'utf8'), 'download\n', 'Reconnect must reuse the verified archive');
      const cache = path.join(f.home, platform === 'darwin-arm64' ? 'Library/Caches' : '.cache', 'dev.tensorviz/runtime', `${entry.sha256}.tar.gz`);
      const corrupt = Buffer.from(archive); corrupt[0] ^= 1; await fs.writeFile(cache, corrupt);
      const refused = await f.launch();
      assert.equal(refused.status, 1); assert.equal(refused.stdout, ''); assert.match(refused.stderr, /damaged/);
      assert.equal(await fs.readFile(f.env.EXEC_LOG, 'utf8'), 'ran\nran\n', 'Damaged bytes must never execute');
      await fs.rm(cache);
      const bad = { ...entry, sha256: '0'.repeat(64) };
      await fs.writeFile(file, JSON.stringify(formatVersion === 1 ? bad : { ...lock, runtimes: { [platform]: bad } }));
      const mismatch = await f.launch();
      assert.equal(mismatch.status, 1); assert.equal(mismatch.stdout, ''); assert.match(mismatch.stderr, /checksum mismatch/);
      assert.equal(await fs.readFile(f.env.EXEC_LOG, 'utf8'), 'ran\nran\n');
      assert.deepEqual(await fs.readdir(path.dirname(cache)), [], 'Failed download must be cleaned');
    } finally { await f.close(); }
  });
}

test('Linux SSH rejects the old Darwin-only pin without a download or cache write', async () => {
  const f = await fixture('Linux', 'x86_64');
  try {
    await fs.writeFile(path.join(f.plugin, 'runtime-lock.json'), JSON.stringify({ formatVersion: 1, version: '0.3.104', platform: 'darwin-arm64' }));
    const result = await f.launch(ssh);
    assert.equal(result.status, 1); assert.equal(result.stdout, ''); assert.match(result.stderr, /No linux-x64 runtime/);
    assert.deepEqual(await fs.readdir(f.home), []);
    assert.equal(await fs.stat(f.env.DOWNLOAD_LOG).catch(() => null), null);
  } finally { await f.close(); }
});

for (const libc of ['glibc 2.34', 'musl 1.2']) test(`${libc} is refused before Linux lock/cache/download`, async () => {
  const f = await fixture('Linux', 'x86_64');
  try {
    await fs.writeFile(path.join(f.root, 'bin/getconf'), `#!/bin/sh\necho '${libc}'\n`, { mode: 0o755 });
    const result = await f.launch(ssh);
    assert.equal(result.status, 1); assert.equal(result.stdout, ''); assert.match(result.stderr, /requires glibc 2.35/);
    assert.deepEqual(await fs.readdir(f.home), []);
    assert.equal(await fs.stat(f.env.DOWNLOAD_LOG).catch(() => null), null);
  } finally { await f.close(); }
});
