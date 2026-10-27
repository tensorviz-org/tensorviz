const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const https = require('node:https');
const { createHash } = require('node:crypto');
const { spawn, execFileSync } = require('node:child_process');

test('thin installer verifies HTTPS downloads, supports offline reuse, rejects tampering and cleans cancelled downloads', { timeout: 30000 }, async () => {
  assert.equal(`${process.platform}-${process.arch}`, 'darwin-arm64', 'Run installer acceptance on its supported platform');
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-installer-'));
  const home = path.join(root, 'home'), plugin = path.join(root, 'plugin with spaces'), tmp = path.join(root, 'sessions');
  let server, running;
  try {
    for (const directory of [home, tmp, path.join(plugin, 'scripts'), path.join(root, 'payload')]) await fs.mkdir(directory, { recursive: true });
    await fs.copyFile(path.resolve(__dirname, '../../integration/launch.sh'), path.join(plugin, 'scripts/launch.sh'));
    await fs.writeFile(path.join(root, 'payload/tensorviz-host'), '#!/bin/sh\nIFS= read -r line\nprintf "verified:%s\\n" "$line"\n', { mode: 0o755 });
    execFileSync('/usr/bin/tar', ['-czf', path.join(root, 'runtime.tar.gz'), '-C', path.join(root, 'payload'), 'tensorviz-host'], { env: { ...process.env, COPYFILE_DISABLE: '1' } });
    const archive = await fs.readFile(path.join(root, 'runtime.tar.gz'));
    const sha256 = createHash('sha256').update(archive).digest('hex');
    execFileSync('openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1', '-subj', '/CN=localhost',
      '-addext', 'subjectAltName=IP:127.0.0.1', '-keyout', path.join(root, 'test.key'), '-out', path.join(root, 'test.pem')], { stdio: 'ignore' });
    let requests = 0, mode = 'ok', downloading;
    server = https.createServer({ key: await fs.readFile(path.join(root, 'test.key')), cert: await fs.readFile(path.join(root, 'test.pem')) }, (req, res) => {
      requests++;
      if (mode === 'interrupted') { res.writeHead(200, { 'Content-Length': archive.length }); res.write(archive.subarray(0, 20)); setTimeout(() => res.destroy(), 20); }
      else if (mode === 'slow') { res.writeHead(200, { 'Content-Length': archive.length }); res.write(archive.subarray(0, 20)); downloading?.(); }
      else { res.writeHead(200, { 'Content-Length': archive.length }); res.end(archive); }
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    const lock = { version: '0.3.0', sha256, bytes: archive.length, url: `https://127.0.0.1:${server.address().port}/runtime.tar.gz` };
    const lockFile = path.join(plugin, 'runtime-lock.json');
    await fs.writeFile(lockFile, JSON.stringify(lock));
    const launch = () => {
      const child = spawn('/bin/sh', [path.join(plugin, 'scripts/launch.sh')], { env: {
        HOME: home, TMPDIR: tmp, PATH: '/usr/bin:/bin', CURL_CA_BUNDLE: path.join(root, 'test.pem'),
      }, stdio: ['pipe', 'pipe', 'pipe'] });
      running = child;
      let stdout = '', stderr = '';
      child.stdout.on('data', b => stdout += b); child.stderr.on('data', b => stderr += b);
      child.stdin.end('hello world\n');
      return { child, done: new Promise(resolve => child.on('close', code => { running = undefined; resolve({ code, stdout, stderr }); })) };
    };
    let result = await launch().done;
    assert.equal(result.code, 0, result.stderr);
    assert.equal(result.stdout, 'verified:hello world\n', 'Download diagnostics must not corrupt MCP stdout');
    assert.equal(requests, 1);
    assert.deepEqual(await fs.readdir(tmp), []);
    result = await launch().done;
    assert.equal(result.code, 0, result.stderr); assert.equal(requests, 1, 'A cached archive needs no network');
    const cache = path.join(home, 'Library/Caches/dev.tensorviz/runtime'), cached = path.join(cache, `${sha256}.tar.gz`);
    await fs.writeFile(cached, 'modified archive');
    result = await launch().done;
    assert.notEqual(result.code, 0); assert.match(result.stderr, /damaged/); assert.equal(result.stdout, '');
    await fs.rm(cached);
    await fs.writeFile(lockFile, JSON.stringify({ ...lock, sha256: '0'.repeat(64) }));
    result = await launch().done;
    assert.notEqual(result.code, 0); assert.match(result.stderr, /checksum mismatch/); assert.equal(result.stdout, '');
    assert.deepEqual(await fs.readdir(cache), []);
    await fs.writeFile(lockFile, JSON.stringify(lock));
    mode = 'interrupted'; result = await launch().done;
    assert.notEqual(result.code, 0); assert.match(result.stderr, /download failed/);
    assert.deepEqual(await fs.readdir(cache), []);
    mode = 'slow';
    const started = new Promise(resolve => downloading = resolve);
    const pending = launch(); await started; pending.child.kill('SIGTERM');
    result = await pending.done;
    assert.equal(result.code, 143);
    assert.deepEqual(await fs.readdir(cache), []);
    assert.deepEqual(await fs.readdir(tmp), []);
  } finally {
    running?.kill('SIGKILL'); server?.closeAllConnections(); server?.close();
    await fs.rm(root, { recursive: true, force: true });
  }
});
