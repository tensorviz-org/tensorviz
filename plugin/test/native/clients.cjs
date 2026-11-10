const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const https = require('node:https');
const { spawn, execFileSync } = require('node:child_process');
const { createHash } = require('node:crypto');

test('Codex and Claude install only the integration, download the compiled runtime, capture and uninstall', { timeout: process.platform === 'linux' ? 600000 : 420000 }, async () => {
  const { nativePlatform, runtimeArchive } = await import('../../scripts/native-platforms.mjs');
  const target = nativePlatform();
  const { exportPublic } = await import('../../scripts/export-public.mjs');
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-native-clients-'));
  const version = (await import('../../scripts/release-version.mjs')).pluginVersion();
  const archive = await fs.readFile(path.resolve(__dirname, `../../dist/${runtimeArchive({ version, platform: target.key })}`));
  let server;
  try {
    execFileSync('openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1', '-subj', '/CN=localhost',
      '-addext', 'subjectAltName=IP:127.0.0.1', '-keyout', path.join(root, 'test.key'), '-out', path.join(root, 'test.pem')], { stdio: 'ignore' });
    let requests = 0;
    server = https.createServer({ key: await fs.readFile(path.join(root, 'test.key')), cert: await fs.readFile(path.join(root, 'test.pem')) }, (_req, res) => {
      requests++; res.writeHead(200, { 'Content-Length': archive.length }); res.end(archive);
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    const runtimeLock = { formatVersion: 1, version, platform: target.key, pythonABI: target.abi,
      url: `https://127.0.0.1:${server.address().port}/runtime.tar.gz`, bytes: archive.length,
      sha256: createHash('sha256').update(archive).digest('hex') };
    const integration = path.join(root, 'integration');
    await exportPublic(integration, { runtimeLock });
    for (const client of ['codex', 'claude']) {
      const before = requests;
      const env = { ...process.env, TENSORVIZ_TEST_PLUGIN: path.join(integration, 'plugins/tensorviz'), TENSORVIZ_TEST_DOWNLOAD_CA: path.join(root, 'test.pem') };
      for (const key of ['NODE_TEST_CONTEXT', 'TENSORVIZ_TEST_CODEX_AGENT', 'TENSORVIZ_TEST_AGENT_AUTH_FILE', 'TENSORVIZ_TEST_INSTALL_PERMISSIONS']) delete env[key];
      const result = await new Promise((resolve, reject) => {
        const child = spawn(process.execPath, ['--test', path.resolve(__dirname, `../clients/${client}.cjs`)], { env, stdio: ['ignore', 'pipe', 'pipe'] });
        let output = '';
        child.stdout.on('data', data => output += data); child.stderr.on('data', data => output += data);
        child.once('error', reject);
        child.once('close', code => resolve({ code, output }));
      });
      assert.equal(result.code, 0, result.output);
      assert.match(result.output, /Captured Tiny and inspected measured shapes/);
      assert.ok(requests > before, `${client} must perform a fresh download into its disposable cache`);
      console.log(result.output.trim());
    }
  } finally { server?.closeAllConnections(); server?.close(); await fs.rm(root, { recursive: true, force: true }); }
});
