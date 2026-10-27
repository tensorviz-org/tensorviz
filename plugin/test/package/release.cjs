const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { createHash } = require('node:crypto');

test('release archive verifies, detects changed/extra files, and captures through both installed clients', { timeout: 180000 }, async () => {
  const { verifyRelease } = await import('../../scripts/package.mjs');
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-release-test-'));
  const version = require('../../package.json').version;
  const archive = path.resolve(__dirname, `../../dist/tensorviz-${version}.tar.gz`);
  try {
    const expected = (await fs.readFile(`${archive}.sha256`, 'utf8')).split(' ')[0];
    assert.equal(createHash('sha256').update(await fs.readFile(archive)).digest('hex'), expected);
    execFileSync('tar', ['-xzf', archive, '-C', root]);
    const extracted = path.join(root, `tensorviz-${version}`), plugin = path.join(extracted, 'plugins/tensorviz');
    const manifest = await verifyRelease(extracted);
    assert.ok(Object.keys(manifest.sources).length > 100);
    assert.equal(manifest.productionTldrawLicense, 'unverified');
    const notices = await fs.readFile(path.join(plugin, 'DEPENDENCY_NOTICES.md'), 'utf8');
    assert.match(notices, /@modelcontextprotocol\/sdk@/);
    assert.match(notices, /tldraw@5.0.1/);
    assert.match(notices, /Copyright \(c\) 2022 WorkOS/);
    assert.ok(!notices.includes('firebase@'), 'Hosted auth must not enter the browser artifact');
    const launcher = path.join(plugin, 'scripts/launch.cjs'), original = await fs.readFile(launcher);
    await fs.appendFile(launcher, '\n// changed\n');
    await assert.rejects(verifyRelease(extracted), /integrity mismatch/);
    await fs.writeFile(launcher, original);
    const extra = path.join(plugin, '.env.local');
    await fs.writeFile(extra, 'not-a-secret');
    await assert.rejects(verifyRelease(extracted), /excluded file/);
    await fs.unlink(extra);
    const hostPath = path.join(plugin, 'runtime/host.cjs'), hostBytes = await fs.readFile(hostPath);
    await fs.writeFile(hostPath, Buffer.alloc(5 * 1024 * 1024));
    await assert.rejects(verifyRelease(extracted), /5 MiB directory limit/);
    await fs.writeFile(hostPath, hostBytes);
    await verifyRelease(extracted);
    const clientEnv = { ...process.env, TENSORVIZ_TEST_PLUGIN: plugin };
    delete clientEnv.NODE_TEST_CONTEXT;
    delete clientEnv.TENSORVIZ_TEST_CODEX_AGENT;
    delete clientEnv.TENSORVIZ_TEST_AGENT_AUTH_FILE;
    for (const client of ['codex', 'claude']) {
      const result = execFileSync(process.execPath, ['--test', path.resolve(__dirname, `../clients/${client}.cjs`)], {
        cwd: root, env: clientEnv, encoding: 'utf8', timeout: 90000,
      });
      assert.match(result, /Captured Tiny and inspected measured shapes/);
      console.log(`Extracted release passed actual ${client} installation, native approval bridge, capture, inspection and uninstall.`);
    }
  } finally { await fs.rm(root, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 }); }
});
