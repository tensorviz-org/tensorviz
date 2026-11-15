const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { spawn, spawnSync } = require('node:child_process');
const { once } = require('node:events');
const adapter = process.env.TENSORVIZ_TEST_ASSETS_ADAPTER || path.resolve(__dirname, '../../out/host/assets.cjs');
const { pinAssets } = require(adapter);

test('Windows asset disposal waits for a child to release its working directory', { skip: process.platform !== 'win32' }, async () => {
  const plugin = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-assets-lock-'));
  let assets, child;
  try {
    await fs.writeFile(path.join(plugin, 'plugin.json'), JSON.stringify({ version: '1.0.0' }));
    await fs.writeFile(path.join(plugin, 'runtime-manifest.json'), '{}');
    await fs.mkdir(path.join(plugin, 'backend')); await fs.mkdir(path.join(plugin, 'browser'));
    assets = await pinAssets(plugin, '1.0.0');
    child = spawn(process.execPath, ['-e', "process.stdin.resume();process.stdin.once('end',()=>setTimeout(()=>process.exit(0),350));console.log('ready')"],
      { cwd: path.join(assets.directory, 'backend'), windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
    await once(child.stdout, 'data');
    const ended = once(child, 'exit');
    child.stdin.end();
    const cleanup = assets.dispose();
    assert.equal(assets.dispose(), cleanup, 'Concurrent disposal shares one cleanup');
    await cleanup;
    assert.deepEqual(await ended, [0, null]);
    assert.equal(await fs.stat(assets.directory).catch(() => null), null);
  } finally {
    if (child && child.exitCode === null) { child.kill(); await once(child, 'exit'); }
    await assets?.dispose();
    if (assets) await fs.rm(assets.directory, { recursive: true, force: true });
    await fs.rm(plugin, { recursive: true, force: true });
  }
});

test('beforeExit waits for asynchronous cleanup exactly once', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-before-exit-'));
  try {
    const receipt = path.join(root, 'receipt');
    const code = "const fs=require('node:fs');const {disposeAssetsBeforeExit}=require(process.argv[1]);let calls=0;disposeAssetsBeforeExit({dispose:()=>new Promise(resolve=>setTimeout(()=>{fs.writeFileSync(process.argv[2],String(++calls));resolve()},100))});";
    const result = spawnSync(process.execPath, ['-e', code, adapter, receipt], { encoding: 'utf8', timeout: 5000, windowsHide: true });
    assert.ifError(result.error); assert.equal(result.status, 0, result.stderr); assert.equal(result.stderr, '');
    assert.equal(await fs.readFile(receipt, 'utf8'), '1');
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});

test('failed asynchronous cleanup reports failure without an uncaught exception', () => {
  const code = "const {disposeAssetsBeforeExit}=require(process.argv[1]);disposeAssetsBeforeExit({dispose:async()=>{throw Object.assign(new Error('private fixture path'),{code:'EBUSY'})}});";
  const result = spawnSync(process.execPath, ['-e', code, adapter], { encoding: 'utf8', timeout: 5000, windowsHide: true });
  assert.ifError(result.error); assert.equal(result.status, 1);
  assert.match(result.stderr, /could not remove its private session resources/);
  assert.doesNotMatch(result.stderr, /private fixture path|Unhandled|Error:|at /);
});
