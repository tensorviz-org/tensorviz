const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { once } = require('node:events');
const { ownedTemporaryDirectory, recoverOwnedTemporary } = require('../../out/host/ownedTemporary.cjs');

test('recovery reclaims a killed owner, preserving live, young, unknown and aliased directories', { timeout: 20000 }, async () => {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-recovery-test-')));
  let child;
  try {
    const implementation = path.resolve(__dirname, '../../out/host/ownedTemporary.cjs');
    const childEnv = { ...process.env }; delete childEnv.NODE_TEST_CONTEXT;
    child = spawn(process.execPath, ['-e', `require(${JSON.stringify(implementation)}).ownedTemporaryDirectory(process.argv[1], 'tensorviz-test-').then(({directory})=>{console.log(directory);setInterval(()=>{},1000)})`, root], { env: childEnv, stdio: ['ignore', 'pipe', 'inherit'] });
    const [output] = await Promise.race([once(child.stdout, 'data'), once(child, 'exit').then(() => { throw new Error('Temporary owner exited before creating its record'); })]);
    const abandoned = output.toString().trim();
    const setAge = async directory => {
      const file = path.join(directory, 'tensorviz-owner.json');
      const record = JSON.parse(await fs.readFile(file, 'utf8'));
      record.createdAt = Date.now() - 120_000;
      await fs.writeFile(file, JSON.stringify(record));
    };
    await setAge(abandoned);
    assert.equal(await recoverOwnedTemporary(root, 'tensorviz-test-'), 0, 'a live PID is never collected');
    const live = await ownedTemporaryDirectory(root, 'tensorviz-test-');
    await setAge(live.directory);
    const young = await ownedTemporaryDirectory(root, 'tensorviz-test-');
    const unknown = path.join(root, 'tensorviz-test-legacy'); await fs.mkdir(unknown);
    await fs.writeFile(path.join(unknown, 'user.txt'), 'preserve unknown contents');
    await fs.symlink(unknown, path.join(root, 'tensorviz-test-alias'));
    const youngMarker = path.join(young.directory, 'tensorviz-owner.json');
    const owner = JSON.parse(await fs.readFile(youngMarker, 'utf8')); owner.pid = child.pid;
    await fs.writeFile(youngMarker, JSON.stringify(owner));
    child.kill('SIGKILL'); await once(child, 'exit'); child = undefined;
    await fs.writeFile(path.join(abandoned, 'plan.json'), 'private test values');
    const next = await ownedTemporaryDirectory(root, 'tensorviz-test-');
    assert.equal(await fs.stat(abandoned).catch(() => null), null, 'next use recovers a provably dead old owner');
    for (const directory of [live.directory, young.directory, unknown]) assert.ok((await fs.stat(directory)).isDirectory());
    assert.equal(await fs.readFile(path.join(unknown, 'user.txt'), 'utf8'), 'preserve unknown contents');
    await setAge(young.directory);
    assert.equal(await recoverOwnedTemporary(root, 'tensorviz-test-'), 1, 'young remnants become recoverable after the grace period');
    await live.dispose(); await next.dispose();
  } finally { child?.kill('SIGKILL'); await fs.rm(root, { recursive: true, force: true }); }
});
