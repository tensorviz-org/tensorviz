const { test } = require('node:test');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const { once } = require('node:events');
const { Backend } = require(process.env.TENSORVIZ_TEST_BACKEND_ADAPTER || '../../out/host/backend.cjs');

test('dispose releases the owner pipe even without a platform kill target', async () => {
  const child = spawn(process.execPath, ['-e', "process.stdin.resume();process.stdin.once('end',()=>process.exit(0));console.log('ready')"],
    { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
  const backend = new Backend('/unused/backend', '/unused/browser');
  try {
    await once(child.stdout, 'data');
    const ended = once(child, 'exit');
    // A failed/unavailable platform termination must not leave the owned pipe open.
    backend.child = { stdin: child.stdin };
    backend.dispose();
    const outcome = await Promise.race([ended, new Promise(resolve => {
      const timer = setTimeout(() => resolve('deadline'), 1000); timer.unref();
    })]);
    assert.deepEqual(outcome, [0, null]);
    assert.equal(backend.ready, false);
    assert.doesNotThrow(() => backend.dispose());
  } finally {
    if (child.exitCode === null) { child.kill(); await once(child, 'exit'); }
  }
});

test('dispose before service startup remains safe', () => {
  const backend = new Backend('/unused/backend', '/unused/browser');
  assert.doesNotThrow(() => { backend.dispose(); backend.dispose(); });
});
