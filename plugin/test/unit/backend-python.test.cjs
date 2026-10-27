const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { backendPythonArgs } = require('../../out/host/backendPython.cjs');

test('source hosts retain isolated script execution; native hosts never fall back to source', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-native-dispatch-'));
  try {
    assert.deepEqual(backendPythonArgs(root, 'standalone_server.py', ['--parent-pipe']),
      ['-I', '-B', path.join(root, 'standalone_server.py'), '--parent-pipe']);
    await fs.writeFile(path.join(root, 'native-backend.json'), JSON.stringify({ abi: 'unsupported-test-abi' }));
    await fs.writeFile(path.join(root, 'standalone_server.py'), 'raise RuntimeError("SOURCE MUST NEVER RUN")');
    const args = backendPythonArgs(root, 'standalone_server.py');
    assert.ok(args.includes('-c'));
    const result = spawnSync('python3', args, { encoding: 'utf8' });
    assert.equal(result.status, 1);
    assert.equal(JSON.parse(result.stdout).code, 'UNSUPPORTED_PYTHON_ABI');
    assert.equal(result.stderr, '');
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});
