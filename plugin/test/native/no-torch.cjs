const { test } = require('node:test');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs/promises');
const path = require('node:path');
const { backendPythonArgs } = require('../../out/host/backendPython.cjs');

// Synchronous subprocesses block the test runner's timeout timer. Bound each
// native probe at the OS process boundary, and identify the active call in CI.
function runProbe(label, python, args) {
  process.stderr.write(`[no-torch] starting ${label}\n`);
  const result = spawnSync(python, args, {
    encoding: 'utf8', timeout: 60000, killSignal: 'SIGKILL', maxBuffer: 1024 * 1024,
  });
  process.stderr.write(`[no-torch] finished ${label}: status=${result.status} signal=${result.signal || 'none'}\n`);
  assert.ifError(result.error && new Error(`${label}: ${result.error.message}`));
  return result;
}

test('native backend starts without torch and attributes capture failure to the selected Python', () => {
  const python = process.env.TENSORVIZ_NO_TORCH_PYTHON || process.env.TENSORVIZ_COMPILER_PYTHON;
  const backend = process.env.TENSORVIZ_TEST_BACKEND;
  assert.ok(python && backend, 'Explicit clean Python and native backend paths are required');
  const clean = runProbe('clean interpreter inspection', python, ['-I', '-c', 'import importlib.util; assert importlib.util.find_spec("torch") is None']);
  assert.equal(clean.status, 0, clean.stdout + clean.stderr);
  const result = runProbe('compiled missing-Torch capture probe', python, backendPythonArgs(backend, 'services/ide/environment_setup.py', ['probe-capture']));
  assert.equal(result.status, 1, result.stdout + result.stderr);
  assert.equal(result.stderr, '', 'Missing optional capture dependencies must not produce a startup traceback');
  const failure = JSON.parse(result.stdout);
  assert.equal(failure.ok, false);
  assert.equal(failure.code, 'MISSING_DEPENDENCIES');
  assert.match(failure.message, /torch/);
  assert.match(failure.message, /runtime is available/);
});

test('an incompatible capture library is attributed to capture, not TensorViz startup', () => {
  const script = `import json, sys, types
sys.path.insert(0, sys.argv[1])
import _tensorviz_backend
# A loadable module with an incompatible API simulates a broken Torch install.
sys.modules['torch'] = types.ModuleType('torch')
sys.exit(_tensorviz_backend.run('services/ide/environment_setup.py', ['probe-capture']))
`;
  const result = runProbe('compiled incompatible-Torch capture probe', process.env.TENSORVIZ_NO_TORCH_PYTHON || process.env.TENSORVIZ_COMPILER_PYTHON,
    ['-I', '-B', '-c', script, process.env.TENSORVIZ_TEST_BACKEND]);
  assert.equal(result.status, 1, result.stdout + result.stderr);
  assert.equal(result.stderr, '');
  const failure = JSON.parse(result.stdout);
  assert.equal(failure.code, 'CAPTURE_RUNTIME_UNSUPPORTED');
  assert.match(failure.message, /TensorViz started/);
});

// Keep native packaging privacy guarantees identical on newly admitted targets.
test('native backend package excludes source files and repository build paths', async () => {
  const backend = process.env.TENSORVIZ_TEST_BACKEND;
  const metadata = JSON.parse(await fs.readFile(path.join(backend, 'native-backend.json'), 'utf8'));
  const bytes = await fs.readFile(path.join(backend, metadata.binary));
  assert.ok(!bytes.includes(Buffer.from(path.resolve(__dirname, '../../..'))),
    'Native backend must not retain the repository build path');
  const names = await fs.readdir(backend, {recursive:true});
  assert.ok(!names.some(name => /\.(py|pyc|pyo)$/.test(name)), 'Native runtime must not ship backend source/bytecode');
});
