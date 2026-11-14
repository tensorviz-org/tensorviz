const { test } = require('node:test');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const { backendPythonArgs } = require('../../out/host/backendPython.cjs');

test('native backend starts without torch and attributes capture failure to the selected Python', () => {
  const python = process.env.TENSORVIZ_NO_TORCH_PYTHON;
  const backend = process.env.TENSORVIZ_TEST_BACKEND;
  assert.ok(python && backend, 'Explicit clean Python and native backend paths are required');
  const clean = spawnSync(python, ['-I', '-c', 'import importlib.util; assert importlib.util.find_spec("torch") is None'], { encoding: 'utf8' });
  assert.equal(clean.status, 0, clean.stdout + clean.stderr);
  const result = spawnSync(python, backendPythonArgs(backend, 'services/ide/environment_setup.py', ['probe-capture']), { encoding: 'utf8' });
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
  const result = spawnSync(process.env.TENSORVIZ_NO_TORCH_PYTHON,
    ['-I', '-B', '-c', script, process.env.TENSORVIZ_TEST_BACKEND], { encoding: 'utf8' });
  assert.equal(result.status, 1, result.stdout + result.stderr);
  assert.equal(result.stderr, '');
  const failure = JSON.parse(result.stdout);
  assert.equal(failure.code, 'CAPTURE_RUNTIME_UNSUPPORTED');
  assert.match(failure.message, /TensorViz started/);
});
