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
