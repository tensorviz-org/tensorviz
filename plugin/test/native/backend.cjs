const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { spawnSync, spawn } = require('node:child_process');
const { createHash } = require('node:crypto');
const { createInterface } = require('node:readline');
const { backendPythonArgs } = require('../../out/host/backendPython.cjs');
const { runEnvironmentHelper } = require('../../../packages/graph-host/dist/projectEnvironmentCore.js');

const python = process.env.TENSORVIZ_TEST_PYTHON;
const backend = process.env.TENSORVIZ_TEST_BACKEND;
assert.ok(python && path.isAbsolute(python) && backend && path.isAbsolute(backend), 'Set explicit native test Python and backend paths.');
const inspect = args => {
  const result = spawnSync(python, backendPythonArgs(backend, 'services/ide/environment_setup.py', args), { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  return JSON.parse(result.stdout);
};

test('compiled environment inspection and model probes work without Python source or public package collisions', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-native-probe-'));
  try {
    assert.ok(await fs.stat(path.join(backend, 'native-backend.json')));
    const entries = await fs.readdir(backend, { recursive: true });
    assert.ok(!entries.some(name => /\.(py|pyc|pyo)$/.test(name)), 'Native runtime must contain no backend Python source or bytecode');
    const inspected = inspect(['inspect']);
    assert.equal(inspected.ok, true, JSON.stringify(inspected));
    assert.equal(inspected.environment.executable, python);
    for (const name of ['services', 'models', 'utils']) {
      await fs.mkdir(path.join(root, name));
      await fs.writeFile(path.join(root, name, '__init__.py'), 'value = 7\n');
    }
    const file = path.join(root, 'model.py');
    await fs.writeFile(file, 'import services, models, utils\nassert services.value == models.value == utils.value == 7\n');
    const probed = inspect(['probe-model', root, file]);
    assert.equal(probed.ok, true, JSON.stringify(probed));
    for (const args of [['probe-capture'], ['probe-model', root, file]]) {
      const probed = await runEnvironmentHelper(python, backend, ['--target-python', python, ...args], () => false, true);
      assert.equal(probed.ok, true, JSON.stringify(probed));
      if (args[0] === 'probe-capture') {
        assert.ok(probed.torch);
        assert.equal(probed.executable, python, 'The worker probe must run in the selected Python');
      }
    }
    const native = spawnSync(python, ['-I', '-B', '-c', 'import sys,pathlib; sys.path.insert(0,sys.argv[1]); import _tensorviz_backend as m; assert type(m.run).__name__ == "compiled_function"; assert not pathlib.Path(m.__file__).exists(); print("native")', backend], { encoding: 'utf8' });
    assert.equal(native.status, 0, native.stderr); assert.equal(native.stdout.trim(), 'native');
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});

test('a compiled backend with an incompatible architecture is rejected before native import', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-native-abi-'));
  try {
    const manifest = JSON.parse(await fs.readFile(path.join(backend, 'native-backend.json')));
    await fs.writeFile(path.join(root, 'native-backend.json'), JSON.stringify({ ...manifest, architecture: 'unsupported-fixture' }));
    const result = spawnSync(python, backendPythonArgs(root, 'standalone_server.py'), { encoding: 'utf8' });
    assert.equal(result.status, 1);
    assert.equal(JSON.parse(result.stdout).code, 'UNSUPPORTED_PYTHON_ABI');
    assert.equal(result.stderr, '');
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});

test('compiled service authenticates HTTP requests and projects source without importing the model', { timeout: 20000 }, async () => {
  const token = 'test-only-private-launch-token-1234567890';
  const child = spawn(python, backendPythonArgs(backend, 'standalone_server.py', ['--parent-pipe']), {
    env: { ...process.env, TENSORVIZ_LOCAL_TOKEN: token }, stdio: ['pipe', 'pipe', 'pipe'],
  });
  let stderr = '';
  child.stderr.on('data', bytes => stderr += bytes);
  const lines = createInterface({ input: child.stdout });
  try {
    const port = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`Service startup timed out: ${stderr}`)), 10000);
      child.once('exit', () => { clearTimeout(timer); reject(new Error(stderr)); });
      lines.once('line', line => { clearTimeout(timer); resolve(JSON.parse(line).port); });
    });
    const origin = `http://127.0.0.1:${port}`;
    let response;
    for (let attempt = 0; attempt < 30; attempt++) {
      try { response = await fetch(`${origin}/api/health`); break; }
      catch { await new Promise(resolve => setTimeout(resolve, 50)); }
    }
    assert.equal(response.status, 401);
    response = await fetch(`${origin}/api/health`, { headers: { 'X-tensorViz-Token': token } });
    assert.equal((await response.json()).host, 'standalone');
    const source = 'raise RuntimeError("STATIC SOURCE MUST NOT EXECUTE")\nfrom torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4,3)\n def forward(self,x): return self.fc(x)\n';
    const digest = createHash('sha256').update(source).digest('hex');
    response = await fetch(`${origin}/api/ide/multifile/project`, { method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-tensorViz-Token': token },
      body: JSON.stringify({ protocol_version: 1, root: { file: 'model.py', cls: 'Tiny', source_digest: digest },
        sources: [{ file: 'model.py', source, digest }] }),
    });
    const projection = await response.text();
    assert.equal(response.status, 200, projection);
    assert.match(projection, /Linear/);
    const closed = new Promise(resolve => child.once('exit', resolve));
    child.stdin.end();
    await closed;
  } finally { lines.close(); child.kill('SIGKILL'); }
});
