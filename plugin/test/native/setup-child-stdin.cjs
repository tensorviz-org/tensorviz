const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { spawn, spawnSync } = require('node:child_process');
const { backendPythonArgs } = require('../../out/host/backendPython.cjs');

const python = process.env.TENSORVIZ_TEST_PYTHON;
const backend = process.env.TENSORVIZ_TEST_BACKEND;
assert.ok(python && path.isAbsolute(python) && backend && path.isAbsolute(backend),
  'Set explicit native test Python and backend paths.');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const alive = pid => {
  try { process.kill(pid, 0); return true; }
  catch (error) { if (error.code === 'ESRCH') return false; throw error; }
};
function start(args) {
  const child = spawn(python, args, { windowsHide: true, detached: process.platform !== 'win32',
    stdio: ['pipe', 'pipe', 'pipe'], env: { ...process.env, TENSORVIZ_PARENT_PIPE: '1' } });
  const state = { child, closed: false, stdout: '', stderr: '', error: undefined };
  child.stdout.on('data', data => { state.stdout = (state.stdout + data).slice(-1024 * 1024); });
  child.stderr.on('data', data => { state.stderr = (state.stderr + data).slice(-1024 * 1024); });
  child.on('error', error => { state.error = error; });
  child.on('close', code => { state.closed = true; state.code = code; });
  return state;
}
async function until(state, condition, milliseconds) {
  const deadline = Date.now() + milliseconds;
  while (!condition()) {
    if (state.error) throw state.error;
    assert.ok(Date.now() < deadline, `Native setup exceeded ${milliseconds}ms: ${state.stderr}`);
    await sleep(50);
  }
}
async function cleanup(state) {
  if (state.child.pid && !state.closed) {
    state.child.stdin.end();
    for (let n = 0; n < 100 && !state.closed; n++) await sleep(50);
    if (!state.closed) {
      if (process.platform === 'win32') {
        const result = spawnSync(path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'taskkill.exe'),
          ['/PID', String(state.child.pid), '/T', '/F'], { windowsHide: true, timeout: 10000 });
        assert.ifError(result.error);
      } else {
        try { process.kill(-state.child.pid, 'SIGKILL'); }
        catch (error) { if (error.code !== 'ESRCH') throw error; }
      }
      await until(state, () => state.closed, 5000);
    }
  }
}

test('compiled create-environment completes with the lifetime stdin pipe held open', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-native-setup-stdin-'));
  const prefix = path.join(root, 'venv');
  const state = start(backendPythonArgs(backend, 'services/ide/environment_setup.py', ['create-environment', prefix]));
  try {
    // ensurepip uses its bundled wheels; no index/network download is needed.
    await until(state, () => state.closed, 60000);
    assert.equal(state.code, 0, state.stdout + state.stderr);
    assert.equal(JSON.parse(state.stdout).ok, true);
    const createdPython = path.join(prefix, ...(process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python']));
    const pip = spawnSync(createdPython, ['-I', '-B', '-m', 'pip', '--version'],
      { encoding: 'utf8', timeout: 15000, stdio: ['ignore', 'pipe', 'pipe'] });
    assert.ifError(pip.error);
    assert.equal(pip.status, 0, pip.stdout + pip.stderr);
  } finally { await cleanup(state); await fs.rm(root, { recursive: true, force: true }); }
});

test('Windows native lifetime EOF still terminates detached-stdin child and grandchild',
  { skip: process.platform !== 'win32' }, async () => {
    const program = `import importlib, json, subprocess, sys, time
sys.path.insert(0, sys.argv[1])
import _tensorviz_backend
importlib.import_module('_tensorviz_backend.services.ingest.worker_lifetime').guard_parent_pipe()
code = 'import json,subprocess,sys,time; p=subprocess.Popen([sys.executable,"-I","-B","-c","import time;time.sleep(60)"],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); print(json.dumps({"grandchild":p.pid}),flush=True); time.sleep(60)'
child = subprocess.Popen([sys.executable, '-I', '-B', '-c', code], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
ids = json.loads(child.stdout.readline()); ids['child'] = child.pid
print(json.dumps(ids), flush=True)
time.sleep(60)
`;
    const state = start(['-I', '-B', '-c', program, backend]);
    try {
      await until(state, () => state.stdout.includes('\n'), 10000);
      const ids = JSON.parse(state.stdout.split('\n')[0]);
      assert.ok(alive(ids.child) && alive(ids.grandchild));
      state.child.stdin.end();
      await until(state, () => state.closed && !alive(ids.child) && !alive(ids.grandchild), 10000);
      assert.notEqual(state.code, 0);
    } finally { await cleanup(state); }
  });
