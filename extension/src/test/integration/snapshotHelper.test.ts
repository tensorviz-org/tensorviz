import * as assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';

const core = require('@tensorviz/graph-host/hostSnapshot');
const childProcess = require('node:child_process');
const realSpawn = childProcess.spawn;
const repo = path.resolve(__dirname, '../../../..');
const backend = path.join(repo, 'backend');
const python = process.env.TENSORVIZ_TEST_PYTHON || path.join(repo, 'venv/bin/python');
const MODEL = 'import torch\nfrom torch import nn\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.a = nn.Linear(8, 8)\n    def forward(self, x):\n        return self.a(x)\n';

/** The answer of the original one-shot snapshot, computed independently. */
function oneShot(request: object): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const child = realSpawn.call(childProcess, python, ['-I', '-B', path.join(backend, 'services/validation/host_snapshot.py')],
      { cwd: backend, env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' } });
    let out = '';
    child.stdout.on('data', (chunk: Buffer) => { out += chunk; });
    child.once('close', (code: number) => code === 0 ? resolve(JSON.parse(out)) : reject(new Error(`exit ${code}`)));
    child.stdin.end(JSON.stringify(request));
  });
}

const alive = (pid: number | undefined) => { if (!pid) return false; try { process.kill(pid, 0); return true; } catch { return false; } };
async function gone(pid: number | undefined, ms = 3000) {
  const end = Date.now() + ms;
  while (Date.now() < end) { if (!alive(pid)) return true; await new Promise(r => setTimeout(r, 25)); }
  return !alive(pid);
}

suite('host snapshot helper with real Python', function () {
  this.timeout(60000);
  let root: string;
  const spawns: string[][] = [];
  const original = childProcess.spawn;
  const savedIdle = core.SNAPSHOT_HELPER_IDLE_MS;
  const request = (extra: object = {}) => ({ root, documents: {}, executable: python, environmentRevision: 0, session: 's',
    workspaceId: 'w', runId: 'r', generation: 1, modelFile: 'model.py', className: 'Model', timeoutMs: 30000,
    execution: { inputs: { args: [{ shape: [2, 8], dtype: 'float32', fill: 'randn' }] } }, ...extra });
  const read = (body: object, signal = new AbortController().signal) => core.readHostSnapshot(python, backend, body, signal);
  const served = () => spawns.filter(args => args.includes('--serve')).length;
  const oneShots = () => spawns.filter(args => args.some(a => a.endsWith('host_snapshot.py')) && !args.includes('--serve')).length;
  // pgrep -f matches its own argv, and execSync runs that argv through a shell, so the
  // old pattern selected the wrapper and process.kill failed with ESRCH. A character
  // class is not in the helper's command line, and spawnSync does not add a shell.
  const helperPids = (): number[] => {
    const result = childProcess.spawnSync('pgrep', ['-f', '[h]ost_snapshot.py --serve'], { encoding: 'utf8' });
    if (result.error) throw result.error;
    if (result.status !== 0 && result.status !== 1) throw new Error(`pgrep exited ${result.status}: ${result.stderr}`);
    return (result.stdout || '').split('\n').map((line: string) => Number(line.trim())).filter((pid: number) => Number.isInteger(pid) && pid > 0);
  };
  const processArgs = (pid: number): string => {
    const result = childProcess.spawnSync('ps', ['-p', String(pid), '-o', 'args='], { encoding: 'utf8' });
    return (result.stdout || '').trim();
  };
  const requireHelper = async (): Promise<number> => {
    const end = Date.now() + 2000;
    let last = '';
    while (Date.now() < end) {
      for (const pid of helperPids()) {
        if (!alive(pid)) continue;
        const args = processArgs(pid);
        last = args;
        if (args.includes('host_snapshot.py') && args.includes('--serve') && !args.includes('pgrep')) return pid;
      }
      await new Promise(r => setTimeout(r, 20));
    }
    assert.fail(`snapshot helper is not running (${last || 'no process'})`);
  };

  suiteSetup(() => assert.ok(existsSync(python), 'set TENSORVIZ_TEST_PYTHON'));
  setup(async () => {
    core.disposeSnapshotHelpers(); // helpers are per host; start each test from none
    root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tv-snapshot-')));
    await fs.writeFile(path.join(root, 'model.py'), MODEL);
    spawns.length = 0;
    childProcess.spawn = (command: string, args: string[], ...rest: unknown[]) => {
      spawns.push(args);
      return original.call(childProcess, command, args, ...rest);
    };
  });
  teardown(async () => {
    childProcess.spawn = original;
    core.disposeSnapshotHelpers();
    core.SNAPSHOT_HELPER_IDLE_MS = savedIdle;
    await fs.rm(root, { recursive: true, force: true });
  });

  test('answers equal the one-shot snapshot and reuse one process', async () => {
    const expected = await oneShot(request());
    assert.deepEqual(await read(request()), expected);
    assert.deepEqual(await read(request()), expected);
    assert.deepEqual(await read(request()), expected);
    assert.equal(served(), 1, 'one helper for three snapshots');
    assert.equal(oneShots(), 0);
  });

  test('a saved file change between snapshots is seen', async () => {
    const before = await read(request()) as { applicabilityDigest: string };
    await fs.writeFile(path.join(root, 'model.py'), MODEL.replace('nn.Linear(8, 8)', 'nn.Linear(8, 8, bias=False)'));
    const after = await read(request()) as { applicabilityDigest: string };
    assert.notEqual(after.applicabilityDigest, before.applicabilityDigest);
    assert.deepEqual(after, await oneShot(request()));
  });

  test('a failing request fails alone; the next one on the same helper succeeds', async () => {
    await read(request());
    await assert.rejects(read(request({ root: path.join(root, 'missing') })), /snapshot_failed/);
    assert.deepEqual(await read(request()), await oneShot(request()));
    assert.equal(served(), 1);
  });

  test('a snapshot while the helper is busy falls back to a one-shot process', async () => {
    await read(request());
    const [a, b] = await Promise.all([read(request()), read(request())]);
    assert.deepEqual(a, b);
    assert.equal(oneShots(), 1);
  });

  test('cancelling mid-snapshot rejects as before, stops the helper, and the next call works', async () => {
    await read(request());
    const pid = await requireHelper();
    const controller = new AbortController();
    const pending = read(request(), controller.signal);
    controller.abort();
    await assert.rejects(pending, /snapshot_failed/);
    assert.ok(await gone(pid), 'the cancelled helper is stopped');
    assert.equal(alive(pid), false);
    assert.ok(!helperPids().includes(pid), 'the cancelled helper is no longer listed');
    assert.deepEqual(await read(request()), await oneShot(request()));
    assert.equal(served(), 2, 'a fresh helper replaces the stopped one');
    assert.notEqual(await requireHelper(), pid, 'the replacement helper is a new process');
  });

  test('a helper that died between snapshots is replaced without failing the call', async () => {
    await read(request());
    const pid = await requireHelper();
    process.kill(pid, 'SIGKILL');
    assert.ok(await gone(pid));
    assert.deepEqual(await read(request()), await oneShot(request()));
  });

  test('an already-aborted signal rejects without starting anything', async () => {
    const controller = new AbortController(); controller.abort();
    await assert.rejects(read(request(), controller.signal), /cancelled/);
    assert.equal(spawns.length, 0);
  });

  test('the idle timeout stops an unused helper', async () => {
    core.SNAPSHOT_HELPER_IDLE_MS = 1000;
    await read(request());
    const pid = await requireHelper();
    assert.ok(await gone(pid, 5000), 'stopped after the idle timeout');
    assert.deepEqual(await read(request()), await oneShot(request()));
  });

  test('dispose stops every helper', async () => {
    await read(request());
    const pid = await requireHelper();
    core.disposeSnapshotHelpers();
    assert.ok(await gone(pid));
  });
});
