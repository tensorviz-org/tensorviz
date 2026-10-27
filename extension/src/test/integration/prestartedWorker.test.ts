import * as assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { TraceWorker } from '../../worker';
import type { TraceOutcome } from '@tensorviz/graph-host/traceWorkerCore';
import { executionSpecSchema } from '@tensorviz/graph-host/executionSpec';

const core = require('@tensorviz/graph-host/traceWorkerCore');
const repo = path.resolve(__dirname, '../../../..');
const backend = path.join(repo, 'backend');
const python = process.env.TENSORVIZ_TEST_PYTHON || path.join(repo, 'venv/bin/python');
const USED = '[worker] using pre-started worker';

function model(layers: number, extra = ''): string {
  const body = Array.from({ length: layers }, (_, i) => `        self.l${i} = nn.Linear(8, 8)`).join('\n');
  const calls = Array.from({ length: layers }, (_, i) => `        x = torch.relu(self.l${i}(x))`).join('\n');
  return `import torch
from torch import nn
# tensorviz: input=2,8
class Model(nn.Module):
    def __init__(self):
        super().__init__()
${body}
    def forward(self, x):
${extra}${calls}
        return x
`;
}

const alive = (pid: number | undefined): boolean => {
  if (!pid) return false;
  try { process.kill(pid, 0); return true; } catch { return false; }
};
async function gone(pid: number | undefined, ms = 3000): Promise<boolean> {
  const end = Date.now() + ms;
  while (Date.now() < end) { if (!alive(pid)) return true; await new Promise(r => setTimeout(r, 25)); }
  return !alive(pid);
}
type Internal = { standby?: { child: { pid?: number } } };
const standbyPid = (worker: TraceWorker): number | undefined => (worker as unknown as Internal).standby?.child.pid;
/** What the run measured: the ordered operations of its semantic capture (this is
 *  where explicit and automatic runs carry the graph; `result.graph` is empty for
 *  them on main too) plus the legacy graph's node count. Never vacuous: a run with
 *  no operations fails here instead of comparing nothing with nothing. */
function signature(outcome: TraceOutcome): string {
  assert.ok(outcome.ok, JSON.stringify(outcome).slice(0, 1500));
  const result = outcome.envelope.result as { graph: { nodes: unknown[] };
    diagnostics: { semantic_capture?: { operations: { target: string }[] } } };
  const operations = (result.diagnostics.semantic_capture?.operations ?? []).map(op => op.target);
  assert.ok(operations.length + result.graph.nodes.length > 0, 'the run must have measured something');
  return JSON.stringify([operations, result.graph.nodes.length]);
}
const ALT_PYTHON = process.env.TENSORVIZ_TEST_ALT_PYTHON
  || path.join(repo, '..', 'tv-incremental', '.venv-verify', 'bin', 'python');

suiteSetup(() => {
  assert.ok(existsSync(python), `these suites spawn a real interpreter: set TENSORVIZ_TEST_PYTHON (looked at ${python})`);
});

suite('pre-started single-use worker with real Python', function () {
  this.timeout(60000);
  let root: string, file: string, logs: string[], worker: TraceWorker;
  let interpreter = python;
  const savedIdle = core.STANDBY_IDLE_MS;
  setup(async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), 'tv-standby-'));
    file = path.join(root, 'model.py');
    await fs.writeFile(file, model(2));
    logs = [];
    interpreter = python;
    worker = new TraceWorker(backend, { appendLine: (line: string) => logs.push(line) } as unknown as vscode.OutputChannel,
      { interpreter: async () => interpreter });
  });
  teardown(async () => {
    const pid = standbyPid(worker);
    worker.dispose();
    core.STANDBY_IDLE_MS = savedIdle;
    delete process.env.TV_STANDBY_TEST_VARIABLE;
    assert.ok(await gone(pid), 'dispose must not leave a standby behind');
    await fs.rm(root, { recursive: true, force: true });
  });
  const request = () => ({ file, cls: 'Model', workspace: root, timeoutMs: 30000 });
  const graph = () => worker.run(request());
  /** Ground truth: the same request on a brand-new worker that has no standby. */
  async function fresh(): Promise<string> {
    const reference = new TraceWorker(backend, { appendLine() {} } as unknown as vscode.OutputChannel,
      { interpreter: async () => interpreter });
    try { return signature(await reference.run(request())); } finally { reference.dispose(); }
  }
  const used = () => logs.filter(line => line === USED).length;

  test('the first run spawns; the next uses the standby and gives the same graph', async () => {
    const first = signature(await graph());
    assert.equal(used(), 0);
    assert.ok(alive(standbyPid(worker)), 'a standby is started after an eligible run');
    const second = signature(await graph());
    assert.equal(used(), 1);
    assert.equal(second, first);
    assert.equal(second, await fresh());
  });

  test('an edited model is imported fresh by the standby, never served stale', async () => {
    const before = signature(await graph());
    await fs.writeFile(file, model(5));
    const after = signature(await graph());
    assert.equal(used(), 1);
    assert.notEqual(after, before, 'the edit must be visible');
    assert.equal(after, await fresh(), 'the standby run must equal a fresh process on the edited file');
    await fs.writeFile(file, model(1));
    const shrunk = signature(await graph());
    assert.equal(used(), 2);
    assert.equal(shrunk, await fresh());
  });

  test('state set by user code in one run does not reach the next', async () => {
    await fs.writeFile(file, model(1, '        torch.set_num_threads(3)\n        assert getattr(torch, "_tv_marker", None) is None\n        torch._tv_marker = 1\n'));
    signature(await graph());
    signature(await graph()); // fails if the marker survived: the process must be new
    assert.equal(used(), 1);
  });

  test('an explicit execution plan run also uses the standby', async () => {
    const execution = executionSpecSchema.parse({ inputs: { args: [{ shape: [2, 8], dtype: 'float32', fill: 'ones' }] } });
    const run = () => worker.run({ file, cls: 'Model', workspace: root, execution, timeoutMs: 30000 });
    const first = signature(await run());
    const second = signature(await run());
    assert.equal(used(), 1);
    assert.equal(second, first);
    const reference = new TraceWorker(backend, { appendLine() {} } as unknown as vscode.OutputChannel, { interpreter: async () => python });
    try { assert.equal(second, signature(await reference.run({ file, cls: 'Model', workspace: root, execution, timeoutMs: 30000 }))); }
    finally { reference.dispose(); }
  });

  test('a changed environment discards the standby and spawns as before', async () => {
    await graph();
    const stale = standbyPid(worker);
    process.env.TV_STANDBY_TEST_VARIABLE = 'changed';
    signature(await graph());
    assert.equal(used(), 0, 'a standby whose torch import saw a different environment is never used');
    assert.ok(await gone(stale), 'the mismatched standby is retired');
    assert.ok(alive(standbyPid(worker)), 'and replaced by one for the new environment');
    signature(await graph());
    assert.equal(used(), 1);
  });

  test('a different interpreter discards the standby', async function () {
    await graph();
    const stale = standbyPid(worker);
    if (!existsSync(ALT_PYTHON)) this.skip();
    interpreter = ALT_PYTHON;
    signature(await graph());
    assert.equal(used(), 0);
    assert.ok(await gone(stale));
  });

  test('a standby that died while idle is skipped without failing the run', async () => {
    await graph();
    const pid = standbyPid(worker)!;
    process.kill(-pid, 'SIGKILL');
    assert.ok(await gone(pid));
    signature(await graph());
    assert.equal(used(), 0);
  });

  test('cancelling a run on a claimed standby stops it', async () => {
    await graph();
    await fs.writeFile(file, model(1, '        import time; time.sleep(20)\n'));
    const pending = graph();
    await new Promise(r => setTimeout(r, 1500));
    assert.equal(used(), 1);
    worker.cancel();
    const outcome = await pending;
    assert.equal(outcome.ok, false);
  });

  test('two back-to-back runs: one claims the standby, the newer one completes', async () => {
    await graph();
    const [older, newer] = [graph(), graph()];
    const results = await Promise.all([older, newer]);
    assert.ok(results[1].ok, JSON.stringify(results[1]).slice(0, 800));
    assert.ok(used() <= 1, 'a standby is single-use');
  });

  test('the idle timeout retires an unused standby', async () => {
    core.STANDBY_IDLE_MS = 300;
    await graph();
    const pid = standbyPid(worker);
    assert.ok(alive(pid));
    assert.ok(await gone(pid, 5000), 'retired after the idle timeout');
    assert.equal(standbyPid(worker), undefined);
    signature(await graph());
    assert.equal(used(), 0);
  });

  test('legacy inline launches never start a standby', async () => {
    const run = () => worker.run({ file, cls: 'Model', input: '2,8', timeoutMs: 30000 });
    signature(await run());
    signature(await run());
    assert.equal(used(), 0);
    assert.equal(standbyPid(worker), undefined);
  });

  test('a backend staged without standby.py keeps the ordinary path and starts nothing', async () => {
    // A real copy (the backend checks its own paths) without the standby, as an
    // older or partial staging would be.
    const mirror = path.join(root, 'backend');
    await fs.cp(backend, mirror, { recursive: true, filter: source => {
      const relative = path.relative(backend, source);
      return relative !== path.join('services', 'ingest', 'standby.py') && !relative.startsWith('tests')
        && !relative.split(path.sep).includes('__pycache__');
    } });
    const partial = new TraceWorker(mirror, { appendLine: (line: string) => logs.push(line) } as unknown as vscode.OutputChannel,
      { interpreter: async () => interpreter });
    // Count spawns rather than look for a live process: a Python started on a missing
    // script exits before any assertion could see it.
    const childProcess = require('node:child_process');
    const original = childProcess.spawn;
    const standbySpawns: string[][] = [];
    childProcess.spawn = (command: string, args: string[], ...rest: unknown[]) => {
      if (args?.some(arg => arg.endsWith('standby.py'))) standbySpawns.push(args);
      return original.call(childProcess, command, args, ...rest);
    };
    try {
      signature(await partial.run(request()));
      signature(await partial.run(request()));
      assert.deepEqual(standbySpawns, [], 'no standby is spawned when its script is missing');
      assert.equal(used(), 0);
      // The same counter does see the standby on the full backend, so it is not vacuous.
      await graph();
      assert.equal(standbySpawns.length, 1);
    } finally { childProcess.spawn = original; partial.dispose(); }
  });

  test('worker stderr from a standby run is still reported', async () => {
    await graph();
    await fs.writeFile(file, model(1, '        import sys; sys.stderr.write("tv-stderr-marker\\n")\n'));
    signature(await graph());
    assert.equal(used(), 1);
    assert.ok(logs.some(line => line.includes('tv-stderr-marker')));
  });
});
