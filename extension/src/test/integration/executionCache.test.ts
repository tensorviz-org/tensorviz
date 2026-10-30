import * as assert from 'node:assert/strict';
import * as childProcess from 'node:child_process';
import { existsSync } from 'node:fs';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { TraceWorker } from '../../worker';
import type { ExecutionCacheSettings } from '@tensorviz/graph-host/executionCache';
import { executionSpecSchema } from '@tensorviz/graph-host/executionSpec';
import { disposeSnapshotHelpers } from '@tensorviz/graph-host/hostSnapshot';

/** Execution cache, shadow mode, against the real worker: the worker always runs, an
 *  identical run is compared with the stored one, and an edit is never a hit. */
const repo = path.resolve(__dirname, '../../../..');
const backend = path.join(repo, 'backend');
const python = process.env.TENSORVIZ_TEST_PYTHON || path.join(repo, 'venv/bin/python');

const MODEL = `import torch
from torch import nn
from helpers import WIDTH
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, WIDTH)
        self.b = nn.Linear(WIDTH, 4)
    def forward(self, x):
        return self.b(torch.relu(self.a(x)))
`;

suiteSetup(() => {
  assert.ok(existsSync(python), `these suites spawn a real interpreter: set TENSORVIZ_TEST_PYTHON (looked at ${python})`);
});

suite('execution cache (shadow) with real Python', function () {
  this.timeout(120_000);
  let root: string, file: string, logs: string[], worker: TraceWorker, settings: ExecutionCacheSettings;
  const execution = executionSpecSchema.parse({ inputs: { args: [{ shape: [2, 8], dtype: 'float32', fill: 'randn' }] } });
  const run = async () => {
    const outcome = await worker.run({ file, cls: 'Model', workspace: root, execution, reuse: 'allow', timeoutMs: 60_000 });
    assert.ok(outcome.ok, JSON.stringify(outcome).slice(0, 1500));
    await worker.executionCacheIdle();
    return outcome;
  };
  const cacheLines = () => logs.filter(line => line.startsWith('[cache]'));

  setup(async () => {
    root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tv-exec-cache-')));
    file = path.join(root, 'model.py');
    await fs.writeFile(file, MODEL);
    await fs.writeFile(path.join(root, 'helpers.py'), 'WIDTH = 16\n');
    logs = [];
    settings = { mode: 'shadow', maxSizeMB: 96 };
    worker = new TraceWorker(backend, { appendLine: (line: string) => logs.push(line) } as unknown as vscode.OutputChannel,
      { interpreter: async () => python, executionCache: () => settings });
  });
  teardown(async () => {
    worker.dispose();
    disposeSnapshotHelpers();
    await fs.rm(root, { recursive: true, force: true });
  });

  test('first run stores, a second identical run reports identical', async () => {
    const first = await run();
    const second = await run();
    assert.deepEqual((first.envelope.result as any).diagnostics.reuse, { storable: true, reasons: [], untrackedReads: 0 },
      'the outcome keeps the verdict only');
    const [stored] = [...(worker as any).resultStore.entries.values()] as Array<{ text: string }>;
    assert.ok(JSON.parse(stored.text).result.diagnostics.reuse.environment.length > 0,
      'the stored entry lists the installed files the run read, for the recheck before serving');
    assert.match(cacheLines()[0], /^\[cache\] shadow miss: stored [a-f0-9]{12}$/);
    assert.equal(cacheLines()[1], '[cache] shadow hit: identical');
    // Shadow never serves: the second envelope is the fresh run's own capture.
    assert.notEqual((second.envelope.result as any).diagnostics.semantic_capture.id,
      (first.envelope.result as any).diagnostics.semantic_capture.id);
    const stats = worker.executionCacheStats();
    assert.equal(stats.identical, 1); assert.equal(stats.different, 0); assert.equal(stats.entries, 1);
  });

  test('an edit in another workspace file makes a different key; A -> B -> A hits identical', async () => {
    await run();
    await fs.writeFile(path.join(root, 'helpers.py'), 'WIDTH = 32\n');
    await run();
    await fs.writeFile(path.join(root, 'helpers.py'), 'WIDTH = 16\n');
    await run();
    const [a, b, again] = cacheLines();
    assert.match(a, /^\[cache\] shadow miss: stored /);
    assert.match(b, /^\[cache\] shadow miss: stored /);
    assert.notEqual(a, b);
    assert.equal(again, '[cache] shadow hit: identical');
  });

  test('a module-level constant change in the model file is a different key', async () => {
    await run();
    await fs.writeFile(file, MODEL.replace('nn.Linear(WIDTH, 4)', 'nn.Linear(WIDTH, 5)'));
    await run();
    assert.equal(cacheLines().filter(line => line.includes('shadow hit')).length, 0);
  });

  test('a model that reads a data file is never stored', async () => {
    await fs.writeFile(path.join(root, 'scale.txt'), '2.0\n');
    await fs.writeFile(file, MODEL.replace('        super().__init__()\n',
      '        super().__init__()\n        self.scale = float(open(__file__.replace("model.py", "scale.txt")).read())\n'));
    const first = await run();
    await run();
    assert.deepEqual((first.envelope.result as any).diagnostics.reuse,
      { storable: false, reasons: ['untracked_workspace_read'], untrackedReads: 1 });
    assert.deepEqual(cacheLines(), ['[cache] shadow miss: not storable (worker:untracked_workspace_read)',
      '[cache] shadow miss: not storable (worker:untracked_workspace_read)']);
  });

  test('mode off does no cache work', async () => {
    settings = { mode: 'off' };
    await run();
    await run();
    assert.deepEqual(cacheLines(), []);
    assert.equal(worker.executionCacheStats().lookups, 0);
  });
});

/** The real document a Canvas receives for an envelope (services.ide.projection.project). */
function projectDocument(envelope: unknown, source: string): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const child = childProcess.execFile(python, ['-B', '-c', 'import json,sys\nfrom services.ide.projection import project\n'
      + 'p=json.load(sys.stdin)\nr=project(dict(nodes=[],edges=[]),p["envelope"],cls="Model",rel_file="model.py",source=p["source"])\n'
      + 'print(r.document.model_dump_json(by_alias=True))'],
    { cwd: backend, timeout: 60_000, maxBuffer: 32 * 1024 * 1024, env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' } },
    (error, stdout) => { try { if (error) reject(error); else resolve(JSON.parse(stdout)); } catch (e) { reject(e); } });
    child.stdin!.end(JSON.stringify({ envelope, source }));
  });
}

suite('execution cache (on) with real Python', function () {
  this.timeout(180_000);
  let root: string, file: string, logs: string[], worker: TraceWorker, settings: ExecutionCacheSettings;
  let launched: string[][] = [];
  // The module object itself (an `import *` namespace is read-only), so the worker's
  // call-time `child_process.spawn` lookup sees the counting wrapper.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const processes = require('node:child_process') as { spawn: typeof childProcess.spawn };
  const realSpawn = processes.spawn;
  const execution = executionSpecSchema.parse({ inputs: { args: [{ shape: [2, 8], dtype: 'float32', fill: 'randn' }] } });
  const run = async (reuse: 'allow' | 'bypass' = 'allow') => {
    const outcome = await worker.run({ file, cls: 'Model', workspace: root, execution, reuse, timeoutMs: 60_000 });
    assert.ok(outcome.ok, JSON.stringify(outcome).slice(0, 1500));
    await worker.executionCacheIdle();
    return outcome;
  };
  /** Python processes started for the model itself (a worker, or a standby that would run it). */
  const workerLaunches = () => launched.filter(args => args.some(arg => /services[\\/]ingest[\\/](bootstrap|standby)\.py$/.test(arg))).length;

  setup(async () => {
    root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tv-exec-cache-on-')));
    file = path.join(root, 'model.py');
    await fs.writeFile(file, MODEL);
    await fs.writeFile(path.join(root, 'helpers.py'), 'WIDTH = 16\n');
    logs = []; launched = [];
    settings = { mode: 'on', maxSizeMB: 96 };
    // Count every process the worker module starts (it calls child_process.spawn at run time).
    processes.spawn = ((command: string, args: readonly string[], options: childProcess.SpawnOptions) => {
      launched.push([command, ...(args ?? [])]);
      return realSpawn(command, args, options);
    }) as typeof childProcess.spawn;
    worker = new TraceWorker(backend, { appendLine: (line: string) => logs.push(line) } as unknown as vscode.OutputChannel,
      { interpreter: async () => python, executionCache: () => settings });
  });
  teardown(async () => {
    processes.spawn = realSpawn;
    worker.dispose();
    disposeSnapshotHelpers();
    await fs.rm(root, { recursive: true, force: true });
  });

  test('a second identical run starts no worker process and projects the identical graph', async () => {
    const first = await run();
    const before = workerLaunches();
    assert.ok(before >= 1, 'the first run executed');
    const second = await run();
    assert.equal(workerLaunches(), before, 'no Python worker (or standby) was started for the served run');
    assert.ok(second.ok && second.reused, 'served');
    assert.deepEqual(second.envelope, first.envelope);
    const source = await fs.readFile(file, 'utf8');
    assert.deepEqual(await projectDocument(second.envelope, source), await projectDocument(first.envelope, source),
      'the Canvas document of the served run equals the fresh run\'s');
    assert.ok(logs.some(line => /^\[cache\] served [0-9a-f]{12}: .*\(model not run again\)$/.test(line)));
    assert.equal(worker.executionCacheStats().served, 1);
  });

  test('an edit runs the worker; A -> B -> A is served', async () => {
    const a = await run();
    await fs.writeFile(path.join(root, 'helpers.py'), 'WIDTH = 32\n');
    const afterA = workerLaunches();
    const b = await run();
    assert.ok(workerLaunches() > afterA, 'the edited run executed');
    assert.ok(b.ok && !b.reused);
    await fs.writeFile(path.join(root, 'helpers.py'), 'WIDTH = 16\n');
    const afterB = workerLaunches();
    const again = await run();
    assert.equal(workerLaunches(), afterB, 'reverting to A is served');
    assert.ok(again.ok && again.reused);
    assert.equal((again.envelope.result as any).diagnostics.semantic_capture.id, (a.envelope.result as any).diagnostics.semantic_capture.id);
  });

  test('a served result carries the open editor\'s current document version', async () => {
    // Undo leaves the file identical (same key) but bumps the editor's version; navigation
    // refuses a node whose recorded documentVersion is not the open document's.
    let version = 1;
    const original = Object.getOwnPropertyDescriptor(vscode.workspace, 'textDocuments')!;
    Object.defineProperty(vscode.workspace, 'textDocuments', { configurable: true,
      get: () => [{ uri: vscode.Uri.file(file), version, isDirty: false }] });
    try {
      const versions = (outcome: Awaited<ReturnType<typeof run>>) => {
        const d = (outcome.envelope.result as any).diagnostics, pick = (files: { path: string; documentVersion: number | null }[]) =>
          files.find(item => item.path === 'model.py')?.documentVersion;
        return [pick(d.execution.resolvedPlan.source.files), pick(d.semantic_capture.source.files),
          outcome.ok && outcome.reused ? pick(outcome.reused.plan.source.files) : undefined];
      };
      const first = await run();
      assert.deepEqual(versions(first), [1, 1, undefined]);
      const before = workerLaunches();
      version = 3;
      const served = await run();
      assert.equal(workerLaunches(), before, 'still served: the content is unchanged');
      assert.ok(served.ok && served.reused);
      assert.deepEqual(versions(served), [3, 3, 3]);
    } finally { Object.defineProperty(vscode.workspace, 'textDocuments', original); }
  });

  test('Run fresh executes even with an identical stored result', async () => {
    await run();
    const before = workerLaunches();
    const fresh = await run('bypass');
    assert.ok(workerLaunches() > before);
    assert.ok(fresh.ok && !fresh.reused);
  });

  test('a model that reads a data file is never served', async () => {
    await fs.writeFile(path.join(root, 'scale.txt'), '2.0\n');
    await fs.writeFile(file, MODEL.replace('        super().__init__()\n',
      '        super().__init__()\n        self.scale = float(open(__file__.replace("model.py", "scale.txt")).read())\n'));
    await run();
    const before = workerLaunches();
    const second = await run();
    assert.ok(workerLaunches() > before);
    assert.ok(second.ok && !second.reused);
  });
});
