import * as assert from 'node:assert/strict';
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { TraceWorkerCore, type TraceRequest } from '@tensorviz/graph-host/traceWorkerCore';
import { disposeSnapshotHelpers } from '@tensorviz/graph-host/hostSnapshot';
import type { ExecutionCacheSettings } from '@tensorviz/graph-host/executionCache';

/** A fake interpreter (a node script) that answers the interpreter probe, the snapshot
 *  helper and the worker the way Python would, driven by `<backend>/state.json`. */
const FAKE = String.raw`
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const argv = process.argv.slice(2);
const state = () => JSON.parse(fs.readFileSync(path.join(process.cwd(), 'state.json'), 'utf8'));
if (argv[0] === '-c') { process.stdout.write('2.7.1 2.0.0\n'); process.exit(0); }
if (argv.some(a => a.endsWith('host_snapshot.py'))) {
  const answer = () => { const s = state(); return JSON.stringify({ applicabilityDigest: s.digest, guards: {},
    plan: { version: 1, id: 'plan-' + crypto.randomUUID(), interpreter: { executable: process.argv[1] },
      execution: { device: 'cpu', seed: s.seed === undefined ? 0 : s.seed, deterministic: true, inputs: {} }, job: null, timeoutMs: 20000 } }); };
  if (argv.includes('--serve')) {
    let buffer = '';
    process.stdin.on('data', chunk => { buffer += chunk; let i; while ((i = buffer.indexOf('\n')) >= 0) { buffer = buffer.slice(i + 1); process.stdout.write(answer() + '\n'); } });
  } else { process.stdin.resume(); process.stdin.on('end', () => process.stdout.write(answer())); }
} else if (argv.some(a => a.endsWith('bootstrap.py'))) {
  const s = state(), plan = JSON.parse(fs.readFileSync(process.env.TENSORVIZ_EXECUTION_PLAN, 'utf8'));
  fs.appendFileSync(path.join(process.cwd(), 'runs.log'), argv.includes('--reuse-audit') ? 'run audited\n' : 'run\n');
  const planId = plan.plan.id;
  process.stdout.write(JSON.stringify({ protocol_version: 1, spans: {}, error: null,
    worker: { planId, planDigest: plan.applicabilityDigest, elapsed_ms: Math.random() * 100 },
    result: { graph: { value: s.value }, diagnostics: {
      timings_ms: { build_ms: Math.random(), trace_ms: Math.random() },
      worker: { planId, elapsed_ms: Math.random() },
      execution: { planDigest: plan.applicabilityDigest, planCurrent: true, resolvedPlan: { id: planId, source: { id: 'source:' + planId, generation: 1 } } },
      semantic_capture: { id: 'capture:' + crypto.randomUUID(), attemptId: 'a-' + crypto.randomUUID(), source: { id: 'source:' + planId, generation: 1 }, sourceBindings: [] },
      semantic_capture_error: null, reuse: s.reuse || { storable: true, reasons: [], untrackedReads: 0, environment: [] } } } }) + '\n');
  process.exit(s.exit || 0);
} else process.exit(2);
`;

/** One run, then wait for its shadow comparison (scheduled after the result is returned). */
async function go(w: TraceWorkerCore, request: TraceRequest) {
  const outcome = await w.run(request);
  await w.executionCacheIdle();
  return outcome;
}

suite('TraceWorkerCore execution cache (shadow hook)', function () {
  this.timeout(30_000);
  let root: string, backend: string, workspace: string, python: string, lines: string[];
  let settings: ExecutionCacheSettings | undefined;
  const state = (value: Record<string, unknown>) => writeFileSync(path.join(backend, 'state.json'), JSON.stringify({ digest: 'a'.repeat(64), value: 1, ...value }));
  const worker = (withOption = true) => new TraceWorkerCore(backend, { appendLine: line => lines.push(line) }, {
    interpreter: async () => python, trusted: () => true, documents: () => ({}),
    ...(withOption ? { executionCache: () => settings } : {}),
  });
  const request = (extra: Partial<TraceRequest> = {}): TraceRequest => ({ file: path.join(workspace, 'm.py'), cls: 'M', workspace,
    execution: { inputs: { args: [] } } as never, reuse: 'allow', ...extra });
  const cacheLines = () => lines.filter(line => line.startsWith('[cache]'));

  setup(() => {
    root = mkdtempSync(path.join(os.tmpdir(), 'tv-cache-hook-'));
    backend = path.join(root, 'backend'); workspace = path.join(root, 'ws');
    mkdirSync(path.join(backend, 'services/validation'), { recursive: true });
    mkdirSync(path.join(backend, 'services/ingest'), { recursive: true });
    mkdirSync(workspace);
    writeFileSync(path.join(backend, 'services/validation/host_snapshot.py'), '# fake\n');
    writeFileSync(path.join(backend, 'services/ingest/bootstrap.py'), '# fake\n');
    writeFileSync(path.join(workspace, 'm.py'), 'class M: pass\n');
    python = path.join(root, 'python');
    writeFileSync(python, `#!${process.execPath}\n${FAKE}`);
    chmodSync(python, 0o755);
    state({});
    lines = []; settings = { mode: 'shadow', maxSizeMB: 96 };
  });
  teardown(() => { disposeSnapshotHelpers(); rmSync(root, { recursive: true, force: true }); });

  test('off (no host setting, as in the plugin host): no cache work and no store', async () => {
    const w = worker(false);
    try {
      for (let i = 0; i < 2; i++) assert.equal((await go(w, request())).ok, true);
      assert.deepEqual(cacheLines(), []);
      // Launched exactly as before: the worker is not asked to audit.
      assert.equal(readFileSync(path.join(backend, 'runs.log'), 'utf8'), 'run\nrun\n');
      assert.deepEqual(w.executionCacheStats(), { lookups: 0, hits: 0, identical: 0, different: 0, misses: 0, stored: 0,
        notStorable: 0, ineligible: 0, hitNotStorable: 0, served: 0, serveRejected: 0, entries: 0, bytes: 0 });
    } finally { w.dispose(); }
  });

  test('off (setting): identical to no setting', async () => {
    settings = { mode: 'off' };
    const w = worker();
    try {
      for (let i = 0; i < 2; i++) assert.equal((await go(w, request())).ok, true);
      assert.deepEqual(cacheLines(), []);
      assert.equal(w.executionCacheStats().lookups, 0);
    } finally { w.dispose(); }
  });

  test('shadow work happens after the caller has its result', async () => {
    const w = worker();
    try {
      const outcome = await w.run(request());
      assert.equal(outcome.ok, true);
      assert.deepEqual(cacheLines(), []);   // not yet: scheduled for the next event-loop turn
      await w.executionCacheIdle();
      assert.match(cacheLines()[0], /^\[cache\] shadow miss: stored /);
    } finally { w.dispose(); }
  });

  test('a caller mutating its envelope cannot change the stored or compared result', async () => {
    const w = worker();
    try {
      const first = await w.run(request());
      (first as { envelope: { result: { graph: { value: number } } } }).envelope.result.graph.value = 99;
      await w.executionCacheIdle();
      const second = await w.run(request());
      (second as { envelope: { result: { graph: { value: number } } } }).envelope.result.graph.value = 42;
      await w.executionCacheIdle();
      assert.equal(cacheLines()[1], '[cache] shadow hit: identical');
    } finally { w.dispose(); }
  });

  test('shadow: first run stores, identical second run is a hit that still runs the worker', async () => {
    const w = worker();
    try {
      const first = await go(w, request());
      const second = await go(w, request());
      assert.ok(first.ok && second.ok);
      assert.equal(readFileSync(path.join(backend, 'runs.log'), 'utf8'), 'run audited\nrun audited\n');
      // The worker ran both times and the caller got the FRESH envelope, not the stored one.
      assert.notEqual((second.envelope.result as any).diagnostics.semantic_capture.id,
        (first.envelope.result as any).diagnostics.semantic_capture.id);
      assert.equal(cacheLines()[0].startsWith('[cache] shadow miss: stored '), true);
      assert.equal(cacheLines()[1], '[cache] shadow hit: identical');
      const stats = w.executionCacheStats();
      assert.equal(stats.hits, 1); assert.equal(stats.identical, 1); assert.equal(stats.different, 0);
      assert.equal(stats.entries, 1);
    } finally { w.dispose(); }
  });

  test('shadow: a different result for the same key is reported with its first path', async () => {
    const w = worker();
    try {
      await go(w, request());
      state({ value: 2 });               // same applicability digest, different output
      await go(w, request());
      assert.equal(cacheLines()[1], '[cache] shadow hit: DIFFERENT (.result.graph.value)');
      assert.equal(w.executionCacheStats().different, 1);
    } finally { w.dispose(); }
  });

  test('shadow: an edit changes the key (no hit); A -> B -> A hits', async () => {
    const w = worker();
    try {
      await go(w, request());
      state({ digest: 'b'.repeat(64) });
      await go(w, request());
      state({});
      await go(w, request());
      const found = cacheLines();
      assert.match(found[0], /^\[cache\] shadow miss: stored /);
      assert.match(found[1], /^\[cache\] shadow miss: stored /);
      assert.notEqual(found[0], found[1]);
      assert.equal(found[2], '[cache] shadow hit: identical');
    } finally { w.dispose(); }
  });

  test('shadow: environment variables are part of the key', async () => {
    const w = worker();
    const saved = process.env.TV_CACHE_TEST_FLAG;
    try {
      await go(w, request());
      process.env.TV_CACHE_TEST_FLAG = '1';
      await go(w, request());
      assert.match(cacheLines()[1], /^\[cache\] shadow miss: stored /);
    } finally { if (saved === undefined) delete process.env.TV_CACHE_TEST_FLAG; else process.env.TV_CACHE_TEST_FLAG = saved; w.dispose(); }
  });

  test('shadow: non-storable runs are never stored', async () => {
    const w = worker();
    try {
      state({ reuse: { storable: false, reasons: ['untracked_workspace_read'], untrackedReads: 1 } });
      await go(w, request());
      await go(w, request());
      assert.deepEqual(cacheLines(), ['[cache] shadow miss: not storable (worker:untracked_workspace_read)',
        '[cache] shadow miss: not storable (worker:untracked_workspace_read)']);
      state({ exit: 4 });
      await go(w, request());
      assert.equal(cacheLines()[2], '[cache] shadow miss: not storable (failed)');
      assert.equal(w.executionCacheStats().entries, 0);
    } finally { w.dispose(); }
  });

  test('shadow: a stored key whose fresh run is not storable is dropped', async () => {
    const w = worker();
    try {
      await go(w, request());
      state({ reuse: { storable: false, reasons: ['write'], untrackedReads: 0 } });
      await go(w, request());
      assert.equal(cacheLines()[1], '[cache] shadow hit: fresh run not storable (worker:write)');
      assert.equal(w.executionCacheStats().entries, 0);
    } finally { w.dispose(); }
  });

  test('shadow: requests that did not opt in, or have no seed, are not eligible', async () => {
    const w = worker();
    try {
      await go(w, request({ reuse: undefined }));
      state({ seed: null });
      await go(w, request());
      assert.deepEqual(cacheLines(), ['[cache] not eligible (not_requested)', '[cache] not eligible (no_seed)']);
      assert.equal(w.executionCacheStats().entries, 0);
    } finally { w.dispose(); }
  });

  test('the mode is read for the run\'s own file, so a folder-level value of a multi-root workspace applies', async () => {
    const asked: unknown[] = [];
    const w = new TraceWorkerCore(backend, { appendLine: line => lines.push(line) }, {
      interpreter: async () => python, trusted: () => true, documents: () => ({}),
      executionCache: (...resource: unknown[]) => { asked.push(...resource); return { mode: 'off' }; },
    });
    try {
      await go(w, request());
      assert.deepEqual(asked, [path.join(workspace, 'm.py')]);
      assert.deepEqual(cacheLines(), []);
    } finally { w.dispose(); }
  });

  test('one store serves every folder: a folder set to off skips only its own runs and keeps other folders\' entries', async () => {
    const other = path.join(root, 'other');
    mkdirSync(other);
    writeFileSync(path.join(other, 'm.py'), 'class M: pass\n');
    // The window level (no resource) is shadow; the other folder sets off, then a tiny size.
    let folder: ExecutionCacheSettings = { mode: 'off' };
    const w = new TraceWorkerCore(backend, { appendLine: line => lines.push(line) }, {
      interpreter: async () => python, trusted: () => true, documents: () => ({}),
      executionCache: resource => resource?.startsWith(other + path.sep) ? folder : settings,
    });
    const elsewhere = () => request({ file: path.join(other, 'm.py'), workspace: other });
    try {
      await go(w, request());
      await go(w, elsewhere());
      assert.equal(w.executionCacheStats().entries, 1, 'the off folder\'s run left the shared store alone');
      // Only the window-level size caps the shared store.
      folder = { mode: 'shadow', maxSizeMB: 1e-6 };
      await go(w, elsewhere());
      assert.equal(w.executionCacheStats().entries, 2);
      await go(w, request());
      assert.equal(cacheLines().at(-1), '[cache] shadow hit: identical');
      // The off folder's run launched as with the cache off: not audited.
      assert.equal(readFileSync(path.join(backend, 'runs.log'), 'utf8'), 'run audited\nrun\nrun audited\nrun audited\n');
    } finally { w.dispose(); }
  });

  test('the window-level off still drops the shared store, whatever the run\'s folder says', async () => {
    let windowLevel: ExecutionCacheSettings = { mode: 'shadow' };
    const w = new TraceWorkerCore(backend, { appendLine: line => lines.push(line) }, {
      interpreter: async () => python, trusted: () => true, documents: () => ({}),
      executionCache: resource => resource === undefined ? windowLevel : settings,
    });
    try {
      await go(w, request());
      assert.equal(w.executionCacheStats().entries, 1);
      windowLevel = { mode: 'off' };
      await go(w, request());
      assert.equal(w.executionCacheStats().entries, 0);
      assert.equal(cacheLines().length, 1, 'no lookup while the window is off');
    } finally { w.dispose(); }
  });

  test('switching the setting to off drops the store without a reload', async () => {
    const w = worker();
    try {
      await go(w, request());
      assert.equal(w.executionCacheStats().entries, 1);
      settings = { mode: 'off' };
      await go(w, request());
      assert.equal(w.executionCacheStats().entries, 0);
      settings = { mode: 'shadow' };
      await go(w, request());
      assert.match(cacheLines()[1], /^\[cache\] shadow miss: stored /);
    } finally { w.dispose(); }
  });
});
