import * as assert from 'node:assert/strict';
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { TraceWorkerCore, WORKER_PROTOCOL_VERSION, type TraceOutcome, type TraceRequest } from '@tensorviz/graph-host/traceWorkerCore';
import { disposeSnapshotHelpers } from '@tensorviz/graph-host/hostSnapshot';
import { ExecutionResultStore, MAX_ENTRY_BYTES, servableEntry, type ExecutionCacheSettings } from '@tensorviz/graph-host/executionCache';
import type { ExecutionSnapshot } from '@tensorviz/graph-host/executionPlanCore';

const FIXTURE = path.join(__dirname, '../../../tests/fixtures/contracts/semantic-capture.json');

/** A fake interpreter (a node script) answering the probe, the snapshot helper and the worker
 *  like Python would, driven by `<backend>/state.json`. Unlike the shadow-hook fake, its
 *  worker prints a real, valid canonical capture for the resolved plan, so a stored entry is
 *  servable. Every worker launch appends to `runs.log`; every snapshot to `snapshots.log`. */
const FAKE = String.raw`
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const argv = process.argv.slice(2);
const state = () => JSON.parse(fs.readFileSync(path.join(process.cwd(), 'state.json'), 'utf8'));
const fixture = () => JSON.parse(fs.readFileSync(state().fixture, 'utf8'));
if (argv[0] === '-c') { process.stdout.write('2.7.1 2.0.0\n'); process.exit(0); }
if (argv.some(a => a.endsWith('host_snapshot.py'))) {
  const answer = () => { const s = state(), c = fixture();
    const id = 'plan-' + crypto.randomUUID();
    return JSON.stringify({ applicabilityDigest: s.digest, guards: {},
      plan: { version: 1, id, interpreter: { executable: process.argv[1] }, source: { ...c.source, id: 'source:' + id,
        files: c.source.files.map(file => s.documentVersion === undefined ? file : { ...file, documentVersion: s.documentVersion }) },
        captureLimits: c.limits, execution: { device: 'cpu', seed: s.seed === undefined ? 0 : s.seed, deterministic: true, inputs: {} },
        job: null, timeoutMs: 20000 } }); };
  // Logged when the request ARRIVES, so a test can act while the answer is still delayed.
  const reply = write => { fs.appendFileSync(path.join(process.cwd(), 'snapshots.log'), 'snapshot\n');
    const delay = state().snapshotDelayMs || 0; delay ? setTimeout(write, delay) : write(); };
  if (argv.includes('--serve')) {
    let buffer = '';
    process.stdin.on('data', chunk => { buffer += chunk; let i; while ((i = buffer.indexOf('\n')) >= 0) { buffer = buffer.slice(i + 1); reply(() => process.stdout.write(answer() + '\n')); } });
  } else { process.stdin.resume(); process.stdin.on('end', () => reply(() => process.stdout.write(answer()))); }
} else if (argv.some(a => a.endsWith('bootstrap.py'))) {
  const s = state(), plan = JSON.parse(fs.readFileSync(process.env.TENSORVIZ_EXECUTION_PLAN, 'utf8'));
  fs.appendFileSync(path.join(process.cwd(), 'runs.log'), 'run\n');
  const planId = plan.plan.id, capture = fixture();
  Object.assign(capture, { id: 'capture:' + crypto.randomUUID(), attemptId: planId, planDigest: plan.applicabilityDigest,
    source: plan.plan.source });
  if (s.partial) { capture.status = 'partial'; capture.coverage.omissions = ['byte_limit']; }
  process.stdout.write(JSON.stringify({ protocol_version: 1, spans: {}, error: null,
    worker: { planId, planDigest: plan.applicabilityDigest, elapsed_ms: Math.random() * 100 },
    result: { graph: { value: s.value === undefined ? 1 : s.value }, diagnostics: {
      timings_ms: { build_ms: Math.random(), trace_ms: Math.random() },
      worker: { planId, elapsed_ms: Math.random() },
      execution: { planDigest: plan.applicabilityDigest, planCurrent: true, resolvedPlan: plan.plan },
      semantic_capture: capture, semantic_capture_error: null,
      reuse: s.reuse || { storable: true, reasons: [], untrackedReads: 0 } } } }) + '\n');
  process.exit(s.exit || 0);
} else process.exit(2);
`;

type Ok = Extract<TraceOutcome, { ok: true }>;
const captureId = (outcome: TraceOutcome) => (outcome.envelope?.result as any)?.diagnostics?.semantic_capture?.id as string;

suite('TraceWorkerCore execution cache (on: serving)', function () {
  this.timeout(30_000);
  let root: string, backend: string, workspace: string, python: string, lines: string[];
  let settings: ExecutionCacheSettings | undefined;
  let current: Record<string, unknown>;
  const state = (value: Record<string, unknown>) => {
    current = { digest: 'a'.repeat(64), fixture: FIXTURE, ...value };
    writeFileSync(path.join(backend, 'state.json'), JSON.stringify(current));
  };
  const worker = () => new TraceWorkerCore(backend, { appendLine: line => lines.push(line) }, {
    interpreter: async () => python, trusted: () => true, documents: () => ({}), executionCache: () => settings,
  });
  const request = (extra: Partial<TraceRequest> = {}): TraceRequest => ({ file: path.join(workspace, 'm.py'), cls: 'M', workspace,
    execution: { inputs: { args: [] } } as never, reuse: 'allow', attemptId: `attempt-${Math.random()}`, ...extra });
  const launches = () => existsSync(path.join(backend, 'runs.log')) ? readFileSync(path.join(backend, 'runs.log'), 'utf8').split('\n').filter(Boolean).length : 0;
  const snapshots = () => existsSync(path.join(backend, 'snapshots.log')) ? readFileSync(path.join(backend, 'snapshots.log'), 'utf8').split('\n').filter(Boolean).length : 0;
  async function go(w: TraceWorkerCore, extra: Partial<TraceRequest> = {}) {
    const outcome = await w.run(request(extra));
    await w.executionCacheIdle();
    return outcome;
  }
  const store = (w: TraceWorkerCore) => (w as unknown as { resultStore: ExecutionResultStore }).resultStore;

  setup(() => {
    root = mkdtempSync(path.join(os.tmpdir(), 'tv-cache-serve-'));
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
    lines = []; settings = { mode: 'on', maxSizeMB: 96 };
  });
  teardown(() => { disposeSnapshotHelpers(); rmSync(root, { recursive: true, force: true }); });

  test('on: an identical second run is served without launching the worker, with the original identity', async () => {
    const w = worker();
    try {
      const first = await go(w) as Ok;
      assert.equal(first.ok, true);
      assert.equal(first.reused, undefined);
      assert.equal(launches(), 1);
      const second = await go(w) as Ok;
      assert.equal(second.ok, true);
      assert.equal(launches(), 1, 'no worker process for a served run');
      assert.ok(second.reused);
      assert.equal(second.executionCurrent, true);
      assert.equal(captureId(second), captureId(first), 'the reused capture keeps its id');
      assert.equal(second.reused.captureId, captureId(first));
      assert.equal(second.reused.attemptId, (first.envelope.result as any).diagnostics.semantic_capture.attemptId);
      assert.equal(second.reused.plan.id, second.reused.attemptId);
      assert.deepEqual(second.envelope, first.envelope, 'the stored envelope is returned as the outcome');
      assert.ok(second.recheckExecution && await second.recheckExecution(new AbortController().signal),
        'the post-projection freshness recheck still re-resolves the plan');
      const stats = w.executionCacheStats();
      assert.equal(stats.served, 1); assert.equal(stats.hits, 1); assert.equal(stats.serveRejected, 0);
      assert.ok(lines.some(line => /^\[cache\] served [0-9a-f]{12}: reused identical result of attempt .* \(no Python ran\)$/.test(line)));
    } finally { w.dispose(); }
  });

  test('on: a served outcome is a fresh copy; mutating it cannot change what is served next', async () => {
    const w = worker();
    try {
      const first = await go(w);
      const served = await go(w) as Ok;
      (served.envelope.result as any).graph.value = 99;
      const again = await go(w) as Ok;
      assert.equal((again.envelope.result as any).graph.value, 1);
      assert.equal(captureId(again), captureId(first));
      assert.equal(launches(), 1);
    } finally { w.dispose(); }
  });

  test('on: an edit misses and runs; A -> B -> A serves A', async () => {
    const w = worker();
    try {
      const a = await go(w);
      state({ digest: 'b'.repeat(64) });
      const b = await go(w) as Ok;
      assert.equal(b.reused, undefined);
      assert.equal(launches(), 2);
      state({});
      const a2 = await go(w) as Ok;
      assert.equal(launches(), 2);
      assert.ok(a2.reused);
      assert.equal(captureId(a2), captureId(a));
      state({ digest: 'b'.repeat(64) });
      const b2 = await go(w) as Ok;
      assert.equal(captureId(b2), captureId(b));
      assert.equal(launches(), 2);
    } finally { w.dispose(); }
  });

  test('on: a served result carries the requesting run\'s document versions, not the stored ones', async () => {
    // The key leaves documentVersion out (the content digest covers it): undo bumps the
    // editor version of an identical file. Navigation compares it to the open document.
    const w = worker();
    try {
      const versions = (outcome: Ok) => {
        const d = (outcome.envelope.result as any).diagnostics;
        return [d.execution.resolvedPlan.source.files, d.semantic_capture.source.files, outcome.reused?.plan.source.files]
          .map(files => files?.map((file: { documentVersion: number | null }) => file.documentVersion));
      };
      state({ documentVersion: 1 });
      const first = await go(w) as Ok;
      assert.deepEqual(versions(first), [[1], [1], undefined]);
      state({ documentVersion: 3 });
      const undone = await go(w) as Ok;
      assert.ok(undone.reused, 'still served: the key is unchanged');
      assert.equal(launches(), 1);
      assert.equal(captureId(undone), captureId(first));
      assert.deepEqual(versions(undone), [[3], [3], [3]]);
      state({ documentVersion: null });
      const closed = await go(w) as Ok;
      assert.ok(closed.reused);
      assert.deepEqual(versions(closed), [[null], [null], [null]]);
      assert.equal(launches(), 1);
      // The stored entry is not rewritten: the original run's versions come back when asked for.
      state({ documentVersion: 1 });
      assert.deepEqual(versions(await go(w) as Ok), [[1], [1], [1]]);
    } finally { w.dispose(); }
  });

  test('on: excluded requests are never served (not opted in, diagnostic job, data kind, fake tier, no seed)', async () => {
    const w = worker();
    try {
      await go(w);                                      // stores an entry for the plain request
      const excluded: Partial<TraceRequest>[] = [{ reuse: undefined }, { job: { kind: 'diagnostic' } as never },
        { kind: 'data', callable: 'loader' }, { tier: 'fake' }];
      for (const extra of excluded) {
        const before = launches();
        const outcome = await go(w, extra);
        assert.equal((outcome as Ok).reused, undefined, JSON.stringify(extra));
        // This tree refuses every diagnostic job before launch: the job schema only
        // names profiling and training-step capture, then rejects both. A job is
        // therefore never served. Data pipelines and the fake tier exit before the
        // fake interpreter prints a run; a request that merely omits reuse still runs.
        if (extra.job) {
          assert.equal(outcome.ok, false, 'a diagnostic job is refused before launch');
          assert.equal(launches(), before, 'a diagnostic job does not launch the worker');
        } else if (!extra.kind && !extra.tier) assert.equal(launches(), before + 1, `${JSON.stringify(extra)} launches the worker`);
      }
      state({ seed: null });
      const before = launches();
      await go(w);
      await go(w);
      assert.equal(launches(), before + 2, 'a run without a seed always executes');
      assert.equal(w.executionCacheStats().served, 0);
    } finally { w.dispose(); }
  });

  test('on: a run the worker reports not storable is never stored, so never served', async () => {
    const w = worker();
    try {
      state({ reuse: { storable: false, reasons: ['untracked_workspace_read'], untrackedReads: 1 } });
      await go(w); await go(w);
      assert.equal(launches(), 2);
      assert.equal(w.executionCacheStats().entries, 0);
    } finally { w.dispose(); }
  });

  test('on: a failed run is not stored and the next identical run executes', async () => {
    const w = worker();
    try {
      state({ exit: 4 });
      await go(w);
      state({});
      const next = await go(w) as Ok;
      assert.equal(next.reused, undefined);
      assert.equal(launches(), 2);
    } finally { w.dispose(); }
  });

  test('Run fresh (reuse: bypass) launches the worker and replaces the stored entry', async () => {
    const w = worker();
    try {
      const first = await go(w);
      const fresh = await go(w, { reuse: 'bypass' }) as Ok;
      assert.equal(launches(), 2, 'bypass always executes');
      assert.equal(fresh.reused, undefined);
      assert.notEqual(captureId(fresh), captureId(first));
      const served = await go(w) as Ok;
      assert.equal(launches(), 2);
      assert.equal(captureId(served), captureId(fresh), 'the fresh result replaced the earlier one');
      assert.ok(lines.includes('[cache] shadow hit: identical'), 'the bypassed run is still compared with the stored one');
    } finally { w.dispose(); }
  });

  type Entry = { text: string; meta?: string };
  const corruptions: Array<[string, (entry: Entry) => Entry]> = [
    ['unparseable text', (entry: { text: string; meta?: string }) => ({ ...entry, text: entry.text.slice(0, -10) })],
    ['corrupt metadata', (entry: { text: string; meta?: string }) => ({ ...entry, meta: '{"attemptId":' })],
    ['tampered capture', (entry: { text: string; meta?: string }) => {
      const envelope = JSON.parse(entry.text);
      envelope.result.diagnostics.semantic_capture.status = 'partial';
      return { ...entry, text: JSON.stringify(envelope) };
    }],
    ['worker-certified stale plan', (entry: { text: string; meta?: string }) => {
      const envelope = JSON.parse(entry.text);
      envelope.result.diagnostics.execution.planCurrent = false;
      return { ...entry, text: JSON.stringify(envelope) };
    }],
    ['not storable', (entry: { text: string; meta?: string }) => {
      const envelope = JSON.parse(entry.text);
      envelope.result.diagnostics.reuse.storable = false;
      return { ...entry, text: JSON.stringify(envelope) };
    }],
  ];
  for (const [name, corrupt] of corruptions) {
    test(`on: a stored entry with ${name} is dropped and the run proceeds fresh`, async () => {
      const w = worker();
      try {
        await go(w);
        const cache = store(w);
        const [key] = [...(cache as unknown as { entries: Map<string, unknown> }).entries.keys()];
        const damaged = corrupt(cache.entry(key)!);
        // Write the damaged entry directly: putText would re-serialise (repair) the metadata.
        (cache as unknown as { entries: Map<string, { text: string; meta?: string; bytes: number }> }).entries.set(key,
          { text: damaged.text, ...(damaged.meta !== undefined ? { meta: damaged.meta } : {}), bytes: damaged.text.length });
        const outcome = await go(w) as Ok;
        assert.equal(outcome.ok, true);
        assert.equal(outcome.reused, undefined);
        assert.equal(launches(), 2, 'ran fresh');
        assert.equal(w.executionCacheStats().serveRejected, 1);
        assert.ok(lines.some(line => line.startsWith('[cache] stored result not servable (')));
        const served = await go(w) as Ok;
        assert.ok(served.reused, 'the fresh run replaced the dropped entry');
        assert.equal(launches(), 2);
      } finally { w.dispose(); }
    });
  }

  test('on: an entry kept for comparison only (incomplete capture) is never served and is not dropped', async () => {
    const w = worker();
    try {
      await go(w);
      const cache = store(w);
      const entries = (cache as unknown as { entries: Map<string, { text: string; meta?: string; bytes: number }> }).entries;
      const [key] = [...entries.keys()];
      const { text } = entries.get(key)!;
      entries.set(key, { text, bytes: text.length });          // what a partial (byte-limit) capture stores
      const outcome = await go(w) as Ok;
      assert.equal(outcome.reused, undefined);
      assert.equal(launches(), 2, 'ran fresh');
      assert.ok(lines.includes('[cache] stored result is comparison-only (capture not complete); running fresh'));
      assert.ok(lines.includes('[cache] shadow hit: identical'), 'still compared');
      assert.equal(w.executionCacheStats().serveRejected, 0);
      assert.ok((await go(w) as Ok).reused, 'the complete fresh run replaced it and is served');
    } finally { w.dispose(); }
  });

  test('a partial capture (e.g. byte limit) is stored for comparison only', async () => {
    const w = worker();
    try {
      state({ partial: true });
      await go(w);
      await go(w);
      assert.equal(launches(), 2, 'never served');
      assert.ok(lines.some(line => /shadow miss: stored [0-9a-f]{12} \(comparison only: capture not complete, never served\)$/.test(line)));
      assert.equal(w.executionCacheStats().served, 0);
    } finally { w.dispose(); }
  });

  test('servableEntry refuses an oversized entry, a digest mismatch and a protocol mismatch', async () => {
    const w = worker();
    try {
      await go(w);
      const cache = store(w);
      const [key] = [...(cache as unknown as { entries: Map<string, unknown> }).entries.keys()];
      const entry = cache.entry(key)!;
      const meta = JSON.parse(entry.meta!);
      const snapshot = { plan: meta.plan, applicabilityDigest: 'a'.repeat(64), guards: {} } as ExecutionSnapshot;
      assert.equal(servableEntry(entry, snapshot, WORKER_PROTOCOL_VERSION).ok, true);
      assert.deepEqual(servableEntry({ ...entry, text: entry.text + ' '.repeat(MAX_ENTRY_BYTES) }, snapshot, WORKER_PROTOCOL_VERSION),
        { ok: false, reason: 'oversized' });
      assert.deepEqual(servableEntry(entry, { ...snapshot, applicabilityDigest: 'b'.repeat(64) }, WORKER_PROTOCOL_VERSION),
        { ok: false, reason: 'plan_digest' });
      assert.deepEqual(servableEntry(entry, snapshot, WORKER_PROTOCOL_VERSION + 1), { ok: false, reason: 'protocol' });
      assert.deepEqual(servableEntry(undefined, snapshot, WORKER_PROTOCOL_VERSION), { ok: false, reason: 'missing' });
    } finally { w.dispose(); }
  });

  test('mode change takes effect on the next run: on -> shadow runs (keeps entries) -> off drops them', async () => {
    const w = worker();
    try {
      await go(w);
      settings = { mode: 'shadow' };
      const shadow = await go(w) as Ok;
      assert.equal(shadow.reused, undefined);
      assert.equal(launches(), 2, 'shadow never serves');
      settings = { mode: 'on' };
      assert.ok((await go(w) as Ok).reused);
      assert.equal(launches(), 2);
      settings = { mode: 'off' };
      await go(w);
      assert.equal(launches(), 3);
      settings = { mode: 'on' };
      await go(w);
      assert.equal(launches(), 4, 'off dropped the store');
    } finally { w.dispose(); }
  });

  test('clearExecutionCache drops every entry; the next run executes', async () => {
    const w = worker();
    try {
      await go(w);
      assert.equal(w.clearExecutionCache(), 1);
      assert.equal(w.executionCacheStats().entries, 0);
      await go(w);
      assert.equal(launches(), 2);
    } finally { w.dispose(); }
  });

  test('a cancel while a hit is being rechecked returns cancelled and no reused result', async () => {
    const w = worker();
    try {
      await go(w);
      state({ snapshotDelayMs: 300 });
      const before = snapshots();
      const pending = w.run(request());
      // Wait for the serve path's own freshness recheck (the run's second snapshot request).
      while (snapshots() < before + 2) await new Promise(resolve => setTimeout(resolve, 10));
      w.cancel();
      const outcome = await pending;
      assert.equal(outcome.ok, false);
      assert.equal((outcome as { cancelled?: boolean }).cancelled, true);
      assert.equal(launches(), 1);
      assert.equal(w.executionCacheStats().served, 0);
    } finally { w.dispose(); }
  });

  test('candidate runs and runs without the cache option never look up', async () => {
    const plain = new TraceWorkerCore(backend, { appendLine: line => lines.push(line) }, {
      interpreter: async () => python, trusted: () => true, documents: () => ({}) });
    try {
      await plain.run(request()); await plain.run(request());
      assert.equal(launches(), 2);
      assert.equal(plain.executionCacheStats().lookups, 0);
    } finally { plain.dispose(); }
  });
});
