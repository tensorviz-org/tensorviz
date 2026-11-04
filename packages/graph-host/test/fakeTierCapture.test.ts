import * as assert from 'node:assert/strict';
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { TraceWorkerCore, fakeTensorLimitation, workerArgs, type TraceOutcome, type TraceRequest } from '@tensorviz/graph-host/traceWorkerCore';
import { requestIneligible } from '@tensorviz/graph-host/executionCache';
import { disposeSnapshotHelpers } from '@tensorviz/graph-host/hostSnapshot';
import type { ExecutionSnapshot } from '@tensorviz/graph-host/executionPlanCore';

/** A fake interpreter (a node script) answering the probe, the capture-preparation helper
 *  and the worker like Python would. Each worker launch appends `<tier> <preparation id>`
 *  to `runs.log`; each preparation appends to `preparations.log`. `state.json` decides
 *  whether a fake-tier run stops on a fake-tensor limitation. */
const FAKE = String.raw`
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const argv = process.argv.slice(2);
const state = () => JSON.parse(fs.readFileSync(path.join(process.cwd(), 'state.json'), 'utf8'));
if (argv[0] === '-c') { process.stdout.write('2.7.1 2.0.0\n'); process.exit(0); }
if (argv.some(a => a.endsWith('host_snapshot.py'))) {
  const answer = () => { const id = 'preparation-' + crypto.randomUUID();
    fs.appendFileSync(path.join(process.cwd(), 'preparations.log'), id + '\n');
    return JSON.stringify({ applicabilityDigest: 'a'.repeat(64), guards: {},
      preparation: { version: 1, id, interpreter: { executable: process.argv[1] }, source: { id: 'source:' + id },
        request: { input: null, tensorInputs: null, overrides: null }, timeoutMs: 20000 } }); };
  if (argv.includes('--serve')) {
    let buffer = '';
    process.stdin.on('data', chunk => { buffer += chunk; let i; while ((i = buffer.indexOf('\n')) >= 0) { buffer = buffer.slice(i + 1); process.stdout.write(answer() + '\n'); } });
  } else { process.stdin.resume(); process.stdin.on('end', () => process.stdout.write(answer())); }
} else if (argv.some(a => a.endsWith('bootstrap.py'))) {
  const s = state(), tier = argv[argv.indexOf('--tier') + 1];
  const plan = JSON.parse(fs.readFileSync(process.env.TENSORVIZ_EXECUTION_PLAN, 'utf8'));
  fs.appendFileSync(path.join(process.cwd(), 'runs.log'), tier + ' ' + plan.preparation.id + ' ' + argv.includes('--capture-preparation') + '\n');
  const stop = tier === 'fake' && s.limitation;
  const error = stop ? { type: 'TraceError', message: 'FakeTensorLimitation: the model reads a tensor value', frames: [], node_id: null,
    fakeTensor: { reason: s.limitation, errorType: 'DataDependentOutputException', operation: 'aten._local_scalar_dense.default', phase: 'forward' } }
    : s.otherError && tier === 'fake' ? { type: 'TraceError', message: 'RuntimeError: shapes cannot be multiplied', frames: [], node_id: null } : null;
  process.stdout.write(JSON.stringify({ protocol_version: 1, spans: {}, error, worker: {},
    result: { graph: { nodes: [], edges: [] }, diagnostics: { tier,
      automatic_capture: { preparationDigest: plan.applicabilityDigest, current: true, importedSources: [] } } } }) + '\n');
  process.exit(error ? 4 : 0);
} else process.exit(2);
`;

suite('TraceWorkerCore fake-tier automatic capture', function () {
  this.timeout(30_000);
  let root: string, backend: string, workspace: string, python: string, lines: string[];
  const state = (value: Record<string, unknown>) => writeFileSync(path.join(backend, 'state.json'), JSON.stringify(value));
  const worker = () => new TraceWorkerCore(backend, { appendLine: line => lines.push(line) }, {
    interpreter: async () => python, trusted: () => true, documents: () => ({}),
  });
  const request = (extra: Partial<TraceRequest> = {}): TraceRequest => ({ file: path.join(workspace, 'm.py'), cls: 'M', workspace, ...extra });
  const log = (name: string) => existsSync(path.join(backend, name))
    ? readFileSync(path.join(backend, name), 'utf8').split('\n').filter(Boolean) : [];

  setup(() => {
    root = mkdtempSync(path.join(os.tmpdir(), 'tv-fake-tier-'));
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
    lines = [];
  });
  teardown(() => { disposeSnapshotHelpers(); rmSync(root, { recursive: true, force: true }); });

  test('workerArgs: automatic preparation accepts the fake tier; explicit execution and candidates still refuse it', () => {
    const args = workerArgs(request({ tier: 'fake' }), 'preparation');
    assert.ok(args.includes('--capture-preparation'));
    assert.deepEqual(args.slice(args.indexOf('--tier'), args.indexOf('--tier') + 2), ['--tier', 'fake']);
    assert.throws(() => workerArgs(request({ tier: 'fake', execution: { inputs: { args: [] } } as never }), 'plan'), /real model tier/);
    assert.throws(() => workerArgs(request({ tier: 'fake' }), 'candidate'), /Invalid candidate request/);
    assert.throws(() => workerArgs(request({ tier: 'fake', kind: 'data', callable: 'f' }), 'preparation'), /Automatic preparation/);
  });

  test('a fake-tier run is an automatic canonical capture (preparation + bootstrap), not the legacy launch', async () => {
    const w = worker();
    try {
      const outcome = await w.run(request({ tier: 'fake' }));
      assert.equal(outcome.ok, true, JSON.stringify(outcome));
      const runs = log('runs.log');
      assert.equal(runs.length, 1);
      assert.match(runs[0], /^fake preparation-\S+ true$/);
      assert.ok(log('preparations.log').includes(runs[0].split(' ')[1]), 'the worker ran the host-prepared capture');
      assert.equal(outcome.fakeTensorFallback, undefined);
    } finally { w.dispose(); }
  });

  test('a real-tier run is unchanged: one run, no fallback field', async () => {
    const w = worker();
    try {
      state({ limitation: 'data_dependent' });
      for (const tier of [undefined, 'real'] as const) {
        const outcome = await w.run(request(tier ? { tier } : {}));
        assert.equal(outcome.ok, true);
        assert.ok(!('fakeTensorFallback' in outcome));
      }
      assert.deepEqual(log('runs.log').map(line => line.split(' ')[0]), ['real', 'real']);
    } finally { w.dispose(); }
  });

  test('a fake-tensor limitation re-runs exactly once with real tensors and the same preparation', async () => {
    const w = worker();
    try {
      state({ limitation: 'data_dependent' });
      let prepared = 0;
      const outcome = await w.run(request({ tier: 'fake', onCapturePrepared: () => { prepared += 1; } }));
      assert.equal(outcome.ok, true, JSON.stringify(outcome));
      assert.deepEqual(outcome.fakeTensorFallback, { reason: 'data_dependent', errorType: 'DataDependentOutputException',
        operation: 'aten._local_scalar_dense.default', phase: 'forward' });
      const runs = log('runs.log').map(line => line.split(' '));
      assert.deepEqual(runs.map(run => run[0]), ['fake', 'real']);
      assert.equal(runs[0][1], runs[1][1], 'the real re-run uses the attempt\'s one preparation');
      // (Later preparations are the host's post-run freshness rechecks, never a run's input.)
      assert.equal(prepared, 2, 'the same preparation is reported again (the ledger keeps the first)');
      assert.ok(lines.some(line => /fake-tensor run stopped: data_dependent .* re-running once with real tensors/.test(line)));
    } finally { w.dispose(); }
  });

  test('any other fake-tier failure is reported as is, without a real re-run', async () => {
    const w = worker();
    try {
      state({ otherError: true });
      const outcome = await w.run(request({ tier: 'fake' }));
      assert.equal(outcome.ok, false);
      assert.match(outcome.envelope?.error?.message ?? '', /shapes cannot be multiplied/);
      assert.equal(outcome.fakeTensorFallback, undefined);
      assert.deepEqual(log('runs.log').map(line => line.split(' ')[0]), ['fake']);
    } finally { w.dispose(); }
  });

  test('a newer request during the fake run suppresses the re-run', async () => {
    const w = worker();
    try {
      state({ limitation: 'data_dependent' });
      let second: Promise<TraceOutcome> | undefined;
      const first = w.run(request({ tier: 'fake', onCapturePrepared: () => { second ??= w.run(request({ tier: 'real' })); } }));
      const [a, b] = await Promise.all([first, (async () => { while (!second) await new Promise(r => setTimeout(r, 5)); return second; })()]);
      assert.ok(!a.ok && a.cancelled, 'the superseded fake run is cancelled');
      assert.equal(a.fakeTensorFallback, undefined);
      assert.equal(b.ok, true);
      assert.deepEqual(log('runs.log').map(line => line.split(' ')[0]), ['real'], 'only the newer request ran');
    } finally { w.dispose(); }
  });

  test('fakeTensorLimitation reads only a well-formed marker on a finished failure', () => {
    const marker = { reason: 'dynamic_output_shape', errorType: 'DynamicOutputShapeException', operation: 'aten.nonzero.default', phase: 'forward' } as const;
    const failed = (fakeTensor: unknown, extra: object = {}): TraceOutcome => ({ ok: false, reason: 'r', detail: 'd', ...extra,
      envelope: { protocol_version: 1, result: null, spans: {}, worker: {}, error: { type: 'TraceError', message: 'm', frames: [], fakeTensor } as never } });
    assert.deepEqual(fakeTensorLimitation(failed(marker)), marker);
    assert.equal(fakeTensorLimitation(failed(undefined)), undefined);
    const lazy = { reason: 'tensor_subclass', errorType: 'RuntimeError', operation: 'UninitializedParameter', phase: 'construction' } as const;
    assert.deepEqual(fakeTensorLimitation(failed(lazy)), lazy, 'a lazy module (tensor subclass) also falls back to real');
    assert.equal(fakeTensorLimitation(failed({ ...marker, reason: 'other' })), undefined);
    assert.equal(fakeTensorLimitation(failed({ ...marker, phase: 'later' })), undefined);
    assert.equal(fakeTensorLimitation(failed(marker, { cancelled: true })), undefined);
    assert.equal(fakeTensorLimitation(failed(marker, { timedOut: true })), undefined);
  });

  test('the execution cache never looks up or stores a fake-tier run', () => {
    const snapshot = { plan: { execution: { seed: 0, deterministic: true, device: 'cpu' }, job: null } } as unknown as ExecutionSnapshot;
    const explicit = request({ reuse: 'allow', execution: { inputs: { args: [] } } as never });
    assert.equal(requestIneligible(explicit, snapshot, false), undefined);
    assert.equal(requestIneligible({ ...explicit, tier: 'fake' }, snapshot, false), 'tier');
    assert.equal(requestIneligible(request({ reuse: 'allow', tier: 'fake' }), undefined, false), 'no_resolved_plan');
  });
});
