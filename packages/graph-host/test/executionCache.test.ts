import * as assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, rmSync, utimesSync, writeFileSync } from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {
  ALLOWED_DOCUMENT_VERSION_LISTS, ALLOWED_GENERATION_PATHS, ALLOWED_TIMING_PATHS, ExecutionResultStore, envDigest, environmentFingerprint,
  firstDifference, machineFacts, normaliseEnvelope, outcomeUnstorable, requestIneligible, resultKey, resultKeyParts, reusedResultLabel,
  shadowDifference, type ResultKeyParts,
} from '@tensorviz/graph-host/executionCache';
import { backendDigest, BACKEND_IDENTITY_FILE, resetBackendDigestCache } from '@tensorviz/graph-host/backendIdentity';
import type { ExecutionSnapshot } from '@tensorviz/graph-host/executionPlanCore';
import type { TraceOutcome, TraceRequest, WorkerEnvelope } from '@tensorviz/graph-host/traceWorkerCore';

const DIGEST = 'a'.repeat(64);
function parts(overrides: Partial<ResultKeyParts> = {}): ResultKeyParts {
  return {
    version: 1, applicability: DIGEST, argv: ['/b/services/ingest/bootstrap.py', '--file', '/w/m.py', '--cls', 'M',
      '--resolved-execution', '--workspace', '/w', '--tier', 'real', '--depth', '2', '--json'],
    backend: { digest: 'source:' + 'b'.repeat(64), protocol: 1 }, env: 'c'.repeat(64),
    machine: { platform: 'darwin', arch: 'arm64', cpus: 12, cpuModel: 'Apple M3 Pro' }, interpreter: '/venv/bin/python',
    ...overrides,
  };
}

suite('execution cache key', () => {
  test('every component changes the key', () => {
    const base = resultKey(parts());
    const variants: Partial<ResultKeyParts>[] = [
      { version: 2 },
      { applicability: 'd'.repeat(64) },
      { argv: parts().argv.map(a => (a === '2' ? '3' : a)) },            // --depth
      { argv: parts().argv.map(a => (a === 'real' ? 'fake' : a)) },      // --tier
      { argv: [...parts().argv, '--kind', 'train_step'] },               // kind
      { argv: [...parts().argv, '--no-frames'] },                        // capture flags
      { argv: parts().argv.map(a => (a === 'M' ? 'N' : a)) },            // class
      { backend: { digest: 'source:' + 'e'.repeat(64), protocol: 1 } },
      { backend: { digest: parts().backend.digest, protocol: 2 } },
      { env: 'f'.repeat(64) },
      { machine: { ...parts().machine, cpus: 8 } },
      { machine: { ...parts().machine, arch: 'x64' } },
      { machine: { ...parts().machine, platform: 'linux' } },
      { machine: { ...parts().machine, cpuModel: 'Apple M1' } },
      { interpreter: '/other/bin/python' },
    ];
    const keys = new Set([base]);
    for (const variant of variants) {
      const key = resultKey(parts(variant));
      assert.notEqual(key, base, JSON.stringify(variant));
      keys.add(key);
    }
    assert.equal(keys.size, variants.length + 1);
  });

  test('object key order does not change the key', () => {
    const a = parts();
    const b = { interpreter: a.interpreter, machine: { cpuModel: a.machine.cpuModel, cpus: a.machine.cpus, arch: a.machine.arch,
      platform: a.machine.platform }, env: a.env, backend: { protocol: 1, digest: a.backend.digest }, argv: a.argv,
    applicability: a.applicability, version: 1 } as ResultKeyParts;
    assert.equal(resultKey(a), resultKey(b));
  });

  test('environment digest covers every variable except the per-run plan and candidate paths', () => {
    const env = { PATH: '/usr/bin', OMP_NUM_THREADS: '4', HOME: '/h', KMP_DUPLICATE_LIB_OK: 'TRUE' };
    const base = envDigest(env);
    assert.equal(envDigest({ ...env, TENSORVIZ_EXECUTION_PLAN: '/tmp/x/plan.json' }), base);
    assert.equal(envDigest({ ...env, TENSORVIZ_CANDIDATE_REQUEST: '/tmp/y' }), base);
    assert.equal(envDigest({ ...env, UNSET: undefined }), base);
    assert.equal(envDigest({ HOME: '/h', KMP_DUPLICATE_LIB_OK: 'TRUE', OMP_NUM_THREADS: '4', PATH: '/usr/bin' }), base);
    for (const changed of [{ OMP_NUM_THREADS: '1' }, { MY_MODEL_FLAG: '1' }, { PYTHONPATH: '/x' }, { HF_HOME: '/c' }]) {
      assert.notEqual(envDigest({ ...env, ...changed }), base, JSON.stringify(changed));
    }
  });

  test('resultKeyParts derives backend, env, machine and interpreter from the run', () => {
    const backendDir = mkdtempSync(path.join(os.tmpdir(), 'tv-backend-'));
    try {
      writeFileSync(path.join(backendDir, 'a.py'), 'x = 1\n');
      resetBackendDigestCache();
      const input = { applicabilityDigest: DIGEST, args: ['--depth', '2'], backendDir, protocol: 1,
        env: { A: '1' }, interpreter: '/p' };
      const p = resultKeyParts(input);
      assert.equal(p.applicability, DIGEST);
      assert.deepEqual(p.argv, ['--depth', '2']);
      assert.match(p.backend.digest, /^source:[a-f0-9]{64}$/);
      assert.equal(p.env, envDigest({ A: '1' }));
      assert.deepEqual(p.machine, machineFacts());
      assert.equal(p.machine.cpus, os.cpus().length);
      assert.equal(p.interpreter, '/p');
    } finally { rmSync(backendDir, { recursive: true, force: true }); }
  });
});

suite('backend digest', () => {
  let dir: string;
  setup(() => { dir = mkdtempSync(path.join(os.tmpdir(), 'tv-backend-')); resetBackendDigestCache(); });
  teardown(() => rmSync(dir, { recursive: true, force: true }));

  test('a checkout hashes every .py, is stable, and changes on any backend edit', () => {
    mkdirSync(path.join(dir, 'services/ingest'), { recursive: true });
    mkdirSync(path.join(dir, '__pycache__'));
    writeFileSync(path.join(dir, 'services/ingest/worker.py'), 'A = 1\n');
    writeFileSync(path.join(dir, 'server.py'), 'B = 2\n');
    writeFileSync(path.join(dir, '__pycache__/x.py'), 'ignored\n');
    writeFileSync(path.join(dir, 'notes.txt'), 'ignored\n');
    mkdirSync(path.join(dir, 'tests'));
    writeFileSync(path.join(dir, 'tests/test_x.py'), 'ignored\n');
    const first = backendDigest(dir);
    assert.match(first, /^source:[a-f0-9]{64}$/);
    assert.equal(backendDigest(dir), first);
    writeFileSync(path.join(dir, 'notes.txt'), 'still ignored\n');
    writeFileSync(path.join(dir, '__pycache__/x.py'), 'still ignored\n');
    writeFileSync(path.join(dir, 'tests/test_x.py'), 'never imported by the worker\n');
    assert.equal(backendDigest(dir), first);
    // Same size, different bytes, forced different mtime: the cached digest must not survive.
    writeFileSync(path.join(dir, 'services/ingest/worker.py'), 'A = 3\n');
    const later = new Date(Date.now() + 5000);
    utimesSync(path.join(dir, 'services/ingest/worker.py'), later, later);
    const edited = backendDigest(dir);
    assert.notEqual(edited, first);
    writeFileSync(path.join(dir, 'services/new.py'), '');
    assert.notEqual(backendDigest(dir), edited);
  });

  test('a staged backend uses its identity file', () => {
    writeFileSync(path.join(dir, 'a.py'), 'x\n');
    writeFileSync(path.join(dir, BACKEND_IDENTITY_FILE), JSON.stringify({ format: 1, digest: 'd'.repeat(64) }));
    assert.equal(backendDigest(dir), `staged:${'d'.repeat(64)}`);
    resetBackendDigestCache();
    writeFileSync(path.join(dir, BACKEND_IDENTITY_FILE), JSON.stringify({ format: 2, digest: 'not-a-digest' }));
    assert.match(backendDigest(dir), /^source:/);
  });
});

function snapshot(execution: Record<string, unknown> = {}, job: unknown = null): ExecutionSnapshot {
  return { applicabilityDigest: DIGEST, guards: {},
    plan: { execution: { device: 'cpu', mode: 'eval', seed: 0, deterministic: true, gradEnabled: false, cwd: '.', inputs: {}, ...execution }, job } } as unknown as ExecutionSnapshot;
}
const request: TraceRequest = { file: '/w/m.py', cls: 'M', workspace: '/w', execution: { inputs: { args: [] } } as never, reuse: 'allow' };

suite('execution cache eligibility', () => {
  test('a seeded deterministic CPU model run of a resolved plan is eligible', () => {
    assert.equal(requestIneligible(request, snapshot(), false), undefined);
  });
  const cases: [string, TraceRequest, ExecutionSnapshot | undefined, boolean, string][] = [
    ['not opted in', { ...request, reuse: undefined }, snapshot(), false, 'not_requested'],
    ['candidate', request, snapshot(), true, 'candidate'],
    ['input preparation', { ...request, prepareInputs: true }, snapshot(), false, 'input_preparation'],
    ['config defaults preparation', { ...request, prepareInputs: true, prepareConfigDefaults: true }, snapshot(), false, 'input_preparation'],
    ['automatic input search (no plan)', { ...request, execution: undefined }, undefined, false, 'no_resolved_plan'],
    ['diagnostic job on the request', { ...request, job: { kind: 'profile' } as never }, snapshot(), false, 'diagnostic_job'],
    ['diagnostic job on the plan', request, snapshot({}, { kind: 'training' }), false, 'diagnostic_job'],
    ['train_step', { ...request, kind: 'train_step' }, snapshot(), false, 'kind'],
    ['data', { ...request, kind: 'data' }, snapshot(), false, 'kind'],
    ['watch steps', { ...request, steps: 3 }, snapshot(), false, 'kind'],
    ['fake tier', { ...request, tier: 'fake' }, snapshot(), false, 'tier'],
    ['no seed', request, snapshot({ seed: null }), false, 'no_seed'],
    ['nondeterministic', request, snapshot({ deterministic: false }), false, 'nondeterministic'],
    ['mps', request, snapshot({ device: 'mps' }), false, 'device'],
    ['cuda', request, snapshot({ device: 'cuda:0' }), false, 'device'],
    ['jax', request, snapshot({ framework: 'jax' }), false, 'jax'],
  ];
  for (const [name, req, snap, candidate, reason] of cases) {
    test(`excluded: ${name}`, () => assert.equal(requestIneligible(req, snap, candidate), reason));
  }
});

function envelope(reuse: unknown = { storable: true, reasons: [], untrackedReads: 0, environment: [] }, extra: Record<string, unknown> = {}): WorkerEnvelope {
  return { protocol_version: 1, result: { graph: {}, diagnostics: { execution: { planDigest: DIGEST, planCurrent: true },
    semantic_capture: { id: 'capture:1' }, semantic_capture_error: null, reuse, ...extra } }, spans: {}, error: null, worker: {} };
}
const okOutcome = (env = envelope()): TraceOutcome => ({ ok: true, envelope: env, exitCode: 0, executionCurrent: true });

suite('execution cache storability', () => {
  test('an accepted, current, complete, storable run is stored', () => {
    assert.equal(outcomeUnstorable(okOutcome(), snapshot()), undefined);
  });
  const cases: [string, TraceOutcome, string][] = [
    ['failed forward (partial graph)', { ok: false, reason: 'x', detail: '', envelope: envelope(), exitCode: 4 }, 'failed'],
    ['cancelled', { ok: false, reason: 'x', detail: '', cancelled: true }, 'cancelled'],
    ['timed out', { ok: false, reason: 'x', detail: '', timedOut: true }, 'timed_out'],
    ['nonzero exit', { ...okOutcome(), exitCode: 4 } as TraceOutcome, 'exit_code'],
    ['host recheck not current', { ...okOutcome(), executionCurrent: false } as TraceOutcome, 'not_current'],
    ['recheck missing', { ...okOutcome(), executionCurrent: undefined } as TraceOutcome, 'not_current'],
    ['worker error', okOutcome({ ...envelope(), error: { type: 'TraceError', message: '', frames: [] } }), 'worker_error'],
    ['plan not current in worker', okOutcome(envelope(undefined, { execution: { planDigest: DIGEST, planCurrent: false } })), 'not_current'],
    ['plan digest mismatch', okOutcome(envelope(undefined, { execution: { planDigest: 'e'.repeat(64), planCurrent: true } })), 'not_current'],
    ['no semantic capture', okOutcome(envelope(undefined, { semantic_capture: null })), 'partial'],
    ['semantic capture error', okOutcome(envelope(undefined, { semantic_capture: null, semantic_capture_error: 'byte_limit' })), 'partial'],
    ['old backend without a reuse report', okOutcome(envelope(null)), 'no_reuse_report'],
    ['untracked read', okOutcome(envelope({ storable: false, reasons: ['untracked_workspace_read'], untrackedReads: 1 })), 'worker:untracked_workspace_read'],
    ['write', okOutcome(envelope({ storable: false, reasons: ['write'], untrackedReads: 0 })), 'worker:write'],
    ['native code', okOutcome(envelope({ storable: false, reasons: ['native_code'], untrackedReads: 0 })), 'worker:native_code'],
    // Without the installed files it read, a stored result could not be rechecked before serving.
    ['storable report without installed-file fingerprints', okOutcome(envelope({ storable: true, reasons: [], untrackedReads: 0 })),
      'no_environment_report'],
    ['malformed installed-file fingerprints', okOutcome(envelope({ storable: true, reasons: [], untrackedReads: 0,
      environment: [['/venv/lib/site-packages/x.py', 12]] })), 'no_environment_report'],
  ];
  for (const [name, outcome, reason] of cases) {
    test(`never stored: ${name}`, () => assert.equal(outcomeUnstorable(outcome, snapshot()), reason));
  }
});

suite('execution result store', () => {
  const sized = (bytes: number): WorkerEnvelope => ({ ...envelope(), spans: { pad: 'x'.repeat(bytes) } });
  test('returns a fresh copy; mutating it does not change the store', () => {
    const store = new ExecutionResultStore();
    assert.equal(store.put('k', envelope()), true);
    const first = store.get('k')!;
    (first.result as { graph: { mutated?: boolean } }).graph.mutated = true;
    assert.deepEqual(store.get('k'), envelope());
    assert.equal(store.get('missing'), undefined);
  });
  test('evicts least recently used beyond the entry cap (default 32)', () => {
    const store = new ExecutionResultStore();
    for (let i = 0; i < 32; i++) store.put(`k${i}`, envelope());
    store.get('k0');                       // k0 becomes most recent
    store.put('k32', envelope());
    assert.equal(store.size, 32);
    assert.equal(store.has('k0'), true);
    assert.equal(store.has('k1'), false);  // oldest unused
    assert.equal(store.has('k32'), true);
  });
  test('evicts to the byte cap and refuses entries over the per-entry cap', () => {
    const store = new ExecutionResultStore({ maxBytes: 3000, maxEntryBytes: 1500 });
    assert.equal(store.put('a', sized(1000)), true);
    assert.equal(store.put('b', sized(1000)), true);
    assert.equal(store.put('c', sized(1000)), true);   // a evicted (3 x ~1.2 KB > 3000)
    assert.equal(store.has('a'), false);
    assert.ok(store.bytes <= 3000);
    assert.equal(store.put('huge', sized(2000)), false);
    assert.equal(store.has('huge'), false);
    store.resize(1500);
    assert.equal(store.size, 1);
    assert.ok(store.bytes <= 1500);
  });
  test('the default per-entry cap is 16 MB', () => {
    const store = new ExecutionResultStore({ maxBytes: 64 * 1024 * 1024 });
    assert.equal(store.put('big', sized(16 * 1024 * 1024)), false);
    assert.equal(store.put('ok', sized(1024 * 1024)), true);
  });
  test('refreshing a key replaces its bytes; clear empties', () => {
    const store = new ExecutionResultStore();
    store.put('k', sized(5000));
    const before = store.bytes;
    store.put('k', sized(10));
    assert.ok(store.bytes < before);
    assert.equal(store.size, 1);
    store.clear();
    assert.equal(store.size, 0);
    assert.equal(store.bytes, 0);
  });
});

/** Shape of a resolved-plan envelope, carrying every field that differs between two runs
 *  of an identical plan in the 69-model corpus measurement. */
function captured(run: { plan: string; source: string; attempt: string; capture: string; generation: number;
  bindings: string[]; elapsed: number; build: number; trace: number; versions: (number | null)[] }): Record<string, unknown> {
  const [b0, b1] = run.bindings;
  const files = () => [{ path: 'models/m.py', digest: 'b'.repeat(64), documentVersion: run.versions[0], role: 'source' },
    { path: 'models/__init__.py', digest: 'c'.repeat(64), documentVersion: run.versions[1], role: 'dependency' }];
  return {
    protocol_version: 1, spans: {}, error: null, worker: { python: '3.12', planId: run.plan, planDigest: DIGEST, elapsed_ms: run.elapsed },
    result: { graph: { nodes: [{ id: 'linear1', shape: [8, 4] }] }, diagnostics: {
      timings_ms: { build_ms: run.build, trace_ms: run.trace },
      worker: { planId: run.plan, elapsed_ms: run.elapsed },
      execution: { planDigest: DIGEST, planCurrent: true,
        resolvedPlan: { id: run.plan, source: { id: run.source, generation: run.generation, workspaceSessionId: 'session-1', files: files() } } },
      run_evidence: { outputs: [{ digest: 'f'.repeat(64), values: [0.25, -1.5] }] },
      semantic_capture: { id: run.capture, attemptId: run.attempt,
        source: { id: run.source, generation: run.generation, workspaceSessionId: 'session-1', files: files() },
        sourceBindings: [{ id: b0, snapshotId: run.source, line: 3 }, { id: b1, snapshotId: run.source, line: 9 }],
        definitions: [{ name: 'M', sourceBindingIds: [b0] }],
        modules: [{ path: 'fc', constructorBindingIds: [b1] }],
        invocations: [{ path: 'fc', sourceBindingIds: [b1, b0] }],
        operations: [{ op: 'aten.addmm', sourceBindingIds: [b1] }] },
      reuse: { storable: true, reasons: [], untrackedReads: 0, environment: [] } } },
  };
}
const runA = { plan: 'plan-a', source: 'source:run-a', attempt: 'attempt-a', capture: 'capture:1111', generation: 3,
  bindings: ['bind-a0', 'bind-a1'], elapsed: 812.4, build: 101.2, trace: 55.1, versions: [1, null] };
const runB = { plan: 'plan-b', source: 'source:run-b', attempt: 'attempt-b', capture: 'capture:2222', generation: 7,
  bindings: ['bind-b0', 'bind-b1'], elapsed: 790.1, build: 99.8, trace: 57.3, versions: [3, 1] };

suite('execution cache allowed differences', () => {
  test('the allowed-difference list is exactly the measured set', () => {
    // Measured: 69 corpus models x 2 runs with different attempt ids in one host process
    // (~/tv-bench/cache/twice-main.json). Everything else was byte-identical. Adding an
    // entry here needs the same evidence.
    assert.deepEqual([...ALLOWED_TIMING_PATHS], ['worker.elapsed_ms', 'result.diagnostics.worker.elapsed_ms',
      'result.diagnostics.timings_ms.build_ms', 'result.diagnostics.timings_ms.trace_ms']);
    assert.deepEqual([...ALLOWED_GENERATION_PATHS], ['result.diagnostics.semantic_capture.source.generation',
      'result.diagnostics.execution.resolvedPlan.source.generation']);
    // Measured: 11 models in installed VS Code, re-run and undo in shadow mode
    // (~/tv-bench/rollout/smoke10): all 13 DIFFERENT verdicts were a file's documentVersion,
    // and with it neutralised 22/22 were identical. The key already leaves it out.
    assert.deepEqual([...ALLOWED_DOCUMENT_VERSION_LISTS], ['result.diagnostics.execution.resolvedPlan.source.files',
      'result.diagnostics.semantic_capture.source.files']);
  });

  test('two runs of an identical plan normalise to the same envelope', () => {
    const a = captured(runA), b = captured(runB);
    assert.notEqual(firstDifference(a, b), undefined);
    assert.equal(firstDifference(normaliseEnvelope(a), normaliseEnvelope(b)), undefined);
    assert.equal(shadowDifference(a, b), undefined);
  });

  test('binding ids normalise by position, so reordered references are still a difference', () => {
    const a = captured(runA), b = captured(runB);
    const semantic = ((b.result as { diagnostics: { semantic_capture: { invocations: { sourceBindingIds: string[] }[] } } })
      .diagnostics.semantic_capture);
    semantic.invocations[0].sourceBindingIds.reverse();
    assert.equal(firstDifference(normaliseEnvelope(a), normaliseEnvelope(b)),
      '.result.diagnostics.semantic_capture.invocations[0].sourceBindingIds[0]');
    assert.equal(shadowDifference(a, b), '.result.diagnostics.semantic_capture.invocations[0].sourceBindingIds[0]');
  });

  const meaningful: [string, (e: any) => void, string][] = [
    ['an output value', e => { e.result.diagnostics.run_evidence.outputs[0].values[1] = -1.25; }, '.result.diagnostics.run_evidence.outputs[0].values[1]'],
    ['a shape', e => { e.result.graph.nodes[0].shape = [8, 5]; }, '.result.graph.nodes[0].shape[1]'],
    ['the plan digest', e => { e.worker.planDigest = 'e'.repeat(64); }, '.worker.planDigest'],
    ['the workspace session', e => { e.result.diagnostics.semantic_capture.source.workspaceSessionId = 'session-2'; },
      '.result.diagnostics.semantic_capture.source.workspaceSessionId'],
    ['a timing field that is not on the list', e => { e.result.diagnostics.timings_ms.forward_ms = 1; }, '.result.diagnostics.timings_ms.forward_ms'],
    ['a new key', e => { e.result.diagnostics.extra = true; }, '.result.diagnostics.extra'],
    ['the reuse report', e => { e.result.diagnostics.reuse.untrackedReads = 1; }, '.result.diagnostics.reuse.untrackedReads'],
    ['an identity token used as data elsewhere', e => { e.result.graph.nodes[0].id = 'capture:2222-other'; }, '.result.graph.nodes[0].id'],
    ['a source file digest', e => { e.result.diagnostics.semantic_capture.source.files[1].digest = 'd'.repeat(64); },
      '.result.diagnostics.semantic_capture.source.files[1].digest'],
    ['a source file role', e => { e.result.diagnostics.execution.resolvedPlan.source.files[0].role = 'dependency'; },
      '.result.diagnostics.execution.resolvedPlan.source.files[0].role'],
    ['an extra source file', e => { e.result.diagnostics.execution.resolvedPlan.source.files.push({ path: 'x.py', digest: 'e'.repeat(64),
      documentVersion: null, role: 'dependency' }); }, '.result.diagnostics.execution.resolvedPlan.source.files[length]'],
    ['a documentVersion outside the source file lists', e => { e.result.graph.nodes[0].documentVersion = 4; },
      '.result.graph.nodes[0].documentVersion'],
    ['a binding line', e => { e.result.diagnostics.semantic_capture.sourceBindings[1].line = 10; },
      '.result.diagnostics.semantic_capture.sourceBindings[1].line'],
  ];
  for (const [name, mutate, at] of meaningful) {
    test(`still a difference: ${name}`, () => {
      const a = captured(runA), b = captured(runB);
      mutate(b);
      assert.equal(firstDifference(normaliseEnvelope(a), normaliseEnvelope(b)), at);
      // The copy-free comparison used by the shadow hook gives the same answer.
      assert.equal(shadowDifference(a, b), at);
    });
  }

  test('an allowed path missing on one side is still a difference', () => {
    const a = captured(runA), b = captured(runB);
    delete (b.worker as Record<string, unknown>).elapsed_ms;
    assert.equal(shadowDifference(a, b), '.worker.elapsed_ms');
    assert.equal(firstDifference(normaliseEnvelope(a), normaliseEnvelope(b)), '.worker.elapsed_ms');
  });

  test('firstDifference reports the first differing path, length and kind changes', () => {
    assert.equal(firstDifference({ a: 1, b: [1, 2] }, { a: 1, b: [1, 2] }), undefined);
    assert.equal(firstDifference({ a: 1, b: [1, 2] }, { a: 1, b: [1, 3] }), '.b[1]');
    assert.equal(firstDifference({ b: [1] }, { b: [1, 2] }), '.b[length]');
    assert.equal(firstDifference({ a: null }, { a: {} }), '.a');
    assert.equal(firstDifference({ a: 1 }, { b: 1 }), '.a');
  });
});

suite('execution cache installed-file fingerprints', () => {
  // The worker records them (reuse_audit.fingerprint) and the host compares them before
  // serving. If the two disagreed on the format, every run that read an installed file would
  // be refused; if they agreed on too little, an edit would be missed.
  test('the host computes the worker\'s fingerprint for a file, a folder and a missing path', function () {
    this.timeout(30_000);   // a cold interpreter start
    const dir = mkdtempSync(path.join(os.tmpdir(), 'tv-fingerprint-'));
    try {
      const file = path.join(dir, 'layers.py');
      writeFileSync(file, 'WIDTH = 3\n');
      const paths = [file, dir, path.join(dir, 'absent.py'), path.join(file, 'below-a-file')];
      const script = 'import importlib.util, json, sys\n'
        + 'spec = importlib.util.spec_from_file_location("reuse_audit", sys.argv[1])\n'
        + 'module = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\n'
        + 'print(json.dumps([module.fingerprint(p) for p in sys.argv[2:]]))\n';
      const audit = path.join(__dirname, '../../../backend/services/ingest/reuse_audit.py');
      const python = spawnSync(process.env.TENSORVIZ_TEST_PYTHON || process.env.PYTHON || 'python3', ['-I', '-c', script, audit, ...paths], { encoding: 'utf8' });
      if (python.error) this.skip();
      assert.equal(python.status, 0, python.stderr);
      const expected = JSON.parse(python.stdout) as (string | null)[];
      assert.deepEqual(paths.map(environmentFingerprint), expected);
      assert.equal(expected[2], null);
      assert.equal(expected[3], null);
      writeFileSync(file, 'WIDTH = 30\n');
      assert.notEqual(environmentFingerprint(file), expected[0], 'an in-place edit changes it');
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });
});

suite('reused result label', () => {
  test('says the model was not run again, not that no Python ran', () => {
    // Serving still runs the user's interpreter to resolve the plan and recheck it.
    const captured = new Date(2026, 9, 25, 14, 2);
    const label = reusedResultLabel(captured.getTime(), new Date(2026, 9, 25, 16, 0).getTime());
    // Same day: the time alone, in the machine's locale (its digits need not be ASCII).
    const time = captured.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    assert.equal(label, `Reused identical result from ${time} (model not run again).`);
    assert.doesNotMatch(label, /Python/);
  });
});
