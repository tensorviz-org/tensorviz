import * as assert from 'node:assert/strict';
import type { GraphEnvironmentCandidate, LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import { inspectModelEnvironment, startModelInspection } from '../dist/modelEnvironment';
import type { EnvironmentRunner, HelperResult } from '../dist/projectEnvironmentCore';

const target: LocalBrowserTargetIdentity = { workspaceFolder: '/project', file: 'pkg/model.py', cls: 'Net',
  documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 0 };
const candidate: GraphEnvironmentCandidate = { id: 'chosen', interpreter: '/project/.venv/bin/python', source: 'Saved',
  status: 'libraries_present', python: '3.12.13', torch: '2.6.0+cu124', fingerprint: 'b'.repeat(64) };
const options = (runner: EnvironmentRunner) => ({ target, backendDir: '/tool/backend', managerPython: '/tool/python',
  signal: new AbortController().signal, runner });

suite('chosen model environment readiness', () => {
  test('known parent-initializer imports become exact reviewed additions with OpenCV display name', async () => {
    const calls: string[][] = [];
    const result = await inspectModelEnvironment(candidate, options(async (python, backend, args, _cancelled, isolated, parentPipe) => {
      assert.equal(python, '/tool/python'); assert.equal(backend, '/tool/backend'); assert.equal(isolated, true); assert.equal(parentPipe, true);
      calls.push(args); return { ok: true, missingModules: [{ module: 'torchmetrics', requirement: 'torchmetrics' },
        { module: 'cv2', requirement: 'opencv-python' }, { module: 'scipy', requirement: 'scipy' }] };
    }));
    assert.equal(result.reason, 'missing_libraries'); assert.deepEqual(result.requirements, ['torchmetrics', 'opencv-python', 'scipy']);
    assert.match(result.detail, /torchmetrics, OpenCV, SciPy/);
    assert.deepEqual(calls, [['--target-python', candidate.interpreter, 'inspect-model', '/project', '/project/pkg/model.py', '[]']]);
  });

  test('server packages never become capture additions; missing worker requirements are combined with model imports', async () => {
    const result = await inspectModelEnvironment({ ...candidate, status: 'missing_libraries', missing: ['pydantic>=2.12,<3'] },
      options(async () => ({ ok: true, missingModules: [{ module: 'cv2', requirement: 'opencv-python' }] })));
    assert.deepEqual(result.requirements, ['pydantic>=2.12,<3', 'opencv-python']);
    assert.ok(!JSON.stringify(result).includes('fastapi'));
  });

  test('unknown and optional imports are not guessed; Torch absence cannot hide a device/build decision in repair', async () => {
    const runner: EnvironmentRunner = async () => ({ ok: true, missingModules: [
      { module: 'opaque', requirement: null }, { module: 'optional', requirement: 'optional', optional: true }] });
    const result = await inspectModelEnvironment(candidate, options(runner));
    assert.equal(result.reason, 'missing_libraries'); assert.equal(result.requirements?.length, 0);
    assert.deepEqual(result.missingModules, ['opaque']); assert.match(result.detail, /correct distribution names/);
    const noTorch = await inspectModelEnvironment({ ...candidate, torch: null, missing: ['torch'] }, options(runner));
    assert.equal(noTorch.requirements, undefined); assert.match(noTorch.detail, /Torch build/);
  });

  test('an exact-wheel-review host offers capture and model libraries for an empty Python without executing it', async () => {
    const calls: string[][] = [];
    const result = await inspectModelEnvironment({ ...candidate, torch: null, status: 'missing_libraries',
      missing: ['torch', 'pydantic>=2.12,<3'] }, { ...options(async (_python, _backend, args) => {
      calls.push(args); return { ok: true, missingModules: [{ module: 'torch', requirement: 'torch' },
        { module: 'diffusers', requirement: 'diffusers==0.35.2' }, { module: 'timm', requirement: 'timm>=1.0,<2' },
        { module: 'cv2', requirement: 'opencv-python' }, { module: 'PIL', requirement: 'pillow' }] };
    }), reviewMissingTorch: true });
    assert.equal(result.reason, 'missing_libraries');
    assert.deepEqual(result.requirements, ['torch', 'pydantic>=2.12,<3', 'diffusers==0.35.2', 'timm>=1.0,<2', 'opencv-python', 'pillow']);
    assert.match(result.detail, /Review the exact PyPI wheel versions before installing/);
    assert.match(result.detail, /specific Torch\/CUDA build/);
    assert.deepEqual(calls.map(args => args[2]), ['inspect-model']);
  });

  test('a declared Git dependency explains the unsupported installer instead of asking for another declaration', async () => {
    const calls: string[][] = [];
    const result = await inspectModelEnvironment({ ...candidate, managed: true }, options(async (_python, _backend, args) => {
      calls.push(args);
      return { ok: true, missingModules: [{ module: 'torch_remat', requirement: null },
        { module: 'optional', requirement: null, optional: true }], unsupportedDeclaredModules: ['torch_remat', 'optional'] };
    }));
    assert.equal(result.reason, 'missing_libraries');
    assert.deepEqual(result.requirements, []);
    assert.deepEqual(result.missingModules, ['torch_remat']);
    assert.match(result.detail, /already declares torch_remat/);
    assert.match(result.detail, /guided installer/);
    assert.match(result.detail, /project's installation instructions/);
    assert.doesNotMatch(result.detail, /Add the correct distribution|optional/);
    assert.equal(calls.length, 1, 'a required model dependency must block worker imports and capture');
  });

  test('package conflicts remain distinct from missing packages; only supported model compatibility can offer a copy', async () => {
    const conflict = await inspectModelEnvironment(candidate, options(async () => ({ ok: true,
      incompatible: ['scipy 1.0 (needs >=1.14)'] })));
    assert.equal(conflict.reason, 'incompatible'); assert.equal(conflict.requirements, undefined);
    const copy = await inspectModelEnvironment(candidate, { ...options(async () => ({ ok: true,
      compatibilityPackages: ['transformers==4.51.3'] })), importFailure: true });
    assert.deepEqual(copy.compatibilityPackages, ['transformers==4.51.3']);
    const workerConflict = await inspectModelEnvironment({ ...candidate, incompatible: ['pydantic 1.0 (needs >=2.12)'] },
      { ...options(async () => ({ ok: true, compatibilityPackages: ['transformers==4.51.3'] })), importFailure: true });
    assert.equal(workerConflict.compatibilityPackages, undefined);
  });

  test('a native host offers the project pins behind a conflict; only the exact accepted pins let the check pass', async () => {
    const runner: EnvironmentRunner = async (_python, _backend, args) => args[2] === 'inspect-model'
      ? { ok: true, incompatible: ['torch 2.7.1 (needs ==2.0.0)'], incompatibleRequirements: ['torch==2.0.0'] } : { ok: true };
    const offered = await inspectModelEnvironment(candidate, { ...options(runner), offerPins: true });
    assert.equal(offered.reason, 'incompatible'); assert.deepEqual(offered.declaredPins, ['torch==2.0.0']);
    // Hosts that cannot act on pins (the plugin, MCP) never receive them.
    assert.equal((await inspectModelEnvironment(candidate, options(runner))).declaredPins, undefined);
    const other = await inspectModelEnvironment(candidate, { ...options(runner), offerPins: true, acceptedPins: ['torch==1.13.0'] });
    assert.equal(other.reason, 'incompatible');
    const accepted = await inspectModelEnvironment(candidate, { ...options(runner), offerPins: true, acceptedPins: ['torch==2.0.0'] });
    assert.equal(accepted.reason, 'ready');
    assert.match(accepted.detail, /^Using installed torch 2\.7\.1 \(needs ==2\.0\.0\) instead of the project pins\./);
    // tensorViz's own capture requirements are never optional.
    const capture = await inspectModelEnvironment({ ...candidate, incompatible: ['pydantic 1.0 (needs >=2.12)'] },
      { ...options(runner), offerPins: true, acceptedPins: ['torch==2.0.0'] });
    assert.equal(capture.reason, 'incompatible'); assert.equal(capture.declaredPins, undefined);
    // A failed import with accepted pins brings the choice back.
    const failed = await inspectModelEnvironment(candidate, { ...options(runner), offerPins: true, acceptedPins: ['torch==2.0.0'], importFailure: true });
    assert.equal(failed.reason, 'incompatible'); assert.deepEqual(failed.declaredPins, ['torch==2.0.0']);
  });

  test('readiness imports the exact worker once per environment fingerprint, without claiming model execution', async () => {
    const calls: string[] = [], workerCache = new Set<string>();
    const runner: EnvironmentRunner = async (_python, _backend, args) => { calls.push(args[2]); return { ok: true }; };
    for (const chosen of [candidate, candidate, { ...candidate, fingerprint: 'c'.repeat(64) }]) {
      const result = await inspectModelEnvironment(chosen, { ...options(runner), workerCache });
      assert.equal(result.reason, 'ready'); assert.match(result.detail, /execution still needs/);
    }
    assert.deepEqual(calls, ['inspect-model', 'probe-capture', 'inspect-model', 'inspect-model', 'probe-capture']);
  });

  test('unsupported worker and observed model import failures return recoverable reasons without retrying', async () => {
    const replies: HelperResult[] = [{ ok: true }, { ok: false, message: 'Unsupported CPython ABI.' }];
    const unsupported = await inspectModelEnvironment(candidate, options(async () => replies.shift()!));
    assert.equal(unsupported.reason, 'worker_failed'); assert.equal(unsupported.detail, 'Unsupported CPython ABI.');
    const model = await inspectModelEnvironment(candidate, { ...options(async () => ({ ok: true })), importFailure: true });
    assert.equal(model.reason, 'worker_failed'); assert.match(model.detail, /model failed to import/);
  });

  test('unavailable Python starts no worker; cancellation discards a late library check', async () => {
    let calls = 0;
    const unavailable = await inspectModelEnvironment({ ...candidate, status: 'unavailable' }, options(async () => { calls++; return { ok: true }; }));
    assert.equal(unavailable.reason, 'unavailable'); assert.equal(calls, 0);
    const controller = new AbortController();
    await assert.rejects(inspectModelEnvironment(candidate, { ...options(async () => { controller.abort(); return { ok: true }; }),
      signal: controller.signal }), /abort/i);
  });
});

suite('model check started ahead of the capture result', () => {
  const ok: HelperResult = { ok: true, missingModules: [] };
  const ready = async (runner: EnvironmentRunner) => {
    const runs: string[][] = [];
    const counted: EnvironmentRunner = async (...args) => { runs.push(args[2]); return runner(...args); };
    return { runs, counted };
  };

  test('a started check is used instead of a second identical one', async () => {
    const { runs, counted } = await ready(async () => ok);
    const inspection = startModelInspection(candidate.interpreter, options(counted));
    const result = await inspectModelEnvironment({ ...candidate, missing: ['x'] }, { ...options(counted), inspection });
    assert.equal(result.reason, 'missing_libraries');
    assert.deepEqual(runs, [['--target-python', candidate.interpreter, 'inspect-model', '/project', '/project/pkg/model.py', '[]']]);
  });

  test('it runs the same arguments, including the missing modules, as the check it replaces', async () => {
    const direct: string[][] = [], started: string[][] = [];
    await inspectModelEnvironment(candidate, { ...options(async (_p, _b, args) => { direct.push(args); return ok; }), missingModules: ['cv2'] });
    await startModelInspection(candidate.interpreter, { ...options(async (_p, _b, args) => { started.push(args); return ok; }), missingModules: ['cv2'] }).settled;
    assert.deepEqual(started, [direct[0]]);
    assert.equal(direct[0][2], 'inspect-model');
  });

  test('a check started for another interpreter is ignored', async () => {
    const { runs, counted } = await ready(async () => ok);
    const inspection = startModelInspection('/other/python', options(counted));
    await inspectModelEnvironment({ ...candidate, missing: ['x'] }, { ...options(counted), inspection });
    assert.deepEqual(runs.map(args => args[1]).sort(), ['/other/python', candidate.interpreter].sort());
  });

  test('an unavailable environment never awaits it, and its failure is not an unhandled rejection', async () => {
    let unhandled = 0;
    const count = () => { unhandled++; };
    process.on('unhandledRejection', count);
    try {
      const inspection = startModelInspection(candidate.interpreter, options(async () => { throw new Error('no python'); }));
      const result = await inspectModelEnvironment({ ...candidate, status: 'unavailable' }, { ...options(async () => ok), inspection });
      assert.equal(result.reason, 'unavailable');
      await new Promise(resolve => setTimeout(resolve, 20));
      assert.equal(unhandled, 0);
    } finally { process.off('unhandledRejection', count); }
  });

  test('a failed started check fails the readiness check exactly as before', async () => {
    const boom = new Error('helper crashed');
    const inspection = startModelInspection(candidate.interpreter, options(async () => { throw boom; }));
    await assert.rejects(inspectModelEnvironment(candidate, { ...options(async () => ok), inspection }), error => error === boom);
    await assert.rejects(inspectModelEnvironment(candidate, options(async () => { throw boom; })), error => error === boom);
  });

  test('an aborted opening stops waiting for it', async () => {
    const controller = new AbortController();
    let release!: (value: HelperResult) => void;
    const runner: EnvironmentRunner = () => new Promise(resolve => { release = resolve; });
    const inspection = startModelInspection(candidate.interpreter, { ...options(runner), signal: controller.signal });
    const pending = inspectModelEnvironment(candidate, { ...options(runner), signal: controller.signal, inspection });
    await new Promise(resolve => setImmediate(resolve));
    controller.abort();
    release(ok);
    await assert.rejects(pending, error => (error as Error).name === 'AbortError');
  });
});
