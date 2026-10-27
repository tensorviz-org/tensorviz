import * as assert from 'node:assert/strict';
import { inspectEnvironmentCandidates, type EnvironmentHint, type HelperResult } from '@tensorviz/graph-host/projectEnvironmentCore';

const hints: EnvironmentHint[] = [
  { interpreter: '/project/.venv/bin/python', source: 'Saved', selected: true },
  { interpreter: '/other/bin/python', source: 'Selected in VS Code', pythonSelected: true },
];
const metadata = (prefix: string): HelperResult => ({ ok: true, python: '3.12.13', fingerprint: 'a'.repeat(64),
  environment: { executable: `${prefix}/bin/python`, prefix, python: '3.12.13', platform: 'linux', architecture: 'x86_64',
    implementation: 'cpython', abi: 'cpython-312-x86_64-linux-gnu', torch: '2.6.0+cu124', pip: null } });

suite('bounded environment candidate metadata', () => {
  test('checks the target through the manager without importing a model or requiring target pip', async () => {
    const rows = await inspectEnvironmentCandidates(hints, { backendDir: '/backend', managerPython: '/service/python',
      signal: new AbortController().signal, runner: async (python, backend, args, _cancelled, isolated, parentPipe) => {
        assert.equal(python, '/service/python'); assert.equal(backend, '/backend');
        assert.deepEqual(args, ['--target-python', args[1], 'inspect-capture']);
        assert.equal(isolated, true); assert.equal(parentPipe, true);
        return metadata(args[1].replace('/bin/python', ''));
      } });
    assert.equal(rows.length, 2); assert.equal(rows[0].selected, true); assert.equal(rows[1].pythonSelected, true);
    assert.equal(rows[0].torch, '2.6.0+cu124'); assert.equal(rows[0].status, 'libraries_present');
    assert.equal(rows[0].fingerprint, 'a'.repeat(64));
  });

  test('an opening checks every candidate for the model in the same probe, and records only required gaps', async () => {
    const seen: string[][] = [];
    const rows = await inspectEnvironmentCandidates(hints, { backendDir: '/backend', managerPython: '/service/python',
      signal: new AbortController().signal, model: { workspaceFolder: '/project', file: 'models/net.py', missingModules: ['einops'] },
      runner: async (_python, _backend, args) => {
        seen.push(args);
        const base = metadata(args[1].replace('/bin/python', ''));
        return args[1] === hints[0].interpreter
          ? { ...base, model: { missingModules: [{ module: 'einops', requirement: 'einops>=0.8' }, { module: 'cv2', requirement: null },
            { module: 'wandb', requirement: 'wandb', optional: true }], incompatible: ['transformers 4.57.6 (needs >=5)'] } }
          : { ...base, model: { missingModules: [], incompatible: [] } };
      } });
    assert.deepEqual(seen.map(args => args.slice(2)), hints.map(() => ['inspect-candidate', '/project', '/project/models/net.py', '["einops"]']));
    assert.deepEqual(rows.map(row => [row.modelChecked, row.modelMissing, row.modelIncompatible]), [
      [true, ['einops>=0.8', 'cv2'], ['transformers 4.57.6 (needs >=5)']],
      [true, undefined, undefined]]);
  });

  test('a candidate whose model could not be checked keeps its environment facts without model fields', async () => {
    const [row] = await inspectEnvironmentCandidates([hints[0]], { backendDir: '/backend', signal: new AbortController().signal,
      model: { workspaceFolder: '/project', file: 'net.py' }, runner: async () => metadata('/project/.venv') });
    assert.equal(row.status, 'libraries_present');
    assert.ok(!('modelChecked' in row) && !('modelMissing' in row) && !('modelIncompatible' in row));
  });

  test('one unavailable candidate is isolated and missing libraries are distinct from version conflicts', async () => {
    const rows = await inspectEnvironmentCandidates([...hints, { interpreter: '/conflict/python', source: 'Project' }], {
      backendDir: '/backend', signal: new AbortController().signal, runner: async python => {
        if (python === hints[0].interpreter) throw new Error('PRIVATE_SUBPROCESS_OUTPUT');
        return python === hints[1].interpreter ? { ...metadata('/other'), missing: ['pydantic>=2.12,<3'] }
          : { ...metadata('/conflict'), incompatible: ['pydantic 1.10 (needs >=2.12,<3)'] };
      } });
    assert.deepEqual(rows.map(row => row.status), ['unavailable', 'missing_libraries', 'incompatible']);
    assert.equal(rows[0].selected, true, 'a missing explicit choice remains visible, never replaced by a working suggestion');
    assert.ok(!JSON.stringify(rows).includes('PRIVATE_SUBPROCESS_OUTPUT'));
  });

  test('deduplicates aliases by environment prefix while preserving venvs that share a base executable', async () => {
    const rows = await inspectEnvironmentCandidates([hints[0], { ...hints[0], source: 'Known', selected: undefined },
      { interpreter: '/project/.venv/bin/python3', source: 'VS Code', pythonSelected: true }, hints[1]], {
      backendDir: '/backend', signal: new AbortController().signal,
      runner: async python => metadata(python.includes('/project/') ? '/project/.venv' : '/other'),
    });
    assert.equal(rows.length, 2); assert.equal(rows[0].source, 'Saved');
    assert.equal(rows[0].selected, true); assert.equal(rows[0].pythonSelected, true);
    assert.equal(rows[1].prefix, '/other');
  });

  test('limits candidates to sixteen and permits at most two concurrent probes', async () => {
    let running = 0, peak = 0, calls = 0;
    const rows = await inspectEnvironmentCandidates(Array.from({ length: 30 }, (_, i) => ({ interpreter: `/env${i}/python`, source: 'Known' })), {
      backendDir: '/backend', signal: new AbortController().signal, runner: async python => {
        calls++; running++; peak = Math.max(peak, running);
        await new Promise(resolve => setTimeout(resolve, 1)); running--; return metadata(python);
      } });
    assert.equal(calls, 16); assert.equal(rows.length, 16); assert.equal(peak, 2);
    assert.equal(rows[0].interpreter, '/env0/python'); assert.equal(rows[15].interpreter, '/env15/python');
  });

  test('cancellation refuses a completed old probe and starts no further candidates', async () => {
    const controller = new AbortController(); let calls = 0;
    await assert.rejects(inspectEnvironmentCandidates(hints, { backendDir: '/backend', signal: controller.signal,
      runner: async () => { calls++; controller.abort(); return metadata('/project'); } }), /abort/i);
    assert.equal(calls, 1);
  });
});
