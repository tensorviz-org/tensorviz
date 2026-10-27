import * as assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import * as path from 'node:path';
import { resourceDevices, executionResources } from '../../executionResources';
suite('execution host observations', () => {
  test('stopping readings before a missing tool reports never signals the whole process group', async () => {
    // Its own process group: a kill(0, …) regression ends only this child, not the test run.
    const script = `const { executionResources } = require(${JSON.stringify(path.join(__dirname, '..', '..', 'executionResources.js'))});
const controller = new AbortController();
executionResources('/missing/tensorviz-python', false, controller.signal).catch(() => {});
controller.abort();
setTimeout(() => process.stdout.write('alive'), 200);`;
    const child = spawn(process.execPath, ['-e', script], { detached: true, stdio: ['ignore', 'pipe', 'inherit'] });
    let stdout = '';
    child.stdout.on('data', chunk => { stdout += chunk; });
    const [code, signal] = await new Promise<[number | null, NodeJS.Signals | null]>(resolve => child.on('close', (c, s) => resolve([c, s])));
    assert.equal(signal, null); assert.equal(code, 0); assert.equal(stdout, 'alive');
  });
  test('CUDA ordinals join by UUID rather than assuming physical GPU index', () => {
    const devices = resourceDevices(JSON.stringify([{ id: 'cuda:0', name: 'Second physical GPU', uuid: 'GPU-b' }, { id: 'cuda:1', name: 'First', uuid: 'GPU-a' }]),
      'GPU-a, 7, 100, 200\nGPU-b, 100, 0, 8000');
    assert.equal(devices[0].utilization, 100); assert.equal(devices[0].memoryFree, 0);
    assert.equal(devices[1].utilization, 7);
  });
  test('unavailable/malformed metrics stay unknown, never idle or free', () => {
    const devices = resourceDevices('[{"id":"cuda:0","name":"GPU","uuid":"GPU-a"}]', 'GPU-a, [N/A], N/A, unknown');
    assert.equal(devices[0].utilization, null); assert.equal(devices[0].memoryFree, null);
    assert.deepEqual(resourceDevices('not json', ''), []);
  });
  test('MIG or unknown UUIDs never inherit a parent GPU reading', () => {
    const devices = resourceDevices('[{"id":"cuda:0","name":"MIG instance","uuid":"GPU-instance"}]', 'GPU-parent, 100, 0, 80000');
    assert.equal(devices[0].utilization, null); assert.equal(devices[0].memoryTotal, null);
  });
  test('host observations work without project Python or NVIDIA tools', async () => {
    const resources = await executionResources('', true, new AbortController().signal);
    assert.ok(resources.host); assert.ok(resources.memoryTotal > 0); assert.equal(resources.remote, true);
    assert.ok(resources.sampledAt <= Date.now());
  });
});
