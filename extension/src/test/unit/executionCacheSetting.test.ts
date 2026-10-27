import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as path from 'node:path';
import { executionCacheSettings } from '../../worker';
import { reset, state } from '../vscodeStub';

const manifest = JSON.parse(readFileSync(path.join(__dirname, '../../../package.json'), 'utf8'));
const properties = manifest.contributes.configuration.properties;

suite('tensorviz.executionCache settings', () => {
  setup(() => reset());

  test('declared: mode off|shadow|on (default off) and maxSizeMB (default 96)', () => {
    const mode = properties['tensorviz.executionCache.mode'];
    assert.deepEqual(mode.enum, ['off', 'shadow', 'on']);
    assert.equal(mode.default, 'off');
    assert.equal(mode.enumDescriptions.length, 3);
    assert.match(mode.enumDescriptions[2], /no Python ran/);
    assert.ok(mode.markdownDescription.includes('Clear Execution Cache'));
    assert.ok(mode.markdownDescription.includes('never'));
    const size = properties['tensorviz.executionCache.maxSizeMB'];
    assert.equal(size.type, 'number');
    assert.equal(size.default, 96);
  });

  test('read per run: default off, shadow or on when set, anything else off', () => {
    assert.deepEqual(executionCacheSettings(), { mode: 'off', maxSizeMB: 96 });
    state.configuration.set('tensorviz.executionCache.mode', 'shadow');
    state.configuration.set('tensorviz.executionCache.maxSizeMB', 12);
    assert.deepEqual(executionCacheSettings(), { mode: 'shadow', maxSizeMB: 12 });
    state.configuration.set('tensorviz.executionCache.mode', 'on');
    assert.equal(executionCacheSettings().mode, 'on');
    state.configuration.set('tensorviz.executionCache.mode', 'always');
    assert.equal(executionCacheSettings().mode, 'off');
  });

  test('tensorViz: Clear Execution Cache is contributed', () => {
    const command = manifest.contributes.commands.find((entry: { command: string }) => entry.command === 'tensorviz.clearExecutionCache');
    assert.deepEqual(command, { command: 'tensorviz.clearExecutionCache', title: 'Clear Execution Cache', category: 'tensorViz' });
  });
});
