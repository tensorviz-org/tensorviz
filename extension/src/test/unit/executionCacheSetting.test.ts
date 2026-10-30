import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as path from 'node:path';
import * as vscode from 'vscode';
import type { ExecutionCacheSettings } from '@tensorviz/graph-host/executionCache';
import { executionCacheSettings, TraceWorker } from '../../worker';
import { reset, state } from '../vscodeStub';

const manifest = JSON.parse(readFileSync(path.join(__dirname, '../../../package.json'), 'utf8'));
const properties = manifest.contributes.configuration.properties;

suite('tensorviz.executionCache settings', () => {
  setup(() => reset());

  test('declared: mode off|shadow|on (default on) and maxSizeMB (default 96)', () => {
    const mode = properties['tensorviz.executionCache.mode'];
    assert.deepEqual(mode.enum, ['off', 'shadow', 'on']);
    assert.equal(mode.default, 'on');
    assert.equal(mode.enumDescriptions.length, 3);
    // Serving still runs Python to resolve and recheck the plan; only the model is not run.
    assert.match(mode.enumDescriptions[2], /model not run again/);
    assert.doesNotMatch(`${mode.enumDescriptions[2]} ${mode.markdownDescription}`, /no Python ran|without running Python/);
    assert.ok(mode.markdownDescription.includes('Clear Execution Cache'));
    assert.ok(mode.markdownDescription.includes('never'));
    const size = properties['tensorviz.executionCache.maxSizeMB'];
    assert.equal(size.type, 'number');
    assert.equal(size.default, 96);
  });

  test('read per run: default on, off or shadow when set, anything else off', () => {
    assert.deepEqual(executionCacheSettings(), { mode: 'on', maxSizeMB: 96 });
    state.configuration.set('tensorviz.executionCache.mode', 'off');
    assert.equal(executionCacheSettings().mode, 'off');
    state.configuration.set('tensorviz.executionCache.mode', 'shadow');
    state.configuration.set('tensorviz.executionCache.maxSizeMB', 12);
    assert.deepEqual(executionCacheSettings(), { mode: 'shadow', maxSizeMB: 12 });
    state.configuration.set('tensorviz.executionCache.mode', 'on');
    assert.equal(executionCacheSettings().mode, 'on');
    state.configuration.set('tensorviz.executionCache.mode', 'always');
    assert.equal(executionCacheSettings().mode, 'off');
  });

  test('read for the run\'s file, so a folder of a multi-root workspace can turn the cache off', () => {
    // The setting is machine-overridable: the Settings UI offers it per folder, and only a
    // read scoped to the file merges that folder's value.
    assert.equal(properties['tensorviz.executionCache.mode'].scope, 'machine-overridable');
    const workspace = vscode.workspace as unknown as { getConfiguration: unknown };
    const original = workspace.getConfiguration;
    const scopes: unknown[] = [];
    workspace.getConfiguration = (_section: string, scope?: { fsPath: string }) => {
      scopes.push(scope);
      return { get: <T>(key: string, fallback: T): T =>
        (key === 'executionCache.mode' && scope?.fsPath?.startsWith('/projects/data-loader/') ? 'off' : fallback) as T };
    };
    try {
      assert.equal(executionCacheSettings(vscode.Uri.file('/projects/model-zoo/net.py') as vscode.Uri).mode, 'on');
      assert.equal(executionCacheSettings(vscode.Uri.file('/projects/data-loader/net.py') as vscode.Uri).mode, 'off');
      // The worker hands the per-run read the run's own file.
      const worker = new TraceWorker('/backend', { appendLine() {} } as unknown as vscode.OutputChannel);
      const host = (worker as unknown as { options: { executionCache(resource?: string): ExecutionCacheSettings | undefined } }).options;
      assert.equal(host.executionCache('/projects/data-loader/net.py')?.mode, 'off');
      assert.equal(host.executionCache('/projects/model-zoo/net.py')?.mode, 'on');
      // With no file it reads the window level, unscoped: that value owns the one shared store.
      scopes.length = 0;
      assert.equal(host.executionCache()?.mode, 'on');
      assert.deepEqual(scopes, [undefined]);
    } finally { workspace.getConfiguration = original; }
  });

  test('tensorViz: Clear Execution Cache is contributed', () => {
    const command = manifest.contributes.commands.find((entry: { command: string }) => entry.command === 'tensorviz.clearExecutionCache');
    assert.deepEqual(command, { command: 'tensorviz.clearExecutionCache', title: 'Clear Execution Cache', category: 'tensorViz' });
  });
});
