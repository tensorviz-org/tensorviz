import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { executionPlanFile, type ExecutionSnapshot, type PreparationSnapshot } from '../../executionPlan';
import { workerArgs } from '../../worker';
import { executionSpecSchema } from '@tensorviz/graph-host/executionSpec';

suite('resolved execution private transport', () => {
  test('saved recipes and diagnostic arguments never enter the resolved worker command', () => {
    const execution = executionSpecSchema.parse({ inputs: { args: [] }, constructorArgs: { label: 'private_recipe' } });
    const args = workerArgs({ file: '/workspace/model.py', cls: 'Model', workspace: '/workspace', execution }, 'plan');
    assert.ok(args.includes('--resolved-execution'));
    assert.ok(!args.includes('--execution') && !args.includes('--diagnostic-job'));
    assert.doesNotMatch(args.join(' '), /private_recipe/);
    assert.throws(() => workerArgs({ file: '/workspace/model.py', cls: 'Model', workspace: '/workspace', execution,
      job: { kind: 'training' } }, 'plan'));
  });
  test('automatic input expressions and constructor values use only the private preparation', async () => {
    const args = workerArgs({ file: '/workspace/model.py', cls: 'Model', workspace: '/workspace',
      input: 'private_input()', overrides: '{"label":"private_override"}' }, 'preparation');
    assert.ok(args.includes('--capture-preparation') && args.includes('--workspace'));
    assert.doesNotMatch(args.join(' '), /private_input|private_override|--input|--overrides/);
    const snapshot = { preparation: { id: 'auto' }, applicabilityDigest: 'b'.repeat(64) } as PreparationSnapshot;
    const result = await executionPlanFile(snapshot);
    try {
      assert.deepEqual(JSON.parse(await fs.readFile(result.file, 'utf8')),
        { version: 1, preparation: snapshot.preparation, applicabilityDigest: snapshot.applicabilityDigest });
    } finally { await result.dispose(); }
    await assert.rejects(fs.stat(result.file), { code: 'ENOENT' });
  });
  test('plan files have private permissions and explicit idempotent cleanup', async () => {
    const snapshot = { plan: { version: 1, id: 'plan' }, applicabilityDigest: 'a'.repeat(64), guards: {} } as ExecutionSnapshot;
    const result = await executionPlanFile(snapshot);
    const directory = path.dirname(result.file);
    try {
      assert.equal(path.dirname(directory), await fs.realpath(os.tmpdir()));
      if (process.platform !== 'win32') {
        assert.equal((await fs.stat(directory)).mode & 0o777, 0o700);
        assert.equal((await fs.stat(result.file)).mode & 0o777, 0o600);
      }
      assert.deepEqual(JSON.parse(await fs.readFile(result.file, 'utf8')),
        { version: 1, plan: snapshot.plan, applicabilityDigest: snapshot.applicabilityDigest });
    } finally { await result.dispose(); }
    await assert.rejects(fs.stat(directory), { code: 'ENOENT' });
    await result.dispose();
  });
  test('oversized identities refuse rather than truncating a bootstrap', async () => {
    const snapshot = { plan: { value: 'x'.repeat(16 * 1024 * 1024) } } as unknown as ExecutionSnapshot;
    await assert.rejects(executionPlanFile(snapshot), /limit/);
  });
});
