import * as assert from 'node:assert/strict';
import { executionSpecSchema, executionJobSchema, trainingExecutionSpecSchema } from '@tensorviz/graph-host/executionSpec';
import { workerArgs } from '@tensorviz/graph-host/traceWorkerCore';

suite('explicit execution requests', () => {
  const tensor = { shape: [2, 8], dtype: 'float32', fill: 'ones' };
  test('accepts keyword-only inputs and records explicit defaults', () => {
    const execution = executionSpecSchema.parse({ inputs: { kwargs: { x: tensor } } });
    assert.equal(execution.device, 'cpu'); assert.equal(execution.mode, 'eval');
    assert.equal(execution.seed, 0); assert.equal(execution.deterministic, true);
    assert.equal(execution.gradEnabled, false); assert.equal(execution.cwd, '.');
    const args = workerArgs({ file: '/ws/model.py', cls: 'Net', workspace: '/ws', execution });
    assert.deepEqual(JSON.parse(args[args.indexOf('--execution') + 1]), execution);
    assert.equal(args[args.indexOf('--workspace') + 1], '/ws');
  });
  test('supports actual fixtures, model factories and checkpoint selection', () => {
    const value = executionSpecSchema.parse({ inputs: { fixture: 'data/batch.pt' },
      setup: { file: 'recipes/model.py', callable: 'build', kwargs: { width: 512 } },
      checkpoint: { file: 'weights.pt', key: 'state_dict' }, device: 'cuda:0', mode: 'preserve' });
    assert.equal(value.checkpoint!.strict, true);
    assert.equal(value.setup!.kwargs.width, 512);
  });
  test('rejects traversal, ambiguous configuration and unsafe allocation', () => {
    for (const value of [
      { inputs: { fixture: '../secret.pt' } }, { inputs: { fixture: '.env' } },
      { inputs: { fixture: 'C:\\secret.pt' } }, { inputs: { fixture: 'a.pt', args: [] } },
      { inputs: { args: [tensor] }, device: 'auto' },
      { inputs: { args: [tensor] }, cwd: '../escape' },
      { inputs: { args: [tensor] }, gradEnabled: 1 },
      { inputs: { args: [{ ...tensor, shape: [1e6, 1e6] }] } },
      { inputs: { args: [{ ...tensor, dtype: 'int64', fill: 'randn' }] } },
      { inputs: { args: Array(33).fill(tensor) } },
      { inputs: { args: [] }, constructorArgs: {}, setup: { file: 'm.py', callable: 'make' } },
      { inputs: { args: [] }, constructorBinding: { args: [], kwargs: {} }, setup: { file: 'm.py', callable: 'make' } },
    ]) assert.equal(executionSpecSchema.safeParse(value).success, false, JSON.stringify(value));
  });
  test('cannot mix explicit setup with legacy input, fake tier or a missing workspace', () => {
    const execution = executionSpecSchema.parse({ inputs: { args: [tensor] } });
    for (const extra of [{}, { workspace: '/ws', input: '2,8' }, { workspace: '/ws', tier: 'fake' as const },
      { workspace: '/ws', kind: 'train_step' as const }]) {
      assert.throws(() => workerArgs({ file: '/ws/model.py', cls: 'Net', execution, ...extra }));
    }
  });
  test('profiling and training-step jobs are rejected', () => {
    assert.equal(trainingExecutionSpecSchema.parse({ inputs: { args: [tensor] } }).mode, 'train');
    for (const value of [{ kind: 'training' }, { kind: 'profile' }, { kind: 'training', loss: { file: 'loss.py', callable: 'compute' } }]) {
      assert.equal(executionJobSchema.safeParse(value).success, false, JSON.stringify(value));
    }
    assert.equal(executionSpecSchema.safeParse({ inputs: { args: [] }, training: { kind: 'training' } }).success, false);
  });

});
