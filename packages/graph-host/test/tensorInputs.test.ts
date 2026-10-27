import * as assert from 'node:assert/strict';
import { tensorInputsSchema } from '@tensorviz/graph-host/tensorInputs';
import { workerArgs } from '@tensorviz/graph-host/traceWorkerCore';

suite('structured runtime inputs', () => {
  const input = { shape: [1, 16], dtype: 'int64' as const, fill: 'ones' as const };
  test('carries integer tensors as JSON, never as a Python expression', () => {
    const args = workerArgs({ file: '/ws/model.py', cls: 'Net', tensorInputs: [input] });
    assert.deepEqual(JSON.parse(args[args.indexOf('--tensor-inputs') + 1]), [input]);
    assert.ok(!args.includes('--input'));
    assert.throws(() => workerArgs({ file: 'x.py', input: 'torch.randn(2)', tensorInputs: [input] }));
    assert.throws(() => workerArgs({ file: 'x.py', kind: 'data', tensorInputs: [input] }));
  });
  test('rejects code, unsupported fields, dtypes, fills and unbounded allocations', () => {
    for (const value of [[], 'torch.ones(2)', [{ ...input, shape: [0] }], [{ ...input, shape: [1.5] }],
      [{ ...input, dtype: 'torch.int64' }], [{ ...input, device: 'cuda' }], [{ ...input, fill: 'randn' }],
      [{ ...input, shape: [1000, 5000] }], Array(9).fill(input), [{ ...input, shape: Array(9).fill(1) }]]) {
      assert.equal(tensorInputsSchema.safeParse(value).success, false, JSON.stringify(value));
    }
    assert.equal(tensorInputsSchema.safeParse([{ ...input, shape: [] }]).success, true);
  });
});
