import * as assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { compareOutputs } from '@tensorviz/graph-host/runNumerics';

function tensor(dtype: string, buffer: Buffer, shape: number[], options: Record<string, unknown> = {}): any {
  const digest = createHash('sha256').update(dtype).update(JSON.stringify(shape)).update(buffer).digest('hex');
  return { digest, allFinite: true, truncated: [], containers: [], leaves: [{ path: 'output', kind: 'tensor', dtype, shape,
    numel: shape.reduce((n, dim) => n * dim, 1), digest, nonfinite: 0,
    values: { encoding: 'base64', byteOrder: 'little', data: buffer.toString('base64') }, ...options }] };
}
function f64(values: number[]): any {
  const buffer = Buffer.alloc(values.length * 8); values.forEach((value, index) => buffer.writeDoubleLE(value, index * 8));
  return tensor('float64', buffer, [values.length]);
}
const result = (a: any, b: any, atol = 0, rtol = 0) => compareOutputs(a, b, atol, rtol) as any;

suite('retained numerical output comparison', () => {
  test('reports tolerances and actual mismatches against the baseline', () => {
    assert.equal(result(f64([1, 2]), f64([1.000001, 2]), 1e-5).allClose, true);
    const changed = result(f64([1, 2]), f64([1, 3]));
    assert.equal(changed.allClose, false); assert.equal(changed.items[0].maxAbsDifference, 1);
    assert.deepEqual(changed.items[0].examples, [{ flatIndex: 1, baseline: 2, candidate: 3 }]);
    assert.equal(changed.toleranceReference, 'baseline');
  });
  test('does not round large integer differences away', () => {
    const a = Buffer.alloc(8), b = Buffer.alloc(8);
    a.writeBigInt64LE(2n ** 60n + 7n); b.writeBigInt64LE(2n ** 60n + 8n);
    const changed = result(tensor('int64', a, [1]), tensor('int64', b, [1]), 100, 100);
    assert.equal(changed.allClose, false); assert.equal(changed.items[0].maxAbsDifference, '1');
    assert.equal(changed.items[0].examples[0].baseline, (2n ** 60n + 7n).toString());
    assert.equal(result(tensor('int64', a, [1]), f64([Number(2n ** 60n)])).items[0].status, 'integer_precision_unavailable');
  });
  test('half, bfloat and big-endian values decode correctly', () => {
    const half = Buffer.alloc(2), bfloat = Buffer.alloc(2), big = Buffer.alloc(8);
    half.writeUInt16LE(0x3c00); bfloat.writeUInt16LE(0x3f80); big.writeDoubleBE(1);
    assert.equal(result(tensor('float16', half, [1]), f64([1])).allClose, true);
    assert.equal(result(tensor('bfloat16', bfloat, [1]), f64([1])).allClose, true);
    const encoded = tensor('float64', big, [1]); encoded.leaves[0].values.byteOrder = 'big';
    assert.equal(result(encoded, f64([1])).allClose, true);
  });
  test('complex error uses magnitude, and shape changes remain visible', () => {
    const a = Buffer.alloc(8), b = Buffer.alloc(8);
    a.writeFloatLE(1); a.writeFloatLE(2, 4); b.writeFloatLE(1); b.writeFloatLE(3, 4);
    const changed = result(tensor('complex64', a, [1]), tensor('complex64', b, [1]));
    assert.equal(changed.items[0].maxAbsDifference, 1); assert.equal(changed.allClose, false);
    assert.equal(result(f64([1, 2]), f64([1])).items[0].status, 'shape_changed');
  });
  test('NaNs and unavailable values cannot prove equivalence', () => {
    const nan = f64([NaN]); nan.leaves[0].nonfinite = 1;
    assert.equal(result(nan, nan).allClose, false);
    const a = f64([1]), b = f64([2]); a.leaves[0].values = null; b.leaves[0].values = null;
    assert.equal(result(a, b).allClose, null);
    assert.equal(result(a, a).allClose, true, 'a full matching finite fingerprint still establishes exact equality');
    assert.equal(result(undefined, a).allClose, null);
  });
  test('truncated structure cannot become a false difference or a false pass', () => {
    const partial = f64([1]); partial.truncated = ['output_structure_limit']; partial.leaves = [];
    assert.equal(result(partial, f64([1])).allClose, null);
    assert.equal(result(partial, partial).allClose, null);
  });
  test('finite float64 overflow does not create an accidental tolerance pass', () => {
    assert.equal(result(f64([1e308]), f64([-1e308]), 0, 1.9).allClose, false);
    assert.equal(result(f64([1e308]), f64([-1e308]), 0, 2.1).allClose, true);
    assert.equal(result(f64([-0]), f64([0])).allClose, true);
  });
});
