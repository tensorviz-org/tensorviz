import type { CaptureSnapshot } from './semanticCapture';

type Operation = CaptureSnapshot['operations'][number];
type Argument = Operation['output'] | undefined;
export interface ShapeCheck {
  status: 'compatible' | 'incompatible' | 'unknown';
  rule: string | null;
  reason: string;
  /** A single output retains the original field; multiple outputs are bound to exact value identities. */
  outputShape: number[] | null;
  outputShapes?: Array<{ valueId: string; shape: number[] }>;
}
export const concreteShape = (shape: unknown): shape is number[] => Array.isArray(shape) && shape.length <= 32
  && shape.every(d => Number.isSafeInteger(d) && d >= 0);
const unknown = (reason: string): ShapeCheck => ({ status: 'unknown', rule: null, reason, outputShape: null });
const bad = (rule: string, reason: string): ShapeCheck => ({ status: 'incompatible', rule, reason, outputShape: null });
const good = (rule: string, shape: number[]): ShapeCheck => concreteShape(shape)
  ? { status: 'compatible', rule, reason: 'dimensions_satisfy_rule', outputShape: shape }
  : unknown('shape_outside_supported_range');
const integer = (arg: Argument): number | undefined => arg?.kind === 'int' && /^-?\d+$/.test(arg.value)
  && Number.isSafeInteger(Number(arg.value)) ? Number(arg.value) : undefined;
const integers = (arg: Argument): number[] | undefined => {
  if (arg?.kind !== 'list' && arg?.kind !== 'tuple') return undefined;
  const values = arg.items.map(integer);
  return values.length <= 32 && values.every(value => value !== undefined) ? values as number[] : undefined;
};
const count = (shape: number[]) => shape.reduce((n, dim) => n * BigInt(dim), 1n);
const equal = (a: number[], b: number[]) => a.length === b.length && a.every((dim, i) => dim === b[i]);
function broadcast(a: number[], b: number[]): number[] | null {
  const output: number[] = [];
  for (let i = 1; i <= Math.max(a.length, b.length); i++) {
    const left = a[a.length - i] ?? 1, right = b[b.length - i] ?? 1;
    if (left !== right && left !== 1 && right !== 1) return null;
    output.unshift(left === 1 ? right : left);
  }
  return output;
}
const unary = new Set(['aten.relu.default', 'aten.gelu.default', 'aten.sigmoid.default', 'aten.tanh.default',
  'aten.neg.default', 'aten.clone.default', 'aten.detach.default']);
const binary = new Set(['add', 'sub', 'mul', 'div'].flatMap(op => [`aten.${op}.Tensor`, `aten.${op}.Scalar`]));

/** Dimension rules for a closed ATen subset. These are conditional arithmetic,
 * not execution, dtype/layout checks, numerical equivalence or gradient proof. */
export function checkOperationShape(op: Operation, shapeOf: (id: string) => number[] | null): ShapeCheck {
  if (op.effects.length) return unknown('mutation_not_supported');
  if (op.status !== 'completed') return unknown('operation_not_completed');
  const arg = (index: number, name: string): Argument => op.arguments.args[index] ?? op.arguments.kwargs[name];
  const tensor = (value: Argument): number[] | null => {
    if (value?.kind !== 'value') return null;
    const shape = shapeOf(value.id); return concreteShape(shape) ? shape : null;
  };
  const operand = (value: Argument): number[] | null => value?.kind === 'int' || value?.kind === 'float' || value?.kind === 'bool'
    ? [] : tensor(value);
  if (op.target === 'aten.native_layer_norm.default') {
    const output = op.output;
    if (output?.kind !== 'tuple' || output.items.length !== 3 || output.items.some(item => item.kind !== 'value')
      || new Set(output.items.map(item => item.kind === 'value' ? item.id : null)).size !== 3) {
      return unknown('layer_norm_three_tensor_outputs_required');
    }
    const input = tensor(arg(0, 'input')), normalized = integers(arg(1, 'normalized_shape'));
    if (!input || !normalized) return unknown('shape_or_dimensions_unavailable');
    const rule = 'native_layer_norm_v1';
    if (!normalized.length || normalized.some(dim => dim < 0)) return bad(rule, 'invalid_normalized_shape');
    if (input.length < normalized.length || !equal(input.slice(-normalized.length), normalized)) {
      return bad(rule, 'normalized_dimension_mismatch');
    }
    for (const [index, name] of [[2, 'weight'], [3, 'bias']] as const) {
      const value = arg(index, name);
      if (value?.kind === 'none') continue;
      const shape = tensor(value);
      if (!shape) return unknown(`${name}_shape_unavailable`);
      if (!equal(shape, normalized)) return bad(rule, `${name}_dimension_mismatch`);
    }
    const stats = [...input.slice(0, input.length - normalized.length), ...normalized.map(() => 1)];
    return { status: 'compatible', rule, reason: 'dimensions_satisfy_rule', outputShape: null,
      outputShapes: output.items.map((item, i) => ({ valueId: (item as { kind: 'value'; id: string }).id,
        shape: i === 0 ? [...input] : [...stats] })) };
  }
  if (op.output?.kind !== 'value') return unknown('single_tensor_output_required');
  const input = tensor(arg(0, 'self'));
  if (unary.has(op.target)) return input ? good('same_shape_v1', [...input]) : unknown('input_shape_unavailable');
  if (binary.has(op.target)) {
    const left = operand(arg(0, 'self')), right = operand(arg(1, 'other'));
    if (!left || !right) return unknown('operand_shape_unavailable');
    const shape = broadcast(left, right);
    return shape ? good('broadcast_v1', shape) : bad('broadcast_v1', 'incompatible_broadcast_dimensions');
  }
  if (op.target === 'aten.mm.default' || op.target === 'aten.bmm.default' || op.target === 'aten.addmm.default') {
    const add = op.target === 'aten.addmm.default', batched = op.target === 'aten.bmm.default';
    const a = tensor(arg(add ? 1 : 0, add ? 'mat1' : 'self'));
    const b = tensor(arg(add ? 2 : 1, 'mat2'));
    const rule = batched ? 'batched_matrix_v1' : 'matrix_v1', rank = batched ? 3 : 2;
    if (!a || !b) return unknown('operand_shape_unavailable');
    if (a.length !== rank || b.length !== rank) return bad(rule, 'matrix_rank_mismatch');
    if (a[rank - 1] !== b[rank - 2]) return bad(rule, 'contracting_dimension_mismatch');
    if (batched && a[0] !== b[0]) return bad(rule, 'batch_dimension_mismatch');
    const shape = [...a.slice(0, -1), b[rank - 1]];
    if (add) {
      if (!input) return unknown('bias_shape_unavailable');
      const biased = broadcast(input, shape);
      if (!biased || !equal(biased, shape)) return bad(rule, 'bias_dimension_mismatch');
    }
    return good(rule, shape);
  }
  if (op.target === 'aten.linear.default') {
    const a = tensor(arg(0, 'input')), weight = tensor(arg(1, 'weight')), bias = arg(2, 'bias');
    if (!a || !weight) return unknown('operand_shape_unavailable');
    if (!a.length || weight.length !== 2 || a[a.length - 1] !== weight[1]) return bad('linear_v1', 'feature_dimension_mismatch');
    const shape = [...a.slice(0, -1), weight[0]];
    if (bias && bias.kind !== 'none') {
      const b = tensor(bias); if (!b) return unknown('bias_shape_unavailable');
      const biased = broadcast(b, shape);
      if (!biased || !equal(biased, shape)) return bad('linear_v1', 'bias_dimension_mismatch');
    }
    return good('linear_v1', shape);
  }
  if (['aten.view.default', 'aten.reshape.default', 'aten._unsafe_view.default'].includes(op.target)) {
    const dims = integers(arg(1, op.target === 'aten.reshape.default' ? 'shape' : 'size'));
    if (!input || !dims) return unknown('shape_or_dimensions_unavailable');
    const inferred = dims.filter(d => d === -1).length;
    if (inferred > 1 || dims.some(d => d < -1)) return bad('reshape_count_v1', 'invalid_inferred_dimension');
    const known = count(dims.filter(d => d !== -1)), total = count(input);
    if (inferred) {
      if (known === 0n || total % known !== 0n) return bad('reshape_count_v1', known === 0n ? 'ambiguous_inferred_dimension' : 'element_count_mismatch');
      const value = total / known;
      if (value > BigInt(Number.MAX_SAFE_INTEGER)) return unknown('shape_outside_supported_range');
      return good('reshape_count_v1', dims.map(d => d === -1 ? Number(value) : d));
    }
    return known === total ? good('reshape_count_v1', dims) : bad('reshape_count_v1', 'element_count_mismatch');
  }
  if (op.target === 'aten.permute.default') {
    const dims = integers(arg(1, 'dims'));
    if (!input || !dims) return unknown('shape_or_dimensions_unavailable');
    const normalized = dims.map(d => d < 0 ? input.length + d : d);
    if (normalized.length !== input.length || new Set(normalized).size !== input.length
      || normalized.some(d => d < 0 || d >= input.length)) return bad('permutation_v1', 'invalid_permutation');
    return good('permutation_v1', normalized.map(d => input[d]));
  }
  if (op.target === 'aten.transpose.int' || op.target === 'aten.t.default') {
    if (!input) return unknown('input_shape_unavailable');
    if (op.target === 'aten.t.default') return input.length <= 2
      ? good('transpose_v1', [...input].reverse()) : bad('transpose_v1', 'transpose_rank_mismatch');
    const a = integer(arg(1, 'dim0')), b = integer(arg(2, 'dim1')), rank = Math.max(1, input.length);
    if (a === undefined || b === undefined) return unknown('dimensions_unavailable');
    const left = a < 0 ? rank + a : a, right = b < 0 ? rank + b : b;
    if (left < 0 || left >= rank || right < 0 || right >= rank) return bad('transpose_v1', 'dimension_out_of_range');
    const shape = [...input]; if (shape.length) [shape[left], shape[right]] = [shape[right], shape[left]];
    return good('transpose_v1', shape);
  }
  if (op.target === 'aten.cat.default') {
    const tensors = arg(0, 'tensors'), dimension = arg(1, 'dim');
    const dim = dimension === undefined ? 0 : integer(dimension);
    if ((tensors?.kind !== 'list' && tensors?.kind !== 'tuple') || dim === undefined) return unknown('tensor_list_or_dimension_unavailable');
    const all = tensors.items.map(tensor);
    if (all.some(shape => shape === null)) return unknown('operand_shape_unavailable');
    const nonempty = (all as number[][]).filter(shape => !equal(shape, [0]));
    // ATen ignores dim when every operand is the special empty 1-D tensor.
    if (all.length && !nonempty.length) return good('concatenate_v1', [0]);
    const shapes = nonempty.length ? nonempty : all as number[][], first = shapes[0];
    if (!first?.length) return bad('concatenate_v1', 'nonempty_nonscalar_list_required');
    const axis = dim < 0 ? first.length + dim : dim;
    if (axis < 0 || axis >= first.length) return bad('concatenate_v1', 'dimension_out_of_range');
    if (shapes.some(shape => shape.length !== first.length || shape.some((d, i) => i !== axis && d !== first[i]))) {
      return bad('concatenate_v1', 'nonconcatenated_dimension_mismatch');
    }
    const result = [...first]; result[axis] = shapes.reduce((sum, shape) => sum + shape[axis], 0);
    return good('concatenate_v1', result);
  }
  return unknown('operator_not_supported');
}
