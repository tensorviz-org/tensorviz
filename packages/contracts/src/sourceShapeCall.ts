/** Optional observed Python-to-ATen argument correspondence. This checks stored
 * evidence, never infers a mapping from a source line or matching tensor shapes. */
import type { components } from './semanticContracts.generated';

type Operation = components['schemas']['Operation'];
type Binding = components['schemas']['SourceBinding'];
type Value = components['schemas']['Value'];

/** The dimension rule alone; this neither executes flatten nor proves layout. */
export function flattenedShape(shape: unknown, start: number, end: number): number[] | null {
  if (!Array.isArray(shape) || shape.length > 32 || shape.some(d => !Number.isSafeInteger(d) || d < 0)
    || !Number.isInteger(start) || !Number.isInteger(end) || start < -32 || start > 31 || end < -32 || end > 31) return null;
  const rank = Math.max(1, shape.length), first = start < 0 ? rank + start : start, last = end < 0 ? rank + end : end;
  if (first < 0 || last >= rank || first > last) return null;
  if (!shape.length) return [1];
  const product = shape.slice(first, last + 1).reduce((n, d) => n * BigInt(d), BigInt(1));
  return product > BigInt(Number.MAX_SAFE_INTEGER) ? null : [...shape.slice(0, first), Number(product), ...shape.slice(last + 1)];
}

function flattenCallValid(op: Operation, bindings: ReadonlyMap<string, Binding>, values: ReadonlyMap<string, Value>): boolean {
  const call = op.sourceFlattenCall!, binding = bindings.get(call.bindingId), args = op.arguments.args;
  if (op.sourceShapeCall != null || op.status !== 'completed' || op.target !== 'aten.view.default' || op.effects.length
    || args.length !== 2 || Object.keys(op.arguments.kwargs).length || args[0].kind !== 'value'
    || (args[1].kind !== 'list' && args[1].kind !== 'tuple') || op.output?.kind !== 'value'
    || op.sourceBindingIds.length !== 1 || op.sourceBindingIds[0] !== call.bindingId
    || binding?.kind !== 'call' || binding.quality !== 'exact' || call.arguments.length > 2) return false;
  const supplied = new Map(call.arguments.map(item => [item.parameter, item]));
  if (supplied.size !== call.arguments.length) return false;
  const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
  const offset = call.callable === 'torch.flatten' ? 1 : call.callable === 'torch.Tensor.flatten' ? 0 : -1;
  if (offset === -1) return false;
  const parameters = [['start_dim', call.startDim, 0], ['end_dim', call.endDim, -1]] as const;
  for (const [index, [name, value, defaultValue]] of parameters.entries()) {
    const item = supplied.get(name);
    if (item ? item.value !== value || (!same(item.sourcePath, ['args', index + offset]) && !same(item.sourcePath, ['kwargs', name]))
      : value !== defaultValue) return false;
  }
  if (supplied.get('end_dim')?.sourcePath[0] === 'args' && !same(supplied.get('start_dim')?.sourcePath, ['args', offset])) return false;
  const dimensions = flattenedShape(values.get(args[0].id)?.shape, call.startDim, call.endDim);
  return dimensions !== null && same(values.get(op.output.id)?.shape, dimensions)
    && same(args[1].items, dimensions.map(value => ({ kind: 'int', value: String(value) })));
}

export function sourceShapeCallValid(op: Operation, bindings: ReadonlyMap<string, Binding>,
  values: ReadonlyMap<string, Value>, operations: ReadonlyMap<string, Operation>): boolean {
  if (op.sourceFlattenCall != null) return flattenCallValid(op, bindings, values);
  const call = op.sourceShapeCall;
  if (call == null) return true;
  const binding = bindings.get(call.bindingId), shape = op.arguments.args[1];
  if (op.status !== 'completed' || op.target !== 'aten.view.default' || op.effects.length
    || op.arguments.args.length !== 2 || Object.keys(op.arguments.kwargs).length
    || op.arguments.args[0].kind !== 'value' || op.output?.kind !== 'value'
    || (shape?.kind !== 'list' && shape?.kind !== 'tuple')
    || call.dimensions.length < 1 || call.dimensions.length > 32 || call.dimensions.length !== shape.items.length
    || op.sourceBindingIds.length !== 1 || op.sourceBindingIds[0] !== call.bindingId
    || binding?.kind !== 'call' || binding.quality !== 'exact') return false;
  const paths = call.dimensions.map(item => item.sourcePath), prefix = paths[0].slice(0, 2);
  const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
  const expected = call.callable === 'torch.reshape'
    ? (same(prefix, ['args', 1]) || same(prefix, ['kwargs', 'shape']))
      && same(paths, paths.map((_, i) => [...prefix, 'items', i]))
    : (call.callable === 'torch.Tensor.reshape' || call.callable === 'torch.Tensor.view')
      && (same(paths, paths.map((_, i) => ['args', i])) || same(paths, paths.map((_, i) => ['args', 0, 'items', i])));
  return !!expected && call.dimensions.every((item, i) => {
    if (item.dimension !== i || item.value.kind !== 'int' || !Number.isSafeInteger(Number(item.value.value))
      || shape.items[i].kind !== 'int' || item.value.value !== (shape.items[i] as typeof item.value).value) return false;
    const read = item.tensorDimension;
    if (read == null) return true;
    const fact = values.get(read.valueId), axis = read.axis >= 0 ? read.axis : (fact?.shape?.length ?? 0) + read.axis;
    if (!Number.isInteger(read.axis) || read.axis < -32 || read.axis > 31 || !fact?.shape
      || axis < 0 || axis >= fact.shape.length || typeof fact.shape[axis] !== 'number'
      || fact.shape[axis] !== Number(item.value.value)) return false;
    return !fact.producer || (operations.get(fact.producer.operationId)?.sequence ?? Infinity) < op.sequence;
  });
}
