import { sourceShapeCallValid } from '@tensorviz/contracts/sourceShapeCall';
import type { CaptureSnapshot } from './semanticCapture';
import type { SourceContextPage } from './sourceContext';
import { checkOperationShape, concreteShape, type ShapeCheck } from './computationShapes';

type Link = { anchorId: string; bindingId: string; ref: { kind: string; id: string } };
type Record = Extract<SourceContextPage['items'][number], { id: string }>;
const bytes = (value: unknown) => Buffer.byteLength(JSON.stringify(value));

/** A local dimension rule with captured input shapes held fixed. Other source
 * effects are not silently mixed into a claim about the edited forward. */
function operationEffect(record: Record, bindingId: string, operation: CaptureSnapshot['operations'][number],
  capture: CaptureSnapshot, bindings: Map<string, CaptureSnapshot['sourceBindings'][number]>,
  values: Map<string, CaptureSnapshot['values'][number]>, operations: Map<string, CaptureSnapshot['operations'][number]>) {
  const change = record.configurationChange;
  if (change?.kind !== 'call') throw new Error('An operation effect requires a source call.');
  const base = { anchorId: record.id, bindingId, operationId: operation.id, target: operation.target,
    status: 'unknown' as 'unknown' | 'derived', reason: '', dimensions: [] as Array<{
      path: Array<string | number>; dimension: number; sourceBefore: number | null; assumedAfter: number | null; observedBefore: number;
      basis: 'source_integer_expression' | 'captured_tensor_dimension'; tensorDimension?: { valueId: string; axis: number };
    }>, localCheck: null as ShapeCheck | null };
  const unknown = (reason: string) => ({ ...base, reason });
  const call = operation.sourceShapeCall;
  if (capture.backend !== 'runtime' || !call || call.bindingId !== bindingId || !sourceShapeCallValid(operation, bindings, values, operations)) {
    return unknown('operation_argument_correspondence_unavailable');
  }
  if (change.hasExpansion || change.omittedArguments || change.omittedArgumentValues) return unknown('expanded_or_omitted_arguments');
  if (!change.argumentValues) return unknown('source_argument_values_unavailable');
  const argumentsByPath = new Map(change.argumentValues.map(item => [JSON.stringify(item.path), item]));
  if (argumentsByPath.size !== change.argumentValues.length) return unknown('ambiguous_source_argument_values');
  const shapeOf = (id: string) => {
    const shape = values.get(id)?.shape;
    return concreteShape(shape) ? [...shape] : null;
  };
  const dimensions = [];
  for (const observed of call.dimensions) {
    const value = argumentsByPath.get(JSON.stringify(observed.sourcePath)), before = Number(observed.value.value);
    const read = observed.tensorDimension, readShape = read ? shapeOf(read.valueId) : null;
    const readAxis = read && readShape ? read.axis >= 0 ? read.axis : readShape.length + read.axis : -1;
    const fromTensor = !!read && value?.status === 'unknown' && value.reason === 'unsupported_expression'
      && readShape !== null && readAxis >= 0 && readAxis < readShape.length;
    const sourceBefore = fromTensor ? before : value?.before ?? null;
    const after = fromTensor ? readShape![readAxis] : value?.after ?? null;
    base.dimensions.push({ path: [...observed.sourcePath], dimension: observed.dimension,
      sourceBefore, assumedAfter: after, observedBefore: before,
      basis: fromTensor ? 'captured_tensor_dimension' : 'source_integer_expression',
      ...(fromTensor ? { tensorDimension: { ...read! } } : {}) });
    if (!fromTensor && (!value || value.status !== 'derived' || read)) return unknown('source_dimension_unknown');
    if (sourceBefore !== before) return unknown('observed_source_value_mismatch');
    dimensions.push({ kind: 'int' as const, value: String(after) });
  }
  const arguments_ = { args: [operation.arguments.args[0], { kind: 'list' as const, items: dimensions }], kwargs: {} };
  const baseline = checkOperationShape(operation, shapeOf), output = operation.output;
  const observedOutput = output?.kind === 'value' ? shapeOf(output.id) : null;
  if (baseline.status !== 'compatible' || !observedOutput || JSON.stringify(baseline.outputShape) !== JSON.stringify(observedOutput)) {
    return unknown('recorded_dimension_rule_mismatch');
  }
  return { ...base, status: 'derived' as const, reason: 'observed_source_dimensions',
    localCheck: checkOperationShape({ ...operation, arguments: arguments_ }, shapeOf) };
}

export function sourceShapeImpact(page: SourceContextPage, capture: CaptureSnapshot, links: Link[]) {
  const calls = page.items.filter((item): item is Record => 'id' in item && item.configurationChange?.kind === 'call');
  const result = { version: 1, evidence: 'conditional_operation_dimensions' as const,
    scope: 'linked_calls_on_this_page_with_fixed_input_shapes' as const, effects: [] as ReturnType<typeof operationEffect>[],
    unlinkedCalls: 0, omittedEffects: 0, forwardCompatibility: 'not_established', checkpointCompatibility: 'not_established',
    assumptions: ['The same recorded callable and operation execute with their captured operand and referenced shape-read tensor shapes.',
      'Only the fully mapped source dimensions are substituted using the selected integer assumption.'],
    limitations: ['A localCheck concerns only this operation with fixed input shapes, not the complete configuration edit.',
      'Upstream constructor, state, data, other argument changes and unobserved paths are not propagated here.',
      'Only observed direct tensor shape reads can resolve dynamic dimensions; no execution, stride, dtype, numerical or checkpoint claim is made.'] };
  const bindings = new Map(capture.sourceBindings.map(item => [item.id, item]));
  const operations = new Map(capture.operations.map(item => [item.id, item]));
  const values = new Map(capture.values.map(value => [value.id, value]));
  let size = bytes(result) + 128;
  for (const call of calls) {
    const matches = links.filter(link => link.anchorId === call.id && link.ref.kind === 'operation');
    if (!matches.length) { result.unlinkedCalls++; continue; }
    for (const match of matches) {
      const operation = operations.get(match.ref.id);
      if (!operation) { result.omittedEffects++; continue; }
      const effect = operationEffect(call, match.bindingId, operation, capture, bindings, values, operations), amount = bytes(effect) + 1;
      if (result.effects.length >= 16 || size + amount > 12 * 1024) { result.omittedEffects++; continue; }
      result.effects.push(effect); size += amount;
    }
  }
  return result;
}
