import { z } from 'zod';
import type { CaptureSnapshot } from './semanticCapture';
import { GraphToolError } from './mcpGraph';
import { ComputationIndex, computationQuerySchema, pageComputation } from './computationQuery';
import { checkOperationShape, concreteShape, type ShapeCheck } from './computationShapes';

export const computationChangeSchema = computationQuerySchema.omit({ anchors: true, direction: true }).extend({
  changes: z.array(z.object({ valueId: z.string().min(1).max(128),
    shape: z.array(z.number().int().min(0).max(Number.MAX_SAFE_INTEGER)).max(32) }).strict()).min(1).max(16),
}).strict();
const bytes = (value: unknown) => Buffer.byteLength(JSON.stringify(value));
const same = (a: number[], b: number[]) => a.length === b.length && a.every((dim, i) => dim === b[i]);
const unknown = (reason: string): ShapeCheck => ({ status: 'unknown', rule: null, reason, outputShape: null });

/** Changes are user-supplied assumptions about exact values, never source edits
 * or observations. The original capture is immutable throughout this analysis. */
export function analyzeComputationChange(capture: CaptureSnapshot, requested: unknown, prepared?: ComputationIndex) {
  const parsed = computationChangeSchema.safeParse(requested);
  if (!parsed.success) throw new GraphToolError('INVALID_QUERY', 'Use bounded exact value identities and concrete nonnegative shapes.');
  const { changes, ...bounds } = parsed.data;
  const assumed = new Map(changes.map(change => [change.valueId, change.shape]));
  if (assumed.size !== changes.length) throw new GraphToolError('INVALID_QUERY', 'Use distinct changed values.');
  const index = prepared ?? new ComputationIndex(capture);
  index.requireCapture(capture);
  const { query, page: selection } = index.select({ ...bounds, anchors: changes.map(change => ({ kind: 'value', id: change.valueId })),
    direction: 'downstream' });
  const values = new Map(capture.values.map(value => [value.id, value]));
  const selected = selection.items.flatMap(item => item.kind === 'entity' ? [item.ref] : []);
  const operationIds = new Set(selected.filter(ref => ref.kind === 'operation').map(ref => ref.id));
  const shapes = new Map<string, number[] | null>(selected.filter(ref => ref.kind === 'value').map(ref => [ref.id, null]));
  const produced = new Map<string, string[]>();
  for (const value of capture.values) if (value.producer && operationIds.has(value.producer.operationId)) {
    shapes.set(value.id, null);
    const ids = produced.get(value.producer.operationId) ?? [];
    ids.push(value.id); produced.set(value.producer.operationId, ids);
  }
  for (const [id, shape] of assumed) shapes.set(id, [...shape]);
  const changed = changes.map(change => values.get(change.valueId)!);
  const possibleAliases = capture.values.filter(value => !assumed.has(value.id) && changed.some(source =>
    source.objectId === value.objectId || source.aliasOf === value.id || value.aliasOf === source.id
    || source.storageId !== null && source.storageId === value.storageId));
  const aliasIds = new Set(possibleAliases.map(value => value.id));
  const shapeOf = (id: string): number[] | null => {
    if (shapes.has(id)) return shapes.get(id)!;
    if (aliasIds.has(id)) return null;
    const shape = values.get(id)?.shape;
    return concreteShape(shape) ? [...shape] : null;
  };
  const checks = new Map<string, ShapeCheck>();
  const operations = capture.operations.filter(op => operationIds.has(op.id)).sort((a, b) => a.sequence - b.sequence);
  for (const op of operations) {
    let check = capture.backend === 'runtime' ? checkOperationShape(op, shapeOf) : unknown('runtime_aten_capture_required');
    const outputs = new Map(check.outputShapes?.map(output => [output.valueId, output.shape]) ?? []);
    if (op.output?.kind === 'value' && check.outputShape) outputs.set(op.output.id, check.outputShape);
    if ([...outputs].some(([id, shape]) => assumed.has(id) && !same(assumed.get(id)!, shape))) {
      check = { status: 'incompatible', rule: check.rule, reason: 'assumed_output_conflicts_with_derived_shape', outputShape: null };
      outputs.clear();
    }
    // Clear every producer output, including explicit seeds, after a failed or unknown rule.
    for (const id of produced.get(op.id) ?? []) shapes.set(id, outputs.get(id) ?? null);
    checks.set(op.id, check);
  }
  const page = pageComputation(selection, query.offset, query.limit);
  const summary = { compatible: 0, incompatible: 0, unknown: 0 };
  for (const check of checks.values()) summary[check.status]++;
  const analysis = {
    version: 1, kind: 'hypothetical_shape_change' as const, scope: 'selected_recorded_path' as const,
    changes, summary, checks: [] as Array<ShapeCheck & { operationId: string }>,
    values: [] as Array<{ id: string; basis: 'assumed' | 'derived' | 'unknown'; derivedShape: number[] | null }>,
    possibleAliases: possibleAliases.slice(0, 16).map(value => value.id), omittedAliases: Math.max(0, possibleAliases.length - 16),
    omittedPageEntries: 0,
    assumptions: ['Recorded operation arguments and control flow stay fixed.',
      'Unmodified operands outside the slice retain their captured shapes unless aliasing is unresolved.',
      'Dimension rules assume ordinary dense tensors; strides, dtype, device and custom dispatch are not verified.'],
    limitations: ['compatible means only that the named dimension rule holds under these assumptions.',
      'Alias effects outside the selected tensor paths, unobserved branches and source/config changes are not resolved here.',
      'No execution, numerical, gradient, checkpoint or candidate-validity claim is made.'],
  };
  let size = bytes(analysis) + 128;
  if (size > 16 * 1024) throw new GraphToolError('RESULT_TOO_LARGE', 'The change assumptions exceed the analysis metadata budget.');
  for (const ref of page.items.flatMap(item => item.kind === 'entity' ? [item.ref] : [])) {
    if (ref.kind === 'operation') {
      const entry = { operationId: ref.id, ...checks.get(ref.id)! };
      if (size + bytes(entry) + 1 > 16 * 1024) { analysis.omittedPageEntries++; continue; }
      analysis.checks.push(entry); size += bytes(entry) + 1;
    } else if (ref.kind === 'value') {
      const shape = shapes.get(ref.id) ?? null;
      const entry = { id: ref.id, basis: (shape ? assumed.has(ref.id) ? 'assumed' : 'derived' : 'unknown') as 'assumed' | 'derived' | 'unknown',
        derivedShape: shape };
      if (size + bytes(entry) + 1 > 16 * 1024) { analysis.omittedPageEntries++; continue; }
      analysis.values.push(entry); size += bytes(entry) + 1;
    }
  }
  return { ...page, analysis };
}
