import { flattenedShape, sourceShapeCallValid } from '@tensorviz/contracts/sourceShapeCall';
import type { CaptureSnapshot } from './semanticCapture';
import type { SourceContextPage } from './sourceContext';
import type { sourceConfigurationImpact } from './sourceConfigurationImpact';
import type { sourceShapeImpact } from './sourceShapeImpact';
import { checkOperationShape, concreteShape, type ShapeCheck } from './computationShapes';

type Operation = CaptureSnapshot['operations'][number];
type Link = { anchorId: string; bindingId: string; ref: { kind: string; id: string } };
type Configuration = ReturnType<typeof sourceConfigurationImpact>;
type Local = ReturnType<typeof sourceShapeImpact>;
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const unknown = (reason: string): ShapeCheck => ({ status: 'unknown', rule: null, reason, outputShape: null });
const dimensions = (shape: number[]) => ({ kind: 'list' as const, items: shape.map(value => ({ kind: 'int' as const, value: String(value) })) });
const outputs = (op: Operation, check: ShapeCheck) => new Map(check.outputShapes?.map(item => [item.valueId, item.shape])
  ?? (op.output?.kind === 'value' && check.outputShape ? [[op.output.id, check.outputShape]] : []));

/** Combine only explicit substitutions on the recorded tape. This remains a
 * conditional dimension calculation, not a prediction of the edited program. */
export function sourceComputationImpact(page: SourceContextPage, capture: CaptureSnapshot, links: Link[],
  configuration: Configuration, local: Local, omittedMatches: number) {
  const result = {
    version: 1, evidence: 'conditional_recorded_path_dimensions' as const,
    scope: 'recorded_capture_with_source_page_substitutions' as const,
    status: 'analyzed' as 'analyzed' | 'unknown', reason: 'bounded_recorded_path',
    coverage: { sourcePageComplete: page.items.length === page.total && page.nextOffset === null,
      sourceBoundaries: page.traversal.boundaries, omittedSourceSeeds: page.traversal.omittedSeeds,
      sourceDiagnostics: page.diagnostics.length + page.omittedDiagnostics,
      captureStatus: capture.status, captureScope: capture.coverage.scope, captureOmissions: [...capture.coverage.omissions],
      omittedMatches, omittedConstructorEffects: configuration.omittedEffects, omittedLocalEffects: local.omittedEffects },
    stateChanges: [] as Array<{ valueId: string; shape: number[] | null; reason: string }>, omittedStateChanges: 0,
    checks: [] as Array<ShapeCheck & { operationId: string; substitution: string }>, omittedChecks: 0,
    summary: { compatible: 0, incompatible: 0, unknown: 0 },
    outputs: [] as Array<{ valueId: string; shape: number[] | null }>, omittedOutputs: 0,
    forwardCompatibility: 'not_established', checkpointCompatibility: 'not_established',
    assumptions: ['Only the displayed source-page substitutions change; other recorded arguments and control flow stay fixed.',
      'Standard modules keep their constructor semantics and forward behavior, with no later attribute/state replacement, hooks or rewiring.',
      'Recorded tensor and parameter identities still correspond; dense tensor dimension rules apply.'],
    limitations: ['Derived shapes are conditional calculations, never observed candidate execution or a complete source-edit verdict.',
      'Source page boundaries, unresolved dependencies and unobserved paths can hide additional effects.',
      'No dtype, stride, device, checkpoint, numeric, gradient or current-runtime claim. Use exact candidate verification.'],
  };
  if (capture.backend !== 'runtime' || capture.operations.length > 512 || capture.values.length > 4096
    || capture.invocations.length > 1024 || capture.modules.length > 1024) {
    return { ...result, status: 'unknown' as const,
      reason: capture.backend !== 'runtime' ? 'runtime_aten_capture_required' : 'recorded_path_analysis_limit' };
  }
  if (omittedMatches || configuration.omittedEffects || local.omittedEffects) {
    return { ...result, status: 'unknown' as const, reason: 'source_correspondence_truncated' };
  }
  if (!configuration.effects.length && !local.effects.length) {
    return { ...result, status: 'unknown' as const, reason: 'no_mapped_source_effects' };
  }
  const values = new Map(capture.values.map(value => [value.id, value]));
  const bindings = new Map(capture.sourceBindings.map(binding => [binding.id, binding]));
  const operations = new Map(capture.operations.map(op => [op.id, op]));
  const invocations = new Map(capture.invocations.map(call => [call.id, call]));
  const records = new Map(page.items.flatMap(item => 'id' in item ? [[item.id, item] as const] : []));
  const liveFiles = new Map(page.files.map(file => [file.file, file.digest]));
  const capturedFiles = new Map(capture.source.files.map(file => [file.path, file.digest]));
  const currentBinding = (id: string) => {
    const binding = bindings.get(id), anchor = binding?.anchors[0];
    return binding?.snapshotId === capture.source.id && binding.quality === 'exact' && binding.anchors.length === 1
      && !!anchor?.tokenDigest && liveFiles.get(anchor.file) === anchor.digest && capturedFiles.get(anchor.file) === anchor.digest;
  };
  const shapeOfBaseline = (id: string) => concreteShape(values.get(id)?.shape) ? [...values.get(id)!.shape as number[]] : null;
  const seeds = new Map<string, { shape: number[] | null; reason: string }>();
  const unresolvedModules = new Set<string>();
  for (const effect of configuration.effects) {
    if (effect.status !== 'derived') unresolvedModules.add(effect.moduleId);
    for (const state of effect.stateShapes) for (const id of state.valueIds) {
      const value = values.get(id), prior = seeds.get(id);
      const reason = state.status !== 'derived' || state.omittedValues || state.omittedObjects || value?.producer
        ? 'constructor_state_unresolved' : prior && !prior.shape ? prior.reason : prior && !same(prior.shape, state.after)
          ? 'conflicting_state_substitutions' : 'standard_constructor_dimensions';
      seeds.set(id, { shape: reason === 'standard_constructor_dimensions' ? [...state.after!] : null, reason });
    }
  }
  for (const module of capture.modules) if (unresolvedModules.has(module.id)) {
    for (const value of capture.values) if (value.state.some(state => module.paths.some(path => !path || state.name.startsWith(`${path}.`)))) {
      seeds.set(value.id, { shape: null, reason: 'constructor_state_unresolved' });
    }
  }
  const unresolvedCalls = new Set<string>();
  for (const invocation of capture.invocations) {
    let call: typeof invocation | undefined = invocation;
    const ancestors = new Set<string>();
    while (call && !ancestors.has(call.id)) {
      if (unresolvedModules.has(call.moduleId)) { unresolvedCalls.add(invocation.id); break; }
      ancestors.add(call.id); call = call.parentId ? invocations.get(call.parentId) : undefined;
    }
  }
  const shapes = new Map(capture.values.filter(value => value.producer).map(value => [value.id, null as number[] | null]));
  for (const [id, seed] of seeds) if (!values.get(id)?.producer) shapes.set(id, seed.shape);
  const changed = [...seeds.keys()].map(id => values.get(id)!);
  for (const value of capture.values) if (!value.producer && !seeds.has(value.id) && changed.some(source => source.objectId === value.objectId
    || source.aliasOf === value.id || value.aliasOf === source.id || source.storageId !== null && source.storageId === value.storageId)) {
    shapes.set(value.id, null);
    seeds.set(value.id, { shape: null, reason: 'unresolved_state_alias' });
  }
  const shapeOf = (id: string) => shapes.has(id) ? shapes.get(id)! : shapeOfBaseline(id);
  const produced = new Map<string, string[]>();
  for (const value of capture.values) if (value.producer) {
    const ids = produced.get(value.producer.operationId) ?? [];
    ids.push(value.id); produced.set(value.producer.operationId, ids);
  }

  const substitute = (op: Operation): { operation?: Operation; failure?: string; basis: string } => {
    const moduleId = op.invocationId ? invocations.get(op.invocationId)?.moduleId : null;
    if (op.invocationId && unresolvedCalls.has(op.invocationId)) return { failure: 'constructor_effect_unresolved', basis: 'unresolved' };
    if (op.sourceShapeCall) {
      const effects = local.effects.filter(effect => effect.operationId === op.id);
      const call = op.sourceShapeCall;
      const sourceDimensions = effects.length === 1 && effects[0].status === 'derived' ? effects[0].dimensions
        : !effects.length && call.dimensions.every(dimension => dimension.tensorDimension)
          && currentBinding(call.bindingId) && sourceShapeCallValid(op, bindings, values, operations)
          ? call.dimensions.map(dimension => ({ tensorDimension: dimension.tensorDimension, assumedAfter: null })) : null;
      if (!sourceDimensions) return { failure: 'source_dimensions_unresolved', basis: 'reshape' };
      const dims = sourceDimensions.map(dimension => {
        const read = dimension.tensorDimension;
        if (!read) return dimension.assumedAfter;
        const shape = shapeOf(read.valueId), axis = read.axis >= 0 ? read.axis : (shape?.length ?? 0) + read.axis;
        return shape && axis >= 0 && axis < shape.length ? shape[axis] : null;
      });
      if (dims.some(dim => dim === null)) return { failure: 'shape_read_dependency_unknown', basis: 'reshape' };
      return { operation: { ...op, arguments: { args: [op.arguments.args[0], dimensions(dims as number[])], kwargs: {} } }, basis: 'source_and_tensor_dimensions' };
    }
    if (op.sourceFlattenCall) {
      const call = op.sourceFlattenCall;
      const matches = links.filter(link => link.ref.kind === 'operation' && link.ref.id === op.id && link.bindingId === call.bindingId);
      const change = matches.length === 1 ? records.get(matches[0].anchorId)?.configurationChange : undefined;
      if (!sourceShapeCallValid(op, bindings, values, operations) || change?.kind !== 'call'
        || change.hasExpansion || change.omittedArguments || change.omittedArgumentValues || !change.argumentValues) {
        return { failure: 'flatten_argument_correspondence_unavailable', basis: 'flatten' };
      }
      let start = call.startDim, end = call.endDim;
      for (const argument of call.arguments) {
        const value = change.argumentValues.find(item => same(item.path, argument.sourcePath));
        if (value?.status !== 'derived' || value.before !== argument.value) return { failure: 'flatten_argument_unknown', basis: 'flatten' };
        if (argument.parameter === 'start_dim') start = value.after; else end = value.after;
      }
      const input = op.arguments.args[0], shape = input.kind === 'value' ? shapeOf(input.id) : null;
      if (!shape) return { failure: 'flatten_input_shape_unknown', basis: 'flatten' };
      const flattened = flattenedShape(shape, start, end);
      if (!flattened) return { failure: 'flatten_dimensions_unsupported_or_invalid', basis: 'flatten' };
      return { operation: { ...op, arguments: { args: [input, dimensions(flattened)], kwargs: {} } }, basis: 'source_flatten_dimensions' };
    }
    if (['aten.view.default', 'aten.reshape.default', 'aten._unsafe_view.default'].includes(op.target)) {
      return { failure: 'shape_argument_semantics_unavailable', basis: 'unresolved' };
    }
    const effects = configuration.effects.filter(effect => effect.moduleId === moduleId);
    if (op.target === 'aten.native_layer_norm.default' && effects.length) {
      const effect = effects[0], argument = effect.arguments.find(arg => arg.parameter === 'normalized_shape');
      const before = argument?.observedBefore, after = argument?.assumedAfter;
      const observed = op.arguments.args[1] ?? op.arguments.kwargs.normalized_shape;
      if (effects.length !== 1 || effect.className !== 'torch.nn.modules.normalization.LayerNorm' || before == null || after == null
        || !same(observed, dimensions([before]))) return { failure: 'layer_norm_argument_correspondence_unavailable', basis: 'layer_norm' };
      for (const [index, name] of [[2, 'weight'], [3, 'bias']] as const) {
        const value = op.arguments.args[index] ?? op.arguments.kwargs[name];
        const state = effect.stateShapes.find(state => state.parameter === name);
        if (value?.kind !== 'none' && (value?.kind !== 'value' || state?.status !== 'derived' || !state.valueIds.includes(value.id))) {
          return { failure: 'layer_norm_state_correspondence_unavailable', basis: 'layer_norm' };
        }
      }
      const args = [...op.arguments.args], kwargs = { ...op.arguments.kwargs };
      if (args.length > 1) args[1] = dimensions([after]); else kwargs.normalized_shape = dimensions([after]);
      return { operation: { ...op, arguments: { args, kwargs } }, basis: 'standard_layer_norm_constructor' };
    }
    if (['aten.permute.default', 'aten.transpose.int', 'aten.cat.default'].includes(op.target)
      && links.some(link => link.ref.kind === 'operation' && link.ref.id === op.id)) {
      return { failure: 'source_argument_effect_unsupported', basis: 'unresolved' };
    }
    return { operation: op, basis: 'recorded_arguments' };
  };

  for (const op of [...capture.operations].sort((a, b) => a.sequence - b.sequence)) {
    const baseline = checkOperationShape(op, shapeOfBaseline), baseOutputs = outputs(op, baseline);
    const substitution = substitute(op);
    const baselineMatches = baseline.status === 'compatible' && baseOutputs.size > 0
      && [...baseOutputs].every(([id, shape]) => same(shapeOfBaseline(id), shape));
    const check = substitution.failure ? unknown(substitution.failure)
      : !baselineMatches ? unknown(baseline.status === 'unknown' ? baseline.reason : 'recorded_dimension_rule_mismatch')
        : checkOperationShape(substitution.operation!, shapeOf);
    const derived = outputs(op, check);
    for (const id of produced.get(op.id) ?? []) shapes.set(id, derived.get(id) ?? null);
    result.summary[check.status]++;
    if (result.checks.length < 32) result.checks.push({ operationId: op.id, substitution: substitution.basis, ...check });
    else result.omittedChecks++;
  }
  for (const [valueId, { shape, reason }] of seeds) {
    if (result.stateChanges.length < 32) result.stateChanges.push({ valueId, shape, reason }); else result.omittedStateChanges++;
  }
  const visit = (argument: Operation['output']): void => {
    if (argument?.kind === 'value') {
      if (result.outputs.length < 16) result.outputs.push({ valueId: argument.id, shape: shapeOf(argument.id) }); else result.omittedOutputs++;
    } else if (argument?.kind === 'list' || argument?.kind === 'tuple') argument.items.forEach(visit);
    else if (argument?.kind === 'dict') argument.entries.forEach(entry => visit(entry.value));
  };
  visit(capture.output);
  while (Buffer.byteLength(JSON.stringify(result)) > 16 * 1024) {
    if (result.checks.length) { result.checks.pop(); result.omittedChecks++; }
    else if (result.stateChanges.length) { result.stateChanges.pop(); result.omittedStateChanges++; }
    else { result.outputs.pop(); result.omittedOutputs++; }
  }
  return result;
}
