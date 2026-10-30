import { initializerSourceArgument } from '@tensorviz/contracts/initializerArguments';
import type { CaptureSnapshot } from './semanticCapture';
import type { SourceContextPage } from './sourceContext';

type Module = CaptureSnapshot['modules'][number];
type Link = { anchorId: string; bindingId: string; ref: { kind: string; id: string } };
type Record = Extract<SourceContextPage['items'][number], { id: string }>;
const bytes = (value: unknown) => Buffer.byteLength(JSON.stringify(value));
const sameShape = (actual: unknown, expected: number[]) => Array.isArray(actual)
  && actual.length === expected.length && actual.every((dimension, i) => dimension === expected[i]);
const integer = (value: { kind: string; value?: unknown } | null | undefined): number | null => {
  if (value?.kind !== 'int' || typeof value.value !== 'string') return null;
  const number = Number(value.value);
  return Number.isSafeInteger(number) ? number : null;
};

/** Expected standard-constructor dimensions, conditional on the observed class
 * and unchanged arguments. Never a prediction of final state after rewiring. */
function constructorEffect(record: Record, bindingId: string, module: Module, capture: CaptureSnapshot) {
  const change = record.configurationChange;
  if (change?.kind !== 'call') throw new Error('A constructor effect requires call arguments.');
  const className = capture.definitions.find(item => item.id === module.definitionId)?.qualifiedName ?? null;
  const base = { anchorId: record.id, bindingId, moduleId: module.id, className,
    status: 'unknown' as 'unknown' | 'derived', reason: '', arguments: [] as Array<{
      path: Array<string | number>; parameter: string | null; sourceBefore: number | null; assumedAfter: number | null; observedBefore: number | null;
    }>, stateShapes: [] as Array<{
      parameter: string; status: 'unknown' | 'derived'; reason: string; before: number[] | null; after: number[] | null;
      names: string[]; omittedNames: number; valueIds: string[]; omittedValues: number; objectIds: string[]; omittedObjects: number;
    }> };
  const unknown = (reason: string) => ({ ...base, reason });
  if (!module.constructorClassVerified || module.constructorBindingIds?.length !== 1 || module.constructorBindingIds[0] !== bindingId) {
    return unknown('constructor_identity_unverified');
  }
  if (change.hasExpansion || change.omittedArguments) return unknown('expanded_or_omitted_arguments');
  if (!change.arguments.length) return unknown('no_scalar_argument_effects');
  const entry = module.initializerArguments;
  if (entry?.status !== 'observed') return unknown('initializer_arguments_unavailable');
  const original = new Map([...(entry.positional ?? []), ...(entry.keywordOnly ?? [])].map(item => [item.name, integer(item.value)]));
  const proposed = new Map(original);
  for (const argument of change.arguments) {
    const [group, slot] = argument.path;
    const parameter = group === 'args' && typeof slot === 'number' ? entry.positional?.[slot]?.name
      : group === 'kwargs' && typeof slot === 'string' ? slot : undefined;
    const observedBefore = integer(initializerSourceArgument(entry, argument.path));
    base.arguments.push({ path: [...argument.path], parameter: parameter ?? null, sourceBefore: argument.before,
      assumedAfter: argument.after, observedBefore });
    if (argument.status !== 'derived') return unknown('source_argument_unknown');
    if (parameter === undefined || observedBefore === null) return unknown('initializer_arguments_unavailable');
    if (observedBefore !== argument.before) return unknown('observed_source_value_mismatch');
    proposed.set(parameter, argument.after);
  }
  const rule = className === 'torch.nn.modules.linear.Linear' ? ['in_features', 'out_features']
    : className === 'torch.nn.modules.normalization.LayerNorm' ? ['normalized_shape'] : null;
  if (!rule) return unknown('unsupported_module');
  if (base.arguments.some(argument => !rule.includes(argument.parameter!))) return unknown('unsupported_initializer_argument');
  if (rule.some(name => original.get(name) == null || proposed.get(name) == null || original.get(name)! < 0 || proposed.get(name)! < 0)) {
    return unknown('invalid_or_unavailable_dimension');
  }
  const dimension = (values: Map<string, number | null>, name: string) => values.get(name)!;
  const shapes = (values: Map<string, number | null>) => rule.length === 2
    ? { weight: [dimension(values, 'out_features'), dimension(values, 'in_features')], bias: [dimension(values, 'out_features')] }
    : { weight: [dimension(values, 'normalized_shape')], bias: [dimension(values, 'normalized_shape')] };
  const before = shapes(original), after = shapes(proposed);
  for (const parameter of ['weight', 'bias'] as const) {
    const names = new Set(module.paths.map(path => path ? `${path}.${parameter}` : parameter));
    const values = capture.values.filter(value => value.state.some(state => state.kind === 'parameter' && names.has(state.name)));
    const observedNames = [...new Set(values.flatMap(value => value.state.filter(state => state.kind === 'parameter').map(state => state.name)))].sort();
    const objectIds = [...new Set(values.map(value => value.objectId))];
    const reason = !values.length ? 'state_not_observed' : objectIds.length !== 1 ? 'state_identity_ambiguous'
      : values.some(value => !sameShape(value.shape, before[parameter])) ? 'recorded_state_shape_mismatch' : 'standard_constructor_dimension_rule';
    const derived = reason === 'standard_constructor_dimension_rule';
    base.stateShapes.push({ parameter, status: derived ? 'derived' : 'unknown', reason,
      before: derived ? before[parameter] : null, after: derived ? after[parameter] : null,
      names: observedNames.slice(0, 16), omittedNames: Math.max(0, observedNames.length - 16),
      valueIds: values.slice(0, 16).map(value => value.id), omittedValues: Math.max(0, values.length - 16),
      objectIds: objectIds.slice(0, 16), omittedObjects: Math.max(0, objectIds.length - 16) });
  }
  return { ...base, status: 'derived' as const, reason: 'supported_constructor_arguments' };
}

export function sourceConfigurationImpact(page: SourceContextPage, capture: CaptureSnapshot, links: Link[]) {
  const calls = page.items.filter((item): item is Record => 'id' in item && item.configurationChange?.kind === 'call');
  const result = { version: 1, evidence: 'conditional_constructor_state_dimensions' as const,
    scope: 'linked_constructor_calls_on_this_page' as const, effects: [] as ReturnType<typeof constructorEffect>[],
    unlinkedCalls: 0, omittedEffects: 0, forwardCompatibility: 'not_established', checkpointCompatibility: 'not_established',
    assumptions: ['The recorded standard module class and constructor semantics stay fixed.',
      'Unchanged initializer arguments retain their observed values; the changed source expression reaches this call.',
      'Expected dimensions concern standard constructor state; later replacement, mutation and sharing are not predicted.'],
    limitations: ['Baseline state associations describe retained values only; missing state means unobserved, not absent.',
      'These partial state dimensions must not be treated as a complete forward-change analysis with unchanged operation arguments.',
      'No candidate execution, checkpoint loading, numerical equivalence or current-runtime applicability is established.'] };
  let size = bytes(result) + 128;
  const modules = new Map(capture.modules.map(module => [module.id, module]));
  for (const call of calls) {
    const matches = links.filter(link => link.anchorId === call.id && link.ref.kind === 'module');
    if (!matches.length || capture.backend !== 'runtime') { result.unlinkedCalls++; continue; }
    for (const match of matches) {
      const module = modules.get(match.ref.id);
      if (!module) { result.omittedEffects++; continue; }
      const effect = constructorEffect(call, match.bindingId, module, capture), amount = bytes(effect) + 1;
      if (result.effects.length >= 16 || size + amount > 12 * 1024) { result.omittedEffects++; continue; }
      result.effects.push(effect); size += amount;
    }
  }
  return result;
}
