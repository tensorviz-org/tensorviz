/** Offline structural/relational gate for the Python-owned capture contract.
 * Imported evidence never authorizes execution, source navigation or editing. */
import { isPortableCapture } from '../types/portableValidators.generated';
import type { components } from '../types/semanticContracts.generated';
import { initializerArgumentsValid } from '../ide/initializerArguments.generated';

export type PortableCapture = components['schemas']['PortableCapture'];
export type CaptureSnapshot = PortableCapture['capture'];
export type Argument = NonNullable<CaptureSnapshot['output']>;
type Arguments = CaptureSnapshot['inputs'];
const structure = isPortableCapture as (value: unknown) => value is PortableCapture;
const LOCAL_PATH = /^(?:\/|[a-z]:[\\/]|\\\\|file:|vscode:)/i;
const PRIVATE_NAME = /^(?:_.*|source|sourceCode|source_code|code|file|file_path|path|location|token|session|sessionId|credential|secret|api[_-]?key|authorization|password)$/i;

export function requireCaptureFact(condition: unknown): asserts condition {
  if (!condition) throw new Error('The saved capture contains inconsistent or unsupported evidence.');
}
export function sameJson(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object' || Array.isArray(a) !== Array.isArray(b)) return false;
  const left = a as Record<string, unknown>, right = b as Record<string, unknown>;
  const keys = Object.keys(left);
  return keys.length === Object.keys(right).length && keys.every(key => Object.hasOwn(right, key) && sameJson(left[key], right[key]));
}
const unique = (items: unknown[]) => new Set(items).size === items.length;
function index<T extends { id: string }>(items: T[]): Map<string, T> {
  const map = new Map(items.map(item => [item.id, item]));
  requireCaptureFact(map.size === items.length);
  return map;
}
export function argumentLeaves(tree: Argument | null, path: Array<string | number> = []): Array<{ path: Array<string | number>; argument: Argument }> {
  if (tree === null) return [];
  if ((tree.kind === 'list' || tree.kind === 'tuple') && tree.items.length) return tree.items.flatMap((item, i) => argumentLeaves(item, [...path, 'items', i]));
  if (tree.kind === 'dict' && tree.entries.length) return tree.entries.flatMap((entry, i) => argumentLeaves(entry.value, [...path, 'entries', i, 'value']));
  return [{ path, argument: tree }];
}
export const inputLeaves = (inputs: Arguments, field: string) => [
  ...inputs.args.flatMap((argument, i) => argumentLeaves(argument, [field, 'args', i])),
  ...Object.entries(inputs.kwargs).flatMap(([key, argument]) => argumentLeaves(argument, [field, 'kwargs', key])),
];

function validateRelations(capture: CaptureSnapshot): void {
  const check: typeof requireCaptureFact = requireCaptureFact;
  check((capture.planDigest === null) !== (capture.preparationDigest === null));
  check((capture.preparationDigest !== null) === (capture.automatic !== null));
  if (capture.automatic) check(capture.automatic.candidateIndex <= capture.automatic.candidatesTried);
  const source = capture.source;
  check(source.sourceMode === 'saved' && unique(source.files.map(file => file.path)));
  const files = new Map(source.files.map(file => [file.path, file]));
  check(files.get(source.modelFile)?.role === 'source');
  const configs = source.files.filter(file => file.role === 'config');
  check(source.configDigest === null ? !configs.length : configs.length === 1 && configs[0].path === 'tensorviz.toml' && configs[0].digest === source.configDigest);
  const definitions = index(capture.definitions), modules = index(capture.modules), calls = index(capture.invocations);
  const ops = index(capture.operations), values = index(capture.values), bindings = index(capture.sourceBindings), diagnostics = index(capture.diagnostics);
  for (const binding of capture.sourceBindings) {
    check(binding.snapshotId === source.id && (binding.quality !== 'exact' || binding.anchors.length === 1));
    for (const anchor of binding.anchors) check(files.get(anchor.file)?.digest === anchor.digest
      && (anchor.endLine > anchor.startLine || (anchor.endLine === anchor.startLine && anchor.endColumn > anchor.startColumn)));
  }
  for (const item of [...capture.definitions, ...capture.invocations, ...capture.operations]) check(unique(item.sourceBindingIds) && item.sourceBindingIds.every(id => bindings.has(id)));
  for (const diagnostic of capture.diagnostics) check(diagnostic.sourceBindingId === null || bindings.has(diagnostic.sourceBindingId));
  const paths = new Set<string>();
  for (const module of capture.modules) {
    check(definitions.has(module.definitionId));
    const constructors = module.constructorBindingIds ?? [];
    check(unique(constructors) && constructors.every(id => bindings.get(id)?.kind === 'constructor'));
    check(!module.constructorClassVerified || (constructors.length === 1 && bindings.get(constructors[0])?.quality === 'exact'));
    check(initializerArgumentsValid(module));
    const entry = module.initializerArguments;
    if (entry) {
      const scalar = (value: NonNullable<typeof entry.varArgs>[number], privateName = false) => {
        if (value.kind !== 'opaque' && value.kind !== 'none') check(!privateName);
      };
      for (const parameter of [...(entry.positional ?? []), ...(entry.keywordOnly ?? [])]) {
        scalar(parameter.value, PRIVATE_NAME.test(parameter.name.replace(/^_+/, '')));
      }
      for (const value of entry.varArgs ?? []) scalar(value);
      for (const [name, value] of Object.entries(entry.varKwargs ?? {})) {
        check(!LOCAL_PATH.test(name)); scalar(value, PRIVATE_NAME.test(name));
      }
    }
    for (const path of module.paths) { check(!paths.has(path)); paths.add(path); }
  }
  const seenCalls = new Set<string>(), sequences = new Set<number>();
  for (const call of capture.invocations) {
    check(modules.has(call.moduleId));
    if (call.parentId !== null) {
      const parent = calls.get(call.parentId);
      check(seenCalls.has(call.parentId) && parent && parent.sequence < call.sequence);
      if (parent.endSequence !== null) check(call.endSequence !== null && call.endSequence < parent.endSequence);
    }
    check(call.endSequence === null ? call.status === 'interrupted' : call.endSequence > call.sequence);
    check(call.parameterCount == null || call.parameterCountUnavailable == null);
    for (const seq of call.endSequence === null ? [call.sequence] : [call.sequence, call.endSequence]) { check(!sequences.has(seq)); sequences.add(seq); }
    seenCalls.add(call.id);
  }
  let previousSequence = -1, opaque = false;
  function tree(argument: Argument | null, before: number, depth = 0, count = { nodes: 0 }, privateName = false): void {
    if (argument === null) return;
    check(depth <= 32 && ++count.nodes <= 16384);
    if (argument.kind === 'value') {
      const value = values.get(argument.id);
      check(value);
      if (value.producer) { const producer = ops.get(value.producer.operationId); check(producer && producer.sequence < before); }
      return;
    }
    if (argument.kind === 'none') return;
    if (argument.kind === 'list' || argument.kind === 'tuple') {
      argument.items.forEach(item => tree(item, before, depth + 1, count, privateName)); return;
    }
    if (argument.kind === 'dict') {
      const keys = new Set<string>();
      for (const entry of argument.entries) {
        const key = entry.key;
        const identity = key.kind === 'none' ? 'none' : key.kind === 'bool' ? `int:${key.value ? '1' : '0'}`
          : key.kind === 'int' ? `int:${key.value}` : `str:${key.value}`;
        check(!keys.has(identity)); keys.add(identity);
        check(!('value' in key && typeof key.value === 'string' && LOCAL_PATH.test(key.value)));
        tree(entry.value, before, depth + 1, count, privateName || ('value' in key && typeof key.value === 'string' && PRIVATE_NAME.test(key.value)));
      }
      return;
    }
    check(!privateName && !('value' in argument && typeof argument.value === 'string' && LOCAL_PATH.test(argument.value)));
    if (argument.kind === 'opaque') opaque = true;
    if (argument.kind === 'float') check(!Object.is(argument.value, -0));
  }
  function args(arguments_: Arguments, before: number): void {
    const count = { nodes: 0 };
    arguments_.args.forEach(argument => tree(argument, before, 0, count));
    Object.entries(arguments_.kwargs).forEach(([name, argument]) => tree(argument, before, 0, count, PRIVATE_NAME.test(name)));
  }
  args(capture.inputs, -1);
  for (const call of capture.invocations) { args(call.arguments, call.sequence); tree(call.output, call.endSequence ?? Infinity); }
  for (const record of [...capture.invocations, ...capture.operations]) {
    check((record.status === 'completed') === (record.output !== null));
    check((record.status === 'failed') === (record.diagnosticId !== null));
    check(record.diagnosticId === null || diagnostics.has(record.diagnosticId));
  }
  for (const op of capture.operations) {
    check(op.sequence > previousSequence && !sequences.has(op.sequence)); previousSequence = op.sequence;
    if (op.invocationId !== null) {
      const call = calls.get(op.invocationId);
      check(call && call.sequence < op.sequence && (call.endSequence === null || op.sequence < call.endSequence));
    }
    args(op.arguments, op.sequence); tree(op.output, op.sequence + 1);
    check(op.status !== 'interrupted' || !op.effects.length);
    const leaves = new Map(inputLeaves(op.arguments, 'arguments').map(entry => [JSON.stringify(entry.path.slice(1)), entry.argument]));
    const effectKeys = new Set<string>();
    for (const effect of op.effects) {
      const key = JSON.stringify([effect.kind, effect.argumentPath, effect.beforeId]);
      check(!effectKeys.has(key)); effectKeys.add(key);
      const argument = leaves.get(JSON.stringify(effect.argumentPath)), prior = values.get(effect.beforeId);
      check(argument?.kind === 'value' && prior);
      check(effect.kind === 'write' ? prior.id === argument.id : prior.id !== argument.id && prior.storageId !== null && prior.storageId === values.get(argument.id)?.storageId);
      tree({ kind: 'value', id: effect.beforeId }, op.sequence); tree(effect.after, op.sequence + 1);
      if (effect.after.kind === 'value') { const next = values.get(effect.after.id); check(next?.previousVersion === prior.id && next.producer?.operationId === op.id); }
    }
  }
  tree(capture.output, Infinity);
  const seenValues = new Set<string>(), latest = new Map<string, CaptureSnapshot['values'][number]>();
  const outputs = new Map(capture.operations.map(op => [op.id, new Map(argumentLeaves(op.output).map(entry => [JSON.stringify(entry.path), entry.argument]))]));
  for (const value of capture.values) {
    check(value.stride === null || value.shape === null || value.stride.length === value.shape.length);
    check((value.origin === 'operation') === (value.producer !== null) && (value.origin !== 'state' || value.state.length));
    for (const ref of [value.aliasOf, value.previousVersion]) check(ref === null || seenValues.has(ref));
    if (value.aliasOf !== null) { const alias = values.get(value.aliasOf)!; check(value.storageId === null || alias.storageId === null || value.storageId === alias.storageId); }
    const prior = latest.get(value.objectId);
    check(value.previousVersion === null ? !prior && value.logicalVersion === 0 : prior && value.previousVersion === prior.id
      && value.logicalVersion === prior.logicalVersion + 1 && (prior.version === null || value.version === null || prior.version <= value.version));
    if (value.producer) {
      const producer = value.producer, op = ops.get(producer.operationId);
      check(op);
      const argument = producer.kind === 'output' ? outputs.get(op.id)?.get(JSON.stringify(producer.path))
        : producer.path.length === 2 && typeof producer.path[0] === 'number' && producer.path[1] === 'after' ? op.effects[producer.path[0]]?.after : undefined;
      check(argument?.kind === 'value' && argument.id === value.id);
    }
    latest.set(value.objectId, value); seenValues.add(value.id);
  }
  const root = capture.rootInvocationId === null ? undefined : calls.get(capture.rootInvocationId);
  check(capture.rootInvocationId === null || root?.parentId === null);
  check(capture.invocations.every(call => call.parentId !== null || call.id === capture.rootInvocationId));
  if (root) check(sameJson(root.arguments, capture.inputs) && sameJson(root.output, capture.output));
  const coverage = capture.coverage;
  check(coverage.operationsSeen >= ops.size && coverage.invocationsSeen >= calls.size && unique(coverage.omissions));
  check(coverage.omissions.length || (coverage.operationsSeen === ops.size && coverage.invocationsSeen === calls.size));
  const byPath = new Map<string, string[]>();
  for (const call of capture.invocations) for (const path of modules.get(call.moduleId)!.paths) {
    const ids = byPath.get(path) ?? []; ids.push(call.id); byPath.set(path, ids);
  }
  check(unique(coverage.modules.map(entry => entry.modulePath)));
  for (const entry of coverage.modules) {
    check(unique(entry.invocationIds) && sameJson([...entry.invocationIds].sort(), [...(byPath.get(entry.modulePath) ?? [])].sort()));
    check(['observed', 'failed'].includes(entry.status) === !!entry.invocationIds.length);
    check((entry.status === 'capture_limited') === (entry.reason !== null));
    if (entry.status === 'failed') check(entry.invocationIds.some(id => calls.get(id)?.status === 'failed'));
  }
  if (capture.status === 'complete') check(root?.status === 'completed' && !opaque && !coverage.omissions.length
    && coverage.operationsSeen === ops.size && coverage.invocationsSeen === calls.size
    && !capture.invocations.some(call => call.status === 'interrupted')
    && !capture.operations.some(op => op.status === 'interrupted' || (op.status === 'failed' && op.effects.length)));
  check(ops.size <= capture.limits.operations && calls.size <= capture.limits.invocations && values.size <= capture.limits.values);
  check(new Blob([JSON.stringify(capture)]).size <= capture.limits.bytes);
}

export function requirePortableCapture(value: unknown): PortableCapture {
  try {
    requireCaptureFact(structure(value));
    const source = value.capture.source;
    requireCaptureFact(source.workspaceSessionId === 'portable' && source.workspaceId === 'portable' && source.modelId === 'portable'
      && source.generation === 0 && /^portable:[0-9a-f]{64}$/.test(source.id)
      && source.files.every((file, i) => file.documentVersion === null && file.path === (file.role === 'config' ? 'tensorviz.toml' : `file-${i}.py`)));
    validateRelations(value.capture);
    return value;
  } catch { throw new Error('This file has invalid, inconsistent or private capture data. Lossless import refused.'); }
}
