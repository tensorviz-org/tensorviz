/** Conservative correspondence between validated captures. Runtime ids, event
 * ordinals, tensor shapes and content similarity are never cross-run identity.
 * These pairs describe recorded roles, not persistent Python object identity. */
import type { components } from './semanticContracts.generated';

type Capture = components['schemas']['CaptureSnapshot'];
type Argument = NonNullable<Capture['output']>;
type Arguments = Capture['inputs'];
type Operation = Capture['operations'][number];
export type CaptureEntityKind = 'module' | 'invocation' | 'operation' | 'value';
export interface CapturePair {
  kind: CaptureEntityKind; beforeId: string; afterId: string;
  basis: 'registered_paths' | 'root_call' | 'unique_module_call' | 'source_call' | 'input_path' | 'state_names' | 'producer_port' | 'dataflow_role';
}
export interface CaptureUncertainGroup { kind: CaptureEntityKind; beforeIds: string[]; afterIds: string[] }
export interface CaptureCorrespondence {
  pairs: CapturePair[];
  unmatched: { before: Record<CaptureEntityKind, string[]>; after: Record<CaptureEntityKind, string[]> };
  uncertain: CaptureUncertainGroup[];
}

/** Objects are unordered JSON fields; ordered kwargs are represented as entries
 * by argumentStructure, and typed mapping entries already carry their order. */
export function captureJson(value: unknown): string {
  if (value === undefined) return 'undefined';
  if (Array.isArray(value)) return `[${value.map(captureJson).join(',')}]`;
  if (value !== null && typeof value === 'object') return `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, item]) => `${JSON.stringify(key)}:${captureJson(item)}`).join(',')}}`;
  return JSON.stringify(value);
}

export function mapCaptureArgument(argument: Argument | null, value: (id: string) => unknown,
  literal: (argument: Argument) => unknown = item => item): unknown {
  if (argument === null) return null;
  if (argument.kind === 'value') return value(argument.id);
  if (argument.kind === 'list' || argument.kind === 'tuple') return { kind: argument.kind,
    items: argument.items.map(item => mapCaptureArgument(item, value, literal)) };
  if (argument.kind === 'dict') return { kind: argument.kind,
    entries: argument.entries.map(entry => ({ key: entry.key, value: mapCaptureArgument(entry.value, value, literal) })) };
  return literal(argument);
}
export function argumentStructure(arguments_: Arguments, value: (id: string) => unknown,
  literal?: (argument: Argument) => unknown): unknown {
  return { args: arguments_.args.map(item => mapCaptureArgument(item, value, literal)),
    kwargs: Object.entries(arguments_.kwargs).map(([name, item]) => [name, mapCaptureArgument(item, value, literal)]) };
}
function append<T>(map: Map<string, T[]>, key: string, item: T): void {
  const items = map.get(key); if (items) items.push(item); else map.set(key, [item]);
}
function valuePaths(arguments_: Arguments): Map<string, string[]> {
  const result = new Map<string, string[]>();
  const visit = (argument: Argument, path: unknown[]) => {
    if (argument.kind === 'value') append(result, argument.id, captureJson(path));
    else if (argument.kind === 'list' || argument.kind === 'tuple') argument.items.forEach((child, i) => visit(child, [...path, argument.kind, i]));
    else if (argument.kind === 'dict') argument.entries.forEach((entry, i) => visit(entry.value, [...path, 'dict', i, entry.key]));
  };
  arguments_.args.forEach((item, i) => visit(item, ['args', i]));
  Object.entries(arguments_.kwargs).forEach(([name, item]) => visit(item, ['kwargs', name]));
  return result;
}
function groups<T>(items: T[], key: (item: T) => string | undefined): Map<string, T[]> {
  const out = new Map<string, T[]>();
  for (const item of items) { const name = key(item); if (name !== undefined) append(out, name, item); }
  return out;
}
function uniquePairs<T>(left: T[], right: T[], keyA: (item: T) => string | undefined,
  keyB: (item: T) => string | undefined, pair: (a: T, b: T) => void): void {
  const a = groups(left, keyA), b = groups(right, keyB);
  for (const [key, items] of Array.from(a.entries())) { const next = b.get(key); if (items.length === 1 && next?.length === 1) pair(items[0], next[0]); }
}

export function correspondCaptures(before: Capture, after: Capture): CaptureCorrespondence {
  const pairs: CapturePair[] = [];
  const maps = () => ({ module: new Map<string, string>(), invocation: new Map<string, string>(),
    operation: new Map<string, string>(), value: new Map<string, string>() });
  const a = maps(), b = maps();
  const pair = (kind: CaptureEntityKind, left: string, right: string, basis: CapturePair['basis']) => {
    if (a[kind].has(left) || b[kind].has(right)) return false;
    const key = `${kind}:${pairs.length}`;
    a[kind].set(left, key); b[kind].set(right, key); pairs.push({ kind, beforeId: left, afterId: right, basis }); return true;
  };
  const moduleKey = (module: Capture['modules'][number]) => {
    // Numeric registrations can shift after an insertion. No instance lineage is
    // present in v1, so even equal numeric paths cannot silently become identity.
    if (!module.paths.length || module.paths.some(path => path.split('.').some(part => /^\d+$/.test(part)))) return undefined;
    return captureJson([...module.paths].sort());
  };
  uniquePairs(before.modules, after.modules, moduleKey, moduleKey,
    (left, right) => pair('module', left.id, right.id, 'registered_paths'));

  const anchorKey = (capture: Capture) => {
    const bindings = new Map(capture.sourceBindings.map(binding => [binding.id, binding]));
    const filesByDigest = groups(capture.source.files, file => file.digest);
    return (ids: string[]): string | undefined => {
      const keys = ids.flatMap(id => {
        const binding = bindings.get(id);
        if (binding?.kind !== 'call' || binding.quality !== 'exact') return [];
        const anchor = binding.anchors[0];
        const file = anchor.file === capture.source.modelFile ? 'selected-model'
          : filesByDigest.get(anchor.digest)?.length === 1 ? `file-content:${anchor.digest}` : undefined;
        return file ? [captureJson([file, anchor.scope, anchor.tokenDigest])] : [];
      });
      return keys.length ? captureJson(keys.sort()) : undefined;
    };
  };
  const anchorA = anchorKey(before), anchorB = anchorKey(after);
  const childrenA = groups(before.invocations, call => call.parentId ?? undefined);
  const childrenB = groups(after.invocations, call => call.parentId ?? undefined);
  const rootA = before.invocations.find(call => call.id === before.rootInvocationId);
  const rootB = after.invocations.find(call => call.id === after.rootInvocationId);
  const callQueue: Array<[Capture['invocations'][number], Capture['invocations'][number]]> = [];
  const callPair = (left: Capture['invocations'][number], right: Capture['invocations'][number], basis: CapturePair['basis']) => {
    if (pair('invocation', left.id, right.id, basis)) callQueue.push([left, right]);
  };
  if (rootA && rootB) callPair(rootA, rootB, 'root_call');
  for (let i = 0; i < callQueue.length; i++) {
    const [left, right] = callQueue[i], ca = childrenA.get(left.id) ?? [], cb = childrenB.get(right.id) ?? [];
    const key = (side: typeof a, anchor: typeof anchorA) => (call: Capture['invocations'][number]) => {
      const module = side.module.get(call.moduleId), source = anchor(call.sourceBindingIds);
      return module && source ? captureJson([module, source]) : undefined;
    };
    uniquePairs(ca, cb, key(a, anchorA), key(b, anchorB), (x, y) => callPair(x, y, 'source_call'));
    // Count all calls, including those already paired: removing them first could
    // turn the remaining ambiguous repeated call into an invented unique call.
    uniquePairs(ca, cb, call => a.module.get(call.moduleId), call => b.module.get(call.moduleId),
      (x, y) => callPair(x, y, 'unique_module_call'));
  }

  const inputA = valuePaths(before.inputs), inputB = valuePaths(after.inputs);
  uniquePairs(before.values.filter(value => value.origin === 'input'), after.values.filter(value => value.origin === 'input'),
    value => inputA.has(value.id) ? captureJson(inputA.get(value.id)) : undefined,
    value => inputB.has(value.id) ? captureJson(inputB.get(value.id)) : undefined,
    (x, y) => pair('value', x.id, y.id, 'input_path'));
  const stateKey = (value: Capture['values'][number]) => value.origin === 'state'
    && !value.state.some(state => state.name.split('.').some(part => /^\d+$/.test(part)))
    ? captureJson([value.logicalVersion, value.state.map(state => captureJson(state)).sort()]) : undefined;
  uniquePairs(before.values, after.values, stateKey, stateKey, (x, y) => pair('value', x.id, y.id, 'state_names'));

  const outputsA = groups(before.values, value => value.producer?.operationId);
  const outputsB = groups(after.values, value => value.producer?.operationId);
  const valueQueue: Array<[string, string]> = [];
  const operationPair = (left: Operation, right: Operation, basis: CapturePair['basis']) => {
    if (!pair('operation', left.id, right.id, basis)) return;
    const key = (value: Capture['values'][number]) => captureJson([value.producer!.kind, value.producer!.path]);
    uniquePairs(outputsA.get(left.id) ?? [], outputsB.get(right.id) ?? [], key, key, (x, y) => {
      if (pair('value', x.id, y.id, 'producer_port')) valueQueue.push([x.id, y.id]);
    });
  };
  const owner = (op: Operation, side: typeof a) => op.invocationId === null ? 'outside-module' : side.invocation.get(op.invocationId);
  const sourceKey = (side: typeof a, anchor: typeof anchorA) => (op: Operation) => {
    const call = anchor(op.sourceBindingIds), parent = owner(op, side);
    return call && parent ? captureJson([parent, call]) : undefined;
  };
  uniquePairs(before.operations, after.operations, sourceKey(a, anchorA), sourceKey(b, anchorB),
    (x, y) => operationPair(x, y, 'source_call'));

  // An operation can have a unique recorded dataflow role within a matched call.
  // All repeated candidates are indexed before choosing any pair. Dependencies
  // wake only their consumers, avoiding rescanning a long chain every round.
  interface Waiting { op: Operation; missing: Set<string>; ready: boolean; key?: string }
  const readyA = new Map<string, Waiting[]>(), readyB = new Map<string, Waiting[]>();
  const watchA = new Map<string, Waiting[]>(), watchB = new Map<string, Waiting[]>();
  const changedKeys = new Set<string>();
  const prepareSide = (capture: Capture, side: typeof a, ready: typeof readyA, watchers: typeof watchA) => {
    const enqueue = (entry: Waiting) => {
      if (entry.ready || entry.missing.size) return;
      entry.ready = true;
      const structure = argumentStructure(entry.op.arguments, id => ({ kind: 'value', role: side.value.get(id) }),
        item => ({ kind: item.kind }));
      entry.key = captureJson([owner(entry.op, side), entry.op.target, structure]);
      append(ready, entry.key, entry); changedKeys.add(entry.key);
    };
    for (const op of capture.operations) {
      if (owner(op, side) === undefined) continue;
      const missing = new Set<string>(); let opaque = false;
      argumentStructure(op.arguments, id => { if (!side.value.has(id)) missing.add(id); }, item => { if (item.kind === 'opaque') opaque = true; });
      if (opaque) continue;
      const entry: Waiting = { op, missing, ready: false };
      for (const id of Array.from(missing)) append(watchers, id, entry);
      enqueue(entry);
    }
    return (id: string) => { for (const entry of watchers.get(id) ?? []) { entry.missing.delete(id); enqueue(entry); } watchers.delete(id); };
  };
  const wakeA = prepareSide(before, a, readyA, watchA), wakeB = prepareSide(after, b, readyB, watchB);
  let valueCursor = 0;
  while (changedKeys.size || valueCursor < valueQueue.length) {
    const keys = Array.from(changedKeys); changedKeys.clear();
    for (const key of keys) {
      const left = readyA.get(key), right = readyB.get(key);
      if (left?.length === 1 && right?.length === 1) operationPair(left[0].op, right[0].op, 'dataflow_role');
    }
    while (valueCursor < valueQueue.length) { const [left, right] = valueQueue[valueCursor++]; wakeA(left); wakeB(right); }
  }

  const unmatched = (capture: Capture, side: typeof a) => ({
    module: capture.modules.filter(item => !side.module.has(item.id)).map(item => item.id),
    invocation: capture.invocations.filter(item => !side.invocation.has(item.id)).map(item => item.id),
    operation: capture.operations.filter(item => !side.operation.has(item.id)).map(item => item.id),
    value: capture.values.filter(item => !side.value.has(item.id)).map(item => item.id),
  });
  const uncertain: CaptureUncertainGroup[] = [];
  const candidates = (capture: Capture, side: typeof a) => {
    const defs = new Map(capture.definitions.map(item => [item.id, item.qualifiedName]));
    const modules = new Map(capture.modules.map(item => [item.id, item]));
    const result = new Map<string, { kind: CaptureEntityKind; ids: string[] }>();
    const add = (kind: CaptureEntityKind, id: string, key: unknown) => {
      if (side[kind].has(id)) return;
      const k = captureJson([kind, key]); const entry = result.get(k) ?? { kind, ids: [] };
      entry.ids.push(id); result.set(k, entry);
    };
    capture.modules.forEach(item => add('module', item.id, defs.get(item.definitionId)));
    capture.invocations.forEach(item => add('invocation', item.id, [defs.get(modules.get(item.moduleId)!.definitionId),
      item.parentId === null ? 'root' : side.invocation.get(item.parentId) ?? 'unmatched-parent']));
    capture.operations.forEach(item => add('operation', item.id, [owner(item, side) ?? 'unmatched-parent', item.target]));
    // Missing tensor correspondence remains explicit in unmatched.value. Matching
    // values by shape or dtype would confuse separate inputs and shared state.
    return result;
  };
  const candidatesB = candidates(after, b);
  for (const [key, left] of Array.from(candidates(before, a).entries())) {
    const right = candidatesB.get(key);
    if (right) uncertain.push({ kind: left.kind, beforeIds: left.ids, afterIds: right.ids });
  }
  return { pairs, unmatched: { before: unmatched(before, a), after: unmatched(after, b) }, uncertain };
}
