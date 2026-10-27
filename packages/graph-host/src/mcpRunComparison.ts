import * as path from 'node:path';
import { createHash } from 'node:crypto';
import { object, type JsonObject } from './mcpGraph';
import { compareOutputs } from './runNumerics';
import { compareSemanticRuns, needsSemanticComparison } from './mcpSemanticComparison';

export interface ComparableRun { runId: string; status: string; graph: JsonObject; evidence?: JsonObject }
const rows = (value: unknown): JsonObject[] => Array.isArray(value) ? value.map(object) : [];
function canonical(value: unknown): string {
  if (value === undefined) return 'undefined';
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value !== null && typeof value === 'object') return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}`;
  return JSON.stringify(value);
}
const equal = (a: unknown, b: unknown) => canonical(a) === canonical(b);
const page = (items: JsonObject[], offset: number, limit: number) => ({ items: items.slice(offset, offset + limit),
  count: items.length, nextOffset: offset + limit < items.length ? offset + limit : null });
function fields(before: JsonObject, after: JsonObject): JsonObject[] {
  return [...new Set([...Object.keys(before), ...Object.keys(after)])].sort().flatMap(field => equal(before[field], after[field]) ? []
    : [{ field, baseline: before[field] ?? null, candidate: after[field] ?? null,
      baselinePresent: Object.hasOwn(before, field), candidatePresent: Object.hasOwn(after, field) }]);
}
function structure(node: JsonObject): JsonObject {
  return { operation: node.operation, moduleClass: node.moduleClass, definition: node.definition,
    kind: node.kind, parameters: node.parameters, repeatCount: node.repeatCount, inputPorts: node.inputPorts, outputPorts: node.outputPorts };
}
function identity(node: JsonObject, run: ComparableRun, modulePaths: Set<unknown>): string | undefined {
  if (typeof node.modulePath === 'string' && modulePaths.has(node.modulePath)) return `module:${node.modulePath}`;
  if (node.nodeId === 'input_node' || node.nodeId === 'output_node') return `boundary:${node.nodeId}`;
  const source = object(node.source);
  if (typeof source.file === 'string' && typeof source.digest === 'string') {
    const root = String(run.graph.canonicalWorkspace ?? object(run.graph.target).workspaceFolder ?? '');
    const file = path.isAbsolute(source.file) ? path.relative(root, source.file) : source.file;
    return canonical(['source', file, source.digest, node.operation]);
  }
  return undefined;
}
function indexed(nodes: JsonObject[], run: ComparableRun): Map<string, JsonObject[]> {
  const result = new Map<string, JsonObject[]>();
  const modulePaths = new Set(rows(object(run.evidence?.execution).coverage).map(module => module.modulePath));
  for (const node of nodes) {
    const key = identity(node, run, modulePaths);
    if (key) result.set(key, [...(result.get(key) ?? []), node]);
  }
  return result;
}
function topology(before: ComparableRun, after: ComparableRun): JsonObject {
  const a = rows(before.graph.nodes), b = rows(after.graph.nodes), ai = indexed(a, before), bi = indexed(b, after);
  const matchedBefore = new Set<unknown>(), matchedAfter = new Set<unknown>();
  const beforeKeys = new Map<unknown, string>(), afterKeys = new Map<unknown, string>();
  const changes: JsonObject[] = [], dimensions: JsonObject[] = [];
  for (const [key, previous] of ai) {
    const next = bi.get(key);
    if (previous.length !== 1 || next?.length !== 1) continue;
    const left = previous[0], right = next[0];
    matchedBefore.add(left.nodeId); matchedAfter.add(right.nodeId);
    beforeKeys.set(left.nodeId, key); afterKeys.set(right.nodeId, key);
    const changed = fields(structure(left), structure(right));
    const node = { identity: key, baselineNodeId: left.nodeId, candidateNodeId: right.nodeId };
    if (changed.length) changes.push({ ...node, change: 'changed', fields: changed });
    const shapes = fields(object(left.shapes), object(right.shapes));
    if (shapes.length) dimensions.push({ ...node, fields: shapes });
  }
  const removed = a.filter(node => !matchedBefore.has(node.nodeId));
  const added = b.filter(node => !matchedAfter.has(node.nodeId));
  // Structural similarity is a possibility, never a silent identity match.
  const groups = new Map<string, { baselineNodeIds: unknown[]; candidateNodeIds: unknown[] }>();
  for (const [nodes, side] of [[removed, 'baselineNodeIds'], [added, 'candidateNodeIds']] as const) {
    for (const node of nodes) {
      const key = canonical(structure(node));
      const group = groups.get(key) ?? { baselineNodeIds: [], candidateNodeIds: [] };
      group[side].push(node.nodeId); groups.set(key, group);
    }
  }
  const uncertain = [...groups.values()].filter(group => group.baselineNodeIds.length && group.candidateNodeIds.length);
  const uncertainBefore = new Set(uncertain.flatMap(group => group.baselineNodeIds));
  const uncertainAfter = new Set(uncertain.flatMap(group => group.candidateNodeIds));
  for (const node of removed) {
    beforeKeys.set(node.nodeId, `baseline:${node.nodeId}`);
    changes.push({ change: 'removed', baseline: node, uncertainIdentity: uncertainBefore.has(node.nodeId) });
  }
  for (const node of added) {
    afterKeys.set(node.nodeId, `candidate:${node.nodeId}`);
    changes.push({ change: 'added', candidate: node, uncertainIdentity: uncertainAfter.has(node.nodeId) });
  }
  const edges = (run: ComparableRun, keys: Map<unknown, string>, uncertainIds: Set<unknown>) => {
    const result = new Map<string, { edge: JsonObject; count: number; uncertain: boolean }>();
    for (const edge of rows(run.graph.edges)) {
      const source = keys.get(edge.sourceNodeId), target = keys.get(edge.targetNodeId);
      const key = canonical([source ?? `unknown:${run.runId}:${edge.sourceNodeId}`,
        target ?? `unknown:${run.runId}:${edge.targetNodeId}`, edge.sourcePort, edge.targetPort]);
      const prior = result.get(key);
      result.set(key, { edge, count: (prior?.count ?? 0) + 1,
        uncertain: !source || !target || uncertainIds.has(edge.sourceNodeId) || uncertainIds.has(edge.targetNodeId) });
    }
    return result;
  };
  const ae = edges(before, beforeKeys, uncertainBefore), be = edges(after, afterKeys, uncertainAfter);
  const connections: JsonObject[] = [];
  for (const key of new Set([...ae.keys(), ...be.keys()])) {
    const left = ae.get(key), right = be.get(key);
    if (left?.count === right?.count) continue;
    connections.push({ baseline: left?.edge ?? null, candidate: right?.edge ?? null,
      baselineCount: left?.count ?? 0, candidateCount: right?.count ?? 0,
      uncertainIdentity: !!(left?.uncertain || right?.uncertain) });
  }
  return { changes, dimensions, connections, matchedNodeCount: matchedBefore.size,
    uncertainMatches: uncertain.map(group => ({ baselineNodeIds: group.baselineNodeIds.slice(0, 200), candidateNodeIds: group.candidateNodeIds.slice(0, 200),
      baselineCount: group.baselineNodeIds.length, candidateCount: group.candidateNodeIds.length,
      truncated: group.baselineNodeIds.length > 200 || group.candidateNodeIds.length > 200 })) };
}
function setupDifferences(before: ComparableRun, after: ComparableRun): JsonObject {
  const controls = (run: ComparableRun): JsonObject => {
    const value = object(run.evidence?.setup);
    return {
      inputs: object(value.inputs).digest, weights: object(value.state).digest,
      mode: value.mode === undefined ? undefined : { mode: value.mode, modules: value.moduleModes, gradEnabled: value.gradEnabled },
      environment: value.environment,
      randomness: value.rngDigest === undefined ? undefined : { seed: value.seed, rng: value.rng ?? value.rngDigest },
      modelSetup: value.setup,
      diagnostic: !Object.keys(value).length ? undefined : value.training !== undefined ? { kind: 'training', ...object(value.training) }
        : { kind: 'forward' },
    };
  };
  const a = controls(before), b = controls(after), details: JsonObject[] = [], summaries: JsonObject = {};
  const digest = (value: unknown) => value == null ? null : createHash('sha256').update(canonical(value)).digest('hex');
  // A page contains scalar setup facts, never entire state dictionaries or module inventories.
  function leaves(value: unknown, prefix: string, result: JsonObject = {}): JsonObject {
    if (value !== null && typeof value === 'object' && Object.keys(value).length) {
      for (const [key, child] of Object.entries(value)) leaves(child, `${prefix}[${JSON.stringify(key)}]`, result);
    } else result[prefix] = value;
    return result;
  }
  for (const field of Object.keys(a)) {
    const baseline = a[field], candidate = b[field];
    const unknown = baseline == null || candidate == null || field === 'randomness' &&
      (object(object(baseline).rng).complete === false || object(object(candidate).rng).complete === false);
    summaries[field] = { status: unknown ? 'unknown' : equal(baseline, candidate) ? 'same' : 'different',
      baselineDigest: digest(baseline), candidateDigest: digest(candidate) };
    details.push(...fields(leaves(baseline, field), leaves(candidate, field)).map(change => ({ category: field, ...change })));
  }
  return { ...summaries, recordedControlsMatch: Object.values(summaries).every(field => object(field).status === 'same'), details,
    detail: 'Matching recorded controls is not proof that custom attributes, external I/O or custom random generators are identical. Use the setup section for changed facts and inspect_run for full setup evidence.' };
}
function observations(before: ComparableRun, after: ComparableRun): JsonObject {
  const a = object(before.evidence?.execution), b = object(after.evidence?.execution);
  const previousCoverage = new Map(rows(a.coverage).map(module => [module.modulePath, module]));
  const nextCoverage = new Map(rows(b.coverage).map(module => [module.modulePath, module]));
  const coverage: JsonObject[] = [];
  for (const key of new Set([...previousCoverage.keys(), ...nextCoverage.keys()])) {
    const left = previousCoverage.get(key), right = nextCoverage.get(key);
    if (!equal(left, right)) coverage.push({ modulePath: key, baseline: left ?? null, candidate: right ?? null });
  }
  const key = (call: JsonObject) => canonical([call.modulePath, call.invocation]);
  const previous = new Map(rows(a.moduleCalls).map(call => [key(call), call]));
  const next = new Map(rows(b.moduleCalls).map(call => [key(call), call]));
  const changes: JsonObject[] = [];
  const complete = (execution: JsonObject) => Array.isArray(execution.moduleCalls) && Array.isArray(execution.truncated) &&
    !execution.truncated.includes('module_calls');
  let unknownValues = 0, baselineOnlyCalls = 0;
  for (const call of [...rows(b.moduleCalls)].sort((x, y) => Number(x.completedIndex ?? Infinity) - Number(y.completedIndex ?? Infinity))) {
    const left = previous.get(key(call));
    if (!left) {
      unknownValues++;
      changes.push({ modulePath: call.modulePath, invocation: call.invocation, kind: 'newly_observed_call',
        candidateCompletedIndex: call.completedIndex ?? null, absenceConfirmed: complete(a) });
      continue;
    }
    const shape = (value: unknown) => rows(value).map(tensor => ({ shape: tensor.shape, dtype: tensor.dtype, device: tensor.device }));
    const fieldsChanged = fields({ status: left.status, inputs: shape(left.inputs), outputs: shape(left.outputs) },
      { status: call.status, inputs: shape(call.inputs), outputs: shape(call.outputs) });
    const oldValues = object(left.values), newValues = object(call.values);
    const valuesKnown = typeof oldValues.digest === 'string' && typeof newValues.digest === 'string';
    if (!valuesKnown) unknownValues++;
    if (fieldsChanged.length || valuesKnown && oldValues.digest !== newValues.digest) changes.push({
      modulePath: call.modulePath, invocation: call.invocation, baselineCompletedIndex: left.completedIndex ?? null,
      candidateCompletedIndex: call.completedIndex ?? null, fields: fieldsChanged,
      kind: fieldsChanged.length ? 'shape_or_coverage' : 'value_fingerprint',
      valuesComparable: valuesKnown, valuesDiffer: valuesKnown ? oldValues.digest !== newValues.digest : null });
  }
  const firstObservedDivergence = changes.find(change => change.absenceConfirmed !== false) ?? null;
  for (const [id, call] of previous) if (!next.has(id)) {
    baselineOnlyCalls++; unknownValues++;
    changes.push({ modulePath: call.modulePath, invocation: call.invocation, kind: 'not_observed_in_candidate',
      baselineCompletedIndex: call.completedIndex ?? null, absenceConfirmed: complete(b) });
  }
  return { coverage, changes, firstObservedDivergence, baselineOnlyCalls, callsWithoutComparableValues: unknownValues,
    evidenceAvailable: { baseline: Array.isArray(a.moduleCalls), candidate: Array.isArray(b.moduleCalls) },
    baselineTruncation: a.truncated ?? null, candidateTruncation: b.truncated ?? null,
    detail: 'The first retained difference is ordered by candidate module completion. Baseline-only calls have no candidate ordering. Value fingerprints detect byte changes; module tolerances are unavailable without retained values. Unobserved calls may differ earlier.' };
}

export function compareRetainedRuns(baseline: ComparableRun, candidate: ComparableRun,
  section: string, offset: number, limit: number, atol: number, rtol: number): JsonObject {
  const outputs = compareOutputs(object(baseline.evidence?.execution).outputs, object(candidate.evidence?.execution).outputs, atol, rtol);
  const { details, ...setup } = setupDifferences(baseline, candidate);
  const base: JsonObject = { baselineRunId: baseline.runId, candidateRunId: candidate.runId,
    outcomes: { baseline: baseline.status, candidate: candidate.status }, setup,
    snapshots: { baseline: { target: baseline.graph.target, traced: baseline.graph.traced, truncated: baseline.graph.truncated },
      candidate: { target: candidate.graph.target, traced: candidate.graph.traced, truncated: candidate.graph.truncated } } };
  if (needsSemanticComparison(baseline) || needsSemanticComparison(candidate)) {
    return compareSemanticRuns(baseline, candidate, section, offset, limit, base, rows(details), outputs);
  }
  const architecture = topology(baseline, candidate), observed = observations(baseline, candidate);
  base.format = 'legacy';
  if (section === 'setup') return { ...base, ...page(rows(details), offset, limit) };
  if (section === 'nodes') return { ...base, changes: page(rows(architecture.changes), offset, limit),
    dimensions: page(rows(architecture.dimensions), offset, limit), uncertainMatches: page(rows(architecture.uncertainMatches), offset, limit) };
  if (section === 'connections') return { ...base, ...page(rows(architecture.connections), offset, limit) };
  if (section === 'coverage') return { ...base, ...page(rows(observed.coverage), offset, limit) };
  if (section === 'outputs') return { ...base, ...outputs, ...page(rows(outputs.items), offset, limit) };
  if (section === 'observations') return { ...base, ...observed, coverage: undefined, changes: page(rows(observed.changes), offset, limit) };
  return { ...base, nodeChangeCount: rows(architecture.changes).length, connectionChangeCount: rows(architecture.connections).length,
    dimensionChangeCount: rows(architecture.dimensions).length, uncertainMatchCount: rows(architecture.uncertainMatches).length,
    coverageChangeCount: rows(observed.coverage).length, matchedNodeCount: architecture.matchedNodeCount,
    firstObservedDivergence: observed.firstObservedDivergence, callsWithoutComparableValues: observed.callsWithoutComparableValues, baselineOnlyCalls: observed.baselineOnlyCalls,
    observationEvidence: { available: observed.evidenceAvailable, baselineTruncation: observed.baselineTruncation, candidateTruncation: observed.candidateTruncation },
    outputs: { ...outputs, items: undefined }, sections: ['setup', 'nodes', 'connections', 'coverage', 'outputs', 'observations'],
    detail: 'This compares captured executions, not every possible model input. Layout and annotations are excluded. Setup differences can explain output differences.' };
}
