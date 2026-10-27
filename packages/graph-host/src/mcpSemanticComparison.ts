/** Bounded MCP presentation of the shared canonical comparer. Exact retained
 * trees remain addressable through inspect_run; display limits never affect diffing. */
import { compareCaptures, type CaptureRecordChange, type CapturedValueUse } from '@tensorviz/contracts/captureComparison';
import type { CaptureSnapshot } from './semanticCapture';
import type { ComparableRun } from './mcpRunComparison';
import { object, type JsonObject } from './mcpGraph';

const rows = (value: unknown): JsonObject[] => Array.isArray(value) ? value.map(object) : [];
const size = (value: unknown) => Buffer.byteLength(JSON.stringify(value));
/** References and bounded inline previews; inspect_run pages the original tree. */
function inline(value: unknown): JsonObject {
  if (value === undefined) return { present: false };
  const serializedBytes = size(value);
  return serializedBytes <= 4096 ? { present: true, value } : { present: true, inline: false, serializedBytes };
}
function page(items: JsonObject[], offset: number, limit: number): JsonObject {
  const selected: JsonObject[] = []; let bytes = 0;
  for (const item of items.slice(offset, offset + limit)) {
    const itemBytes = size(item);
    if (itemBytes > 60 * 1024) throw new Error('Canonical comparison row was not bounded.');
    if (bytes + itemBytes > 60 * 1024) break;
    selected.push(item); bytes += itemBytes;
  }
  return { items: selected, count: items.length, nextOffset: offset + selected.length < items.length ? offset + selected.length : null };
}
export function needsSemanticComparison(run: ComparableRun): boolean {
  const unavailable = run.evidence?.captureUnavailable;
  return !!run.evidence?.capture || Object.keys(object(run.graph.capture)).length > 0
    || typeof unavailable === 'string' && unavailable !== 'not_captured';
}

export function compareSemanticRuns(baseline: ComparableRun, candidate: ComparableRun, section: string,
  offset: number, limit: number, base: JsonObject, setupDetails: JsonObject[], outputs: JsonObject): JsonObject {
  const before = baseline.evidence?.capture as CaptureSnapshot | undefined;
  const after = candidate.evidence?.capture as CaptureSnapshot | undefined;
  const numericSummary = { ...outputs, items: undefined };
  if (!before || !after) {
    const unavailable = { ...base, format: 'unavailable', semanticComparisonAvailable: false, equivalentCapturedComputation: null,
      captureAvailable: { baseline: !!before, candidate: !!after },
      detail: 'Both runs need accepted canonical captures for semantic comparison. Legacy layer summaries cannot supply missing arguments, state or sharing.' };
    if (section === 'setup') return { ...unavailable, ...page(setupDetails.map(detail => ({ ...detail,
      baseline: inline(detail.baseline), candidate: inline(detail.candidate) })), offset, limit) };
    if (section === 'outputs') return { ...unavailable, ...numericSummary, ...page(rows(outputs.items), offset, limit) };
    return { ...unavailable, outputs: numericSummary, sections: ['setup', 'outputs'] };
  }
  const report = compareCaptures(before, after);
  const projection = (run: ComparableRun, capture: CaptureSnapshot) => {
    const map = new Map<string, JsonObject | null>();
    const ports = new Map<string, unknown>(), edges = new Map<string, unknown>();
    for (const node of rows(run.graph.nodes)) {
      const reference = object(node.captureRef);
      if (reference.captureId !== capture.id || reference.sourceId !== capture.source.id) continue;
      const key = JSON.stringify([reference.kind, reference.semanticId]);
      map.set(key, map.has(key) ? null : node);
      for (const port of rows(node.typedPorts)) for (const path of Array.isArray(port.paths) ? port.paths : []) {
        const key = JSON.stringify([node.nodeId, path]);
        ports.set(key, ports.has(key) ? null : port.id);
      }
    }
    for (const edge of rows(run.graph.edges)) {
      const key = JSON.stringify([edge.targetNodeId, edge.targetPort, object(edge.captureEdge).valueId]);
      edges.set(key, edges.has(key) ? null : edge.edgeId);
    }
    const get = (kind: string, id: string) => map.get(JSON.stringify([kind, id]));
    return { get, nodeId: (kind: string, id: string) => get(kind, id)?.nodeId ?? null,
      edgeId: (use: CapturedValueUse) => {
        const node = get(use.entity.kind, use.entity.id);
        const port = ports.get(JSON.stringify([node?.nodeId, use.path]));
        return node && port ? edges.get(JSON.stringify([node.nodeId, port, use.valueId])) ?? null : null;
      } };
  };
  const pa = projection(baseline, before), pb = projection(candidate, after);
  const inspect = (run: ComparableRun, capture: CaptureSnapshot, kind: CaptureRecordChange['kind'], id: string): JsonObject => {
    if (kind === 'capture') return { runId: run.runId, section: 'capture', path: [] };
    if (kind === 'module') return { runId: run.runId, section: 'capture', path: ['modules', capture.modules.findIndex(module => module.id === id)] };
    return { runId: run.runId, section: 'capture', entity: { kind, id }, path: [] };
  };
  const records = (changes: CaptureRecordChange[]) => changes.flatMap(record => record.fields.map(field => ({
    kind: record.kind, baselineId: record.beforeId, candidateId: record.afterId, field: field.field,
    baselineNodeId: pa.nodeId(record.kind, record.beforeId), candidateNodeId: pb.nodeId(record.kind, record.afterId),
    baseline: inline(field.before.value), candidate: inline(field.after.value),
    baselineInspect: inspect(baseline, before, record.kind, record.beforeId),
    candidateInspect: inspect(candidate, after, record.kind, record.afterId),
  })));
  const changed = records(report.changes), observed = records(report.observations), unresolved = records(report.unresolved);
  const use = (item: CapturedValueUse | null, view: typeof pa, run: ComparableRun): JsonObject | null => {
    if (item === null) return null;
    const node = view.get(item.entity.kind, item.entity.id);
    const smallPath = size(item.path) <= 4096;
    return { ...item, path: smallPath ? item.path : inline(item.path), nodeId: node?.nodeId ?? null, edgeId: view.edgeId(item),
      inspect: { runId: run.runId, section: 'capture', entity: item.entity,
        path: smallPath ? item.entity.kind === 'output' ? item.path.slice(1) : item.path : [] } };
  };
  const uses = (changes: typeof report.valueUses.changes) => changes.map(change => ({ baseline: use(change.before, pa, baseline),
    candidate: use(change.after, pb, candidate) }));
  const connections = uses(report.valueUses.changes), uncertainConnections = uses(report.valueUses.unresolved);
  const comparisonBase = { ...base, format: 'semantic-v1', semanticComparisonAvailable: true,
    equivalentCapturedComputation: report.equivalent, coverageComplete: report.coverageComplete,
    capturedControls: Object.fromEntries(report.controls.map(control => [control.field, { status: control.status }])),
    detail: 'Pairs describe recorded roles, not persistent Python objects. Unmatched records and unresolved operands do not prove program additions/removals. Shape, mode, coverage and weight differences are separate. This covers the captured paths only.' };
  if (section === 'setup') return { ...comparisonBase, ...page(setupDetails.map(detail => ({ ...detail,
    baseline: inline(detail.baseline), candidate: inline(detail.candidate) })), offset, limit),
    captureControls: page(report.controls.map(control => ({ field: control.field, status: control.status,
      baseline: inline(control.before), candidate: inline(control.after),
      baselineInspect: { runId: baseline.runId, section: 'capture', path: [] }, candidateInspect: { runId: candidate.runId, section: 'capture', path: [] } })), offset, limit) };
  if (section === 'nodes') return { ...comparisonBase, changes: page(changed, offset, limit), unresolved: page(unresolved, offset, limit) };
  if (section === 'connections') return { ...comparisonBase, representation: 'exact_value_uses', ...page(connections, offset, limit), unresolved: page(uncertainConnections, offset, limit) };
  if (section === 'observations' || section === 'coverage') return { ...comparisonBase,
    ...page(observed.filter(record => section === 'observations' || record.kind === 'capture'), offset, limit),
    unresolved: page(unresolved.filter(record => section !== 'coverage' || record.kind === 'capture' && record.field === 'coverage'), offset, limit) };
  if (section === 'outputs') return { ...comparisonBase, ...numericSummary, ...page(rows(outputs.items), offset, limit) };
  if (section === 'correspondence') {
    const unmatched = (side: 'before' | 'after') => Object.entries(report.correspondence.unmatched[side]).flatMap(([kind, ids]) => ids.map(id => ({kind,id})));
    return { ...comparisonBase, pairs: page(report.correspondence.pairs.map(pair => ({ ...pair,
      baselineNodeId: pa.nodeId(pair.kind, pair.beforeId), candidateNodeId: pb.nodeId(pair.kind, pair.afterId) })), offset, limit),
      baselineUnmatched: page(unmatched('before'), offset, limit), candidateUnmatched: page(unmatched('after'), offset, limit),
      uncertain: page(report.correspondence.uncertain.map(group => ({ kind: group.kind,
        baselineCount: group.beforeIds.length, candidateCount: group.afterIds.length,
        baselineSample: group.beforeIds.slice(0, 10), candidateSample: group.afterIds.slice(0, 10),
        truncated: group.beforeIds.length > 10 || group.afterIds.length > 10 })), offset, limit) };
  }
  return { ...comparisonBase, changedRecordCount: report.changes.length, changedFieldCount: changed.length,
    observedFieldChangeCount: observed.length, unresolvedFieldCount: unresolved.length, connectionChangeCount: connections.length,
    unresolvedConnectionCount: uncertainConnections.length, matchedRecordCount: report.correspondence.pairs.length,
    unmatched: Object.fromEntries(Object.entries(report.correspondence.unmatched).map(([side, kinds]) => [side,
      Object.fromEntries(Object.entries(kinds).map(([kind, ids]) => [kind, ids.length]))])),
    uncertainMatchCount: report.correspondence.uncertain.length, outputs: numericSummary,
    firstObservedDivergence: null, firstObservedDivergenceUnavailable: 'Canonical records do not retain intermediate tensor payloads or establish alignment for every event.',
    sections: ['setup', 'nodes', 'connections', 'coverage', 'outputs', 'observations', 'correspondence'] };
}
