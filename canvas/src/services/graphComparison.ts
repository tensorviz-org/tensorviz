import { ArchitectureNode } from '../types/graphDocument';
import { canonical, EdgeTarget, edgeTarget } from './graphCollaboration';
import { GraphFile } from './graphFile';
import { compareCaptures, type SemanticCaptureComparison } from '../ide/captureComparison.generated';

export interface FieldChange {
  field: string;
  before: { present: boolean; value?: unknown };
  after: { present: boolean; value?: unknown };
}
export interface ChangedLayer { nodeId: string; changes: FieldChange[] }
export interface ChangedConnection { target: EdgeTarget; count: number; beforeCount: number; afterCount: number }
export interface UncertainMatch { beforeIds: string[]; afterIds: string[] }
export interface GraphComparison {
  capture?: SemanticCaptureComparison;
  addedLayers: ArchitectureNode[];
  removedLayers: ArchitectureNode[];
  changedLayers: ChangedLayer[];
  addedConnections: ChangedConnection[];
  removedConnections: ChangedConnection[];
  recordedShapes: ChangedLayer[];
  uncertainMatches: UncertainMatch[];
  modelNamesDiffer: boolean;
  snapshotEvidenceChanged: boolean;
  /** Null means the available comparer cannot judge these captures. */
  architectureChanged: boolean | null;
}
const order = (a: string, b: string): number => a < b ? -1 : a > b ? 1 : 0;
const sortedLayers = (nodes: ArchitectureNode[]) => [...nodes].sort((a, b) => order(a.node_id, b.node_id));
const valueAt = (record: Record<string, unknown>, key: string) => Object.prototype.hasOwnProperty.call(record, key)
  ? { present: true, value: record[key] } : { present: false };
function changes(before: Record<string, unknown>, after: Record<string, unknown>, prefix = ''): FieldChange[] {
  return Array.from(new Set([...Object.keys(before), ...Object.keys(after)])).sort(order).flatMap(key => {
    const a = valueAt(before, key); const b = valueAt(after, key);
    return canonical(a) === canonical(b) ? [] : [{ field: `${prefix}${key}`, before: a, after: b }];
  });
}
function structure(node: ArchitectureNode): Record<string, unknown> {
  return { kind: node.kind, operation: node.op, inputPorts: [...node.in_ports].sort(order), outputPorts: [...node.out_ports].sort(order),
    ...(node.metadata.parentId !== undefined ? { parent: node.metadata.parentId } : {}),
    ...(node.metadata.repeatCount !== undefined ? { repeatCount: node.metadata.repeatCount } : {}) };
}
function shapes(node: ArchitectureNode): Record<string, unknown> {
  return Object.fromEntries(['inputShape', 'outputShape'].filter(key => node.metadata[key] !== undefined).map(key => [key, node.metadata[key]]));
}
function connections(file: GraphFile): Map<string, { target: EdgeTarget; count: number }> {
  const result = new Map<string, { target: EdgeTarget; count: number }>();
  for (const edge of file.document.semantic.edges) {
    const target = edgeTarget(edge); const key = canonical(target);
    const previous = result.get(key);
    result.set(key, { target, count: (previous?.count ?? 0) + 1 });
  }
  return result;
}
/** Potential renames remain additions/removals. Grouping avoids all-pairs scans
 * and never upgrades a structural resemblance into a source-identity match. */
function uncertainMatches(before: ArchitectureNode[], after: ArchitectureNode[]): UncertainMatch[] {
  const groups = new Map<string, UncertainMatch>();
  const collect = (nodes: ArchitectureNode[], side: 'beforeIds' | 'afterIds') => {
    for (const node of nodes) {
      const { parent, ...facts } = structure(node);
      const key = canonical({ ...facts, parameters: node.params });
      const group = groups.get(key) ?? { beforeIds: [], afterIds: [] };
      group[side].push(node.node_id); groups.set(key, group);
    }
  };
  collect(before, 'beforeIds'); collect(after, 'afterIds');
  return Array.from(groups.values()).filter(group => group.beforeIds.length && group.afterIds.length);
}

/** Compare validated portable architecture only. Neither snapshot is mutated. */
export function compareGraphFiles(before: GraphFile, after: GraphFile): GraphComparison {
  if (before.capture && after.capture) {
    const capture = compareCaptures(before.capture.capture, after.capture.capture);
    return { capture, addedLayers: [], removedLayers: [], changedLayers: [], addedConnections: [], removedConnections: [],
      recordedShapes: [], uncertainMatches: [], modelNamesDiffer: before.document.semantic.name !== after.document.semantic.name,
      snapshotEvidenceChanged: before.status !== after.status || before.warningCount !== after.warningCount,
      architectureChanged: capture.equivalent === null ? null : !capture.equivalent };
  }
  // A legacy file cannot reconstruct the missing argument and instance semantics.
  if (before.capture || after.capture) return { addedLayers: [], removedLayers: [], changedLayers: [],
    addedConnections: [], removedConnections: [], recordedShapes: [], uncertainMatches: [],
    modelNamesDiffer: before.document.semantic.name !== after.document.semantic.name,
    snapshotEvidenceChanged: false, architectureChanged: null };
  const beforeNodes = new Map(before.document.semantic.nodes.map(node => [node.node_id, node]));
  const afterNodes = new Map(after.document.semantic.nodes.map(node => [node.node_id, node]));
  const addedLayers = sortedLayers(after.document.semantic.nodes.filter(node => !beforeNodes.has(node.node_id)));
  const removedLayers = sortedLayers(before.document.semantic.nodes.filter(node => !afterNodes.has(node.node_id)));
  const changedLayers: ChangedLayer[] = []; const recordedShapes: ChangedLayer[] = [];
  for (const next of sortedLayers(after.document.semantic.nodes)) {
    const previous = beforeNodes.get(next.node_id);
    if (!previous) continue;
    const fields = [...changes(structure(previous), structure(next)), ...changes(previous.params, next.params, 'parameters.')];
    if (fields.length) changedLayers.push({ nodeId: next.node_id, changes: fields });
    const evidence = changes(shapes(previous), shapes(next));
    if (evidence.length) recordedShapes.push({ nodeId: next.node_id, changes: evidence });
  }
  const beforeEdges = connections(before); const afterEdges = connections(after);
  const addedConnections: ChangedConnection[] = []; const removedConnections: ChangedConnection[] = [];
  for (const key of Array.from(new Set([...Array.from(beforeEdges.keys()), ...Array.from(afterEdges.keys())])).sort(order)) {
    const a = beforeEdges.get(key); const b = afterEdges.get(key);
    const beforeCount = a?.count ?? 0; const afterCount = b?.count ?? 0;
    if (beforeCount === afterCount) continue;
    const item = { target: (b ?? a)!.target, count: Math.abs(afterCount - beforeCount), beforeCount, afterCount };
    (afterCount > beforeCount ? addedConnections : removedConnections).push(item);
  }
  return { addedLayers, removedLayers, changedLayers, addedConnections, removedConnections, recordedShapes,
    uncertainMatches: uncertainMatches(removedLayers, addedLayers), modelNamesDiffer: before.document.semantic.name !== after.document.semantic.name,
    snapshotEvidenceChanged: before.status !== after.status || before.warningCount !== after.warningCount,
    architectureChanged: Boolean(addedLayers.length || removedLayers.length || changedLayers.length || addedConnections.length || removedConnections.length) };
}
