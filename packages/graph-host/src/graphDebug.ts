import { captureJson } from '@tensorviz/contracts/captureCorrespondence';
import { compareCaptures } from '@tensorviz/contracts/captureComparison';
import type { CaptureSnapshot } from './semanticCapture';
import type { GraphDebugContext, ProjectionPayload } from '@tensorviz/contracts/protocol';

export interface GraphDebugBaseline { projection: ProjectionPayload; capture: CaptureSnapshot }
type Node = { node_id: string; metadata?: { captureRef?: { kind: string; semanticId: string } } };
const nodes = (projection: ProjectionPayload): Node[] =>
  (projection.document as { semantic?: { nodes?: Node[] } })?.semantic?.nodes ?? [];

/** Only validated retained captures reach this function. Cross-run ids and shape
 * similarity never establish correspondence or prove a parameter changed. */
export function graphDebugContext(previous: GraphDebugBaseline, projection: ProjectionPayload,
  capture?: CaptureSnapshot): GraphDebugContext {
  const result: GraphDebugContext = { nodePairs: [], changed: [] };
  if (!capture || capture.invocations.some(call => call.status === 'failed') || capture.status !== 'complete') {
    // Keep one bounded prior run, not a growing chain of historical documents.
    if (Buffer.byteLength(JSON.stringify(previous.projection.document)) <= 8 * 1024 * 1024) {
      result.previousDocument = previous.projection.document;
    }
  }
  if (!capture) return result;
  const index = (projection: ProjectionPayload) => new Map(nodes(projection).map(node =>
    [`${node.metadata?.captureRef?.kind}:${node.metadata?.captureRef?.semanticId}`, node.node_id]));
  const left = index(previous.projection), right = index(projection);
  const comparison = compareCaptures(previous.capture, capture);
  const correspondence = comparison.correspondence;
  for (const pair of correspondence.pairs) {
    const beforeId = left.get(`${pair.kind}:${pair.beforeId}`), afterId = right.get(`${pair.kind}:${pair.afterId}`);
    if (beforeId && afterId) result.nodePairs.push({ beforeId, afterId });
  }
  for (const kind of ['input', 'output']) {
    const beforeId = left.get(`${kind}:${previous.capture.id}`), afterId = right.get(`${kind}:${capture.id}`);
    if (beforeId && afterId) result.nodePairs.push({ beforeId, afterId });
  }
  for (const change of comparison.changes.filter(change => change.kind === 'operation')) {
    const nodeId = right.get(`operation:${change.afterId}`);
    const details = change.fields.filter(field => ['target', 'arguments'].includes(field.field))
      .map(field => field.field === 'target' ? 'Operation changed' : 'Recorded arguments changed');
    if (nodeId && details.length) result.changed.push({ nodeId, details });
  }
  const beforeModules = new Map(previous.capture.modules.map(module => [module.id, module]));
  const afterModules = new Map(capture.modules.map(module => [module.id, module]));
  const moduleDetails = new Map<string, string[]>();
  for (const pair of correspondence.pairs.filter(pair => pair.kind === 'module')) {
    const before = beforeModules.get(pair.beforeId)!, after = afterModules.get(pair.afterId)!;
    const a = before.initializerArguments, b = after.initializerArguments;
    if (a?.status !== 'observed' || b?.status !== 'observed') continue;
    const parameters = new Map([...(a.positional ?? []), ...(a.keywordOnly ?? [])].map(entry => [entry.name, entry.value]));
    const details: string[] = [];
    for (const { name, value } of [...(b.positional ?? []), ...(b.keywordOnly ?? [])]) {
      const old = parameters.get(name);
      if (!old || old.kind === 'opaque' || value.kind === 'opaque' || captureJson(old) === captureJson(value)) continue;
      const show = (literal: typeof value) => literal.kind === 'none' ? 'None' : 'value' in literal ? String(literal.value) : '?';
      details.push(`${name.slice(0, 64)}: ${show(old).slice(0, 64)} → ${show(value).slice(0, 64)}`);
    }
    if (details.length) moduleDetails.set(after.id, details);
  }
  // Constructor observations are optional. Named state can independently prove
  // a parameter's measured shape/dtype changed, never its authored scalar value.
  const matchedModules = new Set(correspondence.pairs.filter(pair => pair.kind === 'module').map(pair => pair.afterId));
  const owners = new Map<string, string[]>();
  for (const module of capture.modules.filter(module => matchedModules.has(module.id))) {
    for (const path of module.paths) owners.set(path, [...(owners.get(path) ?? []), module.id]);
  }
  const statePairs = new Set(correspondence.pairs.filter(pair => pair.kind === 'value' && pair.basis === 'state_names')
    .map(pair => pair.afterId));
  const values = new Map(capture.values.map(value => [value.id, value]));
  for (const change of comparison.observations.filter(change => change.kind === 'value' && statePairs.has(change.afterId))) {
    const value = values.get(change.afterId)!;
    if (value.logicalVersion !== 0) continue;
    for (const ref of value.state.filter(ref => ref.kind === 'parameter')) {
      const dot = ref.name.lastIndexOf('.'), path = dot < 0 ? '' : ref.name.slice(0, dot);
      const owner = owners.get(path);
      if (owner?.length !== 1) continue;
      const details = moduleDetails.get(owner[0]) ?? [];
      for (const field of change.fields.filter(field => ['shape', 'dtype'].includes(field.field))) {
        if (!field.before.present || !field.after.present || field.before.value == null || field.after.value == null) continue;
        const show = (item: unknown) => Array.isArray(item) ? `[${item.join(', ')}]` : String(item);
        details.push(`${ref.name.slice(dot + 1, dot + 65)} ${field.field}: ${show(field.before.value).slice(0, 64)} → ${show(field.after.value).slice(0, 64)}`);
      }
      if (details.length) moduleDetails.set(owner[0], Array.from(new Set(details)).slice(0, 16));
    }
  }
  for (const call of capture.invocations) {
    const nodeId = right.get(`invocation:${call.id}`), details = moduleDetails.get(call.moduleId);
    if (nodeId && details?.length) result.changed.push({ nodeId, details: details.slice(0, 16) });
  }
  return result;
}
