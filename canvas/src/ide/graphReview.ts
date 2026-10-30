import type { GraphReviewState, IdeMultifileSpan, ProjectionPayload } from './protocol.generated';
import type { ModelGraph, ModelNode, ModelNodeData, ModelEdge } from '../types';
import { graphDocumentV2ToModelGraph } from '../services/graphDocumentAdapter';
import { reconcileModuleView } from '../services/canvasLayout';
import type { GraphDocumentV2 } from '../types/graphDocument';
import { customCardModel } from '../canvas/customCard';
import { operationDisplayName } from '../canvas/operationDisplayName';
import { sourceParameterValues } from './cardParams';
import { sourceAssignmentCard, sourceAssignmentsForReview, type SourceAssignmentEntry } from './sourceAssignmentCards';

import { declarationLocation, declarationInventoryComplete, compareDeclarations, declarationLocations } from './reviewDeclarations';
/** The phrase a syntax error must show, extracted from a longer review note. */
export function reviewParseFailure(notes: readonly string[] | undefined): string | undefined {
  const note = notes?.find(item => /could not parse \S+:\d+/.test(item));
  return note?.match(/could not parse \S+:\d+/)?.[0];
}

export type ReviewMode = 'before' | 'diff' | 'after';
export type ReviewProjection = ProjectionPayload & { reviewGraph: ModelGraph; reviewId: string; capturedContext?: boolean; proposal?: boolean;
  /** A draft shown in a retained capture: the capture and structure it was proven against. */
  contextKey?: string };
export interface ReviewNode {
  status: 'added' | 'removed' | 'modified' | 'rewired' | 'unresolved' | 'unchanged';
  before?: string; after?: string; beforeName?: string;
  params: Record<string, { before: unknown; after: unknown }>;
  summary?: string;
  /** Proven input-edge changes, separate from this node's source/parameter edits. */
  inputChanges?: { added: number; removed: number };
  /** This declaration has multiple source writes; do not choose one captured instance. */
  ambiguousDeclaration?: boolean;
  /** Source declarations joined to exact recorded constructors, independent of lowered paths. */
  sourceDeclaration?: { before?: string; after?: string };
}
export interface ReviewChange { id: string; label: string; detail: string }
interface ReviewDeclaration { digest: string; ambiguous?: boolean }
export interface GraphReviewPresentation {
  projections: Record<ReviewMode, ReviewProjection>;
  changes: ReviewChange[];
  nodes: Map<string, ReviewNode>;
  counts: Record<ReviewNode['status'], number>;
  edgeCounts: { added: number; removed: number };
  /** Source change IDs can focus retained captured nodes without changing their identity;
   * a list focuses every call of one change together. */
  focusNodes?: Map<string, string | string[]>;
  /** Prefer a located change when other source edits cannot be mapped to the capture. */
  initialChangeId?: string;
  /** Draft only: the forward runs an opaque inherited base the projector did not inspect. */
  inheritedBase?: boolean;
}
const provenance = new Set(['_review_declaration', '_lineno', '_end_lineno', '_col_offset', '_end_col_offset', '_source_file', '_source_line', '_stmt_hash',
  '_shape_provenance', '_in_shape', '_out_shape', '_act_stats', '_grad_stats', '_review_binding', '_source_calls',
  '_review_scope', '_review_order', '_review_statement', '_review_scopes', '_source_arguments', '_source_result_call']);
const canonical = (value: unknown): string => JSON.stringify(value, (_key, item) => item && typeof item === 'object' && !Array.isArray(item)
  ? Object.fromEntries(Object.keys(item).sort().map(key => [key, item[key]])) : item);
const parameters = (node: ModelNode) => {
  const params = Object.fromEntries(Object.entries(sourceParameterValues(node.data.parameters))
    .filter(([key]) => !provenance.has(key) && !(key === '_raw_call' && node.data.parameters._review_call)));
  const call = params._review_call as { arguments?: Record<string, unknown> } | undefined;
  // The parser proves these literal constructor defaults from torch's signature.
  // Include them on both sides so explicit and omitted defaults compare equally.
  for (const [key, value] of Object.entries(call?.arguments ?? {})) {
    if (!(key in params) && (value === null || ['boolean', 'number', 'string'].includes(typeof value))) params[key] = value;
  }
  return params;
};
const spanOf = (projection: ProjectionPayload, id: string) => projection.spanmap[id] as IdeMultifileSpan | undefined;
const title = (node: ModelNode) => customCardModel(node.data.layerType.id, node.data.parameters)?.title
  || operationDisplayName(node.data) || node.data.layerType.displayName || node.data.layerType.name || node.id;
const statementIdentity = (node: ModelNode, projection: ProjectionPayload): string | undefined => {
  const span = spanOf(projection, node.id);
  return span?.digest ? `statement:${span.file}:${node.data.parentId ?? ''}:${canonical(node.data.parameters._review_scope ?? [])}:${span.digest}` : undefined;
};

/** The AST projector preserves named attribute identities; ordinal operation IDs
 * do not survive edits. Unique source bindings can survive expression edits;
 * other anonymous statements require an exact digest. Never use topology or shape. */
function identity(node: ModelNode, projection: ProjectionPayload): string | undefined {
  if (node.type === 'inputNode' || node.type === 'outputNode') return `boundary:${node.type}:${node.id}`;
  const span = spanOf(projection, node.id);
  if (node.data.parameters._review_declaration) return `declaration:${node.data.parameters._review_declaration}:${node.data.parentId ?? ''}`;
  if (span?.kind === 'decl' && !/(?:^|[./:_])\d+(?:$|[./:_])/.test(node.id) && !['@', '#', '[', ']'].some(part => node.id.includes(part))) return `attribute:${span.file}:${node.id}`;
  if (span?.kind === 'call' && node.data.parameters._review_binding) return `binding:${span.file}:${node.data.parentId ?? ''}:${canonical(node.data.parameters._review_binding)}`;
  return statementIdentity(node, projection);
}
function named(node: ModelNode, projection: ProjectionPayload): boolean { return identity(node, projection)?.startsWith('attribute:') === true; }
function buckets(nodes: ModelNode[], projection: ProjectionPayload, keyOf = identity): Map<string, ModelNode[]> {
  const result = new Map<string, ModelNode[]>();
  for (const node of nodes) { const key = keyOf(node, projection); if (key) result.set(key, [...result.get(key) ?? [], node]); }
  return result;
}

/** Pure source insertions require a unique exact embedding of the old operations
 * in the same local call context. Ordinal IDs, line numbers and shapes are not
 * identities. A replacement, reorder or ambiguous repeated anchor fails closed. */
function insertedOperations(before: ModelNode[], after: ModelNode[], left: ProjectionPayload, right: ProjectionPayload): Set<string> {
  const groups = (nodes: ModelNode[], projection: ProjectionPayload) => {
    const result = new Map<string, ModelNode[]>();
    for (const node of nodes) {
      const scopes = node.data.parameters._review_scopes;
      if (!Array.isArray(scopes)) continue;
      for (const entry of scopes) {
        if (typeof entry?.file === 'string' && Array.isArray(entry.scope))
          result.set(canonical([entry.file, node.data.parentId ?? '', entry.scope]), []);
      }
    }
    for (const node of nodes) {
      const params = node.data.parameters, span = spanOf(projection, node.id);
      if (span?.kind !== 'call' || !Array.isArray(params._review_scope) || typeof params._review_order !== 'number') continue;
      const key = canonical([span.file, node.data.parentId ?? '', params._review_scope]);
      result.set(key, [...result.get(key) ?? [], node]);
    }
    for (const group of Array.from(result.values())) group.sort((a, b) => Number(a.data.parameters._review_order) - Number(b.data.parameters._review_order));
    return result;
  };
  const oldGroups = groups(before, left), additions = new Set<string>();
  const signature = (node: ModelNode) => canonical([node.data.layerType.name, node.data.parameters._review_statement, parameters(node)]);
  for (const [key, current] of Array.from(groups(after, right))) {
    const previous = oldGroups.get(key);
    if (!previous || current.length <= previous.length) continue;
    const a = previous.map(signature), b = current.map(signature);
    const earliest: number[] = [], latest: number[] = [];
    let index = 0;
    for (const value of a) { while (index < b.length && b[index] !== value) index++; earliest.push(index++); }
    index = b.length - 1;
    for (const value of [...a].reverse()) { while (index >= 0 && b[index] !== value) index--; latest.unshift(index--); }
    if (!earliest.every((position, i) => position < b.length && position === latest[i])) continue;
    const retained = new Set(earliest);
    current.forEach((node, i) => { if (!retained.has(i)) additions.add(node.id); });
  }
  return additions;
}

type ReviewClassFacts = { defaults: Record<string, string>; digest: string; members?: Record<string, ReviewDeclaration & { span?: IdeMultifileSpan }> };
const classFactsOf = (projection: ProjectionPayload) => (projection.document as GraphDocumentV2).semantic.metadata.reviewClass as ReviewClassFacts | undefined;

/** A draft's added declarations the projector did not lower become visible,
 * unconnected cards at their declaration, never a hidden focus on the class. */
function withDeclaredCards(state: GraphReviewState): { state: GraphReviewState; declared: Set<string> } {
  const left = classFactsOf(state.before)?.members, right = classFactsOf(state.after)?.members;
  const document = state.after.document as GraphDocumentV2;
  const projected = new Set(document.semantic.nodes.map(node => node.node_id));
  const declared = new Set(Object.keys(right ?? {}).filter(name => left && !left[name] && !projected.has(name)
    && !right![name].ambiguous && right![name].span));
  if (!declared.size) return { state, declared };
  const names = Array.from(declared);
  const cards = names.map(name => ({ node_id: name, kind: 'layer', op: 'Custom', params: { _class: name, _review_declared: true },
    in_ports: [], out_ports: [], metadata: { parentId: null, layerType: { id: 'custom_code', name: 'custom_code', parameters: [] } } })) as unknown as GraphDocumentV2['semantic']['nodes'];
  return { declared, state: { ...state, after: { ...state.after,
    document: { ...document, semantic: { ...document.semantic, nodes: [...document.semantic.nodes, ...cards] } },
    spanmap: { ...state.after.spanmap, ...Object.fromEntries(names.map(name => [name, { ...right![name].span!, kind: 'decl' }])) } } } };
}

/** An assignment card navigates to its own statement on each side. */
function withAssignmentSpans(state: GraphReviewState, changes: SourceAssignmentEntry[]): GraphReviewState {
  if (!changes.length) return state;
  const spans = (side: 'before' | 'after') => Object.fromEntries(changes.flatMap(change => change[side] ? [[change.id, change[side]!.span]] : []));
  return { ...state, before: { ...state.before, spanmap: { ...state.before.spanmap, ...spans('before') } },
    after: { ...state.after, spanmap: { ...state.after.spanmap, ...spans('after') } } };
}

export function buildGraphReview(review: GraphReviewState): GraphReviewPresentation {
  const draft = !!review.proposal;
  const assignments = draft ? sourceAssignmentsForReview(review.before.document, review.after.document) : { entries: [], unresolved: false };
  const { state, declared } = draft ? withDeclaredCards(withAssignmentSpans(review, assignments.entries)) : { state: review, declared: new Set<string>() };
  const declarations = compareDeclarations(state.before, state.after, state.renamedFiles);
  const originalDeclarationKey = (side: 'before' | 'after', key: string) => side === 'before'
    ? declarations.originalBeforeKeys.get(key) ?? key : key;
  const located = { before: declarationLocations(declarations.sides.before, declarations.eligible),
    after: declarationLocations(declarations.sides.after, declarations.eligible) };
  const sourceGraph = (side: 'before' | 'after') => {
    const graph = graphDocumentV2ToModelGraph(state[side].document as GraphDocumentV2);
    return { ...graph, nodes: graph.nodes.map(node => {
      const span = spanOf(state[side], node.id), key = !draft && span && located[side].get(declarationLocation(span));
      return key ? { ...node, data: { ...node.data, parameters: { ...node.data.parameters, _review_declaration: key } } } : node;
    }) };
  };
  const before = sourceGraph('before'), after = sourceGraph('after');
  const beforeIdentity = state.renamedFiles ? { ...state.before, spanmap: Object.fromEntries(Object.entries(state.before.spanmap)
    .map(([id, value]) => { const span = value as IdeMultifileSpan; return [id, { ...span, file: state.renamedFiles![span.file] ?? span.file }]; })) } : state.before;
  const left = buckets(before.nodes, beforeIdentity), right = buckets(after.nodes, state.after);
  const matches = new Map<string, ModelNode>(), matchedBefore = new Set<string>();
  // Reserve stable bindings first. Adding a reassignment can remove a binding
  // identity without changing its original statement; exact source still pairs it.
  for (const [key, nodes] of Array.from(right)) {
    const previous = left.get(key);
    if (nodes.length === 1 && previous?.length === 1) {
      matches.set(nodes[0].id, previous[0]); matchedBefore.add(previous[0].id);
    }
  }
  if (before.nodes.some(node => !matchedBefore.has(node.id) && node.data.parameters._review_binding)
      || after.nodes.some(node => !matches.has(node.id) && node.data.parameters._review_binding)) {
    const callStatement = (node: ModelNode, projection: ProjectionPayload) => spanOf(projection, node.id)?.kind === 'call' ? statementIdentity(node, projection) : undefined;
    const oldStatements = buckets(before.nodes, beforeIdentity, callStatement);
    for (const [key, nodes] of Array.from(buckets(after.nodes, state.after, callStatement))) {
      const previous = oldStatements.get(key);
      if (nodes.length === 1 && previous?.length === 1 && !matches.has(nodes[0].id) && !matchedBefore.has(previous[0].id)) {
        matches.set(nodes[0].id, previous[0]); matchedBefore.add(previous[0].id);
      }
    }
  }
  // A draft's opaque inherited forward is neutral context: the same placeholder on
  // both sides, never an added, removed or rewired operation of an uninspected base.
  const inherited = (node: ModelNode) => draft && node.data.parameters._review_inherited === true;
  for (const node of after.nodes) {
    const previous = inherited(node) && !matches.has(node.id) ? before.nodes.find(item => item.id === node.id && inherited(item)) : undefined;
    if (previous && !matchedBefore.has(previous.id)) { matches.set(node.id, previous); matchedBefore.add(previous.id); }
  }
  const leftClass = classFactsOf(state.before), rightClass = classFactsOf(state.after);
  // Class membership proves a named declaration is new even when an external base
  // leaves the comparison incomplete. It says nothing about runtime behaviour.
  const declaredOnly = (node: ModelNode, projection: ProjectionPayload, own?: ReviewClassFacts, other?: ReviewClassFacts) => draft
    && named(node, projection) && !!own?.members?.[node.id] && !own.members[node.id].ambiguous && !!other?.members && !other.members[node.id];
  const beforeIds = new Map<string, string>(), afterIds = new Map<string, string>();
  const records = new Map<string, ReviewNode>();
  const originals = new Map<string, { before?: ModelNode; after?: ModelNode }>();
  const parseFailure = reviewParseFailure(state.notes);
  const complete = !parseFailure && state.modelStatus !== 'unresolved' && !state.before.truncated && !state.after.truncated && !(state.before.diagnostics?.length || state.after.diagnostics?.length);
  const sourceComplete = !parseFailure && state.modelStatus !== 'unresolved' && !state.before.truncated && !state.after.truncated;
  const addedModel = !parseFailure && state.modelStatus === 'added';
  const removedModel = !parseFailure && state.modelStatus === 'removed';
  const inserted = sourceComplete ? insertedOperations(before.nodes, after.nodes, beforeIdentity, state.after) : new Set<string>();
  const deleted = sourceComplete ? insertedOperations(after.nodes, before.nodes, state.after, beforeIdentity) : new Set<string>();
  for (const node of after.nodes) {
    const candidate = matches.get(node.id);
    const id = `review:after:${node.id}`;
    afterIds.set(node.id, id);
    if (candidate) beforeIds.set(candidate.id, id);
    const params: ReviewNode['params'] = {};
    if (candidate) {
      const a = parameters(candidate), b = parameters(node);
      for (const key of Array.from(new Set([...Object.keys(a), ...Object.keys(b)]))) if (canonical(a[key]) !== canonical(b[key])) params[key] = { before: a[key], after: b[key] };
    }
    const replaced = candidate && candidate.data.layerType.name !== node.data.layerType.name;
    const status = inherited(node) ? 'unchanged' : candidate ? (replaced || Object.keys(params).length ? 'modified' : 'unchanged')
      : addedModel || inserted.has(node.id) || (complete && named(node, state.after))
        || declaredOnly(node, state.after, rightClass, leftClass) ? 'added' : 'unresolved';
    const summary = params._review_guards ? 'Execution condition changed' : params._review_helpers ? 'Helper implementation changed'
      : params._raw_call ? 'Source expression changed' : undefined;
    records.set(id, { status, before: candidate?.id, after: node.id, params, ...(summary ? { summary } : {}), ...(replaced ? { beforeName: title(candidate) } : {}) });
    originals.set(id, { before: candidate, after: node });
  }
  for (const node of before.nodes) if (!beforeIds.has(node.id)) {
    const id = `review:before:${node.id}`;
    beforeIds.set(node.id, id);
    records.set(id, { status: inherited(node) ? 'unchanged' : removedModel || deleted.has(node.id) || (complete && named(node, state.before))
      || declaredOnly(node, beforeIdentity, leftClass, rightClass) ? 'removed' : 'unresolved', before: node.id, params: {} });
    originals.set(id, { before: node });
  }
  // Keep nested assignments in review navigation. Their location must be proven
  // by a capture; a disconnected card would imply a place in the architecture.
  for (const change of assignments.entries) {
    const { before: previous, after: current } = change;
    const id = `review:${current ? 'after' : 'before'}:${change.id}`;
    const value = !!previous && !!current && previous.value !== current.value;
    const summary = !current ? 'Assignment removed from source' : !previous ? 'Assignment set in source'
      : value ? 'Assignment value changed' : 'Execution condition changed';
    records.set(id, { status: change.unchanged ? 'unchanged' : 'modified', before: previous && change.id, after: current && change.id,
      params: value ? { value: { before: previous!.value, after: current!.value } } : {},
      summary: `${summary}${(current ?? previous)!.conditional ? ' · conditional' : ''} · not run` });
    originals.set(id, { before: previous && sourceAssignmentCard(change.id, previous), after: current && sourceAssignmentCard(change.id, current) });
  }
  const edgeKey = (edge: ModelEdge, ids: Map<string, string>) => canonical([ids.get(edge.source), edge.sourceHandle ?? edge.data?.sourceHandle ?? 'y', ids.get(edge.target), edge.targetHandle ?? edge.data?.targetHandle ?? 'x']);
  const previousEdges = new Map<string, ModelEdge[]>();
  before.edges.forEach(edge => { const key = edgeKey(edge, beforeIds); previousEdges.set(key, [...previousEdges.get(key) ?? [], edge]); });
  const edges: (ModelEdge & { reviewSides: ('before' | 'after')[] })[] = [];
  let addedEdges = 0, removedEdges = 0;
  const neutral = new Set(Array.from(originals).filter(([, pair]) => inherited(pair.after ?? pair.before!)).map(([id]) => id));
  // An input once fed by the uninspected base cannot prove a rewire.
  const opaqueInputs = new Set([...after.edges.filter(edge => neutral.has(afterIds.get(edge.source)!)).map(edge => afterIds.get(edge.target)!),
    ...before.edges.filter(edge => neutral.has(beforeIds.get(edge.source)!)).map(edge => beforeIds.get(edge.target)!)]);
  const markEndpoint = (id: string, direction: 'added' | 'removed') => {
    const item = id && records.get(id);
    if (!item || opaqueInputs.has(id) || !['unchanged', 'rewired', 'modified'].includes(item.status)) return;
    if (item.status === 'unchanged') item.status = 'rewired';
    item.inputChanges ??= { added: 0, removed: 0 };
    item.inputChanges[direction]++;
  };
  for (const edge of after.edges) {
    const counterpart = previousEdges.get(edgeKey(edge, afterIds))?.shift();
    const source = afterIds.get(edge.source)!, target = afterIds.get(edge.target)!;
    const resolved = records.get(source)?.status !== 'unresolved' && records.get(target)?.status !== 'unresolved' && !neutral.has(source) && !neutral.has(target);
    if (!counterpart && resolved) { addedEdges++; markEndpoint(target, 'added'); }
    edges.push({ ...edge, id: `review:after:edge:${edge.id}`, source, target,
      data: { ...edge.data, ...(!counterpart && resolved ? { diffStatus: 'added' as const } : {}) }, reviewSides: counterpart ? ['before', 'after'] : ['after'] });
  }
  for (const group of Array.from(previousEdges.values())) for (const edge of group) {
    const source = beforeIds.get(edge.source)!, target = beforeIds.get(edge.target)!;
    const resolved = records.get(source)?.status !== 'unresolved' && records.get(target)?.status !== 'unresolved' && !neutral.has(source) && !neutral.has(target);
    if (resolved) { removedEdges++; markEndpoint(target, 'removed'); }
    edges.push({ ...edge, id: `review:before:edge:${edge.id}`, source, target, data: { ...edge.data, ...(resolved ? { diffStatus: 'removed' as const } : {}) }, reviewSides: ['before'] });
  }
  const rootId = 'review:root';
  const prepare = (id: string, original: ModelNode, side: 'before' | 'after', diff: boolean): ModelNode => {
    const ids = side === 'before' ? beforeIds : afterIds;
    const record = records.get(id)!;
    const span = spanOf(state[side], original.id);
    const boundary = ['inputNode', 'outputNode'].includes(original.type ?? '');
    return { ...original, id, selected: false, hidden: false, position: { x: 0, y: 0 }, data: {
      ...original.data, captureLabel: original.data.captureLabel ?? original.id, ...(boundary ? {} : { parentId: original.data.parentId ? ids.get(original.data.parentId) ?? rootId : rootId }),
      ...(original.type === 'moduleNode' ? { moduleInstanceId: id, expanded: true,
        moduleMembers: original.data.moduleMembers?.map(member => ids.get(member)!).filter(Boolean),
        moduleEntryNodeId: ids.get(original.data.moduleEntryNodeId ?? ''), moduleExitNodeId: ids.get(original.data.moduleExitNodeId ?? '') } : {}),
      ...(span ? { sourceLocation: { file: span.file, line: span.start[0] } } : {}),
      shapeEvidence: 'unknown', inputShape: undefined, outputShape: undefined,
      ...(boundary ? { parameters: { ...original.data.parameters, shape: '' } } : {}),
      ...(diff ? { review: record, diffStatus: ['added', 'removed', 'modified', 'rewired'].includes(record.status) ? record.status as ModelNodeData['diffStatus'] : undefined } : { review: { ...record, status: 'unchanged', params: {}, beforeName: undefined, inputChanges: undefined, summary: undefined }, diffStatus: undefined }),
    } };
  };
  for (const record of Array.from(records.values())) {
    for (const side of ['before', 'after'] as const) {
      const span = record[side] && spanOf(state[side], record[side]!);
      const key = !draft && span && located[side].get(declarationLocation(span));
      if (key && declarations.containers.has(key)) { record.status = 'unchanged'; record.params = {}; }
    }
  }
  const counts = { added: 0, removed: 0, modified: 0, rewired: 0, unresolved: 0, unchanged: 0 };
  const changes: ReviewChange[] = [];
  const union = Array.from(originals).map(([id, pair]) => {
    const original = pair.after ?? pair.before!;
    const record = records.get(id)!;
    const sourceDetail = record.summary ?? (Object.keys(record.params).filter(key => !key.startsWith('_')).join(', ') || (record.beforeName ? 'Layer replaced' : 'Constructor arguments changed'));
    if (record.inputChanges) {
      const { added, removed } = record.inputChanges;
      const inputs = `Input connections changed (+${added} / −${removed})`;
      record.summary = record.status === 'rewired' ? `${inputs}. Source and parameters unchanged.` : `${sourceDetail} · ${inputs}`;
    }
    if ((!['inputNode', 'outputNode'].includes(original.type ?? '') || record.status === 'rewired') && !neutral.has(id)) {
      counts[record.status]++;
      // A draft has not run: an addition is a source declaration, not an observed runtime layer.
      const added = !draft ? 'added' : declared.has(original.id) ? 'Declared · not called in the projected forward' : 'Declared in source · not run';
      const assigned = original.data.sourceAssignment;
      if (record.status !== 'unchanged') changes.push({ id, label: assigned ? pair.after ? `${assigned.target} → ${title(original)}` : assigned.target
        : record.beforeName ? `${record.beforeName} → ${title(original)}` : title(original),
        detail: assigned ? `${record.summary} · Location not verified in graph`
          : record.status === 'unresolved' ? 'Correspondence unresolved' : record.status === 'modified' || record.status === 'rewired' ? record.summary ?? sourceDetail
          : record.status === 'added' ? added : record.status });
    }
    return prepare(id, original, pair.after ? 'after' : 'before', true);
  });
  const focusNodes = new Map<string, string>();
  for (const change of assignments.entries) {
    const fact = change.after ?? change.before!;
    const side = change.after ? after : before;
    // A source-level containing call can orient the user without claiming the
    // assignment targets that call. The review row keeps the exact target visible.
    const first = fact.path?.[0];
    const ancestor = first?.kind === 'attribute' ? side.nodes.filter(node => node.id === first.key) : [];
    const id = `review:${change.after ? 'after' : 'before'}:${change.id}`;
    focusNodes.set(id, ancestor.length === 1 ? (change.after ? afterIds : beforeIds).get(ancestor[0].id)! : rootId);
  }
  if (leftClass?.members && rightClass?.members) {
    const projected = new Set([...before.nodes, ...after.nodes].map(node => node.id));
    for (const name of Array.from(new Set([...Object.keys(leftClass.members), ...Object.keys(rightClass.members)]))) {
      const previous: ReviewDeclaration | undefined = leftClass.members[name];
      const current: ReviewDeclaration | undefined = rightClass.members[name];
      // Lowered cards already have constructor/connection comparison. Supplement
      // declarations the projector omitted, rather than inventing source topology.
      const span = spanOf(state.after, name) ?? spanOf(state.before, name);
      const declaration = span && (located.after.get(declarationLocation(span)) ?? located.before.get(declarationLocation(span)));
      if (projected.has(name) || previous?.digest === current?.digest || (declaration && declarations.containers.has(declaration))) continue;
      const ambiguousDeclaration: boolean = !!(previous?.ambiguous || current?.ambiguous);
      const status: ReviewNode['status'] = parseFailure || ambiguousDeclaration ? 'unresolved' : !previous ? 'added' : !current ? 'removed' : 'modified';
      const summary: string = ambiguousDeclaration ? 'Declaration has multiple assignments' : `Declaration ${status}`;
      const id: string = `review:member:${name}`;
      records.set(id, { status, before: previous ? name : undefined, after: current ? name : undefined,
        params: {}, summary, ambiguousDeclaration });
      counts[status]++;
      changes.push({ id, label: name, detail: summary });
      // Without a capture there is no measured node to place this change on.
      focusNodes.set(id, rootId);
    }
  }
  const declarationCompleteness = [state.before, state.after].map(declarationInventoryComplete);
  if (!draft && declarationCompleteness.every(value => value === true)) {
    const evidence = declarations.sides;
    const represented = new Set<string>();
    for (const side of ['before', 'after'] as const) {
      const locations = located[side];
      for (const [id, record] of Array.from(records)) {
        const span = record[side] && spanOf(state[side], record[side]!);
        // An ambiguous member row already reports the several writes that keep its
        // declaration out of capture evidence. That declaration is the same change;
        // another target of a chained write at the same statement is not.
        const label = `${state[side].cls}.${record[side]}`;
        const same = span && record.ambiguousDeclaration && id.startsWith('review:member:') && Object.keys(evidence[side]).find(key =>
          evidence[side][key].label === label && !evidence[side][key].parent && declarationLocation(evidence[side][key].span) === declarationLocation(span));
        if (same) represented.add(same);
        const key = span && locations.get(declarationLocation(span));
        if (!key) continue;
        record.sourceDeclaration = { ...record.sourceDeclaration, [side]: originalDeclarationKey(side, key) };
        represented.add(key);
      }
    }
    for (const key of Array.from(new Set([...Object.keys(evidence.before), ...Object.keys(evidence.after)]))) {
      const previous = evidence.before[key], current = evidence.after[key];
      if (represented.has(key) || declarations.containers.has(key) || previous?.digest === current?.digest) continue;
      const ambiguousDeclaration = !!(previous?.ambiguous || current?.ambiguous) || !declarations.eligible.has(key);
      const status = ambiguousDeclaration ? 'unresolved' : !previous ? 'added' : !current ? 'removed' : 'modified';
      const summary = ambiguousDeclaration ? 'Declaration has multiple assignments' : `Declaration ${status}`;
      const id = `review:declaration:${key}`;
      const sides = { before: previous ? originalDeclarationKey('before', key) : undefined, after: current ? key : undefined };
      records.set(id, { status, ...sides, sourceDeclaration: sides, ambiguousDeclaration, params: {}, summary });
      counts[status]++;
      changes.push({ id, label: (current ?? previous).label, detail: summary });
      focusNodes.set(id, rootId);
    }
  }
  const defaults: ReviewNode['params'] = {};
  if (leftClass && rightClass) {
    for (const key of Array.from(new Set([...Object.keys(leftClass.defaults), ...Object.keys(rightClass.defaults)]))) {
      if (leftClass.defaults[key] !== rightClass.defaults[key]) defaults[key] = { before: leftClass.defaults[key], after: rightClass.defaults[key] };
    }
  }
  const defaultSummary = Object.entries(defaults).map(([key, value]) => `${key} default: ${value.before ?? 'required'} → ${value.after ?? 'required'}`).join(', ');
  const unprojected = (!changes.length || assignments.unresolved) && leftClass && rightClass && leftClass.digest !== rightClass.digest;
  const classChange: ReviewNode | undefined = !draft && declarationCompleteness.includes(false) ? { status: 'unresolved',
    before: '@class', after: '@class', params: {}, summary: 'Source declaration comparison is incomplete. Inspect the code diff.' } : defaultSummary ? { status: 'modified', before: '@class', after: '@class', params: {},
    summary: defaultSummary } : unprojected ? { status: 'unresolved', before: '@class', after: '@class', params: {},
      summary: 'Source changed outside the projected layers. Inspect the code diff; runtime impact is unknown.' } : undefined;
  if (classChange) {
    records.set(rootId, classChange);
    counts[classChange.status]++;
    changes.unshift({ id: rootId, label: state.after.cls, detail: classChange.summary! });
  }
  const summary = [`+${counts.added} added`, `−${counts.removed} removed`, `${counts.modified} modified`, ...(counts.rewired ? [`${counts.rewired} inputs changed`] : []), ...(counts.unresolved ? [`${counts.unresolved} unresolved`] : [])].join(' · ');
  const root: ModelNode = { id: rootId, type: 'moduleNode', position: { x: 0, y: 0 }, data: {
    layerType: { id: 'module', name: state.after.cls, displayName: state.after.cls, description: '', icon: 'Box', category: 'module', parameters: [] },
    parameters: {}, captureLabel: state.after.cls, isGroup: true, expanded: true, moduleInstanceId: rootId, moduleTemplateId: state.after.cls,
    review: classChange ?? { status: 'unchanged', params: {}, summary },
    ...(classChange ? { diffStatus: classChange.status === 'modified' ? 'modified' as const : undefined } : {}),
  } };
  const laid = reconcileModuleView([root, ...union.filter(node => !node.data.sourceAssignment)], edges, { relayout: true });
  const positions = new Map(laid.nodes.map(node => [node.id, node]));
  const projections = {} as Record<ReviewMode, ReviewProjection>;
  for (const mode of ['before', 'diff', 'after'] as const) {
    const nodes = Array.from(originals).flatMap(([id, pair]) => {
      const original = mode === 'before' ? pair.before : mode === 'after' ? pair.after : pair.after ?? pair.before;
      if (!original || original.data.sourceAssignment) return [];
      const node = prepare(id, original, mode === 'before' || !pair.after ? 'before' : 'after', mode === 'diff');
      return [{ ...node, position: positions.get(id)!.position, width: positions.get(id)!.width, height: positions.get(id)!.height }];
    });
    // A draft previews the proposed graph: a removed connection between cards that
    // remain would only duplicate the old path beside the new one.
    const superseded = (edge: typeof edges[number]) => draft && !edge.reviewSides.includes('after')
      && !!originals.get(edge.source)?.after && !!originals.get(edge.target)?.after;
    const graph = { nodes: [positions.get(rootId)!, ...nodes], edges: edges.filter(edge => mode === 'diff' ? !superseded(edge) : edge.reviewSides.includes(mode))
      .map(edge => ({ ...edge, data: { ...edge.data, ...(mode !== 'diff' ? { diffStatus: undefined } : {}) } })) };
    projections[mode] = { ...state[mode === 'before' ? 'before' : 'after'], graphId: `review:${state.id}`, reviewId: state.id,
      viewRestored: true, reviewGraph: graph, ...(draft ? { proposal: true } : {}),
      spanmap: Object.fromEntries(Array.from(records).flatMap(([id, record]) => {
        const side = mode === 'before' || !record.after ? 'before' : 'after';
        const original = record[side]; const span = original && state[side].spanmap[original]; return span ? [[id, span]] : [];
      })),
    };
  }
  return { projections, changes, nodes: records, counts, edgeCounts: { added: addedEdges, removed: removedEdges },
    ...(focusNodes.size ? { focusNodes } : {}), ...(neutral.size ? { inheritedBase: true } : {}) };
}
