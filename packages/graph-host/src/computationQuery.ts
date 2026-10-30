import { z } from 'zod';
import type { CaptureSnapshot } from './semanticCapture';
import { GraphToolError } from './mcpGraph';

const reference = z.discriminatedUnion('kind', [
  z.object({ kind: z.enum(['operation', 'value', 'invocation', 'module', 'definition']),
    id: z.string().min(1).max(128) }).strict(),
  z.object({ kind: z.enum(['parameter', 'buffer']), id: z.string().min(1).max(4096) }).strict(),
]);
export const computationQuerySchema = z.object({
  captureId: z.string().min(1).max(128), sourceId: z.string().min(1).max(128),
  anchors: z.array(reference).min(1).max(16),
  direction: z.enum(['upstream', 'downstream', 'both']).default('both'),
  depth: z.number().int().min(0).max(20).default(4),
  maxEntities: z.number().int().min(1).max(500).default(100),
  offset: z.number().int().min(0).max(1_000_000).default(0),
  limit: z.number().int().min(1).max(100).default(50),
}).strict();
export const computationFindSchema = z.object({
  captureId: z.string().min(1).max(128), sourceId: z.string().min(1).max(128),
  text: z.string().min(1).max(256).regex(/\S/),
  kinds: z.array(z.enum(['module', 'definition', 'operation', 'parameter', 'buffer'])).min(1).max(5)
    .default(['module', 'definition', 'operation', 'parameter', 'buffer']),
  offset: z.number().int().min(0).max(1_000_000).default(0),
  limit: z.number().int().min(1).max(100).default(25),
}).strict();
type Query = z.infer<typeof computationQuerySchema>;
type Ref = z.infer<typeof reference>;
type Path = Array<string | number>;
type Argument = CaptureSnapshot['operations'][number]['output'];
export interface Edge {
  kind: 'edge'; source: Ref; target: Ref;
  relation: 'argument' | 'output' | 'effect' | 'effect_input' | 'previous_version';
  path: Path;
}
interface RecordEntry { ref: Ref; record: unknown; path: Path }
type RecordItem = { kind: 'entity' | 'anchor' | 'context' | 'source'; ref: { kind: string; id: string };
  inspect: { captureId: string; sourceId: string; path: Path }; record?: unknown; omitted?: 'record_bytes' };
type Item = RecordItem | Edge | { kind: 'boundary'; ref: Ref; via: Edge; reason: 'depth_limit' | 'entity_limit' };
const key = (ref: Ref) => JSON.stringify([ref.kind, ref.id]);
const bytes = (value: unknown) => Buffer.byteLength(JSON.stringify(value));
const PAGE_BYTES = 48 * 1024, RECORD_BYTES = 4096, RELATION_LIMIT = 5000;

/** Canonical value references, never projected/layout edges, define traversal.
 * Instances and calls are anchors, not fabricated tensor-flow vertices. */
export class ComputationIndex {
  private readonly records = new Map<string, RecordEntry>();
  private readonly incoming = new Map<string, Edge[]>();
  private readonly outgoing = new Map<string, Edge[]>();
  private readonly edges: Edge[] = [];
  private readonly operationsByCall = new Map<string, Ref[]>();
  private readonly childrenByCall = new Map<string, string[]>();
  private readonly callsByModule = new Map<string, string[]>();
  private readonly modulesByDefinition = new Map<string, string[]>();
  private readonly valuesByState = new Map<string, Ref[]>();
  private readonly bindings = new Map<string, { record: unknown; index: number }>();

  constructor(private readonly capture: CaptureSnapshot) {
    const collections = { operation: 'operations', value: 'values', invocation: 'invocations',
      module: 'modules', definition: 'definitions' } as const;
    for (const [kind, collection] of Object.entries(collections)) {
      capture[collection].forEach((record, index) => {
        const ref = { kind: kind as Ref['kind'], id: record.id };
        this.records.set(key(ref), { ref, record, path: [collection, index] });
      });
    }
    capture.sourceBindings.forEach((record, index) => this.bindings.set(record.id, { record, index }));
    for (const module of capture.modules) this.append(this.modulesByDefinition, module.definitionId, module.id);
    for (const call of capture.invocations) {
      this.append(this.callsByModule, call.moduleId, call.id);
      if (call.parentId !== null) this.append(this.childrenByCall, call.parentId, call.id);
    }
    for (const op of capture.operations) {
      const target: Ref = { kind: 'operation', id: op.id };
      if (op.invocationId !== null) this.append(this.operationsByCall, op.invocationId, target);
      const walk = (argument: Argument, path: Path): void => {
        if (argument === null) return;
        if (argument.kind === 'value') this.edge({ kind: 'edge', source: { kind: 'value', id: argument.id },
          target, relation: 'argument', path });
        else if (argument.kind === 'list' || argument.kind === 'tuple') {
          argument.items.forEach((item, i) => walk(item, [...path, 'items', i]));
        } else if (argument.kind === 'dict') {
          argument.entries.forEach((entry, i) => walk(entry.value, [...path, 'entries', i, 'value']));
        }
      };
      op.arguments.args.forEach((argument, i) => walk(argument, ['args', i]));
      Object.entries(op.arguments.kwargs).forEach(([name, argument]) => walk(argument, ['kwargs', name]));
      op.effects.forEach((effect, i) => this.edge({ kind: 'edge', source: { kind: 'value', id: effect.beforeId },
        target, relation: 'effect_input', path: ['effects', i, 'beforeId'] }));
    }
    for (const [index, value] of capture.values.entries()) {
      const target: Ref = { kind: 'value', id: value.id };
      value.state.forEach((state, stateIndex) => {
        const ref: Ref = { kind: state.kind, id: state.name };
        if (!this.records.has(key(ref))) this.records.set(key(ref), {
          ref, record: state, path: ['values', index, 'state', stateIndex],
        });
        this.append(this.valuesByState, key(ref), target);
      });
      if (value.producer) this.edge({ kind: 'edge', source: { kind: 'operation', id: value.producer.operationId },
        target, relation: value.producer.kind, path: value.producer.path });
      if (value.previousVersion !== null) this.edge({ kind: 'edge', source: { kind: 'value', id: value.previousVersion },
        target, relation: 'previous_version', path: ['previousVersion'] });
    }
  }

  requireCapture(capture: CaptureSnapshot): void {
    if (capture !== this.capture) throw new GraphToolError('INVALID_CAPTURE', 'The analysis index belongs to another capture object.');
  }

  private append<T>(map: Map<string, T[]>, id: string, value: T): void {
    const list = map.get(id);
    if (list) list.push(value); else map.set(id, [value]);
  }

  private edge(edge: Edge): void {
    if (!this.records.has(key(edge.source)) || !this.records.has(key(edge.target))) {
      throw new GraphToolError('INVALID_CAPTURE', 'A computation reference has no retained entity.');
    }
    this.edges.push(edge);
    this.append(this.outgoing, key(edge.source), edge);
    this.append(this.incoming, key(edge.target), edge);
  }

  private require(ref: Ref): RecordEntry {
    const entry = this.records.get(key(ref));
    if (!entry && (ref.kind === 'parameter' || ref.kind === 'buffer')) {
      throw new GraphToolError('STATE_NOT_OBSERVED', 'No value in this capture records that state kind and registration name. This does not establish absence from the model; inspect source and capture coverage.');
    }
    if (!entry) throw new GraphToolError('UNKNOWN_ENTITY', 'Use an entity from this exact canonical capture.');
    return entry;
  }

  private *seeds(ref: Ref): Generator<Ref> {
    this.require(ref);
    if (ref.kind === 'operation' || ref.kind === 'value') { yield ref; return; }
    if (ref.kind === 'parameter' || ref.kind === 'buffer') { yield* this.valuesByState.get(key(ref)) ?? []; return; }
    const modules = ref.kind === 'definition' ? this.modulesByDefinition.get(ref.id) ?? [] : [ref.id];
    const calls = ref.kind === 'invocation' ? [ref.id] : modules.flatMap(id => this.callsByModule.get(id) ?? []);
    const visited = new Set<string>();
    for (let index = 0; index < calls.length; index++) {
      const id = calls[index];
      if (visited.has(id)) continue;
      visited.add(id);
      yield* this.operationsByCall.get(id) ?? [];
      for (const child of this.childrenByCall.get(id) ?? []) calls.push(child);
    }
  }

  private record(kind: 'entity' | 'anchor' | 'context' | 'source', ref: { kind: string; id: string }, record: unknown, path: Path): RecordItem {
    const item = { kind, ref, inspect: { captureId: this.capture.id, sourceId: this.capture.source.id, path } };
    return bytes(record) > RECORD_BYTES ? { ...item, omitted: 'record_bytes' } : { ...item, record };
  }

  /** Literal name discovery returns candidates, never guessed identity or edges. */
  find(requested: unknown) {
    const parsed = computationFindSchema.safeParse(requested);
    if (!parsed.success) throw new GraphToolError('INVALID_QUERY', 'Use bounded literal name text, entity kinds and pagination.');
    const query = parsed.data, capture = this.capture;
    if (query.captureId !== capture.id || query.sourceId !== capture.source.id) {
      throw new GraphToolError('SOURCE_MISMATCH', 'Use the exact retained capture and source identities.');
    }
    const kinds = new Set<string>(query.kinds);
    const matches: Array<{ entry: RecordEntry; matchedFields: string[]; omittedMatchedFields: number }> = [];
    for (const entry of this.records.values()) {
      if (!kinds.has(entry.ref.kind)) continue;
      const matchedFields: string[] = [];
      let count = 0;
      const match = (field: string, value: string) => {
        if (!value.includes(query.text)) return;
        count++;
        if (matchedFields.length < 16) matchedFields.push(field);
      };
      if (entry.ref.kind === 'module') {
        const module = entry.record as CaptureSnapshot['modules'][number];
        module.paths.forEach((name, index) => match(`paths[${index}]`, name));
        const definition = this.require({ kind: 'definition', id: module.definitionId }).record as CaptureSnapshot['definitions'][number];
        match('definition.qualifiedName', definition.qualifiedName);
      } else if (entry.ref.kind === 'definition') {
        match('qualifiedName', (entry.record as CaptureSnapshot['definitions'][number]).qualifiedName);
      } else if (entry.ref.kind === 'operation') {
        match('target', (entry.record as CaptureSnapshot['operations'][number]).target);
      } else {
        match('name', entry.ref.id);
      }
      if (count) matches.push({ entry, matchedFields, omittedMatchedFields: count - matchedFields.length });
    }
    const page = { captureId: capture.id, sourceId: capture.source.id, text: query.text,
      kinds: [...kinds], matching: 'case_sensitive_literal_substring' as const,
      evidence: { backend: capture.backend, status: capture.status, scope: capture.coverage.scope, planDigest: capture.planDigest,
        appliesToCurrentRuntime: 'not_established' as const,
        coverageInspection: { captureId: capture.id, sourceId: capture.source.id, path: ['coverage'] },
        limitation: 'Names belong to this retained capture. No match does not establish absence from the model. Matches are candidates, not identity equivalence or shared state.' },
      total: matches.length, items: [] as Array<RecordItem & { matchedFields: string[]; omittedMatchedFields: number }>,
      nextOffset: null as number | null };
    let size = bytes(page) + 128;
    for (const { entry, matchedFields, omittedMatchedFields } of matches.slice(query.offset, query.offset + query.limit)) {
      const item = { ...this.record('anchor', entry.ref, entry.record, entry.path), matchedFields, omittedMatchedFields };
      const added = bytes(item) + 1;
      if (size + added > PAGE_BYTES) break;
      page.items.push(item); size += added;
    }
    if (size > PAGE_BYTES || (query.offset < matches.length && !page.items.length)) {
      throw new GraphToolError('RESULT_TOO_LARGE', 'The computation lookup metadata exceeds the bounded page size.');
    }
    page.nextOffset = query.offset + page.items.length < matches.length ? query.offset + page.items.length : null;
    return page;
  }

  /** One bounded selection shared by transport paging and deterministic analysis. */
  select(requested: unknown) {
    const parsed = computationQuerySchema.safeParse(requested);
    if (!parsed.success) throw new GraphToolError('INVALID_QUERY', 'Use bounded canonical anchors, traversal depth and pagination.');
    const query: Query = parsed.data, capture = this.capture;
    if (query.captureId !== capture.id || query.sourceId !== capture.source.id) {
      throw new GraphToolError('SOURCE_MISMATCH', 'Use the exact retained capture and source identities.');
    }
    if (new Set(query.anchors.map(key)).size !== query.anchors.length) {
      throw new GraphToolError('INVALID_QUERY', 'Use distinct canonical anchors.');
    }
    const seenSeeds = new Set<string>();
    const selected = new Map<string, { ref: Ref; distance: number }>();
    let omittedSeeds = 0;
    for (const anchor of query.anchors) {
      for (const ref of this.seeds(anchor)) {
        if (seenSeeds.has(key(ref))) continue;
        seenSeeds.add(key(ref));
        if (selected.size === query.maxEntities) { omittedSeeds++; continue; }
        selected.set(key(ref), { ref, distance: 0 });
      }
    }
    // Map iteration visits newly appended entries in deterministic BFS order.
    for (const { ref, distance } of selected.values()) {
      if (distance === query.depth) continue;
      for (const edge of this.neighbors(ref, query.direction)) {
        const other = key(edge.source) === key(ref) ? edge.target : edge.source;
        if (!selected.has(key(other)) && selected.size < query.maxEntities) selected.set(key(other), { ref: other, distance: distance + 1 });
      }
    }
    const items: Item[] = query.anchors.filter(ref => ref.kind !== 'operation' && ref.kind !== 'value')
      .map(ref => { const entry = this.require(ref); return this.record('anchor', ref, entry.record, entry.path); });
    const bindingIds = new Set<string>();
    const collectBindings = (entry: RecordEntry) => {
      const record = entry.record as { sourceBindingIds?: string[]; constructorBindingIds?: string[] };
      for (const id of [...record.sourceBindingIds ?? [], ...record.constructorBindingIds ?? []]) bindingIds.add(id);
    };
    for (const ref of query.anchors) collectBindings(this.require(ref));
    for (const { ref } of selected.values()) {
      const entry = this.require(ref);
      items.push(this.record('entity', ref, entry.record, entry.path));
      collectBindings(entry);
    }
    // Source ownership is context, not a fabricated tensor-flow edge. Include
    // direct call/instance/definition facts so their bindings are discoverable.
    const context = new Map<string, RecordEntry>();
    const owners = (ref: Ref): void => {
      const entry = this.require(ref);
      if (context.has(key(ref))) return;
      context.set(key(ref), entry);
      if (ref.kind === 'operation') {
        const op = entry.record as CaptureSnapshot['operations'][number];
        if (op.invocationId !== null) owners({ kind: 'invocation', id: op.invocationId });
      } else if (ref.kind === 'invocation') {
        owners({ kind: 'module', id: (entry.record as CaptureSnapshot['invocations'][number]).moduleId });
      } else if (ref.kind === 'module') {
        owners({ kind: 'definition', id: (entry.record as CaptureSnapshot['modules'][number]).definitionId });
      }
    };
    for (const ref of [...query.anchors, ...[...selected.values()].map(value => value.ref)]) owners(ref);
    const explicit = new Set(query.anchors.map(key));
    for (const [id, entry] of context) {
      if (!selected.has(id) && !explicit.has(id)) items.push(this.record('context', entry.ref, entry.record, entry.path));
      collectBindings(entry);
    }
    let internalEdges = 0, boundaryEdges = 0, omittedRelations = 0;
    for (const edge of this.edges) {
      const source = selected.get(key(edge.source)), target = selected.get(key(edge.target));
      if (source && target) {
        internalEdges++;
        if (internalEdges + boundaryEdges <= RELATION_LIMIT) items.push(edge); else omittedRelations++;
      } else if ((source && query.direction !== 'upstream') || (target && query.direction !== 'downstream')) {
        const inside = source ?? target!;
        boundaryEdges++;
        if (internalEdges + boundaryEdges <= RELATION_LIMIT) items.push({ kind: 'boundary', ref: source ? edge.target : edge.source,
          via: edge, reason: inside.distance === query.depth ? 'depth_limit' : 'entity_limit' });
        else omittedRelations++;
      }
    }
    for (const id of bindingIds) {
      const binding = this.bindings.get(id);
      if (!binding) throw new GraphToolError('INVALID_CAPTURE', 'A source binding has no retained record.');
      items.push(this.record('source', { kind: 'sourceBinding', id }, binding.record, ['sourceBindings', binding.index]));
    }
    const page = {
      captureId: capture.id, sourceId: capture.source.id, anchors: query.anchors,
      direction: query.direction, depth: query.depth,
      evidence: { backend: capture.backend, status: capture.status, scope: capture.coverage.scope,
        planDigest: capture.planDigest, preparationDigest: capture.preparationDigest,
        appliesToCurrentRuntime: 'not_established' as const,
        omissions: capture.coverage.omissions.slice(0, 32), omissionCount: capture.coverage.omissions.length,
        coverageInspection: { captureId: capture.id, sourceId: capture.source.id, path: ['coverage'] } },
      traversal: { entities: selected.size, internalEdges, boundaryEdges, omittedSeeds, omittedRelations,
        completeWithinRecordedGraph: omittedSeeds === 0 && boundaryEdges === 0 && omittedRelations === 0,
        continuation: 'Page this query with nextOffset. Expand boundary refs using the same capture/source. Inspect exact records for omitted relation details.' },
      total: items.length, items, nextOffset: null as number | null,
    };
    return { query, page };
  }

  query(requested: unknown) {
    const { query, page } = this.select(requested);
    return pageComputation(page, query.offset, query.limit);
  }

  private neighbors(ref: Ref, direction: Query['direction']): Edge[] {
    return [...direction !== 'downstream' ? this.incoming.get(key(ref)) ?? [] : [],
      ...direction !== 'upstream' ? this.outgoing.get(key(ref)) ?? [] : []];
  }
}

/** Callers use the validated query bounds; source records remain whole. */
export function pageComputation(selection: ReturnType<ComputationIndex['select']>['page'], offset: number, limit: number) {
  const page = { ...selection, items: [] as Item[] };
  let size = bytes(page) + 128;
  for (const item of selection.items.slice(offset, offset + limit)) {
    const added = bytes(item) + 1;
    if (size + added > PAGE_BYTES) break;
    page.items.push(item); size += added;
  }
  if (size > PAGE_BYTES || (offset < selection.items.length && page.items.length === 0)) {
    throw new GraphToolError('RESULT_TOO_LARGE', 'The computation query metadata exceeds the bounded page size.');
  }
  page.nextOffset = offset + page.items.length < selection.items.length ? offset + page.items.length : null;
  return page;
}
