import { createHash } from 'node:crypto';
import { z } from 'zod';
import { GraphToolError, type ModelRef } from './mcpGraph';
import type { CaptureSnapshot } from './semanticCapture';
import { sourceConfigurationImpact } from './sourceConfigurationImpact';
import { sourceShapeImpact } from './sourceShapeImpact';
import { sourceComputationImpact } from './sourceComputationImpact';

const digest = z.string().regex(/^[a-f0-9]{64}$/);
const id = z.string().regex(/^source:[a-f0-9]{64}$/);
const file = z.string().min(1).max(512);
const count = z.number().int().min(0).max(100_000);
const integer = z.number().int().min(-Number.MAX_SAFE_INTEGER).max(Number.MAX_SAFE_INTEGER);
export const sourceContextQuerySchema = z.object({
  file, symbol: z.string().min(1).max(256).optional(), line: z.number().int().min(1).optional(),
  anchorIds: z.array(id).min(1).max(16).optional(), indexId: digest.optional(),
  direction: z.enum(['upstream', 'downstream', 'both']).default('both'), depth: z.number().int().min(0).max(8).default(4),
  maxEntities: z.number().int().min(1).max(500).default(100), offset: count.default(0), limit: z.number().int().min(1).max(100).default(50),
  assumeValue: integer.optional(),
}).strict().refine(query => query.anchorIds
  ? !!query.indexId && !query.symbol && !query.line && new Set(query.anchorIds).size === query.anchorIds.length
  : !!query.symbol, 'Use a symbol, or distinct anchors with the exact indexId.')
  .refine(query => query.assumeValue === undefined || !query.anchorIds && query.direction === 'downstream',
    'Configuration changes require a downstream symbol selector.');
export type SourceContextQuery = z.input<typeof sourceContextQuerySchema>;
export interface IndexedSource { file: string; source: string; digest: string }
const relationSchema = z.object({ kind: z.literal('relation'), source: id, target: id,
  relation: z.enum(['import', 'reference', 'assignment', 'argument', 'expression', 'default']),
  certainty: z.enum(['lexical', 'ambiguous', 'possible', 'default_only']),
  argumentPath: z.array(z.union([z.string().max(256), count])).max(3).optional(),
}).strict();
const anchorSchema = z.object({ file, digest, startLine: z.number().int().min(1), endLine: z.number().int().min(1),
  startColumn: z.number().int().min(0).max(262144), endColumn: z.number().int().min(0).max(262144), scope: z.string().max(512), tokenDigest: digest.nullable() }).strict();
const scalarChange = z.union([
  z.object({ status: z.literal('derived'), before: integer, after: integer, reason: z.literal('integer_expression') }).strict(),
  z.object({ status: z.literal('unknown'), before: z.null(), after: z.null(), reason: z.enum([
    'integer_limit', 'unsupported_definition', 'analysis_limit', 'default_parameter', 'unresolved_binding',
    'division_by_zero', 'unsupported_expression',
  ]) }).strict(),
]);
const argumentPath = z.union([z.tuple([z.literal('args'), count]), z.tuple([z.literal('kwargs'), z.string().min(1).max(256)]), z.tuple([z.literal('unpacking')])]);
const leafPath = z.union([...argumentPath.options,
  z.tuple([z.literal('args'), count, z.literal('items'), count]),
  z.tuple([z.literal('kwargs'), z.string().min(1).max(256), z.literal('items'), count]),
  z.tuple([z.literal('unpacking'), z.literal('items'), count]),
]);
const configurationChange = z.union([
  z.object({ kind: z.literal('scalar') }).extend(scalarChange.options[0].shape).strict(),
  z.object({ kind: z.literal('scalar') }).extend(scalarChange.options[1].shape).strict(),
  z.object({ kind: z.literal('call'), arguments: z.array(z.union([
    scalarChange.options[0].extend({ path: argumentPath }), scalarChange.options[1].extend({ path: argumentPath }),
  ])).max(16), omittedArguments: count, hasExpansion: z.boolean(),
    argumentValues: z.array(z.union([scalarChange.options[0].extend({ path: leafPath }),
      scalarChange.options[1].extend({ path: leafPath })])).max(32).optional(),
    omittedArgumentValues: count.optional(),
  }).strict().refine(change => (change.argumentValues === undefined) === (change.omittedArgumentValues === undefined)
    && (!change.argumentValues || new Set(change.argumentValues.map(value => JSON.stringify(value.path))).size === change.argumentValues.length)),
]);
const recordSchema = z.object({ id, kind: z.enum(['definition', 'reference', 'call', 'expression']),
  name: z.string().max(256), anchor: anchorSchema,
  bindingKind: z.enum(['assignment', 'parameter', 'class', 'function', 'import', 'deletion', 'conditional_binding']).optional(),
  resolution: z.enum(['attribute_write', 'unresolved_import', 'unresolved_scope', 'unresolved_name', 'lexical', 'ambiguous', 'possible_attribute', 'unresolved_attribute']).optional(),
  excerpt: z.string().refine(text => Buffer.byteLength(text) <= 2048).optional(),
  omitted: z.literal('excerpt_bytes').optional(), literalSource: z.string().max(256).optional(),
  configurationChange: configurationChange.optional(),
}).strict();
const pageSchema = z.object({ version: z.literal(1), indexId: digest, evidence: z.literal('source_only'), scope: z.literal('supplied_python_files'),
  files: z.array(z.object({ file, digest }).strict()).min(1).max(16),
  assumptions: z.array(z.string().max(512)).max(8), limitations: z.array(z.string().max(512)).max(8),
  diagnostics: z.array(z.object({ file, line: z.number().int().min(1),
    reason: z.enum(['nested_expression_scope', 'dynamic_namespace', 'star_import', 'unresolved_import', 'unsupported_binding_form']),
    module: z.string().max(512).optional() }).strict()).max(32), omittedDiagnostics: count,
  traversal: z.object({ entities: count, boundaries: count, omittedSeeds: count }).strict(), total: count,
  items: z.array(z.union([recordSchema, relationSchema, z.object({ kind: z.literal('boundary'), anchorId: id,
    via: relationSchema, reason: z.enum(['depth_limit', 'entity_limit']) }).strict()])).max(100), nextOffset: count.nullable(),
  changeAnalysis: z.object({ version: z.literal(1), evidence: z.literal('conditional_source_arithmetic'), definitionId: id,
    before: integer, after: integer, scope: z.literal('source_records_on_this_page'),
    assumptions: z.array(z.string().max(512)).max(8), limitations: z.array(z.string().max(512)).max(8) }).strict().optional(),
}).strict();
export type SourceContextPage = z.infer<typeof pageSchema>;
export type SourceContextResult = SourceContextPage & { collection: { availableFiles: number; omittedFiles: number; byteLimit: number; fileLimit: number } };
export type SourceContextReader = (ref: ModelRef, query: SourceContextQuery) => Promise<SourceContextResult>;

/** A response must describe exactly the buffers sent, including its excerpts.
 * Analysis is performed by the authenticated parser; it never supplies file authority. */
export function sourceContextPage(value: unknown, sources: IndexedSource[], requested: SourceContextQuery): SourceContextPage {
  const parsed = pageSchema.safeParse(value), query = sourceContextQuerySchema.parse(requested);
  const refuse = (): never => { throw new GraphToolError('SOURCE_CONTEXT_INVALID', 'The source context response did not match the current buffers and query.'); };
  if (!parsed.success || Buffer.byteLength(JSON.stringify(value)) > 48 * 1024) return refuse();
  const page = parsed.data;
  if ((query.assumeValue !== undefined) !== !!page.changeAnalysis
    || page.changeAnalysis && page.changeAnalysis.after !== query.assumeValue) return refuse();
  const files = [...sources].sort((a, b) => Buffer.compare(Buffer.from(a.file), Buffer.from(b.file))).map(({ file, digest }) => ({ file, digest }));
  const indexId = createHash('sha256').update(JSON.stringify([1, files.map(item => [item.file, item.digest])])).digest('hex');
  if (new Set(files.map(item => item.file)).size !== files.length || sources.some(item => createHash('sha256').update(item.source).digest('hex') !== item.digest)
    || page.indexId !== indexId || query.indexId && query.indexId !== indexId || JSON.stringify(page.files) !== JSON.stringify(files)
    || page.items.length > query.limit || page.traversal.entities > query.maxEntities
    || (page.nextOffset !== null && (page.nextOffset !== query.offset + page.items.length || page.nextOffset >= page.total || !page.items.length))
    || (page.nextOffset === null && query.offset + page.items.length !== Math.max(query.offset, page.total))) return refuse();
  const byFile = new Map(sources.map(source => [source.file, source]));
  for (const item of page.items) {
    if (item.kind === 'relation' || item.kind === 'boundary') continue;
    const source = byFile.get(item.anchor.file), anchor = item.anchor, lines = source?.source.split('\n');
    if (!source || anchor.digest !== source.digest || !lines || anchor.endLine < anchor.startLine || anchor.endLine > lines.length
      || anchor.startColumn > lines[anchor.startLine - 1].length || anchor.endColumn > lines[anchor.endLine - 1].length
      || anchor.startLine === anchor.endLine && anchor.endColumn < anchor.startColumn
      || item.excerpt !== undefined && item.excerpt !== lines.slice(anchor.startLine - 1, anchor.endLine).join('\n')
      || (item.excerpt === undefined) !== (item.omitted === 'excerpt_bytes')) return refuse();
    if (item.configurationChange && (!page.changeAnalysis || (item.kind === 'call') !== (item.configurationChange.kind === 'call'))) return refuse();
    if (item.id === page.changeAnalysis?.definitionId && (item.kind !== 'definition' || item.bindingKind !== 'assignment'
      || item.name !== query.symbol || anchor.file !== query.file || query.line !== undefined && anchor.startLine !== query.line
      || item.configurationChange?.kind !== 'scalar' || item.configurationChange.status !== 'derived'
      || item.configurationChange.before !== page.changeAnalysis.before || item.configurationChange.after !== page.changeAnalysis.after)) return refuse();
  }
  if (page.diagnostics.some(item => !byFile.has(item.file))) return refuse();
  return page;
}

/** Correspondence is exact source identity, never name similarity or current runtime applicability. */
export function sourceCaptureContext(page: SourceContextPage, capture: CaptureSnapshot) {
  const key = (a: z.infer<typeof anchorSchema>) => JSON.stringify([a.file, a.digest, a.startLine, a.startColumn, a.endLine, a.endColumn, a.scope, a.tokenDigest]);
  const files = new Map(capture.source.files.map(item => [item.path, item.digest]));
  const bindings = new Map<string, string[]>();
  for (const binding of capture.sourceBindings) {
    const anchor = binding.anchors[0];
    if (binding.quality !== 'exact' || binding.snapshotId !== capture.source.id || binding.anchors.length !== 1 || !anchor?.tokenDigest
      || files.get(anchor.file) !== anchor.digest) continue;
    const ids = bindings.get(key(anchor)) ?? []; ids.push(binding.id); bindings.set(key(anchor), ids);
  }
  const owners = new Map<string, Array<{ kind: 'operation' | 'invocation' | 'module' | 'definition'; id: string }>>();
  for (const [kind, records] of [['operation', capture.operations], ['invocation', capture.invocations], ['module', capture.modules], ['definition', capture.definitions]] as const) {
    for (const record of records) {
      const refs = record as { sourceBindingIds?: string[]; constructorBindingIds?: string[] };
      for (const bindingId of [...refs.sourceBindingIds ?? [], ...refs.constructorBindingIds ?? []]) {
        const list = owners.get(bindingId) ?? []; list.push({ kind, id: record.id }); owners.set(bindingId, list);
      }
    }
  }
  const links: Array<{ anchorId: string; bindingId: string; ref: { kind: string; id: string } }> = [];
  let omittedMatches = 0, unmatchedRecords = 0, size = 0;
  for (const item of page.items) {
    if (item.kind === 'relation' || item.kind === 'boundary') continue;
    const matched = item.anchor.tokenDigest ? bindings.get(key(item.anchor)) ?? [] : [];
    if (!matched.length) unmatchedRecords++;
    for (const bindingId of matched) for (const ref of owners.get(bindingId) ?? [{ kind: 'sourceBinding', id: bindingId }]) {
      const link = { anchorId: item.id, bindingId, ref }, bytes = Buffer.byteLength(JSON.stringify(link));
      if (links.length >= 64 || size + bytes > 12 * 1024) { omittedMatches++; continue; }
      links.push(link); size += bytes;
    }
  }
  const configurationImpact = page.changeAnalysis ? sourceConfigurationImpact(page, capture, links) : undefined;
  const operationImpact = page.changeAnalysis ? sourceShapeImpact(page, capture, links) : undefined;
  return { captureId: capture.id, sourceId: capture.source.id, evidence: 'exact_source_correspondence' as const,
    scope: 'source_records_on_this_page' as const, appliesToCurrentRuntime: 'not_established' as const,
    captureStatus: capture.status, coverageScope: capture.coverage.scope,
    coverageInspection: { captureId: capture.id, sourceId: capture.source.id, path: ['coverage'] },
    links, omittedMatches, unmatchedRecords,
    ...(configurationImpact && operationImpact ? { configurationImpact, operationImpact,
      computationImpact: sourceComputationImpact(page, capture, links, configurationImpact, operationImpact, omittedMatches) } : {}) };
}
