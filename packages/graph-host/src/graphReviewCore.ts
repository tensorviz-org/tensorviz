import type { IdeMultifileProjectionResponse, IdeMultifileSpan, ProjectionPayload } from '@tensorviz/contracts/protocol';
import type { ReviewModelRoot } from './reviewSnapshotsCore';
import type { SourceSet } from './workspaceSourceCore';

interface ReviewDeclaration { span: IdeMultifileSpan; constructor: IdeMultifileSpan | null; parent?: string; children?: string[] }

/** Both hosts analyze pinned source text and refuse execution-backed replies. */
export async function projectReviewSnapshot(target: Pick<ReviewModelRoot, 'file' | 'cls' | 'project'>, sources: SourceSet,
  request: (body: unknown) => Promise<IdeMultifileProjectionResponse>): Promise<ProjectionPayload> {
  const prefix = target.project ? `${target.project}/` : '';
  const file = target.file.slice(prefix.length), root = sources.entries.find(entry => entry.file === file);
  if (!root) throw new Error('The model source is unavailable in this revision.');
  const response = await request({ protocol_version: 1, root: { file, cls: target.cls, source_digest: root.digest },
    sources: sources.entries, truncated: sources.truncated, review_only: true });
  if (response.source_digest !== root.digest || response.cls !== target.cls || response.traced) {
    throw new Error('The source projection did not match the requested revision.');
  }
  const classSpan = (response.document as { semantic?: { metadata?: { reviewClass?: { span?: IdeMultifileSpan } } } }).semantic?.metadata?.reviewClass?.span;
  const spanmap = { ...response.spanmap, ...(classSpan ? { '@class': classSpan } : {}) };
  // Declarations and dependencies name project-relative files too. Review joins
  // them to this spanmap, to Git renames and to captures: all workspace paths.
  const moved = (span: IdeMultifileSpan) => ({ ...span, file: prefix + span.file });
  const document = response.document as { semantic?: { metadata?: { reviewDeclarations?: { entries: Record<string, ReviewDeclaration> } } } };
  const evidence = prefix ? document.semantic?.metadata?.reviewDeclarations : undefined;
  const keys = new Map(Object.entries(evidence?.entries ?? {}).map(([key, fact]) => {
    const spelled = `@declaration:[${JSON.stringify(fact.span.file)},`;
    return [key, key.startsWith(spelled) ? `@declaration:[${JSON.stringify(prefix + fact.span.file)},${key.slice(spelled.length)}` : key];
  }));
  const declarations = evidence && { ...evidence, entries: Object.fromEntries(Object.entries(evidence.entries).map(([key, fact]) => [keys.get(key)!, {
    ...fact, span: moved(fact.span), constructor: fact.constructor && moved(fact.constructor),
    parent: fact.parent && (keys.get(fact.parent) ?? fact.parent), children: fact.children?.map(child => keys.get(child) ?? child) }])) };
  return { graphId: response.graph_id, revision: response.revision, cls: target.cls, file: target.file,
    sourceDigest: root.digest, traced: false, greyed: false, viewRestored: false, document: declarations
      ? { ...document, semantic: { ...document.semantic, metadata: { ...document.semantic!.metadata, reviewDeclarations: declarations } } } : response.document,
    spanmap: Object.fromEntries(Object.entries(spanmap).map(([id, value]) => [keys.get(id) ?? id, moved(value as IdeMultifileSpan)])),
    projectionDigest: response.projection_digest, dependencies: response.dependencies?.map(entry => ({ ...entry, file: prefix + entry.file })),
    diagnostics: response.diagnostics, truncated: response.truncated };
}

/** A 400 from source projection names the file and line instead of a generic analysis failure. */
export function reviewAnalysisError(file: string, status: number, body: string): string {
  let detail: { error?: unknown; line?: unknown } | undefined;
  try {
    const parsed = JSON.parse(body) as { detail?: { error?: unknown; line?: unknown } };
    detail = parsed.detail && typeof parsed.detail === 'object' ? parsed.detail : undefined;
  } catch { /* The body is not JSON. */ }
  const line = detail?.line;
  if (detail?.error === 'syntax_error' && typeof line === 'number' && Number.isInteger(line) && line > 0) {
    return `could not parse ${file}:${line}`;
  }
  return `Source analysis failed (${status}). Both revisions must contain a supported class with this name.`;
}

export function emptyReviewProjection(cls: string): ProjectionPayload {
  return { graphId: 'review:empty', revision: 0, cls, file: '', sourceDigest: '', traced: false, greyed: false,
    viewRestored: false, spanmap: {}, diagnostics: [], truncated: false, document: { schemaVersion: 2,
      semantic: { graph_id: 'review:empty', name: cls, revision: 0, inputs: [], nodes: [], edges: [], modules: [], operations: [], metadata: {} },
      view: { graph_id: 'review:empty', node_positions: {}, collapsed_modules: [], selection: [], layout: {} } } };
}
