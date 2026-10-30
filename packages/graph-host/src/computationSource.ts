import { createHash } from 'node:crypto';
import type { CaptureSnapshot } from './semanticCapture';
import type { ComputationIndex } from './computationQuery';
import { GraphToolError, type ModelRef } from './mcpGraph';

export type ComputationSourceReader = (ref: ModelRef, file: string) => Promise<{ file: string; source: string; digest: string }>;
type Page = ReturnType<ComputationIndex['query']>;
type Binding = CaptureSnapshot['sourceBindings'][number];
interface SourceEntry {
  bindingId: string;
  status: 'matched' | 'binding_omitted' | 'ambiguous_binding' | 'snapshot_mismatch' | 'source_unavailable'
    | 'source_changed' | 'range_limit' | 'file_limit' | 'context_limit';
  file?: string; digest?: string; startLine?: number; endLine?: number; source?: string;
  reason?: string;
}
export interface ComputationSourceContext {
  status: 'complete' | 'partial' | 'empty' | 'unavailable';
  scope: 'source_bindings_on_this_page';
  reason?: 'reader_unavailable' | 'graph_stale' | 'stale_target';
  evidence: 'capture_digest_matched' | null;
  entries: SourceEntry[];
  omittedEntries: number;
  filesRead: number;
}
const bytes = (value: unknown) => Buffer.byteLength(JSON.stringify(value));
const CONTEXT_BYTES = 8192, FILE_LIMIT = 4, ENTRY_LIMIT = 16;

/** The host supplies its existing trusted, graph-pinned source reader. This
 * function never opens files or executes code, and excerpts only exact bytes. */
export async function computationSourceContext(page: Page, snapshot: CaptureSnapshot['source'], ref: ModelRef,
  read?: ComputationSourceReader, unavailable: 'reader_unavailable' | 'graph_stale' = 'reader_unavailable'): Promise<ComputationSourceContext> {
  const result: ComputationSourceContext = { status: 'unavailable', scope: 'source_bindings_on_this_page',
    ...(!read ? { reason: unavailable } : {}), evidence: null, entries: [], omittedEntries: 0, filesRead: 0 };
  if (!read) return result;
  const sources = page.items.flatMap(item => item.kind === 'source' ? [item] : []);
  const files = new Map<string, string>();
  const available = new Map(snapshot.files.map(file => [file.path, file.digest]));
  const entries: Array<{ entry: SourceEntry; anchor?: Binding['anchors'][number] }> = [];
  for (const item of sources.slice(0, ENTRY_LIMIT)) {
    const entry: SourceEntry = { bindingId: item.ref.id, status: 'binding_omitted' };
    const binding = item.record as Binding | undefined;
    if (!binding) { entries.push({ entry }); continue; }
    if (binding.quality !== 'exact' || binding.anchors.length !== 1) entry.status = 'ambiguous_binding';
    else if (binding.snapshotId !== page.sourceId || snapshot.id !== page.sourceId) entry.status = 'snapshot_mismatch';
    else {
      const anchor = binding.anchors[0];
      Object.assign(entry, { file: anchor.file, digest: anchor.digest });
      if (available.get(anchor.file) !== anchor.digest) entry.status = 'snapshot_mismatch';
      else if (!files.has(anchor.file) && files.size === FILE_LIMIT) entry.status = 'file_limit';
      else { files.set(anchor.file, anchor.digest); entries.push({ entry, anchor }); continue; }
    }
    entries.push({ entry });
  }
  result.omittedEntries = Math.max(0, sources.length - entries.length);
  const reads = new Map<string, { source: Awaited<ReturnType<ComputationSourceReader>> } | { error: string }>(await Promise.all([...files.keys()].map(async file => {
    result.filesRead++;
    try { return [file, { source: await read(ref, file) }] as const; }
    catch (error) { return [file, { error: error instanceof GraphToolError ? error.code : 'SOURCE_UNAVAILABLE' }] as const; }
  })));
  let size = bytes(result) + 128;
  for (const { entry, anchor } of entries) {
    if (anchor) {
      const response = reads.get(anchor.file)!;
      if ('error' in response) { entry.status = 'source_unavailable'; entry.reason = response.error; }
      else {
        const { source } = response;
        if (source.file !== anchor.file || source.digest !== anchor.digest
          || createHash('sha256').update(source.source).digest('hex') !== anchor.digest) entry.status = 'source_changed';
        else {
          const lines = source.source.split('\n');
          const start = Math.max(1, anchor.startLine - 2), end = Math.min(lines.length, anchor.endLine + 2);
          if (anchor.startLine < 1 || anchor.endLine < anchor.startLine || anchor.endLine > lines.length || end - start >= 60) entry.status = 'range_limit';
          else Object.assign(entry, { status: 'matched', startLine: start, endLine: end,
            source: lines.slice(start - 1, end).join('\n') });
        }
      }
    }
    if (size + bytes(entry) + 1 > CONTEXT_BYTES) {
      delete entry.source; delete entry.startLine; delete entry.endLine;
      entry.status = 'context_limit';
    }
    if (size + bytes(entry) + 1 > CONTEXT_BYTES) { result.omittedEntries++; continue; }
    result.entries.push(entry); size += bytes(entry) + 1;
  }
  const matched = result.entries.filter(entry => entry.status === 'matched').length;
  result.evidence = matched ? 'capture_digest_matched' : null;
  result.status = sources.length === 0 ? 'empty' : matched === result.entries.length && result.omittedEntries === 0
    ? 'complete' : matched ? 'partial' : 'unavailable';
  return result;
}
