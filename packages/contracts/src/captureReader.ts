/** Shared host/offline structural reader. Reads validated capture JSON only. */
import type { components } from './semanticContracts.generated';
import type { CaptureInspectionEntry, CaptureInspectionPage, CaptureInspectionQuery } from './protocol';
import { isCaptureInspectionQuery } from './protocol';

type CaptureSnapshot = components['schemas']['CaptureSnapshot'];
const MAX_PAGE_BYTES = 64 * 1024;
const bytes = (value: unknown): number => new Blob([JSON.stringify(value)]).size;
export class CaptureReadError extends Error {
  constructor(readonly code: 'CAPTURE_UNAVAILABLE' | 'INVALID_QUERY' | 'SOURCE_MISMATCH' | 'UNKNOWN_ENTITY' | 'INVALID_PATH' | 'RESULT_TOO_LARGE', message: string) {
    super(message);
  }
}

function describe(key: CaptureInspectionEntry['key'], value: unknown): CaptureInspectionEntry {
  if (value === null) return { key, kind: 'null', value: null };
  if (Array.isArray(value)) return { key, kind: 'array', count: value.length };
  if (typeof value === 'object') {
    const row = (value as Record<string, unknown>);
    const label = Array.from([row.id, row.target ?? row.qualifiedName ?? row.kind ?? row.status].filter(v => typeof v === 'string').join(' · ')).slice(0, 160).join('');
    return { key, kind: 'object', count: Object.keys(row).length, ...(label ? { label } : {}) };
  }
  if (typeof value === 'string' || typeof value === 'boolean' || (typeof value === 'number' && Number.isFinite(value))) {
    return { key, kind: typeof value as 'string' | 'boolean' | 'number', value };
  }
  throw new CaptureReadError('CAPTURE_UNAVAILABLE', 'The retained capture contains an unsupported value.');
}

/** An exact structural view, never a second serialization of the whole record.
 * Typed argument tags, dict entry order, repeated references, state and effects
 * remain intact. Pages can be recursively reconstructed without truncation. */
export function inspectCapture(capture: CaptureSnapshot, requested: CaptureInspectionQuery): CaptureInspectionPage {
  if (!isCaptureInspectionQuery(requested) || bytes(requested.path) > 8192) {
    throw new CaptureReadError('INVALID_QUERY', 'Use a bounded capture path and page size.');
  }
  const query = requested;
  if (capture.id !== query.captureId || capture.source.id !== query.sourceId) {
    throw new CaptureReadError('SOURCE_MISMATCH', 'Use the exact capture and source identity returned by tensorViz.');
  }
  let value: unknown = capture;
  if (query.entity) {
    const { kind, id } = query.entity;
    if (kind === 'input' || kind === 'output') {
      if (id !== capture.id) throw new CaptureReadError('UNKNOWN_ENTITY', 'This boundary does not belong to the capture.');
      value = kind === 'input' ? capture.inputs : capture.output;
    } else {
      const collection: Array<{ id: string }> = kind === 'operation' ? capture.operations : kind === 'invocation' ? capture.invocations : capture.values;
      value = collection.find(record => record.id === id);
      if (value === undefined) throw new CaptureReadError('UNKNOWN_ENTITY', 'This entity does not belong to the capture.');
    }
  }
  for (const key of query.path) {
    if (Array.isArray(value) ? typeof key !== 'number' || key >= value.length
      : value === null || typeof value !== 'object' || typeof key !== 'string' || !Object.hasOwn(value, key)) {
      throw new CaptureReadError('INVALID_PATH', 'This path does not exist in the retained capture.');
    }
    value = (value as Record<string | number, unknown>)[key];
  }
  const node = describe(null, value);
  const entries: Array<[string | number, unknown]> = Array.isArray(value) ? value.map((item, i) => [i, item])
    : node.kind === 'object' ? Object.entries((value as Record<string, unknown>)) : [];
  const page: CaptureInspectionPage = { captureId: capture.id, sourceId: capture.source.id,
    ...(query.entity ? { entity: query.entity } : {}), path: query.path, node, items: [], total: entries.length, nextOffset: null };
  let size = bytes(page) + 64;
  for (const [key, child] of entries.slice(query.offset, query.offset + query.limit)) {
    const item = describe(key, child), added = bytes(item) + 1;
    if (size + added > MAX_PAGE_BYTES) break;
    page.items.push(item); size += added;
  }
  if ((query.offset < entries.length && !page.items.length) || size > MAX_PAGE_BYTES) {
    throw new CaptureReadError('RESULT_TOO_LARGE', 'This value exceeds the capture inspection page limit.');
  }
  page.nextOffset = query.offset + page.items.length < entries.length ? query.offset + page.items.length : null;
  return page;
}
