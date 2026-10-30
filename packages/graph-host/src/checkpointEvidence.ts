/** Bounded observations from the existing explicit checkpoint loader. */
import { z } from 'zod';
import { executionPathSchema, type ExecutionSpec } from './executionSpec';
import type { CandidateCheckpointSummary } from '@tensorviz/contracts/protocol';

const natural = z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER);
const digest = z.string().regex(/^[a-f0-9]{64}$/);
const metadata = z.object({
  kind: z.enum(['parameter', 'buffer', 'stored']),
  shape: z.array(natural).max(32).nullable(), dtype: z.string().max(64).nullable(),
  objectGroup: natural.nullable(), storageGroup: natural.nullable(), storageOffset: natural.nullable(),
  stride: z.array(natural).max(32).nullable(),
  unavailable: z.enum(['unsupported_tensor', 'non_tensor_state', 'metadata_error']).nullable(),
}).strict();
const row = z.object({
  name: z.string().max(2048), registered: metadata.nullable(), stored: metadata.nullable(),
  presence: z.enum(['both', 'registered_only', 'checkpoint_only', 'unknown']),
  shape: z.enum(['same', 'different', 'unknown']), dtype: z.enum(['same', 'different', 'unknown']),
}).strict().refine(item => {
  const presence = item.registered && item.stored ? 'both' : item.registered ? 'registered_only' : item.stored ? 'checkpoint_only' : null;
  const compare = (key: 'shape' | 'dtype') => item.registered?.[key] == null || item.stored?.[key] == null ? 'unknown'
    : JSON.stringify(item.registered[key]) === JSON.stringify(item.stored[key]) ? 'same' : 'different';
  return (item.presence === presence || item.presence === 'unknown' && presence !== 'both' && presence !== null)
    && item.shape === compare('shape') && item.dtype === compare('dtype')
    && (!item.registered || item.registered.kind !== 'stored') && (!item.stored || item.stored.kind === 'stored');
});
export const checkpointEvidenceSchema = z.object({
  version: z.literal(1), sourceDigest: digest,
  checkpoint: z.object({ file: executionPathSchema, digest, key: z.string().min(1).max(128).optional(), strict: z.boolean() }).strict(),
  scope: z.literal('registered_state_before_load'), loaderSemantics: z.literal('not_inferred'),
  loadStatus: z.enum(['succeeded', 'failed']), complete: z.boolean(),
  registeredComplete: z.boolean(), storedComplete: z.boolean(),
  omissions: z.array(z.enum(['module_limit', 'cyclic_registration', 'unsupported_registration', 'name_limit',
    'entry_limit', 'unsupported_mapping', 'unsupported_tensor', 'non_tensor_state', 'metadata_error', 'byte_limit'])).max(10),
  entries: z.array(row).max(8192),
}).strict().refine(report => report.complete === (report.omissions.length === 0)
  && (!report.complete || report.registeredComplete && report.storedComplete)
  && new Set(report.entries.map(item => item.name)).size === report.entries.length
  && report.entries.every(item => item.presence === (item.registered && item.stored ? 'both'
    : item.registered ? report.storedComplete ? 'registered_only' : 'unknown'
    : report.registeredComplete ? 'checkpoint_only' : 'unknown')
    && [item.registered, item.stored].every(value => !value?.unavailable || report.omissions.includes(value.unavailable))));
export type CheckpointEvidence = z.infer<typeof checkpointEvidenceSchema>;
export type CheckpointRequest = { sourceDigest: string; checkpoint?: ExecutionSpec['checkpoint'] };
export const checkpointQuerySchema = z.object({
  offset: z.number().int().min(0).max(8192).default(0),
  limit: z.number().int().min(1).max(200).default(50),
}).strict();

export function readCheckpointEvidence(value: unknown, request?: CheckpointRequest): CheckpointEvidence | undefined {
  if (value == null || !request?.checkpoint) return undefined;
  if (Buffer.byteLength(JSON.stringify(value)) > 1024 * 1024) return undefined;
  const parsed = checkpointEvidenceSchema.safeParse(value);
  if (!parsed.success) return undefined;
  const report = parsed.data;
  if (report.sourceDigest !== request.sourceDigest || report.checkpoint.file !== request.checkpoint.file
    || report.checkpoint.key !== request.checkpoint.key || report.checkpoint.strict !== request.checkpoint.strict) return undefined;
  return report;
}

export function checkpointEvidencePage(report: CheckpointEvidence | undefined, offset: number, limit: number) {
  if (!report) return { checkpoint: null, detail: 'No valid checkpoint comparison was retained for this run. Missing evidence is not a successful load.' };
  const { entries, ...summary } = report;
  const counts = { scope: 'retained_entries', entries: entries.length,
    shapeDifferences: entries.filter(item => item.shape === 'different').length,
    dtypeDifferences: entries.filter(item => item.dtype === 'different').length,
    registeredOnly: entries.filter(item => item.presence === 'registered_only').length,
    checkpointOnly: entries.filter(item => item.presence === 'checkpoint_only').length,
    unknownPresence: entries.filter(item => item.presence === 'unknown').length };
  const items: typeof entries = [];
  let bytes = 0;
  for (const item of entries.slice(offset, offset + limit)) {
    bytes += Buffer.byteLength(JSON.stringify(item));
    if (bytes > 48 * 1024) break;
    items.push(item);
  }
  return { checkpoint: { ...summary, counts }, items, count: entries.length,
    nextOffset: offset + items.length < entries.length ? offset + items.length : null,
    detail: 'Historical metadata before this explicit load, with its actual load outcome. Registered state excludes nonpersistent buffers. Object/storage groups are local to each inventory; shared storage does not imply identical views. Dtype differences can be converted by the loader. Metadata agreement does not verify custom loading hooks, tensor values, current state or a forward pass.' };
}

/** A bounded historical inventory for the native read-only result document. */
export function checkpointReviewSummary(report: CheckpointEvidence): CandidateCheckpointSummary {
  const { scope: _scope, ...counts } = checkpointEvidencePage(report, 0, 1).checkpoint!.counts;
  return { loadStatus: report.loadStatus, complete: report.complete, ...counts };
}

export function checkpointEvidenceText(report: CheckpointEvidence | undefined, entries?: CheckpointEvidence['entries']): string {
  if (!report) return '\n## Checkpoint load\n\nNo valid checkpoint-load report was retained. Missing evidence is not a successful load.\n';
  const page = checkpointEvidencePage(report, 0, 1), counts = page.checkpoint!.counts;
  const metadataText = (value: CheckpointEvidence['entries'][number]['registered']) => !value ? 'Not recorded'
    : `${value.kind}; shape ${JSON.stringify(value.shape)}; dtype ${JSON.stringify(value.dtype)}; `
      + `object group ${value.objectGroup}; storage group ${value.storageGroup}; offset ${value.storageOffset}; `
      + `stride ${JSON.stringify(value.stride)}${value.unavailable ? `; unavailable: ${value.unavailable}` : ''}`;
  return `\n## Checkpoint load\n\nActual load: ${report.loadStatus}\nCheckpoint: ${JSON.stringify(report.checkpoint.file)}\n`
    + `Key: ${JSON.stringify(report.checkpoint.key ?? null)}\nStrict loading: ${report.checkpoint.strict}\n`
    + `Metadata coverage: ${report.complete ? 'complete' : 'partial'}\n`
    + `Retained entries: ${counts.entries}; shape differences: ${counts.shapeDifferences}; dtype differences: ${counts.dtypeDifferences}; `
    + `registered only: ${counts.registeredOnly}; checkpoint only: ${counts.checkpointOnly}; unknown presence: ${counts.unknownPresence}\n`
    + (report.omissions.length ? `Omissions: ${report.omissions.join(', ')}\n` : '')
    + (entries ?? report.entries).map(item => `\n### ${JSON.stringify(item.name)}\n\nRegistered: ${metadataText(item.registered)}\n`
      + `Stored: ${metadataText(item.stored)}\nPresence: ${item.presence}; shape: ${item.shape}; dtype: ${item.dtype}\n`).join('')
    + `\n${page.detail}\n\nCheckpoint digest: ${report.checkpoint.digest}\nCandidate model source digest: ${report.sourceDigest}\n`;
}
