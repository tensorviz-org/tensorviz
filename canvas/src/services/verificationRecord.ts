import { GraphDocumentV2 } from '../types';

/**
 * Persisted verification outcome, carried inside the graph document.
 *
 * The redesign puts a model's health on its card and in the projects table:
 * Verified / N errors / Never run, with the first failure quoted inline. None of
 * that was expressible before, because a verify result lived in ModelBuilder's
 * React state and died on unmount — so "verified", "failing" and "never run"
 * were indistinguishable for every project in the list.
 *
 * It rides in `semantic.metadata`, which graphDocumentAdapter already
 * round-trips in both directions and which the projects list response already
 * carries in full. That is why this needs no schema change, no new column and no
 * extra request per row.
 *
 * `revision` is the graph revision the result was produced against — NOT a wall
 * clock. Comparing it to the project's currentRevision is what lets the UI
 * distinguish "verified" from "verified, then edited", which a timestamp cannot
 * do reliably when saves and verifies interleave.
 */
export type VerificationRecord = {
  ok: boolean;
  /** Graph revision this result describes. */
  revision: number;
  /** Failure count. 0 when ok. */
  errorCount: number;
  /** The first failure, already formatted for display. Absent when ok. */
  firstError?: string;
  /** Node the first failure belongs to, for deep-linking into the canvas. */
  firstErrorNodeId?: string;
  /** torch's own numbers, when the verify got far enough to build the module. */
  summary?: {
    layerCount?: number;
    params?: number;
    sizeMb?: number;
  };
};

const METADATA_KEY = 'verification';

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

/**
 * Reads the record back, tolerating anything. This parses data that was written
 * by an older build or hand-edited, so a malformed blob must degrade to "never
 * run" rather than throw inside a list render.
 */
export function readVerification(
  document?: GraphDocumentV2 | null,
): VerificationRecord | null {
  const raw = document?.semantic?.metadata?.[METADATA_KEY];
  if (!isRecord(raw)) return null;
  if (typeof raw.ok !== 'boolean' || typeof raw.revision !== 'number') return null;
  const summary = isRecord(raw.summary) ? raw.summary : undefined;
  return {
    ok: raw.ok,
    revision: raw.revision,
    errorCount: typeof raw.errorCount === 'number' ? raw.errorCount : raw.ok ? 0 : 1,
    firstError: typeof raw.firstError === 'string' ? raw.firstError : undefined,
    firstErrorNodeId:
      typeof raw.firstErrorNodeId === 'string' ? raw.firstErrorNodeId : undefined,
    summary: summary
      ? {
          layerCount: typeof summary.layerCount === 'number' ? summary.layerCount : undefined,
          params: typeof summary.params === 'number' ? summary.params : undefined,
          sizeMb: typeof summary.sizeMb === 'number' ? summary.sizeMb : undefined,
        }
      : undefined,
  };
}

export type ProjectStatus = 'verified' | 'error' | 'stale' | 'never-run';

/**
 * The four states the redesign's status column distinguishes.
 *
 * 'stale' exists because a pass that predates the current revision is not a
 * pass. Reporting it as Verified would be the "falsely complete" failure — the
 * model has changed since anything checked it.
 */
export function projectStatus(
  record: VerificationRecord | null,
  currentRevision?: number,
): ProjectStatus {
  if (!record) return 'never-run';
  if (typeof currentRevision === 'number' && record.revision < currentRevision) return 'stale';
  return record.ok ? 'verified' : 'error';
}

export const STATUS_LABELS: Record<ProjectStatus, string> = {
  verified: 'Verified',
  error: 'Error',
  stale: 'Stale',
  'never-run': 'Never run',
};

/** Builds the metadata patch to merge into a document's semantic.metadata. */
export function verificationMetadata(record: VerificationRecord): Record<string, unknown> {
  return { [METADATA_KEY]: record as unknown as Record<string, unknown> };
}
