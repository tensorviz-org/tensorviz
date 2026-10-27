import { SavedGraph } from '../types';
import { VerificationRecord, readVerification } from './verificationRecord';

/**
 * Where a verify result is remembered between visits.
 *
 * A verify result used to live in ModelBuilder's React state and die on unmount,
 * which is why /projects could not tell "verified", "failing" and "never run"
 * apart — the redesign needs all three.
 *
 * This keeps it in localStorage rather than in the graph document, deliberately:
 * writing it into the document means writing through the revisioned CAS save
 * queue, and ModelBuilder's save request does not currently carry graph metadata
 * at all (buildSaveRequest passes only { nodes, edges }). Fixing that plumbing to
 * carry a status flag would put a UI change inside the save-authority path, which
 * is a much larger and riskier change than the status column is worth.
 *
 * The honest cost of this choice: the record is per-browser, so a project
 * verified on another machine reads as "never run" here. That is a truthful
 * statement about what THIS machine has checked, which is the local-first claim
 * the rest of the app makes. `readVerification` is still consulted first, so if
 * the record ever does start riding in the document, server truth wins with no
 * change to the call sites.
 */

const KEY = 'pytorchviz.verification.v1';
/** Cap so a long-lived browser cannot grow this blob without bound. */
const MAX_ENTRIES = 200;

type Stored = Record<string, VerificationRecord & { at: number }>;

function load(): Stored {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch {
    // A quota error, a disabled store, or a corrupt blob must not break a render.
    return {};
  }
}

function save(next: Stored): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* Best effort: losing the cache costs a status pill, not the user's work. */
  }
}

export function recordVerification(graphId: string, record: VerificationRecord): void {
  if (!graphId) return;
  const store = load();
  store[graphId] = { ...record, at: Date.now() };
  const ids = Object.keys(store);
  if (ids.length > MAX_ENTRIES) {
    ids
      .sort((a, b) => (store[a].at ?? 0) - (store[b].at ?? 0))
      .slice(0, ids.length - MAX_ENTRIES)
      .forEach((id) => delete store[id]);
  }
  save(store);
}

export function readRecordedVerification(graphId?: string): VerificationRecord | null {
  if (!graphId) return null;
  const entry = load()[graphId];
  if (!entry || typeof entry.ok !== 'boolean' || typeof entry.revision !== 'number') return null;
  const { at, ...record } = entry;
  return record;
}

/**
 * The verification a project list should display: the document's own record when
 * one exists (server truth, synced), otherwise this machine's.
 */
export function resolveVerification(
  project: Pick<SavedGraph, 'id' | 'document'>,
): VerificationRecord | null {
  return readVerification(project.document) ?? readRecordedVerification(project.id);
}

/** Drops a project's record — call when the project itself is deleted. */
export function forgetVerification(graphId: string): void {
  const store = load();
  if (!(graphId in store)) return;
  delete store[graphId];
  save(store);
}
