import { ModelGraph, PendingGraphCommit } from '../types';
import { GraphDocumentV2 } from '../types/graphDocument';
import {
  graphDocumentV2ToModelGraph,
  isGraphDocumentV2,
} from './graphDocumentAdapter';

/**
 * Local crash recovery is deliberately not a second graph authority. A draft
 * store is bound to one authenticated owner when an editor mounts, and only
 * contains the queue envelopes that have not yet been confirmed by the server.
 */
const SCOPED_DRAFT_PREFIX = 'canvas-draft-v3:';
const SCOPED_RECOVERY_PREFIX = 'canvas-recovery-v1:';
// Drafts that can never be rebased onto the saved graph (another tab committed a
// different edit from the same base). Archived under their own prefix, which read()
// deliberately never consults: leaving them live made every subsequent load of that
// graph throw, so the graph could not be opened again in that browser profile at all.
const SCOPED_CONFLICT_PREFIX = 'canvas-conflict-v1:';
const UNSCOPED_V2_PREFIX = 'canvas-draft-v2-';
const RAW_LEGACY_PREFIX = 'canvas-draft-';
export const UNSAVED_DRAFT_ID = '__unsaved__';
export const LOCAL_DEVELOPMENT_DRAFT_OWNER = 'local-dev';

export interface CrashDraft {
  document: GraphDocumentV2;
  baseRevision: number;
  pendingCommits: PendingGraphCommit[];
}

export type DraftReadResult =
  | { kind: 'revisioned'; draft: CrashDraft }
  | { kind: 'legacy-unknown-base'; graph: ModelGraph; sourceKey: string }
  | null;

export type DraftServerReconciliation =
  | { status: 'pending'; draft: CrashDraft }
  | { status: 'committed' }
  | { status: 'conflict' };

function sortedJsonValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortedJsonValue);
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, sortedJsonValue(item)])
    );
  }
  return value;
}

function comparableDocument(document: GraphDocumentV2): string {
  const copy = JSON.parse(JSON.stringify(document)) as GraphDocumentV2;
  copy.semantic.graph_id = '';
  copy.semantic.revision = 0;
  copy.view.graph_id = '';
  return JSON.stringify(sortedJsonValue(copy));
}

/** Compare graph meaning and view state while ignoring server-assigned identity. */
export function graphDocumentsMatch(
  left: GraphDocumentV2,
  right: GraphDocumentV2
): boolean {
  return comparableDocument(left) === comparableDocument(right);
}

function pendingCommit(value: unknown): PendingGraphCommit | null {
  if (!value || typeof value !== 'object') return null;
  const candidate = value as Partial<PendingGraphCommit>;
  if (!isGraphDocumentV2(candidate.document)) return null;
  return {
    document: candidate.document,
    author: candidate.author === 'agent' ? 'agent' : 'user',
    ...(typeof candidate.opSummary === 'string' ? { opSummary: candidate.opSummary } : {}),
    ...(typeof candidate.createKey === 'string' ? { createKey: candidate.createKey } : {}),
  };
}

function parseRevisionedDraft(raw: string, sourceKey: string): DraftReadResult {
  const parsed = JSON.parse(raw) as Partial<CrashDraft>;
  if (!isGraphDocumentV2(parsed.document)) return null;
  if (
    typeof parsed.baseRevision !== 'number'
    || !Number.isInteger(parsed.baseRevision)
    || parsed.baseRevision < 0
  ) {
    return {
      kind: 'legacy-unknown-base',
      graph: graphDocumentV2ToModelGraph(parsed.document),
      sourceKey,
    };
  }
  const pendingCommits = Array.isArray(parsed.pendingCommits)
    ? parsed.pendingCommits.map(pendingCommit).filter(Boolean) as PendingGraphCommit[]
    : [];
  const commits = pendingCommits.length > 0
    ? pendingCommits
    : [{ document: parsed.document, author: 'user' as const }];
  return {
    kind: 'revisioned',
    draft: {
      document: commits[commits.length - 1].document,
      baseRevision: parsed.baseRevision,
      pendingCommits: commits,
    },
  };
}

function stampedCommit(
  id: string,
  baseRevision: number,
  commit: PendingGraphCommit
): PendingGraphCommit {
  const document = JSON.parse(JSON.stringify(commit.document)) as GraphDocumentV2;
  document.semantic.graph_id = id;
  document.semantic.revision = baseRevision;
  document.view.graph_id = id;
  return { ...commit, document };
}

/** Owner-captured local crash journal. A null owner is intentionally disabled. */
export class GraphDraftStore {
  constructor(private readonly ownerUid: string | null) {}

  private scopedKey(id: string): string | null {
    if (!this.ownerUid) return null;
    return `${SCOPED_DRAFT_PREFIX}${encodeURIComponent(this.ownerUid)}:${encodeURIComponent(id)}`;
  }

  private recoveryKey(id: string): string | null {
    if (!this.ownerUid) return null;
    return `${SCOPED_RECOVERY_PREFIX}${encodeURIComponent(this.ownerUid)}:${encodeURIComponent(id)}`;
  }

  private conflictKey(id: string): string | null {
    if (!this.ownerUid) return null;
    return `${SCOPED_CONFLICT_PREFIX}${encodeURIComponent(this.ownerUid)}:${encodeURIComponent(id)}`;
  }

  /**
   * Archive a draft that cannot be reconciled with the saved graph, so the saved
   * graph opens normally on this and every later load. The payload is kept, not
   * deleted — it is the user's unsent work — but it is moved somewhere read() will
   * not pick it up, otherwise the same unresolvable conflict recurs forever.
   */
  preserveConflicted(id: string): void {
    const key = this.scopedKey(id);
    const archive = this.conflictKey(id);
    if (!key || !archive) return;
    try {
      const recoveryKey = this.recoveryKey(id);
      const existing = localStorage.getItem(key)
        ?? (recoveryKey ? localStorage.getItem(recoveryKey) : null);
      if (existing) localStorage.setItem(archive, existing);
      localStorage.removeItem(key);
      if (recoveryKey) localStorage.removeItem(recoveryKey);
      this.removeKnownRevisionedUnscopedDraft(id);
    } catch {
      /* best-effort */
    }
  }

  private canReadUnscopedLegacy(): boolean {
    return this.ownerUid === LOCAL_DEVELOPMENT_DRAFT_OWNER;
  }

  private removeKnownRevisionedUnscopedDraft(id: string): void {
    if (!this.canReadUnscopedLegacy()) return;
    const key = UNSCOPED_V2_PREFIX + id;
    const raw = localStorage.getItem(key);
    if (!raw) return;
    try {
      if (parseRevisionedDraft(raw, key)?.kind === 'revisioned') {
        localStorage.removeItem(key);
      }
    } catch {
      // Malformed and unknown-base recovery payloads remain untouched.
    }
  }

  read(id: string): DraftReadResult {
    const key = this.scopedKey(id);
    if (!key) return null;
    try {
      const scoped = localStorage.getItem(key);
      if (scoped) return parseRevisionedDraft(scoped, key);
      const recoveryKey = this.recoveryKey(id);
      const quarantined = recoveryKey ? localStorage.getItem(recoveryKey) : null;
      if (quarantined) return parseRevisionedDraft(quarantined, recoveryKey!);
      if (!this.canReadUnscopedLegacy()) return null;

      const oldV2Key = UNSCOPED_V2_PREFIX + id;
      const oldV2 = localStorage.getItem(oldV2Key);
      if (oldV2) return parseRevisionedDraft(oldV2, oldV2Key);

      const rawKey = RAW_LEGACY_PREFIX + id;
      const raw = localStorage.getItem(rawKey);
      if (!raw) return null;
      const graph = JSON.parse(raw) as ModelGraph;
      if (!Array.isArray(graph?.nodes) || !Array.isArray(graph?.edges)) return null;
      return { kind: 'legacy-unknown-base', graph, sourceKey: rawKey };
    } catch {
      return null;
    }
  }

  readRevisioned(id: string): CrashDraft | null {
    const result = this.read(id);
    return result?.kind === 'revisioned' ? result.draft : null;
  }

  /** Replace the crash journal with the queue's exact outstanding commit order. */
  replaceJournal(
    id: string,
    baseRevision: number,
    pendingCommits: PendingGraphCommit[]
  ): boolean {
    const key = this.scopedKey(id);
    if (!key) return false;
    if (pendingCommits.length === 0) {
      this.clear(id);
      return true;
    }
    const stamped = pendingCommits.map((commit) => stampedCommit(id, baseRevision, commit));
    try {
      const existing = localStorage.getItem(key);
      const recoveryKey = this.recoveryKey(id);
      if (
        existing
        && recoveryKey
        && parseRevisionedDraft(existing, key)?.kind === 'legacy-unknown-base'
        && localStorage.getItem(recoveryKey) === null
      ) {
        // Keep the exact pre-revision payload outside the live crash journal.
        localStorage.setItem(recoveryKey, existing);
      }
      localStorage.setItem(
        key,
        JSON.stringify({
          document: stamped[stamped.length - 1].document,
          baseRevision,
          pendingCommits: stamped,
        })
      );
      this.removeKnownRevisionedUnscopedDraft(id);
      return true;
    } catch (error) {
      console.warn('Failed to save local draft journal:', error);
      return false;
    }
  }

  clear(id: string): void {
    const key = this.scopedKey(id);
    if (!key) return;
    try {
      const existing = localStorage.getItem(key);
      if (existing && parseRevisionedDraft(existing, key)?.kind === 'legacy-unknown-base') {
        return;
      }
      localStorage.removeItem(key);
      this.removeKnownRevisionedUnscopedDraft(id);
    } catch {
      /* best-effort */
    }
  }

  /**
   * Reconcile a journal with the current server revision. This covers the
   * narrow crash window after CAS succeeds but before the response is received.
   */
  reconcileWithServer(
    id: string,
    draft: CrashDraft,
    savedDocument: GraphDocumentV2,
    savedRevision: number
  ): DraftServerReconciliation {
    if (savedRevision < draft.baseRevision) return { status: 'conflict' };
    const landedCount = savedRevision - draft.baseRevision;
    if (landedCount === 0) return { status: 'pending', draft };
    if (landedCount > draft.pendingCommits.length) return { status: 'conflict' };
    if (!graphDocumentsMatch(
      draft.pendingCommits[landedCount - 1].document,
      savedDocument
    )) {
      return { status: 'conflict' };
    }

    const remaining = draft.pendingCommits.slice(landedCount);
    if (remaining.length === 0) {
      this.clear(id);
      return { status: 'committed' };
    }
    if (!this.replaceJournal(id, savedRevision, remaining)) {
      return { status: 'conflict' };
    }
    const rebased = this.readRevisioned(id);
    return rebased ? { status: 'pending', draft: rebased } : { status: 'conflict' };
  }
}

export function createGraphDraftStore(ownerUid: string | null): GraphDraftStore {
  return new GraphDraftStore(ownerUid);
}

export const localDevelopmentDraftStore = createGraphDraftStore(
  LOCAL_DEVELOPMENT_DRAFT_OWNER
);

export function draftModelGraph(draft: CrashDraft): ModelGraph {
  return graphDocumentV2ToModelGraph(draft.document);
}
