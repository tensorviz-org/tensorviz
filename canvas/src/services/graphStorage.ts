import { SavedGraph, ModelGraph } from '../types';
import {
  ensureDefaultProject,
  listGraphs,
  getGraph as apiGetGraph,
  createGraph,
  commitGraph,
  deleteGraph as apiDeleteGraph,
  GraphRecord,
} from './projectsApi';
import {
  CrashDraft,
  draftModelGraph,
  GraphDraftStore,
  UNSAVED_DRAFT_ID,
} from './localDraft';
import {
  graphDocumentV2ToModelGraph,
  legacyModelGraphToDocumentV2,
  modelGraphToDocumentV2,
} from './graphDocumentAdapter';
import type { GraphSaveMetadata } from './revisionedGraphSaveQueue';
import { ideMode } from '../ide/projection';

// Legacy localStorage blob (v1) — read only for the one-time import into the
// server. No longer the source of truth for the graph list.
const LEGACY_STORAGE_KEY = 'canvas-saved-graphs';
const MIGRATED_FLAG_KEY = 'canvas-migrated';
// "Not now" on the import banner. Deliberately NOT the migrated flag: the legacy
// blob stays untouched and importable, we just stop re-offering it every visit.
const IMPORT_DISMISSED_KEY = 'canvas-migration-dismissed';

// Ids minted locally before the server migration (legacy list + templates).
// Anything else is treated as a server id and update-first.
const LOCAL_ID_RE = /^(graph-|template[-_]|template$|__unsaved__$)/;

function isServerId(id: string): boolean {
  return !LOCAL_ID_RE.test(id);
}

function apiStatus(error: unknown): number | null {
  const message = error instanceof Error ? error.message : String(error);
  const match = message.match(/API Error \((\d{3})\)/);
  return match ? Number(match[1]) : null;
}

function isOfflineError(error: unknown): boolean {
  if (error instanceof TypeError) return true;
  const message = error instanceof Error ? error.message : String(error);
  return /failed to fetch|network(?: request)? failed|\boffline\b|load failed/i.test(message);
}

function recordToSaved(rec: GraphRecord): SavedGraph {
  return {
    id: rec.id,
    name: rec.name,
    description: rec.description,
    document: rec.document,
    currentRevision: rec.currentRevision,
    graph: graphDocumentV2ToModelGraph(rec.document),
    createdAt: rec.createdAt,
    updatedAt: rec.updatedAt,
  };
}

function recoveredDraft(id: string, draft: CrashDraft, record?: GraphRecord): SavedGraph {
  const now = new Date().toISOString();
  const description = draft.document.semantic.metadata?.description;
  return {
    id,
    name: draft.document.semantic.name || record?.name || 'Untitled Model',
    ...(typeof description === 'string' ? { description } : {}),
    graph: draftModelGraph(draft),
    document: draft.document,
    currentRevision: draft.baseRevision,
    createdAt: record?.createdAt || now,
    updatedAt: record?.updatedAt || now,
    recoveredDraft: true,
    recoveredCommits: draft.pendingCommits,
  };
}

/**
 * Server-backed graph persistence. The four live methods (getGraph,
 * getAllGraphs, saveGraph, deleteGraph) are async and route through
 * projectsApi under the owner's default project. localStorage is used only
 * for crash drafts (see localDraft) and the legacy-import blob.
 */
export class GraphStorageService {
  static async getAllGraphs(): Promise<SavedGraph[]> {
    const pid = await ensureDefaultProject();
    const records = await listGraphs(pid);
    return records
      .map(recordToSaved)
      .sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
  }

  static getUnsavedDraft(
    draftStore: GraphDraftStore
  ): SavedGraph | null {
    const draft = draftStore.readRevisioned(UNSAVED_DRAFT_ID);
    return draft ? recoveredDraft(UNSAVED_DRAFT_ID, draft) : null;
  }

  static async getGraph(
    id: string,
    draftStore: GraphDraftStore
  ): Promise<SavedGraph | null> {
    let record: GraphRecord;
    try {
      const pid = await ensureDefaultProject();
      record = await apiGetGraph(pid, id);
    } catch (error) {
      // Only an authoritative 404 means the graph is absent. During an actual
      // network outage a revision-tagged crash draft may be shown, but auth,
      // server and decode failures must reach the load-error UI; treating them
      // as an empty graph could create a duplicate authority on the next edit.
      const draftResult = draftStore.read(id);
      if (apiStatus(error) === 404) {
        if (draftResult) {
          throw new Error('The saved graph no longer exists; its local recovery draft was preserved.');
        }
        return null;
      }
      if (draftResult?.kind === 'revisioned' && isOfflineError(error)) {
        return recoveredDraft(id, draftResult.draft);
      }
      throw error;
    }

    const draftResult = draftStore.read(id);
    if (!draftResult) return recordToSaved(record);
    if (draftResult.kind === 'legacy-unknown-base') {
      // Never attach an unknown-base draft to a saved authority. The raw draft
      // remains available locally while the authoritative graph opens normally.
      return { ...recordToSaved(record), legacyRecoveryAvailable: true };
    }
    const draft = draftResult.draft;
    const reconciliation = draftStore.reconcileWithServer(
      id,
      draft,
      record.document,
      record.currentRevision
    );
    if (reconciliation.status === 'committed') return recordToSaved(record);
    if (reconciliation.status === 'pending') {
      return recoveredDraft(id, reconciliation.draft, record);
    }
    // Unreconcilable: another tab committed a different edit from the same base.
    // Archive the local draft and open the saved graph. Throwing here bricked the
    // graph — the pill told the user to reload, and every reload in that profile
    // hit the load-error screen, whose only control is a Retry that cannot succeed.
    draftStore.preserveConflicted(id);
    return { ...recordToSaved(record), conflictedDraftAvailable: true };
  }

  static async saveGraph(
    graph: ModelGraph,
    name: string,
    description?: string,
    id?: string,
    meta?: GraphSaveMetadata
  ): Promise<SavedGraph> {
    if (ideMode()) {
      throw new Error('Source graphs are managed by VS Code. Export a graph file to save a snapshot.');
    }
    const pid = await ensureDefaultProject();
    const document = modelGraphToDocumentV2(graph, {
      graphId: id,
      name,
      description,
      revision: meta?.expectedRevision ?? 0,
      operations: meta?.operations,
      layout: meta?.layout,
    });

    if (id && isServerId(id)) {
      if (meta?.expectedRevision === undefined) {
        throw new Error('A saved graph update requires its expected revision');
      }
      const record = await commitGraph(pid, id, {
        name,
        document,
        expectedRevision: meta.expectedRevision,
        author: meta.author,
        opSummary: meta.opSummary,
      });
      return recordToSaved(record);
    }

    const record = await createGraph(pid, name, document, {
      ...(meta?.author ? { author: meta.author } : {}),
      ...(meta?.opSummary ? { opSummary: meta.opSummary } : {}),
      ...(meta?.createKey ? { idempotencyKey: meta.createKey } : {}),
    });
    return recordToSaved(record);
  }

  /**
   * Rename a saved graph from a list page, without opening it in the builder.
   *
   * The graph, its stored layout and its operation history are all carried
   * through: modelGraphToDocumentV2 rebuilds the document from what it is
   * handed, so passing the name alone would drop the canvas arrangement and the
   * history. Name lives in the semantic half of the document, so this does
   * commit a new revision — a verified graph reads as 'stale' afterwards, the
   * same as any other edit.
   */
  static async renameGraph(project: SavedGraph, name: string): Promise<SavedGraph> {
    return this.saveGraph(project.graph, name, project.description, project.id, {
      expectedRevision: project.currentRevision,
      author: 'user',
      opSummary: 'Renamed',
      operations: project.document.semantic.operations,
      layout: project.document.view.layout,
    });
  }

  static async deleteGraph(
    id: string,
    draftStore: GraphDraftStore
  ): Promise<boolean> {
    try {
      const pid = await ensureDefaultProject();
      await apiDeleteGraph(pid, id);
      draftStore.clear(id);
      return true;
    } catch (error) {
      console.error('Error deleting graph:', error);
      return false;
    }
  }

  // --- One-time migration of legacy localStorage graphs --------------------

  /**
   * Legacy graphs stored under the v1 localStorage key that have not yet been
   * imported. Returns [] when the blob is absent or the migrated flag is set.
   */
  static getUnmigratedLocalGraphs(): SavedGraph[] {
    try {
      if (localStorage.getItem(MIGRATED_FLAG_KEY) === '1') return [];
      const stored = localStorage.getItem(LEGACY_STORAGE_KEY);
      if (!stored) return [];
      const parsed = JSON.parse(stored);
      return Array.isArray(parsed) ? (parsed as SavedGraph[]) : [];
    } catch (error) {
      console.error('Error reading legacy graphs:', error);
      return [];
    }
  }

  static markMigrated(): void {
    try {
      localStorage.setItem(MIGRATED_FLAG_KEY, '1');
    } catch (error) {
      console.warn('Failed to set migration flag:', error);
    }
  }

  /**
   * "Not now": stop offering the import on later visits. The legacy blob is left
   * in place and is NOT marked migrated, so nothing is dropped — importLocalGraphs
   * still finds and imports it if the offer is ever shown again.
   */
  static dismissLocalGraphImport(): void {
    try {
      localStorage.setItem(IMPORT_DISMISSED_KEY, '1');
    } catch (error) {
      console.warn('Failed to persist import dismissal:', error);
    }
  }

  static isLocalGraphImportDismissed(): boolean {
    try {
      return localStorage.getItem(IMPORT_DISMISSED_KEY) === '1';
    } catch {
      return false;
    }
  }

  /**
   * Create each legacy graph on the server, then set the migrated flag so it
   * is never offered again. Returns the number of graphs imported.
   */
  static async importLocalGraphs(): Promise<number> {
    const legacy = this.getUnmigratedLocalGraphs();
    if (legacy.length === 0) {
      this.markMigrated();
      return 0;
    }
    const pid = await ensureDefaultProject();
    for (const g of legacy) {
      await createGraph(
        pid,
        g.name,
        legacyModelGraphToDocumentV2(g.graph, {
          graphId: g.id,
          name: g.name,
          description: g.description,
          revision: g.currentRevision || 0,
        })
      );
    }
    this.markMigrated();
    return legacy.length;
  }
}
