import React, { useEffect, useMemo, useState } from 'react';
import { ChevronRight, Download, FileArchive, Pencil, Plus, Search, Trash2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import WorkbenchShell from './WorkbenchShell';
import StatusPill from './StatusPill';
import { GraphStorageService } from '../../services/graphStorage';
import * as hostServices from '../../ide/hostServices';
import { SavedGraph } from '../../types';
import { useAuth } from '../../contexts/AuthContext';
import {
  createGraphDraftStore,
  LOCAL_DEVELOPMENT_DRAFT_OWNER,
} from '../../services/localDraft';
import {
  ProjectStatus,
  VerificationRecord,
  projectStatus,
  readVerification,
} from '../../services/verificationRecord';
import {
  GraphMetrics,
  formatParams,
  formatSize,
  projectMetrics,
} from '../../services/graphMetrics';
import { formatEditedAt } from '../../utils/relativeTime';
import CardMenu from './CardMenu';
import { exportProjectAsPython, exportProjectAsZip } from '../../services/exportProjectCode';

/**
 * The projects list, as the table the redesign specifies: validation state, size
 * and edit recency per row.
 *
 * The previous list showed a name, a locale timestamp and a raw node count, so
 * the one question this page exists to answer — which of my models currently
 * build, and which are broken — could only be answered by opening every project
 * in the builder.
 */

type TableFilter = 'all' | 'verified' | 'error' | 'never-run';

type ProjectRow = {
  project: SavedGraph;
  record: VerificationRecord | null;
  status: ProjectStatus;
  metrics: GraphMetrics;
};

const FILTERS: { key: TableFilter; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'verified', label: 'Verified' },
  { key: 'error', label: 'Failing' },
  { key: 'never-run', label: 'Never run' },
];

/**
 * The header says "Sorted by edited ↓", so the rows have to actually be in that
 * order. An unparseable timestamp sorts as 0 rather than NaN, which would make
 * the comparator inconsistent and the resulting order arbitrary.
 */
function editedTime(value: string): number {
  const time = new Date(value).getTime();
  return Number.isNaN(time) ? 0 : time;
}

export default function ProjectsPageV2() {
  const navigate = useNavigate();
  const { user, configured } = useAuth();
  const draftOwner = configured ? user?.uid ?? null : LOCAL_DEVELOPMENT_DRAFT_OWNER;
  const draftStore = useMemo(() => createGraphDraftStore(draftOwner), [draftOwner]);
  const [projects, setProjects] = useState<SavedGraph[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [exportNotice, setExportNotice] = useState<string | null>(null);
  const [exportingId, setExportingId] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [filter, setFilter] = useState<TableFilter>('all');
  const [query, setQuery] = useState('');
  // Bulk cleanup: the list is where a pile of autonamed graphs accumulates, and
  // deleting them one confirm at a time is not a cleanup path.
  const [selected, setSelected] = useState<Set<string>>(new Set());

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    GraphStorageService.getAllGraphs()
      .then((all) => {
        if (!cancelled) {
          setProjects(all);
          setError(null);
        }
      })
      .catch(() => {
        if (!cancelled) setError('Failed to load models.');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // reloadKey is the retry trigger — the error branch bumps it to refetch.
  }, [draftOwner, reloadKey]);

  const rows = useMemo<ProjectRow[]>(
    () =>
      projects
        .map((project) => {
          const record = readVerification(project.document);
          return {
            project,
            record,
            status: projectStatus(record, project.currentRevision),
            metrics: projectMetrics(project),
          };
        })
        .sort((a, b) => editedTime(b.project.updatedAt) - editedTime(a.project.updatedAt)),
    [projects]
  );

  /**
   * Counts describe the whole list, not the current filter, so the strip keeps
   * saying how many models are broken while you are looking at the verified
   * ones. 'stale' (verified, then edited) is deliberately counted only under
   * All: it is not a pass any more, and it has never been run in its current
   * shape either, so putting it under either heading would be a false claim.
   */
  const counts = useMemo<Record<TableFilter, number>>(
    () => ({
      all: rows.length,
      verified: rows.filter((row) => row.status === 'verified').length,
      error: rows.filter((row) => row.status === 'error').length,
      'never-run': rows.filter((row) => row.status === 'never-run').length,
    }),
    [rows]
  );

  const totalLayers = useMemo(
    () => rows.reduce((acc, row) => acc + row.metrics.layers, 0),
    [rows]
  );

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return rows.filter((row) => {
      if (filter !== 'all' && row.status !== filter) return false;
      if (needle && !row.project.name.toLowerCase().includes(needle)) return false;
      return true;
    });
  }, [rows, filter, query]);

  /**
   * Export the row's model, gated on validation, as a single model.py or as
   * the runnable package.
   *
   * Only one export runs at a time per row, and the outcome is always reported:
   * a silent no-op after a click on a Download icon is indistinguishable from a
   * broken button, and a refusal has to name what the user must go and fix.
   */
  const exportProject = async (project: SavedGraph, format: 'python' | 'zip') => {
    setActionError(null);
    setExportNotice(null);
    setExportingId(project.id);
    try {
      const result = await (format === 'zip'
        ? exportProjectAsZip(project)
        : exportProjectAsPython(project));
      if (!result.ok) {
        setActionError(`${project.name} was not exported. ${result.message}`);
        return;
      }
      // An incomplete export still downloads — the file carries the backend's
      // own "# INCOMPLETE EXPORT" banner — but it must not be reported as if it
      // were the whole model.
      setExportNotice(
        result.unsupported.length > 0
          ? `${result.fileName} downloaded, but ${result.unsupported.length} module`
            + `${result.unsupported.length > 1 ? 's are' : ' is'} not reproduced in it: `
            + `${result.unsupported.map((m) => m.class_name).join(', ')}.`
          : `${result.fileName} downloaded.`
      );
    } finally {
      setExportingId(null);
    }
  };

  // Naming has to be cheap from the list. Renaming a graph the builder called
  // 'Untitled Model' otherwise means opening it and double-clicking the title in
  // the builder chrome — which is why 199 of them stayed 'Untitled Model'.
  const renameProject = async (project: SavedGraph) => {
    const next = hostServices.prompt('Name this model', project.name)?.trim();
    if (!next || next === project.name) return;
    try {
      const saved = await GraphStorageService.renameGraph(project, next);
      setActionError(null);
      setProjects((prev) => prev.map((candidate) => (candidate.id === saved.id ? saved : candidate)));
    } catch (error) {
      // Fail loud: the row must not show a name the server never accepted.
      setActionError(
        `${project.name} was not renamed. `
        + (error instanceof Error ? error.message : 'The request did not complete.')
      );
    }
  };

  const removeProject = async (id: string) => {
    if (!hostServices.confirm('Delete this model?')) return;
    try {
      const ok = await GraphStorageService.deleteGraph(id, draftStore);
      // A false return means the delete did not happen. Dropping the row anyway
      // would show the project gone until the next load brought it back, and
      // doing nothing at all made the button look broken.
      if (!ok) {
        setActionError('That model was not deleted — the backend rejected the request.');
        return;
      }
      setActionError(null);
      setProjects((prev) => prev.filter((project) => project.id !== id));
    } catch {
      setActionError('Delete failed — the request did not complete.');
    }
  };

  const toggleSelected = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  /**
   * Select-all acts on the rows actually in front of you, not on the whole
   * list: with a filter or a name search applied, an "all" that reaches rows
   * you cannot see is how a bulk delete takes out the wrong models.
   */
  const visibleIds = visible.map(({ project }) => project.id);
  const allVisibleSelected = visibleIds.length > 0 && visibleIds.every((id) => selected.has(id));
  const toggleAllVisible = () => {
    setSelected((prev) => {
      const next = new Set(prev);
      visibleIds.forEach((id) => (allVisibleSelected ? next.delete(id) : next.add(id)));
      return next;
    });
  };

  /**
   * Delete every selected model, behind one confirm that says how many.
   *
   * Each id is a separate DELETE — there is no bulk endpoint — so the outcomes
   * are partial by nature: rows the server refused stay in the table AND stay
   * selected (so a retry needs no re-picking), and they are named in the error
   * rather than disappearing from the list as though they were gone.
   */
  const removeSelected = async () => {
    const ids = Array.from(selected);
    if (ids.length === 0) return;
    if (!hostServices.confirm(
      `Delete ${ids.length} model${ids.length === 1 ? '' : 's'}? This cannot be undone.`
    )) return;
    const outcomes = await Promise.all(ids.map(async (id) => {
      try {
        return { id, ok: await GraphStorageService.deleteGraph(id, draftStore) };
      } catch {
        return { id, ok: false };
      }
    }));
    const deleted = new Set(outcomes.filter((outcome) => outcome.ok).map((outcome) => outcome.id));
    const failed = outcomes.filter((outcome) => !outcome.ok).map((outcome) => outcome.id);
    setProjects((prev) => prev.filter((project) => !deleted.has(project.id)));
    setSelected(new Set(failed));
    setActionError(
      failed.length === 0
        ? null
        : `${failed.length} of ${ids.length} were not deleted — still here: `
          + `${failed.map((id) => projects.find((p) => p.id === id)?.name ?? id).join(', ')}.`
    );
  };

  const railSlot = (
    <div className="flex flex-col">
      <div className="ui2-eyebrow">Totals</div>
      <div className="ui2-env-row">
        <span>Models</span>
        <span>{projects.length}</span>
      </div>
      <div className="ui2-env-row">
        <span>Layers</span>
        <span>{totalLayers}</span>
      </div>
      <div className="ui2-env-row">
        <span>Verified</span>
        <span style={{ color: counts.verified > 0 ? 'var(--ui2-green)' : 'var(--ui2-text-dim)' }}>
          {counts.verified}
        </span>
      </div>
      <div className="ui2-env-row">
        <span>Failing</span>
        <span style={{ color: counts.error > 0 ? 'var(--ui2-red)' : 'var(--ui2-text-dim)' }}>
          {counts.error}
        </span>
      </div>
    </div>
  );

  return (
    <WorkbenchShell railSlot={railSlot} projectCount={projects.length}>
      <div className="ui2-page">
        {actionError && (
          <section className="ui2-panel ui2-notice-error p-4 mb-5 flex items-center gap-3" role="alert">
            <div className="min-w-0 flex-1 text-sm">{actionError}</div>
            <button className="ui2-chip" onClick={() => setActionError(null)}>
              Dismiss
            </button>
          </section>
        )}

        {exportNotice && (
          <section className="ui2-panel ui2-notice-info p-4 mb-5 flex items-center gap-3" role="status">
            <div className="min-w-0 flex-1 text-sm">{exportNotice}</div>
            <button className="ui2-chip" onClick={() => setExportNotice(null)}>
              Dismiss
            </button>
          </section>
        )}

        <div
          className="flex flex-wrap items-center gap-3 pb-4"
          style={{ borderBottom: '1px solid var(--ui2-rule)' }}
        >
          <div className="ui2-pipe-tabs" role="tablist" aria-label="Filter models by status">
            {FILTERS.map(({ key, label }) => (
              <button
                key={key}
                role="tab"
                aria-selected={filter === key}
                onClick={() => setFilter(key)}
              >
                {label} {counts[key]}
              </button>
            ))}
          </div>
          <div className="ml-auto flex items-center gap-3">
            <span
              style={{ fontFamily: 'var(--font-mono)', fontSize: '11.5px', color: 'var(--ui2-text-dim)' }}
            >
              Sorted by edited ↓
            </span>
            <span
              className="inline-flex items-center gap-2"
              style={{
                border: '1px solid var(--ui2-border)',
                borderRadius: '8px',
                padding: '6px 11px',
                width: '240px',
                background: 'rgba(255, 255, 255, 0.03)',
              }}
            >
              <Search size={14} style={{ color: 'var(--ui2-text-dim)', flex: '0 0 auto' }} />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Filter by name"
                aria-label="Filter models by name"
                style={{
                  flex: '1 1 auto',
                  minWidth: 0,
                  background: 'transparent',
                  border: 0,
                  outline: 'none',
                  color: 'var(--ui2-text)',
                  fontSize: '12.5px',
                }}
              />
            </span>
            {selected.size > 0 && (
              <button className="ui2-chip" onClick={() => { void removeSelected(); }}>
                <Trash2 size={13} className="inline mr-1" />
                Delete {selected.size}
              </button>
            )}
            <button className="ui2-chip ui2-chip-primary" onClick={() => navigate('/builder')}>
              <Plus size={13} className="inline mr-1" />
              New
            </button>
          </div>
        </div>

        {loading ? (
          <div className="ui2-muted pt-4">Loading models…</div>
        ) : error ? (
          <section className="ui2-panel ui2-notice-error p-5 mt-5 flex items-center gap-3" role="alert">
            <div className="min-w-0 flex-1">
              <div className="text-sm font-semibold">{error}</div>
              <div className="ui2-muted text-sm">The model list could not be read.</div>
            </div>
            <button className="ui2-chip" onClick={() => setReloadKey((key) => key + 1)}>
              Retry
            </button>
          </section>
        ) : projects.length === 0 ? (
          <div className="ui2-muted pt-4">No models yet. Create one from the builder.</div>
        ) : (
          <>
            <div className="ui2-data-table ui2-data-table-head">
              <span className="ui2-eyebrow">Model</span>
              <span className="ui2-eyebrow">Status</span>
              <span className="ui2-eyebrow" style={{ textAlign: 'right' }}>
                Layers
              </span>
              {/* Same reserved marker gutter as the cells below, so the header
                  sits over its own figures rather than over the "est." column. */}
              <span
                className="ui2-eyebrow ui2-num-estimable"
                style={{ textAlign: 'right' }}
              >
                Params
              </span>
              <span className="ui2-eyebrow" style={{ textAlign: 'right' }}>
                Size
              </span>
              <span className="ui2-eyebrow">Edited</span>
              <span className="ui2-eyebrow flex justify-end">
                <input
                  type="checkbox"
                  checked={allVisibleSelected}
                  onChange={toggleAllVisible}
                  aria-label="Select all listed models"
                  style={{ width: 13, height: 13, accentColor: 'var(--ui2-accent)', cursor: 'pointer' }}
                />
              </span>
            </div>
            {visible.length === 0 ? (
              <div className="ui2-muted pt-4">No models match this filter.</div>
            ) : (
              visible.map(({ project, record, status, metrics }) => {
                // A failing row quotes its first error where the description
                // would go: the reason it broke outranks what it is. With
                // neither, the cell stays empty — an em dash would imply the
                // description was checked and found missing.
                const firstError = status === 'error' ? record?.firstError : undefined;
                const subline = firstError ?? project.description;
                return (
                  // The row is a div, not a button: it carries Delete and Open
                  // controls, and nesting one button inside another is invalid
                  // HTML with undefined activation behaviour. The first cell is
                  // the primary control, and .ui2-data-table lights the whole
                  // row from that child's hover via :has() — the pattern
                  // .ui2-row already uses, so a pointer parked on Delete does
                  // not make the row look as though it will open.
                  <div key={project.id} className="ui2-data-table">
                    <button
                      className="ui2-data-table-primary"
                      onClick={() => navigate(`/builder/${project.id}`)}
                    >
                      <span className="ui2-card-name block">{project.name}</span>
                      {subline ? (
                        <span
                          className="block"
                          style={{
                            fontSize: '12px',
                            marginTop: '3px',
                            color: firstError ? 'var(--ui2-red)' : 'var(--ui2-text-dim)',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {subline}
                        </span>
                      ) : null}
                    </button>
                    <StatusPill variant="inline" status={status} errorCount={record?.errorCount} />
                    <span className="ui2-num">{metrics.layers}</span>
                    {/* ui2-num-estimable reserves the marker's gutter on every
                        row, so an exact figure and an estimated one share the
                        same right edge. */}
                    <span className="ui2-num ui2-num-estimable">
                      {formatParams(metrics.params)}
                      {/* Marked once per row, on the params cell, and only when
                          there is a figure to qualify: an em dash is a missing
                          number, not an estimate. */}
                      {!metrics.exact && metrics.params > 0 ? (
                        <span className="ui2-metric-estimated">est.</span>
                      ) : null}
                    </span>
                    <span className="ui2-num">{formatSize(metrics.sizeBytes)}</span>
                    <span style={{ fontSize: '12.5px', color: 'var(--ui2-text-dim)' }}>
                      {formatEditedAt(project.updatedAt)}
                    </span>
                    <span className="flex justify-end gap-1.5 items-center">
                      <input
                        type="checkbox"
                        checked={selected.has(project.id)}
                        onChange={() => toggleSelected(project.id)}
                        aria-label={`Select ${project.name}`}
                        style={{ width: 13, height: 13, accentColor: 'var(--ui2-accent)', cursor: 'pointer' }}
                      />
                      {/* A menu, not two more icons: a second export button
                          would need a fourth 28px slot in a track sized for
                          three, and the same ⋯ affordance already carries these
                          actions on the home cards. */}
                      <CardMenu
                        label={`More actions for ${project.name}`}
                        busy={exportingId === project.id}
                        items={[
                          {
                            key: 'rename',
                            label: 'Rename',
                            icon: Pencil,
                            run: () => {
                              void renameProject(project);
                            },
                          },
                          {
                            key: 'export-python',
                            label: 'Export as Python',
                            icon: Download,
                            run: () => {
                              void exportProject(project, 'python');
                            },
                          },
                          {
                            key: 'export-zip',
                            label: 'Export as zip',
                            icon: FileArchive,
                            run: () => {
                              void exportProject(project, 'zip');
                            },
                          },
                        ]}
                      />
                      <button
                        className="ui2-icon-button"
                        style={{ width: '28px', height: '28px' }}
                        aria-label={`Delete ${project.name}`}
                        onClick={() => {
                          void removeProject(project.id);
                        }}
                      >
                        <Trash2 size={13} />
                      </button>
                      <button
                        className="ui2-icon-button"
                        style={{ width: '28px', height: '28px' }}
                        aria-label={`Open ${project.name}`}
                        onClick={() => navigate(`/builder/${project.id}`)}
                      >
                        <ChevronRight size={13} />
                      </button>
                    </span>
                  </div>
                );
              })
            )}
          </>
        )}
      </div>
    </WorkbenchShell>
  );
}
