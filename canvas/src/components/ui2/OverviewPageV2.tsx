import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SquareTerminal, Boxes, FolderSearch, FileCode2, Github, X, Trash2, UploadCloud, Download, FileArchive, Pencil } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { listUserTemplates, deleteUserTemplate, createGraphFromUserTemplate, UserTemplate } from '../../services/userTemplatesApi';
import { galleryTemplateCount } from '../../services/templateGallery';
import * as hostServices from '../../ide/hostServices';
import WorkbenchShell from './WorkbenchShell';
import ModernTemplateGallery from '../ModernTemplateGallery';
import ModelImportDialog from './ModelImportDialog';
import GraphThumb from './GraphThumb';
import StatusPill from './StatusPill';
import { GraphStorageService } from '../../services/graphStorage';
import { hierarchicalToFlat } from '../../services/hierarchicalToFlat';
import { readVerification, projectStatus, ProjectStatus } from '../../services/verificationRecord';
import { projectMetrics, formatParams, formatSize } from '../../services/graphMetrics';
import { formatEditedAt } from '../../utils/relativeTime';
import CardMenu from './CardMenu';
import { exportProjectAsPython, exportProjectAsZip } from '../../services/exportProjectCode';
import { useAuth } from '../../contexts/AuthContext';
import {
  createGraphDraftStore,
  LOCAL_DEVELOPMENT_DRAFT_OWNER,
} from '../../services/localDraft';
import { SavedGraph } from '../../types';

const REPO_URL = 'https://github.com/tensorviz-org/tensorviz';

// Legacy rows can carry a blank/whitespace name, which would render an
// invisible, unclickable-looking title line.
function templateLabel(t: UserTemplate): string {
  return t.name?.trim() || 'Untitled template';
}

/** The rail's pinned dot. Only a real outcome earns a hue — see StatusPill. */
const DOT_COLOR: Record<ProjectStatus, string> = {
  verified: 'var(--ui2-green)',
  error: 'var(--ui2-red)',
  stale: 'var(--ui2-text-dim)',
  'never-run': 'var(--ui2-text-dim)',
};

function thumbStatus(status: ProjectStatus): 'verified' | 'error' | 'unvalidated' {
  if (status === 'error') return 'error';
  if (status === 'verified') return 'verified';
  return 'unvalidated';
}

/**
 * Whole rows only. A 3-column grid showing 4 or 5 cards leaves a hole beside a
 * heading, so Continue renders 1, 2, 3 or 6 — never a partial row. The overflow
 * has a home behind "All N projects".
 */
function continueCount(total: number): number {
  if (total >= 6) return 6;
  if (total >= 3) return 3;
  return total;
}

export default function OverviewPageV2() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<SavedGraph[]>([]);
  const [showTemplateGallery, setShowTemplateGallery] = useState(false);
  const [showModelImport, setShowModelImport] = useState(false);
  const [templateError, setTemplateError] = useState<string | null>(null);
  const [unmigrated, setUnmigrated] = useState<SavedGraph[]>([]);
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);

  const [myTemplates, setMyTemplates] = useState<UserTemplate[]>([]);
  // How many starters the gallery actually serves. null = not counted (the
  // probe failed), which renders as no count at all rather than a "0" that
  // would repeat the very claim this fixes.
  const [galleryCount, setGalleryCount] = useState<number | null>(null);
  const [recentStatus, setRecentStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [templatesFailed, setTemplatesFailed] = useState(false);

  const [query, setQuery] = useState('');
  const [source, setSource] = useState('blank');
  const commandInput = useRef<HTMLInputElement>(null);

  // The card menu acts on a project, so this page now needs the same delete
  // authority the projects table has: deleteGraph clears the owner's local
  // draft alongside the server record.
  const { user, configured } = useAuth();
  const draftOwner = configured ? user?.uid ?? null : LOCAL_DEVELOPMENT_DRAFT_OWNER;
  const draftStore = useMemo(() => createGraphDraftStore(draftOwner), [draftOwner]);
  const [cardError, setCardError] = useState<string | null>(null);
  const [cardNotice, setCardNotice] = useState<string | null>(null);
  const [busyProjectId, setBusyProjectId] = useState<string | null>(null);

  const loadRecent = useCallback(() => {
    setRecentStatus('loading');
    GraphStorageService.getAllGraphs()
      .then((all) => {
        // The whole list, not a page of it: the band needs the total to say how
        // much it is hiding, and the rail needs the same order for PINNED.
        setProjects(all);
        setRecentStatus('ready');
      })
      // Fail loud: an unreachable backend must not render as "you have no projects".
      .catch(() => {
        setProjects([]);
        setRecentStatus('error');
      });
  }, []);

  const loadMyTemplates = useCallback(() => {
    listUserTemplates()
      .then((t) => {
        setMyTemplates(t);
        setTemplatesFailed(false);
      })
      .catch(() => {
        setMyTemplates([]);
        setTemplatesFailed(true);
      });
  }, []);

  useEffect(() => {
    loadRecent();
    loadMyTemplates();
    galleryTemplateCount().then(setGalleryCount).catch(() => setGalleryCount(null));
    if (!GraphStorageService.isLocalGraphImportDismissed()) {
      setUnmigrated(GraphStorageService.getUnmigratedLocalGraphs());
    }
  }, [loadRecent, loadMyTemplates]);

  // ⌘K/Ctrl+K is the shortcut the bar advertises, so it has to actually work.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        commandInput.current?.focus();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  const openMyTemplate = async (t: UserTemplate) => {
    setTemplateError(null);
    // Fail loud: an empty template would otherwise drop the user into a blank
    // builder that reads as "the template is broken".
    if (!t.graph?.nodes?.length) {
      setTemplateError(`“${templateLabel(t)}” has no layers saved in it.`);
      return;
    }
    try {
      const record = await createGraphFromUserTemplate(t);
      navigate(`/builder/${record.id}`);
    } catch (e) {
      setTemplateError(e instanceof Error ? e.message : 'Could not open that template.');
    }
  };

  const removeMyTemplate = async (t: UserTemplate) => {
    if (!hostServices.confirm(`Delete template “${templateLabel(t)}”? This cannot be undone.`)) return;
    try {
      await deleteUserTemplate(t.id);
      loadMyTemplates();
    } catch (e) {
      setTemplateError(e instanceof Error ? e.message : 'Could not delete that template.');
    }
  };

  const importLocalGraphs = async () => {
    setImporting(true);
    setImportError(null);
    try {
      await GraphStorageService.importLocalGraphs();
      setUnmigrated([]);
      loadRecent();
    } catch (e) {
      // Fail loud: the banner stays so the local graphs are not silently lost.
      setImportError(e instanceof Error ? e.message : 'Could not import your local graphs.');
    } finally {
      setImporting(false);
    }
  };

  // "Not now": persist the dismissal so a refresh does not bring the banner
  // back. The legacy blob is NOT marked migrated, so the graphs are still there
  // and still importable — we just stop asking on every visit.
  const dismissMigration = () => {
    GraphStorageService.dismissLocalGraphImport();
    setUnmigrated([]);
  };

  const handleTemplateSelected = async (templateData: any) => {
    setShowTemplateGallery(false);
    setTemplateError(null);
    if (templateData.type === 'popular_model' || templateData.type === 'template') {
      try {
        const graphData = templateData.data.graph;
        const flat = hierarchicalToFlat(graphData);
        if (!flat.nodes?.length) throw new Error('template produced an empty graph');
        const displayName = templateData.model?.name || templateData.data?.template_id || 'Template Model';
        const saved = await GraphStorageService.saveGraph(flat, displayName, `Generated from ${displayName}`);
        navigate(`/builder/${saved.id}`);
      } catch (e) {
        // Fail loud: surface the error instead of silently dropping the user into
        // an empty builder (which reads as "templates are broken").
        console.error('Template processing failed', e);
        setTemplateError(e instanceof Error ? e.message : 'Could not open that template.');
      }
    }
  };

  /**
   * Validate the card's model and download it, as a single model.py or as the
   * runnable package. Shared with the projects table, so a project exports the
   * same way from either page.
   */
  const exportProject = async (project: SavedGraph, format: 'python' | 'zip') => {
    setCardError(null);
    setCardNotice(null);
    setBusyProjectId(project.id);
    let result;
    try {
      result = await (format === 'zip'
        ? exportProjectAsZip(project)
        : exportProjectAsPython(project));
    } finally {
      setBusyProjectId(null);
    }
    if (!result.ok) {
      setCardError(`${project.name} was not exported. ${result.message}`);
      return;
    }
    // A partial export still downloads — the file carries the backend's own
    // "# INCOMPLETE EXPORT" banner — but it is not reported as the whole model.
    setCardNotice(
      result.unsupported.length > 0
        ? `${result.fileName} downloaded, but ${result.unsupported.length} module`
          + `${result.unsupported.length > 1 ? 's are' : ' is'} not reproduced in it: `
          + `${result.unsupported.map((m) => m.class_name).join(', ')}.`
        : `${result.fileName} downloaded.`
    );
  };

  // Naming has to be cheap from the list, or every graph the builder autonamed
  // 'Untitled Model' stays that way: renaming it used to mean opening it and
  // double-clicking the title in the builder chrome.
  const renameProject = async (project: SavedGraph) => {
    const next = hostServices.prompt('Name this model', project.name)?.trim();
    if (!next || next === project.name) return;
    setCardError(null);
    setCardNotice(null);
    try {
      const saved = await GraphStorageService.renameGraph(project, next);
      setProjects((prev) => prev.map((candidate) => (candidate.id === saved.id ? saved : candidate)));
      setCardNotice(`Renamed to ${saved.name}.`);
    } catch (error) {
      // Fail loud: a rename that did not land must not leave the card showing
      // the new name until the next load quietly reverted it.
      setCardError(
        `${project.name} was not renamed. `
        + (error instanceof Error ? error.message : 'The request did not complete.')
      );
    }
  };

  const removeProject = async (project: SavedGraph) => {
    if (!hostServices.confirm(`Delete ${project.name}?`)) return;
    setCardError(null);
    setCardNotice(null);
    try {
      const ok = await GraphStorageService.deleteGraph(project.id, draftStore);
      // A false return means the delete did not happen. Dropping the card anyway
      // would show the project gone until the next load brought it back.
      if (!ok) {
        setCardError(`${project.name} was not deleted — the backend rejected the request.`);
        return;
      }
      setProjects((prev) => prev.filter((candidate) => candidate.id !== project.id));
    } catch {
      setCardError(`${project.name} was not deleted — the request did not complete.`);
    }
  };

  const openBlankGraph = () => navigate('/builder');
  const openTemplateGallery = () => setShowTemplateGallery(true);
  // The two import entry points are distinct choices, so each opens the dialog
  // on the tab it names rather than making the user re-pick.
  const [importMode, setImportMode] = useState<'library' | 'repo'>('library');
  const openModelImport = (mode: 'library' | 'repo' = 'library') => {
    setImportMode(mode);
    setShowModelImport(true);
  };
  const openPasteCode = () => navigate('/builder?import=1');
  const openSandboxes = () => navigate('/sandboxes');

  // The band used to count only the user's own saved templates, so a first run
  // read "0 starter blocks" on the same screen the gallery serves its starter
  // architectures from. The gallery's own count leads; a saved count is added
  // once there is one.
  const templateBandNote = [
    galleryCount === null ? null : `${galleryCount} in the gallery`,
    myTemplates.length > 0 ? `${myTemplates.length} saved` : null,
  ].filter(Boolean).join(' · ');

  const total = projects.length;
  const shown = projects.slice(0, continueCount(total));
  const pinned = projects.slice(0, 3);

  // Substring match over the user's OWN projects only. A library/torchvision
  // catalogue search would need a backend call this page never makes, and
  // faking one would be a hardcoded number pretending to be a live fact.
  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return [];
    return projects.filter((p) => (p.name ?? '').toLowerCase().includes(needle)).slice(0, 6);
  }, [projects, query]);

  const onCommandKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    const first = matches[0];
    if (first) navigate(`/builder/${first.id}`);
    else if (!query.trim()) navigate('/builder');
  };

  const sources: Array<{ id: string; label: string; title?: string; run: () => void }> = [
    { id: 'blank', label: 'Blank graph', title: 'New model', run: openBlankGraph },
    { id: 'template', label: 'Template', run: openTemplateGallery },
    { id: 'library', label: 'Library model', run: () => openModelImport('library') },
    { id: 'repo', label: 'Repo folder', run: () => openModelImport('repo') },
    { id: 'paste', label: 'Paste code', run: openPasteCode },
    { id: 'sandbox', label: 'Sandbox', run: openSandboxes },
  ];

  const importRoutes = [
    { id: 'library', Icon: Boxes, title: 'Import a library model', note: 'From the libraries installed on this machine', run: () => openModelImport('library') },
    { id: 'repo', Icon: FolderSearch, title: 'Scan a repo folder', note: 'Static analysis, then you pick the entrypoint', run: () => openModelImport('repo') },
    { id: 'paste', Icon: FileCode2, title: 'Paste nn.Module code', note: 'Parsed into a graph, never executed', run: openPasteCode },
  ];

  // Rendered by the shell only when the rail is expanded, so it costs nothing
  // on the compact rail. Order is the recents order: most recently edited wins.
  const railSlot =
    pinned.length > 0 ? (
      <div>
        <div className="ui2-eyebrow" style={{ marginBottom: 12 }}>Pinned</div>
        <div className="flex flex-col">
          {pinned.map((p, index) => {
            const status = projectStatus(readVerification(p.document), p.currentRevision);
            return (
              <button
                key={p.id}
                onClick={() => navigate(`/builder/${p.id}`)}
                className="flex items-center gap-2.5 text-left"
                style={{
                  padding: '10px 0',
                  borderBottom: index < pinned.length - 1 ? '1px solid var(--ui2-rule)' : undefined,
                }}
              >
                <span className="ui2-status-dot" style={{ background: DOT_COLOR[status] }} />
                <span className="truncate" style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--ui2-text)' }}>
                  {p.name}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    ) : null;

  return (
    <>
    {showTemplateGallery && (
      <ModernTemplateGallery
        onTemplateSelected={handleTemplateSelected}
        onClose={() => setShowTemplateGallery(false)}
      />
    )}
    {showModelImport && (
      <ModelImportDialog
        initialMode={importMode}
        onClose={() => setShowModelImport(false)}
        onImported={(result) => {
          setShowModelImport(false);
          // The backend already persisted the graph and told us where it lives, so go
          // straight there instead of round-tripping the payload through the client.
          if (result.builderPath) navigate(result.builderPath);
          else loadRecent();
        }}
      />
    )}
    {/* projectCount is omitted until the list actually loaded: a 0 we never
        counted would read in the rail as "you have no projects". */}
    <WorkbenchShell railSlot={railSlot} projectCount={recentStatus === 'ready' ? total : undefined}>
      <div className="ui2-page">
        {cardError && (
          <section className="ui2-panel ui2-notice-error p-4 mb-5 flex items-center gap-3" role="alert">
            <div className="min-w-0 flex-1 text-sm">{cardError}</div>
            <button className="ui2-chip" onClick={() => setCardError(null)}>
              Dismiss
            </button>
          </section>
        )}

        {cardNotice && (
          <section className="ui2-panel ui2-notice-info p-4 mb-5 flex items-center gap-3" role="status">
            <div className="min-w-0 flex-1 text-sm">{cardNotice}</div>
            <button className="ui2-chip" onClick={() => setCardNotice(null)}>
              Dismiss
            </button>
          </section>
        )}

        {templateError && (
          <section className="ui2-panel ui2-notice-error p-5 mb-6 flex items-center gap-3" role="alert">
            <X size={18} style={{ color: 'var(--ui2-red)' }} />
            <div className="min-w-0 flex-1">
              <div className="text-sm font-semibold">Couldn’t complete that action</div>
              <div className="ui2-muted text-sm">{templateError}</div>
            </div>
            <button className="ui2-chip" onClick={() => setTemplateError(null)} aria-label="Dismiss">Dismiss</button>
          </section>
        )}
        {unmigrated.length > 0 && (
          <section
            className={`ui2-panel p-5 mb-6 flex items-center gap-3 ${importError ? 'ui2-notice-error' : 'ui2-notice-info'}`}
            role={importError ? 'alert' : undefined}
          >
            <UploadCloud size={18} style={{ color: importError ? 'var(--ui2-red)' : 'var(--ui2-accent)' }} />
            <div className="min-w-0 flex-1">
              <div className="font-medium">
                Import {unmigrated.length} local graph{unmigrated.length === 1 ? '' : 's'} to your account
              </div>
              <div className="ui2-muted text-sm">
                {importError ?? 'Found graphs saved in this browser. Import them to sync with your account.'}
              </div>
            </div>
            <button className="ui2-chip ui2-chip-primary" onClick={importLocalGraphs} disabled={importing}>
              {importing ? 'Importing…' : 'Import'}
            </button>
            <button className="ui2-chip" onClick={dismissMigration} disabled={importing} aria-label="Not now">
              <X size={13} className="inline mr-1" /> Not now
            </button>
          </section>
        )}

        {/* The command surface replaces the hero, the KPI band and the six-row
            quick-start rail. It is relative so the results panel can float over
            the bands instead of shoving them down on every keystroke. */}
        <div style={{ position: 'relative', zIndex: 5 }}>
          <div className="ui2-eyebrow" style={{ marginBottom: 18 }}>Find a model</div>
          <div className="ui2-command-bar">
            <SquareTerminal size={18} aria-hidden="true" style={{ color: 'var(--ui2-text-dim)', flex: '0 0 auto' }} />
            <input
              ref={commandInput}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={onCommandKeyDown}
              placeholder="Search your models"
              aria-label="Find a model"
              spellCheck={false}
            />
            <span className="ui2-kbd">⌘K</span>
          </div>

          <div className="ui2-pipe-tabs" role="tablist" aria-label="Source" style={{ marginTop: 14 }}>
            {sources.map((item) => (
              <button
                key={item.id}
                role="tab"
                aria-selected={source === item.id}
                title={item.title}
                onClick={() => {
                  setSource(item.id);
                  item.run();
                }}
              >
                {item.label}
              </button>
            ))}
          </div>

          {query.trim().length > 0 && (
            <div
              className="ui2-panel"
              style={{ position: 'absolute', left: 0, right: 0, top: '100%', marginTop: 8, overflow: 'hidden' }}
            >
              <div className="flex items-center gap-3" style={{ padding: '11px 16px', borderBottom: '1px solid var(--ui2-rule)' }}>
                <span className="ui2-eyebrow">Matches in your models</span>
                <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ui2-text-dim)' }}>
                  {matches.length > 0 ? '↵ to open' : 'No matching model'}
                </span>
              </div>
              {matches.length === 0 ? (
                <div style={{ padding: '11px 16px', fontSize: 13, color: 'var(--ui2-text-dim)' }}>
                  No model matches “{query.trim()}”. Try another name or create a blank graph.
                </div>
              ) : (
                matches.map((p) => {
                  const metrics = projectMetrics(p);
                  return (
                    <button key={p.id} className="ui2-candidate-row" onClick={() => navigate(`/builder/${p.id}`)}>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--ui2-text)' }}>{p.name}</span>
                      <span className="ui2-candidate-score">
                        {metrics.layers} layers · {formatParams(metrics.params)}
                      </span>
                    </button>
                  );
                })
              )}
            </div>
          )}
        </div>

        {recentStatus === 'loading' && <p className="ui2-muted" style={{ marginTop: 40 }}>Loading your models…</p>}
        {recentStatus === 'error' && (
          <div className="flex items-center gap-3" style={{ marginTop: 40 }} role="alert">
            <p className="ui2-muted min-w-0 flex-1">
              Couldn’t load your models. They haven’t been lost — the backend didn’t respond.
            </p>
            <button className="ui2-chip" onClick={loadRecent}>Retry</button>
          </div>
        )}

        {shown.length > 0 && (
          <section>
            <div className="ui2-band-head" style={{ marginTop: 40 }}>
              <h2 className="ui2-band-title">Continue</h2>
              {total > shown.length ? (
                <button className="ui2-chip" onClick={() => navigate('/projects')}>All {total} models</button>
              ) : (
                <span className="ui2-band-note">{shown.length} of {total} models</span>
              )}
            </div>
            <div className="ui2-card-grid">
              {shown.map((p) => {
                const record = readVerification(p.document);
                const status = projectStatus(record, p.currentRevision);
                const metrics = projectMetrics(p);
                return (
                  // A div card, not a button: the ⋯ menu is a control of its own
                  // and cannot nest inside another button. The card lights from
                  // .ui2-card-primary only, so a pointer parked on the menu does
                  // not make the card look as though it will open.
                  <div key={p.id} className="ui2-card" style={{ position: 'relative' }}>
                    <button className="ui2-card-primary" onClick={() => navigate(`/builder/${p.id}`)}>
                      <div style={{ position: 'relative' }}>
                        <GraphThumb
                          graph={p.graph}
                          status={thumbStatus(status)}
                          errorNodeId={record?.firstErrorNodeId}
                        />
                        <span style={{ position: 'absolute', left: 12, top: 12 }}>
                          <StatusPill status={status} errorCount={record?.errorCount} />
                        </span>
                      </div>
                      <div className="ui2-card-body">
                        <div className="ui2-card-name">{p.name}</div>
                        <div className="ui2-metric-strip">
                          <span>{metrics.layers} layer{metrics.layers === 1 ? '' : 's'}</span>
                          <span>{formatParams(metrics.params)}</span>
                          <span>{formatSize(metrics.sizeBytes)}</span>
                          {/* An estimate is labelled as one — see graphMetrics. */}
                          {!metrics.exact && <span className="ui2-metric-estimated">est.</span>}
                        </div>
                        {status === 'error' && record?.firstError ? (
                          <div style={{ fontSize: 12, color: 'var(--ui2-red)' }}>{record.firstError}</div>
                        ) : (
                          <div style={{ fontSize: 12, color: 'var(--ui2-text-dim)' }}>Edited {formatEditedAt(p.updatedAt)}</div>
                        )}
                      </div>
                    </button>
                    <CardMenu
                      label={`More actions for ${p.name}`}
                      triggerStyle={{ position: 'absolute', right: 8, top: 8 }}
                      busy={busyProjectId === p.id}
                      items={[
                        {
                          key: 'rename',
                          label: 'Rename',
                          icon: Pencil,
                          run: () => {
                            void renameProject(p);
                          },
                        },
                        {
                          key: 'export-python',
                          label: 'Export as Python',
                          icon: Download,
                          run: () => {
                            void exportProject(p, 'python');
                          },
                        },
                        {
                          key: 'export-zip',
                          label: 'Export as zip',
                          icon: FileArchive,
                          run: () => {
                            void exportProject(p, 'zip');
                          },
                        },
                        {
                          key: 'delete',
                          label: 'Delete model',
                          icon: Trash2,
                          danger: true,
                          run: () => {
                            void removeProject(p);
                          },
                        },
                      ]}
                    />
                  </div>
                );
              })}
            </div>
          </section>
        )}

        <section>
          <div className="ui2-band-head" style={{ marginTop: 34 }}>
            <h2 className="ui2-band-title">Start from a template</h2>
            <div className="flex items-center gap-3">
              {!templatesFailed && templateBandNote && (
                <span className="ui2-band-note">{templateBandNote}</span>
              )}
              <button className="ui2-chip" onClick={openTemplateGallery}>Choose from Template</button>
            </div>
          </div>
          {/* Fail loud: a failed fetch used to make this whole band vanish, which
              is indistinguishable from "you saved no templates". */}
          {templatesFailed ? (
            <div className="flex items-center gap-3" style={{ marginTop: 18 }} role="alert">
              <p className="ui2-muted min-w-0 flex-1">Couldn’t load your saved templates.</p>
              <button className="ui2-chip" onClick={loadMyTemplates}>Retry</button>
            </div>
          ) : myTemplates.length === 0 ? (
            <p className="ui2-muted" style={{ marginTop: 18 }}>
              {galleryCount === null
                ? 'Open the gallery to start from a shipped architecture.'
                : `${galleryCount} starter architectures are in the gallery, ready to open.`
                  + ' None saved to your account yet.'}
            </p>
          ) : (
            <div className="ui2-card-grid ui2-card-grid-4">
              {myTemplates.map((t) => {
                const label = templateLabel(t);
                const layers = t.graph?.nodes?.length ?? 0;
                return (
                  // A div card, not a button: the delete affordance is a control
                  // of its own and cannot nest inside another button.
                  <div key={t.id} className="ui2-card" style={{ position: 'relative' }}>
                    <button
                      onClick={() => openMyTemplate(t)}
                      title="Start a new graph from this template"
                      style={{ display: 'block', width: '100%', textAlign: 'left', cursor: 'pointer' }}
                    >
                      <GraphThumb graph={t.graph} height={88} />
                      <div className="ui2-card-body" style={{ gap: 5 }}>
                        <div className="flex items-baseline gap-2.5">
                          <span className="ui2-card-name" style={{ fontSize: 12.5, minWidth: 0 }}>{label}</span>
                          <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--ui2-text-dim)' }}>
                            {layers}L
                          </span>
                        </div>
                        <div style={{ fontSize: 12, lineHeight: '17px', color: 'var(--ui2-text-dim)' }}>
                          {t.description || 'Personal template'}
                        </div>
                      </div>
                    </button>
                    <button
                      className="ui2-icon-button"
                      onClick={() => removeMyTemplate(t)}
                      aria-label={`Delete template ${label}`}
                      title="Delete template"
                      style={{ position: 'absolute', right: 8, top: 8, width: 28, height: 28 }}
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <section>
          <div className="ui2-band-head" style={{ marginTop: 34 }}>
            <h2 className="ui2-band-title">Or bring your own</h2>
            <span className="ui2-band-note">Nothing is uploaded — the backend runs on this machine</span>
          </div>
          <div className="ui2-card-grid" style={{ gap: 14 }}>
            {importRoutes.map(({ id, Icon, title, note, run }) => (
              <button
                key={id}
                className="ui2-card"
                onClick={run}
                style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '20px 16px' }}
              >
                <Icon size={16} aria-hidden="true" style={{ color: 'var(--ui2-accent)', flex: '0 0 auto' }} />
                <span className="min-w-0">
                  <span style={{ display: 'block', fontSize: 13, color: 'var(--ui2-text)' }}>{title}</span>
                  <span style={{ display: 'block', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ui2-text-dim)', marginTop: 3 }}>
                    {note}
                  </span>
                </span>
              </button>
            ))}
          </div>
          <div style={{ marginTop: 18, display: 'flex', alignItems: 'center', gap: 8 }}>
            <a
              href={REPO_URL}
              target="_blank"
              rel="noopener noreferrer"
              style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--ui2-text-dim)' }}
            >
              <Github size={14} aria-hidden="true" /> View on GitHub
            </a>
          </div>
        </section>
      </div>
    </WorkbenchShell>
    </>
  );
}
