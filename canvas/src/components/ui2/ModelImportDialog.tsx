/**
 * Import a model from an installed library or a local repository.
 *
 * Two modes, because the two cases genuinely differ. A library architecture is a single
 * name, so it is a search box over the catalog. A repository has an unknown entrypoint, so
 * it is a two-step flow: scan (pure static analysis — nothing is executed) and then pick a
 * ranked candidate. The ranking is a heuristic and is surfaced as a list rather than
 * applied silently, because training and loss wrappers legitimately outrank the backbone
 * they contain — they are what train.py instantiates.
 *
 * Styling is the shared dialog frame from styles/ui2.tokens.css (`.ui2-dialog*`,
 * `.ui2-tab-underline`, `.ui2-segmented`, `.ui2-candidate-row`); only the layout wrappers
 * those classes do not cover are inline.
 *
 * Every failure renders the backend's own message plus a retry, never an empty list: a
 * missing path and a repository that defines no models are different statements, and an
 * empty candidate list would say the second one when the truth is the first.
 */
import React, { useCallback, useEffect, useState } from 'react';
import {
  AlertTriangle,
  Boxes,
  ChevronRight,
  Folder,
  FolderSearch,
  Loader2,
  Search,
  X,
} from 'lucide-react';
import {
  CatalogEntry,
  EntrypointCandidate,
  IngestResult,
  getLibraries,
  ingestModel,
  ingestRepo,
  scanRepo,
  searchCatalog,
} from '../../services/ingestApi';
import * as hostServices from '../../ide/hostServices';

type Mode = 'library' | 'repo';

interface Props {
  onClose: () => void;
  onImported: (result: IngestResult) => void;
  /**
   * Which tab to open on. The home page's source strip offers "Library model"
   * and "Repo folder" as separate entry points, so landing on the library tab
   * after a user asked for a repo folder makes them re-pick what they already
   * picked.
   */
  initialMode?: Mode;
}

/** `retry` names the action to re-run, so the notice can offer it without re-deriving it. */
interface Failure {
  headline: string;
  detail: string;
  retry?: 'scan' | 'libraries';
}

const DEPTHS = [1, 2, 3];

/** `scan_only` ranks at most five candidates, so a five-long result is a window, not a total. */
const SCAN_LIMIT = 5;

const MONO = { fontFamily: 'var(--font-mono)' } as const;

const S = {
  hint: { fontSize: 12.5, lineHeight: '19px', color: 'var(--ui2-text-dim)' },
  input: {
    flex: 1, minWidth: 0, background: 'transparent', border: 'none', outline: 'none',
    color: 'var(--ui2-text)', fontFamily: 'var(--font-mono)', fontSize: 13,
  },
  bar: {
    display: 'flex', alignItems: 'center', gap: 10, padding: '9px 13px',
    background: 'rgba(255,255,255,0.03)', border: '1px solid var(--ui2-border-strong)',
    borderRadius: 9, color: 'var(--ui2-text-dim)',
  },
  rowTitle: { ...MONO, display: 'block', fontSize: 13, color: 'var(--ui2-text)' },
  rowMeta: {
    ...MONO, display: 'block', fontSize: 11, lineHeight: '16px', marginTop: 4,
    color: 'var(--ui2-text-dim)', overflow: 'hidden', textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as const,
  },
};

const messageOf = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function ModelImportDialog({ onClose, onImported, initialMode = 'library' }: Props) {
  const [mode, setMode] = useState<Mode>(initialMode);
  const [libraries, setLibraries] = useState<Record<string, boolean> | null>(null);
  const [library, setLibrary] = useState('torchvision');
  const [query, setQuery] = useState('');
  const [copiedInstall, setCopiedInstall] = useState<string | null>(null);
  const [entries, setEntries] = useState<CatalogEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [depth, setDepth] = useState(2);
  const [repo, setRepo] = useState('');
  const [scanned, setScanned] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<EntrypointCandidate[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [retryToken, setRetryToken] = useState(0);

  // A failed probe used to fall back to `{ torchvision: true }`, which asserted an install
  // nothing had verified. Unknown stays unknown and the notice offers the retry.
  useEffect(() => {
    getLibraries()
      .then(setLibraries)
      .catch((e) => setFailure({
        headline: 'Couldn’t list the installed libraries',
        detail: messageOf(e),
        retry: 'libraries',
      }));
  }, [retryToken]);

  // Debounced so typing does not fire a request per keystroke.
  useEffect(() => {
    if (mode !== 'library') return;
    const handle = window.setTimeout(() => {
      searchCatalog({ library, q: query, limit: 40 })
        .then((res) => { setEntries(res.entries); setTotal(res.total); })
        .catch((e) => setFailure({
          headline: `Couldn’t search the ${library} catalog`,
          detail: messageOf(e),
          retry: 'libraries',
        }));
    }, 250);
    return () => window.clearTimeout(handle);
  }, [mode, library, query, retryToken]);

  const finish = useCallback((result: IngestResult) => {
    if (!result.ok) {
      setFailure({ headline: 'Import failed', detail: result.error || 'The backend reported no reason.' });
      // A failed repo import returns the ranked alternatives to choose from instead; when it
      // returns none, the list already on screen is kept so the choice is still available.
      setCandidates((prev) => (result.candidates.length ? result.candidates : prev));
      return;
    }
    onImported(result);
  }, [onImported]);

  const run = useCallback(async (label: string, work: () => Promise<IngestResult>) => {
    setFailure(null);
    setBusy(label);
    try {
      finish(await work());
    } catch (e) {
      setFailure({ headline: label.replace('…', ' failed'), detail: messageOf(e) });
    } finally {
      setBusy(null);
    }
  }, [finish]);

  const doScan = useCallback(async () => {
    const path = repo.trim();
    if (!path) return;
    setFailure(null);
    setCandidates(null);
    setScanned(null);
    setBusy('Scanning repository…');
    try {
      const found = await scanRepo(path);
      setCandidates(found);
      setScanned(path);
    } catch (e) {
      const detail = messageOf(e);
      setFailure({
        headline: /\(404\)/.test(detail)
          ? `Repository not found: ${path}`
          : `Couldn’t scan ${path}`,
        detail,
        retry: 'scan',
      });
    } finally {
      setBusy(null);
    }
  }, [repo]);

  const available = Object.entries(libraries || {}).filter(([, ok]) => ok).map(([k]) => k);
  const missing = Object.entries(libraries || {}).filter(([, ok]) => !ok).map(([k]) => k);

  const depthBlock = (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 9 }}>
        <span className="ui2-eyebrow">Detail depth</span>
        <span className="ui2-segmented">
          {DEPTHS.map((d) => (
            <button onClick={() => setDepth(d)} key={d} type="button" aria-pressed={depth === d}>
              {d}
            </button>
          ))}
        </span>
      </div>
      <div style={S.hint}>
        Blocks arrive collapsed — expand them on the canvas. Depth 1–2 stays legible;
        deeper is bigger, not more informative.
      </div>
    </div>
  );

  return (
    <div
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      className="ui2-dialog-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="Import a repository"
    >
      <div className="ui2-dialog" style={{ width: 720, maxWidth: '100%', maxHeight: '88vh' }}>
        <header className="ui2-dialog-head">
          <div style={{ minWidth: 0 }}>
            <h2 className="ui2-dialog-title">Import a repository</h2>
            <p className="ui2-dialog-lede">No weights are downloaded and nothing leaves this machine.</p>
          </div>
          <button
            onClick={onClose}
            type="button"
            aria-label="Close"
            className="ui2-icon-button"
            style={{ marginLeft: 'auto', flex: '0 0 auto' }}
          >
            <X size={16} />
          </button>
        </header>

        <div
          role="tablist"
          style={{
            display: 'flex', gap: 22, padding: '0 20px', flex: '0 0 auto',
            borderBottom: '1px solid var(--ui2-rule)',
          }}
        >
          <button
            onClick={() => { setMode('library'); setFailure(null); }}
            type="button"
            role="tab"
            aria-selected={mode === 'library'}
            className="ui2-tab-underline"
          >
            <Boxes size={14} /> Installed library
          </button>
          <button
            onClick={() => { setMode('repo'); setFailure(null); }}
            type="button"
            role="tab"
            aria-selected={mode === 'repo'}
            className="ui2-tab-underline"
          >
            <FolderSearch size={14} /> Repository folder
          </button>
        </div>

        <div
          className="ui2-dialog-body ui2-scroll-surface"
          style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 16 }}
        >
          {mode === 'library' ? (
            <>
              <div>
                <div className="ui2-eyebrow" style={{ marginBottom: 9 }}>Library</div>
                {libraries === null ? (
                  <div style={S.hint}>Checking which libraries are installed…</div>
                ) : (
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {available.map((lib) => (
                      <button
                        onClick={() => setLibrary(lib)}
                        key={lib}
                        type="button"
                        aria-pressed={library === lib}
                        className={`ui2-chip${library === lib ? ' ui2-chip-active' : ''}`}
                      >
                        {lib}
                      </button>
                    ))}
                    {missing.map((lib) => (
                      // Was a non-interactive span at 45% opacity whose only remedy
                      // was a `title` attribute: the one thing a user needs — the
                      // command that fixes it — appeared solely on hover, and never
                      // at all on a touch device. The command is on the chip now,
                      // and clicking copies it.
                      <button
                        key={lib}
                        type="button"
                        onClick={() => {
                          void hostServices.copyText(`pip install ${lib}`);
                          setCopiedInstall(lib);
                          window.setTimeout(() => setCopiedInstall(null), 2000);
                        }}
                        className="ui2-chip"
                        title={`Copy: pip install ${lib}`}
                        style={{
                          opacity: 0.8,
                          borderStyle: 'dashed',
                          display: 'inline-flex',
                          gap: 6,
                          alignItems: 'baseline',
                        }}
                      >
                        <span>{lib} · not installed</span>
                        <code style={{ fontSize: 11, opacity: 0.85 }}>
                          {copiedInstall === lib ? 'copied!' : `pip install ${lib}`}
                        </code>
                      </button>
                    ))}
                  </div>
                )}
                {missing.length > 0 && (
                  <div style={{ ...S.hint, marginTop: 8 }}>
                    A greyed-out library is not installed on the machine running the
                    backend. Install it there, restart the backend, and it becomes
                    selectable — nothing is downloaded to your browser.{' '}
                    <a href="/docs/ingestion.html" target="_blank" rel="noreferrer">
                      How importing works
                    </a>
                  </div>
                )}
              </div>

              <div style={S.bar}>
                <Search size={15} style={{ flex: '0 0 auto' }} />
                <input
                  autoFocus
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder={`Search ${library} architectures…`}
                  aria-label="Search architectures"
                  style={S.input}
                />
                <span style={{ ...MONO, fontSize: 11, flex: '0 0 auto' }}>{total}</span>
              </div>

              {depthBlock}

              <div style={{ borderTop: '1px solid var(--ui2-rule)', paddingTop: 16 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6, flexWrap: 'wrap' }}>
                  <span className="ui2-eyebrow">Architectures</span>
                  <span style={{ ...MONO, fontSize: 11, color: 'var(--ui2-text-dim)' }}>
                    showing {entries.length} of {total}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  {entries.map((entry, i) => (
                    <button
                      onClick={() => run(`Importing ${entry.arch}…`, () => ingestModel({ library, arch: entry.arch, depth }))}
                      disabled={!!busy}
                      key={`${entry.library}:${entry.arch}`}
                      type="button"
                      className="ui2-candidate-row"
                      style={i === entries.length - 1 ? { borderBottom: 'none' } : undefined}
                    >
                      <span style={{ minWidth: 0, flex: 1 }}>
                        <span style={S.rowTitle}>{entry.arch}</span>
                        <span style={S.rowMeta}>{entry.library}</span>
                      </span>
                      <span className="ui2-candidate-score">{entry.kind}</span>
                      <ChevronRight size={14} style={{ color: 'var(--ui2-text-dim)', flex: '0 0 auto' }} />
                    </button>
                  ))}
                  {!entries.length && !busy && !failure && (
                    <div style={S.hint}>No {library} architectures match “{query}”.</div>
                  )}
                </div>
              </div>
            </>
          ) : (
            <>
              <div
                style={{
                  display: 'flex', alignItems: 'flex-start', gap: 10, padding: '11px 13px',
                  border: '1px solid color-mix(in srgb, var(--ui2-orange) 40%, transparent)',
                  background: 'color-mix(in srgb, var(--ui2-orange) 12%, transparent)',
                  borderRadius: 9,
                }}
              >
                <AlertTriangle
                  size={15}
                  style={{ color: 'var(--ui2-orange)', flex: '0 0 auto', marginTop: 2 }}
                />
                <span style={{ fontSize: 12.5, lineHeight: '19px', color: 'var(--ui2-text)' }}>
                  Importing a repository <strong style={{ color: 'var(--ui2-orange)' }}>runs its code</strong>
                  {' '}— defining a model means importing its module. Only import repositories you
                  would run locally.
                </span>
              </div>

              <div>
                <div className="ui2-eyebrow" style={{ marginBottom: 9 }}>Folder</div>
                <div style={S.bar}>
                  <Folder size={15} style={{ flex: '0 0 auto' }} />
                  <input
                    autoFocus
                    value={repo}
                    onChange={(e) => setRepo(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') doScan(); }}
                    placeholder="~/code/your-repo"
                    aria-label="Repository path"
                    style={S.input}
                  />
                  <button
                    onClick={doScan}
                    disabled={!repo.trim() || !!busy}
                    type="button"
                    className="ui2-chip"
                    style={{ flex: '0 0 auto' }}
                  >
                    Scan
                  </button>
                </div>
              </div>

              {depthBlock}

              {candidates && (
                <div style={{ borderTop: '1px solid var(--ui2-rule)', paddingTop: 16 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6, flexWrap: 'wrap' }}>
                    <span className="ui2-eyebrow">Ranked candidates</span>
                    <span style={{ ...MONO, fontSize: 11, color: 'var(--ui2-text-dim)' }}>
                      {candidates.length === SCAN_LIMIT
                        ? `top ${SCAN_LIMIT} nn.Module subclasses`
                        : `${candidates.length} nn.Module subclasses`}
                      {' · '}
                      <span style={{ color: 'var(--ui2-green)' }}>nothing executed</span>
                    </span>
                  </div>
                  <div style={{ ...S.hint, marginBottom: 12 }}>
                    The scan is static analysis. The top result is a heuristic, not an answer —
                    training and loss wrappers legitimately outrank the backbone they contain.
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column' }}>
                    {candidates.map((c, i) => {
                      const evidence = [`${c.file}:${c.line}`, ...c.reasons].join(' · ');
                      return (
                        <button
                          onClick={() => run(`Importing ${c.name}…`, () => ingestRepo({ repo: repo.trim(), entrypoint: c.name, depth }))}
                          disabled={!!busy}
                          key={c.name}
                          type="button"
                          className="ui2-candidate-row"
                          data-top={i === 0 ? 'true' : undefined}
                          style={i === candidates.length - 1 ? { borderBottom: 'none' } : undefined}
                        >
                          <span style={{ minWidth: 0, flex: 1 }}>
                            <span style={S.rowTitle}>{c.name}</span>
                            <span style={S.rowMeta} title={evidence}>{evidence}</span>
                          </span>
                          {/* The backend's score is a points total from the ranking rules, not a
                              probability, so it carries its own label rather than reading as one. */}
                          <span className="ui2-candidate-score" title="heuristic rank score — a points total, not a probability">
                            {c.score.toFixed(2)}
                          </span>
                          <ChevronRight size={14} style={{ color: 'var(--ui2-text-dim)', flex: '0 0 auto' }} />
                        </button>
                      );
                    })}
                    {!candidates.length && (
                      <div style={S.hint}>
                        The scan found no nn.Module subclasses in {scanned}. Nothing was executed.
                      </div>
                    )}
                  </div>
                </div>
              )}
            </>
          )}

          {busy && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12.5, color: 'var(--ui2-text-dim)' }}>
              <Loader2 size={14} className="spinning" /> {busy}
            </div>
          )}

          {failure && (
            <div
              role="alert"
              className="ui2-panel ui2-notice-error"
              style={{ display: 'flex', alignItems: 'flex-start', gap: 10, padding: '11px 13px' }}
            >
              <AlertTriangle size={15} style={{ color: 'var(--ui2-red)', flex: '0 0 auto', marginTop: 2 }} />
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ fontSize: 13, color: 'var(--ui2-text)' }}>{failure.headline}</div>
                <div
                  style={{
                    ...MONO, fontSize: 11.5, lineHeight: '17px', marginTop: 4,
                    color: 'var(--ui2-text-dim)', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                  }}
                >
                  {failure.detail}
                </div>
              </div>
              {failure.retry === 'scan' && (
                <button onClick={doScan} disabled={!!busy} type="button" className="ui2-chip" style={{ flex: '0 0 auto' }}>
                  Scan again
                </button>
              )}
              {failure.retry === 'libraries' && (
                <button onClick={() => setRetryToken((t) => t + 1)} type="button" className="ui2-chip" style={{ flex: '0 0 auto' }}>
                  Retry
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
