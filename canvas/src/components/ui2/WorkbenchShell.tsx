import React, { useEffect, useState } from 'react';
import { headerTitle } from '../../ide/header';
import { Cpu, Home, Folder, Box, FlaskConical, Github, FileText, PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen, ChevronDown } from 'lucide-react';
import { useWorkbenchServices } from '../../workbench/services';
import * as hostServices from '../../ide/hostServices';

const REPO_URL = 'https://github.com/tensorviz-org/tensorviz';

type WorkbenchShellProps = {
  children: React.ReactNode;
  rightSidebar?: React.ReactNode;
  title?: string;
  rightSidebarTitle?: string;
  centerScrollable?: boolean;
  /**
   * Page-supplied block in the rail between the nav and the environment
   * readout (Home's PINNED list, the model list's TOTALS). Expanded rail only — the
   * 76px rail has no room for a label, and an icon-only total is a riddle.
   */
  railSlot?: React.ReactNode;
  /** Count beside the Models nav item. Omit when it is not known: a 0 the
   *  page never counted would read as "you have no projects". */
  projectCount?: number;
  /**
   * 'full' (default) renders the persistent left nav rail + full top bar.
   * 'minimal' (builder route) drops the rail — nav moves behind the logo-mark
   * menu — and uses a 40px top bar, so the canvas can be full-bleed.
   * 'ide' is 'minimal' minus every way out of the canvas: the IDE bundle mounts
   * no route but /builder (see App.tsx), so a nav menu and a sign-out there would
   * be controls whose destinations do not exist in that build.
   * 'local' (the external read-only page) drops the top bar as well: that page's
   * own strip already names the target and sync state, so a second header would
   * be a duplicate claim on the same truth — and wasted height at compact sizes.
   */
  chrome?: 'full' | 'minimal' | 'ide' | 'local';
  /** Truthful save-state for the minimal (builder) chrome's status pill. */
  saveState?: 'idle' | 'saving' | 'saved' | 'offline' | 'conflict';
  /** In the IDE the header reports the run, not a save: there is no project store
   *  behind this canvas and nothing is queued for one. When present it replaces
   *  the save pill entirely rather than sitting beside it, because two pills
   *  disagreeing about the same corner of the screen is worse than either. */
  ideStatus?: { tone: 'ok' | 'warn' | 'muted'; text: string; detail: string } | null;
  /** The model's actual name. The builder used to render the literal "Model Graph". */
  modelName?: string;
  /** Commit a new name. Omit to keep the title read-only. */
  onRenameModel?: (name: string) => void;
};

/**
 * The model's name, renameable in place.
 *
 * The builder's title was a hardcoded <span>Model Graph</span>: every model showed
 * the same string whatever it was called, and there was no way to rename one —
 * double-clicking it did nothing, because it was not an input.
 */
function ModelTitle({ name, onRename }: { name: string; onRename?: (name: string) => void }) {
  const [editing, setEditing] = React.useState(false);
  const [draft, setDraft] = React.useState(name);
  React.useEffect(() => { if (!editing) setDraft(name); }, [name, editing]);

  const commit = () => {
    const next = draft.trim();
    setEditing(false);
    if (next && next !== name) onRename?.(next);
    else setDraft(name);
  };

  if (!editing || !onRename) {
    return (
      <span
        className="font-semibold truncate text-sm"
        onDoubleClick={() => onRename && setEditing(true)}
        title={onRename ? `${name} — double-click to rename` : name}
        style={onRename ? { cursor: 'text' } : undefined}
      >
        {name}
      </span>
    );
  }
  return (
    <input
      className="ui2-inspector-input font-semibold text-sm"
      style={{ width: Math.max(120, Math.min(360, draft.length * 8 + 24)) }}
      value={draft}
      autoFocus
      aria-label="Model name"
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (event.key === 'Enter') { event.preventDefault(); commit(); }
        if (event.key === 'Escape') { event.preventDefault(); setDraft(name); setEditing(false); }
      }}
    />
  );
}

/** The IDE's own pill. Same shape as SavePill so the header does not move, and a
 *  `title` carrying the long form, because a failed run's message is the useful
 *  part and does not fit. */
function IdeStatusPill({
  status,
}: {
  status: { tone: 'ok' | 'warn' | 'muted'; text: string; detail: string };
}) {
  const colour =
    status.tone === 'warn'
      ? 'var(--ui2-orange)'
      : status.tone === 'ok'
      ? 'var(--ui2-green)'
      : 'var(--ui2-text-dim)';
  return (
    <span
      className="ui2-saved-pill"
      title={status.detail}
      data-testid="ide-status-pill"
      style={{ color: colour, background: 'transparent' }}
    >
      <span style={{ background: colour }} /> {status.text}
    </span>
  );
}

function SavePill({ state }: { state: 'idle' | 'saving' | 'saved' | 'offline' | 'conflict' }) {
  if (state === 'saving') {
    return (
      <span className="ui2-saved-pill" style={{ color: 'var(--ui2-cyan)', background: 'rgba(84,196,255,0.1)' }}>
        <span style={{ background: 'var(--ui2-cyan)' }} /> Saving…
      </span>
    );
  }
  if (state === 'offline') {
    return (
      <span className="ui2-saved-pill" style={{ color: 'var(--ui2-orange)', background: 'rgba(255,182,90,0.12)' }}>
        <span style={{ background: 'var(--ui2-orange)' }} /> Offline — changes queued
      </span>
    );
  }
  if (state === 'conflict') {
    return (
      <span className="ui2-saved-pill" style={{ color: 'var(--ui2-orange)', background: 'rgba(255,182,90,0.12)' }}>
        <span style={{ background: 'var(--ui2-orange)' }} /> Canvas changed — reload to reconcile
      </span>
    );
  }
  return <span className="ui2-saved-pill"><span /> {state === 'saved' ? 'Saved · just now' : 'Saved'}</span>;
}


const navItems = [
  { icon: Home, label: 'Home', path: '/' },
  // 'Models', not 'Projects': the route lists individual models, and /api/projects
  // returns exactly one real project holding all of them. The route path is left
  // alone — renaming a URL is a different change with its own blast radius.
  { icon: Folder, label: 'Models', path: '/projects' },
  { icon: Box, label: 'Builder', path: '/builder' },
  { icon: FlaskConical, label: 'Sandboxes', path: '/sandboxes' },
];

export default function WorkbenchShell({
  children,
  rightSidebar,
  railSlot,
  projectCount,
  title = 'canvas',
  rightSidebarTitle = 'Inspector',
  centerScrollable = true,
  chrome = 'full',
  saveState = 'idle',
  ideStatus = null,
  modelName,
  onRenameModel,
}: WorkbenchShellProps) {
  const services = useWorkbenchServices();
  const pathname = services?.pathname ?? '';
  const navigate = (path: string) => services?.navigate(path);
  const AccountMenu = services?.AccountMenu;
  const Environment = services?.Environment;
  const [leftCompact, setLeftCompact] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);
  const [navMenuOpen, setNavMenuOpen] = useState(false);
  // 'ide' keeps the minimal LAYOUT — no rail, 40px bar, full-bleed canvas — and
  // differs only in what the bar offers, so the two share every size below.
  // 'local' shares that layout and then drops the bar itself.
  const minimal = chrome === 'minimal' || chrome === 'ide' || chrome === 'local';
  const ideChrome = chrome === 'ide';
  const localChrome = chrome === 'local';
  const activeNav = navItems.find(({ path }) =>
    path === '/' ? pathname === '/' : pathname.startsWith(path),
  );

  useEffect(() => {
    // Edge-triggered: only re-derive the rails when a breakpoint is actually
    // crossed, so a manual toggle survives unrelated resizes (window drag,
    // devtools open, mobile URL-bar show/hide).
    let lastCompact: boolean | null = null;
    let lastRightOpen: boolean | null = null;
    const syncLayoutState = () => {
      const width = window.innerWidth;
      // Expanding the rail costs 188px; below 1400 that would leave the content
      // column NARROWER than it was at a smaller window, so expand later.
      const compact = width < 1400;
      const openRight = width >= 1100;
      if (compact !== lastCompact) {
        lastCompact = compact;
        setLeftCompact(compact);
      }
      if (openRight !== lastRightOpen) {
        lastRightOpen = openRight;
        setRightOpen(openRight);
      }
    };
    syncLayoutState();
    window.addEventListener('resize', syncLayoutState);
    return () => window.removeEventListener('resize', syncLayoutState);
  }, []);

  return (
    <div className="ui2-app-bg h-screen overflow-hidden" style={{ color: 'var(--ui2-text)' }}>
      <div className="grid h-full" style={{ gridTemplateColumns: minimal ? 'minmax(0, 1fr)' : (leftCompact ? '76px minmax(0, 1fr)' : '264px minmax(0, 1fr)'), gap: '0', padding: '0' }}>
        {!minimal && (
        <aside className="ui2-sidebar p-5 flex flex-col" aria-label="Workspace navigation">
          <div className="flex items-center gap-3 pb-5">
            <div className="w-9 h-9 rounded-lg flex items-center justify-center ui2-brand-mark">
              <Cpu size={18} />
            </div>
            {!leftCompact && (
              <div className="min-w-0">
                <div className="ui2-section-title">{title}</div>
              </div>
            )}
          </div>

          <nav className="space-y-1 mb-5" aria-label="Primary">
            {navItems.map(({ icon: Icon, label, path }) => {
              const active = path === '/'
                ? pathname === '/'
                : pathname.startsWith(path);
              const badge = path === '/projects' ? projectCount : undefined;
              return (
              <button
                key={label}
                className="ui2-nav-item w-full flex items-center gap-3 px-3 py-2.5 text-left"
                onClick={() => {
                  navigate(path);
                }}
                aria-label={label}
                aria-current={active ? 'page' : undefined}
                title={label}
                style={{
                  color: active ? 'var(--ui2-text)' : 'var(--ui2-text-dim)',
                  background: active ? 'color-mix(in srgb, var(--ui2-accent) 10%, transparent)' : 'transparent',
                  borderColor: active ? 'color-mix(in srgb, var(--ui2-accent) 50%, transparent)' : 'transparent',
                  cursor: 'pointer',
                }}
              >
                <Icon size={16} />
                {!leftCompact && <span className="text-sm">{label}</span>}
                {!leftCompact && badge !== undefined && (
                  <span className="ml-auto" style={{ fontFamily: 'var(--font-mono)', fontSize: '11.5px', color: 'var(--ui2-text-dim)' }}>
                    {badge}
                  </span>
                )}
              </button>
            );
            })}
          </nav>

          {/* Scrolls rather than clips: a long pinned list must not push the
              environment readout off the bottom of the rail. */}
          {!leftCompact && railSlot ? (
            <div className="min-h-0 overflow-auto ui2-scroll-surface mb-5">{railSlot}</div>
          ) : null}

          <div className="mt-auto flex flex-col gap-2.5">
            {!leftCompact && Environment && <Environment />}
            <div
              className="flex items-center gap-2"
              style={!leftCompact ? { paddingTop: 12, borderTop: '1px solid var(--ui2-rule)' } : undefined}
            >
              {/* The top bar keeps its own UserMenu for the compact rail and the
                  builder's minimal chrome, where this one is not rendered.
                  'rail' opens the popover upward — the foot sits against an
                  overflow-hidden root, so a downward one would be clipped. */}
              {!leftCompact && AccountMenu && <AccountMenu placement="rail" />}
              <button
                className="ui2-icon-button ml-auto"
                onClick={() => setLeftCompact((v) => !v)}
                aria-label={leftCompact ? 'Expand navigation rail' : 'Collapse navigation rail'}
              >
                {leftCompact ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}
              </button>
            </div>
          </div>
        </aside>
        )}

        <div className="min-w-0 min-h-0 flex flex-col overflow-hidden">
          {!localChrome && (minimal ? (
          <header className="ui2-topbar h-10 px-4 flex items-center gap-3 flex-shrink-0">
            {ideChrome ? (
              /* A brand mark, not a menu button. Every destination this menu had
                 — Home, Models, Sandboxes — is a route the IDE bundle does not
                 mount, and a dropdown of dead links is worse than no dropdown. */
              <span className="w-6 h-6 rounded-md flex items-center justify-center ui2-brand-mark flex-shrink-0">
                <Cpu size={14} />
              </span>
            ) : (
            <div className="relative flex-shrink-0">
              <button
                className="flex items-center gap-1.5"
                onClick={() => setNavMenuOpen((v) => !v)}
                aria-label="Navigation menu"
                aria-haspopup="menu"
                aria-expanded={navMenuOpen}
                style={{ cursor: 'pointer' }}
              >
                <span className="w-6 h-6 rounded-md flex items-center justify-center ui2-brand-mark"><Cpu size={14} /></span>
                <ChevronDown size={12} className="ui2-muted" />
              </button>
              {navMenuOpen && (
                <>
                  <div className="fixed inset-0" style={{ zIndex: 40 }} onClick={() => setNavMenuOpen(false)} />
                  <div className="ui2-panel" role="menu" style={{ position: 'absolute', left: 0, top: '100%', marginTop: 6, minWidth: 200, padding: 6, zIndex: 50 }}>
                    {navItems.map(({ icon: Icon, label, path }) => {
                      const active = path === '/' ? pathname === '/' : pathname.startsWith(path);
                      return (
                        <button
                          key={label}
                          role="menuitem"
                          className="ui2-nav-item w-full flex items-center gap-2 px-3 py-2 text-left text-sm"
                          onClick={() => { navigate(path); setNavMenuOpen(false); }}
                          style={{ color: active ? 'var(--ui2-text)' : 'var(--ui2-text-dim)', background: active ? 'color-mix(in srgb, var(--ui2-accent) 10%, transparent)' : 'transparent', cursor: 'pointer' }}
                        >
                          <Icon size={14} /> {label}
                        </button>
                      );
                    })}
                  </div>
                </>
              )}
            </div>
            )}
            <ModelTitle name={headerTitle({ graphName: modelName, ide: Boolean(ideStatus) }).title} onRename={onRenameModel} />
            {ideStatus ? <IdeStatusPill status={ideStatus} /> : <SavePill state={saveState} />}
            <div className="ml-auto flex items-center gap-2">
              {rightSidebar ? (
                <button
                  className="ui2-chip"
                  onClick={() => setRightOpen((value) => !value)}
                  aria-label={rightOpen ? 'Hide right panel' : 'Show right panel'}
                >
                  {rightOpen ? <PanelRightClose size={14} /> : <PanelRightOpen size={14} />}
                </button>
              ) : null}
              {/* No account surface in the IDE: the canvas authenticates with the
                  per-launch API token, and this menu's sign-out navigates to
                  /login — a route that build does not mount. */}
              {!ideChrome && AccountMenu && <AccountMenu />}
            </div>
          </header>
          ) : (
          <header className="ui2-topbar px-6 py-4 flex items-center gap-3 flex-shrink-0">
            <span className="ui2-eyebrow truncate">{activeNav?.label ?? title}</span>
            <div className="ml-auto flex items-center gap-2">
              {rightSidebar ? (
                <button
                  className="ui2-chip"
                  onClick={() => setRightOpen((value) => !value)}
                  aria-label={rightOpen ? 'Hide right panel' : 'Show right panel'}
                >
                  {rightOpen ? <PanelRightClose size={14} /> : <PanelRightOpen size={14} />}
                </button>
              ) : null}
              <button
                className="ui2-chip flex items-center gap-1.5"
                onClick={() => hostServices.openExternal(REPO_URL, '_blank', 'noopener,noreferrer')}
                title="Open the repository on GitHub"
              >
                <Github size={14} /> Repo
              </button>
              <button
                className="ui2-chip flex items-center gap-1.5"
                onClick={() => hostServices.openExternal(`${REPO_URL}/tree/main/docs`, '_blank', 'noopener,noreferrer')}
                title="Open the docs on GitHub"
              >
                <FileText size={14} /> Docs
              </button>
              {/* Rendered here only when the rail's own UserMenu is not: an
                  expanded rail already carries it, at the foot the design shows. */}
              {leftCompact && AccountMenu && <AccountMenu />}
            </div>
          </header>
          ))}

          <div className="grid min-h-0 flex-1 overflow-hidden" style={{ gridTemplateColumns: rightSidebar ? (rightOpen ? 'minmax(0, 1fr) 382px' : 'minmax(0, 1fr) 44px') : 'minmax(0, 1fr)', gap: '0' }}>
            {/* Gutter only on full chrome: the builder's minimal chrome needs a
                full-bleed canvas. */}
            <main className={`min-w-0 min-h-0 flex flex-col ${centerScrollable ? 'overflow-auto ui2-scroll-surface' : 'overflow-hidden'} ${minimal ? '' : 'px-6 py-6'}`}>{children}</main>
            {rightSidebar ? (
              rightOpen ? (
                <aside className="min-h-0 min-w-0 overflow-hidden flex flex-col ui2-right-rail" aria-label={rightSidebarTitle}>
                  <div className="flex items-center justify-between px-5 py-4">
                    <span className="ui2-eyebrow">{rightSidebarTitle}</span>
                    <button
                      className="ui2-chip"
                      onClick={() => setRightOpen(false)}
                      aria-label="Minimize right panel"
                      title="Minimize panel"
                    >
                      <PanelRightClose size={14} />
                    </button>
                  </div>
                  <div className="min-h-0 flex-1 overflow-auto px-5 pb-5 ui2-scroll-surface">
                    {rightSidebar}
                  </div>
                </aside>
              ) : (
                <aside className="ui2-panel min-h-0 min-w-0 overflow-hidden flex flex-col items-center gap-3 py-3" aria-label={rightSidebarTitle}>
                  <button
                    className="ui2-chip"
                    onClick={() => setRightOpen(true)}
                    aria-label="Expand right panel"
                    title="Expand panel"
                  >
                    <PanelRightOpen size={14} />
                  </button>
                  <span className="ui2-eyebrow" style={{ writingMode: 'vertical-rl' }}>{rightSidebarTitle}</span>
                </aside>
              )
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
