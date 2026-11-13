import React from 'react';
import { Check, ChevronDown, LoaderCircle, RefreshCw, Search, Workflow } from 'lucide-react';
import type { LocalBrowserTargetIdentity } from './protocol.generated';
import type { ModelCatalogEntry as LocalModelOption, ModelCatalog as LocalModelCatalog } from './modelCatalog.generated';
import './LocalBrowserModelMenu.css';

export type { ModelCatalogEntry as LocalModelOption, ModelCatalog as LocalModelCatalog } from './modelCatalog.generated';

/** Model selection and its evidence first; file/setup utilities remain secondary. */
export default function LocalBrowserModelMenu({ target, catalog, onSelect, onRefresh, disabledReason, children }: {
  target: LocalBrowserTargetIdentity | null;
  catalog?: LocalModelCatalog;
  onSelect?(id: string): void;
  onRefresh?(): void;
  disabledReason?: string;
  children?: React.ReactNode;
}) {
  const [open, setOpen] = React.useState(false);
  const [query, setQuery] = React.useState('');
  const root = React.useRef<HTMLDivElement>(null);
  const trigger = React.useRef<HTMLButtonElement>(null);
  const id = React.useId();
  const available = !!catalog || !!onSelect || !!onRefresh;
  const models = catalog?.models ?? [];
  const needle = query.trim().toLocaleLowerCase();
  const matching = models.filter(model => `${model.cls} ${model.file}`.toLocaleLowerCase().includes(needle));
  const current = (model: LocalModelOption) => catalog?.activeId !== undefined ? model.id === catalog.activeId
    : !!target && model.cls === target.cls && model.file === target.file;
  const close = () => { setOpen(false); trigger.current?.focus(); };

  React.useEffect(() => {
    if (!open) { setQuery(''); return; }
    const dismiss = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault(); setOpen(false); trigger.current?.focus();
    };
    const first = root.current?.querySelector<HTMLElement>('input[type="search"], button[data-model-id]:not(:disabled)');
    first?.focus();
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('pointerdown', dismiss);
      document.removeEventListener('keydown', escape);
    };
  }, [open]);

  const navigate = (event: React.KeyboardEvent) => {
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
    const buttons = Array.from(root.current?.querySelectorAll<HTMLButtonElement>('button[data-model-id]:not(:disabled)') ?? []);
    if (!buttons.length) return;
    const index = buttons.indexOf(event.target as HTMLButtonElement);
    if (index < 0 && !(event.target instanceof HTMLInputElement)) return;
    if (event.target instanceof HTMLInputElement && event.key !== 'ArrowDown') return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1
      : index < 0 ? 0 : (index + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length;
    buttons[next].focus();
  };
  const section = (opened: boolean) => {
    const items = matching.filter(model => model.opened === opened);
    if (!items.length) return null;
    const label = opened ? 'Opened models' : 'Project models';
    return <section className="local-model-section" aria-label={label}>
      <h3>{label}</h3>
      <ul className="local-model-options">
        {items.map(model => <li key={model.id}><button type="button" data-model-id={model.id}
          aria-current={current(model) ? 'true' : undefined} disabled={!!disabledReason || !onSelect}
          title={disabledReason ?? `${model.cls} · ${model.file}`}
          onClick={() => { if (!current(model)) onSelect?.(model.id); close(); }}>
          <span className="local-model-option-identity"><strong>{model.cls}</strong><code>{model.file}</code></span>
          <span className="local-model-option-evidence" title={model.evidence === 'captured' ? 'Observed during model execution.' : 'Found in source; execution has not been recorded.'}>{model.evidence === 'captured' ? 'Captured' : 'From code'}</span>
          {current(model) && <Check size={14} aria-label="Current model" />}
        </button></li>)}
      </ul>
    </section>;
  };

  return <div ref={root} className="local-browser-model-menu">
    <button ref={trigger} type="button" className="local-browser-model-trigger"
      aria-label={available ? 'Switch model' : 'Model and file actions'} aria-haspopup="dialog" aria-expanded={open} aria-controls={id}
      title={target ? `${target.cls} · ${target.file}` : 'Choose a model'}
      onKeyDown={event => { if (event.key === 'ArrowDown') { event.preventDefault(); setOpen(true); } }}
      onClick={() => setOpen(value => !value)}>
      <span className="local-browser-brand-mark"><Workflow size={16} aria-hidden="true" /></span>
      <span className="local-browser-model-name"><strong>{target && <span className="local-browser-product">tensorViz /</span>}{target?.cls ?? 'tensorViz'}</strong>
        {target && <small>{target.file}</small>}</span>
      <ChevronDown size={14} aria-hidden="true" />
    </button>
    {open && <div id={id} className="local-browser-model-actions local-model-picker" role="dialog" aria-label={available ? 'Switch model' : 'Model actions'} onKeyDown={navigate}>
      {available && <>
      <div className="local-model-picker-heading"><strong>Models</strong>{onRefresh && <button type="button" aria-label="Refresh models"
        disabled={catalog?.status === 'loading'} onClick={onRefresh}><RefreshCw size={13} aria-hidden="true" /></button>}</div>
      {models.length > 8 && <label className="local-model-search"><Search size={13} aria-hidden="true" />
        <input type="search" aria-label="Find a model" placeholder="Find a model" value={query} onChange={event => setQuery(event.target.value)} />
      </label>}
      {catalog?.status === 'loading' && <p className="local-model-status" role="status"><LoaderCircle size={13} aria-hidden="true" />Finding models…</p>}
      {catalog?.status === 'unavailable' && <p className="local-model-status" role="status">{catalog.detail || 'The model list is unavailable. Refresh to try again.'}</p>}
      <div className="local-model-picker-list">{section(true)}{section(false)}</div>
      {catalog?.status === 'ready' && !models.length && <p className="local-model-status">No models found in this workspace.</p>}
      {models.length > 0 && !matching.length && <p className="local-model-status">No matching models.</p>}
      {!catalog && <p className="local-model-status">Open a model to switch between model graphs.</p>}
      {catalog?.truncated && <p className="local-model-status">Showing part of the workspace. Refresh after narrowing the project.</p>}
      {disabledReason && <p className="local-model-status">{disabledReason}</p>}
      </>}
      {target && <details className="local-model-file-details"><summary>File and setup</summary><p>{target.file}</p>{children}</details>}
    </div>}
  </div>;
}
