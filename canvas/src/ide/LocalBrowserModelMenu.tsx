import React from 'react';
import { ChevronDown, Workflow } from 'lucide-react';
import type { LocalBrowserTargetIdentity } from './protocol.generated';

/** The model's file, environment and saved setup, kept out of the toolbar until asked for. */
export default function LocalBrowserModelMenu({ target, children }: { target: LocalBrowserTargetIdentity | null; children?: React.ReactNode }) {
  const [open, setOpen] = React.useState(false);
  const root = React.useRef<HTMLDivElement>(null);
  const trigger = React.useRef<HTMLButtonElement>(null);
  const id = React.useId();

  React.useEffect(() => {
    if (!open) return;
    const dismiss = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      setOpen(false);
      trigger.current?.focus();
    };
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('pointerdown', dismiss);
      document.removeEventListener('keydown', escape);
    };
  }, [open]);

  return <div ref={root} className="local-browser-model-menu">
    <button ref={trigger} type="button" className="local-browser-model-trigger"
      aria-label="Model and file actions" aria-expanded={open} aria-controls={id}
      title={target ? `${target.cls} · ${target.file}` : 'tensorViz'}
      onClick={() => setOpen(value => !value)}>
      <span className="local-browser-brand-mark"><Workflow size={16} aria-hidden="true" /></span>
      <span className="local-browser-model-name"><strong>{target && <span className="local-browser-product">tensorViz /</span>}{target?.cls ?? 'tensorViz'}</strong>
        {target && <small>{target.file}</small>}</span>
      <ChevronDown size={14} aria-hidden="true" />
    </button>
    {open && <div id={id} className="local-browser-model-actions" role="group" aria-label="Model actions">
      {target && <p>{target.file}</p>}
      {children}
    </div>}
  </div>;
}
