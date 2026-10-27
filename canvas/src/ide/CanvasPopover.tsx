import React from 'react';
import { createPortal } from 'react-dom';
import CanvasTooltip from '../components/CanvasTooltip';
import { X } from 'lucide-react';

/** Header/footer popovers stay outside clipping containers, including fullscreen. */
export default function CanvasPopover({ label, title = label, trigger, children, className = '', role = 'dialog', testId, shortcut, help }: {
  label: string;
  help?: string;
  title?: string;
  trigger: React.ReactNode;
  children: (close: () => void) => React.ReactNode;
  className?: string;
  role?: 'dialog' | 'menu';
  testId?: string;
  shortcut?: string;
}) {
  const [open, setOpen] = React.useState(false);
  const [position, setPosition] = React.useState<React.CSSProperties | null>(null);
  const button = React.useRef<HTMLButtonElement>(null);
  const panel = React.useRef<HTMLDivElement>(null);
  const id = React.useId();
  const close = () => { setOpen(false); button.current?.focus(); };

  React.useLayoutEffect(() => {
    if (!open) return;
    const place = () => {
      const rect = button.current?.getBoundingClientRect();
      if (!rect) return;
      const width = Math.min(304, window.innerWidth - 16);
      const above = rect.top > window.innerHeight - rect.bottom;
      setPosition({ width, left: Math.max(8, Math.min(rect.right - width, window.innerWidth - width - 8)),
        ...(above ? { bottom: window.innerHeight - rect.top + 8 } : { top: rect.bottom + 8 }),
        maxHeight: Math.max(80, (above ? rect.top : window.innerHeight - rect.bottom) - 16) });
    };
    const outside = (event: Event) => {
      const target = event.target as Node;
      if (!button.current?.contains(target) && !panel.current?.contains(target)) setOpen(false);
    };
    place();
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(place);
    if (button.current) observer?.observe(button.current);
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    document.addEventListener('pointerdown', outside);
    document.addEventListener('focusin', outside);
    return () => {
      observer?.disconnect();
      window.removeEventListener('resize', place);
      window.removeEventListener('scroll', place, true);
      document.removeEventListener('pointerdown', outside);
      document.removeEventListener('focusin', outside);
    };
  }, [open]);

  const positioned = position !== null;
  React.useEffect(() => { if (open && positioned) panel.current?.focus(); }, [open, positioned]);
  React.useEffect(() => {
    if (!shortcut) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || event.key !== shortcut) return;
      if ((event.target as HTMLElement)?.closest('input, textarea, select, [contenteditable="true"]')) return;
      event.preventDefault();
      event.stopPropagation();
      setOpen(value => !value);
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [shortcut]);

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(); return; }
    if (role !== 'menu' || !['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
    if ((event.target as HTMLElement).matches('input, select, textarea')) return;
    const items = Array.from(panel.current?.querySelectorAll<HTMLButtonElement>('[role^="menuitem"]:not(:disabled)') ?? []);
    if (!items.length) return;
    const current = items.indexOf(document.activeElement as HTMLButtonElement);
    const index = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1
      : current < 0 ? (event.key === 'ArrowUp' ? items.length - 1 : 0)
        : (current + (event.key === 'ArrowUp' ? -1 : 1) + items.length) % items.length;
    event.preventDefault();
    items[index].focus();
  };

  return <>
    <CanvasTooltip text={help ?? (shortcut ? `${label} (${shortcut})` : label)}><button ref={button} type="button" className={`local-canvas-control ${className}`} aria-label={label}
      title={shortcut ? `${label} (${shortcut})` : label} aria-haspopup={role} aria-expanded={open}
      aria-controls={open ? id : undefined} data-testid={testId} onClick={() => setOpen(value => !value)}
      onKeyDown={event => { if (event.key === 'Escape' && open) { event.stopPropagation(); close(); } }}>
      {trigger}
    </button></CanvasTooltip>
    {open && position && createPortal(<div ref={panel} id={id} role={role} aria-label={title} tabIndex={-1}
      className="local-canvas-popover local-canvas-action-panel" style={position} onKeyDown={onKeyDown}>
      <div className="local-canvas-popover-heading"><strong>{title}</strong>
        <button type="button" className="local-canvas-control" aria-label={`Close ${title.toLowerCase()}`} onClick={close}><X size={15} /></button>
      </div>
      {children(close)}
    </div>, document.body)}
  </>;
}
