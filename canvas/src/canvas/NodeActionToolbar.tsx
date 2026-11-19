import React from 'react';
import { Ellipsis, Expand, GitBranch, ListTree, MessageCircle, PanelRightOpen, Pencil, X, type LucideIcon } from 'lucide-react';
import CanvasTooltip from '../components/CanvasTooltip';
import { ExplainNodeButton } from './ExplainNodeButton';
import type { NodeExplanationAction } from './useNodeExplanationAction';
import './nodeActionToolbar.css';

type Point = { x: number; y: number };
/** All inputs are viewport-local pixels, never document/client coordinates. */
export function nodeToolbarPosition(start: Point, end: Point, viewport: { w: number; h: number }, width: number) {
  const top = start.y >= 56 ? start.y - 48 : end.y + 48 < viewport.h ? end.y + 8 : 8;
  return { left: Math.max(8, Math.min((start.x + end.x - width) / 2, viewport.w - width - 8)),
    top: Math.max(8, Math.min(top, viewport.h - 48)) };
}

export function NodeActionButton({ label, help, icon: Icon, ...props }: Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, 'children'> & {
  label: string; help?: string; icon: LucideIcon;
}) {
  return <CanvasTooltip text={help || label}><button {...props} type="button" className="node-action-button" aria-label={label}>
    <Icon size={16} strokeWidth={1.7} aria-hidden="true" />
  </button></CanvasTooltip>;
}

/** Actions stay on the graph. Tooltips work with pointer hover and keyboard focus. */
export function NodeActionToolbar({ position, popoverAbove, label, count, onInspect, onExplain, explainAction, explainDisabled, onDiscuss, onCapture,
  onEnter, onEdit, onEditParameters, onTrace, tracing, onClear, children, onClaim }: {
  position: { left: number; top: number }; count?: number;
  popoverAbove?: boolean;
  label?: string;
  onInspect?(): void; onExplain?(): void; explainDisabled?: string; onDiscuss?(): void; onCapture?(): void;
  explainAction?: NodeExplanationAction;
  onEnter?(): void; onEdit?(): void; onEditParameters?(): void; onTrace(): void; tracing: boolean; onClear(): void;
  children?: React.ReactNode; onClaim?(event: React.PointerEvent): void;
}) {
  const [more, setMore] = React.useState(false);
  const root = React.useRef<HTMLDivElement>(null);
  const menuId = React.useId();
  React.useEffect(() => {
    if (!more) return;
    const outside = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setMore(false); };
    document.addEventListener('pointerdown', outside, true);
    return () => document.removeEventListener('pointerdown', outside, true);
  }, [more]);
  const action = (callback: () => void) => { setMore(false); callback(); };
  return <div ref={root} className="node-action-anchor" data-popover-above={popoverAbove || undefined} style={position} onPointerDown={onClaim} onPointerUp={onClaim}
    onKeyDown={event => {
      if (event.key === 'Escape' && more) { event.preventDefault(); event.stopPropagation(); setMore(false);
        root.current?.querySelector<HTMLButtonElement>('[aria-controls]')?.focus(); }
      if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key) && (event.target as HTMLElement).classList.contains('node-action-button')) {
        const buttons = Array.from(root.current!.querySelectorAll<HTMLButtonElement>('.node-action-button:not(:disabled)'));
        const index = buttons.indexOf(event.target as HTMLButtonElement);
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1
          : (index + (event.key === 'ArrowLeft' ? -1 : 1) + buttons.length) % buttons.length;
        event.preventDefault(); event.stopPropagation(); buttons[next]?.focus();
      }
    }}>
    <div className="node-action-toolbar" role="toolbar" aria-label={label ? `Node actions: ${label}` : 'Node actions'}>
      {count && count > 1 ? <span className="node-action-count">{count} selected</span> : null}
      {onInspect && <NodeActionButton label="Inspect" help="Inspect component · source, inputs and connections" icon={PanelRightOpen} onClick={onInspect} />}
      {onExplain && <ExplainNodeButton action={explainAction ?? { phase: 'idle', run: onExplain, disabledReason: explainDisabled }} />}
      {onDiscuss && <NodeActionButton label="Discuss in chat" icon={MessageCircle} onClick={onDiscuss} />}
      {onCapture && <NodeActionButton label="Captured calls & tensors" help="Browse recorded arguments and tensors · does not run the model" icon={ListTree} onClick={onCapture} />}
      {onEnter && <NodeActionButton label="Open inside" help="Open this component’s internal graph" icon={Expand} onClick={onEnter} />}
      <NodeActionButton label="Trace" help="Trace incoming and outgoing connections" icon={GitBranch} aria-expanded={tracing} onClick={() => { setMore(false); onTrace(); }} />
      <span className="node-action-divider" aria-hidden="true" />
      <NodeActionButton label="More node actions" icon={Ellipsis} aria-expanded={more} aria-controls={menuId}
        onClick={() => { if (tracing) onTrace(); setMore(value => !value); }} />
    </div>
    {more && <div id={menuId} className="node-action-menu" role="group" aria-label="More node actions">
      {onEdit && <button type="button" onClick={() => action(onEdit)}><Pencil size={14} aria-hidden="true" />Edit source</button>}
      {onEditParameters && <button type="button" onClick={() => action(onEditParameters)}><Pencil size={14} aria-hidden="true" />Edit parameters</button>}
      <button type="button" aria-keyshortcuts="Escape" onClick={() => action(onClear)}><X size={14} aria-hidden="true" />Clear selection<kbd aria-hidden="true">Esc</kbd></button>
    </div>}
    {tracing && <div className="node-action-trace">{children}</div>}
  </div>;
}
