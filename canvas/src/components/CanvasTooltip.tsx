import React from 'react';
import { createPortal } from 'react-dom';
import './CanvasTooltip.css';

/** Screen-space help stays readable when its canvas node is zoomed or clipped. */
export default function CanvasTooltip({ text, children, block = false }: {
  text: string;
  children: React.ReactElement<{ title?: string; disabled?: boolean; 'aria-describedby'?: string }>;
  block?: boolean;
}) {
  const id = React.useId();
  const anchor = React.useRef<HTMLSpanElement>(null);
  const tip = React.useRef<HTMLDivElement>(null);
  const timer = React.useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const focused = React.useRef(false);
  const [open, setOpen] = React.useState(false);
  const [position, setPosition] = React.useState({ left: 8, top: 8 });
  const clear = () => { clearTimeout(timer.current); };
  const close = React.useCallback(() => { clearTimeout(timer.current); setOpen(false); }, []);
  const leave = () => { clear(); if (!focused.current) timer.current = setTimeout(close, 120); };
  React.useEffect(() => () => clearTimeout(timer.current), []);
  React.useLayoutEffect(() => {
    if (!open) return;
    const rect = anchor.current!.getBoundingClientRect();
    const bounds = tip.current!.getBoundingClientRect();
    setPosition({ left: Math.max(8, Math.min(rect.left + (rect.width - bounds.width) / 2, window.innerWidth - bounds.width - 8)),
      top: Math.max(8, rect.bottom + bounds.height + 16 > window.innerHeight ? rect.top - bounds.height - 8 : rect.bottom + 8) });
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(); }
    };
    document.addEventListener('keydown', escape, true);
    window.addEventListener('resize', close);
    window.addEventListener('wheel', close, { passive: true });
    window.addEventListener('scroll', close, true);
    return () => {
      document.removeEventListener('keydown', escape, true);
      window.removeEventListener('resize', close);
      window.removeEventListener('wheel', close);
      window.removeEventListener('scroll', close, true);
    };
  }, [open, text, close]);
  const description = [children.props['aria-describedby'], open ? id : undefined].filter(Boolean).join(' ') || undefined;
  return <>
    <span ref={anchor} className={`canvas-tooltip-anchor${block ? ' canvas-tooltip-block' : ''}`}
      tabIndex={children.props.disabled ? 0 : undefined} aria-describedby={children.props.disabled ? description : undefined}
      onPointerEnter={() => { clear(); timer.current = setTimeout(() => setOpen(true), 350); }} onPointerLeave={leave}
      onFocus={() => { focused.current = true; clear(); setOpen(true); }}
      onBlur={() => { focused.current = false; close(); }} onClickCapture={close}>
      {React.cloneElement(children, { title: undefined, 'aria-describedby': description })}
    </span>
    {open && createPortal(<div ref={tip} id={id} role="tooltip" className="canvas-tooltip" style={position}
      onPointerEnter={clear} onPointerLeave={leave}>{text}</div>, document.fullscreenElement ?? document.body)}
  </>;
}
