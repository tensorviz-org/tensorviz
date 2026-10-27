import React from 'react';
import CanvasTooltip from './CanvasTooltip';
import './CanvasSegmentedControl.css';

/** A single keyboard stop for related canvas views; selection never runs a model. */
export default function CanvasSegmentedControl<T extends string>({ label, value, options, onChange, compact = false, className = '' }: {
  label: string;
  value: T;
  options: readonly { value: T; label: string; icon?: React.ReactNode; help?: string }[];
  onChange(value: T): void;
  compact?: boolean;
  className?: string;
}) {
  const root = React.useRef<HTMLDivElement>(null);
  const active = Math.max(0, options.findIndex(option => option.value === value));
  return <div ref={root} className={`canvas-segments ${className}`} role="group" aria-label={label}
    style={{ '--segment-count': options.length, '--segment-active': active } as React.CSSProperties}
    onKeyDown={event => {
      if (event.altKey || event.ctrlKey || event.metaKey || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault(); event.stopPropagation();
      const index = event.key === 'Home' ? 0 : event.key === 'End' ? options.length - 1
        : (active + (event.key === 'ArrowLeft' ? -1 : 1) + options.length) % options.length;
      onChange(options[index].value);
      root.current?.querySelectorAll<HTMLButtonElement>('button')[index]?.focus();
    }}>
    <span className="canvas-segment-selection" aria-hidden="true" />
    {options.map(option => <CanvasTooltip key={option.value} text={option.help ?? option.label}>
      <button type="button" aria-label={option.label} aria-pressed={value === option.value}
        tabIndex={value === option.value ? 0 : -1} onClick={() => onChange(option.value)}>
        {option.icon}{!compact && <span>{option.label}</span>}
      </button>
    </CanvasTooltip>)}
  </div>;
}
