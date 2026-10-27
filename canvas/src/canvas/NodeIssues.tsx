import React from 'react';
import { createPortal } from 'react-dom';
import type { GraphIssue } from '../types';
import { currentProjection, reportNodeClick } from '../ide/projection';
import { graphIssueLabel } from './graphIssues';
import { dispatchBuilderCommand } from '../components/ui2/builderCommand';
import './captureDetails.css';

export function NodeIssues({ issues, label, className, children, style }: {
  issues: GraphIssue[]; label: string; className?: string; children?: React.ReactNode; style?: React.CSSProperties;
}) {
  const [open, setOpen] = React.useState(false);
  const trigger = React.useRef<HTMLButtonElement>(null);
  const dialog = React.useRef<HTMLDialogElement>(null);
  React.useEffect(() => { if (open) dialog.current?.showModal(); }, [open]);
  const close = () => { setOpen(false); trigger.current?.focus({ preventScroll: true }); };
  if (!issues.length) return null;
  const summary = graphIssueLabel(issues);
  return <>
    <button ref={trigger} type="button" className={className} style={{ ...style, pointerEvents: 'auto', cursor: 'pointer' }}
      aria-label={`${summary} in ${label}`} aria-haspopup="dialog" title="Inspect recorded issues"
      onPointerDown={event => event.stopPropagation()}
      onClick={event => { event.stopPropagation(); setOpen(true); }}>{children ?? summary}</button>
    {open && createPortal(<dialog ref={dialog} className="capture-details node-issues" aria-label={`Issues in ${label}`}
      onPointerDown={event => event.stopPropagation()} onPointerUp={event => event.stopPropagation()}
      onClick={event => event.stopPropagation()} onWheel={event => event.stopPropagation()}
      onCancel={event => { event.preventDefault(); close(); }} onKeyDown={event => event.stopPropagation()}>
      <div className="capture-details-content">
        <header><div><h2>{summary}</h2><p>{label} · repeated reports of the same captured failure count once.</p></div>
          <button type="button" aria-label="Close issues" onClick={close}>×</button></header>
        <div className="capture-details-body">{issues.map(issue => <article key={issue.id}>
          <strong>{issue.label}</strong><p style={{ whiteSpace: 'pre-wrap' }}>{issue.message}</p>
          <button type="button" onClick={() => { close(); dispatchBuilderCommand('focus-node', { nodeId: issue.nodeId }); }}>Locate in graph</button>
          {Boolean(currentProjection()?.spanmap?.[issue.nodeId]) && <button type="button" onClick={() => reportNodeClick(issue.nodeId)}>Show source</button>}
        </article>)}</div>
      </div>
    </dialog>, document.body)}
  </>;
}
