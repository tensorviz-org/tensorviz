import React from 'react';
import { ChevronLeft, ChevronRight, FileCode2, RefreshCw, X } from 'lucide-react';
import type { GraphReviewState } from './protocol.generated';
import { reviewParseFailure, type GraphReviewPresentation, type ReviewMode } from './graphReview';
import CanvasSegmentedControl from '../components/CanvasSegmentedControl';
import './graphReview.css';

type CoverageKind = 'model' | 'repository';

/** Coverage buttons sit in the change row; the opened notes need their own width below it. */
function useReviewCoverage({ notes = [], repositoryNotes = [] }: Pick<GraphReviewState, 'notes' | 'repositoryNotes'>, resetKey: string) {
  const [open, setOpen] = React.useState<CoverageKind>();
  const buttons = React.useRef<Partial<Record<CoverageKind, HTMLButtonElement | null>>>({});
  const id = React.useId();
  React.useEffect(() => setOpen(undefined), [resetKey]);
  if (!notes.length && !repositoryNotes.length) return { actions: null, content: null };
  const label = open === 'model' ? 'Coverage limits for this model' : 'Other repository notes';
  const actions = <span className="graph-review-coverage-actions">
    {!!notes.length && <button type="button" ref={element => { buttons.current.model = element; }}
      aria-expanded={open === 'model'} aria-controls={id} onClick={() => setOpen(open === 'model' ? undefined : 'model')}>
      Coverage limits ({notes.length})
    </button>}
    {!!repositoryNotes.length && <button type="button" ref={element => { buttons.current.repository = element; }}
      aria-expanded={open === 'repository'} aria-controls={id} onClick={() => setOpen(open === 'repository' ? undefined : 'repository')}>
      Other repository notes ({repositoryNotes.length})
    </button>}
  </span>;
  const content = open && <div id={id} className="graph-review-coverage-content" role="region" aria-label={label} tabIndex={0}
    onKeyDown={event => { if (event.key === 'Escape') { event.stopPropagation(); setOpen(undefined); buttons.current[open]?.focus(); } }}>
    <p>{open === 'model' ? 'Source analysis limits in this model or its dependencies.'
      : 'Outside this model’s known dependencies. These notes concern discovery of other affected models.'}</p>
    <ul>{(open === 'model' ? notes : repositoryNotes).map((note, i) => <li key={i}>{note}</li>)}</ul>
  </div>;
  return { actions, content };
}

/** Revision-level actions live in the page header beside the compared revisions. */
export function GraphReviewActions({ onClose, onRefresh, onSource, sourceAction }: {
  onClose(): void; onRefresh(): void; onSource(): void; sourceAction?: 'copy';
}) {
  return <span className="graph-review-actions">
    <button type="button" className="local-canvas-control" aria-label={sourceAction === 'copy' ? 'Copy reviewed source reference' : 'Open native code diff'} title={sourceAction === 'copy' ? 'Copy the file, line and reviewed revision' : 'Open the corresponding code diff in VS Code'} onClick={onSource}><FileCode2 size={14} /></button>
    <button type="button" className="local-canvas-control" aria-label="Refresh graph diff" title="Read the selected revisions again" onClick={onRefresh}><RefreshCw size={14} /></button>
    <button type="button" className="local-canvas-control" aria-label="Close graph diff" title="Return to the live model" onClick={onClose}><X size={14} /></button>
  </span>;
}

function ChangeNavigation({ state, presentation, index, onChange }: {
  state: GraphReviewState; presentation: GraphReviewPresentation; index: number; onChange(index: number): void;
}) {
  const { changes, counts, edgeCounts } = presentation;
  const parseFailure = reviewParseFailure(state.notes);
  const stepper = parseFailure && !changes.length ? parseFailure : state.models?.length === 0 ? 'No model changes identified' : changes.length ? `${index + 1} / ${changes.length}` : state.modelStatus === 'unresolved' || state.before.truncated || state.after.truncated || state.before.diagnostics?.length || state.after.diagnostics?.length ? 'Comparison incomplete' : 'No projected changes';
  return <>
    <div className="graph-review-stepper">
      <button type="button" className="local-canvas-control" aria-label="Previous graph change" disabled={!changes.length} onClick={() => onChange((index - 1 + changes.length) % changes.length)}><ChevronLeft size={14} /></button>
      <span role={parseFailure && !changes.length ? 'status' : undefined}>{stepper}</span>
      <button type="button" className="local-canvas-control" aria-label="Next graph change" disabled={!changes.length} onClick={() => onChange((index + 1) % changes.length)}><ChevronRight size={14} /></button>
    </div>
    <span className="graph-review-counts" title={`${edgeCounts.added} added connections · ${edgeCounts.removed} removed connections`}>
      <span className="graph-review-added" aria-label={`${counts.added} added`}>+{counts.added}</span><span className="graph-review-removed" aria-label={`${counts.removed} removed`}>−{counts.removed}</span><span className="graph-review-modified" aria-label={`${counts.modified} modified`}>~{counts.modified}</span>
      {counts.rewired > 0 && <span className="graph-review-rewired" title="Nodes with changed input connections and unchanged source and parameters">{counts.rewired} inputs changed</span>}
      {counts.unresolved > 0 && <span title="These nodes cannot be paired reliably across revisions">{counts.unresolved} unresolved</span>}
    </span>
  </>;
}

/** A pending agent proposal: the source stays unchanged until Accept. No Close or Refresh; a revision replaces the draft. */
export function DraftGraphToolbar({ state, presentation, index, onChange, onSource, onAccept, onDiscard, onContinueEditing }: {
  state: GraphReviewState; presentation: GraphReviewPresentation | null; index: number; onChange(index: number): void; onSource(): void;
  /** False when the request could not be sent, so the decision stays available. */
  onAccept(): boolean; onDiscard(): boolean; onContinueEditing?(): void;
}) {
  const current = presentation?.changes[index];
  const coverage = useReviewCoverage(state, `${state.id}:${state.after.cls}`);
  // One decision per published state: the host answers with a new state, or null when the draft closes.
  const [deciding, setDeciding] = React.useState<'accept' | 'discard' | null>(null);
  React.useEffect(() => setDeciding(null), [state]);
  const staleId = React.useId();
  const decide = (decision: 'accept' | 'discard') => {
    if (!deciding && (decision === 'accept' ? onAccept() : onDiscard())) setDeciding(decision);
  };
  return <section className="graph-review-toolbar graph-review-draft" aria-label="Draft graph preview">
    <div className="graph-review-draft-row">
      <strong title="A proposed source edit. Nothing has run; any retained shapes are from the previous run.">Draft · Not run</strong>
      {presentation && <ChangeNavigation state={state} presentation={presentation} index={index} onChange={onChange} />}
      {current && <span className="graph-review-draft-current" data-status={presentation!.nodes.get(current.id)?.status} aria-live="polite" title={`${current.label} · ${current.detail}`}>
        <strong>{current.label}</strong> <span className="graph-review-draft-detail">{current.detail}</span></span>}
      <button type="button" className="local-canvas-control" aria-label="Open proposed code diff" title="Open the proposed change as a code diff in VS Code" onClick={onSource}><FileCode2 size={14} /></button>
    </div>
    <div className="graph-review-draft-actions" role="group" aria-label="Draft decision" aria-busy={!!deciding}>
      <button type="button" className="graph-review-accept" disabled={state.stale || !!deciding} aria-describedby={state.stale ? staleId : undefined}
        title="Save the accepted source and run the model with its saved inputs and device" onClick={() => decide('accept')}>{deciding === 'accept' ? 'Accepting…' : 'Accept'}</button>
      <button type="button" disabled={!!deciding} title="Remove the draft; the source stays unchanged" onClick={() => decide('discard')}>
        {deciding === 'discard' ? 'Discarding…' : 'Discard'}</button>
      {onContinueEditing && <button type="button" title="Describe a revision in the agent chat" onClick={onContinueEditing}>Continue editing</button>}
    </div>
    {presentation?.inheritedBase && <p className="graph-review-draft-note">The inherited base forward was not inspected; its internals and output shapes are unknown.</p>}
    {state.stale && <p id={staleId} className="graph-review-stale" role="status">Source changed since this draft, so it can’t be accepted. Discard it and request a new change.</p>}
    {coverage.actions && <div className="graph-review-coverage">{coverage.actions}{coverage.content}</div>}
  </section>;
}

export function GraphReviewToolbar({ state, presentation, mode, onMode, index, onChange, onRefresh, onModel, capturedContext }: {
  state: GraphReviewState; presentation: GraphReviewPresentation; mode: ReviewMode; index: number;
  onMode(mode: ReviewMode): void; onChange(index: number): void; onRefresh(): void;
  onModel?(id: string): void;
  capturedContext?: boolean;
}) {
  const { changes, counts, edgeCounts } = presentation;
  const parseFailure = reviewParseFailure(state.notes);
  const coverage = useReviewCoverage(state, `${state.id}:${state.selectedModel ?? state.after.cls}`);
  return <section className="graph-review-toolbar" aria-label="Graph diff review">
    {!!state.models?.length && <label className="graph-review-models">Affected model
      <select aria-label="Affected model" value={state.selectedModel} onChange={event => onModel?.(event.target.value)}>
        {state.models.map(model => <option key={model.id} value={model.id}>{model.label} · {model.file}{model.status !== 'modified' ? ` · ${model.status}` : ''}</option>)}
      </select><span>{state.models.length} model{state.models.length === 1 ? '' : 's'}</span>
    </label>}
    <div className="graph-review-controls">
      {!capturedContext && <CanvasSegmentedControl label="Graph revision" value={mode} onChange={onMode}
        options={[{ value: 'before', label: 'Before' }, { value: 'diff', label: 'Diff' }, { value: 'after', label: 'After' }]} />}
      <div className="graph-review-stepper">
        <button type="button" className="local-canvas-control" aria-label="Previous graph change" disabled={!changes.length} onClick={() => onChange((index - 1 + changes.length) % changes.length)}><ChevronLeft size={14} /></button>
        <span role={parseFailure && !changes.length ? 'status' : undefined}>{parseFailure && !changes.length ? parseFailure : state.models?.length === 0 ? 'No model changes identified' : changes.length ? `${index + 1} / ${changes.length}` : state.modelStatus === 'unresolved' || state.before.truncated || state.after.truncated || state.before.diagnostics?.length || state.after.diagnostics?.length ? 'Comparison incomplete' : 'No projected changes'}</span>
        <button type="button" className="local-canvas-control" aria-label="Next graph change" disabled={!changes.length} onClick={() => onChange((index + 1) % changes.length)}><ChevronRight size={14} /></button>
      </div>
      {parseFailure && !!changes.length && <span className="graph-review-stale" role="status">{parseFailure}</span>}
      {changes[index] && <div className="graph-review-current" data-status={presentation.nodes.get(changes[index].id)?.status} aria-live="polite"
        title={`${changes[index].label} · ${changes[index].detail}`}><strong>{changes[index].label}</strong><span>{changes[index].detail}</span></div>}
      {state.stale && <span className="graph-review-stale" role="status">Source changed since this review. <button type="button" onClick={onRefresh}>Refresh comparison</button></span>}
      {state.updated && <span className="graph-review-stale" role="status">PR updated. <button type="button" onClick={onRefresh}>Refresh to latest changes</button></span>}
      <span className="graph-review-counts" title={`${edgeCounts.added} added connections · ${edgeCounts.removed} removed connections`}>
        <span className="graph-review-added" aria-label={`${counts.added} added`}>+{counts.added}</span><span className="graph-review-removed" aria-label={`${counts.removed} removed`}>−{counts.removed}</span><span className="graph-review-modified" aria-label={`${counts.modified} modified`}>~{counts.modified}</span>
        {counts.rewired > 0 && <span className="graph-review-rewired" title="Nodes with changed input connections and unchanged source and parameters">{counts.rewired} inputs changed</span>}
        {counts.unresolved > 0 && <span title="These nodes cannot be paired reliably across revisions">{counts.unresolved} unresolved</span>}
      </span>
      {coverage.actions}
    </div>
    {coverage.content}
  </section>;
}
