import React from 'react';
import { createPortal } from 'react-dom';
import { CheckCircle2, ChevronDown, CircleAlert, Play, Square, X, CircleSlash } from 'lucide-react';

import CanvasTooltip from '../components/CanvasTooltip';

import { LocalBrowserContext } from './localBrowserContext';
import './LocalBrowserValidate.css';

export default function LocalBrowserValidate({ controls = 'all', compact = false }: { controls?: 'all' | 'run' | 'result'; compact?: boolean }): React.ReactElement | null {
  const context = React.useContext(LocalBrowserContext);
  const validation = context?.validation;
  const [resultOpen, setResultOpen] = React.useState(false);
  const resultButton = React.useRef<HTMLButtonElement>(null);
  const resultPanel = React.useRef<HTMLDivElement>(null);
  const [anchor, setAnchor] = React.useState<React.CSSProperties | null>(null);
  const resultId = React.useId();
  const result = validation?.latestResult ?? validation?.lastResult;
  const evidence = validation?.latestResult?.evidence;
  const displayed = validation?.latestResult?.showingMeasurements ? evidence?.displayedCapture : null;
  React.useEffect(() => setResultOpen(false), [result?.status, result?.detail]);
  React.useLayoutEffect(() => {
    if (!resultOpen) return;
    // The toolbar scrolls horizontally and clips vertical overflow. Its popovers
    // must live on body, positioned from the trigger like the existing More menu.
    const position = () => {
      const rect = resultButton.current?.getBoundingClientRect();
      if (!rect) return;
      const width = Math.min(360, window.innerWidth - 32);
      const above = rect.top >= window.innerHeight - rect.bottom;
      setAnchor({ left: Math.max(16, Math.min(rect.right - width, window.innerWidth - width - 16)), width,
        ...(above ? { bottom: window.innerHeight - rect.top + 8 } : { top: rect.bottom + 8 }),
        maxHeight: Math.max(0, Math.min(380, (above ? rect.top : window.innerHeight - rect.bottom) - 24)) });
    };
    position();
    window.addEventListener('resize', position);
    window.addEventListener('scroll', position, true);
    return () => { window.removeEventListener('resize', position); window.removeEventListener('scroll', position, true); };
  }, [resultOpen]);
  const positioned = anchor !== null;
  React.useEffect(() => { if (resultOpen && positioned) resultPanel.current?.focus(); }, [resultOpen, positioned]);
  const closeResult = () => { setResultOpen(false); resultButton.current?.focus(); };
  if (!validation) {
    return null;
  }
  const { running, disabledReason } = validation;
  const reviewing = context?.page.opening?.execution?.phase === 'review';
  const stopping = context?.page.opening?.execution?.phase === 'cancelling';
  const active = running && !reviewing;
  const environmentPending = running && !!context?.page.opening?.environment && context.page.opening.environment.reason !== 'ready';
  const runLabel = stopping ? 'Cancelling…' : environmentPending ? 'Preparing environment…' : active ? 'Running…' : 'Run model';
  const ResultIcon = result?.status === 'passed' ? CheckCircle2 : result?.status === 'cancelled' ? CircleSlash : CircleAlert;
  return (
    <>
      {controls !== 'result' && <CanvasTooltip text={reviewing ? 'Review device and inputs before Run.' : environmentPending ? 'Environment setup is running. Use Cancel to stop it.' : running ? 'Model execution is in progress. Use Cancel to stop it.' : disabledReason
        ?? (validation.reopens ? 'Run the model again with its saved setup.' : 'Choose the device and review inputs before running.')}><button
        type="button"
        className={controls === 'run' ? `local-canvas-control local-canvas-run${active ? ' local-canvas-capturing' : ''}` : 'ui2-primary-action ui2-primary-action-green'}
        data-testid="local-browser-validate"
        disabled={!reviewing && (running || disabledReason !== null)}
        aria-busy={active}
        onClick={() => validation.run()}
        title={disabledReason ?? (validation.reopens ? 'Run with the saved setup' : 'Review model execution settings')}
        aria-label={runLabel}
      >
        <Play size={14} aria-hidden="true" /><span className="local-canvas-control-label">{runLabel}</span>
      </button></CanvasTooltip>}
      {controls !== 'result' && running && !reviewing && !(environmentPending && context?.navigation?.inputsOpen) && (
        <CanvasTooltip text="Stop the current run. Any previous measured graph remains visible."><button
          type="button"
          className={controls === 'run' ? 'local-canvas-control' : 'ui2-primary-action'}
          data-testid="local-browser-validate-cancel"
          disabled={stopping}
          onClick={() => validation.cancel()}
          title="Stop the run and leave the graph as it is."
          aria-label="Cancel"
        >
          <Square size={13} aria-hidden="true" />{!compact && <span className="local-canvas-control-label">Cancel</span>}
        </button></CanvasTooltip>
      )}
      {controls !== 'run' && result && (
        <div style={{ position: 'relative' }} onKeyDown={event => {
          if (event.key === 'Escape') { closeResult(); event.stopPropagation(); }
        }}>
          <CanvasTooltip text="Inspect the latest run result, the setup used, and whether the displayed measurements come from this attempt."><button ref={resultButton} type="button" className={controls === 'result' ? 'local-canvas-control local-canvas-result-trigger' : 'ui2-chip'}
            data-result={result.status} aria-haspopup="dialog" aria-expanded={resultOpen} aria-controls={resultId}
            aria-label={`Latest validation: ${validation.statusLabel}`} onClick={() => setResultOpen(open => !open)}>
            <ResultIcon size={14} aria-hidden="true" /><span>{validation.statusLabel}</span><ChevronDown size={12} aria-hidden="true" />
          </button></CanvasTooltip>
          {resultOpen && anchor && createPortal(<>
            <div style={{ position: 'fixed', inset: 0, zIndex: controls === 'result' ? 1201 : 54 }} aria-hidden="true" onClick={closeResult} />
            <div ref={resultPanel} id={resultId} className={`ui2-panel canvas-run-result${controls === 'result' ? ' local-canvas-popover' : ''}`} role="dialog" aria-label="Latest validation result" tabIndex={-1}
              style={{ ...anchor, position: 'fixed', boxSizing: 'border-box', color: 'var(--ui2-text)', boxShadow: 'var(--ui2-shadow-float)',
                padding: 0, zIndex: controls === 'result' ? 1202 : 55, overflowY: 'auto', whiteSpace: 'normal' }}>
              <div className="canvas-run-result-heading"><div><span>Latest validation</span><strong><ResultIcon size={17} aria-hidden="true" />{validation.statusLabel}</strong></div>
                <button type="button" className="local-canvas-control" aria-label="Close result" onClick={closeResult}><X size={15} aria-hidden="true" /></button></div>
              <p className="canvas-run-result-detail">{result.detail}</p>
              {evidence && <div className="canvas-run-result-evidence">
                <div>Started by {evidence.origin === 'mcp' ? 'coding agent' : evidence.origin === 'automatic' ? 'trace on save' : 'manual graph action'}
                  {evidence.attemptStatus === 'timed_out' ? ' · Timed out' : ''}</div>
                {evidence.setup && <div>Attempt setup: {evidence.setup.profileName ? `profile ${evidence.setup.profileName}` : 'explicit inputs'}
                  {' · '}{evidence.setup.device}{' · '}{evidence.setup.mode}</div>}
                <div>{displayed
                  ? `Displayed measurements: ${displayed.attemptId === validation?.latestResult?.runId ? 'this attempt' : 'an earlier attempt'} · ${displayed.forwardStatus === 'passed' ? 'forward passed' : 'forward failed'}`
                  : 'No measured capture is displayed.'}</div>
                {displayed && (displayed.format === 'legacy' || displayed.status !== 'complete')
                  && <div>Detailed operation capture is incomplete.</div>}
                {displayed?.unavailable && <div>The full capture exceeded the retention limit.</div>}
              </div>}
            </div>
          </>, document.body)}
        </div>
      )}
    </>
  );
}
