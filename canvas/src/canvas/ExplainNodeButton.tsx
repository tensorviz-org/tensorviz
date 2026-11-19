import React from 'react';
import { CircleAlert, LoaderCircle, Sparkles, Square } from 'lucide-react';
import CanvasTooltip from '../components/CanvasTooltip';
import type { NodeExplanationAction } from './useNodeExplanationAction';
import './explainNodeButton.css';

/** Completion styles the AI sparkle without replacing the action's identity. */
export function ExplainNodeButton({ action }: { action: NodeExplanationAction }) {
  const { phase, detail, disabledReason, run } = action;
  const busy = phase === 'preparing' || phase === 'working';
  const label = busy ? 'Stop explaining' : phase === 'error' ? 'Retry AI explanation'
    : phase === 'complete' || phase === 'stopped' ? 'Explain again' : 'Explain this component';
  const help = disabledReason || (busy ? `${detail || (phase === 'preparing' ? 'Preparing AI explanation…' : 'Generating AI explanation…')} Click to stop.`
    : phase === 'complete' ? 'AI explanation generated. Inspect to read it, or click to explain again.'
      : phase === 'error' ? `${detail || 'AI explanation failed.'} Click to retry.`
        : phase === 'stopped' ? 'AI explanation stopped. Click to explain again.' : 'Explain this component with the selected agent');
  return <CanvasTooltip text={help}><button type="button" className="node-action-button node-explain-action"
    data-state={phase} aria-label={label} aria-busy={busy} disabled={!!disabledReason} onClick={run}>
    {busy ? <LoaderCircle size={16} strokeWidth={1.7} className="node-explain-spinner" aria-hidden="true" />
      : phase === 'error' ? <CircleAlert size={16} strokeWidth={1.7} className="node-explain-alert" aria-hidden="true" />
        : phase === 'stopped' ? <Square size={14} strokeWidth={1.7} aria-hidden="true" />
          : <Sparkles size={16} strokeWidth={1.7} className={phase === 'complete' ? 'node-explain-complete' : undefined} aria-hidden="true" />}
    {busy && <Square size={8} strokeWidth={1.7} className="node-explain-stop" aria-hidden="true" />}
  </button></CanvasTooltip>;
}
