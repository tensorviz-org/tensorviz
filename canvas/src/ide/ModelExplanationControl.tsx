import React from 'react';
import { CircleAlert, LoaderCircle, MessageCircle, Sparkles, X } from 'lucide-react';
import { NodeExplanationContext } from '../canvas/NodeExplanationContext';
import { dispatchBuilderCommand } from '../components/ui2/builderCommand';
import CanvasPopover from './CanvasPopover';
import CanvasTooltip from '../components/CanvasTooltip';
import './modelExplanation.css';

/** Each explicit click explains the visible scope; stored annotations can be hidden separately. */
export default function ModelExplanationControl({ compact }: { compact: boolean }) {
  const context = React.useContext(NodeExplanationContext);
  const model = context?.model;
  if (!model) return null;
  const state = model.state;
  const loading = state?.status === 'loading';
  const completed = state?.completed ?? 0;
  const settled = !!state && !loading;
  const failed = state?.status === 'unavailable' || state?.status === 'cancelled';
  const canToggle = settled && completed > 0;
  const label = loading ? `Explaining view · ${completed}/${state.total}`
    : failed ? 'Retry explanations' : 'Explain this view';
  const help = model.disabledReason ?? (loading ? state.detail ?? `${completed} of ${state.total} components explained. You can keep exploring or stop.`
    : failed && !canToggle && state?.detail ? state.detail
      : 'Explain the nodes visible now, including inputs, outputs and their connections. Uses your configured agent; collapsed children stay untouched.');
  const start = () => dispatchBuilderCommand('explain-model');
  return <div className="model-explanation-control">
    <CanvasTooltip text={help}><button type="button" className="local-canvas-control"
      aria-label={label}
      disabled={!!model.disabledReason || loading} title={help}
      onClick={start}>
      {loading ? <LoaderCircle size={15} className="model-explanation-spinner" aria-hidden="true" /> : <Sparkles size={15} aria-hidden="true" />}
      <span className="local-canvas-control-label">{loading ? `${completed}/${state.total}` : compact && !canToggle && !failed ? 'Explain' : label}</span>
    </button></CanvasTooltip>
    {loading && <CanvasTooltip text="Stop generating. Completed explanations stay available."><button type="button" className="local-canvas-control"
      aria-label="Stop explaining view" title="Stop explaining view" onClick={model.cancel}><X size={14} aria-hidden="true" /></button></CanvasTooltip>}
    {settled && <CanvasPopover label="View explanation details" title="View explanations"
      trigger={failed ? <CircleAlert size={14} aria-hidden="true" /> : <span>{completed}/{state.total}</span>}>
      {close => <div className="model-explanation-details">
        {completed > 0 && <p role="status">{state.status === 'cancelled' ? 'Stopped. ' : ''}{completed} of {state.total} components explained.</p>}
        {state.detail && <p role={failed ? 'alert' : undefined}>{state.detail}</p>}
        {canToggle && <button type="button" className="local-canvas-menu-item" aria-pressed={!!context.annotationsVisible}
          onClick={() => { model.toggle(); close(); }}>
          <Sparkles size={14} aria-hidden="true" /><span>{context.annotationsVisible ? 'Hide explanations' : 'Show explanations'}</span>
        </button>}
        <button type="button" className="local-canvas-menu-item" onClick={() => { start(); close(); }} disabled={!!model.disabledReason}>
          <Sparkles size={14} aria-hidden="true" /><span>{failed ? 'Try again' : 'Explain current view'}</span>
        </button>
        {failed && <button type="button" className="local-canvas-menu-item" onClick={() => {
          window.dispatchEvent(new CustomEvent('tensorviz-agent-compose')); close();
        }}><MessageCircle size={14} aria-hidden="true" /><span>Open agent chat</span></button>}
      </div>}
    </CanvasPopover>}
    {loading && <span className="model-explanation-sr-status" role="status">{completed} of {state.total} components explained</span>}
  </div>;
}
