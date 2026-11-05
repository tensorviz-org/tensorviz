import React from 'react';
import { CircleAlert, LoaderCircle, Sparkles, X } from 'lucide-react';
import { NodeExplanationContext } from '../canvas/NodeExplanationContext';
import { dispatchBuilderCommand } from '../components/ui2/builderCommand';
import CanvasPopover from './CanvasPopover';
import CanvasTooltip from '../components/CanvasTooltip';
import './modelExplanation.css';

/** One model-wide action; node inspection continues in the existing details. */
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
  const label = loading ? `Explaining model · ${completed}/${state.total}`
    : canToggle ? 'Explanations' : failed ? 'Retry explanations' : 'Explain model';
  const help = model.disabledReason ?? (loading ? `${completed} of ${state.total} components explained. You can keep exploring or stop.`
    : canToggle ? `${context.annotationsVisible ? 'Hide' : 'Show'} the generated purpose and flow labels. Full explanations stay in node details.`
      : 'Explain component roles and data flow using this graph and its source. Uses the agent connected in chat.');
  const start = () => dispatchBuilderCommand('explain-model');
  return <div className="model-explanation-control">
    <CanvasTooltip text={help}><button type="button" className="local-canvas-control"
      aria-label={label} aria-pressed={canToggle ? !!context.annotationsVisible : undefined}
      disabled={!!model.disabledReason || loading} title={help}
      onClick={canToggle ? model.toggle : start}>
      {loading ? <LoaderCircle size={15} className="model-explanation-spinner" aria-hidden="true" /> : <Sparkles size={15} aria-hidden="true" />}
      <span className="local-canvas-control-label">{loading ? `${completed}/${state.total}` : compact && !canToggle && !failed ? 'Explain' : label}</span>
    </button></CanvasTooltip>
    {loading && <CanvasTooltip text="Stop generating. Completed explanations stay available."><button type="button" className="local-canvas-control"
      aria-label="Stop explaining model" title="Stop explaining model" onClick={model.cancel}><X size={14} aria-hidden="true" /></button></CanvasTooltip>}
    {settled && <CanvasPopover label="Model explanation details" title="Model explanations"
      trigger={failed ? <CircleAlert size={14} aria-hidden="true" /> : <span>{completed}/{state.total}</span>}>
      {close => <div className="model-explanation-details">
        <p role="status">{state.status === 'cancelled' ? 'Stopped. ' : ''}{completed} of {state.total} components explained.</p>
        {state.detail && <p>{state.detail}</p>}
        <small>Source-based interpretation. Recorded tensor evidence stays separate. Up to 128 visible components per pass; unchanged explanations are reused.</small>
        <button type="button" className="local-canvas-menu-item" onClick={() => { start(); close(); }} disabled={!!model.disabledReason}>
          <Sparkles size={14} aria-hidden="true" /><span>{failed ? 'Try again' : 'Explain current view'}</span>
        </button>
        {failed && <button type="button" className="local-canvas-menu-item" onClick={() => {
          window.dispatchEvent(new CustomEvent('tensorviz-agent-compose')); close();
        }}>Open agent chat</button>}
      </div>}
    </CanvasPopover>}
    {loading && <span className="model-explanation-sr-status" role="status">{completed} of {state.total} components explained</span>}
  </div>;
}
