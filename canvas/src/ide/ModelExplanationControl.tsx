import React from 'react';
import { ChevronDown, CircleAlert, LoaderCircle, MessageCircle, Sparkles, X } from 'lucide-react';
import { NodeExplanationContext } from '../canvas/NodeExplanationContext';
import { dispatchBuilderCommand } from '../components/ui2/builderCommand';
import CanvasPopover from './CanvasPopover';
import CanvasTooltip from '../components/CanvasTooltip';
import { AgentModelMenu, effortLabel } from './AgentModelPicker';
import './modelExplanation.css';

/** Each explicit click explains the visible scope; stored annotations can be hidden separately. */
export default function ModelExplanationControl({ compact }: { compact: boolean }) {
  const context = React.useContext(NodeExplanationContext);
  const model = context?.model;
  if (!model) return null;
  const state = model.state;
  const configuration = model.configuration;
  const settings = configuration?.settings;
  const availability = configuration?.availability;
  const selectedModel = availability?.models.find(entry => entry.id === (settings?.model ?? availability.defaultModel));
  const invalidChoice = availability?.status === 'ready' ? (settings?.model && !selectedModel
    ? `Model “${settings.model}” is unavailable. Choose another model.`
    : settings?.effort && !selectedModel?.efforts.includes(settings.effort)
      ? `Effort “${settings.effort}” is unavailable for this model. Choose another effort.` : undefined) : undefined;
  const loading = state?.status === 'loading';
  const completed = state?.completed ?? 0;
  const settled = !!state && !loading;
  const failed = state?.status === 'unavailable' || state?.status === 'cancelled';
  const canToggle = settled && completed > 0;
  const label = loading ? `Explaining view · ${completed}/${state.total}`
    : failed ? 'Retry explanations' : 'Explain this view';
  const disabledReason = model.disabledReason ?? invalidChoice ?? (availability?.status === 'checking' ? 'Checking available models…' : undefined);
  const chosenModel = selectedModel?.label ?? settings?.model ?? 'Agent default';
  const chosenEffort = settings?.effort ?? selectedModel?.defaultEffort;
  const choiceLabel = [chosenModel, chosenEffort ? effortLabel(chosenEffort) : (selectedModel?.efforts.length ? 'Default effort' : undefined)].filter(Boolean).join(' · ');
  const executionLabel = state?.execution && [state.execution.model, state.execution.effort ? effortLabel(state.execution.effort) : undefined].filter(Boolean).join(' · ');
  const visibleChoice = loading && executionLabel ? executionLabel : choiceLabel;
  const help = disabledReason ?? (loading ? state.detail ?? `${completed} of ${state.total} components explained. You can keep exploring or stop.`
    : failed && !canToggle && state?.detail ? state.detail
      : `Explain the visible nodes and connections${configuration ? ` with ${choiceLabel}` : ' using your configured agent'}. Collapsed children stay untouched.`);
  const start = () => dispatchBuilderCommand('explain-model');
  return <div className="model-explanation-control">
    <div className={`model-explanation-action${configuration ? ' model-explanation-action-configurable' : ''}`}>
    <CanvasTooltip text={help}><button type="button" className="local-canvas-control model-explanation-start"
      aria-label={label}
      disabled={!!disabledReason || loading} title={help}
      onClick={start}>
      {loading ? <LoaderCircle size={15} className="model-explanation-spinner" aria-hidden="true" /> : <Sparkles size={15} aria-hidden="true" />}
      <span className="model-explanation-button-copy"><span className="local-canvas-control-label">{loading ? `${completed}/${state.total}` : compact && !canToggle && !failed ? 'Explain' : label}</span>
        {configuration && <span className="model-explanation-choice">{visibleChoice}</span>}</span>
    </button></CanvasTooltip>
    {configuration && <CanvasPopover label="Choose explanation model and effort" title="Explanation model" width={316}
      className="model-explanation-configure" panelClassName="agent-model-menu" initialFocusSelector="input"
      disabled={loading || configuration.disabled}
      onOpen={() => { if (!availability || availability.status === 'unchecked') configuration.discover(); }}
      help={`Explanation model and effort: ${choiceLabel}. Saved separately from chat.`}
      trigger={<ChevronDown size={13} aria-hidden="true" />}>
      {close => <>
        <AgentModelMenu availability={availability} settings={settings ?? {}} onChange={configuration.choose}
          onDone={close} onRefresh={() => configuration.discover(true)} />
        <div className="model-explanation-settings-footer">
          {configuration.followsChat ? <span>Using chat settings</span>
            : <button type="button" className="local-canvas-menu-item" onClick={() => { configuration.choose(undefined); close(); }}>Use chat settings</button>}
        </div>
      </>}
    </CanvasPopover>}
    </div>
    {loading && <CanvasTooltip text="Stop generating. Completed explanations stay available."><button type="button" className="local-canvas-control"
      aria-label="Stop explaining view" title="Stop explaining view" onClick={model.cancel}><X size={14} aria-hidden="true" /></button></CanvasTooltip>}
    {settled && <CanvasPopover label="View explanation details" title="View explanations"
      trigger={failed ? <CircleAlert size={14} aria-hidden="true" /> : <span>{completed}/{state.total}</span>}>
      {close => <div className="model-explanation-details">
        {executionLabel && <p className="model-explanation-execution">{executionLabel}</p>}
        {completed > 0 && <p role="status">{state.status === 'cancelled' ? 'Stopped. ' : ''}{completed} of {state.total} components explained.</p>}
        {state.detail && <p role={failed ? 'alert' : undefined}>{state.detail}</p>}
        {canToggle && <button type="button" className="local-canvas-menu-item" aria-pressed={!!context.annotationsVisible}
          onClick={() => { model.toggle(); close(); }}>
          <Sparkles size={14} aria-hidden="true" /><span>{context.annotationsVisible ? 'Hide explanations' : 'Show explanations'}</span>
        </button>}
        <button type="button" className="local-canvas-menu-item" onClick={() => { start(); close(); }} disabled={!!disabledReason}>
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
