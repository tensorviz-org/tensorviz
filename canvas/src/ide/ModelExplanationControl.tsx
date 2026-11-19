import React from 'react';
import { ChevronDown, CircleAlert, Ellipsis, LoaderCircle, MessageCircle, Sparkles, Square } from 'lucide-react';
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
  const preflightFailure = model.startFailure;
  const configuration = model.configuration;
  const settings = configuration?.settings;
  const availability = configuration?.availability;
  const selectedModel = availability?.models.find(entry => entry.id === (settings?.model ?? availability.defaultModel));
  const invalidChoice = availability?.status === 'ready' ? (settings?.model && !selectedModel
    ? `Model “${settings.model}” is unavailable. Choose another model.`
    : settings?.effort && !selectedModel?.efforts.includes(settings.effort)
      ? `Effort “${settings.effort}” is unavailable for this model. Choose another effort.` : undefined) : undefined;
  const loading = !preflightFailure && state?.status === 'loading';
  const completed = state?.completed ?? 0;
  const total = state?.total ?? 0;
  const settled = !!state && !loading;
  const failed = !!preflightFailure || state?.status === 'unavailable' || state?.status === 'cancelled';
  const canToggle = settled && completed > 0;
  const succeeded = !preflightFailure && state?.status === 'ready' && completed > 0;
  const cancelled = !preflightFailure && state?.status === 'cancelled';
  const label = loading ? `Stop explaining view · ${completed}/${total}`
    : succeeded ? `View explained · ${completed}/${total}. Explain this view again`
      : cancelled ? 'Explanations stopped. Retry explanations'
        : failed ? 'Explanations unavailable. Retry explanations' : 'Explain this view';
  const buttonCopy = loading ? `Stop · ${completed}/${total}` : succeeded ? 'Explained'
    : cancelled ? 'Stopped · Retry' : failed ? compact ? 'Retry' : 'Retry explanations' : compact ? 'Explain' : 'Explain this view';
  const disabledReason = model.disabledReason ?? invalidChoice ?? (availability?.status === 'checking' ? 'Checking available models…' : undefined);
  const chosenModel = selectedModel?.label ?? settings?.model ?? 'Agent default';
  const chosenEffort = settings?.effort ?? selectedModel?.defaultEffort;
  const choiceLabel = [chosenModel, chosenEffort ? effortLabel(chosenEffort) : (selectedModel?.efforts.length ? 'Default effort' : undefined)].filter(Boolean).join(' · ');
  const executionLabel = state?.execution && [state.execution.model, state.execution.effort ? effortLabel(state.execution.effort) : undefined].filter(Boolean).join(' · ');
  const visibleChoice = loading && executionLabel ? executionLabel : choiceLabel;
  const operationHelp = loading ? `${completed} of ${total} components explained. Click to stop. Completed explanations stay available.`
    : succeeded ? `${completed} of ${total} components explained. Click to explain the current view again.`
      : failed ? `${cancelled ? 'Stopped.' : 'Explanations unavailable.'}${completed > 0 ? ` ${completed} of ${total} components explained. Completed explanations stay available.` : ''} Click to retry.`
        : `Explain the visible nodes and connections${configuration ? ` with ${choiceLabel}` : ' using your configured agent'}. Collapsed children stay untouched.`;
  const help = [operationHelp, preflightFailure?.detail ?? state?.detail, !loading ? disabledReason : undefined].filter(Boolean).join(' ');
  const start = () => dispatchBuilderCommand('explain-model');
  return <div className={`model-explanation-control${compact ? ' model-explanation-compact' : ''}`}>
    <div className={`model-explanation-action${configuration ? ' model-explanation-action-configurable' : ''}`}>
    <CanvasTooltip text={help}><button type="button" className="local-canvas-control model-explanation-start"
      data-state={loading ? 'working' : succeeded ? 'complete' : cancelled ? 'stopped' : failed ? 'error' : 'idle'}
      aria-label={label}
      aria-busy={loading}
      disabled={!loading && !!disabledReason} title={help}
      onClick={loading ? () => model.cancel() : start}>
      {loading ? <LoaderCircle size={15} className="model-explanation-spinner" aria-hidden="true" />
        : cancelled ? <Square size={15} aria-hidden="true" /> : failed ? <CircleAlert size={15} aria-hidden="true" />
          : <Sparkles size={15} className={succeeded ? 'model-explanation-complete' : undefined} aria-hidden="true" />}
      <span className="model-explanation-button-copy"><span className="local-canvas-control-label">{buttonCopy}</span>
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

    <CanvasPopover label="View explanation details" title="View explanations" disabled={!settled}
      className="model-explanation-overflow" trigger={<Ellipsis size={14} aria-hidden="true" />}>
      {close => state && <div className="model-explanation-details">
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
    </CanvasPopover>
    {loading && <span className="model-explanation-sr-status" role="status">{completed} of {total} components explained</span>}
  </div>;
}
