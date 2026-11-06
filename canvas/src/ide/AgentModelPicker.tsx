import React from 'react';
import { ArrowLeft, Check, ChevronDown, ChevronRight, RefreshCw, Search, TriangleAlert } from 'lucide-react';
import type { CanvasAgentAvailability, CanvasAgentModel, CanvasAgentSettings } from './protocol.generated';
import CanvasPopover from './CanvasPopover';
import './agentModelPicker.css';

export interface AgentModelPickerProps {
  availability?: CanvasAgentAvailability;
  settings: CanvasAgentSettings;
  onChange(settings: CanvasAgentSettings): void;
  disabled?: boolean;
  labelPrefix?: string;
  onRefresh?(): void;
  onOpen?(): void;
}

export const effortLabel = (value: string) => {
  const words = value === 'xhigh' ? 'extra high' : value.replace(/_/g, ' ');
  return words.charAt(0).toUpperCase() + words.slice(1);
};

/** Model capabilities come from the connected provider; no local model catalogue. */
export default function AgentModelPicker({ availability, settings, onChange, disabled = false, labelPrefix = 'Agent', onRefresh, onOpen }: AgentModelPickerProps) {
  const models = availability?.models ?? [];
  const current = models.find(model => model.id === (settings.model ?? availability?.defaultModel));
  const missingModel = availability?.status === 'ready' && !!settings.model && !current;
  const missingEffort = availability?.status === 'ready' && !!settings.effort && !current?.efforts.includes(settings.effort);
  const modelLabel = settings.model ? current?.label ?? settings.model : current?.label ?? 'Default model';
  const shownEffort = settings.effort ?? current?.defaultEffort;
  const defaultDescription = current?.description;
  const showEffort = !!current?.efforts.length || !!settings.effort;
  return <div className="agent-model-picker" role="group" aria-label={`${labelPrefix} model and effort`}>
    <CanvasPopover label={`${labelPrefix} model: ${modelLabel}`} title="Choose model"
      help={`${modelLabel}${!settings.model ? ' · Agent default' : ''}${defaultDescription ? ` — ${defaultDescription}` : ''}`}
      className={`agent-model-chip${missingModel ? ' agent-model-chip-invalid' : ''}`} panelClassName="agent-model-menu"
      width={316} disabled={disabled} initialFocusSelector="input" onOpen={onOpen}
      trigger={<>{missingModel && <TriangleAlert size={12} aria-hidden="true" />}<span>{modelLabel}</span><ChevronDown size={12} aria-hidden="true" /></>}>
      {close => <AgentModelMenu availability={availability} settings={settings} onChange={onChange} onDone={close} onRefresh={onRefresh} />}
    </CanvasPopover>
    {showEffort && <CanvasPopover label={`${labelPrefix} effort: ${shownEffort ? effortLabel(shownEffort) : 'Default'}`} title="Reasoning effort"
      help={missingEffort ? `Saved effort “${settings.effort}” is unavailable. Choose another effort.`
        : `${shownEffort ? effortLabel(shownEffort) : 'Default'} effort${!settings.effort ? ' · Agent default' : ''}. Higher effort can take longer.`}
      className={`agent-effort-chip${missingEffort ? ' agent-model-chip-invalid' : ''}`} panelClassName="agent-model-menu" width={252}
      role="menu" disabled={disabled} onOpen={onOpen} trigger={<>{missingEffort && <TriangleAlert size={12} aria-hidden="true" />}<span>{shownEffort ? effortLabel(shownEffort) : 'Default'}</span><ChevronDown size={12} aria-hidden="true" /></>}>
      {close => <EffortOptions model={current} settings={settings} onChoose={next => { onChange(next); close(); }} />}
    </CanvasPopover>}
  </div>;
}

function EffortOptions({ model, settings, onChoose }: { model?: CanvasAgentModel; settings: CanvasAgentSettings; onChoose(settings: CanvasAgentSettings): void }) {
  const modelSettings = settings.model ? { model: settings.model } : {};
  return <div className="agent-effort-options">
    {model && <p className="agent-model-caption">{model.label}</p>}
    <button type="button" className="local-canvas-menu-item" role="menuitemradio" aria-checked={!settings.effort}
      onClick={() => onChoose(modelSettings)}><span>Default{model?.defaultEffort ? ` (${effortLabel(model.defaultEffort)})` : ''}</span>{!settings.effort && <Check size={14} aria-hidden="true" />}</button>
    {model?.efforts.map(effort => <button key={effort} type="button" className="local-canvas-menu-item" role="menuitemradio" aria-checked={settings.effort === effort}
      onClick={() => onChoose({ ...modelSettings, effort })}><span>{effortLabel(effort)}</span>{settings.effort === effort && <Check size={14} aria-hidden="true" />}</button>)}
    {settings.effort && !model?.efforts.includes(settings.effort) && <p className="agent-model-caption agent-model-unavailable">{effortLabel(settings.effort)} is unavailable.</p>}
  </div>;
}

/** Embeddable menu content for task-specific split buttons without nested portals. */
export function AgentModelMenu({ availability, settings, onChange, onRefresh, onDone }: Pick<AgentModelPickerProps, 'availability' | 'settings' | 'onRefresh' | 'onChange'> & { onDone?(): void }) {
  const onChoose = (next: CanvasAgentSettings) => { onChange(next); onDone?.(); };
  const [query, setQuery] = React.useState('');
  const [effortModel, setEffortModel] = React.useState<CanvasAgentModel | null>(null);
  const root = React.useRef<HTMLDivElement>(null);
  const search = React.useRef<HTMLInputElement>(null);
  const back = React.useRef<HTMLButtonElement>(null);
  const previousEffort = React.useRef<string | null>(null);
  const models = availability?.models ?? [];
  const defaultModel = models.find(model => model.id === availability?.defaultModel);
  const listedModels = models.filter(model => model.id !== 'default' || settings.model === 'default');
  const visibleModels = listedModels.filter(model => `${model.label} ${model.id}`.toLowerCase().includes(query.trim().toLowerCase()));
  const unavailable = availability?.status === 'ready' && !!settings.model && !models.some(model => model.id === settings.model);
  React.useEffect(() => {
    if (effortModel) back.current?.focus();
    else if (previousEffort.current) {
      const trigger = Array.from(root.current?.querySelectorAll<HTMLButtonElement>('[data-effort-model]') ?? []).find(button => button.dataset.effortModel === previousEffort.current);
      (trigger ?? search.current)?.focus();
    }
    previousEffort.current = effortModel?.id ?? null;
  }, [effortModel]);
  const keyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Escape' && effortModel) {
      event.preventDefault(); event.stopPropagation(); setEffortModel(null); return;
    }
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
    if ((event.target as HTMLElement).matches('input') && !['ArrowDown', 'ArrowUp'].includes(event.key)) return;
    const items = Array.from(root.current?.querySelectorAll<HTMLButtonElement>('[role^="menuitem"]:not(:disabled)') ?? []);
    if (!items.length) return;
    const current = items.indexOf(document.activeElement as HTMLButtonElement);
    const index = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1
      : current < 0 ? event.key === 'ArrowUp' ? items.length - 1 : 0
        : (current + (event.key === 'ArrowUp' ? -1 : 1) + items.length) % items.length;
    event.preventDefault(); event.stopPropagation(); items[index].focus();
  };
  return <div ref={root} className="agent-model-menu-content" onKeyDown={keyDown}>
    {effortModel ? <>
      <button ref={back} type="button" className="agent-model-back" onClick={() => setEffortModel(null)}><ArrowLeft size={14} aria-hidden="true" />Models</button>
      <div role="menu" aria-label={`Effort for ${effortModel.label}`}><EffortOptions model={effortModel}
        settings={{ model: effortModel.id, ...((settings.model ?? availability?.defaultModel) === effortModel.id && settings.effort ? { effort: settings.effort } : {}) }} onChoose={onChoose} /></div>
    </> : <>
      <label className="agent-model-search"><Search size={14} aria-hidden="true" /><input ref={search} type="search" aria-label="Search models" placeholder="Search models…" value={query}
        onChange={event => setQuery(event.target.value)} /></label>
      <div className="agent-model-results" role="menu" aria-label="Models">
        {!query.trim() && <button type="button" className="local-canvas-menu-item agent-model-default" role="menuitemradio" aria-checked={!settings.model}
          onClick={() => onChoose({})}><span>Agent default{defaultModel && <small>{defaultModel.label}</small>}</span>{!settings.model && <Check size={14} aria-hidden="true" />}</button>}
        {visibleModels.map(model => <div key={model.id} className={`agent-model-row${settings.model === model.id ? ' agent-model-row-selected' : ''}`}>
          <button type="button" className="local-canvas-menu-item" role="menuitemradio" aria-checked={settings.model === model.id} title={model.description || model.id}
            onClick={() => onChoose(settings.model === model.id ? settings : { model: model.id })}>
            <span>{model.label}</span>{settings.model === model.id && <Check size={14} aria-hidden="true" />}
          </button>
          {model.efforts.length > 0 && <button type="button" role="menuitem" className="agent-model-effort-open" aria-label={`Effort for ${model.label}`}
            title={`Choose effort for ${model.label}`} data-effort-model={model.id} onClick={() => setEffortModel(model)}><ChevronRight size={14} aria-hidden="true" /></button>}
        </div>)}
        {query.trim() && !visibleModels.length && <p className="agent-model-caption" role="status">No matching models.</p>}
        {!models.length && !query.trim() && <p className="agent-model-caption" role="status">{availability?.status === 'checking' ? 'Checking available models…' : availability?.detail || 'Connect an agent to list models.'}</p>}
        {unavailable && <p className="agent-model-caption agent-model-unavailable">{settings.model} is unavailable. Choose another model.</p>}
      </div>
      {onRefresh && <div className="agent-model-menu-footer"><button type="button" className="local-canvas-menu-item" disabled={availability?.status === 'checking'} onClick={onRefresh}>
        <RefreshCw size={14} aria-hidden="true" /><span>Refresh models</span></button></div>}
    </>}
  </div>;
}
