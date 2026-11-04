import React from 'react';
import { ChevronDown, Focus } from 'lucide-react';
import CanvasPopover from './CanvasPopover';
import CanvasTooltip from '../components/CanvasTooltip';
import { LocalBrowserContext } from './localBrowserContext';
import type { CodeFocusOptions } from './protocol.generated';
import { setCodeFocusOn, useCodeFocusOn } from './codeFocusOn';

/** One click turns Code focus on or off; how much code it dims sits beside it. */
export default function CodeFocusControls({ compact }: { compact?: boolean }) {
  const context = React.useContext(LocalBrowserContext);
  const focus = context?.codeFocus;
  const on = useCodeFocusOn();
  if (!focus || context?.reviewing) return null;
  const { options, change } = focus;
  const disabled = context?.status?.phase !== 'connected' || context.status.hostLink !== 'connected';
  return <span className="local-canvas-code-focus-controls">
    <CanvasTooltip text="Code focus: selecting a node opens its source beside the Canvas, and double-click or Enter on a layer moves to it. Off, exploring the graph never opens or dims code; source links still open it.">
      <button type="button" className={`local-canvas-control${on ? ' local-canvas-code-focus-active' : ''}`}
        aria-label="Code focus" aria-pressed={on} title="Code focus" onClick={() => setCodeFocusOn(!on)}>
        <Focus size={15} aria-hidden="true" />{!compact && <span className="local-canvas-control-label">Code focus</span>}
      </button>
    </CanvasTooltip>
    <CanvasPopover label="Code focus options" title="Code focus" className="local-canvas-code-focus-options"
      help="While Code focus is on, optionally dim unrelated code, keeping the model class, enclosing functions or selected expressions readable."
      trigger={<ChevronDown size={13} aria-hidden="true" />}>
      {() => <div className="local-canvas-code-focus">
        <label className="local-canvas-code-focus-switch"><input type="checkbox" checked={options.enabled} disabled={disabled}
          onChange={event => change({ ...options, enabled: event.target.checked })} />Dim unrelated code</label>
        <label>Scope<select aria-label="Code focus scope" value={options.scope} disabled={disabled}
          onChange={event => change({ ...options, scope: event.target.value as CodeFocusOptions['scope'] })}>
          <option value="class">Model / class</option><option value="function">Function</option><option value="node">Node / expression</option>
        </select></label>
        {focus.parameter && <label>Parameter source<select aria-label="Parameter source" value={options.references} disabled={disabled}
          onChange={event => change({ ...options, references: event.target.value as CodeFocusOptions['references'] })}>
          <option value="declaration">Declaration</option><option value="uses">Uses</option><option value="both">Both</option>
        </select></label>}
        <label>Surrounding code opacity <span>{options.opacity}%</span><input type="range" min="10" max="60" step="5"
          aria-label="Surrounding code opacity" value={options.opacity} disabled={disabled}
          onChange={event => change({ ...options, opacity: Number(event.target.value) })} /></label>
        <p role="status">{focus.detail}</p>
      </div>}
    </CanvasPopover>
  </span>;
}
