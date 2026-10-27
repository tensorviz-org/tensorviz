import React from 'react';
import { Focus } from 'lucide-react';
import CanvasPopover from './CanvasPopover';
import { LocalBrowserContext } from './localBrowserContext';
import type { CodeFocusOptions } from './protocol.generated';

export default function CodeFocusControls({ compact }: { compact?: boolean }) {
  const context = React.useContext(LocalBrowserContext);
  const focus = context?.codeFocus;
  if (!focus || context?.reviewing) return null;
  const { options, change } = focus;
  const disabled = context?.status?.phase !== 'connected' || context.status.hostLink !== 'connected';
  return <CanvasPopover label="Code focus" help="Dim unrelated code in VS Code. Choose the whole model class, enclosing functions, or selected expressions."
    className={options.enabled ? 'local-canvas-code-focus-active' : ''}
    trigger={<><Focus size={15} aria-hidden="true" />{!compact && <span className="local-canvas-control-label">Focus code</span>}</>}>
    {() => <div className="local-canvas-code-focus">
      <label className="local-canvas-code-focus-switch"><input type="checkbox" checked={options.enabled} disabled={disabled}
        onChange={event => change({ ...options, enabled: event.target.checked })} />Dim unrelated code</label>
      <label>Scope<select aria-label="Code focus scope" value={options.scope} disabled={disabled}
        onChange={event => change({ ...options, scope: event.target.value as CodeFocusOptions['scope'] })}>
        <option value="class">Model / class</option><option value="function">Function</option><option value="node">Node / expression</option>
      </select></label>
      <label>Parameter source<select aria-label="Parameter source" value={options.references} disabled={disabled || !focus.parameter}
        onChange={event => change({ ...options, references: event.target.value as CodeFocusOptions['references'] })}>
        <option value="declaration">Declaration</option><option value="uses">Uses</option><option value="both">Both</option>
      </select></label>
      {!focus.parameter && <p>References apply when the selected node is a parameter. Scope still follows the current selection.</p>}
      <label>Surrounding code opacity <span>{options.opacity}%</span><input type="range" min="10" max="60" step="5"
        aria-label="Surrounding code opacity" value={options.opacity} disabled={disabled}
        onChange={event => change({ ...options, opacity: Number(event.target.value) })} /></label>
      <p role="status">{focus.detail}</p>
    </div>}
  </CanvasPopover>;
}
