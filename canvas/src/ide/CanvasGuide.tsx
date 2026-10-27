import React from 'react';
import { Info } from 'lucide-react';
import CanvasPopover from './CanvasPopover';
import { dispatchBuilderCommand } from '../components/ui2/builderCommand';

export default function CanvasGuide() {
  React.useEffect(() => {
    const readable = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || !event.shiftKey || event.code !== 'Digit2') return;
      if ((event.target as HTMLElement)?.closest('input, textarea, select, [contenteditable="true"]')) return;
      event.preventDefault();
      event.stopPropagation();
      dispatchBuilderCommand('zoom-readable');
    };
    window.addEventListener('keydown', readable, true);
    return () => window.removeEventListener('keydown', readable, true);
  }, []);
  return <CanvasPopover label="Canvas guide and shortcuts" title="Canvas guide" help="Learn what nodes, tensor shapes and connections mean, and see the navigation shortcuts." shortcut="?" trigger={<Info size={15} />}>
    {close => <>
      <section className="local-canvas-menu-section" aria-label="Reading nodes">
        <h3>Reading nodes</h3>
        <p>Modules group their layers and operations. Expand a module to inspect its contents; expansion does not execute Python.</p>
        <p>In a captured graph, input and output rows show shapes measured for that call. Source-only shapes are hints, not measurements. State nodes represent tensors such as parameters and buffers.</p>
        <p>Opening a model runs it once with its saved inputs on the saved device. To run a captured model on another device, choose Run model, then Run. A passing run describes those example inputs, not every possible input.</p>
      </section>
      <section className="local-canvas-menu-section" aria-label="Connections">
        <h3>Connections</h3>
        <div className="local-canvas-legend-row"><i />Tensor flow · arrow points to destination</div>
        <div className="local-canvas-legend-row"><i className="dots" />Residual / skip · still carries data</div>
        <div className="local-canvas-legend-row"><i className="dashes" />Secondary project relationship</div>
        <p>Mixed ×N combines ordinary and residual paths.</p>
        <p>Comparison: + added · − removed · ~ changed.</p>
      </section>
      <section className="local-canvas-menu-section" aria-label="Keyboard shortcuts">
        <h3>Navigate</h3>
        <button className="local-canvas-menu-item" onClick={() => { dispatchBuilderCommand('fit-view'); close(); }}><span>Fit graph</span><kbd>⌘ / Ctrl 0</kbd></button>
        <button className="local-canvas-menu-item" onClick={() => { dispatchBuilderCommand('zoom-readable'); close(); }}><span>Zoom to readable detail</span><kbd>⇧ 2</kbd></button>
        <div className="local-canvas-shortcut"><span>Find a layer</span><kbd>⌘ / Ctrl P</kbd></div>
        <div className="local-canvas-shortcut"><span>Move between layers</span><kbd>[ / ]</kbd></div>
        <div className="local-canvas-shortcut"><span>Open selected layer</span><kbd>Enter</kbd></div>
        <div className="local-canvas-shortcut"><span>Pan canvas</span><kbd>Space + drag</kbd></div>
        <div className="local-canvas-shortcut"><span>Close / clear focus</span><kbd>Esc</kbd></div>
      </section>
    </>}
  </CanvasPopover>;
}
