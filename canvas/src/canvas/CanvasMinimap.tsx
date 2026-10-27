import React, { useId, useRef, useState } from 'react';
import { DefaultMinimap, useEditor } from 'tldraw';
import { ChevronDown, ChevronUp, Map } from 'lucide-react';
import { cameraDuration } from './reducedMotion';
import './CanvasMinimap.css';
import { LocalBrowserContext } from '../ide/localBrowserContext';
import { useCanvasTheme } from '../styles/canvasTheme';

/** Native camera navigation; this never changes graph or source state. */
export function CanvasMinimap({ sidePanelOpen = false }: { sidePanelOpen?: boolean }) {
  const editor = useEditor();
  const local = React.useContext(LocalBrowserContext);
  const [expanded, setExpanded] = useState(!local);
  const toggle = useRef<HTMLButtonElement>(null);
  const id = useId();
  // MinimapManager reads its palette once; remount it when the theme changes.
  const theme = useCanvasTheme();

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    const bounds = editor.getViewportPageBounds();
    const point = { x: bounds.center.x, y: bounds.center.y };
    const options = { animation: { duration: cameraDuration(100) } };
    switch (event.key) {
      case 'ArrowLeft': point.x -= bounds.width / 5; break;
      case 'ArrowRight': point.x += bounds.width / 5; break;
      case 'ArrowUp': point.y -= bounds.height / 5; break;
      case 'ArrowDown': point.y += bounds.height / 5; break;
      case 'Home': editor.zoomToFit(options); break;
      case '+': case '=': editor.zoomIn(undefined, options); break;
      case '-': editor.zoomOut(undefined, options); break;
      case 'Escape':
        setExpanded(false);
        toggle.current?.focus();
        break;
      default: return;
    }
    event.preventDefault();
    event.stopPropagation();
    if (event.key.startsWith('Arrow')) editor.centerOnPoint(point, options);
  };

  return (
    <section
      className="canvas-minimap"
      aria-label="Graph overview"
      data-panel-open={sidePanelOpen}
      data-expanded={expanded}
      onPointerDown={event => event.stopPropagation()}
      // DefaultMinimap must receive pointerup on document.body to end its drag.
      onPointerUp={event => editor.markEventAsHandled(event)}
    >
      <button
        className="canvas-minimap-toggle"
        ref={toggle}
        aria-label={expanded ? 'Hide minimap' : 'Show minimap'}
        aria-expanded={expanded}
        aria-controls={id}
        onClick={() => setExpanded(value => !value)}
      >
        <Map size={14} aria-hidden="true" />
        <span>Overview</span>
        {expanded ? <ChevronUp size={14} aria-hidden="true" /> : <ChevronDown size={14} aria-hidden="true" />}
      </button>
      {expanded && (
        <div
          id={id}
          className="canvas-minimap-navigation"
          role="group"
          aria-label="Minimap navigation"
          title="Click or drag to navigate. Arrow keys pan, + and − zoom, Home fits the graph."
          tabIndex={0}
          onKeyDownCapture={onKeyDown}
        >
          <DefaultMinimap key={theme} />
        </div>
      )}
    </section>
  );
}
