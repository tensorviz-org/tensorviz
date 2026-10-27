import React from 'react';
import { ArrowLeft, Check, ChevronRight, MoreHorizontal, SlidersHorizontal } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { ModelNode } from '../types';
import { PassThroughToggle } from '../components/PassThroughToggle';
import { dispatchBuilderCommand } from '../components/ui2/builderCommand';
import { GraphOverlayControls } from '../canvas/GraphQueryPanel';
import CanvasPopover from './CanvasPopover';
import CanvasTooltip from '../components/CanvasTooltip';
import { CanvasThemePreference, setCanvasThemePreference, useCanvasThemePreference } from '../styles/canvasTheme';

type Action = { label: string; icon: LucideIcon; testid?: string; run(): void; hint: string; disabled?: boolean };

/** Group the existing commands without changing capture, export or layout behavior. */
export default function LocalCanvasMenu({ actions, nodes, viewMode, svgOverview, canvasTool }: {
  actions: Action[];
  nodes: ModelNode[];
  viewMode: string;
  svgOverview: boolean;
  canvasTool?: string;
}) {
  const [display, setDisplay] = React.useState(false);
  const themePreference = useCanvasThemePreference();
  return <CanvasPopover help="Arrange the graph, adjust display overlays, focus a selection, or export the canvas." label="More tools" title={display ? 'Display settings' : 'Canvas'} role={display ? 'dialog' : 'menu'}
    trigger={<MoreHorizontal size={15} />} testId="ui2-action-more">
    {close => {
      const item = (id: string) => {
        const action = actions.find(candidate => candidate.testid === `ui2-action-${id}`);
        if (!action) return null;
        const checked = id === 'view-blocks' ? viewMode === 'overview' : id === 'view-expanded' ? viewMode === 'expanded'
          : id === 'svg-overview' ? svgOverview : id === 'select' ? canvasTool === 'select' : undefined;
        return <CanvasTooltip key={id} text={action.hint} block><button type="button" role={checked === undefined ? 'menuitem' : 'menuitemcheckbox'}
          aria-checked={checked} className="local-canvas-menu-item" disabled={action.disabled} title={action.hint}
          data-testid={action.testid} onClick={() => { action.run(); close(); }}>
          <action.icon size={15} aria-hidden="true" /><span>{action.label}</span>
          {checked && <Check size={14} aria-hidden="true" />}
        </button></CanvasTooltip>;
      };
      return display ? <section className="local-canvas-menu-section local-canvas-display">
        <button className="local-canvas-menu-item" onClick={() => setDisplay(false)}><ArrowLeft size={15} /><span>Back to canvas actions</span></button>
        <label className="local-canvas-theme-choice">Appearance
          <select aria-label="Canvas appearance" value={themePreference}
            onChange={event => setCanvasThemePreference(event.target.value as CanvasThemePreference)}>
            <option value="default">Default</option>
            <option value="light">Light · Paper</option>
            <option value="dark">Dark · Ink</option>
          </select>
        </label>
        <GraphOverlayControls />
      </section> : <>
        <section className="local-canvas-menu-section" role="group" aria-label="View">
          <h3>View</h3>
          {item('view-blocks')}{item('view-expanded')}{item('svg-overview')}
          <PassThroughToggle menuItem className="local-canvas-menu-item" nodes={nodes}
            onToggle={shown => { dispatchBuilderCommand('show-unchanged-calls', { shown }); close(); }} />
          <button role="menuitem" className="local-canvas-menu-item" onClick={() => setDisplay(true)}>
            <SlidersHorizontal size={15} /><span>Display settings</span><ChevronRight size={14} />
          </button>
        </section>
        <section className="local-canvas-menu-section" role="group" aria-label="Tools"><h3>Tools</h3>
          {item('layout')}{item('select')}{item('focus')}
        </section>
        <section className="local-canvas-menu-section" role="group" aria-label="Navigate"><h3>Navigate</h3>
          <div className="local-canvas-menu-zoom">{item('zoom-out')}{item('zoom-in')}</div>
          {item('fullscreen')}
        </section>
        <section className="local-canvas-menu-section" role="group" aria-label="Export">{item('download-svg')}</section>
      </>;
    }}
  </CanvasPopover>;
}
