import React from 'react';
import { createPortal } from 'react-dom';
import { Check, ChevronDown, FileCode2, GitCommitHorizontal, GitCommitVertical, Minus, Moon, Plus, Proportions, Scan, Search, SlidersHorizontal, Sun } from 'lucide-react';
import { LocalBrowserContext } from './localBrowserContext';
import LocalBrowserValidate from './LocalBrowserValidate';
import { dispatchBuilderCommand } from '../components/ui2/builderCommand';
import GraphDetailToggle from '../components/GraphDetailToggle';
import type { GraphDetailLevel } from '../services/graphDetailLevels';
import CanvasPopover from './CanvasPopover';
import CanvasGuide from './CanvasGuide';
import CanvasTooltip from '../components/CanvasTooltip';
import { setCanvasThemePreference, useCanvasTheme } from '../styles/canvasTheme';
import CodeFocusControls from './CodeFocusControls';
import type { LayoutDirection } from '../canvas/edgeAnchors';
import type { LayoutChoice } from './layoutDirection';
import ModelExplanationControl from './ModelExplanationControl';

const DIRECTION_ICON = { horizontal: GitCommitHorizontal, vertical: GitCommitVertical };
const LAYOUT_LABEL = { auto: 'Auto', horizontal: 'Horizontal', vertical: 'Vertical' };

/** Auto, Horizontal or Vertical. The trigger shows the direction on screen. */
function LayoutDirectionControl({ choice, direction }: { choice: LayoutChoice; direction: LayoutDirection }) {
  const Icon = DIRECTION_ICON[direction];
  const label = `Layout direction: ${choice === 'auto' ? `Auto (${direction})` : LAYOUT_LABEL[choice]}`;
  return <CanvasPopover label={label} title="Layout" role="menu" help="Arrange the graph left to right or top to bottom."
    trigger={<Icon size={14} aria-hidden="true" />} testId="ui2-layout-direction">
    {close => <div className="local-canvas-menu-section">
      {(['auto', 'horizontal', 'vertical'] as const).map(option => {
        const OptionIcon = option === 'auto' ? Proportions : DIRECTION_ICON[option];
        return <button key={option} type="button" role="menuitemradio" aria-checked={choice === option} className="local-canvas-menu-item"
          onClick={() => { dispatchBuilderCommand('set-layout-direction', { choice: option }); close(); }}>
          <OptionIcon size={15} aria-hidden="true" /><span>{LAYOUT_LABEL[option]}</span>
          {option === 'auto' && choice === 'auto' && <small className="local-canvas-menu-hint">{LAYOUT_LABEL[direction]}</small>}
          {choice === option && <Check size={14} aria-hidden="true" />}
        </button>;
      })}
    </div>}
  </CanvasPopover>;
}

/** Existing canvas actions, rendered into the page's header and footer. */
export default function LocalCanvasNavigation({ onFind, onReveal, hasSelection, nodeCount, more, fullscreen = false, detailLevel, findButtonRef, layout }: {
  onFind(): void;
  onReveal(): void;
  hasSelection: boolean;
  nodeCount: number;
  more: React.ReactNode;
  fullscreen?: boolean;
  detailLevel?: GraphDetailLevel;
  findButtonRef?: React.RefObject<HTMLButtonElement | null>;
  layout?: { choice: LayoutChoice; direction: LayoutDirection };
}) {
  const context = React.useContext(LocalBrowserContext);
  const navigation = context?.navigation;
  const copySource = context?.sourceAction === 'copy';
  const theme = useCanvasTheme();
  if (!navigation) return null;
  const compact = navigation.compact && !fullscreen;
  const header = <>
      <div className="local-canvas-tool-group" role="group" aria-label="Graph view">
      <CodeFocusControls compact={compact} />
      {!context?.reviewing && <ModelExplanationControl compact={!!compact} />}
      {!context?.reviewing && <GraphDetailToggle level={detailLevel} compact={compact} />}
      <CanvasTooltip text="Find a layer by name and move the view to it. Press ⌘ / Ctrl P."><button ref={findButtonRef} type="button" className="local-canvas-control" aria-label="Find a layer" title="Find a layer (⌘P)"
        onClick={onFind}><Search size={15} aria-hidden="true" /></button></CanvasTooltip>
      </div>
      {!context?.reviewing && <div className="local-canvas-tool-group" role="group" aria-label="Model capture">
      {context?.page.opening && <CanvasTooltip text="Review the model configuration and example tensors saved in tensorviz.toml. Saving runs the model with them."><button type="button" className="local-canvas-control"
        aria-label="Change inputs" aria-expanded={navigation.inputsOpen}
        disabled={context?.validation?.running || context.status?.phase !== 'connected' || context.status?.hostLink !== 'connected'}
        onClick={navigation.changeInputs} title="View or change the example inputs">
        <SlidersHorizontal size={15} aria-hidden="true" />{!compact && <span className="local-canvas-control-label">Inputs</span>}
      </button></CanvasTooltip>}
      {!context?.reviewing && <LocalBrowserValidate controls="run" />}
      </div>}
      <div className="local-canvas-tool-group" role="group" aria-label="Canvas tools">
      <CanvasTooltip text={theme === 'dark' ? 'Switch the Canvas to Light · Paper.' : 'Switch the Canvas to Dark · Ink.'}>
        <button type="button" className="local-canvas-control" role="switch" aria-label="Dark canvas appearance"
          aria-checked={theme === 'dark'} title={theme === 'dark' ? 'Switch to light appearance (Paper)' : 'Switch to dark appearance (Ink)'}
          data-testid="canvas-theme-switch" onClick={() => setCanvasThemePreference(theme === 'dark' ? 'light' : 'dark')}>
          {theme === 'dark' ? <Sun size={15} aria-hidden="true" /> : <Moon size={15} aria-hidden="true" />}
          {!compact && <span className="local-canvas-control-label">{theme === 'dark' ? 'Light' : 'Dark'}</span>}
        </button>
      </CanvasTooltip>
      {!context?.reviewing && more}
      </div>
    </>;
  const footer = <>
      {!context?.reviewing && <LocalBrowserValidate controls="result" />}
      <span className="local-canvas-node-count" title="Model summary">{nodeCount} graph {nodeCount === 1 ? 'node' : 'nodes'}</span>
      {hasSelection && <CanvasTooltip text={copySource ? 'Copy the selected layer’s file and line.' : 'Open the selected layer’s mapped source location in VS Code.'}><button type="button" className="local-canvas-control" onClick={onReveal}
        aria-label={copySource ? 'Copy source reference' : 'Reveal in editor'} title={copySource ? 'Copy the selected layer’s file and line' : 'Jump to the line that declares the selected layer'} data-testid="ui2-view-code">
        <FileCode2 size={14} aria-hidden="true" /><span className="local-canvas-control-label">Source</span>
      </button></CanvasTooltip>}
      {!context?.reviewing && layout && <LayoutDirectionControl {...layout} />}
      <div className="local-canvas-navigation" role="group" aria-label="Zoom and fit">
        <CanvasTooltip text="Zoom out to see more of the graph."><button type="button" className="local-canvas-control" aria-label="Zoom out" title="Zoom out"
          onClick={() => dispatchBuilderCommand('zoom-out')}><Minus size={14} aria-hidden="true" /></button></CanvasTooltip>
        <CanvasTooltip text="Fit the entire graph in the canvas. The graph also fits automatically when it first appears, validation completes, or the canvas is resized. For larger text, choose Readable detail from Zoom options. Press ⌘ / Ctrl 0."><button type="button" className="local-canvas-control" aria-label="Fit graph to view" title="Fit to view (⌘0)"
          data-testid="ui2-action-fit-view" onClick={() => dispatchBuilderCommand('fit-view')}><Scan size={14} aria-hidden="true" /><span>Fit</span></button></CanvasTooltip>
        <CanvasPopover label="Zoom options" help="Choose between fitting the whole graph and zooming to readable node details." trigger={<ChevronDown size={13} />} className="local-canvas-zoom-options">
          {close => <div className="local-canvas-menu-section">
            <button className="local-canvas-menu-item" onClick={() => { dispatchBuilderCommand('fit-view'); close(); }}><span>Fit entire graph</span><kbd>⌘ / Ctrl 0</kbd></button>
            <button className="local-canvas-menu-item" onClick={() => { dispatchBuilderCommand('zoom-readable'); close(); }}><span>Readable detail</span><kbd>⇧ 2</kbd></button>
          </div>}
        </CanvasPopover>
        <CanvasTooltip text="Zoom in to read node labels and tensor shapes."><button type="button" className="local-canvas-control" aria-label="Zoom in" title="Zoom in"
          onClick={() => dispatchBuilderCommand('zoom-in')}><Plus size={14} aria-hidden="true" /></button></CanvasTooltip>
      </div>
      <CanvasGuide />
    </>;
  if (fullscreen) return <>
    <div className="local-canvas-tools local-canvas-fullscreen-header">{header}</div>
    <div className="local-canvas-footer-tools local-canvas-fullscreen-footer">{footer}</div>
  </>;
  return <>
    {navigation.header && createPortal(header, navigation.header)}
    {navigation.footer && createPortal(footer, navigation.footer)}
  </>;
}
