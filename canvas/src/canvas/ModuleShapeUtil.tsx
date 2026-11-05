import { GraphReviewBadge, GraphReviewValue } from './GraphReviewChanges';
import React from 'react';
import CanvasTooltip from '../components/CanvasTooltip';
import {
  BaseBoxShapeUtil,
  HTMLContainer,
  T,
  resizeBox,
  Editor,
  track,
} from 'tldraw';
import { RefreshCcw, Minimize2, Maximize2, CornerDownRight, Ungroup, FileCode2 } from 'lucide-react';
import type { ModuleShape } from './types';
import { NodePorts } from './NodePort';
import { currentProjection, localBrowserMode, reportNodeClick, reportProjectExpand, showingProjectGraph } from '../ide/projection';
import type { ModelNodeData } from '../types';
import { ModuleBoundaryRows } from '../components/ModuleBoundaryRows';
import { MODULE_PARAM_CELLS, moduleContents, moduleEvidenceLabel, moduleParameters } from '../services/modulePresentation';
import { moduleDefinitionNodeId } from './moduleSourceTarget';
import { shapeIdToNodeId } from './canvasShapeId';
import { classifyBlock } from './blockFamily';
import { BlockIdentity } from './BlockIdentity';
import { moduleCardHeader } from './cardMetrics';
import { CardIdentity } from './NodeCardIdentity';
import { NodeExplanationContext } from './NodeExplanationContext';
import { NodeExplanationAnnotation } from './NodeExplanationAnnotation';
import { nodeAnnotationLayout } from './nodeExplanationPresentation';
import { definitionLinkLayout } from './cardContentLayout';
import { graphIsReadOnly } from './graphAccess';
import { NodeIssues } from './NodeIssues';
import { NodeChanges } from './NodeChanges';
import { ownGraphIssues } from './graphIssues';
import { NodeSourceLink } from './NodeSourceControls';
import { LocalBrowserContext } from '../ide/localBrowserContext';

/* ── Status helpers ──────────────────────────────────────────── */

const statusColor: Record<string, string> = {
  ok: 'var(--graph-muted)',
  warning: 'var(--tv-warning)',
  error: 'var(--tv-error)',
};

const statusLabel: Record<string, string> = {
  ok: 'No issues recorded',
  warning: 'Warnings',
  error: 'Errors',
};

/** Evidence describes this boundary only; 'ok' alone never means validated. */
export function statusLabelForModule(status: string, data: ModelNodeData): string {
  const evidence = moduleEvidenceLabel(data);
  if (evidence !== undefined) return status === 'ok' ? evidence : [evidence, statusLabel[status] || statusLabel.ok].filter(Boolean).join(' · ');
  if (status === 'ok' && localBrowserMode()) return '';
  return statusLabel[status] || statusLabel.ok;
}

/* ── Constructor arguments ───────────────────────────────────── */

const parameterText = (value: unknown): string => Array.isArray(value)
  ? `(${value.map(item => item === null ? 'None' : String(item)).join(', ')})`
  : typeof value === 'boolean' ? (value ? 'True' : 'False') : value === null ? 'None' : String(value);

/** Source constructor arguments of a captured call, two per row. Names keep their
 *  exact spelling in the tooltip; the label only swaps underscores for spaces. */
function ModuleParameterGrid({ data }: { data: ModelNodeData }) {
  const entries = moduleParameters(data);
  if (!entries.length) return null;
  const overflow = entries.length > MODULE_PARAM_CELLS;
  const shown = overflow ? entries.slice(0, MODULE_PARAM_CELLS - 1) : entries;
  return <dl className="module-param-grid" aria-label="Constructor arguments">
    {shown.map(([key, value]) => {
      const change = data.review?.params?.[key];
      return <div key={key} title={key}>
        <dt>{key.replace(/_/g, ' ')}</dt>
        <dd>{change ? <GraphReviewValue {...change} format={parameterText} /> : parameterText(value)}</dd>
      </div>;
    })}
    {overflow && <div className="module-param-more">+{entries.length - shown.length} more</div>}
  </dl>;
}

/* ── In-shape button handlers ─────────────────────────────────── */

/**
 * Handlers for a button rendered INSIDE a tldraw shape. Such a button has to
 * claim the gesture on pointerdown; reacting to `click` is not enough.
 *
 * tldraw owns pointer input on the canvas. Its container handles pointerdown and
 * enters `select.pointing_shape`, and as soon as the pointer travels past
 * `dragDistanceSquared` (16, i.e. 4px) it starts translating the shape. A DOM
 * `click` only fires when pointerdown and pointerup land on the same element with
 * no drag in between — so the gesture is lost in both directions: tldraw nudges
 * the module instead, and no click ever reaches the button.
 *
 * That is what made "Expand Block" look broken. At the default zoom-to-fit camera
 * (~0.34) the CTA is only ~12px tall, and a real hand crossing 4px while pressing
 * a 12px strip is completely ordinary — the module twitched and nothing expanded.
 * Verified: a pixel-stationary synthetic click expanded the block every time,
 * while the same click with 6px of travel did nothing. Calling stopPropagation()
 * from the click handler cannot help, because by then the canvas has already
 * taken the gesture (and usually no click is coming at all).
 *
 * So act on pointerdown and stop it there, before tldraw's container sees it.
 */
function canvasButtonHandlers(action: () => void) {
  return {
    onPointerDown: (event: React.PointerEvent) => {
      event.stopPropagation();
      event.preventDefault();
      action();
    },
    // Keyboard activation (Enter/Space) synthesises a click with detail === 0 and
    // no preceding pointerdown, so it still needs handling here. A real pointer
    // click reports detail >= 1 and was already handled above — running it again
    // would toggle the module straight back.
    onClick: (event: React.MouseEvent) => {
      event.stopPropagation();
      if (event.detail === 0) action();
    },
  };
}

/* ── Shape util ──────────────────────────────────────────────── */

export class ModuleShapeUtil extends BaseBoxShapeUtil<ModuleShape> {
  static override type = 'module-node' as const;

  static override props = {
    w: T.number,
    h: T.number,
    nodeData: T.any,
  };

  getDefaultProps(): ModuleShape['props'] {
    return {
      w: 240,
      h: 160,
      nodeData: {
        layerType: {
          id: 'module',
          name: 'Module',
          displayName: 'Module',
          description: 'A neural network module',
          icon: 'Package',
          category: 'module',
          parameters: [],
        },
        parameters: { name: 'Module' },
        expanded: false,
        moduleInstanceId: '',
        moduleTemplateId: 'Module',
        moduleLayerCount: 0,
        moduleStatus: 'ok',
        moduleLayoutState: 'deterministic',
      } as any,
    };
  }

  /* ── Capabilities ────────────────────────────────────────── */

  override canBind() {
    return true;
  }

  // tldraw draws a selected shape's blue box and corner handles from an overlay
  // util (SelectionForegroundOverlayUtil), sized from the shape's STORED box —
  // which a card's rendered height only approximates, so it sat above the Expand
  // row on one card and below the last row on another. These are the per-shape
  // opt-outs the util honours. The card outlines itself: `.selected` and
  // `.hovered` are exactly its edge. The marquee and every other tool are
  // untouched.
  override hideSelectionBoundsFg() {
    return true;
  }

  override hideResizeHandles() {
    return true;
  }

  override hideRotateHandle() {
    return true;
  }

  override canResize() {
    return true;
  }

  override hideInMinimap(shape: ModuleShape) {
    return Boolean(shape.props.nodeData.expanded);
  }

  /* ── Resize ──────────────────────────────────────────────── */

  override onResize = (shape: any, info: any) => {
    return resizeBox(shape, info);
  };

  /* ── Indicator ───────────────────────────────────────────── */

  // The card outlines itself; see LayerShapeUtil.getIndicatorPath.
  getIndicatorPath() {
    return new Path2D();
  }

  /* ── Component ───────────────────────────────────────────── */

  component(shape: ModuleShape) {
    return <ModuleCard shape={shape} editor={this.editor} />;
  }
}

const ModuleCard = track(function ModuleCard({ shape, editor }: { shape: ModuleShape; editor: Editor }) {
    const explanations = React.useContext(NodeExplanationContext);
    const explanationState = (explanations?.annotations ?? explanations?.states)?.[shapeIdToNodeId(shape.id)];
    const explanation = explanations?.annotationsVisible && explanationState?.status === 'ready'
      && !shape.props.nodeData.review && !shape.props.nodeData.sourceReplacement ? explanationState.explanation : undefined;
    const annotationHeight = explanation ? nodeAnnotationLayout(explanation, shape.props.w).height : 0;
    const copySource = React.useContext(LocalBrowserContext)?.sourceAction === 'copy';
    const data = shape.props.nodeData;
    const isSelected = editor.getSelectedShapeIds().includes(shape.id);
    const isHovered = editor.getHoveredShapeId() === shape.id;
    const projectKind = typeof data.parameters?._kind === 'string' ? data.parameters._kind : '';
    const isProjectFrame = showingProjectGraph() && projectKind !== '';
    const contents = moduleContents(isProjectFrame ? { ...data, expandable: true } : data);
    const isExpanded = contents.canExpand && Boolean(data.expanded);
    const moduleStatus = data.moduleDiagnosticStatus || data.moduleStatus || 'ok';
    const moduleColor = graphIsReadOnly(editor) && moduleStatus === 'ok'
      ? 'var(--text-muted)' : statusColor[moduleStatus] || statusColor.ok;
    const repeatCount = data.repeatCount || 0;
    const repeatBlockLabel = data.repeatBlockLabel || 'block';
    const isRepeated = repeatCount > 1;
    const moduleTemplateId = data.moduleTemplateId || data.parameters.name || data.parameters._class || data.layerType?.name || 'module';
    const moduleTitle = String(moduleTemplateId).split('::').pop();
    const moduleInstanceId = data.moduleInstanceId || shapeIdToNodeId(shape.id);
    const header = moduleCardHeader(data, shape.props.w);
    const previousCall = data.sourceReplacement ? data.captureLabel?.match(/^(.*) · call (.+)$/) : undefined;
    const sourceNodeId = moduleDefinitionNodeId(data.moduleMembers, moduleInstanceId, shape.id);
    const issues = data.moduleDiagnostics ?? ownGraphIssues(shapeIdToNodeId(shape.id), data);
    const status = data.sourceReplacement ? 'Source assignment · Not run' : statusLabelForModule(moduleStatus, data);
    const quietSource = !data.captureRef && !data.review && moduleStatus === 'ok' && !issues.length
      && (data.debugEvidence === 'source' || (localBrowserMode() && moduleEvidenceLabel(data) === undefined));
    const sourceEvidenceIcon = quietSource ? <span className="module-source-evidence" role="img"
      aria-label="Read from code; tensor shapes have not been recorded."
      title="Read from code; tensor shapes have not been recorded."><FileCode2 size={12} aria-hidden="true" /></span> : null;
    const statusBadge = issues.length ? <NodeIssues issues={issues} label={data.captureLabel ?? moduleInstanceId}
      className="module-pill status" style={{ borderColor: moduleColor, color: moduleColor }} />
      : status ? <span className="module-pill status" style={{ borderColor: moduleColor, color: moduleColor }}>{status}</span> : null;
    const previewNodes = data.modulePreviewNodes || [];
    const previewEdges = data.modulePreviewEdges || [];
    const previewLabels = previewNodes
      .map((node) => (node.id || '').split('__').pop() || node.id)
      .filter(Boolean)
      .slice(0, 4);
    const definitionFile = typeof data.parameters?._defined_in === 'string'
      ? data.parameters._defined_in
      : '';
    const definition = definitionFile ? definitionLinkLayout(definitionFile, shape.props.w) : undefined;
    const definitionText = definition?.lines.length === 1 ? definition.lines[0]
      : definition?.lines.map((line, index) => <span key={index}>{line}</span>);
    // Only the active project projection owns host-controlled expansion. Class
    // modules also carry `_kind` metadata; that field alone cannot choose the
    // project command channel. Portable and class modules expand locally.
    // A PROJECT frame, not a module the user grouped on this canvas. The project
    // document sets `_kind` on every card it emits — the one channel that survives
    // to the canvas — and a frame wearing it came from the indexer, so the three
    // callbacks below (`onToggleExpand`, `onResetLayout`, ungroup) do not exist for
    // it: there is no module instance in a graph the page cannot edit.

    // The external local-browser page is a read-only projection. Resetting or
    // expanding a view is presentation; ungrouping rewrites graph structure and
    // therefore has no place there (the framed editor keeps the existing action).
    const canUngroup = !localBrowserMode() && !isProjectFrame && !graphIsReadOnly(editor);

    const accent = 'var(--cat-module)';
    // A recognised torch.nn module keeps its family; user classes stay neutral composites.
    const block = classifyBlock(data, 'moduleNode');

    // Collapsing goes back out the way it came in. `Show contents` on the collapsed
    // card posts `projectExpand`; the host rebuilds the document without this frame
    // in its open set and sends it back. Without this the frame opened and could
    // never be closed — `onToggleExpand` is undefined on a project frame, so the
    // Collapse button returned on its first line and looked broken.
    const toggleExpand = canvasButtonHandlers(() => {
      if (isProjectFrame) {
        reportProjectExpand(shapeIdToNodeId(shape.id), !isExpanded);
        return;
      }
      if (!contents.canExpand || !moduleInstanceId || !data.onToggleExpand) return;
      data.onToggleExpand(moduleInstanceId);
    });

    const resetLayout = canvasButtonHandlers(() => {
      if (!moduleInstanceId || !data.onResetLayout) return;
      data.onResetLayout(moduleInstanceId);
    });

    // Ungroup — the inverse of G, and the only way back out of a module. It goes
    // through the builder-command channel rather than a `data.onUngroup` callback
    // like the two above: the same channel BuilderPageV2's More menu dispatches on,
    // so there is one handler in ModelBuilder and no new prop to thread through the
    // reconciler. The card is where it has to be reachable — the module block is
    // the only thing on screen that names the module.
    const ungroup = canvasButtonHandlers(() => {
      if (!moduleInstanceId) return;
      window.dispatchEvent(new CustomEvent('builder-command', {
        detail: { command: 'ungroup-module', payload: { nodeId: moduleInstanceId } },
      }));
    });

    if (data.capturePassThrough) {
      const path = data.captureLabel || moduleInstanceId;
      const hasSource = localBrowserMode() && Boolean(currentProjection()?.spanmap?.[sourceNodeId]);
      return <HTMLContainer style={{ pointerEvents: 'none' }}>
        <div className={`module-node passthrough-node${isSelected ? ' selected' : ''}${isHovered ? ' hovered' : ''}${data.diffStatus ? ` diff-${data.diffStatus}` : ''}`}
          data-node-id={shapeIdToNodeId(shape.id)} data-pass-through="true"
          data-error={moduleStatus === 'error'}
          style={{ width: shape.props.w, height: shape.props.h }}>
          <div className="node-header module-header" style={{ height: header.height }}
            title={!hasSource && localBrowserMode() ? `${path} · Source location not mapped` : undefined}>
            <BlockIdentity block={block} className="node-icon" size={16} />
            <CardIdentity layout={header} title={header.title} subtitle={header.subtitle} module />
            {hasSource && <button type="button" className="icon-button" style={{ pointerEvents: 'auto' }}
              title={`Open source: ${path}`} aria-label={`Open source: ${path}`}
              {...canvasButtonHandlers(() => reportNodeClick(sourceNodeId))}><FileCode2 size={13} /></button>}
          </div>
          <NodeExplanationAnnotation explanation={explanation} width={shape.props.w} />
          {data.review && data.review.status !== 'unchanged' ? <GraphReviewBadge review={data.review} />
            : <div className="passthrough-detail" title="This call returned the same tensor without recording tensor operations.">Input unchanged</div>}
          <NodePorts shape={shape} />
        </div>
      </HTMLContainer>;
    }

    // ── Expanded mode ──────────────────────────────────────
    if (isExpanded) {
      return (
        <HTMLContainer style={{ pointerEvents: 'none' }}>
          <div
            className={`module-container${isSelected ? ' selected' : ''}${data.diffStatus ? ` diff-${data.diffStatus}` : ''}`}
            data-node-id={shapeIdToNodeId(shape.id)}
            data-block={block.kind}
            data-error={moduleStatus === 'error'}
            style={{
              width: '100%',
              height: '100%',
              border: `1.5px dashed ${moduleStatus === 'error' ? 'var(--error)' : isSelected ? accent : 'var(--border-strong)'}`,
              borderRadius: 'var(--radius-lg)',
              background: 'var(--accent-subtle)',
              boxShadow: isSelected ? `0 0 0 2px ${accent}, var(--shadow-md)` : 'none',
              position: 'relative',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <div
              className="module-container-header"
              style={{
                display: 'flex',
                minHeight: header.height,
                flexShrink: 0,
                alignItems: 'center',
                gap: 8,
                padding: '6px 10px',
                borderTopLeftRadius: 'var(--radius-lg)',
                borderTopRightRadius: 'var(--radius-lg)',
                background: 'var(--bg-elevated)',
                borderBottom: `1px solid ${isSelected ? accent : 'var(--border-color)'}`,
              }}
            >
              <BlockIdentity block={block} className="module-frame-glyph" size={15} />
              <NodeChanges data={data} compact />
              <GraphReviewBadge review={data.review} />
              <CardIdentity layout={header} title={header.title} subtitle={header.subtitle} module />
              {sourceEvidenceIcon}
              {isRepeated && (
                <span
                  className="module-pill repeat"
                  style={{ borderColor: accent, color: accent, fontSize: 10 }}
                  title={`${repeatBlockLabel} repeated ${repeatCount}×`}
                >
                  {repeatBlockLabel} ×{repeatCount}
                </span>
              )}
              {/* "Healthy" is a shape-inference verdict. The project graph never ran
                  one, so on a project frame the pill asserted a clean bill of health
                  nothing had checked — and Reset layout drives `onResetLayout`,
                  which a frame the indexer built does not have. */}
              {!isProjectFrame && (
                statusBadge
              )}
              {!isProjectFrame && contents.canExpand && (
                <CanvasTooltip text="Arrange the layers inside this module again. This changes canvas positions without running the model."><button
                  className="expand-button"
                  style={{ pointerEvents: 'auto' }}
                  {...resetLayout}
                  aria-label="Reset module layout"
                >
                  <RefreshCcw size={12} />
                </button></CanvasTooltip>
              )}
              {canUngroup && (
                <button
                  className="expand-button"
                  style={{ pointerEvents: 'auto' }}
                  {...ungroup}
                  title="Ungroup — break this module back into its layers"
                >
                  <Ungroup size={12} />
                </button>
              )}
              <CanvasTooltip text="Fold this module into one card. Its captured tensor inputs and outputs remain visible; no code runs."><button
                className="icon-button"
                style={{ pointerEvents: 'auto' }}
                {...toggleExpand}
                title="Collapse module (static view; does not run code)"
                aria-label={`Collapse ${data.captureLabel ?? (moduleInstanceId || moduleTemplateId)}`}
              >
                <Minimize2 size={13} />
              </button></CanvasTooltip>
            </div>
            <NodeExplanationAnnotation explanation={explanation} width={shape.props.w} />
            {definitionFile && (
              <div className="module-resolution-block" style={{ height: definition?.height, flexShrink: 0 }}>
                <button
                  type="button"
                  className="module-source-link card-content-lines"
                  style={{ pointerEvents: 'auto', height: definition?.height, font: '10px/15px var(--font-mono)' }}
                  {...canvasButtonHandlers(() => reportNodeClick(sourceNodeId))}
                  title={copySource ? `Copy ${definitionFile} source reference` : `Reveal ${definitionFile} in VS Code`}
                >
                  {definitionText}
                </button>
              </div>
            )}
          </div>
        </HTMLContainer>
      );
    }

    // ── Collapsed mode ─────────────────────────────────────
    const nodeClass = `module-node ${isSelected ? 'selected' : ''}${isHovered ? ' hovered' : ''} collapsed${data.diffStatus ? ` diff-${data.diffStatus}` : ''}`;

    return (
      <HTMLContainer style={{ pointerEvents: 'none' }}>
        <div
          className={nodeClass}
          data-node-id={shapeIdToNodeId(shape.id)}
          data-block={block.kind}
          data-error={moduleStatus === 'error'}
          data-detail="inspection"
          style={{ width: shape.props.w, height: shape.props.h,
            borderColor: moduleStatus === 'error' ? 'var(--error)' : isSelected ? accent : 'var(--border-color)',
            boxShadow: isSelected
              ? `0 0 0 2px ${accent}, var(--shadow-md)`
              : 'var(--shadow-sm)',
          }}
        >
          <div className="node-header module-header" style={{ height: header.height }}>
            <BlockIdentity block={block} className="node-icon" size={16} />
            {data.sourceReplacement ? <div className="node-title">
              {/* A replaced call names the source callable, never an observed class, and
                  keeps its captured path and call ordinal. */}
              <div className="layer-name" title={data.sourceReplacement ? `Previous run: ${data.sourceReplacement.capturedType}. Source assigns ${moduleTemplateId} at ${data.sourceReplacement.target}. ${data.sourceReplacement.coverage === 'partial' ? 'Dependency coverage is partial; ' : ''}which calls change is unverified until validation.` : String(moduleTemplateId)}>
                {data.sourceReplacement?.before ? <GraphReviewValue before={data.sourceReplacement.before} after={moduleTitle} /> : moduleTitle}</div>
              <div className={`layer-id${previousCall ? ' source-call-label' : ''}`}>{previousCall
                ? <><span className="source-call-path">{previousCall[1]}</span><span> · previous call {previousCall[2]}</span></>
                : (data.sourceReplacement ? data.captureLabel : data.review?.summary) ?? data.captureLabel ?? (moduleInstanceId || 'module')}</div>
            </div> : <CardIdentity layout={header} title={header.title} subtitle={header.subtitle} module />}
            {!data.sourceReplacement && <div className="module-header-actions">
              {sourceEvidenceIcon}
              <GraphReviewBadge review={data.review} />
              {!isProjectFrame && contents.canExpand && (
                <CanvasTooltip text="Arrange the layers inside this module again. This changes canvas positions without running the model."><button
                  className="expand-button"
                  style={{ pointerEvents: 'auto' }}
                  {...resetLayout}
                  aria-label="Reset module layout"
                >
                  <RefreshCcw size={13} />
                </button></CanvasTooltip>
              )}
              {canUngroup && (
                <button
                  className="expand-button"
                  style={{ pointerEvents: 'auto' }}
                  {...ungroup}
                  title="Ungroup — break this module back into its layers"
                >
                  <Ungroup size={13} />
                </button>
              )}
                {(isProjectFrame || contents.canExpand) && <CanvasTooltip text="Show the layers and operations inside this module. This only changes the view; no code runs."><button
                  className="icon-button"
                  style={{ pointerEvents: 'auto' }}
                  {...toggleExpand}
                  title={contents.detail || 'Expand module (static view; does not run code)'}
                  aria-label={`Expand ${data.captureLabel ?? (moduleInstanceId || moduleTemplateId)}`}
                  disabled={!isProjectFrame && !contents.canExpand}
                >
                <Maximize2 size={14} />
              </button></CanvasTooltip>}
            </div>}
          </div>

          <NodeExplanationAnnotation explanation={explanation} width={shape.props.w} />
          <div className="node-body" style={{ height: `calc(100% - ${header.height + annotationHeight}px)` }}>
            <NodeChanges data={data} />
            {data.review && <ModuleParameterGrid data={data} />}
            {!isProjectFrame && <ModuleBoundaryRows data={data} width={shape.props.w} />}

            {(contents.label || isRepeated || statusBadge) && <div className="module-pill-row">
              {contents.label && <span className="module-pill">{contents.label}</span>}
              {isRepeated && (
                <span
                  className="module-pill repeat"
                  style={{ borderColor: accent, color: accent }}
                  title={`${repeatBlockLabel} repeated ${repeatCount}×`}
                >
                  {repeatBlockLabel} ×{repeatCount}
                </span>
              )}
              {statusBadge}
            </div>}

            {data.sourceReplacement && <div className="module-source-uncertainty">
              Affected calls unverified{data.sourceReplacement.coverage === 'partial' && <><br />Partial source coverage</>}
            </div>}

            <NodeSourceLink nodeId={shapeIdToNodeId(shape.id)} data={data} width={shape.props.w} />

            {definitionFile && (
              <button
                type="button"
                className="module-source-link card-content-lines"
                style={{ pointerEvents: 'auto', height: definition?.height, flexShrink: 0, font: '10px/15px var(--font-mono)' }}
                {...canvasButtonHandlers(() => reportNodeClick(sourceNodeId))}
                title={copySource ? `Copy ${definitionFile} source reference` : `Reveal ${definitionFile} in VS Code`}
              >
                {definitionText}
              </button>
            )}

            {data.moduleEntryNodeId && data.moduleExitNodeId && (
              <div className="module-route">
                <CornerDownRight size={12} />
                <span>{data.moduleEntryNodeId.split('__').pop()} {'->'} {data.moduleExitNodeId.split('__').pop()}</span>
              </div>
            )}

            {previewNodes.length > 0 && (
              <div className="module-mini-map" aria-label="module preview">
                <svg viewBox="0 0 140 52" width="140" height="52">
                  {previewEdges.map((edge, index) => {
                    const source = previewNodes.find((node) => node.id === edge.source);
                    const target = previewNodes.find((node) => node.id === edge.target);
                    if (!source || !target) return null;
                    return (
                      <line
                        key={`${edge.source}-${edge.target}-${index}`}
                        x1={8 + source.x * 124}
                        y1={6 + source.y * 40}
                        x2={8 + target.x * 124}
                        y2={6 + target.y * 40}
                        stroke="var(--text-muted)"
                        strokeWidth="1.2"
                        opacity="0.6"
                      />
                    );
                  })}
                  {previewNodes.map((node) => (
                    <circle
                      key={node.id}
                      cx={8 + node.x * 124}
                      cy={6 + node.y * 40}
                      r="2.6"
                      fill="var(--graph-muted)"
                    />
                  ))}
                </svg>
              </div>
            )}

            {previewLabels.length > 0 && (
              <div className="module-sublayer-list">
                {previewLabels.map((label, index) => (
                  <span key={`${label}-${index}`} className="module-sublayer-chip">{label}</span>
                ))}
              </div>
            )}

          </div>
        </div>
        <NodePorts shape={shape} />
      </HTMLContainer>
    );
});
