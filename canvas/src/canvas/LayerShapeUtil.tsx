import { GraphReviewBadge, GraphReviewValue } from './GraphReviewChanges';
import { shapeIdToNodeId } from './canvasShapeId';
import { operationTensorRows } from './operationTensorData';
import { OperationTensorRows } from './OperationTensorRows';
import { scopedNodeLabel } from './nodeScope';
import { graphSearchResult, searchStateFor } from './graphQueryState';
/**
 * LayerShapeUtil renders layer and operation cards on the shared tldraw canvas.
 */
import { BaseBoxShapeUtil, HTMLContainer, T, Editor, track } from 'tldraw';
import React from 'react';
import * as LucideIcons from 'lucide-react';
import { LayerShape } from './types';
import { LayerType } from '../types';
import { NodePorts } from './NodePort';
import { opSummary } from './opSummary';
import { operationDisplayName } from './operationDisplayName';
import { cardParams } from '../ide/cardParams';
import { customCardModel } from './customCard';
import { reportNodeClick, reportProjectExpand } from '../ide/projection';
import { useGraphDetail } from './graphDetail';
import { TensorSummary } from './TensorSummary';
import { changedDimensions, tensorEvidence, tensorShapes, coverageLabel } from './graphSemantics';
import { ownGraphIssues, nodeFailureLines } from './graphIssues';
import { NodeChanges } from './NodeChanges';
import { NodeIssues } from './NodeIssues';
import { NodeSourceLink } from './NodeSourceControls';
import { operationGroupCard } from '../services/operationGroups';
import { classifyBlock } from './blockFamily';
import { BlockIdentity } from './BlockIdentity';

import { assignmentEvidenceShown } from './cardMetrics';
export { layerCardHeight, IO_CARD_HEIGHT } from './cardMetrics';

/** One tensor's numbers, as the trace worker measured them (task.md §6.1, M8). */
interface RuntimeStats {
  mean: number;
  std: number;
  absmax: number;
  nan: number;
  inf: number;
  dtype?: string;
  device?: string;
}

const readStats = (value: unknown): RuntimeStats | undefined => {
  if (!value || typeof value !== 'object') return undefined;
  const stats = value as Partial<RuntimeStats>;
  if (typeof stats.mean !== 'number' || typeof stats.std !== 'number') return undefined;
  return {
    mean: stats.mean,
    std: stats.std,
    absmax: typeof stats.absmax === 'number' ? stats.absmax : 0,
    nan: typeof stats.nan === 'number' ? stats.nan : 0,
    inf: typeof stats.inf === 'number' ? stats.inf : 0,
  };
};

/** Three significant figures is what fits, and what a mean is worth reading to. */
const compact = (value: number): string => {
  if (!Number.isFinite(value)) return '—';
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e4)) return value.toExponential(1);
  return value.toFixed(magnitude >= 100 ? 0 : magnitude >= 1 ? 2 : 3);
};

export class LayerShapeUtil extends BaseBoxShapeUtil<LayerShape> {
  static override type = 'layer' as const;

  static override props = {
    w: T.number,
    h: T.number,
    nodeData: T.any,
  };

  getDefaultProps(): LayerShape['props'] {
    return {
      w: 200,
      h: 120,
      nodeData: {
        layerType: {
          id: 'linear',
          name: 'Linear',
          displayName: 'Linear',
          description: 'Linear layer',
          icon: 'Box',
          category: 'linear',
          parameters: [],
        },
        parameters: {},
      },
    };
  }

  // A card is sized from its content by cardMetrics; a resize handle would let a
  // user promise a size the card will not keep, and the handles were the last of
  // tldraw's blue furniture on a selected card.
  override canResize() {
    return false;
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

  override canBind() {
    return true;
  }

  component(shape: LayerShape) {
    return <LayerCard shape={shape} editor={this.editor} />;
  }

  // tldraw 5 strokes selected and hovered shapes from this path in its stock
  // blue, drawn from its JavaScript theme where Canvas tokens cannot reach. A
  // card outlines itself (`.selected`, `.hovered`), so the path is empty, as it
  // already is for connections.
  getIndicatorPath() {
    return new Path2D();
  }
}

const LayerCard = track(function LayerCard({ shape, editor }: { shape: LayerShape; editor: Editor }) {
    const display = operationGroupCard(shape.props.nodeData);
    const data = { ...display, ...tensorShapes(display) };
    const zoomDetail = useGraphDetail(editor);
    // A source assignment has no tensors to summarise at low zoom: its body is
    // the value, the line and "not run", so it keeps that body at every tier.
    const detail = data.review || data.sourceAssignment ? 'inspection' : zoomDetail;
    const evidence = tensorEvidence(data);
    const coverage = coverageLabel(data);
    const layerType = data.layerType as LayerType;
    const isSelected = editor.getSelectedShapeIds().includes(shape.id);
    // A signal read inside component(), so only the card under the pointer
    // re-renders — the same way isSelected works. tldraw's own hover outline is
    // hidden (TldrawCanvas.hiddenComponents) because it is drawn from the stored
    // box and not from the card; this is drawn from the card.
    const isHovered = editor.getHoveredShapeId() === shape.id;

    if (!layerType || typeof layerType !== 'object') {
      return (
        <HTMLContainer style={{ pointerEvents: 'none' }}>
          <div className="layer-node error">
            <div className="node-header">
              <LucideIcons.AlertCircle size={16} />
              <span>Unknown Layer</span>
            </div>
          </div>
        </HTMLContainer>
      );
    }

    const changedAxes = changedDimensions(data.inputShape, data.outputShape);

    const formatShape = (s?: number[]) => (s ? `(${s.join(', ')})` : '');

    const renderShape = (s?: number[]) => {
      if (!s) return '';
      if (!changedAxes.length) return formatShape(s);
      return (
        <>
          {'('}
          {s.map((extent, index) => (
            <React.Fragment key={index}>
              {index > 0 && ', '}
              {changedAxes.includes(index) ? <em className="ui2-node-axis-hot">{extent}</em> : extent}
            </React.Fragment>
          ))}
          {')'}
        </>
      );
    };

    const formatParameterValue = (value: any) => {
      if (Array.isArray(value)) return `(${value.map(item => item === null ? 'None' : String(item)).join(', ')})`;
      if (typeof value === 'boolean') return value ? 'True' : 'False';
      if (typeof value === 'number') return value.toString();
      return String(value);
    };

    // One derived line for ops whose parameters do not read as the operation —
    // a Slice showing `dim/start/length` never says "slice". Read-only: a card is
    // ~64px wide at the default fit-view camera, which is what made the module
    // Expand CTA unusable, so cards display and the Inspector edits.
    const summary = opSummary(layerType.id, data.parameters);
    const operationName = operationDisplayName(data);
    const assignmentTarget = data.sourceAssignment && `sets ${data.sourceAssignment.target}${data.sourceAssignment.conditional ? ' · conditional' : ''}`;

    // What the run actually measured, if the graph was traced with --capture
    // (task.md M8). Shapes say the model is wired; these say it is working.
    const actStats = readStats(data.parameters?._act_stats);
    const gradStats = readStats(data.parameters?._grad_stats);
    // Amber, not red: a NaN is not a broken graph — the wiring is fine and the
    // numbers are not — and drawing it as an error would put it in the same
    // colour as a shape mismatch, which the user fixes a completely different way.
    const poisoned = [actStats, gradStats].some((stats) => stats && (stats.nan > 0 || stats.inf > 0));
    const resolutionSeverity = data.resolutionSeverity;
    const resolutionDetail = data.resolutionDetail;
    // Status keeps its own colour; the block family is the marker beside the title.
    const statusAccent = data.error || resolutionSeverity === 'error'
      ? 'var(--error)'
      : poisoned || resolutionSeverity === 'warning'
        ? 'var(--warning, #f5c451)'
        : 'var(--tv-text-2)';
    const block = classifyBlock(data, 'layerNode');
    const diffClass = data.diffStatus ? ` diff-${data.diffStatus}` : '';
    const nodeClass = `layer-node ui2-node-card ${isSelected ? 'selected' : ''}${isHovered ? ' hovered' : ''} ${data.error ? 'error' : ''}${diffClass}`;
    // `custom_code` / `python`: the two ops the registry deliberately does not
    // model (task.md M6). Non-null for those and null for every other layer, so
    // the ordinary card below is unchanged.
    const custom = customCardModel(layerType.id, data.parameters);
    const nodeId = shapeIdToNodeId(shape.id);
    const issues = ownGraphIssues(nodeId, data);
    const searchState = searchStateFor(nodeId, graphSearchResult.get());
    // The statistics are rendered as their own row, not as parameters: they are
    // objects, and `+2 more params` counting them would also desynchronise
    // cardMetrics (which excludes them for the same reason). A custom card also
    // drops the `_`-prefixed provenance keys, which are what it is built FROM —
    // `_class` and `_source_line` are the card, not rows on it — while keeping
    // the real constructor arguments beside them.
    // ONE filter, shared with cardMetrics via `cardParams`. This used to
    // be a third copy of the rule that dropped `_`-prefixed provenance only for a CUSTOM
    // card, so every parsed node in the IDE drew `_source_file`, `_class`, `_lineno` and
    // `_params_M` as if the user had passed them to a constructor — a Module card showed
    // six of them and "+12 more params". cardParams' own docstring records that this
    // exact drift happened once before; this is the copy that had not been reconciled.
    // Web never showed it because a template's nodes carry no provenance at all: the
    // same model has 0 such keys from the gallery and 9 to 11 after being parsed.
    // `contents` is the other half of the same filter: a `nn.ModuleList` arrives as
    // `_modules: [{op, params}, …]`, which is not a row a card can draw — it rendered as
    // `modules  [object Object],…` — but how many of what IS worth one line.
    const { entries, contents } = cardParams(data.parameters);
    const entryValues = Object.fromEntries(entries);
    const paramEntries = data.review ? Array.from(new Set([...Object.keys(data.review.params).filter(key => !key.startsWith('_')), ...entries.map(([key]) => key)])).map(key => [key, entryValues[key]] as const) : entries;
    const measuredRows = Boolean(data.inputShape || data.outputShape);
    const operationRows = operationTensorRows(data);

    /** A row inside the card that takes a click. The card itself is
     *  pointer-events:none so tldraw owns the gesture; these two opt back in and
     *  claim the event before .tl-canvas can start dragging the shape, the same
     *  way NodePort does. */
    const rowButtonStyle: React.CSSProperties = {
      pointerEvents: 'auto',
      appearance: 'none',
      width: '100%',
      margin: 0,
      padding: 0,
      border: 0,
      background: 'transparent',
      font: 'inherit',
      cursor: 'pointer',
      textAlign: 'left',
    };
    const claim = (event: React.PointerEvent) => {
      editor.markEventAsHandled(event);
      event.stopPropagation();
    };

    const statRow = (label: string, stats: RuntimeStats) => (
      <div
        className="ui2-node-shape-row"
        data-testid={`node-stats-${label}`}
        style={stats.nan > 0 || stats.inf > 0 ? { color: 'var(--warning, #f5c451)' } : undefined}
      >
        {stats.nan > 0 || stats.inf > 0 ? (
          <LucideIcons.AlertTriangle size={11} />
        ) : (
          <LucideIcons.Activity size={11} />
        )}
        <span>{label}</span>
        <span className="ui2-node-shape">
          {stats.nan > 0 || stats.inf > 0
            ? `${stats.nan > 0 ? `${stats.nan} NaN` : ''}${stats.nan > 0 && stats.inf > 0 ? ' · ' : ''}${stats.inf > 0 ? `${stats.inf} Inf` : ''}`
            : `μ ${compact(stats.mean)} σ ${compact(stats.std)}`}
        </span>
      </div>
    );

    return (
      <HTMLContainer style={{ pointerEvents: 'none', overflow: 'visible' }}>
        <div
          className={nodeClass}
          data-detail={detail}
          data-node-id={nodeId}
          data-search={searchState}
          data-source-assignment={data.sourceAssignment ? '' : undefined}
          data-block={block.kind}
          style={{ width: shape.props.w, height: shape.props.h }}
        >
          <div className="ui2-node-header">
            <BlockIdentity block={block} />
            <div className="ui2-node-title-wrap">
              {/* The catalogue names a Linear and an Embedding; it has no name for a
                  `ModuleList`, and a frame child titled from the catalogue alone came
                  out blank. The document always supplies a label, so it is the next
                  thing asked before falling back to the id. */}
              <div className="ui2-node-title">
                {/* A placeholder never beats a real name. The custom-code branch
                    titles a card `_class || 'Custom Code'`, and a function frame's
                    step has no class — its name is the document's label,
                    `F.cross_entropy → loss`, which is the whole reason to draw it. */}
                {data.review?.beforeName ? <GraphReviewValue before={data.review.beforeName} after={operationName || layerType.displayName || layerType.name} /> : custom
                  ? (custom.title === 'Custom Code' && (data as { label?: string }).label) || custom.title
                  : (operationName || layerType.displayName || (data as { label?: string }).label || nodeId)}
              </div>
              {!operationName && <div className="ui2-node-subtitle" title={assignmentTarget}>{assignmentTarget
                ?? custom?.callable ?? data.captureLabel ?? `#${scopedNodeLabel(shapeIdToNodeId(shape.id), data.parentId)}`}</div>}
            </div>
            {/* The assignment's "not run" is a body row: as a header badge it took the
                width the exact target needs. */}
            {data.review && data.review.status !== 'unchanged' ? <GraphReviewBadge review={data.review} /> : data.sourceAssignment ? null
            : issues.length ?<NodeIssues issues={issues} label={data.captureLabel ?? nodeId} className="graph-state-badge"
              style={{ color: statusAccent, background: 'transparent', border: 0 }}><LucideIcons.AlertTriangle size={13} /></NodeIssues>
            : <span className="graph-state-badge" data-state={data.error || resolutionSeverity === 'error' ? 'error' : poisoned || resolutionSeverity === 'warning' ? 'warning' : evidence}
              aria-label={data.error || resolutionDetail || (poisoned ? 'Non-finite activation or gradient values' : evidence === 'stale' ? 'Tensor shapes: stale' : coverage || `Tensor shapes: ${evidence}`)}
              title={data.error || resolutionDetail || (poisoned ? 'Non-finite activation or gradient values' : evidence === 'stale' ? 'Tensor shapes: stale' : coverage || `Tensor shapes: ${evidence}`)}>
              {data.error || poisoned || resolutionSeverity === 'warning' || resolutionSeverity === 'error' ? <LucideIcons.AlertTriangle size={13} /> : data.debugEvidence === 'previous' ? 'Previous run' : data.debugEvidence === 'source' ? 'Source only' : data.diffStatus || (evidence === 'stale' ? evidence : coverageLabel(data, true) || (measuredRows && evidence !== 'measured' ? evidence : ''))}
            </span>}
          </div>

          {data.operationGroup && data.operationGroup.kind !== 'slice' && <button type="button" className="node-operation-group"
            style={{ pointerEvents: 'auto' }} onPointerDown={claim}
            onClick={event => { event.stopPropagation(); data.onRevealOperations?.(nodeId); }}>
            Show {data.operationGroup.members.length} operations <LucideIcons.ChevronDown size={12} />
          </button>}

          {detail === 'inspection' ? (<div className="ui2-node-body">
            {summary && <div className="ui2-node-expr">{summary}</div>}

            {(paramEntries.length > 0 || custom?.snippet || contents) && (
              <div className="ui2-node-list">
                {custom?.snippet && (
                  <div className="ui2-node-row" data-testid="node-custom-snippet">
                    <span className="ui2-node-key">call</span>
                    <span
                      className="ui2-node-value"
                      // The shared mono stack: this row
                      // is measured as an ordinary .ui2-node-row (cardMetrics), and
                      // a different family is a different line height.
                      style={{ fontFamily: 'var(--font-mono)' }}
                    >
                      {custom.snippet}
                    </span>
                  </div>
                )}
                {paramEntries.slice(0, 6).map(([key, value]) => (
                  <div key={key} className="ui2-node-row">
                    <span className="ui2-node-key">{key}</span>
                    <span className="ui2-node-value">{data.review?.params[key] ? <GraphReviewValue {...data.review.params[key]} format={formatParameterValue} /> : formatParameterValue(value)}</span>
                  </div>
                ))}
                {paramEntries.length > 6 && (
                  <div className="ui2-node-hint">+{paramEntries.length - 6} more params</div>
                )}
                {contents && <div className="ui2-node-hint">{contents}</div>}
              </div>
            )}

            {/* A shape on a research card is worth drawing when something
                MEASURED it — Validate traces the class and the numbers are real.
                What must not happen is inference inventing one, which is refused
                where it is computed (ModelBuilder.triggerShapeInference): a card
                that hid measured shapes to avoid invented ones threw away the
                answer along with the guess. */}
            {operationRows !== undefined && <OperationTensorRows data={data} />}
            {((operationRows === undefined && measuredRows) ||
              actStats ||
              gradStats ||
              custom?.location) && (
              <div className="ui2-node-shapes">
                {operationRows === undefined && measuredRows && data.inputShape && (
                  <div className="ui2-node-shape-row ui2-node-tensor-row" data-direction="in">
                    <LucideIcons.ArrowDown size={11} />
                    <span>In</span>
                    <span className="ui2-node-shape">{renderShape(data.inputShape)}</span>
                  </div>
                )}
                {operationRows === undefined && measuredRows && data.outputShape && (
                  <div className="ui2-node-shape-row ui2-node-tensor-row" data-direction="out">
                    <LucideIcons.ArrowUp size={11} />
                    <span>Out</span>
                    <span className="ui2-node-shape">{renderShape(data.outputShape)}</span>
                  </div>
                )}
                {actStats && statRow('Act', actStats)}
                {gradStats && statRow('Grad', gradStats)}
                {custom?.location && !data.sourceLocation && (
                  <button
                    type="button"
                    className="ui2-node-shape-row"
                    data-testid="node-defined-in"
                    title={`Reveal ${custom.location} in the editor`}
                    style={rowButtonStyle}
                    onPointerDown={claim}
                    // The node → code navigation the bridge already carries (M3):
                    // the host owns the file and reveals the span, so this says
                    // which node and nothing about where.
                    onClick={() => reportNodeClick(nodeId)}
                  >
                    <LucideIcons.FileCode2 size={11} />
                    <span>defined in</span>
                    <span className="ui2-node-shape">{custom.location}</span>
                  </button>
                )}
                {custom?.validatable && custom.validatable.state !== 'unsupported' && (
                  // A status, not a button. The run itself is not wired yet, and an
                  // enabled control whose press does nothing is worse than no
                  // control: it teaches the reader that the card lies. What the row
                  // is for today is the half that was missing — saying whether this
                  // node CAN be checked, before anyone presses anything.
                  <div
                    className="ui2-node-shape-row"
                    data-testid="node-validate"
                    title={custom.validatable.detail || `${custom.title} can be checked`}
                  >
                    <LucideIcons.CircleCheck size={11} />
                    <span>
                      {custom.validatable.state === 'ready' ? 'Ready to validate' : 'Cannot validate'}
                    </span>
                  </div>
                )}
                {custom?.projectExpandable && (
                  // "Expand", not "Show contents". It was asked for three times as
                  // "where is the expand button?" while this row was on screen —
                  // the chevron and the word are what a reader scans for, and the
                  // layers icon read as a label rather than a control.
                  <button
                    type="button"
                    className="ui2-node-expand-row"
                    data-testid="node-project-expand"
                    title={`Expand ${custom.title} — show what it contains`}
                    style={rowButtonStyle}
                    onPointerDown={claim}
                    onClick={() => reportProjectExpand(nodeId, true)}
                  >
                    <LucideIcons.ChevronRight size={12} />
                    <span>Expand</span>
                  </button>
                )}

              </div>
            )}
            <NodeChanges data={data} />
            {operationRows === undefined && <NodeSourceLink nodeId={nodeId} data={data} />}
            {assignmentEvidenceShown(data) && <div className="node-assignment-evidence"
              title="Set in source. Not run: what it changes at runtime is unknown until the model is validated.">Set in source · not run</div>}

            {data.error && (
              <div className="ui2-node-error">
                <LucideIcons.AlertTriangle size={11} />
                <span className="graph-failure-facts">{nodeFailureLines(data).map(line => <span key={line} title={line}>{line}</span>)}</span>
              </div>
            )}
            {/* Why this node cannot be checked, and the one line that would fix it.
                cardMetrics has reserved 34px for this sentence since the verdict
                shipped and nothing drew it — a card measured taller than its own
                content, which is the desync that file's whole comment is about. The
                remedy reached the card model too and was thrown away, so a reader
                was told "Cannot validate" and left to work out what to write. */}
            {custom?.validatable?.state === 'needs' && custom.validatable.detail && (
              <div className="ui2-node-resolution info" data-testid="node-validate-detail">
                <span>{custom.validatable.detail}</span>
                {custom.validatable.remedy && (
                  <code className="ui2-node-remedy">{custom.validatable.remedy}</code>
                )}
              </div>
            )}
          </div>) : <TensorSummary data={data} compact={shape.props.h < 170} />}
        </div>
        <NodePorts shape={shape} />
      </HTMLContainer>
    );
});
