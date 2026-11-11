import React from 'react';
import { BaseBoxShapeUtil, HTMLContainer, T, Editor, track, useSvgExportContext } from 'tldraw';
import { ModelEdgeShape } from './types';
import { edgeControlPoints } from './edgeCurve';
import { selectedEdgeId } from './edgeSelection';
import { graphFocus } from './graphFocus';
import { NodeExplanationContext } from './NodeExplanationContext';
import { graphNodes } from './graphQueryState';
import { applicableViewExplanation, viewExplanationConnection, viewConnectionLabelLayout } from './viewExplanationPresentation';

/** The pipeline relationships, as research/contracts.py names them. Shared with
 *  the layout, which straightens these and lets the others go slack. */
const FLOW_RELATIONSHIPS = new Set([
  'reads', 'transforms', 'feeds', 'trains', 'evaluates', 'produces', 'logs', 'derives_from',
]);

export class ModelEdgeShapeUtil extends BaseBoxShapeUtil<ModelEdgeShape> {
  static override type = 'model-edge' as const;

  static override props = {
    w: T.number,
    h: T.number,
    startX: T.number,
    startY: T.number,
    endX: T.number,
    endY: T.number,
    curveOffset: T.number.optional(),
    startSide: T.string.optional(),
    endSide: T.string.optional(),
    diffStatus: T.string.optional(),
    edgeId: T.string.optional(),
    sourceId: T.string.optional(),
    targetId: T.string.optional(),
    role: T.string.optional(),
    connectionCount: T.number.optional(),
    /** What the project document says this edge MEANS — `feeds`, `builds`, … .
     *  Absent on a per-class graph, whose edges are all the same kind of claim. */
    relationship: T.string.optional(),
  };

  getDefaultProps(): ModelEdgeShape['props'] {
    return {
      w: 1,
      h: 1,
      startX: 0,
      startY: 0,
      endX: 1,
      endY: 1,
    };
  }

  override canBind() {
    return false;
  }

  override canEdit() {
    return false;
  }

  override canResize() {
    return false;
  }

  // A connector's bounding rectangle would obscure the node overview.
  override hideInMinimap() {
    return true;
  }

  component(shape: ModelEdgeShape) {
    return <ModelConnection shape={shape} editor={this.editor} />;
  }

  getIndicatorPath() { return new Path2D(); }
}

const ModelConnection = track(function ModelConnection({ shape, editor }: { shape: ModelEdgeShape; editor: Editor }) {
    const { w, h, startX, startY, endX, endY, curveOffset = 0, startSide, endSide, diffStatus, edgeId, relationship, role, connectionCount = 1 } = shape.props;
    const exporting = useSvgExportContext();
    const explanations = React.useContext(NodeExplanationContext);
    const view = applicableViewExplanation(explanations?.model?.state, explanations?.annotationsVisible, graphNodes.get());
    const meaning = viewExplanationConnection(view, shape.props.sourceId, shape.props.targetId);
    const focus = graphFocus(editor).focus.get();
    const focused = !exporting && focus.active && focus.edgeIds.has(edgeId ?? '');
    // A focused edge that touches the selection is coloured by which way its
    // arrow points — incoming vs outgoing — because the arrowhead itself
    // disappears wherever cards overlap. Edges between two seeds, and traced
    // edges that never touch the seeds, keep the neutral neighborhood colour.
    const incoming = focused && focus.incomingEdgeIds.has(edgeId ?? '');
    const outgoing = focused && focus.outgoingEdgeIds.has(edgeId ?? '');
    const quiet = !exporting && focus.active && !focused && !diffStatus;
    // component() runs under useStateTracking, so reading the atom subscribes
    // this edge alone — a click restyles two edges, not the whole canvas.
    const isSelected = Boolean(edgeId) && selectedEdgeId.get() === edgeId;
    // Direction-aware, and shared with the code that sizes this shape's box —
    // see edgeCurve.ts. The previous unconditional horizontal control points made
    // a backward edge double back over itself.
    const { c1x, c1y, c2x, c2y } = edgeControlPoints(startX, startY, endX, endY, curveOffset, startSide, endSide);
    const path = `M ${startX} ${startY} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${endX} ${endY}`;
    const angle = Math.atan2(endY - c2y, endX - c2x) * 180 / Math.PI;
    const labelX = (startX + 3 * c1x + 3 * c2x + endX) / 8;
    const labelY = (startY + 3 * c1y + 3 * c2y + endY) / 8;
    const meaningLayout = meaning && viewConnectionLabelLayout(meaning.label, Math.abs(endX - startX) - 24);

    const diffClass = (diffStatus === 'added' ? ' diff-added' : diffStatus === 'removed' ? ' diff-removed' : diffStatus === 'modified' ? ' diff-modified' : '') + (isSelected ? ' selected' : '') + (focused ? ' neighborhood' : '') + (incoming ? ' in-edge' : outgoing ? ' out-edge' : '') + (quiet && !isSelected ? ' unrelated' : '') + (role === 'residual' ? ' residual' : '');
    // A project graph draws its pipeline and its call graph on one canvas. Drawn
    // alike they are a mesh; the dataflow is what a reader came for, so the rest
    // goes quiet rather than away — `builds` is still true and still clickable, it
    // just stops competing with the spine for attention. The layout makes the same
    // distinction with edge weight; see canvasLayout.edgeLayoutWeight.
    const flowClass = relationship && !FLOW_RELATIONSHIPS.has(relationship) ? ' model-edge-path-quiet' : '';

    return (
      <HTMLContainer style={{ pointerEvents: 'none', overflow: 'visible' }}>
        <svg
          width={w}
          height={h}
          viewBox={`0 0 ${w} ${h}`}
          className="model-edge-svg"
          data-edge-id={edgeId}
          data-role={role || 'ordinary'}
          data-focus={focused ? (incoming ? 'incoming' : outgoing ? 'outgoing' : focus.direction) : quiet ? 'unrelated' : 'none'}
          aria-hidden="true"
        >
          <path className={`model-edge-path${diffClass}${flowClass}`} d={path} />
          <circle className={`model-edge-terminal${diffClass}`} cx={startX} cy={startY} r="2" />
          <path className={`model-edge-direction${diffClass}`} d="M -11 -4 L -4 0 L -11 4" transform={`translate(${endX} ${endY}) rotate(${angle})`} />
          {(connectionCount > 1 || diffStatus) && <text className={`model-edge-label${diffClass}`} x={(startX + 3 * c1x + 3 * c2x + endX) / 8} y={(startY + 3 * c1y + 3 * c2y + endY) / 8 - 9} textAnchor="middle">
            {diffStatus === 'added' ? '+ ' : diffStatus === 'removed' ? '− ' : diffStatus === 'modified' ? '~ ' : ''}{connectionCount > 1 ? `${role === 'mixed' ? 'Mixed ' : ''}×${connectionCount}` : diffStatus}
          </text>}
          {meaning && meaningLayout && <g data-agent-connection="true" className={`model-edge-explanation${quiet && !isSelected ? ' unrelated' : ''}`}>
            <title>{meaning.label}{meaning.detail ? `. ${meaning.detail}` : ''}</title>
            {meaningLayout.lines.map((line, index) => <text key={index} x={labelX} y={labelY + 18 + index * 15} textAnchor="middle">{line}</text>)}
          </g>}
        </svg>
      </HTMLContainer>
    );
});
