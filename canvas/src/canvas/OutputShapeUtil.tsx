import { shapeIdToNodeId } from './canvasShapeId';
import { BlockIdentity } from './BlockIdentity';
/**
 * OutputShapeUtil — tldraw custom shape wrapping the existing OutputNode UI.
 * Identical visual to the React Flow version, minus Handle components.
 */
import { BaseBoxShapeUtil, HTMLContainer, T, Editor, track } from 'tldraw';
import React from 'react';
import * as LucideIcons from 'lucide-react';
import { OutputShape } from './types';
import { NodePorts } from './NodePort';
import { CaptureBoundaryRows } from './CaptureBoundaryRows';
import { GraphReviewBadge } from './GraphReviewChanges';
import { cardHeaderLayout } from './cardIdentity';
import { CardIdentity } from './NodeCardIdentity';
import { ioCardContent } from './cardMetrics';

export class OutputShapeUtil extends BaseBoxShapeUtil<OutputShape> {
  static override type = 'output-node' as const;

  static override props = {
    w: T.number,
    h: T.number,
    nodeData: T.any,
  };

  getDefaultProps(): OutputShape['props'] {
    return {
      w: 200,
      h: cardHeaderLayout('Output').height + 2,
      nodeData: {
        layerType: {
          id: 'output',
          name: 'Output',
          displayName: 'Output',
          description: 'Output tensor',
          icon: 'ArrowLeft',
          category: 'io',
          parameters: [],
        },
        parameters: {},
      },
    };
  }

  // Like layer and module cards: the card keeps its measured size and outlines
  // itself, so tldraw's blue selection box and handles stay off.
  override canResize() {
    return false;
  }

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

  component(shape: OutputShape) {
    return <OutputShapeCard shape={shape} editor={this.editor} />;
  }

  // The card outlines itself; see LayerShapeUtil.getIndicatorPath.
  getIndicatorPath() {
    return new Path2D();
  }
}

const OutputShapeCard = track(function OutputShapeCard({ shape, editor }: { shape: OutputShape; editor: Editor }) {
    const data = shape.props.nodeData;
    const isSelected = editor.getSelectedShapeIds().includes(shape.id);
    const header = cardHeaderLayout('Output', data.captureLabel, shape.props.w);
    const content = ioCardContent(data, 'output');

    const formatShape = (s?: number[]) => {
      if (!s) return '';
      return `(${s.join(', ')})`;
    };

    const nodeClass = `output-node ${isSelected ? 'selected' : ''}${data.diffStatus ? ` diff-${data.diffStatus}` : ''}`;

    return (
      <HTMLContainer style={{ pointerEvents: 'none', overflow: 'visible' }}>
        <div className={`${nodeClass} ui2-node-card`} data-detail="compact" data-node-id={shapeIdToNodeId(shape.id)} data-block="output" style={{ width: shape.props.w, height: shape.props.h }}>
          <div className="ui2-node-header" style={{ height: header.height }}>
            <BlockIdentity block={{ kind: 'output', label: 'Output' }} />
            <CardIdentity layout={header} title="Output" subtitle={data.captureLabel} />
            {data.review ? <GraphReviewBadge review={data.review} /> : data.diffStatus && <span className="ui2-node-badge">{data.diffStatus}</span>}
          </div>

          {content.bodyHeight > 0 && <div className="ui2-node-body">
            {data.capturePorts ? <CaptureBoundaryRows ports={data.capturePorts} /> : content.rowCount > 0 && <div className="ui2-node-list">
              {data.outputShape && (
                <div className="ui2-node-shape-row">
                  <LucideIcons.Layers size={11} />
                  <span>Shape</span>
                  <span className="ui2-node-shape">{formatShape(data.outputShape)}</span>
                </div>
              )}
              {content.name && (
                <div className="ui2-node-shape-row">
                  <LucideIcons.Tag size={12} />
                  <span>Name</span>
                  <span className="ui2-node-shape">{content.name}</span>
                </div>
              )}
            </div>}

            {content.errorLines.length > 0 && (
              <div className="ui2-node-error" style={{ height: content.errorHeight, boxSizing: 'border-box' }}>
                <LucideIcons.AlertTriangle size={12} />
                <span className="graph-failure-facts">{content.errorLines.map((line, index) => <span key={index}>{line}</span>)}</span>
              </div>
            )}


          </div>}
        </div>
        <NodePorts shape={shape} />
      </HTMLContainer>
    );
});
