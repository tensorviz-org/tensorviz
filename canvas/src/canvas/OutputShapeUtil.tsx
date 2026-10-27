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
import { useGraphDetail } from './graphDetail';
import { CaptureBoundaryRows } from './CaptureBoundaryRows';
import { GraphReviewBadge } from './GraphReviewChanges';

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
      h: 100,
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
    const detail = useGraphDetail(editor);
    const isSelected = editor.getSelectedShapeIds().includes(shape.id);

    const formatShape = (s?: number[]) => {
      if (!s) return '';
      return `(${s.join(', ')})`;
    };

    const nodeClass = `output-node ${isSelected ? 'selected' : ''}${data.diffStatus ? ` diff-${data.diffStatus}` : ''}`;

    return (
      <HTMLContainer style={{ pointerEvents: 'none', overflow: 'visible' }}>
        <div className={`${nodeClass} ui2-node-card`} data-detail={detail} data-node-id={shapeIdToNodeId(shape.id)} data-block="output" style={{ width: shape.props.w, height: shape.props.h }}>
          <div className="ui2-node-header">
            <BlockIdentity block={{ kind: 'output', label: 'Output' }} />
            <div className="ui2-node-title-wrap">
              <div className="ui2-node-title">Output</div>
              <div className="ui2-node-subtitle">{data.captureLabel ?? `#${shapeIdToNodeId(shape.id)}`}</div>
            </div>
            {data.review ? <GraphReviewBadge review={data.review} /> : data.diffStatus && <span className="ui2-node-badge">{data.diffStatus}</span>}
          </div>

          <div className="ui2-node-body">
            {data.capturePorts ? <CaptureBoundaryRows ports={data.capturePorts} /> : <div className="ui2-node-list">
              {data.outputShape && (
                <div className="ui2-node-shape-row">
                  <LucideIcons.Layers size={11} />
                  <span>Shape</span>
                  <span className="ui2-node-shape">{formatShape(data.outputShape)}</span>
                </div>
              )}
              {data.parameters.name && (
                <div className="ui2-node-shape-row">
                  <LucideIcons.Tag size={12} />
                  <span>Name</span>
                  <span className="ui2-node-shape">{data.parameters.name}</span>
                </div>
              )}
            </div>}

            {data.error && (
              <div className="ui2-node-error">
                <LucideIcons.AlertTriangle size={12} />
                <span>{data.error}</span>
              </div>
            )}


          </div>
        </div>
        <NodePorts shape={shape} />
      </HTMLContainer>
    );
});
