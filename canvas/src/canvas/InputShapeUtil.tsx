import { shapeIdToNodeId } from './canvasShapeId';
import { BlockIdentity } from './BlockIdentity';
/**
 * InputShapeUtil — tldraw custom shape wrapping the existing InputNode UI.
 * Identical visual to the React Flow version, minus Handle components.
 */
import { BaseBoxShapeUtil, HTMLContainer, T, Editor, track } from 'tldraw';
import React from 'react';
import * as LucideIcons from 'lucide-react';
import { InputShape } from './types';
import { displayInputShape } from '../utils/inputShape';
import { inputProvenanceNote } from '../ide/cardParams';
import { NodePorts } from './NodePort';
import { useGraphDetail } from './graphDetail';
import { CaptureBoundaryRows } from './CaptureBoundaryRows';
import { GraphReviewBadge } from './GraphReviewChanges';

export class InputShapeUtil extends BaseBoxShapeUtil<InputShape> {
  static override type = 'input-node' as const;

  static override props = {
    w: T.number,
    h: T.number,
    nodeData: T.any,
  };

  getDefaultProps(): InputShape['props'] {
    return {
      w: 200,
      h: 100,
      nodeData: {
        layerType: {
          id: 'input',
          name: 'Input',
          displayName: 'Input',
          description: 'Input tensor',
          icon: 'ArrowRight',
          category: 'io',
          parameters: [],
        },
        parameters: { shape: '1,3,224,224' },
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

  component(shape: InputShape) {
    return <InputShapeCard shape={shape} editor={this.editor} />;
  }

  // The card outlines itself; see LayerShapeUtil.getIndicatorPath.
  getIndicatorPath() {
    return new Path2D();
  }
}

const InputShapeCard = track(function InputShapeCard({ shape, editor }: { shape: InputShape; editor: Editor }) {
    const data = shape.props.nodeData;
    const detail = useGraphDetail(editor);
    const isSelected = editor.getSelectedShapeIds().includes(shape.id);

    // Was `shapeStr.split(',')` with a catch that returned the raw value: given a list
    // it threw, fell through, and React rendered [1, 64, 48] as the string "16448" by
    // concatenating the array's children. The shared parser accepts either form.
    // Null when there is no shape to state — an absent value, an empty string, or
    // anything unparseable. The card then says "Not recorded", like every other card
    // on an untraced canvas, instead of asserting an ImageNet tensor nobody supplied.
    const formatShape = (raw: unknown) => displayInputShape(raw);

    const nodeClass = `input-node ${isSelected ? 'selected' : ''}${data.diffStatus ? ` diff-${data.diffStatus}` : ''}`;

    // Every shape downstream of this card was measured on this one tensor. When the
    // backend picked it rather than the user, the card says so — otherwise a search
    // result reads exactly like a `# tensorviz: input=` the user wrote.
    const provenanceNote = inputProvenanceNote(data.parameters);

    return (
      <HTMLContainer style={{ pointerEvents: 'none', overflow: 'visible' }}>
        <div className={`${nodeClass} ui2-node-card`} data-detail={detail} data-node-id={shapeIdToNodeId(shape.id)} data-block="input" style={{ width: shape.props.w, height: shape.props.h }}>
          <div className="ui2-node-header">
            <BlockIdentity block={{ kind: 'input', label: 'Input' }} />
            <div className="ui2-node-title-wrap">
              <div className="ui2-node-title">Input</div>
              <div className="ui2-node-subtitle">{data.captureLabel ?? `#${shapeIdToNodeId(shape.id)}`}</div>
            </div>
            {data.review ? <GraphReviewBadge review={data.review} /> : data.diffStatus && <span className="ui2-node-badge">{data.diffStatus}</span>}
          </div>

          <div className="ui2-node-body">
            {data.capturePorts ? <CaptureBoundaryRows ports={data.capturePorts} /> : <div className="ui2-node-list">
              <div className="ui2-node-shape-row">
                <LucideIcons.Layers size={11} />
                <span>Shape</span>
                <span className="ui2-node-shape">{formatShape(data.parameters.shape ?? data.inputShape) ?? 'Not recorded'}</span>
              </div>
              {data.parameters.name && (
                <div className="ui2-node-shape-row">
                  <LucideIcons.Tag size={12} />
                  <span>Name</span>
                  <span className="ui2-node-shape">{data.parameters.name}</span>
                </div>
              )}
              {provenanceNote && (
                <div
                  className="ui2-node-shape-row"
                  data-testid="input-provenance"
                  style={{ color: 'var(--ui2-orange)' }}
                >
                  <LucideIcons.HelpCircle size={11} />
                  <span>Source</span>
                  <span className="ui2-node-shape" style={{ color: 'var(--ui2-orange)' }}>
                    {provenanceNote.label}
                  </span>
                </div>
              )}
            </div>}


          </div>
        </div>
        <NodePorts shape={shape} />
      </HTMLContainer>
    );
});
