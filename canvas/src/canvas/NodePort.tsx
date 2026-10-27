/**
 * NodePort — a react-flow-style handle: an always-visible dot on a card's edge
 * that starts a connection.
 *
 * It must claim the gesture on pointerdown. editor.markEventAsHandled makes
 * .tl-canvas's own onPointerDown bail before it calls setPointerCapture and
 * before the state chart runs, so SelectTool never enters select.pointing_shape
 * and the 4px drag-to-translate threshold never fires — without it the card
 * moves instead of a connection starting.
 *
 * Always visible rather than revealed on hover: getHoveredShapeId() is reactive
 * and a shape's component() runs under useStateTracking, so reading hover state
 * inside a card would re-render every card on every hover.
 *
 * On the read-only local page a port is a topology fact (where an edge lands),
 * not an affordance: no title promise, no gesture claim, and the inert class
 * lets the theme strip pointer capture, crosshair, and hover growth.
 */
import React from 'react';
import { TLShape, TLShapeId, useEditor, useValue } from 'tldraw';
import { ConnectableNode, ConnectionSide, hasPort } from './connectivity';
import { connectableFromShape, startConnection, useConnectionGraph } from './connectionDrag';
import { localBrowserMode } from '../ide/projection';
import { graphIsReadOnly } from './graphAccess';
import { ModelNodeData } from '../types';
import { displayedPorts, portLabel, portType } from './capturePorts';
import { AnchorSide, portAnchorPoint } from './edgeAnchors';
import { capturedPortSide } from './edgeAnchors';
import { portSides } from './portLayout';
import { isFailingPort } from './graphIssues';
import { shapeIdToNodeId } from './canvasShapeId';

interface NodePortProps {
  shapeId: TLShapeId;
  node: ConnectableNode;
  side: ConnectionSide;
  /** Which of the card's four sides this dot sits on. */
  anchorSide: AnchorSide;
}

/** The side a card draws each direction on. A source leaves right or bottom, a
 *  target enters left or top, so the side an edge uses is also a dot the user can
 *  grab. */
const SIDES: Record<ConnectionSide, AnchorSide[]> = {
  target: ['left', 'top'],
  source: ['right', 'bottom'],
};

export const NodePort: React.FC<NodePortProps> = ({ shapeId, node, side, anchorSide }) => {
  const editor = useEditor();
  const graph = useConnectionGraph();

  if (localBrowserMode() || graphIsReadOnly(editor) || editor.getIsReadonly()) {
    return (
      <div
        className={`model-node-port model-node-port-${anchorSide} model-node-port-inert`}
        data-testid={`node-port-${side}`}
      />
    );
  }

  return (
    <div
      className={`model-node-port model-node-port-${anchorSide}`}
      data-testid={`node-port-${side}`}
      title={side === 'source' ? 'Drag to connect' : 'Drop a connection here'}
      onPointerDown={(event) => {
        editor.markEventAsHandled(event);
        event.stopPropagation();
        startConnection(editor, graph, {
          node,
          side,
          anchorSide,
          shapeId,
          origin: { x: event.clientX, y: event.clientY },
        });
      }}
    />
  );
};

/**
 * The ports one card renders. Which ones exist IS the direction rule: an Input
 * has no target port, an Output no source port, and a collapsed module frame
 * gets neither until it can name both its entry and exit children.
 *
 * A captured-computation card keeps one inert dot per tensor, placed on the side
 * that tensor's edge uses and distributed along it, so the dot stays where the
 * edge lands.
 */
export const NodePorts: React.FC<{ shape: TLShape }> = ({ shape }) => {
  const sides = useValue(portSides);
  const props = shape.props as { nodeData?: ModelNodeData; h?: number; w?: number };
  // A source assignment is not a step in the data flow: no handle to connect.
  if (props.nodeData?.sourceAssignment) return null;
  if (props.nodeData?.captureRef) {
    const data = props.nodeData;
    const nodeSides = sides.get(shapeIdToNodeId(shape.id));
    const box = {
      x: 0,
      y: 0,
      w: props.w ?? 200,
      h: props.h ?? 120,
    };
    return <>{displayedPorts(data).map(port => {
      const side = capturedPortSide(nodeSides, port.direction, port.id);
      const sameDirection = displayedPorts(data).filter(item => item.direction === port.direction);
      const index = sameDirection.findIndex(item => item.id === port.id);
      const point = portAnchorPoint(box, side, index, sameDirection.length);
      return <div key={port.id}
        className="model-node-port model-node-port-floating model-node-port-inert"
        data-testid="capture-port" data-port-id={port.id}
        data-failed-port={isFailingPort(data, port) || undefined}
        title={`${portLabel(port, index)}: ${portType(port)}`}
        style={{ left: point.x, top: point.y, transform: 'translate(-50%, -50%)', pointerEvents: 'auto', cursor: 'help',
          ...(port.argumentKind !== 'value' ? { borderRadius: 2, opacity: .65 } : {}) }}
      />;
    })}</>;
  }
  const node = connectableFromShape(shape);
  if (!node) return null;
  return (
    <>
      {hasPort(node, 'target') && SIDES.target.map(anchorSide => (
        <NodePort key={anchorSide} shapeId={shape.id} node={node} side="target" anchorSide={anchorSide} />
      ))}
      {hasPort(node, 'source') && SIDES.source.map(anchorSide => (
        <NodePort key={anchorSide} shapeId={shape.id} node={node} side="source" anchorSide={anchorSide} />
      ))}
    </>
  );
};
