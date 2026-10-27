import React from 'react';
import { Box, useEditor, useValue } from 'tldraw';
import type { ModelNode } from '../types';
import { nodeIdToShapeId } from './canvasShapeId';
import { SourceEditSession } from '../ide/SourceEditSession';
import { cameraDuration } from './reducedMotion';
import type { LayerShape, ModuleShape } from './types';

/** A temporary expanded node body. The session survives same-model reprojection. */
export function InlineSourceEditor({ nodes, onClose }: { nodes: ModelNode[]; onClose(): void }) {
  const editor = useEditor();
  const shapeId = nodeIdToShapeId(nodes[0].id);
  const lastBounds = React.useRef(editor.getShapePageBounds(shapeId));
  const original = React.useRef(editor.getShape(shapeId));
  const viewportSize = useValue('inline source editor viewport', () => {
    const bounds = editor.getViewportScreenBounds();
    return `${bounds.w}:${bounds.h}`;
  }, [editor]);
  React.useLayoutEffect(() => {
    const shape = original.current;
    if (!shape || (shape.type !== 'layer' && shape.type !== 'module-node')) return;
    const expanded = shape.props.nodeData.expanded;
    if (!expanded) editor.updateShape<LayerShape | ModuleShape>({ id: shapeId, type: shape.type, props: { w: 340, h: 430 } });
    return () => {
      const current = editor.getShape(shapeId);
      if (!expanded && current?.type === shape.type && current.props.w === 340 && current.props.h === 430) {
        editor.updateShape<LayerShape | ModuleShape>({ id: shapeId, type: shape.type, props: { w: shape.props.w, h: shape.props.h } });
      }
    };
  }, [editor, shapeId]);
  React.useLayoutEffect(() => {
    const bounds = editor.getShapePageBounds(shapeId) ?? lastBounds.current;
    if (bounds) editor.zoomToBounds(new Box(bounds.x, bounds.y, 340, 430), { targetZoom: 1, inset: 32, animation: { duration: cameraDuration(160) } });
  }, [editor, shapeId, viewportSize]);
  const placement = useValue('inline source editor position', () => {
    const bounds = editor.getShapePageBounds(shapeId);
    if (bounds) lastBounds.current = bounds;
    const point = lastBounds.current;
    if (!point) return null;
    const screen = editor.pageToViewport(point);
    return { x: screen.x, y: screen.y, zoom: editor.getZoomLevel() };
  }, [editor, shapeId]);
  if (!placement) return null;
  return <div className="inline-source-editor" style={{ left: placement.x, top: placement.y,
    transform: `scale(${placement.zoom})` }}>
    <SourceEditSession nodes={nodes} onClose={onClose} />
  </div>;
}
