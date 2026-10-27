/**
 * Tldraw shape types for the tensorViz canvas.
 *
 * Each custom shape wraps our existing React node components and stores
 * the same ModelNodeData payload so the rest of the app never sees the
 * difference between old React Flow nodes and new tldraw shapes.
 */
import { TLBaseShape } from 'tldraw';
import { ModelNodeData } from '../types';
import type { AnchorSide } from './edgeAnchors';

/* ── Shape prop types ────────────────────────────────────────── */

export type LayerShapeProps = {
  w: number;
  h: number;
  nodeData: ModelNodeData;
};

export type InputShapeProps = {
  w: number;
  h: number;
  nodeData: ModelNodeData;
};

export type OutputShapeProps = {
  w: number;
  h: number;
  nodeData: ModelNodeData;
};

export type ModuleShapeProps = {
  w: number;
  h: number;
  nodeData: ModelNodeData;
};

export type ModelEdgeShapeProps = {
  w: number;
  h: number;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  curveOffset?: number;
  startSide?: AnchorSide;
  endSide?: AnchorSide;
  diffStatus?: 'added' | 'removed' | 'modified';
  /** ModelEdge.id, so a click can name the edge it hit. The shape's other props
   *  are pure geometry and cannot identify the edge they draw. */
  edgeId?: string;
  sourceId?: string;
  targetId?: string;
  role?: 'residual' | 'mixed';
  connectionCount?: number;
  /** What the project document says this edge means — `feeds`, `builds`, … .
   *  Absent on a per-class graph, whose edges are all the same kind of claim. */
  relationship?: string;
};

/* ── Module augmentation to register custom shapes with tldraw ── */

declare module '@tldraw/tlschema' {
  interface TLGlobalShapePropsMap {
    'layer': LayerShapeProps;
    'input-node': InputShapeProps;
    'output-node': OutputShapeProps;
    'module-node': ModuleShapeProps;
    'model-edge': ModelEdgeShapeProps;
  }
}

/* ── TLBaseShape aliases ─────────────────────────────────────── */

export type LayerShape = TLBaseShape<'layer', LayerShapeProps>;
export type InputShape = TLBaseShape<'input-node', InputShapeProps>;
export type OutputShape = TLBaseShape<'output-node', OutputShapeProps>;
export type ModuleShape = TLBaseShape<'module-node', ModuleShapeProps>;
export type ModelEdgeShape = TLBaseShape<'model-edge', ModelEdgeShapeProps>;
