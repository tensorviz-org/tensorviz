const MODEL_SHAPE_TYPES = new Set(['input-node', 'output-node', 'module-node', 'layer']);

export const EDGE_SHAPE_TYPE = 'model-edge';

export const isModelShape = (shape: { type: string } | undefined | null): boolean =>
  Boolean(shape && MODEL_SHAPE_TYPES.has(shape.type));

export const isManagedCanvasShape = (shape: { type: string } | undefined | null): boolean =>
  isModelShape(shape) ||
  shape?.type === EDGE_SHAPE_TYPE ||
  shape?.type === 'arrow';
