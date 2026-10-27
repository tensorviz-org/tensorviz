import type { TLShapeId } from 'tldraw';

// Graph documents store semantic identities, never tldraw records. Keep the
// ephemeral shape namespace lossless and separate nodes from generated edges.
export const nodeIdToShapeId = (nodeId: string): TLShapeId =>
  `shape:node:${encodeURIComponent(nodeId)}` as TLShapeId;

export const shapeIdToNodeId = (shapeId: string): string =>
  shapeId.startsWith('shape:node:')
    ? decodeURIComponent(shapeId.slice('shape:node:'.length))
    : shapeId.replace(/^shape:/, '');

export const edgeIdToShapeId = (edge: { id?: string; source: string; target: string }): TLShapeId =>
  `shape:edge:${encodeURIComponent(JSON.stringify([edge.id ?? null, edge.source, edge.target]))}` as TLShapeId;
