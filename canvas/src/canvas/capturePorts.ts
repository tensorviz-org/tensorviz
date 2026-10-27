/** Shared pin/edge geometry. Semantic identities never depend on card layout. */
import type { CapturePort, ModelNodeData } from '../types';
import { AnchorBox, AnchorPoint, AnchorSide, anchorPoint, portAnchorPoint } from './edgeAnchors';

type PortData = Pick<ModelNodeData, 'capturePorts' | 'captureDisplayPorts'>;

export function displayedPorts(data: PortData | undefined): CapturePort[] {
  return data?.captureDisplayPorts ?? data?.capturePorts ?? [];
}

export function portCardHeight(data: PortData | undefined, base: number): number {
  const ports = displayedPorts(data);
  const count = Math.max(ports.filter(p => p.direction === 'in').length, ports.filter(p => p.direction === 'out').length);
  return Math.max(base, 48 + count * 16);
}

/** Where one edge endpoint lands on a card's side. When the handle names a
 *  captured port, the point is that port's evenly distributed slot along the
 *  side, so several ports sharing a side stay distinct; otherwise it is the
 *  middle of the side. Shared by the committed edge, the drag ghost and the
 *  rendered port dots so all three agree on the same point. */
export function endpointOnSide(
  data: PortData | undefined,
  handleId: string | null | undefined,
  box: AnchorBox,
  side: AnchorSide,
): AnchorPoint {
  const ports = displayedPorts(data);
  const port = ports.find(item => item.id === handleId);
  if (!port) return anchorPoint(box, side, 0.5);
  const sameDirection = ports.filter(item => item.direction === port.direction);
  return portAnchorPoint(box, side, sameDirection.findIndex(item => item.id === port.id), sameDirection.length);
}

export function portLabel(port: CapturePort, index: number): string {
  if (!port.paths.length) return `${port.direction === 'in' ? 'Input' : 'Output'} ${index + 1}`;
  return port.paths.map(path => path.map((part, i) => typeof part === 'number' ? `[${part}]` : `${i ? '.' : ''}${part}`).join('')).join(' / ');
}

export function portType(port: CapturePort): string {
  if (port.argumentKind !== 'value') return port.argumentKind;
  const shape = port.shape === null ? 'shape unknown' : `(${port.shape.join(', ')})`;
  return `${shape} · ${port.dtype?.replace(/^torch\./, '') ?? 'dtype unknown'}`;
}
