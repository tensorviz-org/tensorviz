import { zoomToReadable } from '../../canvas/readableZoom';
import { ModelNode } from '../../types';

const node = (id: string, x: number): ModelNode => ({ id, type: 'layerNode', position: { x, y: 0 },
  data: { parameters: {}, layerType: { id: 'relu', name: 'ReLU' } as any } });
const editor = () => ({ getViewportPageBounds: () => ({ center: { x: 1000, y: 100 } }), zoomToBounds: jest.fn() });

it('zooms to the visible card nearest the camera without moving or changing the graph', () => {
  const e = editor(), nodes = [node('distant', 0), node('near', 900), { ...node('hidden', 950), hidden: true }];
  const before = JSON.stringify(nodes);
  zoomToReadable(e as any, nodes);
  expect(e.zoomToBounds).toHaveBeenCalledWith(expect.objectContaining({ id: 'near', x: 900 }), expect.objectContaining({ targetZoom: 1 }));
  expect(JSON.stringify(nodes)).toBe(before);
});

it('uses a selected nested node in SVG mode where there are no tldraw shapes', () => {
  const e = editor();
  const parent = { ...node('parent', 200), type: 'moduleNode', data: { parameters: {}, expanded: true } } as ModelNode;
  const child = { ...node('child', 40), parentNode: 'parent', selected: true };
  zoomToReadable(e as any, [parent, child, node('near', 900)]);
  expect(e.zoomToBounds).toHaveBeenCalledWith(expect.objectContaining({ id: 'child', x: 240 }), expect.objectContaining({ targetZoom: 1 }));
});

it('does not move the camera when no graph is available', () => {
  const e = editor();
  zoomToReadable(e as any, []);
  expect(e.zoomToBounds).not.toHaveBeenCalled();
});

it('targets a readable card rather than refitting an enormous expanded container', () => {
  const e = editor();
  const frame = { ...node('root', 0), type: 'moduleNode', style: { width: 2000, height: 2000 },
    data: { parameters: {}, expanded: true }, selected: true } as ModelNode;
  zoomToReadable(e as any, [frame, node('near', 900)]);
  expect(e.zoomToBounds).toHaveBeenCalledWith(expect.objectContaining({ id: 'near' }), expect.anything());
});
