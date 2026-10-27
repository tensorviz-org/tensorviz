import type { Editor } from 'tldraw';
import type { ModelNode } from '../types';
import { graphSvgBounds } from '../services/graphSvg';
import { cameraDuration } from './reducedMotion';

/** Use logical bounds too: an SVG overview has no selectable tldraw shapes. */
export function zoomToReadable(editor: Editor, nodes: ModelNode[]) {
  const visible = graphSvgBounds(nodes);
  const frames = new Set(nodes.filter(node => node.type === 'moduleNode' && node.data.expanded).map(node => node.id));
  const cards = visible.filter(box => !frames.has(box.id));
  const boxes = cards.length ? cards : visible;
  if (!boxes.length) return;
  const selected = new Set(nodes.filter(node => node.selected || node.data.isSelected).map(node => node.id));
  const center = editor.getViewportPageBounds().center;
  const distance = (box: typeof boxes[number]) => Math.hypot(box.x + box.w / 2 - center.x, box.y + box.h / 2 - center.y);
  const box = boxes.find(candidate => selected.has(candidate.id))
    ?? boxes.reduce((nearest, candidate) => distance(candidate) < distance(nearest) ? candidate : nearest);
  editor.zoomToBounds(box, { targetZoom: 1, inset: 80, animation: { duration: cameraDuration(220) } });
}
