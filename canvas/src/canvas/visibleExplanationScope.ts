import type { ModelNode } from '../types';

export interface ExplanationBounds { x: number; y: number; w: number; h: number }

/** Snapshot what can actually be seen at click time. Camera movement never requests work. */
export function visibleExplanationNodeIds(
  nodes: readonly ModelNode[],
  viewport: ExplanationBounds,
  bounds: (id: string) => ExplanationBounds | undefined,
): string[] {
  if (viewport.w <= 0 || viewport.h <= 0) return [];
  return nodes.filter(node => {
    if (node.hidden) return false;
    const box = bounds(node.id);
    return box && box.w > 0 && box.h > 0
      && box.x < viewport.x + viewport.w && box.x + box.w > viewport.x
      && box.y < viewport.y + viewport.h && box.y + box.h > viewport.y;
  }).map(node => node.id);
}
