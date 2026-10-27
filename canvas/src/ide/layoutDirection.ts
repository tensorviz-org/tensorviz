import type { LayoutDirection } from '../canvas/edgeAnchors';

/** What the Layout control stores. Auto follows the canvas shape. */
export type LayoutChoice = 'auto' | LayoutDirection;

export const isLayoutChoice = (value: unknown): value is LayoutChoice =>
  value === 'auto' || value === 'horizontal' || value === 'vertical';

/** Auto's direction for a canvas of this size. Once resolved, a direction is
 * kept inside a band around square so dragging a divider near the boundary
 * cannot flip the whole layout back and forth. Zero size keeps `previous`. */
export function autoLayoutDirection(width: number, height: number, previous?: LayoutDirection): LayoutDirection | undefined {
  if (!(width > 0) || !(height > 0)) return previous;
  const ratio = width / height;
  if (previous === 'vertical') return ratio >= 1.15 ? 'horizontal' : 'vertical';
  if (previous === 'horizontal') return ratio <= 0.87 ? 'vertical' : 'horizontal';
  return ratio >= 1 ? 'horizontal' : 'vertical';
}
