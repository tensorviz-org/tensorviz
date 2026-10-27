/**
 * Edge anchors. Live graphs use stable layout sides (computePortSides below).
 * The position-based helpers remain for loose connection previews and
 * thumbnails that do not carry the graph's hierarchy.
 *
 * A card used to wire out of its right edge and into its left edge regardless of
 * where the other card sat, so a vertical or backward connection had to double
 * back to reach a side it was nowhere near and the reader could not tell where
 * the edge was going. Instead each endpoint picks one of the card's four sides:
 *
 *   - a source edge leaves `right` or `bottom`;
 *   - a target edge enters `left` or `top`.
 *
 * The vertical side is chosen only when the other end is on the dominant axis on
 * the side that matches the flow (a source picks `bottom` when its target is
 * below; a target picks `top` when its source is above). That keeps the chosen
 * side facing the other card, and — because it is the same dominant-axis rule
 * `edgeCurve` already uses for its control points — the curve still leaves and
 * arrives perpendicular to the side. A backward, level edge keeps the horizontal
 * pair and bows, exactly as it did before.
 *
 * Deliberately free of tldraw imports so the rule stays unit-testable.
 */

export type AnchorSide = 'left' | 'right' | 'top' | 'bottom';
export type PortDirection = 'in' | 'out';

export interface AnchorBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface AnchorPoint {
  x: number;
  y: number;
}

export const boxCenter = (box: AnchorBox): AnchorPoint => ({
  x: box.x + box.w / 2,
  y: box.y + box.h / 2,
});

/** The side a source edge leaves: `bottom` only when the target is below on the
 *  dominant axis, otherwise `right`. */
export function sourceSide(box: AnchorBox, toward: AnchorPoint): 'right' | 'bottom' {
  const center = boxCenter(box);
  const dx = toward.x - center.x;
  const dy = toward.y - center.y;
  if (Math.abs(dy) > Math.abs(dx) && dy > 0) return 'bottom';
  return 'right';
}

/** The side a target edge enters: `top` only when the source is above on the
 *  dominant axis, otherwise `left`. */
export function targetSide(box: AnchorBox, toward: AnchorPoint): 'left' | 'top' {
  const center = boxCenter(box);
  const dx = toward.x - center.x;
  const dy = toward.y - center.y;
  if (Math.abs(dy) > Math.abs(dx) && dy < 0) return 'top';
  return 'left';
}

export function sideForDirection(
  direction: PortDirection,
  box: AnchorBox,
  toward: AnchorPoint,
): AnchorSide {
  return direction === 'out' ? sourceSide(box, toward) : targetSide(box, toward);
}

/** A point on a side, `fraction` of the way along it (0 = the first corner). */
export function anchorPoint(box: AnchorBox, side: AnchorSide, fraction = 0.5): AnchorPoint {
  switch (side) {
    case 'left':
      return { x: box.x, y: box.y + box.h * fraction };
    case 'right':
      return { x: box.x + box.w, y: box.y + box.h * fraction };
    case 'top':
      return { x: box.x + box.w * fraction, y: box.y };
    default:
      return { x: box.x + box.w * fraction, y: box.y + box.h };
  }
}

/** Evenly spaced positions for `count` ports along one side, corners left clear.
 *  A lone port sits at the centre. */
export function portFraction(index: number, count: number): number {
  if (count <= 1) return 0.5;
  return (index + 1) / (count + 1);
}

export function portAnchorPoint(
  box: AnchorBox,
  side: AnchorSide,
  index: number,
  count: number,
): AnchorPoint {
  return anchorPoint(box, side, portFraction(index, count));
}

/* ── Stable sides for laid-out cards ────────────────────────────────────── */

export interface NodePortSides {
  in: AnchorSide;
  out: AnchorSide;
  /** A named tensor used by a skip connection keeps one side for every fan-out. */
  ports?: Record<string, AnchorSide>;
}

/** A chosen canvas orientation. Absent means the legacy mix: a left-to-right
 * outer graph with module contents stacked top to bottom. */
export type LayoutDirection = 'horizontal' | 'vertical';

/** Stable sides follow the layout direction, never card positions, so dragging
 * never flips ports. A module focus projection's own direction wins, because it
 * placed those cards. Without a direction the outer graph runs left to right
 * while module contents run top to bottom. */
export function computePortSides(
  nodes: readonly { id: string; parentNode?: string; data?: { parentId?: string; focusDirection?: 'LR' | 'TB' } }[],
  direction?: LayoutDirection,
): Map<string, NodePortSides> {
  return new Map(nodes.map(node => [node.id,
    (node.data?.focusDirection ? node.data.focusDirection === 'TB'
      : direction ? direction === 'vertical' : Boolean(node.parentNode || node.data?.parentId))
      ? { in: 'top', out: 'bottom' } : { in: 'left', out: 'right' }]));
}

/** Skip connections inside frames run beside the flow: right of a column, below a row. */
export const skipLaneSide = (direction?: LayoutDirection): AnchorSide =>
  direction === 'horizontal' ? 'bottom' : 'right';

export function capturedPortSide(sides: NodePortSides | undefined, direction: PortDirection, handle?: string | null): AnchorSide {
  return (handle && sides?.ports?.[handle]) || sides?.[direction] || (direction === 'in' ? 'left' : 'right');
}
