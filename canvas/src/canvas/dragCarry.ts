/**
 * When a module frame moves, its children must move with it — tldraw applies no
 * parent transform to these shapes (they are flat children of the page and their
 * positions are absolute), so the frame would otherwise slide off its contents.
 *
 * But that carry is only correct for a USER gesture. The bridge's sync effects
 * also write shape geometry, and those writes are already absolute and already
 * carry each child's own new position — `absolutePositionOf` resolves
 * module-relative → absolute in the same pass. Carrying children again on top of
 * a programmatic write adds the frame's own displacement to children that were
 * already correct.
 *
 * That is precisely what pushed an expanded module's layers out of its frame:
 * expanding a module moves its frame during the relayout, so every child was
 * displaced by that frame delta (twice, in practice, because the frame's geometry
 * is written by both sync effects). Measured on F5-TTS:
 *
 *   ConvNeXt-V2  frame y 922 → 137 (Δ −785); first child landed at −1361
 *                instead of 209 — exactly 2 × 785 too high.
 *   Conv1D       frame y 1396 → 1225 (Δ −171); first child landed 341px
 *                too high (≈ 2 × 171).
 *
 * X was unaffected in both cases only because neither frame moved horizontally.
 * The next relayout recomputed child positions from the graph and snapped them
 * back inside the frame, which is why the breakage looked intermittent — it
 * survived only until the next expand/collapse.
 */

export interface DragCarryInput {
  /** True while a bridge sync effect is writing shape geometry. */
  isSyncingToEditor: boolean;
  /** tldraw shape type of the shape that changed. */
  shapeType: string;
  dx: number;
  dy: number;
}

/**
 * Whether an observed shape geometry change should drag a module's children along.
 * True only for a real user move of a module frame.
 */
export function shouldCarryModuleChildren({
  isSyncingToEditor,
  shapeType,
  dx,
  dy,
}: DragCarryInput): boolean {
  // A programmatic write already positioned the children correctly.
  if (isSyncingToEditor) return false;
  if (shapeType !== 'module-node') return false;
  return dx !== 0 || dy !== 0;
}
