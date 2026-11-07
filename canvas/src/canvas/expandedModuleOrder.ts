import { getIndicesBetween } from 'tldraw';
import type { TLParentId, TLShape, TLShapeId, TLShapePartial } from 'tldraw';

/**
 * Plan sequential singleton sendToBack calls without repeatedly materializing
 * every sibling. Call after edge reordering, inside the projection transaction.
 *
 * The caller supplies the existing depth-sorted module sequence. Actual tldraw
 * parents, not model ancestry, determine which siblings share an index space.
 * This is final-state equivalence for a stable store, not an event-by-event
 * replacement: intermediate edits (and partial progress on failure) are batched.
 */
export function planExpandedModuleOrder(
  editor: {
    getShape(id: TLShapeId): TLShape | undefined;
    getSortedChildIdsForParent(parentId: TLParentId): TLShapeId[];
  },
  ids: readonly TLShapeId[],
): TLShapePartial[] {
  const bottomByParent = new Map<TLParentId, Pick<TLShape, 'id' | 'index'> | undefined>();
  const updates = new Map<TLShapeId, TLShapePartial>();

  for (const id of ids) {
    const shape = editor.getShape(id);
    if (!shape) continue;

    if (!bottomByParent.has(shape.parentId)) {
      // Sorted child ids may contain stale records. Match tldraw's compact()
      // behavior, but stop once the first live sibling has been found.
      let bottom: TLShape | undefined;
      for (const childId of editor.getSortedChildIdsForParent(shape.parentId)) {
        bottom = editor.getShape(childId);
        if (bottom) break;
      }
      bottomByParent.set(shape.parentId, bottom);
    }
    const bottom = bottomByParent.get(shape.parentId);
    if (bottom?.id === id) continue;

    // Allocate ONE index per effective singleton move. Bulk allocation uses a
    // different index sequence. A repeated id may move again after another id.
    const index = getIndicesBetween(undefined, bottom?.index, 1)[0];
    updates.set(id, { id, type: shape.type, index });
    bottomByParent.set(shape.parentId, { id, index });
  }
  return Array.from(updates.values());
}
