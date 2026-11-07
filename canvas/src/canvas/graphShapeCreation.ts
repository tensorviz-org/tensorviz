import { Editor, ShapeUtil, TLCreateShapePartial, TLShape } from 'tldraw';

// In tldraw 5.0.1 this public base method always returns false. Capture its
// identity so later prototype overrides also take the ordinary discovery path.
const cannotReceiveChildren = ShapeUtil.prototype.canReceiveNewChildrenOfType;

const usesDefaultChildRejection = (util: ShapeUtil): boolean => {
  // Reading the method directly could invoke a custom getter before tldraw
  // reaches that shape. An accessor is an override too: fall back without
  // invoking it, even if it would return the default method.
  for (let owner: object | null = util; owner; owner = Object.getPrototypeOf(owner)) {
    const descriptor = Object.getOwnPropertyDescriptor(owner, 'canReceiveNewChildrenOfType');
    if (descriptor) return 'value' in descriptor && descriptor.value === cannotReceiveChildren;
  }
  return false;
};

const isPlainGraphPartial = (shape: unknown): boolean => {
  if (!shape || typeof shape !== 'object') return false;
  const prototype = Object.getPrototypeOf(shape);
  if (prototype !== Object.prototype && prototype !== null) return false;
  // Preflight and the eventual spread must not invoke input getters earlier
  // than createShapes would. Bridge partials are ordinary data-only literals.
  for (const key of Reflect.ownKeys(shape)) {
    const descriptor = Object.getOwnPropertyDescriptor(shape, key);
    if (!descriptor || !descriptor.enumerable || !('value' in descriptor)) return false;
  }
  if ('parentId' in shape) return false;
  const own = (key: string) => Object.getOwnPropertyDescriptor(shape, key)?.value;
  return typeof own('id') === 'string' && typeof own('type') === 'string'
    && Number.isFinite(own('x')) && Number.isFinite(own('y'));
};

const isPlainGraphBatch = (shapes: unknown[]): boolean => {
  // Restrict the optimization to the dense, ordinary arrays the bridge builds.
  // Index getters, holes, custom methods and array subclasses keep their native
  // access order and behavior in createShapes.
  if (Object.getPrototypeOf(shapes) !== Array.prototype
    || Reflect.ownKeys(shapes).length !== shapes.length + 1) return false;
  for (let i = 0; i < shapes.length; i++) {
    const descriptor = Object.getOwnPropertyDescriptor(shapes, i);
    if (!descriptor || !('value' in descriptor) || !isPlainGraphPartial(descriptor.value)) return false;
  }
  return true;
};

/**
 * Create the bridge's positioned, page-space graph shapes without repeating
 * tldraw's parent search when every existing shape is known to reject children.
 * Call inside the projection transaction, after deletions and immediately before
 * creation. Never retain the result of this check across batches.
 */
export function createGraphShapes<TShape extends TLShape>(
  editor: Editor,
  shapes: TLCreateShapePartial<TShape>[],
): void {
  // Keep tldraw authoritative for invalid inputs, readonly no-ops and the page
  // limit (including its alert). None of these need a parent search.
  if (!Array.isArray(shapes) || shapes.length === 0 || editor.getIsReadonly()
    || !editor.canCreateShapes(shapes)) {
    editor.createShapes(shapes);
    return;
  }

  const pageId = editor.getCurrentPageId();
  // Explicit parents, unpositioned or malformed partials, and focused groups
  // keep the original batch unchanged, including tldraw's transforms/errors.
  if (editor.getFocusedGroupId() !== pageId || !isPlainGraphBatch(shapes)) {
    editor.createShapes(shapes);
    return;
  }

  // Use the same snapshot tldraw's createShapes uses, including descendants and
  // hidden/offscreen/locked shapes. Do not invoke custom predicates or visibility
  // callbacks to decide whether an override happens to reject these shapes.
  for (const shape of editor.getCurrentPageShapesSorted()) {
    if (!usesDefaultChildRejection(editor.getShapeUtil(shape))) {
      editor.createShapes(shapes);
      return;
    }
  }

  // The default predicate rejects every possible parent, so tldraw would choose
  // this page without a coordinate transform. Leave batching, order, indices,
  // defaults, hooks and store writes entirely to the original API.
  editor.createShapes(shapes.map(shape => ({ ...shape, parentId: pageId })));
}
