/** Source/architecture access is separate from tldraw's layout lock. */
const readOnlyViews = new WeakMap<object, { readonly current: boolean }>();
export function registerGraphAccess(editor: object, readOnly: { readonly current: boolean }): void {
  readOnlyViews.set(editor, readOnly);
}
export function graphIsReadOnly(editor: object): boolean {
  return readOnlyViews.get(editor)?.current ?? false;
}
