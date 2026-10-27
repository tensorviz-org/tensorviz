/**
 * The rule behind the tab-close warning, in one importable place.
 *
 * A parameter edit reaches the server about 4s after it is made (autosave
 * debounce plus round trip). Closing the tab inside that window used to destroy
 * the edit silently: no dialog, no message, and nothing afterwards to say it had
 * happened — while reload and in-app navigation were both covered by the local
 * draft queue. That asymmetry is what made it dangerous, because the app had
 * taught users the "Saved" chip could be trusted.
 *
 * It lives here rather than inline in ModelBuilder so the rule can be tested on
 * what it ANSWERS. The component cannot be mounted under this jest transform, so
 * an inline handler could only ever have been pinned by reading its source text.
 */

/** The part of the save queue this rule needs. */
export interface UncommittedWorkSource {
  /** An edit in flight, queued behind one, or parked after a conflict. */
  hasUncommittedWork(): boolean;
  /** The canvas differs from the last snapshot the server accepted. */
  needsSave(request: unknown): boolean;
}

/**
 * Apply the unload rule to an event.
 *
 * Returns true when the browser should prompt. Setting both `preventDefault`
 * and a string `returnValue` is what browsers actually honour; the wording of
 * the dialog is the browser's and cannot be set by the page.
 */
export function warnIfUnsaved(
  event: Pick<BeforeUnloadEvent, 'preventDefault' | 'returnValue'>,
  queue: UncommittedWorkSource | null | undefined,
  currentRequest?: unknown
): boolean {
  // No queue means no editor mounted, so there is nothing of the user's to lose.
  // Never prompt on a clean exit: a warning users learn to dismiss is a warning
  // they will dismiss on the one occasion it mattered.
  if (!queue) return false;

  // BOTH questions have to be asked, and asking only the first was the whole
  // reason the first attempt at this fix did nothing. The queue can only see an
  // edit once the autosave debounce has handed it over, and that handover is the
  // SMALLER half of the window: measured on the live app, closing the tab 300ms
  // after a layer insert found hasUncommittedWork() false while the edit was
  // still sitting in React state behind the debounce timer — exactly the moment
  // the loss was worst, and exactly the moment the old guard stayed silent.
  //
  // needsSave() is the same dirty check the autosave effect itself uses, so the
  // guard and the thing it is guarding cannot disagree about what "unsaved"
  // means.
  const dirty = currentRequest !== undefined && queue.needsSave(currentRequest);
  if (!dirty && !queue.hasUncommittedWork()) return false;
  event.preventDefault();
  // eslint-disable-next-line no-param-reassign
  event.returnValue = '';
  return true;
}
