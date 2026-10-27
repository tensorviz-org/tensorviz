/**
 * Which node has its parameters open for editing.
 *
 * Parameter editing used to live in the Inspector: a 320px panel docked to the
 * left edge that reserved canvas width whether or not anything was selected, and
 * that opened on the pointer-DOWN that begins a drag — so moving a card put a
 * panel over the canvas you were moving it on. It is gone. The one thing it held
 * that exists nowhere else, the catalogue-driven parameter fields, now opens ON
 * the node: a popover anchored to the card you clicked, in the same
 * `InFrontOfTheCanvas` slot and with the same projection rules as the
 * edge-insert picker.
 *
 * An atom rather than React state for the same reason `edgeInsert` uses one: the
 * affordance is set from a tldraw event handler inside the editor tree and read
 * by a component mounted in one of tldraw's own slots, with no common ancestor
 * that owns state.
 */
import { atom } from 'tldraw';

/** The tldraw shape id whose parameters are open, or null. */
export const parameterTarget = atom<string | null>('nodeParameterTarget', null);

/** Open the editor on `shapeId`. Only a drag-free click on an editable canvas calls this. */
export function openNodeParameters(shapeId: string): void {
  parameterTarget.set(shapeId);
}

export function closeNodeParameters(): void {
  parameterTarget.set(null);
}
