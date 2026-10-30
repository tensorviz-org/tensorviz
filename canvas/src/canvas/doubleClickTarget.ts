/**
 * Which shape a double-click activates.
 *
 * tldraw's click detector reports a double-click with the target of the pointer
 * events it came from, and those arrive on the canvas: card bodies ignore the
 * pointer (pointerEvents:'none') and leave hit-testing to tldraw. The select tool
 * resolves the shape for itself, on a copy of the event, so the event an 'event'
 * listener receives never names it. Resolve it with the select tool's own
 * pointer-down hit test instead: that is the card the same gesture selected, the
 * card rather than the expanded module surface beneath it, and nothing on empty
 * canvas, where an earlier selection must not be activated.
 */
import { getHitShapeOnCanvasPointerDown, type Editor, type TLEventInfo, type TLShapeId } from 'tldraw';

/**
 * The shape a finished double-click landed on, or null. Only the 'up' phase
 * counts: 'down' and 'settle' belong to the same gesture. As in tldraw's select
 * tool, a Shift or Ctrl double-click toggles the selection instead.
 */
export function doubleClickedShapeId(editor: Editor, info: TLEventInfo): TLShapeId | null {
  if (info.type !== 'click' || info.name !== 'double_click' || info.phase !== 'up') return null;
  if (info.shiftKey || info.ctrlKey) return null;
  if (info.target === 'shape') return info.shape.id;
  return info.target === 'canvas' ? getHitShapeOnCanvasPointerDown(editor)?.id ?? null : null;
}
