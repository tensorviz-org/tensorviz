/**
 * Which edge the user clicked.
 *
 * An edge is unselectable by construction — created with isLocked, rendered with
 * pointerEvents:'none' and an empty indicator path — so it can never enter
 * tldraw's own selection. Deliberately: giving edges a wide pointer-events hit
 * band would swallow marquee-select, drag-to-pan and click-empty-to-deselect.
 * The selection therefore lives in an atom instead, read inside
 * ModelEdgeShapeUtil.component() — which runs under useStateTracking, so only
 * the two affected edges re-render — and by ModelBuilder's Delete handler.
 */
import { Editor, atom } from 'tldraw';
import { EDGE_SHAPE_TYPE } from './canvasShapePolicy';
import { distanceToEdgeCurve, edgeCurveMidpoint } from './edgeCurve';

/** ModelEdge.id of the clicked edge, or null. */
export const selectedEdgeId = atom<string | null>('selectedEdgeId', null);

/** Screen-space click slop, mirroring react-flow's interactionWidth = 20. */
export const EDGE_HIT_PX = 10;

/**
 * The edge whose STROKE is nearest `page`, within `tolerance` page units, or
 * null. Distance-based, not box-based: a long S-curve's bounding box is mostly
 * empty space.
 */
/**
 * The nearest edge AND the page-space midpoint of its curve, which is where the
 * insert affordance sits. Bowed and mixed-side curves do not share the
 * straight chord's midpoint, so use the same cubic as the renderer.
 */
export function findEdgeWithMidpoint(
  editor: Editor,
  page: { x: number; y: number },
  tolerance: number,
): { edgeId: string; point: { x: number; y: number } } | null {
  let best: { edgeId: string; point: { x: number; y: number } } | null = null;
  let bestDistance = tolerance;

  const shapes = editor.getCurrentPageShapes();
  for (let i = 0; i < shapes.length; i += 1) {
    const shape = shapes[i];
    if (shape.type !== EDGE_SHAPE_TYPE) continue;
    const props = (shape as any).props;
    if (!props?.edgeId || props.relationship === 'unchanged input') continue;
    const localX = page.x - (shape as any).x;
    const localY = page.y - (shape as any).y;
    const distance = distanceToEdgeCurve(
      localX, localY, props.startX, props.startY, props.endX, props.endY, 48, props.curveOffset ?? 0, props.startSide, props.endSide,
    );
    if (distance >= bestDistance) continue;
    bestDistance = distance;
    const midpoint = edgeCurveMidpoint(props.startX, props.startY, props.endX, props.endY,
      props.curveOffset ?? 0, props.startSide, props.endSide);
    best = {
      edgeId: props.edgeId as string,
      point: {
        x: (shape as any).x + midpoint.x,
        y: (shape as any).y + midpoint.y,
      },
    };
  }
  return best;
}

export function findEdgeAtPoint(
  editor: Editor,
  page: { x: number; y: number },
  tolerance: number,
): string | null {
  let best: string | null = null;
  let bestDistance = tolerance;

  const shapes = editor.getCurrentPageShapes();
  for (let i = 0; i < shapes.length; i += 1) {
    const shape = shapes[i];
    if (shape.type !== EDGE_SHAPE_TYPE) continue;
    const props = (shape as any).props;
    if (!props?.edgeId || props.relationship === 'unchanged input') continue;
    // props.startX/endX are in the shape's local box, so the point comes back
    // out of page space the same way.
    const distance = distanceToEdgeCurve(
      page.x - (shape as any).x,
      page.y - (shape as any).y,
      props.startX,
      props.startY,
      props.endX,
      props.endY,
      48, props.curveOffset ?? 0, props.startSide, props.endSide,
    );
    if (distance >= bestDistance) continue;
    bestDistance = distance;
    best = props.edgeId as string;
  }
  return best;
}
