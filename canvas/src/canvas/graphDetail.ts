import { computed, Editor, useSvgExportContext } from 'tldraw';
import { detailTier, DetailTier } from './detailTier';

const tiers = new WeakMap<Editor, ReturnType<typeof createTier>>();
function createTier(editor: Editor) {
  return computed<DetailTier>('tensorViz readable detail', previous =>
    detailTier(editor.getEfficientZoomLevel() * 200, typeof previous === 'string' ? previous : undefined));
}

/** One discrete signal per editor, not a subscription per pixel of camera travel. */
export function useGraphDetail(editor: Editor): DetailTier {
  const exporting = useSvgExportContext();
  let tier = tiers.get(editor);
  if (!tier) { tier = createTier(editor); tiers.set(editor, tier); }
  return exporting ? 'inspection' : tier.get();
}
