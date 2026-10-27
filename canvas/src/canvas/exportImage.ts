/**
 * The canvas as a picture you can paste somewhere.
 *
 * To the CLIPBOARD, not to a file, and that is the whole design. A VS Code webview
 * blocks a download — `<a download>`, a blob: link and a script-driven save all do
 * nothing there — so a Save button would work in the browser page and silently fail
 * inside the editor, which is the surface most people use. Writing the file from the
 * extension instead would mean the host gaining a write path it does not have, for a
 * picture.
 *
 * `copyText` already routes through the host bridge when there is one, so an SVG
 * reaches the system clipboard from both surfaces through a path that already
 * exists.
 *
 * SVG rather than PNG wherever there is a choice: it is text, so it diffs, it scales,
 * and it can be read without opening it.
 */
import type { Editor, TLShapeId } from 'tldraw';

import { copyText, saveFile } from '../ide/hostServices';
import { EDGE_SHAPE_TYPE, isModelShape } from './canvasShapePolicy';

export type ExportFormat = 'svg' | 'png';

/** How a graph becomes a picture; shared by the clipboard and file paths. */
const IMAGE_OPTIONS = {
  background: true,
  darkMode: true,
  padding: 32,
  // 1, not the screen's ratio: an export is read at whatever size it is placed at,
  // and a 3x raster of a 40-node graph is a file nobody wants in a README.
  pixelRatio: 1,
} as const;

export interface ExportOutcome {
  ok: boolean;
  /** What to tell the user. Always set — a copy that silently did nothing is the
   *  failure this whole path exists to avoid. */
  message: string;
}

/**
 * Which shapes belong in the picture.
 *
 * The managed ones: nodes and the edges between them. Not tldraw's own furniture,
 * and not the chat card — a graph exported into a README should be the graph, and a
 * conversation someone had beside it is not part of the model.
 */
export function exportableShapeIds(shapes: Array<{ id: TLShapeId; type: string }>): TLShapeId[] {
  // NOT `isManagedCanvasShape`, which also admits the chat card: that predicate
  // answers "does this canvas own it", and this one answers "is it the graph".
  return shapes
    .filter((shape) => isModelShape(shape) || shape.type === EDGE_SHAPE_TYPE)
    .map((shape) => shape.id);
}

/** The data URL's payload, decoded. `toImageDataUrl` answers base64 for both
 *  formats, so an SVG arrives encoded even though it is text. */
export function decodeDataUrl(url: string): string {
  const comma = url.indexOf(',');
  const head = url.slice(0, comma);
  const body = url.slice(comma + 1);
  return head.includes(';base64') ? atob(body) : decodeURIComponent(body);
}

export async function exportCanvasImage(
  editor: Editor | null | undefined,
  format: ExportFormat,
): Promise<ExportOutcome> {
  if (!editor) {
    return { ok: false, message: 'The canvas is not ready yet.' };
  }
  const ids = exportableShapeIds(editor.getCurrentPageShapes());
  if (ids.length === 0) {
    return { ok: false, message: 'Nothing on the canvas to export.' };
  }

  try {
    const exported = await editor.toImageDataUrl(ids, { format, ...IMAGE_OPTIONS });

    if (format === 'svg') {
      await copyText(decodeDataUrl(exported.url));
      return { ok: true, message: `Copied ${ids.length} shapes as SVG.` };
    }

    // PNG has no text form, so it goes to the clipboard as an image — which not
    // every surface allows. Saying which one failed beats a button that appears to
    // work.
    const blob = await (await fetch(exported.url)).blob();
    const clipboard = navigator.clipboard as { write?: (items: unknown[]) => Promise<void> };
    if (typeof ClipboardItem === 'undefined' || !clipboard?.write) {
      return { ok: false, message: 'This surface cannot copy an image. Use SVG instead.' };
    }
    await clipboard.write([new ClipboardItem({ 'image/png': blob })]);
    return { ok: true, message: `Copied ${ids.length} shapes as PNG.` };
  } catch (error) {
    return { ok: false, message: `Export failed: ${String(error)}` };
  }
}

/**
 * The canvas as a file.
 *
 * In the browser this is an ordinary download; inside a VS Code webview the bytes
 * go over the bridge and the host writes them into `tensorviz-exports/` and opens
 * the file (see `hostServices.saveFile`), because `a[download]` is inert there.
 */
export async function saveCanvasImage(
  editor: Editor | null | undefined,
  format: ExportFormat,
): Promise<ExportOutcome> {
  if (!editor) {
    return { ok: false, message: 'The canvas is not ready yet.' };
  }
  const ids = exportableShapeIds(editor.getCurrentPageShapes());
  if (ids.length === 0) {
    return { ok: false, message: 'Nothing on the canvas to export.' };
  }

  try {
    const exported = await editor.toImageDataUrl(ids, { format, ...IMAGE_OPTIONS });
    if (format === 'svg') {
      saveFile('graph.svg', new Blob([decodeDataUrl(exported.url)], { type: 'image/svg+xml' }));
    } else {
      saveFile('graph.png', await (await fetch(exported.url)).blob());
    }
    return { ok: true, message: `Saved ${ids.length} shapes as ${format.toUpperCase()}.` };
  } catch (error) {
    return { ok: false, message: `Export failed: ${String(error)}` };
  }
}
