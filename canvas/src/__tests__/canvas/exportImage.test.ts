jest.mock('../../ide/hostServices', () => ({ copyText: jest.fn(), saveFile: jest.fn() }));

import { copyText, saveFile } from '../../ide/hostServices';
import { saveCanvasImage } from '../../canvas/exportImage';

const editorWith = (shapes: Array<{ id: string; type: string }>) => ({
  getCurrentPageShapes: () => shapes,
  toImageDataUrl: jest.fn(async () => ({ url: `data:image/svg+xml;base64,${btoa('<svg/>')}` })),
}) as never;

beforeEach(() => {
  (copyText as jest.Mock).mockClear();
  (saveFile as jest.Mock).mockClear();
});

test('saving writes the SVG to a file, not the clipboard', async () => {
  const outcome = await saveCanvasImage(editorWith([{ id: 'shape:node', type: 'layer' }]), 'svg');
  expect(outcome.ok).toBe(true);
  expect(saveFile).toHaveBeenCalledWith('graph.svg', expect.any(Blob));
  expect(copyText).not.toHaveBeenCalled();
});

test('an empty canvas is not saved', async () => {
  const outcome = await saveCanvasImage(editorWith([]), 'svg');
  expect(outcome.ok).toBe(false);
  expect(saveFile).not.toHaveBeenCalled();
});
