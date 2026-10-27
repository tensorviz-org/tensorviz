/**
 * The canvas as a picture, and where it goes.
 *
 * To the clipboard rather than to a file, which is the decision worth recording: a
 * VS Code webview blocks a download — `<a download>`, blob: links and script-driven
 * saves all do nothing there — so a Save button would work in the browser page and
 * silently fail inside the editor, which is the surface most people use.
 *
 * The parts tested here are the ones that do not need tldraw, which cannot be
 * mounted under the CRA jest transform.
 */
import { decodeDataUrl, exportableShapeIds } from '../../canvas/exportImage';

const shape = (id: string, type: string) => ({ id: id as never, type });

describe('what belongs in the picture', () => {
  it('takes the nodes and the edges between them', () => {
    const ids = exportableShapeIds([
      shape('shape:node:a', 'layer'),
      shape('shape:node:b', 'module-node'),
      shape('shape:edge:x', 'model-edge'),
    ]);
    expect(ids).toHaveLength(3);
  });

  it('leaves out the chat card and tldraw’s own furniture', () => {
    // A graph exported into a README should be the graph. A conversation someone
    // had beside it is not part of the model, and neither is a selection box.
    const ids = exportableShapeIds([
      shape('shape:node:a', 'layer'),
      shape('shape:chat', 'chat-node'),
      shape('shape:draw', 'draw'),
      shape('shape:text', 'text'),
    ]);
    expect(ids).toEqual(['shape:node:a']);
  });
});

describe('reading back what tldraw hands over', () => {
  it('decodes a base64 payload, which is what an SVG arrives as', () => {
    // `toImageDataUrl` answers base64 for both formats, so an SVG is encoded even
    // though it is text — copied raw, the clipboard would hold the base64.
    const svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>';
    expect(decodeDataUrl(`data:image/svg+xml;base64,${btoa(svg)}`)).toBe(svg);
  });

  it('decodes the percent-encoded form too', () => {
    const svg = '<svg><g id="a b"/></svg>';
    expect(decodeDataUrl(`data:image/svg+xml,${encodeURIComponent(svg)}`)).toBe(svg);
  });
});
