import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import type { Editor } from 'tldraw';
import GraphSvgLayer, { clampOverviewCamera } from '../../canvas/GraphSvgLayer';
import { graphSvgBounds } from '../../services/graphSvg';
import type { ModelNode } from '../../types';
import { configureCanvasTheme, setCanvasHostTheme } from '../../styles/canvasTheme';

beforeEach(() => { localStorage.clear(); configureCanvasTheme('host'); });

// jsdom has no PointerEvent, so RTL's pointer helpers would carry no coordinates.
if (typeof window.PointerEvent === 'undefined') {
  (window as unknown as { PointerEvent: typeof MouseEvent }).PointerEvent = class extends MouseEvent {};
}

const node = (id: string, x: number): ModelNode => ({
  id, type: 'layerNode', position: { x, y: 40 },
  data: { layerType: { id: 'linear', name: 'Linear' }, parameters: { _module_path: id } },
} as ModelNode);

const nodes = [node('a', 20), node('b', 320)];

const fakeEditor = (over: Record<string, unknown> = {}): Editor => ({
  getCamera: () => ({ x: 5, y: 7, z: 2 }),
  getViewportScreenBounds: () => ({ x: 0, y: 0, width: 800, height: 600 }),
  getCameraOptions: () => ({ zoomSteps: [0.05, 0.1, 0.5, 1, 2, 4] }),
  setCameraOptions: jest.fn(),
  on: jest.fn(),
  off: jest.fn(),
  store: { listen: () => () => {} },
  screenToPage: (point: { x: number; y: number }) => point,
  ...over,
}) as unknown as Editor;

const renderLayer = (editor: Editor, onSelect = jest.fn()) => {
  render(<GraphSvgLayer nodes={nodes} edges={[]} selectedIds={new Set()} onSelect={onSelect} editor={editor} />);
  return { layer: screen.getByTestId('graph-svg-layer'), content: screen.getByTestId('graph-svg-content'), onSelect };
};

test('follows the tldraw camera, so gestures are tldraw gestures', () => {
  const { content } = renderLayer(fakeEditor());
  expect(content.style.transform).toContain('scale(2)');
});

test('takes no pointer events, so wheel/pan reach the tldraw canvas', () => {
  const { layer } = renderLayer(fakeEditor());
  expect(layer.style.pointerEvents).toBe('none');
});

test.each([0.05, 1, 4])('diff remains visible at zoom %s, including on selected nodes, and clears with review', z => {
  const changed = { ...nodes[0], data: { ...nodes[0].data, diffStatus: 'modified' as const } };
  const hidden = { ...changed, id: 'hidden', hidden: true };
  const editor = fakeEditor({ getCamera: () => ({ x: 5, y: 7, z }) });
  const { container, rerender } = render(<GraphSvgLayer nodes={[changed, nodes[1], hidden]} edges={[]}
    selectedIds={new Set(['a'])} onSelect={jest.fn()} editor={editor} />);
  const outlines = container.querySelectorAll<HTMLElement>('[data-diff-node-id]');
  expect(outlines).toHaveLength(1);
  expect(outlines[0].dataset.diffNodeId).toBe('a');
  expect(outlines[0].style.outline).toBe(`${2 / z}px solid var(--tv-modified)`);
  expect(outlines[0].style.pointerEvents).toBe('none');
  const selection = container.querySelector<HTMLElement>('.graph-svg-selection')!;
  expect(selection.style.outline).toBe(`${1.5 / z}px solid var(--tv-selection)`);
  expect(selection.style.outlineOffset).toBe(`${3.5 / z}px`);
  for (const property of ['left', 'top', 'width', 'height'] as const) {
    expect(selection.style[property]).toBe(outlines[0].style[property]);
  }
  rerender(<GraphSvgLayer nodes={nodes} edges={[]} onSelect={jest.fn()} editor={editor} />);
  expect(container.querySelector('[data-diff-node-id]')).toBeNull();
});

test.each([
  { error: 'Shape mismatch' }, { moduleDiagnosticStatus: 'error' as const }, { moduleStatus: 'error' as const },
])('diff highlighting does not hide a node failure from %o', failure => {
  const failed = { ...nodes[0], data: { ...nodes[0].data, diffStatus: 'modified' as const, ...failure } };
  const { container } = render(<GraphSvgLayer nodes={[failed]} edges={[]} onSelect={jest.fn()} editor={fakeEditor()} />);
  expect(container.querySelector<HTMLElement>('[data-diff-node-id]')?.style.outline).toBe('1px solid var(--tv-error)');
});

test('overview outlines distinguish changed inputs from source modifications', () => {
  const changed = { ...nodes[0], data: { ...nodes[0].data, diffStatus: 'rewired' as const } };
  const { container } = render(<GraphSvgLayer nodes={[changed]} edges={[]} onSelect={jest.fn()} editor={fakeEditor()} />);
  expect(container.querySelector<HTMLElement>('[data-diff-node-id]')?.style.outline).toBe('1px solid var(--tv-inputs-changed)');
});

test('a mounted overview changes Ink to Paper and back without resetting its camera, selection or review', async () => {
  const createDescriptor = Object.getOwnPropertyDescriptor(URL, 'createObjectURL');
  const revokeDescriptor = Object.getOwnPropertyDescriptor(URL, 'revokeObjectURL');
  const blobs = new Map<string, Blob>();
  const revoke = jest.fn();
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: (blob: Blob) => {
    const url = `blob:overview-${blobs.size}`; blobs.set(url, blob); return url;
  } });
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revoke });
  const changed = { ...nodes[0], data: { ...nodes[0].data, diffStatus: 'rewired' as const,
    review: { status: 'rewired' as const, params: {}, inputChanges: { added: 1, removed: 0 } } } };
  const failed = { ...nodes[1], data: { ...nodes[1].data, diffStatus: 'modified' as const, error: 'Shape mismatch' } };
  const camera = { x: 5, y: 7, z: 2 };
  const setCamera = jest.fn();
  const editor = fakeEditor({ getCamera: () => camera, setCamera });
  const onSelect = jest.fn();
  const view = render(<GraphSvgLayer nodes={[changed, failed]} edges={[]} selectedIds={new Set(['a'])}
    onSelect={onSelect} editor={editor} />);
  try {
    const image = screen.getByRole('img', { name: 'Expanded graph overview' });
    const content = screen.getByTestId('graph-svg-content');
    const selection = view.container.querySelector<HTMLElement>('.graph-svg-selection')!;
    const changedOutline = view.container.querySelector<HTMLElement>('[data-diff-node-id="a"]')!;
    const errorOutline = view.container.querySelector<HTMLElement>('[data-diff-node-id="b"]')!;
    const retainedTransform = content.style.transform;
    const retainedSelection = selection.style.cssText;
    const readSvg = () => new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(reader.error);
      reader.readAsText(blobs.get(image.getAttribute('src')!)!);
    });
    const initialUrl = image.getAttribute('src');
    const ink = await readSvg();
    expect(ink).toContain('fill="#171817"');
    expect(ink).toContain('>inputs changed</text>');
    expect(ink).toContain('stroke="#ff9298"');

    act(() => setCanvasHostTheme('light'));
    const paperUrl = image.getAttribute('src');
    const paper = await readSvg();
    expect(paperUrl).not.toBe(initialUrl);
    expect(paper).toContain('fill="#f3f2ee"');
    expect(paper).toContain('fill="#1f52b4"');
    expect(paper).toContain('stroke="#c62f45"');
    const bounds = (svg: string) => new DOMParser().parseFromString(svg, 'image/svg+xml').documentElement.getAttribute('viewBox');
    expect(bounds(paper)).toBe(bounds(ink));

    act(() => setCanvasHostTheme('dark'));
    expect(await readSvg()).toBe(ink);
    expect(screen.getByTestId('graph-svg-content')).toBe(content);
    expect(screen.getByRole('img', { name: 'Expanded graph overview' })).toBe(image);
    expect(content.style.transform).toBe(retainedTransform);
    expect(view.container.querySelector('.graph-svg-selection')).toBe(selection);
    expect(selection.style.cssText).toBe(retainedSelection);
    expect(changedOutline.style.outline).toBe('1px solid var(--tv-inputs-changed)');
    expect(errorOutline.style.outline).toBe('1px solid var(--tv-error)');
    expect(setCamera).not.toHaveBeenCalled();
    expect(onSelect).not.toHaveBeenCalled();
    expect(revoke).toHaveBeenCalledWith(initialUrl);
    expect(revoke).toHaveBeenCalledWith(paperUrl);
  } finally {
    view.unmount();
    if (createDescriptor) Object.defineProperty(URL, 'createObjectURL', createDescriptor);
    else delete (URL as Partial<typeof URL>).createObjectURL;
    if (revokeDescriptor) Object.defineProperty(URL, 'revokeObjectURL', revokeDescriptor);
    else delete (URL as Partial<typeof URL>).revokeObjectURL;
  }
});

test('keeps 40% of the smaller of graph and viewport overlapping on each axis', () => {
  const box = { x: 0, y: 0, w: 1000, h: 500 };
  const viewport = { width: 800, height: 600 };
  const left = clampOverviewCamera({ x: 100000, y: 100000, z: 1 }, viewport, box);
  expect(left.x).toBe(480); // 0.4 of the viewport width stays covered
  expect(left.y).toBe(400); // 0.4 of 500 visible at the top edge
  const right = clampOverviewCamera({ x: -100000, y: -100000, z: 1 }, viewport, box);
  expect(right.x).toBe(-680); // trailing 40% of the viewport remains covered
  expect(right.y).toBe(-300);
  const inside = clampOverviewCamera({ x: 10, y: 20, z: 1 }, viewport, box);
  expect(inside).toMatchObject({ x: 10, y: 20 });
});

test.each([0.05, 1, 4])('a large graph can pan to either end at zoom %s', z => {
  const box = { x: 100, y: 200, w: 2500, h: 66000 };
  const viewport = { width: 800, height: 600 };
  for (const y of [-box.y, viewport.height / z - box.y - box.h]) {
    const camera = { x: -box.x, y, z };
    expect(clampOverviewCamera(camera, viewport, box)).toEqual(camera);
  }
  const top = clampOverviewCamera({ x: 0, y: 1e6, z }, viewport, box);
  const bottom = clampOverviewCamera({ x: 0, y: -1e6, z }, viewport, box);
  expect((top.y + box.y) * z).toBeCloseTo(viewport.height * 0.6);
  expect((bottom.y + box.y + box.h) * z).toBeCloseTo(viewport.height * 0.4);
});

test('publishes the corrected camera on mount and when detail changes the graph bounds', () => {
  let camera = { x: -100000, y: -100000, z: 1 };
  const editor = fakeEditor({ getCamera: () => camera,
    setCamera: (next: typeof camera) => { camera = next; } });
  const { rerender } = render(<GraphSvgLayer nodes={nodes} edges={[]} onSelect={jest.fn()} editor={editor} />);
  const expectAligned = (currentNodes: ModelNode[]) => {
    const bounds = graphSvgBounds(currentNodes);
    const minX = Math.min(...bounds.map(b => b.x)) - 24;
    const minY = Math.min(...bounds.map(b => b.y)) - 24;
    expect(screen.getByTestId('graph-svg-content').style.transform)
      .toBe(`translate(${(minX + camera.x) * camera.z}px, ${(minY + camera.y) * camera.z}px) scale(${camera.z})`);
  };
  expectAligned(nodes);
  const smaller = [nodes[0]];
  rerender(<GraphSvgLayer nodes={smaller} edges={[]} onSelect={jest.fn()} editor={editor} />);
  expectAligned(smaller);
});

test('an out-of-range camera is pushed back through setCamera', () => {
  const setCamera = jest.fn();
  const editor = fakeEditor({ getCamera: () => ({ x: -100000, y: -100000, z: 1 }), setCamera });
  renderLayer(editor);
  expect(setCamera).toHaveBeenCalledTimes(1);
  expect(setCamera.mock.calls[0][0].x).toBeGreaterThan(-100000);
});

test('SVG zoom can fit a tall capture, follows resize, and restores interactive zoom limits', () => {
  const setCameraOptions = jest.fn();
  let viewport = { x: 0, y: 0, width: 800, height: 600 };
  let resize: () => void;
  const editor = fakeEditor({ setCameraOptions, getViewportScreenBounds: () => viewport,
    on: jest.fn((event, listener) => { if (event === 'resize') resize = listener; }) });
  const original = editor.getCameraOptions().zoomSteps;
  const tall = [nodes[0], { ...nodes[1], position: { x: 320, y: 66000 } }];
  const { unmount } = render(<GraphSvgLayer nodes={tall} edges={[]} onSelect={jest.fn()} editor={editor} />);
  const height = Number.parseFloat(screen.getByTestId('graph-svg-content').style.height);
  const firstSteps = setCameraOptions.mock.calls[0][0].zoomSteps;
  expect(firstSteps[0]).toBeCloseTo((viewport.height - 48) / height);
  expect(firstSteps[0]).toBeLessThan(original[0]);
  expect(firstSteps.slice(1)).toEqual(original);
  viewport = { ...viewport, height: 300 };
  // A sidebar divider or input panel changes the editor, not the browser window.
  act(() => resize());
  const resizedSteps = setCameraOptions.mock.calls[setCameraOptions.mock.calls.length - 1][0].zoomSteps;
  expect(resizedSteps[0]).toBeCloseTo((viewport.height - 48) / height);
  unmount();
  expect(setCameraOptions).toHaveBeenLastCalledWith({ zoomSteps: original });
  expect(editor.off).toHaveBeenCalledWith('resize', resize!);
});

test('a click selects through the same camera', () => {
  const target = graphSvgBounds(nodes).find((box) => box.id === 'b')!;
  const editor = fakeEditor({
    screenToPage: () => ({ x: target.x + target.w / 2, y: target.y + target.h / 2 }),
  });
  const { layer, onSelect } = renderLayer(editor);
  layer.getBoundingClientRect = () => ({ left: 0, top: 0, right: 800, bottom: 600,
    width: 800, height: 600, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;
  fireEvent.pointerDown(window, { button: 0, clientX: 100, clientY: 100 });
  fireEvent.pointerUp(window, { button: 0, clientX: 100, clientY: 100 });
  expect(onSelect).toHaveBeenCalledWith('b');
});

test('double-click opens local detail through the camera and ignores clicks outside the canvas', () => {
  const onActivate = jest.fn();
  render(<GraphSvgLayer nodes={nodes} edges={[]} onSelect={jest.fn()} onActivate={onActivate}
    editor={fakeEditor({ screenToPage: () => ({ x: 330, y: 50 }) })} />);
  screen.getByTestId('graph-svg-layer').getBoundingClientRect = () => ({ left: 0, top: 50,
    right: 800, bottom: 600 } as DOMRect);
  fireEvent.doubleClick(window, { clientX: 100, clientY: 100 });
  expect(onActivate).toHaveBeenCalledWith('b');
  fireEvent.doubleClick(window, { clientX: 100, clientY: 20 });
  expect(onActivate).toHaveBeenCalledTimes(1);
});
