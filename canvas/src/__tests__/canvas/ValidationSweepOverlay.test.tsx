import React from 'react';
import { render, act } from '@testing-library/react';
import { ValidationSweepOverlay } from '../../canvas/ValidationSweepOverlay';

// Stub the adapter so importing the overlay doesn't pull in the tldraw ESM bundle.
jest.mock('../../canvas/canvasShapeId', () => ({
  nodeIdToShapeId: (id: string) => `shape:${id}`,
}));

const NODE_X: Record<string, number> = { a: 0, b: 100, c: 200 };

function makeEditor() {
  return {
    getShape: (shapeId: string) => {
      const id = shapeId.replace('shape:', '');
      if (!(id in NODE_X)) return undefined;
      return { x: NODE_X[id], y: 0, props: { w: 80, h: 40 } };
    },
    pageToViewport: ({ x, y }: { x: number; y: number }) => ({ x, y }),
    sideEffects: { registerAfterChangeHandler: () => () => {} },
  } as any;
}

const makeNodes = (errIds: string[] = []) =>
  ['a', 'b', 'c'].map((id) => ({
    id,
    type: 'layer',
    position: { x: NODE_X[id], y: 0 },
    data: { error: errIds.includes(id) ? 'shape mismatch' : undefined },
  })) as any;

// a → b → c (topological order is a, b, c regardless of input array order)
const EDGES = [
  { source: 'b', target: 'c' },
  { source: 'a', target: 'b' },
] as any;

describe('ValidationSweepOverlay', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    (window as any).matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
  });
  afterEach(() => {
    act(() => { jest.runOnlyPendingTimers(); });
    jest.useRealTimers();
  });

  it('renders nothing until a sweep is triggered (runId 0)', () => {
    const { container } = render(
      <ValidationSweepOverlay editor={makeEditor()} nodes={makeNodes()} edges={EDGES} runId={0} />,
    );
    expect(container.querySelector('.ui2-sweep-overlay')).toBeNull();
  });

  it('lights a halo per node in topological order, painting errored nodes red', () => {
    const { container } = render(
      <ValidationSweepOverlay editor={makeEditor()} nodes={makeNodes(['b'])} edges={EDGES} runId={1} />,
    );
    act(() => { jest.advanceTimersByTime(300); });
    const halos = Array.from(container.querySelectorAll('.ui2-sweep-halo'));
    expect(halos).toHaveLength(3);
    // topo order a,b,c → b errored → ok, err, ok
    expect(halos[0].className).toContain('ok');
    expect(halos[1].className).toContain('err');
    expect(halos[2].className).toContain('ok');
    // positioned from the fake editor's page→viewport mapping
    expect((halos[0] as HTMLElement).style.left).toBe('0px');
    expect((halos[2] as HTMLElement).style.left).toBe('200px');
  });

  it('uses the same sweep for measured nodes and mapped errors, leaving unreached nodes unlit', () => {
    const nodes = makeNodes(['b']);
    nodes[0].data.parameters = { _shape_provenance: 'observed' };
    nodes[1].data.parameters = { _shape_provenance: 'unreached' };
    nodes[2].data.parameters = { _shape_provenance: 'unreached' };
    const editor = makeEditor();
    const { container, rerender } = render(
      <ValidationSweepOverlay editor={editor} nodes={nodes} edges={EDGES} runId={1} measuredOnly />,
    );
    act(() => { jest.advanceTimersByTime(300); });
    expect(container.querySelectorAll('.ui2-sweep-halo.ok')).toHaveLength(1);
    expect(container.querySelectorAll('.ui2-sweep-halo.err')).toHaveLength(1);
    act(() => { jest.advanceTimersByTime(5000); });
    rerender(<ValidationSweepOverlay editor={editor} nodes={nodes} edges={EDGES} runId={2} measuredOnly />);
    act(() => { jest.advanceTimersByTime(300); });
    expect(container.querySelectorAll('.ui2-sweep-halo')).toHaveLength(2);
  });

  it('tears itself down after the sweep finishes', () => {
    const { container } = render(
      <ValidationSweepOverlay editor={makeEditor()} nodes={makeNodes()} edges={EDGES} runId={1} />,
    );
    act(() => { jest.advanceTimersByTime(5000); });
    expect(container.querySelector('.ui2-sweep-overlay')).toBeNull();
  });

  it('keeps validation halos on their nodes while automatic fit changes the camera', () => {
    const editor = makeEditor();
    const handlers = new Map<string, () => void>();
    const dispose = jest.fn();
    editor.sideEffects.registerAfterChangeHandler = (type: string, handler: () => void) => {
      handlers.set(type, handler);
      return dispose;
    };
    let zoom = 1;
    editor.pageToViewport = ({ x, y }: { x: number; y: number }) => ({ x: x * zoom, y: y * zoom });
    const { container, unmount } = render(
      <ValidationSweepOverlay editor={editor} nodes={makeNodes()} edges={EDGES} runId={1} />,
    );
    act(() => { jest.advanceTimersByTime(300); });
    const halo = container.querySelectorAll<HTMLElement>('.ui2-sweep-halo')[2];
    expect(halo.style.left).toBe('200px');
    expect(halo.style.width).toBe('80px');
    act(() => { zoom = 0.5; handlers.get('camera')?.(); });
    expect(halo.style.left).toBe('100px');
    expect(halo.style.width).toBe('40px');
    unmount();
    expect(dispose).toHaveBeenCalledTimes(2);
  });

  it('does not play when the user prefers reduced motion', () => {
    (window as any).matchMedia = () => ({ matches: true, addEventListener() {}, removeEventListener() {} });
    const { container } = render(
      <ValidationSweepOverlay editor={makeEditor()} nodes={makeNodes()} edges={EDGES} runId={1} />,
    );
    act(() => { jest.advanceTimersByTime(2000); });
    expect(container.querySelector('.ui2-sweep-overlay')).toBeNull();
  });
});
