import { execFileSync } from 'child_process';
import path from 'path';

type Shape = { id: string; type: 'layer' | 'module-node'; x: number; y: number; w: number; h: number };
type Gesture = { click: [number, number] } | { doubleClick: [number, number]; shift?: boolean; onShape?: string };
type Result = { activations: string[]; clicks: { phase: string; target: string; shape?: string }[]; selected: string[] };

// Each scenario starts Node and loads tldraw, which takes about a second.
jest.setTimeout(30000);

/**
 * tldraw does not load under the CRA Jest transform, so each scenario runs in Node
 * against the installed tldraw: real pointer events, click detector and hit test.
 */
function perform(shapes: Shape[], gestures: Gesture[]): Result {
  const script = path.join(__dirname, '../../test/tldrawGestures.cjs');
  return JSON.parse(execFileSync(process.execPath, [script, JSON.stringify({ shapes, gestures })],
    { cwd: process.cwd(), encoding: 'utf8', timeout: 20000 }));
}

const layer: Shape = { id: 'fc', type: 'layer', x: 100, y: 100, w: 200, h: 100 };
const expandedModule: Shape = { id: 'block', type: 'module-node', x: 0, y: 0, w: 700, h: 400 };

it('activates the card under a double-click once, although tldraw reports it on the canvas', () => {
  const { activations, clicks, selected } = perform([layer], [{ doubleClick: [150, 150] }]);
  // No phase of the gesture names the shape: card bodies leave the pointer to tldraw.
  expect(clicks).toEqual(['down', 'up', 'settle'].map(phase => ({ phase, target: 'canvas' })));
  expect(activations).toEqual(['shape:fc']);
  expect(selected).toEqual(['shape:fc']);
});

it('activates nothing when the double-click lands on empty canvas', () => {
  expect(perform([layer], [{ click: [150, 150] }, { doubleClick: [600, 500] }]).activations).toEqual([]);
});

it('activates the card above an expanded module, and the module from its own surface', () => {
  expect(perform([expandedModule, layer], [{ doubleClick: [150, 150] }]).activations).toEqual(['shape:fc']);
  expect(perform([expandedModule, layer], [{ doubleClick: [500, 300] }]).activations).toEqual(['shape:block']);
});

it('leaves a Shift double-click to toggle the selection', () => {
  expect(perform([layer], [{ doubleClick: [150, 150], shift: true }]).activations).toEqual([]);
});

it('still activates a double-click reported on the shape itself', () => {
  expect(perform([layer], [{ doubleClick: [150, 150], onShape: 'fc' }]).activations).toEqual(['shape:fc']);
});
