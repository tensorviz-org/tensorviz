import { isManagedCanvasShape } from '../../canvas/canvasShapePolicy';

describe('useTldrawBridge managed canvas shapes', () => {
  it('does not preserve obsolete chat shapes', () => {
    expect(isManagedCanvasShape({ type: 'chat-node' })).toBe(false);
  });

  it('still removes unsupported user-created shapes', () => {
    expect(isManagedCanvasShape({ type: 'geo' })).toBe(false);
  });
});
