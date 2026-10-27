import { findEdgeAtPoint, findEdgeWithMidpoint } from '../../canvas/edgeSelection';
import type { Editor } from 'tldraw';

jest.mock('tldraw', () => ({ atom: () => ({ set: jest.fn() }) }));

test('input annotations cannot be selected or offered as editable computation edges', () => {
  const shape = { type: 'model-edge', x: 0, y: 0, props: {
    edgeId: 'annotation', relationship: 'unchanged input', startX: 0, startY: 0, endX: 100, endY: 0,
  } };
  const editor = { getCurrentPageShapes: () => [shape] } as unknown as Editor;
  expect(findEdgeAtPoint(editor, { x: 50, y: 0 }, 10)).toBeNull();
  expect(findEdgeWithMidpoint(editor, { x: 50, y: 0 }, 10)).toBeNull();
  shape.props.relationship = '';
  expect(findEdgeAtPoint(editor, { x: 50, y: 0 }, 10)).toBe('annotation');
  expect(findEdgeWithMidpoint(editor, { x: 50, y: 0 }, 10)?.edgeId).toBe('annotation');
});
