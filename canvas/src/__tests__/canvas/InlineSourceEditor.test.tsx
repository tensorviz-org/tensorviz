import React from 'react';
import { render } from '@testing-library/react';
import { InlineSourceEditor } from '../../canvas/InlineSourceEditor';

let mockShape: any;
let mockViewport = { w: 1280, h: 720 };
const mockEditor = {
  getShape: () => mockShape,
  getShapePageBounds: () => ({ x: 20, y: 40 }),
  getViewportScreenBounds: () => mockViewport,
  pageToViewport: (point: unknown) => point,
  getZoomLevel: () => 1,
  zoomToBounds: jest.fn(),
  updateShape: jest.fn(),
};
jest.mock('tldraw', () => ({
  Box: class {},
  useEditor: () => mockEditor,
  useValue: (_name: string, read: () => unknown) => read(),
}));
jest.mock('../../canvas/reducedMotion', () => ({ cameraDuration: () => 0 }));
jest.mock('../../ide/SourceEditSession', () => ({ SourceEditSession: () => <div>source form</div> }));

beforeEach(() => {
  mockEditor.updateShape.mockImplementation(update => { mockShape = { ...mockShape, props: { ...mockShape.props, ...update.props } }; });
  mockShape = { id: 'shape:node:fc', type: 'layer', props: { w: 200, h: 140, nodeData: {} } };
  mockViewport = { w: 1280, h: 720 };
});

test('closing a temporary node form restores the original card size', () => {
  const view = render(<InlineSourceEditor nodes={[{ id: 'fc' }] as any} onClose={() => {}} />);
  expect(mockShape.props).toMatchObject({ w: 340, h: 430 });
  view.unmount();
  expect(mockShape.props).toMatchObject({ w: 200, h: 140 });
});

test('closing after reprojection preserves the replacement card size', () => {
  const view = render(<InlineSourceEditor nodes={[{ id: 'fc' }] as any} onClose={() => {}} />);
  mockShape = { ...mockShape, props: { ...mockShape.props, w: 260, h: 300 } };
  view.unmount();
  expect(mockShape.props).toMatchObject({ w: 260, h: 300 });
  expect(mockEditor.updateShape).toHaveBeenCalledTimes(1);
});

test('editing an expanded module preserves its frame and child layout', () => {
  mockShape = { ...mockShape, type: 'module-node', props: { w: 800, h: 1600, nodeData: { expanded: true } } };
  const view = render(<InlineSourceEditor nodes={[{ id: 'fc' }] as any} onClose={() => {}} />);
  view.unmount();
  expect(mockEditor.updateShape).not.toHaveBeenCalled();
});

test('resizing the sidebar keeps the active form in view without resizing its node again', () => {
  const view = render(<InlineSourceEditor nodes={[{ id: 'fc' }] as any} onClose={() => {}} />);
  mockViewport = { w: 380, h: 720 };
  view.rerender(<InlineSourceEditor nodes={[{ id: 'fc' }] as any} onClose={() => {}} />);
  expect(mockEditor.zoomToBounds).toHaveBeenCalledTimes(2);
  expect(mockEditor.updateShape).toHaveBeenCalledTimes(1);
});
