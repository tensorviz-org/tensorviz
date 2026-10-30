import React from 'react';
import { render } from '@testing-library/react';
import { LayerShapeUtil } from '../../canvas/LayerShapeUtil';
import { TensorSummary } from '../../canvas/TensorSummary';
import { ModelNode } from '../../types';

jest.mock('tldraw', () => ({
  BaseBoxShapeUtil: class {}, HTMLContainer: ({ children }: any) => <div>{children}</div>,
  T: { number: {}, any: {} }, track: (component: any) => component,
}));
jest.mock('../../canvas/graphDetail', () => ({ useGraphDetail: () => 'inspection' }));
jest.mock('../../canvas/NodePort', () => ({ NodePorts: () => null }));
jest.mock('../../canvas/NodeSourceControls', () => ({ NodeSourceLink: () => null }));
jest.mock('../../canvas/graphQueryState', () => ({
  graphSearchResult: { get: () => ({}) }, searchStateFor: () => undefined,
}));

it.each(['layer', 'summary'])('%s card resolves the current trace instead of cached dimensions', renderer => {
  const node = { id: 'fc1', type: 'layerNode', position: { x: 0, y: 0 }, data: {
    layerType: { id: 'linear', name: 'Linear', category: 'linear', parameters: [] },
    parameters: { _shape_provenance: 'observed', _in_shape: '3x4', _out_shape: '3x4' },
    inputShape: [2, 4], outputShape: [2, 4], shapeEvidence: 'measured',
    outputTensor: { shape: [3, 4], dtype: 'float32', logicalBytes: 48 },
  } } as ModelNode;
  const util = new LayerShapeUtil();
  (util as any).editor = { getSelectedShapeIds: () => [], getHoveredShapeId: () => null };
  const card = () => renderer === 'layer' ? util.component({ id: 'shape:fc1', props: { nodeData: node.data, w: 240, h: 140 } } as any)
    : <TensorSummary data={node.data} />;
  const view = render(card());
  expect(view.container.textContent).toContain('(3, 4)');
  expect(view.container.textContent).not.toContain('(2, 4)');
  node.data = { ...node.data, parameters: { _shape_provenance: 'observed' } }; // incomplete replacement
  view.rerender(card());
  expect(view.container.textContent).not.toContain('3, 4');
  expect(view.container.textContent).not.toContain('2, 4');
});
