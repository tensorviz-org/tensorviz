import React from 'react';
import { render, screen } from '@testing-library/react';
import { LayerShapeUtil } from '../../canvas/LayerShapeUtil';
import { layerCardHeader, nodeCardSize } from '../../canvas/cardMetrics';
import type { LayerShape } from '../../canvas/types';
import type { ModelNode, ModelNodeData } from '../../types';

jest.mock('tldraw', () => ({
  BaseBoxShapeUtil: class {}, HTMLContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  T: { number: {}, any: {} }, track: (component: unknown) => component,
}));
jest.mock('../../canvas/graphDetail', () => ({ useGraphDetail: () => { throw new Error('Node content must not depend on zoom.'); } }));
jest.mock('../../canvas/graphQueryState', () => ({ graphSearchResult: { get: () => ({}) }, searchStateFor: () => undefined }));
jest.mock('../../canvas/NodePort', () => ({ NodePorts: () => null }));
jest.mock('../../ide/projection', () => ({ localBrowserMode: () => true, currentProjection: () => ({}), reportNodeClick: jest.fn(), reportProjectExpand: jest.fn() }));

const name = 'WNCausalConvolutionWithProjectSpecificTemporalConditioningAndLongParameterNames';
const args = Object.fromEntries(Array.from({ length: 200 }, (_, index) => [`constructor_option_${index}`, index]));
const dataFor = (id: string, parameters: Record<string, unknown>): ModelNodeData => ({
  layerType: { id, name, displayName: name, category: 'misc', description: '', icon: 'Box', parameters: [] },
  parameters, debugEvidence: 'source', shapeEvidence: 'unknown',
});

function card(data: ModelNodeData, zoom: number) {
  const util = new LayerShapeUtil();
  (util as unknown as { editor: object }).editor = {
    getSelectedShapeIds: () => [], getHoveredShapeId: () => null, getEfficientZoomLevel: () => zoom,
  };
  const node: ModelNode = { id: 'source-call', type: 'layerNode', position: { x: 0, y: 0 }, data };
  const shape = { id: 'shape:source-call', props: { nodeData: data, ...nodeCardSize(node) } } as LayerShape;
  return util.component(shape);
}

it.each(['linear', 'custom_code'])('keeps the full %s identity and the same DOM at every zoom, even with 200 arguments', id => {
  const parameters = id === 'custom_code' ? { _class: name, ...args } : args;
  const data = dataFor(id, parameters);
  const original = JSON.stringify(data);
  const view = render(card(data, 0.2));
  const first = view.container.innerHTML;
  for (const zoom of [1, 3]) {
    view.rerender(card(data, zoom));
    expect(view.container.innerHTML).toBe(first);
  }
  const title = view.container.querySelector('.ui2-node-title')!;
  expect(title.textContent).toBe(name);
  expect(title.children.length).toBeGreaterThan(1);
  expect(title.textContent).not.toContain('…');
  expect(view.container.querySelector('.ui2-node-list')).toBeNull();
  expect(view.container.querySelector('.ui2-node-body')).toBeNull();
  expect(view.container).not.toHaveTextContent('constructor_option_');
  expect(view.container).not.toHaveTextContent('more params');
  expect(JSON.stringify(data)).toBe(original);
});

it('reserves wrapped identity height rather than constructor argument height', () => {
  const baseline = dataFor('custom_code', { _class: name });
  const many = { ...baseline, parameters: { ...baseline.parameters, ...args } };
  const size = (data: ModelNodeData) => nodeCardSize({ id: 'call', type: 'layerNode', position: { x: 0, y: 0 }, data });
  expect(size(many)).toEqual(size(baseline));
  expect(size(many).h).toBe(layerCardHeader(many).height + 2);
  expect(size(many).h).toBeGreaterThan(53);
});

it('keeps a full custom function callable and recorded tensor shapes at every zoom', () => {
  const callable = 'self.compute_candidate_representation_scores_for_temporally_conditioned_questions';
  const data = dataFor('python', { ...args, _raw_call: `${callable}(features, question)`,
    _source_result_call: { name: callable.split('.').pop() } });
  data.inputShape = [2, 16]; data.outputShape = [2, 8]; data.shapeEvidence = 'measured';
  data.debugEvidence = undefined;
  const view = render(card(data, 0.2));
  const first = view.container.innerHTML;
  const technicalName = view.container.querySelector('.ui2-node-title')!;
  expect(technicalName.textContent).toBe(callable.replace(/^self\./, ''));
  expect(technicalName.children.length).toBeGreaterThan(1);
  expect(view.container.querySelector('[data-direction="in"]')).toHaveTextContent('(2, 16)');
  expect(view.container.querySelector('[data-direction="out"]')).toHaveTextContent('(2, 8)');
  view.rerender(card(data, 3));
  expect(view.container.innerHTML).toBe(first);
  expect(screen.queryByText('Not recorded')).not.toBeInTheDocument();
  expect(view.container.querySelector('.ui2-node-list')).toBeNull();
});
