import React from 'react';
import { render, screen } from '@testing-library/react';
import { InputShapeUtil } from '../../canvas/InputShapeUtil';
import { OutputShapeUtil } from '../../canvas/OutputShapeUtil';
import { ioCardContent, ioCardHeight, IO_CARD_HEIGHT, nodeCardSize } from '../../canvas/cardMetrics';
import { cardHeaderLayout } from '../../canvas/cardIdentity';
import type { InputShape, OutputShape } from '../../canvas/types';
import type { CapturePort, ModelNode, ModelNodeData } from '../../types';

jest.mock('tldraw', () => ({
  BaseBoxShapeUtil: class {}, HTMLContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  T: { number: {}, any: {} }, track: (component: unknown) => component,
}));
jest.mock('../../canvas/NodePort', () => ({ NodePorts: () => null }));
jest.mock('../../canvas/graphDetail', () => ({ useGraphDetail: () => { throw new Error('I/O cards must not change with zoom.'); } }));

const dataFor = (kind: 'input' | 'output', changes: Partial<ModelNodeData> = {}): ModelNodeData => ({
  layerType: { id: kind, name: kind, displayName: kind, category: 'io', description: '', icon: 'Box', parameters: [] },
  parameters: {}, ...changes,
});
const port = (id: string): CapturePort => ({
  id, nodeId: 'result', direction: 'in', argumentKind: 'value', paths: [['output']], shape: [2, 16],
  dtype: 'torch.float32', device: 'cpu', memberPortId: null, valueId: null,
});

function card(kind: 'input' | 'output', data: ModelNodeData, zoom = 1) {
  const util = kind === 'input' ? new InputShapeUtil() : new OutputShapeUtil();
  (util as unknown as { editor: object }).editor = { getSelectedShapeIds: () => [], getEfficientZoomLevel: () => zoom };
  const shape = { id: 'shape:root', props: { nodeData: data, w: 200, h: ioCardHeight(data, kind) } };
  return kind === 'input' ? (util as InputShapeUtil).component(shape as InputShape) : (util as OutputShapeUtil).component(shape as OutputShape);
}

it('renders an empty output as a header, with no reserved body or synthetic node id', () => {
  const data = dataFor('output');
  const view = render(card('output', data));
  expect(view.container.querySelector('.ui2-node-body')).toBeNull();
  expect(view.container.querySelector('.ui2-node-subtitle')).toBeNull();
  expect(view.container.querySelector('.output-node')).toHaveStyle({ height: `${cardHeaderLayout('Output').height + 2}px` });
  expect(view.container).not.toHaveTextContent('#root');
});

it.each(['input', 'output'] as const)('retains the explicit empty captured %s interface without extra blank space', kind => {
  const data = dataFor(kind, { capturePorts: [] });
  const view = render(card(kind, data));
  expect(screen.getByText('No boundary value recorded')).toBeVisible();
  expect(view.container.querySelectorAll('.ui2-node-list')).toHaveLength(1);
  expect(view.container).not.toHaveTextContent('Not recorded');
  expect(ioCardHeight(data)).toBe(86);
  expect(ioCardContent(data).rowCount).toBe(1);
});

it('keeps a truthful unknown-shape row for an untraced input', () => {
  const data = dataFor('input');
  render(card('input', data));
  expect(screen.getByText('Not recorded')).toBeVisible();
  expect(ioCardHeight(data)).toBe(IO_CARD_HEIGHT);
  expect(screen.queryByText('(1, 3, 224, 224)')).not.toBeInTheDocument();
});

it.each([
  [dataFor('input', { parameters: { shape: '2,16' } }), 86],
  [dataFor('input', { parameters: { shape: '2,16', name: 'features' } }), 106],
  [dataFor('input', { parameters: { shape: '2,16', _input_provenance: 'seed-from-first-layer' } }), 106],
  [dataFor('input', { parameters: { shape: '2,16', name: 'features', _input_provenance: 'seed-from-first-layer' } }), 127],
  [dataFor('output', { parameters: { name: 'scores' } }), 86],
  [dataFor('output', { outputShape: [2, 16] }), 86],
  [dataFor('output', { outputShape: [2, 16], parameters: { name: 'scores' } }), 106],
  [dataFor('output', { capturePorts: [port('a')] }), 105],
  [dataFor('output', { capturePorts: [port('a'), port('b')] }), 145],
] as Array<[ModelNodeData, number]>)('sizes only the actual interface rows (%j)', (data, height) => {
  const kind = data.layerType.id as 'input' | 'output';
  const view = render(card(kind, data));
  expect(ioCardHeight(data)).toBe(height);
  expect(view.container.querySelector('.ui2-node-card')).toHaveStyle({ height: `${height}px` });
});

it.each([0.2, 1, 3])('keeps the same output content at zoom %s', zoom => {
  const data = dataFor('output', { capturePorts: [port('a')] });
  const view = render(card('output', data, zoom));
  expect(view.container.querySelector('.output-node')).toHaveAttribute('data-detail', 'compact');
  expect(screen.getByTitle('(2, 16) · float32')).toBeVisible();
  expect(screen.getByTitle('(2, 16) · float32').textContent).toBe('(2, 16) · float32');
  expect(view.container.querySelectorAll('.capture-boundary-row')).toHaveLength(1);
});

it.each(['input', 'output'] as const)('accounts for every error line on the %s card', kind => {
  const error = 'The recorded root interface could not be captured because a required tensor value was unavailable.'.repeat(2);
  const data = dataFor(kind, { error });
  const content = ioCardContent(data);
  const view = render(card(kind, data));
  const failure = view.container.querySelector('.graph-failure-facts')!;
  expect(failure.textContent).toBe(error);
  expect(failure.children).toHaveLength(content.errorLines.length);
  expect(content.errorLines.length).toBeGreaterThan(1);
  expect(view.container.querySelector('.ui2-node-error')).toHaveStyle({ height: `${content.errorHeight}px` });
  if (kind === 'output') expect(view.container.querySelector('.ui2-node-list')).toBeNull();
});

it('wraps the complete recorded interface label and grows the matching geometry', () => {
  const captureLabel = 'outputs.long_project_specific_dictionary_key_for_candidate_representation_scores';
  const data = dataFor('output', { captureLabel });
  const view = render(card('output', data));
  const label = view.container.querySelector('.ui2-node-subtitle')!;
  expect(label.textContent).toBe(captureLabel);
  expect(label.children.length).toBeGreaterThan(1);
  expect(ioCardHeight(data)).toBe(cardHeaderLayout('Output', captureLabel).height + 2);
  expect(view.container.querySelector('.ui2-node-body')).toBeNull();
});

it('uses graph node kind when an older output snapshot carries a mismatched layer type', () => {
  const data = dataFor('input');
  const node: ModelNode = { id: 'result', type: 'outputNode', position: { x: 0, y: 0 }, data };
  const view = render(card('output', data));
  expect(nodeCardSize(node)).toEqual({ w: 200, h: 53 });
  expect(view.container.querySelector('.ui2-node-body')).toBeNull();
});

it('measures reviewed I/O identity against its actual wider card', () => {
  const data = dataFor('output', { captureLabel: 'outputs.long_project_specific_dictionary_key_for_candidate_scores',
    review: { status: 'modified', params: {} } });
  const node: ModelNode = { id: 'result', type: 'outputNode', position: { x: 0, y: 0 }, data };
  expect(nodeCardSize(node)).toEqual({ w: 280, h: cardHeaderLayout('Output', data.captureLabel, 280).height + 2 });
  expect(nodeCardSize(node).h).toBeLessThan(ioCardHeight(data, 'output', 200));
});
