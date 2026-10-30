import React from 'react';
import { render, screen } from '@testing-library/react';
import { LayerShapeUtil } from '../../canvas/LayerShapeUtil';
import { nodeCardSize } from '../../canvas/cardMetrics';
import type { DetailTier } from '../../canvas/detailTier';
import { NodePorts } from '../../canvas/NodePort';
import { NodeSourceLink } from '../../canvas/NodeSourceControls';
import { sourceAssignmentCard } from '../../ide/sourceAssignmentCards';
import type { SourceAssignmentFact } from '../../ide/sourceAssignmentCards';

jest.mock('tldraw', () => ({
  BaseBoxShapeUtil: class {}, HTMLContainer: ({ children }: any) => <div>{children}</div>,
  T: { number: {}, any: {} }, track: (component: any) => component,
  atom: (_name: string, value: unknown) => ({ get: () => value, set: () => undefined }),
  useValue: (value: { get(): unknown }) => value.get(), useEditor: () => ({}),
}));
let mockDetail: DetailTier = 'inspection';
jest.mock('../../canvas/graphDetail', () => ({ useGraphDetail: () => mockDetail }));
jest.mock('../../canvas/graphQueryState', () => ({
  graphSearchResult: { get: () => ({}) }, searchStateFor: () => undefined,
}));
const mockSpanmap: Record<string, unknown> = {};
jest.mock('../../ide/projection', () => ({ localBrowserMode: () => true, currentProjection: () => ({ spanmap: mockSpanmap }),
  reportNodeClick: jest.fn(), reportProjectExpand: jest.fn() }));

const KEY = '@assignment:layer3[0].relu';
const fact: SourceAssignmentFact = { target: 'layer3[0].relu', digest: 'd', title: 'nn.GELU', value: 'nn.GELU()', ambiguous: false,
  conditional: true, span: { file: 'vision_models.py', start: [10, 8], end: [10, 42], kind: 'decl' } as SourceAssignmentFact['span'] };
const card = (data = sourceAssignmentCard(KEY, fact).data) => {
  const util = new LayerShapeUtil();
  (util as any).editor = { getSelectedShapeIds: () => [], getHoveredShapeId: () => null, markEventAsHandled: () => undefined };
  return util.component({ id: `shape:${KEY}`, props: { nodeData: data, ...nodeCardSize({ type: 'layerNode', data } as any) } } as any);
};

afterEach(() => { mockDetail = 'inspection'; for (const key of Object.keys(mockSpanmap)) delete mockSpanmap[key]; });

test('the live card says what was set, that it did not run, and offers no edit, ports or shapes', () => {
  const view = render(card());
  // The header is the name and the exact target, with nothing beside them to
  // squeeze either into an ellipsis; "not run" is a body row.
  expect(view.container.querySelector('.ui2-node-header')).toHaveTextContent(/^nn\.GELUsets layer3\[0\]\.relu · conditional$/);
  expect(screen.getByText('sets layer3[0].relu · conditional')).toHaveAttribute('title', 'sets layer3[0].relu · conditional');
  expect(screen.getByText('Set in source · not run').closest('.ui2-node-body')).not.toBeNull();
  expect(screen.getByText('nn.GELU()')).toBeVisible();
  expect(view.container.querySelector('[data-source-assignment]')).not.toBeNull();
  expect(screen.queryByRole('button', { name: /^Edit/ })).toBeNull();
  expect(view.container.querySelector('.model-node-port')).toBeNull();
  expect(view.container.textContent).not.toMatch(/Tensor shapes|Source only|calls|\bIn\b|\bOut\b/);
  // The host has no span for this display-only id: the line is named, not linked.
  expect(screen.queryByRole('button', { name: 'vision_models.py:10' })).toBeNull();
  expect(view.container.querySelector('.node-source-location[data-static]')).toHaveTextContent('vision_models.py:10');
});

test.each<DetailTier>(['overview', 'working', 'inspection'])('at %s zoom the card keeps its source body and never grows a tensor-shape section', tier => {
  mockDetail = tier;
  const view = render(card());
  expect(view.container.querySelector('.graph-tensor-summary')).toBeNull();
  expect(view.container.textContent).not.toMatch(/tensor shape|not recorded|unknown/i);
  expect(screen.getByText('nn.GELU()')).toBeVisible();
  expect(screen.getByText('Set in source · not run')).toBeVisible();
  expect(view.container.querySelector('.node-source-location')).toHaveTextContent('vision_models.py:10');
});

test('in a draft the card carries the amber modified badge and the value change', () => {
  const data = { ...sourceAssignmentCard(KEY, { ...fact, title: 'nn.SiLU', value: 'nn.SiLU()' }).data, diffStatus: 'modified' as const,
    review: { status: 'modified' as const, params: { value: { before: 'nn.GELU()', after: 'nn.SiLU()' } }, summary: 'Assignment value changed · not run' } };
  mockSpanmap[KEY] = fact.span;
  const view = render(card(data));
  expect(screen.getByLabelText('Graph change: modified')).toHaveAttribute('data-status', 'modified');
  // The badge says what changed; "Set in source" would misread a removal.
  expect(screen.queryByText('Set in source · not run')).toBeNull();
  expect(view.container.querySelector('.graph-review-value del')).toHaveTextContent('nn.GELU()');
  expect(view.container.querySelector('.graph-review-value ins')).toHaveTextContent('nn.SiLU()');
  // A review span routes the link through the existing reveal path.
  expect(screen.getByRole('button', { name: 'vision_models.py:10' })).toBeVisible();
});

test('the box reserves the evidence row and keeps the draft width after Accept', () => {
  const live = sourceAssignmentCard(KEY, fact);
  // 51 header + 2 borders + 16 padding, the source and evidence rows (16.5 + 7 gap each) and the 16.5 value row.
  expect(nodeCardSize(live)).toEqual({ w: 280, h: 133 });
  expect(nodeCardSize({ ...live, data: { ...live.data, sourceAssignment: undefined } }).w).toBe(200);
  expect(nodeCardSize({ ...live, data: { ...live.data, review: { status: 'modified', params: {} } } })).toEqual({ w: 280, h: 120 });
});

test('an assignment card has no connection handles, unlike a captured card', () => {
  const shape = (nodeData: object) => ({ id: 'shape:n', props: { nodeData, w: 200, h: 120 } }) as any;
  const port = { id: 'in0', direction: 'in', argumentKind: 'value', paths: [], shape: [1, 4], dtype: 'torch.float32' };
  const captured = render(<NodePorts shape={shape({ captureRef: {}, capturePorts: [port] })} />);
  expect(captured.container.querySelectorAll('[data-testid="capture-port"]')).toHaveLength(1);
  const assignment = render(<NodePorts shape={shape({ ...sourceAssignmentCard(KEY, fact).data, captureRef: {}, capturePorts: [port] })} />);
  expect(assignment.container.innerHTML).toBe('');
});

test('an ordinary node without a projected span still gets no source link', () => {
  const view = render(<NodeSourceLink nodeId="fc" data={{ parameters: {}, sourceLocation: { file: 'model.py', line: 3 } } as any} />);
  expect(view.container.innerHTML).toBe('');
});
