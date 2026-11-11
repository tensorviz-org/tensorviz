import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import fixture from '../../test/fixtures/canvas/nestedAssignmentCapture.json';
import { ModuleShapeUtil } from '../../canvas/ModuleShapeUtil';
import { moduleCardHeight, moduleCardHeader } from '../../canvas/cardMetrics';
import { buildGraphReview } from '../../ide/graphReview';
import { draftAssignmentReview } from '../../ide/capturedGraphReview';
import { assignmentContext } from '../../ide/assignmentContext';
import { reportNodeClick } from '../../ide/projection';
import type { GraphReviewState, ProjectionPayload } from '../../ide/protocol.generated';
import type { ModelNodeData } from '../../types';

jest.mock('tldraw', () => ({
  BaseBoxShapeUtil: class {}, HTMLContainer: ({ children }: any) => <div>{children}</div>,
  T: { number: {}, any: {} }, track: (component: any) => component,
}));
jest.mock('../../canvas/NodePort', () => ({ NodePorts: () => null }));
let mockDetail = 'inspection';
jest.mock('../../canvas/graphDetail', () => ({ useGraphDetail: () => mockDetail }));
jest.mock('../../canvas/graphAccess', () => ({ graphIsReadOnly: () => true }));
const KEY = '@assignment:blocks[0].relu';
jest.mock('../../ide/projection', () => ({ localBrowserMode: () => true,
  currentProjection: () => ({ spanmap: { '@assignment:blocks[0].relu': { file: 'model.py', start: [15, 8] } } }),
  showingProjectGraph: () => false, reportProjectExpand: jest.fn(), reportNodeClick: jest.fn() }));
afterEach(() => { mockDetail = 'inspection'; (reportNodeClick as jest.Mock).mockClear(); });

const nodes = (fixture.capture.document as any).semantic.nodes as any[];
const id = (label: string) => nodes.find(node => node.metadata.captureLabel === label).node_id as string;
const state = { id: 'draft', proposal: { id: 'draft' }, beforeLabel: 'Current code', afterLabel: 'Proposed code',
  before: fixture.before, after: fixture.after, changedFiles: ['model.py'], stale: false,
  liveContext: { workspaceFolder: '/ws', file: 'model.py', cls: 'Net' } } as GraphReviewState;
const draftNode = (nodeId: string): ModelNodeData => draftAssignmentReview(state, buildGraphReview(state), fixture.capture as ProjectionPayload)!
  .projections.diff.reviewGraph.nodes.find(node => node.id === nodeId)!.data;
const card = (nodeId: string, data: ModelNodeData) => {
  const util = new ModuleShapeUtil({} as any);
  (util as any).editor = { getSelectedShapeIds: () => [], getHoveredShapeId: () => null };
  return render(util.component({ id: `shape:${nodeId}`, props: { nodeData: data, w: 240, h: 200 } } as any));
};

test.each(['overview', 'working', 'inspection'])('at %s zoom a replaced call reads ReLU → nn.GELU, keeps its call ordinal and says Not run', tier => {
  mockDetail = tier;
  const call = id('blocks.0.relu · call 2/3');
  const view = card(call, draftNode(call));
  expect(view.container.querySelector('.layer-name .graph-review-value del')).toHaveTextContent(/^ReLU$/);
  expect(view.container.querySelector('.layer-name .graph-review-value ins')).toHaveTextContent(/^nn\.GELU$/);
  expect(view.container.querySelector('.layer-id')).toHaveTextContent(/^blocks\.0\.relu · previous call 2\/3$/);
  expect(screen.getByText('Source assignment · Not run')).toBeVisible();
  expect(screen.getByText(/Affected calls unverified/)).toBeVisible();
  expect(view.container.querySelector('.layer-name')).toHaveAttribute('title', expect.stringContaining('which calls change is unverified until validation'));
  // The old ReLU's measured interface and contents are not shown as GELU's.
  expect(screen.queryByRole('group', { name: 'Input and output shapes' })).toBeNull();
  expect(view.container.textContent).not.toMatch(/\(2, 4\)|float32|Previous run|Passed|Ready/);
  expect(screen.queryByRole('button', { name: /Expand|Reset module layout/ })).toBeNull();
});

test('its source link opens the assignment line, not the ReLU library line', () => {
  const call = id('blocks.0.relu · call 1/3');
  card(call, draftNode(call));
  fireEvent.click(screen.getByRole('button', { name: 'model.py:15' }));
  expect(reportNodeClick).toHaveBeenCalledWith(KEY);
});

test('after Accept the same call card remains, without the draft badge', () => {
  const call = id('blocks.0.relu · call 3/3');
  const data = assignmentContext(fixture.capture.document, fixture.accepted.document)!.graph.nodes.find(node => node.id === call)!.data;
  const view = card(call, data);
  expect(view.container.querySelector('.layer-name .graph-review-value')).toHaveTextContent('ReLU→nn.GELU');
  expect(screen.getByText('Source assignment · Not run')).toBeVisible();
  expect(screen.getByText(/Affected calls unverified/)).toBeVisible();
  expect(screen.queryByLabelText(/^Graph change/)).toBeNull();
});

test('surrounding captured calls say Previous run, never current', () => {
  const linear = id('blocks.0.fc');
  card(linear, draftNode(linear));
  expect(screen.getByText('Previous run')).toBeVisible();
  expect(screen.queryByText('Not run')).toBeNull();
});


test('partial coverage is disclosed without claiming the source callable ran', () => {
  const call = id('blocks.0.relu · call 1/3');
  const data = draftNode(call);
  const view = card(call, { ...data, sourceReplacement: { ...data.sourceReplacement!, coverage: 'partial' } });
  expect(screen.getByText('Source assignment · Not run')).toBeVisible();
  expect(screen.getByText(/Affected calls unverified/)).toBeVisible();
  expect(screen.getByText(/Partial source coverage/)).toBeVisible();
  expect(view.container.querySelector('.layer-name')).toHaveAttribute('title', expect.stringContaining('Dependency coverage is partial'));
  expect(view.container.querySelector('.layer-name')).toHaveAttribute('title', expect.stringContaining('Previous run: ReLU'));
});

test.each(['complete', 'partial'] as const)('%s assignment cards reserve space for uncertainty and the source action', coverage => {
  const data = draftNode(id('blocks.0.relu · call 1/3'));
  const replaced = { ...data, sourceReplacement: { ...data.sourceReplacement!, coverage } };
  // 14px per visible uncertainty line and an 8px section gap. The source action
  // was clipped in the installed app when this section had no layout budget.
  // The rest of this fixture needs 131px, including the source action. Removing
  // sourceReplacement for a comparison would reintroduce the old tensor rows.
  expect(moduleCardHeight(replaced)).toBe(131 + (coverage === 'partial' ? 28 : 14) + 8 + moduleCardHeader(replaced).height - 56);
});
