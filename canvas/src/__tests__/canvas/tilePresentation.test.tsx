import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { InputShapeUtil } from '../../canvas/InputShapeUtil';
import { OutputShapeUtil } from '../../canvas/OutputShapeUtil';
import { LayerShapeUtil } from '../../canvas/LayerShapeUtil';
import { layerCardHeight } from '../../canvas/cardMetrics';
import { CapturePort, ModelNodeData } from '../../types';
import fixture from '../../test/fixtures/canvas/capture-document.json';

jest.mock('tldraw', () => ({
  BaseBoxShapeUtil: class {}, HTMLContainer: ({ children }: any) => <div>{children}</div>,
  T: { number: {}, any: {} }, track: (component: any) => component,
}));
let mockDetail = 'overview';
jest.mock('../../canvas/graphDetail', () => ({ useGraphDetail: () => mockDetail }));
jest.mock('../../canvas/NodePort', () => ({ NodePorts: () => null }));
jest.mock('../../canvas/NodeSourceControls', () => ({ NodeEditButton: () => null, NodeSourceLink: () => null }));
jest.mock('../../canvas/graphQueryState', () => ({
  graphSearchResult: { get: () => ({}) }, searchStateFor: () => undefined,
  graphOverlayScale: { get: () => ({ values: new Map() }) },
}));
jest.mock('../../ide/projection', () => ({ currentProjection: () => null, reportNodeClick: jest.fn(), reportProjectExpand: jest.fn() }));

const data = (extra: Partial<ModelNodeData> = {}): ModelNodeData => ({
  layerType: { id: 'linear', name: 'Linear', displayName: 'Linear', category: 'linear', icon: 'Box', description: '', parameters: [] },
  parameters: {}, inputShape: [1, 16], outputShape: [1, 10], shapeEvidence: 'measured', ...extra,
});

function renderCard(util: InputShapeUtil | OutputShapeUtil | LayerShapeUtil, nodeData: ModelNodeData) {
  (util as any).editor = { getSelectedShapeIds: () => [], getHoveredShapeId: () => null };
  return render(util.component({ id: 'shape:tile', props: { nodeData, w: 240, h: 140 } } as any));
}

describe.each(['overview', 'working', 'inspection'])('%s zoom', detail => {
  beforeEach(() => { mockDetail = detail; });

  it('shows what a nested conditional selection does with its exact callable', () => {
    renderCard(new LayerShapeUtil({} as any), data({
      layerType: { ...data().layerType, id: 'python', name: 'Python' },
      captureLabel: 'python@DinoVisionTransformer#2',
      parameters: { _raw_call: 'torch.where(masks.unsqueeze(-1), token.to(x.dtype).unsqueeze(0), x)',
        _source_calls: ['unsqueeze', 'to', 'unsqueeze', 'where'],
        _source_result_call: { name: 'where', target: 'torch.where' } },
    }));
    expect(screen.getByText('Select by condition')).toBeVisible();
    expect(screen.getByText('torch.where', { exact: true })).toBeVisible();
    expect(screen.queryByText('python@DinoVisionTransformer#2')).not.toBeInTheDocument();
    expect(screen.queryByText(/unsqueeze →/)).not.toBeInTheDocument();
  });

  it.each([InputShapeUtil, OutputShapeUtil, LayerShapeUtil])('labels changed inputs consistently on %p', Util => {
    const { container } = renderCard(new Util({} as any), data({ diffStatus: 'rewired', review: {
      status: 'rewired', params: {}, inputChanges: { added: 1, removed: 1 },
      summary: 'Input connections changed (+1 / −1). Source and parameters unchanged.',
    } }));
    const badge = screen.getByLabelText('Graph change: inputs changed');
    expect(badge).toHaveTextContent('inputs changed');
    expect(badge).toHaveAttribute('title', expect.stringContaining('Source and parameters unchanged'));
    expect(container.querySelector('.diff-rewired')).not.toBeNull();
    expect(screen.queryByText(/modified/)).not.toBeInTheDocument();
  });

  it('keeps both reasons visible when parameters and inputs change', () => {
    renderCard(new LayerShapeUtil({} as any), data({ diffStatus: 'modified', review: {
      status: 'modified', params: { out_features: { before: 8, after: 10 } }, inputChanges: { added: 1, removed: 1 },
      summary: 'out_features · Input connections changed (+1 / −1)',
    } }));
    expect(screen.getByLabelText('Graph change: modified + inputs')).toHaveTextContent('~ modified + inputs');
  });

  it.each([false, true])('renders source operations with the visualization vocabulary (review=%s)', review => {
    const { container } = renderCard(new LayerShapeUtil(), data({
      layerType: { ...data().layerType, id: 'permute', name: 'torch.permute', displayName: 'torch.permute' },
      captureLabel: 'permute@DinoVisionTransformer#0', parameters: { dims: [0, 3, 1, 2] },
      inputShape: undefined, outputShape: undefined, shapeEvidence: 'unknown',
      ...(review ? { review: { status: 'unchanged' as const, params: {} } } : {}),
    }));
    expect(container.querySelector('.ui2-node-title')).toHaveTextContent('Permute axes');
    expect(screen.queryByText('permute@DinoVisionTransformer#0')).not.toBeInTheDocument();
    expect(container.querySelector('.ui2-node-expr')).toBeNull();
    if (detail === 'inspection') expect(screen.getByText('(0, 3, 1, 2)')).toBeVisible();
  });

  it('shows symbolic source shapes on review cards without claiming measured shapes', () => {
    const { container } = renderCard(new LayerShapeUtil(), data({
      layerType: { ...data().layerType, id: 'reshape', name: 'torch.reshape' },
      parameters: { _source_arguments: { shape: '(1, M, M, dim)' } },
      inputShape: undefined, outputShape: undefined, shapeEvidence: 'unknown',
      review: { status: 'unchanged', params: {} },
    }));
    expect(container.querySelector('.ui2-node-title')).toHaveTextContent('Reshape');
    expect(container.querySelectorAll('.ui2-node-tensor-row')).toHaveLength(0);
    if (detail === 'inspection') expect(screen.getByText('(1, M, M, dim)')).toBeVisible();
  });

  it.each(['input', 'output'])('keeps the %s boundary shape and dtype visible', kind => {
    const port = fixture.semantic.nodes.flatMap(node => node.metadata.capturePorts ?? [])[0];
    renderCard(kind === 'input' ? new InputShapeUtil() : new OutputShapeUtil(), data({
      capturePorts: [{ ...port, shape: [1, 16], dtype: 'torch.float32' }] as ModelNodeData['capturePorts'],
    }));
    expect(screen.getByTitle('(1, 16) · float32')).toBeVisible();
    expect(screen.getByTitle('(1, 16) · float32')).toHaveTextContent('(1, 16) · float32');
  });

  it('keeps operation shapes visible without routine measured badges', () => {
    const { container } = renderCard(new LayerShapeUtil(), data());
    const rows = container.querySelectorAll('.ui2-node-tensor-row, .graph-shape-flow > span');
    expect(rows).toHaveLength(2);
    expect(rows[0]).toBeVisible();
    expect(rows[1]).toBeVisible();
    expect(rows[0].textContent?.replace(/\s/g, '')).toContain('1,16');
    expect(rows[1].textContent?.replace(/\s/g, '')).toContain('1,10');
    expect(screen.queryByText(/^measured$/i)).not.toBeInTheDocument();
  });

  it('renders every captured input at this zoom even when the single-input summary is absent', () => {
    const captured = fixture.semantic.nodes.find(node => node.metadata.captureRef.semanticId === 'add2')!;
    renderCard(new LayerShapeUtil(), data({
      ...captured.metadata, capturePorts: captured.metadata.capturePorts as ModelNodeData['capturePorts'],
      captureRef: captured.metadata.captureRef as ModelNodeData['captureRef'],
      layerType: data().layerType, parameters: { ...captured.params,
        _operation_display: { options: { dim: '0', start: '0', end: '9223372036854775807' } } },
      sourceLocation: { file: 'models/tiny_transformer.py', line: 25 },
    }));
    const rows = screen.getByRole('group', { name: 'Operation input and output shapes' });
    expect(rows.querySelectorAll('[data-direction="in"]')).toHaveLength(2);
    expect(rows.querySelectorAll('[data-direction="out"]')).toHaveLength(1);
    expect(rows).toHaveTextContent('(2, 4)');
    expect(screen.queryByRole('group', { name: 'Operation arguments' })).not.toBeInTheDocument();
    expect(screen.queryByText('9223372036854775807')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'models/tiny_transformer.py:25' })).not.toBeInTheDocument();
  });

  it.each([['aten.mean.dim', 'Mean'], ['aten.view.default', 'Reshape'], ['aten.bmm.default', 'Batch matrix multiply']])
  ('renames the actual %s tile without an explanation panel', (target, title) => {
    const { container } = renderCard(new LayerShapeUtil(), data({
      captureRef: { kind: 'operation' } as ModelNodeData['captureRef'], captureLabel: target,
    }));
    expect(container.querySelector('.ui2-node-title')).toHaveTextContent(title);
    expect(screen.queryByText(target)).not.toBeInTheDocument();
    expect(container.querySelector('.ui2-node-expr')).toBeNull();
  });

  it('labels a layout group and opens its original operations locally', () => {
    const reveal = jest.fn();
    renderCard(new LayerShapeUtil(), data({ operationGroup: { members: ['a', 'tile'], labels: ['permute', 'view'],
      inputShape: [2, 8], outputShape: [8, 2] }, onRevealOperations: reveal }));
    expect(screen.getByText('Tensor layout')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Show 2 operations' }));
    expect(reveal).toHaveBeenCalledWith('tile');
  });

  it('renames an older graph tile even without capture metadata', () => {
    const target = 'aten.max_pool2d_with_indices.default';
    const { container } = renderCard(new LayerShapeUtil(), data({
      layerType: { ...data().layerType, name: target, displayName: target },
    }));
    expect(container.querySelector('.ui2-node-title')).toHaveTextContent('Max pool 2D');
    expect(screen.queryByText(target)).not.toBeInTheDocument();
  });
});

it('retains stale attribution on an overview operation without drawing its source note', () => {
  mockDetail = 'overview';
  renderCard(new LayerShapeUtil(), data({ shapeEvidence: 'stale', resolutionDetail: 'Check the input width' }));
  expect(screen.getAllByText('stale').length).toBeGreaterThan(0);
  expect(screen.queryByText('Check the input width')).not.toBeInTheDocument();
});

// Static-analysis notes belong to the footer's source-notes list. A sentence inside the
// card was clipped by the card's own height and pushed the card's content aside.
describe.each(['overview', 'working', 'inspection'])('source notes at %s zoom', detail => {
  const note = 'transformers is external to the workspace; its classes stay opaque on the static graph unless they are canonical framework operations';
  beforeAll(() => { HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); }; });
  beforeEach(() => { mockDetail = detail; });

  it('keeps an external-package note off the card, leaving it on the status tooltip', () => {
    const { container } = renderCard(new LayerShapeUtil(), data({ debugEvidence: 'source',
      resolutionCode: 'EXTERNAL_PACKAGE', resolutionSeverity: 'info', resolutionDetail: note }));
    expect(container).not.toHaveTextContent('external to the workspace');
    expect(container.querySelector(`[title="${note}"]`)).toHaveTextContent('Source only');
  });

  it('flags a resolution warning with the issue badge instead of a sentence', () => {
    const warning = 'models.blocks is a workspace module but is not in the supplied source set (truncated set or missing file)';
    const { container } = renderCard(new LayerShapeUtil(), data({
      resolutionCode: 'UNRESOLVED_IMPORT', resolutionSeverity: 'warning', resolutionDetail: warning }));
    expect(container).not.toHaveTextContent('not in the supplied source set');
    fireEvent.click(screen.getByLabelText('1 warning in tile'));
    expect(screen.getByRole('dialog')).toHaveTextContent(warning);
  });
});

it('measures a card with a source note exactly like the same card without one', () => {
  const base = data({ parameters: { in_features: 16, out_features: 10 } });
  const noted = { ...base, resolutionSeverity: 'info' as const, resolutionDetail: 'torchvision is external to the workspace' };
  expect(layerCardHeight(noted)).toBe(layerCardHeight(base));
  const port = (direction: 'in' | 'out') => ({ id: direction, nodeId: 'tile', direction, shape: [1, 16], dtype: 'torch.float32',
    device: 'cpu', valueId: direction, memberPortId: null, argumentKind: 'value',
    paths: direction === 'in' ? [['arguments', 'args', 0]] : [['output']] }) as unknown as CapturePort;
  const operation = data({ captureRef: { captureId: 'run', kind: 'operation', operationId: 'op:0' } as ModelNodeData['captureRef'],
    capturePorts: [port('in'), port('out')] });
  expect(layerCardHeight({ ...operation, resolutionDetail: 'torchvision is external to the workspace' })).toBe(layerCardHeight(operation));
});
