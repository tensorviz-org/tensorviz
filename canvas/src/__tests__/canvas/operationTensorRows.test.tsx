import React from 'react';
import { render, screen } from '@testing-library/react';
import { TensorSummary } from '../../canvas/TensorSummary';
import { layerCardHeight } from '../../canvas/cardMetrics';
import { ModelNodeData, CapturePort } from '../../types';

const port = (id: string, direction: 'in' | 'out', shape: number[] | null, index = 0): CapturePort => ({
  id, nodeId: 'op', direction, shape, dtype: 'torch.float32', device: 'cpu',
  valueId: id, memberPortId: null, argumentKind: 'value',
  paths: direction === 'in' ? [['arguments', 'args', index]] : [['output']],
});
const data = (ports: CapturePort[], extra: Partial<ModelNodeData> = {}): ModelNodeData => ({
  layerType: { id: 'captured_operation', name: 'aten.addmm.default', displayName: 'Matrix multiply + add',
    category: 'misc', icon: 'Box', description: '', parameters: [] },
  captureRef: { version: 1, captureId: 'capture', sourceId: 'source', kind: 'operation', semanticId: 'op' },
  capturePorts: ports, shapeEvidence: 'measured', parameters: { _shape_provenance: 'observed', _out_shape: [8, 96] }, ...extra,
});

it('shows every measured operand when a matrix operation has no single input summary', () => {
  render(<TensorSummary data={data([port('bias', 'in', [96]), port('x', 'in', [8, 32], 1),
    port('weight', 'in', [32, 96], 2), port('result', 'out', [8, 96])])} />);
  const rows = screen.getByRole('group', { name: 'Operation input and output shapes' });
  expect(rows.querySelectorAll('[data-direction="in"]')).toHaveLength(3);
  for (const value of ['(96)', '(8, 32)', '(32, 96)', '(8, 96)']) expect(rows).toHaveTextContent(value);
});

it('keeps operation operand numbering and uses readable input names in tooltips', () => {
  render(<TensorSummary data={data([port('bias', 'in', [96]), port('x', 'in', [8, 32], 1), port('result', 'out', [8, 96])])} />);
  const rows = screen.getByRole('group', { name: 'Operation input and output shapes' });
  expect(Array.from(rows.querySelectorAll('.operation-tensor-label'), label => label.textContent)).toEqual(['IN 0', 'IN 1', 'OUT']);
  expect(rows.querySelector('[data-direction="in"]')).toHaveAttribute('title', 'Input 1 · float32');
});

it('shows only tensor shapes even when old capture data contains operation options', () => {
  const literal = { ...port('size', 'in', null, 1), valueId: null, argumentKind: 'int' as const, dtype: null };
  render(<TensorSummary data={data([port('x', 'in', [8, 96]), literal, port('out', 'out', [8, 1, 96])], {
    parameters: { _shape_provenance: 'observed', _out_shape: [8, 1, 96],
      _operation_display: { argumentNames: ['self', 'size'], options: { size: '[8, 1, 96]' } } },
  })} />);
  expect(screen.getByRole('group', { name: 'Operation input and output shapes' }).querySelectorAll('[data-direction="in"]')).toHaveLength(1);
  expect(screen.queryByRole('group', { name: 'Operation arguments' })).not.toBeInTheDocument();
  expect(screen.queryByText('size')).not.toBeInTheDocument();
  expect(screen.getByText('(8, 1, 96)')).toBeVisible();
});

it('preserves repeated operands, scalar tensors, unknown shapes and multiple outputs', () => {
  const repeated = port('x', 'in', []);
  const output = port('tuple0', 'out', [0, 4]); output.paths = [['output', 'items', 0]];
  render(<TensorSummary data={data([repeated, { ...repeated, id: 'x-again', paths: [['arguments', 'args', 1]] },
    port('unknown', 'in', null, 2), output, { ...port('tuple1', 'out', []), paths: [['output', 'items', 1]] }],
    { parameters: { _shape_provenance: 'observed' } })} />);
  const rows = screen.getByRole('group', { name: 'Operation input and output shapes' });
  expect(rows.querySelectorAll('[data-direction="in"]')).toHaveLength(3);
  expect(rows.querySelectorAll('[data-direction="out"]')).toHaveLength(2);
  expect(rows).toHaveTextContent('()'); expect(rows).toHaveTextContent('(0, 4)');
  expect(rows).toHaveTextContent('Not recorded');
  expect(screen.queryByText('unknown')).not.toBeInTheDocument();
});

it('keeps the recorded input beside a failed operation without inventing an output', () => {
  render(<TensorSummary data={data([port('x', 'in', [8, 32])], {
    parameters: { _shape_provenance: 'failed' }, error: 'Invalid reshape',
    captureDiagnostic: { code: 'invalid_shape', actualShape: [8, 32], expectedShape: [8, 96] } as ModelNodeData['captureDiagnostic'],
  })} />);
  const rows = screen.getByRole('group', { name: 'Operation input and output shapes' });
  expect(rows).toHaveTextContent('(8, 32)');
  expect(rows.querySelectorAll('[data-direction="out"]')).toHaveLength(0);
  expect(screen.getByText('Incompatible shape')).toBeVisible();
});

it('retains stale attribution and sizes cards for all input rows', () => {
  const one = data([port('x', 'in', [8, 32]), port('y', 'out', [8, 96])]);
  const many = data([port('x', 'in', [8, 32]), port('w', 'in', [32, 96], 1), port('b', 'in', [96], 2), port('y', 'out', [8, 96])], { shapeEvidence: 'stale' });
  render(<TensorSummary data={many} />);
  expect(screen.getByText('stale')).toBeVisible();
  expect(layerCardHeight(many)).toBeGreaterThanOrEqual(layerCardHeight(one) + 54);
});
