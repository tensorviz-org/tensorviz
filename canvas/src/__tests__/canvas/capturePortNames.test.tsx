import React from 'react';
import { render, screen } from '@testing-library/react';
import { CaptureBoundaryRows } from '../../canvas/CaptureBoundaryRows';
import { portLabel } from '../../canvas/capturePorts';
import { moduleBoundaryRows } from '../../services/modulePresentation';
import type { CapturePort, ModelNodeData } from '../../types';

const port = (paths: CapturePort['paths'], extra: Partial<CapturePort> = {}): CapturePort => ({
  id: 'p', nodeId: 'node', direction: 'in', paths, argumentKind: 'value', valueId: 'v', memberPortId: null,
  shape: [1, 4, 768], dtype: 'torch.float32', device: 'cpu', ...extra,
});

it.each([
  [[['inputs', 'args', 0]], 'out', 'Input 1'],
  [[['inputs', 'kwargs', 'attention_mask']], 'out', 'attention_mask'],
  [[['inputs', 'kwargs', 'items']], 'out', 'items'],
  [[['inputs', 'kwargs', 'entries']], 'out', 'entries'],
  [[['arguments', 'args', 1, 'items', 0]], 'in', 'Input 2[0]'],
  [[['output', 'items', 2]], 'in', 'Output 3'],
  [[['output', 'entries', 2, 'value']], 'in', 'Output 3'],
  [[['output']], 'in', 'Output'],
] as const)('keeps older path %j readable and uses its semantic direction', (paths, direction, expected) => {
  expect(portLabel(port(paths.map(path => [...path]), { direction }), 2)).toBe(expected);
});

it('uses dictionary keys consistently on model and Output cards without changing identity or shape', () => {
  const outputs = ['patch_tokens', 'cls_token', 'pooled', 'grid_size'].map((displayName, i) =>
    port([['output', 'entries', i, 'value']], { id: `out-${i}`, displayName }));
  const before = JSON.stringify(outputs);
  render(<CaptureBoundaryRows ports={outputs} />);
  for (const output of outputs) expect(screen.getByText(output.displayName!)).toBeVisible();
  expect(screen.getAllByTitle('(1, 4, 768) · float32')).toHaveLength(4);
  expect(screen.queryByText(/entries|\.value/)).not.toBeInTheDocument();
  const rows = moduleBoundaryRows({ capturePorts: [port([['arguments', 'args', 0]], { id: 'input' }),
    ...outputs.map(output => ({ ...output, direction: 'out' as const }))] } as ModelNodeData);
  expect(rows.map(row => row.label)).toEqual(['In', 'Out patch_tokens', 'Out cls_token', 'Out pooled', 'Out grid_size']);
  expect(JSON.stringify(outputs)).toBe(before);
});

it('numbers unnamed outputs independently of preceding input ports', () => {
  const rows = moduleBoundaryRows({ capturePorts: [port([['arguments', 'args', 0]]),
    ...[0, 1, 2].map(i => port([['output', 'entries', i, 'value']], { id: `out-${i}`, direction: 'out' }))] } as ModelNodeData);
  expect(rows.map(row => row.label)).toEqual(['In', 'Out 1', 'Out 2', 'Out 3']);
});

it('keeps a named single output and shared input aliases visible', () => {
  expect(moduleBoundaryRows({ capturePorts: [port([['output', 'entries', 0, 'value']],
    { direction: 'out', displayName: 'logits' })] } as ModelNodeData)[0].label).toBe('Out logits');
  expect(portLabel(port([['inputs', 'kwargs', 'left'], ['inputs', 'kwargs', 'right']],
    { direction: 'out', displayName: 'left / right' }), 0)).toBe('left / right');
});
