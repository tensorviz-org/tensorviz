import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { NodeIssues } from '../../canvas/NodeIssues';
import { ownGraphIssues, isFailingPort, nodeFailureLines } from '../../canvas/graphIssues';
import { reportNodeClick } from '../../ide/projection';
jest.mock('../../ide/projection', () => ({ currentProjection: () => ({ spanmap: { failed: {} } }), reportNodeClick: jest.fn() }));

beforeAll(() => { HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); }; });
test('shape facts remain exact and only the addressed operation port is marked', () => {
  const data: any = { parameters: {}, captureRef: { kind: 'operation' },
    captureDiagnostic: { code: 'invalid_shape', field: ['args', 1], expectedShape: [null, 8], actualShape: [2, 4] } };
  const port: any = { direction: 'in', paths: [['arguments', 'args', 1]] };
  expect(nodeFailureLines(data)).toEqual(['Incompatible shape', 'Expected [?, 8]', 'Received [2, 4]']);
  expect(isFailingPort(data, port)).toBe(true);
  expect(isFailingPort(data, { ...port, paths: [['arguments', 'args', 0]] })).toBe(false);
  expect(isFailingPort({ ...data, captureRef: { kind: 'invocation' } }, port)).toBe(false);
  expect(isFailingPort({ ...data, debugEvidence: 'previous' }, port)).toBe(false);
  expect(ownGraphIssues('old', { ...data, debugEvidence: 'previous', error: 'Old error' })).toEqual([]);
});
test('the issue count opens the same records, with known dtype evidence and source/focus actions', () => {
  const issues = ownGraphIssues('failed', { layerType: {}, parameters: {}, captureLabel: 'token_embedding',
    captureRef: { captureId: 'run' }, captureDiagnostic: { id: 'dtype', code: 'invalid_dtype', field: ['args', 1], actualDtype: 'torch.float32' } } as any);
  issues.push({ id: 'warning', nodeId: 'unmapped', label: 'Other block', severity: 'warning', message: 'Source is unresolved.' });
  render(<NodeIssues label="decoder" issues={issues} />);
  const trigger = screen.getByRole('button', { name: '1 error · 1 warning in decoder' });
  fireEvent.click(trigger);
  expect(screen.getByRole('dialog', { name: 'Issues in decoder' })).toBeVisible();
  expect(screen.getByText(/Actual dtype: torch.float32/)).toHaveTextContent('Input: args.1');
  expect(screen.queryByText(/Expected dtype:/)).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Show source' }));
  expect(reportNodeClick).toHaveBeenCalledWith('failed');
  const onCommand = jest.fn(); window.addEventListener('builder-command', onCommand);
  try {
    fireEvent.click(screen.getAllByRole('button', { name: 'Locate in graph' })[0]);
    expect(onCommand).toHaveBeenCalledWith(expect.objectContaining({ detail: { command: 'focus-node', payload: { nodeId: 'failed' } } }));
    expect(screen.queryByRole('dialog')).toBeNull(); expect(trigger).toHaveFocus();
  } finally { window.removeEventListener('builder-command', onCommand); }
});

test('dialog pointer events cannot start a canvas gesture and swallow its buttons', () => {
  const canvasPointer = jest.fn();
  render(<div onPointerDown={canvasPointer} onPointerUp={canvasPointer}>
    <NodeIssues label="fc" issues={[{ id: 'error', nodeId: 'fc', label: 'fc', severity: 'error', message: 'Failed' }]} />
  </div>);
  fireEvent.click(screen.getByRole('button', { name: '1 error in fc' }));
  const close = screen.getByRole('button', { name: 'Close issues' });
  fireEvent.pointerDown(close);
  fireEvent.pointerUp(close);
  expect(canvasPointer).not.toHaveBeenCalled();
  fireEvent.click(close);
  expect(screen.queryByRole('dialog')).toBeNull();
});
