import React from 'react';
import { render, screen } from '@testing-library/react';
import { detailTier } from '../../canvas/detailTier';
import { TensorSummary } from '../../canvas/TensorSummary';
import { ModelNodeData } from '../../types';

const data = (extra: Partial<ModelNodeData> = {}): ModelNodeData => ({
  layerType: { id: 'linear', name: 'Linear', displayName: 'Linear', category: 'linear', icon: 'Box', description: '', parameters: [] },
  parameters: {}, ...extra,
});

it('uses screen readability and hysteresis instead of flickering at a zoom boundary', () => {
  expect(detailTier(80)).toBe('overview');
  expect(detailTier(150)).toBe('working');
  expect(detailTier(220)).toBe('inspection');
  expect(detailTier(115, 'overview')).toBe('overview');
  expect(detailTier(105, 'working')).toBe('working');
  expect(detailTier(185, 'inspection')).toBe('inspection');
  expect(detailTier(195, 'working')).toBe('working');
  expect(detailTier(210, 'working')).toBe('inspection');
});

it('presents constructor widths as parameters without claiming measured tensors', () => {
  render(<TensorSummary data={data({ parameters: { in_features: 768, out_features: 3072 } })} />);
  expect(screen.getByText('Features')).toBeInTheDocument();
  expect(screen.getByText('Parameters')).toBeInTheDocument();
  expect(screen.queryByText('measured')).not.toBeInTheDocument();
  expect(screen.getByText('3072')).toHaveClass('graph-dimension-changed');
});

it('keeps recorded dimensions, evidence and mismatch information together', () => {
  render(<TensorSummary data={data({ inputShape: [2, 16, 768], outputShape: [2, 16, 3072], shapeEvidence: 'stale', error: 'Expected 768; received 512' })} />);
  expect(screen.getByText('stale')).toBeInTheDocument();
  expect(screen.getByText('3072')).toHaveClass('graph-dimension-changed');
  expect(screen.getByText('Shape check failed').parentElement).toHaveAttribute('title', 'Expected 768; received 512');
});

it('does not invent a tensor for an unmeasured operation', () => {
  render(<TensorSummary data={data()} />);
  expect(screen.getByText('unknown')).toBeInTheDocument();
  expect(screen.getByText('Not recorded')).toBeInTheDocument();
});

it('shows captured failure dimensions at working zoom instead of an unrecorded output placeholder', () => {
  render(<TensorSummary compact data={data({ error: 'Forward pass failed.', captureDiagnostic: {
    id: 'diagnostic:0', phase: 'forward', code: 'invalid_shape', field: ['args', 1], names: [],
    sourceBindingId: null, expectedShape: [null, 16], actualShape: [1, 8],
    expectedDtype: null, actualDtype: null, remedy: 'inspect_failure',
  } })} />);
  expect(screen.getByText('Incompatible shape')).toBeVisible();
  expect(screen.getByText('Expected [?, 16]')).toBeVisible();
  expect(screen.getByText('Received [1, 8]')).toBeVisible();
  expect(screen.queryByText('Not recorded')).not.toBeInTheDocument();
});

it('shows a state value without an empty input row or routine measured label', () => {
  render(<TensorSummary data={data({ outputShape: [10, 16], shapeEvidence: 'measured' })} />);
  expect(screen.getByText('OUT')).toBeVisible();
  expect(screen.queryByText('IN')).not.toBeInTheDocument();
  expect(screen.queryByText('Not recorded')).not.toBeInTheDocument();
  expect(screen.queryByText('measured')).not.toBeInTheDocument();
});

it('keeps coverage concise with its full meaning in the existing evidence badge', () => {
  render(<TensorSummary data={data({ parameters: { _shape_provenance: 'not_observed',
    _capture_coverage: { status: 'not_observed' }, in_features: 4, out_features: 8 } })} />);
  expect(screen.getByText('Not observed')).toHaveAttribute('title', 'Not observed in this run');
  expect(screen.queryByText('measured')).not.toBeInTheDocument();
});

it('never hides stale attribution behind an earlier coverage result', () => {
  render(<TensorSummary data={data({ shapeEvidence: 'stale', parameters: { _shape_provenance: 'state',
    _out_shape: '4', _capture_coverage: { status: 'state', stateValueIds: ['value:0'] } } })} />);
  expect(screen.getByText('stale')).toBeInTheDocument();
  expect(screen.queryByText('State')).not.toBeInTheDocument();
});
