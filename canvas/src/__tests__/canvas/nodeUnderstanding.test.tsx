import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { NodeUnderstanding } from '../../canvas/NodeUnderstanding';
import { NodeExplanationContext, type NodeExplanationContextValue } from '../../canvas/NodeExplanationContext';
import { nodeEvidenceLabel, nodeInputOutput, nodePresentation } from '../../canvas/nodePresentation';
import type { ModelNode, ModelNodeData } from '../../types';
import type { NodeExplanationState } from '../../ide/nodeExplanation.generated';

jest.mock('../../ide/projection', () => ({ localBrowserMode: () => false }));

const node: ModelNode = { id: 'python@Model#4', type: 'layerNode', position: { x: 0, y: 0 }, data: {
  layerType: { id: 'python', name: 'Python', displayName: 'Python', category: 'misc', description: '', icon: 'Code', parameters: [] },
  parameters: { _raw_call: 'scores = self._score_hidden(candidates, question)', _source_result_call: { name: '_score_hidden' } },
  captureLabel: 'python@Model#4', shapeEvidence: 'unknown', debugEvidence: 'source', inputPorts: ['candidates', 'question'], outputPorts: ['scores'],
} };
const ready: NodeExplanationState = { requestId: 'request', nodeId: node.id, revision: 'revision', status: 'ready', explanation: {
  title: 'Compare representations', purpose: 'Compare each candidate representation with the question.',
  inputs: ['Candidate features', 'Question features'], outputs: ['Raw comparison scores'],
  steps: ['Batch the representations.', 'Pass the batches through the head.'], uncertainties: ['Actual tensor sizes have not been recorded.'],
  citations: [{ id: 'source', file: 'model.py', sourceDigest: 'a'.repeat(64), startLine: 20, endLine: 26 }],
} };
const context = (state: NodeExplanationState): NodeExplanationContextValue => ({ states: { [node.id]: state }, retry: jest.fn(), reveal: jest.fn() });

it('keeps source-only meaning unknown while exposing the exact call and graph names', () => {
  const view = render(<NodeUnderstanding node={node} />);
  expect(screen.getByText('From source · shapes not recorded')).toBeVisible();
  expect(screen.getAllByText('self._score_hidden', { exact: true })[0]).toBeVisible();
  expect(screen.getByLabelText('Graph inputs and outputs')).toHaveTextContent('candidates · question');
  expect(screen.queryByText(/Compare each candidate/)).not.toBeInTheDocument();
  expect(screen.queryByText('python@Model#4')).not.toBeInTheDocument();
  fireEvent.click(screen.getByText('More details'));
  expect(screen.getByText(node.data.parameters._raw_call)).toBeVisible();
  expect(view.container.querySelector('dialog')).toBeNull();
  expect(screen.queryByRole('button', { name: 'Discuss in chat' })).not.toBeInTheDocument();
});

it('separates the generated explanation from graph evidence and reveals only citation identities', () => {
  const value = context(ready);
  const before = JSON.stringify(node);
  render(<NodeExplanationContext.Provider value={value}><NodeUnderstanding node={node} /></NodeExplanationContext.Provider>);
  const interpretation = screen.getByLabelText('Interpreted inputs and outputs');
  expect(interpretation).toHaveTextContent('Candidate features · Question features');
  expect(screen.getByText(ready.explanation!.purpose)).toBeVisible();
  expect(screen.getAllByText('Agent interpretation')[0]).toBeVisible();
  fireEvent.click(screen.getByText('More details'));
  expect(screen.getByLabelText('Graph inputs and outputs')).toHaveTextContent('candidates · question');
  expect(screen.getByLabelText('How it works')).toHaveTextContent('Pass the batches through the head.');
  expect(screen.getByText(ready.explanation!.uncertainties[0])).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'model.py:20–26' }));
  expect(value.reveal).toHaveBeenCalledWith(ready, 'source');
  expect(JSON.stringify(node)).toBe(before);
});

it('keeps source facts usable while loading, then permits a deliberate retry after failure', () => {
  const value = context({ ...ready, status: 'loading', explanation: undefined });
  const view = render(<NodeExplanationContext.Provider value={value}><NodeUnderstanding node={node} /></NodeExplanationContext.Provider>);
  expect(screen.getByRole('img', { name: 'Reading node source' })).toBeVisible();
  expect(screen.getByLabelText('Graph inputs and outputs')).toHaveTextContent('scores');
  const failed: NodeExplanationState = { ...ready, status: 'unavailable', explanation: undefined, detail: 'Source changed. Select the node again.' };
  const next = context(failed);
  view.rerender(<NodeExplanationContext.Provider value={next}><NodeUnderstanding node={node} /></NodeExplanationContext.Provider>);
  expect(screen.queryByRole('img', { name: 'Reading node source' })).not.toBeInTheDocument();
  expect(next.retry).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Retry explanation' }));
  expect(next.retry).toHaveBeenCalledWith(failed);
  expect(screen.queryByText(ready.explanation!.purpose)).not.toBeInTheDocument();
});

it('uses only registered layer descriptions, without treating arbitrary source names as meaning', () => {
  const known: ModelNodeData = { ...node.data, layerType: { ...node.data.layerType, id: 'linear', name: 'Linear', displayName: 'Linear', description: 'Unverified project-specific purpose' }, parameters: {} };
  const description = nodePresentation(known, 'fc').purpose;
  expect(description).toBeTruthy();
  expect(description).not.toContain('project-specific');
  expect(nodePresentation(node.data, node.id)).toMatchObject({ title: '_score_hidden', callable: 'self._score_hidden', purpose: undefined });
  expect(nodePresentation({ ...known, captureLabel: 'encoder.layers.2.fc' }, 'fc').subtitle).toBe('encoder.layers.2.fc');
  expect(nodePresentation({ ...known, layerType: { ...known.layerType, id: 'module', name: 'models.blocks.Projection' },
    captureRef: { kind: 'invocation' } as ModelNodeData['captureRef'], moduleTemplateId: 'Projection' }, 'block').callable).toBe('models.blocks.Projection');
  const view = render(<NodeUnderstanding node={{ ...node, id: 'fc', data: known }} />);
  expect(view.container.querySelector('.node-understanding-callable')).toBeNull();
  fireEvent.click(screen.getByText('More details'));
  expect(screen.getByText('Linear', { exact: true })).toBeVisible();
});

it('does not turn source-only cached sizes into recorded tensor evidence', () => {
  const unobserved = { ...node.data, inputShape: [99, 8], outputShape: [99, 1], parameters: { _shape_provenance: 'not_observed' } };
  expect(nodeEvidenceLabel(unobserved)).toBe('From source · shapes not recorded');
  expect(nodeInputOutput(unobserved)).toEqual({ inputs: ['candidates', 'question'], outputs: ['scores'] });
  const measured = { ...node.data, inputPorts: ['features'], shapeEvidence: 'measured' as const, inputShape: [2, 8], outputShape: [2, 1], parameters: {} };
  expect(nodeInputOutput(measured)).toEqual({ inputs: ['features (2, 8)'], outputs: ['scores (2, 1)'] });
});

it('opens chat with the current selection reported first, preserving the existing composer draft', () => {
  const commands: string[] = [];
  const record = (event: Event) => commands.push(event.type === 'builder-command' ? (event as CustomEvent).detail.command : event.type);
  window.addEventListener('builder-command', record); window.addEventListener('tensorviz-agent-compose', record);
  try {
    render(<NodeExplanationContext.Provider value={context(ready)}><NodeUnderstanding node={node} /></NodeExplanationContext.Provider>);
    fireEvent.click(screen.getByRole('button', { name: 'Discuss in chat' }));
    expect(commands).toEqual(['report-selection', 'tensorviz-agent-compose']);
    expect(within(screen.getByLabelText('About this node')).queryByRole('textbox')).not.toBeInTheDocument();
  } finally {
    window.removeEventListener('builder-command', record); window.removeEventListener('tensorviz-agent-compose', record);
  }
});
