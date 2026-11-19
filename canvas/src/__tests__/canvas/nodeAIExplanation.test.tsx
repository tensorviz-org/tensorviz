import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { NodeAIExplanation } from '../../canvas/NodeAIExplanation';
import type { ModelExplanationState, NodeExplanationState } from '../../ide/nodeExplanation.generated';

const result: NodeExplanationState = { requestId: 'selection', nodeId: 'attention', revision: 'revision', status: 'ready', explanation: {
  title: 'Mix token information', purpose: 'Combine the value representations using attention weights.',
  inputs: ['Queries, keys and values'], outputs: ['Contextual representations'], steps: ['Compare query and key representations.'],
  concepts: [{ term: 'Attention weights', explanation: 'Relative contributions from the value representations.' }],
  uncertainties: ['The masking policy is not established by this excerpt.'], citations: [
    { id: 'source-authority:attention:58', file: 'models/attention.py', sourceDigest: 'a'.repeat(64), startLine: 58, endLine: 65 },
  ],
} };

it('renders a static AI reading surface with no operation-status feedback', () => {
  render(<NodeAIExplanation result={result} onReveal={jest.fn()} />);
  const panel = screen.getByRole('region', { name: 'AI explanation' });
  expect(within(panel).getByRole('heading', { name: 'AI explanation' })).toBeVisible();
  expect(within(panel).getByText(result.explanation!.purpose)).toBeVisible();
  expect(within(panel).getByText('AI-generated interpretation · verify against source')).toBeVisible();
  expect(within(panel).queryByRole('status')).not.toBeInTheDocument();
  expect(within(panel).queryByRole('alert')).not.toBeInTheDocument();
  expect(panel).not.toHaveAttribute('aria-busy');
  expect(panel).not.toHaveAttribute('data-state');
  for (const label of ['Generating', 'Generated', 'Unavailable', 'Stopped', 'No explanation generated',
    'Previous explanation', 'Try again', 'Stop explaining', 'Technical details']) {
    expect(within(panel).queryByText(label, { exact: true })).not.toBeInTheDocument();
  }
  expect(within(panel).getAllByRole('button')).toHaveLength(1);
});

it('renders nothing without a result', () => {
  const { container } = render(<NodeAIExplanation onReveal={jest.fn()} />);
  expect(container).toBeEmptyDOMElement();
});

it.each<NodeExplanationState['status']>(['queued', 'loading', 'unavailable'])('renders nothing for a %s result, even with retained explanation data', status => {
  const { container } = render(<NodeAIExplanation result={{ ...result, status, detail: 'Operation feedback belongs on Explain.' }} onReveal={jest.fn()} />);
  expect(container).toBeEmptyDOMElement();
});

it('renders nothing for a ready result without an explanation', () => {
  const { container } = render(<NodeAIExplanation result={{ ...result, explanation: undefined }} onReveal={jest.fn()} />);
  expect(container).toBeEmptyDOMElement();
});

it('preserves interpreted I/O and exact source authority IDs', () => {
  const reveal = jest.fn();
  render(<NodeAIExplanation result={result} onReveal={reveal} />);
  const io = screen.getByLabelText('Interpreted inputs and outputs');
  expect(io).toHaveTextContent('Queries, keys and values');
  expect(io).toHaveTextContent('Contextual representations');
  expect(reveal).not.toHaveBeenCalled();
  const source = screen.getByRole('button', { name: 'models/attention.py:58–65' });
  expect(source).toHaveTextContent('attention.py:58–65');
  expect(source).toHaveAttribute('title', 'models/attention.py:58–65');
  fireEvent.click(source);
  expect(reveal).toHaveBeenCalledTimes(1);
  expect(reveal).toHaveBeenCalledWith('source-authority:attention:58');
});

it('keeps steps, concepts and uncertainties accessible under Explanation details', () => {
  render(<NodeAIExplanation result={result} onReveal={jest.fn()} />);
  const summary = screen.getByText('Explanation details');
  const details = summary.closest('details')!;
  expect(details).not.toHaveAttribute('open');
  fireEvent.click(summary);
  expect(details).toHaveAttribute('open');
  expect(screen.getByLabelText('How it works')).toHaveTextContent(result.explanation!.steps[0]);
  expect(screen.getByText(result.explanation!.uncertainties[0])).toBeVisible();
  const concept = within(screen.getByRole('region', { name: 'Related concepts' })).getByText('Attention weights');
  fireEvent.click(concept);
  expect(concept.closest('details')).toHaveAttribute('open');
  expect(screen.getByText(result.explanation!.concepts![0].explanation)).toBeVisible();
});

it('omits empty optional result sections', () => {
  render(<NodeAIExplanation result={{ ...result, explanation: { ...result.explanation!, inputs: [], outputs: [], steps: [], concepts: [], uncertainties: [], citations: [] } }} onReveal={jest.fn()} />);
  expect(screen.getByText(result.explanation!.purpose)).toBeVisible();
  expect(screen.queryByText('Explanation details')).not.toBeInTheDocument();
  expect(screen.queryByLabelText('Interpreted inputs and outputs')).not.toBeInTheDocument();
  expect(screen.queryByLabelText('Explanation sources')).not.toBeInTheDocument();
});

it('omits execution metadata unless explicitly provided', () => {
  const { container } = render(<NodeAIExplanation result={result} onReveal={jest.fn()} />);
  expect(container.querySelector('.node-ai-metadata')).not.toBeInTheDocument();
  expect(screen.queryByText(/Codex|Claude|model-a/)).not.toBeInTheDocument();
});

it.each<[NonNullable<ModelExplanationState['execution']>, string]>([
  [{ provider: 'codex', model: 'model-a', effort: 'high' }, 'Codex · model-a · high'],
  [{ provider: 'claude' }, 'Claude'],
])('shows only supplied actual execution metadata: %j', (execution, label) => {
  const { container } = render(<NodeAIExplanation result={result} execution={execution} onReveal={jest.fn()} />);
  expect(screen.getByText(label)).toBeVisible();
  expect(container.querySelector('.node-ai-metadata')).toHaveTextContent(label);
  expect(container.querySelector('.node-ai-metadata')!.textContent).toBe(label);
});

it('does not render execution metadata without a verified ready explanation', () => {
  const { container } = render(<NodeAIExplanation execution={{ provider: 'codex', model: 'model-a' }} onReveal={jest.fn()} />);
  expect(container).toBeEmptyDOMElement();
});
