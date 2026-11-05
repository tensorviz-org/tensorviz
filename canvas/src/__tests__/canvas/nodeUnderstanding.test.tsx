import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { NodeUnderstanding } from '../../canvas/NodeUnderstanding';
import { NodeExplanationContext, type NodeExplanationContextValue } from '../../canvas/NodeExplanationContext';
import { nodeEvidenceLabel, nodeInputOutput, nodePresentation, sourceDocumentationSummary } from '../../canvas/nodePresentation';
import type { ModelNode, ModelNodeData } from '../../types';
import type { NodeExplanationState, NodeSourceDocumentation } from '../../ide/nodeExplanation.generated';

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
const documentation: Extract<NodeSourceDocumentation, { status: 'resolved' }> = {
  status: 'resolved', kind: 'method', qualifiedName: 'models.spectral.SpectralMixer.blend',
  signature: 'def blend(\n    self,\n    features: tuple[Tensor, Tensor],\n    *,\n    scale: float = 0.5,\n    labels: Mapping[str, int] | None = None,\n) -> dict[str, Tensor]:',
  docstring: 'Blend the two feature streams using the declared scale.\n\nArgs:\n    features: A pair of feature tensors.\n    scale: The weighting applied to the second stream.\n\nReturns:\n    A dictionary containing the blended features.',
  truncated: false, citation: { id: 'definition', file: 'models/spectral.py', sourceDigest: 'b'.repeat(64), startLine: 40, endLine: 61 },
};

it('keeps every parameter and the full technical identity available in the selection details', () => {
  const callable = 'research.experimental.video_question_answering.ExtremelyLongCandidateRepresentationScoringHead';
  const parameters = Object.fromEntries(Array.from({ length: 200 }, (_, index) => [`candidate_projection_parameter_${index}`, { dimensions: [2, 32], enabled: true }]));
  const selected = { ...node, data: { ...node.data, layerType: { ...node.data.layerType, id: 'custom_code' }, parameters: { ...parameters, _class: callable } } };
  render(<NodeUnderstanding node={selected} />);
  expect(screen.getByLabelText('Parameter values')).not.toBeVisible();
  fireEvent.click(screen.getByText('More details'));
  expect(screen.getByText(callable, { exact: true })).toBeVisible();
  const region = screen.getByRole('region', { name: 'Parameter values' });
  expect(region).toHaveAttribute('tabindex', '0');
  expect(within(region).getByText('candidate_projection_parameter_199')).toBeVisible();
  expect(region.querySelectorAll('dt')).toHaveLength(200);
});

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

it('makes author documentation useful without an available agent, preserving the full declared signature and citation', () => {
  const state: NodeExplanationState = { ...ready, status: 'unavailable', explanation: undefined, detail: 'The agent is not signed in.', documentation };
  const value = context(state);
  render(<NodeExplanationContext.Provider value={value}><NodeUnderstanding node={node} /></NodeExplanationContext.Provider>);
  const overview = screen.getByLabelText('Source documentation overview');
  expect(overview).toHaveTextContent('Source documentation');
  expect(overview).toHaveTextContent('Blend the two feature streams using the declared scale.');
  expect(overview).not.toHaveTextContent('Args:');
  expect(screen.getByText('The agent is not signed in.')).not.toBeVisible();
  expect(screen.getByRole('button', { name: 'Retry explanation' })).not.toBeVisible();
  fireEvent.click(within(overview).getByRole('button', { name: 'models/spectral.py:40–61' }));
  expect(value.reveal).toHaveBeenCalledWith(state, 'definition');
  fireEvent.click(screen.getByText('More details'));
  const signature = screen.getByRole('region', { name: 'Declared signature' });
  expect(signature.textContent).toBe(documentation.signature);
  expect(signature).toHaveAttribute('tabindex', '0');
  expect(screen.getByRole('region', { name: 'Full source documentation' }).textContent).toBe(documentation.docstring);
  expect(screen.getByText(documentation.qualifiedName)).toBeVisible();
  expect(screen.getByRole('button', { name: 'Retry explanation' })).toBeVisible();
});

it('keeps source documentation distinct and inspectable alongside a ready agent interpretation', () => {
  const value = context({ ...ready, documentation });
  render(<NodeExplanationContext.Provider value={value}><NodeUnderstanding node={node} /></NodeExplanationContext.Provider>);
  expect(screen.getByText(ready.explanation!.purpose)).toBeVisible();
  expect(screen.queryByLabelText('Source documentation overview')).not.toBeInTheDocument();
  expect(screen.getByLabelText('Source documentation details')).not.toBeVisible();
  fireEvent.click(screen.getByText('More details'));
  expect(screen.getByLabelText('Source documentation details')).toHaveTextContent('Source documentation');
  expect(screen.getByRole('region', { name: 'Full source documentation' })).toBeVisible();
  expect(screen.getByLabelText('Interpreted inputs and outputs')).toHaveTextContent('Candidate features');
  expect(screen.getByLabelText('Graph inputs and outputs')).toHaveTextContent('candidates');
  fireEvent.click(within(screen.getByLabelText('Source documentation details')).getByRole('button', { name: 'models/spectral.py:40–61' }));
  expect(value.reveal).toHaveBeenCalledWith(value.states[node.id], 'definition');
});

it('shows a custom class declaration without inventing a forward signature or purpose when it has no docstring', () => {
  const source: NodeSourceDocumentation = { ...documentation, kind: 'class', qualifiedName: 'models.spectral.OptionalSpectralMixer',
    signature: 'class OptionalSpectralMixer(BaseMixer[Tensor]):', docstring: undefined };
  const data: ModelNodeData = { ...node.data, layerType: { ...node.data.layerType, id: 'module', name: source.qualifiedName, displayName: 'OptionalSpectralMixer' }, parameters: {}, moduleTemplateId: 'OptionalSpectralMixer' };
  render(<NodeExplanationContext.Provider value={context({ ...ready, status: 'unavailable', explanation: undefined, documentation: source })}>
    <NodeUnderstanding node={{ ...node, data }} /></NodeExplanationContext.Provider>);
  expect(screen.queryByLabelText('Source documentation overview')).not.toBeInTheDocument();
  expect(screen.queryByText(/Custom module group/)).not.toBeInTheDocument();
  expect(screen.queryByText(ready.explanation!.purpose)).not.toBeInTheDocument();
  fireEvent.click(screen.getByText('More details'));
  expect(screen.getByRole('region', { name: 'Class declaration' }).textContent).toBe(source.signature);
  expect(screen.queryByRole('region', { name: 'Declared signature' })).not.toBeInTheDocument();
  expect(screen.queryByRole('region', { name: 'Full source documentation' })).not.toBeInTheDocument();
});

it('shows unknown imported definitions quietly, without claiming a docstring or running a model', () => {
  const state: NodeExplanationState = { ...ready, status: 'unavailable', explanation: undefined,
    documentation: { status: 'unavailable', reason: 'unresolved_import' } };
  render(<NodeExplanationContext.Provider value={context(state)}><NodeUnderstanding node={node} /></NodeExplanationContext.Provider>);
  const reason = screen.getByText('The imported definition is unavailable in the current source.');
  expect(reason).not.toBeVisible();
  fireEvent.click(screen.getByText('More details'));
  expect(reason).toBeVisible();
  expect(screen.queryByRole('region', { name: 'Declared signature' })).not.toBeInTheDocument();
  expect(screen.queryByRole('region', { name: 'Full source documentation' })).not.toBeInTheDocument();
});

it('keeps a resolved imported function declaration distinct from its local call alias', () => {
  const imported: NodeSourceDocumentation = { ...documentation, kind: 'function', qualifiedName: 'research.ops.spectral.blend',
    signature: 'def blend(features: Tensor, axes: tuple[int, ...] = (0, 2)) -> Tensor:',
    docstring: 'Mix the selected axes in the frequency domain.' };
  const importedNode: ModelNode = { ...node, data: { ...node.data, parameters: {
    _raw_call: 'mixed = spectral_blend(features, axes=(0, 2))', _source_result_call: { name: 'spectral_blend' },
  } } };
  render(<NodeExplanationContext.Provider value={context({ ...ready, status: 'unavailable', explanation: undefined, documentation: imported })}>
    <NodeUnderstanding node={importedNode} /></NodeExplanationContext.Provider>);
  expect(screen.getByLabelText('Source documentation overview')).toHaveTextContent(imported.docstring!);
  fireEvent.click(screen.getByText('More details'));
  expect(screen.getByRole('region', { name: 'Declared signature' }).textContent).toBe(imported.signature);
  expect(screen.getByText(imported.qualifiedName)).toBeVisible();
  expect(screen.getByText('mixed = spectral_blend(features, axes=(0, 2))')).toBeVisible();
});

it('renders bounded source excerpts as plain text and identifies truncation', () => {
  const docstring = 'Author description with <script>alert("source")</script>.\n\n' + 'Detailed source notes.\n'.repeat(100);
  const source = { ...documentation, docstring, truncated: true };
  const onKeyDown = jest.fn();
  const view = render(<div onKeyDown={onKeyDown}><NodeExplanationContext.Provider value={context({ ...ready, documentation: source })}><NodeUnderstanding node={node} /></NodeExplanationContext.Provider></div>);
  fireEvent.click(screen.getByText('More details'));
  const full = screen.getByRole('region', { name: 'Full source documentation' });
  expect(full.textContent).toBe(docstring);
  expect(full).toHaveAttribute('tabindex', '0');
  expect(view.container.querySelector('script')).toBeNull();
  expect(screen.getByText('Source documentation was truncated.')).toBeVisible();
  expect(fireEvent.keyDown(full, { key: 'PageDown' })).toBe(true);
  expect(onKeyDown).not.toHaveBeenCalled();
  fireEvent.keyDown(full, { key: 'Escape' });
  expect(onKeyDown).toHaveBeenCalledTimes(1);
});

it('quotes only the opening source paragraph and bounds long summaries at a word boundary', () => {
  expect(sourceDocumentationSummary('  First line\ncontinued here.\n\nArgs:\n    x: input')).toBe('First line continued here.');
  expect(sourceDocumentationSummary(' \n\n ')).toBeUndefined();
  const summary = sourceDocumentationSummary('LongDocumentation '.repeat(40))!;
  expect(summary.length).toBeLessThanOrEqual(321);
  expect(summary.endsWith('LongDocumentation…')).toBe(true);
});
