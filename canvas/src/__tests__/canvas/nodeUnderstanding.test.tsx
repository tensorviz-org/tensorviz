import React from 'react';
import { readFileSync } from 'fs';
import { resolve } from 'path';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { NodeUnderstanding } from '../../canvas/NodeUnderstanding';
import { NodeExplanationContext, type NodeExplanationContextValue } from '../../canvas/NodeExplanationContext';
import { nodeEvidenceLabel, nodeInputOutput, nodePresentation, sourceDocumentationSummary } from '../../canvas/nodePresentation';
import type { CapturePort, ModelNode, ModelNodeData } from '../../types';
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

it('defines a compact reading layout with aligned insets and subordinate composition', () => {
  const inspector = readFileSync(resolve(__dirname, '../../canvas/nodeInspector.css'), 'utf8');
  const understanding = readFileSync(resolve(__dirname, '../../canvas/nodeUnderstanding.css'), 'utf8');
  expect(inspector).toMatch(/\.node-inspector-header\s*\{[^}]*padding: 12px 16px;/);
  expect(inspector).toMatch(/\.node-inspector-body\s*\{[^}]*padding: 12px 16px 16px;/);
  expect(inspector).toMatch(/\.node-inspector h2\s*\{[^}]*font: 600 16px\/22px var\(--font-ui\);/);
  expect(inspector).toMatch(/\.node-inspector \.graph-block-composition\s*\{[^}]*flex-wrap: wrap;[^}]*font: 10px\/15px var\(--font-ui\);/);
  expect(inspector).not.toMatch(/\.node-inspector-actions\s*\{/);
  expect(inspector).toMatch(/\.module-focus-frame:has\(\.node-inspector\) > \.module-focus-options\s*\{\s*display: none;/);
  expect(understanding).toMatch(/\.node-understanding\s*\{[^}]*font: 12px\/18px var\(--font-ui\);/);
  expect(understanding).toMatch(/\.node-understanding \.node-understanding-neighbors \.node-understanding-connection\s*\{[^}]*flex-direction: row;[^}]*background: transparent;/);
});

it('pads connection hover targets while keeping text and dividers aligned with the section', () => {
  const css = readFileSync(resolve(__dirname, '../../canvas/nodeUnderstanding.css'), 'utf8');
  expect(css).toMatch(/\.node-understanding \.node-understanding-neighbors \.node-understanding-connection\s*\{[^}]*padding: 8px;[^}]*margin-inline: -8px;/);
  expect(css).toMatch(/\.node-understanding-connection-body\s*\{[^}]*gap: 2px;/);
  expect(css).toMatch(/\.node-understanding-connection \+ \.node-understanding-connection::before\s*\{[^}]*left: 8px;[^}]*right: 8px;/);
  expect(css).toMatch(/\.node-understanding-flow-heading\s*\{[^}]*padding: 0 0 6px;/);
});

it('wraps long exact callable names at punctuation without truncating their text', () => {
  const target = 'torch.nn.functional.scaled_dot_product_attention';
  const peer = { ...node, id: 'attention', data: { ...node.data,
    parameters: { _source_result_call: { name: 'scaled_dot_product_attention', target } } } };
  render(<NodeUnderstanding node={node} nodes={[node, peer]} edges={[{ id: 'edge', source: node.id, target: peer.id }]} onNavigate={jest.fn()} />);
  const exact = screen.getByText(target, { selector: 'code' });
  expect(exact.textContent).toBe(target);
  expect(exact.querySelector('wbr')).not.toBeNull();
});

it('adds expandable terminology without replacing source facts or inventing connections', () => {
  const state = { ...ready, explanation: { ...ready.explanation!, concepts: [{ term: 'VAE', explanation: 'A variational autoencoder represents data through a learned latent distribution.' }] } };
  render(<NodeExplanationContext.Provider value={context(state)}><NodeUnderstanding node={node} /></NodeExplanationContext.Provider>);
  fireEvent.click(screen.getByText('Explanation details'));
  expect(screen.getByText('VAE').closest('details')).not.toHaveAttribute('open');
  fireEvent.click(screen.getByText('VAE'));
  expect(screen.getByText(state.explanation.concepts[0].explanation)).toBeVisible();
  expect(screen.queryByLabelText('Connected components')).toBeNull();
  expect(screen.getByLabelText('Graph inputs and outputs')).toHaveTextContent('candidates');
});

it('groups exact graph neighbors by flow direction with readable rows that reveal each peer', () => {
  const upstream: ModelNode = { ...node, id: 'python@Model#2', data: { ...node.data,
    sourceLocation: { file: 'models/attention.py', line: 42 },
    parameters: { _raw_call: 'q = q.view(batch, heads, width)', _source_result_call: { name: 'view' } } } };
  const downstream: ModelNode = { ...node, id: 'python@Model#8', data: { ...node.data,
    parameters: { _raw_call: 'torch.matmul(scores, values)', _source_result_call: { name: 'matmul', target: 'torch.matmul' } } } };
  const navigate = jest.fn();
  render(<NodeUnderstanding node={node} nodes={[upstream, node, downstream]} edges={[
    { id: 'in', source: upstream.id, target: node.id }, { id: 'out', source: node.id, target: downstream.id },
    { id: 'duplicate', source: upstream.id, target: node.id },
    { id: 'unavailable', source: 'missing-node', target: node.id },
  ]} onNavigate={navigate} />);
  const connections = screen.getByLabelText('Connected components');
  expect(within(connections).getAllByRole('button')).toHaveLength(2);
  const incoming = within(connections).getByRole('region', { name: 'Incoming connections' });
  expect(within(incoming).getByText('Incoming')).toBeVisible();
  expect(within(incoming).getByText('To selected node')).toBeVisible();
  const input = within(incoming).getByRole('button', { name: 'Show in graph: Incoming from q.view' });
  expect(within(input).getByText('q.view', { exact: true }).tagName).toBe('CODE');
  expect(input).toHaveAttribute('title', 'q = q.view(batch, heads, width)');
  expect(within(input).queryByText('q = q.view(batch, heads, width)')).not.toBeInTheDocument();
  expect(within(input).getByText('models/attention.py:42')).toBeVisible();
  expect(within(input).queryByText('Show in graph')).not.toBeInTheDocument();
  expect(input.querySelector('svg')).toHaveAttribute('aria-hidden', 'true');
  expect(input).not.toHaveTextContent(/Reshape|query|python@/);
  fireEvent.click(input);
  expect(navigate).toHaveBeenLastCalledWith(upstream.id);
  const outgoing = within(connections).getByRole('region', { name: 'Outgoing connections' });
  expect(within(outgoing).getByText('Outgoing')).toBeVisible();
  expect(within(outgoing).getByText('From selected node')).toBeVisible();
  const output = within(outgoing).getByRole('button', { name: 'Show in graph: Outgoing to torch.matmul' });
  expect(within(output).getByText('Matrix multiply')).toBeVisible();
  expect(within(output).getByText('torch.matmul', { exact: true }).tagName).toBe('CODE');
  fireEvent.click(output);
  expect(navigate).toHaveBeenLastCalledWith(downstream.id);
  expect(navigate).toHaveBeenCalledTimes(2);
  expect(screen.queryByText('Receives from')).not.toBeInTheDocument();
  expect(screen.queryByText('Provides to')).not.toBeInTheDocument();
});
it('distinguishes connected instances instead of repeating their shared class name', () => {
  const upstream = { ...node, id: 'block1', data: { ...node.data, parameters: {}, captureLabel: 'transformer.h.1',
    layerType: { ...node.data.layerType, id: 'module', name: 'model.Block' } } };
  const downstream = { ...upstream, id: 'block3', data: { ...upstream.data, captureLabel: 'transformer.h.3' } };
  render(<NodeUnderstanding node={node} nodes={[upstream, node, downstream]} edges={[
    { id: 'in', source: upstream.id, target: node.id }, { id: 'out', source: node.id, target: downstream.id },
  ]} onNavigate={jest.fn()} />);
  const connections = screen.getByLabelText('Connected components');
  const input = within(connections).getByRole('button', { name: 'Show in graph: Incoming from transformer.h.1' });
  const output = within(connections).getByRole('button', { name: 'Show in graph: Outgoing to transformer.h.3' });
  expect(within(input).getByText('transformer.h.1', { exact: true })).toBeVisible();
  expect(within(output).getByText('transformer.h.3', { exact: true })).toBeVisible();
  expect(within(input).getByText('model.Block', { exact: true })).toBeVisible();
});

it('preserves recorded source file and line on connection rows without inventing missing locations', () => {
  const upstream: ModelNode = { ...node, id: 'source-view', data: { ...node.data, parameters: {
    _raw_call: 'q.view(shape)', _source_result_call: { name: 'view' }, _source_file: 'research/model.py', _source_line: 81,
  } } };
  const downstream: ModelNode = { ...node, id: 'source-output', data: { ...node.data, parameters: { _raw_call: 'consume(q)' } } };
  render(<NodeUnderstanding node={node} nodes={[upstream, downstream]} edges={[
    { id: 'in', source: upstream.id, target: node.id }, { id: 'out', source: node.id, target: downstream.id },
  ]} />);
  const input = screen.getByRole('button', { name: 'Show in graph: Incoming from q.view' });
  expect(within(input).getByText('research/model.py:81')).toBeVisible();
  expect(input).toBeDisabled();
  const output = screen.getByRole('button', { name: 'Show in graph: Outgoing to consume' });
  expect(output).toBeDisabled();
  expect(output.querySelector('.node-understanding-connection-location')).toBeNull();
  expect(output).not.toHaveTextContent(/unknown|not recorded|source-output/i);
});

it.each([
  { _source_file: 'call.py', _decl_line: 12 },
  { _decl_file: 'definition.py', _source_line: 42 },
  { _source_file: 'call.py', _source_line: 0 },
  { _source_file: 'call.py', _source_line: '42' },
])('omits incomplete or invalid source-location pairs instead of combining unrelated evidence: %j', parameters => {
  const peer: ModelNode = { ...node, id: 'opaque-neighbor-key', data: { ...node.data,
    parameters: { _raw_call: 'q.view(shape)', ...parameters },
  } };
  render(<NodeUnderstanding node={node} nodes={[peer]} edges={[{ id: 'in', source: peer.id, target: node.id }]} />);
  const connection = screen.getByRole('button', { name: 'Show in graph: Incoming from q.view' });
  expect(connection.querySelector('.node-understanding-connection-location')).toBeNull();
  expect(connection).not.toHaveTextContent(peer.id);
});

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
  expect(screen.getByText('Read from code · tensor shapes not recorded')).toBeVisible();
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
  expect(screen.getByRole('region', { name: 'AI explanation' })).toBeVisible();
  expect(screen.getByText('AI-generated interpretation · verify against source')).toBeVisible();
  fireEvent.click(screen.getByText('More details'));
  fireEvent.click(screen.getByText('Explanation details'));
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
  expect(screen.getByRole('status')).toHaveTextContent('Reading component');
  expect(screen.getByRole('status')).toBeVisible();
  expect(screen.getByRole('status').closest('.node-understanding-evidence')).toBeNull();
  expect(screen.getByRole('status')).toHaveAttribute('title', expect.stringContaining('does not run the model'));
  expect(screen.getByLabelText('Graph inputs and outputs')).toHaveTextContent('scores');
  const failed: NodeExplanationState = { ...ready, status: 'unavailable', explanation: undefined, detail: 'Source changed. Select the node again.' };
  const next = context(failed);
  view.rerender(<NodeExplanationContext.Provider value={next}><NodeUnderstanding node={node} /></NodeExplanationContext.Provider>);
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
  expect(next.retry).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Retry explanation' }));
  expect(next.retry).toHaveBeenCalledWith(failed);
  expect(screen.queryByText(ready.explanation!.purpose)).not.toBeInTheDocument();
});

it('leaves operation feedback to the parent button without hiding source facts or citation navigation', () => {
  const state: NodeExplanationState = { ...ready, status: 'unavailable', explanation: undefined,
    detail: 'Explanation failed.', documentation };
  const value = context(state);
  render(<NodeExplanationContext.Provider value={value}><NodeUnderstanding node={node} hideActions /></NodeExplanationContext.Provider>);
  fireEvent.click(screen.getByText('More details'));
  expect(screen.queryByText('Explanation failed.')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Retry explanation' })).not.toBeInTheDocument();
  expect(screen.getByLabelText('Graph inputs and outputs')).toHaveTextContent('candidates');
  fireEvent.click(within(screen.getByLabelText('Source documentation details')).getByRole('button', { name: 'models/spectral.py:40–61' }));
  expect(value.reveal).toHaveBeenCalledWith(state, 'definition');
  expect(value.retry).not.toHaveBeenCalled();
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
  expect(nodeEvidenceLabel(unobserved)).toBe('Read from code · tensor shapes not recorded');
  expect(nodeInputOutput(unobserved)).toEqual({ inputs: ['candidates', 'question'], outputs: ['scores'] });
  const measured = { ...node.data, inputPorts: ['features'], shapeEvidence: 'measured' as const, inputShape: [2, 8], outputShape: [2, 1], parameters: {} };
  expect(nodeInputOutput(measured)).toEqual({ inputs: ['features (2, 8)'], outputs: ['scores (2, 1)'] });
});

it('omits internal port IDs without replacing them with numbered names and preserves measured shapes', () => {
  const data: ModelNodeData = { ...node.data, shapeEvidence: 'measured', inputPorts: ['port:input-id'], outputPorts: ['port:output-id'],
    parameters: { _shape_provenance: 'observed', _in_shape: [1, 8, 768], _out_shape: [1, 8, 768] } };
  const before = JSON.stringify(data);
  expect(nodeInputOutput(data)).toEqual({ inputs: ['(1, 8, 768)'], outputs: ['(1, 8, 768)'] });
  expect(nodeInputOutput({ ...data, parameters: { _shape_provenance: 'not_observed' } })).toEqual({ inputs: [], outputs: [] });
  expect(nodeInputOutput({ ...data, inputPorts: ['features', 'port:input-id'], parameters: {} })).toEqual({ inputs: ['features'], outputs: [] });
  expect(JSON.stringify(data)).toBe(before);
});

it('shows only available captured names, paths and shapes without generated port labels', () => {
  const port: CapturePort = { id: 'port:input-id', nodeId: node.id, direction: 'in', argumentKind: 'value',
    valueId: 'value:input-id', memberPortId: null, device: 'cpu', paths: [], shape: [1, 8, 768], dtype: 'float32' };
  const data: ModelNodeData = { ...node.data, parameters: {}, inputPorts: [], outputPorts: [], shapeEvidence: 'measured',
    capturePorts: [port] };
  expect(nodeInputOutput(data)).toEqual({ inputs: ['(1, 8, 768)'], outputs: [] });
  expect(nodeInputOutput({ ...data, capturePorts: [{ ...port, shape: null }] })).toEqual({ inputs: [], outputs: [] });
  expect(nodeInputOutput({ ...data, capturePorts: [{ ...port, displayName: 'hidden_states' }] })).toEqual({ inputs: ['hidden_states (1, 8, 768)'], outputs: [] });
  expect(nodeInputOutput({ ...data, capturePorts: [{ ...port, paths: [['arguments', 'args', 'items', 0]] }] })).toEqual({ inputs: ['args[0] (1, 8, 768)'], outputs: [] });
  expect(nodeInputOutput({ ...data, capturePorts: [{ ...port, paths: [['arguments', 'kwargs', 'hidden_states']] }] })).toEqual({ inputs: ['kwargs.hidden_states (1, 8, 768)'], outputs: [] });
  expect(nodeInputOutput({ ...data, capturePorts: [{ ...port, displayName: 'port:input-id' }] })).toEqual({ inputs: ['(1, 8, 768)'], outputs: [] });
  expect(nodeInputOutput({ ...data, capturePorts: [{ ...port, shape: null, paths: [['arguments', 'kwargs', 'entries', 0, 'value']] }] })).toEqual({ inputs: [], outputs: [] });
});

it('labels source-only x and y as source names rather than tensor shapes', () => {
  const selected: ModelNode = { ...node, data: { ...node.data, inputPorts: ['x'], outputPorts: ['y'] } };
  const view = render(<NodeUnderstanding node={selected} />);
  const source = screen.getByLabelText('Graph inputs and outputs');
  expect(within(source).getByText('Inputs')).toBeVisible();
  expect(within(source).getByText('Outputs')).toBeVisible();
  expect(source).toHaveAccessibleDescription('Source names');
  expect(screen.getAllByText('Source names', { exact: true })).toHaveLength(1);
  expect(within(source).getByText('x', { exact: true })).toBeVisible();
  expect(within(source).getByText('y', { exact: true })).toBeVisible();
  expect(source).not.toHaveTextContent(/shape|measured/i);
  view.rerender(<NodeUnderstanding node={{ ...selected, data: { ...selected.data,
    shapeEvidence: 'measured', inputShape: [2, 8], outputShape: [2, 4],
  } }} />);
  const measured = screen.getByLabelText('Graph inputs and outputs');
  expect(within(measured).getByText('Inputs')).toBeVisible();
  expect(within(measured).getByText('Outputs')).toBeVisible();
  expect(measured).toHaveAccessibleDescription('Names · recorded shapes');
  expect(screen.getAllByText('Names · recorded shapes', { exact: true })).toHaveLength(1);
  expect(within(measured).getByText('x (2, 8)')).toBeVisible();
  expect(within(measured).getByText('y (2, 4)')).toBeVisible();
});

it('omits unavailable input and output rows instead of rendering placeholders', () => {
  const data: ModelNodeData = { ...node.data, parameters: {}, inputPorts: [], outputPorts: [] };
  const view = render(<NodeUnderstanding node={{ ...node, data }} />);
  expect(screen.queryByLabelText('Graph inputs and outputs')).not.toBeInTheDocument();
  expect(screen.queryByText('Not recorded')).not.toBeInTheDocument();
  view.rerender(<NodeUnderstanding node={{ ...node, data: { ...data, outputPorts: ['scores'] } }} />);
  const io = screen.getByLabelText('Graph inputs and outputs');
  expect(within(io).getByText('Outputs')).toBeVisible();
  expect(within(io).getByText('scores')).toBeVisible();
  expect(within(io).queryByText('Inputs')).not.toBeInTheDocument();
  expect(io).not.toHaveTextContent('→');
});

it('leaves chat entry to the graph toolbar rather than the reading surface', () => {
  render(<NodeExplanationContext.Provider value={context(ready)}><NodeUnderstanding node={node} /></NodeExplanationContext.Provider>);
  expect(screen.queryByRole('button', { name: 'Discuss in chat' })).not.toBeInTheDocument();
  expect(within(screen.getByLabelText('About this node')).queryByRole('textbox')).not.toBeInTheDocument();
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

it('preserves a retained interpretation beside current authored documentation without requesting generation', () => {
  const current: NodeExplanationState = { ...ready, requestId: 'documentation-read', status: 'unavailable',
    explanation: undefined, documentation };
  const retained: NodeExplanationState = { ...ready, requestId: 'verified-annotation',
    documentation: { ...documentation, qualifiedName: 'old.definition', docstring: 'Outdated author documentation.' } };
  const start = jest.fn();
  const value: NodeExplanationContextValue = { ...context(current), annotations: { [node.id]: retained },
    model: { start, cancel: jest.fn(), toggle: jest.fn() } };
  render(<NodeExplanationContext.Provider value={value}><NodeUnderstanding node={node} /></NodeExplanationContext.Provider>);
  expect(screen.getByText(ready.explanation!.purpose)).toBeVisible();
  expect(screen.getByLabelText('Interpreted inputs and outputs')).toHaveTextContent('Candidate features');
  fireEvent.click(screen.getByText('More details'));
  const authored = screen.getByLabelText('Source documentation details');
  expect(within(authored).getByText(documentation.qualifiedName)).toBeVisible();
  expect(within(authored).getByRole('region', { name: 'Full source documentation' }).textContent).toBe(documentation.docstring);
  expect(screen.queryByText('Outdated author documentation.')).not.toBeInTheDocument();
  expect(screen.getByLabelText('How it works')).toHaveTextContent(ready.explanation!.steps[0]);
  fireEvent.click(within(authored).getByRole('button', { name: 'models/spectral.py:40–61' }));
  expect(value.reveal).toHaveBeenLastCalledWith(current, documentation.citation.id);
  fireEvent.click(within(screen.getByLabelText('Explanation sources')).getByRole('button', { name: 'model.py:20–26' }));
  expect(value.reveal).toHaveBeenLastCalledWith(retained, ready.explanation!.citations[0].id);
  expect(value.retry).not.toHaveBeenCalled();
  expect(start).not.toHaveBeenCalled();
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

it('does not show an explanation error or retry prompt before any explanation was requested', () => {
  const state: NodeExplanationState = { requestId: 'selection', nodeId: node.id, revision: 'revision', status: 'unavailable' };
  render(<NodeExplanationContext.Provider value={context(state)}><NodeUnderstanding node={node} /></NodeExplanationContext.Provider>);
  expect(screen.queryByText('Explanation unavailable.')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Retry explanation' })).not.toBeInTheDocument();
});

it('quotes only the opening source paragraph and bounds long summaries at a word boundary', () => {
  expect(sourceDocumentationSummary('  First line\ncontinued here.\n\nArgs:\n    x: input')).toBe('First line continued here.');
  expect(sourceDocumentationSummary(' \n\n ')).toBeUndefined();
  const summary = sourceDocumentationSummary('LongDocumentation '.repeat(40))!;
  expect(summary.length).toBeLessThanOrEqual(321);
  expect(summary.endsWith('LongDocumentation…')).toBe(true);
});
