import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import type { ModelNode } from '../../types';
import type { ModelExplanationState, NodeExplanationCitation, NodeExplanationState } from '../../ide/nodeExplanation.generated';
import { NodeExplanationContext, type NodeExplanationContextValue } from '../../canvas/NodeExplanationContext';
import { ViewExplanationOverview } from '../../canvas/ViewExplanationOverview';
import { applicableViewExplanation, viewCitationState, viewConnectionLabelLayout, viewExplanationConnection } from '../../canvas/viewExplanationPresentation';
import { graphToSvg } from '../../services/graphSvg';

const node = (id: string, x = 0): ModelNode => ({ id, type: 'layerNode', position: { x, y: 0 }, data: {
  layerType: { id: 'python', name: id, displayName: id, category: 'misc', description: '', icon: 'Code', parameters: [] },
  parameters: { _class: id },
} });
const nodes = [node('sr_cond'), node('CausalDecoder', 450)];
const citation: NodeExplanationCitation = { id: 'view-proof', file: 'layers/audio_vae_v2.py', sourceDigest: 'a'.repeat(64), startLine: 316, endLine: 330 };
const progress: ModelExplanationState = { requestId: 'view-request', revision: 'revision', status: 'ready', completed: 2, total: 2,
  view: { summary: 'Generate a waveform from audio features, using a bucket to condition decoding.',
    nodeIds: nodes.map(node => node.id), connections: [{ sourceNodeId: 'sr_cond', targetNodeId: 'CausalDecoder',
      label: 'Sample-rate conditioning', detail: 'The bucket selects conditioning before decoder blocks.', citations: [citation] }],
    callouts: [{ title: 'Why this branch exists', body: 'The bucket selects how decoder features are conditioned.',
      nodeIds: ['sr_cond', 'CausalDecoder'], citations: [citation] }] } };
const ready: NodeExplanationState = { requestId: 'node-request', nodeId: 'CausalDecoder', revision: 'revision', status: 'ready',
  explanation: { title: 'Generate audio waveform', purpose: 'Expand audio features into waveform samples.',
    inputs: ['Audio features'], outputs: ['Audio waveform'], steps: [], uncertainties: [], citations: [{ ...citation, id: 'node-proof' }] } };
const context = (): NodeExplanationContextValue => ({ states: { CausalDecoder: ready }, annotationsVisible: true,
  annotations: { CausalDecoder: ready }, model: { state: progress, start: jest.fn(), cancel: jest.fn(), toggle: jest.fn() },
  retry: jest.fn(), reveal: jest.fn() });
const tree = (value: NodeExplanationContextValue, extra: Partial<React.ComponentProps<typeof ViewExplanationOverview>> = {}) =>
  <NodeExplanationContext.Provider value={value}><ViewExplanationOverview nodes={nodes} focused={false} onNavigate={jest.fn()} {...extra} /></NodeExplanationContext.Provider>;

describe('visible-view explanation presentation', () => {
  it('keeps commentary scoped to the visible snapshot while allowing ordinary pan and zoom', () => {
    expect(applicableViewExplanation(progress, true, nodes)).toBe(progress.view);
    expect(applicableViewExplanation(progress, false, nodes)).toBeUndefined();
    expect(applicableViewExplanation(progress, true, nodes.slice(1))).toBeUndefined();
    expect(applicableViewExplanation(progress, true, [{ ...nodes[0], hidden: true }, nodes[1]])).toBeUndefined();
    expect(applicableViewExplanation(progress, true, [...nodes, node('another-visible-component')])).toBe(progress.view);
    expect(viewExplanationConnection(progress.view, 'sr_cond', 'CausalDecoder')).toBe(progress.view!.connections[0]);
    expect(viewExplanationConnection(progress.view, 'CausalDecoder', 'sr_cond')).toBeUndefined();
    expect(viewExplanationConnection(progress.view, 'hidden', 'CausalDecoder')).toBeUndefined();
  });

  it('wraps the entire connection meaning without clipping or changing its wording', () => {
    const label = 'Condition this decoder using the selected sample-rate bucket and the complete technical input';
    for (const width of [0, 56, 100, 180, 1000]) {
      const layout = viewConnectionLabelLayout(label, width);
      expect(layout.lines.join('')).toBe(label);
      expect(layout.width).toBeGreaterThanOrEqual(56);
      expect(layout.width).toBeLessThanOrEqual(180);
      expect(layout.height).toBe(layout.lines.length * 15);
    }
  });

  it('resolves only the exact current source proof and uses its host-issued citation ID', () => {
    expect(viewCitationState(citation, 'revision', [{ CausalDecoder: ready }], ['CausalDecoder']))
      .toEqual({ state: ready, citationId: 'node-proof' });
    for (const update of [{ file: 'other.py' }, { sourceDigest: 'b'.repeat(64) }, { startLine: 315 }, { endLine: 331 }]) {
      expect(viewCitationState({ ...citation, ...update }, 'revision', [{ CausalDecoder: ready }], [])).toBeUndefined();
    }
    expect(viewCitationState(citation, 'new-revision', [{ CausalDecoder: ready }], [])).toBeUndefined();
    expect(viewCitationState(citation, 'revision', [{ CausalDecoder: { ...ready, status: 'loading', explanation: undefined } }], [])).toBeUndefined();
  });

  it('puts summary and linked callouts on canvas and uses existing node selection', () => {
    const navigate = jest.fn();
    render(tree(context(), { onNavigate: navigate }));
    const overview = screen.getByLabelText('Explanation of this view');
    expect(within(overview).getByText(progress.view!.summary)).toBeVisible();
    expect(screen.getByText('Why this branch exists')).toBeVisible();
    expect(screen.getByText(progress.view!.callouts[0].body)).not.toBeVisible();
    expect(screen.getByRole('button', { name: 'Select sr_cond' })).not.toBeVisible();
    fireEvent.click(screen.getByText('Why this branch exists'));
    expect(screen.getByText(progress.view!.callouts[0].body)).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Select sr_cond' }));
    expect(navigate).toHaveBeenCalledWith('sr_cond');
    fireEvent.click(screen.getByText('Connection meanings'));
    expect(screen.getByText('Sample-rate conditioning')).toBeVisible();
    expect(screen.getByText(progress.view!.connections[0].detail!)).toBeVisible();
  });

  it('keeps each callout closed until requested and expands independently without hiding the summary', () => {
    const value = context();
    const another = { ...progress.view!.callouts[0], title: 'How the stage is called', body: 'The surrounding loop executes each stored stage.' };
    value.model!.state = { ...progress, view: { ...progress.view!, callouts: [...progress.view!.callouts, another] } };
    render(tree(value));
    const first = screen.getByText('Why this branch exists').closest('details')!;
    const second = screen.getByText(another.title).closest('details')!;
    expect(first).not.toHaveAttribute('open');
    expect(second).not.toHaveAttribute('open');
    fireEvent.click(screen.getByText(another.title));
    expect(first).not.toHaveAttribute('open');
    expect(second).toHaveAttribute('open');
    expect(screen.getByText(another.body)).toBeVisible();
    expect(screen.getByText(progress.view!.summary)).toBeVisible();
    fireEvent.click(screen.getByText(another.title));
    expect(second).not.toHaveAttribute('open');
    expect(screen.getByText(another.body)).not.toBeVisible();
  });

  it('collapses while inspecting a node and keeps an explicit way to reopen the overview', () => {
    const value = context();
    const rendered = render(tree(value));
    rendered.rerender(tree(value, { focused: true, selectedNodeIds: ['CausalDecoder'] }));
    const toggle = screen.getByRole('button', { name: /This view/ });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByText(progress.view!.summary)).toBeNull();
    fireEvent.click(toggle);
    expect(screen.getByText(progress.view!.summary)).toBeVisible();
  });

  it('distinguishes repeated module classes using their source instance names', () => {
    const containers = nodes.map((item, index) => ({ ...item, data: { ...item.data,
      parameters: { _class: 'ModuleList' }, moduleInstanceId: index ? 'sr_cond_model' : 'model' } }));
    const navigate = jest.fn();
    render(tree(context(), { nodes: containers, onNavigate: navigate }));
    fireEvent.click(screen.getByText('Why this branch exists'));
    expect(screen.getByRole('button', { name: 'Select ModuleList · model' })).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Select ModuleList · sr_cond_model' }));
    expect(navigate).toHaveBeenCalledWith('CausalDecoder');
  });

  it('selects a citation owner and waits for its current result before revealing source', () => {
    const value = context(), navigate = jest.fn();
    value.states = {};
    const rendered = render(tree(value, { onNavigate: navigate }));
    fireEvent.click(screen.getByText('Why this branch exists'));
    fireEvent.click(screen.getAllByRole('button', { name: 'Open layers/audio_vae_v2.py:316–330' })[0]);
    expect(navigate).toHaveBeenCalledWith('CausalDecoder');
    expect(value.reveal).not.toHaveBeenCalled();
    const pending = { ...value, states: { CausalDecoder: { ...ready, requestId: 'selected', status: 'loading' as const, explanation: undefined } } };
    rendered.rerender(tree(pending, { focused: true, selectedNodeIds: ['CausalDecoder'], onNavigate: navigate }));
    expect(value.reveal).not.toHaveBeenCalled();
    const selected = { ...ready, requestId: 'selected' };
    rendered.rerender(tree({ ...value, states: { CausalDecoder: selected } }, { focused: true, selectedNodeIds: ['CausalDecoder'], onNavigate: navigate }));
    expect(value.reveal).toHaveBeenCalledTimes(1);
    expect(value.reveal).toHaveBeenCalledWith(selected, 'node-proof');
  });

  it('waits for a new host selection request instead of revealing retained ready batch evidence', () => {
    const value = context(), navigate = jest.fn();
    const retained = { ...ready, requestId: progress.requestId };
    value.states = { CausalDecoder: retained };
    value.annotations = { CausalDecoder: retained };
    const rendered = render(tree(value, { onNavigate: navigate }));
    fireEvent.click(screen.getByText('Why this branch exists'));
    fireEvent.click(screen.getAllByRole('button', { name: 'Open layers/audio_vae_v2.py:316–330' })[0]);
    expect(navigate).toHaveBeenCalledWith('CausalDecoder');
    // Canvas selection is immediate; the host still owns the old batch proof.
    rendered.rerender(tree(value, { focused: true, selectedNodeIds: ['CausalDecoder'], onNavigate: navigate }));
    expect(value.reveal).not.toHaveBeenCalled();
    const queued = { ...ready, requestId: 'host-selection', status: 'queued' as const, explanation: undefined };
    rendered.rerender(tree({ ...value, states: { CausalDecoder: queued } }, { focused: true, selectedNodeIds: ['CausalDecoder'], onNavigate: navigate }));
    expect(value.reveal).not.toHaveBeenCalled();
    const selected = { ...ready, requestId: 'host-selection' };
    rendered.rerender(tree({ ...value, states: { CausalDecoder: selected } }, { focused: true, selectedNodeIds: ['CausalDecoder'], onNavigate: navigate }));
    expect(value.reveal).toHaveBeenCalledTimes(1);
    expect(value.reveal).toHaveBeenCalledWith(selected, 'node-proof');
  });

  it('reveals immediately for a current selected owner and retires pending links when the view changes', () => {
    const value = context();
    const rendered = render(tree(value, { selectedNodeIds: ['CausalDecoder'] }));
    fireEvent.click(screen.getByText('Why this branch exists'));
    fireEvent.click(screen.getAllByRole('button', { name: 'Open layers/audio_vae_v2.py:316–330' })[0]);
    expect(value.reveal).toHaveBeenCalledWith(ready, 'node-proof');
    rendered.unmount();
    const other = context(); other.states = {};
    const waiting = render(tree(other));
    fireEvent.click(screen.getByText('Why this branch exists'));
    fireEvent.click(screen.getAllByRole('button', { name: 'Open layers/audio_vae_v2.py:316–330' })[0]);
    const changed = { ...other, model: { ...other.model!, state: { ...progress, requestId: 'different-view' } } };
    waiting.rerender(tree(changed));
    waiting.rerender(tree({ ...changed, states: { CausalDecoder: ready } }, { selectedNodeIds: ['CausalDecoder'] }));
    expect(other.reveal).not.toHaveBeenCalled();
  });

  it('does not expose commentary or source controls when proofs or scope are unavailable', () => {
    const value = context(); value.states = {}; value.annotations = {};
    const rendered = render(tree(value));
    expect(screen.queryByRole('button', { name: /^Open layers/ })).toBeNull();
    rendered.rerender(tree({ ...value, annotationsVisible: false }));
    expect(screen.queryByLabelText('Explanation of this view')).toBeNull();
    rendered.rerender(tree(context(), { nodes: nodes.slice(1) }));
    expect(screen.queryByLabelText('Explanation of this view')).toBeNull();
  });

  it('exports meanings only on existing directed edges in the same scope, safely escaping generated text', () => {
    const state = { ...progress, view: { ...progress.view!, connections: [{ ...progress.view!.connections[0], label: 'Bucket <select> & condition' }] } };
    const edge = { id: 'condition', source: 'sr_cond', target: 'CausalDecoder' };
    const options = { annotationsVisible: true, viewExplanation: state };
    const svg = graphToSvg(nodes, [edge], options);
    const doc = new DOMParser().parseFromString(svg, 'image/svg+xml');
    expect(doc.querySelectorAll('[data-agent-connection]')).toHaveLength(1);
    expect(Array.from(doc.querySelectorAll('[data-agent-connection] text'), text => text.textContent).join('')).toBe('Bucket <select> & condition');
    expect(doc.querySelector('select')).toBeNull();
    expect(graphToSvg(nodes, [], options)).not.toContain('data-agent-connection');
    expect(graphToSvg(nodes, [{ ...edge, source: edge.target, target: edge.source }], options)).not.toContain('data-agent-connection');
    expect(graphToSvg(nodes, [edge], { ...options, annotationsVisible: false })).not.toContain('data-agent-connection');
    expect(graphToSvg([{ ...nodes[0], hidden: true }, nodes[1]], [edge], options)).not.toContain('data-agent-connection');
  });
});
