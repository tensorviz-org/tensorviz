import React from 'react';
import { render, screen, within } from '@testing-library/react';
import { NodeExplanationAnnotation } from '../../canvas/NodeExplanationAnnotation';
import { explainedGraphPresentation, nodeAnnotationLayout } from '../../canvas/nodeExplanationPresentation';
import { ioCardContent, ioCardHeight, nodeCardSize } from '../../canvas/cardMetrics';
import type { ModelNode } from '../../types';
import type { NodeExplanation, NodeExplanationState } from '../../ide/nodeExplanation.generated';

const explanation: NodeExplanation = { title: 'Score candidates', purpose: 'Compare candidate representations with the question.',
  inputs: ['Candidate features', 'Question features'], outputs: ['Raw candidate scores'], steps: [], uncertainties: [],
  citations: [{ id: 'source', file: 'model.py', sourceDigest: 'a'.repeat(64), startLine: 10, endLine: 20 }] };
const state = (id: string, value = explanation): NodeExplanationState => ({ requestId: 'request', nodeId: id, revision: 'revision', status: 'ready', explanation: value });
const node = (id: string, x = 0, y = 0): ModelNode => ({ id, type: 'layerNode', position: { x, y },
  data: { layerType: { id: 'python', name: 'Python', category: 'misc' }, parameters: { _raw_call: 'self._score_hidden(candidates, question)' } } } as ModelNode);

it('leaves source nodes and layout untouched until model explanations are explicitly shown', () => {
  const nodes = [node('score')];
  expect(explainedGraphPresentation(nodes, { score: state('score') }, false).nodes).toBe(nodes);
  expect(explainedGraphPresentation(nodes, {}, true).nodes).toBe(nodes);
});

it('adds measured space and displaces colliding siblings without editing graph records', () => {
  const nodes = [node('score'), node('next', 0, 100), node('parallel', 700, 0)];
  const before = JSON.stringify(nodes);
  const view = explainedGraphPresentation(nodes, { score: state('score') }, true);
  const first = view.nodes[0], second = view.nodes[1];
  const firstSize = view.sizes.get(first.id)!;
  expect(firstSize.h).toBe(nodeCardSize(nodes[0], firstSize.w).h + nodeAnnotationLayout(explanation, firstSize.w).height);
  expect(second.position.y).toBeGreaterThanOrEqual(first.position.y + firstSize.h + 24);
  expect(view.nodes[2].position).toEqual(nodes[2].position);
  expect(view.offsets.get('next')!.y).toBe(second.position.y - nodes[1].position.y);
  expect(first.data).toBe(nodes[0].data);
  expect(JSON.stringify(nodes)).toBe(before);
});

it('reserves annotation clearance inside expanded frames and grows around enriched children', () => {
  const frame: ModelNode = { ...node('frame'), type: 'moduleNode', style: { width: 600, height: 200 },
    data: { ...node('frame').data, moduleTemplateId: 'VideoQuestionModel', moduleLayerCount: 1, expandable: true, expanded: true } };
  const child = { ...node('child', 24, 80), parentNode: 'frame' };
  const nodes = [frame, child, node('next', 0, 250)];
  const view = explainedGraphPresentation(nodes, { frame: state('frame'), child: state('child') }, true);
  const shownFrame = view.nodes[0], shownChild = view.nodes[1];
  expect(shownChild.position.y).toBe(child.position.y + nodeAnnotationLayout(explanation, 600).height);
  expect(view.sizes.get('frame')!.h).toBeGreaterThanOrEqual(shownChild.position.y + view.sizes.get('child')!.h + 20);
  expect(view.nodes[2].position.y).toBeGreaterThan(shownFrame.position.y + view.sizes.get('frame')!.h);
  expect(nodes[1].position.y).toBe(80);
});

it('keeps review and interface cards free of generated interpretations', () => {
  const review = { ...node('review'), data: { ...node('review').data, review: { status: 'unchanged', params: {} } } } as ModelNode;
  const input = { ...node('input'), type: 'inputNode' };
  const nodes = [review, input];
  expect(explainedGraphPresentation(nodes, { review: state('review'), input: state('input') }, true).nodes).toBe(nodes);
});

it('wraps complete generated titles and shows compact flow with full values available', () => {
  const value = { ...explanation, title: 'CompareCandidatesAgainstEveryQuestionRepresentationWithoutChangingTheirOrder',
    inputs: ['A long description of a candidate representation whose complete meaning remains available in the selection details.', 'Question'] };
  const layout = nodeAnnotationLayout(value, 280);
  const { container } = render(<NodeExplanationAnnotation explanation={value} width={280} />);
  expect(layout.titleLines.join('')).toBe(value.title);
  expect(container.querySelector('section')).toHaveStyle({ height: `${layout.height}px` });
  expect(within(screen.getByLabelText('Agent interpretation')).getByText('In')).toBeVisible();
  expect(screen.getByTitle(value.inputs.join(' · '))).toBeInTheDocument();
  expect(screen.getByTitle(value.purpose)).toBeInTheDocument();
  expect(layout.inputLines.join('')).toBe(value.inputs.join(' · '));
  expect(container.querySelector('strong')!.textContent).toBe(value.title);
});

it('measures long root names and high-rank shapes at the actual card width', () => {
  const data = { ...node('input').data, parameters: { name: 'conditioning_audio_features_'.repeat(8) },
    inputShape: Array.from({ length: 24 }, (_, index) => index + 1) };
  const narrow = ioCardContent(data, 'input', 200), wide = ioCardContent(data, 'input', 420);
  expect(narrow.values.name!.lines.join('')).toBe(data.parameters.name);
  expect(narrow.values.shape!.lines.join('')).toBe(`(${data.inputShape.join(', ')})`);
  expect(ioCardHeight(data, 'input', 200)).toBeGreaterThan(ioCardHeight(data, 'input', 420));
  expect(narrow.bodyHeight).toBeGreaterThan(wide.bodyHeight);
});
