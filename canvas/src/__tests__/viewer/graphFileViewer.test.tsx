import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import GraphFileViewer from '../../viewer/GraphFileViewer';
import { createGraphFile } from '../../services/graphFile';
import { graphDocumentV2ToModelGraph } from '../../services/graphDocumentAdapter';
import { initialGraphFileView, revealGraphNode } from '../../viewer/graphFileView';
import { reconcileModuleView } from '../../services/canvasLayout';
import { computePortSides } from '../../canvas/edgeAnchors';
import { nestedGraph } from '../../test/fixtures/graphFileFixture';
import { capturedFile } from '../../test/fixtures/capturedGraphFileFixture';

jest.mock('../../viewer/GraphFileCanvas', () => ({ __esModule: true, default: ({ file }: any) => <div data-testid="graph-canvas">{file.document.semantic.nodes.length} nodes</div> }));
const file = () => new File([JSON.stringify(createGraphFile(nestedGraph(), { name: 'NestedNet', status: 'static' }))], 'model.tensorviz.json', { type: 'application/json' });
const choose = (file: File) => fireEvent.change(screen.getByLabelText('Graph file'), { target: { files: [file] } });

it('opens a file with no account, session or backend and states snapshot provenance', async () => {
  const fetchSpy = jest.fn();
  const previousFetch = global.fetch;
  global.fetch = fetchSpy;
  try {
    render(<GraphFileViewer />);
    expect(screen.getByText(/No upload, account or Python setup/)).toBeInTheDocument();
    choose(file());
    expect(await screen.findByText('NestedNet')).toBeInTheDocument();
    expect(screen.getByText(/Static snapshot/)).toBeInTheDocument();
    expect(screen.getByTestId('graph-canvas')).toHaveTextContent('5 nodes');
    expect(fetchSpy).not.toHaveBeenCalled();
  } finally { global.fetch = previousFetch; }
});

it('keeps the last valid graph when a replacement file is malformed', async () => {
  render(<GraphFileViewer />); choose(file());
  await screen.findByText('NestedNet');
  choose(new File(['not JSON'], 'bad.tensorviz.json'));
  expect(await screen.findByRole('alert')).toHaveTextContent('previous graph is still open');
  expect(screen.getByTestId('graph-canvas')).toHaveTextContent('5 nodes');
  choose(file());
  await waitFor(() => expect(screen.queryByRole('alert')).toBeNull());
});

it('identifies saved JAX programs as inferred without claiming an executed path', async () => {
  const value = capturedFile();
  Object.assign(value.capture!.capture, { backend: 'jaxpr', jaxVersion: '0.10.2', torchVersion: null });
  value.capture!.capture.coverage.scope = 'traced_program';
  value.document.semantic.nodes.forEach(node => { node.metadata.shapeEvidence = 'inferred'; });
  render(<GraphFileViewer />);
  choose(new File([JSON.stringify(value)], 'jax.tensorviz.json'));
  expect(await screen.findByText(/inferred shapes, not revalidated here/)).toBeInTheDocument();
  expect(screen.getByText('Complete capture of the traced program')).toBeInTheDocument();
  expect(screen.queryByText(/one execution path/)).toBeNull();
});

it('accepts file drop and rejects multiple files without clearing the graph', async () => {
  render(<GraphFileViewer />);
  fireEvent.drop(screen.getByRole('main'), { dataTransfer: { files: [file()], types: ['Files'] } });
  await screen.findByText('NestedNet');
  fireEvent.drop(screen.getByRole('main'), { dataTransfer: { files: [file(), file()], types: ['Files'] } });
  expect(screen.getByRole('alert')).toHaveTextContent('Choose one');
  expect(screen.getByText('NestedNet')).toBeInTheDocument();
});

it('searching a hidden layer reveals every ancestor without changing snapshot semantics', () => {
  const file = createGraphFile(nestedGraph(), { name: 'Net', status: 'static' });
  const before = JSON.stringify(file);
  const graph = graphDocumentV2ToModelGraph(file.document);
  const view = revealGraphNode(graph, 'fc');
  expect(view.nodes.find(n => n.id === 'block')?.data.expanded).toBe(true);
  expect(view.nodes.find(n => n.id === 'inner')?.data.expanded).toBe(true);
  expect(view.nodes.find(n => n.id === 'inner')?.data.moduleLayerCount).toBe(1);
  expect(view.nodes.find(n => n.id === 'fc')?.hidden).toBe(false);
  expect(JSON.stringify(file)).toBe(before);
  expect(view.edges.filter(e => !e.data?.synthetic).map(e => [e.source, e.target])).toEqual(graph.edges.map(e => [e.source, e.target]));
});

it('opens the exported positions exactly, including nested children', () => {
  const source = nestedGraph();
  source.nodes[1].data.expanded = true; source.nodes[2].data.expanded = true;
  source.nodes[3].position = { x: 190, y: 175 };
  const file = createGraphFile(source, { name: 'Net', status: 'static' });
  const rendered = initialGraphFileView(file);
  for (const node of source.nodes) expect(rendered.nodes.find(n => n.id === node.id)?.position).toEqual(node.position);
  expect(Number(rendered.nodes.find(n => n.id === 'inner')?.style?.width)).toBeGreaterThan(190);
});

it.each(['horizontal', 'vertical'] as const)('opens and expands a %s export with matching positions and ports', direction => {
  const source = nestedGraph();
  source.nodes.forEach(node => { if (node.type === 'moduleNode') node.data.expanded = true; });
  const laid = reconcileModuleView(source.nodes, source.edges, { relayout: true, direction });
  const file = createGraphFile(laid, { name: 'Net', status: 'static', direction });
  const rendered = initialGraphFileView(file);
  for (const node of laid.nodes) expect(rendered.nodes.find(n => n.id === node.id)?.position).toEqual(node.position);
  const expanded = revealGraphNode(rendered, 'fc', file.document.view.layout.direction);
  expect(computePortSides(expanded.nodes, file.document.view.layout.direction).get('fc')).toEqual(
    direction === 'horizontal' ? { in: 'left', out: 'right' } : { in: 'top', out: 'bottom' });
  const input = expanded.nodes.find(node => node.id === 'input')!;
  const output = expanded.nodes.find(node => node.id === 'output')!;
  expect(output.position[direction === 'horizontal' ? 'x' : 'y']).toBeGreaterThan(input.position[direction === 'horizontal' ? 'x' : 'y']);
});
