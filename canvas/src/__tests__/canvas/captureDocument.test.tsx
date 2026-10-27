import React from 'react';
import { render, screen } from '@testing-library/react';
import fixture from '../../test/fixtures/canvas/capture-document.json';
import { GraphDocumentV2, ModelGraph } from '../../types';
import { graphDocumentV2ToModelGraph, modelGraphToDocumentV2 } from '../../services/graphDocumentAdapter';
import { reconcileModuleView } from '../../services/canvasLayout';
import { portCardHeight, endpointOnSide } from '../../canvas/capturePorts';
import { CaptureBoundaryRows } from '../../canvas/CaptureBoundaryRows';
import { moduleContents } from '../../services/modulePresentation';
import { overlayScale } from '../../canvas/graphOverlay';
import { applyProjection } from '../../ide/projection';
import { ProjectionPayload } from '../../ide/protocol.generated';

jest.mock('../../ide/bridge', () => ({ postToHost: jest.fn() }));
beforeEach(() => { window.__TENSORVIZ_LOCAL_BROWSER__ = true; });
afterEach(() => { delete window.__TENSORVIZ_LOCAL_BROWSER__; });

const convert = () => graphDocumentV2ToModelGraph(fixture as GraphDocumentV2);
const context = { name: 'Model', graphId: 'captured-model', revision: 0 };
const semantic = (graph: ModelGraph) => modelGraphToDocumentV2(graph, context).semantic;

test('actual Python-produced document retains semantic references, typed ports and repeated arguments', () => {
  const graph = convert();
  const restored = semantic(graph);
  expect(restored.nodes.map(n => n.metadata.captureRef)).toEqual(fixture.semantic.nodes.map(n => n.metadata.captureRef));
  expect(restored.nodes.map(n => n.metadata.capturePorts)).toEqual(fixture.semantic.nodes.map(n => n.metadata.capturePorts));
  expect(restored.edges.map(e => e.metadata.captureEdge)).toEqual(fixture.semantic.edges.map(e => e.metadata.captureEdge));
  const add = graph.nodes.find(n => n.data.captureRef?.semanticId === 'add2')!;
  const incoming = graph.edges.filter(e => e.target === add.id);
  expect(incoming).toHaveLength(2);
  expect(new Set(incoming.map(e => e.targetHandle)).size).toBe(2);
  expect(new Set(incoming.map(e => e.sourceHandle)).size).toBe(1);
});

test('canonical metric evidence survives conversion/collapse and disappears on stale projection', () => {
  const graph = convert();
  const root = graph.nodes.find(n => n.data.captureRef?.kind === 'invocation')!;
  expect(overlayScale(graph.nodes, 'params').values.get(root.id)?.value).toBe(0);
  expect(overlayScale(graph.nodes, 'activation')).toMatchObject({ min: 32, max: 32 });
  const restored = graphDocumentV2ToModelGraph(modelGraphToDocumentV2(graph, context));
  expect(overlayScale(restored.nodes, 'activation')).toEqual(overlayScale(graph.nodes, 'activation'));
  const collapsed = reconcileModuleView(graph.nodes.map(n => n.id === root.id ? { ...n, data: { ...n.data, expanded: false } } : n), graph.edges);
  expect(overlayScale(collapsed.nodes, 'activation').values.get(root.id)?.value).toBe(32);
  expect(overlayScale(collapsed.nodes, 'activation').values.size).toBeLessThan(overlayScale(graph.nodes, 'activation').values.size);
  const stale = applyProjection({ document: fixture, graphId: 'captured-model', file: 'model.py', cls: 'Model',
    sourceDigest: 'same-source', traced: true, greyed: true, viewRestored: true } as unknown as ProjectionPayload, graph.nodes)!;
  expect(overlayScale(stale.nodes, 'activation').values.size).toBe(0);
  expect(overlayScale(stale.nodes, 'params').values.size).toBe(0);
});

test('collapse only changes display ports and round trips without persisting synthetic facts', () => {
  const graph = convert();
  const before = JSON.stringify(graph);
  const root = graph.nodes.find(n => n.data.captureRef?.kind === 'invocation')!;
  const collapsed = reconcileModuleView(graph.nodes.map(n => n.id === root.id ? { ...n, data: { ...n.data, expanded: false } } : n), graph.edges);
  expect(JSON.stringify(graph)).toBe(before);
  expect(semantic({ ...graph, ...collapsed })).toEqual(semantic(graph));
  const boundary = collapsed.nodes.find(n => n.id === root.id)!.data.captureDisplayPorts!;
  expect(boundary).toHaveLength(2);
  expect(boundary.every(p => p.id === `boundary:${p.memberPortId}` && p.paths.length === 0)).toBe(true);
  expect(collapsed.edges.filter(e => e.data?.synthetic).every(e => e.data!.originalEdgeIds!.length === 1)).toBe(true);
  const expanded = reconcileModuleView(collapsed.nodes.map(n => n.id === root.id ? { ...n, data: { ...n.data, expanded: true } } : n), collapsed.edges);
  expect(semantic({ ...graph, ...expanded })).toEqual(semantic(graph));
  expect(expanded.edges.every(e => !e.hidden && !e.data?.synthetic)).toBe(true);
  expect(moduleContents(root.data)).toMatchObject({ canExpand: true, label: '' });
});

test('two argument positions crossing a collapsed boundary stay two connections', () => {
  const graph = convert();
  const add1 = graph.nodes.find(n => n.data.captureRef?.semanticId === 'add1')!;
  const root = graph.nodes.find(n => n.data.captureRef?.kind === 'invocation')!;
  // Move the producer outside the displayed region; collapse must still retain
  // both destination ports rather than combining edges by card pair.
  add1.data.parentId = undefined;
  root.data.expanded = false;
  const projected = reconcileModuleView(graph.nodes, graph.edges);
  const repeated = projected.edges.filter(e => e.data?.synthetic && e.source === add1.id);
  expect(repeated).toHaveLength(2);
  expect(new Set(repeated.map(e => e.targetHandle)).size).toBe(2);
  const shown = projected.nodes.find(n => n.id === root.id)!;
  const box = { x: 0, y: 0, w: 215, h: 240 };
  const points = repeated.map(e => endpointOnSide(shown.data, e.targetHandle, box, 'left'));
  expect(points[0].y).not.toBe(points[1].y);
  expect(projected.edges.filter(e => !e.data?.synthetic).map(e => e.data?.captureEdge)).toEqual(graph.edges.map(e => e.data?.captureEdge));
});

test('Whisper-style boundaries show separate shapes/dtypes, including a scalar and literal', () => {
  const port = convert().nodes.find(n => n.type === 'inputNode')!.data.capturePorts![0];
  const ports = [
    { ...port, id: 'mel', shape: [1, 80, 32], dtype: 'torch.float32' },
    { ...port, id: 'tokens', paths: [['inputs', 'args', 1]], shape: [1, 4], dtype: 'torch.int64' },
    { ...port, id: 'scalar', paths: [['output']], shape: [] },
    { ...port, id: 'literal', paths: [['output', 'items', 1]], argumentKind: 'int' as const, valueId: null, shape: null, dtype: null },
  ];
  render(<CaptureBoundaryRows ports={ports} />);
  expect(screen.getByTitle('(1, 80, 32) · float32')).toHaveTextContent('(1, 80, 32) · float32');
  expect(screen.getByTitle('(1, 4) · int64')).toHaveTextContent('(1, 4) · int64');
  expect(screen.getByTitle('() · float32')).toHaveTextContent('() · float32');
  expect(screen.getByText('int')).toBeInTheDocument();
  expect(portCardHeight({ capturePorts: Array.from({ length: 30 }, (_, i) => ({ ...port, id: String(i) })) }, 120)).toBeGreaterThan(480);
});
