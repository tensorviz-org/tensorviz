const mockLayouts: string[][] = [];

jest.mock('dagre', () => {
  const actual = jest.requireActual('dagre');
  return {
    ...actual,
    layout: (...args: any[]) => {
      mockLayouts.push(args[0].nodes());
      return actual.layout(...args);
    },
  };
});
jest.mock('../../ide/bridge', () => ({ postToHost: jest.fn() }));

import { applyProjection } from '../../ide/projection';
import { mergeProjection } from '../../ide/applyProjection';
import { ProjectionPayload } from '../../ide/protocol.generated';
import { graphDocumentV2ToModelGraph } from '../../services/graphDocumentAdapter';
import { reconcileModuleView } from '../../services/canvasLayout';
import { GraphDocumentV2 } from '../../types/graphDocument';
import captureFixture from '../../test/fixtures/canvas/capture-document.json';

let nextGraph = 0;
const payload = (collapsed = false, nested = false): ProjectionPayload => {
  const document = JSON.parse(JSON.stringify(captureFixture)) as GraphDocumentV2;
  if (nested) {
    // Wrap the real captured fixture in one more frame and give every incoming
    // node a nonzero position. Neither nesting nor tracer positions may make
    // the discarded first top-level pass affect the final geometry.
    const inner = document.semantic.nodes.find(node => node.kind === 'module')!;
    const outerId = 'invocation:outer';
    document.semantic.nodes.push({ ...inner, node_id: outerId, in_ports: [], out_ports: [],
      metadata: { ...inner.metadata, captureRef: { ...(inner.metadata.captureRef as object), semanticId: 'call:outer' },
        capturePorts: [], moduleInstanceId: outerId, moduleMembers: [inner.node_id] } });
    inner.metadata.parentId = outerId;
    document.semantic.nodes.forEach((node, index) => { document.view.node_positions[node.node_id] = { x: 900 + index * 30, y: 600 + index * 20 }; });
  }
  if (collapsed) document.view.collapsed_modules = document.semantic.nodes.filter(node => node.kind === 'module').map(node => node.node_id);
  return { graphId: `initial-layout:${nextGraph++}`, cls: 'Net', file: 'net.py', sourceDigest: 'digest', revision: 1,
    traced: true, greyed: false, viewRestored: false, spanmap: {}, document };
};
const inputId = captureFixture.semantic.nodes.find(node => node.kind === 'input')!.node_id;
const topLevelCalls = () => mockLayouts.filter(ids => ids.includes(inputId)).length;

beforeEach(() => { mockLayouts.length = 0; window.__TENSORVIZ_LOCAL_BROWSER__ = false; });
afterEach(() => { window.__TENSORVIZ_LOCAL_BROWSER__ = false; });

describe.each([undefined, 'horizontal', 'vertical'] as const)('first capture layout (%s)', direction => {
  it.each([[false, false], [true, false], [false, true], [true, true]])('places the top level once with identical geometry (collapsed=%s, nested=%s)', (collapsed, nested) => {
    const projection = payload(collapsed, nested);
    // The previous two-pass path is an independent geometry oracle. Use real
    // dagre underneath the call counter so boundary alignment is covered too.
    const graph = graphDocumentV2ToModelGraph(projection.document as GraphDocumentV2);
    const reconciled = reconcileModuleView(graph.nodes, graph.edges, { relayout: true, direction });
    const expected = mergeProjection({ ...graph, ...reconciled }, [], { direction });
    expect(topLevelCalls()).toBe(2);
    const oldCalls = mockLayouts.length;
    mockLayouts.length = 0;

    const actual = applyProjection(projection, [], direction)!;
    expect(topLevelCalls()).toBe(1);
    expect(mockLayouts).toHaveLength(oldCalls - 1);
    expect(actual.nodes).toEqual(expected.nodes);
    expect(actual.edges).toEqual(expected.edges);
  });
});

it('keeps moved positions and selection on a same-capture refresh without layout', () => {
  const projection = payload();
  const first = applyProjection(projection, [])!;
  const moved = first.nodes.map(node => ({ ...node, selected: true,
    position: { x: node.position.x + 80, y: node.position.y + 40 } }));
  mockLayouts.length = 0;
  const refreshed = applyProjection(projection, moved)!;
  expect(mockLayouts).toHaveLength(0);
  expect(refreshed.nodes.map(node => node.position)).toEqual(moved.map(node => node.position));
  expect(refreshed.nodes.every(node => node.selected)).toBe(true);
});

it('does not auto-place a restored view', () => {
  const projection = payload();
  projection.viewRestored = true;
  const document = projection.document as GraphDocumentV2;
  document.semantic.nodes.forEach((node, index) => { document.view.node_positions[node.node_id] = { x: 100 + index * 50, y: 700 }; });
  const restored = applyProjection(projection, [])!;
  expect(mockLayouts).toHaveLength(0);
  expect(restored.nodes.map(node => node.position)).toEqual(restored.nodes.map(node => document.view.node_positions[node.id]));
});

it.each([false, true])('settles temporary badges once unless the user moved a card (moved=%s)', moved => {
  window.__TENSORVIZ_LOCAL_BROWSER__ = true;
  const projection = payload(true);
  const first = applyProjection({ ...projection, greyed: true }, [])!;
  expect(topLevelCalls()).toBe(1);
  const before = moved ? first.nodes.map(node => ({ ...node,
    position: { x: node.position.x + 20, y: node.position.y + 10 } })) : first.nodes;
  mockLayouts.length = 0;
  const settled = applyProjection(projection, before)!;
  expect(topLevelCalls()).toBe(moved ? 0 : 1);
  if (moved) expect(settled.nodes.map(node => node.position)).toEqual(before.map(node => node.position));
  mockLayouts.length = 0;
  const refreshed = applyProjection(projection, settled.nodes)!;
  expect(mockLayouts).toHaveLength(0);
  expect(refreshed.nodes.map(node => node.position)).toEqual(settled.nodes.map(node => node.position));
});

it('still reconciles a new capture over an existing view', () => {
  const projection = payload();
  const first = applyProjection(projection, [])!;
  const moved = first.nodes.map(node => ({ ...node, position: { x: node.position.x + 80, y: node.position.y + 40 } }));
  const document = JSON.parse(JSON.stringify(projection.document)) as GraphDocumentV2;
  for (const node of document.semantic.nodes) (node.metadata.captureRef as { captureId: string }).captureId = 'capture:next';
  mockLayouts.length = 0;
  const refreshed = applyProjection({ ...projection, document }, moved)!;
  expect(topLevelCalls()).toBe(1);
  expect(refreshed.nodes.map(node => node.position)).toEqual(moved.map(node => node.position));
});
