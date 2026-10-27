import { applyCanvasFocus, planCanvasFocus, queuedEditorFocus } from '../../canvas/focusSelection';
import type { ModelNode, ModelEdge } from '../../types';
import type { Editor } from 'tldraw';
import { selectedEdgeId } from '../../canvas/edgeSelection';
import { nodeIdToShapeId } from '../../canvas/canvasShapeId';
jest.mock('tldraw', () => ({ Box: { Common: (bounds: unknown) => bounds } }));
jest.mock('../../canvas/edgeSelection', () => ({ selectedEdgeId: { set: jest.fn() } }));
const nodes = [{ id: 'block', type: 'moduleNode', data: { expanded: false, parameters: { width: 8 } } },
  { id: 'block.fc', data: { parentId: 'block' } }, { id: 'out', data: {} }] as ModelNode[];
const edges = [{ id: 'e1', source: 'block.fc', target: 'out' }] as ModelEdge[];

test('an editor click waits until its node is on the canvas, then selects it', () => {
  expect(queuedEditorFocus('block.fc', ['block'])).toEqual({ pending: 'block.fc', focus: null });
  expect(queuedEditorFocus('block.fc', ['block', 'block.fc'])).toEqual({ pending: null, focus: 'block.fc' });
});

test('MCP and quick jump reveal a folded pass-through call along with its ancestors', () => {
  const graph = nodes.map(node => node.id === 'block.fc' ? { ...node, data: { ...node.data,
    capturePassThrough: { valueId: 'v', inputPortId: 'in', sourceNodeId: 'out', sourcePortId: 'out' } } } : node);
  const plan = planCanvasFocus(graph, edges, ['block.fc'], [])!;
  expect(plan.nodes[0].data.expanded).toBe(true);
  expect(plan.nodes[1].data.passThroughShown).toBe(true);
  expect(graph[1].data.passThroughShown).toBeUndefined();
});

test('reveals a nested connection without mutating architecture or original layout state', () => {
  const plan = planCanvasFocus(nodes, edges, [], ['e1'])!;
  expect(plan.nodes[0].data.expanded).toBe(true);
  expect(nodes[0].data.expanded).toBe(false);
  expect(plan.nodes[0].data.parameters).toEqual({ width: 8 });
  expect(plan.edge).toBe(edges[0]);
  expect(planCanvasFocus(nodes, edges, ['missing'], [])).toBeNull();
});

test('does not claim focus before the expanded shape exists', () => {
  const editor = { getShapePageBounds: () => undefined, select: jest.fn() } as unknown as Editor;
  expect(applyCanvasFocus(editor, planCanvasFocus(nodes, edges, ['block.fc'], [])!)).toBe(false);
  expect(editor.select).not.toHaveBeenCalled();
});

test('selects exact node shape identities and shows the requested bounds', () => {
  const editor = { getShapePageBounds: jest.fn(() => ({ x: 0, y: 0, w: 100, h: 80 })), select: jest.fn(), setCurrentTool: jest.fn(), zoomToBounds: jest.fn() } as unknown as Editor;
  expect(applyCanvasFocus(editor, planCanvasFocus(nodes, edges, ['block.fc', 'out'], [])!)).toBe(true);
  expect(editor.select).toHaveBeenCalledWith(nodeIdToShapeId('block.fc'), nodeIdToShapeId('out'));
  expect(editor.zoomToBounds).toHaveBeenCalled();
  expect(selectedEdgeId.set).toHaveBeenCalledWith(null);
});
