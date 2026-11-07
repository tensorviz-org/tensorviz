import dagre from 'dagre';
import { reconcileModuleView } from '../../services/canvasLayout';
import { ModelNode, ModelEdge } from '../../types';
import { hierarchicalToFlat } from '../../services/hierarchicalToFlat';
import { autoModularize } from '../../services/autoModularize';

const frame = (id: string, parentId?: string): ModelNode => ({
  id, type: 'moduleNode', position: { x: 40, y: 80 },
  data: { layerType: { id: 'module', name: 'Module', displayName: 'Module', description: '', icon: 'Package', category: 'module', parameters: [] }, parameters: {},
    parentId, expanded: true, moduleLayoutState: 'custom' },
} as ModelNode);
const child = (id: string, index: number, parentId = 'inner'): ModelNode => ({
  id, type: 'layerNode', position: { x: 30 + index * 350, y: 90 + index * 250 },
  data: { layerType: { id: 'linear', name: 'Linear', displayName: 'Linear', description: '', icon: 'Box', category: 'linear', parameters: [] }, parameters: {}, parentId },
} as ModelNode);
const graph = (count = 3) => ({
  nodes: [frame('outer'), frame('inner', 'outer'), ...Array.from({ length: count }, (_, i) => child(`c${i}`, i))],
  edges: [
    ...Array.from({ length: count - 1 }, (_, i) => ({ id: `e${i}`, source: `c${i}`, target: `c${i + 1}` })),
    { id: 'skip', source: 'c0', target: `c${count - 1}` },
  ] as ModelEdge[],
});

afterEach(() => jest.restoreAllMocks());

it.each([undefined, 'horizontal', 'vertical'] as const)('restores nested saved geometry without invoking Dagre: %s', direction => {
  const input = graph(), before = JSON.stringify(input);
  const layout = jest.spyOn(dagre, 'layout');
  const restored = reconcileModuleView(input.nodes, input.edges, { preserveChildPositions: true, direction });
  expect(layout).not.toHaveBeenCalled();
  for (const node of restored.nodes) expect(node.position).toEqual(input.nodes.find(n => n.id === node.id)!.position);
  expect(reconcileModuleView(restored.nodes, restored.edges, { preserveChildPositions: true, direction })).toEqual(restored);
  expect(layout).not.toHaveBeenCalled();
  expect(JSON.stringify(input)).toBe(before);
});

it.each([8, 32, 128])('does no discarded Dagre work when restoring %i children', count => {
  const input = graph(count), layout = jest.spyOn(dagre, 'layout');
  reconcileModuleView(input.nodes, input.edges, { preserveChildPositions: true });
  expect(layout).not.toHaveBeenCalled();
});

it.each([undefined, null])('uses the full child-layout fallback for a missing position (%s)', position => {
  const input = graph();
  input.nodes[3].position = position as any;
  const before = JSON.stringify(input), layout = jest.spyOn(dagre, 'layout');
  const result = reconcileModuleView(input.nodes, input.edges, { preserveChildPositions: true });
  expect(layout).toHaveBeenCalledTimes(1);
  const laidGraph = layout.mock.calls[0][0];
  expect(laidGraph.nodes()).toEqual(['c0', 'c1', 'c2']);
  expect(result.nodes.find(n => n.id === 'c0')!.position).toEqual(input.nodes[2].position);
  // Reconcile's restore assembly retains malformed input positions too; this
  // optimization must not silently repair them or alter the existing fallback.
  expect(result.nodes.find(n => n.id === 'c1')!.position).toBe(position);
  expect(JSON.stringify(input)).toBe(before);
});

it('keeps first-occurrence saved positions for duplicate child ids', () => {
  const input = graph();
  input.nodes.push(child('c0', 10));
  const layout = jest.spyOn(dagre, 'layout');
  const result = reconcileModuleView(input.nodes, input.edges, { preserveChildPositions: true });
  expect(layout).not.toHaveBeenCalled();
  expect(result.nodes.filter(n => n.id === 'c0').map(n => n.position)).toEqual([input.nodes[2].position, input.nodes[input.nodes.length - 1].position]);
  const unique = graph();
  const expected = reconcileModuleView(unique.nodes, unique.edges, { preserveChildPositions: true });
  expect(result.nodes.find(n => n.id === 'inner')!.style).toEqual(expected.nodes.find(n => n.id === 'inner')!.style);
});

it('does not substitute a later duplicate for a missing first position', () => {
  const input = graph();
  input.nodes[2].position = undefined as any;
  input.nodes.push(child('c0', 10));
  const layout = jest.spyOn(dagre, 'layout');
  const result = reconcileModuleView(input.nodes, input.edges, { preserveChildPositions: true });
  expect(layout).toHaveBeenCalledTimes(1);
  expect(result.nodes.find(n => n.id === 'c0')!.position).not.toEqual(input.nodes[input.nodes.length - 1].position);
});

it('still lays out children when saved positions were not requested', () => {
  const input = graph(), layout = jest.spyOn(dagre, 'layout');
  reconcileModuleView(input.nodes, input.edges, { preserveChildPositions: false });
  expect(layout).toHaveBeenCalledTimes(2);
});

it('preserves custom children during an explicit top-level relayout', () => {
  const input = graph(), layout = jest.spyOn(dagre, 'layout');
  const result = reconcileModuleView(input.nodes, input.edges, { relayout: true, preserveChildPositions: true });
  expect(layout).toHaveBeenCalledTimes(1);
  expect(layout.mock.calls[0][0].nodes()).toEqual(['outer']);
  expect(result.nodes.filter(n => n.data.parentId).map(n => n.position)).toEqual(input.nodes.slice(1).map(n => n.position));
});

it.each(['bert_base.hierarchical.json', 'f5tts_real.json'])('restores expanded real-fixture modules without discarded layouts: %s', filename => {
  const fixture = require(`../../test/fixtures/canvas/${filename}`);
  let logical = hierarchicalToFlat(fixture);
  if (filename === 'f5tts_real.json') logical = autoModularize(logical.nodes, logical.edges);
  // F5 is a grouped/expanded restore variant, not its original flat opening.
  logical = { ...logical, nodes: logical.nodes.map(n => n.type === 'moduleNode' ? { ...n, data: { ...n.data, expanded: true } } : n) };
  const before = JSON.stringify(logical), layout = jest.spyOn(dagre, 'layout');
  const result = reconcileModuleView(logical.nodes, logical.edges, { preserveChildPositions: true });
  expect(layout).not.toHaveBeenCalled();
  expect(result.nodes).toHaveLength(logical.nodes.length);
  expect(JSON.stringify(logical)).toBe(before);
});
