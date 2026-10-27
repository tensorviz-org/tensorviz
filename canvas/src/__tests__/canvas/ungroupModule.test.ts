/**
 * Grouping layers into a module used to be irreversible.
 *
 * Pressing G collapsed a selection into a module wrapper and nothing anywhere
 * undid it — the user guide said so ("there's no ungroup button yet") — and since
 * the model-history stack lives in a ref and dies on reload, a group made
 * yesterday was permanent.
 *
 * The hard part is not removing the wrapper, it is the boundary edges. Grouping
 * REWRITES them (`input -> conv` becomes `input -> module-17`) and keeps no record
 * of the child, so a naive ungroup leaves edges pointing at a node that no longer
 * exists. These tests exist to make sure a half-ungroup that quietly drops or
 * dangles an edge can never ship: every case below asserts on the resulting
 * topology, not on the wrapper being gone.
 */
import { ModelEdge, ModelNode } from '../../types';
import { ungroupModuleGraph } from '../../services/ungroupModule';
import { buildInferablePayload } from '../../services/shapeInferencePayload';

const layer = (id: string, parentId?: string): ModelNode => ({
  id,
  type: 'layerNode',
  position: { x: 0, y: 0 },
  data: {
    layerType: { id: 'linear', name: 'Linear' },
    parameters: {},
    ...(parentId ? { parentId } : {}),
  },
} as ModelNode);

const wrapper = (id: string, instanceId?: string, extra: Record<string, any> = {}): ModelNode => ({
  id,
  type: 'moduleNode',
  position: { x: 400, y: 200 },
  data: {
    layerType: { id: 'module', name: 'Module' },
    parameters: { name: 'CustomModule' },
    isGroup: true,
    moduleInstanceId: instanceId || id,
    ...extra,
  },
} as ModelNode);

const edge = (id: string, source: string, target: string, data?: any): ModelEdge =>
  ({ id, source, target, ...(data ? { data } : {}) } as ModelEdge);

const wiring = (edges: ModelEdge[]) =>
  edges.map((e) => `${e.source}->${e.target}`).sort();

/** input -> [conv -> relu] -> output, with the pair grouped the way G groups it. */
const grouped = () => ({
  nodes: [
    wrapper('module-1'),
    layer('input'),
    layer('conv', 'module-1'),
    layer('relu', 'module-1'),
    layer('output'),
  ],
  edges: [
    edge('e1', 'input', 'module-1'),
    edge('e2', 'conv', 'relu'),
    edge('e3', 'module-1', 'output'),
  ],
});

describe('ungrouping a module', () => {
  it('restores the exact chain the module replaced', () => {
    const { nodes, edges } = grouped();
    const result = ungroupModuleGraph(nodes, edges, 'module-1')!;
    expect(result).not.toBeNull();
    expect(wiring(result.edges)).toEqual(['conv->relu', 'input->conv', 'relu->output']);
    expect(result.restored).toBe(2);
  });

  it('leaves no edge pointing at a node that is gone', () => {
    // The failure mode this whole feature has to avoid: the wrapper is deleted
    // and its boundary edges still name it, so the graph is silently severed.
    const { nodes, edges } = grouped();
    const result = ungroupModuleGraph(nodes, edges, 'module-1')!;
    const ids = new Set(result.nodes.map((n) => n.id));
    for (const e of result.edges) {
      expect(ids.has(e.source)).toBe(true);
      expect(ids.has(e.target)).toBe(true);
    }
    expect(result.nodes.some((n) => n.id === 'module-1')).toBe(false);
  });

  it('keeps every edge — the count cannot drop', () => {
    const { nodes, edges } = grouped();
    const result = ungroupModuleGraph(nodes, edges, 'module-1')!;
    expect(result.edges).toHaveLength(3);
  });

  it('hands the children back to the parent canvas', () => {
    const { nodes, edges } = grouped();
    const result = ungroupModuleGraph(nodes, edges, 'module-1')!;
    const conv = result.nodes.find((n) => n.id === 'conv')!;
    expect(conv.data.parentId).toBeUndefined();
    expect(conv.parentNode).toBeUndefined();
    expect(conv.extent).toBeUndefined();
    expect(conv.hidden).toBe(false);
  });

  it('makes child positions absolute again', () => {
    // Children are positioned relative to their wrapper. Left relative, every
    // restored layer would jump to the canvas origin.
    const nodes = [
      wrapper('module-1'),
      { ...layer('conv', 'module-1'), position: { x: 20, y: 56 } } as ModelNode,
      layer('relu', 'module-1'),
    ];
    const result = ungroupModuleGraph(nodes, [edge('e2', 'conv', 'relu')], 'module-1')!;
    expect(result.nodes.find((n) => n.id === 'conv')!.position).toEqual({ x: 420, y: 256 });
  });

  it('produces exactly the flat graph the backend was already validating', () => {
    // This is the correctness argument for the whole feature. The wrapper is not
    // a runnable leaf, so every backend payload has always dropped it and rewired
    // its boundary edges onto the same entry/exit children. Ungroup reuses that
    // rule, so it cannot introduce a topology the graph did not already have.
    const { nodes, edges } = grouped();
    const payload = buildInferablePayload(nodes, edges);
    const result = ungroupModuleGraph(nodes, edges, 'module-1')!;
    expect(wiring(result.edges)).toEqual(wiring(payload.inferableEdges));
  });

  it('honours the entry/exit ids the wrapper carries, over the derived ones', () => {
    // Backend- and template-authored modules name their own boundary children.
    const nodes = [
      wrapper('module-1', 'module-1', { moduleEntryNodeId: 'relu', moduleExitNodeId: 'conv' }),
      layer('input'),
      layer('conv', 'module-1'),
      layer('relu', 'module-1'),
      layer('output'),
    ];
    const edges = [edge('e1', 'input', 'module-1'), edge('e3', 'module-1', 'output')];
    const result = ungroupModuleGraph(nodes, edges, 'module-1')!;
    expect(wiring(result.edges)).toEqual(['conv->output', 'input->relu']);
  });

  it('is addressable by module instance id, which is all the canvas card knows', () => {
    const nodes = [wrapper('module-node-9', 'auto-mod-3'), layer('conv', 'module-node-9')];
    const result = ungroupModuleGraph(nodes, [], 'auto-mod-3');
    expect(result?.restored).toBe(1);
  });

  it('drops synthetic display edges instead of restoring them as real ones', () => {
    // The reconciler invents these to connect a COLLAPSED box to its neighbours.
    // Keeping one would double the input edge and add a link the model never had.
    const nodes = [wrapper('module-1'), layer('input'), layer('conv', 'module-1')];
    const edges = [
      edge('e1', 'input', 'conv'),
      edge('synth__input__module-1', 'input', 'module-1', { synthetic: true }),
    ];
    const result = ungroupModuleGraph(nodes, edges, 'module-1')!;
    expect(wiring(result.edges)).toEqual(['input->conv']);
  });

  it('restores the hidden instances of a repeated (xN) block', () => {
    // A repeated module renders one canonical child and hides the rest. Once the
    // wrapper is gone they are ordinary layers and must be visible.
    const nodes = [
      wrapper('module-1'),
      layer('block0', 'module-1'),
      { ...layer('block1', 'module-1'), data: { ...layer('block1', 'module-1').data, repeatHidden: true } } as ModelNode,
    ];
    const result = ungroupModuleGraph(nodes, [], 'module-1')!;
    expect(result.restored).toBe(2);
    expect(result.nodes.every((n) => !n.hidden)).toBe(true);
    expect(result.nodes.find((n) => n.id === 'block1')!.data.repeatHidden).toBeUndefined();
  });

  it('leaves other modules on the canvas completely alone', () => {
    const nodes = [
      wrapper('module-1'),
      wrapper('module-2'),
      layer('conv', 'module-1'),
      layer('norm', 'module-2'),
    ];
    const edges = [edge('e1', 'module-1', 'module-2')];
    const result = ungroupModuleGraph(nodes, edges, 'module-1')!;
    expect(result.nodes.some((n) => n.id === 'module-2')).toBe(true);
    expect(result.nodes.find((n) => n.id === 'norm')!.data.parentId).toBe('module-2');
    expect(wiring(result.edges)).toEqual(['conv->module-2']);
  });

  it('refuses rather than deleting a wrapper it cannot restore', () => {
    // Nothing to hand back means the boundary edges have nowhere to land, so
    // removing the wrapper would lose them. Say no; the caller shows a toast.
    expect(ungroupModuleGraph([wrapper('module-1')], [edge('e', 'a', 'module-1')], 'module-1')).toBeNull();
    expect(ungroupModuleGraph([layer('conv')], [], 'conv')).toBeNull();
    expect(ungroupModuleGraph([], [], 'module-1')).toBeNull();
  });
});
