import * as fs from 'fs';
import * as path from 'path';
import { buildInferablePayload } from '../../services/shapeInferencePayload';
import { hierarchicalToFlat } from '../../services/hierarchicalToFlat';
import { reconcileModuleView } from '../../services/canvasLayout';
import { ModelNode, ModelEdge } from '../../types';

/**
 * Guards the canvas shape-inference payload against the "module child +
 * everything downstream shows 'no input connections / unreachable from input'"
 * bug. Module wrappers are dropped before shape inference; their boundary edges
 * (a -> module, module -> b) must be REWIRED onto the module's entry/exit child,
 * otherwise the child and all downstream nodes get orphaned. These tests fail if
 * the rewiring regresses to a plain wrapper-strip + edge-drop.
 */

const node = (id: string, type: string, data: any = {}): ModelNode =>
  ({ id, type, position: { x: 0, y: 0 }, data: { parameters: {}, ...data } } as ModelNode);

const edge = (source: string, target: string): ModelEdge =>
  ({ id: `${source}->${target}`, source, target } as ModelEdge);

const targetsOf = (edges: ModelEdge[]) => new Set(edges.map((e) => e.target));
const sourcesOf = (edges: ModelEdge[]) => new Set(edges.map((e) => e.source));

describe('buildInferablePayload — module boundary rewiring', () => {
  it('retains a wired opaque boundary when no real children can replace it', () => {
    const nodes = [node('in', 'inputNode'), node('blocks', 'moduleNode', {
      layerType: { id: 'modulelist', name: 'ModuleList' },
      parameters: { _modules: [{ op: 'Linear' }] }, moduleMembers: [],
    }), node('out', 'outputNode')];
    const edges = [edge('in', 'blocks'), edge('blocks', 'out')];
    const result = buildInferablePayload(nodes, edges);
    expect(result.inferableNodes).toEqual(nodes);
    expect(result.inferableEdges).toEqual(edges);
  });

  it('rewires a single-child module to its entry/exit child and drops the wrapper', () => {
    const nodes = [
      node('in', 'inputNode'),
      node('a', 'layerNode'),
      node('mod', 'moduleNode', { isGroup: true, moduleEntryNodeId: 'child', moduleExitNodeId: 'child' }),
      node('child', 'layerNode', { parentId: 'mod' }),
      node('b', 'layerNode'),
      node('out', 'outputNode'),
    ];
    const edges = [edge('in', 'a'), edge('a', 'mod'), edge('mod', 'b'), edge('b', 'out')];

    const { inferableNodes, inferableEdges } = buildInferablePayload(nodes, edges);

    // wrapper dropped, child kept
    expect(inferableNodes.find((n) => n.id === 'mod')).toBeUndefined();
    expect(inferableNodes.find((n) => n.id === 'child')).toBeDefined();
    // boundary edges rewired onto the child (this is what the old code failed)
    expect(inferableEdges).toContainEqual(expect.objectContaining({ source: 'a', target: 'child' }));
    expect(inferableEdges).toContainEqual(expect.objectContaining({ source: 'child', target: 'b' }));
    // nothing references the dropped wrapper anymore
    expect(sourcesOf(inferableEdges).has('mod')).toBe(false);
    expect(targetsOf(inferableEdges).has('mod')).toBe(false);
    // every non-input node is reachable (has an incoming edge)
    const incoming = targetsOf(inferableEdges);
    for (const n of inferableNodes) {
      if (n.type === 'inputNode') continue;
      expect(incoming.has(n.id)).toBe(true);
    }
  });

  it('rewires a multi-child module entry/exit topologically when metadata is absent', () => {
    const nodes = [
      node('in', 'inputNode'),
      node('mod', 'moduleNode', { isGroup: true }), // no entry/exit metadata
      // children intentionally NOT in topological order in the array
      node('c2', 'layerNode', { parentId: 'mod' }),
      node('c0', 'layerNode', { parentId: 'mod' }),
      node('c1', 'layerNode', { parentId: 'mod' }),
      node('out', 'outputNode'),
    ];
    const edges = [
      edge('in', 'mod'),
      edge('mod', 'out'),
      // internal chain c0 -> c1 -> c2
      edge('c0', 'c1'),
      edge('c1', 'c2'),
    ];

    const { inferableEdges } = buildInferablePayload(nodes, edges);

    // entry = c0 (no internal incoming), exit = c2 (no internal outgoing)
    expect(inferableEdges).toContainEqual(expect.objectContaining({ source: 'in', target: 'c0' }));
    expect(inferableEdges).toContainEqual(expect.objectContaining({ source: 'c2', target: 'out' }));
    // internal chain preserved
    expect(inferableEdges).toContainEqual(expect.objectContaining({ source: 'c0', target: 'c1' }));
    expect(inferableEdges).toContainEqual(expect.objectContaining({ source: 'c1', target: 'c2' }));
  });

  it('leaves a plain module-free graph untouched', () => {
    const nodes = [node('in', 'inputNode'), node('a', 'layerNode'), node('out', 'outputNode')];
    const edges = [edge('in', 'a'), edge('a', 'out')];
    const { inferableNodes, inferableEdges } = buildInferablePayload(nodes, edges);
    expect(inferableNodes).toHaveLength(3);
    expect(inferableEdges).toHaveLength(2);
  });

  it('keeps a real hierarchical template (bert_base) fully connected through its module', () => {
    const payload = JSON.parse(
      fs.readFileSync(path.join(__dirname, '../../test/fixtures/canvas/bert_base.hierarchical.json'), 'utf8')
    );
    const flat = hierarchicalToFlat(payload);
    const { inferableNodes, inferableEdges } = buildInferablePayload(
      flat.nodes as ModelNode[],
      flat.edges as ModelEdge[]
    );

    // the module wrapper is gone, its 12 children remain
    expect(inferableNodes.find((n) => n.id === 'encoder_stack')).toBeUndefined();
    const children = inferableNodes.filter((n) => n.data?.parentId === 'encoder_stack');
    expect(children.length).toBe(12);

    // no edge still points at the dropped wrapper
    expect(sourcesOf(inferableEdges).has('encoder_stack')).toBe(false);
    expect(targetsOf(inferableEdges).has('encoder_stack')).toBe(false);

    // every non-input node has an incoming edge (nothing orphaned / unreachable)
    const incoming = targetsOf(inferableEdges);
    const inputId = inferableNodes.find((n) => n.type === 'inputNode')?.id;
    for (const n of inferableNodes) {
      if (n.id === inputId) continue;
      expect(incoming.has(n.id)).toBe(true);
    }
  });
});

/**
 * GPT-2 small is a decoder-only template whose `decoder_stack` module ships
 * with input_node_id/output_node_id == null, so the rewiring must fall back to
 * deriving entry/exit topologically. This drives the FULL canvas pipeline
 * (flatten -> reconcileModuleView -> buildInferablePayload) in BOTH collapsed
 * and expanded states — the states a user actually toggles — and asserts the
 * 12 decoder layers + lm_head + output stay reachable. Before the fix every one
 * of those nodes showed "no input connections" / "unreachable from input".
 */
describe('buildInferablePayload — GPT-2 decoder-only (null module entry/exit)', () => {
  const payload = JSON.parse(
    fs.readFileSync(path.join(__dirname, '../../test/fixtures/canvas/gpt2_small.hierarchical.json'), 'utf8')
  );

  const assertFullyConnected = (expanded: boolean) => {
    const flat = hierarchicalToFlat(payload);
    // Toggle the module's expand state the way the canvas does.
    const staged = flat.nodes.map((n) =>
      n.id === 'decoder_stack' ? ({ ...n, data: { ...n.data, expanded } } as ModelNode) : n
    );
    const view = reconcileModuleView(staged as ModelNode[], flat.edges as ModelEdge[], {
      relayout: true,
    });
    const { inferableNodes, inferableEdges } = buildInferablePayload(
      view.nodes as ModelNode[],
      view.edges as ModelEdge[]
    );

    // wrapper dropped, 12 decoder layers survive
    expect(inferableNodes.find((n) => n.id === 'decoder_stack')).toBeUndefined();
    const layers = inferableNodes.filter((n) => n.data?.parentId === 'decoder_stack');
    expect(layers.length).toBe(12);
    // no edge still points at the dropped wrapper
    expect(sourcesOf(inferableEdges).has('decoder_stack')).toBe(false);
    expect(targetsOf(inferableEdges).has('decoder_stack')).toBe(false);
    // entry (layer_0) is fed from the embeddings; exit (layer_11) feeds lm_head
    expect(inferableEdges).toContainEqual(
      expect.objectContaining({ source: 'token_embeddings', target: 'decoder_stack__layer_0' })
    );
    expect(inferableEdges).toContainEqual(
      expect.objectContaining({ source: 'decoder_stack__layer_11', target: 'lm_head' })
    );
    // every non-input node is reachable (has an incoming edge)
    const incoming = targetsOf(inferableEdges);
    const inputId = inferableNodes.find((n) => n.type === 'inputNode')?.id;
    for (const n of inferableNodes) {
      if (n.id === inputId) continue;
      expect(incoming.has(n.id)).toBe(true);
    }
  };

  it('stays fully connected when the decoder stack is collapsed', () => {
    assertFullyConnected(false);
  });

  it('stays fully connected when the decoder stack is expanded', () => {
    assertFullyConnected(true);
  });
});
