import { mergeProjection } from '../../ide/applyProjection';
import { ModelGraph, ModelNode } from '../../types';

function node(
  id: string,
  overrides: Partial<ModelNode> = {},
): ModelNode {
  return {
    id,
    type: 'layerNode',
    position: { x: 10, y: 20 },
    data: { label: id },
    ...overrides,
  } as ModelNode;
}

function moduleNode(id: string, expanded: boolean, position = { x: 10, y: 20 }): ModelNode {
  return {
    id,
    type: 'moduleNode',
    position,
    data: { label: id, expanded },
  } as ModelNode;
}

function graph(nodes: ModelNode[]): ModelGraph {
  return { nodes, edges: [] } as unknown as ModelGraph;
}

describe('mergeProjection — frame openness', () => {
  it('keeps a frame the user expanded across a re-projection', () => {
    // The defect this guards: the source says nothing about which frames are
    // open, so a keystroke re-projection carrying `expanded: false` would close
    // every frame the user had opened. Harmless for one class; for a whole
    // project it closes the entire canvas on every keypress.
    const existing = [moduleNode('Net', true)];
    const incoming = graph([moduleNode('Net', false)]);

    const merged = mergeProjection(incoming, existing);

    expect((merged.nodes[0].data as { expanded: boolean }).expanded).toBe(true);
  });

  it('keeps a frame the user collapsed, too', () => {
    const merged = mergeProjection(graph([moduleNode('Net', true)]), [moduleNode('Net', false)]);
    expect((merged.nodes[0].data as { expanded: boolean }).expanded).toBe(false);
  });

  it('lets a structural push win when it is authoritative', () => {
    // A trace that brought new children has to open the frame holding them, or
    // the deeper layers stay hidden until the user collapses and reopens by hand.
    const merged = mergeProjection(graph([moduleNode('Net', true)]), [moduleNode('Net', false)], {
      trustOpenness: true,
    });
    expect((merged.nodes[0].data as { expanded: boolean }).expanded).toBe(true);
  });

  it('does not invent openness for a node that never had it', () => {
    const merged = mergeProjection(graph([node('fc1')]), [node('fc1')]);
    expect((merged.nodes[0].data as { expanded?: boolean }).expanded).toBeUndefined();
  });

  it('only applies to module frames, not plain cards', () => {
    const previous = node('fc1', { data: { label: 'fc1', expanded: true } } as Partial<ModelNode>);
    const merged = mergeProjection(graph([node('fc1')]), [previous]);
    expect((merged.nodes[0].data as { expanded?: boolean }).expanded).toBeUndefined();
  });
});

describe('mergeProjection — what it already guaranteed', () => {
  it('keeps a position the user dragged and replaces the data', () => {
    const existing = [node('fc1', { position: { x: 400, y: 500 } })];
    const incoming = graph([node('fc1', { data: { label: 'renamed' } } as Partial<ModelNode>)]);

    const merged = mergeProjection(incoming, existing);

    expect(merged.nodes[0].position).toEqual({ x: 400, y: 500 });
    expect((merged.nodes[0].data as { label: string }).label).toBe('renamed');
  });

  it('reports added and removed ids so the caller can patch', () => {
    const merged = mergeProjection(graph([node('a'), node('c')]), [node('a'), node('b')]);
    expect(merged.added).toEqual(['c']);
    expect(merged.removed).toEqual(['b']);
  });

  it('keeps a selection through a re-projection', () => {
    const merged = mergeProjection(graph([node('a')]), [node('a', { selected: true })]);
    expect(merged.nodes[0].selected).toBe(true);
  });
});
