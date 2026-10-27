/**
 * The merge that makes "the graph follows the code" liveable (task.md M3).
 *
 * The IDE re-projects the file on every settled keystroke. Two behaviours decide
 * whether that is useful or unusable, and both are asserted here:
 *
 *   * a card the user placed keeps its position across a re-projection — the
 *     code-import path (`handleImport`) clears the canvas and relayouts, which is
 *     right once and destroys the user's arrangement on a loop;
 *   * a node the edit created is placed, and it is the only one that moves.
 *
 * These are unit-level on purpose: the equivalent through the canvas needs a
 * live tldraw editor, which the browser specs own.
 */
import { mergeProjection } from '../../ide/applyProjection';
import { ModelEdge, ModelGraph, ModelNode } from '../../types';

const layer = (id: string, x = 0, y = 0): ModelNode =>
  ({
    id,
    type: 'layerNode',
    position: { x, y },
    data: { layerType: { id: 'linear', name: 'Linear', parameters: [] }, parameters: {} },
  } as unknown as ModelNode);

const edge = (source: string, target: string): ModelEdge =>
  ({ id: `edge_${source}_${target}`, source, target } as ModelEdge);

const graph = (nodes: ModelNode[], edges: ModelEdge[] = []): ModelGraph =>
  ({ nodes, edges } as ModelGraph);

describe('mergeProjection', () => {
  it('keeps the position of every node the canvas already has', () => {
    const existing = [layer('l1', 640, 220), layer('l2', 980, 220)];
    const incoming = graph([layer('l1'), layer('l2')], [edge('l1', 'l2')]);

    const merged = mergeProjection(incoming, existing);

    expect(merged.nodes.map((node) => node.position)).toEqual([
      { x: 640, y: 220 },
      { x: 980, y: 220 },
    ]);
    expect(merged.added).toEqual([]);
    expect(merged.removed).toEqual([]);
  });

  it('takes structure from the projection while keeping layout from the canvas', () => {
    const existing = [layer('l1', 640, 220)];
    const projected = {
      ...layer('l1'),
      data: { layerType: { id: 'conv2d', name: 'Conv2d', parameters: [] }, parameters: { k: 3 } },
    } as unknown as ModelNode;

    const merged = mergeProjection(graph([projected]), existing);

    expect(merged.nodes[0].data.layerType.id).toBe('conv2d');
    expect(merged.nodes[0].data.parameters).toEqual({ k: 3 });
    expect(merged.nodes[0].position).toEqual({ x: 640, y: 220 });
  });

  it('lays out a new node and moves nothing else', () => {
    const existing = [layer('l1', 640, 220), layer('l2', 980, 220)];
    const incoming = graph(
      [layer('l1'), layer('l2'), layer('extra')],
      [edge('l1', 'l2'), edge('l2', 'extra')],
    );

    const merged = mergeProjection(incoming, existing);

    expect(merged.added).toEqual(['extra']);
    expect(merged.nodes.find((node) => node.id === 'l1')!.position).toEqual({ x: 640, y: 220 });
    expect(merged.nodes.find((node) => node.id === 'l2')!.position).toEqual({ x: 980, y: 220 });
    const placed = merged.nodes.find((node) => node.id === 'extra')!.position;
    expect(placed).not.toEqual({ x: 0, y: 0 });
  });

  it('drops a node the file no longer declares', () => {
    const existing = [layer('l1', 100, 100), layer('gone', 300, 100)];

    const merged = mergeProjection(graph([layer('l1')]), existing);

    expect(merged.nodes.map((node) => node.id)).toEqual(['l1']);
    expect(merged.removed).toEqual(['gone']);
  });

  it('never wipes the canvas the way an import does', () => {
    // The regression this exists for: a projection is not an import. Applying one
    // must leave a graph on the canvas, not an empty one followed by a relayout.
    const existing = [layer('l1', 640, 220), layer('l2', 980, 220)];
    const merged = mergeProjection(graph([layer('l1'), layer('l2')], [edge('l1', 'l2')]), existing);

    expect(merged.nodes).toHaveLength(2);
    expect(merged.edges).toHaveLength(1);
  });

  it('honours a position the projection brought for a node the canvas lacks', () => {
    // The `.tensorviz/views` sidecar arrives inside the projection document, so a
    // node reopened in a new window must land where the user left it rather than
    // being relaid out. `viewRestored` is how the host says the positions came
    // from that sidecar; this used to be inferred from the positions themselves,
    // which is what made a never-arranged graph open in tracer-emission order.
    const merged = mergeProjection(graph([layer('l1', 1240, 480)]), [], { viewRestored: true });

    expect(merged.added).toEqual(['l1']);
    expect(merged.nodes[0].position).toEqual({ x: 1240, y: 480 });
  });

  it('lays out a graph nobody has arranged when it is opened', () => {
    // The UX this exists for. A projection always carries positions, so on an
    // empty canvas nothing counted as unplaced and dagre never ran: the graph
    // opened in whatever order the tracer emitted nodes, cards overlapping and
    // edges crossing, and every user reached for the layout button by hand.
    const merged = mergeProjection(
      graph(
        [layer('l1', 40, 40), layer('l2', 60, 44), layer('l3', 52, 41)],
        [edge('l1', 'l2'), edge('l2', 'l3')],
      ),
      [],
      { viewRestored: false },
    );

    const xs = merged.nodes.map((node) => node.position.x);
    // Laid out along the edges rather than left in a heap: three distinct columns,
    // in flow order, which the emitted positions above are not.
    expect(new Set(xs).size).toBe(3);
    expect(xs[0]).toBeLessThan(xs[1]);
    expect(xs[1]).toBeLessThan(xs[2]);
  });

  it('does not lay out again once the canvas has the graph', () => {
    // The guard that keeps this from being the destructive relayout the file
    // header warns about: re-projection happens on every settled keystroke, and
    // a whole-graph layout there would move every card while the user types.
    const existing = [layer('l1', 640, 220), layer('l2', 980, 220)];
    const merged = mergeProjection(
      graph([layer('l1', 0, 0), layer('l2', 0, 0)], [edge('l1', 'l2')]),
      existing,
      { viewRestored: false },
    );

    expect(merged.nodes.map((node) => node.position)).toEqual([
      { x: 640, y: 220 },
      { x: 980, y: 220 },
    ]);
  });

  it('keeps the selection a re-projection would otherwise clear', () => {
    const selected = { ...layer('l1', 640, 220), selected: true } as ModelNode;

    const merged = mergeProjection(graph([layer('l1')]), [selected]);

    expect(merged.nodes[0].selected).toBe(true);
  });
});
