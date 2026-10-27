/**
 * `nn.ModuleList` has to arrive as a block you can open, not a single flat card.
 *
 * The user's report: "it doesn't change into an expandable/collapsable block and instead
 * shows one single unit". The tracer folds four identical `nn.Linear` siblings into one
 * node marked `_repeat_count: 4`, and that node reached the canvas as a bare `Linear`
 * card — nothing naming the container, nothing to expand.
 */
import { ModelGraph, ModelNode } from '../../types';
import { layoutTopLevel, reconcileModuleView } from '../../services/canvasLayout';
import { repeatFrameId, wrapRepeatBlocks } from '../../ide/repeatBlocks';

const layer = (id: string, parameters: Record<string, unknown>): ModelNode => ({
  id,
  type: 'layerNode',
  position: { x: 0, y: 0 },
  data: {
    layerType: {
      id: 'linear', name: 'Linear', displayName: 'Linear',
      description: '', icon: 'Box', category: 'basic', parameters: [],
    },
    parameters,
  },
}) as ModelNode;

const graph = (nodes: ModelNode[], edges: any[] = []): ModelGraph =>
  ({ nodes, edges } as unknown as ModelGraph);

const folded = () => layer('blocks.0', {
  in_features: 8, out_features: 8,
  _module_path: 'blocks.0', _repeat_count: 4, _repeat_reason: 'module-list',
});

describe('wrapRepeatBlocks', () => {
  it('gives a folded ModuleList a module frame that can be expanded', () => {
    const out = wrapRepeatBlocks(graph([folded()]));
    const frame = out.nodes.find((n) => n.id === repeatFrameId('blocks.0'))!;
    expect(frame).toBeTruthy();
    expect(frame.type).toBe('moduleNode');
    expect(frame.data.expandable).toBe(true);
    expect(frame.data.expanded).toBe(false);
    expect(frame.data.repeatCount).toBe(4);
  });

  it('names the frame after the attribute the user wrote, not the layer class', () => {
    const out = wrapRepeatBlocks(graph([folded()]));
    const frame = out.nodes.find((n) => n.id === repeatFrameId('blocks.0'))!;
    // `self.blocks = nn.ModuleList(...)` — the block is "blocks". Calling it "Linear"
    // would name the thing inside it and lose the only word the user would search for.
    expect(frame.data.layerType.displayName).toBe('blocks');
    expect(frame.data.repeatBlockLabel).toBe('Linear');
  });

  it('parents the folded card into its frame so the canvas can hide it when collapsed', () => {
    const out = wrapRepeatBlocks(graph([folded()]));
    const member = out.nodes.find((n) => n.id === 'blocks.0')!;
    expect(member.data.parentId).toBe(repeatFrameId('blocks.0'));
  });

  it('holds ONE canonical block, because nothing measured the four running in series', () => {
    const out = wrapRepeatBlocks(graph([folded()]));
    const frame = out.nodes.find((n) => n.id === repeatFrameId('blocks.0'))!;
    expect(frame.data.moduleLayerCount).toBe(1);
    expect(out.nodes.filter((n) => n.type !== 'moduleNode')).toHaveLength(1);
    // The ×4 is the whole of what the graph knows; four chained cards would be three
    // edges nobody measured.
    expect(frame.data.moduleEntryNodeId).toBe('blocks.0');
    expect(frame.data.moduleExitNodeId).toBe('blocks.0');
  });

  it('leaves the edges naming the member, for the reconciler to reroute', () => {
    const edges = [{ id: 'e1', source: 'l1', target: 'blocks.0' },
                   { id: 'e2', source: 'blocks.0', target: 'l2' }];
    const out = wrapRepeatBlocks(graph([layer('l1', {}), folded(), layer('l2', {})], edges));
    expect(out.edges).toEqual(edges);
  });

  it('is a no-op on a file with nothing folded', () => {
    const input = graph([layer('l1', { in_features: 8 })]);
    expect(wrapRepeatBlocks(input)).toBe(input);
  });

  it('ignores a count of one — a card standing for itself is not a block', () => {
    const input = graph([layer('l1', { _repeat_count: 1 })]);
    expect(wrapRepeatBlocks(input)).toBe(input);
  });

  it('does not nest a second frame around a card that already has a parent', () => {
    const child = folded();
    child.data.parentId = 'someone-elses-block';
    const out = wrapRepeatBlocks(graph([child]));
    expect(out.nodes.filter((n) => n.type === 'moduleNode')).toHaveLength(0);
    expect(out.nodes[0].data.parentId).toBe('someone-elses-block');
  });

  it('keeps the frame id stable, so a keystroke re-projection lands on the same frame', () => {
    const a = wrapRepeatBlocks(graph([folded()])).nodes.map((n) => n.id);
    const b = wrapRepeatBlocks(graph([folded()])).nodes.map((n) => n.id);
    expect(a).toEqual(b);
  });
});

/**
 * The frame has to claim the space it draws.
 *
 * Measured on the QA fixture through the installed host on 2026-09-13: the
 * `repeat:stack.0` frame's card drew 250x235 while its shape box was 250x140, and the
 * `x.reshape(...)` card sat at +188 — inside it by 47, over the Expand Block CTA.
 *
 * The collapsed card now draws only its 56px header, 2px border, 22px body padding,
 * the 27px pill row (layer count, repeat, status) and the 14px entry→exit route with
 * its 8px gap; the mini-map, sublayer chips and footer CTA are gone. Its footprint is
 * derived from those sections (cardMetrics.moduleCardHeight), and rendered cards are
 * checked against their boxes in QA.
 */
const DRAWN_COLLAPSED_FRAME_HEIGHT = 129;

describe('a repeat frame claims the space it draws', () => {
  // The QA fixture's shape: the folded block and the `x.reshape(...)` card share a
  // rank, so dagre stacks them vertically one `nodesep` apart — which is where the
  // frame's own drawn height decides whether they touch.
  const laidOut = () => {
    const edges = [{ id: 'e1', source: 'l1', target: 'blocks.0' },
                   { id: 'e2', source: 'blocks.0', target: 'l3' },
                   { id: 'e3', source: 'l1', target: 'sib' },
                   { id: 'e4', source: 'sib', target: 'l3' }];
    const wrapped = wrapRepeatBlocks(
      graph([layer('l1', {}), folded(), layer('sib', {}), layer('l3', {})], edges));
    const placed = layoutTopLevel(wrapped.nodes, wrapped.edges as any);
    const positioned = wrapped.nodes.map((node) =>
      placed.has(node.id) ? { ...node, position: placed.get(node.id)! } : node);
    return reconcileModuleView(positioned, wrapped.edges as any, { relayout: false });
  };

  it('reserves the frame’s drawn footprint, not the bare card’s', () => {
    const frame = laidOut().nodes.find((n) => n.id === repeatFrameId('blocks.0'))! as any;
    expect(frame.style.height).toBeGreaterThanOrEqual(DRAWN_COLLAPSED_FRAME_HEIGHT);
  });

  it('leaves no neighbour standing inside the frame', () => {
    const { nodes } = laidOut();
    const frame = nodes.find((n) => n.id === repeatFrameId('blocks.0'))!;
    const box = {
      left: frame.position.x,
      right: frame.position.x + (frame.style!.width as number),
      top: frame.position.y,
      // What the card DRAWS, which is what the user sees the neighbour land on.
      bottom: frame.position.y + DRAWN_COLLAPSED_FRAME_HEIGHT,
    };
    const neighbours = nodes.filter((n) => !n.data.parentId && n.id !== frame.id);
    expect(neighbours.length).toBeGreaterThan(0);
    for (const n of neighbours) {
      const overlaps =
        n.position.x < box.right && n.position.x + 200 > box.left &&
        n.position.y < box.bottom && n.position.y + 120 > box.top;
      expect({ id: n.id, overlaps }).toEqual({ id: n.id, overlaps: false });
    }
  });
});
