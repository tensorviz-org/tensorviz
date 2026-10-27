/**
 * A folded repeat becomes an expandable block.
 *
 * The tracer does not emit one node per `nn.ModuleList` member. `introspect.build_tree`
 * groups structurally identical siblings and recurses into the FIRST one only, so
 * `self.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])` arrives as a single
 * node carrying `_repeat_count: 4` — deliberately, because a Stable Diffusion UNet drawn
 * one card per sibling is unreadable.
 *
 * What reached the canvas, though, was a bare `Linear` card: one flat unit, with no way
 * to tell it apart from a lone layer and nothing to open. This wraps it in the module
 * frame the rest of the product already uses for repeats, so it collapses and expands
 * like any other block.
 *
 * The frame holds ONE canonical member, not N. That is the same rule `autoModularize`
 * applies to a repeated stack it infers itself (`repeatHidden`, whose docstring says it
 * outright: "an expanded ×N module shows ONE representative block instead of all N
 * copies"), and here it is also the only honest reading. The tracer measured all four
 * Linears — `io` and `call_counts` carry every one — but it folds their dataflow onto the
 * canonical node, so nothing in the graph says the four run in SERIES. Drawing four cards
 * in a chain would mean inventing three edges nobody measured, which is the same mistake
 * as the `modules=[...]` leak this block was reported for.
 *
 * Nothing here touches the semantic graph: the wrapper is a display frame, and a display
 * frame is grouping, not computation. The member keeps its id,
 * its params and its edges, so codegen, verify and the validator see exactly what they
 * saw before.
 */
import { ModelEdge, ModelGraph, ModelNode } from '../types';

/** The projection's marker for "this card stands for N identical blocks". */
export const REPEAT_COUNT_KEY = '_repeat_count';
const MODULE_PATH_KEY = '_module_path';

/** Deterministic, so a re-projection on a keystroke merges onto the same frame. */
export const repeatFrameId = (nodeId: string): string => `repeat:${nodeId}`;

/** Is this id one of the frames above, rather than a node the projection sent?
 *
 * A repeat frame is drawn, not projected: nothing in the document has this id, and the
 * canvas invented it to hold a folded block. Anything comparing the canvas against the
 * projection has to leave it out, or the two will never agree — `structureDrifted` did
 * not, so every model with a folded block was permanently "edited" and every save said
 * so. */
export const isRepeatFrameId = (id: string): boolean => id.startsWith('repeat:');

function repeatCountOf(node: ModelNode): number {
  const raw = node.data?.parameters?.[REPEAT_COUNT_KEY];
  const count = typeof raw === 'string' ? Number(raw) : raw;
  return typeof count === 'number' && Number.isFinite(count) && count > 1 ? count : 0;
}

/**
 * What to call the block. `_module_path` is the attribute the member lives under
 * (`blocks.0`), so its parent segment is the name the user actually wrote — `blocks`,
 * not `Linear`. Falls back to the layer's own name when the path says nothing useful.
 */
function frameLabel(node: ModelNode): string {
  const path = node.data?.parameters?.[MODULE_PATH_KEY];
  if (typeof path === 'string' && path.includes('.')) {
    const container = path.slice(0, path.lastIndexOf('.'));
    if (container) return container;
  }
  return node.data?.layerType?.displayName || node.data?.layerType?.name || node.id;
}

function frameFor(node: ModelNode, count: number): ModelNode {
  const label = frameLabel(node);
  const blockName = node.data?.layerType?.displayName || node.data?.layerType?.name || 'block';
  return {
    id: repeatFrameId(node.id),
    type: 'moduleNode',
    position: { ...node.position },
    data: {
      layerType: {
        id: 'module',
        name: 'Module',
        displayName: label,
        description: `${blockName} repeated ${count}×`,
        icon: 'Package',
        category: 'module',
        parameters: [],
      },
      parameters: { name: label },
      isGroup: true,
      expandable: true,
      expanded: false,
      moduleInstanceId: repeatFrameId(node.id),
      moduleTemplateId: label,
      moduleEntryNodeId: node.id,
      moduleExitNodeId: node.id,
      // ONE canonical block, exactly as autoModularize counts a repeated stack:
      // the ×N badge is what conveys the repetition.
      moduleLayerCount: 1,
      moduleStatus: 'ok',
      moduleLayoutState: 'deterministic',
      moduleMembers: [node.id],
      repeatCount: count,
      repeatBlockLabel: blockName,
    },
  } as ModelNode;
}

/**
 * Wrap every folded repeat in the graph in its own module frame.
 *
 * Returns the graph unchanged when there is nothing folded, so an ordinary file pays
 * nothing for this. A node that already has a parent is left alone: it belongs to a
 * block someone else built, and nesting a second frame around it would fight them.
 */
export function wrapRepeatBlocks(graph: ModelGraph): ModelGraph {
  const frames: ModelNode[] = [];
  const nodes: ModelNode[] = graph.nodes.map((node) => {
    if (node.data?.parentId || node.type === 'moduleNode') return node;
    const count = repeatCountOf(node);
    if (!count) return node;
    frames.push(frameFor(node, count));
    return { ...node, data: { ...node.data, parentId: repeatFrameId(node.id) } } as ModelNode;
  });
  if (frames.length === 0) return graph;
  // Edges keep naming the member. `reconcileModuleView` hides a collapsed module's
  // child-targeted edges and synthesizes the frame-level ones, the same way it does
  // for every other block; rerouting them here would give it two of each.
  return { ...graph, nodes: [...frames, ...nodes] as ModelNode[], edges: graph.edges as ModelEdge[] };
}
