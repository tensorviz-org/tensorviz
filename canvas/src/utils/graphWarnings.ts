/**
 * The warnings the diagnostics drawer shows for a graph's shape, before any
 * backend call.
 *
 * Lives here rather than inside BuilderPageV2 so it can be tested directly:
 * BuilderPageV2 pulls in the canvas and is not mountable or importable under the
 * CRA jest transform, which is why this had only ever been pinned as an
 * 800-character window of its own source text.
 *
 * ── the bug this exists to close ─────────────────────────────────────────────
 * A missing Input was never reported, because the per-node loop skips the
 * terminals. Deleting the Input left only "Linear - No incoming edge", which names
 * the symptom and not the cause; and a lone layer on a blank canvas failed Validate
 * with a shape error the user could not act on. Validate pushes a tensor of the
 * Input node's shape through the graph, so without one there is nothing to push and
 * no way to know what shape was assumed — which is why the warning says so.
 */

/** The subset of a canvas node this rule reads. */
export interface WarnableNode {
  id: string;
  data?: { layerType?: { id?: string; displayName?: string } };
}

/** The subset of a canvas edge this rule reads. */
export interface WarnableEdge {
  source: string;
  target: string;
}

export interface GraphWarning {
  nodeId: string;
  name: string;
  message: string;
}

/** The drawer renders at most this many rows. */
const MAX_WARNINGS = 24;

export const NO_INPUT_WARNING =
  'No Input node. Validate pushes a tensor of the Input node\'s shape through the '
  + 'graph, so shape results here are based on a fallback shape rather than yours. '
  + 'Add an Input layer and connect it to your first layer.';

export function graphWarnings(
  nodes: WarnableNode[],
  edges: WarnableEdge[],
): GraphWarning[] {
  const inCount = new Map<string, number>();
  const outCount = new Map<string, number>();
  for (const edge of edges) {
    inCount.set(edge.target, (inCount.get(edge.target) || 0) + 1);
    outCount.set(edge.source, (outCount.get(edge.source) || 0) + 1);
  }
  const warnings: GraphWarning[] = [];
  const terminals = nodes.map((node) => String(node?.data?.layerType?.id || '').toLowerCase());
  // An empty canvas is a new model, not a broken one.
  if (nodes.length > 0 && !terminals.includes('input')) {
    warnings.push({
      nodeId: nodes[0]?.id || '',
      name: 'Model',
      message: NO_INPUT_WARNING,
    });
  }
  for (const node of nodes) {
    const layerId = String(node?.data?.layerType?.id || '').toLowerCase();
    const incoming = inCount.get(node.id) || 0;
    const outgoing = outCount.get(node.id) || 0;
    if (layerId === 'input' || layerId === 'output') continue;
    if (incoming === 0 && outgoing === 0) {
      warnings.push({
        nodeId: node.id,
        name: node.data?.layerType?.displayName || node.id,
        message: 'Isolated node not connected to graph path.',
      });
    } else if (incoming === 0) {
      warnings.push({
        nodeId: node.id,
        name: node.data?.layerType?.displayName || node.id,
        message: 'No incoming edge from upstream layer.',
      });
    } else if (outgoing === 0) {
      warnings.push({
        nodeId: node.id,
        name: node.data?.layerType?.displayName || node.id,
        message: 'No outgoing edge to downstream layer.',
      });
    }
  }
  return warnings.slice(0, MAX_WARNINGS);
}
