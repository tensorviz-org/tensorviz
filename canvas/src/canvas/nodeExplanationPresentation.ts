import type { ModelNode } from '../types';
import type { NodeExplanation, NodeExplanationState } from '../ide/nodeExplanation.generated';
import { layerCardHeight, moduleCardHeight, moduleExpandedHeaderHeight, nodeCardSize } from './cardMetrics';
import { wrapCardText } from './cardIdentity';

/** Keep every role readable; the model prompt asks for concise role names. */
function roleSummary(values: readonly string[]): string | undefined {
  return values.length ? values.join(' · ') : undefined;
}

export function nodeAnnotationLayout(explanation: NodeExplanation, width: number) {
  const titleLines = wrapCardText(explanation.title, Math.max(80, width - 48), 12);
  const purposeLines = wrapCardText(explanation.purpose, Math.max(60, width - 24), 12);
  const input = roleSummary(explanation.inputs), output = roleSummary(explanation.outputs);
  const inputLines = input ? wrapCardText(input, Math.max(60, width - 64), 11) : [];
  const outputLines = output ? wrapCardText(output, Math.max(60, width - 64), 11) : [];
  return { titleLines, purposeLines, input, output, inputLines, outputLines,
    height: 21 + titleLines.length * 16 + (purposeLines.length ? 6 + purposeLines.length * 17 : 0)
      + (inputLines.length ? 6 + inputLines.length * 15 : 0)
      + (outputLines.length ? 4 + outputLines.length * 15 : 0) };
}

export function nodeAnnotation(node: ModelNode, states: Readonly<Record<string, NodeExplanationState>>, visible: boolean) {
  const state = states[node.id];
  return visible && !node.hidden && !node.data.review && !node.data.sourceAssignment && !node.data.sourceReplacement
    && state?.status === 'ready' ? state.explanation : undefined;
}

export interface ExplainedGraphPresentation {
  nodes: ModelNode[];
  sizes: ReadonlyMap<string, { w: number; h: number }>;
  /** Remove these offsets before a user drag reaches authoritative model state. */
  offsets: ReadonlyMap<string, { x: number; y: number }>;
}

/** Make space only in the displayed scene. Never change model data, persisted
 * layout, graph topology, selection or camera. Keep sibling order and grow
 * expanded frames around their actual displayed children.
 */
export function explainedGraphPresentation(source: ModelNode[], states: Readonly<Record<string, NodeExplanationState>>, visible: boolean): ExplainedGraphPresentation {
  const sizes = new Map<string, { w: number; h: number }>();
  const offsets = new Map<string, { x: number; y: number }>();
  if (!visible || !source.some(node => nodeAnnotation(node, states, visible))) return { nodes: source, sizes, offsets };
  const byId = new Map(source.map(node => [node.id, node]));
  const displayed = new Map<string, ModelNode>();
  const groups = new Map<string, ModelNode[]>();
  for (const node of source) {
    if (node.hidden) continue;
    const parent = node.parentNode && byId.has(String(node.parentNode)) && !byId.get(String(node.parentNode))?.hidden ? String(node.parentNode) : '';
    if (!groups.has(parent)) groups.set(parent, []);
    groups.get(parent)!.push(node);
  }
  const visiting = new Set<string>(), done = new Set<string>();
  const layout = (parent: string, headerSpace = 0): void => {
    if (visiting.has(parent) || done.has(parent)) return;
    visiting.add(parent);
    const placed: Array<{ node: ModelNode; w: number; h: number }> = [];
    const members = [...groups.get(parent) ?? []].sort((a, b) => a.position.y - b.position.y || a.position.x - b.position.x || a.id.localeCompare(b.id));
    for (const original of members) {
      const explanation = nodeAnnotation(original, states, visible);
      const base = nodeCardSize(original);
      let w = explanation ? Math.max(280, base.w) : base.w;
      let h: number;
      if (original.type === 'moduleNode' && original.data.expanded) {
        layout(original.id);
        for (const child of groups.get(original.id) ?? []) {
          const shown = displayed.get(child.id), size = sizes.get(child.id);
          if (!shown || !size) continue;
          w = Math.max(w, shown.position.x + size.w + 20);
        }
        const extra = (explanation ? nodeAnnotationLayout(explanation, w).height : 0)
          + moduleExpandedHeaderHeight(original.data, w) - moduleExpandedHeaderHeight(original.data, base.w);
        h = base.h + extra;
        for (const child of groups.get(original.id) ?? []) {
          const shown = displayed.get(child.id), size = sizes.get(child.id);
          if (!shown || !size) continue;
          const shifted = { ...shown, position: { ...shown.position, y: shown.position.y + extra } };
          displayed.set(child.id, shifted);
          const before = offsets.get(child.id)!;
          offsets.set(child.id, { x: before.x, y: before.y + extra });
          h = Math.max(h, shifted.position.y + size.h + 20);
        }
      } else {
        const measured = original.type === 'moduleNode' ? moduleCardHeight(original.data, w)
          : original.type === 'inputNode' || original.type === 'outputNode' ? nodeCardSize(original, w).h : layerCardHeight(original.data, w);
        h = measured + (explanation ? nodeAnnotationLayout(explanation, w).height : 0);
      }
      let y = original.position.y + headerSpace;
      for (const previous of placed) {
        if (original.position.x < previous.node.position.x + previous.w + 20
          && original.position.x + w + 20 > previous.node.position.x
          && y < previous.node.position.y + previous.h + 24) y = previous.node.position.y + previous.h + 24;
      }
      const node: ModelNode = { ...original, position: { x: original.position.x, y }, style: { ...original.style, width: w, height: h } };
      displayed.set(node.id, node); sizes.set(node.id, { w, h });
      offsets.set(node.id, { x: node.position.x - original.position.x, y: node.position.y - original.position.y });
      placed.push({ node, w, h });
    }
    visiting.delete(parent); done.add(parent);
  };
  layout('');
  // Malformed imported parentage must not make interpretation hide nodes.
  return { nodes: source.map(node => displayed.get(node.id) ?? node), sizes, offsets };
}
