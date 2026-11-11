import type { ModelNode } from '../types';
import type { ModelExplanationState, NodeExplanationCitation, NodeExplanationState, ViewExplanation } from '../ide/nodeExplanation.generated';
import { wrapCardText } from './cardIdentity';

/** A retained result belongs only to a view that still contains its visible snapshot. */
export function applicableViewExplanation(state: ModelExplanationState | undefined, visible: boolean | undefined,
  nodes: readonly ModelNode[]): ViewExplanation | undefined {
  if (!visible || !state?.view) return undefined;
  const displayed = new Set(nodes.filter(node => !node.hidden).map(node => node.id));
  return state.view.nodeIds.every(id => displayed.has(id)) ? state.view : undefined;
}

/** The caller supplies an actual edge's directed endpoints; interpretations never add edges. */
export function viewExplanationConnection(view: ViewExplanation | undefined, sourceId?: string, targetId?: string) {
  return sourceId && targetId ? view?.connections.find(connection => connection.sourceNodeId === sourceId
    && connection.targetNodeId === targetId) : undefined;
}

export function viewConnectionLabelLayout(label: string, availableWidth: number) {
  const width = Math.max(56, Math.min(180, availableWidth));
  const lines = wrapCardText(label, width, 11);
  return { lines, width, height: lines.length * 15 };
}

/** A view proof can use a different host-issued ID for the same exact source window. */
export function viewCitationState(citation: NodeExplanationCitation, revision: string,
  states: readonly Readonly<Record<string, NodeExplanationState>>[], preferredNodeIds: readonly string[]) {
  const ids = [...new Set([...preferredNodeIds, ...states.flatMap(group => Object.keys(group))])];
  for (const group of states) for (const id of ids) {
    const state = group[id];
    if (state?.status !== 'ready' || state.revision !== revision) continue;
    const actual = state.explanation?.citations.find(proof => proof.file === citation.file
      && proof.sourceDigest === citation.sourceDigest && proof.startLine === citation.startLine
      && proof.endLine === citation.endLine);
    if (actual) return { state, citationId: actual.id };
  }
  return undefined;
}
