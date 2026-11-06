import type { GraphReviewState, ProjectionPayload } from './protocol.generated';
import type { GraphDocumentV2, ModelNode } from '../types';

export const DROPOUT_INSERT_HINT = 'In eval captures, Dropout is recorded but hidden as an unchanged call. Reveal it with More tools → Show unchanged calls (N). LayerNorm and Linear stay visible.';
export const isDropoutLayer = (type: string) => /^(?:.*\.)?(?:dropout(?:1d|2d|3d)?|alphadropout|featurealphadropout)$/i.test(type);

export function reviewAddsDropout(state: GraphReviewState): boolean {
  const before = state.before.document as GraphDocumentV2;
  const after = state.after.document as GraphDocumentV2;
  const ids = new Set(before?.semantic?.nodes.map(node => node.node_id));
  return !!after?.semantic?.nodes.some(node => !ids.has(node.node_id) && isDropoutLayer(node.op));
}

/** Scoped to the accepted source; ordinary opens never arm a reveal. */
export function createDropoutInsertionReveal() {
  let pending: { file: string; cls: string; sourceDigest: string; ids: Set<string> } | undefined;
  return {
    accept(state: GraphReviewState, live?: ProjectionPayload | null) {
      pending = reviewAddsDropout(state) && state.after.sourceDigest ? {
        file: state.after.file, cls: state.after.cls, sourceDigest: state.after.sourceDigest,
        ids: new Set((live?.document as GraphDocumentV2 | undefined)?.semantic?.nodes.map(node => node.node_id)),
      } : undefined;
    },
    clear() { pending = undefined; },
    apply(payload: ProjectionPayload, nodes: ModelNode[]): ModelNode[] {
      if (!pending || !payload.traced || payload.greyed) return nodes;
      if (payload.file !== pending.file || payload.cls !== pending.cls) { pending = undefined; return nodes; }
      if (payload.sourceDigest !== pending.sourceDigest) return nodes;
      const accepted = pending;
      pending = undefined;
      const ops = new Map((payload.document as GraphDocumentV2).semantic.nodes.map(node => [node.node_id, node.op]));
      return nodes.map(node => !accepted.ids.has(node.id) && node.data.capturePassThrough && isDropoutLayer(ops.get(node.id) ?? '')
        ? { ...node, data: { ...node.data, passThroughShown: true } } : node);
    },
  };
}
export const dropoutInsertionReveal = createDropoutInsertionReveal();
