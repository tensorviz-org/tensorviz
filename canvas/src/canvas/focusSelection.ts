import { Box, type Editor } from 'tldraw';
import type { ModelNode, ModelEdge } from '../types';
import { nodeIdToShapeId, edgeIdToShapeId } from './canvasShapeId';
import { selectedEdgeId } from './edgeSelection';
import { revealOperationGroup } from '../services/operationGroups';

export interface CanvasFocusPlan {
  nodes: ModelNode[];
  nodeIds: string[];
  edge?: ModelEdge;
}
/** Reveal ancestors without changing graph semantics, including nested connections. */
export function planCanvasFocus(nodes: ModelNode[], edges: ModelEdge[], nodeIds: string[], edgeIds: string[], expandAncestors = true): CanvasFocusPlan | null {
  if ((!nodeIds.length && !edgeIds.length) || (nodeIds.length && edgeIds.length) || edgeIds.length > 1 || new Set(nodeIds).size !== nodeIds.length) return null;
  const edge = edgeIds.length ? edges.find(item => item.id === edgeIds[0]) : undefined;
  if (edgeIds.length && !edge) return null;
  const targets = edge ? [edge.source, edge.target] : nodeIds;
  const byId = new Map(nodes.map(node => [node.id, node]));
  if (targets.some(id => !byId.has(id))) return null;
  for (const id of targets) nodes = revealOperationGroup(nodes, id);
  const ancestors = new Set<string>();
  for (const id of targets) {
    let node = byId.get(id);
    const seen = new Set<string>([id]);
    let parent = node?.data?.parentId ?? node?.parentNode;
    while (parent) {
      if (seen.has(parent)) return null;
      seen.add(parent); ancestors.add(parent);
      node = byId.get(parent);
      if (!node) return null;
      parent = node.data?.parentId ?? node.parentNode;
    }
  }
  return { nodes: nodes.map(node => {
    const expand = expandAncestors && ancestors.has(node.id) && !node.data.expanded;
    const reveal = targets.includes(node.id) && node.data.capturePassThrough && !node.data.passThroughShown;
    return expand || reveal ? { ...node, data: { ...node.data, ...(expand ? { expanded: true } : {}),
      ...(reveal ? { passThroughShown: true } : {}) } } : node;
  }), nodeIds, edge };
}

/** An editor focus that arrived before its node was on the canvas. */
export function queuedEditorFocus(pending: string | null, ids: readonly string[]): { pending: string | null; focus: string | null } {
  if (!pending || !ids.includes(pending)) return { pending, focus: null };
  return { pending: null, focus: pending };
}

/** False while expanded shapes have not reached tldraw yet; callers retry after render. */
export function applyCanvasFocus(editor: Editor, plan: CanvasFocusPlan, inset = 64): boolean {
  const ids = plan.edge ? [edgeIdToShapeId(plan.edge)] : plan.nodeIds.map(nodeIdToShapeId);
  const bounds = ids.map(id => editor.getShapePageBounds(id));
  if (bounds.some(bound => !bound)) return false;
  editor.setCurrentTool('select');
  if (plan.edge) { editor.selectNone(); selectedEdgeId.set(plan.edge.id); }
  else { editor.select(...ids); selectedEdgeId.set(null); }
  editor.zoomToBounds(Box.Common(bounds as Box[]), { targetZoom: 1, inset, animation: { duration: 0 } });
  return true;
}
