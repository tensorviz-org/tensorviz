import { SavedGraphView } from '../services/graphCollaboration';
import { GraphFile } from '../services/graphFile';
import { graphDocumentV2ToModelGraph } from '../services/graphDocumentAdapter';
import { ModelGraph } from '../types';
import type { LayoutDirection } from '../canvas/edgeAnchors';
import { reconcileModuleView } from '../services/canvasLayout';
import { revealOperationGroup } from '../services/operationGroups';

/** Opening ancestors is view state only. The snapshot is never rewritten. */
export function revealGraphNode(graph: ModelGraph, id: string, direction?: LayoutDirection): ModelGraph {
  const byId = new Map(graph.nodes.map(node => [node.id, node]));
  const ancestors = new Set<string>();
  let parent = byId.get(id)?.data.parentId;
  while (parent && !ancestors.has(parent)) {
    ancestors.add(parent);
    parent = byId.get(parent)?.data.parentId;
  }
  return reconcileModuleView(revealOperationGroup(graph.nodes, id).map(node => ({
    ...node,
    data: { ...node.data, ...(ancestors.has(node.id) ? { expanded: true } : {}), ...(node.id === id ? { repeatHidden: false, passThroughShown: true } : {}) },
  })), graph.edges, { relayout: ancestors.size > 0, direction });
}

export function initialGraphFileView(file: Pick<GraphFile, 'document'>): ModelGraph {
  const graph = graphDocumentV2ToModelGraph(file.document);
  return reconcileModuleView(graph.nodes, graph.edges, { relayout: false, preserveChildPositions: true, direction: file.document.view.layout.direction });
}

/** Saved walkthrough views may reference an older architecture. Restore known
 * nodes only; original targets remain on the step for visible context warnings. */
export function restoreGraphFileView(graph: ModelGraph, view: SavedGraphView, direction?: LayoutDirection): ModelGraph {
  const expanded = new Set(view.expandedModuleIds);
  const targets = new Set(view.selectedEdge ? [view.selectedEdge.source, view.selectedEdge.target] : view.selectedNodeIds);
  const byId = new Map(graph.nodes.map(node => [node.id, node]));
  let nodes = graph.nodes;
  for (const id of Array.from(targets)) {
    nodes = revealOperationGroup(nodes, id);
    const visited = new Set<string>();
    let parent = byId.get(id)?.data.parentId;
    while (parent && !visited.has(parent)) {
      visited.add(parent); expanded.add(parent); parent = byId.get(parent)?.data.parentId;
    }
  }
  return reconcileModuleView(nodes.map(node => ({ ...node,
    position: view.nodePositions[node.id] ?? node.position,
    data: { ...node.data, ...(node.type === 'moduleNode' ? { expanded: expanded.has(node.id) } : {}),
      ...(targets.has(node.id) ? { repeatHidden: false, passThroughShown: true } : {}) },
  })), graph.edges, { relayout: false, preserveChildPositions: true, direction });
}

/** Keep the same page-space center when a recipient has a different viewport. */
export function cameraForViewport(view: SavedGraphView, viewport: { width: number; height: number }): SavedGraphView['camera'] {
  if (!view.viewport) return view.camera;
  return { ...view.camera,
    x: view.camera.x + (viewport.width - view.viewport.width) / (2 * view.camera.z),
    y: view.camera.y + (viewport.height - view.viewport.height) / (2 * view.camera.z),
  };
}

/** Focus review targets at a readable scale in the unobscured canvas area. */
export function cameraForTarget(bounds: { x: number; y: number; w: number; h: number }, viewport: { width: number; height: number }, panelWidth = 0): SavedGraphView['camera'] {
  const width = Math.max(200, viewport.width - (viewport.width > 740 ? panelWidth : 0));
  const height = Math.max(180, viewport.height);
  const z = Math.max(.01, Math.min(1.25, (width - 96) / Math.max(1, bounds.w), (height - 128) / Math.max(1, bounds.h)));
  return { x: width / (2 * z) - (bounds.x + bounds.w / 2), y: height / (2 * z) - (bounds.y + bounds.h / 2), z };
}
