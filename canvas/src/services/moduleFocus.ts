import type { ModelEdge, ModelGraph, ModelNode } from '../types';
import { dagreLayout, reconcileModuleView } from './canvasLayout';
import { nodeCardSize } from '../canvas/cardMetrics';
import { displayedPorts } from '../canvas/capturePorts';
import { passThroughHidden } from './passThroughPresentation';

export type FocusDirection = 'LR' | 'TB';
export type FocusOrientation = 'auto' | FocusDirection;
export interface FocusViewport { width: number; height: number }
export interface FocusLayout {
  direction: FocusDirection;
  positions: Map<string, { x: number; y: number }>;
  bounds: { x: number; y: number; w: number; h: number };
}

export const focusNodeName = (node: ModelNode): string => String(node.data.captureLabel
  || node.data.parameters.name || node.data.layerType.displayName || node.data.layerType.name || node.id);

function navigationIndex(nodes: readonly ModelNode[]) {
  const byId = new Map(nodes.map(node => [node.id, node]));
  const children = new Map<string, ModelNode[]>();
  for (const node of nodes) if (node.data.parentId) {
    if (!children.has(node.data.parentId)) children.set(node.data.parentId, []);
    children.get(node.data.parentId)!.push(node);
  }
  return { byId, children };
}

/** A cycle or broken ownership chain cannot become navigable ancestry. */
export function moduleFocusPath(nodes: readonly ModelNode[], id: string | null, index = navigationIndex(nodes)): ModelNode[] {
  const { byId } = index;
  const path: ModelNode[] = [];
  const seen = new Set<string>();
  while (id) {
    const node = byId.get(id);
    if (!node || node.type !== 'moduleNode' || seen.has(id)) return [];
    seen.add(id); path.unshift(node); id = node.data.parentId || null;
  }
  return path;
}

export function canEnterModule(nodes: readonly ModelNode[], id: string, index = navigationIndex(nodes)): boolean {
  const node = index.byId.get(id);
  return node?.type === 'moduleNode' && node.data.expandable !== false && !node.data.repeatHidden
    && moduleFocusPath(nodes, id, index).length > 0
    && (index.children.get(id) ?? []).some(child => !child.data.repeatHidden && !passThroughHidden(child) && !child.data.operationGroupHiddenBy);
}

export function enterableModuleIds(nodes: readonly ModelNode[]): Set<string> {
  const index = navigationIndex(nodes);
  return new Set(nodes.filter(node => canEnterModule(nodes, node.id, index)).map(node => node.id));
}

/** Re-capture namespaces display IDs by capture ID. Retain navigation only for
 * the same source and uniquely matching recorded invocation, label, type and
 * owner. This does not establish operation/edge correspondence or selection. */
export function recapturedModuleMap(before: readonly ModelNode[], after: readonly ModelNode[], sameSourceProjection = false): Map<string, string> {
  const key = (node: ModelNode) => {
    const ref = node.data.captureRef;
    return node.type === 'moduleNode' && ref?.kind === 'invocation'
      ? JSON.stringify([sameSourceProjection ? 'same-source-projection' : ref.sourceId, ref.semanticId, node.data.captureLabel, node.data.layerType.name]) : undefined;
  };
  const group = (nodes: readonly ModelNode[]) => {
    const groups = new Map<string, ModelNode[]>();
    for (const node of nodes) {
      const value = key(node); if (!value) continue;
      groups.set(value, [...groups.get(value) ?? [], node]);
    }
    return groups;
  };
  const left = group(before), right = group(after), candidates = new Map<string, ModelNode>();
  left.forEach((nodes, value) => { const next = right.get(value); if (nodes.length === 1 && next?.length === 1) candidates.set(nodes[0].id, next[0]); });
  const result = new Map<string, string>();
  const beforeIndex = navigationIndex(before), afterIndex = navigationIndex(after);
  for (const node of before) {
    const next = candidates.get(node.id);
    if (!next) continue;
    const oldPath = moduleFocusPath(before, node.id, beforeIndex), newPath = moduleFocusPath(after, next.id, afterIndex);
    if (oldPath.length && oldPath.length === newPath.length && oldPath.every((parent, i) => candidates.get(parent.id)?.id === newPath[i].id)) result.set(node.id, next.id);
  }
  return result;
}

/** Exact multi-target focus across different owners needs the full overview.
 * Never report a folded module as if its hidden operation had been selected. */
export function focusScopeForTargets(nodes: ModelNode[], edges: ModelEdge[], ids: readonly string[]): string | null {
  const targets = ids.map(id => nodes.find(node => node.id === id));
  const parent = targets[0]?.data.parentId;
  if (!parent || !targets.every(node => node?.data.parentId === parent)) return null;
  const projection = projectModuleFocus(nodes, edges, parent);
  return projection && ids.every(id => projection.nodes.some(node => node.id === id)) ? parent : null;
}

/** Keep a matching registered path when comparing sibling invocations. An
 * ambiguous repeated call is deliberately not chosen by name alone. */
export function matchingFocusSibling(nodes: readonly ModelNode[], current: string, ancestor: string, sibling: string): string {
  const lookup = navigationIndex(nodes);
  const path = moduleFocusPath(nodes, current, lookup);
  const index = path.findIndex(node => node.id === ancestor);
  if (index < 0 || !canEnterModule(nodes, sibling, lookup)) return current;
  const original = path[index];
  const replacement = nodes.find(node => node.id === sibling)!;
  if (original.data.parentId !== replacement.data.parentId) return current;
  const key = (node: ModelNode) => String(node.data.parameters._module_path || focusNodeName(node)).split('.').pop();
  let target = sibling;
  for (const old of path.slice(index + 1)) {
    const matches = (lookup.children.get(target) ?? []).filter(node => node.type === 'moduleNode'
      && key(node) === key(old) && node.data.layerType.id === old.data.layerType.id && canEnterModule(nodes, node.id, lookup));
    if (matches.length !== 1) break;
    target = matches[0].id;
  }
  return target;
}

/** Derive a scene, never a replacement semantic graph. The existing reconciler
 * owns pass-through folding, operation groups and captured boundary-port maps.
 * Outside endpoints become explicitly marked presentation-only boundary cards.
 * Every displayed wire retains its original captured edge/port/value evidence. */
export function projectModuleFocus(nodes: ModelNode[], edges: ModelEdge[], scopeId: string): ModelGraph | null {
  if (!canEnterModule(nodes, scopeId)) return null;
  const opened = new Set(moduleFocusPath(nodes, scopeId).map(node => node.id));
  const graph = reconcileModuleView(nodes.map(node => node.type === 'moduleNode'
    ? { ...node, data: { ...node.data, expanded: opened.has(node.id) } } : node), edges, { skipChildLayout: true });
  const children = graph.nodes.filter(node => node.data.parentId === scopeId && !node.hidden);
  if (!children.length) return null;
  const inside = new Set(children.map(node => node.id));
  const byId = new Map(graph.nodes.map(node => [node.id, node]));
  const boundaries = new Map<string, ModelNode>();
  const wires: ModelEdge[] = [];
  const boundary = (external: ModelNode, direction: 'in' | 'out', handle?: string | null) => {
    const id = `focus-boundary:${encodeURIComponent(JSON.stringify([scopeId, direction, external.id, handle || '']))}`;
    if (!boundaries.has(id)) {
      const port = displayedPorts(external.data).find(item => item.id === handle);
      const ports = port ? [{ ...port, nodeId: id, memberPortId: port.memberPortId || port.id }] : undefined;
      const type = direction === 'in' ? 'input' : 'output';
      boundaries.set(id, { id, type: `${type}Node`, position: { x: 0, y: 0 },
        data: { layerType: { ...external.data.layerType, id: type, name: type, displayName: type },
          parameters: {}, captureLabel: `${direction === 'in' ? 'From' : 'To'} ${focusNodeName(external)}`,
          focusBoundary: { direction, externalNodeId: external.id },
          ...(ports ? { capturePorts: ports, captureDisplayPorts: ports } : {}),
        } });
    }
    return id;
  };
  for (const edge of graph.edges) {
    if (edge.hidden) continue;
    const sourceInside = inside.has(edge.source), targetInside = inside.has(edge.target);
    if (!sourceInside && !targetInside) continue;
    const source = byId.get(edge.source), target = byId.get(edge.target);
    if (!source || !target) continue;
    if (source.hidden || target.hidden || (!sourceInside && source.data.parentId === scopeId)
      || (!targetInside && target.data.parentId === scopeId)) continue;
    const sourceId = sourceInside ? source.id : boundary(source, 'in', edge.sourceHandle ?? edge.data?.sourceHandle);
    const targetId = targetInside ? target.id : boundary(target, 'out', edge.targetHandle ?? edge.data?.targetHandle);
    wires.push({ ...edge, source: sourceId, target: targetId,
      data: { ...edge.data, synthetic: true, originalEdgeIds: edge.data?.originalEdgeIds ?? (edge.data?.synthetic ? [] : [edge.id]) } });
  }
  return { nodes: [...children, ...Array.from(boundaries.values())].map(node => ({ ...node,
    parentNode: undefined, extent: undefined, expandParent: undefined, hidden: false,
    data: { ...node.data, parentId: undefined },
  })), edges: wires };
}

/** Position/selection changes do not need another pair of DAG layouts. */
export function focusLayoutKey(graph: ModelGraph): string {
  return JSON.stringify([graph.nodes.map(node => [node.id, nodeCardSize(node)]),
    graph.edges.map(edge => [edge.source, edge.target, edge.data?.diffStatus === 'removed'])]);
}

export function layoutModuleFocus(graph: ModelGraph, viewport: FocusViewport, orientation: FocusOrientation = 'auto'): FocusLayout {
  const sizes = new Map(graph.nodes.map(node => [node.id, nodeCardSize(node)]));
  const candidate = (direction: FocusDirection): FocusLayout => {
    // Comparison-only boundary cards still need a rank beside their peer.
    // This affects placement, never which connections count as an active path.
    const positions = dagreLayout(graph.nodes.map(node => node.id), sizes,
      graph.edges, { rankdir: direction, ranksep: 72, nodesep: 56, marginx: 32, marginy: 32 });
    let w = 0, h = 0;
    positions.forEach((pos, id) => { const size = sizes.get(id)!; w = Math.max(w, pos.x + size.w + 32); h = Math.max(h, pos.y + size.h + 32); });
    return { direction, positions, bounds: { x: 0, y: 0, w: Math.max(1, w), h: Math.max(1, h) } };
  };
  if (orientation !== 'auto') return candidate(orientation);
  const horizontal = candidate('LR'), vertical = candidate('TB');
  const score = (layout: FocusLayout) => Math.min(Math.max(1, viewport.width - 64) / layout.bounds.w,
    Math.max(1, viewport.height - 64) / layout.bounds.h);
  // Near ties retain the familiar reading direction. The caller stores this
  // decision per scope; a sidebar resize never rotates the graph automatically.
  return score(vertical) > score(horizontal) * 1.12 ? vertical : horizontal;
}

export function placeModuleFocus(graph: ModelGraph, layout: FocusLayout): ModelGraph {
  return { nodes: graph.nodes.map(node => ({ ...node, position: layout.positions.get(node.id) || node.position,
    data: { ...node.data, focusDirection: layout.direction } })), edges: graph.edges };
}
