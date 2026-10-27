import type { ModelEdge, ModelNode } from '../types';
import { AnchorBox, computePortSides, capturedPortSide, LayoutDirection, skipLaneSide } from './edgeAnchors';
import { endpointOnSide } from './capturePorts';
import { bottomLanePull, edgeBow, edgeCurveBounds, rightLanePull } from './edgeCurve';

/** Display-only shortcuts: an alternate visible path reaches the same target. */
export function skipEdgeIds(edges: readonly ModelEdge[], isVisible: (id: string) => boolean): Set<string> {
  // Removed review connections are comparison annotations, not active paths.
  const visible = edges.filter(edge => edge.data?.diffStatus !== 'removed'
    && !edge.hidden && isVisible(edge.source) && isVisible(edge.target));
  const outgoing = new Map<string, ModelEdge[]>();
  for (const edge of visible) {
    if (!outgoing.has(edge.source)) outgoing.set(edge.source, []);
    outgoing.get(edge.source)!.push(edge);
  }
  // A search cannot reach an earlier node in a DAG. This prevents every
  // ordinary edge in a full model from traversing the rest of the program.
  const incoming = new Map<string, number>();
  for (const edge of visible) {
    if (!incoming.has(edge.source)) incoming.set(edge.source, 0);
    incoming.set(edge.target, (incoming.get(edge.target) ?? 0) + 1);
  }
  const queue = Array.from(incoming.keys()).filter(id => incoming.get(id) === 0);
  const order = new Map<string, number>();
  for (let index = 0; index < queue.length; index++) {
    const id = queue[index]; order.set(id, index);
    for (const edge of outgoing.get(id) ?? []) {
      incoming.set(edge.target, incoming.get(edge.target)! - 1);
      if (incoming.get(edge.target) === 0) queue.push(edge.target);
    }
  }
  const skips = new Set<string>();
  for (const edge of visible) {
    const pending = (outgoing.get(edge.source) ?? []).filter(other => other.target !== edge.target).map(other => other.target);
    const seen = new Set([edge.source]);
    while (pending.length) {
      const id = pending.pop()!;
      if (id === edge.target) { skips.add(edge.id); break; }
      if (seen.has(id)) continue;
      const targetOrder = order.get(edge.target), currentOrder = order.get(id);
      if (targetOrder !== undefined && currentOrder !== undefined && currentOrder > targetOrder) continue;
      seen.add(id);
      pending.push(...(outgoing.get(id) ?? []).map(other => other.target));
    }
  }
  return skips;
}

/** Shared by live sync, dragging and SVG export. Topology is cached; boxes are
 * read at routing time so a dragged card never uses its previous position.
 * `direction` must be the one the nodes were laid out with. */
export function createEdgeRouter(nodes: readonly ModelNode[], edges: readonly ModelEdge[] = [], direction?: LayoutDirection) {
  const visible = nodes.filter(node => !node.hidden);
  const byId = new Map(nodes.map(node => [node.id, node]));
  const sides = computePortSides(visible, direction);
  // A module focus projection lays out its own scope, so its direction wins.
  const laneOf = (node: ModelNode) => skipLaneSide(node.data.focusDirection
    ? (node.data.focusDirection === 'LR' ? 'horizontal' : 'vertical') : direction);
  const parent = (node: ModelNode) => node.parentNode || node.data.parentId || '';
  const children = new Map<string, ModelNode[]>();
  for (const node of visible) {
    const key = parent(node);
    if (!children.has(key)) children.set(key, []);
    children.get(key)!.push(node);
  }
  // A visible shortcut around an alternate path gets a side lane. Topology,
  // not drag coordinates, selects ports; one shared tensor never gets two dots.
  const skips = skipEdgeIds(edges, id => sides.has(id));
  for (const edge of edges) {
    const source = byId.get(edge.source), target = byId.get(edge.target);
    if (!skips.has(edge.id)) continue;
    if (!source || !target || (!source.data.focusDirection && (!parent(source) || !parent(target)))) { skips.delete(edge.id); continue; }
    const laneSide = laneOf(source);
    for (const [node, direction, handle] of [
      [source, 'out', edge.sourceHandle ?? edge.data?.sourceHandle],
      [target, 'in', edge.targetHandle ?? edge.data?.targetHandle],
    ] as const) {
      const side = sides.get(node.id);
      if (!side) continue;
      if (handle) side.ports = { ...side.ports, [handle]: laneSide };
      else side[direction] = laneSide;
    }
  }
  const ancestors = (id: string) => {
    const result = new Set<string>();
    let node = byId.get(id);
    while (node && !result.has(node.id)) { result.add(node.id); node = byId.get(parent(node)); }
    return result;
  };
  return {
    sides,
    route(edge: ModelEdge, boxFor: (id: string) => AnchorBox | undefined) {
      const source = byId.get(edge.source), target = byId.get(edge.target);
      const from = boxFor(edge.source), to = boxFor(edge.target);
      if (!source || !target || !from || !to) return undefined;
      const startSide = capturedPortSide(sides.get(source.id), 'out', edge.sourceHandle ?? edge.data?.sourceHandle);
      const endSide = capturedPortSide(sides.get(target.id), 'in', edge.targetHandle ?? edge.data?.targetHandle);
      const start = endpointOnSide(source.data, edge.sourceHandle ?? edge.data?.sourceHandle, from, startSide);
      const end = endpointOnSide(target.data, edge.targetHandle ?? edge.data?.targetHandle, to, endSide);
      const obstacles = () => {
        const excluded = new Set([...Array.from(ancestors(source.id)), ...Array.from(ancestors(target.id))]);
        const siblings = new Set([...(children.get(parent(source)) ?? []), ...(children.get(parent(target)) ?? [])]);
        return Array.from(siblings).flatMap(node => {
          const box = excluded.has(node.id) ? undefined : boxFor(node.id);
          return box ? [box] : [];
        });
      };
      const curveOffset = edge.id.startsWith('review:') && edge.data?.diffStatus === 'removed' ? 220
        : skips.has(edge.id) ? (laneOf(source) === 'bottom' ? bottomLanePull : rightLanePull)(start.x, start.y, end.x, end.y, obstacles())
          : edgeBow(start.x, start.y, end.x, end.y, startSide, endSide);
      const bounds = edgeCurveBounds(start.x, start.y, end.x, end.y, 28, curveOffset, startSide, endSide);
      return { start, end, startSide, endSide, curveOffset, bounds };
    },
  };
}

export function routedShapeGeometry(route: NonNullable<ReturnType<ReturnType<typeof createEdgeRouter>['route']>>) {
  const { start, end, startSide, endSide, curveOffset, bounds: { minX, minY, maxX, maxY } } = route;
  return { x: minX, y: minY, props: {
    w: Math.max(1, maxX - minX), h: Math.max(1, maxY - minY),
    startX: start.x - minX, startY: start.y - minY,
    endX: end.x - minX, endY: end.y - minY,
    startSide, endSide, curveOffset,
  } };
}
