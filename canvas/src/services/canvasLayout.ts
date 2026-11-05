/**
 * canvasLayout — single source of truth for turning the flat *logical* graph
 * (module nodes carrying `data.expanded`, children carrying `data.parentId`)
 * into ReactFlow *render* nodes/edges.
 *
 * It does three jobs the old inline mapper in ModelBuilder did NOT:
 *   1. Collapse-aware sub-flows: children of a collapsed module are hidden and
 *      their edges re-routed to the module node; children of an expanded
 *      module are parented (`extent:'parent'`) and laid out inside it.
 *   2. Real DAG layout via dagre (size-aware, left-to-right), run on load and
 *      after every expand/collapse so large backbones don't pile at the origin.
 *   3. layerType re-resolution + repeat-badge metadata, so template/import
 *      nodes get icons/labels and repeated stacks show "×N".
 *
 * The same function backs template load, import, grouping, expand/collapse and
 * the manual Layout button — replacing three divergent code paths.
 */
import dagre from 'dagre';
import { nodeCardSize, moduleCardHeight, moduleCardHeader, moduleCardWidth } from '../canvas/cardMetrics';
import { endpointOnSide } from '../canvas/capturePorts';
import { skipEdgeIds } from '../canvas/edgeRouting';
import type { LayoutDirection } from '../canvas/edgeAnchors';
import { CapturePort, ModelNode, ModelEdge, ModelNodeData } from '../types';
import { graphNodeOrigin } from './graphOverview';
import { getLayerByType } from '../data/layerTypes';
import { passThroughConnections, passThroughHidden } from './passThroughPresentation';
import { deriveOperationGroups } from './operationGroups';
import { mergeGraphIssues, ownGraphIssues } from '../canvas/graphIssues';
import type { GraphIssue } from '../types';

const MODULE_COLLAPSED_W = 250;
/**
 * The footprint a COLLAPSED module claims. One number, three consumers: the spacing
 * dagre gives its neighbours (`TYPE_SIZE.moduleNode`), the tldraw shape box written
 * from `style.height`, and therefore the selection box and hit area too.
 *
 * It was 140 while the card drew far more than that. A collapsed module card is
 * auto-height — header, pill row, optional route line, "Collapsed block view" and the
 * Expand Block CTA — so it overflowed its own box by the bottom third, and dagre put
 * the next card one `nodesep` (48) below 140, i.e. INSIDE the drawn card, right over
 * the CTA. That is the same mistake `layerCardHeight` exists to prevent for layer
 * cards, which measure their own rows.
 *
 * 240 is measured, not guessed: on the QA fixture, through the installed host on
 * 2026-09-13, `repeat:stack.0` drew 250x235 and `blocks.0` 250x212 while both shape
 * boxes were 250x140. A module card with a `defined in …` link, a preview minimap or
 * sublayer chips still draws taller than this — the honest fix is a measured
 * `moduleCardHeight` beside `layerCardHeight`, which is more than this repair.
 */
const MODULE_COLLAPSED_H = 240;
const MODULE_HEADER = 52;     // reserved for the expanded module's header bar
const SKIP_LANE_RESERVE = 48; // keeps incoming skip lanes clear of frame header controls
const MODULE_PAD = 20;

const CHILD_SIZE = { w: 215, h: 96 };


type Pos = { x: number; y: number };

/** Enrich a node's layerType from the registry (icon/category/params) while
 *  PRESERVING any semantic label the template gave it — so "MLP", "Patch
 *  Embedding", "Class Token" survive instead of being reset to the raw op. */
function resolveLayerType(orig: any): any {
  const reg = orig?.id ? getLayerByType(orig.id) : undefined;
  if (!reg) return orig;
  return {
    ...reg,
    name: orig?.name || reg.name,
    displayName: orig?.displayName || orig?.name || reg.displayName,
    icon: orig?.icon || reg.icon,
  };
}

/** Run a left-to-right layered dagre layout. `sizes` gives per-node footprint;
 *  returns top-left positions keyed by node id. */
/**
 * Which edges are the story, and which are scenery.
 *
 * A project graph carries two kinds of edge and they are not the same claim.
 * **Dataflow** — `feeds`, `trains`, `reads`, `produces` — is the pipeline: the
 * order things actually happen in, and what a reader opened the graph for.
 * **`builds`** and `calls` say that one definition constructs or calls another,
 * which is true of nearly every pair of definitions in a codebase and says nothing
 * about sequence.
 *
 * Ranked together, the call graph bends the pipeline out of shape: a `data` node
 * that also builds two helpers is pulled to the centroid of all three, so the three
 * pipeline stages end up on three different rows and the whole thing reads as a
 * mesh with no line to follow. Measured on `data → model → trainer` plus two
 * helpers: ranked together the stages land at y 228 / 60 / 144; ranked on dataflow
 * alone, 60 / 60 / 60.
 *
 * (Edge `weight` is the obvious lever and does not work: dagre 0.8.5 ignores it in
 * coordinate assignment. Weight 1 and weight 100 produce byte-identical output.)
 *
 * A per-class graph has no `_relationship` on any edge, so it has no pipeline by
 * this test and is laid out exactly as it always was.
 */
const FLOW_RELATIONSHIPS = new Set([
  'reads',
  'transforms',
  'feeds',
  'trains',
  'evaluates',
  'produces',
  'logs',
  'derives_from',
]);

export interface LayoutEdge {
  source: string;
  target: string;
}

/** `_relationship` is what the project document says an edge MEANS. It rides
 *  beside the label because the label may be the attribute the source named
 *  (`transformer.h`), which cannot be read back into a relationship. */
export function isFlowEdge(edge: { data?: unknown }): boolean {
  const relationship = (edge.data as { _relationship?: string } | undefined)?._relationship;
  return Boolean(relationship && FLOW_RELATIONSHIPS.has(relationship));
}

/** Vertical gap between the pipeline and the scaffolding drawn beneath it. */
const STORY_BAND_GAP = 120;

/**
 * The pipeline on its own row, and everything else below it.
 *
 * Two passes, because they answer different questions. The first ranks the nodes
 * dataflow reaches, which is the sequence. The second lays out what is left using
 * only the edges among THEM — the call graph keeps its own shape rather than being
 * flattened into a column — and drops the result beneath the spine.
 *
 * Falls back to one pass over everything when the graph has no dataflow at all,
 * which is every per-class graph and any project whose edges are all `builds`.
 */
function layoutByStory(
  ids: string[],
  sizes: Map<string, { w: number; h: number }>,
  edges: Array<LayoutEdge & { flow: boolean }>,
  direction?: LayoutDirection,
): Map<string, Pos> {
  const rankdir = direction === 'vertical' ? 'TB' : 'LR';
  const flow = edges.filter((edge) => edge.flow);
  if (flow.length === 0) {
    return dagreLayout(ids, sizes, edges, { rankdir });
  }

  const spine = new Set<string>();
  for (const edge of flow) {
    spine.add(edge.source);
    spine.add(edge.target);
  }
  const rest = ids.filter((id) => !spine.has(id));
  const placed = dagreLayout(ids.filter((id) => spine.has(id)), sizes, flow, { rankdir });
  if (rest.length === 0) {
    return placed;
  }

  // The scaffolding sits beside the spine on its cross axis: below a row,
  // right of a column.
  const across = rankdir === 'TB' ? 'x' : 'y';
  let floor = 0;
  placed.forEach((pos, id) => {
    const size = sizes.get(id);
    if (size) floor = Math.max(floor, pos[across] + (across === 'x' ? size.w : size.h));
  });

  const below = dagreLayout(rest, sizes, edges.filter(
    (edge) => !spine.has(edge.source) && !spine.has(edge.target),
  ), { rankdir });
  below.forEach((pos, id) => {
    placed.set(id, { ...pos, [across]: pos[across] + floor + STORY_BAND_GAP });
  });
  return placed;
}

export function dagreLayout(
  ids: string[],
  sizes: Map<string, { w: number; h: number }>,
  edges: LayoutEdge[],
  opts: { rankdir?: 'LR' | 'TB'; nodesep?: number; ranksep?: number; marginx?: number; marginy?: number } = {}
): Map<string, Pos> {
  // Full JAX programs exceed dagre's recursive DFS stack. Rank large graphs
  // iteratively; this only changes positions, never graph records or edges.
  if (ids.length > 1000 || edges.length > 4000) {
    const next = new Map(ids.map(id => [id, [] as string[]]));
    const incoming = new Map(ids.map(id => [id, 0]));
    const rank = new Map(ids.map(id => [id, 0]));
    for (const edge of edges) {
      if (edge.source === edge.target || !next.has(edge.source) || !next.has(edge.target)) continue;
      next.get(edge.source)!.push(edge.target);
      incoming.set(edge.target, incoming.get(edge.target)! + 1);
    }
    const queue = ids.filter(id => incoming.get(id) === 0);
    for (let index = 0; index < queue.length; index++) {
      const id = queue[index];
      for (const target of next.get(id)!) {
        rank.set(target, Math.max(rank.get(target)!, rank.get(id)! + 1));
        incoming.set(target, incoming.get(target)! - 1);
        if (incoming.get(target) === 0) queue.push(target);
      }
    }
    // Cyclic components have no topological order; retain them in a separate
    // band rather than recursing forever or omitting their nodes.
    let finalRank = 1;
    for (const value of Array.from(rank.values())) finalRank = Math.max(finalRank, value + 1);
    const bands = new Map<number, string[]>();
    for (const id of ids) {
      const level = incoming.get(id)! > 0 ? finalRank : rank.get(id)!;
      const band = bands.get(level) ?? [];
      band.push(id); bands.set(level, band);
    }
    const vertical = opts.rankdir === 'TB';
    let along = (vertical ? opts.marginy : opts.marginx) ?? 24;
    const out = new Map<string, Pos>();
    for (const level of Array.from(bands.keys()).sort((a, b) => a - b)) {
      let across = (vertical ? opts.marginx : opts.marginy) ?? 24, width = 0;
      for (const id of bands.get(level)!) {
        const size = sizes.get(id) ?? CHILD_SIZE;
        out.set(id, vertical ? { x: across, y: along } : { x: along, y: across });
        across += (vertical ? size.w : size.h) + (opts.nodesep ?? 48);
        width = Math.max(width, vertical ? size.h : size.w);
      }
      along += width + (opts.ranksep ?? 90);
    }
    return out;
  }
  const g = new dagre.graphlib.Graph();
  g.setGraph({
    rankdir: opts.rankdir ?? 'LR',
    nodesep: opts.nodesep ?? 48,
    ranksep: opts.ranksep ?? 90,
    marginx: opts.marginx ?? 24,
    marginy: opts.marginy ?? 24,
  });
  g.setDefaultEdgeLabel(() => ({}));

  const idSet = new Set(ids);
  for (const id of ids) {
    const s = sizes.get(id) || CHILD_SIZE;
    g.setNode(id, { width: s.w, height: s.h });
  }
  for (const e of edges) {
    if (idSet.has(e.source) && idSet.has(e.target) && e.source !== e.target) {
      g.setEdge(e.source, e.target);
    }
  }

  dagre.layout(g);

  const out = new Map<string, Pos>();
  for (const id of ids) {
    const n = g.node(id);
    if (!n) continue;
    // dagre returns node centres; ReactFlow positions are top-left.
    out.set(id, { x: n.x - n.width / 2, y: n.y - n.height / 2 });
  }
  return out;
}

interface ChildLayout {
  positions: Map<string, Pos>;
  width: number;
  height: number;
}

/** Lay a module's internal children out locally (relative to the module
 *  origin), leaving room for the header bar. */
function layoutChildren(
  children: ModelNode[],
  internalEdges: ModelEdge[],
  moduleSizes: Map<string, { w: number; h: number }>,
  preservePositions = false,
  sideReserve = 0,
  direction?: LayoutDirection,
  moduleData?: ModelNodeData,
): ChildLayout {
  if (children.length === 0) {
    return { positions: new Map(), width: MODULE_COLLAPSED_W, height: MODULE_COLLAPSED_H };
  }
  const ids = children.map((c) => c.id);
  // Size each child from the SAME function that renders its card. CHILD_SIZE's
  // h: 96 was a stale estimate: with ranksep 30 it packed children on a 126px
  // pitch, while a real card is 157px tall (2 params + 2 shape rows) since the
  // card height was made truthful. Every child therefore overlapped the next by
  // ~31px, which is what made an expanded module look like a jumbled stack —
  // the children WERE inside the frame, they were just laid on top of each other.
  const sizeOf = (c: ModelNode) => moduleSizes.get(c.id) ?? nodeCardSize(c);
  const childSizes = new Map(children.map((c) => [c.id, sizeOf(c)] as const));
  const sizes = childSizes;
  // Without a chosen direction, stack a module's internals top-to-bottom (the
  // design's vertical sub-layer column) so the module stays narrow beside its
  // neighbours in the top-level row. A horizontal canvas runs them in a row.
  const row = direction === 'horizontal';
  const local = dagreLayout(ids, sizes, internalEdges, {
    rankdir: row ? 'LR' : 'TB',
    nodesep: 24,
    ranksep: 48,
    marginx: MODULE_PAD,
    marginy: MODULE_PAD,
  });

  let maxRight = 0;
  let maxBottom = 0;
  const positions = new Map<string, Pos>();
  for (const id of ids) {
    const p = local.get(id) || { x: MODULE_PAD, y: MODULE_PAD };
    const saved = preservePositions ? children.find(child => child.id === id)?.position : undefined;
    const placed = saved ? { ...saved } : { x: p.x + MODULE_PAD, y: p.y + MODULE_HEADER };
    positions.set(id, placed);
    const sz = childSizes.get(id) || CHILD_SIZE;
    maxRight = Math.max(maxRight, placed.x + sz.w);
    maxBottom = Math.max(maxBottom, placed.y + sz.h);
  }
  // Reserve symmetrically on automatic layout, keeping the inner column centred
  // under its upstream card. Restored positions only gain right-hand clearance.
  // A row's lane runs below it, so the reserve moves to the height.
  const reserve = sideReserve * (preservePositions ? 1 : 2);
  const width = Math.max(340, maxRight + MODULE_PAD) + (row ? 0 : reserve);
  if (sideReserve && !preservePositions && !row) {
    const minLeft = Math.min(...Array.from(positions.values()).map(position => position.x));
    const shift = width / 2 - (minLeft + maxRight) / 2;
    positions.forEach(position => { position.x += shift; });
  }
  if (sideReserve && !preservePositions && row) positions.forEach(position => { position.y += sideReserve; });
  const headerClearance = moduleData ? Math.max(0, moduleCardHeader(moduleData, width).height - MODULE_HEADER) : 0;
  const wrappedHeader = moduleData && moduleCardHeader(moduleData, width);
  const neededClearance = preservePositions ? (positions.size && wrappedHeader && wrappedHeader.titleLines.length > 1
    ? Math.max(0, wrappedHeader.height + MODULE_PAD - Math.min(...Array.from(positions.values()).map(position => position.y))) : 0) : headerClearance;
  if (neededClearance) positions.forEach(position => { position.y += neededClearance; });
  return {
    positions,
    // Floor the expanded width so the module header fits without truncating.
    width,
    height: Math.max(MODULE_COLLAPSED_H, maxBottom + MODULE_PAD + (row ? reserve : 0) + neededClearance),
  };
}

/** Heuristic: a freshly-flattened template/import graph has degenerate
 *  positions (everything at {0,0} or overlapping). Used to decide whether to
 *  auto-run dagre on load without clobbering a user-arranged saved canvas. */
export function graphNeedsLayout(nodes: ModelNode[]): boolean {
  const topLevel = nodes.filter((n) => !n.data.parentId);
  if (topLevel.length < 2) return false;
  const seen = new Set<string>();
  let atOrigin = 0;
  for (const n of topLevel) {
    const pos = n.position || { x: 0, y: 0 };
    if (Math.abs(pos.x) < 1 && Math.abs(pos.y) < 1) atOrigin += 1;
    const key = `${Math.round(pos.x)},${Math.round(pos.y)}`;
    if (seen.has(key)) return true; // two nodes overlap → needs layout
    seen.add(key);
  }
  return atOrigin >= 2;
}

export interface ReconcileOptions {
  /** A separate scope projection owns placement; only derive visibility/ports. */
  skipChildLayout?: boolean;
  /** Hold an explicitly toggled module's header in place while making room. */
  anchorNodeId?: string;
  /** Re-run dagre on the top level (and expanded children). When false, keep
   *  existing positions (e.g. reloading a canvas the user arranged by hand). */
  relayout?: boolean;
  /** Restore a portable snapshot exactly before any user-requested relayout. */
  preserveChildPositions?: boolean;
  /** Canvas orientation for the top level and expanded contents. Absent keeps
   *  the legacy left-to-right graph with top-to-bottom module contents. */
  direction?: LayoutDirection;
}

/**
 * Reconcile a logical flat graph into a ReactFlow render graph: hide/parent
 * children by their module's expanded state, size modules, lay everything out,
 * re-route edges around collapsed modules, and re-resolve layerType.
 */
export function reconcileModuleView(
  nodes: ModelNode[],
  inputEdges: ModelEdge[],
  opts: ReconcileOptions = {}
): { nodes: ModelNode[]; edges: ModelEdge[] } {
  nodes = deriveOperationGroups(nodes, inputEdges);
  // Expansion is a view of existing contents, never a request to execute Python.
  // Recompute even on restore/focus so saved open state cannot create an empty frame.
  const childCounts = new Map<string, number>();
  for (const node of nodes) {
    const parent = node.data.parentId;
    if (parent && parent !== node.id && !node.data.repeatHidden) {
      childCounts.set(parent, (childCounts.get(parent) || 0) + 1);
    }
  }
  nodes = nodes.map(node => {
    node = { ...node, data: { ...node.data } };
    if (node.type !== 'moduleNode') return node;
    const count = childCounts.get(node.id) || 0;
    // Source project frames can load contents through the host. Keep their
    // existing state; only local captured contents are bounded here.
    if (count === 0 && !node.data.moduleCapture && !node.data.parameters?._shape_provenance && !node.data.shapeEvidence) return node;
    return { ...node, data: { ...node.data,
      expandable: count > 0,
      expanded: count > 0 && Boolean(node.data.expanded),
      moduleLayerCount: count > 0 ? node.data.moduleLayerCount || count : 0,
    } };
  });
  // Drop synthetic display edges from a previous reconcile so re-running (on
  // every expand/collapse, fed the live render edges) is idempotent and never
  // accumulates wrapper edges.
  const edges = inputEdges.filter((e) => !e.data?.synthetic);
  const annotations = passThroughConnections(nodes);
  const activeEdges = edges.filter(edge => edge.data?.diffStatus !== 'removed');
  const layoutEdges = [...activeEdges, ...annotations];
  const moduleIdList = nodes.filter((n) => n.type === 'moduleNode').map((n) => n.id);
  const moduleIds = new Set(moduleIdList);

  // parentId is only meaningful when it points at an actual module node.
  const byId = new Map(nodes.map((n) => [n.id, n] as const));
  const expandedSet = new Set(
    nodes.filter((n) => n.type === 'moduleNode' && n.data.expanded).map((n) => n.id)
  );
  const expandedOf = (id: string): boolean => expandedSet.has(id);

  const parentOf = (n: ModelNode): string | undefined => {
    const pid = n.data.parentId;
    return pid && moduleIds.has(pid) ? pid : undefined;
  };

  // Memoized: moduleDepth is asked once per module in each of the two sorts and
  // ancestorsExpanded once per child, and both walk the parent chain. Recomputing
  // them per call made a fully-expanded graph (hundreds of frames) quadratic in
  // tree depth for no reason — the shape is fixed for the whole reconcile.
  const depthMemo = new Map<string, number>();
  const moduleDepth = (id: string): number => {
    const cached = depthMemo.get(id);
    if (cached !== undefined) return cached;
    let depth = 0;
    let current = byId.get(id);
    const seen = new Set<string>();
    while (current) {
      const parent = parentOf(current);
      if (!parent || seen.has(parent)) break;
      seen.add(parent);
      depth += 1;
      current = byId.get(parent);
    }
    depthMemo.set(id, depth);
    return depth;
  };

  const childrenByModule = new Map<string, ModelNode[]>();
  for (const id of moduleIdList) childrenByModule.set(id, []);
  for (const n of nodes) {
    const pid = parentOf(n);
    if (pid) childrenByModule.get(pid)!.push(n);
  }
  const ancestorsExpandedMemo = new Map<string, boolean>();
  const ancestorsExpanded = (node: ModelNode): boolean => {
    const cached = ancestorsExpandedMemo.get(node.id);
    if (cached !== undefined) return cached;
    let expanded = true;
    let parent = parentOf(node);
    const seen = new Set<string>();
    while (parent && !seen.has(parent)) {
      seen.add(parent);
      if (!expandedOf(parent)) { expanded = false; break; }
      const parentNode = byId.get(parent);
      parent = parentNode ? parentOf(parentNode) : undefined;
    }
    ancestorsExpandedMemo.set(node.id, expanded);
    return expanded;
  };

  // This mapping affects presentation only. Captured member endpoints stay on
  // the original edges, and each argument use gets its own collapsed endpoint.
  const displayNodeMemo = new Map<string, string>();
  const displayNode = (id: string): string => {
    const cached = displayNodeMemo.get(id);
    if (cached !== undefined) return cached;
    let node = byId.get(id);
    if (!node) return id;
    let shown = node.data.operationGroupHiddenBy ?? id;
    const seen = new Set<string>();
    let parent = parentOf(node);
    while (parent && !seen.has(parent)) {
      seen.add(parent);
      if (!expandedOf(parent)) shown = parent;
      node = byId.get(parent) ?? node;
      parent = parentOf(node);
    }
    displayNodeMemo.set(id, shown);
    return shown;
  };
  const ports = new Map(nodes.flatMap(node => (node.data.capturePorts ?? []).map(port => [port.id, port] as const)));
  const boundaryPorts = new Map<string, Map<string, CapturePort>>();
  const displayedPort = (portId: string, shownNode: string): string => {
    const original = ports.get(portId);
    if (!original || original.nodeId === shownNode) return portId;
    const id = `boundary:${portId}`;
    if (!boundaryPorts.has(shownNode)) boundaryPorts.set(shownNode, new Map());
    boundaryPorts.get(shownNode)!.set(id, { ...original, id, nodeId: shownNode, memberPortId: portId, paths: [] });
    return id;
  };
  for (const edge of edges) if (edge.data?.captureEdge) {
    const source = displayNode(edge.source), target = displayNode(edge.target);
    if (source === target) continue;
    displayedPort(edge.data.captureEdge.sourcePortId, source);
    displayedPort(edge.data.captureEdge.targetPortId, target);
  }
  for (const edge of annotations) if (ancestorsExpanded(byId.get(edge.target)!)) {
    displayedPort(edge.sourceHandle!, displayNode(edge.source));
  }
  for (const node of nodes) if (node.data.captureRef) {
    node.data = { ...node.data, captureDisplayPorts: boundaryPorts.has(node.id)
      ? [...Array.from(boundaryPorts.get(node.id)!.values()), ...(node.data.operationGroup
        ? (node.data.capturePorts ?? []).filter(port => port.direction === 'out') : [])]
      : node.data.captureRef.kind === 'invocation' && node.data.expanded ? [] : node.data.capturePorts ?? [] };
  }

  // Bucket every layout edge to the ONE expanded module whose direct canonical
  // children its endpoints resolve to. Computing this per module (the old
  // `layoutEdges.flatMap` inside the loop below) rescanned every edge for every
  // expanded frame — O(modules × edges) — which dominated a fully-expanded graph.
  // Each node's ancestor→direct-child map is built once; an edge can be internal
  // to at most its lowest common expanded module.
  const canonicalChildren = new Map<string, Set<string>>();
  for (const id of moduleIdList) {
    const set = new Set<string>();
    for (const child of childrenByModule.get(id) || []) {
      if (!child.data?.repeatHidden && !passThroughHidden(child) && !child.data.operationGroupHiddenBy) set.add(child.id);
    }
    canonicalChildren.set(id, set);
  }
  const childUnder = new Map<string, Map<string, string>>();
  for (const node of nodes) {
    const under = new Map<string, string>();
    let child = node.data.operationGroupHiddenBy ?? node.id;
    let parent = parentOf(node);
    const seen = new Set<string>();
    while (parent && !seen.has(parent)) {
      seen.add(parent);
      if (canonicalChildren.get(parent)?.has(child)) under.set(parent, child);
      child = parent;
      const parentNode = byId.get(parent);
      parent = parentNode ? parentOf(parentNode) : undefined;
    }
    childUnder.set(node.id, under);
  }
  const internalByModule = new Map<string, ModelEdge[]>();
  for (const edge of layoutEdges) {
    const source = childUnder.get(edge.source);
    const target = childUnder.get(edge.target);
    if (!source || !target) continue;
    const [small, large] = source.size <= target.size ? [source, target] : [target, source];
    small.forEach((child, moduleId) => {
      const other = large.get(moduleId);
      if (other === undefined || other === child) return;
      const mapped: ModelEdge = { ...edge, source: source.get(moduleId)!, target: target.get(moduleId)! };
      const bucket = internalByModule.get(moduleId);
      if (bucket) bucket.push(mapped);
      else internalByModule.set(moduleId, [mapped]);
    });
  }

  const isHiddenChild = (id: string): boolean => {
    const node = byId.get(id);
    if (!node) return false;
    if (passThroughHidden(node)) return true;
    if (node.data.operationGroupHiddenBy) return true;
    if (node.data?.repeatHidden) return true; // hidden repeat instance
    return !ancestorsExpanded(node);
  };

  // A skip entering an expanded frame needs a clear lane beside its header.
  // Use the same displayed topology as the router, including collapsed ends.
  const displayedEdges = layoutEdges.map(edge => ({ ...edge,
    source: displayNode(edge.source), target: displayNode(edge.target), hidden: false,
  })).filter(edge => edge.source !== edge.target);
  const skips = skipEdgeIds(displayedEdges, id => byId.has(id) && !isHiddenChild(id));
  const laneFrames = new Set<string>();
  for (const edge of displayedEdges) {
    if (!skips.has(edge.id) || !parentOf(byId.get(edge.source)!) || !parentOf(byId.get(edge.target)!)) continue;
    const sourceAncestors = childUnder.get(edge.source);
    childUnder.get(edge.target)?.forEach((_child, id) => {
      if (expandedOf(id) && !sourceAncestors?.has(id)) laneFrames.add(id);
    });
  }

  // Per-module child layout + resulting module footprint. A repeated module
  // (×N) lays out and is sized for ONE canonical instance — its other instances
  // carry `repeatHidden` and are never rendered, so an expanded ×N block shows a
  // single representative block instead of all N copies (mirroring the exported
  // `nn.ModuleList([Block() for _ in range(N)])`).
  const childLayouts = new Map<string, ChildLayout>();
  const moduleSize = new Map<string, { w: number; h: number }>();
  // Inner modules are sized first so an expanded parent reserves the actual
  // footprint of its nested module, not the footprint of an ordinary layer.
  const bottomUpModules = [...moduleIdList].sort((left, right) =>
    moduleDepth(right) - moduleDepth(left) || left.localeCompare(right));
  const diagnostics = new Map<string, GraphIssue[]>();
  for (const id of bottomUpModules) {
    const descendants = (childrenByModule.get(id) || []).flatMap(child =>
      diagnostics.get(child.id) ?? ownGraphIssues(child.id, child.data));
    diagnostics.set(id, mergeGraphIssues(descendants, ownGraphIssues(id, byId.get(id)!.data, !descendants.length)));
  }
  for (const id of bottomUpModules) {
    // A collapsed module is sized for its boundary rows and hides its children,
    // so its internal dagre layout would be computed and then thrown away.
    // Skipping it matters: an imported UNet is ~80 modules, and every reconcile
    // (i.e. every expand/collapse) paid ~80 dagre runs for nothing.
    const moduleNode = byId.get(id)!;
    if (!expandedOf(id) || !ancestorsExpanded(moduleNode)) {
      moduleSize.set(id, { w: moduleCardWidth(moduleNode.data), h: moduleCardHeight(moduleNode.data) });
      continue;
    }
    if (opts.skipChildLayout) continue;
    const children = childrenByModule.get(id) || [];
    const canonical = children.filter((c) => !c.data?.repeatHidden && !passThroughHidden(c) && !c.data.operationGroupHiddenBy);
    const internalEdges = internalByModule.get(id) || [];
    const layout = layoutChildren(canonical, internalEdges, moduleSize,
      opts.preserveChildPositions && (!opts.relayout || moduleNode.data.moduleLayoutState === 'custom'),
      laneFrames.has(id) ? SKIP_LANE_RESERVE : 0, opts.direction, moduleNode.data);
    childLayouts.set(id, layout);
    moduleSize.set(id, { w: layout.width, h: layout.height });
  }

  // ── Top-level layout (modules + standalone nodes) ──────────────────────
  const topLevel = nodes.filter((n) => !parentOf(n));
  const topIds = topLevel.filter(n => !n.data.operationGroupHiddenBy).map((n) => n.id);
  const topSizes = new Map<string, { w: number; h: number }>();
  for (const n of topLevel) {
    topSizes.set(n.id, moduleSize.get(n.id) ?? nodeCardSize(n));
  }

  // Map any endpoint to the top-level node that represents it (a child folds
  // into its module) so the top-level DAG is connected.
  const toTop = (id: string): string | undefined => {
    let node = byId.get(id);
    if (!node) return undefined;
    const seen = new Set<string>();
    while (parentOf(node) && !seen.has(node.id)) {
      seen.add(node.id);
      node = byId.get(parentOf(node)!) ?? node;
    }
    return node.id;
  };
  const topEdges: Array<LayoutEdge & { flow: boolean }> = [];
  for (const e of activeEdges) {
    const s = toTop(e.source);
    const t = toTop(e.target);
    if (s && t && s !== t) topEdges.push({ source: s, target: t, flow: isFlowEdge(e) });
  }

  const doRelayout = opts.relayout ?? false;
  const topPos = doRelayout
    ? layoutByStory(topIds, topSizes, topEdges, opts.direction)
    : new Map(topLevel.map((n) => [n.id, n.position || { x: 0, y: 0 }] as const));

  // ── Assemble render nodes (parents before children) ───────────────────
  const outNodes: ModelNode[] = [];

  for (const n of topLevel) {
    const resolved = n.data.captureRef ? n.data.layerType : resolveLayerType(n.data.layerType);
    const pos = topPos.get(n.id) || n.position || { x: 0, y: 0 };
    if (n.type === 'moduleNode') {
      const size = moduleSize.get(n.id) || { w: MODULE_COLLAPSED_W, h: MODULE_COLLAPSED_H };
      const childCount = (childrenByModule.get(n.id) || []).length;
      outNodes.push({
        ...n,
        position: pos,
        parentNode: undefined,
        extent: undefined,
        style: { ...(n.style || {}), width: size.w, height: size.h },
        zIndex: 0,
        data: {
          ...n.data,
          layerType: resolved,
          moduleLayerCount: n.data.moduleLayerCount || childCount,
        },
      } as ModelNode);
    } else {
      outNodes.push({
        ...n,
        hidden: n.data.captureRef?.kind === 'operation' ? Boolean(n.data.operationGroupHiddenBy) : n.hidden,
        position: pos,
        parentNode: undefined,
        extent: undefined,
        zIndex: 1,
        data: { ...n.data, layerType: resolved },
      } as ModelNode);
    }
  }

  const topDownModules = [...moduleIdList].sort((left, right) =>
    moduleDepth(left) - moduleDepth(right) || left.localeCompare(right));
  for (const id of topDownModules) {
    const expanded = expandedOf(id);
    // Absent for a collapsed module (no layout was run); its children are
    // hidden, so they fall back to the module origin and get real positions the
    // moment it expands and reconcile runs again.
    const layout = childLayouts.get(id);
    for (const child of childrenByModule.get(id) || []) {
      const resolved = child.data.captureRef ? child.data.layerType : resolveLayerType(child.data.layerType);
      // Non-canonical instances of a repeated module are kept (for the payload)
      // but never shown — only the canonical block renders when expanded.
      const repeatHidden = Boolean(child.data?.repeatHidden);
      const hidden = !expanded || !ancestorsExpanded(child) || repeatHidden || passThroughHidden(child) || Boolean(child.data.operationGroupHiddenBy);
      const childModuleSize = child.type === 'moduleNode' ? moduleSize.get(child.id) : undefined;
      outNodes.push({
        ...child,
        parentNode: id,
        extent: 'parent',
        expandParent: (expanded && !repeatHidden) || undefined,
        hidden,
        position: layout?.positions.get(child.id) || child.position || { x: MODULE_PAD, y: MODULE_HEADER },
        ...(childModuleSize
          ? { style: { ...(child.style || {}), width: childModuleSize.w, height: childModuleSize.h } }
          : {}),
        zIndex: 2,
        data: { ...child.data, layerType: resolved, parentId: id },
      } as ModelNode);
    }
  }

  // ── Edge visibility + synthesized wrapper edges ───────────────────────
  // Real edges touching a child of a collapsed module are HIDDEN (not
  // rerouted/dropped) so re-expanding restores them verbatim. For a collapsed
  // module whose boundary edges point at its children (the auto-modularize /
  // hierarchical-template case), hiding alone would leave the box looking
  // disconnected — so we additionally synthesize display-only edges between the
  // collapsed-module boxes (and their visible neighbours). These carry
  // `data.synthetic` and are stripped before any backend payload, so the
  // inferable graph stays identical to the flat one. When boundary edges
  // already point at the wrapper (manual grouping / backend root_edges) nothing
  // is folded, so no synthetic edge is added and there is no duplication.
  // Older exports include both a wrapper summary and the real child boundary.
  // Hide only the redundant summary while its child connection is visible.
  const coveredBoundaries = new Set<string>();
  const boundaryKey = (source: string, target: string, handle: string | null | undefined, incoming: boolean) =>
    JSON.stringify([source, target, handle ?? '', incoming]);
  for (const edge of edges) {
    if (edge.data?.captureEdge || edge.data?.diffStatus === 'removed'
      || isHiddenChild(edge.source) || isHiddenChild(edge.target)) continue;
    for (const incoming of [true, false]) {
      let node = byId.get(incoming ? edge.target : edge.source);
      const seen = new Set<string>();
      while (node && parentOf(node) && !seen.has(node.id)) {
        seen.add(node.id);
        const parent = parentOf(node)!;
        if (expandedOf(parent)) coveredBoundaries.add(boundaryKey(
          incoming ? edge.source : parent, incoming ? parent : edge.target,
          incoming ? edge.sourceHandle ?? edge.data?.sourceHandle : edge.targetHandle ?? edge.data?.targetHandle, incoming));
        node = byId.get(parent);
      }
    }
  }
  const coveredWrapper = (edge: ModelEdge) => !edge.data?.captureEdge && edge.data?.diffStatus !== 'removed' && (
    coveredBoundaries.has(boundaryKey(edge.source, edge.target, edge.sourceHandle ?? edge.data?.sourceHandle, true))
    || coveredBoundaries.has(boundaryKey(edge.source, edge.target, edge.targetHandle ?? edge.data?.targetHandle, false)));
  const outEdges: ModelEdge[] = edges.map((e) => ({
    ...e,
    hidden: isHiddenChild(e.source) || isHiddenChild(e.target) || coveredWrapper(e),
  }));

  const folded = new Map<string, ModelEdge>();
  const synthEdges: ModelEdge[] = [];
  for (const edge of annotations) if (!isHiddenChild(edge.target)) {
    const source = displayNode(edge.source);
    synthEdges.push({ ...edge, source, sourceHandle: displayedPort(edge.sourceHandle!, source), hidden: false });
  }
  for (const e of edges) {
    const ds = displayNode(e.source);
    const dt = displayNode(e.target);
    if (ds === dt) continue; // internal to one collapsed module
    if (ds === e.source && dt === e.target) continue; // nothing folded → real edge shows
    if (e.data?.captureEdge) {
      synthEdges.push({ ...e, id: `synth__${e.id}`, source: ds, target: dt, hidden: false,
        sourceHandle: displayedPort(e.data.captureEdge.sourcePortId, ds),
        targetHandle: displayedPort(e.data.captureEdge.targetPortId, dt),
        data: { ...e.data, synthetic: true, originalEdgeIds: [e.id] } });
      continue;
    }
    const key = `${ds}__${dt}`;
    const previous = folded.get(key);
    if (previous) {
      previous.data!.originalEdgeIds!.push(e.id);
      if (previous.data!.diffStatus !== e.data?.diffStatus && (previous.data!.diffStatus || e.data?.diffStatus)) previous.data!.diffStatus = 'modified';
      if (previous.data!.role !== e.data?.role) {
        previous.data!.role = 'mixed';
        delete previous.data!.roleEvidence;
      }
      continue;
    }
    const synthetic: ModelEdge = { id: `synth__${key}`, source: ds, target: dt,
      data: { synthetic: true, role: e.data?.role, roleEvidence: e.data?.roleEvidence, originalEdgeIds: [e.id], diffStatus: e.data?.diffStatus } };
    folded.set(key, synthetic);
    synthEdges.push(synthetic);
  }

  for (const node of outNodes) if (node.type === 'moduleNode') {
    const issues = diagnostics.get(node.id) ?? [];
    node.data = { ...node.data, moduleDiagnostics: issues, moduleDiagnosticCount: issues.length,
      moduleDiagnosticStatus: issues.some(issue => issue.severity === 'error') ? 'error' : issues.length ? 'warning' : undefined };
  }
  if (doRelayout) {
    alignBoundaryCards(outNodes, [...outEdges, ...synthEdges], topPos, opts.direction);
    for (const node of outNodes) if (!node.parentNode && topPos.has(node.id)) node.position = topPos.get(node.id)!;
  }
  if (opts.anchorNodeId) {
    const before = graphNodeOrigin(nodes, opts.anchorNodeId);
    const after = graphNodeOrigin(outNodes, opts.anchorNodeId);
    if (before && after) for (const node of outNodes) if (!node.parentNode) {
      node.position = { x: node.position.x + before.x - after.x, y: node.position.y + before.y - after.y };
    }
  }
  return { nodes: outNodes, edges: [...outEdges, ...synthEdges] };
}

/** An expanded column's entry and exit are not at its centre. Keep external
 * boundary cards beside the actual children, leaving short hooks in the empty
 * header/footer bands. With a chosen direction both ends share one axis, so the
 * card lines up with the child's port instead. This only runs for an explicitly
 * requested layout. */
function alignBoundaryCards(nodes: ModelNode[], edges: ModelEdge[], positions: Map<string, Pos>, direction?: LayoutDirection): void {
  // Cards move across the flow; ports sit on the flow's entry and exit sides.
  const across = direction === 'vertical' ? 'x' : 'y';
  const [inSide, outSide] = direction === 'vertical' ? ['top', 'bottom'] as const : ['left', 'right'] as const;
  const placed = nodes.map(node => positions.has(node.id) ? { ...node, position: positions.get(node.id)! } : node);
  const byId = new Map(placed.map(node => [node.id, node]));
  const parent = (node: ModelNode) => node.parentNode || node.data.parentId;
  const top = placed.filter(node => !parent(node) && !node.hidden);
  const expandedRoot = (id: string): ModelNode | undefined => {
    let node = byId.get(id);
    if (!node || !parent(node)) return undefined;
    const seen = new Set<string>();
    while (node && parent(node) && !seen.has(node.id)) {
      seen.add(node.id); node = byId.get(parent(node)!);
    }
    return node?.type === 'moduleNode' && node.data.expanded ? node : undefined;
  };
  const visibleEdges = edges.filter(edge => !edge.hidden && edge.data?.diffStatus !== 'removed');
  const move = (node: ModelNode, value: number): boolean => {
    const p = { ...(positions.get(node.id) ?? node.position), [across]: value }, size = nodeCardSize(node);
    const collision = top.some(other => {
      if (other.id === node.id) return false;
      const q = positions.get(other.id) ?? other.position, s = nodeCardSize(other);
      return p.x < q.x + s.w + 48 && p.x + size.w + 48 > q.x
        && p.y < q.y + s.h + 48 && p.y + size.h + 48 > q.y;
    });
    if (!collision) positions.set(node.id, p);
    return !collision;
  };
  const boundaryRoot = (links: ModelEdge[], entering: boolean) => {
    const roots = links.map(edge => expandedRoot(entering ? edge.target : edge.source));
    return roots.length > 0 && roots[0] && roots.every(root => root?.id === roots[0]!.id);
  };
  for (const node of top) {
    if (node.type === 'moduleNode') continue;
    const outgoing = visibleEdges.filter(edge => edge.source === node.id);
    const incoming = visibleEdges.filter(edge => edge.target === node.id);
    const entering = Boolean(boundaryRoot(outgoing, true));
    const leaving = Boolean(boundaryRoot(incoming, false));
    if (entering === leaving) continue; // Ambiguous bridge between two frames.
    const links = entering ? outgoing : incoming;
    const peers = links.map(edge => entering ? edge.target : edge.source);
    const size = nodeCardSize(node);
    const candidates = links.flatMap((edge, i) => {
      const peer = byId.get(peers[i])!;
      const origin = graphNodeOrigin(placed, peer.id);
      if (!origin) return [];
      const handle = entering ? edge.sourceHandle ?? edge.data?.sourceHandle : edge.targetHandle ?? edge.data?.targetHandle;
      const anchor = endpointOnSide(node.data, handle, { x: 0, y: 0, ...size }, entering ? outSide : inSide);
      if (!direction) return [origin.y + (entering ? -12 : nodeCardSize(peer).h + 24) - anchor.y];
      const peerHandle = entering ? edge.targetHandle ?? edge.data?.targetHandle : edge.sourceHandle ?? edge.data?.sourceHandle;
      const peerAnchor = endpointOnSide(peer.data, peerHandle, { x: 0, y: 0, ...nodeCardSize(peer) }, entering ? inSide : outSide);
      return [origin[across] + peerAnchor[across] - anchor[across]];
    });
    if (!candidates.length) continue;
    const value = entering ? Math.min(...candidates) : Math.max(...candidates);
    if (!move(node, value)) continue;
    // Continue the same flow through unambiguous external chains.
    let current = node;
    const seen = new Set([current.id]);
    while (current) {
      const currentId = current.id;
      const onward = visibleEdges.filter(edge => (entering ? edge.target : edge.source) === currentId);
      if (onward.length !== 1) break;
      const edge = onward[0], next = byId.get(entering ? edge.source : edge.target);
      if (!next || next.hidden || parent(next) || next.type === 'moduleNode' || seen.has(next.id)) break;
      const back = visibleEdges.filter(edge => (entering ? edge.source : edge.target) === next.id);
      if (back.length !== 1) break;
      const currentHandle = entering ? edge.targetHandle ?? edge.data?.targetHandle : edge.sourceHandle ?? edge.data?.sourceHandle;
      const nextHandle = entering ? edge.sourceHandle ?? edge.data?.sourceHandle : edge.targetHandle ?? edge.data?.targetHandle;
      const currentAnchor = endpointOnSide(current.data, currentHandle,
        { ...(positions.get(current.id) ?? current.position), ...nodeCardSize(current) }, entering ? inSide : outSide);
      const nextAnchor = endpointOnSide(next.data, nextHandle, { x: 0, y: 0, ...nodeCardSize(next) }, entering ? outSide : inSide);
      if (!move(next, currentAnchor[across] - nextAnchor[across])) break;
      seen.add(next.id); current = next;
    }
  }
}

/**
 * Dagre layout for the TOP LEVEL only (nodes without a `parentNode`), leaving
 * module children untouched. Size-aware: a module's current rendered footprint
 * (style.width/height — large when expanded) feeds the spacing so an expanded
 * block no longer overruns its neighbour. Safe for any graph shape; used by the
 * manual "Layout" button. Returns new top-left positions keyed by node id.
 */
export function layoutTopLevel(nodes: ModelNode[], edges: ModelEdge[], direction?: LayoutDirection): Map<string, Pos> {
  // A node is a child if it's parented in the render graph (`parentNode`) or in
  // the logical graph (`data.parentId`) — so this works on either shape.
  const childParent = (n: ModelNode): string | undefined => n.parentNode || n.data.parentId;
  const topLevel = nodes.filter((n) => !childParent(n));
  const topIds = topLevel.map((n) => n.id);
  const topIdSet = new Set(topIds);

  const sizes = new Map(topLevel.map((n) => [n.id, nodeCardSize(n)] as const));

  // Fold any endpoint up to its top-level ancestor so the DAG stays connected.
  const parentNodeOf = new Map(nodes.map((n) => [n.id, childParent(n)] as const));
  const toTop = (id: string): string | undefined => {
    let cur: string | undefined = id;
    const guard = new Set<string>();
    while (cur && !topIdSet.has(cur) && !guard.has(cur)) {
      guard.add(cur);
      cur = parentNodeOf.get(cur) || undefined;
    }
    return cur && topIdSet.has(cur) ? cur : undefined;
  };
  const topEdges: Array<LayoutEdge & { flow: boolean }> = [];
  for (const e of edges) {
    if (e.data?.diffStatus === 'removed') continue;
    const s = toTop(e.source);
    const t = toTop(e.target);
    if (s && t && s !== t) topEdges.push({ source: s, target: t, flow: isFlowEdge(e) });
  }

  const positions = layoutByStory(topIds, sizes, topEdges, direction);
  alignBoundaryCards(nodes, edges, positions, direction);
  return positions;
}
