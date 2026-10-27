/**
 * autoModularize — deterministically induce module structure in a FLAT logical
 * graph by detecting repeated blocks and linear pipelines, then wrapping each in
 * a collapsible `moduleNode`.
 *
 * This is a *display/structure* transform, not a semantic one. It returns a new
 * logical graph where:
 *   - a new `moduleNode` wraps each detected block (children carry `parentId`),
 *   - children KEEP their original edges (we do NOT reroute boundary edges onto
 *     the wrapper the way manual grouping does),
 *   - modules start collapsed (`expanded: false`).
 *
 * Keeping the real child-targeted edges is what makes this safe for models like
 * F5-TTS whose blocks consume cross-boundary conditioning signals (a DiT block
 * reads both the sequence `x` AND the global timestep `t_emb`). The single
 * entry/exit rewrite in shapeInferencePayload would destroy that topology; by
 * leaving the edges on the real children, the verify/validate payload stays
 * byte-identical to the flat graph. The collapsed module still looks connected
 * because reconcileModuleView synthesizes display-only wrapper edges from those
 * same real edges.
 *
 * Detection is pure code (no model): longest-path period detection for repeated
 * stacks (the ×N case) + strict 1-in/1-out chain folding for the rest.
 */
import { ModelNode, ModelEdge } from '../types';
import { getLayerByType } from '../data/layerTypes';

export interface DetectedGroup {
  moduleId: string;
  label: string;
  memberIds: string[];
  repeatCount: number; // >1 for a repeated stack, 1 for a one-off chain/block
  canonicalCount: number; // nodes in ONE instance (what an expanded module shows)
  entryId: string;
  exitId: string;
}

export interface ModularizeResult {
  nodes: ModelNode[];
  edges: ModelEdge[];
  groups: DetectedGroup[];
}

export interface AutoModularizeOptions {
  /**
   * Graph-level metadata, read for `template_instances`. Declared structure beats
   * inferred structure: it records what `instantiate_template` actually built, whereas
   * inferring it back from repeated op signatures guessed 3 groups on a 107-node build
   * carrying 12 declared blocks. Omitted → behaviour is exactly as before.
   */
  metadata?: Record<string, unknown>;
}

/** One entry of `graph.metadata.template_instances`, as written by the resolver. */
interface DeclaredInstance {
  instance_id?: unknown;
  template_id?: unknown;
  entry_node_id?: unknown;
  exit_node_id?: unknown;
}

const declaredInstances = (
  metadata: Record<string, unknown> | undefined,
): DeclaredInstance[] => {
  const raw = metadata?.template_instances;
  if (!Array.isArray(raw)) return [];
  return raw.filter((entry): entry is DeclaredInstance => !!entry && typeof entry === 'object');
};

/** Repetition signature: the op identity. Two nodes are "the same kind of step"
 *  if they share this, regardless of display labels. */
function opSig(n: ModelNode): string {
  return n.data.layerType?.id || n.data.layerType?.name || (n.type as string) || '?';
}

const isWrappable = (n: ModelNode): boolean => n.type === 'layerNode';

/** Ops that make a good human-facing block name when present in a block. */
const HEADLINE_PRIORITY = [
  'multiheadattention', 'attention', 'lstm', 'gru', 'rnn',
  'conv2d', 'conv1d', 'convtranspose2d', 'embedding',
  'sinusoidal_pos_embedding', 'rotary_embedding', 'grn',
  'linear', 'layernorm', 'batchnorm2d',
];

/** Derive a readable block label from its members. Prefer a name prefix the
 *  template already encodes (e.g. "DiT 0: ..." -> "DiT", "ConvNeXt-V2 0: ..."
 *  -> "ConvNeXt-V2"); else fall back to the most distinctive op's display name. */
function deriveLabel(members: ModelNode[]): string {
  const firstName = String(members[0]?.data?.parameters?.name || '');
  // "DiT 0: ..." -> "DiT", "ConvNeXt-V2 0: ..." -> "ConvNeXt-V2". Require
  // whitespace before the index digit so an embedded version number ("V2")
  // isn't mistaken for the instance index.
  const m = firstName.match(/^([A-Za-z][\w.-]*?)\s+\d/);
  if (m && m[1].length >= 2) return m[1];

  let best: ModelNode | undefined;
  let bestRank = Infinity;
  for (const n of members) {
    const rank = HEADLINE_PRIORITY.indexOf(opSig(n));
    if (rank !== -1 && rank < bestRank) {
      bestRank = rank;
      best = n;
    }
  }
  const head = best || members[0];
  const reg = getLayerByType(opSig(head));
  return reg?.displayName || head?.data?.parameters?.name || opSig(head) || 'Block';
}

/** Topologically order node ids (Kahn). Returns null if the graph has a cycle. */
function topoOrder(
  ids: string[],
  succ: Map<string, string[]>,
  indeg: Map<string, number>,
): string[] | null {
  const deg = new Map(ids.map((id) => [id, indeg.get(id) || 0] as const));
  const queue = ids.filter((id) => (deg.get(id) || 0) === 0);
  const order: string[] = [];
  while (queue.length) {
    const id = queue.shift()!;
    order.push(id);
    for (const t of succ.get(id) || []) {
      const d = (deg.get(t) || 0) - 1;
      deg.set(t, d);
      if (d === 0) queue.push(t);
    }
  }
  return order.length === ids.length ? order : null;
}

/** Longest path (by node count) through the DAG, as an ordered id list. This is
 *  the "backbone" we run period detection over — repeated blocks chained in
 *  series all land on it. */
function longestPath(
  ids: string[],
  succ: Map<string, string[]>,
  pred: Map<string, string[]>,
  indeg: Map<string, number>,
): string[] {
  const order = topoOrder(ids, succ, indeg);
  if (!order) return [];
  const dist = new Map<string, number>();
  const from = new Map<string, string | undefined>();
  for (const id of order) {
    let bestD = 0;
    let bestP: string | undefined;
    for (const p of pred.get(id) || []) {
      const d = (dist.get(p) || 0) + 1;
      if (d > bestD) {
        bestD = d;
        bestP = p;
      }
    }
    dist.set(id, bestD);
    from.set(id, bestP);
  }
  let end: string | undefined;
  let max = -1;
  for (const id of ids) {
    if ((dist.get(id) || 0) > max) {
      max = dist.get(id) || 0;
      end = id;
    }
  }
  const path: string[] = [];
  let cur = end;
  while (cur) {
    path.push(cur);
    cur = from.get(cur);
  }
  return path.reverse();
}

interface Run {
  start: number;
  period: number;
  count: number;
}

// Merge/residual ops — a repeated block ends on one of these (the skip rejoins),
// it does not start on one. Used to fix the phase of a detected repeat.
const MERGE_SIGS = new Set(['add', 'cat', 'concat', 'mul', 'sum', 'stack', 'sub', 'prod']);

function isRepeatAt(sigs: string[], start: number, period: number, count: number): boolean {
  if (start < 0 || start + period * count > sigs.length) return false;
  for (let r = 1; r < count; r++) {
    for (let k = 0; k < period; k++) {
      if (sigs[start + k] !== sigs[start + r * period + k]) return false;
    }
  }
  return true;
}

/** Period detection finds the repeating *region* but its phase is ambiguous: a
 *  block whose last op shares a signature with a preceding node (e.g. a residual
 *  `add`) can be detected one step early, swallowing that node. Slide the start
 *  forward within one period to the earliest phase that does NOT begin on a
 *  merge op, keeping the same repeat count. Cosmetic only (payload is unaffected
 *  either way) — it makes block boundaries and labels line up with intent. */
function refinePhase(sigs: string[], run: Run): Run {
  const score = (s: number): number => (MERGE_SIGS.has(sigs[s]) ? 1 : 0);
  let bestStart = run.start;
  let bestScore = score(run.start);
  for (let off = 1; off < run.period && bestScore > 0; off++) {
    const s = run.start + off;
    if (!isRepeatAt(sigs, s, run.period, run.count)) continue;
    if (score(s) < bestScore) {
      bestStart = s;
      bestScore = score(s);
    }
  }
  return { start: bestStart, period: run.period, count: run.count };
}

/** Find maximal consecutive repeats in a signature sequence. Greedy left to
 *  right: at each position pick the repeat covering the most cells (smallest
 *  period on ties → most granular blocks). Requires count≥2 and span≥4. */
function findRepeats(sigs: string[]): Run[] {
  const runs: Run[] = [];
  let i = 0;
  while (i < sigs.length) {
    let best: Run | null = null;
    const maxPeriod = Math.floor((sigs.length - i) / 2);
    for (let p = 1; p <= maxPeriod; p++) {
      let count = 1;
      while (i + (count + 1) * p <= sigs.length) {
        let matches = true;
        for (let k = 0; k < p; k++) {
          if (sigs[i + count * p + k] !== sigs[i + k]) {
            matches = false;
            break;
          }
        }
        if (!matches) break;
        count++;
      }
      if (count >= 2) {
        const span = count * p;
        const bestSpan = best ? best.count * best.period : 0;
        if (span > bestSpan || (span === bestSpan && best && p < best.period)) {
          best = { start: i, period: p, count };
        }
      }
    }
    if (best && best.count * best.period >= 4) {
      runs.push(best);
      i += best.count * best.period;
    } else {
      i += 1;
    }
  }
  return runs;
}

let moduleSeq = 0;

function makeModuleNode(moduleId: string, label: string, repeatCount: number, memberCount: number, entry: string, exit: string): ModelNode {
  return {
    id: moduleId,
    type: 'moduleNode',
    position: { x: 0, y: 0 },
    data: {
      layerType: {
        id: 'module',
        name: 'Module',
        displayName: label,
        description: 'Auto-detected block',
        icon: 'Package',
        category: 'module',
        parameters: [],
      },
      parameters: { name: label },
      isGroup: true,
      expanded: false,
      moduleInstanceId: moduleId,
      moduleTemplateId: label,
      moduleEntryNodeId: entry,
      moduleExitNodeId: exit,
      moduleLayerCount: memberCount,
      moduleStatus: 'ok',
      moduleLayoutState: 'deterministic',
      repeatCount: repeatCount > 1 ? repeatCount : 0,
      repeatBlockLabel: repeatCount > 1 ? label : undefined,
    },
  } as ModelNode;
}

/**
 * Detect repeated stacks + linear chains in a flat graph and wrap them in
 * collapsed module nodes. No-op (returns input unchanged) if the graph already
 * has modules/parented children or nothing groupable is found.
 */
export function autoModularize(
  nodes: ModelNode[],
  edges: ModelEdge[],
  options: AutoModularizeOptions = {},
): ModularizeResult {
  const noop = (): ModularizeResult => ({ nodes, edges, groups: [] });

  // Already modular → leave it (don't clobber a hierarchical template).
  if (nodes.some((n) => n.type === 'moduleNode' || n.data?.parentId)) return noop();

  const realEdges = edges.filter((e) => !e.data?.synthetic);
  const ids = nodes.map((n) => n.id);
  const idSet = new Set(ids);
  const byId = new Map(nodes.map((n) => [n.id, n] as const));

  const succ = new Map<string, string[]>();
  const pred = new Map<string, string[]>();
  const indeg = new Map<string, number>();
  const outdeg = new Map<string, number>();
  for (const id of ids) {
    succ.set(id, []);
    pred.set(id, []);
    indeg.set(id, 0);
    outdeg.set(id, 0);
  }
  for (const e of realEdges) {
    if (!idSet.has(e.source) || !idSet.has(e.target)) continue;
    succ.get(e.source)!.push(e.target);
    pred.get(e.target)!.push(e.source);
    outdeg.set(e.source, (outdeg.get(e.source) || 0) + 1);
    indeg.set(e.target, (indeg.get(e.target) || 0) + 1);
  }

  const groups: DetectedGroup[] = [];
  const grouped = new Set<string>();
  const parentOf = new Map<string, string>();
  // Members of a repeated stack beyond the FIRST instance — kept for codegen/
  // verify but hidden on the canvas so an expanded ×N module shows one block.
  const repeatHidden = new Set<string>();

  // ── Pass 0: blocks the backend already DECLARED ─────────────────────────
  // `template_instances` records what instantiate_template built, so it is real
  // structure rather than a guess. Seeding the same maps the later passes already
  // consult means they simply skip these nodes — no other line of this function
  // changes. Members are identified by the `<instance_id>__` prefix the resolver
  // mints, and the longest instance_id is matched first so a prefix that is itself a
  // prefix of another cannot steal its nodes.
  const instances = declaredInstances(options.metadata)
    .map((entry) => ({
      instanceId: typeof entry.instance_id === 'string' ? entry.instance_id : '',
      templateId: typeof entry.template_id === 'string' ? entry.template_id : '',
      entryId: typeof entry.entry_node_id === 'string' ? entry.entry_node_id : '',
      exitId: typeof entry.exit_node_id === 'string' ? entry.exit_node_id : '',
    }))
    .filter((entry) => entry.instanceId)
    .sort((a, b) => b.instanceId.length - a.instanceId.length);

  for (const instance of instances) {
    const prefix = `${instance.instanceId}__`;
    const members = ids.filter(
      (id) =>
        !grouped.has(id) &&
        (id === instance.instanceId || id.startsWith(prefix)) &&
        isWrappable(byId.get(id)!),
    );
    // A single node is not a block, and an instance whose members were mostly deleted
    // by later edits legitimately stops rendering as a card.
    if (members.length < 2) continue;
    const moduleId = `auto-mod-${instance.instanceId}`;
    const entryId = members.includes(instance.entryId) ? instance.entryId : members[0];
    const exitId = members.includes(instance.exitId)
      ? instance.exitId
      : members[members.length - 1];
    members.forEach((id) => {
      grouped.add(id);
      parentOf.set(id, moduleId);
    });
    groups.push({
      moduleId,
      label: instance.templateId || deriveLabel(members.map((id) => byId.get(id)!)),
      memberIds: members,
      repeatCount: 1,
      canonicalCount: members.length,
      entryId,
      exitId,
    });
  }

  // ── Pass 1: repeated stacks (×N) along the backbone ────────────────────
  const spine = longestPath(ids, succ, pred, indeg).filter((id) => isWrappable(byId.get(id)!));
  const spineSet = new Set(spine);
  const spineSigs = spine.map((id) => opSig(byId.get(id)!));
  for (const rawRun of findRepeats(spineSigs)) {
    const run = refinePhase(spineSigs, rawRun);
    const runMembers = spine.slice(run.start, run.start + run.period * run.count);
    if (runMembers.some((id) => grouped.has(id))) continue;
    const moduleId = `auto-mod-${moduleSeq++}`;
    const label = deriveLabel(runMembers.slice(0, run.period).map((id) => byId.get(id)!));

    // Assign each node to an instance index. Spine members are the backbone;
    // off-spine side branches (e.g. a block's AdaLayerNorm-Zero modulation
    // Linear) are pulled into instance j when ALL their successors are in j.
    const instOf = new Map<string, number>();
    runMembers.forEach((id, idx) => instOf.set(id, Math.floor(idx / run.period)));
    let changed = true;
    while (changed) {
      changed = false;
      for (const id of ids) {
        if (instOf.has(id) || grouped.has(id) || spineSet.has(id) || !isWrappable(byId.get(id)!)) continue;
        const succs = succ.get(id) || [];
        if (succs.length === 0) continue;
        const insts = new Set(succs.map((s) => instOf.get(s)));
        if (insts.size === 1 && !insts.has(undefined)) {
          instOf.set(id, succs.map((s) => instOf.get(s)!)[0]);
          changed = true;
        }
      }
    }

    const canonicalIds: string[] = [];
    const repeatIds: string[] = [];
    instOf.forEach((inst, id) => {
      grouped.add(id);
      parentOf.set(id, moduleId);
      if (inst === 0) canonicalIds.push(id);
      else { repeatIds.push(id); repeatHidden.add(id); }
    });
    groups.push({
      moduleId, label,
      memberIds: [...canonicalIds, ...repeatIds],
      repeatCount: run.count,
      canonicalCount: canonicalIds.length,
      entryId: runMembers[0],
      exitId: runMembers[run.period - 1],
    });
  }

  // ── Pass 2: fold remaining strict 1-in/1-out pipelines (length ≥ 3) ─────
  const chainable = (id: string): boolean => {
    // An edge may name a node that is not in `nodes` (a graph mid-edit, or a document
    // whose nodes were pruned). The non-null assertion here threw
    // "Cannot read properties of undefined" out of isWrappable and took the whole
    // canvas projection down with it.
    const node = byId.get(id);
    if (!node) return false;
    return (
      !grouped.has(id) &&
      isWrappable(node) &&
      (indeg.get(id) || 0) === 1 &&
      (outdeg.get(id) || 0) === 1
    );
  };

  const next = new Map<string, string>();
  const prev = new Map<string, string>();
  for (const e of realEdges) {
    if (chainable(e.source) && chainable(e.target)) {
      next.set(e.source, e.target);
      prev.set(e.target, e.source);
    }
  }
  for (const id of ids) {
    if (!chainable(id) || prev.has(id) || grouped.has(id)) continue;
    const chain: string[] = [];
    let cur: string | undefined = id;
    while (cur && chainable(cur) && !grouped.has(cur)) {
      chain.push(cur);
      cur = next.get(cur);
    }
    if (chain.length < 3) continue;
    const moduleId = `auto-mod-${moduleSeq++}`;
    const label = deriveLabel(chain.map((c) => byId.get(c)!));
    chain.forEach((c) => {
      grouped.add(c);
      parentOf.set(c, moduleId);
    });
    groups.push({
      moduleId, label, memberIds: chain, repeatCount: 1, canonicalCount: chain.length,
      entryId: chain[0], exitId: chain[chain.length - 1],
    });
  }

  if (groups.length === 0) return noop();

  // ── Build the new logical graph: wrappers + parentId on children ────────
  const moduleNodes = groups.map((g) =>
    makeModuleNode(
      g.moduleId,
      g.label,
      g.repeatCount,
      g.canonicalCount, // header shows ONE block's layer count (×N badge conveys repeats)
      g.entryId,
      g.exitId,
    ),
  );
  const childNodes = nodes.map((n) => {
    const pid = parentOf.get(n.id);
    if (!pid) return n;
    const data: any = { ...n.data, parentId: pid };
    if (repeatHidden.has(n.id)) data.repeatHidden = true;
    return { ...n, data } as ModelNode;
  });

  return { nodes: [...moduleNodes, ...childNodes], edges, groups };
}
