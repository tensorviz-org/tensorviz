"""graph_modularizer — turn a FLAT graph into a HierarchicalGraph whose repeated
blocks are factored into a single reusable module instantiated ×N.

This is the backend sibling of the frontend `autoModularize.ts`. It exists for
MODULAR CODE GENERATION only: the export path runs a flat graph (e.g. the
267-node F5-TTS) through here to get a hierarchy where the 22 DiT blocks become
one `DiTBlock` class instantiated 22× as an `nn.ModuleList`, instead of 220
inlined layers.

It is deliberately confined to the export/codegen path — the verify/validate
gates keep running on the original flat graph, so this never has to satisfy the
single-entry assumptions baked into the verification/shape-inference flatteners.

Detection mirrors the frontend exactly (longest-path period detection + phase
refinement). On top of that it extracts a *canonical* block (instance 0), works
out which external inputs are the chaining/primary input vs shared conditioning
inputs (e.g. the F5-TTS timestep embedding `t_emb` that fans into every block),
and rewires the root graph by folding every instance of a repeated stack into a
single module node.
"""
from typing import Any, Dict, List, Optional, Tuple

from models.graph_models import (
    Edge,
    HierarchicalGraph,
    ModuleDefinition,
    Node,
)

MERGE_SIGS = {"add", "cat", "concat", "mul", "sum", "stack", "sub", "prod"}
MIN_SPAN = 4  # don't bother grouping runs shorter than this many nodes

HEADLINE_PRIORITY = [
    "multiheadattention", "attention", "lstm", "gru", "rnn",
    "conv2d", "conv1d", "convtranspose2d", "embedding",
    "sinusoidal_pos_embedding", "rotary_embedding", "grn",
    "linear", "layernorm", "batchnorm2d",
]


# ── dict accessors (the flat payload is plain dicts) ──────────────────────────
def _nid(n: Dict) -> str:
    return n["id"]


def _ntype(n: Dict) -> str:
    return n.get("type", "")


def _op(n: Dict) -> str:
    lt = (n.get("data") or {}).get("layerType") or {}
    return lt.get("id") or lt.get("name") or _ntype(n) or "?"


def _name(n: Dict) -> str:
    return str(((n.get("data") or {}).get("parameters") or {}).get("name") or "")


def _sig(n: Dict) -> str:
    """Repeat-detection signature: op **plus its constructor params**.

    Op alone is not enough. Folding keeps only instance 0
    (``canonical_spine = members[0:period]``) and emits
    ``nn.ModuleList([Block() for _ in range(N)])``, so two spans that merely share
    op types get collapsed onto the first span's parameters. A 3-layer MLP
    ``Linear(16,32) ReLU Linear(32,32) ReLU`` is two op-identical spans of 2, hits
    ``MIN_SPAN``, and generated code that reused ``Linear(16,32)`` twice raised
    ``mat1 and mat2 shapes cannot be multiplied (1x32 and 16x32)``. That reached
    the user-facing ``POST /api/models/generate-code`` (``server_v2.py:315``), not
    just the AI path.

    Display-only keys are excluded so genuinely identical blocks still fold: a
    per-instance ``name`` ("DiTBlock 1"/"DiTBlock 2") must not split an F5-TTS
    DiTBlock x22 run. This mirrors ``code_generation.py``, which skips ``name``
    and ``_``-prefixed keys as non-constructor.
    """
    params = (n.get("data") or {}).get("parameters") or {}
    material = sorted(
        (str(k), repr(v))
        for k, v in params.items()
        if k != "name" and not str(k).startswith("_")
    )
    return _op(n) + "|" + ";".join(f"{k}={v}" for k, v in material)


def _is_wrappable(n: Dict) -> bool:
    return _ntype(n) == "layerNode"


# ── detection (faithful port of autoModularize.ts) ────────────────────────────
def _topo_order(ids: List[str], succ: Dict[str, List[str]], indeg: Dict[str, int]) -> Optional[List[str]]:
    deg = {i: indeg.get(i, 0) for i in ids}
    queue = [i for i in ids if deg[i] == 0]
    order: List[str] = []
    while queue:
        i = queue.pop(0)
        order.append(i)
        for t in succ.get(i, []):
            deg[t] -= 1
            if deg[t] == 0:
                queue.append(t)
    return order if len(order) == len(ids) else None


def _longest_path(ids, succ, pred, indeg) -> List[str]:
    order = _topo_order(ids, succ, indeg)
    if not order:
        return []
    dist: Dict[str, int] = {}
    came: Dict[str, Optional[str]] = {}
    for i in order:
        best_d, best_p = 0, None
        for p in pred.get(i, []):
            d = dist.get(p, 0) + 1
            if d > best_d:
                best_d, best_p = d, p
        dist[i] = best_d
        came[i] = best_p
    end, mx = None, -1
    for i in ids:
        if dist.get(i, 0) > mx:
            mx, end = dist.get(i, 0), i
    path: List[str] = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = came.get(cur)
    path.reverse()
    return path


def _is_repeat_at(sigs: List[str], start: int, period: int, count: int) -> bool:
    if start < 0 or start + period * count > len(sigs):
        return False
    for r in range(1, count):
        for k in range(period):
            if sigs[start + k] != sigs[start + r * period + k]:
                return False
    return True


def _find_repeats(sigs: List[str]) -> List[Tuple[int, int, int]]:
    """Return (start, period, count) runs. Greedy left-to-right, max span, then
    smallest period on ties; requires count>=2 and span>=MIN_SPAN."""
    runs: List[Tuple[int, int, int]] = []
    i = 0
    n = len(sigs)
    while i < n:
        best = None  # (period, count)
        best_span = 0
        max_period = (n - i) // 2
        for p in range(1, max_period + 1):
            count = 1
            while i + (count + 1) * p <= n:
                ok = all(sigs[i + count * p + k] == sigs[i + k] for k in range(p))
                if not ok:
                    break
                count += 1
            if count >= 2:
                span = count * p
                if span > best_span or (span == best_span and best and p < best[0]):
                    best, best_span = (p, count), span
        if best and best[0] * best[1] >= MIN_SPAN:
            runs.append((i, best[0], best[1]))
            i += best[0] * best[1]
        else:
            i += 1
    return runs


def _refine_phase(
    sigs: List[str], ops: List[str], start: int, period: int, count: int
) -> int:
    """Slide the start within one period to the earliest phase that does NOT
    begin on a merge op (a repeated block ends, not starts, on a residual add).

    ``sigs`` are the params-aware detection signatures and ``ops`` the bare op
    names, kept separate on purpose: the merge test below matches against
    ``MERGE_SIGS``, which holds op names, while ``_is_repeat_at`` must compare the
    same signatures detection used or it could accept a phase that is not actually
    a repeat.
    """
    def score(s: int) -> int:
        return 1 if ops[s] in MERGE_SIGS else 0
    best_start, best_score = start, score(start)
    off = 1
    while off < period and best_score > 0:
        s = start + off
        if _is_repeat_at(sigs, s, period, count) and score(s) < best_score:
            best_start, best_score = s, score(s)
        off += 1
    return best_start


def _derive_label(members: List[Dict]) -> str:
    import re
    first = _name(members[0]) if members else ""
    m = re.match(r"^([A-Za-z][\w.-]*?)\s+\d", first)
    if m and len(m.group(1)) >= 2:
        return m.group(1)
    best, best_rank = members[0] if members else None, 1e9
    for n in members:
        try:
            r = HEADLINE_PRIORITY.index(_op(n))
        except ValueError:
            continue
        if r < best_rank:
            best, best_rank = n, r
    return (_name(best) or _op(best) or "Block") if best else "Block"


# ── construction ──────────────────────────────────────────────────────────────
def _module_id_from_label(label: str, taken: set) -> str:
    import re
    base = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower() or "block"
    base = f"{base}_block"
    mid, i = base, 1
    while mid in taken:
        i += 1
        mid = f"{base}_{i}"
    taken.add(mid)
    return mid


def _make_module_node(module_id: str, label: str, repeat_count: int) -> Dict:
    return {
        "id": module_id,
        "type": "moduleNode",
        "position": {"x": 0.0, "y": 0.0},
        "data": {
            "layerType": {"id": "module", "name": "Module", "parameters": []},
            "parameters": {"name": label, "repeat_count": repeat_count, "repeat_block_label": label},
        },
    }


def modularize_for_codegen(flat: Dict[str, Any]) -> Optional[HierarchicalGraph]:
    """Convert a flat {nodes, edges} graph into a HierarchicalGraph whose
    repeated blocks are factored into reusable ×N modules. Returns None when no
    repeated block is found (caller falls back to flat codegen)."""
    nodes: List[Dict] = list(flat.get("nodes") or [])
    edges: List[Dict] = list(flat.get("edges") or [])
    if not nodes:
        return None
    # The canvas may send a FRONTEND display-grouped graph (moduleNode wrappers +
    # parentId children + synthetic display edges). Flatten it back to the raw
    # layer graph and re-detect repeats from there — this is what makes the
    # "Export Code"/"Export Package" buttons produce the modular ×N form instead
    # of the literal frontend groups. (A real backend HierarchicalGraph arrives
    # as {root_nodes, modules} with no "nodes" key and never reaches here.)
    if any(_ntype(n) == "moduleNode" for n in nodes) or any((n.get("data") or {}).get("parentId") for n in nodes):
        flat_nodes = []
        for n in nodes:
            if _ntype(n) == "moduleNode":
                continue
            data = {k: v for k, v in (n.get("data") or {}).items() if k != "parentId"}
            flat_nodes.append({**n, "data": data})
        nodes = flat_nodes
        edges = [e for e in edges if not (e.get("data") or {}).get("synthetic")]
        if not nodes:
            return None

    by_id = {_nid(n): n for n in nodes}
    ids = list(by_id.keys())
    succ: Dict[str, List[str]] = {i: [] for i in ids}
    pred: Dict[str, List[str]] = {i: [] for i in ids}
    indeg: Dict[str, int] = {i: 0 for i in ids}
    for e in edges:
        s, t = e.get("source"), e.get("target")
        if s in by_id and t in by_id:
            succ[s].append(t)
            pred[t].append(s)
            indeg[t] += 1

    spine_all = _longest_path(ids, succ, pred, indeg)
    spine = [i for i in spine_all if _is_wrappable(by_id[i])]
    sigs = [_sig(by_id[i]) for i in spine]
    spine_ops = [_op(by_id[i]) for i in spine]

    grouped: Dict[str, str] = {}       # node id -> module id
    module_defs: List[ModuleDefinition] = []
    module_nodes: List[Dict] = []
    taken_ids: set = set()

    for (raw_start, period, count) in _find_repeats(sigs):
        start = _refine_phase(sigs, spine_ops, raw_start, period, count)
        members = spine[start:start + period * count]
        if any(m in grouped for m in members):
            continue
        # Belt and braces: only instance 0 survives folding, so refuse to group
        # unless every instance is materially identical. _sig already guarantees
        # this, but the cost of being wrong is silently shipping runnable code for
        # a different architecture, which is the worst failure this module has.
        if any(
            [sigs[start + k] for k in range(period)]
            != [sigs[start + r * period + k] for k in range(period)]
            for r in range(1, count)
        ):
            continue
        inst_sets = [set(members[j * period:(j + 1) * period]) for j in range(count)]
        # Off-spine expansion: a block-internal node that branches off the spine
        # (e.g. F5-TTS's per-block AdaLayerNorm-Zero modulation Linear, fed by the
        # shared t_emb) isn't on the longest path. Pull such a node into instance j
        # when ALL its successors live in instance j. Iterate to a fixpoint.
        spine_set = set(spine)
        inst_of: Dict[str, int] = {}
        for j, s in enumerate(inst_sets):
            for nid in s:
                inst_of[nid] = j
        changed = True
        while changed:
            changed = False
            for nid in ids:
                # Only OFF-spine side branches are block-internal; on-spine nodes
                # are the backbone (the primary chain) and must stay out.
                if nid in inst_of or nid in grouped or nid in spine_set or not _is_wrappable(by_id[nid]):
                    continue
                tgt = {inst_of.get(t) for t in succ.get(nid, [])}
                if len(tgt) == 1 and None not in tgt and succ.get(nid):
                    j = next(iter(tgt))
                    inst_of[nid] = j
                    inst_sets[j].add(nid)
                    changed = True
        instances = [list(s) for s in inst_sets]
        canonical_spine = members[0:period]
        canonical = canonical_spine + [n for n in inst_sets[0] if n not in set(canonical_spine)]
        canon_set = set(canonical)
        all_instance_set = set().union(*inst_sets)

        # A folded run emits exactly ONE tensor — the last instance's exit — because
        # every member collapses onto a single module node and the root rewiring
        # dedupes `member -> external` down to one `module -> external` edge. So if
        # any OTHER member also feeds a node outside the run, folding drops that
        # edge. The four Conv2d pyramid levels of an FPN head each reach the same
        # concat: the exported `torch.cat` received one tensor where the verified
        # graph had four, and the head's Conv2d then saw 3 channels instead of 12.
        # Flat codegen reproduces such a graph correctly, so decline to fold.
        run_exit = members[-1]
        if any(
            target not in all_instance_set
            for member in all_instance_set
            if member != run_exit
            for target in succ.get(member, [])
        ):
            continue
        label = _derive_label([by_id[i] for i in canonical_spine])
        module_id = _module_id_from_label(label, taken_ids)

        internal_edges = [e for e in edges if e.get("source") in canon_set and e.get("target") in canon_set]
        boundary_in = [e for e in edges if e.get("target") in canon_set and e.get("source") not in canon_set]

        # Classify each external source: feeds >1 instance of this run ->
        # conditioning (e.g. t_emb); feeds only the canonical instance -> primary.
        def feeds_instances(src: str) -> int:
            hit = 0
            for inst in instances:
                inst_set = set(inst)
                if any(e.get("source") == src and e.get("target") in inst_set for e in edges):
                    hit += 1
            return hit

        cond_sources, primary_candidates = [], []
        seen_src: set = set()
        for e in boundary_in:
            s = e.get("source")
            if s in seen_src:
                continue
            seen_src.add(s)
            (cond_sources if feeds_instances(s) > 1 else primary_candidates).append(s)
        if not primary_candidates:
            continue  # no chaining input → not a real stack; skip

        # primary = the spine predecessor of the canonical entry when available.
        spine_pred = spine[start - 1] if start > 0 else None
        primary_source = spine_pred if spine_pred in primary_candidates else primary_candidates[0]
        # any extra single-instance sources beyond the primary are treated as
        # conditioning too (keeps the block arity faithful).
        cond_sources += [s for s in primary_candidates if s != primary_source]

        input_node_ids = [primary_source] + cond_sources
        input_arg_names = ["x"] + [f"cond{i}" for i in range(len(cond_sources))]
        output_node_id = canonical_spine[-1]  # block exit is the last SPINE node

        module_defs.append(ModuleDefinition(
            module_id=module_id,
            parent_id=None,
            internal_nodes=[Node(**by_id[i]) for i in canonical],
            internal_edges=[Edge(**e) for e in internal_edges],
            input_node_id=canonical_spine[0],
            output_node_id=output_node_id,
            is_modular_block=True,
            input_node_ids=input_node_ids,
            input_arg_names=input_arg_names,
            conditioning_source_ids=cond_sources,
            boundary_edges=[Edge(**e) for e in boundary_in],
            parameters={"name": label, "repeat_count": count, "repeat_block_label": label},
        ))
        module_nodes.append(_make_module_node(module_id, label, count))
        for m in all_instance_set:
            grouped[m] = module_id

    if not module_defs:
        return None

    # ── Rewire root: fold every instance of a stack into its module node ──
    def fold(i: str) -> str:
        return grouped.get(i, i)

    root_nodes = [n for n in nodes if _nid(n) not in grouped] + module_nodes
    root_edges_seen: set = set()
    root_edges: List[Dict] = []
    eidx = 0
    for e in edges:
        fs, ft = fold(e.get("source")), fold(e.get("target"))
        if fs == ft:
            continue  # internal to a collapsed stack
        key = (fs, ft)
        if key in root_edges_seen:
            continue
        root_edges_seen.add(key)
        root_edges.append({"id": f"re_{eidx}", "source": fs, "target": ft})
        eidx += 1

    node_hierarchy = {nid: mid for nid, mid in grouped.items()}
    return HierarchicalGraph(
        root_nodes=[Node(**n) for n in root_nodes],
        root_edges=[Edge(**e) for e in root_edges],
        modules={m.module_id: m for m in module_defs},
        node_hierarchy=node_hierarchy,
    )
