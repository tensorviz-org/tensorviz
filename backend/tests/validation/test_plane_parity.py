"""Tier-6 plane-parity corpus (docs/tensorviz-plan.md §5.5, §9 Tier 6, §11.5).

The keystone — the moat's definition, made executable. Over a FIXED corpus of graphs
the ANALYTIC plane (``canonical_validation``) and the TORCH plane (``verify_model``,
real forward) must agree on the *verdict*: both valid, or both invalid — ideally
blaming the same node.

Corpus (deterministic, no randomness so the set is reproducible run to run):
  (a) VALID — templates expected valid in BOTH planes:
        * the 6 headline gallery models (ViT/BERT/GPT-2/T5/ResNet/GAN), and
        * a sorted-registry stride sample (same construction as test_export_roundtrip).
  (b) INVALID — hand-crafted graphs each carrying ONE deliberate single-node shape
      error (wrong Linear in_features, Conv2d in_channels mismatch, LayerNorm
      normalized_shape, mismatched concat/add branches, ...). Each records the node
      the error must be blamed on; both planes must flag it invalid AND (secondary)
      blame that same node.

PRIMARY assertion: analytic verdict == torch verdict.
SECONDARY (only where both planes agree the graph is invalid): they blame the same
node. This is the softer check — collected and reported, and any divergence would be
documented in NODE_BLAME_GAPS (empty today: all 12 crafted graphs blame-match).

Verdict disagreements are documented in KNOWN_PARITY_GAPS with a one-line reason and
marked xfail(strict=False); the suite is GREEN today and each gap auto-flips to XPASS
the moment the underlying plane divergence is closed. Nothing is silently skipped.

The two planes are fed the SAME logical graph through each plane's real product entry
point: the analytic plane validates the canonical ``ArchitectureGraph`` (the
agent-ops / validate-route path); the torch plane runs ``verify_model`` on the flat/
hierarchical canvas payload (the /api/models/verify path).

Run: KMP_DUPLICATE_LIB_OK=TRUE <viz-python> -m pytest tests/test_plane_parity.py -q
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # must precede torch import
os.environ.setdefault("AUTH_DISABLED", "true")



import pytest

from api.routers.validate import ValidationRequest, _build_canonical_graph
from models.architecture_models import ArchitectureEdge, ArchitectureGraph, ArchitectureNode
from scripts.run_template_validation_sweep import _build_graph_for_sequence
from services.canonical_template_registry import CanonicalTemplateRegistry
from services.canonical_validation import CanonicalValidationService
from services.template_service import TemplateService
from services.verification_service import (
    _arch_node_to_flat,
    _expand_module_templates,
    _flatten_for_torch,
    _reconstruct_hierarchy,
    verify_model,
)

_REGISTRY = CanonicalTemplateRegistry()
_VALIDATOR = CanonicalValidationService()

# The 6 headline gallery models (docs/tensorviz-plan.md §5.4), instantiated as full hierarchical graphs.
GALLERY = ["vit_base", "bert_base", "gpt2_small", "t5_base", "resnet50", "biggan"]


def _registry_sample(n_target: int = 20) -> list:
    """Deterministic stride sample of the canonical registry: sort every contract id,
    take every ``step``-th. No randomness — the same ids come out every run so
    KNOWN_PARITY_GAPS stays stable."""
    ids = sorted(c.template_id for c in _REGISTRY.all_contracts())
    step = max(1, len(ids) // n_target)
    return ids[::step]


SAMPLE = _registry_sample()


# ── Known verdict-parity gaps (all analytic-valid / torch-invalid): the analytic
# plane treats these as valid but the real torch forward can't (custom op with no
# torch.nn class, a multi-input block fed into a single-input op, or a genuine
# analytic/torch shape drift). xfail(strict=False) so each auto-flips to XPASS when
# the plane divergence is closed; test_parity_coverage fails on a stale key.
#
# NOTE ON COVERAGE: this gate can only document ids that fall in _corpus() (gallery +
# the stride sample above) — test_parity_coverage fails on any KNOWN_PARITY_GAPS key
# not in the corpus. Two divergences here are just the stride-sampled REPRESENTATIVES
# of a larger family that diverges in full:
#   * bidirectional-RNN feature-doubling — 13 blocks diverge, only the 2 below
#     (lstm_8layer_bi_block, rnn_8layer_bi_block) land in the sample.
# The COMPLETE family (and the count of what escaped this sample) is enumerated and
# gated in tests/test_qa_plane_parity_sweep.py, which parametrizes every block.
#
# The FPN-head family used to be the second such representative pair
# (fcos_fpn_l6_head_block, yolo_fpn_l4_head_block). All 16 blocks now agree on both
# planes: the divergence was never "a multi-input block fed into a single-input op" —
# the fan-in was internal to the template, which had no merge node at all, and
# template_expansion_pack now concatenates the pyramid levels before the heads. Both
# ids are pruned from this dict; test_qa_plane_parity_sweep.py asserts the whole family
# forwards on both planes rather than deleting the coverage. ─────────────────────────
KNOWN_PARITY_GAPS = {
    "deepfm_512_block": "torch verify can't build custom op 'factorization_machine' (no torch.nn class); analytic treats it as a shape-preserving passthrough",
    "gnn_lstm_1hop_block": "torch verify can't build custom op 'neighbor_sampler' (no torch.nn class); analytic treats it as a shape-preserving passthrough",
    "gnn_sum_1hop_block": "torch verify can't build custom op 'neighbor_sampler' (no torch.nn class); analytic treats it as a shape-preserving passthrough",
}

# ── Crafted-invalid graphs where the two planes DO agree invalid but might blame
# different nodes. Empty today — all 12 blame-match. A future divergence goes here
# with a reason; test_parity_coverage fails on a stale key. ──────────────────────────
NODE_BLAME_GAPS: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Canonical-graph construction helpers
# ─────────────────────────────────────────────────────────────────────────────
def _node(node_id: str, kind: str, op: str, **params) -> ArchitectureNode:
    return ArchitectureNode(
        node_id=node_id,
        kind=kind,  # type: ignore[arg-type]
        op=op,
        params=params,
        in_ports=[] if kind == "input" else ["x"],
        out_ports=[] if kind == "output" else ["y"],
    )


def _edge(src: str, tgt: str) -> ArchitectureEdge:
    return ArchitectureEdge(
        edge_id=f"{src}->{tgt}", source_node_id=src, source_port="y",
        target_node_id=tgt, target_port="x",
    )


def _graph(graph_id: str, nodes, edges, input_shape) -> ArchitectureGraph:
    return ArchitectureGraph(
        graph_id=graph_id, name=graph_id, revision=0,
        inputs=[{"name": "x", "shape": input_shape}], nodes=nodes, edges=edges,
    )


def _arch_to_payload(graph: ArchitectureGraph) -> dict:
    """Flatten a canonical ArchitectureGraph to the flat canvas payload verify_model
    consumes (id / type / data.layerType / data.parameters), the same conversion the
    Tier-5 export test uses for registry blocks."""
    return {
        "nodes": [_arch_node_to_flat(n) for n in graph.nodes],
        "edges": [
            {"id": e.edge_id, "source": e.source_node_id, "target": e.target_node_id}
            for e in graph.edges
        ],
    }


def _input_shape_from_root(root_nodes) -> list:
    for n in root_nodes:
        if n.type == "inputNode":
            raw = (n.data.parameters or {}).get("shape")
            if isinstance(raw, str) and raw.strip():
                return [int(v) for v in raw.split(",")]
            if isinstance(raw, (list, tuple)) and raw:
                return [int(v) for v in raw]
    return [1, 3, 224, 224]


def _gallery_arch_graph(hg_payload: dict, input_shape) -> ArchitectureGraph:
    """Analytic-plane projection of a hierarchical gallery graph: flatten to leaf
    nodes with the verify path's own flattener (``_flatten_for_torch``), then rebuild
    the canonical ArchitectureGraph exactly as the /validate_shapes route does
    (``_build_canonical_graph``). Both are real product functions, so this mirrors
    validating a flattened canvas graph analytically — and the torch plane flattens
    the same hierarchical payload identically, so both planes see the same leaves."""
    payload = _reconstruct_hierarchy(_expand_module_templates(hg_payload))
    flat = _flatten_for_torch(payload)
    nodes = []
    for n in flat.nodes:
        lt = n.data.layerType
        nodes.append({
            "id": n.id,
            "type": n.type,
            "position": {"x": 0.0, "y": 0.0},
            "data": {
                "layerType": lt.model_dump() if hasattr(lt, "model_dump") else lt,
                "parameters": dict(n.data.parameters or {}),
            },
        })
    edges = [
        {"id": e.id or f"{e.source}->{e.target}", "source": e.source, "target": e.target}
        for e in flat.edges
    ]
    return _build_canonical_graph(
        ValidationRequest(nodes=nodes, edges=edges, input_shape=input_shape)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hand-crafted INVALID corpus — one deliberate single-node shape error each.
# Value: (ArchitectureGraph, input_shape, expected_blamed_node_id).
# ─────────────────────────────────────────────────────────────────────────────
def _crafted_invalid() -> dict:
    out: dict = {}

    # Linear with wrong in_features straight off the input vector.
    out["bad_linear_infeatures"] = (
        _graph("bad_linear_infeatures",
               [_node("input", "input", "Input", shape=[1, 64]),
                _node("lin", "layer", "nn.Linear", in_features=128, out_features=10),
                _node("output", "output", "Output")],
               [_edge("input", "lin"), _edge("lin", "output")], [1, 64]),
        [1, 64], "lin")

    # Conv2d in_channels don't match the 3-channel input.
    out["bad_conv_inchannels"] = (
        _graph("bad_conv_inchannels",
               [_node("input", "input", "Input", shape=[1, 3, 32, 32]),
                _node("conv", "layer", "nn.Conv2d", in_channels=8, out_channels=16, kernel_size=3, padding=1),
                _node("output", "output", "Output")],
               [_edge("input", "conv"), _edge("conv", "output")], [1, 3, 32, 32]),
        [1, 3, 32, 32], "conv")

    # Conv2d fed a rank-2 vector (needs [N, C, H, W]).
    out["bad_conv_rank"] = (
        _graph("bad_conv_rank",
               [_node("input", "input", "Input", shape=[1, 128]),
                _node("conv", "layer", "nn.Conv2d", in_channels=128, out_channels=16, kernel_size=3),
                _node("output", "output", "Output")],
               [_edge("input", "conv"), _edge("conv", "output")], [1, 128]),
        [1, 128], "conv")

    # LayerNorm normalized_shape doesn't match the trailing input dim (256).
    out["bad_layernorm"] = (
        _graph("bad_layernorm",
               [_node("input", "input", "Input", shape=[1, 16, 256]),
                _node("ln", "layer", "nn.LayerNorm", normalized_shape=128),
                _node("output", "output", "Output")],
               [_edge("input", "ln"), _edge("ln", "output")], [1, 16, 256]),
        [1, 16, 256], "ln")

    # BatchNorm2d num_features (8) doesn't match the conv's 16 output channels.
    out["bad_batchnorm"] = (
        _graph("bad_batchnorm",
               [_node("input", "input", "Input", shape=[1, 3, 32, 32]),
                _node("conv", "layer", "nn.Conv2d", in_channels=3, out_channels=16, kernel_size=3, padding=1),
                _node("bn", "layer", "nn.BatchNorm2d", num_features=8),
                _node("output", "output", "Output")],
               [_edge("input", "conv"), _edge("conv", "bn"), _edge("bn", "output")], [1, 3, 32, 32]),
        [1, 3, 32, 32], "bn")

    # Linear in_features wrong after a conv+flatten of known width.
    out["bad_linear_after_flatten"] = (
        _graph("bad_linear_after_flatten",
               [_node("input", "input", "Input", shape=[1, 3, 8, 8]),
                _node("conv", "layer", "nn.Conv2d", in_channels=3, out_channels=4, kernel_size=3, padding=1),
                _node("flat", "layer", "nn.Flatten"),
                _node("lin", "layer", "nn.Linear", in_features=999, out_features=10),
                _node("output", "output", "Output")],
               [_edge("input", "conv"), _edge("conv", "flat"), _edge("flat", "lin"), _edge("lin", "output")],
               [1, 3, 8, 8]),
        [1, 3, 8, 8], "lin")

    # LSTM input_size (128) doesn't match the rank-3 feature dim (256).
    out["bad_rnn_inputsize"] = (
        _graph("bad_rnn_inputsize",
               [_node("input", "input", "Input", shape=[1, 16, 256]),
                _node("rnn", "layer", "nn.LSTM", input_size=128, hidden_size=64, batch_first=True),
                _node("output", "output", "Output")],
               [_edge("input", "rnn"), _edge("rnn", "output")], [1, 16, 256]),
        [1, 16, 256], "rnn")

    # MaxPool2d fed a rank-2 vector.
    out["bad_maxpool_rank"] = (
        _graph("bad_maxpool_rank",
               [_node("input", "input", "Input", shape=[1, 128]),
                _node("mp", "layer", "nn.MaxPool2d", kernel_size=2),
                _node("output", "output", "Output")],
               [_edge("input", "mp"), _edge("mp", "output")], [1, 128]),
        [1, 128], "mp")

    # GroupNorm num_groups (5) doesn't divide num_channels (16).
    out["bad_groupnorm"] = (
        _graph("bad_groupnorm",
               [_node("input", "input", "Input", shape=[1, 16, 8, 8]),
                _node("gn", "layer", "nn.GroupNorm", num_groups=5, num_channels=16),
                _node("output", "output", "Output")],
               [_edge("input", "gn"), _edge("gn", "output")], [1, 16, 8, 8]),
        [1, 16, 8, 8], "gn")

    # torch.add of two branches with mismatched channel counts (16 vs 8).
    out["bad_add_mismatch"] = (
        _graph("bad_add_mismatch",
               [_node("input", "input", "Input", shape=[1, 3, 16, 16]),
                _node("cA", "layer", "nn.Conv2d", in_channels=3, out_channels=16, kernel_size=3, padding=1),
                _node("cB", "layer", "nn.Conv2d", in_channels=3, out_channels=8, kernel_size=3, padding=1),
                _node("add", "op", "torch.add"),
                _node("output", "output", "Output")],
               [_edge("input", "cA"), _edge("input", "cB"), _edge("cA", "add"), _edge("cB", "add"), _edge("add", "output")],
               [1, 3, 16, 16]),
        [1, 3, 16, 16], "add")

    # torch.cat of two branches whose non-concat spatial dims differ (k=3 vs k=5 pad=1).
    out["bad_cat_mismatch"] = (
        _graph("bad_cat_mismatch",
               [_node("input", "input", "Input", shape=[1, 3, 16, 16]),
                _node("cA", "layer", "nn.Conv2d", in_channels=3, out_channels=16, kernel_size=3, padding=1),
                _node("cB", "layer", "nn.Conv2d", in_channels=3, out_channels=8, kernel_size=5, padding=1),
                _node("cat", "op", "torch.cat", dim=1),
                _node("output", "output", "Output")],
               [_edge("input", "cA"), _edge("input", "cB"), _edge("cA", "cat"), _edge("cB", "cat"), _edge("cat", "output")],
               [1, 3, 16, 16]),
        [1, 3, 16, 16], "cat")

    # Mid-stack Linear whose in_features (128) doesn't match the previous width (64).
    out["bad_linear_mid_stack"] = (
        _graph("bad_linear_mid_stack",
               [_node("input", "input", "Input", shape=[1, 32]),
                _node("l1", "layer", "nn.Linear", in_features=32, out_features=64),
                _node("l2", "layer", "nn.Linear", in_features=128, out_features=10),
                _node("output", "output", "Output")],
               [_edge("input", "l1"), _edge("l1", "l2"), _edge("l2", "output")], [1, 32]),
        [1, 32], "l2")

    return out


CRAFTED_INVALID = _crafted_invalid()


# ─────────────────────────────────────────────────────────────────────────────
# Build the (analytic ArchitectureGraph, torch payload, input_shape) triple per entry
# ─────────────────────────────────────────────────────────────────────────────
def _gallery_pair(model_id: str):
    hg = TemplateService().instantiate_popular_model(model_id, {})
    input_shape = _input_shape_from_root(hg.root_nodes)
    payload = hg.model_dump()
    return _gallery_arch_graph(payload, input_shape), payload, input_shape


def _registry_pair(template_id: str):
    graph, _ = _build_graph_for_sequence(_REGISTRY, _VALIDATOR, [template_id], 0, enable_adapters=False)
    input_shape = next(list(n.params.get("shape", [])) for n in graph.nodes if n.kind == "input")
    return graph, _arch_to_payload(graph), input_shape


def _invalid_pair(name: str):
    graph, input_shape, _expected = CRAFTED_INVALID[name]
    return graph, _arch_to_payload(graph), input_shape


_PAIR = {"gallery": _gallery_pair, "registry": _registry_pair, "invalid": _invalid_pair}


def _verdicts(kind: str, ident: str):
    """Return (analytic_valid, analytic_blame, torch_valid, torch_blame) for a corpus
    entry. Analytic uses ``strict=False`` — strict promotes warnings to errors and adds
    guards that have no torch-forward analogue, so the honest cross-plane comparison is
    non-strict (mirrors the /validate_shapes route + verify_model's real forward)."""
    arch, payload, input_shape = _PAIR[kind](ident)

    result = _VALIDATOR.validate(graph=arch, strict=False)
    analytic_valid = result.valid
    analytic_blame = {nid for nid, nr in result.node_results.items() if nr.status == "error"}

    res = verify_model(payload, input_shape)
    torch_valid = bool(res.get("ok"))
    torch_blame = {e["node_id"] for e in (res.get("errors") or []) if e.get("node_id")}

    return analytic_valid, analytic_blame, torch_valid, torch_blame


def _corpus():
    return (
        [("gallery", m) for m in GALLERY]
        + [("registry", t) for t in SAMPLE]
        + [("invalid", n) for n in CRAFTED_INVALID]
    )


def _params():
    out = []
    for kind, ident in _corpus():
        marks = []
        if ident in KNOWN_PARITY_GAPS:
            marks.append(pytest.mark.xfail(strict=False, reason=KNOWN_PARITY_GAPS[ident]))
        out.append(pytest.param(kind, ident, id=ident, marks=marks))
    return out


@pytest.mark.parametrize("kind,ident", _params())
def test_plane_parity(kind, ident):
    analytic_valid, analytic_blame, torch_valid, torch_blame = _verdicts(kind, ident)

    # PRIMARY: the two planes agree valid-vs-invalid.
    assert analytic_valid == torch_valid, (
        f"{ident}: verdicts disagree — analytic={analytic_valid} torch={torch_valid} "
        f"(analytic_blame={sorted(analytic_blame)} torch_blame={sorted(torch_blame)})"
    )

    # SECONDARY (softer): where a crafted graph is invalid in BOTH planes, both must
    # blame the node the error was planted on. Divergences are documented in
    # NODE_BLAME_GAPS rather than silently tolerated.
    if kind == "invalid" and not analytic_valid and not torch_valid and ident not in NODE_BLAME_GAPS:
        expected = CRAFTED_INVALID[ident][2]
        assert expected in analytic_blame and expected in torch_blame, (
            f"{ident}: node-blame diverges — expected both planes to blame '{expected}' "
            f"(analytic_blame={sorted(analytic_blame)} torch_blame={sorted(torch_blame)})"
        )


def test_parity_coverage(capsys):
    """Independent walk of the whole corpus: prints the verdict-parity tally and the
    node-blame match count, and is the real regression gate — fails if a NON-gap graph
    stops agreeing, if a crafted-invalid graph's blame diverges outside NODE_BLAME_GAPS,
    or if either gap set references an entry no longer in the corpus (a stale key)."""
    corpus = _corpus()
    total = len(corpus)
    verdict_agree = 0
    both_invalid = 0
    blame_matches = 0
    unexpected_disagree, unexpected_agree, blame_divergence = [], [], []

    for kind, ident in corpus:
        analytic_valid, analytic_blame, torch_valid, torch_blame = _verdicts(kind, ident)
        agree = analytic_valid == torch_valid
        verdict_agree += int(agree)

        if not analytic_valid and not torch_valid:
            both_invalid += 1
            if kind == "invalid":
                expected = CRAFTED_INVALID[ident][2]
                if expected in analytic_blame and expected in torch_blame:
                    blame_matches += 1
                elif ident not in NODE_BLAME_GAPS:
                    blame_divergence.append(ident)
            elif analytic_blame & torch_blame:
                blame_matches += 1

        if not agree and ident not in KNOWN_PARITY_GAPS:
            unexpected_disagree.append(ident)
        if agree and ident in KNOWN_PARITY_GAPS:
            unexpected_agree.append(ident)

    with capsys.disabled():
        print(
            f"\n[Tier-6 plane parity] analytic==torch verdict on {verdict_agree}/{total} "
            f"corpus graphs ({blame_matches} node-blame matches over {both_invalid} both-invalid; "
            f"{len(KNOWN_PARITY_GAPS)} known verdict gaps documented)"
        )
        if unexpected_agree:
            print("FIXED (prune from KNOWN_PARITY_GAPS): " + ", ".join(sorted(unexpected_agree)))

    assert not unexpected_disagree, (
        f"Non-gap corpus graphs where planes disagree on the verdict: {unexpected_disagree}"
    )
    assert not blame_divergence, (
        f"Crafted-invalid graphs where the planes blame different nodes "
        f"(add to NODE_BLAME_GAPS with a reason if intended): {blame_divergence}"
    )

    corpus_ids = {ident for _, ident in corpus}
    stale = set(KNOWN_PARITY_GAPS) - corpus_ids
    assert not stale, f"KNOWN_PARITY_GAPS references entries not in the corpus: {stale}"
    stale_blame = set(NODE_BLAME_GAPS) - set(CRAFTED_INVALID)
    assert not stale_blame, f"NODE_BLAME_GAPS references unknown crafted graphs: {stale_blame}"
