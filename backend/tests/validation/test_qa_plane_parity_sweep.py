"""QA regression suite for the plane-parity corpus & validation sweep (PRs #33, #38).

These lock in the two headline behaviours the sweep + plane-parity work added, and
target the GAPS the existing ``tests/test_plane_parity.py`` leaves open. That suite
compares the analytic plane (``canonical_validation``) against the torch plane
(``verify_model``) over a *stride-sampled* registry slice, so most templates never
reach it. This file instead:

  1. Runs the FULL single-template validation sweep (all 306 registry templates) and
     asserts single==306/306 in BOTH standard and strict mode — the "validation moat"
     the sweep exists to defend. (Combination checks are out of scope: ~93k directed
     pairs with ~20 known combo issues; the sweep documents those and we do not.)
  2. Spot-checks that graphs the analytic plane greenlights genuinely forward under a
     real torch build+forward (the parity moat, exercised end-to-end).
  3. Documents and counts the full bidirectional-RNN (13) analytic/torch divergence —
     most of which escapes ``KNOWN_PARITY_GAPS`` because it falls outside its stride
     sample — so no divergence stays invisible to the gate. The FPN-head family (16)
     was the second such divergence and is now fixed; all 16 blocks stay parametrized
     as the positive case.

Both planes are fed the SAME logical graph through their real product entry points:
the analytic plane validates the canonical ``ArchitectureGraph`` (agent-ops / validate
path), the torch plane runs ``verify_model`` on the flattened canvas payload
(/api/models/verify path) — the same conversion Tier-5/Tier-6 tests use.

Run: KMP_DUPLICATE_LIB_OK=TRUE <viz-python> -m pytest tests/test_qa_plane_parity_sweep.py -q
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # must precede torch import
os.environ.setdefault("AUTH_DISABLED", "true")



import pytest

from scripts.run_template_validation_sweep import _build_graph_for_sequence, _run_case
from services.canonical_template_registry import CanonicalTemplateRegistry
from services.canonical_validation import CanonicalValidationService
from services.verification_service import _arch_node_to_flat, verify_model

# KNOWN_PARITY_GAPS is the gate whose under-coverage these bugs are about; importing it
# lets the corpus test assert exactly which divergences the existing gate documents.
from tests.validation.test_plane_parity import KNOWN_PARITY_GAPS

_REGISTRY = CanonicalTemplateRegistry()
_VALIDATOR = CanonicalValidationService()

# Snapshot of the registry at the PR-33/#38 baseline. The invariants below are
# passed==total and total==len(contracts); this extra pin makes a corpus size change
# fail loudly so the "single=306/306" claim is re-baselined consciously.
EXPECTED_TEMPLATE_COUNT = 306

# The full bidirectional-RNN block family (Bug 1) and FPN-head family (Bug 2). Held as
# literals so test_bug_corpus_matches_registry can catch registry drift, and so the
# parity parametrizations below stay exhaustive rather than sampled.
BIDIRECTIONAL_BLOCKS = [
    "bilstm_block",
    "gru_2layer_bi_block", "gru_4layer_bi_block", "gru_6layer_bi_block", "gru_8layer_bi_block",
    "lstm_2layer_bi_block", "lstm_4layer_bi_block", "lstm_6layer_bi_block", "lstm_8layer_bi_block",
    "rnn_2layer_bi_block", "rnn_4layer_bi_block", "rnn_6layer_bi_block", "rnn_8layer_bi_block",
]
FPN_HEAD_BLOCKS = [
    "detr_fpn_l3_head_block", "detr_fpn_l4_head_block", "detr_fpn_l5_head_block", "detr_fpn_l6_head_block",
    "fcos_fpn_l3_head_block", "fcos_fpn_l4_head_block", "fcos_fpn_l5_head_block", "fcos_fpn_l6_head_block",
    "retina_fpn_l3_head_block", "retina_fpn_l4_head_block", "retina_fpn_l5_head_block", "retina_fpn_l6_head_block",
    "yolo_fpn_l3_head_block", "yolo_fpn_l4_head_block", "yolo_fpn_l5_head_block", "yolo_fpn_l6_head_block",
]

# ── Full documentation of the remaining analytic/torch parity gap (Bug 1). Every block
# in the family is analytic-valid / torch-invalid TODAY (reproduced across the whole
# family, not the stride sample): the 13 bidirectional-RNN blocks each hit a 2*hidden
# Linear mismatch. The production shape logic is intentionally NOT changed here
# (sweep-gated / risky), so the gap stands — but it is now documented in FULL, closing
# the coverage hole where test_plane_parity's stride-sampled KNOWN_PARITY_GAPS reached
# only 2-of-13 and left the other 11 divergences invisible. The uniform per-family
# reason is accurate because the divergence has a single root cause across the family.
# Bug 2, the 16-block FPN-head arity divergence, is fixed — see below. ────────────────
# BIDIRECTIONAL_PARITY_GAPS is gone: the family no longer diverges. The documented cause
# was right — a bidirectional RNN concatenates its two directions, so its output width is
# 2*hidden_size — but the conclusion, that the production rule could not be touched, was
# not: `_estimate_dynamic_output_shape` now doubles the width and all 13 blocks agree on
# both planes. test_bidirectional_rnn_family_agrees_on_both_planes asserts that instead.

# FPN_HEAD_PARITY_GAPS is gone: the family no longer diverges. The documented cause was
# wrong — an FPN head is not a multi-input block, it takes one external input and the
# fan-in was internal, which merge ops already express. The template had no merge node
# and wired all L pyramid levels straight into a Conv2d; template_expansion_pack now
# concatenates them on the channel axis first. All 16 blocks are still parametrized
# below, as the positive case, so a regression fails rather than disappearing.

# Templates the analytic plane accepts that ALSO forward cleanly under real torch — the
# parity moat's positive case. Deliberately spans conv / residual / graph / dilated-conv
# families so a plane regression in any one surfaces here.
ANALYTIC_PASS_TEMPLATES = [
    "cnn_conv2_k3_plain_block",
    "resnet_basic_block",
    "gcn_layer",
    "aspp_block",
]


def _is_bidirectional(tid: str) -> bool:
    return "bi_block" in tid or "bilstm" in tid


def _is_fpn_head(tid: str) -> bool:
    return "fpn" in tid and "head" in tid


def _single_verdicts(template_id: str):
    """Build the single-template sweep graph and return (analytic_valid, torch_valid).

    Mirrors tests/test_plane_parity.py's registry harness exactly: the analytic plane
    validates the canonical graph non-strict (the honest cross-plane comparison — strict
    adds guards with no torch-forward analogue), and the torch plane runs verify_model on
    the flattened canvas payload built via the real _arch_node_to_flat converter.
    """
    graph, _ = _build_graph_for_sequence(_REGISTRY, _VALIDATOR, [template_id], 0, enable_adapters=False)
    input_shape = next(list(n.params.get("shape", [])) for n in graph.nodes if n.kind == "input")
    payload = {
        "nodes": [_arch_node_to_flat(n) for n in graph.nodes],
        "edges": [
            {"id": e.edge_id, "source": e.source_node_id, "target": e.target_node_id}
            for e in graph.edges
        ],
    }
    analytic_valid = _VALIDATOR.validate(graph=graph, strict=False).valid
    res = verify_model(payload, input_shape)
    return analytic_valid, bool(res.get("ok")), res


# ─────────────────────────────────────────────────────────────────────────────
# Core PR behaviour: the single-template validation sweep is fully green.
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("strict", [False, True], ids=["standard", "strict"])
def test_single_template_sweep_all_pass(strict):
    """Every registered template validates on its own — single==306/306 per pass.

    This is the sweep's reason to exist: the analytic validation moat must be green for
    all 306 canonical templates individually, in both standard mode and strict mode
    (strict promotes warnings to errors and adds extra guards). A single template that
    stops validating would regress the moat and fail here with the offending id.
    """
    contracts = _REGISTRY.all_contracts()
    total = 0
    passed = 0
    failures = []
    for idx, contract in enumerate(contracts, start=1):
        total += 1
        ok, issue = _run_case(
            registry=_REGISTRY,
            validator=_VALIDATOR,
            sequence=(contract.template_id,),
            graph_index=idx,
            strict=strict,
            enable_adapters=False,
        )
        if ok:
            passed += 1
        else:
            failures.append((contract.template_id, issue.issue_type if issue else "?",
                             issue.summary if issue else ""))

    assert not failures, f"single-template sweep ({'strict' if strict else 'standard'}) regressions: {failures[:10]}"
    assert passed == total, f"sweep pass count {passed} != total {total}"
    assert total == len(contracts) == EXPECTED_TEMPLATE_COUNT, (
        f"corpus drifted: swept {total} templates, registry has {len(contracts)}, "
        f"baseline {EXPECTED_TEMPLATE_COUNT} — re-baseline the single=306/306 claim consciously"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core PR behaviour: analytic-accepted graphs genuinely torch-forward (parity moat).
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("template_id", ANALYTIC_PASS_TEMPLATES)
def test_analytic_pass_templates_actually_torch_run(template_id):
    """A graph the analytic plane greenlights must build and forward under real torch.

    The whole point of the two-plane design is that an analytic 'valid' is trustworthy.
    For these representative accepted templates both planes must agree valid — if torch
    could not actually forward one, the analytic moat would be painting a broken graph
    green (the failure mode the bidirectional-RNN bug below still exhibits).
    """
    analytic_valid, torch_valid, res = _single_verdicts(template_id)
    assert analytic_valid, f"{template_id}: analytic plane unexpectedly rejected an accepted template"
    assert torch_valid, (
        f"{template_id}: analytic-accepted graph failed the real torch forward — "
        f"parity moat broken: {res.get('errors')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Guard: the bug corpora below stay exhaustive against the live registry.
# ─────────────────────────────────────────────────────────────────────────────
def test_bug_corpus_matches_registry():
    """The bidirectional / FPN-head families this file gates must equal the live
    registry's — otherwise a newly-added divergent block would slip the parity checks
    below (exactly how the documented bugs escaped the stride-sampled gate). Fails loud
    if the registry gains or drops a matching template."""
    ids = {c.template_id for c in _REGISTRY.all_contracts()}
    assert set(BIDIRECTIONAL_BLOCKS) == {t for t in ids if _is_bidirectional(t)}
    assert set(FPN_HEAD_BLOCKS) == {t for t in ids if _is_fpn_head(t)}


# ─────────────────────────────────────────────────────────────────────────────
# Bug 1 (medium), now CLOSED: the analytic RNN rule ignored bidirectional
# feature-doubling, so a downstream Linear sized for 1*hidden was analytic-valid while
# torch raised "mat1 and mat2 shapes cannot be multiplied". These 13 blocks were pinned
# as documented gaps; `_estimate_dynamic_output_shape` now doubles the width for
# bidirectional=True, so — exactly as the old pin's docstring instructed — they are kept
# parametrized as the POSITIVE case instead. Same treatment Bug 2 got below.
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("template_id", BIDIRECTIONAL_BLOCKS)
def test_bidirectional_rnn_family_agrees_on_both_planes(template_id):
    """Every bidirectional-RNN block validates analytically AND forwards under torch.

    These 13 blocks were the file's first documented divergence: a bidirectional RNN
    concatenates its two directions, so its output width is 2*hidden_size, and the
    analytic rule returned hidden_size. The projection that follows was then sized for
    half the tensor it would actually receive — analytic-valid, and dead on the first
    matmul under real torch.

    `_estimate_dynamic_output_shape` now doubles the width when bidirectional is set, so
    the planes agree. Kept parametrized over the WHOLE family rather than the stride
    sample, which reached only 2 of the 13 — that gap is how eleven of them escaped
    notice in the first place.
    """
    analytic_valid, torch_valid, res = _single_verdicts(template_id)
    assert analytic_valid and torch_valid, (
        f"{template_id}: both planes must accept this block — got "
        f"analytic={analytic_valid} torch={torch_valid}: {res.get('errors')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bug 2 (low), now CLOSED: FPN-head multi-input Conv arity. This was 16 pinned
# divergences; the template gained the merge node its heads needed, so the same 16
# blocks are kept parametrized as the positive case instead.
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("template_id", FPN_HEAD_BLOCKS)
def test_fpn_head_family_agrees_on_both_planes(template_id):
    """Every FPN-head block validates analytically AND forwards under real torch.

    These 16 blocks used to be the file's second documented divergence: the analytic
    plane accepted L pyramid levels fanning into a single-input Conv2d while torch
    raised 'Conv2d received N inputs but expects exactly 1'. The template now
    concatenates the levels on the channel axis before the heads, so the two planes
    agree. Kept parametrized over the WHOLE family (not the stride sample, which
    reaches 2 of 16) so re-breaking any level count fails here.
    """
    analytic_valid, torch_valid, res = _single_verdicts(template_id)
    assert analytic_valid and torch_valid, (
        f"{template_id}: both planes must accept this block — got "
        f"analytic={analytic_valid} torch={torch_valid}: {res.get('errors')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Coverage: the full divergent families are documented and counted, and the split
# between what the stride-sampled gate reaches and what escaped it is pinned.
# ─────────────────────────────────────────────────────────────────────────────
def test_documented_parity_gap_counts():
    """No divergent family is documented any more, and the stride-sampled gate must not
    still be carrying either of them.

    This file used to gate two: 16 FPN-head blocks and 13 bidirectional-RNN blocks. Both
    are fixed, so what is pinned here is the absence — if a plane fix regresses, the
    positive tests above fail first, and if someone re-adds a stale entry to
    KNOWN_PARITY_GAPS this catches it.
    """
    gated_bi = {t for t in KNOWN_PARITY_GAPS if _is_bidirectional(t)}
    gated_fpn = {t for t in KNOWN_PARITY_GAPS if _is_fpn_head(t)}
    assert not gated_bi, (
        f"the bidirectional-RNN family agrees on both planes now — prune "
        f"{sorted(gated_bi)} from test_plane_parity.KNOWN_PARITY_GAPS"
    )
    assert not gated_fpn, (
        f"the FPN-head family agrees on both planes now — prune {sorted(gated_fpn)} "
        f"from test_plane_parity.KNOWN_PARITY_GAPS"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Coverage regression: this file gates the FULL divergent set, not a stride sample.
# ─────────────────────────────────────────────────────────────────────────────
def test_this_file_supersets_the_documented_parity_gaps():
    """Every bidirectional / FPN-head divergence that test_plane_parity's
    KNOWN_PARITY_GAPS documents must fall inside the families this file exhaustively
    parametrizes — so the xfail parity suites above genuinely superset that stride-sampled
    gate and close its coverage hole (the root of both bugs: 2-of-13 bi and 2-of-16 FPN
    documented, the rest escaping).

    Forward-safe: a plane fix only removes divergences, never adds bi/FPN ids, so this
    never fires on a fix. It fires only if the gate ever documents a bi/FPN id this file
    fails to exercise — a real coverage inconsistency worth catching.

    Both families are fixed now, so both sets are legitimately empty and the containment
    below holds vacuously. The assertion that the bi set is NON-empty was removed rather
    than inverted: it existed to stop the gate silently forgetting a live divergence, and
    with the divergence gone it would have demanded the gate document a bug that no longer
    exists — which is how a stale pin becomes a reason not to fix something.
    """
    documented_bi = {t for t in KNOWN_PARITY_GAPS if _is_bidirectional(t)}
    documented_fpn = {t for t in KNOWN_PARITY_GAPS if _is_fpn_head(t)}

    assert documented_bi <= set(BIDIRECTIONAL_BLOCKS), (
        f"gate documents bi ids this file does not exercise: {sorted(documented_bi - set(BIDIRECTIONAL_BLOCKS))}"
    )
    # The FPN-head family is fixed, so the gate should document none of it; any id it
    # documents again must still be one this file parametrizes.
    assert documented_fpn <= set(FPN_HEAD_BLOCKS), (
        f"gate documents FPN ids this file does not exercise: {sorted(documented_fpn - set(FPN_HEAD_BLOCKS))}"
    )
