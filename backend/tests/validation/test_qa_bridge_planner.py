"""QA regression tests for the bridge planner's adapter-insertion policy.

Focus (PRs #43, #44, #63): the rank-2 native-acceptance fix and the product
placeholder walk.

The bridge planner used to collapse each op to a single "required rank"
(``required_rank_for_op``), which forced a spurious rank change onto tensors that
the target already accepts. The fix (``_op_accepts_batched_rank``) consults the
op registry so an op that *natively* accepts the current batched rank is left
alone — with a deliberate exception for sequence models (LSTM/GRU/RNN/
Transformer*), whose registry rank-2 acceptance is the *unbatched* form and so
must still gain a batch-sequence dim on the always-batched canvas.

These tests target gaps not covered by ``test_bridge_planner.py`` (which tests
the rank-4/rank-2 transitions and ``resolve_*`` helpers in isolation): the
BatchNorm1d no-adapter path, the sequence-model seq-dim expansion semantics, the
``_op_accepts_batched_rank`` distinction itself, and direct graph validation.
"""

from __future__ import annotations



import pytest

from models.architecture_models import (
    ArchitectureEdge,
    ArchitectureGraph,
    ArchitectureNode,
)
from services.bridge_planner import (
    _op_accepts_batched_rank,
    insert_adapter_chain,
)
from services.canonical_validation import CanonicalValidationService


# ─── helpers ───────────────────────────────────────────────────────────────────

def _input_node(shape: list[int]) -> ArchitectureNode:
    return ArchitectureNode(
        node_id="input", kind="input", op="Input",
        params={"shape": shape}, in_ports=[], out_ports=["y"],
    )


def _edge(src: str, tgt: str) -> ArchitectureEdge:
    return ArchitectureEdge(
        edge_id=f"{src}:y->{tgt}:x",
        source_node_id=src, source_port="y",
        target_node_id=tgt, target_port="x",
    )


def _adapt(source_shape, target_op, target_params, prefix="br"):
    """Run insert_adapter_chain from a fresh single-input graph."""
    nodes: list[ArchitectureNode] = [_input_node(source_shape)]
    edges: list[ArchitectureEdge] = []
    new_source, ins_nodes, ins_edges, issue = insert_adapter_chain(
        source_shape=source_shape,
        nodes=nodes,
        edges=edges,
        source_node_id="input",
        target_op=target_op,
        target_params=target_params,
        adapter_prefix=prefix,
    )
    return new_source, ins_nodes, ins_edges, issue


# ─── _op_accepts_batched_rank: the core distinction the fix introduced ─────────

class TestOpAcceptsBatchedRank:
    """The registry-aware check is what stops the planner force-adapting an
    already-valid input. BatchNorm1d natively takes a *batched* rank-2 [N,C];
    sequence models list rank-2 only in their *unbatched* form, so on the batched
    canvas they must be treated as NOT accepting rank-2."""

    def test_batchnorm1d_accepts_batched_rank2(self):
        assert _op_accepts_batched_rank("nn.BatchNorm1d", 2) is True

    def test_batchnorm1d_accepts_rank3(self):
        assert _op_accepts_batched_rank("nn.BatchNorm1d", 3) is True

    def test_lstm_rank2_is_unbatched_not_accepted(self):
        # LSTM lists rank 2 in the registry, but that is (L, H_in) unbatched;
        # a canvas [N, F] must still gain a seq dim, so this must be False.
        assert _op_accepts_batched_rank("nn.LSTM", 2) is False

    def test_gru_rank2_is_unbatched_not_accepted(self):
        assert _op_accepts_batched_rank("nn.GRU", 2) is False

    def test_conv1d_does_not_accept_rank2(self):
        # Conv1d requires rank 3 exactly — rank 2 is genuinely unacceptable.
        assert _op_accepts_batched_rank("nn.Conv1d", 2) is False


# ─── rank-2 into an op that natively accepts it: NO adapter ────────────────────

class TestRank2NativeAcceptNoAdapter:
    """The headline fix: a rank-2 [N,C] feeding BatchNorm1d is already valid, so
    the planner must insert nothing. A regression here (reverting to the collapsed
    required-rank logic) would splice a Linear+Unflatten that mangles the tensor."""

    def test_batchnorm1d_rank2_inserts_no_adapter(self):
        new_source, ins_nodes, ins_edges, issue = _adapt(
            [8, 64], "nn.BatchNorm1d", {"num_features": 64},
        )
        assert issue is None
        assert new_source == "input", "source must be handed to BN1d directly"
        assert ins_nodes == []
        assert ins_edges == []

    def test_batchnorm1d_rank2_direct_connection_validates(self):
        """End-to-end proof the no-adapter decision is correct: [8,64] → BN1d(64)
        validates with no bridge, so any inserted adapter would be gratuitous."""
        nodes = [
            _input_node([8, 64]),
            ArchitectureNode(node_id="bn", kind="layer", op="nn.BatchNorm1d",
                             params={"num_features": 64}, in_ports=["x"], out_ports=["y"]),
            ArchitectureNode(node_id="output", kind="output", op="Output",
                             params={}, in_ports=["x"], out_ports=[]),
        ]
        edges = [_edge("input", "bn"), _edge("bn", "output")]
        graph = ArchitectureGraph(
            graph_id="g", name="n", revision=0,
            inputs=[{"name": "x", "shape": [8, 64]}],
            nodes=nodes, edges=edges, modules=[], metadata={},
        )
        result = CanonicalValidationService().validate(graph=graph, strict=False)
        assert result.node_results["bn"].status == "ok"


# ─── rank-2 into a sequence model: seq dim IS added ────────────────────────────

class TestRank2SequenceModelAddsSeqDim:
    """Counterpart to the BatchNorm1d path: sequence models must gain a batched
    sequence dim (rank 2 → 3), inserted as Linear + Unflatten([1, feature])."""

    @pytest.mark.parametrize(
        "op,params",
        [
            ("nn.LSTM", {"input_size": 300, "hidden_size": 64}),
            ("nn.GRU", {"input_size": 300, "hidden_size": 64}),
            ("nn.RNN", {"input_size": 300, "hidden_size": 64}),
            ("nn.TransformerEncoderLayer", {"d_model": 300, "nhead": 6}),
        ],
    )
    def test_rank2_gains_sequence_dim(self, op, params):
        new_source, ins_nodes, ins_edges, issue = _adapt([8, 300], op, params)
        assert issue is None
        assert new_source != "input", f"{op} rank-2 must be expanded to rank-3"
        unflatten = [n for n in ins_nodes if n.op == "nn.Unflatten"]
        assert len(unflatten) == 1, f"{op}: exactly one Unflatten expected"
        size = unflatten[0].params["unflattened_size"]
        # A single-step sequence dim is prepended: [feature] → [1, feature].
        assert size[0] == 1, f"{op}: leading seq dim must be 1, got {size}"
        assert len(size) == 2, f"{op}: result must be rank-3 (dim1 = [1, F]); got {size}"

    def test_batchnorm1d_and_lstm_diverge_on_identical_rank2_input(self):
        """The whole point of the fix: the *same* [8,300] input is left untouched
        for BatchNorm1d but expanded for LSTM."""
        bn_new, bn_nodes, _, _ = _adapt([8, 300], "nn.BatchNorm1d", {"num_features": 300}, "bn")
        lstm_new, lstm_nodes, _, _ = _adapt([8, 300], "nn.LSTM", {"input_size": 300, "hidden_size": 64}, "ls")
        assert bn_nodes == [] and bn_new == "input"
        assert lstm_nodes and lstm_new != "input"


# ─── rank-3 BatchNorm1d: correct boundary + the confirmed bug ──────────────────

class TestRank3BatchNorm1dLength:
    def test_long_sequence_no_adapter(self):
        """Control / boundary: a rank-3 [8,64,16] into BatchNorm1d(64) is already
        valid and the planner correctly inserts nothing (last-dim length >= 16)."""
        new_source, ins_nodes, ins_edges, issue = _adapt(
            [8, 64, 16], "nn.BatchNorm1d", {"num_features": 64},
        )
        assert issue is None
        assert new_source == "input"
        assert ins_nodes == []

    def test_short_sequence_must_not_insert_spurious_adapter(self):
        """A rank-3 [8,64,10] is already valid for BatchNorm1d(64) — no adapter
        should be inserted (fixed: 1d-norm ops keep their sequence length)."""
        new_source, ins_nodes, ins_edges, issue = _adapt(
            [8, 64, 10], "nn.BatchNorm1d", {"num_features": 64},
        )
        assert new_source == "input"
        assert ins_nodes == []
        assert ins_edges == []
