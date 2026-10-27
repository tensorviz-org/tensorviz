"""Regression tests for the Step 0 / Step 0.5 validation fixes.

- RMSNorm templates must get `normalized_shape` auto-filled (Step 0).
- Element-wise placeholder ops (`mul`) must be treated as shape-preserving so their
  templates validate in strict mode (Step 0.5). `sum` was in that list too until the
  passthrough was found to be masking a wrong aggregation axis in the MPNN template.

These guard against the regression where newly added RMSNorm templates and the
`torch.mul`/`torch.sum` ops failed canonical validation.
"""

from models.architecture_models import (
    ArchitectureEdge,
    ArchitectureGraph,
    ArchitectureNode,
)
from services.canonical_template_registry import CanonicalTemplateRegistry
from services.canonical_validation import CanonicalValidationService


def _validate_template(template_id: str, input_shape):
    registry = CanonicalTemplateRegistry()
    validator = CanonicalValidationService()
    nodes, edges, entry, exit_, _cfg = registry.instantiate(template_id, f"t1_{template_id}", {})
    graph_nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": input_shape}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
        *nodes,
    ]
    graph_edges = [
        *edges,
        ArchitectureEdge(edge_id=f"input:y->{entry}:x", source_node_id="input", source_port="y", target_node_id=entry, target_port="x"),
        ArchitectureEdge(edge_id=f"{exit_}:y->output:x", source_node_id=exit_, source_port="y", target_node_id="output", target_port="x"),
    ]
    graph = ArchitectureGraph(
        graph_id=f"g_{template_id}",
        name=template_id,
        revision=0,
        inputs=[{"name": "x", "shape": input_shape}],
        nodes=graph_nodes,
        edges=graph_edges,
        metadata={},
    )
    return validator.validate(graph=graph, strict=True)


def test_rmsnorm_template_gets_normalized_shape():
    """Step 0: an rmsnorm node must have normalized_shape auto-filled."""
    registry = CanonicalTemplateRegistry()
    nodes, _edges, _entry, _exit, _cfg = registry.instantiate(
        "transformer_rmsnorm_block", "t1_transformer_rmsnorm_block", {}
    )
    rms_nodes = [n for n in nodes if "rms" in n.node_id.split("__")[-1]]
    assert rms_nodes, "expected at least one rmsnorm node"
    for node in rms_nodes:
        assert node.params.get("normalized_shape"), f"{node.node_id} missing normalized_shape"


def test_rmsnorm_transformer_validates_strict():
    """Step 0: the rmsnorm transformer block validates in strict mode."""
    result = _validate_template("transformer_rmsnorm_block", [1, 16, 256])
    assert result.valid, [f"{nid}: {r.message}" for nid, r in result.node_results.items() if r.status != "ok"]


def test_message_passing_block_validates_strict():
    """Step 0.5: the MPNN aggregate sums over nodes; block validates strict.

    The axis is what this pins. Now that torch.sum carries the real reduction, summing
    the FEATURE axis (the old `dim: 1`) leaves the update_fn Linear a last dim of 1 and
    the block goes invalid — measured at both shapes below — which is precisely the
    failure the passthrough rule used to hide.

    Also checked at num_nodes > 1, not because [1, 64] misses the wrong axis (it does
    not) but because [1, 64] is degenerate: reducing a single node is shape-preserving,
    so it cannot show the fix works on a real graph. NEITHER shape distinguishes
    `keepdim=True` from `keepdim=False` — the validator accepts the resulting rank-1
    tensor either way. keepdim is held by the template's declared input contract
    ("rank-2 node features [num_nodes, in_features]"), not by this assertion.
    """
    for input_shape in ([1, 64], [3, 64]):
        result = _validate_template("message_passing_block", input_shape)
        assert result.valid, [
            f"{input_shape} {nid}: {r.message}"
            for nid, r in result.node_results.items()
            if r.status != "ok"
        ]


def test_t5_ffn_block_validates_strict():
    """Step 0.5: `mul` (gated FFN) is shape-preserving; block validates strict."""
    result = _validate_template("t5_ffn_block", [1, 16, 256])
    assert result.valid, [f"{nid}: {r.message}" for nid, r in result.node_results.items() if r.status != "ok"]
