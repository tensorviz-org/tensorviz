from scripts.run_template_validation_sweep import _build_graph_for_sequence
from services.canonical_template_registry import CanonicalTemplateRegistry
from services.canonical_validation import CanonicalValidationService


def test_sweep_graph_builder_connects_previous_exit_to_all_template_roots():
    registry = CanonicalTemplateRegistry()
    validator = CanonicalValidationService()

    graph, unsupported_transition = _build_graph_for_sequence(
        registry=registry,
        validator=validator,
        sequence=["dual_encoder_fusion_block"],
        graph_index=1,
        enable_adapters=False,
    )
    assert unsupported_transition is None

    edge_ids = {edge.edge_id for edge in graph.edges}
    assert any("input:y->" in edge_id and "__image_proj:x" in edge_id for edge_id in edge_ids)
    assert any("input:y->" in edge_id and "__text_proj:x" in edge_id for edge_id in edge_ids)
