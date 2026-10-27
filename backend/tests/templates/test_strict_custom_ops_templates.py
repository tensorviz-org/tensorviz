from models.architecture_models import ArchitectureEdge, ArchitectureGraph, ArchitectureNode
from services.canonical_template_registry import CanonicalTemplateRegistry
from services.canonical_validation import CanonicalValidationService
from scripts.run_template_validation_sweep import _shape_for_template


def _validate_template_strict(template_id: str) -> bool:
    registry = CanonicalTemplateRegistry()
    validator = CanonicalValidationService()
    nodes, edges, entry_node_id, exit_node_id, _cfg = registry.instantiate(template_id, f"test_{template_id}", {})
    input_shape = _shape_for_template(registry, template_id)
    graph = ArchitectureGraph(
        graph_id=f"strict_{template_id}",
        name=template_id,
        revision=0,
        inputs=[{"name": "x", "shape": input_shape}],
        nodes=[
            ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": input_shape}, in_ports=[], out_ports=["y"]),
            *nodes,
            ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
        ],
        edges=[
            ArchitectureEdge(
                edge_id=f"input:y->{entry_node_id}:x",
                source_node_id="input",
                source_port="y",
                target_node_id=entry_node_id,
                target_port="x",
            ),
            *edges,
            ArchitectureEdge(
                edge_id=f"{exit_node_id}:y->output:x",
                source_node_id=exit_node_id,
                source_port="y",
                target_node_id="output",
                target_port="x",
            ),
        ],
        metadata={},
    )
    return validator.validate(graph, strict=True).valid


def test_strict_validation_allows_known_custom_template_ops():
    template_ids = [
        "llama2_block",
        "bifpn_block",
        "performer_block",
        "wav2vec2_encoder_block",
        "hubert_encoder_block",
        "vae_encoder_decoder",
    ]
    for template_id in template_ids:
        assert _validate_template_strict(template_id), f"Template should pass strict validation: {template_id}"
