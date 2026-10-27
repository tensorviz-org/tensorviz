"""Regression tests for the Vision Transformer (ViT-B/16) popular model.

These guard the runnable architecture the frontend gallery renders:
Input -> Patch Embedding (Conv2d) -> Tokenize (Flatten) -> Transformer Block
(x12, expanded; one nn.TransformerEncoderLayer) -> Class Token (pool) ->
Flatten -> Linear -> Output, with the frontend display keys
(_title/_subtitle/_shape/_accent) in node.data.parameters, AND that it both
shape-infers and runs a real forward pass.
"""


import pytest
import torch


from services.template_service import TemplateService  # noqa: E402
from services.verification_service import verify_model, _flatten_for_torch  # noqa: E402
from services.code_generation import CodeGenerationService  # noqa: E402


@pytest.fixture(scope="module")
def graph():
    return TemplateService().instantiate_popular_model("vit_base", {})


def test_vit_is_discoverable_as_popular_model():
    svc = TemplateService()
    assert "vit_base" in svc.get_all_popular_models()
    model = svc.get_popular_model_by_id("vit_base")
    assert model is not None
    assert model.template_id == "vision_transformer"


def test_vit_top_level_nodes_in_order(graph):
    expected = [
        ("input_node", "inputNode", "input"),
        ("patch_embedding", "layerNode", "conv2d"),
        ("tokenize", "layerNode", "flatten"),
        ("transformer_block", "moduleNode", "module"),
        ("pool", "layerNode", "adaptiveavgpool1d"),
        ("flatten_out", "layerNode", "flatten"),
        ("classifier_head", "layerNode", "linear"),
        ("output_node", "outputNode", "output"),
    ]
    actual = [(n.id, n.type, n.data.layerType.id) for n in graph.root_nodes]
    assert actual == expected


def test_vit_top_level_edges_form_the_chain(graph):
    edges = {(e.source, e.target) for e in graph.root_edges}
    assert edges == {
        ("input_node", "patch_embedding"),
        ("patch_embedding", "tokenize"),
        ("tokenize", "transformer_block"),
        ("transformer_block", "pool"),
        ("pool", "flatten_out"),
        ("flatten_out", "classifier_head"),
        ("classifier_head", "output_node"),
    }


def test_transformer_block_repeats_12_with_runnable_encoder(graph):
    mod = graph.modules.get("transformer_block")
    assert mod is not None, "transformer_block module definition missing"
    assert mod.parameters.get("repeat_count") == 12
    assert mod.parameters.get("repeat_block_label") == "EncoderLayer"
    # One self-contained, single-input encoder layer (so the dry forward runs).
    assert len(mod.internal_nodes) == 1
    enc = mod.internal_nodes[0]
    assert enc.data.layerType.id == "transformerencoderlayer"
    # entry == exit == the encoder layer
    assert mod.input_node_id == enc.id
    assert mod.output_node_id == enc.id


def test_transformer_block_renders_expanded_by_default(graph):
    payload = graph.model_dump()
    dumped = next(n for n in payload["root_nodes"] if n["id"] == "transformer_block")
    assert dumped["data"].get("expanded") is True


def test_reserved_display_keys_present_on_root_layers(graph):
    expected = {
        "patch_embedding": ("Conv2d 16×16", "[B, 768, 14, 14]", "var(--cat-conv)"),
        "pool": ("pool", "[B, 768, 1]", "var(--cat-activation)"),
        "classifier_head": ("1000", "[B, 1000]", "var(--error)"),
    }
    by_id = {n.id: n for n in graph.root_nodes}
    for node_id, (sub, shape, accent) in expected.items():
        p = by_id[node_id].data.parameters
        assert p.get("_subtitle") == sub, node_id
        assert p.get("_shape") == shape, node_id
        assert p.get("_accent") == accent, node_id


def test_real_torch_params_are_numeric(graph):
    """Config substitution must yield ints, not strings (else torch breaks)."""
    conv = next(n for n in graph.root_nodes if n.id == "patch_embedding").data.parameters
    assert conv["out_channels"] == 768 and isinstance(conv["out_channels"], int)
    assert conv["kernel_size"] == 16 and isinstance(conv["kernel_size"], int)
    enc = graph.modules["transformer_block"].internal_nodes[0].data.parameters
    assert isinstance(enc["d_model"], int)
    assert enc["d_model"] % enc["nhead"] == 0  # heads must divide the model dim


def test_vit_passes_shape_inference_and_forward(graph):
    """The whole point of the runnable template: a real dry forward succeeds."""
    result = verify_model(graph.model_dump(), [1, 3, 224, 224])
    assert result["ok"] is True, result.get("errors")
    assert result["stage"] == "complete"
    assert result["output_shape"][-1] == 1000


def test_vit_exported_code_runs(graph):
    """Exported PyTorch code instantiates and forwards end-to-end."""
    flat = _flatten_for_torch(graph.model_dump())
    code = CodeGenerationService().generate_flat_code(flat, [1, 3, 224, 224], "ViTModel")
    ns: dict = {}
    exec(code, ns)  # noqa: S102 - generated code under test
    model = ns["ViTModel"]()
    y = model(torch.randn(1, 3, 224, 224))
    assert tuple(y.shape) == (1, 1000)
