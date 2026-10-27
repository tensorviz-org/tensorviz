"""Regression tests for the GPT-2 Small popular model (decoder-only transformer).

GPT-2 is the decoder-only counterpart to ViT/nanoDiT: a token-embedding feeds a
stack of nn.TransformerDecoderLayer blocks, then an LM head projects back to the
vocabulary. These guard that it (1) passes the verify_model shape check + dry
forward and (2) exports to PyTorch that instantiates and forwards end-to-end.

The exported-code case is the one that regressed: nn.TransformerDecoderLayer's
forward(tgt, memory) needs two positional inputs, so codegen must emit
`self.layer(x, x)` (mirroring core.layer_builder.call_module) rather than the
single-arg `self.layer(x)` used for encoder layers.
"""


import pytest
import torch


from models.graph_models import HierarchicalGraph  # noqa: E402
from services.code_generation import CodeGenerationService  # noqa: E402
from services.template_service import TemplateService  # noqa: E402
from services.verification_service import verify_model  # noqa: E402

INPUT_SHAPE = [1, 1024]
VOCAB = 50257


@pytest.fixture(scope="module")
def graph():
    return TemplateService().instantiate_popular_model("gpt2_small", {})


def test_gpt2_is_discoverable(graph):
    svc = TemplateService()
    assert "gpt2_small" in svc.get_all_popular_models()
    assert svc.get_popular_model_by_id("gpt2_small").template_id == "transformer_decoder_only"


def test_gpt2_decoder_stack_repeats_12(graph):
    mod = graph.modules.get("decoder_stack")
    assert mod is not None
    assert mod.parameters.get("repeat_count") == 12
    assert len(mod.internal_nodes) == 12
    assert all(n.data.layerType.id == "transformerdecoderlayer" for n in mod.internal_nodes)


def test_gpt2_passes_shape_inference_and_forward(graph):
    result = verify_model(graph.model_dump(), INPUT_SHAPE)
    assert result["ok"] is True, result.get("errors")
    assert result["stage"] == "complete"
    assert result["output_shape"][-1] == VOCAB


def test_gpt2_exported_code_runs(graph):
    """Exported PyTorch instantiates and forwards end-to-end — the decoder
    layers must be called with (tgt, memory), not a single tensor."""
    code = CodeGenerationService().generate_hierarchical_code(graph, INPUT_SHAPE, "GPT2Small")
    ns: dict = {}
    exec(code, ns)  # noqa: S102 - generated code under test
    model = ns["GPT2Small"]()
    y = model(torch.randint(0, VOCAB, tuple(INPUT_SHAPE)))
    assert y.shape[-1] == VOCAB
