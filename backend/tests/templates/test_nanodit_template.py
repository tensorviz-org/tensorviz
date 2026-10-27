"""Regression tests for the nanoDiT (Diffusion Transformer) popular model.

These guard the runnable architecture the frontend gallery renders:
Input -> Patch Embedding (Conv2d) -> Tokenize (Flatten) -> DiT Block
(x12, expanded; one pre-norm nn.TransformerEncoderLayer) -> Final Norm
(LayerNorm) -> Detokenize (Unflatten) -> Unpatchify (ConvTranspose2d) ->
Output, with the frontend display keys (_subtitle/_shape/_accent) in
node.data.parameters.

Unlike a classifier, a DiT predicts a tensor with the SAME shape as its
input (the noise prediction), so the chain patchifies, runs the transformer
backbone, then unpatchifies back to image space. The model must (1) pass the
verify_model shape check + dry forward, (2) pass canonical validation with no
errors, and (3) export to PyTorch that instantiates and forwards end-to-end.

Ported from https://github.com/sayakpaul/nanoDiT (DiT, Peebles & Xie 2022).
"""


import pytest
import torch


from models.architecture_models import ArchitectureEdge, ArchitectureGraph, ArchitectureNode  # noqa: E402
from services.canonical_validation import CanonicalValidationService  # noqa: E402
from services.code_generation import CodeGenerationService  # noqa: E402
from services.template_service import TemplateService  # noqa: E402
from services.verification_service import _flatten_for_torch, verify_model  # noqa: E402

INPUT_SHAPE = [1, 3, 32, 32]


@pytest.fixture(scope="module")
def graph():
    return TemplateService().instantiate_popular_model("nanodit", {})


def test_nanodit_is_discoverable_as_popular_model():
    svc = TemplateService()
    assert "nanodit" in svc.get_all_popular_models()
    model = svc.get_popular_model_by_id("nanodit")
    assert model is not None
    assert model.template_id == "diffusion_transformer"
    assert svc.get_template_by_id("diffusion_transformer") is not None


def test_nanodit_top_level_nodes_in_order(graph):
    expected = [
        ("input_node", "inputNode", "input"),
        ("patch_embedding", "layerNode", "conv2d"),
        ("tokenize", "layerNode", "flatten"),
        ("dit_blocks", "moduleNode", "module"),
        ("final_norm", "layerNode", "layernorm"),
        ("detokenize", "layerNode", "unflatten"),
        ("unpatchify", "layerNode", "convtranspose2d"),
        ("output_node", "outputNode", "output"),
    ]
    actual = [(n.id, n.type, n.data.layerType.id) for n in graph.root_nodes]
    assert actual == expected


def test_nanodit_top_level_edges_form_the_chain(graph):
    edges = {(e.source, e.target) for e in graph.root_edges}
    assert edges == {
        ("input_node", "patch_embedding"),
        ("patch_embedding", "tokenize"),
        ("tokenize", "dit_blocks"),
        ("dit_blocks", "final_norm"),
        ("final_norm", "detokenize"),
        ("detokenize", "unpatchify"),
        ("unpatchify", "output_node"),
    }


def test_dit_block_repeats_12_with_runnable_prenorm_encoder(graph):
    mod = graph.modules.get("dit_blocks")
    assert mod is not None, "dit_blocks module definition missing"
    assert mod.parameters.get("repeat_count") == 12
    assert mod.parameters.get("repeat_block_label") == "DiTBlock"
    # One self-contained, single-input encoder layer (so the dry forward runs).
    assert len(mod.internal_nodes) == 1
    blk = mod.internal_nodes[0]
    assert blk.data.layerType.id == "transformerencoderlayer"
    # DiT blocks are pre-norm (LN before attn/MLP), unlike a vanilla encoder.
    assert blk.data.parameters.get("norm_first") is True
    # entry == exit == the encoder layer
    assert mod.input_node_id == blk.id
    assert mod.output_node_id == blk.id


def test_dit_blocks_render_expanded_by_default(graph):
    payload = graph.model_dump()
    dumped = next(n for n in payload["root_nodes"] if n["id"] == "dit_blocks")
    assert dumped["data"].get("expanded") is True


def test_real_torch_params_are_numeric(graph):
    """Config substitution must yield ints, not strings (else torch breaks)."""
    conv = next(n for n in graph.root_nodes if n.id == "patch_embedding").data.parameters
    assert conv["out_channels"] == 384 and isinstance(conv["out_channels"], int)
    assert conv["kernel_size"] == 2 and isinstance(conv["kernel_size"], int)
    # Unpatchify must mirror the patch embed channels so output == input shape.
    deconv = next(n for n in graph.root_nodes if n.id == "unpatchify").data.parameters
    assert deconv["in_channels"] == 384 and deconv["out_channels"] == 3
    blk = graph.modules["dit_blocks"].internal_nodes[0].data.parameters
    assert isinstance(blk["d_model"], int)
    assert blk["d_model"] % blk["nhead"] == 0  # heads must divide the model dim


def test_nanodit_passes_shape_inference_and_forward(graph):
    """The whole point of the runnable template: a real dry forward succeeds,
    and a DiT returns a tensor with the SAME shape as its input."""
    result = verify_model(graph.model_dump(), INPUT_SHAPE)
    assert result["ok"] is True, result.get("errors")
    assert result["stage"] == "complete"
    assert list(result["output_shape"]) == INPUT_SHAPE


def _to_canonical_graph(graph) -> ArchitectureGraph:
    """Flatten the hierarchical graph the way the verify/validate paths do and
    wrap it as a canonical ArchitectureGraph (mirrors api/routers/validate.py)."""
    flat = _flatten_for_torch(graph.model_dump())
    nodes = []
    for n in flat.nodes:
        kind = "input" if n.type == "inputNode" else "output" if n.type == "outputNode" else "layer"
        op = {"input": "Input", "output": "Output"}.get(kind) or f"nn.{n.data.layerType.name}"
        params = {k: v for k, v in (n.data.parameters or {}).items()
                  if k != "name" and not str(k).startswith("_")}
        if kind == "input":
            params["shape"] = INPUT_SHAPE
        nodes.append(ArchitectureNode(
            node_id=n.id, kind=kind, op=op, params=params,
            in_ports=[] if kind == "input" else ["x"],
            out_ports=[] if kind == "output" else ["y"],
        ))
    edges = [ArchitectureEdge(
        edge_id=e.id or f"{e.source}->{e.target}",
        source_node_id=e.source, source_port="y",
        target_node_id=e.target, target_port="x",
    ) for e in flat.edges]
    return ArchitectureGraph(
        graph_id="nanodit_test", name="nanodit", revision=0,
        inputs=[{"name": "x", "shape": INPUT_SHAPE}],
        nodes=nodes, edges=edges, modules=[], metadata={},
    )


@pytest.mark.parametrize("strict", [False, True])
def test_nanodit_passes_canonical_validation(graph, strict):
    """No validation error: the runnable leaf chain validates cleanly under the
    same engine the /validate_shapes endpoint uses (both strict and lenient)."""
    result = CanonicalValidationService().validate(_to_canonical_graph(graph), strict=strict)
    bad = {nid: r.message for nid, r in result.node_results.items() if r.status == "error"}
    assert result.valid, (result.global_errors, bad)


def test_nanodit_exported_code_runs(graph):
    """Exported PyTorch code instantiates and forwards end-to-end, returning a
    same-shape tensor (the DiT noise prediction)."""
    code = CodeGenerationService().generate_hierarchical_code(graph, INPUT_SHAPE, "NanoDiT")
    ns: dict = {}
    exec(code, ns)  # noqa: S102 - generated code under test
    model = ns["NanoDiT"]()
    y = model(torch.randn(*INPUT_SHAPE))
    assert list(y.shape) == INPUT_SHAPE
