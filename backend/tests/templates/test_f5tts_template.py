"""Regression tests for the F5-TTS template + popular model.

F5-TTS (Chen et al., 2024, arXiv:2410.06885; swivid/F5-TTS) is a non-
autoregressive flow-matching TTS model with a DiT backbone. The builder renders
the FULL, faithful multi-branch graph (every block as a real, correctly-named
node):

  - 4 inputs: text token ids, noisy mel (flow sample), reference mel, timestep
  - text encoder: Char Embedding + ConvNeXt-V2 blocks (DWConv1d, LayerNorm,
    PWConv, GELU, GRN, residual add)
  - timestep branch: SinusPositionEmbedding -> Linear -> SiLU -> Linear
  - input embedding: Concat(noisy ‖ cond ‖ text) -> Linear
  - conv position embedding: Conv1d k31 + Mish (residual)
  - `depth` DiT blocks: AdaLayerNorm-Zero (conditioned on timestep) -> RoPE ->
    Self-Attention -> gated residual add -> FF AdaLN -> FeedForward -> gated
    residual add
  - AdaLayerNorm-Final -> Output Projection -> mel flow output

These tests pin the COMPLETE block inventory and per-type counts (so no block
can be silently dropped) and assert both canvas gates report zero errors:
verify_model (dry torch forward) and the /validate_shapes engine. The custom
blocks (SinusPositionEmbedding / GRN / RoPE / AdaLayerNorm) are shape-preserving
and recognized by both gates; the conditioning + residual wiring is explicit.
"""

from collections import Counter

import pytest


from api.routers.validate import ValidationRequest, validate_shapes  # noqa: E402
from services.template_service import TemplateService  # noqa: E402
from services.verification_service import verify_model  # noqa: E402

INPUT_SHAPE = [1, 256]
OUTPUT_SHAPE = [1, 256, 100]
DEPTH = 22
CONV_LAYERS = 4


@pytest.fixture(scope="module")
def svc():
    return TemplateService()


@pytest.fixture(scope="module")
def graph(svc):
    return svc.instantiate_template("f5_tts", svc.get_template_by_id("f5_tts").base_config)


def _op_counts(g) -> Counter:
    return Counter(n.data.layerType.id for n in g.root_nodes)


# ── discoverability ──────────────────────────────────────────────────────────
def test_f5tts_is_discoverable_template_and_popular_model(svc):
    assert "f5_tts" in svc.get_templates_by_category("generative")
    assert "f5tts" in svc.get_all_popular_models()
    model = svc.get_popular_model_by_id("f5tts")
    assert model is not None and model.template_id == "f5_tts"
    assert svc.get_template_by_id("f5_tts") is not None


# ── multi-branch I/O ──────────────────────────────────────────────────────────
def test_f5tts_has_four_inputs_and_one_output(graph):
    inputs = [n for n in graph.root_nodes if n.type == "inputNode"]
    outputs = [n for n in graph.root_nodes if n.type == "outputNode"]
    assert {n.id for n in inputs} == {"in_text", "in_noisy", "in_cond", "in_time"}
    assert len(outputs) == 1


# ── full block inventory + per-type counts (don't-miss-anything guard) ────────
def test_f5tts_full_block_inventory(graph):
    c = _op_counts(graph)
    # text encoder
    assert c["embedding"] == 1                       # Char Embedding
    assert c["grn"] == CONV_LAYERS                   # one GRN per ConvNeXt-V2 block
    # timestep embedding branch
    assert c["sinusoidal_pos_embedding"] == 1
    assert c["silu"] == 1
    # DiT backbone: one RoPE + one attention per block, two AdaLN per block
    assert c["rotary_embedding"] == DEPTH
    assert c["multiheadattention"] == DEPTH
    assert c["ada_layer_norm"] == 2 * DEPTH
    assert c["ada_layer_norm_final"] == 1
    assert c["dropout"] == DEPTH                      # one per FeedForward
    # fusion / position embedding
    assert c["cat"] == 1                              # InputEmbedding concat
    assert c["mish"] == 2                             # ConvPositionEmbedding
    # convs: ConvNeXt depthwise (x conv_layers) + 2 conv-position-embedding convs
    assert c["conv1d"] == CONV_LAYERS + 2
    # residual adds: convnext (x conv_layers) + convpos (1) + 2 per DiT block
    assert c["add"] == CONV_LAYERS + 1 + 2 * DEPTH
    # the building blocks every transformer FF / projection needs
    assert c["linear"] > 0 and c["gelu"] > 0 and c["layernorm"] == CONV_LAYERS
    # AdaLayerNorm-Zero modulation projections: one Linear(dim->6*dim) per DiT
    # block + a final Linear(dim->2*dim) — the bulk of the model's parameters.
    mods = [n for n in graph.root_nodes if n.id.endswith("_mod")]
    assert len(mods) == DEPTH + 1
    dit0_mod = next(n for n in graph.root_nodes if n.id == "dit0_mod")
    p = dit0_mod.data.parameters
    assert p["out_features"] == 6 * p["in_features"]


def test_f5tts_conditioning_and_residuals_are_wired(graph):
    """The timestep embedding feeds each block's AdaLayerNorm-Zero modulation
    projection (one per block + final), which in turn modulates both adaLNs; and
    each DiT block's residual adds take the block input as a parent."""
    edges = {(e.source, e.target) for e in graph.root_edges}
    t_emb = "t_lin2"
    # timestep feeds every modulation projection: DEPTH block mods + the final one
    mod_targets = [t for (s, t) in edges if s == t_emb]
    assert len(mod_targets) == DEPTH + 1
    assert ("t_lin2", "dit0_mod") in edges
    # the per-block modulation drives both the attn and FF adaLNs
    assert ("dit0_mod", "dit0_adaln1") in edges
    assert ("dit0_mod", "dit0_adaln2") in edges
    # block 0 gated residuals reference the block input (cpe_add) and attn/ff out
    assert ("cpe_add", "dit0_adaln1") in edges
    assert ("cpe_add", "dit0_add1") in edges          # gated residual (attn)
    assert ("dit0_attn", "dit0_add1") in edges
    assert ("dit0_add1", "dit0_add2") in edges        # gated residual (ff)


def test_real_torch_params_are_numeric(graph):
    """Config substitution must yield ints, not strings (else torch breaks)."""
    emb = next(n for n in graph.root_nodes if n.id == "text_embed").data.parameters
    assert emb["embedding_dim"] == 512 and isinstance(emb["embedding_dim"], int)
    proj = next(n for n in graph.root_nodes if n.id == "out_proj").data.parameters
    assert proj["in_features"] == 1024 and proj["out_features"] == 100
    attn = next(n for n in graph.root_nodes if n.id == "dit0_attn").data.parameters
    assert attn["embed_dim"] % attn["num_heads"] == 0


# ── canvas gates: verify_model + /validate_shapes (zero errors) ──────────────
def _payload(g):
    gd = g.model_dump()
    nodes = [{"id": n["id"], "type": n["type"], "position": n.get("position", {"x": 0, "y": 0}),
              "data": {"layerType": n["data"].get("layerType"),
                       "parameters": n["data"].get("parameters", {})}}
             for n in gd["root_nodes"]]
    edges = [{"id": e.get("id") or f'{e["source"]}->{e["target"]}',
              "source": e["source"], "target": e["target"]} for e in gd["root_edges"]]
    return ValidationRequest(nodes=nodes, edges=edges, input_shape=INPUT_SHAPE)


def test_f5tts_passes_verify_and_validate(graph):
    """Both canvas gates report zero errors; output is the mel flow [B, N, 100]."""
    result = verify_model(graph.model_dump(), INPUT_SHAPE)
    assert result["ok"] is True, result.get("errors")
    assert result["stage"] == "complete"
    assert list(result["output_shape"]) == OUTPUT_SHAPE

    validation = validate_shapes(_payload(graph))
    assert validation["status"] == "valid", validation.get("errors")


def test_f5tts_popular_model_verifies_and_validates(svc):
    g = svc.instantiate_popular_model("f5tts", {})
    result = verify_model(g.model_dump(), INPUT_SHAPE)
    assert result["ok"] is True, result.get("errors")
    validation = validate_shapes(_payload(g))
    assert validation["status"] == "valid", validation.get("errors")
