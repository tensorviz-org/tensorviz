"""Tests for canonical complex template registry."""


import pytest


from services.canonical_template_registry import CanonicalTemplateRegistry  # noqa: E402


def test_palette_inventory_remains_available_without_the_paper_pipeline(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from server_v2 import app

    monkeypatch.setenv("PAPER_DYNAMIC_TEMPLATES_REGISTRY_PATH", str(tmp_path / "templates.json"))
    response = TestClient(app).get("/api/papers/template-inventory")

    assert response.status_code == 200
    inventory = response.json()
    assert inventory["summary"]["template_total"] == len(inventory["templates"])
    assert any(item["template_id"] == "full_transformer_block" for item in inventory["templates"])


def test_registry_contracts_include_all_supported_templates():
    template_ids = {
        contract.template_id
        for contract in CanonicalTemplateRegistry().all_contracts()
    }
    for template_id in [
        "full_transformer_block",
        "vit_block",
        "unet_block",
        "seq2seq_attention_block",
        "dual_encoder_fusion_block",
    ]:
        assert template_id in template_ids


@pytest.mark.parametrize(
    "template_id,config,entry_suffix,exit_suffix",
    [
        ("full_transformer_block", {"d_model": 128, "num_heads": 8, "ffn_dim": 256}, "norm1", "add_mlp"),
        ("vit_block", {"patch_dim": 64, "d_model": 128, "num_heads": 8, "mlp_dim": 256}, "patch_embed", "add_mlp"),
        ("unet_block", {"in_channels": 3, "encoder_channels": 32, "bottleneck_channels": 64, "decoder_channels": 32}, "enc_conv1", "fuse_conv"),
        ("seq2seq_attention_block", {"input_size": 128, "encoder_hidden": 128, "decoder_hidden": 128, "num_heads": 8}, "encoder", "projection"),
        ("dual_encoder_fusion_block", {"image_dim": 256, "text_dim": 256, "fusion_dim": 128}, "image_proj", "fuse_add"),
    ],
)
def test_registry_instantiates_known_templates(template_id, config, entry_suffix, exit_suffix):
    registry = CanonicalTemplateRegistry()
    nodes, edges, entry_id, exit_id, applied_config = registry.instantiate(
        template_id=template_id,
        instance_id="block1",
        config=config,
    )
    assert len(nodes) >= 5
    assert len(edges) >= 4
    assert entry_id == f"block1__{entry_suffix}"
    assert exit_id == f"block1__{exit_suffix}"
    assert isinstance(applied_config, dict)


def test_registry_post_checks_reject_invalid_transformer_head_divisibility():
    registry = CanonicalTemplateRegistry()
    with pytest.raises(ValueError) as exc:
        registry.instantiate(
            template_id="full_transformer_block",
            instance_id="bad_block",
            config={"d_model": 130, "num_heads": 8, "ffn_dim": 256},
        )
    assert "failed post-checks" in str(exc.value)


def test_registry_loads_expansion_pack_with_100_plus_templates():
    registry = CanonicalTemplateRegistry()
    contracts = registry.all_contracts()
    dynamic_ids = registry.list_dynamic_template_ids()
    assert len(contracts) >= 250
    assert len(dynamic_ids) >= 230
    for required_id in [
        "convnext_block",
        "efficientnet_mbconv_block",
        "graphsage_block",
        "transformer_moe_block",
        "rotary_attention_block",
        "diffusion_residual_block",
    ]:
        assert required_id in [c.template_id for c in contracts]


@pytest.mark.parametrize(
    "template_id",
    [
        "convnext_block",
        "tfm_enc_layernorm_gelu_h8_block",
        "gnn_attention_3hop_block",
        "lstm_4layer_bi_block",
        "multimodal_cross_attn_768_block",
        "linformer_block",
        "performer_block",
        "bigbird_block",
        "flash_attention_block",
        "multi_query_attention_block",
        "wav2vec2_encoder_block",
        "whisper_encoder_decoder_block",
        "qformer_fusion_block",
        "flamingo_gated_cross_attn_block",
    ],
)
def test_registry_instantiates_expansion_templates(template_id):
    registry = CanonicalTemplateRegistry()
    nodes, edges, entry_id, exit_id, applied_config = registry.instantiate(
        template_id=template_id,
        instance_id="exp1",
        config={},
    )
    assert len(nodes) >= 3
    assert len(edges) >= 2
    assert entry_id.startswith("exp1__")
    assert exit_id.startswith("exp1__")
    assert isinstance(applied_config, dict)


def test_all_dynamic_templates_use_instance_prefixed_member_ids():
    registry = CanonicalTemplateRegistry()
    dynamic_ids = registry.list_dynamic_template_ids()
    assert len(dynamic_ids) >= 200

    for template_id in dynamic_ids:
        nodes, _, entry_id, exit_id, _ = registry.instantiate(
            template_id=template_id,
            instance_id="expand_guard",
            config={},
        )
        assert entry_id.startswith("expand_guard__")
        assert exit_id.startswith("expand_guard__")
        assert nodes, f"{template_id} produced no nodes"
        for node in nodes:
            assert node.node_id.startswith("expand_guard__"), (
                f"{template_id} node id '{node.node_id}' is not instance-prefixed; "
                "module expansion grouping depends on this convention."
            )
