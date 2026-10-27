from services.canonical_template_registry import CanonicalTemplateRegistry


def test_dynamic_template_infers_linear_and_embedding_defaults():
    registry = CanonicalTemplateRegistry()
    nodes, _edges, _entry, _exit, _cfg = registry.instantiate("deepfm_128_block", "test_deepfm", {})

    linear_nodes = [node for node in nodes if node.op in {"nn.Linear", "Linear"}]
    embedding_nodes = [node for node in nodes if node.op in {"nn.Embedding", "Embedding"}]

    assert linear_nodes, "Expected at least one linear node in deepfm_128_block"
    assert embedding_nodes, "Expected at least one embedding node in deepfm_128_block"

    linear_params = linear_nodes[0].params
    embedding_params = embedding_nodes[0].params
    assert isinstance(linear_params.get("in_features"), int) and linear_params["in_features"] > 0
    assert isinstance(linear_params.get("out_features"), int) and linear_params["out_features"] > 0
    assert isinstance(embedding_params.get("num_embeddings"), int) and embedding_params["num_embeddings"] > 0
    assert isinstance(embedding_params.get("embedding_dim"), int) and embedding_params["embedding_dim"] > 0


def test_dynamic_template_infers_attention_and_layernorm_defaults():
    registry = CanonicalTemplateRegistry()
    nodes, _edges, _entry, _exit, _cfg = registry.instantiate(
        "transformer_encoder_prenorm_block", "test_transformer_encoder", {}
    )

    attn_nodes = [node for node in nodes if node.op in {"nn.MultiheadAttention", "MultiheadAttention"}]
    ln_nodes = [node for node in nodes if node.op in {"nn.LayerNorm", "LayerNorm"}]

    assert attn_nodes, "Expected attention node in transformer_encoder_prenorm_block"
    assert ln_nodes, "Expected layer norm node in transformer_encoder_prenorm_block"

    attn_params = attn_nodes[0].params
    ln_params = ln_nodes[0].params
    assert isinstance(attn_params.get("embed_dim"), int) and attn_params["embed_dim"] > 0
    assert isinstance(attn_params.get("num_heads"), int) and attn_params["num_heads"] > 0

    normalized_shape = ln_params.get("normalized_shape")
    assert isinstance(normalized_shape, (list, int))
    if isinstance(normalized_shape, list):
        assert normalized_shape and all(isinstance(v, int) and v > 0 for v in normalized_shape)
    else:
        assert normalized_shape > 0


def test_dynamic_template_auto_heals_orphan_node_edges():
    registry = CanonicalTemplateRegistry()
    registry.register_dynamic_template(
        template_id="test_orphan_dynamic",
        name="Orphan Dynamic",
        family="test",
        nodes=[
            {"suffix": "a", "layer_op": "nn.Linear", "default_params": {"in_features": 64, "out_features": 64}},
            {"suffix": "b", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "c", "layer_op": "nn.Linear", "default_params": {"in_features": 64, "out_features": 64}},
        ],
        edges=[
            {"source_suffix": "a", "target_suffix": "b"},
            # "c" intentionally orphaned to verify auto-heal logic.
        ],
        required_nodes=["Linear", "ReLU", "Linear"],
        required_edges=["a -> b"],
        allowed_residual_patterns=[],
        training_minimums=[],
        input_expectation="rank-2 tensor [batch, features]",
        persist=False,
    )

    _nodes, edges, _entry, _exit, _cfg = registry.instantiate("test_orphan_dynamic", "orphan_case", {})
    edge_ids = {edge.edge_id for edge in edges}
    assert "orphan_case__b:y->orphan_case__c:x" in edge_ids


def test_dynamic_template_cat_updates_following_conv_channels():
    registry = CanonicalTemplateRegistry()
    registry.register_dynamic_template(
        template_id="test_cat_conv_channels",
        name="Cat Conv Channels",
        family="test",
        nodes=[
            {"suffix": "a", "layer_op": "nn.Conv2d", "default_params": {"in_channels": 3, "out_channels": 3, "kernel_size": 1}},
            {"suffix": "b", "layer_op": "nn.Conv2d", "default_params": {"in_channels": 3, "out_channels": 3, "kernel_size": 1}},
            {"suffix": "cat", "layer_op": "torch.cat", "default_params": {"dim": 1}},
            {"suffix": "proj", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
        ],
        edges=[
            {"source_suffix": "a", "target_suffix": "cat"},
            {"source_suffix": "b", "target_suffix": "cat"},
            {"source_suffix": "cat", "target_suffix": "proj"},
        ],
        required_nodes=["Conv2d", "Conv2d", "cat", "Conv2d"],
        required_edges=["a -> cat", "b -> cat", "cat -> proj"],
        allowed_residual_patterns=[],
        training_minimums=[],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        persist=False,
    )

    nodes, _edges, _entry, _exit, _cfg = registry.instantiate("test_cat_conv_channels", "cat_case", {})
    proj = next(node for node in nodes if node.node_id == "cat_case__proj")
    assert proj.params.get("in_channels") == 6
