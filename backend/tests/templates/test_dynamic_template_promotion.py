

from services.canonical_template_registry import CanonicalTemplateRegistry


def test_dynamic_template_registry_persists_and_instantiates(monkeypatch, tmp_path):
    registry_path = tmp_path / "dynamic-templates.json"
    monkeypatch.setenv("PAPER_DYNAMIC_TEMPLATES_REGISTRY_PATH", str(registry_path))

    registry = CanonicalTemplateRegistry()
    registry.register_dynamic_template(
        template_id="tts_encoder_decoder_block",
        name="TTS Encoder Decoder Block",
        family="tts",
        nodes=[
            {"suffix": "identity", "layer_op": "nn.Identity", "default_params": {}},
            {"suffix": "linear1", "layer_op": "nn.Linear", "default_params": {"in_features": 64, "out_features": 64}},
            {"suffix": "relu", "layer_op": "nn.ReLU", "default_params": {}},
        ],
        edges=[
            {"source_suffix": "identity", "target_suffix": "linear1"},
            {"source_suffix": "linear1", "target_suffix": "relu"},
        ],
        required_nodes=["nn.Linear", "nn.ReLU"],
        required_edges=["identity -> linear1", "linear1 -> relu"],
        allowed_residual_patterns=[],
        training_minimums=["in_features > 0"],
        input_expectation="rank-2 tensor [batch, features]",
        metadata={"source": "test"},
    )

    reloaded = CanonicalTemplateRegistry()
    assert "tts_encoder_decoder_block" in reloaded.list_dynamic_template_ids()

    nodes, edges, entry_id, exit_id, applied = reloaded.instantiate(
        template_id="tts_encoder_decoder_block",
        instance_id="tts_block_1",
        config={},
    )
    assert len(nodes) == 3
    assert len(edges) == 2
    assert entry_id == "tts_block_1__identity"
    assert exit_id == "tts_block_1__relu"
    assert applied == {}
