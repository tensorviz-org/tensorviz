from scripts.run_template_validation_sweep import _run_case
from services.canonical_template_registry import CanonicalTemplateRegistry
from services.canonical_validation import CanonicalValidationService


def test_adapter_probe_cache_does_not_corrupt_final_validation():
    registry = CanonicalTemplateRegistry()
    validator = CanonicalValidationService()

    passed, issue = _run_case(
        registry=registry,
        validator=validator,
        sequence=("aspp_block", "cnn_conv2_k3_plain_block"),
        graph_index=123,
        strict=False,
        enable_adapters=True,
    )
    assert passed
    assert issue is None


def test_adapter_aligns_channels_for_rank4_boundaries():
    registry = CanonicalTemplateRegistry()
    validator = CanonicalValidationService()

    passed, issue = _run_case(
        registry=registry,
        validator=validator,
        sequence=("aspp_block", "cnn_conv2_k3_plain_block"),
        graph_index=456,
        strict=False,
        enable_adapters=True,
    )
    assert passed, issue.summary if issue else "Expected adapter bridge to pass"


def test_adapter_targets_first_compute_node_when_template_entry_is_input():
    registry = CanonicalTemplateRegistry()
    validator = CanonicalValidationService()

    # Regression guard: graph templates often start with an Input node.
    # Adapter planning must align to the first compute node (Linear here),
    # not the Input stub, otherwise feature alignment is skipped.
    passed, issue = _run_case(
        registry=registry,
        validator=validator,
        sequence=("aspp_block", "gcn_layer"),
        graph_index=789,
        strict=False,
        enable_adapters=True,
    )
    assert passed, issue.summary if issue else "Expected Input->compute adapter planning to pass"


def test_adapter_builds_conv1d_ready_sequence_length_for_audio_frontends():
    registry = CanonicalTemplateRegistry()
    validator = CanonicalValidationService()

    for idx, target in enumerate(("hubert_encoder_block", "wav2vec2_encoder_block"), start=1):
        passed, issue = _run_case(
            registry=registry,
            validator=validator,
            sequence=("aspp_block", target),
            graph_index=900 + idx,
            strict=False,
            enable_adapters=True,
        )
        assert passed, f"{target} bridge failed: {issue.summary if issue else 'unknown issue'}"


def test_aspp_to_autoregressive_decoder_reports_channels_and_bridges_cleanly():
    """The pair the gate's one combination check runs, and what it really means.

    An older note recorded this pair as "a Conv2d head fed a non-4-D
    tensor". It is not: ASPP's five branches all leave [1, 3, 224, 224] and concat
    to 15 channels, so the decoder's head — instantiated at in_channels=3, like
    every rank-4 template — is a 4-D tensor short of channels. Adapters off, that
    graph is genuinely invalid and must say why in channel terms; the chaining path
    the product actually uses (plan_adapter_transition, as in operation_resolver)
    aligns the channels and the same pair validates.
    """
    registry = CanonicalTemplateRegistry()
    validator = CanonicalValidationService()

    raw_passed, raw_issue = _run_case(
        registry=registry,
        validator=validator,
        sequence=("aspp_block", "autoregressive_decoder_block"),
        graph_index=1000,
        strict=False,
        enable_adapters=False,
    )
    assert raw_passed is False and raw_issue is not None
    assert "in_channels mismatch: expected 3, got 15" in raw_issue.summary

    bridged_passed, bridged_issue = _run_case(
        registry=registry,
        validator=validator,
        sequence=("aspp_block", "autoregressive_decoder_block"),
        graph_index=1001,
        strict=True,
        enable_adapters=True,
    )
    assert bridged_passed, bridged_issue.summary if bridged_issue else "bridge failed"
