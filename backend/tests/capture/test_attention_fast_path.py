"""Attention modules trace the same operations on every tier and with any instrumentation.

PyTorch runs nn.MultiheadAttention / nn.TransformerEncoderLayer / the nested-tensor
nn.TransformerEncoder as fused inference kernels unless a module hook, a
TorchFunctionMode or a tensor subclass is present. On `main` the legacy real tier
(which installs activation-capture hooks) therefore drew an encoder layer's
sub-layers while the fake tier drew one fused `_transformer_encoder_layer_fwd` and
lost their edges (perceiver_like, cvt_hybrid). trace_model now turns the fast path
off for the traced call only.
"""
import warnings

import pytest
import torch
from torch import nn

from services.ingest import trace as trace_module
from services.ingest.trace import CaptureSpec, capture_scope, trace_model

D = 8
FUSED = ("aten._native_multi_head_attention", "aten._transformer_encoder_layer_fwd", "aten._nested_tensor_from_mask")
PADDING = torch.tensor([[False] * 5, [False] * 3 + [True] * 2])


def attention(**options):
    return nn.MultiheadAttention(D, 2, **options)


CASES = {
    "encoder_layer": (lambda: nn.TransformerEncoderLayer(D, 2, 16, batch_first=True),
                      lambda: ((torch.randn(2, 5, D),), {})),
    "encoder_layer_norm_first": (lambda: nn.TransformerEncoderLayer(D, 2, 16, batch_first=True, norm_first=True),
                                 lambda: ((torch.randn(2, 5, D),), {})),
    "encoder_padding_mask": (lambda: nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 2, 16, batch_first=True), 2),
                             lambda: ((torch.randn(2, 5, D),), {"src_key_padding_mask": PADDING})),
    "mha_need_weights": (lambda: attention(batch_first=True), lambda: ((torch.randn(2, 5, D),) * 3, {})),
    "mha_no_weights": (lambda: attention(batch_first=True),
                       lambda: ((torch.randn(2, 5, D),) * 3, {"need_weights": False})),
    "mha_sequence_first": (lambda: attention(), lambda: ((torch.randn(5, 2, D),) * 3, {"need_weights": False})),
    "mha_key_padding_mask": (lambda: attention(batch_first=True),
                             lambda: ((torch.randn(2, 5, D),) * 3, {"need_weights": False, "key_padding_mask": PADDING})),
    "mha_attention_mask": (lambda: attention(batch_first=True),
                           lambda: ((torch.randn(2, 5, D),) * 3,
                                    {"need_weights": False, "attn_mask": torch.zeros(5, 5, dtype=torch.bool)})),
}


def operations(model, args, kwargs, tier, hooked=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # PyTorch's nested-tensor prototype warning
        if hooked:  # what the worker's real tier always has: activation-capture hooks
            with capture_scope(CaptureSpec(activations=True)):
                result = trace_model(model, args, kwargs, tier=tier)
        else:
            result = trace_model(model, args, kwargs, tier=tier)
    return result.error, [op["op"] for op in result.ops]


@pytest.mark.parametrize("name", CASES)
def test_every_tier_records_the_decomposed_attention(name):
    build, inputs = CASES[name]
    torch.manual_seed(0)
    model = build().eval()
    args, kwargs = inputs()
    fake = operations(model, args, kwargs, "fake")
    real = operations(model, args, kwargs, "real")
    hooked = operations(model, args, kwargs, "real", hooked=True)
    assert fake[0] is None and real[0] is None and hooked[0] is None, (fake[0], real[0], hooked[0])
    assert fake[1] == real[1] == hooked[1]
    assert not [op for op in fake[1] if op.startswith(FUSED)]
    assert any(op.startswith(("aten.addmm", "aten.linear", "aten.mm", "aten.bmm")) for op in fake[1])


def test_the_flag_is_restored_after_success_failure_and_an_explicit_choice():
    mha = torch.backends.mha
    model = nn.TransformerEncoderLayer(D, 2, 16, batch_first=True).eval()
    assert mha.get_fastpath_enabled() is True
    trace_model(model, (torch.randn(2, 5, D),), tier="real")
    assert mha.get_fastpath_enabled() is True
    failed = trace_model(model, (torch.randn(2, 5, D + 1),), tier="real")
    assert failed.error and mha.get_fastpath_enabled() is True
    mha.set_fastpath_enabled(False)
    try:
        trace_model(model, (torch.randn(2, 5, D),), tier="real")
        assert mha.get_fastpath_enabled() is False
    finally:
        mha.set_fastpath_enabled(True)


def test_the_flag_is_off_only_while_the_traced_call_runs():
    seen = []

    class Probe(nn.Module):
        def forward(self, x):
            seen.append(torch.backends.mha.get_fastpath_enabled())
            return x + 1
    trace_model(Probe(), (torch.ones(2),), tier="real")
    trace_model(Probe(), (torch.ones(2),), tier="fake")
    assert seen == [False, False]
    assert torch.backends.mha.get_fastpath_enabled() is True


def test_nested_tensor_inputs_behave_as_before():
    # Tracing a nested tensor fails in the tracer itself ("NestedTensorImpl doesn't
    # support sizes") with or without the fast path; the change neither fixes nor
    # breaks it, and the flag is still restored.
    model = attention(batch_first=True).eval()
    nested = torch.nested.nested_tensor([torch.randn(3, D), torch.randn(5, D)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errors = {tier: trace_model(model, (nested,) * 3, {"need_weights": False}, tier=tier).error
                  for tier in ("real", "fake")}
    assert errors["real"] and errors["real"] == errors["fake"]
    assert torch.backends.mha.get_fastpath_enabled() is True


def test_canonical_real_captures_change_only_by_the_encoder_eligibility_check(monkeypatch):
    """The recorder's TorchFunctionMode already kept the canonical (product real-tier)
    capture off the fused kernels, so it is identical with or without the switch,
    with one exception. nn.TransformerEncoder called with src_key_padding_mask (while
    use_nested_tensor is on) checks nested-tensor eligibility by running
    logical_not + _nested_tensor_from_mask_left_aligned before it looks at
    has_torch_function. With the fast path off that check is skipped, so those two
    bookkeeping operations, which compute nothing the model returns, are no longer
    recorded."""
    from services.semantic.contracts import CaptureLimits, SourceSnapshot
    from services.semantic.recorder import CaptureContext
    context = CaptureContext(source=SourceSnapshot(version=1, id="source:0", workspaceSessionId="session:0",
        workspaceId="workspace:0", modelId="model:0", modelFile="model.py", className="Model", generation=0,
        files=[dict(path="model.py", digest="a" * 64, documentVersion=1, role="source")],
        dependencyScope="complete", configDigest=None, environmentDigest="b" * 64, sourceMode="saved"),
        plan_digest="c" * 64, attempt_id="attempt:0",
        limits=CaptureLimits(operations=10000, invocations=10000, values=10000, bytes=4 * 1024 * 1024))

    def captured(name):
        build, inputs = CASES[name]
        torch.manual_seed(0)
        model = build().eval()
        args, kwargs = inputs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = trace_model(model, args, kwargs, tier="real", semantic_context=context)
        assert result.error is None and result.semantic_capture["status"] == "complete"
        return {key: value for key, value in result.semantic_capture.items() if key != "id"}

    with_switch = {name: captured(name) for name in CASES}
    monkeypatch.setattr(trace_module._AttentionFastPathOff, "__enter__", lambda self: None)
    without_switch = {name: captured(name) for name in CASES}
    changed = [name for name in CASES if with_switch[name] != without_switch[name]]
    assert changed == ["encoder_padding_mask"]
    targets = lambda capture: [op["target"] for op in capture["operations"]]
    check = ["aten.logical_not.default", "aten._nested_tensor_from_mask_left_aligned.default"]
    before, after = targets(without_switch["encoder_padding_mask"]), targets(with_switch["encoder_padding_mask"])
    assert [op for op in before if op in check] == check and not [op for op in after if op in check]
    assert [op for op in before if op not in check] == after
    for capture in with_switch.values():
        assert not [op for op in capture["operations"] if op["target"].startswith(FUSED)]


@pytest.mark.parametrize("tier", ["real", "fake"])
def test_an_error_in_the_traced_call_reports_the_frames_of_an_untoggled_call(monkeypatch, tier):
    """The switch is entered around the call, not wrapped around it, so a model that
    raises reports exactly the frames it reported before (error_frames feed the error
    site the canvas points at), and the flag is still restored."""
    class Raises(nn.Module):
        def forward(self, x):
            raise RuntimeError("boom")

    def frames():
        result = trace_model(Raises(), (torch.randn(2, D),), {}, tier=tier)
        assert result.error == "RuntimeError: boom"
        return [(name, line) for _, line, name in result.error_frames]

    prior = torch.backends.mha.get_fastpath_enabled()
    toggled = frames()
    assert torch.backends.mha.get_fastpath_enabled() is prior
    monkeypatch.setattr(trace_module._AttentionFastPathOff, "__enter__", lambda self: None)
    assert frames() == toggled
    assert [name for name, _ in toggled][:2] == ["trace_model", "<lambda>"]
