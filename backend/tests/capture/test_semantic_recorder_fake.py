"""The canonical recorder observes FakeTensors exactly as it observes real tensors.

A fake run carries shape, dtype, device, stride and identity but no values. These
tests build the same model twice with the same seed, once with real tensors and once
inside a FakeTensorMode (how a fake-tier run will construct it), and require the two
captures to be identical apart from the capture id and the explicit
``valuesUnavailable: "fake_tensor"`` marker. Real captures must not gain that field.

``trace_model(tier="real")`` is called inside a caller-owned FakeTensorMode on
purpose: it runs the canonical recorder with no fake mode of its own, which is the
recorder path a fake-tier capture uses once construction happens inside the mode.
"""
import json

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from services.ingest import execution_setup
from services.ingest.trace import trace_model
from services.semantic.contracts import CaptureLimits, CaptureSnapshot, CapturePreparation, SourceSnapshot
from services.semantic.recorder import CaptureContext, automatic_context


@pytest.fixture
def context():
    return CaptureContext(source=SourceSnapshot(version=1, id="source:0", workspaceSessionId="session:0",
        workspaceId="workspace:0", modelId="model:0", modelFile="model.py", className="Model", generation=0,
        files=[dict(path="model.py", digest="a" * 64, documentVersion=1, role="source")],
        dependencyScope="complete", configDigest=None, environmentDigest="b" * 64, sourceMode="saved"),
        plan_digest="c" * 64, attempt_id="attempt:0",
        limits=CaptureLimits(operations=10000, invocations=10000, values=10000, bytes=4 * 1024 * 1024))


def run(build, inputs, context, fake):
    """Construct, make inputs and capture, all inside one FakeTensorMode when ``fake``."""
    def body():
        torch.manual_seed(0)
        model = build()
        args, kwargs = inputs()
        return model, trace_model(model, args, kwargs, tier="real", semantic_context=context)
    if not fake:
        return body()
    with FakeTensorMode(allow_non_fake_inputs=True):
        return body()


def both(build, inputs, context):
    real_model, real = run(build, inputs, context, fake=False)
    fake_model, fake = run(build, inputs, context, fake=True)
    assert all(type(p) is not FakeTensor for p in real_model.parameters())
    assert all(type(p) is FakeTensor for p in fake_model.parameters())
    assert real.semantic_capture is not None and fake.semantic_capture is not None, (real.error, fake.error)
    return real, fake


def comparable(capture):
    return {key: value for key, value in capture.items() if key not in ("id", "valuesUnavailable")}


def assert_identical_except_marker(real, fake):
    assert "valuesUnavailable" not in real.semantic_capture
    assert fake.semantic_capture["valuesUnavailable"] == "fake_tensor"
    assert json.dumps(comparable(fake.semantic_capture), sort_keys=True) == json.dumps(comparable(real.semantic_capture), sort_keys=True)
    for capture in (real.semantic_capture, fake.semantic_capture):
        assert CaptureSnapshot.model_validate(capture).model_dump(mode="json") == capture


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 6)
        self.norm = nn.BatchNorm1d(6)
        self.register_buffer("scale", torch.full((6,), 2.0))

    def forward(self, x, *, bias=None):
        h = torch.relu_(self.norm(self.linear(x)))     # in-place write
        h = h * self.scale + (0 if bias is None else bias)
        left, right = h.split(3, dim=1)                # views, multi-output
        joined = torch.cat([right, left], dim=1).view(2, -1)
        # Nested outputs: dict, tuple, list, a None and a literal.
        return {"joined": joined, "parts": (left, [right, None]), "rank": joined.dim()}


def test_fake_capture_equals_real_capture_except_the_values_marker(context):
    real, fake = both(Block, lambda: ((torch.randn(4, 6),), {"bias": torch.ones(6)}), context)
    assert real.error is None and fake.error is None
    assert real.semantic_capture["status"] == fake.semantic_capture["status"] == "complete"
    assert_identical_except_marker(real, fake)
    values = fake.semantic_capture["values"]
    assert {value["device"] for value in values} == {"cpu"}
    assert any(value["origin"] == "state" for value in values) and any(value["origin"] == "input" for value in values)
    # The facts are metadata only: no field of a value carries tensor contents.
    assert set(values[0]) == {"id", "objectId", "storageId", "version", "shape", "dtype", "device", "requiresGrad",
                              "aliasOf", "previousVersion", "logicalVersion", "stride", "storageOffset",
                              "logicalBytes", "producer", "origin", "state"}


def test_parameter_counts_are_measured_on_fake_parameters(context):
    real, fake = both(lambda: nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)),
                      lambda: ((torch.randn(3, 4),), {}), context)
    counts = [call.get("parameterCount") for call in fake.semantic_capture["invocations"]]
    assert counts[0] == 4 * 8 + 8 + 8 * 2 + 2
    assert counts == [call.get("parameterCount") for call in real.semantic_capture["invocations"]]
    assert not any(call["parameterCountUnavailable"] for call in fake.semantic_capture["invocations"])


def test_device_queries_are_not_recorded_as_operations(context):
    class UsesDevice(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x):
            # Each `.device` on a FakeTensor is a dispatched prim.device call.
            return self.linear(x) + torch.zeros(3, device=x.device, dtype=self.linear.weight.dtype)
    real, fake = both(UsesDevice, lambda: ((torch.randn(2, 3),), {}), context)
    targets = [op["target"] for op in fake.semantic_capture["operations"]]
    assert not any(target.startswith("prim.") for target in targets)
    assert_identical_except_marker(real, fake)


def test_shape_failure_diagnostic_is_the_same_on_fake_tensors(context):
    class Mismatch(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(5, 2))

        def forward(self, x):
            return torch.mm(x, self.weight)
    real, fake = both(Mismatch, lambda: ((torch.ones(3, 4),), {}), context)
    assert real.error and fake.error
    assert fake.semantic_capture["diagnostics"] == real.semantic_capture["diagnostics"]
    assert fake.semantic_capture["diagnostics"][0]["code"] == "invalid_shape"
    assert fake.semantic_capture["diagnostics"][0]["actualShape"] == [3, 4]
    assert_identical_except_marker(real, fake)


def test_unregistered_parameter_is_incomplete_on_both_tiers(context):
    class Dynamic(nn.Module):
        def forward(self, x):
            return x * nn.Parameter(torch.ones(2))
    real, fake = both(Dynamic, lambda: ((torch.ones(2),), {}), context)
    for result in (real, fake):
        assert result.semantic_capture["status"] == "partial"
        assert "unsupported_type" in result.semantic_capture["coverage"]["omissions"]
    # Known difference: nn.Parameter(fake) is built with extra dispatched aten.detach
    # calls (FakeTensor is a subclass; Parameter.__new__ detaches it). Only
    # parameters created during the forward are affected, and those are already partial.
    # Torch 2.14 records one detach on a real tensor and two on a FakeTensor.
    real_ops = [op["target"] for op in real.semantic_capture["operations"]]
    fake_ops = [op["target"] for op in fake.semantic_capture["operations"]]
    assert real_ops == ["aten.ones.default", "aten.detach.default", "aten.mul.Tensor"]
    assert fake_ops == ["aten.ones.default", "aten.detach.default", "aten.detach.default", "aten.mul.Tensor"]


def test_unsupported_tensor_types_stay_opaque_on_both_tiers(context):
    class Sparse(nn.Module):
        def forward(self, x, mask):
            return x + 1, mask
    def inputs():
        return (torch.ones(2, 2), torch.eye(2).to_sparse()), {}
    real, fake = both(Sparse, inputs, context)
    arguments = fake.semantic_capture["inputs"]["args"]
    assert arguments[1] == {"kind": "opaque", "reason": "unsupported_type", "typeName": arguments[1]["typeName"]}
    assert real.semantic_capture["inputs"]["args"][1]["kind"] == "opaque"
    assert_identical_except_marker(real, fake)


def test_fake_state_alone_is_enough_to_mark_the_capture(context):
    class IgnoresState(nn.Module):
        def __init__(self):
            super().__init__()
            self.unused = nn.Linear(2, 2)

        def forward(self):
            return 3
    real, fake = both(IgnoresState, lambda: ((), {}), context)
    assert not fake.semantic_capture["values"]
    assert_identical_except_marker(real, fake)


def test_real_captures_serialize_exactly_as_before(context):
    _, real = run(Block, lambda: ((torch.randn(4, 6),), {}), context, fake=False)
    payload = real.semantic_capture
    assert "valuesUnavailable" not in payload
    # Round trips keep the field absent (never null), so stored real captures are unchanged.
    parsed = CaptureSnapshot.model_validate(payload)
    assert parsed.valuesUnavailable is None
    assert "valuesUnavailable" not in parsed.model_dump(mode="json")
    assert "valuesUnavailable" not in json.loads(parsed.model_dump_json())


def test_values_marker_rejects_other_reasons(context):
    _, fake = run(Block, lambda: ((torch.randn(4, 6),), {}), context, fake=True)
    with pytest.raises(ValueError):
        CaptureSnapshot.model_validate({**fake.semantic_capture, "valuesUnavailable": "meta_tensor"})


def test_automatic_context_has_no_value_digests_for_fake_inputs_and_state(monkeypatch):
    monkeypatch.setattr(execution_setup.ctypes, "string_at", lambda *_: pytest.fail("fake tensors must not be read"))
    preparation = CapturePreparation.model_construct(source=None, id="preparation:0", captureLimits=None)
    monkeypatch.setattr("services.semantic.execution_plan.applicability_digest", lambda _preparation: "d" * 64)
    with FakeTensorMode():
        model = nn.Linear(4, 2)
        x = torch.ones(3, 4)
    context = automatic_context(preparation, model, (x,), {}, 1, "declared")
    assert context.input_digest is None and context.state_digest is None
    assert context.preparation_digest == "d" * 64


def test_values_unavailable_declares_no_schema_default():
    """A real capture must stay byte-identical through every consumer. The host's
    portable-export check validates with Ajv `useDefaults`, which would write a
    declared default into the capture, so the field is optional with no default."""
    from services.semantic.contracts import CaptureSnapshot
    schema = CaptureSnapshot.model_json_schema()
    field = schema["properties"]["valuesUnavailable"]
    assert "default" not in field and "valuesUnavailable" not in schema["required"]
    assert {"const": "fake_tensor", "type": "string"} in field["anyOf"]
