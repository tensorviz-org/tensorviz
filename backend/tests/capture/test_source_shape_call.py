"""Actual source/call/operand correspondence, not shape equality alone."""
from copy import deepcopy
from dataclasses import replace
import gc
import json
from pathlib import Path
import weakref

import pytest
from pydantic import ValidationError
import torch

from services.semantic.contracts import CaptureSnapshot
from services.semantic.portable import portable_capture
from tests.capture.test_semantic_construction import construct
from tests.source_review.test_semantic_source_bindings import run

SOURCE = '''import torch
class Model(torch.nn.Module):
    def forward(self, h):
        return EXPRESSION
def create():
    return Model()
'''


@pytest.mark.parametrize("expression,callable,paths", [
    ("torch.reshape(h, (4, 2))", "torch.reshape", [["args", 1, "items", 0], ["args", 1, "items", 1]]),
    ("torch.reshape(input=h, shape=[4, 2])", "torch.reshape", [["kwargs", "shape", "items", 0], ["kwargs", "shape", "items", 1]]),
    ("torch.reshape(h, shape=(4, 2))", "torch.reshape", [["kwargs", "shape", "items", 0], ["kwargs", "shape", "items", 1]]),
    ("h.reshape(4, 2)", "torch.Tensor.reshape", [["args", 0], ["args", 1]]),
    ("h.reshape([4, 2])", "torch.Tensor.reshape", [["args", 0, "items", 0], ["args", 0, "items", 1]]),
    ("h.view(4, -1)", "torch.Tensor.view", [["args", 0], ["args", 1]]),
    ("h.view((4, 2))", "torch.Tensor.view", [["args", 0, "items", 0], ["args", 0, "items", 1]]),
])
def test_direct_calls_map_every_observed_dimension_to_its_source_argument(tmp_path, expression, callable, paths):
    model, context, _ = construct(tmp_path, SOURCE.replace("EXPRESSION", expression))
    capture = run(model, context)
    assert capture.status == "complete"
    operation, = capture.operations
    proof = operation.sourceShapeCall
    assert proof is not None
    assert proof.callable == callable
    assert proof.bindingId == operation.sourceBindingIds[0]
    assert [item.sourcePath for item in proof.dimensions] == paths
    assert [item.value for item in proof.dimensions] == operation.arguments.args[1].items
    assert [item.dimension for item in proof.dimensions] == [0, 1]
    assert portable_capture(capture.model_dump(mode="json")).capture.operations[0].sourceShapeCall == proof


@pytest.mark.parametrize("expression", [
    "h.view(*(4, 2))", "torch.reshape(**{'input': h, 'shape': (4, 2)})",
    "torch.Tensor.view(h, 4, 2)", "(lambda x: x.view(4, 2))(h)",
    "h.transpose(0, 1).reshape(4, 2)", "h.flatten()",
])
def test_unsupported_or_expanded_source_cannot_inherit_dimension_correspondence(tmp_path, expression):
    model, context, _ = construct(tmp_path, SOURCE.replace("EXPRESSION", expression))
    capture = run(model, context)
    assert capture.status == "complete"
    assert all(operation.sourceShapeCall is None for operation in capture.operations)


def test_scalar_tuple_and_noncontiguous_decomposition_are_not_confused(tmp_path):
    # Reshape on a directly named noncontiguous input executes clone + unsafe_view.
    source = SOURCE.replace("EXPRESSION", "h.reshape(8)")
    model, context, _ = construct(tmp_path, source)
    from services.ingest.trace import trace_model
    result = trace_model(model, (torch.ones(4, 2).t(),), tier="real", semantic_context=context)
    capture = CaptureSnapshot.model_validate(result.semantic_capture)
    assert capture.status == "complete"
    assert len(capture.operations) > 1
    assert all(operation.sourceShapeCall is None for operation in capture.operations)


@pytest.mark.parametrize("limit", ["MAX_CALLS", "MAX_BYTES"])
def test_optional_provenance_limits_preserve_the_completed_computation(tmp_path, monkeypatch, limit):
    model, context, _ = construct(tmp_path, SOURCE.replace("EXPRESSION", "h.view(4, 2)"))
    monkeypatch.setattr("services.semantic.source_shape_call." + limit, 0)
    capture = run(model, context)
    assert capture.status == "complete" and capture.coverage.omissions == []
    assert capture.operations[0].sourceShapeCall is None
    assert "sourceShapeCall" not in capture.model_dump(mode="json")["operations"][0]


def test_dimension_and_source_identity_must_survive_contract_validation(tmp_path):
    model, context, _ = construct(tmp_path, SOURCE.replace("EXPRESSION", "h.view(4, 2)"))
    original = run(model, context).model_dump(mode="json")
    for mutate in (
        lambda x: x["operations"][0]["sourceShapeCall"].update(bindingId="wrong"),
        lambda x: x["operations"][0]["sourceShapeCall"]["dimensions"][0].update(dimension=1),
        lambda x: x["operations"][0]["sourceShapeCall"]["dimensions"][0].update(sourcePath=["args", -1]),
        lambda x: x["operations"][0]["sourceShapeCall"]["dimensions"][0]["value"].update(value="7"),
        lambda x: x["operations"][0]["sourceShapeCall"]["dimensions"].pop(),
        lambda x: x["sourceBindings"][-1].update(quality="candidate"),
    ):
        value = deepcopy(original); mutate(value)
        with pytest.raises(ValidationError): CaptureSnapshot.model_validate(value)


@pytest.mark.parametrize("expression", ["h.view(4, 2)", "h.view(h.shape[-1], h.shape[0])"])
def test_observation_does_not_retain_input_tensors_or_source_text(tmp_path, expression):
    model, context, _ = construct(tmp_path, SOURCE.replace("EXPRESSION", expression))
    from services.ingest.trace import trace_model
    value = torch.ones(2, 4); reference = weakref.ref(value)
    result = trace_model(model, (value,), tier="real", semantic_context=context)
    capture = CaptureSnapshot.model_validate(result.semantic_capture)
    proof = capture.operations[0].sourceShapeCall
    assert proof is not None
    assert "return" not in json.dumps(proof.model_dump(mode="json"))
    del value, result
    gc.collect()
    assert reference() is None


def test_failed_view_retains_failure_without_argument_proof(tmp_path):
    model, context, _ = construct(tmp_path, SOURCE.replace("EXPRESSION", "h.view(3, 3)"))
    from services.ingest.trace import trace_model
    result = trace_model(model, (torch.ones(2, 4),), tier="real", semantic_context=context)
    capture = CaptureSnapshot.model_validate(result.semantic_capture)
    assert result.error is not None and capture.status != "complete"
    assert any(operation.status == "failed" for operation in capture.operations)
    assert all(operation.sourceShapeCall is None for operation in capture.operations)


def test_direct_builtin_alias_keeps_callable_identity(tmp_path):
    source = SOURCE.replace("class Model", "reshape = torch.reshape\nclass Model").replace("EXPRESSION", "reshape(h, (4, 2))")
    model, context, _ = construct(tmp_path, source)
    assert run(model, context).operations[0].sourceShapeCall.callable == "torch.reshape"


@pytest.mark.parametrize("expression", ["h.view(h.shape[0], h.shape[-1])",
    "torch.reshape(h, [h.shape[+0], h.shape[1]])"])
def test_direct_tensor_shape_reads_reference_prior_captured_values(tmp_path, expression):
    model, context, _ = construct(tmp_path, SOURCE.replace("EXPRESSION", expression))
    capture = run(model, context)
    call = capture.operations[0].sourceShapeCall
    assert call is not None
    assert [d.tensorDimension.valueId for d in call.dimensions] == [capture.inputs.args[0].id] * 2
    assert [d.tensorDimension.axis for d in call.dimensions] == ([0, -1] if expression.startswith("h.view") else [0, 1])
    assert portable_capture(capture).capture.operations[0].sourceShapeCall == call


def test_shape_read_can_refer_to_a_different_tensor_without_guessing_by_shape(tmp_path):
    source = SOURCE.replace("return EXPRESSION", "x = h + 0\n        return h.view(x.shape[0], x.shape[1])")
    model, context, _ = construct(tmp_path, source)
    capture = run(model, context)
    addition, view = capture.operations
    assert addition.output.id != capture.inputs.args[0].id
    assert all(d.tensorDimension.valueId == addition.output.id for d in view.sourceShapeCall.dimensions)
    original = capture.model_dump(mode="json")
    for ref in ({"valueId": "absent", "axis": 0}, {"valueId": addition.output.id, "axis": 32},
                {"valueId": addition.output.id, "axis": -3}, {"valueId": addition.output.id, "axis": 1},
                {"valueId": view.output.id, "axis": 0}):
        payload = deepcopy(original)
        payload["operations"][1]["sourceShapeCall"]["dimensions"][0]["tensorDimension"] = ref
        with pytest.raises(ValidationError): CaptureSnapshot.model_validate(payload)


def test_shape_read_origin_is_observed_before_a_later_argument_rebinds_its_name(tmp_path):
    source = '''import torch
READ = torch.ones(2, 4)
OTHER = torch.ones(2, 4)
def replace():
    global READ
    READ = OTHER
    return 4
class Model(torch.nn.Module):
    def forward(self, h):
        h = h + READ * 0
        y = OTHER + 0
        return h.view(READ.shape[0], replace())
def create(): return Model()
'''
    model, context, _ = construct(tmp_path, source)
    capture = run(model, context)
    first, _, other, view = capture.operations
    assert first.arguments.args[0].id != other.arguments.args[0].id
    read = view.sourceShapeCall.dimensions[0].tensorDimension
    assert read is not None and read.valueId == first.arguments.args[0].id


@pytest.mark.parametrize("body", [
    "for _ in range(2): h = h.view(h.shape[0], 4)\n        return h",
    "if h is not None: return h.view(h.shape[0], 4)",
    "return helper(h)",
    "return Block().forward(h)",
])
def test_shape_read_proof_excludes_repeated_conditional_and_unobserved_frames(tmp_path, body):
    helpers = '''def helper(h): return h.view(h.shape[0], 4)
class Block(torch.nn.Module):
    def forward(self, h): return h.view(h.shape[0], 4)
'''
    source = SOURCE.replace("class Model", helpers + "class Model").replace("return EXPRESSION", body)
    model, context, _ = construct(tmp_path, source)
    capture = run(model, context)
    assert capture.status == "complete"
    calls = [op.sourceShapeCall for op in capture.operations if op.sourceShapeCall]
    assert calls and all(d.tensorDimension is None for call in calls for d in call.dimensions)


def test_repeated_module_calls_keep_each_read_in_its_own_invocation(tmp_path, monkeypatch):
    from services.semantic.recorder import SemanticRecorder
    pending, original = [], SemanticRecorder.snapshot
    def snapshot(recorder):
        pending.append((len(recorder.source_shape_reads), recorder.source_shape_read_bytes))
        return original(recorder)
    monkeypatch.setattr(SemanticRecorder, "snapshot", snapshot)
    source = '''import torch
class Block(torch.nn.Module):
    def forward(self, h): return h.view(h.shape[0], 4)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
    def forward(self, h):
        x = self.block(h)
        y = x + 1
        return self.block(y)
def create(): return Model()
'''
    model, context, _ = construct(tmp_path, source)
    capture = run(model, context)
    first, add, second = capture.operations
    assert first.sourceShapeCall.dimensions[0].tensorDimension.valueId == capture.inputs.args[0].id
    assert second.sourceShapeCall.dimensions[0].tensorDimension.valueId == add.output.id
    assert pending == [(0, 0)]


@pytest.mark.parametrize("limit", ["MAX_READ_CALLS", "MAX_READ_BYTES"])
def test_shape_read_limits_remove_only_optional_provenance(tmp_path, monkeypatch, limit):
    model, context, _ = construct(tmp_path, SOURCE.replace("EXPRESSION", "h.view(h.shape[0], 4)"))
    monkeypatch.setattr("services.semantic.source_shape_call." + limit, 0)
    capture = run(model, context)
    assert capture.status == "complete" and capture.coverage.omissions == []
    assert capture.operations[0].sourceShapeCall is not None
    assert all(d.tensorDimension is None for d in capture.operations[0].sourceShapeCall.dimensions)


@pytest.mark.parametrize("expression,setup", [
    ("h.view(other.shape[0], 4)", "other = torch.ones(2, 4)\n"),
    ("h.view(other.shape[0], 4)", "class Other:\n    shape = (2, 4)\nother = Other()\n"),
    ("h.view([h][0].shape[0], 4)", ""),
    ("h.view(h.size(0), 4)", ""),
])
def test_unobserved_or_unsupported_shape_sources_do_not_gain_value_correspondence(tmp_path, expression, setup):
    source = SOURCE.replace("class Model", setup + "class Model").replace("EXPRESSION", expression)
    model, context, _ = construct(tmp_path, source)
    capture = run(model, context)
    call = capture.operations[-1].sourceShapeCall
    assert call is not None and all(d.tensorDimension is None for d in call.dimensions)


@pytest.mark.parametrize("attribute", ["shape", "__getattribute__"])
def test_shape_read_observation_never_calls_replaced_attribute_protocols(monkeypatch, attribute):
    from types import SimpleNamespace
    from services.semantic import source_shape_call
    value = torch.ones(2, 4)
    callbacks = []
    def getter(self, *args):
        callbacks.append("shape")
        return (2, 4)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, attribute, property(getter) if attribute == "shape" else getter)
        assert source_shape_call.tensor_read(SimpleNamespace(), value, 0) is None
    assert callbacks == []


def test_optional_metadata_cannot_displace_operations_at_final_byte_limit(tmp_path, monkeypatch):
    from services.semantic.recorder import SemanticRecorder
    accounted, original = [], SemanticRecorder.snapshot
    def snapshot(recorder):
        result = original(recorder)
        accounted.append(recorder.bytes)
        return result
    monkeypatch.setattr(SemanticRecorder, "snapshot", snapshot)
    source = SOURCE.replace("return EXPRESSION", "for _ in range(80): h = h.view(2, 4)\n        return h")
    model, context, _ = construct(tmp_path, source)
    with monkeypatch.context() as patch:
        patch.setattr("services.semantic.source_shape_call.MAX_CALLS", 0)
        baseline = run(model, context)
        context = replace(context, limits=context.limits.model_copy(update={"bytes": accounted[-1] + 128}))
        baseline = run(model, context)
    capture = run(model, context)
    assert len(capture.operations) == len(baseline.operations) == 80
    assert capture.status == "complete" and not capture.coverage.omissions
    assert len(capture.model_dump_json().encode()) <= context.limits.bytes
    assert any(operation.sourceShapeCall is None for operation in capture.operations)


def test_shared_reader_fixture_is_real_complete_evidence_and_refuses_each_contradiction():
    from services.semantic.contracts import PortableCapture
    fixture = json.loads((Path(__file__).resolve().parents[3] / "tests/fixtures/contracts/source-shape-call.json").read_text())
    assert CaptureSnapshot.model_validate(fixture["portable"]["capture"]).status == "complete"
    assert PortableCapture.model_validate(fixture["portable"]).capture.operations[0].sourceShapeCall is not None
    for case in fixture["cases"]:
        value = deepcopy(fixture["portable"]["capture"])
        for patch in case["patches"]:
            target = value
            for key in patch["path"][:-1]: target = target[key]
            target[patch["path"][-1]] = patch["value"]
        with pytest.raises(ValidationError, match="."):
            CaptureSnapshot.model_validate(value)


@pytest.mark.parametrize("boundary", ["callable", "receiver", "dimension"])
def test_unsupported_types_do_not_invoke_metaclass_equality(boundary, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from services.semantic import source_shape_call
    callbacks = []
    class Meta(type):
        def __eq__(cls, other):
            callbacks.append("equality")
            return False
        __hash__ = type.__hash__
    class Unsupported(metaclass=Meta): pass
    opaque = Unsupported()
    recorder = SimpleNamespace(sources=SimpleNamespace(direct_shape_call=lambda *args: ({"id": "call:0"}, 0)),
        source_shape_calls={}, source_shape_reads={}, stack=[], mute=nullcontext)
    func, receiver, dimension = torch.Tensor.view, torch.ones(2, 4), 8
    if boundary == "callable":
        func = opaque
        monkeypatch.setattr(source_shape_call, "_RESHAPE", func)
    if boundary == "receiver": receiver = opaque
    if boundary == "dimension": dimension = opaque
    assert source_shape_call.begin(recorder, func, (receiver, dimension), {}) is None
    assert callbacks == [], "Rejecting optional metadata must not call user protocols."
