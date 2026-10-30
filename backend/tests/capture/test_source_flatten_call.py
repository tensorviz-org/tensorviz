"""Observed flatten parameters must survive the real dispatcher and both readers."""
from copy import deepcopy
import gc
import json
from pathlib import Path
import weakref

import pytest
from pydantic import ValidationError
import torch

from services.semantic.contracts import CaptureSnapshot
from services.semantic.portable import portable_capture
from services.ingest.trace import trace_model
from tests.capture.test_source_shape_call import SOURCE
from tests.capture.test_semantic_construction import construct
from tests.source_review.test_semantic_source_bindings import run


@pytest.mark.parametrize("expression,start,end,paths", [
    ("h.flatten()", 0, -1, []),
    ("h.flatten(0)", 0, -1, [["args", 0]]),
    ("h.flatten(-2, -1)", -2, -1, [["args", 0], ["args", 1]]),
    ("h.flatten(start_dim=0, end_dim=1)", 0, 1, [["kwargs", "start_dim"], ["kwargs", "end_dim"]]),
    ("h.flatten(end_dim=1)", 0, 1, [["kwargs", "end_dim"]]),
    ("torch.flatten(h)", 0, -1, []),
    ("torch.flatten(h, 0, 1)", 0, 1, [["args", 1], ["args", 2]]),
    ("torch.flatten(input=h, start_dim=-2)", -2, -1, [["kwargs", "start_dim"]]),
    ("torch.flatten(h, 0, end_dim=-1)", 0, -1, [["args", 1], ["kwargs", "end_dim"]]),
])
def test_actual_flatten_calls_retain_supplied_parameters_and_defaults(tmp_path, expression, start, end, paths):
    model, context, _ = construct(tmp_path, SOURCE.replace("EXPRESSION", expression))
    capture = run(model, context)
    assert capture.status == "complete"
    operation, = capture.operations
    call = operation.sourceFlattenCall
    assert call is not None and operation.sourceShapeCall is None
    assert (call.startDim, call.endDim) == (start, end)
    assert [arg.sourcePath for arg in call.arguments] == paths
    assert call.output_dimensions([2, 4]) == [8]
    assert call.bindingId == operation.sourceBindingIds[0]
    assert portable_capture(capture).capture.operations[0].sourceFlattenCall == call


@pytest.mark.parametrize("expression", ["h.flatten(1)", "h.flatten(0, 0)", "h.flatten(*(0, 1))",
    "torch.flatten(**{'input': h})", "torch.Tensor.flatten(h)", "h.transpose(0, 1).flatten()"])
def test_noop_expansion_indirect_receivers_and_decompositions_have_no_flatten_proof(tmp_path, expression):
    model, context, _ = construct(tmp_path, SOURCE.replace("EXPRESSION", expression))
    capture = run(model, context)
    assert capture.status == "complete"
    assert all(op.sourceFlattenCall is None for op in capture.operations)


@pytest.mark.parametrize("shape", [(), (0, 4), (2, 3, 4)])
def test_scalar_empty_and_higher_rank_flattens_use_actual_output_dimensions(tmp_path, shape):
    model, context, _ = construct(tmp_path, SOURCE.replace("EXPRESSION", "h.flatten()"))
    result = trace_model(model, (torch.ones(shape),), tier="real", semantic_context=context)
    capture = CaptureSnapshot.model_validate(result.semantic_capture)
    assert capture.status == "complete"
    call = capture.operations[0].sourceFlattenCall
    assert call is not None
    assert call.output_dimensions(list(shape)) == list(torch.ones(shape).flatten().shape)


def test_flatten_proof_is_optional_bounded_and_does_not_retain_tensors(tmp_path, monkeypatch):
    model, context, _ = construct(tmp_path, SOURCE.replace("EXPRESSION", "h.flatten()"))
    value = torch.ones(2, 4); reference = weakref.ref(value)
    result = trace_model(model, (value,), tier="real", semantic_context=context)
    capture = CaptureSnapshot.model_validate(result.semantic_capture)
    assert capture.operations[0].sourceFlattenCall is not None
    del value, result
    gc.collect()
    assert reference() is None
    monkeypatch.setattr("services.semantic.source_shape_call.MAX_BYTES", 0)
    capture = run(model, context)
    assert capture.status == "complete" and capture.coverage.omissions == []
    assert "sourceFlattenCall" not in capture.model_dump(mode="json")["operations"][0]


@pytest.mark.parametrize("expression", ["h.flatten(2)", "h.flatten(1, 0)", "h.flatten(-3)"])
def test_failed_flatten_retains_the_actual_failure_without_proof(tmp_path, expression):
    model, context, _ = construct(tmp_path, SOURCE.replace("EXPRESSION", expression))
    result = trace_model(model, (torch.ones(2, 4),), tier="real", semantic_context=context)
    capture = CaptureSnapshot.model_validate(result.semantic_capture)
    assert result.error is not None and capture.status != "complete"
    assert all(op.sourceFlattenCall is None for op in capture.operations)


def test_rejected_flatten_dimension_does_not_invoke_user_protocols():
    from services.semantic.source_shape_call import flatten_call
    callbacks = []
    class Meta(type):
        def __eq__(cls, other):
            callbacks.append("equality")
            return False
        __hash__ = type.__hash__
    class Opaque(metaclass=Meta):
        def __index__(self):
            callbacks.append("index")
            return 0
    assert flatten_call({"id": "call:0"}, "flatten", (torch.ones(2, 4), Opaque()), {}) is None
    assert callbacks == []


def test_shared_flatten_fixture_refuses_each_contradiction():
    fixture = json.loads((Path(__file__).resolve().parents[3] / "tests/fixtures/contracts/source-flatten-call.json").read_text())
    for item in fixture["captures"]:
        capture = CaptureSnapshot.model_validate(item["portable"]["capture"])
        assert capture.status == "complete" and capture.operations[0].sourceFlattenCall is not None
    for case in fixture["cases"]:
        value = deepcopy(fixture["captures"][0]["portable"]["capture"])
        for patch in case["patches"]:
            target = value
            for key in patch["path"][:-1]: target = target[key]
            target[patch["path"][-1]] = patch["value"]
        with pytest.raises(ValidationError): CaptureSnapshot.model_validate(value)
