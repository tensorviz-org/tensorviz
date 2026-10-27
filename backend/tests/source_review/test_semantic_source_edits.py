"""Capabilities prove source scope and operand identity before native review."""
from copy import deepcopy
import hashlib

import pytest
from pydantic import ValidationError

from services.semantic.contracts import LiteralEditCapability, SetLiteralEdit
from services.semantic.source_edits import CapturedEditResolver, SourceEditRefused
from tests.source_review.test_semantic_source_bindings import prepared, run


SOURCE = '''import torch
class Model(torch.nn.Module):
    def forward(self, h):
        a = h + 1  # preserve me
        b = a + a
        return b * 2
'''


def setup(tmp_path, source=SOURCE):
    namespace, context = prepared(tmp_path, source)
    record = run(namespace["Model"](), context)
    return record, CapturedEditResolver(record, {"model.py": source})


def intent(capability, value=None):
    return dict(kind="set_literal", capabilityId=capability.id,
        target=capability.target.model_dump(mode="json"), argumentPath=capability.argumentPath,
        expected=capability.expected.model_dump(mode="json"), value=value or {"kind": "int", "value": "2"},
        scope="source_expression")


def apply(source, result):
    after = source
    for edit in sorted(result["edits"], key=lambda edit: edit['range']['start'], reverse=True):
        lines = after.split("\n")
        start, end = edit["range"]["start"], edit["range"]["end"]
        assert start[0] == end[0]
        line = lines[start[0] - 1].encode("utf-16-le")
        lines[start[0] - 1] = (line[:start[1] * 2] + edit["newText"].encode("utf-16-le") + line[end[1] * 2:]).decode("utf-16-le")
        after = "\n".join(lines)
    assert hashlib.sha256(after.encode()).hexdigest() == result["source_digest"]
    return after


def test_exact_rhs_changes_no_other_bytes_and_recaptures_repeated_operand_unchanged(tmp_path):
    record, resolver = setup(tmp_path)
    cap = resolver.literal_capability(record.operations[0].id)
    assert cap.argumentPath == ["args", 1]
    assert cap.binding.kind == "argument" and cap.binding.anchors[0].startLine == 4
    assert cap.expression.anchors[0].tokenDigest != cap.binding.anchors[0].tokenDigest
    assert cap.scope.affects == "all_executions" and cap.scope.includesUnobservedExecutions
    result = resolver.plan_literal(intent(cap))
    assert result["validation"] == "static_only"
    after = apply(SOURCE, result)
    assert after == SOURCE.replace("h + 1", "h + 2")
    new, _ = setup(tmp_path, after)
    assert new.operations[0].arguments.args[1].value == "2"
    assert new.operations[1].arguments.args[0] == new.operations[1].arguments.args[1]
    assert new.operations[2].arguments == record.operations[2].arguments


@pytest.mark.parametrize("expression,target", [("h - 2", "aten.sub.Tensor"),
    ("h * 2", "aten.mul.Tensor"), ("h / 2", "aten.div.Tensor")])
def test_each_binary_recipe_maps_the_actual_second_argument(tmp_path, expression, target):
    source = SOURCE.replace("h + 1", expression)
    record, resolver = setup(tmp_path, source)
    cap = resolver.literal_capability(record.operations[0].id)
    assert record.operations[0].target == target and cap.expected.value == "2"
    assert apply(source, resolver.plan_literal(intent(cap, {"kind": "int", "value": "3"}))) == source.replace(expression, expression[:-1] + "3")


def test_sharing_counts_observations_and_never_specializes_one_loop_iteration_or_instance(tmp_path):
    source = '''import torch
class Block(torch.nn.Module):
    def forward(self, h):
        for _ in range(2):
            h = h + 1
        return h
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = Block(), Block()
    def forward(self, h):
        return self.b(self.a(h))
'''
    record, resolver = setup(tmp_path, source)
    cap = resolver.literal_capability(record.operations[2].id)
    assert cap.scope.observedOperations == 4
    assert cap.scope.observedInvocations == 2
    assert cap.scope.observedModuleInstances == 2
    after = apply(source, resolver.plan_literal(intent(cap)))
    new, _ = setup(tmp_path, after)
    assert all(op.arguments.args[1].value == "2" for op in new.operations)
    forged = intent(cap)
    forged["scope"] = "selected_invocation"
    with pytest.raises(ValidationError):
        resolver.plan_literal(forged)


@pytest.mark.parametrize("old,new", [("0x01", {"kind": "int", "value": "1"}),
    ("+ 1", {"kind": "int", "value": "1"}), ("1_000", {"kind": "int", "value": "1000"}),
    ("1.00", {"kind": "float", "value": 1.0}), ("-0.0", {"kind": "float", "value": "-0"})])
def test_noop_preserves_literal_spelling_and_every_byte(tmp_path, old, new):
    source = SOURCE.replace("h + 1", "h + " + old)
    record, resolver = setup(tmp_path, source)
    cap = resolver.literal_capability(record.operations[0].id)
    result = resolver.plan_literal(intent(cap, new))
    assert result["edits"] == []
    assert apply(source, result) == source


def test_physical_line_utf16_and_crlf_survive_precise_patch(tmp_path):
    source = SOURCE.replace("a = h + 1", 'label = "😀\u2028"; a = h + 1').replace("\n", "\r\n")
    record, resolver = setup(tmp_path, source)
    cap = resolver.literal_capability(record.operations[0].id)
    assert cap.binding.anchors[0].startLine == 4
    assert apply(source, resolver.plan_literal(intent(cap))) == source.replace("h + 1", "h + 2")


@pytest.mark.parametrize("expression", ["h + h", "1 + h", "h + (1 + 2)",
    "h + float(1)", "h.add(1)", "h // 2", "h + True", "torch.nn.functional.gelu(h)"])
def test_unsupported_expressions_do_not_infer_source_args_from_aten_positions(tmp_path, expression):
    record, resolver = setup(tmp_path, SOURCE.replace("h + 1", expression))
    with pytest.raises(SourceEditRefused, match="unsupported_expression"):
        resolver.literal_capability(record.operations[0].id)


def test_multiline_signed_literal_keeps_comments_by_refusing_the_simple_span_recipe(tmp_path):
    source = SOURCE.replace("h + 1", "h + -(\n            # units\n            1\n        )")
    record, resolver = setup(tmp_path, source)
    with pytest.raises(SourceEditRefused, match="unsupported_expression"):
        resolver.literal_capability(record.operations[0].id)


@pytest.mark.parametrize("field,value", [("capabilityId", "other"), ("argumentPath", ["args", 0]),
    ("expected", {"kind": "int", "value": "999"})])
def test_stale_capability_expected_value_or_operand_cannot_create_patch(tmp_path, field, value):
    record, resolver = setup(tmp_path)
    request = intent(resolver.literal_capability(record.operations[0].id))
    request[field] = value
    with pytest.raises(SourceEditRefused, match="stale_capability"):
        resolver.plan_literal(request)


@pytest.mark.parametrize("field", ["captureId", "sourceId", "kind"])
def test_exact_base_identity_is_required(tmp_path, field):
    record, resolver = setup(tmp_path)
    request = intent(resolver.literal_capability(record.operations[0].id))
    request["target"][field] = "invocation" if field == "kind" else "other"
    with pytest.raises(SourceEditRefused, match="stale_capability"):
        resolver.plan_literal(request)


def test_saved_source_changes_missing_source_and_forged_token_anchor_are_refused(tmp_path):
    record, _ = setup(tmp_path)
    with pytest.raises(SourceEditRefused, match="stale_source"):
        CapturedEditResolver(record, {"model.py": SOURCE.replace("h + 1", "h + 2")})
    with pytest.raises(SourceEditRefused, match="source_unavailable"):
        CapturedEditResolver(record, {}).literal_capability(record.operations[0].id)
    payload = record.model_dump(mode="json")
    binding = next(b for b in payload["sourceBindings"] if b["id"] == record.operations[0].sourceBindingIds[0])
    binding["anchors"][0]["tokenDigest"] = "f" * 64
    with pytest.raises(SourceEditRefused, match="source_binding_mismatch"):
        CapturedEditResolver(payload, {"model.py": SOURCE}).literal_capability(record.operations[0].id)


def test_decomposition_sharing_binding_but_different_runtime_args_is_not_independently_editable(tmp_path):
    record, _ = setup(tmp_path)
    payload = record.model_dump(mode="json")
    payload["operations"][1]["sourceBindingIds"] = payload["operations"][0]["sourceBindingIds"]
    with pytest.raises(SourceEditRefused, match="derived_operation"):
        CapturedEditResolver(payload, {"model.py": SOURCE}).literal_capability(record.operations[0].id)


def test_source_budgets_cannot_silently_enable_unverified_locations(tmp_path, monkeypatch):
    record, _ = setup(tmp_path)
    monkeypatch.setattr("services.semantic.source_edits.MAX_SITES", 0)
    with pytest.raises(SourceEditRefused, match="source_limit"):
        CapturedEditResolver(record, {"model.py": SOURCE}).literal_capability(record.operations[0].id)
    monkeypatch.setattr("services.semantic.source_edits.MAX_FILE_BYTES", 1)
    with pytest.raises(SourceEditRefused, match="source_limit"):
        CapturedEditResolver(record, {"model.py": SOURCE})


def test_resolver_pins_its_base_and_does_not_mutate_the_capture_or_source_map(tmp_path):
    record, _ = setup(tmp_path)
    sources = {"model.py": SOURCE}
    resolver = CapturedEditResolver(record, sources)
    cap = resolver.literal_capability(record.operations[0].id)
    sources["model.py"] = "changed"
    record.source.id = "changed"
    assert resolver.literal_capability(cap.target.semanticId) == cap
    assert resolver.plan_literal(intent(cap))["expected_digest"] == hashlib.sha256(SOURCE.encode()).hexdigest()


def test_capability_contract_rejects_cross_snapshot_and_noncontained_argument_bindings(tmp_path):
    record, resolver = setup(tmp_path)
    payload = resolver.literal_capability(record.operations[0].id).model_dump(mode="json")
    for change in [lambda p: p["binding"].update(snapshotId="other"),
                   lambda p: p["binding"].update(quality="candidate"),
                   lambda p: p.update(argumentPath=["args", 0]),
                   lambda p: p["binding"]["anchors"][0].update(startLine=5, endLine=5),
                   lambda p: p["binding"]["anchors"][0].update(file="other.py")]:
        altered = deepcopy(payload)
        change(altered)
        with pytest.raises(ValidationError):
            LiteralEditCapability.model_validate(altered)
    for value in [{"kind": "float", "value": "inf"}, {"kind": "int", "value": "1 + danger()"},
                  {"kind": "float", "value": float("nan")}, {"kind": "bool", "value": True}]:
        with pytest.raises(ValidationError):
            SetLiteralEdit.model_validate(intent(LiteralEditCapability.model_validate(payload), value))
