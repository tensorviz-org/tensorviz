"""Semantic records retain meaning before any display grouping or source edit."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from pydantic import TypeAdapter, ValidationError
import pytest

from scripts.export_validation_contracts import schema_document
from services.semantic.contracts import (
    Argument, CaptureSnapshot, RunAttempt, SemanticContracts, walk_arguments,
)

DIGEST = "a" * 64


def ref(name):
    return {"kind": "value", "id": name}


def value(name, producer=None, path=None, **changes):
    return {"id": name, "objectId": "object:" + name, "storageId": "storage:" + name,
            "version": 0, "logicalVersion": 0, "shape": [2, 4], "dtype": "float32", "device": "cpu",
            "requiresGrad": False, "aliasOf": None, "previousVersion": None,
            "stride": [4, 1], "storageOffset": 0, "state": [],
            "origin": "operation" if producer else "input",
            "producer": {"operationId": producer, "kind": "output", "path": path or []} if producer else None, **changes}


@pytest.fixture
def snapshot():
    inputs = {"args": [ref("h")], "kwargs": {}}
    ops = [{"id": "add1", "sequence": 1, "invocationId": "call:0", "target": "aten.add.Tensor",
            "schemaSignature": "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
            "arguments": {"args": [ref("h"), {"kind": "int", "value": "1"}], "kwargs": {}},
            "output": ref("a"), "effects": [], "status": "completed", "diagnosticId": None, "sourceBindingIds": []},
           {"id": "add2", "sequence": 2, "invocationId": "call:0", "target": "aten.add.Tensor",
            "schemaSignature": None, "arguments": {"args": [ref("a"), ref("a")], "kwargs": {}},
            "output": ref("b"), "effects": [], "status": "completed", "diagnosticId": None, "sourceBindingIds": []}]
    return {"version": 1, "id": "capture:0", "attemptId": "attempt:0",
            "source": {"version": 1, "id": "source:0", "workspaceSessionId": "session:0", "workspaceId": "workspace:0",
                       "modelId": "model:0", "modelFile": "model.py", "className": "Model", "generation": 0,
                       "files": [{"path": "model.py", "digest": DIGEST, "documentVersion": 1, "role": "source"}],
                       "dependencyScope": "complete", "configDigest": None, "environmentDigest": DIGEST, "sourceMode": "saved"},
            "planDigest": DIGEST, "preparationDigest": None, "automatic": None, "inputDigest": DIGEST, "stateDigest": DIGEST,
            "backend": "runtime", "backendVersion": "1", "torchVersion": "2.7.1",
            "normalizations": ["dispatcher operations"], "status": "complete", "rootInvocationId": "call:0",
            "inputs": inputs, "output": ref("b"), "definitions": [{"id": "definition:0", "qualifiedName": "model.Model", "sourceBindingIds": []}],
            "modules": [{"id": "module:0", "definitionId": "definition:0", "paths": [""]}],
            "invocations": [{"id": "call:0", "moduleId": "module:0", "parentId": None, "sequence": 0, "endSequence": 3,
                             "arguments": inputs, "output": ref("b"), "status": "completed", "training": False,
                             "gradEnabled": False, "diagnosticId": None, "sourceBindingIds": []}],
            "operations": ops, "values": [value("h"), value("a", "add1"), value("b", "add2")],
            "sourceBindings": [], "diagnostics": [],
            "coverage": {"scope": "executed_path", "operationsSeen": 2, "invocationsSeen": 1,
                         "omissions": [], "modules": [{"modulePath": "", "status": "observed", "invocationIds": ["call:0"], "reason": None}]},
            "limits": {"operations": 100, "invocations": 100, "values": 100, "bytes": 1000000}}


def test_repeated_operands_and_typed_literals_survive_json_roundtrip(snapshot):
    parsed = CaptureSnapshot.model_validate_json(json.dumps(snapshot))
    assert parsed.model_dump(exclude_unset=True) == snapshot
    args = list(walk_arguments(parsed.operations[1].arguments))
    assert [(path, argument.id) for path, argument in args] == [(('args', 0), 'a'), (('args', 1), 'a')]
    assert parsed.operations[0].arguments.args[1].value == "1"


def test_optional_metrics_distinguish_old_unknown_zero_and_unavailable(snapshot):
    parsed = CaptureSnapshot.model_validate(snapshot)
    assert parsed.values[0].logicalBytes is None
    assert parsed.invocations[0].parameterCount is None
    snapshot["values"][0]["logicalBytes"] = 0
    snapshot["invocations"][0]["parameterCount"] = 0
    parsed = CaptureSnapshot.model_validate(snapshot)
    assert parsed.values[0].logicalBytes == parsed.invocations[0].parameterCount == 0
    snapshot["invocations"][0]["parameterCountUnavailable"] = "item_limit"
    with pytest.raises(ValidationError, match="cannot also be unavailable"):
        CaptureSnapshot.model_validate(snapshot)


@pytest.mark.parametrize("number", [-1, True, 1.5, 2**53])
def test_metrics_require_exact_nonnegative_safe_integers(snapshot, number):
    snapshot["values"][0]["logicalBytes"] = number
    snapshot["invocations"][0]["parameterCount"] = number
    with pytest.raises(ValidationError) as error:
        CaptureSnapshot.model_validate(snapshot)
    assert {e["loc"] for e in error.value.errors()} == {
        ("values", 0, "logicalBytes"), ("invocations", 0, "parameterCount")}


def test_automatic_capture_has_a_preparation_instead_of_an_invented_plan(snapshot):
    snapshot.update(planDigest=None, preparationDigest=DIGEST,
                    automatic=dict(provenance="search-candidate", candidateIndex=2, candidatesTried=5))
    assert CaptureSnapshot.model_validate(snapshot).automatic.candidateIndex == 2
    snapshot["planDigest"] = DIGEST
    with pytest.raises(ValidationError, match="exactly one"):
        CaptureSnapshot.model_validate(snapshot)
    snapshot.update(planDigest=None, automatic=None)
    with pytest.raises(ValidationError, match="selected candidate"):
        CaptureSnapshot.model_validate(snapshot)
    snapshot["automatic"] = dict(provenance="search-candidate", candidateIndex=6, candidatesTried=5)
    with pytest.raises(ValidationError, match="must have been tried"):
        CaptureSnapshot.model_validate(snapshot)


def test_nested_outputs_have_distinct_exact_paths(snapshot):
    output = {"kind": "dict", "entries": [{"key": {"kind": "str", "value": "scales"}, "value":
              {"kind": "tuple", "items": [ref("b"), ref("c")]}}]}
    snapshot["operations"][1]["output"] = output
    snapshot["output"] = snapshot["invocations"][0]["output"] = output
    snapshot["values"][2]["producer"]["path"] = ["entries", 0, "value", "items", 0]
    snapshot["values"].append(value("c", "add2", ["entries", 0, "value", "items", 1]))
    parsed = CaptureSnapshot.model_validate(snapshot)
    assert len(parsed.values) == 4
    assert parsed.values[-1].producer.path != parsed.values[-2].producer.path
    snapshot["values"][-1]["producer"]["path"][-1] = 0
    with pytest.raises(ValidationError, match="producer output path"):
        CaptureSnapshot.model_validate(snapshot)


@pytest.mark.parametrize("change,match", [
    (lambda s: s["operations"][1]["arguments"]["args"].append(ref("missing")), "unknown value"),
    (lambda s: s["operations"][0]["arguments"]["args"].append(ref("b")), "precede its use"),
    (lambda s: s["values"].append(deepcopy(s["values"][0])), "duplicate semantic"),
    (lambda s: s["operations"][1].update(sequence=1), "unique ascending"),
    (lambda s: s["invocations"][0].update(parentId="call:0"), "earlier parent"),
    (lambda s: s["operations"][0].update(invocationId="missing"), "unknown containing"),
    (lambda s: s["values"][1]["producer"].update(operationId="missing"), "producer must precede"),
    (lambda s: s["values"][1].update(aliasOf="b"), "earlier value"),
    (lambda s: s["values"][1].update(aliasOf="h"), "share storage"),
    (lambda s: s["values"][1].update(previousVersion="h"), "same object's version"),
    (lambda s: s["source"].update(sourceMode="buffers"), "saved source"),
    (lambda s: s["coverage"]["modules"][0].update(status="not_observed"), "invocation evidence"),
    (lambda s: s["coverage"].update(operationsSeen=3), "omission reason"),
    (lambda s: s["coverage"].update(omissions=["operation_limit"]), "complete capture"),
    (lambda s: s["limits"].update(values=2), "semantic budget"),
    (lambda s: s["limits"].update(bytes=1024), "byte budget"),
    (lambda s: s.update(version=True), "version must be an integer"),
    (lambda s: s.update(unexpected=1), "Extra inputs"),
])
def test_invalid_semantics_cannot_be_published_as_valid(snapshot, change, match):
    change(snapshot)
    with pytest.raises(ValidationError, match=match):
        CaptureSnapshot.model_validate(snapshot)


def test_same_instance_can_have_multiple_invocations(snapshot):
    # Reusing the module is distinct from deduplicating its two calls.
    root = snapshot["invocations"][0]
    root["endSequence"] = 7
    for index, op in enumerate(snapshot["operations"]):
        call_id = f"nested:{index}"
        op.update(sequence=2 + index * 3, invocationId=call_id)
        snapshot["invocations"].append({**deepcopy(root), "id": call_id, "parentId": "call:0",
                                       "sequence": 1 + index * 3, "endSequence": 3 + index * 3,
                                       "arguments": deepcopy(op["arguments"]), "output": deepcopy(op["output"])})
    snapshot["coverage"]["invocationsSeen"] = 3
    snapshot["coverage"]["modules"][0]["invocationIds"] = ["call:0", "nested:0", "nested:1"]
    parsed = CaptureSnapshot.model_validate(snapshot)
    assert len(parsed.modules) == 1
    assert len(parsed.invocations) == 3
    assert parsed.operations[0].invocationId != parsed.operations[1].invocationId


def test_observed_call_cannot_be_relabelled_unobserved(snapshot):
    snapshot["coverage"]["modules"][0].update(status="not_observed", invocationIds=[])
    with pytest.raises(ValidationError, match="cannot omit observed"):
        CaptureSnapshot.model_validate(snapshot)


def test_aliases_and_mutations_are_not_new_independent_tensors(snapshot):
    snapshot["values"][1].update(storageId="storage:h", aliasOf="h")
    snapshot["values"][2].update(objectId="object:a", storageId="storage:h", aliasOf="a", previousVersion="a", version=1, logicalVersion=1)
    parsed = CaptureSnapshot.model_validate(snapshot)
    assert parsed.values[-1].previousVersion == "a"
    assert parsed.values[-1].objectId == parsed.values[-2].objectId


@pytest.fixture
def effect_snapshot(snapshot):
    op = snapshot["operations"][0]
    op.update(target="custom.bump", arguments={"args": [ref("h")], "kwargs": {}}, output={"kind": "none"},
              effects=[dict(kind="write", argumentPath=["args", 0], beforeId="h", after=ref("a"))])
    snapshot["values"][1].update(objectId="object:h", storageId="storage:h", previousVersion="h", logicalVersion=1,
                                producer=dict(operationId="add1", kind="effect", path=[0, "after"]))
    return snapshot


def test_nonreturned_effect_preserves_actual_and_logical_versions(effect_snapshot):
    capture = CaptureSnapshot.model_validate(effect_snapshot)
    before, after = capture.values[:2]
    assert before.version == after.version == 0
    assert after.logicalVersion == before.logicalVersion + 1
    assert capture.operations[0].output.kind == "none"
    assert capture.operations[1].arguments.args[0].id == after.id
    assert after.producer.kind == "effect"


@pytest.mark.parametrize("change,match", [
    (lambda s: s["operations"][0]["effects"][0].update(argumentPath=["args", 1]), "known writable argument"),
    (lambda s: s["operations"][0]["effects"][0].update(beforeId="b"), "exact argument value"),
    (lambda s: s["operations"][0]["effects"][0].update(after=ref("h")), "advance its prior value"),
    (lambda s: s["operations"][0]["effects"][0].update(after=ref("missing")), "unknown value"),
    (lambda s: s["values"][1]["producer"].update(path=[1, "after"]), "producer output path"),
    (lambda s: s["values"][1]["producer"].update(kind="output"), "producer output path"),
    (lambda s: s["values"][1].update(logicalVersion=0), "duplicate object/logical-version"),
    (lambda s: s["values"][1].update(logicalVersion=2), "same object's version"),
    (lambda s: s["operations"][0]["effects"].append(deepcopy(s["operations"][0]["effects"][0])), "duplicate mutation effect"),
    (lambda s: s["values"][0].update(version=1), "same object's version"),
    (lambda s: s["operations"][0]["effects"][0].update(kind="alias_update", beforeId="b"), "share its writable"),
])
def test_effects_cannot_invent_dataflow_or_mutation_versions(effect_snapshot, change, match):
    change(effect_snapshot)
    with pytest.raises(ValidationError, match=match):
        CaptureSnapshot.model_validate(effect_snapshot)


def test_opaque_argument_requires_partial_capture(snapshot):
    snapshot["operations"][0]["arguments"]["kwargs"]["custom"] = {"kind": "opaque", "typeName": "Custom", "reason": "unsupported_type"}
    with pytest.raises(ValidationError, match="complete capture"):
        CaptureSnapshot.model_validate(snapshot)
    snapshot["status"] = "partial"
    snapshot["coverage"]["omissions"] = ["unsupported_type"]
    assert CaptureSnapshot.model_validate(snapshot).status == "partial"


def test_forward_failure_keeps_the_prefix_and_diagnostic_without_fabricating_outputs(snapshot):
    diagnostic = {"id": "diagnostic:0", "phase": "forward", "code": "assertion_failed", "field": [],
                  "names": [], "sourceBindingId": None, "remedy": "inspect_failure"}
    snapshot["diagnostics"] = [diagnostic]
    snapshot["invocations"][0].update(status="failed", output=None, diagnosticId="diagnostic:0")
    snapshot.update(status="partial", output=None)
    snapshot["coverage"]["modules"][0]["status"] = "failed"
    parsed = CaptureSnapshot.model_validate(snapshot)
    assert len(parsed.operations) == 2
    assert parsed.output is None
    snapshot["invocations"][0]["diagnosticId"] = None
    with pytest.raises(ValidationError, match="failed invocations need"):
        CaptureSnapshot.model_validate(snapshot)


def test_source_binding_is_revision_specific(snapshot):
    snapshot["sourceBindings"] = [{"id": "binding:0", "snapshotId": "source:0", "kind": "argument", "quality": "exact",
        "anchors": [{"file": "model.py", "digest": DIGEST, "startLine": 1, "startColumn": 2,
                     "endLine": 1, "endColumn": 3, "scope": "Model.forward", "tokenDigest": DIGEST}]}]
    snapshot["operations"][0]["sourceBindingIds"] = ["binding:0"]
    CaptureSnapshot.model_validate(snapshot)
    snapshot["sourceBindings"][0]["anchors"][0]["digest"] = "b" * 64
    with pytest.raises(ValidationError, match="exact file digest"):
        CaptureSnapshot.model_validate(snapshot)


@pytest.mark.parametrize("argument", [
    {"kind": "int", "value": "9007199254740993"}, {"kind": "float", "value": "nan"},
    {"kind": "float", "value": 1.0}, {"kind": "symbolic", "type": "int", "expression": "s0 + 1"},
    {"kind": "none"}, {"kind": "bool", "value": True},
])
def test_literals_preserve_types_without_executing_expressions(argument):
    assert TypeAdapter(Argument).validate_json(json.dumps(argument)).model_dump() == argument


@pytest.mark.parametrize("argument", [
    {"kind": "int", "value": 1}, {"kind": "int", "value": "01"}, {"kind": "int", "value": "-0"},
    {"kind": "bool", "value": 1}, {"kind": "float", "value": True},
    {"kind": "dict", "entries": [{"key": {"kind": "bool", "value": True}, "value": {"kind": "none"}},
                                 {"key": {"kind": "int", "value": "1"}, "value": {"kind": "none"}}]},
])
def test_lossy_literals_and_duplicate_mapping_keys_refuse(argument):
    with pytest.raises(ValidationError):
        TypeAdapter(Argument).validate_json(json.dumps(argument))


def test_timeout_is_terminal_and_can_retain_no_new_capture():
    attempt = {"version": 1, "id": "attempt:2", "sourceId": "source:0", "planDigest": DIGEST,
               "origin": "manual", "requestDigest": DIGEST,
               "phase": "forward", "status": "timed_out", "startedAt": 10, "finishedAt": 20,
               "captureIds": [], "diagnostics": []}
    assert RunAttempt.model_validate(attempt).status == "timed_out"
    attempt["finishedAt"] = None
    with pytest.raises(ValidationError, match="terminal attempts"):
        RunAttempt.model_validate(attempt)


def test_a_reused_capture_is_one_of_the_attempts_captures():
    attempt = {"version": 1, "id": "attempt:3", "sourceId": "source:3", "planDigest": DIGEST,
               "origin": "manual", "requestDigest": DIGEST, "phase": "capture", "status": "succeeded",
               "startedAt": 10, "finishedAt": 20, "captureIds": ["capture:1"], "diagnostics": []}
    assert RunAttempt.model_validate(attempt).reusedCaptureId is None, "an attempt that executed reuses nothing"
    attempt["reusedCaptureId"] = "capture:1"
    assert RunAttempt.model_validate(attempt).reusedCaptureId == "capture:1"
    attempt["reusedCaptureId"] = "capture:other"
    with pytest.raises(ValidationError, match="reused capture"):
        RunAttempt.model_validate(attempt)


def test_setup_refusal_does_not_invent_an_executable_plan():
    attempt = {"version": 1, "id": "attempt:setup", "origin": "manual", "requestDigest": DIGEST,
               "sourceId": None, "planDigest": None, "phase": "configuration", "status": "needs_setup",
               "startedAt": 1, "finishedAt": 2, "captureIds": [], "diagnostics": []}
    assert RunAttempt.model_validate(attempt).planDigest is None
    attempt["sourceId"] = "source:0"
    assert RunAttempt.model_validate(attempt).sourceId == "source:0", "preparation knows source before choosing inputs"
    attempt.update(sourceId=None, planDigest=DIGEST)
    with pytest.raises(ValidationError, match="requires source"):
        RunAttempt.model_validate(attempt)


def test_exported_schema_matches_python_and_both_consumers():
    root = Path(__file__).parent.resolve().parents[2]
    schema = root / "tests/fixtures/contracts/semantic.schema.json"
    assert schema.read_text() == json.dumps(schema_document(SemanticContracts), indent=2, sort_keys=True) + "\n"
    assert (root / "packages/contracts/src/semanticContracts.generated.ts").read_bytes() == (root / "canvas/src/types/semanticContracts.generated.ts").read_bytes()


def test_host_capture_fixture_is_the_same_validated_canonical_snapshot(snapshot):
    root = Path(__file__).parent.resolve().parents[2]
    fixture = json.loads((root / "tests/fixtures/contracts/semantic-capture.json").read_text())
    for value in snapshot["values"]:
        value["logicalBytes"] = 32
    snapshot["invocations"][0]["parameterCount"] = 0
    assert fixture == snapshot
    assert CaptureSnapshot.model_validate(fixture).status == "complete"


def test_contract_import_never_imports_torch_or_workspace_code():
    process = subprocess.run([sys.executable, "-B", "-c",
                              "import sys; from services.semantic.contracts import SemanticContracts; "
                              "from services.semantic.source_join import CaptureJoin; "
                              "from services.semantic.projection import project_capture; assert 'torch' not in sys.modules"],
                             cwd=Path(__file__).parent.resolve().parents[1], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
