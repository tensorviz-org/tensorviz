"""Portable capture preparation preserves computation, not live workspace authority."""
from copy import deepcopy
import subprocess
import sys
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from services.semantic.contracts import PortableCapture, CaptureSnapshot
from services.semantic.portable import portable_capture
from tests.capture.test_semantic_contracts import snapshot, effect_snapshot


def test_computation_is_exact_while_source_names_and_host_identity_are_detached(snapshot):
    snapshot["source"].update(workspaceSessionId="PRIVATE_SESSION", workspaceId="PRIVATE_WORKSPACE", modelId="PRIVATE_MODEL",
                              modelFile="secret_project/private_model.py")
    snapshot["source"]["files"][0]["path"] = "secret_project/private_model.py"
    snapshot["sourceBindings"] = [dict(id="binding:0", snapshotId=snapshot["source"]["id"], kind="argument", quality="exact",
        anchors=[dict(file="secret_project/private_model.py", digest="a" * 64, startLine=10, startColumn=4,
                      endLine=10, endColumn=5, scope="Model.forward", tokenDigest="b" * 64)])]
    snapshot["operations"][0]["sourceBindingIds"] = ["binding:0"]
    original = deepcopy(snapshot)
    portable = portable_capture(snapshot)
    assert snapshot == original
    assert portable.sourceNames == "anonymized"
    for field in ("inputs", "output", "operations", "invocations", "values", "modules", "definitions", "coverage", "diagnostics"):
        assert getattr(portable.capture, field) == getattr(CaptureSnapshot.model_validate(original), field)
    assert "secret_project" not in portable.model_dump_json()
    assert "PRIVATE_" not in portable.model_dump_json()
    assert portable.capture.source.files[0].path == "file-0.py"
    assert portable.capture.source.files[0].documentVersion is None
    assert portable.capture.sourceBindings[0].anchors[0].file == "file-0.py"
    assert portable.capture.sourceBindings[0].snapshotId == portable.capture.source.id
    assert PortableCapture.model_validate_json(portable.model_dump_json()) == portable
    assert portable_capture(snapshot) == portable


def test_known_config_identity_survives_without_user_paths(snapshot):
    snapshot["source"]["configDigest"] = "b" * 64
    snapshot["source"]["files"].append(dict(path="tensorviz.toml", role="config", digest="b" * 64, documentVersion=7))
    portable = portable_capture(snapshot)
    assert [f.path for f in portable.capture.source.files] == ["file-0.py", "tensorviz.toml"]
    assert portable.capture.source.configDigest == "b" * 64


@pytest.mark.parametrize("argument", [
    dict(kind="str", value="/Users/private/model.py"), dict(kind="str", value="C:\\private\\model.py"),
    dict(kind="str", value="file:///private/model.py"),
    dict(kind="dict", entries=[dict(key=dict(kind="str", value="password"), value=dict(kind="str", value="hidden"))]),
])
def test_private_literal_refuses_instead_of_losing_an_operand(snapshot, argument):
    snapshot["operations"][0]["arguments"]["args"][1] = argument
    original = deepcopy(snapshot)
    with pytest.raises(ValueError, match="lossless export refused"):
        portable_capture(snapshot)
    assert snapshot == original


def test_named_tensor_sources_stay_semantic_references_without_exporting_payloads(snapshot):
    snapshot["operations"][0]["arguments"]["kwargs"]["source"] = dict(kind="value", id="h")
    portable = portable_capture(snapshot)
    assert portable.capture.operations[0].arguments.kwargs["source"].id == "h"


def test_mutation_effects_aliases_and_partial_coverage_stay_exact(effect_snapshot):
    original = CaptureSnapshot.model_validate(effect_snapshot)
    portable = portable_capture(original).capture
    assert portable.operations == original.operations
    assert portable.values == original.values
    assert portable.coverage == original.coverage
    assert portable.output == original.output


def test_imported_portable_payload_cannot_bypass_literal_refusal(snapshot):
    portable = portable_capture(snapshot).model_dump(mode="json")
    portable["capture"]["operations"][0]["arguments"]["kwargs"]["token"] = dict(kind="int", value="1234")
    with pytest.raises(ValidationError, match="lossless export refused"):
        PortableCapture.model_validate(portable)


def test_shared_portable_fixture_matches_python_producer():
    root = Path(__file__).parent.resolve().parents[2]
    fixtures = root / "tests/fixtures/contracts"
    payload = json.loads((fixtures / "semantic-capture.json").read_text())
    assert portable_capture(payload).model_dump(mode="json") == json.loads((fixtures / "portable-capture.json").read_text())


@pytest.mark.parametrize("change", [
    lambda c: c["source"].update(workspaceId="live-workspace"),
    lambda c: c["source"].update(generation=1),
    lambda c: c["source"]["files"][0].update(documentVersion=1),
    lambda c: c["source"].update(modelFile="local_model.py", files=[dict(path="local_model.py", digest="a" * 64, documentVersion=None, role="source")]),
])
def test_portable_contract_cannot_claim_live_source_authority(snapshot, change):
    portable = portable_capture(snapshot).model_dump(mode="json")
    change(portable["capture"])
    with pytest.raises(ValidationError, match="detach live source"):
        PortableCapture.model_validate(portable)


def test_portable_preparation_imports_no_torch_or_user_code():
    result = subprocess.run([sys.executable, "-B", "-c", "import sys; from services.semantic.portable import portable_capture; assert 'torch' not in sys.modules"],
                            cwd=Path(__file__).parent.resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("case", json.loads((Path(__file__).parent.resolve().parents[2] /
    "canvas/src/test/fixtures/portable-contract-cases.json").read_text()), ids=lambda case: case["name"])
def test_browser_import_cases_agree_with_python_contract(case):
    portable = json.loads((Path(__file__).parent.resolve().parents[2] /
        "tests/fixtures/contracts/portable-capture.json").read_text())
    for patch in case["patches"]:
        target = portable["capture"]
        for key in patch["path"][:-1]:
            target = target[key]
        target[patch["path"][-1]] = patch["value"]
    if case["valid"]:
        PortableCapture.model_validate(portable)
    else:
        with pytest.raises(ValidationError):
            PortableCapture.model_validate(portable)
