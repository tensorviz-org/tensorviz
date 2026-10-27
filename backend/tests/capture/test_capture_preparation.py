"""Known saved facts precede input search without inventing a resolved recipe."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

from services.semantic.contracts import CapturePreparation, ResolvedExecutionPlan
from services.semantic.execution_plan import applicability_digest, verify_saved_plan
from services.validation.host_snapshot import snapshot


MODEL = "from pathlib import Path\nPath(__file__).with_suffix('.executed').touch()\nclass Model: pass\n"
PROFILE = '''version = 1
[profiles.small]
kind = "model"
model = "other:Other"
mode = "eval"
device = "cpu"
grad_enabled = false
init = {args = [], kwargs = {}}
inputs = {args = [], kwargs = {}}
'''


@pytest.fixture
def request_data(tmp_path):
    (tmp_path / "model.py").write_text(MODEL)
    return dict(root=str(tmp_path), executable=sys.executable, environmentRevision=0,
        session="session", workspaceId="workspace", runId="attempt", generation=1,
        modelFile="model.py", className="Model", timeoutMs=30000,
        captureRequest=dict(input=None, tensorInputs=None, overrides=None))


def prepared(request):
    return CapturePreparation.model_validate(snapshot(request)["preparation"])


@pytest.mark.parametrize("config", [None, PROFILE])
def test_automatic_preparation_is_static_without_a_resolved_recipe(request_data, config):
    root = Path(request_data["root"])
    if config is not None:
        (root / "tensorviz.toml").write_text(config)
    result = snapshot(request_data)
    preparation = CapturePreparation.model_validate(result["preparation"])
    assert preparation.configuration == ("absent" if config is None else "no_match")
    assert preparation.source.sourceMode == "saved" and preparation.source.dependencyScope == "partial"
    assert preparation.request.model_dump() == request_data["captureRequest"]
    assert result["applicabilityDigest"] == applicability_digest(preparation)
    assert "plan" not in result and "execution" not in preparation.model_dump()
    with pytest.raises(ValidationError):
        ResolvedExecutionPlan.model_validate(preparation.model_dump())
    verify_saved_plan(preparation, str(root))
    assert not (root / "model.executed").exists()


@pytest.mark.parametrize("problem", ["invalid", "matching", "script", "package", "cwd", "dirty_config", "dirty_dependency"])
def test_explicit_or_unsaved_configuration_never_becomes_an_automatic_guess(request_data, problem):
    root = Path(request_data["root"])
    config = PROFILE
    if problem == "invalid":
        config = "not [valid toml"
    if problem == "matching":
        config = PROFILE.replace("other:Other", "model:Model")
    if problem == "script":
        config = '''version = 1
[profiles.saved]
kind = "script"
script = "model.py"
args = []
capture = {model = "__main__:Model", call = 1}
'''
    if problem == "package":
        (root / "model").mkdir()
        (root / "model/__init__.py").write_text(MODEL)
        request_data["modelFile"] = "model/__init__.py"
        config = PROFILE.replace("other:Other", "model:Model")
    if problem == "cwd":
        (root / "nested").mkdir()
        (root / "nested/model.py").write_text(MODEL)
        request_data["modelFile"] = "nested/model.py"
        config = PROFILE.replace('kind = "model"', 'kind = "model"\ncwd = "nested"').replace("other:Other", "model:Model")
    if problem == "dirty_config":
        request_data["documents"] = {"tensorviz.toml": {"version": 2, "dirty": True}}
    if problem == "dirty_dependency":
        request_data["documents"] = {"helper.py": {"version": 2, "dirty": True}}
    (root / "tensorviz.toml").write_text(config)
    with pytest.raises(ValueError):
        prepared(request_data)
    assert not list(root.rglob("*.executed"))


@pytest.mark.parametrize("change", ["source", "config", "environment"])
def test_automatic_preparation_rechecks_saved_identity_before_import(request_data, change):
    preparation = prepared(request_data)
    root = Path(request_data["root"])
    if change == "source":
        (root / "model.py").write_text(MODEL + "# changed\n")
    if change == "config":
        (root / "tensorviz.toml").write_text(PROFILE)
    if change == "environment":
        preparation.environment.probe_digest = "0" * 64
    with pytest.raises(ValueError):
        verify_saved_plan(preparation, str(root))
    assert not (root / "model.executed").exists()


@pytest.mark.parametrize("change", ["input", "tensors", "overrides", "source", "config", "environment", "timeout", "limit"])
def test_preparation_applicability_covers_request_and_saved_facts(request_data, change):
    first = prepared(request_data)
    if change == "input":
        request_data["captureRequest"]["input"] = "2,4"
    if change == "tensors":
        request_data["captureRequest"]["tensorInputs"] = [dict(shape=[2, 4], dtype="float32", fill="ones")]
    if change == "overrides":
        request_data["captureRequest"]["overrides"] = {"width": 4}
    if change == "source":
        (Path(request_data["root"]) / "model.py").write_text(MODEL + "# different\n")
    if change == "config":
        (Path(request_data["root"]) / "tensorviz.toml").write_text(PROFILE)
    if change == "environment":
        request_data["environmentRevision"] = 1
    if change == "timeout":
        request_data["timeoutMs"] = 2000
    second = prepared(request_data)
    if change == "limit":
        second.captureLimits.operations = 1
    assert applicability_digest(first) != applicability_digest(second)


def test_preparation_occurrence_identity_is_distinct_from_applicability(request_data):
    first = prepared(request_data)
    request_data.update(runId="next", generation=2, documents={"model.py": {"version": 2, "dirty": False}})
    second = prepared(request_data)
    assert first.id != second.id and first.source.id != second.source.id
    assert applicability_digest(first) == applicability_digest(second)


@pytest.mark.parametrize("change", ["competing", "explicit", "profile", "oversize", "nonfinite", "escape", "dirty", "deleted"])
def test_invalid_requests_refuse_without_model_imports(request_data, change):
    root = Path(request_data["root"])
    if change == "competing":
        request_data["captureRequest"].update(input="2,4", tensorInputs=[dict(shape=[2, 4], dtype="float32", fill="ones")])
    if change == "explicit":
        request_data["execution"] = {"inputs": {"args": []}}
    if change == "profile":
        request_data["profile"] = "small"
    if change == "oversize":
        request_data["captureRequest"]["tensorInputs"] = [dict(shape=[1000000, 1000000], dtype="float32", fill="ones")]
    if change == "nonfinite":
        request_data["captureRequest"]["overrides"] = {"scale": float("nan")}
    if change == "escape":
        request_data["modelFile"] = "../model.py"
    if change == "dirty":
        request_data["documents"] = {"model.py": {"version": 2, "dirty": True}}
    if change == "deleted":
        (root / "model.py").unlink()
    with pytest.raises(ValueError):
        prepared(request_data)
    assert not (root / "model.executed").exists()


def test_preparation_schema_rejects_forged_environment_or_configuration(request_data):
    valid = prepared(request_data).model_dump()
    for change in (lambda p: p["source"].update(sourceMode="buffers"),
                   lambda p: p["source"].update(environmentDigest="0" * 64),
                   lambda p: p.update(configuration="no_match")):
        candidate = deepcopy(valid)
        change(candidate)
        with pytest.raises(ValidationError):
            CapturePreparation.model_validate(candidate)
