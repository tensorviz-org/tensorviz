"""Saved profiles and explicit execution construct the exact same object-backed model."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from services.validation.host_snapshot import snapshot

MODEL = '''from torch import nn
class Dimensions:
    def __init__(self, width): self.width = width
class Net(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.fc = nn.Linear(dims.width, 2)
    def forward(self, x): return self.fc(x)
'''
FACTORY = '''from factory_net import Net, Dimensions
def build(width=4): return Net(Dimensions(width))
'''
PROFILE = '''version = 1
[profiles.object]
kind = "model"
model = "factory_net:Net"
mode = "eval"
device = "cpu"
grad_enabled = false
seed = 7
factory = { file = "recipe.py", callable = "build", kwargs = { width = 4 } }
[profiles.object.inputs]
args = [{kind="tensor", shape=[2,4], dtype="float32", requires_grad=false, generator={kind="full", value=1}}]
kwargs = {}
'''


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "factory_net.py").write_text(MODEL)
    (tmp_path / "recipe.py").write_text(FACTORY)
    (tmp_path / "tensorviz.toml").write_text(PROFILE)
    return tmp_path


def resolve(root, **extra):
    return snapshot(dict(root=str(root), executable=sys.executable, environmentRevision=0,
                         session="s", workspaceId="w", runId="r", generation=1, profile="object", **extra))["resolution"]


def test_factory_preflight_is_static_and_tracks_its_source(workspace):
    (workspace / "factory_net.py").write_text("raise RuntimeError('must not import')\n" + MODEL)
    (workspace / "recipe.py").write_text("raise RuntimeError('must not import')\n" + FACTORY)
    result = resolve(workspace)
    assert result["status"] == "resolved", result
    assert result["spec"]["profile"]["init"] is None
    assert {f["path"]: f["role"] for f in result["spec"]["files"]}["recipe.py"] == "source"
    before = result["spec"]["identity"]["sourceDigest"]
    (workspace / "recipe.py").write_text(FACTORY + "# changed\n")
    assert resolve(workspace)["spec"]["identity"]["sourceDigest"] != before


@pytest.mark.parametrize("change,code", [
    (lambda root: (root / "recipe.py").unlink(), "missing_file"),
    (lambda root: (root / "recipe.py").write_text(FACTORY.replace("width=4", "width, required")), "missing"),
    (lambda root: (root / "recipe.py").write_text("def broken("), "invalid_source"),
    (lambda root: (root / "tensorviz.toml").write_text(PROFILE.replace('file = "recipe.py"', 'file = "../recipe.py"')), "invalid"),
    (lambda root: (root / "tensorviz.toml").write_text(PROFILE.replace('[profiles.object.inputs]', '[profiles.object.init]\nargs=[]\nkwargs={}\n[profiles.object.inputs]')), "invalid"),
])
def test_bad_factory_refuses_without_default_construction(workspace, change, code):
    change(workspace)
    result = resolve(workspace)
    assert result["status"] == "needs_setup"
    assert any(r["code"] == code and "factory" in r["field"] for r in result["requirements"]), result


def test_unsaved_factory_is_a_saved_source_gate(workspace):
    result = resolve(workspace, documents={"recipe.py": {"version": 2, "dirty": True}})
    assert result["status"] == "needs_setup"
    assert any(r["code"] == "unsaved" and r["field"][-2:] == ["factory", "file"] for r in result["requirements"])


@pytest.mark.parametrize("wrong_class", [False, True])
@pytest.mark.parametrize("python_inputs", [False, True])
def test_both_workers_use_the_same_factory_and_exact_class_guard(workspace, wrong_class, python_inputs):
    if python_inputs:
        (workspace / "inputs.py").write_text("import torch\ndef graph_inputs(): return {'x': torch.ones(2, 4)}\n")
        (workspace / "tensorviz.toml").write_text(PROFILE.split('[profiles.object.inputs]')[0]
            + 'inputs = {factory = {file = "inputs.py", callable = "graph_inputs", kwargs = {}}}\n')
    if wrong_class:
        # A subclass can have exactly the same name and source as the selected class.
        # Comparing names/files alone would accept this different runtime class.
        (workspace / "factory_net.py").write_text(MODEL + "Other = type('Net', (Net,), {})\n")
        (workspace / "recipe.py").write_text(FACTORY.replace("import Net, Dimensions", "import Other, Dimensions").replace("return Net(", "return Other("))
    result = resolve(workspace)
    assert result["status"] == "resolved", result
    spec_file = workspace / "spec.json"
    spec_file.write_text(json.dumps(result["spec"]))
    script = '''import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import torch
from services.validation.contracts import RunSpec
from services.validation.model_run import execute_model
from services.ingest.execution_setup import prepare_execution
root = Path(sys.argv[2])
spec = RunSpec.model_validate_json((root / 'spec.json').read_text())
record = execute_model(spec, workspace_root=str(root))
profile = spec.profile
from services.semantic.execution_plan import profile_execution
execution = profile_execution(profile)
try:
    prepared = prepare_execution(str(root / 'factory_net.py'), 'Net', str(root), execution)
    with torch.no_grad(): output = prepared.model(*prepared.args, **prepared.kwargs)
    explicit = list(output.shape)
except ValueError:
    explicit = 'refused'
print(json.dumps(dict(profile=record.process.state, shape=explicit, requirements=[r.code for r in record.requirements])))
'''
    process = subprocess.run([sys.executable, "-B", "-c", script, str(Path(__file__).parent.resolve().parents[1]), str(workspace)],
                             cwd=workspace, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    if wrong_class:
        assert result["profile"] == "failed" and result["shape"] == "refused", result
    else:
        assert result["profile"] == "running" and result["shape"] == [2, 2], result
