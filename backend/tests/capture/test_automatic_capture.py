"""Actual automatic workers retain the selected call with verified saved provenance."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from services.semantic.contracts import CaptureSnapshot, CapturePreparation
from services.semantic.execution_plan import consume_plan_file, consume_preparation_file
from services.validation.host_snapshot import snapshot


BACKEND = Path(__file__).parent.resolve().parents[1]


def run(tmp_path, source, request=None, before=None, explicit=False):
    model = tmp_path / "model.py"
    model.write_text(source)
    data = dict(root=str(tmp_path), executable=sys.executable, environmentRevision=0,
        session="test", workspaceId="test", runId="auto", generation=1,
        modelFile="model.py", className="Model", timeoutMs=15000)
    if explicit:
        data["execution"] = dict(inputs=dict(args=[dict(shape=[1, 4], dtype="float32", fill="ones")]))
    else:
        data["captureRequest"] = {"input": None, "tensorInputs": None, "overrides": None, **(request or {})}
    prepared = snapshot(data)
    key = "plan" if explicit else "preparation"
    private = tmp_path / "private.json"
    private.write_text(json.dumps(dict(version=1, **{key: prepared[key]}, applicabilityDigest=prepared["applicabilityDigest"])))
    private.chmod(0o600)
    if before is not None:
        before(model)
    process = subprocess.run([sys.executable, "-B", str(BACKEND / "services/ingest/bootstrap.py"),
        "--resolved-execution" if explicit else "--capture-preparation",
        "--workspace", str(tmp_path), "--file", str(model), "--cls", "Model", "--json"],
        cwd=BACKEND, env={**os.environ, "TENSORVIZ_EXECUTION_PLAN": str(private), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=30)
    return process.returncode, json.loads(process.stdout), prepared


def capture(envelope):
    diagnostics = envelope["result"]["diagnostics"]
    assert diagnostics["semantic_capture_error"] is None, diagnostics["semantic_capture_error"]
    return CaptureSnapshot.model_validate(diagnostics["semantic_capture"])


@pytest.mark.parametrize("explicit", [False, True])
def test_planned_and_automatic_workers_bind_actual_expressions_to_the_verified_source(tmp_path, explicit):
    code, envelope, _ = run(tmp_path, '''import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
    def forward(self, h):
        a = self.fc(h)
        return a + a
''', explicit=explicit)
    assert code == 0, envelope.get("error")
    record = capture(envelope)
    assert record.definitions[0].sourceBindingIds
    layer = next(module for module in record.modules if 'fc' in module.paths)
    constructor = next(binding for binding in record.sourceBindings if binding.id == layer.constructorBindingIds[0])
    assert constructor.kind == 'constructor' and constructor.anchors[0].scope == 'Model.__init__'
    assert constructor.anchors[0].startLine == 5
    add = next(op for op in record.operations if op.target == "aten.add.Tensor")
    assert len(add.sourceBindingIds) == 1
    binding = next(binding for binding in record.sourceBindings if binding.id == add.sourceBindingIds[0])
    assert binding.kind == "call" and binding.quality == "exact"
    anchor = binding.anchors[0]
    assert (anchor.file, anchor.startLine, anchor.startColumn, anchor.endColumn) == ("model.py", 8, 15, 20)
    assert binding.snapshotId == record.source.id and anchor.digest == record.source.files[0].digest


def test_input_fingerprinting_never_iterates_or_formats_custom_containers(tmp_path):
    from services.semantic.recorder import automatic_context

    class HostileList(list):
        def __iter__(self):
            raise AssertionError("custom iteration must not run")

        def __repr__(self):
            raise AssertionError("custom repr must not run")

    class HostileDict(dict):
        def items(self):
            raise AssertionError("custom items must not run")

        def __repr__(self):
            raise AssertionError("custom repr must not run")

    (tmp_path / "model.py").write_text("class Model: pass\n")
    prepared = snapshot(dict(root=str(tmp_path), executable=sys.executable, environmentRevision=0,
        session="test", workspaceId="test", runId="auto", generation=1,
        modelFile="model.py", className="Model", timeoutMs=15000,
        captureRequest=dict(input=None, tensorInputs=None, overrides=None)))
    preparation = CapturePreparation.model_validate(prepared["preparation"])
    for value in (HostileList([torch.ones(1)]), HostileDict(x=torch.ones(1))):
        context = automatic_context(preparation, torch.nn.Identity(), (value,), {}, 1, "declared")
        assert context.input_digest is None
        assert context.state_digest is not None


def test_automatic_worker_preserves_seeded_call_and_workspace_imports(tmp_path):
    (tmp_path / "services").mkdir()
    (tmp_path / "services/__init__.py").write_text("WIDTH = 4\n")
    code, envelope, prepared = run(tmp_path, '''from pathlib import Path
import torch
from services import WIDTH
assert Path.cwd() == Path(__file__).parent
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(WIDTH, 2)
    def forward(self, x): return self.fc(x) + 1
''')
    assert code == 0, envelope.get("error")
    record = capture(envelope)
    assert record.status == "complete" and record.planDigest is None
    assert record.source.model_dump() == prepared["preparation"]["source"]
    assert record.preparationDigest == prepared["applicabilityDigest"]
    assert record.attemptId == "auto" and record.inputDigest and record.stateDigest
    assert record.automatic.model_dump() == dict(provenance="seed-from-first-layer", candidateIndex=1, candidatesTried=1)
    assert next(v for v in record.values if v.id == record.inputs.args[0].id).shape == [1, 4]
    assert envelope["result"]["diagnostics"]["automatic_capture"]["current"] is True


def test_search_retains_the_actual_selected_candidate_and_precall_state(tmp_path):
    code, envelope, _ = run(tmp_path, '''import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("attempts", torch.tensor(0))
    def forward(self, x):
        self.attempts.add_(1)
        if tuple(x.shape) != (1, 3, 32, 32): raise RuntimeError("try another shape")
        return x + self.attempts
''')
    assert code == 0, envelope.get("error")
    record = capture(envelope)
    assert record.status == "complete"
    assert record.automatic.model_dump() == dict(provenance="search-candidate", candidateIndex=3, candidatesTried=3)
    assert next(v for v in record.values if v.id == record.inputs.args[0].id).shape == [1, 3, 32, 32]
    from services.ingest.execution_setup import state_identity
    expected = torch.nn.Module()
    expected.register_buffer("attempts", torch.tensor(2))
    assert record.stateDigest == state_identity(expected)["digest"], "failed earlier calls changed the state before candidate 3"
    assert any("reuses model state" in line for line in record.normalizations)


def test_failed_search_retains_the_deepest_candidate_without_relabelling_it_as_the_last(tmp_path):
    code, envelope, _ = run(tmp_path, '''import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deep = torch.nn.Sequential(torch.nn.ReLU())
    def forward(self, x):
        if x.shape[-1] == 224: x = self.deep(x)
        raise RuntimeError("all candidates fail")
''')
    assert code == 4
    record = capture(envelope)
    assert record.status == "partial" and record.output is None
    assert record.automatic.candidateIndex == 1 and record.automatic.candidatesTried == 8
    assert next(v for v in record.values if v.id == record.inputs.args[0].id).shape == [1, 3, 224, 224]


@pytest.mark.parametrize("declared", [False, True])
def test_declared_and_requested_inputs_preserve_the_single_failed_call(tmp_path, declared):
    source = '''import torch
{comment}
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)
    def forward(self, x): return self.fc(x)
'''.format(comment="# tensorviz: input=2,3" if declared else "")
    code, envelope, _ = run(tmp_path, source, request=None if declared else {"input": "2,3"})
    assert code == 4
    record = capture(envelope)
    assert record.automatic.model_dump() == dict(provenance="declared" if declared else "explicit", candidateIndex=1, candidatesTried=1)
    assert record.status == "partial" and record.diagnostics[0].code == "invalid_shape"


def test_stale_preparation_refuses_before_any_model_import(tmp_path):
    code, envelope, _ = run(tmp_path, '''from pathlib import Path
Path(__file__).with_suffix('.executed').touch()
''', before=lambda file: file.write_text(file.read_text() + "# changed\n"))
    assert code == 2 and envelope["result"] is None
    assert not (tmp_path / "model.executed").exists()


def test_newer_saved_bytes_do_not_relabel_the_captured_original_source(tmp_path):
    code, envelope, prepared = run(tmp_path, '''from pathlib import Path
import torch
class Model(torch.nn.Module):
    def forward(self, x):
        file = Path(__file__)
        file.write_text(file.read_text() + "# changed during forward\\n")
        return x + 1
''', request={"input": "1,4"})
    assert code == 0
    record = capture(envelope)
    assert record.source.model_dump() == prepared["preparation"]["source"]
    assert envelope["result"]["diagnostics"]["automatic_capture"]["current"] is False


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("aliased", [False, True])
def test_newly_imported_source_cannot_certify_a_foreign_saved_snapshot(tmp_path, explicit, aliased):
    if aliased:
        actual = tmp_path / "actual"
        actual.mkdir()
        link = tmp_path / "workspace"
        link.symlink_to(actual, target_is_directory=True)
        tmp_path = link
    code, envelope, _ = run(tmp_path, '''from pathlib import Path
import torch
class Model(torch.nn.Module):
    def forward(self, x):
        Path(__file__).with_name('late.py').write_text('VALUE = 1\\n')
        from late import VALUE
        return x + VALUE
''', request={"input": "1,4"}, explicit=explicit)
    assert code == 0, envelope.get("error")
    diagnostics = envelope["result"]["diagnostics"]
    assert diagnostics["semantic_capture"] is None
    assert diagnostics["semantic_capture_error"] == "source_mismatch"
    assert (diagnostics["execution"]["planCurrent"] if explicit else diagnostics["automatic_capture"]["current"]) is False


def test_preparation_transport_cannot_be_consumed_as_a_resolved_plan(tmp_path, monkeypatch):
    _, _, prepared = run(tmp_path, '''import torch
class Model(torch.nn.Module):
    def forward(self): return torch.ones(1)
''')
    file = tmp_path / "private.json"
    monkeypatch.setenv("TENSORVIZ_EXECUTION_PLAN", str(file))
    with pytest.raises(ValueError, match="envelope"):
        consume_plan_file()
    monkeypatch.setenv("TENSORVIZ_EXECUTION_PLAN", str(file))
    actual, digest = consume_preparation_file()
    assert actual == CapturePreparation.model_validate(prepared["preparation"])
    assert digest == prepared["applicabilityDigest"]
    body = json.loads(file.read_text())
    body["applicabilityDigest"] = "0" * 64
    file.write_text(json.dumps(body))
    monkeypatch.setenv("TENSORVIZ_EXECUTION_PLAN", str(file))
    with pytest.raises(ValueError, match="fingerprint"):
        consume_preparation_file()
