"""Fake-tier slice 2: one FakeTensorMode for construction, inputs and forward (L1.3), and
the canonical (automatic) capture on the fake tier (L1.4).

The worker tests run the real product path: ``host_snapshot`` prepares the automatic
capture, then ``bootstrap.py --capture-preparation --tier fake|real``. A fake-tier capture
must draw the same graph as the real tier (same rendered GraphDocumentV2), mark its values
unavailable, allocate no parameter storage, and stop with a clear, machine-readable reason
when the model needs tensor values. Real-tier envelopes must not gain any field.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from services.ingest import repo_scan
from services.ingest.service import ingest_source
from services.ingest.spec import IngestDiagnostics, ModelRef
from services.ingest.trace import fake_limitation_message, fake_tensor_limitation
from services.semantic.contracts import CaptureSnapshot
from services.semantic.document import capture_document
from services.validation.host_snapshot import snapshot

BACKEND = Path(__file__).parent.resolve().parents[1]
EXIT_OK, EXIT_CONSTRUCT_FAILED, EXIT_TRACE_FAILED = 0, 3, 4

MODEL = '''import torch
from torch import nn
SCALE = torch.full((8,), 0.5)          # module-level constant: must stay a real tensor
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 3)
        self.register_buffer("bias", torch.zeros(8))
        with open(__file__ + ".built", "a") as built:  # what construction saw
            built.write(f"{type(self.fc.weight).__name__} {type(SCALE).__name__}\\n")
    def forward(self, x):
        h = torch.relu(self.norm(self.fc(x))) * SCALE + self.bias
        return self.head(h + x)
# tensorviz: input=2,8
'''


def built(path):
    """(weight type, module constant type) the last construction of ``path`` saw."""
    return tuple(Path(f"{path}.built").read_text().splitlines()[-1].split())


def worker(tmp_path, source, tier, cls="Model"):
    """The automatic capture exactly as the host runs it: prepare, then bootstrap."""
    model = tmp_path / "model.py"
    model.write_text(source)
    prepared = snapshot(dict(root=str(tmp_path), executable=sys.executable, environmentRevision=0,
        session="test", workspaceId="test", runId="auto", generation=1, modelFile="model.py",
        className=cls, timeoutMs=60000, captureRequest=dict(input=None, tensorInputs=None, overrides=None)))
    private = tmp_path / f"private-{tier}.json"
    private.write_text(json.dumps(dict(version=1, preparation=prepared["preparation"],
                                       applicabilityDigest=prepared["applicabilityDigest"])))
    private.chmod(0o600)
    process = subprocess.run([sys.executable, "-B", str(BACKEND / "services/ingest/bootstrap.py"),
        "--capture-preparation", "--workspace", str(tmp_path), "--file", str(model), "--cls", cls,
        "--tier", tier, "--depth", "2", "--json"],
        cwd=BACKEND, env={**os.environ, "TENSORVIZ_EXECUTION_PLAN": str(private), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=120)
    return process.returncode, json.loads(process.stdout)


def document(capture):
    """The GraphDocumentV2 the canvas draws, with the run-varying capture id pinned."""
    capture = dict(capture, id="capture:fixed")
    digests = {f["path"]: f["digest"] for f in capture["source"]["files"]}
    doc = capture_document(capture, name=capture["source"]["className"], graph_id="g",
        model_file=capture["source"]["modelFile"], source_digests=digests,
        source_graph={"nodes": [], "edges": []}, source_spans={})[0]
    doc = doc.model_dump(mode="json") if hasattr(doc, "model_dump") else doc
    doc = doc.get("document", doc)
    semantic = doc["semantic"] if "semantic" in doc else doc
    return dict(nodes=sorted(json.dumps(node, sort_keys=True) for node in semantic["nodes"]),
                edges=sorted(json.dumps(edge, sort_keys=True) for edge in semantic["edges"]))


# ─────────────────────────────────────────────── L1.3: one FakeTensorMode ───────
def test_instantiate_constructs_inside_the_mode_but_imports_outside_it(tmp_path, monkeypatch):
    (tmp_path / "netmod.py").write_text(MODEL)
    monkeypatch.syspath_prepend(str(tmp_path))
    scan = repo_scan.RepoScan(str(tmp_path))
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model, _ = repo_scan.instantiate(scan, "Model", "class", "Model", {}, file=str(tmp_path / "netmod.py"),
                                     construction_mode=mode)
    assert built(tmp_path / "netmod.py") == ("FakeTensor", "Tensor"), "module scope must not run under the mode"
    assert type(type(model).__init__.__globals__["SCALE"]) is torch.Tensor
    assert all(type(p) is FakeTensor for p in model.parameters())
    assert all(type(b) is FakeTensor for b in model.buffers())


def test_instantiate_without_a_mode_is_unchanged(tmp_path, monkeypatch):
    (tmp_path / "netmod2.py").write_text(MODEL)
    monkeypatch.syspath_prepend(str(tmp_path))
    model, _ = repo_scan.instantiate(repo_scan.RepoScan(str(tmp_path)), "Model", "class", "Model", {},
                                     file=str(tmp_path / "netmod2.py"))
    assert all(type(p) is torch.nn.Parameter for p in model.parameters())


def test_legacy_fake_ingest_builds_no_real_weights(tmp_path):
    (tmp_path / "legacy_net.py").write_text(MODEL)
    result = ingest_source(ModelRef(file=str(tmp_path / "legacy_net.py"), cls="Model"), tier="fake")
    assert built(tmp_path / "legacy_net.py") == ("FakeTensor", "Tensor"), "the fake tier materialised real weights"
    assert result.error is None and result.graph["nodes"]
    assert result.diagnostics.tier == "fake"


def test_real_ingest_still_builds_real_weights(tmp_path):
    (tmp_path / "real_net.py").write_text(MODEL)
    result = ingest_source(ModelRef(file=str(tmp_path / "real_net.py"), cls="Model"), tier="real")
    assert built(tmp_path / "real_net.py") == ("Parameter", "Tensor")
    assert result.error is None


def test_legacy_fake_ingest_inside_a_callers_mode_does_not_nest_a_second_one(tmp_path):
    (tmp_path / "nested_net.py").write_text(MODEL)
    with FakeTensorMode(allow_non_fake_inputs=True):
        result = ingest_source(ModelRef(file=str(tmp_path / "nested_net.py"), cls="Model"), tier="fake")
    assert result.error is None and result.graph["nodes"]


# ─────────────────────────────────────── L1.4: canonical capture on fake ───────
def test_fake_tier_automatic_capture_draws_the_real_tier_graph(tmp_path):
    real_code, real = worker(tmp_path, MODEL, "real")
    fake_code, fake = worker(tmp_path, MODEL, "fake")
    assert (real_code, fake_code) == (EXIT_OK, EXIT_OK), (real.get("error"), fake.get("error"))
    real_capture = real["result"]["diagnostics"]["semantic_capture"]
    fake_capture = fake["result"]["diagnostics"]["semantic_capture"]
    assert fake_capture is not None, fake["result"]["diagnostics"].get("semantic_capture_error")
    assert fake["result"]["diagnostics"]["tier"] == "fake"
    assert fake_capture["valuesUnavailable"] == "fake_tensor" and "valuesUnavailable" not in real_capture
    assert fake_capture["status"] == real_capture["status"] == "complete"
    assert fake_capture["stateDigest"] is None and real_capture["stateDigest"] is not None
    CaptureSnapshot.model_validate(fake_capture)
    assert document(fake_capture) == document(real_capture)
    assert fake["result"]["graph"] == {"nodes": [], "edges": []}, "canonical, not the legacy folded graph"
    assert fake["result"]["diagnostics"]["automatic_capture"]["current"] is True


def test_real_tier_envelopes_gain_no_fake_tier_fields(tmp_path):
    code, envelope = worker(tmp_path, MODEL, "real")
    assert code == EXIT_OK
    assert "fake_tensor" not in envelope["result"]["diagnostics"]
    assert "fake_tensor" not in IngestDiagnostics(tier="real").model_dump(mode="json")
    code, envelope = worker(tmp_path, MODEL.replace("return self.head(h + x)", "return self.head(h[:, :5])"), "real")
    assert code == EXIT_TRACE_FAILED and "fakeTensor" not in envelope["error"]


DATA_DEPENDENT = '''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(4, 4), nn.Linear(4, 4)
    def forward(self, x):
        if x.sum().item() > 0:
            return self.a(x)
        return self.b(x)
# tensorviz: input=2,4
'''


def test_a_value_read_stops_the_fake_tier_with_a_clear_reason(tmp_path):
    code, envelope = worker(tmp_path, DATA_DEPENDENT, "fake")
    assert code == EXIT_TRACE_FAILED
    error = envelope["error"]
    assert error["message"].startswith("FakeTensorLimitation: the model reads a tensor value")
    assert "tensorviz.traceTier = real" in error["message"]
    assert error["fakeTensor"] == dict(reason="data_dependent", errorType="DataDependentOutputException",
                                       operation="aten._local_scalar_dense.default", phase="forward")
    assert (Path(error["file"]).name, error["line"]) == ("model.py", 8), "the error site is the user's line"
    assert envelope["result"]["diagnostics"]["fake_tensor"]["limitation"]["reason"] == "data_dependent"
    assert envelope["result"]["diagnostics"]["semantic_capture"]["status"] == "partial"
    code, envelope = worker(tmp_path, DATA_DEPENDENT, "real")
    assert code == EXIT_OK and envelope["error"] is None


def test_a_value_dependent_shape_stops_the_fake_tier(tmp_path):
    code, envelope = worker(tmp_path, DATA_DEPENDENT.replace(
        "if x.sum().item() > 0:\n            return self.a(x)\n        return self.b(x)",
        "return self.a(x[x > 0].reshape(-1, 1).expand(-1, 4))"), "fake")
    assert code == EXIT_TRACE_FAILED
    assert envelope["error"]["fakeTensor"]["reason"] == "dynamic_output_shape"
    assert envelope["error"]["fakeTensor"]["phase"] == "forward"


def test_a_value_read_during_construction_stops_the_fake_tier(tmp_path):
    code, envelope = worker(tmp_path, DATA_DEPENDENT.replace(
        "self.a, self.b = nn.Linear(4, 4), nn.Linear(4, 4)",
        "self.a, self.b = nn.Linear(4, 4), nn.Linear(4, 4)\n        self.width = int(self.a.weight.abs().sum().item() > -1) * 4"), "fake")
    assert code == EXIT_CONSTRUCT_FAILED
    assert envelope["error"]["fakeTensor"]["phase"] == "construction"
    assert envelope["error"]["fakeTensor"]["reason"] == "data_dependent"
    assert envelope["error"]["message"].startswith("FakeTensorLimitation:")


def test_a_genuine_model_error_on_the_fake_tier_fails_at_the_same_site_without_a_marker(tmp_path):
    """Not a fake-tensor limitation, so no marker and no real re-run. The message is the
    fake kernel's own wording ("a and b must have same reduction dim"), not the CPU
    kernel's ("mat1 and mat2 shapes cannot be multiplied"); the site is the same line."""
    broken = MODEL.replace("return self.head(h + x)", "return self.head(h[:, :5])")
    fake_code, fake = worker(tmp_path, broken, "fake")
    real_code, real = worker(tmp_path, broken, "real")
    assert fake_code == real_code == EXIT_TRACE_FAILED
    assert "fakeTensor" not in fake["error"] and not fake["error"]["message"].startswith("FakeTensorLimitation")
    assert (fake["error"]["file"], fake["error"]["line"]) == (real["error"]["file"], real["error"]["line"])


def test_the_input_search_stops_at_a_fake_tensor_limitation(tmp_path):
    code, envelope = worker(tmp_path, DATA_DEPENDENT.replace("# tensorviz: input=2,4\n", ""), "fake")
    assert code == EXIT_TRACE_FAILED and envelope["error"]["fakeTensor"]["reason"] == "data_dependent"
    assert envelope["result"]["diagnostics"]["timings_ms"]["input_candidates_tried"] == 1


def test_the_legacy_fake_tier_keeps_its_error_text(tmp_path):
    """Without a capture preparation (the legacy fold) nothing is reworded or marked."""
    model = tmp_path / "model.py"
    model.write_text(DATA_DEPENDENT)
    process = subprocess.run([sys.executable, "-B", str(BACKEND / "services/ingest/bootstrap.py"), "--file", str(model),
        "--cls", "Model", "--workspace", str(tmp_path), "--tier", "fake", "--json"], cwd=BACKEND,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, timeout=120)
    envelope = json.loads(process.stdout)
    assert envelope["error"]["message"].startswith("DataDependentOutputException")
    assert "fakeTensor" not in envelope["error"] and "fake_tensor" not in envelope["result"]["diagnostics"]


# ───────────────────────────────────────────────────── classification ───────
def test_fake_tensor_limitation_classifies_only_fake_tensor_failures():
    with FakeTensorMode():
        x = torch.ones(3)
        try:
            x.sum().item()
        except Exception as exc:
            read = exc
        try:
            torch.nonzero(x)
        except Exception as exc:
            shape = exc
    assert fake_tensor_limitation(read)["reason"] == "data_dependent"
    assert fake_tensor_limitation(shape)["reason"] == "dynamic_output_shape"
    assert fake_tensor_limitation(shape)["operation"] == "aten.nonzero.default"
    try:
        raise RuntimeError("constructing Model failed") from read
    except RuntimeError as wrapped:
        assert fake_tensor_limitation(wrapped)["errorType"] == "DataDependentOutputException"
    assert fake_tensor_limitation(ValueError("mat1 and mat2 shapes cannot be multiplied")) is None
    assert fake_limitation_message(fake_tensor_limitation(shape)).startswith(
        "FakeTensorLimitation: the model produces an output whose shape depends on tensor values")


LAZY = '''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.LazyLinear(4)
    def forward(self, x):
        return self.a(x)
# tensorviz: input=2,8
'''


def test_a_lazy_module_stops_the_fake_tier_with_a_clear_reason(tmp_path):
    # nn.LazyLinear creates an UninitializedParameter (a tensor subclass), which PyTorch
    # refuses inside FakeTensorMode with a plain RuntimeError.
    code, envelope = worker(tmp_path, LAZY, "fake")
    assert code == EXIT_CONSTRUCT_FAILED
    error = envelope["error"]
    assert error["fakeTensor"] == dict(reason="tensor_subclass", errorType="RuntimeError",
                                       operation="UninitializedParameter", phase="construction")
    assert error["message"].startswith("FakeTensorLimitation: the model creates a tensor subclass")
    code, envelope = worker(tmp_path, LAZY, "real")
    assert code == EXIT_OK and envelope["error"] is None


def test_only_the_fake_tensor_subclass_runtime_error_is_classified():
    subclass = RuntimeError("Creating a new Tensor subclass UninitializedParameter but the raw Tensor object "
                            "is already associated to a python object of type FakeTensor")
    assert fake_tensor_limitation(subclass)["reason"] == "tensor_subclass"
    assert fake_tensor_limitation(RuntimeError("Creating a new Tensor subclass X but something else")) is None
    assert fake_tensor_limitation(RuntimeError("python object of type FakeTensor")) is None
    assert fake_tensor_limitation(ValueError(str(subclass))) is None
