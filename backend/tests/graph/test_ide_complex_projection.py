"""Complex-model acceptance: static identity, dirty dependencies, errors and CPU execution."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tests.graph.test_ide_multifile_projection import client  # shared isolated FastAPI fixture

FIXTURE = Path(__file__).parent.parent / "fixtures" / "ide_complex_repo"
ROOT = "complex_models/network.py"


def payload(overrides=None):
    overrides = overrides or {}
    sources = []
    for file in sorted(FIXTURE.rglob("*.py")):
        name = file.relative_to(FIXTURE).as_posix()
        source = overrides.get(name, file.read_text())
        sources.append(dict(file=name, source=source, digest=hashlib.sha256(source.encode()).hexdigest()))
    root = next(entry for entry in sources if entry["file"] == ROOT)
    return dict(protocol_version=1, sources=sources, root=dict(
        file=ROOT, cls="ComplexNet", document_version=1, source_digest=root["digest"],
    ))


def project(client, overrides=None):
    response = client.post("/api/ide/multifile/project", json=payload(overrides))
    assert response.status_code == 200, response.text
    return response.json()


def test_complex_projection_retains_all_repeated_and_nested_identities(client):
    data = project(client)
    nodes = data["document"]["semantic"]["nodes"]
    ids = [node["node_id"] for node in nodes]
    assert len(ids) == len(set(ids)) == 102
    assert {"branch.fc", "branch_fc"} <= set(ids)
    view = data["document"]["view"]["layout"]["node_data"]
    for trunk in ("trunk1", "trunk2"):
        for stage in ("stage1", "stage2"):
            for block in ("block1", "block2"):
                for branch in ("first", "second"):
                    prefix = f"{trunk}.{stage}.{block}.{branch}"
                    assert view[prefix]["moduleMembers"] == [f"{prefix}.{layer}" for layer in ("conv", "norm", "act")]
                    assert view[f"{prefix}.conv"]["depth"] == 4
                    assert view[f"{prefix}.conv"]["parentId"] == prefix
                    span = data["spanmap"][f"{prefix}.conv"]
                    assert span["file"] == "complex_models/blocks.py"
                    assert "self.conv =" in (FIXTURE / span["file"]).read_text().splitlines()[span["start"][0] - 1]
    assert data["diagnostics"] == []
    assert data["traced"] is False
    assert data["spanmap"]["branch.fc"]["file"] != data["spanmap"]["branch_fc"]["file"]
    assert not any(name.startswith("complex_models") for name in sys.modules)


def test_dirty_dependency_refresh_updates_all_sixteen_instances_without_changing_ids(client):
    before = project(client)
    source = (FIXTURE / "complex_models/blocks.py").read_text()
    after = project(client, {"complex_models/blocks.py": source.replace("nn.ReLU()", "nn.GELU()")})
    old_nodes = {node["node_id"]: node for node in before["document"]["semantic"]["nodes"]}
    new_nodes = {node["node_id"]: node for node in after["document"]["semantic"]["nodes"]}
    assert old_nodes.keys() == new_nodes.keys()
    activations = [identity for identity in old_nodes if identity.endswith(".act")]
    assert len(activations) == 16
    assert all(old_nodes[identity]["op"] == "ReLU" and new_nodes[identity]["op"] == "GELU" for identity in activations)
    assert before["source_digest"] == after["source_digest"]
    assert before["projection_digest"] != after["projection_digest"]
    assert (FIXTURE / "complex_models/blocks.py").read_text() == source


def test_broken_imported_source_keeps_an_actionable_static_warning(client):
    data = project(client, {"complex_models/heads.py": "def broken(:\n"})
    assert data["diagnostics"]
    assert any(diagnostic.get("file") == "complex_models/heads.py" for diagnostic in data["diagnostics"])
    nodes = {node["node_id"]: node for node in data["document"]["semantic"]["nodes"]}
    assert "branch" in nodes and "branch_fc" in nodes
    assert "branch.fc" not in nodes
    assert data["traced"] is False


def test_complex_fixture_runs_a_real_cpu_forward_and_backward_pass():
    # Explicit execution belongs only to this test. The static projection above
    # must never import the package or turn its result into a runtime-health claim.
    program = """
import json, sys, torch
sys.path.insert(0, sys.argv[1])
from complex_models import ComplexNet
torch.set_num_threads(1)
torch.manual_seed(7)
model = ComplexNet().eval()
out = model(torch.randn(2, 3, 16, 16))
assert tuple(out.shape) == (2, 10)
assert torch.isfinite(out).all()
out.square().mean().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
print(json.dumps({'shape': list(out.shape), 'parameters': sum(p.numel() for p in model.parameters())}))
"""
    run = subprocess.run([sys.executable, "-B", "-c", program, str(FIXTURE)], capture_output=True, text=True, timeout=45)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result["shape"] == [2, 10]
    assert result["parameters"] > 35000
