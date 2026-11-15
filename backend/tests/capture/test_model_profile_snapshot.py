"""Installed F10 fixtures: absence is not a malformed or failed saved snapshot."""
from pathlib import Path
import sys

import pytest

from services.validation.host_snapshot import snapshot

FIXTURES = Path(__file__).parent.parent / "fixtures" / "model_profiles"


def test_documented_toml_is_the_resolved_two_profile_fixture():
    guide = Path(__file__).parent.resolve().parents[2] / "docs" / "model-validation-profiles.md"
    example = guide.read_text().split("```toml\n", 1)[1].split("```", 1)[0]
    assert example.strip() == (FIXTURES / "valid.toml").read_text().strip()


def prepare(root, profile=None, **extra):
    return snapshot(dict(root=str(root), executable=sys.executable, environmentRevision=0,
                         session="s", workspaceId="w", runId="r", generation=1,
                         profile=profile, **extra))


@pytest.fixture
def project(tmp_path):
    # Importing project code during preflight must fail loudly.
    (tmp_path / "audit_models.py").write_text(
        "raise RuntimeError('snapshot must not import source')\n" + (FIXTURES / "audit_models.py").read_text())
    return tmp_path


def test_absence_has_the_exact_missing_requirement(project):
    result = prepare(project)
    assert result["profiles"] == []
    assert result["profileDefinitions"] == {}
    requirements = result["resolution"]["requirements"]
    assert len(requirements) == 1
    assert (requirements[0]["code"], requirements[0]["field"]) == ("missing", ["tensorviz.toml"])


@pytest.mark.parametrize("config,code,field", [
    ((FIXTURES / "invalid.toml").read_text(), "unknown_key", ["profiles", "audit_small", "timeout_ms"]),
    ("version = [private-source", "invalid", ["tensorviz.toml"]),
    ((FIXTURES / "valid.toml").read_text().replace("timeout_ms = 30000", "timeout_ms = 99"),
     "invalid", ["profiles", "audit_small", "limits", "timeout_ms"]),
    ("", "missing", ["version"]),
])
def test_malformed_is_never_absence(project, config, code, field):
    (project / "tensorviz.toml").write_text(config)
    result = prepare(project)
    assert result["profiles"] == []
    assert result["profileDefinitions"] == {}
    assert result["resolution"]["status"] == "needs_setup"
    assert any(r["code"] == code and r["field"] == field for r in result["resolution"]["requirements"])
    assert "private-source" not in str(result)
    if code == "unknown_key":
        assert "limits" in result["resolution"]["requirements"][0]["remedy"]


def test_valid_inventory_is_not_execution_authority_and_selected_bindings_remain_exact(project):
    (project / "tensorviz.toml").write_text((FIXTURES / "valid.toml").read_text())
    listing = prepare(project)
    assert listing["profiles"] == ["audit_small", "audit_large"]
    assert listing["resolution"]["status"] == "needs_setup"
    assert listing["resolution"]["requirements"][0]["code"] == "ambiguous"
    for name, batch in [("audit_small", 2), ("audit_large", 3)]:
        selected = prepare(project, name)["resolution"]
        assert selected["status"] == "resolved"
        assert selected["spec"]["profile"] == listing["profileDefinitions"][name]
        assert selected["spec"]["profile"]["inputs"]["args"][0]["shape"] == [batch, 4]
        assert selected["spec"]["profile"]["limits"]["timeout_ms"] == 30000


def test_selected_signature_and_unsaved_config_fail_with_requirements(project):
    (project / "tensorviz.toml").write_text((FIXTURES / "valid.toml").read_text())
    source = project / "audit_models.py"
    source.write_text(source.read_text().replace("def __init__(self):", "def __init__(self, width):"))
    result = prepare(project, "audit_large")["resolution"]
    assert result["status"] == "needs_setup"
    assert any(r["code"] == "missing" and r["field"] == ["profiles", "audit_large", "init", "kwargs", "width"]
               for r in result["requirements"])
    dirty = prepare(project, documents={"tensorviz.toml": {"version": 2, "dirty": True}})
    assert any(r["code"] == "unsaved" for r in dirty["resolution"]["requirements"])


def test_oversize_validation_record_limit_points_to_graph_budget(project):
    config = (FIXTURES / 'valid.toml').read_text().replace('timeout_ms = 30000',
        'timeout_ms = 30000, max_record_bytes = 16777216')
    (project / 'tensorviz.toml').write_text(config)
    requirements = prepare(project, 'audit_small')['resolution']['requirements']
    remedy = next(r['remedy'] for r in requirements if r['field'][-1] == 'max_record_bytes')
    assert '4194304' in remedy and 'validation records' in remedy
    assert 'limits.max_capture_bytes' in remedy


def test_existing_empty_variadic_profile_returns_to_setup_without_importing(tmp_path):
    (tmp_path / "model.py").write_text('''raise RuntimeError("must not import during preflight")
from torch import nn
class Net(nn.Module):
    def forward_features(self, x, masks=None): return x
    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret
''')
    (tmp_path / "tensorviz.toml").write_text('''version = 1
[profiles.saved]
kind = "model"
model = "model:Net"
mode = "eval"
device = "cpu"
grad_enabled = false
init = {args = [], kwargs = {}}
inputs = {args = [], kwargs = {}}
''')
    result = prepare(tmp_path, "saved")["resolution"]
    assert result["status"] == "needs_setup"
    assert result["requirements"][0]["field"] == ["profiles", "saved", "inputs", "kwargs", "x"]


@pytest.mark.parametrize("kind", ["directory", "dangling", "escape", "unreadable"])
def test_existing_but_unreadable_config_is_a_failed_snapshot(project, tmp_path_factory, monkeypatch, kind):
    config = project / "tensorviz.toml"
    if kind == "directory":
        config.mkdir()
    elif kind == "dangling":
        config.symlink_to(project / "absent")
    elif kind == "escape":
        other = tmp_path_factory.mktemp("outside") / "config.toml"
        other.write_text((FIXTURES / "valid.toml").read_text())
        config.symlink_to(other)
    else:
        config.write_text((FIXTURES / "valid.toml").read_text())
        original = Path.open

        def denied(self, *args, **kwargs):
            if self == config:
                raise PermissionError("private path")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(ValueError, match="configuration snapshot failed"):
        prepare(project)


def test_saved_input_factory_is_checked_without_importing_and_tracks_changes(tmp_path):
    (tmp_path / 'model.py').write_text('from torch import nn\nclass Net(nn.Module):\n    def forward(self, x): return x\n')
    (tmp_path / '.tensorviz').mkdir()
    helper = tmp_path / '.tensorviz/inputs.py'
    helper.write_text('raise RuntimeError("do not import")\ndef graph_inputs(size=4): return {}\n')
    (tmp_path / 'tensorviz.toml').write_text('''version = 1
[profiles.saved]
kind = "model"
model = "model:Net"
mode = "eval"
device = "cpu"
grad_enabled = false
init = {args = [], kwargs = {}}
inputs = {factory = {file = ".tensorviz/inputs.py", callable = "graph_inputs", kwargs = {size = 4}}}
''')
    first = prepare(tmp_path, 'saved')['resolution']
    assert first['status'] == 'resolved', first
    identity = next(f for f in first['spec']['files'] if f['path'] == '.tensorviz/inputs.py')
    assert identity['role'] == 'source'
    helper.write_text(helper.read_text() + '# change\n')
    second = prepare(tmp_path, 'saved')['resolution']
    assert next(f for f in second['spec']['files'] if f['path'] == identity['path'])['digest'] != identity['digest']
    helper.write_text('def graph_inputs(required): return {}\n')
    invalid = prepare(tmp_path, 'saved')['resolution']
    assert invalid['status'] == 'needs_setup'
    assert any(r['field'][:4] == ['profiles', 'saved', 'inputs', 'factory'] for r in invalid['requirements'])
