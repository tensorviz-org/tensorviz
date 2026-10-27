"""Saved profile limits, not page timers, define the execution budget."""
import sys

import pytest
from pydantic import ValidationError

from services.validation.contracts import Limits
from services.validation.host_snapshot import snapshot


def saved_profile(tmp_path, timeout):
    # Snapshot must resolve this statically; importing it would fail the test.
    (tmp_path / "net.py").write_text(
        "raise RuntimeError('snapshot must never execute source')\n"
        "class Net:\n    def forward(self, x):\n        return x\n"
    )
    config = """version = 1
[profiles.net]
kind = "model"
model = "net:Net"
mode = "eval"
device = "cpu"
grad_enabled = false
init = { args = [], kwargs = {} }
inputs = { args = [{ kind = "tensor", shape = [1, 4], dtype = "float32", requires_grad = false, generator = { kind = "normal", mean = 0.0, std = 1.0, seed = 0 } }], kwargs = {} }
"""
    if timeout is not None:
        config += f"[profiles.net.limits]\ntimeout_ms = {timeout}\n"
    (tmp_path / "tensorviz.toml").write_text(config)
    return snapshot(dict(root=str(tmp_path), executable=sys.executable, environmentRevision=0,
                         session="s", workspaceId="w", runId="r", generation=0, profile="net"))


@pytest.mark.parametrize("timeout,expected", [(None, 60_000), (100, 100), (300_000, 300_000), (86_400_000, 86_400_000)])
def test_host_snapshot_carries_the_resolved_bounded_profile_deadline(tmp_path, timeout, expected):
    prepared = saved_profile(tmp_path, timeout)
    assert prepared["profiles"] == ["net"]
    result = prepared["resolution"]
    assert result["status"] == "resolved", result
    assert result["spec"]["profile"]["limits"]["timeout_ms"] == expected


@pytest.mark.parametrize("timeout", ["99", "86400001", "0", "-1", "true", "300000.5", '"300000"', "inf", "nan"])
def test_invalid_profile_deadlines_cannot_authorize_unbounded_execution(tmp_path, timeout):
    prepared = saved_profile(tmp_path, timeout)
    result = prepared["resolution"]
    assert result["status"] == "needs_setup"
    assert "spec" not in result
    assert any(item["field"] == ["profiles", "net", "limits", "timeout_ms"] for item in result["requirements"])


def test_contract_bounds_and_default_are_the_same_in_schema_and_resolved_limits():
    field = Limits.model_json_schema()["properties"]["timeout_ms"]
    assert (field["minimum"], field["maximum"], field["default"]) == (100, 86_400_000, 60_000)
    assert Limits().timeout_ms == 60_000
    with pytest.raises(ValidationError):
        Limits(timeout_ms=None)
