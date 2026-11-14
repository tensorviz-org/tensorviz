"""Managed service repair cannot plan or install model frameworks."""
from pathlib import Path
from services.ide import environment_setup as setup


def test_managed_update_defaults_to_service_manifest(tmp_path, monkeypatch):
    calls = []
    def plan(directory, requirements):
        calls.append((directory, requirements))
        return {"digest": "reviewed-service-plan"}
    monkeypatch.setattr(setup, "plan", plan)
    monkeypatch.setattr(setup, "install", lambda directory, digest: (directory, digest))
    assert setup.update_managed(tmp_path) == (tmp_path, "reviewed-service-plan")
    assert calls == [(tmp_path, setup.SERVICE_REQUIREMENTS)]


def test_service_manifest_has_no_frameworks_or_capture_include():
    text = setup.SERVICE_REQUIREMENTS.read_text()
    dependencies = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith('#')]
    assert dependencies
    assert not any(line.startswith(('-r', '--', 'torch', 'tensorflow', 'jax', 'onnx')) for line in dependencies)
    assert any(line.startswith('websockets') for line in dependencies)
