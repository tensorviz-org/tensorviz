"""The documented launcher must not expose an unauthenticated backend."""

import asyncio
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from bind_policy import (
    assert_bind_allowed,
    candidate_bind_hosts,
    ensure_safe_bind,
    is_loopback_bind,
    public_bind_error,
)


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.2"])
def test_loopback_names_are_safe_without_a_token(host):
    assert is_loopback_bind(host) is True
    assert public_bind_error(host, token="") is None


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.10", "example.com", ""])
def test_public_hosts_require_a_launch_token(host):
    assert public_bind_error(host, token="")
    assert public_bind_error(host, token="   ")
    assert public_bind_error(host, token="launch-secret") is None


def test_ensure_safe_bind_exits_before_a_public_listen(monkeypatch):
    monkeypatch.delenv("TENSORVIZ_LOCAL_TOKEN", raising=False)
    with pytest.raises(SystemExit) as raised:
        ensure_safe_bind("0.0.0.0")
    assert raised.value.code == 1
    monkeypatch.setenv("TENSORVIZ_LOCAL_TOKEN", "launch-secret")
    assert ensure_safe_bind("0.0.0.0") == "0.0.0.0"
    monkeypatch.delenv("TENSORVIZ_LOCAL_TOKEN", raising=False)
    monkeypatch.delenv("HOST", raising=False)
    assert ensure_safe_bind(None) == "127.0.0.1"


def test_launchers_do_not_default_to_every_interface():
    start = (ROOT / "start.sh").read_text(encoding="utf-8")
    example = (ROOT / "backend" / ".env.example").read_text(encoding="utf-8")
    server = (ROOT / "backend" / "server_v2.py").read_text(encoding="utf-8")
    assert "--host 0.0.0.0" not in start
    assert 'HOST="${HOST:-127.0.0.1}"' in start
    assert "HOST=0.0.0.0" not in example
    assert "AUTH_DISABLED=true" not in example
    assert "HOST=127.0.0.1" in example
    assert 'host="0.0.0.0"' not in server
    assert "ensure_safe_bind" in server
    assert "install_uvicorn_bind_guard" in server


def test_candidate_hosts_follow_the_socket_uvicorn_will_use(monkeypatch):
    monkeypatch.delenv("TENSORVIZ_LOCAL_TOKEN", raising=False)
    assert candidate_bind_hosts(None) == ["127.0.0.1"]
    assert candidate_bind_hosts("localhost") == ["localhost"]
    assert candidate_bind_hosts("0.0.0.0", uds="/tmp/tensorviz.sock") == []
    assert candidate_bind_hosts("0.0.0.0", []) == []
    loopback = socket.socket()
    loopback.bind(("127.0.0.1", 0))
    try:
        assert candidate_bind_hosts("0.0.0.0", [loopback]) == ["127.0.0.1"]
        assert_bind_allowed(candidate_bind_hosts("0.0.0.0", [loopback]))
    finally:
        loopback.close()
    assert_bind_allowed(["127.0.0.1", "localhost", "::1"])
    with pytest.raises(SystemExit) as raised:
        assert_bind_allowed(["0.0.0.0"])
    assert raised.value.code == 1
    monkeypatch.setenv("AUTH_DISABLED", "true")
    with pytest.raises(SystemExit):
        assert_bind_allowed(["::"])
    monkeypatch.setenv("TENSORVIZ_LOCAL_TOKEN", "launch-secret")
    assert_bind_allowed(["0.0.0.0", "::"])


def test_server_import_guards_uvicorn_startup(monkeypatch):
    import server_v2  # noqa: F401 — installing the guard is the side effect under test
    import uvicorn.server

    startup = uvicorn.server.Server.startup
    assert getattr(startup, "_tensorviz_bind_guarded", False) is True
    monkeypatch.setenv("TENSORVIZ_LOCAL_TOKEN", "")
    monkeypatch.setenv("AUTH_DISABLED", "true")

    class Config:
        host = "0.0.0.0"
        uds = None
        fd = None

    class Fake:
        config = Config()

    with pytest.raises(SystemExit) as raised:
        asyncio.run(startup(Fake()))
    assert raised.value.code == 1


def _launcher_env(**overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TENSORVIZ_LOCAL_TOKEN"] = ""
    env["AUTH_DISABLED"] = "true"
    env.pop("HOST", None)
    env.pop("UVICORN_HOST", None)
    env.update(overrides)
    return env


def test_uvicorn_cli_refuses_a_public_bind_without_a_token(tmp_path):
    stderr = tmp_path / "uvicorn-public.err"
    with stderr.open("w", encoding="utf-8") as err:
        proc = subprocess.run(
            [sys.executable, "-m", "uvicorn", "server_v2:app", "--host", "0.0.0.0",
             "--port", "8765", "--lifespan", "off"],
            cwd=ROOT / "backend",
            env=_launcher_env(),
            stdout=subprocess.DEVNULL,
            stderr=err,
            timeout=180,
        )
    logged = stderr.read_text(encoding="utf-8", errors="replace")
    assert proc.returncode != 0
    assert "TENSORVIZ_LOCAL_TOKEN" in logged
    assert "0.0.0.0" in logged
    assert "Uvicorn running" not in logged


def test_uvicorn_cli_serves_loopback_without_a_token(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    stderr = tmp_path / "uvicorn-loopback.err"
    err = stderr.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server_v2:app", "--host", "127.0.0.1",
         "--port", str(port), "--lifespan", "off"],
        cwd=ROOT / "backend",
        env=_launcher_env(),
        stdout=subprocess.DEVNULL,
        stderr=err,
    )
    try:
        deadline = time.monotonic() + 180
        body = b""
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise AssertionError(stderr.read_text(encoding="utf-8", errors="replace"))
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    body = response.read()
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(0.2)
        else:
            raise AssertionError(stderr.read_text(encoding="utf-8", errors="replace"))
        assert b'"status"' in body
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        err.close()
