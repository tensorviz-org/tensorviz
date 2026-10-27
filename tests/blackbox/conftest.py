"""Shared harness for the black-box suite.

The rule this suite exists to hold: **nothing here imports the product.** No
`from services...`, no `from api...`, no TestClient. Every assertion goes over
HTTP to a server started as a separate process, exactly as the extension talks to
it. That is what makes these tests independent of the internal working — they
survive a refactor that renames every module, and they fail when the product's
observable behaviour changes, which is the only thing a user can see.

Run against a server you already have:

    TENSORVIZ_BASE_URL=http://127.0.0.1:8765 pytest tests/blackbox

or let the suite start its own (the default).
"""
from __future__ import annotations

import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
STARTUP_TIMEOUT_S = 180


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _healthy(base_url: str, timeout: float = 2.0) -> bool:
    try:
        return requests.get(f"{base_url}/health", timeout=timeout).ok
    except requests.RequestException:
        return False


@pytest.fixture(scope="session")
def base_url() -> Iterator[str]:
    """A live backend, either one already running or one this suite owns."""
    configured = os.environ.get("TENSORVIZ_BASE_URL", "").rstrip("/")
    if configured:
        if not _healthy(configured):
            pytest.fail(
                f"TENSORVIZ_BASE_URL={configured} is not answering /health. "
                "Start it, or unset the variable to let the suite start its own."
            )
        yield configured
        return

    port = _free_port()
    env = {
        **os.environ,
        "BACKEND_PORT": str(port),
        "AUTH_DISABLED": "true",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
    }
    # Deliberately the same interpreter running pytest: a suite that silently
    # tested a different Python than the one under test would be worse than none.
    #
    # The log goes to a FILE, never a pipe. A pipe nobody drains fills its OS
    # buffer after a few hundred requests, and the server then blocks forever on
    # its next write — the suite ran to 48% and hung, looking exactly like a
    # product deadlock. Nothing reads this stream until a failure needs
    # explaining, so it must not be able to apply back-pressure.
    log = tempfile.NamedTemporaryFile(
        prefix="tensorviz-blackbox-", suffix=".log", delete=False, mode="w+"
    )
    proc = subprocess.Popen(
        [sys.executable, "server_v2.py"],
        cwd=str(BACKEND_DIR),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )

    def server_log() -> str:
        try:
            return pathlib.Path(log.name).read_text(errors="replace")
        except OSError:
            return "(the server log could not be read)"

    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + STARTUP_TIMEOUT_S
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"the backend exited before answering:\n{server_log()[-4000:]}")
            if _healthy(url):
                break
            time.sleep(0.5)
        else:
            proc.kill()
            pytest.fail(
                f"the backend did not answer /health within {STARTUP_TIMEOUT_S}s:\n"
                f"{server_log()[-4000:]}"
            )
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        os.unlink(log.name)


class Api:
    """The IDE's HTTP surface, and nothing privileged behind it."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def _post(self, route: str, body: Dict[str, Any], timeout: float = 120.0):
        return requests.post(f"{self.base_url}{route}", json=body, timeout=timeout)

    def parse(self, source: str, cls: Optional[str] = None, file: str = "model.py"):
        return self._post("/api/ide/parse", {"file": file, "source": source, "cls": cls})

    def project(
        self,
        source: str,
        cls: Optional[str] = None,
        file: str = "model.py",
        trace_result: Optional[Dict[str, Any]] = None,
    ):
        body: Dict[str, Any] = {"file": file, "source": source, "cls": cls}
        if trace_result is not None:
            body["trace_result"] = trace_result
        return self._post("/api/ide/project", body)

    def apply(
        self,
        source: str,
        cls: str,
        ops: list,
        spanmap: Dict[str, Any],
        source_digest: str,
        file: str = "model.py",
    ):
        return self._post(
            "/api/ide/apply",
            {
                "file": file,
                "cls": cls,
                "source": source,
                "source_digest": source_digest,
                "spanmap": spanmap,
                "ops": ops,
            },
        )

    def health(self):
        return requests.get(f"{self.base_url}/health", timeout=10)


@pytest.fixture(scope="session")
def api(base_url: str) -> Api:
    return Api(base_url)


@pytest.fixture(scope="session")
def projected(api: Api):
    """`(document, spanmap, source_digest)` for a source, straight from the API.

    Every graph→code test needs a projection first, because the spanmap and digest
    an apply must carry are the product's own output. Deriving them any other way
    would be testing our reimplementation instead of the product.
    """

    def _projected(source: str, cls: str, file: str = "model.py"):
        response = api.parse(source, cls, file)
        assert response.status_code == 200, response.text
        parsed = response.json()
        return parsed["document"], parsed["spanmap"], parsed["source_digest"]

    return _projected


def apply_edits(source: str, edits: list) -> str:
    """Apply `/api/ide/apply`'s edits to a buffer, the way the editor does.

    The product returns ranges, never a new file, because the extension turns them
    into a `WorkspaceEdit` so the change lands in the user's undo stack. A
    round-trip test therefore has to do the editor's half, and doing it wrong
    would make the product look broken when it is not — so it is written once,
    here, rather than five times across the suite.

    Positions are 1-based `[line, column]`, ends exclusive. Edits are applied
    last-first so that an earlier range's offsets are still valid when it is used.
    """
    lines = source.splitlines(keepends=True)

    def offset(position) -> int:
        line, column = int(position[0]), int(position[1])
        return sum(len(item) for item in lines[: line - 1]) + column

    ordered = sorted(edits, key=lambda edit: offset(edit["range"]["start"]), reverse=True)
    text = source
    for edit in ordered:
        start = offset(edit["range"]["start"])
        end = offset(edit["range"]["end"])
        text = text[:start] + edit["newText"] + text[end:]
    return text


@pytest.fixture(scope="session")
def edited(api: Api, projected):
    """Run ops through the product and hand back the buffer they produce.

    The whole graph→code half of the round trip in one call: project the source to
    get the spanmap and digest the product itself computed, apply the ops, then
    apply the returned edits. Returns `(new_source, refused)` — `refused` is not an
    error, it is the product declining an edit it cannot express in source, and a
    test that ignored it would read a refusal as a success.
    """

    def _edited(source: str, cls: str, ops: list, file: str = "model.py"):
        _document, spanmap, digest = projected(source, cls, file)
        response = api.apply(source, cls, ops, spanmap, digest, file)
        assert response.status_code == 200, response.text
        body = response.json()
        return apply_edits(source, body["edits"]), body["refused"]

    return _edited
