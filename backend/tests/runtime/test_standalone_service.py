"""The standalone boundary keeps canonical projection without hosted services."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient

from standalone_server import create_app
from tests.capture.test_semantic_contracts import snapshot  # noqa: F401 - pytest fixture


TOKEN = "standalone-test-token-that-is-not-a-production-secret"


def payload(source=None):
    source = source or "from torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, 3)\n def forward(self, x):\n  return self.fc(x)\n"
    digest = hashlib.sha256(source.encode()).hexdigest()
    return {"protocol_version": 1, "root": {"file": "model.py", "cls": "Tiny", "source_digest": digest},
            "sources": [{"file": "model.py", "source": source, "digest": digest}], "truncated": False}


def test_source_projection_needs_token_and_never_imports_the_model(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORVIZ_LOCAL_TOKEN", TOKEN)
    marker = tmp_path / "imported"
    request = payload(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n" + payload()["sources"][0]["source"])
    with TestClient(create_app(TOKEN), base_url="http://127.0.0.1") as client:
        route = "/api/ide/multifile/project"
        assert client.post(route, json=request).status_code == 401
        reply = client.post(route, json=request, headers={"X-tensorViz-Token": TOKEN})
        assert reply.status_code == 200, reply.text
        graph = reply.json()
        assert graph["traced"] is False
        assert graph["document"]["semantic"]["nodes"]
        assert graph["source_digest"] == request["root"]["source_digest"]
        assert not marker.exists()
        assert client.get("/api/projects", headers={"X-tensorViz-Token": TOKEN}).status_code == 404
        assert client.get("/api/health", headers={"host": "attacker.example", "X-tensorViz-Token": TOKEN}).status_code == 400


def test_standalone_import_graph_has_no_hosted_auth_or_store():
    root = Path(__file__).parent.resolve().parents[1]
    script = "import sys; sys.path.insert(0, sys.argv[1]); import standalone_server; import json; print(json.dumps(sorted(sys.modules)))"
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(root)],
                            capture_output=True, text=True, check=True, cwd=root, timeout=30)
    loaded = json.loads(result.stdout)
    assert "auth" not in loaded
    assert "services.store_factory" not in loaded
    assert not any(name.startswith("firebase") for name in loaded)


def test_static_discovery_uses_shared_inheritance_analysis_without_imports(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORVIZ_LOCAL_TOKEN", TOKEN)
    marker = tmp_path / "imported"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n" + payload()["sources"][0]["source"]
    body = payload(source)
    request = {"protocol_version": 1, "sources": body["sources"], "truncated": True}
    with TestClient(create_app(TOKEN), base_url="http://127.0.0.1") as client:
        assert client.post("/api/ide/review/index", json=request).status_code == 401
        reply = client.post("/api/ide/review/index", json=request, headers={"X-tensorViz-Token": TOKEN})
        assert reply.status_code == 200, reply.text
        assert reply.json()["models"][0]["cls"] == "Tiny"
        assert reply.json()["truncated"] is True
        assert not marker.exists()


def test_malformed_source_and_streamed_oversize_are_refused(monkeypatch):
    monkeypatch.setenv("TENSORVIZ_LOCAL_TOKEN", TOKEN)
    with TestClient(create_app(TOKEN), base_url="http://127.0.0.1") as client:
        headers = {"X-tensorViz-Token": TOKEN}
        request = payload()
        request["sources"][0]["digest"] = "0" * 64
        assert client.post("/api/ide/multifile/project", json=request, headers=headers).status_code == 400
        assert client.post("/local/bootstrap", content=iter([b"x" * 3000, b"y" * 3000])).status_code == 413
        assert client.post("/api/ide/graph-setup/discover", json={"secret": "private"}, headers=headers).json() == {
            "detail": "Invalid local-service request."}


def test_portable_capture_for_publish_needs_token_and_never_echoes_literals(monkeypatch, snapshot):
    import standalone_server
    from services.semantic.portable import portable_capture
    monkeypatch.setenv("TENSORVIZ_LOCAL_TOKEN", TOKEN)
    headers = {"X-tensorViz-Token": TOKEN}
    route = "/api/ide/capture/portable"
    with TestClient(create_app(TOKEN), base_url="http://127.0.0.1") as client:
        assert client.post(route, json=snapshot).status_code == 401
        reply = client.post(route, json=snapshot, headers=headers)
        assert reply.status_code == 200, reply.text
        assert reply.json() == portable_capture(snapshot).model_dump(mode="json")
        assert reply.headers["cache-control"] == "no-store"
        secret = "PRIVATE_TOKEN_MUST_NOT_APPEAR"
        private = json.loads(json.dumps(snapshot))
        private["operations"][0]["arguments"]["kwargs"]["token"] = {"kind": "str", "value": secret}
        refused = client.post(route, json=private, headers=headers)
        assert refused.status_code == 400 and secret not in refused.text
        monkeypatch.setattr(standalone_server, "PORTABLE_CAPTURE_MAX_BYTES", 10)
        limited = client.post(route, json=snapshot, headers=headers)
        assert limited.status_code == 413 and "Nothing was truncated" in limited.text


def test_standalone_broker_does_not_offer_native_source_proposals(monkeypatch):
    monkeypatch.setenv("TENSORVIZ_LOCAL_TOKEN", TOKEN)
    with TestClient(create_app(TOKEN), base_url="http://127.0.0.1") as client:
        reply = client.post("/local/sessions", headers={"X-tensorViz-Token": TOKEN},
                            json={"launchOrigin": "http://127.0.0.1"})
        assert reply.status_code == 200
        assert "canProposeSourceEdits" not in reply.json()["capabilities"]


def test_service_exits_when_the_launchers_pipe_closes():
    root = Path(__file__).parent.resolve().parents[1]
    child = subprocess.Popen([sys.executable, "-I", "-B", str(root / "standalone_server.py"), "--parent-pipe"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env={**os.environ, "TENSORVIZ_LOCAL_TOKEN": TOKEN}, text=True)
    try:
        # A real bound service has started; losing its owner must revoke it even
        # without a graceful MCP shutdown or SIGTERM handler in the launcher.
        import selectors
        with selectors.DefaultSelector() as ready:
            ready.register(child.stdout, selectors.EVENT_READ)
            assert ready.select(timeout=30), "service did not report its bound port"
        assert json.loads(child.stdout.readline())["port"] > 0
        child.stdin.close()
        assert child.wait(timeout=5) == 0
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


FORWARD = "http://127.0.0.1:18080"


def test_browser_view_accepts_a_forwarded_loopback_port_only_from_its_own_page(monkeypatch):
    """ssh -L and editor forwards give the page a local port the service cannot know."""
    import pytest
    from starlette.websockets import WebSocketDisconnect
    from services.ide.local_browser_broker import LOCAL_BROWSER_CLOSE_CODES, view_cookie_name
    monkeypatch.setenv("TENSORVIZ_LOCAL_TOKEN", TOKEN)
    headers = {"X-tensorViz-Token": TOKEN}
    launch = "http://127.0.0.1:45791"

    def bootstrap(credential, origin, host):
        return client.post("/local/bootstrap", headers={"Origin": origin, "Host": host},
                           json={"workspaceSessionId": session["workspaceSessionId"], "credential": credential})

    def view_link():
        reply = client.post(f"/local/sessions/{session['workspaceSessionId']}/view-links", headers=headers)
        assert reply.status_code == 200, reply.text
        return reply.json()["viewCredential"]

    with TestClient(create_app(TOKEN), base_url=launch) as client:
        session = client.post("/local/sessions", headers=headers, json={"launchOrigin": launch}).json()
        assert client.get("/api/health", headers={**headers, "Host": "localhost:18081"}).status_code == 200
        for origin in (FORWARD, "http://localhost:18081"):
            accepted = bootstrap(view_link(), origin, origin.removeprefix("http://"))
            assert accepted.status_code == 200, accepted.text
            assert accepted.headers["set-cookie"].startswith(view_cookie_name(origin) + "=")

        # Another local page, a rebinding name and a non-http scheme are refused
        # before the single-use credential is read, so the link stays usable.
        credential = view_link()
        for origin, host, status in [("http://127.0.0.1:8888", "127.0.0.1:18080", 403),
                                     ("http://attacker.example:18080", "127.0.0.1:18080", 403),
                                     ("http://attacker.example:18080", "attacker.example:18080", 400),
                                     ("https://127.0.0.1:18080", "127.0.0.1:18080", 403)]:
            assert bootstrap(credential, origin, host).status_code == status, (origin, host)
        accepted = bootstrap(credential, FORWARD, "127.0.0.1:18080")
        assert accepted.status_code == 200, accepted.text
        key = accepted.headers["set-cookie"].split(";", 1)[0].split("=", 1)[1]

        def socket(origin, host):
            return client.websocket_connect("/local/ws", headers={
                "Origin": origin, "Host": host, "Cookie": f"{view_cookie_name(origin)}={key}"})

        with socket(FORWARD, "127.0.0.1:18080") as view:
            view.send_json({"type": "viewHandshake", "protocolVersion": 1, "role": "view", "messageId": "m-1",
                            "workspaceSessionId": session["workspaceSessionId"]})
            assert view.receive_json()["type"] == "connected"
        # The key belongs to the forward that bootstrapped it, not another forward
        # or a page on another port reaching the service directly.
        for origin, host in [(FORWARD, "127.0.0.1:45791"), ("http://127.0.0.1:18082", "127.0.0.1:18082"), (launch, "127.0.0.1:45791")]:
            with pytest.raises(WebSocketDisconnect) as refused:
                with socket(origin, host):
                    pass
            assert refused.value.code == LOCAL_BROWSER_CLOSE_CODES["INVALID_ORIGIN"], (origin, host)


def test_requested_port_is_used_when_free_and_falls_back_when_taken():
    from standalone_server import listen
    probe = listen()
    port = probe.getsockname()[1]
    probe.close()
    held = listen(port)
    try:
        assert held.getsockname() == ("127.0.0.1", port)
        # A second session must still open its graph; the host reports the difference.
        fallback = listen(port)
        try:
            assert fallback.getsockname()[1] not in (0, port)
        finally:
            fallback.close()
    finally:
        held.close()


def test_service_reports_the_port_requested_by_the_host():
    from standalone_server import listen
    probe = listen()
    port = probe.getsockname()[1]
    probe.close()
    root = Path(__file__).parent.resolve().parents[1]
    child = subprocess.Popen([sys.executable, "-I", "-B", str(root / "standalone_server.py"), "--parent-pipe", "--port", str(port)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env={**os.environ, "TENSORVIZ_LOCAL_TOKEN": TOKEN}, text=True)
    try:
        assert json.loads(child.stdout.readline())["port"] == port
    finally:
        child.stdin.close()
        try:
            child.wait(timeout=5)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
