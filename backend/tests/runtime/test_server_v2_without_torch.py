"""The extension's backend starts and serves its static routes without PyTorch.

``server_v2`` is started the way the VS Code extension starts it (uvicorn, IDE
token, workspace environment) in a child process where every ``torch`` import
either is missing or fails like a broken native install. Static routes must answer
as usual. Routes that run tensors must answer 503 ``pytorch_unavailable`` naming
the feature and the reason, and nothing may execute the user's model.

Also runnable without pytest, in an environment with or without PyTorch:
    python tests/runtime/test_server_v2_without_torch.py
"""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

try:
    import pytest
except ImportError:  # Script mode: the parametrised test is called directly below.
    pytest = None

BACKEND = Path(__file__).resolve().parents[2]
TOKEN = "server-v2-no-torch-test-not-a-production-secret"

RUNNER = r'''
import importlib.abc, json, os, socket, sys
backend, failure = sys.argv[1:3]

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision'}:
            if failure == 'broken':
                raise RuntimeError('Test fixture: incompatible PyTorch native library')
            raise ModuleNotFoundError(f"No module named '{fullname}'", name=fullname)

sys.meta_path.insert(0, BlockTorch())
sys.path.insert(0, backend)
os.chdir(backend)
import uvicorn
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('127.0.0.1', 0))
print(json.dumps({'port': sock.getsockname()[1]}), flush=True)
# The same application string the extension passes to uvicorn.
server = uvicorn.Server(uvicorn.Config('server_v2:app', log_level='warning'))
server.run(sockets=[sock])
'''

MODEL = '''from pathlib import Path
Path({marker!r}).touch()
import torch
from torch import nn


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.rnn = nn.LSTM(3, 5, batch_first=True)

    def forward(self, x):
        return self.rnn(self.fc(x).unsqueeze(1))[0]
'''


def _graph():
    """Input -> Conv2d -> ReLU -> Output, in the canvas's own node format."""
    def node(node_id, kind, layer, name, parameters, x):
        return {"id": node_id, "type": kind, "position": {"x": x, "y": 0},
                "data": {"layerType": {"id": layer, "name": name, "parameters": []}, "parameters": parameters}}

    return {
        "nodes": [
            node("input", "inputNode", "input", "Input", {"shape": "1,3,32,32"}, 0),
            node("conv1", "layerNode", "conv2d", "nn.Conv2d",
                 {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "padding": 1}, 100),
            node("relu1", "layerNode", "relu", "nn.ReLU", {}, 200),
            node("output", "outputNode", "output", "Output", {}, 300),
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "conv1"},
            {"id": "e2", "source": "conv1", "target": "relu1"},
            {"id": "e3", "source": "relu1", "target": "output"},
        ],
    }


class _Server:
    def __init__(self, failure, workspace):
        state = workspace / ".tensorviz"
        state.mkdir()
        env = {
            **os.environ,
            # The variables extension/src/config.ts backendEnv() sets.
            "AUTH_DISABLED": "true",
            "TENSORVIZ_LOCAL_TOKEN": TOKEN,
            "TENSORVIZ_WORKSPACE_DIR": str(workspace),
            "TENSORVIZ_ALLOWED_ROOTS": str(workspace),
            "USER_HISTORY_DB_PATH": str(state / "state.db"),
            "TENSORVIZ_STATIC_DIR": "",
            "PYTHONUNBUFFERED": "1",
        }
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", RUNNER, str(BACKEND), failure],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            line = executor.submit(self.process.stdout.readline)
            try:
                port = json.loads(line.result(timeout=60))["port"]
            except Exception:
                self.close()
                raise AssertionError("server_v2 did not start: " + self.process.stderr.read()[-4000:])
        self.origin = f"http://127.0.0.1:{port}"
        # Do not route a loopback request through a user's HTTP proxy.
        self.client = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, method, route, payload=None, token=True):
        headers = {"x-tensorviz-token": TOKEN} if token else {}
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        call = urllib.request.Request(self.origin + route, data=data, headers=headers, method=method)
        for attempt in range(300):
            try:
                with self.client.open(call, timeout=60) as reply:
                    return reply.status, json.load(reply) if "json" in reply.headers.get("Content-Type", "") else reply.read()
            except urllib.error.HTTPError as error:
                return error.code, json.load(error)
            except (ConnectionError, urllib.error.URLError):
                # uvicorn imports server_v2 after the socket is bound.
                if attempt == 299 or self.process.poll() is not None:
                    raise AssertionError("server_v2 stopped: " + self.process.stderr.read()[-4000:])
                time.sleep(0.05)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process.stdout.close()
        self.process.stderr.close()


def _static_routes(server, workspace, marker):
    status, health = server.request("GET", "/health", token=False)
    assert status == 200 and health["status"] == "healthy", health
    status, health = server.request("GET", "/health")
    assert health["environment"]["torch"] is None and health["environment"]["libraries"] == [], health

    source = (workspace / "model.py").read_text()
    digest = __import__("hashlib").sha256(source.encode()).hexdigest()
    sources = [{"file": "model.py", "source": source, "digest": digest}]
    root = {"file": "model.py", "cls": "Tiny", "source_digest": digest}
    snapshot = {"protocol_version": 1, "root": root, "sources": sources, "truncated": False}

    status, projection = server.request("POST", "/api/ide/multifile/project", snapshot)
    assert status == 200, projection
    assert projection["traced"] is False and projection["document"]["semantic"]["nodes"], projection
    status, parsed = server.request("POST", "/api/ide/parse", {"file": "model.py", "source": source, "cls": "Tiny"})
    assert status == 200 and parsed["document"]["semantic"]["nodes"], parsed
    status, review = server.request("POST", "/api/ide/review/index",
                                    {"protocol_version": 1, "sources": sources, "truncated": False})
    assert status == 200 and review["models"][0]["cls"] == "Tiny", review
    status, setup = server.request("POST", "/api/ide/graph-setup/discover", snapshot)
    assert status == 200, setup
    status, classes = server.request("POST", "/api/ide/workspace/classes", {"root": str(workspace)})
    assert status == 200 and [c["name"] for c in classes["classes"]] == ["Tiny"], classes
    assert classes["classes"][0]["import_path"] == "model", classes
    status, buffer = server.request("POST", "/api/ide/buffer-classes", {"file": "model.py", "source": source})
    assert status == 200 and buffer["classes"][0]["name"] == "Tiny", buffer
    status, scan = server.request("POST", "/api/ingest/repo/scan", {"repo": str(workspace)})
    assert status == 200 and scan["candidates"], scan

    # The LSTM's constructor defaults come from torch.nn when it can load; without
    # it the parser keeps the explicit arguments rather than failing the parse.
    status, code = server.request("POST", "/api/code/parse", {"code": source})
    assert status == 200 and code["graphs"][0]["graph"]["nodes"], code

    status, categories = server.request("GET", "/api/templates/categories")
    assert status == 200 and categories["categories"], categories
    # Catalogue filtering runs structural validation, which must not need PyTorch:
    # an import failure there would silently empty every category.
    listed = [server.request("GET", f"/api/templates/categories/{c['id']}")[1] for c in categories["categories"]]
    assert any(entry["templates"] for entry in listed), listed

    graph = _graph()
    status, valid = server.request("POST", "/api/models/validate", {"graph": graph, "input_shape": [1, 3, 32, 32]})
    assert status == 200 and valid["valid"] is True, valid
    status, generated = server.request("POST", "/api/models/generate-code",
                                       {"graph": graph, "input_shape": [1, 3, 32, 32], "class_name": "Net"})
    assert status == 200 and "nn.Conv2d" in generated["code"], generated

    assert not marker.exists(), "A static route executed the user's model"


def _torch_routes(server, failure):
    graph = _graph()
    shaped = {"graph": graph, "input_shape": [1, 3, 32, 32]}
    routes = [
        ("POST", "/api/models/infer-shapes", shaped, "Shape inference"),
        ("POST", "/api/validate_shapes", {**graph, "input_shape": [1, 3, 32, 32]}, "Shape validation"),
        ("POST", "/api/models/verify", shaped, "Model verification"),
        ("POST", "/api/models/export-package", {**shaped, "class_name": "Net"}, "Package export"),
        ("POST", "/api/models/export-onnx", shaped, "ONNX export"),
        ("GET", "/api/ingest/libraries", None, "Model library import"),
        ("GET", "/api/ingest/catalog", None, "Model library import"),
        # Asked twice: a broken install must report the same verdict every time.
        ("POST", "/api/models/infer-shapes", shaped, "Shape inference"),
    ]
    for method, route, payload, feature in routes:
        status, body = server.request(method, route, payload)
        assert status == 503, (route, status, body)
        detail = body["detail"]
        assert detail["code"] == "pytorch_unavailable", (route, detail)
        assert detail["reason"] == failure and detail["feature"] == feature, (route, detail)
        assert feature in detail["message"] and "PyTorch" in detail["message"], (route, detail)
        assert detail["hint"] and detail["interpreter"], (route, detail)
        if failure == "broken":
            assert "incompatible PyTorch native library" in detail["error"], (route, detail)

    # IDE mode still refuses the in-process importer before PyTorch is considered.
    status, body = server.request("POST", "/api/ingest/model", {"library": "torchvision", "arch": "resnet18"})
    assert status == 403 and body["detail"]["code"] == "ide_mode_uses_worker", body


def _run(failure):
    with tempfile.TemporaryDirectory(prefix="tensorviz-server-v2-no-torch-") as temporary:
        workspace = Path(temporary).resolve()
        marker = workspace / "model-executed"
        (workspace / "model.py").write_text(MODEL.format(marker=str(marker)))
        server = _Server(failure, workspace)
        try:
            _static_routes(server, workspace, marker)
            _torch_routes(server, failure)
            assert server.process.poll() is None, "server_v2 exited while serving"
        finally:
            server.close()


def _ide_router_lazy_imports():
    """The IDE router imports most of its services inside handlers. Every one of them
    must import with PyTorch blocked, or the route that needs it fails without it."""
    import ast

    tree = ast.parse((BACKEND / "api" / "routers" / "ide.py").read_text())
    modules = sorted({node.module for node in ast.walk(tree)
                      if isinstance(node, ast.ImportFrom) and node.module and node.level == 0})
    blocker = RUNNER.split("import uvicorn")[0]
    script = blocker + "for name in sys.argv[3:]:\n    __import__(name)\n"
    for failure in ("missing", "broken"):
        result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(BACKEND), failure, *modules],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr[-4000:]


if pytest is not None:
    @pytest.mark.parametrize("failure", ["missing", "broken"])
    def test_server_v2_serves_static_routes_without_pytorch(failure):
        _run(failure)

    def test_ide_router_lazy_imports_load_without_pytorch():
        _ide_router_lazy_imports()

    def test_in_process_routes_report_missing_pytorch(monkeypatch):
        """Outside IDE mode the importer runs in-process, so it needs PyTorch too."""
        from fastapi.testclient import TestClient
        import server_v2

        # `import torch` raises ModuleNotFoundError(name="torch") while this is None.
        monkeypatch.setitem(sys.modules, "torch", None)
        client = TestClient(server_v2.app, base_url="http://127.0.0.1")
        for route, payload in (("/api/ingest/model", {"library": "torchvision", "arch": "resnet18", "save": False}),
                               ("/api/ingest/repo", {"repo": str(BACKEND), "save": False})):
            response = client.post(route, json=payload)
            assert response.status_code == 503, response.text
            assert response.json()["detail"]["reason"] == "missing"
            assert response.json()["detail"]["feature"] == "Model import"


if __name__ == "__main__":
    for case in ("missing", "broken"):
        _run(case)
    _ide_router_lazy_imports()
    print("server_v2 served its static routes and reported PyTorch as missing and as broken.")
