"""Smoke the actual HTTP backend and worker in an environment without AI SDKs.

Run in a fresh venv with requirements-project.txt installed. No pytest dependency.
The smoke owns its database, socket and model fixture, and never loads .env files.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid

BACKEND = Path(__file__).resolve().parents[1]
# FastAPI can depend on the inert OpenTelemetry API without installing an SDK.
PROVIDER_MODULES = ("openai", "pydantic_ai", "langgraph", "langchain_core", "opentelemetry.sdk")
MODEL = '''from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
    def forward(self, x):
        return self.fc(x)
'''


def main() -> None:
    present = []
    for name in PROVIDER_MODULES:
        try:
            if importlib.util.find_spec(name) is not None:
                present.append(name)
        except ModuleNotFoundError:
            pass  # A missing parent package also means its SDK is absent.
    if present:
        raise RuntimeError(f"Use a clean environment without these packages: {', '.join(present)}")
    sys.path.insert(0, str(BACKEND))
    with tempfile.TemporaryDirectory(prefix="tensorviz-provider-free-") as temporary:
        token = uuid.uuid4().hex
        os.environ.update({
            "PYTHON_DOTENV_DISABLED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "USER_HISTORY_DB_PATH": str(Path(temporary) / "state.sqlite3"),
            "TENSORVIZ_LOCAL_TOKEN": token,
            "TENSORVIZ_STATIC_DIR": "",
            "AUTH_DISABLED": "false",
        })
        import requests
        import uvicorn
        from server_v2 import app

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        address = f"http://127.0.0.1:{listener.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        session = requests.Session()
        session.trust_env = False
        session.headers["X-tensorViz-Token"] = token
        try:
            deadline = time.monotonic() + 15
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert server.started, "backend did not start"
            health = session.get(address + "/health", timeout=5)
            assert health.status_code == 200
            assert health.json()["status"] == "healthy" and "ai_runtime" not in health.json()
            # HTTP health alone passes when uvicorn cannot serve WebSockets.
            from websockets.sync.client import connect
            minted = session.post(address + "/local/sessions",
                json={"launchOrigin": address}, timeout=5)
            assert minted.status_code == 200
            material = minted.json()
            bootstrapped = session.post(address + "/local/bootstrap",
                headers={"Origin": address}, json={"workspaceSessionId": material["workspaceSessionId"],
                    "credential": material["viewCredential"]}, timeout=5)
            assert bootstrapped.status_code == 200
            for role, cookie in [
                ("host", f'{material["cookieName"]}={material["hostCookieKey"]}'),
                ("view", "; ".join(f"{name}={value}" for name, value in session.cookies.items())),
            ]:
                with connect(address.replace("http:", "ws:") + "/local/ws", origin=address,
                    additional_headers={"Cookie": cookie},
                    open_timeout=5, close_timeout=3) as connection:
                    connection.send(json.dumps({"type": f"{role}Handshake", "role": role,
                        "workspaceSessionId": material["workspaceSessionId"],
                        "protocolVersion": material["protocolVersion"], "messageId": f"smoke-{role}"}))
                    greeting = json.loads(connection.recv(timeout=5))
                    assert greeting["type"] == "connected" and greeting["role"] == role
            for path in ("/api/ai/turns", "/api/agent/explain-graph", "/api/ai/sessions"):
                assert session.post(address + path, json={}, timeout=5).status_code == 404, path
            # If static projection imports the submitted model, this must fail.
            source = MODEL + "\nraise RuntimeError('static inspection must not execute models')\n"
            target = Path(temporary) / "net.py"
            target.write_text(source, encoding="utf-8")
            payload = {"file": str(target), "cls": "Net", "source": source}
            response = session.post(address + "/api/ide/project", json=payload, timeout=10)
            assert response.status_code == 200, response.text
            projected = response.json()
            assert projected["traced"] is False and "fc" in projected["spanmap"]
            apply_payload = {**payload, "source_digest": projected["source_digest"],
                "spanmap": projected["spanmap"], "ops": [{"op": "set_param", "op_id": "check",
                    "node_id": "fc", "param": "out_features", "value": 3}]}
            response = session.post(address + "/api/ide/apply", json=apply_payload, timeout=10)
            assert response.status_code == 200, response.text
            edit = response.json()
            assert edit["edits"] and edit["refused"] == []
            assert edit["source_digest"] != projected["source_digest"]
            assert target.read_text(encoding="utf-8") == source, "backend wrote model source"
            # Execution is a separate, explicit worker call with bounded tensor inputs.
            target.write_text(MODEL, encoding="utf-8")
            process = subprocess.run([sys.executable, "-m", "services.ingest.worker",
                "--file", str(target), "--cls", "Net", "--tensor-inputs",
                json.dumps([{"shape": [1, 4], "dtype": "float32", "fill": "ones"}])],
                cwd=BACKEND, env=dict(os.environ), capture_output=True, text=True, timeout=60)
            envelope = json.loads(process.stdout)
            assert process.returncode == 0 and envelope["error"] is None, envelope.get("error")
            assert envelope["result"]["graph"]["nodes"]
            assert not any(name == module or name.startswith(module + '.')
                           for name in sys.modules for module in PROVIDER_MODULES)
            print("PASS: provider-free HTTP/WebSocket startup, removed endpoints, static projection, source proposal, real trace")
        finally:
            session.close()
            server.should_exit = True
            thread.join(timeout=10)
            listener.close()
            assert not thread.is_alive(), "backend did not stop"


if __name__ == "__main__":
    main()
