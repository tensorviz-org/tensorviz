"""IDE mode: the per-launch token, the static build, and the closed importer.

Three properties, all of them conditional on the environment the VS Code
extension sets (task.md M2, decisions D2/D12):

  * with TENSORVIZ_LOCAL_TOKEN set, every /api/* request must carry the secret —
    the IDE's backend runs with AUTH_DISABLED on a loopback port, so the token is
    the only thing between it and anything else on the machine;
  * with TENSORVIZ_STATIC_DIR set, the production React build is served from this
    origin, mounted LAST so it cannot shadow the API;
  * the in-process importer (/api/ingest/model, /api/ingest/repo), which runs
    user and library code inside this process, is refused in IDE mode — that work
    belongs to the trace worker in the user's own interpreter (M4).

And the property that keeps every other caller unaffected: with neither variable
set nothing above is registered, which is what start.sh, `make e2e` and the rest
of this suite rely on.

server_v2 reads both variables at import time, so these tests reload the module
under a patched environment and reload it back afterwards — the backend suite
runs in one process, and a leaked token would 401 every other HTTP test.
"""

import importlib
import os

import pytest
from fastapi.testclient import TestClient


import server_v2  # noqa: E402

TOKEN = "test-token-8f2c"


def _reload_with(env: dict) -> None:
    """Reload server_v2 with exactly `env` applied on top of the real environment."""
    previous = {key: os.environ.get(key) for key in env}
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        importlib.reload(server_v2)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture
def restore_server():
    """Guarantee the module every other test imported is the one left behind."""
    yield
    _reload_with({"TENSORVIZ_LOCAL_TOKEN": None, "TENSORVIZ_STATIC_DIR": None})


@pytest.fixture
def ide_client(restore_server) -> TestClient:
    _reload_with({"TENSORVIZ_LOCAL_TOKEN": TOKEN, "TENSORVIZ_STATIC_DIR": None})
    return TestClient(server_v2.app)


def test_no_token_no_middleware():
    """The default app is unchanged: no header, no 401, and it says so."""
    client = TestClient(server_v2.app)
    assert client.get("/api/layer-types").status_code == 200
    capabilities = client.get("/health").json()["capabilities"]
    assert capabilities == {"ide": False, "static": False}


def test_api_requires_the_launch_token(ide_client: TestClient):
    unauthenticated = ide_client.get("/api/layer-types")
    assert unauthenticated.status_code == 401
    assert unauthenticated.json()["detail"]["code"] == "local_token_required"

    authenticated = ide_client.get("/api/layer-types", headers={"X-tensorViz-Token": TOKEN})
    assert authenticated.status_code == 200


def test_a_wrong_token_is_not_a_partial_match(ide_client: TestClient):
    """A prefix of the token is as rejected as no token at all."""
    response = ide_client.get("/api/layer-types", headers={"X-tensorViz-Token": TOKEN[:-1]})
    assert response.status_code == 401


def test_health_is_reachable_without_the_token(ide_client: TestClient):
    """The extension probes /health before it has anywhere to put a header, and
    the readiness loop must not read a 401 as "not up yet" forever."""
    response = ide_client.get("/health")
    assert response.status_code == 200
    assert response.json()["capabilities"]["ide"] is True


def test_ide_mode_disables_inprocess_ingest(ide_client: TestClient):
    """Authenticated and still refused: the token proves who is calling, not that
    importing a repository into this process became safe."""
    response = ide_client.post(
        "/api/ingest/repo",
        json={"repo": "/tmp"},
        headers={"X-tensorViz-Token": TOKEN},
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "ide_mode_uses_worker"

    model = ide_client.post(
        "/api/ingest/model",
        json={"library": "torchvision", "arch": "resnet18"},
        headers={"X-tensorViz-Token": TOKEN},
    )
    assert model.status_code == 403
    assert model.json()["detail"]["code"] == "ide_mode_uses_worker"


def test_scan_is_not_refused(ide_client: TestClient):
    """Only the two routes that instantiate are closed. /api/ingest/repo/scan reads
    source without importing it, so refusing it would be theatre — it answers on
    its own terms (a 4xx for a path that is not a repository)."""
    response = ide_client.post(
        "/api/ingest/repo/scan",
        json={"repo": "/tmp"},
        headers={"X-tensorViz-Token": TOKEN},
    )
    assert response.status_code != 403


def test_unauthenticated_health_hides_runtime_versions_in_token_mode(ide_client: TestClient):
    """A readiness probe can tell that the process is up. It cannot fingerprint it."""
    public = ide_client.get("/health").json()
    assert public["status"] == "healthy"
    assert public["capabilities"]["ide"] is True
    assert "torch" not in public["environment"]
    assert "python" not in public["environment"]
    assert "libraries" not in public["environment"]

    authed = ide_client.get("/health", headers={"X-tensorViz-Token": TOKEN}).json()
    assert "torch" in authed["environment"]
    assert authed["environment"]["python"]


def test_token_mode_does_not_publish_the_schema(ide_client: TestClient):
    assert ide_client.get("/openapi.json").status_code == 404
    assert ide_client.get("/docs").status_code == 404
    assert ide_client.get("/redoc").status_code == 404


def test_schema_stays_available_without_a_launch_token():
    client = TestClient(server_v2.app)
    assert client.get("/openapi.json").status_code == 200


def test_a_rebinding_host_is_rejected_and_a_loopback_host_is_not():
    client = TestClient(server_v2.app)
    assert client.get("/health", headers={"host": "attacker.example"}).status_code == 400
    assert client.get("/health", headers={"host": "192.0.2.10"}).status_code == 400
    assert TestClient(server_v2.app, base_url="http://127.0.0.1").get("/health").status_code == 200


def test_a_numeric_host_is_allowed_only_when_a_launch_token_is_configured(ide_client: TestClient):
    assert ide_client.get("/health", headers={"host": "192.0.2.10"}).status_code == 200
    assert ide_client.get("/health", headers={"host": "attacker.example"}).status_code == 400


def test_static_build_is_served_without_shadowing_the_api(tmp_path, restore_server):
    build = tmp_path / "app"
    build.mkdir()
    (build / "index.html").write_text('<div id="root"></div>', encoding="utf-8")
    _reload_with({"TENSORVIZ_LOCAL_TOKEN": None, "TENSORVIZ_STATIC_DIR": str(build)})
    client = TestClient(server_v2.app)

    index = client.get("/")
    assert index.status_code == 200
    assert 'id="root"' in index.text
    assert client.get("/health").json()["capabilities"]["static"] is True

    # The whole reason the mount is the last statement in server_v2: a mount on
    # "/" matches every path, so a route registered after it would never be hit.
    assert client.get("/api/layer-types").status_code == 200
