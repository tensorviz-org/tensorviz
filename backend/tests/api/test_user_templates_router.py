"""HTTP tests for the owner-scoped personal-templates API (Phase 5 §7)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


from api.routers import user_templates  # noqa: E402
from auth import CurrentUser, get_current_user  # noqa: E402
from services.project_store import ProjectStore  # noqa: E402


def _make_app(tmp_path: Path) -> FastAPI:
    user_templates._store = ProjectStore(str(tmp_path / "t.sqlite3"))
    app = FastAPI()
    app.include_router(user_templates.router)
    return app


def _as_user(app: FastAPI, uid: str) -> None:
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(uid=uid)


def test_user_template_crud_and_graph_roundtrip(tmp_path: Path):
    app = _make_app(tmp_path)
    _as_user(app, "userA")
    client = TestClient(app)

    graph = {"nodes": [{"id": "in"}, {"id": "fc"}], "edges": [{"source": "in", "target": "fc"}]}
    created = client.post("/api/user-templates", json={"name": "My MLP", "canonical_graph": graph, "description": "d"})
    assert created.status_code == 200
    tid = created.json()["template"]["id"]

    listed = client.get("/api/user-templates").json()["templates"]
    assert len(listed) == 1 and listed[0]["name"] == "My MLP"

    fetched = client.get(f"/api/user-templates/{tid}").json()["template"]
    assert fetched["canonical_graph"] == graph  # decoded round-trip
    assert fetched["description"] == "d"

    assert client.delete(f"/api/user-templates/{tid}").json() == {"ok": True}
    assert client.get("/api/user-templates").json()["templates"] == []
    assert client.get(f"/api/user-templates/{tid}").status_code == 404


def test_user_templates_are_owner_scoped(tmp_path: Path):
    app = _make_app(tmp_path)
    client = TestClient(app)

    _as_user(app, "userA")
    tid = client.post("/api/user-templates", json={"name": "A's template"}).json()["template"]["id"]

    # userB cannot see or fetch userA's template.
    _as_user(app, "userB")
    assert client.get("/api/user-templates").json()["templates"] == []
    assert client.get(f"/api/user-templates/{tid}").status_code == 404
    assert client.delete(f"/api/user-templates/{tid}").status_code == 404

    # userA still has it.
    _as_user(app, "userA")
    assert len(client.get("/api/user-templates").json()["templates"]) == 1


def test_create_requires_name(tmp_path: Path):
    app = _make_app(tmp_path)
    _as_user(app, "userA")
    client = TestClient(app)
    assert client.post("/api/user-templates", json={"name": ""}).status_code == 422
