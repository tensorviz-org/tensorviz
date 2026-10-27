"""QA regression tests for the owner-scoped personal-templates API (PRs #54,#55,#57).

Complements ``test_user_templates_router.py`` (which covers the happy-path CRUD +
graph round-trip, basic owner-scoping, and empty-name rejection). These tests pin
behaviors that existing coverage leaves unguarded: the exact response payload
shape, create-time defaults, serialization fidelity of complex graphs, the
``_decode`` corrupt-JSON fallback, newest-first list ordering, the soft-delete
guard, and per-uid list isolation. Two currently-broken behaviors are pinned as
``xfail`` so a future fix surfaces as an unexpected pass instead of a silent
regression.
"""

from __future__ import annotations

import sqlite3
import uuid
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


def _force_created_at(created_at_by_id: dict[str, str]) -> None:
    """Overwrite created_at on stored rows to make ordering assertions deterministic."""
    with sqlite3.connect(str(user_templates._store._db_path)) as conn:
        for tid, ts in created_at_by_id.items():
            conn.execute(
                "UPDATE app_user_templates SET created_at = ? WHERE id = ?",
                (ts, tid),
            )
        conn.commit()


# ── payload shape / contract ────────────────────────────────────────────────


def test_created_template_payload_shape(tmp_path: Path):
    """The create response must expose the full owner-scoped template contract.

    The frontend "My Templates" list and the duplicate-to-project flow read these
    exact fields; dropping or renaming one would break them silently.
    """
    app = _make_app(tmp_path)
    _as_user(app, "userA")
    client = TestClient(app)

    graph = {"nodes": [{"id": "in"}], "edges": []}
    resp = client.post(
        "/api/user-templates",
        json={"name": "Shape", "canonical_graph": graph, "description": "desc"},
    )
    assert resp.status_code == 200
    template = resp.json()["template"]

    assert set(template.keys()) == {
        "id",
        "owner_uid",
        "name",
        "description",
        "canonical_graph",
        "created_at",
        "updated_at",
    }
    assert template["owner_uid"] == "userA"  # scoped to the authenticated caller
    assert template["name"] == "Shape"
    assert template["description"] == "desc"
    assert template["canonical_graph"] == graph  # returned decoded, not as a JSON string
    assert isinstance(template["id"], str) and template["id"]
    assert isinstance(template["created_at"], str) and template["created_at"]
    assert isinstance(template["updated_at"], str) and template["updated_at"]


def test_create_defaults_empty_graph_and_null_description(tmp_path: Path):
    """Name-only create stores an empty canonical graph and a null description.

    A template saved before any graph exists must round-trip as ``{}`` (an empty
    object), not the literal string ``"{}"`` and not ``None``.
    """
    app = _make_app(tmp_path)
    _as_user(app, "userA")
    client = TestClient(app)

    template = client.post("/api/user-templates", json={"name": "Bare"}).json()["template"]
    assert template["canonical_graph"] == {}
    assert template["description"] is None

    # The default survives a GET round-trip too.
    fetched = client.get(f"/api/user-templates/{template['id']}").json()["template"]
    assert fetched["canonical_graph"] == {}
    assert fetched["description"] is None


def test_canonical_graph_roundtrips_nested_payload(tmp_path: Path):
    """A structurally rich graph must survive JSON serialize/deserialize exactly.

    Guards against a storage-encoding change that silently mangles nested objects,
    number types, or unicode in a saved template.
    """
    app = _make_app(tmp_path)
    _as_user(app, "userA")
    client = TestClient(app)

    graph = {
        "nodes": [
            {"id": "conv", "params": {"kernel": 3, "stride": 1, "bias": True}},
            {"id": "act", "label": "relu → gelu"},
        ],
        "edges": [{"source": "conv", "target": "act", "weight": 0.5}],
        "meta": {"tags": ["cnn", "vision"], "depth": 12},
    }
    tid = client.post(
        "/api/user-templates", json={"name": "Nested", "canonical_graph": graph}
    ).json()["template"]["id"]

    fetched = client.get(f"/api/user-templates/{tid}").json()["template"]
    assert fetched["canonical_graph"] == graph


def test_decode_falls_back_to_empty_on_corrupt_stored_json(tmp_path: Path):
    """A row whose canonical_graph is not valid JSON is served as ``{}``, not a 500.

    ``_decode`` defensively swallows a decode error so one corrupt row cannot break
    the whole My-Templates fetch. Written by hitting the store directly because the
    API always stores well-formed JSON.
    """
    app = _make_app(tmp_path)
    _as_user(app, "userA")
    client = TestClient(app)

    row = user_templates._store.create_user_template(
        "userA", name="Corrupt", canonical_graph="}{ not json", description=None
    )

    fetched = client.get(f"/api/user-templates/{row['id']}").json()["template"]
    assert fetched["canonical_graph"] == {}


# ── ordering ────────────────────────────────────────────────────────────────


def test_list_returns_newest_first(tmp_path: Path):
    """The list is newest-first, so the most recently saved template surfaces on top.

    created_at is forced to distinct values so the assertion cannot pass by luck of
    insertion order.
    """
    app = _make_app(tmp_path)
    _as_user(app, "userA")
    client = TestClient(app)

    a = client.post("/api/user-templates", json={"name": "oldest"}).json()["template"]["id"]
    b = client.post("/api/user-templates", json={"name": "middle"}).json()["template"]["id"]
    c = client.post("/api/user-templates", json={"name": "newest"}).json()["template"]["id"]
    _force_created_at(
        {
            a: "2026-01-01T00:00:01.000000Z",
            b: "2026-01-01T00:00:02.000000Z",
            c: "2026-01-01T00:00:03.000000Z",
        }
    )

    names = [t["name"] for t in client.get("/api/user-templates").json()["templates"]]
    assert names == ["newest", "middle", "oldest"]


# ── delete semantics ────────────────────────────────────────────────────────


def test_double_delete_returns_404(tmp_path: Path):
    """A soft-deleted template is gone: the second delete 404s and it never re-lists.

    Pins the ``deleted_at IS NULL`` guard so a delete is not silently reported as
    successful twice.
    """
    app = _make_app(tmp_path)
    _as_user(app, "userA")
    client = TestClient(app)

    tid = client.post("/api/user-templates", json={"name": "Once"}).json()["template"]["id"]
    assert client.delete(f"/api/user-templates/{tid}").json() == {"ok": True}
    assert client.delete(f"/api/user-templates/{tid}").status_code == 404
    assert client.get(f"/api/user-templates/{tid}").status_code == 404
    assert client.get("/api/user-templates").json()["templates"] == []


def test_unknown_id_returns_404(tmp_path: Path):
    """Fetching or deleting a never-existed id 404s rather than 200-with-empty."""
    app = _make_app(tmp_path)
    _as_user(app, "userA")
    client = TestClient(app)

    ghost = uuid.uuid4().hex
    assert client.get(f"/api/user-templates/{ghost}").status_code == 404
    assert client.delete(f"/api/user-templates/{ghost}").status_code == 404


# ── per-uid isolation ───────────────────────────────────────────────────────


def test_templates_listing_is_per_uid(tmp_path: Path):
    """Each user's list contains only their own templates when both users have some.

    Extends the existing owner-scoping test (which only had one user own anything)
    to the both-populated case, where a leak would actually be observable.
    """
    app = _make_app(tmp_path)
    client = TestClient(app)

    _as_user(app, "userA")
    client.post("/api/user-templates", json={"name": "A1"})
    client.post("/api/user-templates", json={"name": "A2"})

    _as_user(app, "userB")
    client.post("/api/user-templates", json={"name": "B1"})
    assert {t["name"] for t in client.get("/api/user-templates").json()["templates"]} == {"B1"}

    _as_user(app, "userA")
    assert {t["name"] for t in client.get("/api/user-templates").json()["templates"]} == {"A1", "A2"}


# ── documented bugs (kept GREEN via xfail) ──────────────────────────────────


def test_whitespace_only_name_is_rejected(tmp_path: Path):
    """A blank (whitespace-only) name is not a usable label and should 422.

    Consistency with the empty-string case, which the router already rejects.
    """
    app = _make_app(tmp_path)
    _as_user(app, "userA")
    client = TestClient(app)

    assert client.post("/api/user-templates", json={"name": "   "}).status_code == 422


def test_list_tie_break_is_deterministic_newest_first(tmp_path: Path):
    """With identical created_at, the more recently saved template must still lead.

    Intent: ordering is stable and newest-first even under equal timestamps (e.g.
    a ``created_at DESC, id DESC`` / rowid tie-break).
    """
    app = _make_app(tmp_path)
    _as_user(app, "userA")
    client = TestClient(app)

    first = client.post("/api/user-templates", json={"name": "first"}).json()["template"]["id"]
    second = client.post("/api/user-templates", json={"name": "second"}).json()["template"]["id"]
    _force_created_at(
        {
            first: "2026-01-01T00:00:00.000000Z",
            second: "2026-01-01T00:00:00.000000Z",
        }
    )

    names = [t["name"] for t in client.get("/api/user-templates").json()["templates"]]
    # "second" was saved after "first"; newest-first means it must appear earlier.
    assert names.index("second") < names.index("first")
