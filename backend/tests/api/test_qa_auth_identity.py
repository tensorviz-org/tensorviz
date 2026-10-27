"""Durable regression tests for Auth & identity (local token, gated routers, CORS).

Covers:

* The server-level auth GATE (``app.include_router(..., dependencies=[Depends(
  get_current_user)])`` as wired in ``server_v2.py``): unauthenticated callers
  are rejected with 401, ``AUTH_DISABLED`` bypasses the gate, and a bearer
  token does not.
* CORS allowlist behaviour + the env-driven ``CORS_ORIGINS`` parsing.

Owner scoping for the remaining personal-template API is exercised in
``test_user_templates_router.py``.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient


from auth import get_current_user  # noqa: E402


# ── Gated-router harness (mirrors server_v2.py) ────────────────────────────


def _gated_app() -> FastAPI:
    """A tiny router behind the same auth gate server_v2 uses."""
    router = APIRouter()

    @router.get("/ping")
    @router.post("/ping")
    def ping():
        return {"ok": True}

    app = FastAPI()
    app.include_router(router, dependencies=[Depends(get_current_user)])
    return app


# ── The auth GATE on stateful routers ──────────────────────────────────────


def test_auth_disabled_bypasses_gated_router(monkeypatch):
    """AUTH_DISABLED=true lets an un-tokened request through the gate (dev principal)."""
    monkeypatch.setenv("AUTH_DISABLED", "true")
    client = TestClient(_gated_app(), raise_server_exceptions=False)

    assert client.get("/ping").status_code == 200
    assert client.post("/ping").status_code == 200


def test_gated_router_rejects_missing_token(monkeypatch):
    """With auth enabled, the app-level gate 401s BEFORE the handler runs."""
    monkeypatch.setenv("AUTH_DISABLED", "false")
    client = TestClient(_gated_app(), raise_server_exceptions=False)

    got = client.get("/ping")
    assert got.status_code == 401
    assert (got.headers.get("www-authenticate") or "").lower() == "x-tensorviz-token"

    # The mutating endpoint is gated too — no forging without a token.
    posted = client.post("/ping")
    assert posted.status_code == 401


def test_gated_router_rejects_bearer_token(monkeypatch):
    """A bearer token is not a local launch token and does not pass the gate."""
    monkeypatch.setenv("AUTH_DISABLED", "false")
    client = TestClient(_gated_app(), raise_server_exceptions=False)

    got = client.get("/ping", headers={"Authorization": "Bearer good-token"})
    assert got.status_code == 401
    assert "does not accept bearer tokens" in got.json()["detail"]


# ── CORS allowlist (server_v2.py:45-46) ─────────────────────────────────────


def _parse_cors_origins() -> list[str]:
    """Mirror of server_v2.py's CORS_ORIGINS parsing (comma-split, strip, drop empties)."""
    return [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]


def _cors_app(origins: list[str]) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/ping")
    def ping():
        return {"ok": True}

    return app


def test_cors_allows_configured_origin_and_blocks_others(monkeypatch):
    """Only origins on the CORS_ORIGINS allowlist get an Access-Control-Allow-Origin echo."""
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000, https://app.tensorviz.io")
    origins = _parse_cors_origins()
    assert origins == ["http://localhost:3000", "https://app.tensorviz.io"]

    client = TestClient(_cors_app(origins))

    for allowed in origins:
        r = client.get("/ping", headers={"Origin": allowed})
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == allowed

    # A non-allowlisted origin is NOT granted the CORS header (browser blocks it).
    denied = client.get("/ping", headers={"Origin": "http://evil.example"})
    assert denied.headers.get("access-control-allow-origin") is None

    # And a CORS preflight from a foreign origin is rejected outright.
    preflight = client.options(
        "/ping",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert preflight.status_code == 400


def test_cors_origins_env_parsing(monkeypatch):
    """CORS_ORIGINS parsing: default fallback, comma-split, whitespace + empty pruning."""
    # Unset -> defaults to the local frontend (secure, non-empty default).
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert _parse_cors_origins() == ["http://localhost:3000"]

    # Messy value: surrounding spaces stripped, blank segments dropped.
    monkeypatch.setenv("CORS_ORIGINS", " http://a.com , ,https://b.com, ")
    assert _parse_cors_origins() == ["http://a.com", "https://b.com"]
