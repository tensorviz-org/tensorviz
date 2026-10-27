"""Tests for the local auth dependency."""
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import auth
from auth import CurrentUser, get_current_user


def _client() -> TestClient:
    app = FastAPI()

    @app.get("/whoami")
    def whoami(user: CurrentUser = Depends(get_current_user)):
        return {"uid": user.uid, "email": user.email}

    return TestClient(app, raise_server_exceptions=False)


def test_auth_enabled_by_default(monkeypatch):
    monkeypatch.delenv("AUTH_DISABLED", raising=False)
    assert auth.auth_disabled() is False


@pytest.mark.parametrize("value", ["true", "1", "yes", "on", "TRUE"])
def test_auth_disabled_returns_dev_user(monkeypatch, value):
    monkeypatch.setenv("AUTH_DISABLED", value)
    r = _client().get("/whoami")
    assert r.status_code == 200
    assert r.json()["uid"] == auth.DEV_UID


def test_missing_token_is_401(monkeypatch):
    monkeypatch.setenv("AUTH_DISABLED", "false")
    r = _client().get("/whoami")
    assert r.status_code == 401


def test_bearer_token_is_401(monkeypatch):
    monkeypatch.setenv("AUTH_DISABLED", "false")
    r = _client().get("/whoami", headers={"Authorization": "Bearer garbage"})
    assert r.status_code == 401
    assert "does not accept bearer tokens" in r.json()["detail"]
