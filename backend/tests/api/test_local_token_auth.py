"""Per-workspace identity: the ``LocalTokenAuth`` branch of auth.py (task.md M9).

Why this matters, not just what it does. The IDE's backend has no user: it is
started by an editor, on loopback, for one folder. Before M9 every such backend
answered as the same ``local-dev-user``, so two workspaces open on one machine
wrote into one owner's rows and each could read and overwrite the other's
projects. The token now identifies *the workspace* — one uid per folder, derived
from the folder and nothing else — and an unauthenticated caller on that port is
refused rather than silently promoted to the dev principal.

The three modes are ordered, and the order is asserted here: AUTH_DISABLED stays
the outermost branch so the suite and start.sh keep the principal they had.
"""

import hashlib
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import auth
from auth import CurrentUser, get_current_user
from middleware import LOCAL_TOKEN_HEADER

TOKEN = "launch-secret-4f2b"


def _client() -> TestClient:
    app = FastAPI()

    @app.get("/whoami")
    def whoami(user: CurrentUser = Depends(get_current_user)):
        return {"uid": user.uid, "email": user.email}

    return TestClient(app, raise_server_exceptions=False)


def _expected_uid(folder: Path) -> str:
    return "workspace:" + hashlib.sha256(str(folder.resolve()).encode("utf-8")).hexdigest()


@pytest.fixture
def ide_env(monkeypatch, tmp_path: Path) -> Path:
    """The environment the extension leaves behind: a launch token and a state DB
    inside the opened folder (extension/src/config.ts backendEnv)."""
    monkeypatch.delenv("AUTH_DISABLED", raising=False)
    monkeypatch.setenv("TENSORVIZ_LOCAL_TOKEN", TOKEN)
    monkeypatch.setenv("USER_HISTORY_DB_PATH", str(tmp_path / ".tensorviz" / "state.sqlite3"))
    return tmp_path


def test_token_resolves_to_the_workspace_uid(ide_env: Path):
    """The whole point: the request is the folder, and says so."""
    got = _client().get("/whoami", headers={LOCAL_TOKEN_HEADER: TOKEN})
    assert got.status_code == 200
    assert got.json()["uid"] == _expected_uid(ide_env)


def test_uid_hides_the_folder_it_came_from(ide_env: Path):
    """A uid is stored in every row and shipped to the model; a raw path there
    would leak the user's home directory layout into data we persist."""
    uid = _client().get("/whoami", headers={LOCAL_TOKEN_HEADER: TOKEN}).json()["uid"]
    assert uid.startswith("workspace:")
    assert str(ide_env) not in uid
    assert ide_env.name not in uid


def test_missing_token_is_401_when_auth_is_not_disabled(ide_env: Path):
    """Nothing else on the machine that can reach the loopback port gets in."""
    assert _client().get("/whoami").status_code == 401


def test_wrong_token_is_401(ide_env: Path):
    assert _client().get("/whoami", headers={LOCAL_TOKEN_HEADER: "not-the-token"}).status_code == 401


def test_two_workspaces_are_two_owners(monkeypatch, tmp_path: Path):
    """The regression this milestone exists to prevent: two folders, two owners."""
    monkeypatch.delenv("AUTH_DISABLED", raising=False)
    monkeypatch.setenv("TENSORVIZ_LOCAL_TOKEN", TOKEN)

    uids = []
    for name in ("alpha", "beta"):
        folder = tmp_path / name
        monkeypatch.setenv("USER_HISTORY_DB_PATH", str(folder / ".tensorviz" / "state.sqlite3"))
        uids.append(_client().get("/whoami", headers={LOCAL_TOKEN_HEADER: TOKEN}).json()["uid"])

    assert uids[0] != uids[1]


def test_uid_is_stable_across_launches(ide_env: Path, monkeypatch):
    """A new launch token every time must not orphan the last launch's projects."""
    first = _client().get("/whoami", headers={LOCAL_TOKEN_HEADER: TOKEN}).json()["uid"]

    monkeypatch.setenv("TENSORVIZ_LOCAL_TOKEN", "a-different-launch-secret")
    second = _client().get("/whoami", headers={LOCAL_TOKEN_HEADER: "a-different-launch-secret"})
    assert second.json()["uid"] == first


def test_auth_disabled_still_wins(ide_env: Path, monkeypatch):
    """Ordering, asserted: AUTH_DISABLED is checked first, so the test suite and
    start.sh keep the dev principal even if a token is lying around. Flipping
    this order is a product decision (see the report), not an accident."""
    monkeypatch.setenv("AUTH_DISABLED", "true")
    got = _client().get("/whoami", headers={LOCAL_TOKEN_HEADER: TOKEN})
    assert got.json()["uid"] == auth.DEV_UID


def test_unknown_workspace_fails_loud(monkeypatch):
    """A token with no workspace behind it must not fall back to a shared uid:
    that is how one workspace silently adopts another's rows."""
    monkeypatch.delenv("AUTH_DISABLED", raising=False)
    monkeypatch.setenv("TENSORVIZ_LOCAL_TOKEN", TOKEN)
    monkeypatch.delenv("USER_HISTORY_DB_PATH", raising=False)

    got = _client().get("/whoami", headers={LOCAL_TOKEN_HEADER: TOKEN})
    assert got.status_code == 503
    assert "USER_HISTORY_DB_PATH" in got.json()["detail"]


@pytest.mark.parametrize("relative", [(".tensorviz", "state.sqlite3"), ("state.sqlite3",)])
def test_workspace_folder_strips_the_state_dir(monkeypatch, tmp_path: Path, relative):
    """The folder is recovered from the state DB path, so ``<ws>/.tensorviz/db``
    and a bare ``<ws>/db`` both name ``<ws>`` — not ``<ws>/.tensorviz``."""
    monkeypatch.setenv("USER_HISTORY_DB_PATH", str(tmp_path.joinpath(*relative)))
    assert auth.workspace_folder() == tmp_path.resolve()


def test_no_token_rejects_without_a_bearer_identity(monkeypatch):
    """With no launch token set, an unauthenticated caller gets a 401.

    Bearer tokens are not a second identity provider."""
    monkeypatch.delenv("AUTH_DISABLED", raising=False)
    monkeypatch.delenv("TENSORVIZ_LOCAL_TOKEN", raising=False)

    got = _client().get("/whoami")
    assert got.status_code == 401
    assert (got.headers.get("www-authenticate") or "").lower() == "x-tensorviz-token"


def test_explicit_workspace_identity_is_independent_of_private_state(monkeypatch, tmp_path):
    workspace = tmp_path / "source"
    monkeypatch.setenv("TENSORVIZ_WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("USER_HISTORY_DB_PATH", str(tmp_path / "vscode-storage/.tensorviz/state.sqlite3"))
    assert auth.workspace_folder() == workspace.resolve()
