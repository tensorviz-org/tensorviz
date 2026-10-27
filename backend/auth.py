"""Local authentication for the API.

`get_current_user` is a FastAPI dependency.

Set ``AUTH_DISABLED=true`` to bypass verification for local dev + tests — it
returns a fixed dev principal. Auth is ENABLED by default (secure default);
disable it explicitly.

Set ``TENSORVIZ_LOCAL_TOKEN`` (what the VS Code extension does, one secret per
launch) for the IDE mode: a request carrying that secret in the
``X-tensorViz-Token`` header is the workspace itself, and gets the uid
``workspace:<sha256 of the workspace folder>`` — so two folders opened against
one machine own separate rows without either of them signing in. Anything else
is a 401. The modes are checked in the order written in :func:`get_current_user`.

Bearer tokens are not accepted. Hosted account sign-in is not part of this
backend; the IDE and the standalone server use the local launch token.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request, status

from middleware import LOCAL_TOKEN_HEADER

AUTH_DISABLED_ENV = "AUTH_DISABLED"
DEV_UID = "local-dev-user"

LOCAL_TOKEN_ENV = "TENSORVIZ_LOCAL_TOKEN"
#: Source identity is explicit because the extension keeps state outside the workspace.
WORKSPACE_DIR_ENV = "TENSORVIZ_WORKSPACE_DIR"
#: Older launchers infer source identity from their workspace-local state DB.
WORKSPACE_DB_ENV = "USER_HISTORY_DB_PATH"
#: The state directory that variable sits in; stripped to recover the folder.
WORKSPACE_STATE_DIRNAME = ".tensorviz"
WORKSPACE_UID_PREFIX = "workspace:"

def auth_disabled() -> bool:
    return os.getenv(AUTH_DISABLED_ENV, "false").strip().lower() in {"1", "true", "yes", "on"}


def local_token() -> str:
    """The per-launch secret, or "" when this backend was not started by the IDE."""
    return os.getenv(LOCAL_TOKEN_ENV, "").strip()


def workspace_folder() -> Optional[Path]:
    """The folder the IDE launched this backend against, or None if unknowable.

    Current extensions pass source identity independently of their private state
    DB. Older launchers retain the database-path fallback.
    """
    source_root = os.getenv(WORKSPACE_DIR_ENV, "").strip()
    if source_root:
        return Path(source_root).expanduser().resolve()
    raw = os.getenv(WORKSPACE_DB_ENV, "").strip()
    if not raw:
        return None
    state_dir = Path(raw).expanduser().resolve().parent
    return state_dir.parent if state_dir.name == WORKSPACE_STATE_DIRNAME else state_dir


def workspace_uid(folder: Path) -> str:
    """``workspace:<sha256(folder)>`` — stable per folder, and not the folder itself."""
    digest = hashlib.sha256(str(folder).encode("utf-8")).hexdigest()
    return f"{WORKSPACE_UID_PREFIX}{digest}"


@dataclass(frozen=True)
class CurrentUser:
    uid: str
    email: Optional[str] = None


def _local_token_user(request: Request, token: str) -> CurrentUser:
    """The workspace principal for an IDE-launched backend. Raises 401/503."""
    supplied = request.headers.get(LOCAL_TOKEN_HEADER, "")
    # compare_digest, not ==: same reason as middleware.LocalTokenMiddleware.
    if not secrets.compare_digest(supplied, token):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            f"Missing or invalid {LOCAL_TOKEN_HEADER} header.",
        )
    folder = workspace_folder()
    if folder is None:
        # Loud, not a fallback uid: guessing here would attribute this
        # workspace's writes to whatever the guess collided with.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"{LOCAL_TOKEN_ENV} is set but {WORKSPACE_DB_ENV} is not, so the "
            "workspace this backend belongs to is unknown.",
        )
    return CurrentUser(uid=workspace_uid(folder))


def get_current_user(request: Request) -> CurrentUser:
    """Return the local caller. Raises 401/503 otherwise.

    Order matters: AUTH_DISABLED stays the outermost branch so the test suite
    and start.sh see exactly the principal they saw before. The extension
    currently sets BOTH variables (extension/src/config.ts), so per-workspace
    identity only reaches the product once the launcher stops forcing
    AUTH_DISABLED.
    """
    if auth_disabled():
        return CurrentUser(uid=DEV_UID, email="local@dev")

    token = local_token()
    if token:
        return _local_token_user(request, token)

    raise HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "Missing or invalid local token. This backend does not accept bearer tokens.",
        headers={"WWW-Authenticate": LOCAL_TOKEN_HEADER},
    )
