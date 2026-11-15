"""The local-browser surface: session mint, browser bootstrap, broker socket.

This is the HTTP half of the extension-webpage step 2; the relay itself lives in
services/ide/local_browser_broker.py and the message contract in
docs/extension-local-webpage-protocol.md.

Three endpoints, three different callers:

* ``POST /local/sessions`` — the extension mints a workspace session. This is
  the only route guarded by the per-launch ``TENSORVIZ_LOCAL_TOKEN`` secret
  (checked at request time, like auth.py, so the suite can toggle it per test).
  Without that variable the route answers 404: no sessions can exist, which
  makes bootstrap and the socket unreachable in practice — the same inert-by-
  default property middleware.py's IDE mode keeps.
* ``POST /local/bootstrap`` — the browser page, once, trading the URL-fragment
  credential for the HttpOnly session cookie. Origin-checked against the launch
  origin (or, on the plugin's standalone service, a same-origin forward of it to
  another loopback port) so an unrelated local page can never pass.
* ``WS /local/ws`` — the broker socket, authenticating on that cookie alone.

The router is deliberately NOT in the ``Depends(get_current_user)`` group: the
browser cannot hold the launch token and the broker answers with its own
session-scoped credentials instead of a uid. The per-launch token check on the
mint route is enforced here directly, at request time, rather than by
LocalTokenMiddleware, which only covers ``/api/*``.
"""

from __future__ import annotations

import os
import secrets

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from middleware import LOCAL_TOKEN_HEADER
from services.ide.local_browser_broker import (
    view_cookie_name,
    LOCAL_BROWSER_DEFAULT_TTL_SECONDS,
    LOCAL_BROWSER_MAX_TTL_SECONDS,
    LOCAL_BROWSER_MIN_TTL_SECONDS,
    BootstrapRefusal,
    LocalBrowserBroker,
)

router = APIRouter(prefix="/local", tags=["local-browser"])

broker = LocalBrowserBroker()


def _broker(connection) -> LocalBrowserBroker:
    return getattr(connection.app.state, "local_browser_broker", broker)


class CreateLocalBrowserSessionRequest(BaseModel):
    """What the extension knows when it opens a page: where the browser will
    come from, and how long the session may live."""

    launchOrigin: str = Field(min_length=1, max_length=256)
    ttlSeconds: float = Field(
        default=float(LOCAL_BROWSER_DEFAULT_TTL_SECONDS),
        ge=LOCAL_BROWSER_MIN_TTL_SECONDS,
        le=LOCAL_BROWSER_MAX_TTL_SECONDS,
    )


class BootstrapRequest(BaseModel):
    """The fragment credential, in the body — never the URL, so it can never
    reach an access log, a referrer header, or browser history."""

    workspaceSessionId: str = Field(min_length=1, max_length=256)
    credential: str = Field(min_length=1, max_length=512)


def _refusal(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": {"code": code, "message": message}})


@router.post("/sessions")
async def create_local_browser_session(request: Request, body: CreateLocalBrowserSessionRequest):
    launch_token = os.getenv("TENSORVIZ_LOCAL_TOKEN", "").strip()
    if not launch_token:
        return _refusal(404, "not_found", "Not found.")
    supplied = request.headers.get(LOCAL_TOKEN_HEADER, "")
    # compare_digest, not ==: the token is a secret and this comparison runs on
    # the only route that mints workspace credentials.
    if not secrets.compare_digest(supplied, launch_token):
        return _refusal(
            401,
            "local_token_required",
            f"This backend was started by the tensorViz IDE and requires the per-launch {LOCAL_TOKEN_HEADER} header.",
        )
    try:
        return _broker(request).create_session(body.launchOrigin, body.ttlSeconds)
    except ValueError:
        return _refusal(400, "invalid_launch_origin", "launchOrigin must be an http(s) origin without path, query or fragment.")


@router.post("/sessions/{workspace_session_id}/view-links")
async def create_local_browser_view_link(workspace_session_id: str, request: Request):
    launch_token = os.getenv("TENSORVIZ_LOCAL_TOKEN", "").strip()
    if not launch_token:
        return _refusal(404, "not_found", "Not found.")
    if not secrets.compare_digest(request.headers.get(LOCAL_TOKEN_HEADER, ""), launch_token):
        return _refusal(401, "local_token_required", "The per-launch token is required.")
    try:
        return JSONResponse(_broker(request).create_view_link(workspace_session_id), headers={"Cache-Control": "no-store"})
    except BootstrapRefusal as refusal:
        return _refusal(refusal.status_code, refusal.code, refusal.message)


@router.post("/bootstrap")
def bootstrap_local_browser(request: Request, body: BootstrapRequest):
    try:
        session, cookie_key, handoff_id, view_origin = _broker(request).bootstrap_view(
            body.workspaceSessionId, body.credential, request.headers.get("origin"), request.headers.get("host")
        )
    except BootstrapRefusal as refusal:
        return _refusal(refusal.status_code, refusal.code, refusal.message)
    info = _broker(request).session_info(session, "view")
    if handoff_id is not None:
        info["handoffId"] = handoff_id
    response = JSONResponse(info, headers={"Cache-Control": "no-store"})
    response.set_cookie(
        view_cookie_name(view_origin),
        cookie_key,
        path="/local",
        httponly=True,
        samesite="strict",
        secure=view_origin.startswith("https://"),
    )
    return response


@router.websocket("/ws")
async def local_browser_socket(websocket: WebSocket):
    await _broker(websocket).handle_connection(websocket)
