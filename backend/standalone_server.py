"""Standalone local service: static graph preparation and authenticated browser relay.

Started by the stdio host, with an inherited launch secret and a loopback socket
on an OS-assigned port, or on the port the host asks for (``TENSORVIZ_BROWSER_PORT``)
when it is free. This entry never loads hosted authentication or a project store.
Model execution remains in the separately authorized trace worker.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import threading

# The launcher uses -I; neither the user's project nor cwd belongs on sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from api.routers import local_browser
from middleware import LocalTokenMiddleware
from services.ide import multifile, review
from services.ide.graph_setup import discover
from services.ide.graph_setup_profiles import prepare_profile, restore_config_profile
from services.ide.local_browser_broker import LOOPBACK_BROWSER_HOSTS
from services.ide.local_browser_protocol import LOCAL_BROWSER_CAPTURE_EXPORT_MAX_BYTES
from services.ide.local_browser_wire import MAX_WIRE_BYTES
from services.ide.workspace_projection import ProjectionError, project_workspace

#: The same bounds as the editor backend's /api/ide/capture/portable.
PORTABLE_CAPTURE_INPUT_MAX_BYTES = 16 * 1024 * 1024
PORTABLE_CAPTURE_MAX_BYTES = LOCAL_BROWSER_CAPTURE_EXPORT_MAX_BYTES


class LimitedBody:
    """Bound actual streamed bytes, including requests without Content-Length."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = 24 * 1024 * 1024 if scope["path"].startswith("/api/") else 4096
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            if len(body) + len(event.get("body", b"")) > limit:
                return await JSONResponse({"detail": "Request body exceeds the limit."}, 413)(scope, receive, send)
            body.extend(event.get("body", b""))
            if not event.get("more_body"):
                break
        delivered = False
        async def bounded_receive():
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, bounded_receive, send)


class SetupDiscovery(multifile.MultifileProjectionRequest):
    profile: dict | None = None


class SetupProfile(BaseModel):
    sources: multifile.MultifileProjectionRequest
    config: str = Field(max_length=262144)
    profile: dict
    profile_name: str | None = Field(default=None, max_length=64)
    config_object: dict | None = None
    config_defaults: bool = False


def create_app(token: str, assets: Path | None = None) -> FastAPI:
    if len(token) < 32:
        raise ValueError("A private launch token is required.")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    # The page may arrive through an SSH or editor port forward on another local
    # port, often as localhost; the broker binds each view to that exact origin.
    app.state.local_browser_broker = local_browser.LocalBrowserBroker(source_proposals=False, forwarded_loopback=True)
    app.add_middleware(LimitedBody)
    app.add_middleware(LocalTokenMiddleware, token=token)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=sorted(LOOPBACK_BROWSER_HOSTS))
    app.include_router(local_browser.router)

    @app.exception_handler(RequestValidationError)
    async def invalid_body(_request, _error):
        return JSONResponse({"detail": "Invalid local-service request."}, 400)

    @app.post("/api/ide/multifile/project")
    def project(body: multifile.MultifileProjectionRequest):
        try:
            return project_workspace(body)
        except ProjectionError as error:
            raise HTTPException(error.status_code, error.detail) from None

    @app.post("/api/ide/review/index")
    def index(body: review.ReviewIndexRequest):
        try:
            return review.index_review(body)
        except multifile.MultifileError as error:
            raise HTTPException(400, error.detail) from None

    @app.post("/api/ide/graph-setup/discover")
    def setup(body: SetupDiscovery):
        try:
            result = discover(body)
            if body.profile is not None:
                result["restoredDraft"] = restore_config_profile(body, body.profile, result)
            return result
        except (ValueError, TypeError, RecursionError):
            raise HTTPException(400, "Model setup could not be read from this source snapshot.") from None

    @app.post("/api/ide/graph-setup/profile")
    def profile(body: SetupProfile):
        try:
            return prepare_profile(body.sources, body.config, body.profile, body.profile_name,
                                   body.config_object, config_defaults=body.config_defaults)
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
            raise HTTPException(400, "Setup could not be prepared. Check the fields and tensorviz.toml. Nothing was written.") from None

    @app.post("/api/ide/capture/portable")
    async def portable(request: Request):
        """Detach retained evidence for a graph file without opening files or executing
        user code. Errors never echo the submitted capture's private literals."""
        from services.semantic.portable import portable_capture
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > PORTABLE_CAPTURE_INPUT_MAX_BYTES:
                raise HTTPException(413, "Retained capture exceeds the portable preparation input limit. Nothing was truncated.")
            body.extend(chunk)
        try:
            capture = portable_capture(json.loads(body)).model_dump_json()
        except (ValueError, TypeError, RecursionError):
            raise HTTPException(400, "Lossless export refused: capture data is invalid or contains a private/path-like literal.") from None
        if len(capture.encode("utf-8")) > PORTABLE_CAPTURE_MAX_BYTES:
            raise HTTPException(413, "Retained capture exceeds the 16 MiB live-export limit. Nothing was truncated.")
        return Response(capture, media_type="application/json", headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    def health():
        return {"host": "standalone", "protocolVersion": 1}

    if assets is not None:
        app.mount("/", StaticFiles(directory=assets, html=True), name="browser")
    return app


def listen(port: int = 0) -> socket.socket:
    """A loopback listener on ``port``, else on an OS-assigned port.

    A fixed port lets an SSH ``LocalForward`` be set up once. When it is taken
    (another session holds it) or not permitted, the graph must still open, so
    the service falls back and the host tells the user which port it got.
    """
    for candidate in ([port] if port else []) + [0]:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if candidate and os.name != "nt":
            # Rebind at once after a restart (TIME_WAIT) but never beside a live
            # listener. Windows' SO_REUSEADDR would also allow the latter.
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", candidate))
            listener.listen(128)
        except OSError:
            listener.close()
            if candidate:
                continue
            raise
        return listener
    raise OSError("No loopback port is available.")


def main():
    import uvicorn
    token = os.environ.get("TENSORVIZ_LOCAL_TOKEN", "")
    arguments = sys.argv[1:]
    if arguments[:1] == ["--parent-pipe"]:
        arguments = arguments[1:]
        # EOF also follows a killed/crashed launcher; no orphan service keeps its
        # browser credentials live. The pipe carries no commands or user data.
        def watch_parent():
            while sys.stdin.buffer.read(1):
                pass
            os._exit(0)
        threading.Thread(target=watch_parent, daemon=True).start()
    port = 0
    if arguments[:1] == ["--port"] and len(arguments) > 1:
        # The host validates TENSORVIZ_BROWSER_PORT; anything else is ignored.
        port = int(arguments[1]) if arguments[1].isdecimal() and 0 < int(arguments[1]) < 65536 else 0
        arguments = arguments[2:]
    assets = Path(arguments[0]).resolve() if arguments else None
    app = create_app(token, assets)
    listener = listen(port)
    # Private parent-child pipe, never the MCP connection or a log.
    print(json.dumps({"port": listener.getsockname()[1]}), flush=True)
    # The broker's own frame bound, which Publish's graph file needs.
    uvicorn.Server(uvicorn.Config(app, log_level="critical", access_log=False,
                                 ws_max_size=MAX_WIRE_BYTES)).run(sockets=[listener])


if __name__ == "__main__":
    main()
