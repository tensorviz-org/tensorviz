"""Standalone local service: static graph preparation and authenticated browser relay.

Started by the stdio host, with an inherited launch secret and an OS-assigned
loopback socket. This entry never loads hosted authentication or a project store.
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
from services.ide.local_browser_protocol import LOCAL_BROWSER_CAPTURE_EXPORT_MAX_BYTES
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
    app.state.local_browser_broker = local_browser.LocalBrowserBroker(source_proposals=False)
    app.add_middleware(LimitedBody)
    app.add_middleware(LocalTokenMiddleware, token=token)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1"])
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
    assets = Path(arguments[0]).resolve() if arguments else None
    app = create_app(token, assets)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    # Private parent-child pipe, never the MCP connection or a log.
    print(json.dumps({"port": listener.getsockname()[1]}), flush=True)
    uvicorn.Server(uvicorn.Config(app, log_level="critical", access_log=False,
                                 ws_max_size=8 * 1024 * 1024)).run(sockets=[listener])


if __name__ == "__main__":
    main()
