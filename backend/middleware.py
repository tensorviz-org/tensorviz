"""HTTP request/response logging, body-size and IDE-mode middleware."""

import ipaddress
import logging
import os
import secrets
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger("uvicorn.error")

#: Query parameters whose VALUE must never reach a log line. The launch token is the
#: only thing standing between a local process and full control of this backend, and
#: it travels as a query parameter on the canvas frame URL — so it landed in the log
#: in the clear, twice per open. The extension pipes every backend line into the
#: user-visible tensorViz output channel, which is the first thing a person copies
#: into a bug report or shows on a screen share, and VS Code persists that channel to
#: disk. The secret left the machine at exactly the moment someone asked for help.
REDACTED_QUERY_PARAMS = frozenset({"token", "credential", "access_token", "api_key"})

#: What stands in for a redacted value. Not the empty string: "token=" reads as a
#: request that carried no token, which is a different bug report.
REDACTION = "REDACTED"


def redact_query(query: str) -> str:
    """``token=abc&depth=2`` -> ``token=REDACTED&depth=2``.

    Structure is preserved on purpose — which parameters were sent, in what order, is
    the half of the query a log is read for. Only the secret values go.
    """
    if not query:
        return query
    parts = []
    for pair in query.split("&"):
        name, sep, _ = pair.partition("=")
        if sep and name.lower() in REDACTED_QUERY_PARAMS:
            parts.append(f"{name}={REDACTION}")
        else:
            parts.append(pair)
    return "&".join(parts)


class RedactAccessLogFilter(logging.Filter):
    """Take the secret out of uvicorn's OWN access line.

    Redacting in LoggingMiddleware is not enough: uvicorn logs
    ``127.0.0.1:54846 - "GET /?token=... HTTP/1.1" 404`` from its own logger, with the
    query in the path argument, on a code path this application never touches.

    A filter on the logger rather than a ``--no-access-log`` flag on the command line,
    for two reasons: the access log is worth keeping, and the backend is started four
    different ways (the extension, start.sh, make, a developer's own uvicorn) — a flag
    would protect only the one that remembered to pass it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 3:
            return True
        target = args[2]
        if not isinstance(target, str) or "?" not in target:
            return True
        path, _, query = target.partition("?")
        redacted = redact_query(query)
        if redacted != query:
            record.args = args[:2] + (f"{path}?{redacted}",) + args[3:]
        return True


def install_log_redaction() -> None:
    """Idempotent: safe to call at import and again at startup.

    Attached to the LOGGER, not a handler, because uvicorn reconfigures handlers from
    its own dictConfig after this module is imported — and dictConfig replaces a
    logger's handlers without clearing its filters.
    """
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(existing, RedactAccessLogFilter) for existing in access.filters):
        access.addFilter(RedactAccessLogFilter())


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started_at = time.perf_counter()
        logger.info(
            "http.request.start request_id=%s method=%s path=%s query=%s",
            request_id,
            request.method,
            request.url.path,
            redact_query(request.url.query) or "-",
        )
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            logger.exception(
                "http.request.error request_id=%s method=%s path=%s duration_ms=%s",
                request_id,
                request.method,
                request.url.path,
                duration_ms,
            )
            raise

        duration_ms = int((time.perf_counter() - started_at) * 1000)
        response.headers["x-request-id"] = request_id
        logger.info(
            "http.request.done request_id=%s method=%s path=%s status=%s duration_ms=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


# The largest real payload measured on this backend is the F5-TTS template at
# 141 KiB (290 root nodes); a GraphDocumentV2 wrapping an imported model is the
# next biggest thing this API carries. 32 MiB leaves that two orders of magnitude
# of headroom while still refusing a body no endpoint could mean.
MAX_BODY_BYTES = 32 * 1024 * 1024


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Refuse an oversized body before anything buffers or parses it.

    Content-Length only: a chunked request omits the header, and counting bytes
    off request.stream() here would consume the body FastAPI still has to read to
    validate it. This is a ceiling on declared size, not complete coverage.

    Registered inside CORSMiddleware (see server_v2) so the 413 still carries the
    CORS headers the browser needs to read it, and inside LoggingMiddleware so the
    rejection is logged like any other response rather than vanishing.
    """

    async def dispatch(self, request: Request, call_next):
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                size = int(declared)
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
            if size > MAX_BODY_BYTES:
                logger.warning(
                    "http.request.body_too_large method=%s path=%s bytes=%s limit=%s",
                    request.method,
                    request.url.path,
                    size,
                    MAX_BODY_BYTES,
                )
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body too large (limit {MAX_BODY_BYTES} bytes)"},
                )
        return await call_next(request)


# ---------------------------------------------------------------------------
# IDE mode (task.md M2 / D12)
# ---------------------------------------------------------------------------
#
# Both middlewares below are registered by server_v2 only when
# TENSORVIZ_LOCAL_TOKEN is set, which is exactly when the VS Code extension
# started this backend. With the variable unset they are never constructed, so
# start.sh, `make e2e` and the whole test suite see the app they saw before.

LOCAL_TOKEN_HEADER = "x-tensorviz-token"


def local_token_matches(headers, token: str) -> bool:
    """Constant-time check of the per-launch header. Empty token never matches."""
    if not token:
        return False
    supplied = headers.get(LOCAL_TOKEN_HEADER, "")
    return secrets.compare_digest(supplied, token)


def _host_name(header: str) -> str:
    """Hostname from a Host header, without the port."""
    raw = header.strip().lower()
    if raw.startswith("["):
        end = raw.find("]")
        return raw[1:end] if end != -1 else raw
    if raw.count(":") == 1:
        return raw.rsplit(":", 1)[0]
    return raw


def host_header_allowed(header: str, *, token_configured: bool) -> bool:
    """Reject DNS-rebinding names. Loopback is always allowed.

    A numeric address is allowed only when a launch token is configured, which
    is what makes an explicit non-loopback bind usable from another machine
    without accepting ``Host: attacker.example``.
    """
    name = _host_name(header)
    if not name:
        return False
    if os.environ.get("PYTEST_CURRENT_TEST") and name == "testserver":
        return True
    if name == "localhost":
        return True
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    return token_configured


class LoopbackHostMiddleware:
    """Refuse requests whose Host header is not a loopback name.

    Binds to 127.0.0.1 still accept a browser request whose DNS name resolves
    to loopback. The Host header is the only thing that distinguishes that
    rebinding request from a local one.
    """

    def __init__(self, app, token_configured: bool = False):
        self.app = app
        self.token_configured = token_configured

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            raw = ""
            for key, value in scope.get("headers", []):
                if key == b"host":
                    raw = value.decode("latin-1")
                    break
            if not host_header_allowed(raw, token_configured=self.token_configured):
                if scope["type"] == "http":
                    response = JSONResponse(
                        status_code=400,
                        content={"detail": "Requests must use a loopback Host header (or the bind address when a launch token is set)."},
                    )
                    await response(scope, receive, send)
                    return
                await send({"type": "websocket.close", "code": 1008})
                return
        await self.app(scope, receive, send)

# The two routes that import and instantiate user or library code inside this
# process (api/routers/ingest.py). In the IDE that work belongs to the trace
# worker in the *user's* interpreter, so the route is refused rather than left
# as a way to run arbitrary code in the backend the editor spawned.
_IN_PROCESS_INGEST_PATHS = frozenset({"/api/ingest/model", "/api/ingest/repo"})


class LocalTokenMiddleware(BaseHTTPMiddleware):
    """Require a per-launch shared secret on /api/* when the IDE set one.

    The IDE's backend runs with AUTH_DISABLED on a loopback port, so anything
    else on the machine that can reach that port can drive it. The extension
    generates one random token per launch, passes it in the environment, and
    hands it to the webview; every /api/* request must carry it.

    Deliberately NOT covering /health (the extension's readiness probe runs
    before the app exists) or the static build at / (the browser cannot put a
    header on a document load; the token reaches the app in the iframe URL).
    Registered inside CORSMiddleware so a preflight is answered above this and
    a rejection still carries the CORS headers.
    """

    def __init__(self, app, token: str):
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/"):
            supplied = request.headers.get(LOCAL_TOKEN_HEADER, "")
            # compare_digest, not ==: the token is a secret and the comparison
            # runs on every request.
            if not secrets.compare_digest(supplied, self._token):
                logger.warning(
                    "http.request.local_token_rejected method=%s path=%s",
                    request.method,
                    request.url.path,
                )
                return JSONResponse(
                    status_code=401,
                    content={
                        "detail": {
                            "code": "local_token_required",
                            "message": (
                                "This backend was started by the tensorViz IDE and requires "
                                f"the per-launch {LOCAL_TOKEN_HEADER} header."
                            ),
                        }
                    },
                )
        return await call_next(request)


class InProcessIngestDisabledMiddleware(BaseHTTPMiddleware):
    """Refuse the in-process importer while the IDE is the caller.

    Kept out of api/routers/ingest.py on purpose: the router is shared with the
    hosted web product, where importing a library architecture in-process is the
    feature. What changes in the IDE is the trust boundary, and a boundary is a
    property of the process, not of the handler.
    """

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path in _IN_PROCESS_INGEST_PATHS:
            return JSONResponse(
                status_code=403,
                content={
                    "detail": {
                        "code": "ide_mode_uses_worker",
                        "message": (
                            "In IDE mode models are traced by the worker subprocess in your own "
                            "interpreter, so this backend does not import model code."
                        ),
                    }
                },
            )
        return await call_next(request)
