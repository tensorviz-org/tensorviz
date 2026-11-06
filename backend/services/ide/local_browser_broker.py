"""The authenticated local-browser message broker (extension-webpage step 2).

The product report (docs/extension-local-webpage-product-report.md §5) makes the
loopback backend the typed relay between the one VS Code extension host and the
browser views of a workspace session, and docs/extension-local-webpage-protocol.md
freezes the message contract. This module is the relay: it validates every
message against that contract once, enforces the one-host/many-views topology,
delivers host state in ``hostSequence`` order, and routes each view's
``snapshotRequest`` to the host and the host's ``snapshot`` back to the view
that requested it. Typed source proposals and validation use requestId to return
results only to their originating view. The broker never reads or writes model
source; native VS Code review remains the only way to apply a source proposal.

Credential model (docs/extension-local-webpage-protocol.md "Security Boundary"):

* Sessions are minted over ``POST /local/sessions`` by the extension, which
  proves itself with the per-launch ``TENSORVIZ_LOCAL_TOKEN`` secret that a
  browser page cannot know. The mint response carries the *view credential*
  for the URL fragment and an opaque *host cookie key*; the extension holds
  both in memory and never writes them to the workspace.
* The page posts the fragment credential once, same-origin, to
  ``POST /local/bootstrap`` and gets an opaque view key in an
  ``HttpOnly; SameSite=Strict; Path=/local`` session cookie. The cookie value
  is never the fragment credential: after bootstrap the fragment secret exists
  nowhere the browser can still reach. The host cookie key never enters a
  browser flow at all, so a stolen view credential cannot claim the host role.
* The WebSocket route authenticates only on that cookie plus an exact ``Origin``
  match against the launch origin: a browser socket can present nothing else.
* Sessions expire. The TTL timer closes every socket with CREDENTIAL_EXPIRED
  and revokes every credential, so a browser tab left open overnight does not
  keep a live line into the workspace after the extension's window is gone. The
  session itself stays as an inert expired tombstone for a bounded retention
  window so a late bootstrap can be answered CREDENTIAL_EXPIRED — actionable —
  instead of the same INVALID_CREDENTIALS an unknown session gets; the reaper
  then deletes it so sessions minted over the process's life cannot pile up.

Two refusal styles are deliberate. Transport-level failures (no cookie, wrong
cookie, expired session, wrong origin) refuse before ``accept()`` because no
authenticated channel exists yet to carry an error. Protocol-level failures
(wrong version, wrong session, malformed or misdirected messages) fail *after*
accept with the typed ``error`` message the contract defines; handshake failures
then close with the matching code, while post-handshake violations are refused
individually so one malformed frame cannot kill a healthy session.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import secrets
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit

import anyio
from starlette.websockets import WebSocket, WebSocketDisconnect

from services.ide.local_browser_protocol import (
    LOCAL_BROWSER_MESSAGE_TYPES,
    LOCAL_BROWSER_PROTOCOL_VERSION,
    LocalBrowserRole,
    is_local_browser_message,
)
from services.ide.local_browser_wire import MAX_WIRE_BYTES, decode_message, encode_message

#: The fixed host cookie; view cookies add an exact-origin suffix. The path
#: scope is what lets the two share it while the static page at "/" never sees it.
LOCAL_BROWSER_COOKIE = "tensorviz_local_browser"


def view_cookie_name(launch_origin: str) -> str:
    """Browser cookies ignore ports; keep each forwarded window independent."""
    origin_key = hashlib.sha256(launch_origin.encode("utf-8")).hexdigest()[:16]
    return f"{LOCAL_BROWSER_COOKIE}_{origin_key}"


#: Sessions are meant to die with the VS Code window; eight hours covers a work
#: day, and a day is the most an extension may ask for. The floor is one second:
#: a shorter TTL can only ever be a bug or an attack on the expiry property.
LOCAL_BROWSER_DEFAULT_TTL_SECONDS = 8 * 60 * 60
LOCAL_BROWSER_MIN_TTL_SECONDS = 1
LOCAL_BROWSER_MAX_TTL_SECONDS = 24 * 60 * 60

#: How long an expired session stays as a tombstone before the reaper deletes
#: it. A bootstrap lands within seconds of expiry in anything but a forgotten
#: tab; an hour covers that straggler without letting one mint per window add
#: up to permanent growth of the sessions map.
LOCAL_BROWSER_TOMBSTONE_RETENTION_SECONDS = 60 * 60

#: The wire bound is unchanged; large graphs use bounded gzip JSON envelopes.
LOCAL_BROWSER_MAX_MESSAGE_BYTES = MAX_WIRE_BYTES

#: Application close codes (4000-4999) paired 1:1 with the typed protocol error
#: code sent first; the reason phrase is the error code itself so a denied
#: upgrade still names the cause on clients that surface it.
LOCAL_BROWSER_CLOSE_CODES = {
    "INVALID_CREDENTIALS": 4401,
    "CREDENTIAL_EXPIRED": 4402,
    "INVALID_ORIGIN": 4403,
    "PROTOCOL_VERSION_MISMATCH": 4404,
    "SESSION_MISMATCH": 4405,
    "INVALID_MESSAGE": 4406,
    "UNKNOWN_MESSAGE": 4407,
    "HOST_ALREADY_CONNECTED": 4408,
}

#: The frozen read-only capability set: the browser page can look and navigate,
#: and nothing more. Fresh dicts are handed out by ``_session_info``.
_READ_ONLY_CAPABILITIES = {
    "readOnly": True,
    "canNavigateSource": True,
    "canWriteSource": False,
    "canRunAgent": False,
    "canEditGraph": False,
    "canProposeSourceEdits": True,
}

#: Types a client may never inject because the broker owns them: ``connected``
#: and ``connectionState`` describe links the broker alone can see, and a second
#: handshake is a protocol violation.
_BROKER_OWNED_TYPES = frozenset(
    {"connected", "connectionState", "viewState", "hostHandshake", "viewHandshake"}
)

#: Each role has an explicit direction table. Source and validation results
#: are routed through pending_requests to their originating view only.
_HOST_TO_ALL_VIEWS = frozenset(
    {"graphReviewState", "graphReviewFocus", "activeTarget", "projection", "projectIndex", "projectDocument", "syncState", "staleState", "focusNode", "validationState", "graphOpening", "ack", "error"}
)
_VIEW_TO_HOST = frozenset({"agentRequest", "codeFocusRequest", "graphReviewRequest", "graphReviewApplied", "graphSetupRequest", "projectFocus", "projectExpand", "projectDeclare", "viewSelection", "viewFocusResult", "snapshotRequest", "revealNode", "sourceEditRequest", "sourceEditCancel", "candidateRequest", "validateRequest", "validateCancel", "captureInspectRequest", "captureInspectCancel", "captureExportRequest", "captureExportCancel", "graphPublishRequest", "graphPublishCancel", "ack", "error"})

#: Terminal source-edit and validation results. Not ordered state, so these never carry
#: ``hostSequence`` and never enter ``_SEQUENCED_HOST_TYPES``.
_REQUEST_TERMINAL_TYPES = frozenset({"sourceEditResult", "validateResult", "captureInspectResult", "captureExportResult", "graphPublishResult"})

#: Progress the host reports for one view's request before its terminal result.
_REQUEST_PROGRESS_TYPES = frozenset({"sourceEditReview", "graphPublishState"})

#: Each request's cancellation, which is also what the broker sends when the asking view leaves.
_REQUEST_CANCELS = {"sourceEditRequest": "sourceEditCancel", "validateRequest": "validateCancel", "captureInspectRequest": "captureInspectCancel", "captureExportRequest": "captureExportCancel", "graphPublishRequest": "graphPublishCancel"}

#: Ordered host state: these carry ``hostSequence`` and must strictly increase.
_SEQUENCED_HOST_TYPES = frozenset(
    {"graphOpening", "activeTarget", "projection", "projectIndex", "projectDocument", "syncState", "staleState", "focusNode", "snapshot"}
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_launch_origin(value: str) -> str:
    """Canonicalize a launch origin to exactly what a browser puts in Origin.

    Scheme and host are case-folded and a lone trailing slash removed, because
    the comparison later is a byte-for-byte string equality against the header.
    Anything with credentials, a path, a query or a fragment is not an origin
    and is refused loudly at session creation rather than mismatching later.
    """

    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("launchOrigin must be an http(s) origin")
    if parsed.username or parsed.password:
        raise ValueError("launchOrigin must not carry credentials")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("launchOrigin must not carry a path, query or fragment")
    try:
        parsed.port  # raises ValueError on a malformed port
    except ValueError as exc:
        raise ValueError("launchOrigin has a malformed port") from exc
    return f"{parsed.scheme}://{parsed.netloc.lower()}"


class BootstrapRefusal(RuntimeError):
    """A bootstrap rejection. ``code`` is the frozen protocol error code so the
    page and the contract speak one vocabulary; the message is static text that
    never quotes the credential, the origin, or any other request content."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass
class _Connection:
    connection_id: int
    role: LocalBrowserRole | None
    websocket: WebSocket
    handoff_id: str | None = None


@dataclass
class _Session:
    workspace_session_id: str
    view_credential: str
    launch_origin: str
    expires_at: datetime
    host: _Connection | None = None
    views: dict[int, _Connection] = field(default_factory=dict)
    last_host_sequence: int | None = None
    #: The last roster level THIS host epoch was told. None means "not yet told",
    #: which is why a fresh host is levelled at handshake rather than left to infer
    #: presence from silence. Edge-triggered after that: a second view attaching is
    #: not news, and the host is told `detached` only when the LAST view leaves.
    last_view_state: str | None = None
    view_control: bool = False
    # (view connection id, snapshotRequest message) in the order asked. A view
    # only ever needs the newest snapshot, so a repeat request replaces the old.
    pending_snapshot_requests: deque[tuple[int, dict[str, Any]]] = field(default_factory=deque)
    # requestId -> (view connection id, original typed request). This
    # is how results find their owner, how a duplicate in-flight
    # requestId is refused, and how a view drop tells the host to abandon work
    # nobody is waiting for anymore.
    pending_requests: dict[str, tuple[int, dict[str, Any]]] = field(default_factory=dict)
    view_links: dict[str, tuple[str, datetime]] = field(default_factory=dict)
    handoffs: set[str] = field(default_factory=set)
    expiry_task: asyncio.Task | None = None


class LocalBrowserBroker:
    """One in-memory registry of workspace sessions and their live sockets."""

    def __init__(self, *, source_proposals: bool = True) -> None:
        self._capabilities = dict(_READ_ONLY_CAPABILITIES)
        if not source_proposals:
            self._capabilities.pop("canProposeSourceEdits", None)
        self._sessions: dict[str, _Session] = {}
        # Opaque cookie value -> (session id, role). Keeping the map here rather
        # than re-minting the cookie from the credential is what lets expiry
        # revoke every outstanding key in one sweep.
        self._cookie_keys: dict[str, tuple[str, str]] = {}
        self._connection_counter = 0
        self._message_counter = 0

    # -- session lifecycle ------------------------------------------------

    def create_session(self, launch_origin: str, ttl_seconds: float) -> dict[str, Any]:
        """Mint a session for the extension. Returns everything the extension
        needs to arm both roles; the view credential is the only value meant to
        leave the extension's memory (inside the URL fragment)."""

        origin = normalize_launch_origin(launch_origin)
        now = _utcnow()
        session = _Session(
            workspace_session_id=secrets.token_urlsafe(24),
            view_credential=secrets.token_urlsafe(32),
            launch_origin=origin,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        self._sessions[session.workspace_session_id] = session
        host_cookie_key = self._new_cookie_key(session, "host")
        session.expiry_task = asyncio.create_task(
            self._expire_after(session.workspace_session_id, ttl_seconds)
        )
        return {
            "workspaceSessionId": session.workspace_session_id,
            "launchOrigin": session.launch_origin,
            "expiresAt": _iso_utc(session.expires_at),
            "protocolVersion": LOCAL_BROWSER_PROTOCOL_VERSION,
            "capabilities": dict(self._capabilities),
            "viewCredential": session.view_credential,
            "hostCookieKey": host_cookie_key,
            "cookieName": LOCAL_BROWSER_COOKIE,
        }

    def create_view_link(self, workspace_session_id: str) -> dict[str, Any]:
        session = self._sessions.get(workspace_session_id)
        if session is None:
            raise BootstrapRefusal(401, "INVALID_CREDENTIALS", "Unknown session.")
        if self._is_expired(session):
            raise BootstrapRefusal(410, "CREDENTIAL_EXPIRED", "Reopen the browser session.")
        now = _utcnow()
        session.view_links = {key: value for key, value in session.view_links.items() if value[1] > now}
        if len(session.view_links) >= 16 or len(session.handoffs) >= 128:
            raise BootstrapRefusal(429, "BUSY", "Reopen the browser session before creating more views.")
        credential, handoff_id = secrets.token_urlsafe(32), secrets.token_urlsafe(24)
        expires = min(session.expires_at, now + timedelta(seconds=120))
        session.view_links[credential] = (handoff_id, expires)
        return {"workspaceSessionId": workspace_session_id, "launchOrigin": session.launch_origin,
                "handoffId": handoff_id, "viewCredential": credential, "expiresAt": _iso_utc(expires)}

    def bootstrap_view(
        self, workspace_session_id: str, credential: str, origin: str | None
    ) -> tuple[_Session, str, str | None]:
        """Trade the fragment credential for a view cookie key.

        The refusal order is origin before credential on purpose: the origin
        check is the cheap fence that keeps unrelated local pages from ever
        reaching the secret comparison.
        """

        session = self._sessions.get(workspace_session_id)
        if session is None:
            raise BootstrapRefusal(401, "INVALID_CREDENTIALS", "Unknown session credential.")
        if origin != session.launch_origin:
            raise BootstrapRefusal(403, "INVALID_ORIGIN", "The request origin is not the launch origin.")
        if self._is_expired(session):
            raise BootstrapRefusal(410, "CREDENTIAL_EXPIRED", "The session has expired. Reopen the graph from VS Code.")
        handoff_id = None
        if not secrets.compare_digest(credential, session.view_credential):
            grant = session.view_links.pop(credential, None)
            if grant is None:
                raise BootstrapRefusal(401, "INVALID_CREDENTIALS", "Unknown session credential.")
            handoff_id, expires = grant
            if expires <= _utcnow():
                raise BootstrapRefusal(410, "CREDENTIAL_EXPIRED", "Request a new browser view link.")
            session.handoffs.add(handoff_id)
        return session, self._new_cookie_key(session, "view"), handoff_id

    def session_info(self, session: _Session, role: str) -> dict[str, Any]:
        return {
            "workspaceSessionId": session.workspace_session_id,
            "role": role,
            "protocolVersion": LOCAL_BROWSER_PROTOCOL_VERSION,
            "capabilities": dict(self._capabilities),
            "expiresAt": _iso_utc(session.expires_at),
        }

    # -- websocket handling -----------------------------------------------

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Authenticate, register, relay, and clean up one socket."""

        session, role, connection = await self._authenticate_socket(websocket)
        if session is None:
            return
        registered = False
        try:
            registered = await self._handshake(session, role, connection)
            if registered:
                await self._relay_loop(session, connection)
        except WebSocketDisconnect:
            pass
        finally:
            if registered:
                # Closing a view must deliver both its MCP disconnect and the
                # final roster level, even when the socket task is cancelled.
                with anyio.CancelScope(shield=True):
                    await self._drop_connection(session, connection)

    async def _authenticate_socket(
        self, websocket: WebSocket
    ) -> tuple[_Session | None, str, _Connection]:
        """Refuse transport-level failures before accept: there is no
        authenticated channel yet, so no protocol error can be delivered."""

        origin = websocket.headers.get("origin", "")
        # Host transports send the fixed cookie explicitly. Browser views use a
        # name scoped to their exact launch origin, including the forwarded port.
        cookie_key = websocket.cookies.get(view_cookie_name(origin), "") or websocket.cookies.get(LOCAL_BROWSER_COOKIE, "")
        key_entry = self._cookie_keys.get(cookie_key) if cookie_key else None
        if key_entry is None:
            await self._deny(websocket, "INVALID_CREDENTIALS")
            return None, "", None
        session_id, role = key_entry
        session = self._sessions.get(session_id)
        if session is None:
            await self._deny(websocket, "INVALID_CREDENTIALS")
            return None, "", None
        if self._is_expired(session):
            await self._deny(websocket, "CREDENTIAL_EXPIRED")
            return None, "", None
        if websocket.headers.get("origin") != session.launch_origin:
            await self._deny(websocket, "INVALID_ORIGIN")
            return None, "", None
        self._connection_counter += 1
        connection = _Connection(self._connection_counter, None, websocket)
        await websocket.accept()
        return session, role, connection

    @staticmethod
    async def _receive_client_event(connection: _Connection) -> dict[str, Any]:
        """One raw ASGI event from an accepted socket.

        ``WebSocket.receive()`` reports a clean close in-band as a
        ``websocket.disconnect`` event; only ``receive_text``/``receive_json``
        translate that into ``WebSocketDisconnect``. This broker needs the raw
        event for its size cap and JSON staging, so it does the translation
        itself: handing the event to ``_validate_incoming`` instead would
        mis-label a close as a malformed message, then loop back onto a
        disconnected socket, which raises outside the disconnect handler.
        """

        data = await connection.websocket.receive()
        if data.get("type") == "websocket.disconnect":
            raise WebSocketDisconnect(code=data.get("code", 1000), reason=data.get("reason"))
        return data

    async def _handshake(
        self, session: _Session, role: str, connection: _Connection
    ) -> bool:
        """Validate the first message, register the peer, and emit link state.

        Only after this point do refusals arrive as typed contract errors."""

        try:
            data = await self._receive_client_event(connection)
        except WebSocketDisconnect:
            return False
        message, code = self._validate_incoming(session, data)
        if code is None and message["type"] not in {"hostHandshake", "viewHandshake"}:
            code, message = "INVALID_MESSAGE", None
        if code is None and message["role"] != role:
            # The cookie key proves a role; a handshake claiming the other one
            # is presenting a credential for something it is not.
            code, message = "INVALID_CREDENTIALS", None
        if code is not None:
            await self._refuse(connection, session, code)
            return False

        handoff_id = message.get("handoffId") if role == "view" else None
        if handoff_id is not None:
            if handoff_id not in session.handoffs or any(view.handoff_id == handoff_id for view in session.views.values()):
                await self._refuse(connection, session, "INVALID_MESSAGE")
                return False
            connection.handoff_id = handoff_id
        connection.role = role  # type: ignore[assignment]
        info = self.session_info(session, role)
        if role == "view":
            info["viewId"] = str(connection.connection_id)
            if handoff_id is not None:
                info["handoffId"] = handoff_id
        await self._send_broker_message(
            connection, session, "connected", role=role, session=info
        )
        if role == "host":
            if session.host is not None:
                await self._refuse(connection, session, "HOST_ALREADY_CONNECTED")
                return False
            session.host = connection
            session.view_control = message.get("viewControl") is True
            session.last_host_sequence = None  # a new host epoch restarts ordering
            # Level the roster before anything else: a new host knows nothing about
            # which tabs are open, and inferring presence from silence is how the
            # host came to publish into an empty roster in the first place.
            session.last_view_state = None
            await self._notify_view_state(session)
            await self._notify_connection_state(session, "connected")
            await self._flush_snapshot_requests(session)
            for view in list(session.views.values()):
                await self._notify_view(session, view, True)
        else:
            session.views[connection.connection_id] = connection
            await self._notify_view(session, connection, True)
            await self._send_broker_message(
                connection,
                session,
                "connectionState",
                state="connected" if session.host is not None else "disconnected",
            )
            await self._notify_view_state(session)
        return True

    async def _notify_view(self, session: _Session, view: _Connection, connected: bool) -> None:
        host = session.host
        if host is not None and session.view_control:
            message = self._new_broker_message(session, "viewState", viewId=str(view.connection_id), connected=connected,
                                               **({"handoffId": view.handoff_id} if view.handoff_id else {}))
            if not await self._send(host, message):
                await self._drop_connection(session, host)

    async def _relay_loop(self, session: _Session, connection: _Connection) -> None:
        while True:
            data = await self._receive_client_event(connection)
            # Registration may have been taken away while this loop waited: a
            # failed send drops the peer (views were told "disconnected"), and
            # expiry clears the roster. Relaying anyway would let a ghost host
            # keep feeding state after the broker declared it gone, so exit —
            # handle_connection's finally runs the idempotent drop again.
            if connection.role == "host" and session.host is not connection:
                return
            if connection.role == "view" and connection.connection_id not in session.views:
                return
            message, code = self._validate_incoming(session, data)
            if code is None and message["type"] in _BROKER_OWNED_TYPES:
                code, message = "INVALID_MESSAGE", None
            if code is not None:
                # One bad frame must not kill a healthy session: refuse the
                # frame, keep the peer.
                await self._send_error(connection, session, code)
                continue
            if connection.role == "host":
                await self._relay_from_host(session, connection, message)
            else:
                await self._relay_from_view(session, connection, message)

    # -- relay -------------------------------------------------------------

    async def _relay_from_host(
        self, session: _Session, connection: _Connection, message: dict[str, Any]
    ) -> None:
        message_type = message["type"]
        if message_type in {"candidateResult", "codeFocusState", "agentState"}:
            view = next((item for item in session.views.values() if str(item.connection_id) == message["viewId"]), None)
            if view is not None and not await self._send(view, message):
                await self._drop_connection(session, view)
            return
        if message_type == "focusView":
            if not session.view_control:
                await self._send_error(connection, session, "INVALID_MESSAGE")
                return
            view = next((item for item in session.views.values() if str(item.connection_id) == message["viewId"]), None)
            if view is not None and await self._send(view, message):
                return
            reply = self._new_broker_message(session, "viewFocusResult", viewId=message["viewId"],
                target=message["target"], requestId=message["requestId"], ok=False, detail="The requested canvas view is not connected.")
            await self._send(connection, reply)
            if view is not None:
                await self._drop_connection(session, view)
            return
        if message_type in _REQUEST_TERMINAL_TYPES or message_type in _REQUEST_PROGRESS_TYPES:
            await self._deliver_reply(session, message)
            return
        if message_type == "snapshot":
            pass  # routed below, not broadcast
        elif message_type not in _HOST_TO_ALL_VIEWS:
            await self._send_error(connection, session, "INVALID_MESSAGE")
            return
        if message_type in _SEQUENCED_HOST_TYPES and not await self._accept_host_sequence(
            session, connection, message
        ):
            return
        if message_type == "snapshot":
            await self._deliver_snapshot(session, connection, message)
            return
        await self._broadcast_to_views(session, message)

    async def _relay_from_view(
        self, session: _Session, connection: _Connection, message: dict[str, Any]
    ) -> None:
        message_type = message["type"]
        if message_type not in _VIEW_TO_HOST:
            await self._send_error(connection, session, "INVALID_MESSAGE")
            return
        if message_type in {"agentRequest", "codeFocusRequest", "graphReviewApplied", "viewSelection", "viewFocusResult", "candidateRequest"} or (message_type == "sourceEditRequest" and "viewId" in message):
            if not session.view_control or message["viewId"] != str(connection.connection_id):
                await self._send_error(connection, session, "INVALID_MESSAGE")
                return
        if message_type in _REQUEST_CANCELS:
            await self._route_request(session, connection, message)
            return
        if message_type in _REQUEST_CANCELS.values():
            await self._route_cancel(session, connection, message)
            return
        if message_type == "snapshotRequest":
            self._queue_snapshot_request(session, connection, message)
        host = session.host
        if host is None:
            if message_type != "snapshotRequest":
                await self._send_error(connection, session, "INVALID_MESSAGE")
            return
        if not await self._send(host, message):
            # The socket died without a close frame; treat it as a disconnect so
            # views learn the truth now rather than on the next read failure.
            await self._drop_connection(session, host)

    async def _accept_host_sequence(
        self, session: _Session, connection: _Connection, message: dict[str, Any]
    ) -> bool:
        """Strictly increasing per host epoch. A repeat or rewind is refused and
        dropped: relaying it would let a view see the same sequence twice, which
        is exactly what the ordering guarantee exists to prevent."""

        sequence = message["hostSequence"]
        if session.last_host_sequence is not None and sequence <= session.last_host_sequence:
            await self._send_error(connection, session, "INVALID_MESSAGE")
            return False
        session.last_host_sequence = sequence
        return True

    def _queue_snapshot_request(
        self, session: _Session, connection: _Connection, message: dict[str, Any]
    ) -> None:
        # A view that asks twice only needs the newest answer.
        kept = [entry for entry in session.pending_snapshot_requests if entry[0] != connection.connection_id]
        kept.append((connection.connection_id, message))
        session.pending_snapshot_requests = deque(kept)

    async def _deliver_snapshot(
        self, session: _Session, connection: _Connection, message: dict[str, Any]
    ) -> None:
        while session.pending_snapshot_requests:
            view_id, _request = session.pending_snapshot_requests.popleft()
            view = session.views.get(view_id)
            if view is None:
                continue  # the asker left; try the next oldest request
            await self._send(view, message)
            return
        await self._send_error(connection, session, "INVALID_MESSAGE")

    async def _flush_snapshot_requests(self, session: _Session) -> None:
        """A (re)connected host hears every unanswered request, oldest first."""
        host = session.host
        if host is None:
            return
        for view_id, request in list(session.pending_snapshot_requests):
            if view_id in session.views and not await self._send(host, request):
                await self._drop_connection(session, host)
                return

    # -- request routing -------------------------------------------------------

    async def _route_request(
        self, session: _Session, connection: _Connection, message: dict[str, Any]
    ) -> None:
        """Forward one view's request to the host, remembering who asked.

        Unlike a snapshotRequest this is interactive, so it refuses outright
        while no host is attached (the page enables requests only while connected),
        and a requestId already in flight is a protocol violation — from one
        view a duplicate is a resend bug; across views it is a confusion
        neither can distinguish after routing.
        """

        request_id = message["requestId"]
        host = session.host
        if host is None or request_id in session.pending_requests or len(session.pending_requests) >= 32:
            await self._send_error(connection, session, "INVALID_MESSAGE")
            return
        # Remember who asked, not what they sent: a publish request carries the graph file.
        session.pending_requests[request_id] = (connection.connection_id, {key: value for key, value in message.items() if key != "file"})
        if not await self._send(host, message):
            session.pending_requests.pop(request_id, None)
            await self._drop_connection(session, host)

    async def _route_cancel(
        self, session: _Session, connection: _Connection, message: dict[str, Any]
    ) -> None:
        """Forward the asking view's own cancellation. Stale, foreign, and
        already-answered cancels are no-ops, never refusals: the answer may
        simply have raced the cancel, and another view's request is not this
        view's to abandon."""

        request_id = message["requestId"]
        pending = session.pending_requests.get(request_id)
        host = session.host
        if pending is None or pending[0] != connection.connection_id or host is None:
            return
        expected = _REQUEST_CANCELS[pending[1]["type"]]
        if message["type"] != expected or message["target"] != pending[1]["target"]:
            return
        session.pending_requests.pop(request_id, None)
        if not await self._send(host, message):
            await self._drop_connection(session, host)

    async def _deliver_reply(self, session: _Session, message: dict[str, Any]) -> None:
        """Route one source-edit or validation result to the asking view only. A reply naming
        no pending request is not an error: the view cancelled or disconnected,
        the host simply heard too late, and the answer has no destination left."""

        pending = session.pending_requests.get(message["requestId"])
        if pending is None:
            return
        expected = {
            "sourceEditRequest": {"sourceEditResult", "sourceEditReview"},
            "validateRequest": {"validateResult"},
            "captureInspectRequest": {"captureInspectResult"},
            "captureExportRequest": {"captureExportResult"},
            "graphPublishRequest": {"graphPublishState", "graphPublishResult"},
        }[pending[1]["type"]]
        if message["type"] not in expected or message["target"] != pending[1]["target"]:
            return
        if message["type"] in _REQUEST_TERMINAL_TYPES:
            session.pending_requests.pop(message["requestId"], None)
        view_id, _request = pending
        view = session.views.get(view_id)
        if view is None:
            return  # the drop already told the host to abandon this request
        if not await self._send(view, message):
            await self._drop_connection(session, view)

    # -- teardown ----------------------------------------------------------

    async def _drop_connection(self, session: _Session, connection: _Connection) -> None:
        if connection.role == "host" and session.host is connection:
            session.host = None
            # The level belongs to a host epoch, not to the session: the next host
            # is told afresh at its handshake.
            session.last_view_state = None
            session.view_control = False
            # A new host epoch answers nothing the old one was asked: the views
            # resolve their pending requests locally on the disconnect notification.
            session.pending_requests.clear()
            await self._notify_connection_state(session, "disconnected")
        elif connection.role == "view":
            existed = session.views.pop(connection.connection_id, None)
            if existed is not None:
                await self._notify_view(session, connection, False)
            kept = [
                entry
                for entry in session.pending_snapshot_requests
                if entry[0] != connection.connection_id
            ]
            session.pending_snapshot_requests = deque(kept)
            # Cancel this view's pending reviews and validation when it leaves.
            abandons = [
                (request_id, request)
                for request_id, (view_id, request) in session.pending_requests.items()
                if view_id == connection.connection_id
            ]
            for request_id, _request in abandons:
                session.pending_requests.pop(request_id, None)
            host = session.host
            if host is not None:
                for request_id, request in abandons:
                    cancel = self._new_broker_message(
                        session, _REQUEST_CANCELS[request["type"]],
                        target=request["target"], requestId=request_id
                    )
                    if not await self._send(host, cancel):
                        await self._drop_connection(session, host)
                        break
            # Last, after this view's own requests are settled: the host cannot see
            # this socket die — closing a browser tab leaves the host's socket
            # connected — so without this it kept believing a page was there and
            # published retargets into an empty roster. Stated after the cancels so a
            # request-level answer is never preceded by link-level news.
            await self._notify_view_state(session)

    async def _expire_after(self, session_id: str, ttl_seconds: float) -> None:
        try:
            await asyncio.sleep(max(ttl_seconds, 0))
        except asyncio.CancelledError:
            return
        await self._expire_session(session_id)

    async def _expire_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        for key, (key_session_id, _role) in list(self._cookie_keys.items()):
            if key_session_id == session_id:
                self._cookie_keys.pop(key, None)
        close_code = LOCAL_BROWSER_CLOSE_CODES["CREDENTIAL_EXPIRED"]
        live = [session.host] if session.host is not None else []
        live.extend(session.views.values())
        session.host = None
        session.views.clear()
        session.view_links.clear()
        session.handoffs.clear()
        session.pending_snapshot_requests.clear()
        session.pending_requests.clear()
        for connection in live:
            with contextlib.suppress(Exception):
                await connection.websocket.close(code=close_code, reason="CREDENTIAL_EXPIRED")
        session.expiry_task = asyncio.create_task(self._reap_tombstone(session_id))

    async def _reap_tombstone(self, session_id: str) -> None:
        """Delete the expired tombstone once its retention window closes.

        The tombstone exists so a straggling bootstrap hears CREDENTIAL_EXPIRED
        instead of INVALID_CREDENTIALS; without the reaper every session ever
        minted would sit in ``_sessions`` for the life of the process. Read the
        retention at call time so tests can shrink it.
        """

        try:
            await asyncio.sleep(LOCAL_BROWSER_TOMBSTONE_RETENTION_SECONDS)
        except asyncio.CancelledError:
            return
        self._sessions.pop(session_id, None)

    # -- message plumbing --------------------------------------------------

    def _validate_incoming(
        self, session: _Session, data: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Stage a raw socket event into (message, None) or (None, error code).

        Order matters: parse, then name the type, then version, then session,
        then the frozen validator. That order is what distinguishes a genuinely
        unknown message from a well-formed message from another session.
        """

        text = data.get("text")
        if text is None:
            return None, "INVALID_MESSAGE"  # binary frames are not the protocol
        if len(text.encode("utf-8")) > LOCAL_BROWSER_MAX_MESSAGE_BYTES:
            return None, "INVALID_MESSAGE"
        try:
            message = decode_message(text)
        except (ValueError, UnicodeDecodeError):
            return None, "INVALID_MESSAGE"
        if not isinstance(message, dict):
            return None, "INVALID_MESSAGE"
        message_type = message.get("type")
        if not isinstance(message_type, str):
            return None, "INVALID_MESSAGE"
        if message_type not in LOCAL_BROWSER_MESSAGE_TYPES:
            return None, "UNKNOWN_MESSAGE"
        version = message.get("protocolVersion")
        if isinstance(version, bool) or version != LOCAL_BROWSER_PROTOCOL_VERSION:
            return None, "PROTOCOL_VERSION_MISMATCH"
        if message.get("workspaceSessionId") != session.workspace_session_id:
            return None, "SESSION_MISMATCH"
        if not is_local_browser_message(message, session.workspace_session_id):
            return None, "INVALID_MESSAGE"
        if self._is_expired(session):
            return None, "CREDENTIAL_EXPIRED"
        return message, None

    async def _broadcast_to_views(self, session: _Session, message: dict[str, Any]) -> None:
        for view in list(session.views.values()):
            if not await self._send(view, message):
                await self._drop_connection(session, view)

    async def _notify_view_state(self, session: _Session) -> None:
        """Tell the host when the LAST view leaves. Track the level either way.

        Only `detached` is ever sent. The host treats anything it has not been told
        as "a page may be there" and retargets, which is the right default and the
        behaviour every other build has — so an `attached` edge would be a frame
        carrying nothing the host acts on, at the cost of a frame between every
        `connect_view` and the next host read in three test suites.

        The level is still tracked on attach, because that is what makes the NEXT
        close an edge: without it the roster would go empty, be told once, and never
        be told again.
        """
        state = "attached" if session.views else "detached"
        if session.last_view_state == state:
            return
        session.last_view_state = state
        if state == "detached" and session.host is not None:
            await self._send_broker_message(session.host, session, "viewState", state=state)

    async def _notify_connection_state(self, session: _Session, state: str) -> None:
        for view in list(session.views.values()):
            message = self._new_broker_message(session, "connectionState", state=state)
            if not await self._send(view, message):
                await self._drop_connection(session, view)

    async def _send_error(
        self, connection: _Connection | None, session: _Session, code: str
    ) -> None:
        if connection is None:
            return
        await self._send_broker_message(connection, session, "error", code=code, message=_ERROR_TEXT[code])

    async def _refuse(self, connection: _Connection, session: _Session, code: str) -> None:
        """Handshake-stage refusal: typed error first, close with the paired code."""
        await self._send_error(connection, session, code)
        with contextlib.suppress(Exception):
            await connection.websocket.close(
                code=LOCAL_BROWSER_CLOSE_CODES[code], reason=code
            )

    async def _deny(self, websocket: WebSocket, code: str) -> None:
        """Pre-accept refusal: nothing about the peer is trusted yet."""
        with contextlib.suppress(Exception):
            await websocket.close(code=LOCAL_BROWSER_CLOSE_CODES[code], reason=code)

    async def _send_broker_message(
        self, connection: _Connection, broker_session: _Session, message_type: str, **fields: Any
    ) -> None:
        await self._send(connection, self._new_broker_message(broker_session, message_type, **fields))

    def _new_broker_message(
        self, broker_session: _Session, message_type: str, **fields: Any
    ) -> dict[str, Any]:
        self._message_counter += 1
        return {
            "type": message_type,
            "protocolVersion": LOCAL_BROWSER_PROTOCOL_VERSION,
            "workspaceSessionId": broker_session.workspace_session_id,
            "messageId": f"broker-{self._message_counter}",
            **fields,
        }

    async def _send(self, connection: _Connection | None, message: dict[str, Any]) -> bool:
        if connection is None:
            return False
        try:
            await connection.websocket.send_text(encode_message(message))
        except Exception:
            return False
        return True

    def _new_cookie_key(self, session: _Session, role: str) -> str:
        key = secrets.token_urlsafe(32)
        self._cookie_keys[key] = (session.workspace_session_id, role)
        return key

    @staticmethod
    def _is_expired(session: _Session) -> bool:
        return _utcnow() >= session.expires_at

_ERROR_TEXT = {
    "PROTOCOL_VERSION_MISMATCH": "Protocol version 1 is required.",
    "SESSION_MISMATCH": "The message names a different workspace session.",
    "INVALID_ORIGIN": "The request origin is not the launch origin.",
    "INVALID_CREDENTIALS": "The session credential is unknown, wrong, or for the other role.",
    "CREDENTIAL_EXPIRED": "The session has expired. Reopen the graph from VS Code.",
    "HOST_ALREADY_CONNECTED": "A host is already connected to this session.",
    "UNKNOWN_MESSAGE": "Unknown message type.",
    "INVALID_MESSAGE": "Malformed, misdirected, or out-of-order message.",
}
