"""The local-browser session, bootstrap and broker (extension-webpage step 2).

Why this is tested at the socket and not at the validator: step 1 proved the
contract distinguishes valid messages from invalid ones; what only the broker
can prove is the *machine around them* — who may open a socket, which role a
cookie authorizes, who hears what in which order, and what dies when. Those are
the properties docs/extension-local-webpage-protocol.md's security boundary
asserts, so each test here names the boundary line it holds.

The app under test is a private FastAPI instance: the broker reads
TENSORVIZ_LOCAL_TOKEN at request time (like auth.py), so no module reload is
needed and the rest of the suite is untouched. Two integration tests at the
bottom load the real server_v2 to prove the router is actually mounted.
"""

import contextlib
import logging
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


from middleware import LOCAL_TOKEN_HEADER, LoggingMiddleware  # noqa: E402
from services.ide.local_browser_protocol import (  # noqa: E402
    LOCAL_BROWSER_ERROR_CODES,
    is_local_browser_message,
)
from services.ide.local_browser_broker import (  # noqa: E402
    LOCAL_BROWSER_CLOSE_CODES,
    LOCAL_BROWSER_COOKIE,
    normalize_launch_origin,
    view_cookie_name,
)
from api.routers.local_browser import router as local_browser_router  # noqa: E402

TOKEN = "launch-token-9f3b"
ORIGIN = "http://127.0.0.1:8001"
DIGEST = "0123456789abcdef" * 4


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("TENSORVIZ_LOCAL_TOKEN", TOKEN)
    return monkeypatch


@pytest.fixture
def client(env):
    app = FastAPI()
    app.include_router(local_browser_router)
    # Entering the client gives the whole test one shared portal — one event
    # loop, like the uvicorn loop the broker runs on in production. Without it
    # every request and socket gets its own short-lived loop, which cancels a
    # session's expiry timer as soon as the mint request ends.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def mint(client, ttl_seconds=3600):
    response = client.post(
        "/local/sessions",
        json={"launchOrigin": ORIGIN, "ttlSeconds": ttl_seconds},
        headers={LOCAL_TOKEN_HEADER: TOKEN},
    )
    assert response.status_code == 200, response.text
    return response.json()


def bootstrap(client, session, credential=None, origin=ORIGIN):
    credential = session["viewCredential"] if credential is None else credential
    headers = {} if origin is None else {"Origin": origin}
    return client.post(
        "/local/bootstrap",
        json={"workspaceSessionId": session["workspaceSessionId"], "credential": credential},
        headers=headers,
    )


def cookie_key_of(bootstrap_response) -> str:
    set_cookie = bootstrap_response.headers["set-cookie"]
    return set_cookie.split(";", 1)[0].split("=", 1)[1]


def ws_headers(key, origin=ORIGIN):
    headers = {"Cookie": f"{LOCAL_BROWSER_COOKIE}={key}"}
    if origin is not None:
        headers["Origin"] = origin
    return headers


def envelope(type_, session_id, n, **fields):
    return {
        "type": type_,
        "protocolVersion": 1,
        "workspaceSessionId": session_id,
        "messageId": f"m-{n}",
        **fields,
    }


def target():
    return {
        "workspaceFolder": "workspace-a",
        "file": "model.py",
        "cls": "Net",
        "documentVersion": 3,
        "sourceDigest": DIGEST,
        "backendRevision": 7,
    }


def handshake(sock, role, session_id, n=1):
    expected = "hostHandshake" if role == "host" else "viewHandshake"
    sock.send_json(envelope(expected, session_id, n, role=role))
    return sock.receive_json()


def connect_host(client, session, n=1, expect_detached_level=True):
    """Connect a host and absorb the roster level that follows its handshake.

    A new host epoch knows nothing about which tabs are open, so the broker states
    the roster — but only when it is EMPTY, which is the answer the host acts on.
    Absorbing it here keeps every caller reading the frame it means to read.

    Pass False when no level is coming: a registration the broker refuses never gets
    one, and neither does a host that connects while a view is already attached.
    An unconditional read would block forever on both.
    """
    sock = client.websocket_connect("/local/ws", headers=ws_headers(session["hostCookieKey"]))
    sock.__enter__()
    connected = handshake(sock, "host", session["workspaceSessionId"], n)
    if expect_detached_level:
        level = sock.receive_json()
        assert level["type"] == "viewState", level
        assert level["state"] == "detached", level
    return sock, connected


def connect_view(client, key, session, n=50):
    sock = client.websocket_connect("/local/ws", headers=ws_headers(key))
    sock.__enter__()
    connected = handshake(sock, "view", session["workspaceSessionId"], n)
    link = sock.receive_json()  # broker always states host presence after connecting
    return sock, connected, link


@contextlib.contextmanager
def entered(*socks):
    """Exit-only lifecycle for sockets connect_host/connect_view return already
    entered: entering one again would start a second ASGI connection that hangs
    waiting for a handshake nobody sends, so a scope means "exit exactly once."
    """

    try:
        yield
    finally:
        for sock in socks:
            sock.__exit__(None, None, None)


def assert_valid(message, session_id):
    assert is_local_browser_message(message, session_id), message


# --- session minting: the only launch-token-gated route ---------------------


def test_mint_requires_launch_token_configured(monkeypatch):
    monkeypatch.delenv("TENSORVIZ_LOCAL_TOKEN", raising=False)
    app = FastAPI()
    app.include_router(local_browser_router)
    unconfigured = TestClient(app)
    response = unconfigured.post("/local/sessions", json={"launchOrigin": ORIGIN})
    assert response.status_code == 404


def test_mint_requires_the_launch_token(client):
    missing = client.post("/local/sessions", json={"launchOrigin": ORIGIN})
    assert missing.status_code == 401
    assert missing.json()["detail"]["code"] == "local_token_required"

    wrong = client.post(
        "/local/sessions",
        json={"launchOrigin": ORIGIN},
        headers={LOCAL_TOKEN_HEADER: "not-the-token"},
    )
    assert wrong.status_code == 401


def test_mint_returns_both_role_credentials_and_read_only_capabilities(client):
    session = mint(client)
    assert session["workspaceSessionId"] != ""
    assert session["launchOrigin"] == ORIGIN
    assert session["viewCredential"] != session["hostCookieKey"]
    assert session["cookieName"] == LOCAL_BROWSER_COOKIE
    assert session["protocolVersion"] == 1
    assert session["capabilities"] == {
        "readOnly": True,
        "canNavigateSource": True,
        "canWriteSource": False,
        "canRunAgent": False,
        "canEditGraph": False,
        "canProposeSourceEdits": True,
    }
    assert session["expiresAt"].endswith("Z")


def test_mint_rejects_a_non_origin(client):
    response = client.post(
        "/local/sessions",
        json={"launchOrigin": "http://127.0.0.1:8001/some/page"},
        headers={LOCAL_TOKEN_HEADER: TOKEN},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_launch_origin"


def test_launch_origin_normalization():
    assert normalize_launch_origin("HTTP://127.0.0.1:8001/") == "http://127.0.0.1:8001"
    with pytest.raises(ValueError):
        normalize_launch_origin("not-a-url")


# --- bootstrap: fragment credential -> HttpOnly cookie ----------------------


def test_bootstrap_sets_the_scoped_cookie_and_never_returns_a_credential(client):
    session = mint(client)
    response = bootstrap(client, session)
    assert response.status_code == 200
    set_cookie = response.headers["set-cookie"]
    assert set_cookie.startswith(view_cookie_name(ORIGIN) + "=")
    for attribute in ("HttpOnly", "SameSite=strict", "Path=/local"):
        assert attribute in set_cookie
    assert "Secure" not in set_cookie  # http launch origin only
    body = response.text
    assert session["viewCredential"] not in body
    assert cookie_key_of(response) not in body
    info = response.json()
    assert info["role"] == "view"
    assert info["workspaceSessionId"] == session["workspaceSessionId"]


def test_bootstrap_refusals_name_their_protocol_code(client):
    session = mint(client)

    assert bootstrap(client, session, credential="wrong").status_code == 401
    unknown = client.post(
        "/local/bootstrap",
        json={"workspaceSessionId": "no-such-session", "credential": "anything"},
        headers={"Origin": ORIGIN},
    )
    assert unknown.status_code == 401
    assert unknown.json()["detail"]["code"] == "INVALID_CREDENTIALS"
    assert session["viewCredential"] not in unknown.text

    assert bootstrap(client, session, origin="http://evil.local").status_code == 403
    wrong_origin = bootstrap(client, session, origin="http://evil.local")
    assert wrong_origin.json()["detail"]["code"] == "INVALID_ORIGIN"
    assert bootstrap(client, session, origin=None).status_code == 403

    # The host cookie key is not a bootstrap credential: the browser flow can
    # never mint itself host authority.
    host_key_attempt = bootstrap(client, session, credential=session["hostCookieKey"])
    assert host_key_attempt.status_code == 401


# --- websocket authentication -----------------------------------------------


def test_socket_refuses_missing_or_wrong_cookie_before_accept(client):
    mint(client)
    with pytest.raises(WebSocketDisconnect) as missing:
        client.websocket_connect("/local/ws", headers={"Origin": ORIGIN}).__enter__()
    assert missing.value.code == LOCAL_BROWSER_CLOSE_CODES["INVALID_CREDENTIALS"]
    assert missing.value.reason == "INVALID_CREDENTIALS"

    with pytest.raises(WebSocketDisconnect) as wrong:
        client.websocket_connect(
            "/local/ws", headers=ws_headers("forged-key")
        ).__enter__()
    assert wrong.value.code == LOCAL_BROWSER_CLOSE_CODES["INVALID_CREDENTIALS"]


def test_socket_refuses_wrong_or_missing_origin_before_accept(client):
    session = mint(client)
    view_key = cookie_key_of(bootstrap(client, session))
    for origin in ("http://evil.local", None):
        with pytest.raises(WebSocketDisconnect) as refused:
            client.websocket_connect(
                "/local/ws", headers=ws_headers(view_key, origin=origin)
            ).__enter__()
        assert refused.value.code == LOCAL_BROWSER_CLOSE_CODES["INVALID_ORIGIN"]


def test_ssh_forwarded_origin_is_exact_for_bootstrap_and_both_socket_roles(client):
    external = "http://127.0.0.1:53128"
    response = client.post("/local/sessions", json={"launchOrigin": external},
                           headers={LOCAL_TOKEN_HEADER: TOKEN})
    assert response.status_code == 200
    session = response.json()
    assert bootstrap(client, session, origin=ORIGIN).status_code == 403
    accepted = bootstrap(client, session, origin=external)
    assert accepted.status_code == 200
    assert "HttpOnly" in accepted.headers["set-cookie"]
    for role, key in [("host", session["hostCookieKey"]), ("view", cookie_key_of(accepted))]:
        for wrong in [ORIGIN, "http://127.0.0.1:53129", "http://localhost:53128", None]:
            with pytest.raises(WebSocketDisconnect) as refused:
                with client.websocket_connect("/local/ws", headers=ws_headers(key, origin=wrong)):
                    pass
            assert refused.value.code == LOCAL_BROWSER_CLOSE_CODES["INVALID_ORIGIN"]
        with client.websocket_connect("/local/ws", headers=ws_headers(key, origin=external)) as sock:
            assert handshake(sock, role, session["workspaceSessionId"])["type"] == "connected"


def test_browser_cookies_keep_two_forwarded_window_origins_independent(client):
    first = mint(client)
    bootstrap(client, first)
    second_origin = "http://127.0.0.1:8002"
    response = client.post("/local/sessions", json={"launchOrigin": second_origin},
                           headers={LOCAL_TOKEN_HEADER: TOKEN})
    assert response.status_code == 200
    second = response.json()
    bootstrap(client, second, origin=second_origin)
    # Browsers share cookies across ports. A second SSH window must not replace
    # the first window's credentials or prevent its socket reconnecting.
    for session, origin in [(first, ORIGIN), (second, second_origin)]:
        with client.websocket_connect("/local/ws", headers={"Origin": origin}) as view:
            connected = handshake(view, "view", session["workspaceSessionId"])
            assert connected["session"]["workspaceSessionId"] == session["workspaceSessionId"]
            assert view.receive_json()["state"] == "disconnected"


def test_socket_accepts_the_declared_subprotocol_none_and_jar_cookie(client):
    """httpx keeps the Set-Cookie in the client jar; a browser does the same."""
    session = mint(client)
    bootstrap(client, session)
    with client.websocket_connect("/local/ws", headers={"Origin": ORIGIN}) as view:
        connected = handshake(view, "view", session["workspaceSessionId"])
        assert connected["type"] == "connected"
        assert_valid(connected, session["workspaceSessionId"])
        assert view.receive_json()["state"] == "disconnected"


# --- handshake refusals: typed error, then close ----------------------------


def test_wrong_protocol_version_is_refused_with_the_typed_error(client):
    session = mint(client)
    sock = client.websocket_connect("/local/ws", headers=ws_headers(session["hostCookieKey"]))
    with sock:
        sock.send_json(envelope("hostHandshake", session["workspaceSessionId"], 1, role="host",
                               protocolVersion=2))
        error = sock.receive_json()
        assert_valid(error, session["workspaceSessionId"])
        assert error["code"] == "PROTOCOL_VERSION_MISMATCH"
        with pytest.raises(WebSocketDisconnect) as closed:
            sock.receive_json()
        assert closed.value.code == LOCAL_BROWSER_CLOSE_CODES["PROTOCOL_VERSION_MISMATCH"]


def test_cross_session_handshake_is_refused(client):
    session_a, session_b = mint(client), mint(client)
    view_key = cookie_key_of(bootstrap(client, session_a))
    sock = client.websocket_connect("/local/ws", headers=ws_headers(view_key))
    with sock:
        sock.send_json(envelope("viewHandshake", session_b["workspaceSessionId"], 1, role="view"))
        assert sock.receive_json()["code"] == "SESSION_MISMATCH"
        with pytest.raises(WebSocketDisconnect) as closed:
            sock.receive_json()
        assert closed.value.code == LOCAL_BROWSER_CLOSE_CODES["SESSION_MISMATCH"]


def test_malformed_and_unknown_messages_are_refused(client):
    session = mint(client)

    sock = client.websocket_connect("/local/ws", headers=ws_headers(session["hostCookieKey"]))
    with sock:
        sock.send_text("not json")
        assert sock.receive_json()["code"] == "INVALID_MESSAGE"
        with pytest.raises(WebSocketDisconnect):
            sock.receive_json()

    sock = client.websocket_connect("/local/ws", headers=ws_headers(session["hostCookieKey"]))
    with sock:
        sock.send_json(envelope("nonsenseType", session["workspaceSessionId"], 1))
        assert sock.receive_json()["code"] == "UNKNOWN_MESSAGE"


def test_a_view_credential_cannot_claim_the_host_role(client):
    session = mint(client)
    view_key = cookie_key_of(bootstrap(client, session))
    sock = client.websocket_connect("/local/ws", headers=ws_headers(view_key))
    with sock:
        sock.send_json(envelope("hostHandshake", session["workspaceSessionId"], 1, role="host"))
        assert sock.receive_json()["code"] == "INVALID_CREDENTIALS"
        with pytest.raises(WebSocketDisconnect) as closed:
            sock.receive_json()
        assert closed.value.code == LOCAL_BROWSER_CLOSE_CODES["INVALID_CREDENTIALS"]


def test_session_info_fields_present_in_connected(client):
    session = mint(client)
    sock, connected = connect_host(client, session)
    with entered(sock):
        assert connected["role"] == "host"
        assert connected["session"]["expiresAt"] == session["expiresAt"]
        assert_valid(connected, session["workspaceSessionId"])


# --- topology, relay, ordering ----------------------------------------------


def test_duplicate_host_is_refused_while_views_stay_connected(client):
    session = mint(client)
    host_a, _ = connect_host(client, session)
    view_key = cookie_key_of(bootstrap(client, session))
    view, _, link = connect_view(client, view_key, session)
    assert link["state"] == "connected"

    # Refused before registration, so it is never levelled: asking for the frame
    # would block on one the broker has no reason to send.
    host_b, connected_b = connect_host(client, session, n=900, expect_detached_level=False)
    with entered(host_b):
        assert connected_b["type"] == "connected"  # never registered as host
        error = host_b.receive_json()
        assert error["code"] == "HOST_ALREADY_CONNECTED"
        assert_valid(error, session["workspaceSessionId"])
        with pytest.raises(WebSocketDisconnect) as closed:
            host_b.receive_json()
        assert closed.value.code == LOCAL_BROWSER_CLOSE_CODES["HOST_ALREADY_CONNECTED"]

    with entered(host_a, view):
        host_a.send_json(envelope("activeTarget", session["workspaceSessionId"], 610,
                                  hostSequence=1, target=target()))
        assert view.receive_json()["type"] == "activeTarget"  # the real host still works


def test_two_views_both_receive_host_state_in_sequence(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    keys = [cookie_key_of(bootstrap(client, session)) for _ in range(2)]
    connected_views = [connect_view(client, key, session, n=60 + i) for i, key in enumerate(keys)]
    views = [sock for sock, _, _ in connected_views]
    # Each view hears about the already-connected host as it connects.
    assert [link["state"] for _, _, link in connected_views] == ["connected", "connected"]

    with entered(host, *views):
        for sequence in (1, 2, 3):
            host.send_json(envelope("projection", session["workspaceSessionId"], 100 + sequence,
                                    hostSequence=sequence, target=target(), projection={},
                                    traced=False, greyed=False))
        for view in views:
            received = [view.receive_json() for _ in range(3)]
            assert [m["hostSequence"] for m in received] == [1, 2, 3]
            assert all(m["type"] == "projection" for m in received)


def test_compressed_graph_keeps_order_and_cannot_bypass_session_or_role(client):
    from services.ide.local_browser_wire import decode_message, encode_message, MAX_WIRE_BYTES
    session = mint(client)
    host, _ = connect_host(client, session)
    view, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    sid = session["workspaceSessionId"]
    graph = {"evidence": "measured\n" * 1_000_000}
    message = envelope("projection", sid, 100, hostSequence=1, target=target(),
                       projection=graph, traced=True, greyed=False)
    encoded = encode_message(message)
    assert len(encoded.encode()) < MAX_WIRE_BYTES
    with entered(host, view):
        host.send_text(encode_message({**message, "workspaceSessionId": "other"}))
        assert host.receive_json()["code"] == "SESSION_MISMATCH"
        view.send_text(encoded)
        assert view.receive_json()["code"] == "INVALID_MESSAGE"
        host.send_text(encoded)
        host.send_json(envelope("syncState", sid, 101, hostSequence=2, target=target(), state="runtime-traced"))
        assert decode_message(view.receive_text()) == message
        assert view.receive_json()["hostSequence"] == 2


def test_view_messages_reach_the_host_and_are_never_echoed(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    view_key = cookie_key_of(bootstrap(client, session))
    view, _, link = connect_view(client, view_key, session)
    assert link["state"] == "connected"

    with entered(host, view):
        view.send_json(envelope("revealNode", session["workspaceSessionId"], 200,
                                target=target(), nodeId="node-1"))
        received = host.receive_json()
        assert received["type"] == "revealNode"
        assert received["nodeId"] == "node-1"


def test_out_of_order_host_messages_are_dropped_and_reported(client):
    """A view must never see the same or an older sequence twice: the graph it
    renders is only trustworthy if host state moved forward."""
    session = mint(client)
    host, _ = connect_host(client, session)
    view_key = cookie_key_of(bootstrap(client, session))
    view, _, link = connect_view(client, view_key, session)
    assert link["state"] == "connected"

    with entered(host, view):
        for index, sequence in enumerate((5, 4, 5, 6)):
            host.send_json(envelope("focusNode", session["workspaceSessionId"],
                                    300 + sequence, hostSequence=sequence,
                                    target=target(), nodeId=f"n-{sequence}"))
            # Only the rewind (4) and the repeat (second 5) are refused; the
            # first 5 and the 6 are fresh and relay. Values alone cannot name
            # the refused frames — the sequence value 5 appears on both sides.
            if index in (1, 2):
                error = host.receive_json()
                assert error["code"] == "INVALID_MESSAGE"
        assert [view.receive_json()["hostSequence"] for _ in range(2)] == [5, 6]


def test_misdirected_messages_are_refused_to_the_sender(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    view_key = cookie_key_of(bootstrap(client, session))
    view, _, _ = connect_view(client, view_key, session)

    with entered(host, view):
        view.send_json(envelope("projection", session["workspaceSessionId"], 210,
                                hostSequence=99, target=target(), projection={},
                                traced=False, greyed=False))
        assert view.receive_json()["code"] == "INVALID_MESSAGE"
        host.send_json(envelope("snapshotRequest", session["workspaceSessionId"], 310,
                                role="host", lastHostSequence=0))
        assert host.receive_json()["code"] == "INVALID_MESSAGE"
        # Both connections survive their own mistake.
        host.send_json(envelope("focusNode", session["workspaceSessionId"], 311,
                                hostSequence=1, target=target(), nodeId="n"))
        assert view.receive_json()["type"] == "focusNode"


def test_broker_owned_messages_cannot_be_injected(client):
    session = mint(client)
    view_key = cookie_key_of(bootstrap(client, session))
    view, _, _ = connect_view(client, view_key, session)
    with entered(view):
        forged = envelope("connectionState", session["workspaceSessionId"], 220, state="disconnected")
        view.send_json(forged)
        assert view.receive_json()["code"] == "INVALID_MESSAGE"


# --- host lifecycle ----------------------------------------------------------


def test_host_disconnect_marks_views_disconnected_and_reconnect_restores_ordering(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    view_key = cookie_key_of(bootstrap(client, session))
    view, _, link = connect_view(client, view_key, session)
    assert link["state"] == "connected"

    with entered(view):
        host.__exit__(None, None, None)
        assert view.receive_json()["state"] == "disconnected"

        # A view is still attached, so this epoch is told nothing.
        host_b, _ = connect_host(client, session, n=950, expect_detached_level=False)
        with entered(host_b):
            assert view.receive_json()["state"] == "connected"
            # New epoch: ordering restarts, so a fresh host's sequence 1 is valid.
            host_b.send_json(envelope("activeTarget", session["workspaceSessionId"], 960,
                                      hostSequence=1, target=target()))
            assert view.receive_json()["hostSequence"] == 1


def test_view_messages_without_a_host_are_loud_not_silent(client):
    session = mint(client)
    view_key = cookie_key_of(bootstrap(client, session))
    view, _, link = connect_view(client, view_key, session)
    assert link["state"] == "disconnected"
    with entered(view):
        view.send_json(envelope("revealNode", session["workspaceSessionId"], 230,
                                target=target(), nodeId="n"))
        assert view.receive_json()["code"] == "INVALID_MESSAGE"


# --- ask routing: per-view request/response ----------------------------------


def request(type_, session, n, request_id, **fields):
    envelope_ = envelope(type_, session["workspaceSessionId"], n,
                         target=target(), requestId=request_id, **fields)
    if type_ == "sourceEditRequest":
        envelope_.setdefault("edit", {"kind": "set_param", "nodeId": "enc1", "param": "out_features", "value": 8})
    if type_ == "sourceEditResult":
        envelope_.setdefault("status", "applied")
    return envelope_


def test_source_request_roundtrip_routes_answers_to_the_asking_view_only(client):
    """Two views, two in-flight questions, answers returned out of order: each
    answer must land on the view whose requestId it names. Any broadcast or
    cross-wire misrouting breaks a first-receive assertion below."""
    session = mint(client)
    host, _ = connect_host(client, session)
    view_a_key = cookie_key_of(bootstrap(client, session))
    view_a, _, _ = connect_view(client, view_a_key, session, n=50)
    view_b_key = cookie_key_of(bootstrap(client, session))
    view_b, _, _ = connect_view(client, view_b_key, session, n=60)

    with entered(host, view_a, view_b):
        view_a.send_json(request("sourceEditRequest", session, 100, "req-1"))
        view_b.send_json(request("sourceEditRequest", session, 101, "req-2"))
        assert host.receive_json()["requestId"] == "req-1"
        assert host.receive_json()["requestId"] == "req-2"

        # Out of order: req-2 answered first. A broadcast or misroute surfaces
        # immediately as the wrong message on a first receive.
        host.send_json(request("sourceEditResult", session, 200, "req-2", detail="three layers"))
        assert view_b.receive_json()["requestId"] == "req-2"
        host.send_json(request("sourceEditResult", session, 201, "req-1", detail="enc1 downsamples"))
        assert view_a.receive_json()["requestId"] == "req-1"
        # And the typed failure half routes the same way.
        view_a.send_json(request("sourceEditRequest", session, 102, "req-3"))
        assert host.receive_json()["requestId"] == "req-3"
        host.send_json(request("sourceEditResult", session, 202, "req-3", status="refused",
                           detail="no projection exists for this target yet"))
        failure = view_a.receive_json()
        assert failure["requestId"] == "req-3"
        assert failure["status"] == "refused"


def test_duplicate_source_request_id_is_refused(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    view_key = cookie_key_of(bootstrap(client, session))
    view, _, _ = connect_view(client, view_key, session)

    with entered(host, view):
        view.send_json(request("sourceEditRequest", session, 110, "req-1"))
        assert host.receive_json()["type"] == "sourceEditRequest"
        view.send_json(request("sourceEditRequest", session, 111, "req-1"))
        assert view.receive_json()["code"] == "INVALID_MESSAGE"
        # The host saw exactly one copy; after a legal answer the table is
        # empty, so the id is reusable — a resend must not brick the page.
        host.send_json(request("sourceEditResult", session, 210, "req-1", detail="done"))
        assert view.receive_json()["type"] == "sourceEditResult"
        view.send_json(request("sourceEditRequest", session, 112, "req-1"))
        assert host.receive_json()["requestId"] == "req-1"


def test_owner_cancel_abandons_and_the_late_answer_dies_quietly(client):
    """After a cancel the pending entry is gone: the host's late reply has no
    destination and must reach nobody — proven by the following traffic, which
    any leaked delivery or protocol error would interrupt."""
    session = mint(client)
    host, _ = connect_host(client, session)
    view_key = cookie_key_of(bootstrap(client, session))
    view, _, _ = connect_view(client, view_key, session)
    other_key = cookie_key_of(bootstrap(client, session))
    other, _, _ = connect_view(client, other_key, session, n=60)

    with entered(host, view, other):
        view.send_json(request("sourceEditRequest", session, 120, "req-1"))
        assert host.receive_json()["type"] == "sourceEditRequest"
        view.send_json(request("sourceEditCancel", session, 121, "req-1"))
        cancel = host.receive_json()
        assert cancel["type"] == "sourceEditCancel"
        assert cancel["requestId"] == "req-1"
        # The late answer has no pending entry left: dropped, nobody errors.
        host.send_json(request("sourceEditResult", session, 220, "req-1", detail="late"))
        # A cancel that names another view's request abandons nothing.
        other.send_json(request("sourceEditRequest", session, 122, "req-9"))
        assert host.receive_json()["requestId"] == "req-9"
        view.send_json(request("sourceEditCancel", session, 123, "req-9"))
        # Positive read: the host's next message is healthy ordered state, not
        # a refusal triggered by the foreign cancel or the late reply.
        host.send_json(envelope("focusNode", session["workspaceSessionId"], 221,
                                hostSequence=1, target=target(), nodeId="n"))
        assert other.receive_json()["type"] == "focusNode"
        assert view.receive_json()["type"] == "focusNode"
        # req-9 is still pending for its real owner and answers normally.
        host.send_json(request("sourceEditResult", session, 222, "req-9", detail="fine"))
        assert other.receive_json()["type"] == "sourceEditResult"


def test_view_disconnect_synthesizes_cancels_for_its_pending_requests(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    leaving_key = cookie_key_of(bootstrap(client, session))
    leaving, _, _ = connect_view(client, leaving_key, session, n=50)
    staying_key = cookie_key_of(bootstrap(client, session))
    staying, _, _ = connect_view(client, staying_key, session, n=60)

    leaving.send_json(request("sourceEditRequest", session, 130, "req-gone"))
    assert host.receive_json()["requestId"] == "req-gone"
    staying.send_json(request("sourceEditRequest", session, 131, "req-stays"))
    assert host.receive_json()["requestId"] == "req-stays"

    leaving.__exit__(None, None, None)
    with entered(host, staying):
        # The host is told to abandon exactly the departed view's request.
        cancel = host.receive_json()
        assert cancel["type"] == "sourceEditCancel"
        assert cancel["requestId"] == "req-gone"
        # The staying view's request survives and still answers.
        host.send_json(request("sourceEditResult", session, 230, "req-stays", detail="two edges"))
        assert staying.receive_json()["type"] == "sourceEditResult"


def test_publish_progress_and_result_reach_only_the_asking_view_and_cancel_when_it_leaves(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    asking, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=1800)
    other, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=1801)
    graph = {"id": "AbCdEfGhIjKlMnOpQrSt", "title": "Net", "url": "https://tensorviz.dev/explore/graph/?id=AbCdEfGhIjKlMnOpQrSt",
             "shareUrl": "https://tensorviz.dev/explore/graph/?id=AbCdEfGhIjKlMnOpQrSt&share=1"}
    asking.send_json(request("graphPublishRequest", session, 1802, "publish-a", file="H4sIAAAAAAAA"))
    forwarded = host.receive_json()
    assert forwarded["type"] == "graphPublishRequest" and forwarded["file"] == "H4sIAAAAAAAA"
    host.send_json(request("graphPublishState", session, 1803, "publish-a", phase="signing_in", detail="Sign in.",
                           signInUrl="https://tensorviz.dev/explore/connect/?state=abc"))
    assert asking.receive_json()["phase"] == "signing_in"
    host.send_json(request("graphPublishState", session, 1804, "publish-a", phase="uploading", detail="Publishing Net."))
    assert asking.receive_json()["phase"] == "uploading", "progress does not end the request"
    host.send_json(request("graphPublishResult", session, 1805, "publish-a", status="published", detail="Published.", graph=graph))
    assert asking.receive_json()["graph"] == graph
    # Answered: a late state has no destination left.
    host.send_json(request("graphPublishState", session, 1806, "publish-a", phase="uploading", detail="Late."))
    asking.send_json(request("graphPublishRequest", session, 1807, "publish-left", file="H4sIAAAAAAAA"))
    assert host.receive_json()["requestId"] == "publish-left"

    asking.__exit__(None, None, None)
    with entered(host, other):
        cancel = host.receive_json()
        assert (cancel["type"], cancel["requestId"]) == ("graphPublishCancel", "publish-left")
        host.send_json(envelope("focusNode", session["workspaceSessionId"], 1808, hostSequence=1, target=target(), nodeId="n"))
        assert other.receive_json()["type"] == "focusNode", "publish traffic never reaches another view"


def test_source_request_without_a_host_and_unknown_host_replies_are_handled(client):
    session = mint(client)
    view_key = cookie_key_of(bootstrap(client, session))
    view, _, link = connect_view(client, view_key, session)
    assert link["state"] == "disconnected"
    with entered(view):
        view.send_json(request("sourceEditRequest", session, 140, "req-1"))
        assert view.receive_json()["code"] == "INVALID_MESSAGE"

    host, _ = connect_host(client, session)
    # The closed view is gone for good; reconnect the page on the same cookie.
    view, _, _ = connect_view(client, view_key, session)
    with entered(host, view):
        # State refresh first: the view re-attaches through the snapshot path,
        # which the ask routing must not disturb.
        view.send_json(envelope("snapshotRequest", session["workspaceSessionId"], 141,
                                role="view", lastHostSequence=0))
        assert host.receive_json()["type"] == "snapshotRequest"
        # A reply naming a request nobody asked is dropped, and the next legal
        # view message still relays — no error interrupts the session.
        host.send_json(request("sourceEditResult", session, 240, "req-unknown",
                           status="refused", detail="no such request"))
        view.send_json(request("sourceEditRequest", session, 142, "req-2"))
        assert host.receive_json()["requestId"] == "req-2"


def test_host_cannot_propose_and_views_cannot_answer(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    view_key = cookie_key_of(bootstrap(client, session))
    view, _, _ = connect_view(client, view_key, session)

    with entered(host, view):
        host.send_json(request("sourceEditRequest", session, 250, "req-1"))
        assert host.receive_json()["code"] == "INVALID_MESSAGE"
        view.send_json(request("sourceEditResult", session, 150, "req-1", detail="impersonation"))
        assert view.receive_json()["code"] == "INVALID_MESSAGE"


# --- snapshot routing and reconnect ------------------------------------------


def test_snapshot_roundtrip_ties_request_to_answer(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    view_key = cookie_key_of(bootstrap(client, session))
    view, _, _ = connect_view(client, view_key, session)

    with entered(host, view):
        view.send_json(envelope("snapshotRequest", session["workspaceSessionId"], 240,
                                role="view", lastHostSequence=4))
        request = host.receive_json()
        assert request["type"] == "snapshotRequest"
        assert request["lastHostSequence"] == 4

        host.send_json(envelope("snapshot", session["workspaceSessionId"], 320, role="host",
                                hostSequence=5,
                                state={"activeTarget": None, "projection": None, "traced": False,
                                       "greyed": False, "syncState": None,
                                       "connectionState": "connected"}))
        snapshot = view.receive_json()
        assert_valid(snapshot, session["workspaceSessionId"])
        assert snapshot["hostSequence"] == 5


def test_view_refresh_recovers_the_host_snapshot(client):
    """Browser refresh: same cookie, brand new socket, one request, one answer —
    and the snapshot is delivered to the refresh, not to any other tab."""
    session = mint(client)
    host, _ = connect_host(client, session)
    view_key = cookie_key_of(bootstrap(client, session))

    with entered(connect_view(client, view_key, session)[0]):
        pass  # the pre-refresh tab: connected once, then gone
    # That tab was the only one, so the host is told the roster emptied.
    level = host.receive_json()
    assert level["type"] == "viewState" and level["state"] == "detached", level

    with entered(host):
        second, _, link = connect_view(client, view_key, session, n=70)
        with entered(second):
            assert link["state"] == "connected"
            second.send_json(envelope("snapshotRequest", session["workspaceSessionId"], 71,
                                      role="view", lastHostSequence=0))
            assert host.receive_json()["type"] == "snapshotRequest"
            host.send_json(envelope("snapshot", session["workspaceSessionId"], 330, role="host",
                                    hostSequence=1,
                                    state={"activeTarget": target(), "projection": {},
                                           "traced": False, "greyed": False,
                                           "syncState": "source-only",
                                           "connectionState": "connected"}))
            snapshot = second.receive_json()
            assert snapshot["type"] == "snapshot"
            assert snapshot["state"]["activeTarget"]["cls"] == "Net"


def test_a_snapshot_with_no_pending_request_is_refused(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    with entered(host):
        host.send_json(envelope("snapshot", session["workspaceSessionId"], 340, role="host",
                                hostSequence=1,
                                state={"activeTarget": None, "projection": None, "traced": False,
                                       "greyed": False, "syncState": None,
                                       "connectionState": "connected"}))
        assert host.receive_json()["code"] == "INVALID_MESSAGE"


def test_requests_queued_while_hostless_flush_when_the_host_connects(client):
    session = mint(client)
    view_key = cookie_key_of(bootstrap(client, session))
    view, _, link = connect_view(client, view_key, session)
    assert link["state"] == "disconnected"

    with entered(view):
        view.send_json(envelope("snapshotRequest", session["workspaceSessionId"], 250,
                                role="view", lastHostSequence=0))
        # A view is already attached, so this host is told no level.
        host, _ = connect_host(client, session, expect_detached_level=False)
        with entered(host):
            assert view.receive_json()["state"] == "connected"
            assert host.receive_json()["type"] == "snapshotRequest"
            host.send_json(envelope("snapshot", session["workspaceSessionId"], 350, role="host",
                                    hostSequence=1,
                                    state={"activeTarget": None, "projection": None,
                                           "traced": False, "greyed": False, "syncState": None,
                                           "connectionState": "connected"}))
            assert view.receive_json()["type"] == "snapshot"


# --- expiry -------------------------------------------------------------------


def test_expiry_closes_sockets_and_revokes_every_credential(client):
    session = mint(client, ttl_seconds=1)
    view_key = cookie_key_of(bootstrap(client, session))
    host, _ = connect_host(client, session)
    view, _, _ = connect_view(client, view_key, session)

    time.sleep(1.5)
    with entered(host, view):
        for sock, name in ((host, "host"), (view, "view")):
            with pytest.raises(WebSocketDisconnect) as closed:
                sock.receive_json()
            assert closed.value.code == LOCAL_BROWSER_CLOSE_CODES["CREDENTIAL_EXPIRED"], name
            assert closed.value.reason == "CREDENTIAL_EXPIRED"

    expired_bootstrap = bootstrap(client, session)
    assert expired_bootstrap.status_code == 410
    assert expired_bootstrap.json()["detail"]["code"] == "CREDENTIAL_EXPIRED"
    with pytest.raises(WebSocketDisconnect) as closed:
        client.websocket_connect("/local/ws", headers=ws_headers(view_key)).__enter__()
    assert closed.value.code == LOCAL_BROWSER_CLOSE_CODES["INVALID_CREDENTIALS"]
    with pytest.raises(WebSocketDisconnect) as host_closed:
        client.websocket_connect(
            "/local/ws", headers=ws_headers(session["hostCookieKey"])
        ).__enter__()
    assert host_closed.value.code == LOCAL_BROWSER_CLOSE_CODES["INVALID_CREDENTIALS"]


def test_expired_tombstones_are_reaped_after_a_bounded_retention(client, monkeypatch):
    """The tombstone's whole point is an actionable CREDENTIAL_EXPIRED for a
    late bootstrap. It must not live forever: a broker that keeps every session
    it ever minted grows without bound behind a long-lived backend."""
    import services.ide.local_browser_broker as broker_module

    monkeypatch.setattr(broker_module, "LOCAL_BROWSER_TOMBSTONE_RETENTION_SECONDS", 1.0)
    session = mint(client, ttl_seconds=1)
    time.sleep(1.5)
    tombstone = bootstrap(client, session)
    assert tombstone.status_code == 410  # still inside the retention window
    assert tombstone.json()["detail"]["code"] == "CREDENTIAL_EXPIRED"
    time.sleep(0.8)  # the reaper has now run: well past 1s TTL + 1s retention
    reaped = bootstrap(client, session)
    assert reaped.status_code == 401
    assert reaped.json()["detail"]["code"] == "INVALID_CREDENTIALS"


# --- redaction -----------------------------------------------------------------


def _log_text(caplog):
    return "\n".join(record.getMessage() for record in caplog.records)


def test_credentials_never_reach_logs_or_error_bodies(env, caplog):
    """Every secret this feature mints, exercised end to end, must appear in no
    log line and no error body — the one invariant the security section states."""
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)
    app.include_router(local_browser_router)

    with TestClient(app, raise_server_exceptions=False) as client:
        with caplog.at_level(logging.INFO, logger="uvicorn.error"):
            session = mint(client)
            view_key = cookie_key_of(bootstrap(client, session))
            bootstrap(client, session, credential="wrong")
            bootstrap(client, session, origin="http://evil.local")
            with pytest.raises(WebSocketDisconnect):
                client.websocket_connect("/local/ws", headers=ws_headers("forged")).__enter__()
            view, _, _ = connect_view(client, view_key, session)
            with entered(view):
                view.send_text("garbage")
                view.receive_json()

    logs = _log_text(caplog)
    for secret in (session["viewCredential"], session["hostCookieKey"], view_key):
        assert secret not in logs
        assert len(secret) > 20  # otherwise the assertion above says nothing


def test_messages_are_capped_before_parsing(client):
    session = mint(client)
    view_key = cookie_key_of(bootstrap(client, session))
    view, _, _ = connect_view(client, view_key, session)
    with entered(view):
        view.send_json(envelope("revealNode", session["workspaceSessionId"], 260,
                                target=target(), nodeId="x" * (8 * 1024 * 1024 + 16)))
        assert view.receive_json()["code"] == "INVALID_MESSAGE"
        view.send_json(envelope("revealNode", session["workspaceSessionId"], 261,
                                target=target(), nodeId="n"))
        assert view.receive_json()["code"] == "INVALID_MESSAGE"  # host absent, but parsed


# --- wiring into the real server ----------------------------------------------


def test_server_v2_mounts_the_broker_inert_without_a_launch_token(monkeypatch):
    monkeypatch.delenv("TENSORVIZ_LOCAL_TOKEN", raising=False)
    import server_v2

    client = TestClient(server_v2.app)
    response = client.post("/local/sessions", json={"launchOrigin": ORIGIN})
    assert response.status_code == 404


def test_server_v2_broker_roundtrip_with_launch_token(monkeypatch):
    monkeypatch.setenv("TENSORVIZ_LOCAL_TOKEN", TOKEN)
    import server_v2

    client = TestClient(server_v2.app)
    session = mint(client)
    response = bootstrap(client, session)
    assert response.status_code == 200
    view_key = cookie_key_of(response)
    with client.websocket_connect("/local/ws", headers=ws_headers(view_key)) as sock:
        connected = handshake(sock, "view", session["workspaceSessionId"])
        assert connected["session"]["workspaceSessionId"] == session["workspaceSessionId"]
        assert sock.receive_json()["state"] == "disconnected"


def test_every_close_code_pairs_with_a_contract_error_code():
    assert set(LOCAL_BROWSER_CLOSE_CODES) <= set(LOCAL_BROWSER_ERROR_CODES)


def test_host_is_told_when_its_last_view_goes_away(client):
    """The reopen bug: closing the browser tab leaves the host's own socket up, so
    without this frame the host kept believing a page was there and published
    retargets into an empty roster — "Open as Graph" appeared to do nothing."""
    session = mint(client)
    host, _ = connect_host(client, session)  # levelled `detached`: no views yet
    with entered(host):
        key = cookie_key_of(bootstrap(client, session))
        view, _, link = connect_view(client, key, session)
        assert link["state"] == "connected"

        view.__exit__(None, None, None)
        level = host.receive_json()
        assert level["type"] == "viewState"
        assert level["state"] == "detached"
        assert_valid(level, session["workspaceSessionId"])


def test_a_second_view_closing_is_not_news_but_the_last_one_is(client):
    """Edge-triggered, not per-socket: two tabs closing produce one `detached`, and
    it arrives only when the roster is actually empty."""
    session = mint(client)
    host, _ = connect_host(client, session)
    with entered(host):
        first, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
        second, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=80)

        first.__exit__(None, None, None)  # one tab left: nothing to say
        second.__exit__(None, None, None)  # now the roster is empty

        level = host.receive_json()
        assert level["type"] == "viewState" and level["state"] == "detached", level


def connect_control_host(client, session, expect_detached_level=True):
    sock = client.websocket_connect('/local/ws', headers=ws_headers(session['hostCookieKey']))
    sock.__enter__()
    sock.send_json(envelope('hostHandshake', session['workspaceSessionId'], 1001, role='host', viewControl=True))
    assert sock.receive_json()['type'] == 'connected'
    if expect_detached_level:
        level = sock.receive_json()
        assert level['type'] == 'viewState' and level['state'] == 'detached'
    return sock


def test_view_control_binds_selection_and_focus_to_the_actual_socket(client):
    session = mint(client)
    sid = session['workspaceSessionId']
    host = connect_control_host(client, session)
    with entered(host):
        key1 = cookie_key_of(bootstrap(client, session))
        key2 = cookie_key_of(bootstrap(client, session))
        first, first_connected, _ = connect_view(client, key1, session)
        second, second_connected, _ = connect_view(client, key2, session)
        with entered(first, second):
            id1 = first_connected['session']['viewId']
            id2 = second_connected['session']['viewId']
            assert id1 != id2
            assert host.receive_json()['viewId'] == id1
            assert host.receive_json()['viewId'] == id2
            selection = envelope('viewSelection', sid, 1002, viewId=id1, target=target(), nodeIds=['fc'], edgeIds=[])
            first.send_json(selection)
            assert host.receive_json() == selection
            second.send_json(selection)
            assert second.receive_json()['code'] == 'INVALID_MESSAGE'
            options = dict(enabled=True, scope='class', references='both', opacity=25)
            request = envelope('codeFocusRequest', sid, 1100, viewId=id1, target=target(), options=options)
            first.send_json(request)
            assert host.receive_json() == request
            second.send_json(request)
            assert second.receive_json()['code'] == 'INVALID_MESSAGE'
            host.send_json(request)
            assert host.receive_json()['code'] == 'INVALID_MESSAGE'
            state = envelope('codeFocusState', sid, 1101, viewId=id1, target=target(), options=options, parameter=True, detail='Direct references.')
            host.send_json(state)
            assert first.receive_json() == state
            first.send_json(state)
            assert first.receive_json()['code'] == 'INVALID_MESSAGE'
            review_receipt = envelope('graphReviewApplied', sid, 1007, viewId=id1, reviewId='review-1', mode='diff')
            first.send_json(review_receipt)
            assert host.receive_json() == review_receipt
            second.send_json(review_receipt)
            assert second.receive_json()['code'] == 'INVALID_MESSAGE'
            host.send_json(review_receipt)
            assert host.receive_json()['code'] == 'INVALID_MESSAGE'
            focus = envelope('focusView', sid, 1003, viewId=id1, target=target(), requestId='focus-1', nodeIds=['fc'], edgeIds=[])
            host.send_json(focus)
            assert first.receive_json() == focus
            # This broadcast must be the other view's next message: focus was not broadcast.
            update = envelope('activeTarget', sid, 1004, hostSequence=1, target=target())
            host.send_json(update)
            assert second.receive_json() == update
            assert first.receive_json() == update
            answer = envelope('viewFocusResult', sid, 1005, viewId=id1, target=target(), requestId='focus-1', ok=True, detail='Focused.')
            first.send_json(answer)
            assert host.receive_json() == answer


def test_view_selection_relays_code_focus_unchanged(client):
    session = mint(client)
    sid = session['workspaceSessionId']
    host = connect_control_host(client, session)
    with entered(host):
        view, connected, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
        with entered(view):
            view_id = connected['session']['viewId']
            assert host.receive_json()['viewId'] == view_id
            selection = dict(viewId=view_id, target=target(), nodeIds=['fc'], edgeIds=[])
            numbers = iter(range(1200, 1300))

            def relayed(fields):
                # A selection without Code focus always reaches VS Code, so the host's next
                # message shows whether the one before it was relayed or refused.
                message = envelope('viewSelection', sid, next(numbers), **fields)
                legacy = envelope('viewSelection', sid, next(numbers), **selection)
                view.send_json(message)
                view.send_json(legacy)
                received = host.receive_json()
                if received == legacy:
                    return False
                assert received == message and host.receive_json() == legacy
                return True

            assert relayed(dict(selection, follow=False))
            assert relayed(dict(selection, follow=True))
            assert relayed(dict(selection, nodeIds=[], tooLarge=True, follow=True))
            assert not relayed(dict(selection, follow='edit'))
            assert view.receive_json()['code'] == 'INVALID_MESSAGE'


def test_canvas_agent_requests_bind_to_the_socket_and_replies_are_private_to_the_view(client):
    session = mint(client)
    sid = session['workspaceSessionId']
    host = connect_control_host(client, session)
    with entered(host):
        first, first_connected, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
        second, second_connected, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
        with entered(first, second):
            id1 = first_connected['session']['viewId']
            assert host.receive_json()['viewId'] == id1
            assert host.receive_json()['viewId'] == second_connected['session']['viewId']
            request = envelope('agentRequest', sid, 1002, viewId=id1, action='sync')
            first.send_json(request)
            assert host.receive_json() == request
            second.send_json(request)
            assert second.receive_json()['code'] == 'INVALID_MESSAGE'
            reply = envelope('agentState', sid, 1003, viewId=id1, state={
                'sessionId': 'chat-1', 'revision': 1, 'provider': 'codex',
                'status': 'idle', 'messages': [], 'activity': '',
            })
            host.send_json(reply)
            assert first.receive_json() == reply
            first.send_json(reply)
            assert first.receive_json()['code'] == 'INVALID_MESSAGE'
            update = envelope('activeTarget', sid, 1004, hostSequence=1, target=target())
            host.send_json(update)
            assert second.receive_json() == update, 'a targeted transcript must not be broadcast'
            assert first.receive_json() == update


def test_view_control_reports_closed_views_and_refuses_missing_focus_target(client):
    session = mint(client)
    sid = session['workspaceSessionId']
    host = connect_control_host(client, session)
    with entered(host):
        view, connected, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
        view_id = connected['session']['viewId']
        assert host.receive_json()['connected'] is True
        view.__exit__(None, None, None)
        dropped = host.receive_json()
        assert dropped['type'] == 'viewState' and dropped['viewId'] == view_id and dropped['connected'] is False
        level = host.receive_json()
        assert level['type'] == 'viewState' and level['state'] == 'detached'
        host.send_json(envelope('focusView', sid, 1010, viewId=view_id, target=target(), requestId='gone', nodeIds=['fc'], edgeIds=[]))
        result = host.receive_json()
        assert result['type'] == 'viewFocusResult' and result['ok'] is False


def test_view_control_replays_existing_roster_to_a_new_host(client):
    session = mint(client)
    view, connected, link = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    with entered(view):
        assert link['state'] == 'disconnected'
        host = connect_control_host(client, session, expect_detached_level=False)
        with entered(host):
            roster = host.receive_json()
            assert roster['type'] == 'viewState' and roster['viewId'] == connected['session']['viewId']
            assert view.receive_json()['state'] == 'connected'


def test_a_host_without_view_control_cannot_disconnect_a_view_by_requesting_focus(client):
    session = mint(client)
    sid = session['workspaceSessionId']
    host, _ = connect_host(client, session)
    view, connected, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    with entered(view, host):
        host.send_json(envelope('focusView', sid, 1015, viewId=connected['session']['viewId'], target=target(), requestId='unsupported', nodeIds=['fc'], edgeIds=[]))
        assert host.receive_json()['code'] == 'INVALID_MESSAGE'
        update = envelope('activeTarget', sid, 1016, hostSequence=1, target=target())
        host.send_json(update)
        assert view.receive_json() == update


def test_project_wire_routes_both_directions_with_ordering_and_role_refusal(client):
    """Project traffic crosses the real broker; shape-only tests missed this gate."""
    session = mint(client)
    key = cookie_key_of(bootstrap(client, session))
    host, _ = connect_host(client, session)
    view, _, _ = connect_view(client, key, session)
    sid = session['workspaceSessionId']
    project = dict(workspaceFolder='/work', projectRoot='', experimentId='train', indexDigest=DIGEST, indexRevision=1)
    with entered(host, view):
        for n, kind, fields in [(1, 'projectIndex', dict(index={}, truncated=False, indexed=1, discovered=1)),
                                (2, 'projectDocument', dict(document={}, spanmap={}, expanded=['net']))]:
            message = envelope(kind, sid, n, project=project, hostSequence=n, **fields)
            host.send_json(message)
            assert view.receive_json() == message
            host.send_json(message)
            assert host.receive_json()['code'] == 'INVALID_MESSAGE', 'replayed project state must not be relayed'
            view.send_json(message)
            assert view.receive_json()['code'] == 'INVALID_MESSAGE', 'views cannot publish project state'
        for n, kind, fields in [(3, 'projectFocus', dict(experimentId='other')),
                                (4, 'projectExpand', dict(nodeId='net', expanded=True)),
                                (5, 'projectDeclare', dict(declaration='edge', args=['model', 'trainer', 'feeds']))]:
            message = envelope(kind, sid, n, project=project, **fields)
            view.send_json(message)
            assert host.receive_json() == message
            host.send_json(message)
            assert host.receive_json()['code'] == 'INVALID_MESSAGE', 'hosts cannot impersonate view requests'
            view.send_json(dict(message, target={}))
            assert view.receive_json()['code'] == 'INVALID_MESSAGE'


@pytest.mark.parametrize("kind", ["captureInspect", "captureExport"])
def test_capture_read_is_private_to_requesting_view_and_cancellable(client, kind):
    session = mint(client)
    host, _ = connect_host(client, session)
    a, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=1700)
    b, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=1701)
    query = dict(captureId='capture:one', sourceId='source:one', path=['operations'], offset=0, limit=10)
    request_fields = dict(query=query) if kind == 'captureInspect' else dict(reference={k: query[k] for k in ('captureId', 'sourceId')})
    result_fields = dict(page=None) if kind == 'captureInspect' else dict(capture=None, **request_fields)
    with entered(host, a, b):
        a.send_json(request(kind + 'Request', session, 1702, 'capture-a', **request_fields))
        assert host.receive_json()['requestId'] == 'capture-a'
        b.send_json(request(kind + 'Request', session, 1703, 'capture-b', **request_fields))
        assert host.receive_json()['requestId'] == 'capture-b'
        # A different tab cannot cancel another tab's read.
        b.send_json(request(kind + 'Cancel', session, 1704, 'capture-a'))
        host.send_json(request(kind + 'Result', session, 1705, 'capture-b', **result_fields, detail='No longer retained.'))
        assert b.receive_json()['requestId'] == 'capture-b'
        host.send_json(request(kind + 'Result', session, 1706, 'capture-a', **result_fields, detail='No longer retained.'))
        assert a.receive_json()['requestId'] == 'capture-a'
        a.send_json(request(kind + 'Request', session, 1707, 'cancelled', **request_fields))
        assert host.receive_json()['requestId'] == 'cancelled'
        a.send_json(request(kind + 'Cancel', session, 1708, 'cancelled'))
        assert host.receive_json()['type'] == kind + 'Cancel'
        host.send_json(request(kind + 'Result', session, 1709, 'cancelled', **result_fields, detail='Late page.'))
        host.send_json(envelope('focusNode', session['workspaceSessionId'], 1710, hostSequence=1, target=target(), nodeId='n'))
        assert a.receive_json()['type'] == 'focusNode', 'a cancelled result must be dropped'
        assert b.receive_json()['type'] == 'focusNode', 'capture details must never broadcast'


def mint_view_link(client, session):
    response = client.post(f"/local/sessions/{session['workspaceSessionId']}/view-links", headers={LOCAL_TOKEN_HEADER: TOKEN})
    assert response.status_code == 200, response.text
    assert response.headers['cache-control'] == 'no-store'
    return response.json()


def test_agent_view_link_requires_host_token_and_is_single_use(client):
    session = mint(client)
    path = f"/local/sessions/{session['workspaceSessionId']}/view-links"
    assert client.post(path).status_code == 401
    link = mint_view_link(client, session)
    assert 'hostCookieKey' not in link
    assert link['viewCredential'] != session['viewCredential']
    assert bootstrap(client, session, link['viewCredential'], origin='https://elsewhere.example').status_code == 403
    other = mint(client)
    assert bootstrap(client, other, link['viewCredential']).status_code == 401
    response = bootstrap(client, session, link['viewCredential'])
    assert response.status_code == 200
    assert response.json()['handoffId'] == link['handoffId']
    assert response.json()['role'] == 'view'
    assert bootstrap(client, session, link['viewCredential']).status_code == 401
    # Existing manual browser links continue to work.
    assert 'handoffId' not in bootstrap(client, session).json()


def test_agent_view_link_expiry_and_pending_limit(client, monkeypatch):
    from datetime import timedelta
    from api.routers.local_browser import broker
    import services.ide.local_browser_broker as module
    session = mint(client)
    link = mint_view_link(client, session)
    now = module._utcnow()
    monkeypatch.setattr(module, '_utcnow', lambda: now + timedelta(seconds=121))
    assert bootstrap(client, session, link['viewCredential']).status_code == 410
    monkeypatch.setattr(module, '_utcnow', lambda: now)
    for _ in range(16):
        mint_view_link(client, session)
    path = f"/local/sessions/{session['workspaceSessionId']}/view-links"
    assert client.post(path, headers={LOCAL_TOKEN_HEADER: TOKEN}).status_code == 429
    assert len(broker._sessions[session['workspaceSessionId']].view_links) == 16


def test_handoff_tracks_actual_socket_and_reconnect_without_upgrading_role(client):
    session = mint(client)
    sid = session['workspaceSessionId']
    link = mint_view_link(client, session)
    key = cookie_key_of(bootstrap(client, session, link['viewCredential']))
    host = connect_control_host(client, session)
    with entered(host):
        ids = []
        for n in range(2):
            with client.websocket_connect('/local/ws', headers=ws_headers(key)) as view:
                view.send_json(envelope('viewHandshake', sid, 1100 + n, role='view', handoffId=link['handoffId']))
                connected = view.receive_json()
                assert_valid(connected, sid)
                ids.append(connected['session']['viewId'])
                assert view.receive_json()['type'] == 'connectionState'
                roster = host.receive_json()
                assert_valid(roster, sid)
                assert roster['handoffId'] == link['handoffId']
                assert roster['viewId'] == ids[-1]
                with client.websocket_connect('/local/ws', headers=ws_headers(key)) as duplicate:
                    duplicate.send_json(envelope('viewHandshake', sid, 1110, role='view', handoffId=link['handoffId']))
                    assert duplicate.receive_json()['code'] == 'INVALID_MESSAGE'
            assert host.receive_json()['connected'] is False
            assert host.receive_json()['state'] == 'detached'
        assert ids[0] != ids[1]
        with client.websocket_connect('/local/ws', headers=ws_headers(key)) as view:
            view.send_json(envelope('hostHandshake', sid, 1111, role='host'))
            assert view.receive_json()['code'] == 'INVALID_CREDENTIALS'
        with client.websocket_connect('/local/ws', headers=ws_headers(key)) as view:
            view.send_json(envelope('viewHandshake', sid, 1112, role='view', handoffId='unissued'))
            assert view.receive_json()['code'] == 'INVALID_MESSAGE'
