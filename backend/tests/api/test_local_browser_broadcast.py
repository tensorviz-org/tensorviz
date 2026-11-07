"""Broadcast encoding is shared without weakening the broker boundary."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

import services.ide.local_browser_broker as broker_module
from services.ide.local_browser_broker import LocalBrowserBroker, _Connection, _Session
from services.ide.local_browser_wire import COMPRESSION_THRESHOLD, decode_message, encode_message
from tests.api.test_local_browser_broker import (
    bootstrap, client, connect_host, connect_view, cookie_key_of, entered, env,
    envelope, mint, target,
)


@pytest.mark.parametrize("large", [False, True])
def test_host_broadcast_encodes_once_for_all_views(client, monkeypatch, large):
    session = mint(client)
    host, _ = connect_host(client, session)
    views = [connect_view(client, cookie_key_of(bootstrap(client, session)), session)[0]
             for _ in range(3)]
    message = envelope("projection", session["workspaceSessionId"], 100,
                       hostSequence=1, target=target(), traced=True, greyed=False,
                       projection={"label": "入力 👋" * (COMPRESSION_THRESHOLD if large else 1)})
    wire = encode_message(message)
    calls = []

    def counted_encode(value):
        if value.get("messageId") == message["messageId"]:
            calls.append(value)
        return encode_message(value)

    monkeypatch.setattr(broker_module, "encode_message", counted_encode)
    with entered(host, *views):
        host.send_text(wire)
        for view in views:
            received = view.receive_text()
            assert received == wire
            assert decode_message(received) == message
        assert calls == [message]


class RecordingSocket:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.sent = []

    async def send_text(self, text):
        if self.fail:
            raise RuntimeError("socket closed")
        self.sent.append(text)


def session_with_views(*sockets):
    session = _Session("test-session", "test-credential", "http://127.0.0.1:8001",
                       datetime.now(timezone.utc) + timedelta(hours=1))
    session.views = {i: _Connection(i, "view", sock)
                     for i, sock in enumerate(sockets, start=1)}
    return session


def test_failed_recipient_does_not_stop_broadcast_or_reencode(monkeypatch):
    first, failed, last = RecordingSocket(), RecordingSocket(fail=True), RecordingSocket()
    session = session_with_views(first, failed, last)
    session.pending_requests["pending"] = (2, {"type": "validateRequest", "target": {}})
    session.pending_snapshot_requests.append((2, {}))
    message = {"type": "ack", "label": "入力 👋"}
    calls = []

    def counted_encode(value):
        calls.append(value)
        return encode_message(value)

    monkeypatch.setattr(broker_module, "encode_message", counted_encode)
    asyncio.run(LocalBrowserBroker()._broadcast_to_views(session, message))
    assert calls == [message]
    assert first.sent == last.sent == [encode_message(message)]
    assert first.sent[0] is last.sent[0]
    assert list(session.views) == [1, 3]
    assert session.pending_requests == {}
    assert not session.pending_snapshot_requests


def test_encoding_failure_keeps_send_failure_cleanup(monkeypatch):
    sockets = [RecordingSocket(), RecordingSocket()]
    session = session_with_views(*sockets)
    host = RecordingSocket()
    session.host = _Connection(0, "host", host)
    session.view_control = True
    session.last_view_state = "attached"
    session.pending_requests["pending"] = (1, {"type": "validateRequest", "target": target()})
    session.pending_snapshot_requests.append((1, {}))
    message = {"type": "ack"}
    calls = []

    def refuse_encoding(value):
        if value is message:
            calls.append(value)
            raise ValueError("wire message budget exceeded")
        return encode_message(value)

    monkeypatch.setattr(broker_module, "encode_message", refuse_encoding)
    asyncio.run(LocalBrowserBroker()._broadcast_to_views(session, message))
    assert calls == [message]
    assert not session.views
    assert not session.pending_requests
    assert not session.pending_snapshot_requests
    assert all(not socket.sent for socket in sockets)
    notices = [decode_message(text) for text in host.sent]
    assert [notice["type"] for notice in notices] == [
        "viewState", "validateCancel", "viewState", "viewState",
    ]
    assert notices[0]["viewId"] == "1" and notices[0]["connected"] is False
    assert notices[1]["requestId"] == "pending" and notices[1]["target"] == target()
    assert notices[2]["viewId"] == "2" and notices[2]["connected"] is False
    assert notices[3]["state"] == "detached"


def test_empty_broadcast_does_not_encode(monkeypatch):
    def unexpected_encode(message):
        pytest.fail("No recipients should mean no encoding work")

    monkeypatch.setattr(broker_module, "encode_message", unexpected_encode)
    asyncio.run(LocalBrowserBroker()._broadcast_to_views(session_with_views(), {"type": "ack"}))
