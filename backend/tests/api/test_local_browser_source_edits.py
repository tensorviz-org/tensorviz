"""Source proposals use per-view routing; no view can accept or forge a reply."""
from tests.api.test_local_browser_broker import (
    client, env, mint, connect_host, bootstrap, cookie_key_of, connect_view, entered, request as envelope_request,
)


def request(session, request_id="edit-1", n=101):
    return envelope_request("sourceEditRequest", session, n, request_id,
               edit={"kind": "remove_layer", "nodeId": "act"})


def test_source_reply_cannot_be_forged_or_cross_wired_to_envelope_request(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    view, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    with entered(host, view):
        view.send_json(request(session))
        assert host.receive_json()["type"] == "sourceEditRequest"
        # Wrong reply type and wrong target must leave the pending edit intact.
        host.send_json(envelope_request("validateResult", session, 102, "edit-1", status="passed", detail="wrong channel"))
        wrong = envelope_request("sourceEditResult", session, 103, "edit-1", status="applied", detail="wrong target")
        wrong["target"]["documentVersion"] += 1
        host.send_json(wrong)
        forged = envelope_request("sourceEditResult", session, 104, "edit-1", status="applied", detail="forged")
        view.send_json(forged)
        assert view.receive_json()["code"] == "INVALID_MESSAGE"
        host.send_json(envelope_request("sourceEditResult", session, 105, "edit-1", status="cancelled", detail="Review dismissed."))
        result = view.receive_json()
        assert result["type"] == "sourceEditResult" and result["status"] == "cancelled"


def test_other_view_cannot_cancel_proposal_and_only_requester_gets_reply(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    view_a, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=50)
    view_b, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=60)
    with entered(host, view_a, view_b):
        view_a.send_json(request(session))
        assert host.receive_json()["requestId"] == "edit-1"
        view_b.send_json(envelope_request("sourceEditCancel", session, 102, "edit-1"))
        view_b.send_json(request(session, "edit-2", 103))
        assert host.receive_json()["requestId"] == "edit-2"
        host.send_json(envelope_request("sourceEditResult", session, 104, "edit-2", status="refused", detail="Another review is open."))
        assert view_b.receive_json()["requestId"] == "edit-2"
        host.send_json(envelope_request("sourceEditResult", session, 105, "edit-1", status="applied", detail="Applied in VS Code."))
        assert view_a.receive_json()["requestId"] == "edit-1"


def test_view_disconnect_cancels_its_source_review_on_the_host(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    view, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    with entered(host):
        view.send_json(request(session))
        assert host.receive_json()["type"] == "sourceEditRequest"
        view.__exit__(None, None, None)
        assert host.receive_json()["type"] == "sourceEditCancel"
