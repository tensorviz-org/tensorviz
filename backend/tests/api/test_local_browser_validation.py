"""Manual validation belongs to the requesting view and can never forge an source-edit result."""
from tests.api.test_local_browser_broker import (
    client, env, mint, connect_host, bootstrap, cookie_key_of, connect_view, entered, request, envelope, target,
)


def test_validation_routes_only_to_requester_and_rejects_wrong_reply_kinds(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    a, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=50)
    b, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=60)
    with entered(host, a, b):
        a.send_json(request("validateRequest", session, 101, "v1"))
        assert host.receive_json()["type"] == "validateRequest"
        b.send_json(request("validateCancel", session, 102, "v1"))
        b.send_json(request("validateRequest", session, 103, "v2"))
        assert host.receive_json()["requestId"] == "v2"
        host.send_json(request("sourceEditResult", session, 105, "v1", status="applied", detail="wrong kind"))
        wrong = request("validateResult", session, 106, "v1", status="passed", detail="wrong source")
        wrong["target"]["documentVersion"] += 1
        host.send_json(wrong)
        b.send_json(request("validateResult", session, 107, "v1", status="passed", detail="forged"))
        assert b.receive_json()["code"] == "INVALID_MESSAGE"
        host.send_json(request("validateResult", session, 108, "v2", status="refused", detail="busy"))
        assert b.receive_json()["requestId"] == "v2"
        host.send_json(request("validateResult", session, 109, "v1", status="passed", detail="checked"))
        assert a.receive_json()["status"] == "passed"


def test_structured_failure_reaches_live_views_and_reconnect_snapshots(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    view, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    opening = dict(id="opening", target=target(), phase="failed", hasCapture=True,
                   detail="Incompatible shape.", failure=dict(kind="execution"))
    with entered(host, view):
        host.send_json(envelope("graphOpening", session["workspaceSessionId"], 100, hostSequence=1, state=opening))
        assert view.receive_json()["state"] == opening
        view.send_json(request("validateRequest", session, 101, "v1"))
        assert host.receive_json()["type"] == "validateRequest"
        host.send_json(request("validateResult", session, 102, "v1", status="failed", detail="Shape mismatch.", failure=opening["failure"]))
        assert view.receive_json()["failure"] == opening["failure"]
        view.send_json(envelope("snapshotRequest", session["workspaceSessionId"], 103, role="view", lastHostSequence=1))
        assert host.receive_json()["type"] == "snapshotRequest"
        state = dict(activeTarget=target(), projection={"revision": 7}, traced=True, greyed=False,
                     syncState="runtime-traced", connectionState="connected", opening=opening)
        host.send_json(envelope("snapshot", session["workspaceSessionId"], 104, role="host", hostSequence=2, state=state))
        assert view.receive_json()["state"]["opening"] == opening


def test_validation_disconnect_cancels_and_payload_cannot_supply_code(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    view, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    with entered(host):
        view.send_json(request("validateRequest", session, 101, "v1", source="print(1)"))
        assert view.receive_json()["code"] == "INVALID_MESSAGE"
        view.send_json(request("validateRequest", session, 102, "v1"))
        assert host.receive_json()["type"] == "validateRequest"
        view.__exit__(None, None, None)
        assert host.receive_json()["type"] == "validateCancel"


def test_mcp_validation_feedback_is_host_owned_and_reaches_both_views(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    a, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=50)
    b, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=60)
    def feedback(n, **fields):
        value = request("validationState", session, n, "unused", runId="run-1", status="running", detail="Tracing.")
        value.pop("requestId")
        value.update(fields)
        return value
    with entered(host, a, b):
        a.send_json(feedback(101))
        assert a.receive_json()["code"] == "INVALID_MESSAGE"
        host.send_json(feedback(102))
        assert a.receive_json()["status"] == b.receive_json()["status"] == "running"
        host.send_json(feedback(103, status="failed", projectionRevision=7))
        assert a.receive_json()["projectionRevision"] == b.receive_json()["projectionRevision"] == 7


def test_reconnect_snapshot_restores_latest_failure_without_borrowing_capture_revision(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    view, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    with entered(host, view):
        view.send_json(envelope('snapshotRequest', session['workspaceSessionId'], 150, role='view', lastHostSequence=0))
        assert host.receive_json()['type'] == 'snapshotRequest'
        summary = dict(runId='latest-timeout', target=target(), status='failed', detail='Timed out. Earlier capture retained.')
        state = dict(activeTarget=target(), projection={'revision': 7}, traced=True, greyed=False,
                     syncState='runtime-traced', connectionState='connected', validation=summary)
        host.send_json(envelope('snapshot', session['workspaceSessionId'], 151, role='host', hostSequence=1, state=state))
        restored = view.receive_json()['state']
        assert restored['validation'] == summary
        assert restored['projection']['revision'] == 7
        assert 'projectionRevision' not in restored['validation']
