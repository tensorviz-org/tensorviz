from tests.api.test_local_browser_broker import (
    client, env, mint, connect_host, bootstrap, cookie_key_of, connect_view, entered, envelope, target,
)


def test_opening_state_is_host_owned_and_setup_requests_reach_only_host(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    a, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=50)
    b, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=60)
    state = dict(id="open-1", target=target(), phase="needs_setup", detail="Choose the intended inputs.", hasCapture=False)
    with entered(host, a, b):
        a.send_json(envelope("graphOpening", session["workspaceSessionId"], 101, hostSequence=1, state=state))
        assert a.receive_json()["code"] == "INVALID_MESSAGE"
        host.send_json(envelope("graphOpening", session["workspaceSessionId"], 102, hostSequence=1, state=state))
        assert a.receive_json()["state"] == b.receive_json()["state"] == state
        a.send_json(envelope("graphSetupRequest", session["workspaceSessionId"], 103, target=target(),
                             openingId="open-1", requestId="form-1", action={"kind": "retry"}))
        assert host.receive_json()["action"] == {"kind": "retry"}


def test_opening_state_can_be_restored_before_first_projection(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    view, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    opening = dict(id="open-1", target=target(), phase="needs_setup", detail="Complete setup.", hasCapture=False)
    with entered(host, view):
        view.send_json(envelope("snapshotRequest", session["workspaceSessionId"], 150, role="view", lastHostSequence=0))
        assert host.receive_json()["type"] == "snapshotRequest"
        state = dict(activeTarget=None, projection=None, traced=False, greyed=False,
                     syncState=None, connectionState="connected", opening=opening)
        host.send_json(envelope("snapshot", session["workspaceSessionId"], 151, role="host", hostSequence=1, state=state))
        assert view.receive_json()["state"]["opening"] == opening


def test_model_environment_creation_capability_and_request_cross_the_broker(client):
    session = mint(client)
    host, _ = connect_host(client, session)
    view, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    opening = dict(id="open-create", target=target(), phase="needs_setup", detail="Create model Python.", hasCapture=False,
                   environment=dict(reason="unavailable", detail="No model Python is available.", candidates=[], canCreate=True))
    with entered(host, view):
        host.send_json(envelope("graphOpening", session["workspaceSessionId"], 200, hostSequence=1, state=opening))
        assert view.receive_json()["state"] == opening
        view.send_json(envelope("graphSetupRequest", session["workspaceSessionId"], 201, target=target(),
                                openingId="open-create", requestId="create-1", action={"kind": "create_environment"}))
        assert host.receive_json()["action"] == {"kind": "create_environment"}
        state = dict(activeTarget=None, projection=None, traced=False, greyed=False,
                     syncState=None, connectionState="connected", opening=opening)
        view.send_json(envelope("snapshotRequest", session["workspaceSessionId"], 202, role="view", lastHostSequence=1))
        assert host.receive_json()["type"] == "snapshotRequest"
        host.send_json(envelope("snapshot", session["workspaceSessionId"], 203, role="host", hostSequence=2, state=state))
        assert view.receive_json()["state"]["opening"]["environment"]["canCreate"] is True
        view.send_json(envelope("graphSetupRequest", session["workspaceSessionId"], 204, target=target(),
                                openingId="open-create", requestId="create-forged", action={"kind": "create_environment", "interpreter": "/foreign/python"}))
        assert view.receive_json()["code"] == "INVALID_MESSAGE"


def test_execution_review_replays_and_run_reaches_the_host(client):
    from tests.api.test_execution_review_protocol import state as execution_state
    session = mint(client)
    host, _ = connect_host(client, session)
    view, _, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    opening = dict(id="open-run", target=target(), phase="awaiting_execution", detail="Review execution.",
                   hasCapture=False, execution=execution_state())
    with entered(host, view):
        host.send_json(envelope("graphOpening", session["workspaceSessionId"], 300, hostSequence=1, state=opening))
        assert view.receive_json()["state"] == opening
        view.send_json(envelope("snapshotRequest", session["workspaceSessionId"], 301, role="view", lastHostSequence=1))
        assert host.receive_json()["type"] == "snapshotRequest"
        snapshot = dict(activeTarget=None, projection=None, traced=False, greyed=False,
                        syncState=None, connectionState="connected", opening=opening)
        host.send_json(envelope("snapshot", session["workspaceSessionId"], 302, role="host", hostSequence=2, state=snapshot))
        assert view.receive_json()["state"]["opening"]["execution"]["phase"] == "review"
        action = dict(kind="execution", reviewId="review1", action="run", device="cuda:0")
        view.send_json(envelope("graphSetupRequest", session["workspaceSessionId"], 303, target=target(),
                                openingId="open-run", requestId="run-1", action=action))
        assert host.receive_json()["action"] == action
