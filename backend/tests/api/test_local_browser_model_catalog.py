"""Project classes are source evidence and selection uses host-owned identities."""
import pytest

from services.ide.local_browser_protocol import is_local_browser_message
from tests.api.test_local_browser_broker import (
    client, env, mint, bootstrap, cookie_key_of, connect_view, connect_control_host,
    entered, envelope, target,
)


def catalog():
    return dict(status="ready", activeId="audio", models=[dict(id="audio", cls="AudioVAEV2",
                file="layers/audio_vae_v2.py", evidence="source", opened=False)])


def request(action="list", session="session", view="view"):
    return envelope("modelCatalogRequest", session, 1, viewId=view, target=target(), requestId="models",
                    action=action, **({"modelId": "audio"} if action == "select" else {}))


def result(value=None, session="session", view="view"):
    return envelope("modelCatalog", session, 2, viewId=view, target=target(), requestId="models",
                    catalog=catalog() if value is None else value)


@pytest.mark.parametrize("status", ["loading", "ready", "unavailable"])
def test_model_catalogue_has_explicit_bounded_states(status):
    assert is_local_browser_message(result(dict(catalog(), status=status)))
    assert is_local_browser_message(result(dict(status=status, models=[])))
    assert is_local_browser_message(result(dict(catalog(), detail="x" * 320, truncated=True)))
    value = catalog()
    value["models"][0].update(evidence="captured", opened=True)
    assert is_local_browser_message(result(value))


@pytest.mark.parametrize("update", [
    dict(id=""), dict(id="x" * 129), dict(cls=""), dict(cls="x" * 513), dict(file="x" * 4097),
    dict(file="/etc/passwd"), dict(file="../model.py"), dict(file="a/../model.py"), dict(file="a//model.py"),
    dict(file="./model.py"), dict(file="file:///model.py"), dict(file="C:\\model.py"), dict(file="model\n.py"),
    dict(opened="true"), dict(evidence="inferred"), dict(uri="open this"),
])
def test_model_catalogue_rows_reject_unbounded_or_unproven_execution_authority(update):
    value = catalog()
    value["models"][0].update(update)
    assert not is_local_browser_message(result(value))


def test_model_catalogue_limits_match_typescript_and_require_unique_host_ids():
    value = catalog()
    entry = value["models"][0]
    for update in [dict(models=[entry, entry]), dict(activeId="unknown"), dict(status={}), dict(status="captured"),
                   dict(detail="x" * 321), dict(truncated=1), dict(models=[entry] * 2001), dict(command="execute")]:
        assert not is_local_browser_message(result(dict(value, **update)))
    value["models"] = [dict(entry, id=str(index)) for index in range(2000)]
    del value["activeId"]
    assert is_local_browser_message(result(value))
    for field, maximum in [("id", 128), ("cls", 512), ("file", 4096)]:
        value = catalog()
        value.pop("activeId")
        value["models"][0][field] = "🙂" * (maximum // 2)
        assert is_local_browser_message(result(value))
        value["models"][0][field] += "🙂"
        assert not is_local_browser_message(result(value))


def test_model_catalogue_requests_are_exact_and_workspace_bound():
    assert is_local_browser_message(request(), "session")
    assert is_local_browser_message(request("select"), "session")
    assert not is_local_browser_message(request(), "other-session")
    for update in [dict(action="select"), dict(modelId="audio"), dict(action="select", modelId=""),
                   dict(action="select", modelId="x" * 129), dict(action="execute"), dict(viewId=""),
                   dict(target=None), dict(requestId=""), dict(file="/private/model.py"), dict(cls="Anything")]:
        assert not is_local_browser_message(dict(request(), **update))


@pytest.mark.parametrize("action", ["list", "select"])
def test_model_catalogues_route_privately_and_selection_cannot_spoof_another_view(client, action):
    session = mint(client)
    sid = session["workspaceSessionId"]
    host = connect_control_host(client, session)
    first, first_connected, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    second, second_connected, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=60)
    with entered(first, second, host):
        first_id, second_id = first_connected["session"]["viewId"], second_connected["session"]["viewId"]
        assert host.receive_json()["viewId"] == first_id
        assert host.receive_json()["viewId"] == second_id
        message = request(action, sid, first_id)
        second.send_json(message)
        assert second.receive_json()["code"] == "INVALID_MESSAGE"
        host.send_json(message)
        assert host.receive_json()["code"] == "INVALID_MESSAGE"
        first.send_json(message)
        assert host.receive_json() == message
        reply = result(session=sid, view=first_id)
        first.send_json(reply)
        assert first.receive_json()["code"] == "INVALID_MESSAGE"
        host.send_json(reply)
        assert first.receive_json() == reply
        second_reply = dict(reply, viewId=second_id, messageId="second-result")
        host.send_json(second_reply)
        assert second.receive_json() == second_reply
