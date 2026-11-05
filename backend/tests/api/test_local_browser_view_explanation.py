"""View explanations are bounded, source-linked presentation for exact visible nodes."""
from copy import deepcopy

import pytest

from services.ide.local_browser_protocol import is_local_browser_message
from tests.api.test_local_browser_node_explanation import model_result_message, explanation
from tests.api.test_local_browser_broker import (
    client, env, mint, bootstrap, cookie_key_of, connect_view, connect_control_host, entered,
)


def view_explanation():
    citation = explanation()["citations"][0]
    return dict(summary="Decode features, using a sample-rate bucket to condition decoder blocks.",
                nodeIds=["features", "sample-rate", "decoder", "waveform"],
                connections=[dict(sourceNodeId="features", targetNodeId="decoder", label="Learned audio features",
                                  detail="The decoder receives these features as x.", citations=[citation])],
                callouts=[dict(title="Why this branch exists", body="The bucket selects decoder conditioning.",
                               nodeIds=["sample-rate", "decoder"], citations=[citation])])


def result(value=None, **kwargs):
    return model_result_message(dict(requestId="request", revision="revision", status="ready", completed=4,
                                     total=4, view=view_explanation() if value is None else value), **kwargs)


@pytest.mark.parametrize("status", ["loading", "ready", "unavailable", "cancelled"])
def test_view_explanation_retains_the_whole_snapshot_for_partial_results(status):
    message = result()
    message["state"].update(status=status, completed=4 if status == "ready" else 1)
    assert is_local_browser_message(message, "session")
    assert not is_local_browser_message(message, "other-session")
    message["state"]["total"] = 5
    assert not is_local_browser_message(message)
    del message["state"]["view"]
    assert is_local_browser_message(message)


def test_view_explanation_rejects_hidden_duplicate_and_unbounded_references():
    value = view_explanation()
    connection, callout = value["connections"][0], value["callouts"][0]
    for update in [dict(nodeIds=[]), dict(nodeIds=["features", "features"]), dict(nodeIds=[""]),
                   dict(nodeIds=["x" * 1025]), dict(nodeIds=[f"node-{i}" for i in range(129)]),
                   dict(connections=[dict(connection, sourceNodeId="hidden")]),
                   dict(connections=[dict(connection, targetNodeId="hidden")]),
                   dict(connections=[connection, connection]),
                   dict(callouts=[dict(callout, nodeIds=["hidden"])]),
                   dict(callouts=[dict(callout, nodeIds=[])]),
                   dict(callouts=[dict(callout, nodeIds=["decoder", "decoder"])]),
                   dict(callouts=[callout] * 7), dict(execute=True), dict(callouts=None), dict(connections=None)]:
        assert not is_local_browser_message(result(dict(value, **update))), update
    node_ids = [f"node-{i}" for i in range(128)]
    value["nodeIds"] = node_ids
    value["connections"] = [dict(connection, sourceNodeId=node_id, targetNodeId=node_ids[(i + 1) % 128])
                            for i, node_id in enumerate(node_ids)]
    value["callouts"] = [dict(callout, nodeIds=node_ids)] * 6
    message = result(value)
    message["state"].update(completed=128, total=128)
    assert is_local_browser_message(message)
    value["connections"].append(connection)
    assert not is_local_browser_message(message)


@pytest.mark.parametrize("field,maximum,section", [
    ("summary", 640, None), ("label", 120, "connections"), ("detail", 320, "connections"),
    ("title", 80, "callouts"), ("body", 480, "callouts"),
])
def test_view_explanation_plain_text_bounds_match_typescript(field, maximum, section):
    for text, valid in [("x" * maximum, True), ("🙂" * (maximum // 2), True),
                        ("x" * (maximum + 1), False), ("🙂" * (maximum // 2 + 1), False),
                        ("", False), (" padded", False), ("line\nbreak", False), ("text\x00", False), (None, False)]:
        value = view_explanation()
        if section is None:
            value[field] = text
        else:
            value[section][0][field] = text
        assert is_local_browser_message(result(value)) == valid


@pytest.mark.parametrize("section", ["connections", "callouts"])
def test_view_explanation_requires_distinct_safe_host_citations_and_exact_fields(section):
    citation = explanation()["citations"][0]
    for citations in [[], [citation, citation], [dict(citation, file="/private/model.py")],
                      [dict(citation, startLine=0)], [dict(citation, execute=True)],
                      [dict(citation, id=str(i)) for i in range(9)]]:
        value = view_explanation()
        value[section][0]["citations"] = citations
        assert not is_local_browser_message(result(value))
    value = view_explanation()
    value[section][0]["citations"] = [dict(citation, id=str(i)) for i in range(8)]
    assert is_local_browser_message(result(value))
    value[section][0]["command"] = "run model"
    assert not is_local_browser_message(result(value))


def test_view_explanations_use_the_existing_private_host_result_channel(client):
    session = mint(client)
    sid = session["workspaceSessionId"]
    host = connect_control_host(client, session)
    first, first_connected, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    second, second_connected, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=60)
    with entered(first, second, host):
        first_id, second_id = first_connected["session"]["viewId"], second_connected["session"]["viewId"]
        assert host.receive_json()["viewId"] == first_id
        assert host.receive_json()["viewId"] == second_id
        message = result(session=sid, view=first_id)
        first.send_json(message)
        assert first.receive_json()["code"] == "INVALID_MESSAGE"
        host.send_json(message)
        assert first.receive_json() == message
        second_message = deepcopy(message)
        second_message.update(viewId=second_id, messageId="second-result")
        host.send_json(second_message)
        assert second.receive_json() == second_message
