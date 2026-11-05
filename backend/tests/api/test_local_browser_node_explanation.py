"""Source explanations remain bounded presentation private to their requesting view."""
from copy import deepcopy

import pytest

from services.ide.local_browser_protocol import is_local_browser_message
from tests.api.test_local_browser_broker import (
    client, env, mint, bootstrap, cookie_key_of, connect_view, connect_control_host,
    entered, envelope, target,
)


def explanation():
    return dict(title="Score candidates", purpose="Compare candidate features with question features.",
                inputs=["Candidate features", "Question features"], outputs=["Raw scores"],
                steps=["Pad and batch candidates.", "Batch queries and call the scoring head."],
                uncertainties=["Tensor shapes have not been recorded."],
                citations=[dict(id="helper", file="timejev/asymmetric_video_memory.py", sourceDigest="a" * 64,
                                startLine=247, endLine=251)])


def explanation_state(status="ready"):
    value = dict(requestId="request", nodeId="python@Model#0", revision="revision", status=status)
    if status == "ready":
        value["explanation"] = explanation()
    return value


def result_message(state=None, session="session", view="view"):
    return envelope("nodeExplanation", session, 1, viewId=view, target=target(),
                    state=explanation_state() if state is None else state)


def request_message(action="retry", session="session", view="view"):
    return envelope("nodeExplanationRequest", session, 2, viewId=view, target=target(),
                    nodeId="python@Model#0", requestId="request", action=action,
                    **({"citationId": "helper"} if action == "reveal" else {}))


@pytest.mark.parametrize("status", ["queued", "loading", "ready", "unavailable"])
def test_explanation_lifecycle_has_a_result_only_when_ready(status):
    state = explanation_state(status)
    assert is_local_browser_message(result_message(state), "session")
    if status == "ready":
        del state["explanation"]
    else:
        state["explanation"] = explanation()
    assert not is_local_browser_message(result_message(state), "session")


@pytest.mark.parametrize("field,count,length", [("inputs", 6, 160), ("outputs", 4, 160),
                                               ("steps", 6, 320), ("uncertainties", 4, 240)])
def test_explanation_list_limits_and_plain_text(field, count, length):
    valid = explanation_state()
    valid["explanation"][field] = ["x" * length] * count
    assert is_local_browser_message(result_message(valid))
    for value in [None, "text", ["x"] * (count + 1), ["x" * (length + 1)], [""], [" padded"], ["x\ny"], [False]]:
        state = deepcopy(valid)
        state["explanation"][field] = value
        assert not is_local_browser_message(result_message(state)), value


@pytest.mark.parametrize("field,maximum", [("title", 80), ("purpose", 320)])
def test_explanation_text_uses_the_same_utf16_budget_as_the_canvas(field, maximum):
    state = explanation_state()
    state["explanation"][field] = "🙂" * (maximum // 2)
    assert is_local_browser_message(result_message(state))
    for value in ["🙂" * (maximum // 2 + 1), "x" * (maximum + 1), "", "padded ", "\ufeffpadded", "x\0y", "x\ty", "x\u007fy"]:
        state["explanation"][field] = value
        assert not is_local_browser_message(result_message(state)), value


@pytest.mark.parametrize("update", [
    {"file": "/model.py"}, {"file": "C:\\model.py"}, {"file": "\\\\server\\model.py"},
    {"file": "../model.py"}, {"file": "pkg/../../model.py"}, {"file": "pkg\\..\\model.py"},
    {"file": "file:///model.py"}, {"file": "https://example.com/model.py"}, {"file": "bad\n.py"},
    {"id": ""}, {"id": "i" * 129}, {"sourceDigest": "a" * 63}, {"sourceDigest": "A" * 64},
    {"startLine": 0}, {"startLine": True}, {"startLine": 1.5}, {"endLine": 246},
    {"endLine": 9007199254740992}, {"command": "open model.py"},
])
def test_explanation_citations_reject_nonlocal_paths_bad_spans_and_unknown_fields(update):
    state = explanation_state()
    state["explanation"]["citations"][0].update(update)
    assert not is_local_browser_message(result_message(state))


def test_explanations_need_distinct_citations_and_cannot_add_execution_fields():
    state = explanation_state()
    first = state["explanation"]["citations"][0]
    for value in [[], [first, first], [dict(first, id=str(index)) for index in range(9)], None]:
        state["explanation"]["citations"] = value
        assert not is_local_browser_message(result_message(state))
    state = explanation_state()
    state["explanation"]["command"] = "run model"
    assert not is_local_browser_message(result_message(state))
    state = explanation_state()
    state["unsupported"] = True
    assert not is_local_browser_message(result_message(state))
    for status in [None, 1, {}, ["ready"], "finished"]:
        assert not is_local_browser_message(result_message(dict(explanation_state(), status=status)))


@pytest.mark.parametrize("field,maximum", [("requestId", 128), ("nodeId", 1024), ("revision", 256), ("detail", 320)])
def test_explanation_state_identifiers_and_detail_are_bounded(field, maximum):
    state = explanation_state()
    state[field] = "x" * maximum
    assert is_local_browser_message(result_message(state))
    for value in ["", None, " ", "x\ny", "x" * (maximum + 1)]:
        state[field] = value
        assert not is_local_browser_message(result_message(state))


def test_explanation_requests_are_exact_and_workspace_bound():
    assert is_local_browser_message(request_message(), "session")
    assert is_local_browser_message(request_message("reveal"), "session")
    assert not is_local_browser_message(request_message(), "other-session")
    for update in [{"action": "execute"}, {"action": "reveal"}, {"citationId": "helper"}, {"viewId": ""},
                   {"nodeId": "n" * 1025}, {"requestId": "r" * 129}, {"command": "run model"}, {"target": None}]:
        assert not is_local_browser_message(dict(request_message(), **update)), update
    for citation_id in ["", None, "c" * 129]:
        assert not is_local_browser_message(dict(request_message("reveal"), citationId=citation_id))


@pytest.mark.parametrize("action", ["retry", "reveal"])
def test_explanation_requests_bind_to_the_socket_and_results_stay_private(client, action):
    session = mint(client)
    sid = session["workspaceSessionId"]
    host = connect_control_host(client, session)
    first, first_connected, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    second, second_connected, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=60)
    with entered(first, second, host):
        first_id, second_id = first_connected["session"]["viewId"], second_connected["session"]["viewId"]
        assert host.receive_json()["viewId"] == first_id
        assert host.receive_json()["viewId"] == second_id
        request = request_message(action, sid, first_id)
        second.send_json(request)
        assert second.receive_json()["code"] == "INVALID_MESSAGE"
        first.send_json(dict(request, workspaceSessionId="another-workspace"))
        assert first.receive_json()["code"] == "SESSION_MISMATCH"
        host.send_json(request)
        assert host.receive_json()["code"] == "INVALID_MESSAGE"
        first.send_json(request)
        assert host.receive_json() == request
        reply = result_message(session=sid, view=first_id)
        first.send_json(reply)
        assert first.receive_json()["code"] == "INVALID_MESSAGE"
        host.send_json(reply)
        assert first.receive_json() == reply
        # A second targeted reply proves the first explanation never entered the other tab's queue.
        second_reply = dict(reply, viewId=second_id, messageId="second-result")
        host.send_json(second_reply)
        assert second.receive_json() == second_reply
