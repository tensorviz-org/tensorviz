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


def source_documentation():
    return dict(status="resolved", qualifiedName="UncommonBlock.forward", kind="method",
                signature="def forward(self, x, /, *, normalize=False):",
                docstring="Compare the input representations.\n\nArgs:\n    x: Input features.", truncated=False,
                citation=explanation()["citations"][0])


@pytest.mark.parametrize("status", ["queued", "loading", "ready", "unavailable"])
def test_source_documentation_does_not_require_an_agent_result(status):
    state = dict(explanation_state(status), documentation=source_documentation())
    assert is_local_browser_message(result_message(state))
    state["documentation"] = dict(status="unavailable", reason="ambiguous")
    assert is_local_browser_message(result_message(state))


def test_source_documentation_is_bounded_and_cannot_supply_execution_authority():
    docs = source_documentation()
    for update in [dict(signature="x" * 2049), dict(signature="def f():\npass"), dict(docstring="x" * 4097),
                   dict(docstring="text\x00"), dict(docstring=" padded"), dict(truncated="false"),
                   dict(kind="runtime"), dict(execute="model()"), dict(citation=dict(docs["citation"], file="/private.py"))]:
        assert not is_local_browser_message(result_message(dict(explanation_state("unavailable"), documentation=dict(docs, **update))))
    for invalid in [dict(status="unavailable", reason="guess"), dict(status="unavailable", reason="ambiguous", signature="invented")]:
        assert not is_local_browser_message(result_message(dict(explanation_state("unavailable"), documentation=invalid)))
    docs["docstring"] = "\u0085author text\u0085"
    assert is_local_browser_message(result_message(dict(explanation_state("unavailable"), documentation=docs)))


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


def model_result_message(state=None, session="session", view="view"):
    return envelope("modelExplanation", session, 3, viewId=view, target=target(),
                    state=dict(requestId="model-request", revision="revision", status="loading", completed=0, total=128)
                    if state is None else state)


def model_request_message(action="start", session="session", view="view"):
    return envelope("modelExplanationRequest", session, 4, viewId=view, target=target(),
                    requestId="model-request", action=action, nodeIds=["node"] if action == "start" else [])


def test_optional_terminology_is_bounded_data():
    state = explanation_state()
    concept = dict(term="VAE", explanation="A variational autoencoder learns a latent representation.")
    for concepts in [[], [concept], [dict(term="x" * 80, explanation="x" * 320)] * 4]:
        state["explanation"]["concepts"] = concepts
        assert is_local_browser_message(result_message(state))
    for concepts in [None, "text", [concept] * 5, [{}], [dict(concept, term="")],
                     [dict(concept, term="x" * 81)], [dict(concept, explanation="x" * 321)],
                     [dict(concept, explanation="line\nbreak")], [dict(concept, command="execute")]]:
        state["explanation"]["concepts"] = concepts
        assert not is_local_browser_message(result_message(state)), concepts


@pytest.mark.parametrize("status", ["loading", "ready", "unavailable", "cancelled"])
def test_model_explanation_progress_lifecycle(status):
    state = dict(model_result_message()["state"], status=status)
    assert is_local_browser_message(model_result_message(state), "session")
    assert is_local_browser_message(model_result_message(dict(state, completed=10000, total=10000, detail="x" * 320)))


@pytest.mark.parametrize("update", [
    dict(completed=-1), dict(completed=129), dict(completed=0.5), dict(completed=True),
    dict(completed=float("nan")), dict(total=10001), dict(total=-1), dict(total=float("inf")), dict(total=1.5),
    dict(requestId=""), dict(revision=""), dict(status="queued"), dict(status={}),
    dict(detail=""), dict(detail="x" * 321), dict(command="execute"),
])
def test_model_explanation_progress_rejects_invalid_counts_and_unknown_fields(update):
    state = dict(model_result_message()["state"], **update)
    assert not is_local_browser_message(model_result_message(state))


def test_model_explanation_requests_bind_bounded_unique_nodes_to_the_request():
    request = model_request_message()
    assert is_local_browser_message(request, "session")
    assert is_local_browser_message(dict(request, nodeIds=[f"node-{index}" for index in range(128)]))
    assert is_local_browser_message(model_request_message("cancel"))
    assert not is_local_browser_message(request, "other-session")
    for update in [dict(nodeIds=[]), dict(nodeIds=["node", "node"]), dict(nodeIds=[""]),
                   dict(nodeIds=["n" * 1025]), dict(nodeIds=[f"node-{index}" for index in range(129)]),
                   dict(nodeIds=[1]), dict(nodeIds=None), dict(action="cancel"), dict(action="execute"),
                   dict(viewId=""), dict(target=None), dict(requestId=""), dict(requestId="r" * 129), dict(command="execute")]:
        assert not is_local_browser_message(dict(request, **update)), update


@pytest.mark.parametrize("action", ["start", "cancel"])
def test_model_explanation_routing_is_private_and_socket_bound(client, action):
    session = mint(client)
    sid = session["workspaceSessionId"]
    host = connect_control_host(client, session)
    first, first_connected, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    second, second_connected, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=60)
    with entered(first, second, host):
        first_id, second_id = first_connected["session"]["viewId"], second_connected["session"]["viewId"]
        assert host.receive_json()["viewId"] == first_id
        assert host.receive_json()["viewId"] == second_id
        request = model_request_message(action, sid, first_id)
        second.send_json(request)
        assert second.receive_json()["code"] == "INVALID_MESSAGE"
        first.send_json(dict(request, workspaceSessionId="another-workspace"))
        assert first.receive_json()["code"] == "SESSION_MISMATCH"
        host.send_json(request)
        assert host.receive_json()["code"] == "INVALID_MESSAGE"
        first.send_json(request)
        assert host.receive_json() == request
        reply = model_result_message(session=sid, view=first_id)
        first.send_json(reply)
        assert first.receive_json()["code"] == "INVALID_MESSAGE"
        host.send_json(reply)
        assert first.receive_json() == reply
        second_reply = dict(reply, viewId=second_id, messageId="second-result")
        host.send_json(second_reply)
        assert second.receive_json() == second_reply
