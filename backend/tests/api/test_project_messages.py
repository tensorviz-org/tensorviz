"""The five project-plane messages, validated the way every other message is.

The rule these guard: the project plane and the class plane must not be able to
impersonate each other. A `projectIndex` carrying a class `target`, or a class
message carrying a project identity, would slip past a check written for the other
and end up acted on with the wrong freshness rules.
"""


import pytest


from services.ide import local_browser_protocol as protocol  # noqa: E402

SESSION = "session-1"
DIGEST = "b" * 64


def project(**overrides):
    base = {
        "workspaceFolder": "/w",
        "projectRoot": "",
        "experimentId": "exp:1",
        "indexDigest": DIGEST,
        "indexRevision": 1,
    }
    base.update(overrides)
    return base


def envelope(message_type, **fields):
    return {
        "type": message_type,
        "protocolVersion": 1,
        "workspaceSessionId": SESSION,
        "messageId": "m-1",
        **fields,
    }


def valid(message):
    return protocol.is_local_browser_message(message, SESSION)


# ── host-originated, ordered ────────────────────────────────────────────────────


def test_project_index_is_accepted():
    assert valid(
        envelope("projectIndex", hostSequence=1, project=project(), index={}, truncated=False)
    )


def test_project_document_is_accepted():
    assert valid(
        envelope(
            "projectDocument",
            hostSequence=2,
            project=project(),
            document={},
            spanmap={},
            expanded=["net"],
        )
    )


@pytest.mark.parametrize("message_type", ["projectIndex", "projectDocument"])
def test_ordered_host_state_requires_a_sequence(message_type):
    """Without it the page can render a document against an index it never saw."""
    payload = envelope(
        message_type,
        project=project(),
        index={},
        truncated=False,
        document={},
        spanmap={},
        expanded=[],
    )
    assert not valid(payload)


def test_a_project_message_may_not_carry_a_class_target():
    payload = envelope(
        "projectIndex",
        hostSequence=1,
        project=project(),
        index={},
        truncated=False,
        target={
            "workspaceFolder": "/w",
            "file": "a.py",
            "cls": "Net",
            "documentVersion": 1,
            "sourceDigest": "a" * 64,
            "backendRevision": 0,
        },
    )
    assert not valid(payload)


def test_truncation_must_be_stated_explicitly():
    assert not valid(
        envelope("projectIndex", hostSequence=1, project=project(), index={})
    )


# ── view-originated, unordered ──────────────────────────────────────────────────


def test_project_expand_is_accepted():
    assert valid(
        envelope("projectExpand", project=project(), nodeId="net", expanded=True)
    )


def test_project_focus_is_accepted():
    assert valid(envelope("projectFocus", project=project(), experimentId="exp:2"))


def test_clearing_the_focus_is_a_real_request():
    assert valid(envelope("projectFocus", project=project(), experimentId=""))


@pytest.mark.parametrize("message_type", ["projectExpand", "projectFocus"])
def test_view_traffic_never_carries_a_host_sequence(message_type):
    """Only ordered host state is sequenced; a view claiming one is malformed."""
    payload = envelope(
        message_type,
        hostSequence=1,
        project=project(),
        nodeId="net",
        expanded=True,
        experimentId="e",
    )
    assert not valid(payload)


def test_expand_requires_a_node_and_a_boolean():
    assert not valid(envelope("projectExpand", project=project(), expanded=True))
    assert not valid(envelope("projectExpand", project=project(), nodeId="net"))
    assert not valid(
        envelope("projectExpand", project=project(), nodeId="", expanded=True)
    )


# ── cross-session and malformed identity ────────────────────────────────────────


def test_a_message_from_another_session_is_refused():
    payload = envelope("projectExpand", project=project(), nodeId="net", expanded=True)
    payload["workspaceSessionId"] = "someone-else"
    assert not valid(payload)


def test_a_malformed_project_identity_is_refused():
    assert not valid(
        envelope(
            "projectExpand",
            project=project(indexDigest="short"),
            nodeId="net",
            expanded=True,
        )
    )


def test_an_expanded_list_with_duplicates_is_refused():
    assert not valid(
        envelope(
            "projectDocument",
            hostSequence=1,
            project=project(),
            document={},
            spanmap={},
            expanded=["net", "net"],
        )
    )


def test_an_unbounded_expanded_list_is_refused():
    assert not valid(
        envelope(
            "projectDocument",
            hostSequence=1,
            project=project(),
            document={},
            spanmap={},
            expanded=[f"n{i}" for i in range(5000)],
        )
    )


def test_expansion_exact_ceiling_and_declaration_utf16_bounds():
    assert valid(envelope('projectDocument', project=project(), hostSequence=0,
        document={}, spanmap={}, expanded=[f'n{i}' for i in range(4096)]))
    assert not valid(envelope('projectDocument', project=project(), hostSequence=0,
        document={}, spanmap={}, expanded=[f'n{i}' for i in range(4097)]))
    assert valid(envelope('projectDeclare', project=project(), declaration='experiment', args=['😀' * 256]))
    assert not valid(envelope('projectDeclare', project=project(), declaration='experiment', args=['😀' * 257]))


def test_digest_must_not_accept_a_terminal_newline():
    assert not valid(envelope('projectFocus', project=project(indexDigest=DIGEST + '\n'), experimentId=''))
