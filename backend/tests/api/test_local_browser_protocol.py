"""Behavioral checks for the external-browser broker protocol."""


import pytest

from services.ide.local_browser_protocol import (
    LOCAL_BROWSER_ERROR_CODES,
    LOCAL_BROWSER_MESSAGE_TYPES,
    LOCAL_BROWSER_CAPTURE_EXPORT_MAX_BYTES,
    LOCAL_BROWSER_PROTOCOL_VERSION,
    LOCAL_BROWSER_ROLES,
    is_local_browser_message,
    local_browser_fixture_corpus,
)


@pytest.fixture(scope="module")
def corpus():
    return local_browser_fixture_corpus()


def test_version_roles_messages_and_errors_match_the_frozen_contract():
    assert LOCAL_BROWSER_PROTOCOL_VERSION == 1
    assert LOCAL_BROWSER_ROLES == {"host", "view"}
    assert LOCAL_BROWSER_MESSAGE_TYPES == {
        "graphReviewRequest", "graphReviewState", "graphReviewFocus", "graphReviewApplied",
        "graphOpening", "graphSetupRequest",
        "codeFocusRequest", "codeFocusState",
        "agentRequest", "agentState", "nodeExplanation", "nodeExplanationRequest", "modelExplanation", "modelExplanationRequest",
        "viewState", "viewSelection", "focusView", "viewFocusResult",
        "connected",
        "hostHandshake",
        "viewHandshake",
        "snapshotRequest",
        "snapshot",
        "activeTarget",
        "projection",
        # The project plane. Added alongside the class plane rather than replacing
        # it: a project identity and a class identity are different types with
        # different freshness rules, and these four never carry a class target.
        "projectIndex",
        "projectDocument",
        "projectExpand",
        "projectFocus",
        "projectDeclare",
        "connectionState",
        "viewState",
        "syncState",
        "staleState",
        "revealNode",
        "focusNode",
        "sourceEditRequest",
        "sourceEditCancel",
        "sourceEditResult",
        "sourceEditReview", "candidateRequest", "candidateResult",
        "captureInspectRequest", "captureInspectCancel", "captureInspectResult",
        "captureExportRequest", "captureExportCancel", "captureExportResult",
        "validateRequest",
        "validateCancel",
        "validateResult",
        "validationState",
        "ack",
        "error",
    }
    assert LOCAL_BROWSER_ERROR_CODES == {
        "PROTOCOL_VERSION_MISMATCH",
        "SESSION_MISMATCH",
        "INVALID_ORIGIN",
        "INVALID_CREDENTIALS",
        "CREDENTIAL_EXPIRED",
        "HOST_ALREADY_CONNECTED",
        "STALE_TARGET",
        "UNKNOWN_MESSAGE",
        "INVALID_MESSAGE",
    }


def test_accepts_every_canonical_valid_message(corpus):
    expected_workspace_session_id, valid, _ = corpus
    assert valid
    for message in valid:
        assert is_local_browser_message(message, expected_workspace_session_id) is True, message


def test_portable_transfer_budget_counts_utf8_not_characters(corpus):
    session, valid, _ = corpus
    message = next(item for item in valid if item['type'] == 'captureExportResult')
    oversized = {**message, 'capture': {'version': 1, 'sourceNames': 'anonymized', 'capture': {'label': '界' * (LOCAL_BROWSER_CAPTURE_EXPORT_MAX_BYTES // 3 + 1)}}}
    assert is_local_browser_message(oversized, session) is False


def test_refuses_every_canonical_invalid_message(corpus):
    expected_workspace_session_id, _, invalid = corpus
    assert invalid
    for message in invalid:
        assert is_local_browser_message(message, expected_workspace_session_id) is False, message


def test_view_selection_carries_code_focus_as_an_optional_boolean(corpus):
    session, valid, _ = corpus
    selection = next(item for item in valid if item['type'] == 'viewSelection' and 'tooLarge' not in item and 'follow' not in item)
    overflow = next(item for item in valid if item['type'] == 'viewSelection' and item.get('tooLarge') is True and 'follow' not in item)
    focus = next(item for item in valid if item['type'] == 'focusView')
    for message in (selection, overflow):
        assert is_local_browser_message(message, session) is True  # views that send no Code focus
        for follow in (False, True):
            assert is_local_browser_message({**message, 'follow': follow}, session) is True
        # 0 and 1 compare equal to the booleans in Python but are not JSON booleans.
        for follow in (None, 0, 1, 'edit', [True]):
            assert is_local_browser_message({**message, 'follow': follow}, session) is False
    assert is_local_browser_message({**focus, 'follow': True}, session) is False


def test_target_identity_requires_workspace_folder_and_raw_digest():
    target = {
        "workspaceFolder": "workspace-a",
        "file": "model.py",
        "cls": "Net",
        "documentVersion": 3,
        "sourceDigest": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "backendRevision": 7,
    }
    projection = {
        "type": "projection",
        "protocolVersion": LOCAL_BROWSER_PROTOCOL_VERSION,
        "workspaceSessionId": "session-1",
        "messageId": "message-1",
        "hostSequence": 4,
        "target": target,
        "projection": {},
        "traced": False,
        "greyed": False,
    }

    assert is_local_browser_message(projection, "session-1") is True
    assert is_local_browser_message({**projection, "target": {**target, "workspaceFolder": ""}}, "session-1") is False
    assert is_local_browser_message({**projection, "target": {**target, "sourceDigest": f"sha256:{target['sourceDigest']}"}}, "session-1") is False
    assert is_local_browser_message({**projection, "target": {**target, "documentVersion": -1}}, "session-1") is False
    assert is_local_browser_message({**projection, "target": {**target, "backendRevision": -1}}, "session-1") is False


def test_optional_strings_reject_json_null_and_ordering_is_role_aware():
    target = {
        "workspaceFolder": "workspace-a",
        "file": "model.py",
        "cls": "Net",
        "documentVersion": 3,
        "sourceDigest": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "backendRevision": 7,
    }
    projection = {
        "type": "projection",
        "protocolVersion": LOCAL_BROWSER_PROTOCOL_VERSION,
        "workspaceSessionId": "session-1",
        "messageId": "message-1",
        "hostSequence": 4,
        "target": target,
        "projection": {},
        "traced": False,
        "greyed": False,
    }
    reveal = {
        "type": "revealNode",
        "protocolVersion": LOCAL_BROWSER_PROTOCOL_VERSION,
        "workspaceSessionId": "session-1",
        "messageId": "message-2",
        "target": target,
        "nodeId": "node-1",
    }

    assert is_local_browser_message(projection, "session-1") is True
    sync_state = {
        "type": "syncState",
        "protocolVersion": LOCAL_BROWSER_PROTOCOL_VERSION,
        "workspaceSessionId": "session-1",
        "messageId": "message-3",
        "hostSequence": 4,
        "target": target,
        "state": "runtime-traced",
    }

    assert is_local_browser_message({**projection, "hostSequence": None}, "session-1") is False
    assert is_local_browser_message(sync_state, "session-1") is True
    assert is_local_browser_message({**sync_state, "reason": None}, "session-1") is False
    assert is_local_browser_message(reveal, "session-1") is True
    assert is_local_browser_message({**reveal, "hostSequence": 1}, "session-1") is False


def test_handshake_and_reconnect_snapshot_shapes_are_explicit():
    request = {
        "type": "snapshotRequest",
        "role": "view",
        "protocolVersion": LOCAL_BROWSER_PROTOCOL_VERSION,
        "workspaceSessionId": "session-1",
        "messageId": "message-2",
        "lastHostSequence": 4,
    }

    assert is_local_browser_message(request, "session-1") is True
    assert is_local_browser_message({**request, "lastHostSequence": -1}, "session-1") is False


def test_desktop_parity_repair_is_enforced():
    target = {
        "workspaceFolder": "workspace-a",
        "file": "model.py",
        "cls": "Net",
        "documentVersion": 3,
        "sourceDigest": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "backendRevision": 7,
    }
    handshake = {
        "type": "hostHandshake",
        "protocolVersion": LOCAL_BROWSER_PROTOCOL_VERSION,
        "workspaceSessionId": "session-1",
        "messageId": "message-1",
        "role": "host",
    }
    error = {
        "type": "error",
        "protocolVersion": LOCAL_BROWSER_PROTOCOL_VERSION,
        "workspaceSessionId": "session-1",
        "messageId": "message-2",
        "code": "SESSION_MISMATCH",
        "message": "wrong session",
    }
    ack = {
        "type": "ack",
        "protocolVersion": LOCAL_BROWSER_PROTOCOL_VERSION,
        "workspaceSessionId": "session-1",
        "messageId": "message-3",
        "target": target,
        "ackId": "message-4",
        "ok": False,
    }

    assert is_local_browser_message(handshake, "session-1") is True
    assert is_local_browser_message({**handshake, "session": None}, "session-1") is False
    assert is_local_browser_message(error, "session-1") is True
    assert is_local_browser_message({**error, "target": None}, "session-1") is False
    assert is_local_browser_message({**error, "target": {}}, "session-1") is False
    assert is_local_browser_message({**ack, "errorCode": "STALE_TARGET"}, "session-1") is True
    assert is_local_browser_message({**ack, "errorCode": None}, "session-1") is False


def test_atomic_snapshot_state_is_internally_consistent():
    target = {
        "workspaceFolder": "workspace-a",
        "file": "model.py",
        "cls": "Net",
        "documentVersion": 3,
        "sourceDigest": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "backendRevision": 7,
    }

    def snapshot(state):
        return {
            "type": "snapshot",
            "protocolVersion": LOCAL_BROWSER_PROTOCOL_VERSION,
            "workspaceSessionId": "session-1",
            "messageId": "message-1",
            "role": "host",
            "hostSequence": 5,
            "state": state,
        }

    empty_state = {
        "activeTarget": None,
        "projection": None,
        "traced": False,
        "greyed": False,
        "syncState": None,
        "connectionState": "connected",
    }
    stale_state = {
        "activeTarget": target,
        "projection": {},
        "traced": False,
        "greyed": True,
        "syncState": "stale",
        "reason": "source changed",
        "connectionState": "connected",
    }

    assert is_local_browser_message(snapshot(empty_state), "session-1") is True
    assert is_local_browser_message(snapshot(stale_state), "session-1") is True
    assert is_local_browser_message(snapshot({**empty_state, "syncState": "source-only"}), "session-1") is False
    assert is_local_browser_message(snapshot({**empty_state, "projection": {}}), "session-1") is False
    assert is_local_browser_message(snapshot({**stale_state, "reason": None}), "session-1") is False
    assert is_local_browser_message(snapshot({**stale_state, "syncState": "source-only"}), "session-1") is False


def test_cross_session_messages_are_refused_even_when_structurally_valid():
    message = {
        "type": "hostHandshake",
        "protocolVersion": LOCAL_BROWSER_PROTOCOL_VERSION,
        "workspaceSessionId": "session-2",
        "messageId": "message-1",
        "role": "host",
    }

    assert is_local_browser_message(message) is True
    assert is_local_browser_message(message, "session-1") is False
    assert is_local_browser_message(message, "") is False


def test_setup_source_and_agent_draft_actions_are_bounded():
    from services.ide.graph_setup_protocol import action, fields
    assert action({"kind": "agent_help", "draft": {"profile": {}}})
    assert not action({"kind": "agent_help", "draft": []})
    assert not action({"kind": "agent_help", "instructions": "arbitrary"})
    assert action({"kind": "reveal_setup_source", "file": "model.py", "line": 5})
    for line in (0, -1, True, "5"):
        assert not action({"kind": "reveal_setup_source", "file": "model.py", "line": line})
    field = {"name": "x", "kind": "keyword", "required": True}
    assert fields([{**field, "source": {"file": "model.py", "line": 5}}])
    assert not fields([{**field, "source": {"file": "model.py", "line": 0}}])
