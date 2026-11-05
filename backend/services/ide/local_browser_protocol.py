"""The checked Python consumer of the external-browser broker contract.

The TypeScript file in ``extension/src/protocol.ts`` is the canonical contract.
This module intentionally repeats its runtime constants so importing a protocol
does not drag FastAPI or Node tooling into the backend. The shared JSON fixture
corpus in ``tests/fixtures/contracts/local-browser-protocol.json`` is checked
by every consumer to detect drift across the covered contract cases.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Final, Literal

LOCAL_BROWSER_PROTOCOL_VERSION: Final[int] = 1
LOCAL_BROWSER_CAPTURE_EXPORT_MAX_BYTES: Final[int] = 16 * 1024 * 1024

LocalBrowserRole = Literal["host", "view"]
LocalBrowserErrorCode = Literal[
    "PROTOCOL_VERSION_MISMATCH",
    "SESSION_MISMATCH",
    "INVALID_ORIGIN",
    "INVALID_CREDENTIALS",
    "CREDENTIAL_EXPIRED",
    "HOST_ALREADY_CONNECTED",
    "STALE_TARGET",
    "UNKNOWN_MESSAGE",
    "INVALID_MESSAGE",
]

LOCAL_BROWSER_ROLES: Final[frozenset[str]] = frozenset({"host", "view"})
LOCAL_BROWSER_MESSAGE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "agentRequest", "agentState", "nodeExplanation", "nodeExplanationRequest",
        "graphReviewRequest", "graphReviewState", "graphReviewFocus", "graphReviewApplied",
        "graphOpening", "graphSetupRequest",
        "connected",
        "viewState", "viewSelection", "focusView", "viewFocusResult", "codeFocusRequest", "codeFocusState",
        "hostHandshake",
        "viewHandshake",
        "snapshotRequest",
        "snapshot",
        "activeTarget",
        "projection",
        "projectIndex",
        "projectDocument",
        "projectExpand",
        "projectFocus",
        "projectDeclare",
        "connectionState",
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
)

LOCAL_BROWSER_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
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
)

_SOURCE_DIGEST: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _is_bounded_string(value: Any, max_length: int) -> bool:
    return _is_non_empty_string(value) and len(value) <= max_length


def _node_explanation_text(value: Any, maximum: int) -> bool:
    # Match JavaScript string ceilings, including astral characters as two units.
    whitespace = "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
    return (isinstance(value, str) and bool(value)
            and len(value.encode("utf-16-le", "surrogatepass")) // 2 <= maximum
            and value.strip(whitespace) == value and re.search(r"[\x00-\x1f\x7f]", value) is None)


def _node_explanation_citation(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"id", "file", "sourceDigest", "startLine", "endLine"}:
        return False
    file, start, end = value["file"], value["startLine"], value["endLine"]
    return (_node_explanation_text(value["id"], 128) and _node_explanation_text(file, 4096)
            and re.match(r"^(?:/|\\|[A-Za-z][A-Za-z0-9+.-]*:)", file) is None
            and ".." not in re.split(r"[/\\]", file)
            and isinstance(value["sourceDigest"], str) and _SOURCE_DIGEST.fullmatch(value["sourceDigest"]) is not None
            and _is_non_negative_integer(start) and 1 <= start <= 9007199254740991
            and _is_non_negative_integer(end) and start <= end <= 9007199254740991)


def _node_explanation(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"title", "purpose", "inputs", "outputs", "steps", "uncertainties", "citations"}:
        return False
    def items(field, count, length):
        entries = value[field]
        return isinstance(entries, list) and len(entries) <= count and all(_node_explanation_text(item, length) for item in entries)
    citations = value["citations"]
    return (_node_explanation_text(value["title"], 80) and _node_explanation_text(value["purpose"], 320)
            and items("inputs", 6, 160) and items("outputs", 4, 160)
            and items("steps", 6, 320) and items("uncertainties", 4, 240)
            and isinstance(citations, list) and 1 <= len(citations) <= 8
            and all(_node_explanation_citation(item) for item in citations)
            and len({item["id"] for item in citations}) == len(citations))


def _node_explanation_state(value: Any) -> bool:
    return (isinstance(value, dict)
            and set(value) <= {"requestId", "nodeId", "revision", "status", "detail", "explanation"}
            and _node_explanation_text(value.get("requestId"), 128)
            and _node_explanation_text(value.get("nodeId"), 1024)
            and _node_explanation_text(value.get("revision"), 256)
            and isinstance(value.get("status"), str) and value["status"] in ("queued", "loading", "ready", "unavailable")
            and ("detail" not in value or _node_explanation_text(value["detail"], 320))
            and (_node_explanation(value.get("explanation")) if value["status"] == "ready" else "explanation" not in value))


def _is_non_negative_integer(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        return False
    return value >= 0


def _source_edit_is_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    def name(v):
        return isinstance(v, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", v) is not None
    def node(v):
        return _is_bounded_string(v, 256)
    def literal(v, depth=0):
        return depth <= 8 and (v is None or isinstance(v, bool)
            or (isinstance(v, (int, float)) and not isinstance(v, bool) and -1.7976931348623157e308 <= v <= 1.7976931348623157e308)
            or (isinstance(v, str) and len(v) <= 1024)
            or (isinstance(v, list) and len(v) <= 64 and all(literal(item, depth + 1) for item in v)))
    kind = value.get("kind")
    if kind in ("set_literal", "set_constructor_literal", "set_configuration_literal", "replace_activation", "reconnect_argument", "insert_unary_op", "delete_unary_op"):
        if not node(value.get("nodeId")):
            return False
        from services.semantic.contracts import SetLiteralEdit, SetConstructorLiteralEdit, SetConfigurationLiteralEdit, ReplaceActivationEdit, ReconnectArgumentEdit, InsertUnaryEdit, DeleteUnaryEdit
        try:
            contract = {"set_literal": SetLiteralEdit, "set_constructor_literal": SetConstructorLiteralEdit,
                        "set_configuration_literal": SetConfigurationLiteralEdit,
                        "replace_activation": ReplaceActivationEdit, "reconnect_argument": ReconnectArgumentEdit, "insert_unary_op": InsertUnaryEdit, "delete_unary_op": DeleteUnaryEdit}[kind]
            edit = contract.model_validate({key: item for key, item in value.items() if key != "nodeId"})
            if kind in ("set_constructor_literal", "set_configuration_literal", "reconnect_argument", "insert_unary_op", "delete_unary_op"):
                return True  # Contract checks target identity and exact argument path.
            return (edit.target.kind in ("operation", "invocation") if kind == "replace_activation"
                    else edit.target.kind == "operation" and edit.argumentPath == ["args", 1])
        except ValueError:
            return False
    if kind == "set_param":
        return set(value) == {"kind", "nodeId", "param", "value"} and node(value["nodeId"]) and name(value["param"]) and literal(value["value"])
    if kind == "remove_layer":
        return set(value) == {"kind", "nodeId"} and node(value["nodeId"])
    if kind == "add_layer":
        params = value.get("params")
        return (set(value) == {"kind", "afterNodeId", "nodeId", "layerType", "params"}
            and node(value["afterNodeId"]) and name(value["nodeId"]) and name(value["layerType"])
            and isinstance(params, dict) and len(params) <= 32
            and all(name(key) and literal(val) for key, val in params.items()))
    if kind == "group_layers":
        nodes = value.get("nodeIds")
        return (set(value) == {"kind", "nodeIds", "name"} and name(value["name"])
            and isinstance(nodes, list) and 2 <= len(nodes) <= 128 and all(node(item) for item in nodes)
            and len(set(nodes)) == len(nodes))
    return False


def _capture_query_is_valid(value):
    if not isinstance(value, dict):
        return False
    entity = value.get("entity")
    path = value.get("path")
    count = lambda v: _is_non_negative_integer(v) and v <= 1_000_000
    return (len(value) == (5 if "entity" not in value else 6)
        and _is_bounded_string(value.get("captureId"), 128) and _is_bounded_string(value.get("sourceId"), 128)
        and ("entity" not in value or isinstance(entity, dict) and len(entity) == 2
            and entity.get("kind") in ["operation", "invocation", "value", "input", "output"] and _is_bounded_string(entity.get("id"), 128))
        and isinstance(path, list) and len(path) <= 128 and len(json.dumps(path, ensure_ascii=False, separators=(",", ":"))) <= 8192
        and all(isinstance(key, str) and len(key) <= 128 or count(key) for key in path)
        and count(value.get("offset")) and count(value.get("limit")) and 1 <= value["limit"] <= 200)


def _capture_page_is_valid(value):
    if not isinstance(value, dict) or set(value) != ({"captureId", "sourceId", "path", "node", "items", "total", "nextOffset"} | ({"entity"} if "entity" in value else set())):
        return False
    query = {key: value[key] for key in ["captureId", "sourceId", "entity", "path"] if key in value}
    query.update(offset=0, limit=1)
    total, offset, items = value.get("total"), value.get("nextOffset"), value.get("items")
    count = lambda v: _is_non_negative_integer(v) and v <= 1_000_000
    if (not _capture_query_is_valid(query) or not count(total)
        or not (offset is None or count(offset) and offset < total)
        or not isinstance(items, list) or len(items) > min(200, total)
        or len(json.dumps(value, ensure_ascii=False, separators=(",", ":"))) > 65536):
        return False
    def entry(item, root):
        if not isinstance(item, dict) or not set(item) <= {"key", "kind", "value", "count", "label"} or "key" not in item:
            return False
        key, kind, scalar = item["key"], item.get("kind"), item.get("value")
        if (key is not None if root else not (isinstance(key, str) and len(key) <= 128 or count(key))):
            return False
        if "label" in item and not (isinstance(item["label"], str) and len(item["label"]) <= 160):
            return False
        if kind in ["object", "array"]:
            return "value" not in item and count(item.get("count"))
        return ("count" not in item and "label" not in item and "value" in item
            and (scalar is None if kind == "null" else isinstance(scalar, str) and len(scalar) <= 4096 if kind == "string"
                else isinstance(scalar, (int, float)) and not isinstance(scalar, bool) and -1.7976931348623157e308 <= scalar <= 1.7976931348623157e308 if kind == "number"
                else kind == "boolean" and isinstance(scalar, bool)))
    node = value.get("node")
    return (entry(node, True) and all(entry(item, False) for item in items)
        and (node["count"] == total if node["kind"] in ["object", "array"] else total == 0 and offset is None)
        and len({json.dumps(item["key"]) for item in items}) == len(items))


def _is_protocol_version(value: Any) -> bool:
    return _is_non_negative_integer(value) and value == LOCAL_BROWSER_PROTOCOL_VERSION


def _has_required_string(value: dict[str, Any], key: str) -> bool:
    return _is_non_empty_string(value.get(key))


def _has_optional_string(value: dict[str, Any], key: str) -> bool:
    return key not in value or _is_non_empty_string(value.get(key))


def _has_optional_target(value: dict[str, Any]) -> bool:
    return "target" not in value or _target_is_valid(value["target"])


def _has_optional_error_code(value: dict[str, Any]) -> bool:
    if "errorCode" not in value:
        return True
    error_code = value["errorCode"]
    return isinstance(error_code, str) and error_code in LOCAL_BROWSER_ERROR_CODES


def _candidate_postconditions_is_valid(value):
    from ..semantic.contracts import CandidatePostconditions
    try:
        if (not isinstance(value, dict) or isinstance(value.get("version"), bool)
                or value.get("version") != 1 or not _project_count(value.get("observedOperations"))
                or "observedInitializers" in value and not _project_count(value["observedInitializers"])):
            return False
        # JSON numbers have one representation in JS, and its string ceiling is
        # measured in UTF-16 units. Keep these transport details out of the
        # Python-owned evidence model, whose counts remain strict integers.
        for key in ("executionId", "captureId", "bindingId"):
            item = value.get(key)
            if isinstance(item, str) and len(item.encode("utf-16-le", errors="surrogatepass")) > 256:
                return False
        uses = value.get("configurationUses", [])
        if not isinstance(uses, list) or len(uses) > 256:
            return False
        normalized = []
        for use in uses:
            if not isinstance(use, dict) or not _project_string(use.get("bindingId"), 128):
                return False
            path = use.get("sourceArgumentPath")
            if isinstance(path, list) and len(path) == 2 and path[0] == "args" and _project_count(path[1]):
                path = ["args", int(path[1])]
            normalized.append({**use, "sourceArgumentPath": path})
        CandidatePostconditions.model_validate({**value, "version": 1, "observedOperations": int(value["observedOperations"]),
            "observedInitializers": int(value.get("observedInitializers", 0)), "configurationUses": normalized})
        return True
    except (ValueError, TypeError):
        return False


def _candidate_checkpoint_summary_is_valid(value):
    keys = {"loadStatus", "complete", "entries", "shapeDifferences", "dtypeDifferences", "registeredOnly", "checkpointOnly", "unknownPresence"}
    if (not isinstance(value, dict) or set(value) != keys or value["loadStatus"] not in ("succeeded", "failed")
            or not isinstance(value["complete"], bool) or not _project_count(value["entries"]) or value["entries"] > 8192):
        return False
    if not all(_project_count(value[key]) and value[key] <= value["entries"] for key in keys - {"loadStatus", "complete", "entries"}):
        return False
    both = value["entries"] - value["registeredOnly"] - value["checkpointOnly"] - value["unknownPresence"]
    return (both >= 0 and value["shapeDifferences"] <= both and value["dtypeDifferences"] <= both
        and (not value["complete"] or value["unknownPresence"] == 0))


def _candidate_checkpoint_page_is_valid(value):
    return (isinstance(value, dict) and set(value) == {"runId", "offset", "count", "nextOffset", "text"}
        and _is_bounded_string(value["runId"], 128)
        and _project_count(value["offset"]) and value["offset"] <= 8192
        and _project_count(value["count"]) and value["count"] <= 8192
        and (value["nextOffset"] is None or _project_count(value["nextOffset"])
            and value["offset"] < value["nextOffset"] < value["count"])
        and _is_bounded_string(value["text"], 131072))


def _candidate_summary_is_valid(value):
    return (isinstance(value, dict) and len(value) == 9 + ("runId" in value) + ("postconditions" in value) + ("checkpoint" in value)
        and ("runId" not in value or _is_bounded_string(value["runId"], 128))
        and value.get("staticCheck") == "passed"
        and ((value.get("semanticPostconditions") == "not_checked") if "postconditions" not in value else
            _candidate_postconditions_is_valid(value["postconditions"])
            and value.get("semanticPostconditions") == value["postconditions"]["status"]
            and value.get("runtime") in ("succeeded", "failed") and value.get("cleanup") == "complete"
            and (value["postconditions"]["status"] == "unavailable" or value.get("capture") == "complete"))
        and value.get("runtime") in ("not_run", "running", "succeeded", "failed", "refused", "cancelled", "timed_out")
        and value.get("freshness") in ("unverified", "current", "stale")
        and value.get("capture") in ("unavailable", "complete", "partial", "failed")
        and value.get("inputs") in ("same", "different", "unavailable")
        and value.get("state") in ("same", "different", "unavailable")
        and ("checkpoint" not in value or _candidate_checkpoint_summary_is_valid(value["checkpoint"])
            and value.get("runtime") in ("succeeded", "failed") and value.get("cleanup") == "complete"
            and (value["checkpoint"]["loadStatus"] != "failed" or value.get("runtime") == "failed" and value.get("capture") == "unavailable"))
        and value.get("cleanup") in ("not_run", "pending", "complete", "failed", "unavailable")
        and isinstance(value.get("diagnostics"), list) and len(value["diagnostics"]) <= 16
        and all(isinstance(code, str) and re.fullmatch(r"[a-z_]{1,80}", code) for code in value["diagnostics"]))


def _candidate_record_is_valid(value):
    if not isinstance(value, dict) or set(value) != {"runId", "startedAt", "finishedAt", "review", "summary", "executionId", "capture", "unavailable"}:
        return False
    capture = value["capture"]
    return (_is_bounded_string(value["runId"], 128) and _project_count(value["startedAt"])
        and (value["finishedAt"] is None or (_project_count(value["finishedAt"]) and value["finishedAt"] >= value["startedAt"]))
        and value["review"] in ("open", "historical") and _candidate_summary_is_valid(value["summary"])
        and value["summary"].get("runId") == value["runId"]
        and (value["executionId"] is None or _is_bounded_string(value["executionId"], 128))
        and (capture is None or (isinstance(capture, dict) and set(capture) == {"captureId", "sourceId"}
            and all(_is_bounded_string(item, 128) for item in capture.values())))
        and (value["summary"].get("checkpoint", {}).get("loadStatus") != "failed" or capture is None)
        and ("postconditions" not in value["summary"] or
            value["summary"]["postconditions"]["executionId"] == value["executionId"]
            and (value["summary"]["postconditions"]["captureId"] is None or capture is None
                or value["summary"]["postconditions"]["captureId"] == capture["captureId"]))
        and value["unavailable"] in (None, "not_captured", "invalid_capture", "too_large", "evicted"))


def _candidate_action_is_valid(value):
    if not isinstance(value, dict):
        return False
    kind = value.get("kind")
    if kind in ("list", "reconcile"):
        return set(value) == {"kind"}
    if kind == "run":
        return set(value) == {"kind", "retryId"} and _is_bounded_string(value["retryId"], 128)
    if kind == "cancel":
        return set(value) == {"kind", "runId"} and _is_bounded_string(value["runId"], 128)
    if kind == "checkpoint":
        return (set(value) == {"kind", "runId", "offset", "limit"} and _is_bounded_string(value["runId"], 128)
            and _project_count(value["offset"]) and value["offset"] <= 8192
            and _project_count(value["limit"]) and 1 <= value["limit"] <= 200)
    return (kind == "inspect" and set(value) == ({"kind", "runId", "query"} if "query" in value else {"kind", "runId"})
        and _is_bounded_string(value.get("runId"), 128) and ("query" not in value or _capture_query_is_valid(value["query"])))


def _capture_reconciliation_is_valid(value):
    if (not isinstance(value, dict) or set(value) != {"version", "baseCaptureId", "captureId", "sourceId", "reviewedDigest",
            "dependencyScope", "status", "reason", "inputs", "state", "postconditions"}
            or isinstance(value["version"], bool) or value["version"] != 1
            or not all(_project_string(value[key], 128) for key in ("baseCaptureId", "captureId", "sourceId"))
            or not isinstance(value["reviewedDigest"], str) or _SOURCE_DIGEST.fullmatch(value["reviewedDigest"]) is None
            or value["dependencyScope"] not in ("complete", "partial") or value["status"] not in ("checked", "unavailable")
            or value["reason"] not in ("reviewed_source", "source_reverted", "source_changed", "dependencies_changed", "environment_changed",
                "different_model", "capture_not_new", "source_unavailable")
            or value["inputs"] not in ("same", "different", "unavailable") or value["state"] not in ("same", "different", "unavailable")):
        return False
    checked = value["status"] == "checked"
    return checked == (value["reason"] == "reviewed_source") and (
        _candidate_postconditions_is_valid(value["postconditions"]) and value["postconditions"]["captureId"] == value["captureId"] if checked
        else value["postconditions"] is None and value["inputs"] == value["state"] == "unavailable")


def _applied_source_is_valid(value):
    if not isinstance(value, dict) or set(value) != {"receipt", "checkedAt", "source", "captureFreshness", "reconciliation", "unavailable"}:
        return False
    receipt = value["receipt"]
    if (not isinstance(receipt, dict) or set(receipt) != {"appliedAt", "file", "beforeDigest", "afterDigest", "unavailable"}
            or not _project_count(receipt["appliedAt"]) or not _project_string(receipt["file"], 4096)
            or not all(isinstance(receipt[key], str) and _SOURCE_DIGEST.fullmatch(receipt[key]) for key in ("beforeDigest", "afterDigest"))
            or receipt["beforeDigest"] == receipt["afterDigest"] or receipt["unavailable"] not in (None, "too_large", "evicted")
            or not _project_count(value["checkedAt"]) or value["checkedAt"] < receipt["appliedAt"]
            or value["source"] not in ("reviewed_unsaved", "reviewed_saved", "reverted", "changed", "unavailable")
            or value["captureFreshness"] not in ("current", "stale", "unavailable")
            or value["unavailable"] not in (None, "source_not_saved", "source_changed", "source_unavailable", "evidence_unavailable",
                "capture_unavailable", "check_stale", "check_failed")):
        return False
    result = value["reconciliation"]
    if result is None:
        return value["unavailable"] is not None and value["captureFreshness"] != "current"
    return (value["unavailable"] is None and receipt["unavailable"] is None and value["source"] == "reviewed_saved"
        and value["captureFreshness"] != "stale" and _capture_reconciliation_is_valid(result)
        and result["reviewedDigest"] == receipt["afterDigest"])


def _target_is_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    source_digest = value.get("sourceDigest")
    return (
        _has_required_string(value, "workspaceFolder")
        and _has_required_string(value, "file")
        and _has_required_string(value, "cls")
        and _is_non_negative_integer(value.get("documentVersion"))
        and isinstance(source_digest, str)
        and _SOURCE_DIGEST.fullmatch(source_digest) is not None
        and _is_non_negative_integer(value.get("backendRevision"))
    )



def _is_string_list(value: Any, limit: int) -> bool:
    """A bounded list of non-empty strings, with no duplicates.

    Bounded because an expansion set arrives from the page: an unbounded list is a
    cheap way to make the host allocate.
    """
    if not isinstance(value, list) or len(value) > limit:
        return False
    if not all(_is_non_empty_string(item) for item in value):
        return False
    return len(set(value)) == len(value)


def _project_string(value: Any, limit: int, *, empty: bool = False) -> bool:
    # JavaScript strings count UTF-16 code units, including astral characters.
    return (isinstance(value, str) and (empty or bool(value))
        and len(value.encode("utf-16-le", errors="surrogatepass")) // 2 <= limit)


def _project_count(value: Any) -> bool:
    return _is_non_negative_integer(value) and value <= 9007199254740991


def project_identity_is_valid(value: Any) -> bool:
    """The project counterpart of ``_target_is_valid``.

    A separate check, not a relaxation of the class one. ``file``,
    ``documentVersion`` and ``sourceDigest`` have no project analogue, and making
    them optional would weaken the all-fields equality that refuses a stale target —
    the rule stopping a page one revision behind from steering an editor.

    ``experimentId`` may be empty, because "no experiment focused" is a real state,
    but it must be present and a string: absent is a malformed message, not a
    default. Unknown keys are refused for the same reason every other validator here
    refuses them.
    """
    if not isinstance(value, dict):
        return False
    if set(value) != {
        "workspaceFolder",
        "projectRoot",
        "experimentId",
        "indexDigest",
        "indexRevision",
    }:
        return False
    index_digest = value.get("indexDigest")
    return (
        _project_string(value.get("workspaceFolder"), 4096)
        and _project_string(value.get("projectRoot"), 4096, empty=True)
        and _project_string(value.get("experimentId"), 512, empty=True)
        and isinstance(index_digest, str)
        and _SOURCE_DIGEST.fullmatch(index_digest) is not None
        and _project_count(value.get("indexRevision"))
    )


def _capabilities_is_valid(value: Any) -> bool:
    base = {
        "readOnly": True,
        "canNavigateSource": True,
        "canWriteSource": False,
        "canRunAgent": False,
        "canEditGraph": False,
    }
    return value == base or value == {**base, "canProposeSourceEdits": True}


def _session_is_valid(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and _has_required_string(value, "workspaceSessionId")
        and isinstance(value.get("role"), str)
        and value.get("role") in LOCAL_BROWSER_ROLES
        and _is_protocol_version(value.get("protocolVersion"))
        and _capabilities_is_valid(value.get("capabilities"))
        and _has_required_string(value, "expiresAt")
        and ("viewId" not in value or _is_bounded_string(value["viewId"], 128))
        and ("handoffId" not in value or _is_bounded_string(value["handoffId"], 128))
    )


def _snapshot_state_is_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    from .graph_setup_protocol import opening
    if "opening" in value and not opening(value["opening"], _target_is_valid):
        return False
    if "validation" in value:
        summary = value["validation"]
        target = value.get("activeTarget")
        if (not _validation_summary_is_valid(summary) or not isinstance(target, dict)
                or any(target.get(key) != summary["target"].get(key) for key in ("workspaceFolder", "file", "cls"))):
            return False
    if value.get("activeTarget") is None:
        return (
            "activeTarget" in value
            and "projection" in value
            and "syncState" in value
            and value.get("projection") is None
            and value.get("traced") is False
            and value.get("greyed") is False
            and value.get("syncState") is None
            and "reason" not in value
            and _is_connection_state(value.get("connectionState"))
        )
    return (
        _target_is_valid(value.get("activeTarget"))
        and "projection" in value
        and isinstance(value.get("traced"), bool)
        and isinstance(value.get("greyed"), bool)
        and _is_sync_state(value.get("syncState"))
        and (value.get("syncState") != "stale" or _has_required_string(value, "reason"))
        and (value.get("syncState") == "stale" or "reason" not in value)
        and _is_connection_state(value.get("connectionState"))
    )


def _validation_summary_is_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    status, revision = value.get("status"), value.get("projectionRevision")
    return (len(value) == 4 + ("projectionRevision" in value) + ("evidence" in value)
        and _target_is_valid(value.get("target")) and _is_bounded_string(value.get("runId"), 128)
        and isinstance(status, str) and status in {"running", "passed", "failed", "refused", "cancelled"}
        and _is_bounded_string(value.get("detail"), 512)
        and ("evidence" not in value or _validation_evidence_is_valid(value["evidence"], status))
        and ("projectionRevision" not in value or (status in {"passed", "failed"}
            and type(revision) is int and 0 <= revision <= 9007199254740991)))


def _validation_evidence_is_valid(value: Any, status: str) -> bool:
    if not isinstance(value, dict) or len(value) != 8:
        return False
    def identity(record):
        source, digest = record.get("sourceId"), record.get("planDigest")
        return ("sourceId" in record and "planDigest" in record
            and (source is None or _is_bounded_string(source, 256))
            and (digest is None or isinstance(digest, str) and _SOURCE_DIGEST.fullmatch(digest) is not None)
            and (digest is None or source is not None))
    outcomes = {"running": {"requested", "running"}, "passed": {"succeeded"}, "failed": {"failed", "timed_out"},
                "refused": {"needs_setup", "failed", "cancelled", "timed_out"}, "cancelled": {"cancelled"}}
    if (value.get("origin") not in ("manual", "mcp", "automatic")
            or not isinstance(value.get("attemptStatus"), str) or value["attemptStatus"] not in outcomes[status]
            or not identity(value) or any(key not in value or (value[key] is not None and not _is_bounded_string(value[key], 256))
                for key in ("lastCompleteCaptureId", "lastPartialCaptureId"))):
        return False
    capture = value.get("displayedCapture")
    if "displayedCapture" not in value or (capture is not None and (not isinstance(capture, dict) or len(capture) != 8
            or not _is_bounded_string(capture.get("id"), 256) or not _is_bounded_string(capture.get("attemptId"), 256)
            or capture.get("format") not in ("legacy", "semantic") or capture.get("status") not in ("complete", "partial", "failed")
            or capture.get("format") == "legacy" and capture.get("status") != "partial"
            or capture.get("forwardStatus") not in ("passed", "failed") or not identity(capture)
            or "unavailable" not in capture or capture["unavailable"] not in (None, "retention_limit"))):
        return False
    setup = value.get("setup")
    return "setup" in value and (setup is None or (isinstance(setup, dict) and len(setup) == 4 and value["planDigest"] is not None
        and setup.get("origin") in ("profile", "explicit") and "profileName" in setup
        and (_is_bounded_string(setup["profileName"], 128) if setup["origin"] == "profile" else setup["profileName"] is None)
        and isinstance(setup.get("device"), str) and re.fullmatch(r"cpu|mps|cuda(?::[0-9]{1,3})?", setup["device"]) is not None
        and setup.get("mode") in ("eval", "train", "preserve")))


def _is_view_state(value: Any) -> bool:
    """The broker's word on whether any browser view holds a socket.

    Deliberately NOT ``connectionState``'s vocabulary: that one has three states and
    describes the host, this one has two and describes the roster. Sharing the words
    would let a mis-routed frame validate against the wrong reader.
    """
    return isinstance(value, str) and value in {"attached", "detached"}


def _is_connection_state(value: Any) -> bool:
    return isinstance(value, str) and value in {"connected", "reconnecting", "disconnected"}


def _is_sync_state(value: Any) -> bool:
    return isinstance(value, str) and value in {"source-only", "runtime-traced", "stale"}


def _envelope_is_valid(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and _is_protocol_version(value.get("protocolVersion"))
        and _has_required_string(value, "workspaceSessionId")
        and _has_required_string(value, "messageId")
        and isinstance(value.get("type"), str)
        and value.get("type") in LOCAL_BROWSER_MESSAGE_TYPES
    )


def _agent_settings(value: Any) -> bool:
    return (isinstance(value, dict) and set(value) <= {"model", "effort"}
            and ("model" not in value or _is_bounded_string(value["model"], 256))
            and ("effort" not in value or _is_bounded_string(value["effort"], 32)))


def _agent_availability(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    models = value.get("models")
    return (value.get("status") in ("unchecked", "checking", "ready", "notInstalled", "signInRequired", "error")
            and isinstance(value.get("detail"), str) and len(value["detail"]) <= 2048
            and ("defaultModel" not in value or _is_bounded_string(value["defaultModel"], 256))
            and ("checkedAt" not in value or _is_non_negative_integer(value["checkedAt"]))
            and isinstance(models, list) and len(models) <= 100
            and all(isinstance(model, dict) and _is_bounded_string(model.get("id"), 256)
                    and _is_bounded_string(model.get("label"), 256)
                    and isinstance(model.get("description"), str) and len(model["description"]) <= 1024
                    and isinstance(model.get("efforts"), list) and len(model["efforts"]) <= 16
                    and all(_is_bounded_string(effort, 32) for effort in model["efforts"])
                    and ("defaultEffort" not in model or _is_bounded_string(model["defaultEffort"], 32)) for model in models))


def is_local_browser_message(value: Any, expected_workspace_session_id: str | None = None) -> bool:
    """Validate one complete message shape without dispatching it.

    This is deliberately structural rather than Pydantic-generated: the broker
    has not been implemented yet, and a tight validator now gives step 2 one
    refusal point instead of letting each handler invent its own tolerance.
    """

    if not _envelope_is_valid(value):
        return False

    if expected_workspace_session_id is not None:
        if not _is_non_empty_string(expected_workspace_session_id):
            return False
        if value["workspaceSessionId"] != expected_workspace_session_id:
            return False

    message_type = value["type"]

    if message_type == "nodeExplanation":
        return (len(value) == 7 and _is_bounded_string(value.get("viewId"), 128)
                and _target_is_valid(value.get("target")) and _node_explanation_state(value.get("state")))
    if message_type == "nodeExplanationRequest":
        action = value.get("action")
        return (len(value) == (10 if action == "reveal" else 9)
                and set(value) <= {"type", "protocolVersion", "workspaceSessionId", "messageId", "viewId", "target", "nodeId", "requestId", "action", "citationId"}
                and _is_bounded_string(value.get("viewId"), 128) and _target_is_valid(value.get("target"))
                and _is_bounded_string(value.get("nodeId"), 1024) and _is_bounded_string(value.get("requestId"), 128)
                and (action == "retry" or action == "reveal" and _is_bounded_string(value.get("citationId"), 128)))
    if message_type == "agentRequest":
        if not _is_bounded_string(value.get("viewId"), 128):
            return False
        action = value.get("action")
        if action in ("sync", "clear"):
            return len(value) == 6
        if action == "discover":
            return (set(value) <= {"type", "protocolVersion", "workspaceSessionId", "messageId", "viewId", "action", "refresh"}
                    and ("refresh" not in value or isinstance(value["refresh"], bool)))
        if action == "configure":
            return (len(value) == (9 if "requestId" in value else 8)
                    and ("requestId" not in value or _is_bounded_string(value["requestId"], 128))
                    and value.get("provider") in ("codex", "claude") and _agent_settings(value.get("settings")))
        if not _is_bounded_string(value.get("requestId"), 128):
            return False
        if action == "stop":
            return len(value) == 7
        nodes = value.get("nodeIds")
        return (action == "send" and len(value) == (12 if "settings" in value else 11) + (1 if "draftId" in value else 0)
                and ("draftId" not in value or _is_bounded_string(value["draftId"], 128))
                and ("settings" not in value or _agent_settings(value["settings"])) and value.get("provider") in ("codex", "claude")
                and _is_bounded_string(value.get("prompt"), 16000) and _target_is_valid(value.get("target"))
                and isinstance(nodes, list) and len(nodes) <= 100
                and all(_is_bounded_string(node, 1024) for node in nodes))
    if message_type == "agentState":
        state = value.get("state")
        if len(value) != 6 or not _is_bounded_string(value.get("viewId"), 128) or not isinstance(state, dict):
            return False
        context = state.get("context")
        messages = state.get("messages")
        return (_is_bounded_string(state.get("sessionId"), 128) and _is_non_negative_integer(state.get("revision"))
                and state.get("provider") in ("codex", "claude") and state.get("status") in ("idle", "running", "stopped", "error")
                and ("ack" not in state or _is_bounded_string(state["ack"], 128))
                and ("selections" not in state or (isinstance(state["selections"], dict)
                    and all(key in ("codex", "claude") and _agent_settings(settings) for key, settings in state["selections"].items())))
                and ("agents" not in state or (isinstance(state["agents"], dict) and set(state["agents"]) == {"codex", "claude"}
                    and all(_agent_availability(agent) for agent in state["agents"].values())))
                and isinstance(state.get("activity"), str) and len(state["activity"]) <= 2048
                and ("requestId" not in state or _is_bounded_string(state["requestId"], 128))
                and ("context" not in state or (isinstance(context, dict) and _target_is_valid(context.get("target"))
                    and isinstance(context.get("nodeIds"), list) and len(context["nodeIds"]) <= 100
                    and all(_is_bounded_string(node, 1024) for node in context["nodeIds"])))
                and isinstance(messages, list) and len(messages) <= 40
                and all(isinstance(turn, dict) and _is_bounded_string(turn.get("id"), 128)
                    and ("settings" not in turn or _agent_settings(turn["settings"]))
                    and turn.get("role") in ("user", "assistant") and isinstance(turn.get("text"), str)
                    and len(turn["text"]) <= 32000 for turn in messages))
    if message_type == "graphReviewApplied":
        return (len(value) == 7 and _is_bounded_string(value.get("reviewId"), 128)
                and _is_bounded_string(value.get("viewId"), 128) and value.get("mode") in ("before", "diff", "after"))
    if message_type == "graphReviewRequest":
        action = value.get("action")
        return (action in ("open", "openPR", "refresh", "close", "reveal", "selectModel", "acceptDraft", "discardDraft")
                and set(value) <= {"type", "protocolVersion", "workspaceSessionId", "messageId", "action", "reviewId", "side", "nodeId", "modelId"}
                and (action in ("open", "openPR") or _is_bounded_string(value.get("reviewId"), 128))
                and (action != "selectModel" or _is_bounded_string(value.get("modelId"), 128))
                and (action != "reveal" or (value.get("side") in ("before", "after")
                    and ("nodeId" not in value or _is_bounded_string(value["nodeId"], 1024)))))
    if message_type == "graphReviewState":
        state = value.get("state")
        return "state" in value and (state is None or (isinstance(state, dict) and _is_bounded_string(state.get("id"), 128)
                and ("proposal" not in state or (isinstance(state["proposal"], dict) and _is_bounded_string(state["proposal"].get("id"), 128)
                    and ("unavailable" not in state["proposal"] or isinstance(state["proposal"]["unavailable"], bool))
                    and ("canRevise" not in state["proposal"] or isinstance(state["proposal"]["canRevise"], bool))))
                and isinstance(state.get("before"), dict) and isinstance(state.get("after"), dict)
                and isinstance(state.get("beforeLabel"), str) and isinstance(state.get("afterLabel"), str)
                and isinstance(state.get("stale"), bool) and isinstance(state.get("changedFiles"), list)
                and all(isinstance(file, str) for file in state["changedFiles"])))
    if message_type == "graphReviewFocus":
        return (_is_bounded_string(value.get("reviewId"), 128) and value.get("side") in ("before", "after")
                and isinstance(value.get("nodeIds"), list) and len(value["nodeIds"]) <= 100
                and all(_is_bounded_string(node, 1024) for node in value["nodeIds"]))
    if message_type in {"codeFocusRequest", "codeFocusState"}:
        options = value.get("options")
        return (len(value) == (7 if message_type == "codeFocusRequest" else 9)
            and _is_bounded_string(value.get("viewId"), 128) and _target_is_valid(value.get("target"))
            and isinstance(options, dict) and set(options) == {"enabled", "scope", "references", "opacity"}
            and isinstance(options["enabled"], bool) and options["scope"] in ("class", "function", "node")
            and options["references"] in ("declaration", "uses", "both")
            and type(options["opacity"]) is int and 10 <= options["opacity"] <= 60
            and (message_type == "codeFocusRequest" or (isinstance(value.get("parameter"), bool)
                and _is_bounded_string(value.get("detail"), 512))))

    if message_type == "viewState":
        if "viewId" not in value and "connected" not in value:
            return "hostSequence" not in value and "target" not in value and _is_view_state(value.get("state"))
        return (len(value) == (7 if "handoffId" in value else 6)
            and ("handoffId" not in value or _is_bounded_string(value["handoffId"], 128))
            and _is_bounded_string(value.get("viewId"), 128) and isinstance(value.get("connected"), bool))
    if message_type in {"viewSelection", "focusView", "viewFocusResult"}:
        if not _is_bounded_string(value.get("viewId"), 128) or not _target_is_valid(value.get("target")):
            return False
        if message_type == "viewFocusResult":
            return (len(value) == 9 and _is_bounded_string(value.get("requestId"), 128)
                and isinstance(value.get("ok"), bool) and _is_bounded_string(value.get("detail"), 512))
        def ids(items):
            return isinstance(items, list) and len(items) <= 100 and all(_is_bounded_string(item, 512) for item in items) and len(set(items)) == len(items)
        overflow = value.get("tooLarge") is True
        if "tooLarge" in value and not (message_type == "viewSelection" and overflow and value.get("nodeIds") == [] and value.get("edgeIds") == []):
            return False
        # Code focus (True: on, False: off) travels on selections only;
        # views that send none keep code focus's own behaviour.
        if "follow" in value and not (message_type == "viewSelection" and isinstance(value["follow"], bool)):
            return False
        optional = 1 if message_type == "focusView" else int(overflow) + int("follow" in value)
        return (len(value) == 8 + optional and ids(value.get("nodeIds")) and ids(value.get("edgeIds"))
            and (message_type != "focusView" or _is_bounded_string(value.get("requestId"), 128)))

    if message_type == "graphOpening":
        from .graph_setup_protocol import opening
        return len(value) == 6 and _is_non_negative_integer(value.get("hostSequence")) and opening(value.get("state"), _target_is_valid)
    if message_type == "graphSetupRequest":
        from .graph_setup_protocol import action
        return (len(value) == 8 and _target_is_valid(value.get("target")) and _is_bounded_string(value.get("openingId"), 128)
                and _is_bounded_string(value.get("requestId"), 128) and action(value.get("action")))
    if message_type == "validationState":
        return _validation_summary_is_valid({key: item for key, item in value.items()
            if key not in {"type", "protocolVersion", "workspaceSessionId", "messageId"}})

    if message_type in {"captureExportRequest", "captureExportCancel", "captureExportResult"}:
        if not _target_is_valid(value.get("target")) or not _is_bounded_string(value.get("requestId"), 128):
            return False
        if message_type == "captureExportCancel":
            return len(value) == 6
        reference = value.get("reference")
        if not (isinstance(reference, dict) and set(reference) == {"captureId", "sourceId"}
                and all(_is_bounded_string(item, 128) for item in reference.values())):
            return False
        if message_type == "captureExportRequest":
            return len(value) == 7
        capture = value.get("capture")
        return (len(value) == 9 and "capture" in value and _is_bounded_string(value.get("detail"), 512)
            and (capture is None or (isinstance(capture, dict) and set(capture) == {"version", "sourceNames", "capture"}
                and capture["version"] == 1 and not isinstance(capture["version"], bool)
                and capture["sourceNames"] == "anonymized" and isinstance(capture["capture"], dict)
                and len(json.dumps(capture, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= LOCAL_BROWSER_CAPTURE_EXPORT_MAX_BYTES)))

    if message_type in {"captureInspectRequest", "captureInspectCancel", "captureInspectResult"}:
        if not _target_is_valid(value.get("target")) or not _is_bounded_string(value.get("requestId"), 128):
            return False
        if message_type == "captureInspectRequest":
            return len(value) == 7 and _capture_query_is_valid(value.get("query"))
        if message_type == "captureInspectCancel":
            return len(value) == 6
        return (len(value) == 8 and "page" in value and (value["page"] is None or _capture_page_is_valid(value["page"]))
            and _is_bounded_string(value.get("detail"), 512))

    if message_type in {"validateRequest", "validateCancel", "validateResult"}:
        if not _target_is_valid(value.get("target")) or not _is_bounded_string(value.get("requestId"), 128):
            return False
        if message_type == "validateResult":
            from .graph_setup_protocol import failure
            from .graph_setup_protocol import execution_reuse
            return (len(value) == 8 + ("failure" in value) + ("reused" in value)
                and isinstance(value.get("status"), str)
                and ("reused" not in value or execution_reuse(value["reused"]) and value["status"] == "passed")
                and ("failure" not in value or failure(value["failure"]))
                and value["status"] in {"passed", "failed", "refused", "cancelled"}
                and _is_bounded_string(value.get("detail"), 512))
        if message_type == "validateRequest":
            return (len(value) == 6 + ("fresh" in value)
                and ("fresh" not in value or value["fresh"] is True))
        return len(value) == 6

    if message_type in {"candidateRequest", "candidateResult"}:
        if not _target_is_valid(value.get("target")) or not all(_is_bounded_string(value.get(key), 128) for key in ("viewId", "requestId", "proposalId")):
            return False
        if message_type == "candidateRequest":
            return len(value) == 9 and _candidate_action_is_valid(value.get("action"))
        result = value.get("result")
        return (len(value) == 10 and "result" in value and _is_bounded_string(value.get("detail"), 512)
            and (result is None or (isinstance(result, dict) and set(result) == ({"candidates", "page"}
                | ({"applied"} if "applied" in result else set()) | ({"checkpoint"} if "checkpoint" in result else set()))
                and isinstance(result["candidates"], list) and len(result["candidates"]) <= 32
                and all(_candidate_record_is_valid(item) for item in result["candidates"])
                and ("applied" not in result or _applied_source_is_valid(result["applied"]))
                and ("checkpoint" not in result or _candidate_checkpoint_page_is_valid(result["checkpoint"])
                    and result["page"] is None and "applied" not in result and len(result["candidates"]) == 1
                    and result["candidates"][0]["runId"] == result["checkpoint"]["runId"]
                    and result["candidates"][0]["summary"].get("checkpoint", {}).get("entries") == result["checkpoint"]["count"])
                and (result["page"] is None or _capture_page_is_valid(result["page"])))))

    if message_type in {"sourceEditRequest", "sourceEditCancel", "sourceEditResult", "sourceEditReview"}:
        if not _target_is_valid(value.get("target")) or not _is_bounded_string(value.get("requestId"), 128):
            return False
        if message_type == "sourceEditCancel":
            return len(value) == 6
        if message_type == "sourceEditReview":
            return len(value) == 7 and "candidate" in value and (value["candidate"] is None or _candidate_summary_is_valid(value["candidate"]))
        if message_type == "sourceEditResult":
            return (len(value) == 8 and isinstance(value.get("status"), str) and value.get("status") in {"applied", "cancelled", "refused"}
                and _is_bounded_string(value.get("detail"), 512))
        return (len(value) == (8 if "viewId" in value else 7)
            and ("viewId" not in value or _is_bounded_string(value["viewId"], 128)) and _source_edit_is_valid(value.get("edit"))
            and len(json.dumps(value["edit"], ensure_ascii=False, separators=(",", ":"))) <= 16384)

    if message_type in {"connected", "hostHandshake", "viewHandshake", "snapshotRequest"}:
        if "hostSequence" in value or "target" in value:
            return False
        if message_type in {"hostHandshake", "viewHandshake"} and "session" in value:
            return False

    if message_type == "connected":
        session = value.get("session")
        return (
            isinstance(value.get("role"), str)
            and value.get("role") in LOCAL_BROWSER_ROLES
            and _session_is_valid(session)
            and session["workspaceSessionId"] == value["workspaceSessionId"]
            and session["role"] == value["role"]
        )
    if message_type == "hostHandshake":
        return value.get("role") == "host" and ("viewControl" not in value or value["viewControl"] is True)
    if message_type == "viewHandshake":
        return value.get("role") == "view" and ("handoffId" not in value or _is_bounded_string(value["handoffId"], 128))
    if message_type == "snapshotRequest":
        return value.get("role") == "view" and _is_non_negative_integer(
            value.get("lastHostSequence")
        )
    if message_type == "snapshot":
        return (
            value.get("role") == "host"
            and _is_non_negative_integer(value.get("hostSequence"))
            and _snapshot_state_is_valid(value.get("state"))
        )
    if message_type == "activeTarget":
        return (
            _is_non_negative_integer(value.get("hostSequence"))
            and _target_is_valid(value.get("target"))
            and "lastValidProjection" not in value
            and "nodeId" not in value
        )
    if message_type == "projection":
        return (
            _is_non_negative_integer(value.get("hostSequence"))
            and _target_is_valid(value.get("target"))
            and "projection" in value
            and isinstance(value.get("traced"), bool)
            and isinstance(value.get("greyed"), bool)
        )
    if message_type in {"projectIndex", "projectDocument", "projectFocus", "projectExpand", "projectDeclare"}:
        base = {"type", "protocolVersion", "workspaceSessionId", "messageId", "project"}
        def keys(required, optional=()):
            return set(required) <= set(value) <= base | set(required) | set(optional)
        if not project_identity_is_valid(value.get("project")):
            return False
        if message_type == "projectIndex":
            return (keys({"hostSequence", "index", "truncated"}, {"indexed", "discovered"})
                and _project_count(value.get("hostSequence"))
                and isinstance(value.get("index"), dict)
                and isinstance(value.get("truncated"), bool)
                and all(key not in value or _project_count(value[key]) for key in ("indexed", "discovered")))
        if message_type == "projectDocument":
            return (keys({"hostSequence", "document", "spanmap", "expanded"})
                and _project_count(value.get("hostSequence"))
                and isinstance(value.get("document"), dict) and isinstance(value.get("spanmap"), dict)
                and _is_string_list(value.get("expanded"), 4096)
                and all(_project_string(item, 512) for item in value["expanded"]))
        if message_type == "projectFocus":
            return keys({"experimentId"}) and _project_string(value.get("experimentId"), 512, empty=True)
        if message_type == "projectExpand":
            return (keys({"nodeId", "expanded"}) and _project_string(value.get("nodeId"), 512)
                and isinstance(value.get("expanded"), bool))
        args = value.get("args")
        arity = {"node": (2,), "edge": (2, 3), "experiment": (1,)}
        declaration = value.get("declaration")
        return (keys({"declaration", "args"}) and isinstance(declaration, str)
            and isinstance(args, list) and len(args) in arity.get(declaration, ())
            and all(_project_string(arg, 512) and bool(arg.strip(" \t\n\r\v\f\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff")) for arg in args))
    if message_type == "connectionState":
        return (
            "hostSequence" not in value
            and "target" not in value
            and _is_connection_state(value.get("state"))
        )
    if message_type == "syncState":
        return (
            _is_non_negative_integer(value.get("hostSequence"))
            and _target_is_valid(value.get("target"))
            and _is_sync_state(value.get("state"))
            and _has_optional_string(value, "reason")
        )
    if message_type == "staleState":
        return (
            _is_non_negative_integer(value.get("hostSequence"))
            and _target_is_valid(value.get("target"))
            and "lastValidProjection" in value
            and _has_required_string(value, "reason")
        )
    if message_type == "revealNode":
        return (
            "hostSequence" not in value
            and _target_is_valid(value.get("target"))
            and _has_required_string(value, "nodeId")
        )
    if message_type == "focusNode":
        return (
            _is_non_negative_integer(value.get("hostSequence"))
            and _target_is_valid(value.get("target"))
            and _has_required_string(value, "nodeId")
        )
    if message_type == "ack":
        return (
            "hostSequence" not in value
            and _target_is_valid(value.get("target"))
            and _has_required_string(value, "ackId")
            and isinstance(value.get("ok"), bool)
            and _has_optional_error_code(value)
            and _has_optional_string(value, "detail")
        )
    if message_type == "error":
        return (
            "hostSequence" not in value
            and _has_optional_target(value)
            and isinstance(value.get("code"), str)
            and value.get("code") in LOCAL_BROWSER_ERROR_CODES
            and _has_required_string(value, "message")
        )
    return False


# resolve() first: `parents` does not collapse a `..` segment, so importing this
# module through an unnormalised sys.path entry (a bare `tests/..` join, which
# several test files use) made this path point inside tests/ and the corpus vanish.
FIXTURE_PATH = Path(__file__).resolve().parents[3] / "tests/fixtures/contracts/local-browser-protocol.json"

FixtureMessage = Any


def local_browser_fixture_corpus() -> tuple[str, list[FixtureMessage], list[FixtureMessage]]:
    """Load the canonical valid and invalid messages used by all consumers."""
    corpus = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    expected_workspace_session_id = corpus.get("expectedWorkspaceSessionId")
    valid = corpus.get("valid")
    invalid = corpus.get("invalid")
    if not isinstance(valid, list) or not isinstance(invalid, list):
        raise ValueError("local browser fixture corpus must contain valid and invalid arrays")
    if not _is_non_empty_string(expected_workspace_session_id):
        raise ValueError("local browser fixture corpus must declare expectedWorkspaceSessionId")
    return expected_workspace_session_id, valid, invalid
