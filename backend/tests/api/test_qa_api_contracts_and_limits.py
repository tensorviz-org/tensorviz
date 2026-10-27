"""HTTP gates on the request contract: what the API says, and what it accepts.

Every payload here is one the Ship-Risk audit sent live and got the wrong answer
for (Q-M6 / Q-M7 / Q-M8 / Q-M9). None of them had an HTTP-level test, which is how
four separate defects survived a green suite:

  * every validation failure said the same word ("Invalid request field"), so
    "you forgot `name`" and "`code` must be a string" were indistinguishable;
  * malformed bodies came back 500 with raw exception text ("'str' object has no
    attribute 'get'"), reporting the caller's typo as a server fault;
  * twelve POST endpoints declared no request schema at all, so nothing could be
    rejected at the edge; and
  * nothing had an upper bound — a 100 KB graph name was accepted and stored.

The properties worth stating outright, because they are what the fixes mean:

  * a validation error has to say WHICH rule was broken, without echoing content
    a custom validator interpolated from the request;
  * a malformed request is 4xx and a server fault does not hand the caller its
    own stack; and
  * every limit answers 4xx — nothing is silently truncated on the way to a store.
"""

import json
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient


import server_v2  # noqa: E402
from api.routers import user_templates  # noqa: E402
from middleware import MAX_BODY_BYTES  # noqa: E402
from services.project_store import ProjectStore  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    """The whole app, because the validation-error handler is registered on it —
    but with the template store pointed at a scratch database so the run does
    not write into the shared dev one."""
    db = str(tmp_path_factory.mktemp("contracts") / "p.sqlite3")
    user_templates._store = ProjectStore(db)
    return TestClient(server_v2.app)


def _minimal_graph() -> Dict[str, Any]:
    """One Input node. Enough to get past `_require_nodes` and reach the real work."""
    return {
        "nodes": [{
            "id": "i",
            "type": "inputNode",
            "position": {"x": 0.0, "y": 0.0},
            "data": {
                "layerType": {"id": "input", "name": "Input", "parameters": []},
                "parameters": {"shape": "1,3,32,32"},
            },
        }],
        "edges": [],
    }


def _errors(response) -> list:
    detail = response.json()["detail"]
    assert isinstance(detail, list), detail
    return detail


# ── Q-M6: a validation error has to say which rule was broken ───────────────

def test_a_missing_field_says_it_is_missing(client: TestClient) -> None:
    """The audit's payload. `{}` omits `name`; the answer said "Invalid request
    field", which is not what "missing" means — it points the caller at a value
    they never sent."""
    response = client.post("/api/user-templates", json={})

    assert response.status_code == 422
    errors = _errors(response)
    assert errors[0]["type"] == "missing"
    assert errors[0]["loc"] == ["body", "name"]
    assert errors[0]["msg"] == "Field required"


def test_a_wrong_type_says_what_type_was_expected(client: TestClient) -> None:
    """The audit's second payload: `{"code": 42}` on /api/code/parse."""
    response = client.post("/api/code/parse", json={"code": 42})

    assert response.status_code == 422
    errors = _errors(response)
    assert errors[0]["type"] == "string_type"
    assert errors[0]["loc"] == ["body", "code"]
    assert "string" in errors[0]["msg"].lower()


def test_an_over_long_field_says_what_the_limit_is(client: TestClient) -> None:
    """A length rejection is useless if it will not name the length. This is the
    other half of Q-M9: the cap only helps if the caller can act on it."""
    response = client.post("/api/user-templates", json={"name": "a" * 100_000})

    assert response.status_code == 422
    errors = _errors(response)
    assert errors[0]["type"] == "string_too_long"
    assert "200" in errors[0]["msg"], errors[0]["msg"]


def test_a_custom_validator_message_is_still_redacted(client: TestClient) -> None:
    """The redaction existed for a reason and must survive the fix.

    A `ValueError` raised inside a validator interpolates arbitrary request
    content into `msg` — that is what tests/test_server_validation_privacy.py
    gates. Only pydantic's own static templates are passed through, so this
    value_error stays behind the constant.
    """
    response = client.post("/api/user-templates", json={"name": "   "})

    assert response.status_code == 422
    errors = _errors(response)
    assert errors[0]["type"] == "value_error"
    assert errors[0]["msg"] == "Invalid request field"
    # ctx can carry the submitted value; it is never echoed for any type.
    assert set(errors[0]) == {"type", "loc", "msg"}


# ── Q-M8: the twelve untyped POSTs now declare a request contract ───────────

_TYPED_POSTS = {
    "/api/templates/{template_id}/instantiate": "InstantiateTemplateRequest",
    "/api/popular-models/{model_id}/instantiate": "InstantiatePopularModelRequest",
    "/api/models/customize": "CustomizeModelRequest",
    "/api/models/infer-shapes": "GraphPayloadRequest",
    "/api/models/generate-code": "GenerateCodeRequest",
    "/api/models/validate": "GraphPayloadRequest",
    "/api/models/verify": "VerifyModelRequest",
    "/api/models/export-package": "ExportPackageRequest",
    "/api/models/export-onnx": "ExportOnnxRequest",
    "/api/code/export": "ExportCodeRequest",
    "/api/module-templates/{template_id}/instantiate": "InstantiateModuleTemplateRequest",
    "/api/canonical-templates/{template_id}/instantiate": "InstantiateTemplateRequest",
}


@pytest.mark.parametrize("path,model", sorted(_TYPED_POSTS.items()))
def test_the_endpoint_publishes_a_request_schema(client: TestClient, path: str, model: str) -> None:
    """`request: Dict[str, Any]` publishes `{"type": "object"}` — a contract that
    permits anything, which is the same as having none. A named $ref is what makes
    the body checkable at the edge and readable by a client generator."""
    spec = client.get("/openapi.json").json()
    schema = spec["paths"][path]["post"]["requestBody"]["content"]["application/json"]["schema"]

    assert schema.get("$ref") == f"#/components/schemas/{model}", schema


def test_the_gallery_can_still_send_the_key_the_backend_ignores(client: TestClient) -> None:
    """ModernTemplateGallery posts `model_name` to two instantiate routes that
    never read it. Extra-key tolerance is load-bearing: extra="forbid" here would
    422 both gallery flows."""
    response = client.post(
        "/api/templates/vision_transformer/instantiate",
        json={"config": {}, "model_name": "Custom_vision_transformer"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["graph"]["root_nodes"]


# ── Q-M7: a malformed body is the caller's error, not a server fault ────────

_MALFORMED = [
    ("/api/models/customize", {"base_graph": "hello"}),
    ("/api/models/customize", {"base_graph": {"root_nodes": "hello", "root_edges": []}}),
    ("/api/models/verify", {"graph": "hello"}),
    ("/api/models/export-onnx", {"graph": "hello"}),
    ("/api/models/export-onnx", {"graph": _minimal_graph(), "opset": "abc"}),
    ("/api/models/export-package", {"graph": _minimal_graph(), "class_name": 123}),
    ("/api/code/export", {"graph": "hello"}),
    ("/api/templates/vision_transformer/instantiate", {"config": "abc"}),
    ("/api/module-templates/multi_head_attention/instantiate", {"parameters": "abc"}),
    ("/api/canonical-templates/vit_block/instantiate", {"config": "abc"}),
]


@pytest.mark.parametrize("path,payload", _MALFORMED)
def test_a_malformed_body_is_a_client_error(client: TestClient, path: str, payload: dict) -> None:
    """Each of these answered 500 — two of them a bare "Internal Server Error"
    from a coercion that sat outside the try block."""
    response = client.post(path, json=payload)
    assert 400 <= response.status_code < 500, f"{response.status_code}: {response.text}"


@pytest.mark.parametrize("path,payload", _MALFORMED)
def test_a_malformed_body_does_not_get_the_servers_internals_back(
    client: TestClient, path: str, payload: dict
) -> None:
    """Interpolating `{str(e)}` into `detail` hands the caller Python's own
    diagnostics and, from torch/onnx internals, absolute filesystem paths."""
    body = client.post(path, json=payload).text
    for leak in ("Traceback (most recent call last)", "object has no attribute",
                 "/home/", "site-packages"):
        assert leak not in body, f"{path} leaked {leak!r}: {body[:400]}"


def test_an_unknown_popular_model_does_not_answer_two_different_ways(client: TestClient) -> None:
    """GET said 404 and POST said 500 for the identical bad id, because the
    instantiate route had no pre-check and the not-found ValueError fell into the
    catch-all."""
    assert client.get("/api/popular-models/bogus-model").status_code == 404
    assert client.post("/api/popular-models/bogus-model/instantiate", json={"config": {}}).status_code == 404


def test_infer_shapes_does_not_swallow_its_own_400(client: TestClient) -> None:
    """It was the only endpoint in the file with no `except HTTPException: raise`,
    so its own 400 was caught by the catch-all and re-emitted as
    `500 "Internal error during shape inference: 400: Graph data is required"`."""
    response = client.post("/api/models/infer-shapes", json={"graph": {}})

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Graph data is required"


# ── Q-M9: limits, and every one of them a 4xx ──────────────────────────────

def test_a_too_deep_input_shape_is_refused(client: TestClient) -> None:
    """`input_shape` was an unbounded list of unbounded ints on every model-plane
    endpoint."""
    response = client.post(
        "/api/models/validate",
        json={"graph": _minimal_graph(), "input_shape": [1] * 9},
    )

    assert response.status_code == 422, response.text
    assert _errors(response)[0]["type"] == "too_long"


def test_an_absurd_input_dimension_is_refused(client: TestClient) -> None:
    """A dimension this size describes a tensor no forward pass can allocate, so
    accepting it only defers the failure to somewhere it reads as a server fault."""
    response = client.post(
        "/api/models/validate",
        json={"graph": _minimal_graph(), "input_shape": [1, 3, 10**9, 10**9]},
    )

    assert response.status_code == 422, response.text
    assert _errors(response)[0]["type"] == "less_than_equal"


def test_a_body_over_the_ceiling_is_refused_before_it_is_parsed(client: TestClient) -> None:
    """There was no body-size limit anywhere in the app: `grep add_middleware`
    found only CORS and logging, and uvicorn was launched with host and port only.
    The declared length is what is checked, so the body itself stays small here —
    the point is that the request never reaches the endpoint."""
    response = client.post(
        "/api/models/validate",
        content=b'{"graph":{}}',
        headers={
            "content-type": "application/json",
            "content-length": str(MAX_BODY_BYTES + 1),
        },
    )

    assert response.status_code == 413, response.text
    assert "too large" in response.json()["detail"]


def test_a_body_under_the_ceiling_still_goes_through(client: TestClient) -> None:
    """The cap has to clear real payloads: the largest template graph on this
    backend (F5-TTS, 290 root nodes) serializes to ~141 KiB."""
    payload = json.dumps({"graph": _minimal_graph()}).encode()
    assert len(payload) < MAX_BODY_BYTES

    response = client.post(
        "/api/models/validate",
        content=payload,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200, response.text


def test_an_unparseable_content_length_is_refused(client: TestClient) -> None:
    """A Content-Length that is not a number cannot be compared to the ceiling, so
    it must be rejected rather than waved through the check."""
    import asyncio

    from starlette.requests import Request

    from middleware import BodySizeLimitMiddleware

    async def _never_called(_request):  # pragma: no cover - must not run
        raise AssertionError("the request was passed on with an unparseable length")

    request = Request({
        "type": "http",
        "method": "POST",
        "path": "/api/models/validate",
        "query_string": b"",
        "headers": [(b"content-length", b"not-a-number")],
    })
    response = asyncio.run(
        BodySizeLimitMiddleware(app=None).dispatch(request, _never_called)
    )

    assert response.status_code == 400
