"""Regression gate: invalid source payloads never reach logs or responses verbatim."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError


from server_v2 import validation_exception_handler  # noqa: E402


def test_validation_handler_never_logs_or_returns_request_content(caplog):
    canary = "SOURCE_CONTENT_SECRET_CANARY"
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/ide/project",
            "query_string": b"",
            "headers": [],
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "scheme": "http",
        }
    )
    error = RequestValidationError(
        [
            {
                "type": "value_error",
                "loc": ("body", "source"),
                "msg": f"invalid private value {canary}",
                "input": {
                    "source": canary,
                },
            }
        ]
    )

    with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
        response = asyncio.run(validation_exception_handler(request, error))

    assert response.status_code == 422
    assert canary not in caplog.text
    assert canary.encode() not in response.body
    assert "body" not in json.loads(response.body)
    assert b'"msg":"Invalid request field"' in response.body
