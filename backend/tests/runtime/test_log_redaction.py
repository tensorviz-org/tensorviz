"""The launch token never reaches a log line.

It is the only thing standing between a local process and full control of this
backend, and it travels as a query parameter on the canvas frame URL that
``extension/src/shell.ts`` builds. It was logged in the clear, twice per open:
once by this application's own LoggingMiddleware and once by uvicorn's access
logger, which formats the path and query on a code path the application never
touches.

Why that is worse than it sounds: the extension pipes every backend line into the
user-visible tensorViz output channel, VS Code persists that channel to disk in the
profile, and it is the first thing a person copies into a bug report or shows on a
screen share. The secret left the machine at exactly the moment someone asked for help.

Run: KMP_DUPLICATE_LIB_OK=TRUE <viz-python> -m pytest tests/test_log_redaction.py -q
"""

from __future__ import annotations

import logging

import pytest


from middleware import (  # noqa: E402
    REDACTION,
    RedactAccessLogFilter,
    install_log_redaction,
    redact_query,
)

SECRET = "tok-ade82628f5ca56b64988e9c8a30cf384"


@pytest.mark.parametrize("name", ["token", "credential", "access_token", "api_key"])
def test_every_secret_parameter_is_redacted(name: str) -> None:
    redacted = redact_query(f"{name}={SECRET}")
    assert SECRET not in redacted
    assert redacted == f"{name}={REDACTION}"


def test_the_rest_of_the_query_survives() -> None:
    # Which parameters were sent, in what order, is the half of a query a log is read
    # for. Dropping the whole string would fix the leak by making the log useless.
    redacted = redact_query(f"depth=2&{'token'}={SECRET}&tier=real")
    assert redacted == f"depth=2&token={REDACTION}&tier=real"


def test_the_match_is_case_insensitive_on_the_name() -> None:
    assert SECRET not in redact_query(f"TOKEN={SECRET}")


@pytest.mark.parametrize("query", ["", "a=1", "token", "flag"])
def test_a_query_with_nothing_to_hide_is_left_alone(query: str) -> None:
    # `token` with no `=` is a bare flag, not a value; rewriting it would invent one.
    assert redact_query(query) == query


def _access_record(target: str) -> logging.LogRecord:
    """The record uvicorn's access logger actually emits."""
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:54846", "GET", target, "1.1", 200),
        exc_info=None,
    )


def test_uvicorns_own_access_line_is_redacted() -> None:
    record = _access_record(f"/?token={SECRET}")

    assert RedactAccessLogFilter().filter(record) is True
    assert SECRET not in record.getMessage()
    assert f"/?token={REDACTION}" in record.getMessage()


def test_an_access_line_with_no_query_is_untouched() -> None:
    record = _access_record("/api/ide/runs")
    before = record.getMessage()

    assert RedactAccessLogFilter().filter(record) is True
    assert record.getMessage() == before


def test_the_filter_never_swallows_a_line() -> None:
    # A logging filter that returns False DROPS the record. Redaction must never cost
    # the log a line — an access log with holes in it is its own debugging problem.
    for target in ["/", "/?token=x", "/api/x?a=1", None]:
        record = _access_record(target) if target is not None else _access_record("/")
        if target is None:
            record.args = ()  # a malformed record must not raise or vanish either
        assert RedactAccessLogFilter().filter(record) is True


def test_installing_twice_does_not_stack_filters() -> None:
    access = logging.getLogger("uvicorn.access")
    before = [f for f in access.filters if isinstance(f, RedactAccessLogFilter)]
    install_log_redaction()
    install_log_redaction()
    after = [f for f in access.filters if isinstance(f, RedactAccessLogFilter)]
    assert len(after) == 1, after
    assert len(before) <= 1
