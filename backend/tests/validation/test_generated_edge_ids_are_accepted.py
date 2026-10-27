"""The system must accept the identifiers it generates.

The operation resolver mints edge ids of the form ``src:out->dst:in`` (see
``_resolve_connect``), but the resource-identifier grammar excluded ``>``. So any
operation naming an existing edge was rejected by the system's own validator.
Asking the AI to "insert a LayerNorm right before the output" has to remove the
edge feeding the output, and the chat showed the raw pattern:

    Error: remove_connection edge_id must match '^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$'

reported 10 times. That message was also the whole of what the user was told.
"""

from __future__ import annotations


import pytest


from core.graph_input_safety import (  # noqa: E402
    UnsafeGraphInput,
    require_resource_identifier,
)


@pytest.mark.parametrize(
    "edge_id",
    [
        "mlp_tiny__n2:out->output_node:in",
        "input_node:out->mlp_tiny__n0:in",
        "a:out->b:in",
    ],
)
def test_accepts_edge_ids_the_resolver_generates(edge_id: str) -> None:
    assert require_resource_identifier(edge_id, label="remove_connection edge_id") == edge_id


def test_still_accepts_plain_node_ids() -> None:
    for value in ["output_node", "mlp_tiny__n0", "block.0", "layer:1", "a-b"]:
        assert require_resource_identifier(value, label="node_id") == value


@pytest.mark.parametrize(
    "value",
    [
        "has space",
        'has"quote',
        "has'quote",
        "-leading-hyphen",
        ".leading-dot",
        "semi;colon",
        "paren()",
        "new\nline",
        "x" * 129,
        "",
        None,
        42,
    ],
)
def test_still_rejects_everything_unsafe(value) -> None:
    # Widening the grammar by one character must not open it generally.
    with pytest.raises(UnsafeGraphInput):
        require_resource_identifier(value, label="edge_id")


def test_the_rejection_message_is_readable_rather_than_a_regex() -> None:
    # This text reaches the user in the AI chat.
    with pytest.raises(UnsafeGraphInput) as excinfo:
        require_resource_identifier("has space", label="remove_connection edge_id")
    message = str(excinfo.value)
    assert "^[A-Za-z0-9_]" not in message
    assert "{0,127}" not in message
    assert "is not a usable identifier" in message
    # It names the offending value, so the user can see what was wrong.
    assert "has space" in message
