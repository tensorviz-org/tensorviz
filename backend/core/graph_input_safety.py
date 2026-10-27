"""Safety checks for graph values that can become generated Python tokens.

Graph values are data, not source code.  Any value used as an identifier in an
export must therefore pass a deliberately small grammar before simulation or
code generation can interpolate it.
"""

from __future__ import annotations

import inspect
import keyword
import re
from functools import lru_cache
from typing import Any, FrozenSet, Mapping, Optional

from core.op_registry import OpSpec, spec_for


# '>' is included because the operation resolver itself MINTS edge ids of the form
# "src:out->dst:in". Excluding it meant the system generated identifiers it then
# refused to accept back: any operation naming an existing edge (inserting a layer
# before the output has to remove the edge feeding it) failed validation, and the
# chat showed the raw pattern.
_RESOURCE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.:>-]{0,127}$")
_LAYER_OPERATION = re.compile(
    r"^(?:(?:torch\.)?nn\.|torch\.)?[A-Za-z_][A-Za-z0-9_]*$"
)
_PYTHON_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class UnsafeGraphInput(ValueError):
    """Graph data cannot safely cross the simulation/export boundary."""


def require_resource_identifier(value: Any, *, label: str) -> str:
    """Return a bounded, printable graph identifier or raise.

    Dots, colons, hyphens and '>' are retained because saved/imported graphs use
    them for hierarchical node references and for generated edge ids.  Quotes,
    whitespace, controls, and Python punctuation are intentionally excluded.
    """

    if not isinstance(value, str) or not _RESOURCE_IDENTIFIER.fullmatch(value):
        # Described in words, not as a regex: this message reaches the user in the
        # AI chat, where "must match '^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$'" told
        # them nothing they could act on.
        shown = value if isinstance(value, str) else type(value).__name__
        raise UnsafeGraphInput(
            f"{label} is not a usable identifier ({shown!r}). It must start with a "
            "letter, digit or underscore, be at most 128 characters, and contain "
            "only letters, digits, and the characters _ . : - >"
        )
    return value


def require_python_identifier(value: Any, *, label: str) -> str:
    """Return an ASCII, non-keyword Python identifier or raise."""

    if (
        not isinstance(value, str)
        or not _PYTHON_IDENTIFIER.fullmatch(value)
        or keyword.iskeyword(value)
    ):
        raise UnsafeGraphInput(f"{label} is not a safe Python identifier")
    return value


def require_layer_operation_syntax(
    value: Any, *, label: str = "layer operation"
) -> str:
    """Return an operation token that cannot escape a generated attribute path."""

    if not isinstance(value, str) or not _LAYER_OPERATION.fullmatch(value):
        raise UnsafeGraphInput(f"{label} is not a safe layer operation")
    return value


def require_registered_layer_operation(
    value: Any,
    *,
    label: str = "layer operation",
) -> OpSpec:
    """Resolve a syntactically safe operation through the canonical registry."""

    value = require_layer_operation_syntax(value, label=label)
    spec = spec_for(value)
    if spec is None:
        raise UnsafeGraphInput(f"{label} {value!r} is not a supported operation")
    return spec


@lru_cache(maxsize=256)
def known_constructor_parameters(layer_operation: str) -> Optional[FrozenSet[str]]:
    """Best-effort keyword allowlist for a registered ``torch.nn`` class.

    PyTorch exposes precise signatures for most modules.  A few extension and
    recurrent classes publish only ``*args, **kwargs``; those return ``None`` so
    callers still enforce the lexical identifier grammar without pretending we
    have a complete allowlist.
    """

    spec = require_registered_layer_operation(layer_operation)
    canonical = spec.canonical
    if not canonical.startswith("nn."):
        return None

    try:
        import torch

        target = getattr(torch.nn, canonical.removeprefix("nn."))
        signature = inspect.signature(target)
    except (AttributeError, TypeError, ValueError):
        return None

    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return None
    return frozenset(
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    )


def validate_layer_parameters(
    layer_operation: str,
    parameters: Mapping[str, Any],
    *,
    label: str = "layer parameters",
    allow_display_metadata: bool = False,
) -> None:
    """Validate constructor keys lexically and, where possible, semantically."""

    spec = require_registered_layer_operation(layer_operation)
    known = known_constructor_parameters(layer_operation)
    for name in parameters:
        require_python_identifier(name, label=f"{label} key")
        if allow_display_metadata and (name == "name" or name.startswith("_")):
            continue
        if name.startswith("_"):
            raise UnsafeGraphInput(f"{label} key {name!r} is reserved")
        if known is not None and name not in known:
            raise UnsafeGraphInput(
                f"{label} key {name!r} is not accepted by "
                f"{spec.canonical}"
            )
