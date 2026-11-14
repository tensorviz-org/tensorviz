"""PyTorch is loaded by the routes that run tensors, never by the server itself.

Most of this backend reads source and graphs: IDE projection, the review index,
graph-setup discovery, templates, structural validation and code generation. It
must start, and keep answering, in a Python environment where PyTorch is missing
or fails to load. The routes that execute layers (shape inference, verification,
ONNX export and in-process ingest) call :func:`load` first. A missing or broken
PyTorch then changes only their answer: a 503 that names the feature and says why
PyTorch could not load.
"""
from __future__ import annotations

import importlib
import logging
import sys
from types import ModuleType
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger("uvicorn.error")

#: Long enough for a dlopen or ABI message, short enough for a notification.
_MAX_ERROR_CHARS = 300

#: A failed native import can leave half-initialised extension modules behind,
#: so a second attempt reports a different and less useful error. Keep the first.
#: A missing package is not cached: installing it and retrying is cheap and valid.
_broken: Optional[BaseException] = None
_logged = set()


class PyTorchUnavailable(HTTPException):
    """A route needs PyTorch and this Python environment cannot load it.

    An HTTPException, so every handler's existing ``except HTTPException: raise``
    passes it through and any app that includes a router renders it the same way.
    """

    def __init__(self, feature: str, reason: str, error: Optional[BaseException] = None):
        if reason == "missing":
            message = f"{feature} needs PyTorch, which is not installed in the backend's Python environment."
        else:
            message = f"{feature} needs PyTorch, which is installed in the backend's Python environment but failed to load."
        detail = {
            "code": "pytorch_unavailable",
            "reason": reason,
            "feature": feature,
            "message": message,
            "hint": (
                "Graph projection, templates and code generation work without PyTorch. "
                "Install or repair PyTorch in this interpreter, then restart the tensorViz backend."
            ),
            "interpreter": sys.executable,
        }
        if error is not None:
            detail["error"] = describe(error)
        self.reason = reason
        self.feature = feature
        super().__init__(status_code=503, detail=detail)


def describe(error: BaseException) -> str:
    """The exception's type and first line, bounded. No traceback and no locals."""
    first = str(error).strip().splitlines()[0] if str(error).strip() else ""
    text = f"{type(error).__name__}: {first}" if first else type(error).__name__
    return text if len(text) <= _MAX_ERROR_CHARS else text[: _MAX_ERROR_CHARS - 1] + "…"


def _unavailable(feature: str, reason: str, error: BaseException) -> PyTorchUnavailable:
    if reason not in _logged:
        _logged.add(reason)
        logger.warning("pytorch.unavailable reason=%s feature=%s error=%s", reason, feature, describe(error))
    return PyTorchUnavailable(feature, reason, error)


def require_torch(feature: str) -> ModuleType:
    """Import and return ``torch``, or raise :class:`PyTorchUnavailable` naming ``feature``."""
    global _broken
    if _broken is not None:
        raise _unavailable(feature, "broken", _broken)
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise _unavailable(feature, "missing", exc) from exc
        # torch itself was found and one of its own modules or dependencies was not.
        _broken = exc
        raise _unavailable(feature, "broken", exc) from exc
    except Exception as exc:
        # OSError for a missing native library, RuntimeError or ImportError for an
        # ABI mismatch, AttributeError for a half-upgraded install: all are "broken".
        _broken = exc
        raise _unavailable(feature, "broken", exc) from exc
    return torch


def load(module: str, feature: str) -> ModuleType:
    """Import a backend module that needs PyTorch, or raise :class:`PyTorchUnavailable`.

    PyTorch is checked first, so any other import error from ``module`` is a real
    bug and propagates as one. The exception is a ``torch`` submodule that ``import
    torch`` did not reach, such as a private API missing from this PyTorch version.
    """
    require_torch(feature)
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        if (exc.name or "").split(".")[0] == "torch":
            raise _unavailable(feature, "broken", exc) from exc
        raise
