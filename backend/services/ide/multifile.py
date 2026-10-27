"""The versioned multi-file source-projection contract (steps 1-3 of
CURRENT_TASK.md: "P0 Reliable Multi-File Model Graphs").

This module freezes the shape of ``POST /api/ide/multifile/project`` — request,
response, digests, diagnostics and limits. The resolver that makes the projection
truly cross-file lives in ``services/ide/module_index.py`` (step 2) and
``services/ide/crossfile.py`` (step 3); this module's half of the bargain never
changed while they landed:

* **No execution.** Sources arrive as text supplied by the extension (dirty
  editor buffers win over disk). Nothing in the set is imported, executed or
  written back. The backend never opens workspace paths for this feature.
* **Path safety.** Every path is a normalised workspace-relative POSIX path:
  no absolute paths, no ``..``/``.``/empty segments, no backslashes or drive
  letters, no non-Python files. Violations are refused deterministically with a
  coded error, never corrected or silently dropped.
* **Digests.** ``source_digest`` protects the ROOT buffer alone — it is the same
  digest ``/api/ide/apply`` compares before editing the root file, and nothing
  re-purposes it. ``projection_digest`` identifies the projection over the
  *entire* validated source set: an edit to any supplied dependency changes it,
  which is what lets the host notice a dirty dependency even before resolution
  consumes it.
* **Limits are explicit.** The source-set ceilings are abuse guards (a malformed
  client cannot make this endpoint allocate without bound) and refuse the call;
  the ``truncated`` flag is the client's own admission that the candidate set is
  incomplete, which the response echoes AND reports as a visible diagnostic, so
  a partial graph never looks complete.
* **Diagnostics are structured.** Codes are closed (``DIAGNOSTIC_CODES``), carry
  severity, file, line, import text and detail. The step-1 baseline entries (one
  UNRESOLVED_IMPORT per opaque imported class, emitted by a helper this module no
  longer ships) were superseded by step 3: the expander now emits a diagnostic
  exactly when a call site cannot be expanded — with the reason — and never for
  one that was.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, Field

#: The only request version this build understands. Older/newer clients get a
#: deterministic ``protocol_version_mismatch`` refusal rather than a half-answer.
IDE_PROJECTION_PROTOCOL_VERSION = 1

#: Ceilings. Per-source matches /api/ide/parse's own ceiling; the totals make a
#: malformed client refuse early instead of allocating without bound.
MAX_PATH = 4096
MAX_SOURCE_BYTES = 1_000_000
MAX_SOURCES = 256
MAX_TOTAL_SOURCE_BYTES = 8_000_000

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DRIVE = re.compile(r"[A-Za-z]:")

# ── diagnostic codes (closed set; the generator and every consumer share it) ──

UNRESOLVED_IMPORT = "UNRESOLVED_IMPORT"
UNRESOLVED_DYNAMIC = "UNRESOLVED_DYNAMIC"
AMBIGUOUS_BINDING = "AMBIGUOUS_BINDING"
CYCLE_BOUNDARY = "CYCLE_BOUNDARY"
EXTERNAL_PACKAGE = "EXTERNAL_PACKAGE"
DEPENDENCY_SYNTAX_ERROR = "DEPENDENCY_SYNTAX_ERROR"
SOURCE_SET_TRUNCATED = "SOURCE_SET_TRUNCATED"
UNSUPPORTED_IMPORT_FORM = "UNSUPPORTED_IMPORT_FORM"
UNSUPPORTED_INHERITANCE = "UNSUPPORTED_INHERITANCE"
RESOURCE_LIMIT = "RESOURCE_LIMIT"

DIAGNOSTIC_CODES = frozenset(
    {
        UNRESOLVED_IMPORT,
        UNRESOLVED_DYNAMIC,
        AMBIGUOUS_BINDING,
        CYCLE_BOUNDARY,
        EXTERNAL_PACKAGE,
        DEPENDENCY_SYNTAX_ERROR,
        SOURCE_SET_TRUNCATED,
        UNSUPPORTED_IMPORT_FORM,
        UNSUPPORTED_INHERITANCE,
        RESOURCE_LIMIT,
    }
)

SEVERITIES = frozenset({"info", "warning", "error"})


class MultifileError(ValueError):
    """A contract violation with a stable machine code for the client.

    The router turns it into a 400 whose detail carries ``code`` and ``detail``;
    neither half is optional, because "the source set was wrong" is the answer a
    host shows the user and it has to be exact about which part.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


class MultifileRoot(BaseModel):
    """The class the projection is *about*: workspace-relative file, qualified
    class, host document version, and the root buffer's own digest."""

    file: str = Field(min_length=1, max_length=MAX_PATH)
    cls: str = Field(min_length=1, max_length=200)
    document_version: Optional[int] = Field(default=None, ge=0)
    source_digest: str = Field(min_length=64, max_length=64)
    # What a previous trace of THIS class measured its input to be, remembered by
    # the host. Optional and additive: a host that does not send it gets exactly
    # the projection it got before. Ranked below `# tensorviz: input=` by
    # `utils/code_parser.py`, so it is a hint and never an override.
    measured_input_shape: Optional[List[int]] = None


class MultifileSourceEntry(BaseModel):
    """One Python source: normalised relative path, full text, digest, and the
    host document version when the file is open (the dirty buffer's version)."""

    file: str = Field(min_length=1, max_length=MAX_PATH)
    source: str = Field(max_length=MAX_SOURCE_BYTES)
    digest: str = Field(min_length=64, max_length=64)
    document_version: Optional[int] = Field(default=None, ge=0)


class MultifileProjectionRequest(BaseModel):
    """Everything a projection needs, supplied as a complete unit.

    ``truncated`` is the host saying "I could not enumerate the whole candidate
    set" — honest incompleteness that the response echoes and reports, not an
    error. ``trace_result`` is the optional executed-tier envelope, identical to
    ``/api/ide/project``'s.
    """

    protocol_version: int
    root: MultifileRoot
    sources: List[MultifileSourceEntry] = Field(min_length=1, max_length=MAX_SOURCES)
    truncated: bool = False
    trace_result: Optional[Dict[str, Any]] = None
    # Immutable Git analysis must not replace the live graph's stored document.
    review_only: bool = False


def normalize_workspace_path(raw: str) -> str:
    """Validate a path as normalised workspace-relative POSIX, or refuse loudly.

    "Normalise" on this contract means REJECT the un-normalised spellings: the
    host (VS Code) is the one place that knows the workspace root, and letting a
    server-side fix-up second-guess it is how ``models/../etc`` slips through.
    """

    if not isinstance(raw, str) or not raw:
        raise MultifileError("empty_path", "a source path was empty")
    if len(raw) > MAX_PATH:
        raise MultifileError(
            "path_too_long", f"path exceeds {MAX_PATH} characters: {raw[:64]}…"
        )
    if "\\" in raw:
        raise MultifileError(
            "not_normalised_path",
            f"backslashes are not a path separator on this contract: {raw!r}",
        )
    if raw.startswith("/") or raw.startswith("~") or _DRIVE.match(raw):
        raise MultifileError(
            "absolute_path", f"path must be workspace-relative, not absolute: {raw!r}"
        )
    segments = raw.split("/")
    for segment in segments:
        if segment == "":
            raise MultifileError(
                "not_normalised_path", f"path has an empty segment: {raw!r}"
            )
        if segment in (".", ".."):
            raise MultifileError(
                "traversal", f"path must not contain {segment!r} segments: {raw!r}"
            )
    if not raw.endswith(".py"):
        raise MultifileError("not_python", f"source entries are Python files: {raw!r}")
    return raw


def digest_of_text(text: str) -> str:
    """The sha256 hex digest of one source buffer."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_sources(
    entries: List[MultifileSourceEntry],
) -> Dict[str, MultifileSourceEntry]:
    """Path/digest/duplicate/budget checks for the supplied source set.

    Returns ``{normalised path: entry}`` in first-seen order. Rejects duplicates
    outright — even byte-identical ones — because "the same path twice" is how
    conflicting contents try to smuggle a second index past the first, and a
    strict refusal is cheaper than comparing them forever.
    """

    by_path: Dict[str, MultifileSourceEntry] = {}
    total = 0
    for raw in entries:
        path = normalize_workspace_path(raw.file)
        source_bytes = len(raw.source.encode("utf-8"))
        if source_bytes > MAX_SOURCE_BYTES:
            raise MultifileError(
                RESOURCE_LIMIT,
                f"source {path} exceeds {MAX_SOURCE_BYTES} bytes",
            )
        if not _SHA256.match(raw.digest):
            raise MultifileError(
                "digest_malformed",
                f"digest for {path} is not a sha256 hex string",
            )
        if digest_of_text(raw.source) != raw.digest:
            raise MultifileError(
                "digest_mismatch",
                f"digest for {path} does not match the supplied source text",
            )
        if path in by_path:
            same = by_path[path].source == raw.source
            raise MultifileError(
                "duplicate_path",
                f"{path} appears twice in the source set"
                + (" with identical bytes" if same else " with different bytes"),
            )
        total += source_bytes
        if total > MAX_TOTAL_SOURCE_BYTES:
            raise MultifileError(
                RESOURCE_LIMIT,
                f"source set exceeds {MAX_TOTAL_SOURCE_BYTES} bytes total",
            )
        by_path[path] = raw
    return by_path


def validate_root(
    root: MultifileRoot, sources: Mapping[str, MultifileSourceEntry]
) -> MultifileRoot:
    """The root must name a Python file inside the supplied set, digest included."""
    path = normalize_workspace_path(root.file)
    if not _SHA256.match(root.source_digest):
        raise MultifileError(
            "digest_malformed", "root source_digest is not a sha256 hex string"
        )
    entry = sources.get(path)
    if entry is None:
        raise MultifileError(
            "root_not_in_sources",
            f"root {path} is not among the supplied sources",
        )
    if entry.digest != root.source_digest:
        raise MultifileError(
            "root_digest_mismatch",
            f"root digest does not match the supplied source for {path}",
        )
    return root


def projection_digest(digests: Mapping[str, str]) -> str:
    """One digest over the complete validated source set.

    ``sha256`` of the newline-joined ``path\\0digest`` lines, sorted, under a
    versioned prefix. This is deliberately NOT the root digest: an edit to any
    supplied dependency changes it, which is what a host compares to detect that
    the whole projection — not just the root file — is stale.
    """

    material = "\n".join(
        [f"ide-multifile-projection/v{IDE_PROJECTION_PROTOCOL_VERSION}"]
        + [f"{path}\0{digests[path]}" for path in sorted(digests)]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def with_file_digests(
    spanmap: Mapping[str, Mapping[str, Any]], digests: Mapping[str, str]
) -> Dict[str, Dict[str, Any]]:
    """Copy the spanmap, adding ``file_digest`` beside each span's ``file``.

    The span's existing ``file``/``digest`` pair says WHERE the statement is and
    WHICH statement it is; ``file_digest`` says which source buffer the span was
    computed from — the check a host needs before it lets a node click open the
    file. ``None`` when the span's file was not in the supplied set (a trace
    frame in torch's own source, say), because a digest must never be invented.
    """

    enriched: Dict[str, Dict[str, Any]] = {}
    for node_id, span in spanmap.items():
        entry = dict(span)
        entry["file_digest"] = digests.get(str(span.get("file") or ""))
        enriched[node_id] = entry
    return enriched


def truncation_diagnostic() -> Dict[str, Any]:
    """The client said the source set is incomplete; the graph may be missing modules."""
    return {
        "code": SOURCE_SET_TRUNCATED,
        "severity": "warning",
        "file": "",
        "line": 0,
        "import_text": "",
        "detail": (
            "The workspace source set was truncated before this projection ran: "
            "the graph may be missing modules whose source was not supplied. "
            "Keep all candidate files open, or reduce the workspace, to lift the limit."
        ),
    }
