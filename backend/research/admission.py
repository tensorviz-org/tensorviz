"""What `POST /api/ide/project/index` will accept, and why it is not the multifile shape.

The multi-file projection endpoint caps its request at `MAX_SOURCES = 256` entries, as
a pydantic `max_length`. That is right for one model's dependency closure and fatal
for a project: entry 257 is a 422, and a five-hundred-file research repo is ordinary.
Raising that number alone would only move the wall, because the real cost is shipping
the text of every file on every tick.

So a project-index request is two lists with two different budgets:

* **`manifest`** — one `{path, digest}` per admitted file. Small entries, high cap.
  This is the complete picture: it says what exists, so the backend can tell an
  unreached file from an absent one.

* **`bodies`** — `{path, source}` only for digests the backend does not already hold.
  Large entries, low cap. After the first index of a workspace, a keystroke sends a
  manifest and one body.

The cache is content-addressed and lives only in memory, keyed by digest. It holds no
paths, so nothing about a user's workspace layout survives in it, and it never
persists — a restarted backend simply asks for bodies again.

Path safety is borrowed wholesale from `services/ide/multifile.py`: normalised
workspace-relative POSIX paths, refused rather than repaired.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

#: A manifest entry is a path and a digest, so the ceiling is about file *count*
#: rather than payload. Ten thousand files is roughly half a megabyte of manifest.
MAX_MANIFEST_ENTRIES = 10_000

#: Bodies are the expensive half. The cap is per request, not per workspace: a large
#: first index simply takes several rounds, each one admitting more.
MAX_BODIES = 512
MAX_BODY_BYTES = 1_000_000
MAX_TOTAL_BODY_BYTES = 16_000_000

#: How many source bodies the process keeps, across all workspaces. Bounded so a
#: long-lived backend cannot grow without limit; LRU because the files being edited
#: are the ones being asked for.
SOURCE_CACHE_ENTRIES = 20_000

#: Parsed files held across indexes. Generous, because an entry is small beside the
#: source it was read from and the whole point is that a workspace stays warm: a
#: repository of 600 files evicting its own entries between keystrokes would re-parse
#: exactly what this exists to avoid.
PARSE_CACHE_ENTRIES = 20_000


class AdmissionError(ValueError):
    """A request refused before any parsing happened. Carries a stable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str = Field(min_length=1, max_length=4096)
    digest: str = Field(min_length=64, max_length=64)


class BodyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str = Field(min_length=1, max_length=4096)
    source: str


class ProjectIndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: int = 1
    root: str = Field(min_length=1)
    manifest: List[ManifestEntry] = Field(default_factory=list, max_length=MAX_MANIFEST_ENTRIES)
    bodies: List[BodyEntry] = Field(default_factory=list, max_length=MAX_BODIES)
    #: The caller's own admission that its candidate set was already incomplete.
    truncated: bool = False


class MissingBodies(BaseModel):
    """The backend's answer when it cannot satisfy a manifest from cache.

    Not an error: the extension replies with the requested bodies. This is the
    handshake that keeps a per-keystroke index from re-uploading a repository.
    """

    model_config = ConfigDict(extra="forbid")

    missing: List[str]


def digest_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ParseCache:
    """`(path, content digest)` → the facts `analyse_file` read out of it.

    Parsing is where the time goes, and by a margin that leaves nothing else worth
    optimising first: measured over the validation corpus, it is **88–93%** of a
    whole-repo index — 1.32s of torchtitan's 1.49s across 596 files, 1.00s of trl's
    1.08s across 353. Building every experiment's graph from the parsed facts is the
    remaining 0.08–0.18s.

    So a keystroke does not need a reverse-import index to recompute only the
    experiments whose closure contains the edited file; it needs to not re-parse the
    595 files that did not change. This is that, and the graph rebuild stays whole —
    simpler, and already inside the budget a keystroke has.

    Keyed on the path AS WELL as the digest, unlike `SourceCache`: the facts include
    the file's own relative path (its definitions' ids are `file::qualname`), so two
    identical `__init__.py` files do NOT have identical facts and sharing one entry
    between them would put one package's nodes in another package's graph.
    """

    def __init__(self, capacity: int = PARSE_CACHE_ENTRIES) -> None:
        self._entries: "OrderedDict[Tuple[str, str], object]" = OrderedDict()
        self._capacity = capacity
        self.hits = 0
        self.misses = 0

    def get(self, path: str, digest: str) -> Optional[object]:
        key = (path, digest)
        if key not in self._entries:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return self._entries[key]

    def put(self, path: str, digest: str, facts: object) -> None:
        key = (path, digest)
        self._entries[key] = facts
        self._entries.move_to_end(key)
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


class SourceCache:
    """Digest → source text, bounded and in-memory only.

    Deliberately keyed by content alone: two workspaces containing an identical
    `__init__.py` share one entry, and nothing here records where a file lives.
    """

    def __init__(self, capacity: int = SOURCE_CACHE_ENTRIES) -> None:
        self._entries: "OrderedDict[str, str]" = OrderedDict()
        self._capacity = capacity

    def get(self, digest: str) -> Optional[str]:
        if digest not in self._entries:
            return None
        self._entries.move_to_end(digest)
        return self._entries[digest]

    def put(self, digest: str, source: str) -> None:
        self._entries[digest] = source
        self._entries.move_to_end(digest)
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()


def validate_request(request: ProjectIndexRequest, normalise) -> Tuple[Dict[str, str], List[str]]:
    """Check budgets and paths, then say which digests are still needed.

    Returns `(manifest as {path: digest}, duplicate-free ordered paths)`. Raises
    `AdmissionError` with a stable code rather than repairing anything — a path we
    had to fix is a path we might have fixed wrongly.
    """
    if request.protocol_version != 1:
        raise AdmissionError(
            "protocol_version_mismatch",
            f"expected protocol version 1, received {request.protocol_version}",
        )

    total_body_bytes = 0
    for body in request.bodies:
        size = len(body.source.encode("utf-8"))
        if size > MAX_BODY_BYTES:
            raise AdmissionError(
                "source_too_large", f"{body.file} exceeds {MAX_BODY_BYTES} bytes"
            )
        total_body_bytes += size
    if total_body_bytes > MAX_TOTAL_BODY_BYTES:
        raise AdmissionError(
            "payload_too_large",
            f"bodies total {total_body_bytes} bytes, over {MAX_TOTAL_BODY_BYTES}",
        )

    manifest: Dict[str, str] = {}
    order: List[str] = []
    for entry in request.manifest:
        path = normalise(entry.file)
        if path in manifest:
            raise AdmissionError("duplicate_path", f"{path} appears twice in the manifest")
        manifest[path] = entry.digest
        order.append(path)
    return manifest, order


def resolve_sources(
    manifest: Mapping[str, str],
    bodies: Mapping[str, str],
    cache: SourceCache,
    normalise,
) -> Tuple[Dict[str, str], List[str]]:
    """Assemble `{path: source}` from supplied bodies plus the cache.

    Anything still missing comes back as a list of paths rather than being guessed
    at or skipped — an index built on a partially-resolved set would report files as
    unreached when they were merely absent, which is a lie the user cannot debug.
    """
    for raw_path, source in bodies.items():
        path = normalise(raw_path)
        declared = manifest.get(path)
        actual = digest_of(source)
        if declared is not None and declared != actual:
            raise AdmissionError(
                "digest_mismatch",
                f"{path} does not match the digest its manifest entry declared",
            )
        cache.put(actual, source)

    sources: Dict[str, str] = {}
    missing: List[str] = []
    for path, digest in manifest.items():
        cached = cache.get(digest)
        if cached is None:
            missing.append(path)
        else:
            sources[path] = cached
    return sources, sorted(missing)
