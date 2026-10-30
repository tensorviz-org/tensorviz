"""The IDE plane: code in, projection out (task.md §5, M3).

Six endpoints, all static. ``/parse`` reads one buffer with ``ast`` and returns
the graph the canvas draws plus the spanmap that lets a node point back at a
line; ``/project`` is the same call with a trace merged in (M4);
``/multifile/project`` is the cross-file tier (CURRENT_TASK.md): the extension
supplies the workspace source set and gets the root class expanded into its
imported submodules, with qualified instance ids, per-file spans and digests,
and coded diagnostics for everything that deliberately stayed opaque;
``/workspace`` indexes a folder's ``nn.Module`` classes so the editor can put a
CodeLens above each one; ``/apply`` turns a canvas edit back into source
edits (M5) — and
returns them, because the extension is what writes files, not this process;
``/runs`` lists the run records the extension has appended under
``.tensorviz/runs/`` (M9), which is the one directory this process reads and the
only thing in the workspace it will ever tell you about.

**Nothing here executes the user's code.** ``/api/ingest/*`` imports and
instantiates a model in this process on purpose (documented as local trust); the
IDE deliberately does not, because in the IDE the source under the cursor is
half-written by definition. Execution lives in the M4 worker subprocess, in the
user's own interpreter, behind a per-workspace trust prompt (task.md §8.5).

Every projection is also upserted into ``ProjectStore`` at
``USER_HISTORY_DB_PATH`` under a derived graph id. That record is a cache of a
projection of the file — never the truth about the model — and it exists so the
AI harness has a revision to anchor a turn to (task.md D7, M7).
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from auth import CurrentUser, get_current_user, workspace_folder
from models.graph_models import CodeParseRequest
from research import admission, codegen, compare, document, frames, indexer
from services.ide import anchor, code_apply, multifile, projection, review
from services.ide.local_browser_protocol import LOCAL_BROWSER_CAPTURE_EXPORT_MAX_BYTES as PORTABLE_CAPTURE_MAX_BYTES
from services.ingest.document import to_document
from services.store_factory import get_project_store
from utils.code_parser import CodeParserService

router = APIRouter(prefix="/api/ide", tags=["ide"])

_store = get_project_store()
_parser = CodeParserService()

# A source buffer, a filesystem path. Both ceilings exist so a malformed client
# cannot make this endpoint allocate without bound; the source ceiling matches
# /api/code/parse's own.
_MAX_SOURCE = 1_000_000
_MAX_PATH = 4096

# Independent of capture retention and final graph-file limits. Leave headroom
# below the broker's 8 MiB frame for the authenticated result envelope.
PORTABLE_CAPTURE_INPUT_MAX_BYTES = 16 * 1024 * 1024
CAPTURE_EDIT_INPUT_MAX_BYTES = 24 * 1024 * 1024
# Two retained 16 MiB captures, up to 8 MiB of original source and a 1 MiB
# reviewed file; independent of browser frames (this endpoint is host-only).
CAPTURE_RECONCILE_INPUT_MAX_BYTES = 48 * 1024 * 1024
SOURCE_CONTEXT_INPUT_MAX_BYTES = 8 * 1024 * 1024


@router.post("/source-context")
async def source_context(request: Request, user: CurrentUser = Depends(get_current_user)):
    """Static lexical context from a bounded host-owned source inventory."""
    from services.semantic.source_context import query_source_context, SourceContextRefused
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > SOURCE_CONTEXT_INPUT_MAX_BYTES:
            raise HTTPException(413, "Source context exceeds the preparation limit.")
        body.extend(chunk)
    try:
        result = query_source_context(json.loads(body))
        rendered = json.dumps(result, ensure_ascii=False)
        if len(rendered.encode()) > 64 * 1024:
            raise HTTPException(413, "Source context exceeds the response limit.")
        return Response(rendered, media_type="application/json", headers={"Cache-Control": "no-store"})
    except SourceContextRefused as error:
        raise HTTPException(422, dict(error="source_context_refused", reason=str(error))) from None
    except (ValueError, TypeError, RecursionError, SyntaxError, UnicodeError):
        raise HTTPException(400, "Invalid source context request.") from None


@router.post("/capture/edit")
async def prepare_captured_edit(request: Request, user: CurrentUser = Depends(get_current_user)):
    """Host-only static proposal; returned ranges still require native review."""
    from services.semantic.source_edits import CapturedEditResolver, SourceEditRefused
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > CAPTURE_EDIT_INPUT_MAX_BYTES:
            raise HTTPException(413, "Captured source edit exceeds the preparation limit.")
        body.extend(chunk)
    try:
        payload = json.loads(body)
        if (not isinstance(payload, dict) or set(payload) != {"capture", "sources", "intent"}
                or not isinstance(payload["sources"], dict)):
            raise ValueError("invalid request")
        result = CapturedEditResolver(payload["capture"], payload["sources"]).plan(payload["intent"])
        rendered = json.dumps(result)
        if len(rendered.encode()) > 64 * 1024:
            raise HTTPException(413, "Captured source proposal exceeds the response limit.")
        return Response(rendered, media_type="application/json", headers={"Cache-Control": "no-store"})
    except SourceEditRefused as error:
        raise HTTPException(422, dict(error="capture_edit_refused", reason=str(error))) from None
    except (ValueError, TypeError, RecursionError):
        # Pydantic/parser errors can contain source and literals. Never echo them.
        raise HTTPException(400, "Invalid captured source proposal.") from None


@router.post("/capture/reconcile")
async def reconcile_captured_edit(request: Request, user: CurrentUser = Depends(get_current_user)):
    """Check observed applied-source effects without reading files or running code."""
    from services.semantic.source_reconciliation import reconcile_capture
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > CAPTURE_RECONCILE_INPUT_MAX_BYTES:
            raise HTTPException(413, "Source reconciliation exceeds the preparation limit.")
        body.extend(chunk)
    try:
        result = reconcile_capture(json.loads(body))
        rendered = result.model_dump_json()
        if len(rendered.encode()) > 64 * 1024:
            raise HTTPException(413, "Source reconciliation exceeds the response limit.")
        return Response(rendered, media_type="application/json", headers={"Cache-Control": "no-store"})
    except (ValueError, TypeError, RecursionError, KeyError):
        # Invalid requests may contain private source and constructor literals.
        raise HTTPException(400, "Invalid captured source reconciliation.") from None


@router.post("/capture/portable")
async def prepare_portable_capture(request: Request, user: CurrentUser = Depends(get_current_user)):
    """Detach retained evidence without opening files or executing user code.

    Parse inside the safe error boundary so validation errors cannot echo private
    argument literals. Count streamed bytes even without Content-Length.
    """
    from services.semantic.portable import portable_capture
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > PORTABLE_CAPTURE_INPUT_MAX_BYTES:
            raise HTTPException(413, "Retained capture exceeds the portable preparation input limit. Nothing was truncated.")
        body.extend(chunk)
    try:
        capture = portable_capture(json.loads(body)).model_dump_json()
    except (ValueError, TypeError, RecursionError):
        raise HTTPException(400, "Lossless export refused: capture data is invalid or contains a private/path-like literal.") from None
    if len(capture.encode("utf-8")) > PORTABLE_CAPTURE_MAX_BYTES:
        raise HTTPException(413, "Retained capture exceeds the 16 MiB live-export limit. Nothing was truncated.")
    return Response(capture, media_type="application/json", headers={"Cache-Control": "no-store"})


class ParseRequest(BaseModel):
    """One editor buffer. ``source`` is sent, never read from disk: the buffer the
    user is looking at is usually dirtier than the file."""

    file: str = Field(min_length=1, max_length=_MAX_PATH)
    source: str = Field(max_length=_MAX_SOURCE)
    cls: Optional[str] = Field(default=None, max_length=200)
    # What a PREVIOUS trace of this class measured, remembered by the extension in
    # `.tensorviz/views/<rel>#<Class>.view.json`. Without it the drawing tier
    # guesses the input independently of the worker, so the Input node can
    # disagree with the measured shapes on the cards below it. It is a hint and
    # ranks like one: `# tensorviz: input=` still wins (models/graph_models.py).
    measured_input_shape: Optional[List[int]] = None


class ProjectRequest(ParseRequest):
    """A buffer plus, optionally, what running it produced.

    ``trace_result`` is the worker envelope of task.md §6.3 verbatim — the JSON the
    extension read off ``services.ingest.worker``'s stdout and persisted under
    ``.tensorviz/cache``. It is passed through rather than re-derived because this
    process must never execute the workspace (task.md §8.5): the trace happened in
    the user's interpreter, in a subprocess the extension owns.
    """

    trace_result: Optional[Dict[str, Any]] = None
    source_review: bool = False


class WorkspaceClassesRequest(BaseModel):
    root: str = Field(min_length=1, max_length=_MAX_PATH)


class BufferClassesRequest(BaseModel):
    """One buffer, for the "which ``nn.Module`` is this cursor inside" question.

    Same contract as ``ParseRequest``: ``source`` is the buffer as it stands —
    the editor sends it, nothing goes near the disk, so an unsaved class is as
    visible to this endpoint as a saved one. ``file`` only ever appears in error
    text.
    """

    file: str = Field(min_length=1, max_length=_MAX_PATH)
    source: str = Field(max_length=_MAX_SOURCE)
    include_unresolved_bases: bool = False
    check_compilation: bool = False

#: How many run files ``GET /runs`` will open, and how many lines it will read out of
#: each. `.tensorviz/runs/` is append-only and nobody prunes it, so an unbounded read
#: is a workspace that gets slower every time it is traced.
_MAX_RUN_FILES = 64
_MAX_RUN_LINES = 500


class ApplyRequest(BaseModel):
    """A canvas edit, as the operations that express it (task.md M5).

    ``spanmap`` and ``source_digest`` come from the projection the user is
    looking at; ``source`` is the buffer as it stands now. The two are compared
    before anything is edited, because a span computed against a buffer the user
    has since typed into points at the wrong line, and applying an edit there
    corrupts the file silently.
    """

    file: str = Field(min_length=1, max_length=_MAX_PATH)
    cls: str = Field(min_length=1, max_length=200)
    source: str = Field(max_length=_MAX_SOURCE)
    source_digest: str = Field(min_length=1, max_length=128)
    spanmap: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    ops: List[Dict[str, Any]] = Field(min_length=1, max_length=200)


def _reject_file_traversal(file: str) -> None:
    """Refuse a ``file`` that walks out of the workspace with ``..`` segments.

    The buffer is still the caller's text. The path is stored and later joined
    by the extension, so a segment of ``..`` is rejected rather than normalised.
    """
    parts = file.replace("\\", "/").split("/")
    if any(part == ".." for part in parts):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "path_traversal",
                "message": "file must not contain '..' path segments",
            },
        )


def _configured_scan_roots() -> List[Path]:
    """Workspace directories this process is allowed to index."""
    raw: List[str] = []
    workspace = os.getenv("TENSORVIZ_WORKSPACE_DIR", "").strip()
    if workspace:
        raw.append(workspace)
    extra = os.getenv("TENSORVIZ_ALLOWED_ROOTS", "")
    raw.extend(part.strip() for part in extra.split(os.pathsep) if part.strip())
    folder = workspace_folder()
    if folder is not None:
        raw.append(str(folder))
    roots: List[Path] = []
    seen = set()
    for candidate in raw:
        try:
            resolved = Path(candidate).expanduser().resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(resolved)
    return roots


def _within_root(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_scan_root(root: str) -> str:
    """Resolve ``root`` and keep it inside the configured workspace.

    With no workspace configured (the test suite and a loopback ``start.sh``
    that was not given a folder), any existing directory is still accepted.
    Token mode with no folder is refused: a launch secret must not become a
    pass to index ``/``.
    """
    try:
        resolved = Path(root).expanduser().resolve()
    except OSError as exc:
        raise HTTPException(status_code=400, detail="root is not a usable directory") from exc
    if not resolved.is_dir():
        raise HTTPException(status_code=404, detail=f"{root} is not a directory")
    allowed = _configured_scan_roots()
    if not allowed:
        if os.getenv("TENSORVIZ_LOCAL_TOKEN", "").strip():
            raise HTTPException(
                status_code=403,
                detail=(
                    "Refusing to scan an arbitrary directory. Set TENSORVIZ_WORKSPACE_DIR "
                    "or TENSORVIZ_ALLOWED_ROOTS to the opened workspace."
                ),
            )
        return str(resolved)
    if not any(_within_root(resolved, base) for base in allowed):
        raise HTTPException(
            status_code=403,
            detail="root is outside the configured workspace",
        )
    return str(resolved)


def _parse_class(request: ParseRequest) -> tuple[str, Dict[str, Any]]:
    """The requested class's ``{nodes, edges}``, or a 4xx explaining why not."""
    try:
        parsed = _parser.parse_pytorch_code(
            CodeParseRequest(
                code=request.source,
                file=request.file,
                measured_input_shape=request.measured_input_shape,
            )
        )
    except SyntaxError as error:
        # The editor shows this as a Diagnostic on the offending line and keeps
        # the last good projection greyed, so the message and the line matter.
        raise HTTPException(
            status_code=400,
            detail={
                "error": "syntax_error",
                "message": str(error.msg),
                "line": error.lineno or 1,
                "column": (error.offset or 1) - 1,
            },
        ) from error
    if not parsed.get("success"):
        raise HTTPException(status_code=422, detail=parsed.get("error") or "could not parse")

    entries = [entry for entry in parsed.get("graphs") or [] if entry.get("graph") is not None]
    if not entries:
        raise HTTPException(status_code=422, detail="no nn.Module class in this file")

    if request.cls:
        chosen = next((e for e in entries if e["name"] == request.cls), None)
        if chosen is None:
            raise HTTPException(status_code=404, detail=f"{request.cls} is not in {request.file}")
    else:
        # The class nobody else instantiates is the one the file is about; the
        # frontend importer already picks it this way (apiService.ts).
        chosen = next((e for e in entries if e.get("is_top_level")), entries[0])

    graph = chosen["graph"]
    return chosen["name"], {
        "nodes": [node.model_dump() for node in graph.nodes],
        "edges": [edge.model_dump() for edge in graph.edges],
    }


def _project(request: ParseRequest, uid: str, trace: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cls, ast_graph = _parse_class(request)
    try:
        result = projection.project(
            ast_graph, trace, cls=cls, rel_file=request.file, source=request.source
        )
    except projection.TraceRejected as error:
        # A trace this backend cannot read is the client's to fix (usually a worker
        # from a different build), so it is a 400 naming the mismatch — never a
        # silent fall back to the static tier, which would answer `traced: false`
        # while the user watches a progress notification that says otherwise.
        detail: Dict[str, Any] = {"error": "trace_rejected", "message": str(error)}
        if error.trace_error is not None:
            # The run raised before it could produce a graph. The same shape a
            # successful-but-partial run returns under `trace_error`, so the editor
            # puts its diagnostic on the line that raised either way rather than
            # having a second code path for the case where nothing was drawn.
            detail["trace_error"] = error.trace_error
        raise HTTPException(status_code=400, detail=detail) from error
    except projection.ProjectionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    if isinstance(request, ProjectRequest) and request.source_review:
        from dataclasses import replace
        if trace is not None:
            raise HTTPException(status_code=400, detail="source review cannot include a runtime trace")
        graph_id = projection.source_review_graph_id(result.graph_id)
        document = result.document.model_copy(deep=True)
        document.semantic.graph_id = graph_id
        result = replace(result, document=document, graph_id=graph_id)

    revision = anchor.upsert_projection(
        _store,
        owner_uid=uid,
        graph_id=result.graph_id,
        name=cls,
        document=result.document.model_dump(mode="json", by_alias=True),
    )
    return {
        "cls": cls,
        "document": result.document.model_dump(mode="json", by_alias=True),
        "spanmap": result.spanmap,
        "source_digest": result.source_digest,
        "graph_id": result.graph_id,
        "revision": revision,
        "traced": result.traced,
        # The run raised, and this is where. Without it a half-finished trace and a
        # model with dead code arrive as the same document — the same dimmed cards,
        # nothing saying which — so the canvas can only draw the partial graph as
        # though it were the whole model (task.md §5, §6.3). Null when nothing did.
        "trace_error": result.error,
        # The project every IDE projection is anchored to. Returned because the
        # extension has to create agent sessions against it and has no other way
        # to learn the id — nothing else on this surface mentions a project, and
        # a host that guessed would anchor the loop to a project of its own.
        "project_id": anchor.project_id(_store, uid),
    }


@router.post("/parse")
def parse_buffer(
    request: ParseRequest, user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """AST tier: structure, spans and dataflow, in milliseconds and with no torch.

    Shapes carried by this document are ``_shape_provenance: "static"`` — read
    off the source, never observed. The canvas badges them stale until M4's trace
    replaces them.
    """
    _reject_file_traversal(request.file)
    return _project(request, user.uid)


@router.post("/project")
def project_buffer(
    request: ProjectRequest, user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """The same projection with an executed trace merged in.

    Identical to ``/parse`` while ``trace_result`` is null. With one, the trace wins
    the node set, the edges, the shapes and the repeat counts, and the parse supplies
    the spans, the raw call text and every node the run never reached — which stays
    on the canvas, dimmed, because a trace must not delete a layer the user can see
    in their own file (task.md §5).
    """
    _reject_file_traversal(request.file)
    return _project(request, user.uid, request.trace_result)




class GraphSetupDiscoveryRequest(multifile.MultifileProjectionRequest):
    profile: Optional[Dict[str, Any]] = None


@router.post("/graph-setup/discover")
def discover_graph_setup(
    request: GraphSetupDiscoveryRequest,
    user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """Inspect the host's source snapshot without importing or running the model."""
    from services.ide.graph_setup import discover
    from services.ide.graph_setup_profiles import restore_config_profile
    try:
        result = discover(request)
        if request.profile is not None:
            result["restoredDraft"] = restore_config_profile(request, request.profile, result)
        return result
    except (ValueError, TypeError, RecursionError):
        raise HTTPException(400, "Model setup could not be read from the supplied source snapshot.") from None


class GraphSetupProfileRequest(BaseModel):
    sources: multifile.MultifileProjectionRequest
    config: str = Field(max_length=262144)
    profile: Dict[str, Any]
    profile_name: Optional[str] = Field(default=None, max_length=64)
    config_object: Optional[Dict[str, Any]] = None
    config_defaults: bool = False


@router.post("/graph-setup/profile")
def prepare_graph_profile(request: GraphSetupProfileRequest, user: CurrentUser = Depends(get_current_user)):
    from services.ide.graph_setup_profiles import prepare_profile
    try:
        return prepare_profile(request.sources, request.config, request.profile,
                               request.profile_name, request.config_object, config_defaults=request.config_defaults)
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
        raise HTTPException(400, "Setup could not be prepared. Check the selected configuration, required fields and existing tensorviz.toml. Nothing was written.") from None


@router.post("/review/index")
def review_index(request: review.ReviewIndexRequest, user: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    try:
        return review.index_review(request)
    except multifile.MultifileError as error:
        raise HTTPException(status_code=400, detail={"code": error.code, "detail": error.detail}) from error


@router.post("/multifile/project")
def multifile_project(
    request: multifile.MultifileProjectionRequest,
    user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    from services.ide.workspace_projection import ProjectionError, project_workspace

    def persist(graph_id, name, document):
        return anchor.upsert_projection(_store, owner_uid=user.uid, graph_id=graph_id,
                                        name=name, document=document), anchor.project_id(_store, user.uid)

    try:
        result = project_workspace(request, persist)
        # Review projections are never stored, but keep the extension project identity.
        result["project_id"] = anchor.project_id(_store, user.uid)
        return result
    except ProjectionError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error

@router.post("/apply")
def apply_edits(
    request: ApplyRequest, user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Graph operations in, source edits out. **This writes nothing.**

    The extension applies the returned ranges as a ``WorkspaceEdit`` — which is
    what puts the change in the user's undo stack and in their dirty buffer,
    where they can see it before it reaches disk. FastAPI's only workspace write
    stays ``.tensorviz/state.sqlite3`` (task.md §8.5).

    A refusal is a 200 with an empty-handed ``refused`` entry, not an error: a
    batch can be half-applicable, and the canvas needs the reason and the range
    to show the user why the node it drew cannot be edited where it stands. The
    one exception is a stale spanmap, which invalidates every span in the
    request at once and therefore answers 409.
    """
    if projection.digest_of(request.source) != request.source_digest:
        raise HTTPException(
            status_code=409,
            detail={
                "code": code_apply.SPANMAP_STALE,
                "message": (
                    "the spans were computed from a different buffer; re-project "
                    "this file and try the edit again"
                ),
                "source_digest": projection.digest_of(request.source),
            },
        )

    try:
        result = code_apply.apply_operations(
            request.source,
            request.spanmap,
            request.ops,
            cls=request.cls,
            file=request.file,
        )
    except SyntaxError as error:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "syntax_error",
                "message": str(error.msg),
                "line": error.lineno or 1,
                "column": (error.offset or 1) - 1,
            },
        ) from error
    except code_apply.ApplyError as error:
        # A transform in code_apply rendered source Python cannot parse. That is
        # this server's bug, and it must never be handed to a WorkspaceEdit.
        raise HTTPException(status_code=500, detail=str(error)) from error

    return {
        "edits": result.edits,
        "refused": result.refused,
        "source_digest": projection.digest_of(result.source),
        "spanmap": _spanmap_after(request, result.source),
    }


def _spanmap_after(request: ApplyRequest, source: str) -> Dict[str, Any]:
    """The spanmap of the edited buffer, so the next edit has fresh spans.

    Recomputed here rather than left to a round trip because every span below the
    edit has moved and the client would otherwise send stale ones straight back.
    Deliberately NOT ``_project``: that upserts the projection and bumps the AI
    harness's revision, and an edit the user has not saved yet is not a new
    revision of anything (task.md D7).
    """
    if source == request.source:
        return dict(request.spanmap)
    try:
        cls, ast_graph = _parse_class(
            ParseRequest(file=request.file, source=source, cls=request.cls)
        )
        return projection.project(
            ast_graph, None, cls=cls, rel_file=request.file, source=source
        ).spanmap
    except (HTTPException, projection.ProjectionError):
        # The edit applied but the result does not project — a class renamed out
        # from under us, say. The edits are still right; the client re-parses the
        # buffer it is about to hold anyway.
        return {}


def _required_args(init: ast.FunctionDef) -> int:
    """How many arguments this ``__init__`` demands, ``self`` excluded."""
    positional = [a.arg for a in init.args.args if a.arg not in ("self", "cls")]
    required = len(positional) - len(init.args.defaults)
    required += sum(1 for default in init.args.kw_defaults if default is None)
    return max(0, required)


def _ctor_required(class_node: ast.ClassDef, classes: Dict[str, Dict[str, Any]]) -> int:
    """How many arguments constructing this class demands, ``self`` excluded.

    The M4 worker needs this to know whether it can construct the class with no
    arguments; the CodeLens uses it to say "Open as Graph" versus asking first.

    A class that defines no ``__init__`` inherits one, and the inherited one is
    what ``Deep()`` actually calls. Reading only the class's own body answered 0
    for ``class Deep(Block)`` where ``Block.__init__`` demands ``width``, so the
    CodeLens offered the one-click path and the worker died constructing it —
    after the user had already granted trust to run their code. So the bases are
    walked, depth-first and left to right, which is the MRO closely enough for
    the single-inheritance chains this question is asked about.

    Bases are resolved against the repo scan's own index, which is keyed by class
    name; a base defined outside the workspace (``nn.Module``, a pip-installed
    block) resolves to nothing and the walk ends at 0, because answering
    otherwise would mean guessing at a signature we cannot read without importing
    the package.
    """
    seen: set = set()
    queue = [class_node]
    while queue:
        node = queue.pop(0)
        if node.name in seen:
            continue
        seen.add(node.name)
        init = next(
            (n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
            None,
        )
        if init is not None:
            return _required_args(init)
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                name = base.id
            elif isinstance(base, ast.Attribute):
                # ``models.blocks.Block`` is indexed under ``Block``, the same way
                # the scan closes "is an nn.Module" over the last dotted segment.
                name = base.attr
            else:
                continue
            info = classes.get(name)
            if info is not None:
                bases.append(info["node"])
        queue = bases + queue
    return 0


@router.get("/runs")
def list_runs(
    root: str, user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Every run record the extension has written under ``<root>/.tensorviz/runs``.

    The direction of the arrow is the point (task.md §6.5): this process **reads**
    ``runs/`` and never writes it. A record is one line of JSON appended by the
    extension after a worker finished, so the file is append-only, crash-safe by
    construction, and readable while it is being written.

    A line that does not validate as a ``TraceRecord`` is COUNTED, not skipped
    silently and not fatal: a run written by an older build is exactly the thing a
    user needs told about, and dropping it would make the list quietly shorter than
    the directory.
    """
    workspace = os.path.abspath(os.path.expanduser(root))
    runs_dir = os.path.join(workspace, ".tensorviz", "runs")
    if not os.path.isdir(runs_dir):
        # Not a 404: a workspace that has never been traced has no runs, which is an
        # empty list and not an error. The editor asks this on every window open.
        return {"root": workspace, "records": [], "unreadable": 0}

    # Imported here rather than at module scope: `services.ingest.spec` is a pure
    # pydantic module, but it lives in the package whose __init__ pulls the tracer in.
    from services.ingest.spec import TraceRecord

    records: List[Dict[str, Any]] = []
    unreadable = 0
    files = sorted(
        name for name in os.listdir(runs_dir) if name.endswith(".jsonl")
    )[:_MAX_RUN_FILES]
    for name in files:
        try:
            with open(os.path.join(runs_dir, name), encoding="utf-8") as handle:
                lines = [line for line in handle.read().splitlines() if line.strip()]
        except OSError:
            unreadable += 1
            continue
        for line in lines[:_MAX_RUN_LINES]:
            try:
                records.append(TraceRecord.model_validate_json(line).model_dump(mode="json"))
            except Exception:
                unreadable += 1
    # Newest first, which is the order a Runs view lists them in and the order that
    # makes the last thing the user did the first thing they see.
    records.sort(key=lambda record: str(record.get("started_at") or ""), reverse=True)
    return {"root": workspace, "records": records, "unreadable": unreadable}


@router.post("/workspace/classes")
def workspace_classes(
    request: WorkspaceClassesRequest, user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Every ``nn.Module`` subclass under ``root``, by AST only.

    ``repo_scan.scan_repo`` follows base aliases and workspace import bindings by
    declaration identity, so unrelated classes sharing a name stay independent.
    It never imports either file — which is the requirement: an editor asks this on
    every open and save, and importing a half-written training script to answer
    would execute it.

    Each entry carries the ``import_path`` that would import it, because that is
    what the palette needs to WRITE: adding a workspace block to another file is
    one declaration and one import, and a class name alone cannot produce the
    second (task.md M6).

    One entry per *declaration*, from ``scan.class_occurrences``, not one per class
    name: a workspace that declares the same name in two files gets a row for each.
    Reading the name-keyed index instead kept only the file scanned last, and the
    "Open as Graph" lens — which matches rows to the open document by path — then
    went missing on every other copy.
    """
    root = _resolve_scan_root(request.root)

    # Imported here, not at module scope: services.ingest pulls in torch and the
    # whole tracer, and this router's other two endpoints need neither.
    from services.ingest.repo_scan import module_path_for, scan_repo
    from services.ingest.service import package_root

    scan = scan_repo(root)
    classes: List[Dict[str, Any]] = []
    for info in sorted(
        scan.class_occurrences, key=lambda entry: (entry["name"], entry["file"])
    ):
        name = info["name"]
        if not info["is_module"]:
            continue
        # From the PACKAGE root, not from `root`: a user opens the folder they are
        # working in, and `models/blocks.py` scanned with `models/` as the root
        # would answer `blocks` — an import that resolves from nowhere. Walking out
        # of the package is the same rule the trace worker imports by
        # (services/ingest/service.package_root), so the palette writes the import
        # the worker would have used.
        absolute = os.path.join(root, info["file"])
        try:
            import_path = module_path_for(package_root(absolute), absolute)
        except (ValueError, OSError):
            import_path = None
        classes.append(
            {
                "name": name,
                "file": info["file"].replace(os.sep, "/"),
                "line": info["line"],
                "bases": info["bases"],
                "ctor_required": _ctor_required(info["node"], scan.classes) if isinstance(info["node"], ast.ClassDef) else 0,
                "import_path": import_path,
            }
        )
    # `truncated` is the honest half of a bounded walk. The index is answered on every
    # open and save from an editor that passes no timeout, so the walk has hard bounds —
    # and an index silently missing the user's own model reads as "tensorViz cannot see
    # my model" with nothing to act on. Saying it was cut short is what makes it
    # actionable: point the folder at the project rather than the home directory.
    return {
        "root": root,
        "classes": classes,
        "file_count": scan.file_count,
        "truncated": scan.truncated,
    }


@router.post("/buffer-classes")
def buffer_classes(
    request: BufferClassesRequest, user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Every ``nn.Module`` subclass in one buffer, with its 1-based line span, by AST only.

    The difference from ``workspace/classes`` is the input: that one indexes the folder on
    disk for lenses and the Models view, while this one answers about the buffer the editor
    holds RIGHT NOW — the resolution "Open as Graph" needs when the user has just typed a
    class and saved nothing. The ``nn.Module`` rule is the same one
    (``repo_scan.scan_source`` and ``scan_repo`` share it), so the picker never offers a
    class the projection could refuse. Explicit opening may additionally request
    labelled top-level subclasses with unresolved bases; their runtime type still
    has to pass the capture worker.
    """
    # Imported here, not at module scope, for the same reason workspace/classes does it:
    # services.ingest's module scope pulls in torch and the whole tracer.
    from services.ingest.repo_scan import scan_source

    try:
        if request.check_compilation:
            # AST parsing alone accepts e.g. a top-level return. Compile the
            # reviewed bytes to check that boundary without executing them.
            compile(request.source.encode("utf-8"), request.file, "exec", dont_inherit=True)
        classes = scan_source(request.source, include_unresolved_bases=request.include_unresolved_bases)
    except SyntaxError as error:
        # The exact shape /api/ide/parse reports, so the editor's answer to "the buffer
        # does not parse" is one code path whichever endpoint noticed first.
        raise HTTPException(
            status_code=400,
            detail={
                "error": "syntax_error",
                "message": str(error.msg),
                "line": error.lineno or 1,
                "column": (error.offset or 1) - 1,
            },
        ) from error
    return {"classes": classes}


# ── the project plane ───────────────────────────────────────────────────────────
#
# Everything above answers about ONE class. These answer about the whole project:
# which experiments exist, what feeds what, and which files nothing reaches.
# `docs/project-graph-plan.md` is the architecture; the same rules bind it — nothing
# is imported or executed, and the workspace is never read from here. The extension
# supplies source text, because it is the only component that knows about trust,
# unsaved buffers and `.gitignore`.

#: Process-wide and content-addressed, so a keystroke re-index ships a manifest and a
#: body rather than the repository. Bounded; see research/admission.py.
_source_cache = admission.SourceCache()

#: Parsed files, held across requests so a re-index re-parses only what changed.
#: Beside the source cache and for the same reason: the manifest/bodies handshake
#: already keeps a keystroke from re-UPLOADING the tree, and this keeps it from
#: re-PARSING it — measured at 1.50s → 0.23s over torchtitan's 596 files, 1.20s →
#: 0.07s over trl's 353. Process-local and bounded; nothing here is persisted.
_parse_cache = admission.ParseCache()


@router.post("/project/index")
def project_index(request: admission.ProjectIndexRequest) -> Dict[str, Any]:
    """Index a workspace folder into one typed graph per experiment.

    Deliberately NOT modelled on ``/multifile/project``'s admission: that endpoint
    caps its source list at 256 entries, which a five-hundred-file research repo
    passes on day one. Here a `manifest` of `{path, digest}` carries the complete
    picture cheaply, and `bodies` carry text only for digests this process does not
    already hold. A request whose manifest names digests we lack is answered with
    ``missing`` — a handshake, not an error — and the extension sends those bodies.
    """
    try:
        manifest, _order = admission.validate_request(
            request, multifile.normalize_workspace_path
        )
        sources, missing = admission.resolve_sources(
            manifest,
            {body.file: body.source for body in request.bodies},
            _source_cache,
            multifile.normalize_workspace_path,
        )
    except admission.AdmissionError as error:
        raise HTTPException(
            status_code=400, detail={"error": error.code, "message": error.message}
        ) from error
    except multifile.MultifileError as error:
        # Path safety is shared with the multi-file tier rather than re-implemented,
        # so one bad path means the same thing on both endpoints.
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_path", "message": str(error)},
        ) from error

    if missing:
        # 200, not an error: the caller is expected to answer with these bodies.
        # Indexing a partially-resolved set would report present files as unreached,
        # which is a lie the user has no way to debug.
        return {"status": "need_bodies", "missing": missing}

    index = indexer.index_project(sources, truncated=request.truncated, parse_cache=_parse_cache)
    return {
        "status": "ok",
        "index": index.model_dump(mode="json"),
        "digest": index.digest(),
        "indexed": len(sources),
        "cached_sources": len(_source_cache),
    }


class ProjectDocumentRequest(admission.ProjectIndexRequest):
    """A project index request, plus which experiment to draw and how far open."""

    experiment_id: str = Field(min_length=1)
    expanded: List[str] = Field(default_factory=list, max_length=4096)


@router.post("/project/document")
def project_document(request: ProjectDocumentRequest) -> Dict[str, Any]:
    """Draw one experiment for the canvas.

    Separate from ``/project/index`` because the two change at different rates: an
    index moves whenever any file does, while a document only needs rebuilding when
    the focused experiment or its expansion set changes. Only expanded frames
    project their contents, which is what keeps a large project cheap to hold.
    """
    try:
        manifest, _order = admission.validate_request(
            request, multifile.normalize_workspace_path
        )
        sources, missing = admission.resolve_sources(
            manifest,
            {body.file: body.source for body in request.bodies},
            _source_cache,
            multifile.normalize_workspace_path,
        )
    except admission.AdmissionError as error:
        raise HTTPException(
            status_code=400, detail={"error": error.code, "message": error.message}
        ) from error
    except multifile.MultifileError as error:
        raise HTTPException(
            status_code=400, detail={"error": "invalid_path", "message": str(error)}
        ) from error

    if missing:
        return {"status": "need_bodies", "missing": missing}

    index = indexer.index_project(sources, truncated=request.truncated, parse_cache=_parse_cache)
    experiment = next(
        (e for e in index.experiments if e.experiment_id == request.experiment_id), None
    )
    if experiment is None:
        # 404 rather than an empty document: "this experiment is gone" and "this
        # experiment is empty" are different answers and the page acts on them
        # differently.
        raise HTTPException(
            status_code=404,
            detail={
                "error": "unknown_experiment",
                "message": f"{request.experiment_id} is not in this index",
            },
        )

    built = document.build_experiment_document(
        experiment.graph,
        name=experiment.label,
        expanded=request.expanded,
        layers_for=lambda node: _layers_for(node, sources),
        steps_for=lambda node: _steps_for(node, sources),
    )
    return {
        "status": "ok",
        "document": built.model_dump(mode="json"),
        "spanmap": built.spanmap,
        "expanded": built.expanded,
        "digest": index.digest(),
    }


def _steps_for(node: Any, sources: Dict[str, str]) -> List[Dict[str, Any]]:
    """An expanded FUNCTION frame's children: the statements of its body (step 5.3).

    Pure AST, like the rest of `research/` — the body is read, never run. A trainer
    that will not parse yields no steps rather than raising: one bad function must
    not blank a project.
    """
    if node.source is None:
        return []
    source = sources.get(node.source.file)
    if source is None:
        return []
    # `definition_id` is `file::qualname`, and qualname is what names the body —
    # `train` for a function, `Trainer.train` for a method.
    qualname = str(node.definition_id).split("::", 1)[-1]
    steps = frames.steps_of(source, qualname)
    if not steps and "." not in qualname:
        return []
    return [
        {
            "id": step.id,
            "label": f"{step.label} → {step.writes}" if step.writes else step.label,
            "type": "step",
            "op": "custom_code",
            "params": {k: v for k, v in (("role", step.role),) if v},
            "line": step.lineno,
            "end_line": step.end_lineno,
            "reads": sorted(step.reads),
        }
        for step in steps
    ]


def _layers_for(node: Any, sources: Dict[str, str]) -> List[Dict[str, Any]]:
    """An expanded model frame's children, from the same parser ``/parse`` uses.

    Static only — the class is read, never imported. A class that will not parse
    yields no children rather than raising: one bad model must not blank a project.
    """
    if node.source is None:
        return []
    source = sources.get(node.source.file)
    if source is None:
        return []
    try:
        parsed = _parser.parse_pytorch_code(
            CodeParseRequest(code=source, file=node.source.file, model_name=node.label)
        )
    except SyntaxError:
        return []
    if not parsed.get("success"):
        return []
    for entry in parsed.get("graphs") or []:
        if entry.get("name") != node.label or entry.get("graph") is None:
            continue
        layers = []
        for raw in entry["graph"].nodes:
            data = getattr(raw, "data", None)
            layer_type = getattr(data, "layerType", None) if data else None
            identifier = str(getattr(raw, "id", ""))
            if identifier in {"input", "output"}:
                continue
            layers.append(
                {
                    "id": identifier,
                    # `or identifier`, not a default argument: the parser answers
                    # `label: None` for a ModuleList, and a None label titled the card
                    # with nothing at all.
                    "label": (getattr(data, "label", None) if data else None) or identifier,
                    "type": getattr(layer_type, "name", "layer") if layer_type else "layer",
                    "op": getattr(layer_type, "id", "custom_code") if layer_type else "custom_code",
                    "params": dict(getattr(data, "parameters", {}) or {}) if data else {},
                    "line": getattr(raw, "lineno", None),
                    "reads": [],
                }
            )
        # The parser's edges are the model's forward pass — `token -> add -> blocks
        # -> norm -> head`. They were read and then dropped, so an open frame laid
        # nine children in a row with nothing joining them, and dagre, given no
        # edges, put every one at rank 0 side by side: a frame two thousand pixels
        # wide for a model that reads top to bottom. Carried in the `reads` form
        # the document already draws for a function frame, so one emitter serves
        # both kinds of frame. Endpoints that are not children (`input`, `output`)
        # are dropped with their edges rather than left dangling.
        present = {layer["id"]: layer for layer in layers}
        for edge in getattr(entry["graph"], "edges", None) or []:
            source, target = str(getattr(edge, "source", "")), str(getattr(edge, "target", ""))
            if source in present and target in present and source != target:
                present[target]["reads"].append(source)
        return layers
    return []


class DeclareRequest(BaseModel):
    """Confirm a guess, or state something the code does not."""

    file: str = Field(min_length=1)
    source: str
    source_digest: str = Field(min_length=64, max_length=64)
    #: 'node' | 'edge' | 'experiment'
    kind: str = Field(min_length=1)
    #: node: [node_kind, label]; edge: [from, to, relationship]; experiment: [name]
    args: List[str] = Field(min_length=1, max_length=3)
    #: Every label the current graph has, so a dangling edge is refused here rather
    #: than written and left to fail the next index's validation.
    known_labels: List[str] = Field(default_factory=list, max_length=4096)


@router.post("/project/declare")
def project_declare(request: DeclareRequest) -> Dict[str, Any]:
    """Write an SDK declaration, and return it as a patch the host reviews.

    This is the step that turns a guess into source text. The indexer infers that
    one node feeds another; the user confirms; this writes the `tensorviz.edge(...)`
    call, and the next index reads it as ``declared`` — which outranks every rule, so
    nothing guesses there again.

    Like ``/apply``, it returns contents rather than writing them: the extension puts
    the change in front of the user as a native diff they accept or discard.
    """
    if projection.digest_of(request.source) != request.source_digest:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "source_stale",
                "message": "the file changed since this declaration was proposed",
            },
        )

    try:
        declaration = _declaration_for(request)
        edit = codegen.declare(
            multifile.normalize_workspace_path(request.file),
            request.source,
            declaration,
            known_labels=request.known_labels or None,
        )
    except codegen.CodegenError as error:
        # A refusal is an answer, not a failure: the reason names what the user
        # would have to change for it to be writable.
        raise HTTPException(
            status_code=400, detail={"error": error.code, "message": error.message}
        ) from error
    except multifile.MultifileError as error:
        raise HTTPException(
            status_code=400, detail={"error": "invalid_path", "message": str(error)}
        ) from error

    return {
        "status": "ok",
        "file": edit.path,
        "contents": edit.contents,
        "expected_digest": edit.expected_digest,
        "line": edit.line,
        "statement": edit.statement,
    }


def _declaration_for(request: DeclareRequest):
    if request.kind == "node" and len(request.args) == 2:
        return codegen.DeclareNode(request.args[0], request.args[1])
    if request.kind == "edge" and len(request.args) in (2, 3):
        relationship = request.args[2] if len(request.args) == 3 else "feeds"
        return codegen.DeclareEdge(request.args[0], request.args[1], relationship)
    if request.kind == "experiment" and len(request.args) == 1:
        return codegen.DeclareExperiment(request.args[0])
    raise codegen.CodegenError(
        "malformed_declaration",
        f"{request.kind!r} with {len(request.args)} argument(s) is not a declaration",
    )


class CompareRequest(admission.ProjectIndexRequest):
    """Two experiments of one index, compared."""

    left: str = Field(min_length=1)
    right: str = Field(min_length=1)


@router.post("/project/compare")
def project_compare(request: CompareRequest) -> Dict[str, Any]:
    """What differs between two experiments.

    Answered from two present-tense graphs — no history, no version store, nothing
    on disk — which is why it fits a read-only extension at all: comparing what two
    experiments *are* needs only that both were indexed.
    """
    try:
        manifest, _order = admission.validate_request(
            request, multifile.normalize_workspace_path
        )
        sources, missing = admission.resolve_sources(
            manifest,
            {body.file: body.source for body in request.bodies},
            _source_cache,
            multifile.normalize_workspace_path,
        )
    except admission.AdmissionError as error:
        raise HTTPException(
            status_code=400, detail={"error": error.code, "message": error.message}
        ) from error

    if missing:
        return {"status": "need_bodies", "missing": missing}

    index = indexer.index_project(sources, truncated=request.truncated, parse_cache=_parse_cache)
    by_id = {e.experiment_id: e for e in index.experiments}
    for wanted in (request.left, request.right):
        if wanted not in by_id:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "unknown_experiment",
                    "message": f"{wanted} is not in this index",
                },
            )
    if request.left == request.right:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "same_experiment",
                "message": "an experiment does not differ from itself",
            },
        )

    diff = compare.compare(by_id[request.left], by_id[request.right])
    return {"status": "ok", "diff": diff.model_dump(mode="json")}
