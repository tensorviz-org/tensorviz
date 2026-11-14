"""Model ingestion API: import a library architecture or a repository as a graph.

Sits alongside ``/api/code/parse`` (the single-file AST importer) but works on a different
primitive: instead of parsing source text it *instantiates* the model and interrogates the
object, which is what makes multi-file packages and whole repositories reachable.

Ingestion executes model-definition code — that is inherent to instantiating a model, and
is why the repository endpoints are documented as local-trust operations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.pytorch_requirement import load as load_pytorch_service
from auth import CurrentUser, get_current_user
from services.ingest.document import to_document
from services.ingest.repo_scan import StubbedDependencyError, rank_entrypoints, scan_repo
from services.ingest.spec import ModelRef
from services.store_factory import get_project_store

# services.ingest.service and .resolvers import PyTorch at module scope. They are
# loaded by the routes that instantiate models, so the server and the static
# repository scan below start without PyTorch.

router = APIRouter(prefix="/api/ingest", tags=["ingest"])
_store = get_project_store()

DEFAULT_PROJECT_NAME = "Imported models"


# `repo` and `entrypoint` are filesystem paths, so their ceiling is PATH_MAX;
# everything else here is a library/architecture identifier or a display name.
_MAX_PATH = 4096
_MAX_NAME = 200


class IngestModelRequest(BaseModel):
    library: str = Field(max_length=64, description="torchvision | transformers | diffusers | timm")
    arch: str = Field(min_length=1, max_length=256)
    depth: int = Field(default=2, ge=1, le=12)
    name: Optional[str] = Field(default=None, max_length=_MAX_NAME)
    overrides: Dict[str, Any] = Field(default_factory=dict)
    save: bool = Field(default=True, description="persist as a graph and return its id")
    project_id: Optional[str] = None


class IngestRepoRequest(BaseModel):
    repo: str = Field(min_length=1, max_length=_MAX_PATH, description="path to a local repository")
    entrypoint: Optional[str] = Field(
        default=None, max_length=_MAX_PATH, description="omit to use the top-ranked candidate"
    )
    depth: int = Field(default=2, ge=1, le=12)
    name: Optional[str] = Field(default=None, max_length=_MAX_NAME)
    overrides: Dict[str, Any] = Field(default_factory=dict)
    save: bool = True
    project_id: Optional[str] = None


class ScanRepoRequest(BaseModel):
    repo: str = Field(min_length=1, max_length=_MAX_PATH)


@router.get("/libraries")
def get_libraries(user: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    """Which model libraries this deployment can ingest from.

    ``torchvision`` ships with the backend requirements; the others are optional extras.
    """
    resolvers = load_pytorch_service("services.ingest.resolvers", "Model library import")
    return {"libraries": resolvers.available_libraries()}


@router.get("/catalog")
def get_catalog(
    library: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None, description="substring filter on the arch name"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """Paginated architecture index. Paginated on purpose: the full index across all four
    libraries is thousands of entries and must not be served as one blob."""
    resolvers = load_pytorch_service("services.ingest.resolvers", "Model library import")
    entries = resolvers.catalog(library)
    if q:
        needle = q.lower()
        entries = [e for e in entries if needle in e.arch.lower()]
    total = len(entries)
    window = entries[offset : offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": [e.model_dump() for e in window],
    }


@router.post("/model")
def ingest_model(
    request: IngestModelRequest, user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Import an architecture from an installed library. No weights are downloaded."""
    ref = ModelRef(
        library=request.library,
        arch=request.arch,
        depth=request.depth,
        name=request.name,
        overrides=request.overrides,
    )
    result = _run(ref)
    return _respond(result, user.uid, request.save, request.project_id)


@router.post("/repo/scan")
def scan_repository(
    request: ScanRepoRequest, user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Rank a repository's candidate model classes WITHOUT importing or executing it.

    Ranking is a heuristic and deliberately not auto-applied: training and loss wrappers
    legitimately outrank backbones because they are what ``train.py`` instantiates, so the
    caller confirms which candidate to import.
    """
    try:
        # services.ingest.scan_only, without loading the PyTorch-backed service module.
        candidates = rank_entrypoints(scan_repo(request.repo))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"repository not found: {request.repo}")
    return {"candidates": [c.model_dump() for c in candidates]}


@router.post("/repo")
def ingest_repository(
    request: IngestRepoRequest, user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Import a model from a local repository. Executes that repository's code."""
    ref = ModelRef(
        repo=request.repo,
        entrypoint=request.entrypoint,
        depth=request.depth,
        name=request.name,
        overrides=request.overrides,
    )
    result = _run(ref)
    if not result.ok:
        # Constructor arguments could not be recovered: return the ranked alternatives so
        # the caller can choose another entrypoint or supply overrides.
        return {
            "ok": False,
            "error": result.error,
            "candidates": [c.model_dump() for c in result.candidates],
            "diagnostics": result.diagnostics.model_dump(),
        }
    return _respond(result, user.uid, request.save, request.project_id)


def _run(ref: ModelRef):
    service = load_pytorch_service("services.ingest.service", "Model import")
    from services.ingest.resolvers import MissingLibraryError

    try:
        return service.ingest(ref)
    except service.IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except StubbedDependencyError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except MissingLibraryError as exc:
        # Not a server fault and not a bad request: this deployment simply lacks an
        # optional extra. Its own message already carries the install command.
        raise HTTPException(status_code=422, detail=str(exc))
    except (ImportError, ModuleNotFoundError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"a dependency required to define this model is unavailable: {exc}",
        )
    except Exception as exc:  # surfaced verbatim: ingestion failures are diagnostic
        raise HTTPException(status_code=500, detail=f"ingestion failed: {type(exc).__name__}: {exc}")


def _respond(result, uid: str, save: bool, project_id: Optional[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": result.ok,
        "name": result.name,
        "graph": result.graph,
        "diagnostics": result.diagnostics.model_dump(),
        "candidates": [c.model_dump() for c in result.candidates],
    }
    if not save:
        return payload

    pid = project_id or _default_project(uid)
    document = to_document(result.name, result.graph)
    graph = _store.create_graph(
        uid,
        pid,
        result.name,
        document.model_dump(mode="json"),
        author="user",
        op_summary=f"ingested {result.diagnostics.provenance}",
    )
    if graph is None:
        raise HTTPException(status_code=404, detail=f"project not found: {pid}")
    payload["project_id"] = pid
    payload["graph_id"] = graph["id"]
    payload["builder_path"] = f"/builder/{graph['id']}"
    return payload


def _default_project(uid: str) -> str:
    """Save into the caller's FIRST existing project.

    Deliberately not a dedicated "Imported models" project: the builder deep-link
    ``/builder/{graph_id}`` is resolved by the client against the project it currently has
    selected, so a graph parked in a project the client does not know about 404s on open.
    A new project is created only when the user has none at all.
    """
    projects = _store.list_projects(uid)
    if projects:
        return projects[0]["id"]
    return _store.create_project(uid, DEFAULT_PROJECT_NAME)["id"]
