"""
Refactored FastAPI server for PyTorch model visualization.
Uses modular backend architecture for better maintainability.
"""
import logging
import os
import platform
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from middleware import (
    BodySizeLimitMiddleware,
    InProcessIngestDisabledMiddleware,
    LocalTokenMiddleware,
    LoggingMiddleware,
    LoopbackHostMiddleware,
    install_log_redaction,
    local_token_matches,
)

# Import modular services. Nothing imported here may load PyTorch: the server and
# its static routes start without it. Routes that run tensors load their service
# through api.pytorch_requirement, which answers 503 when PyTorch cannot load.
from api.pytorch_requirement import load as load_pytorch_service
from core.graph_input_safety import UnsafeGraphInput
from services.template_service import TemplateService
from services.code_generation import CodeGenerationService
from services.model_roadmap import get_complex_model_roadmap
from models.graph_models import (
    CodeParseRequest,
    CustomizeModelRequest,
    ExportCodeRequest,
    ExportOnnxRequest,
    ExportPackageRequest,
    GenerateCodeRequest,
    Graph,
    GraphPayloadRequest,
    HierarchicalGraph,
    InstantiateModuleTemplateRequest,
    InstantiatePopularModelRequest,
    InstantiateTemplateRequest,
    VerifyModelRequest,
)
from utils.model_validation import ModelValidationService
from utils.code_parser import CodeParserService


# Load environment variables from project .env at process startup. Also honor
# backend/.env for local backend-only secrets.
_project_root = Path(__file__).resolve().parents[1]
load_dotenv(_project_root / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

from api.routers import (
    diff,
    ide,
    ingest,
    local_browser,
    user_templates,
    validate,
)
from auth import get_current_user


# Read before constructing the app: IDE/token mode does not publish schemas.
_local_token = os.getenv("TENSORVIZ_LOCAL_TOKEN", "").strip()
_publish_schema = not _local_token
app = FastAPI(
    title="PyTorch Model Visualizer API",
    version="2.0.0",
    docs_url="/docs" if _publish_schema else None,
    redoc_url="/redoc" if _publish_schema else None,
    openapi_url="/openapi.json" if _publish_schema else None,
)
logger = logging.getLogger("uvicorn.error")
# CORS: env-driven allowlist (comma-separated), default the local frontend.
_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
# Registration order is reversed by Starlette, so the FIRST registered is the
# innermost. The body-size guard goes first on purpose: wrapped by CORS its 413 is
# readable in the browser, and wrapped by the logger the rejection still appears in
# the request log.
app.add_middleware(BodySizeLimitMiddleware)
# IDE mode (task.md M2, decisions D2/D12). Both are conditional on the launcher
# having set TENSORVIZ_LOCAL_TOKEN, so a backend started by start.sh, by `make
# e2e` or by the test suite is byte-for-byte the app it was before. They are
# registered here, between the body guard and CORS, so a CORS preflight is
# answered above them (a 401 on OPTIONS would break every browser request) and
# a rejection is still logged and still readable in the browser.
if _local_token:
    app.add_middleware(InProcessIngestDisabledMiddleware)
    app.add_middleware(LocalTokenMiddleware, token=_local_token)
app.add_middleware(CORSMiddleware, allow_origins=_cors_origins, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(LoggingMiddleware)
# Outermost: a rebinding Host is refused before it is logged as a real request.
app.add_middleware(LoopbackHostMiddleware, token_configured=bool(_local_token))

# Before uvicorn serves anything: the launch token rides the canvas frame URL as a
# query parameter, and uvicorn's access log would print it in the clear. The
# extension pipes every backend line into the user-visible tensorViz output channel,
# which VS Code persists to disk and which is the first thing a person copies into a
# bug report — so an unredacted line is the secret leaving the machine at the moment
# someone asks for help.
install_log_redaction()

# The production React build, served by this process so the canvas has an http://
# origin (tldraw 5 treats a vscode-webview: origin as unlicensed production and
# unmounts the editor five seconds in — task.md D2). Read here, mounted at the
# very bottom of this module: a mount on "/" swallows every route registered
# after it.
_static_dir = os.getenv("TENSORVIZ_STATIC_DIR", "").strip()

# Auth: gate the stateful/mutating routers. get_current_user authenticates the
# caller (uid); per-uid data scoping is enforced in the stores (Phase 1.2).
_auth = [Depends(get_current_user)]
app.include_router(validate.router, prefix="/api", dependencies=_auth)
app.include_router(diff.router, dependencies=_auth)
app.include_router(user_templates.router, dependencies=_auth)  # personal reusable templates
app.include_router(ingest.router, dependencies=_auth)  # import installed models / repositories
app.include_router(ide.router, dependencies=_auth)  # code <-> graph projection for the IDE (task.md M3)
# The local-browser broker is NOT in the _auth group: the browser cannot hold
# the per-launch token, so the router enforces its own session-scoped
# credential model (mint guarded by the token; bootstrap and socket by the
# session cookie). See api/routers/local_browser.py.
app.include_router(local_browser.router)

# Initialize services
template_service = TemplateService()
code_generation_service = CodeGenerationService()
model_validation_service = ModelValidationService()
code_parser_service = CodeParserService()

# Pydantic's built-in validation messages are static templates that never embed
# the submitted value ("Field required", "Input should be a valid string",
# "String should have at most 8000 characters"), so returning them verbatim is
# safe and is the only way the caller learns WHICH rule it broke. Messages from
# custom validators (value_error / assertion_error) interpolate arbitrary request
# content, so those stay redacted — see tests/api/test_server_validation_privacy.py.
# `ctx` and `input` are never echoed for any type: ctx can carry the input value.
_SAFE_VALIDATION_ERROR_TYPES = frozenset({
    "missing", "string_type", "string_too_short", "string_too_long",
    "string_pattern_mismatch", "int_type", "int_parsing", "float_type",
    "float_parsing", "bool_type", "bool_parsing", "list_type", "dict_type",
    "model_type", "model_attributes_type", "extra_forbidden",
    "greater_than", "greater_than_equal", "less_than", "less_than_equal",
    "too_short", "too_long", "literal_error", "enum", "json_invalid",
    "union_tag_invalid", "union_tag_not_found",
})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    safe_errors = []
    for error in exc.errors():
        error_type = str(error.get("type", "invalid_request"))
        safe_errors.append({
            "type": error_type,
            "loc": [str(part) for part in error.get("loc", ())],
            "msg": (
                str(error.get("msg", ""))
                if error_type in _SAFE_VALIDATION_ERROR_TYPES
                else "Invalid request field"
            ),
        })
    logger.warning(
        "http.request.validation_failed method=%s path=%s fields=%s",
        request.method,
        request.url.path,
        [".".join(error["loc"]) for error in safe_errors],
    )
    return JSONResponse(
        status_code=422,
        content={"detail": safe_errors},
    )


@app.get("/health")
def health_check(request: Request):
    """Health check endpoint.

    Also reports the environment this backend is actually running in. The UI's
    sidebar states that the backend is local and that no weights are downloaded;
    that claim is only checkable if the versions and the device come from the
    running interpreter rather than from a string in the frontend. Every field is
    computed defensively — a missing optional library must degrade the readout,
    not fail the health check that tells the UI whether the backend is up.

    In token mode the extension probes this route before it can send a header, so
    the probe stays open. Torch and Python versions are omitted until the launch
    token is presented: they are a fingerprint of the machine, not a readiness bit.
    """
    authenticated = not _local_token or local_token_matches(request.headers, _local_token)
    try:
        import torch

        torch_version = torch.__version__
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        torch_version = None
        device = None

    try:
        from services.ingest.resolvers import available_libraries

        libraries = sorted(name for name, present in available_libraries().items() if present)
    except Exception:
        libraries = []

    environment = {"device": device}
    if authenticated:
        environment = {
            "torch": torch_version,
            "device": device,
            "python": platform.python_version(),
            "libraries": libraries,
        }

    return {
        "status": "healthy",
        "backend": "canvas-backend-v2",
        "features": ["hierarchical_models", "template_system", "code_generation"],
        "environment": environment,
        # What this process can do, so a caller does not have to infer it from a
        # 404. `ide` means the per-launch token is enforced on /api/*; `static`
        # means the React build is served from this origin.
        "capabilities": {
            "ide": bool(_local_token),
            "static": bool(_static_dir),
        },
    }


@app.get("/api/roadmap/complex-models")
def get_complex_models_roadmap():
    """Recommended high-priority model families and validations to implement next."""
    return get_complex_model_roadmap()


# Template and Model Discovery Endpoints
@app.get("/api/templates/categories")
def get_template_categories():
    """Get all available template categories"""
    try:
        return {"categories": template_service.get_all_categories()}
    except Exception:
        logger.exception("get_template_categories failed")
        raise HTTPException(status_code=500, detail="Error fetching categories")


def _buildable_template(template_id: str) -> bool:
    """Whether a catalogue template can actually be turned into code.

    Asks the real question — the same structural validation `/api/models/generate-code`
    runs before it emits anything — rather than a proxy. A first attempt checked whether
    every node's op resolved in the op registry, which is the right rule for the canonical
    blocks and the wrong one here: these graphs are hierarchical, and reading ops out of
    them dropped thirteen templates that generate perfectly well.

    The reason for filtering at all is F5-TTS: `sinusoidal_pos_embedding` and `grn` arrive
    carrying a name and a colour and no dimensions, so there is nothing to build them from,
    and the template could be placed and never exported or run. The owner's call on the
    canonical blocks applies for the same reason.

    Cached: the answer cannot change while the process lives, and validation is the
    expensive half.
    """
    cached = _BUILDABLE_TEMPLATE_CACHE.get(template_id)
    if cached is not None:
        return cached
    ok = True
    try:
        graph = template_service.instantiate_template(template_id, {})
        payload = graph.dict() if hasattr(graph, "dict") else graph
        result = _structural_validation(payload, [1, 3, 224, 224])
        ok = bool(result.get("valid"))
    except Exception:
        # Unbuildable for some other reason is still unbuildable, but say so by omission
        # rather than by failing the whole catalogue request.
        ok = False
    _BUILDABLE_TEMPLATE_CACHE[template_id] = ok
    return ok


#: template_id -> whether it can be turned into code. Filled on first use.
_BUILDABLE_TEMPLATE_CACHE: Dict[str, bool] = {}


def _only_real_modules(entries):
    """Drop module blocks whose card would 404 when clicked.

    Twelve of the fifteen advertised ids — `resnet_block`, `lstm_cell`,
    `attention_mechanism`, `generator_block` and the rest — resolve to nothing. They read
    like an older naming scheme the module catalogue moved past, and nothing ever
    reconciled the two.

    The authority is `get_module_template_by_id`, which is what
    `/api/module-templates/{id}/instantiate` itself calls, so the listing cannot advertise
    a card that route would refuse. A first attempt filtered against the CANONICAL registry
    instead — a different catalogue that backs a different endpoint — and dropped all
    fifteen, including `multi_head_attention`, `feed_forward_network` and
    `transformer_encoder_block`, which instantiate perfectly well. Ask the endpoint's own
    authority, not a neighbouring one.
    """
    if not isinstance(entries, dict):
        return entries
    return {
        mid: meta
        for mid, meta in entries.items()
        if template_service.get_module_template_by_id(mid) is not None
    }


def _buildable_popular(model_id: str) -> bool:
    """The same question for a popular architecture, which instantiates on its own route."""
    cached = _BUILDABLE_POPULAR_CACHE.get(model_id)
    if cached is not None:
        return cached
    ok = True
    try:
        graph = template_service.instantiate_popular_model(model_id, {})
        payload = graph.dict() if hasattr(graph, "dict") else graph
        ok = bool(_structural_validation(payload, [1, 3, 224, 224]).get("valid"))
    except Exception:
        ok = False
    _BUILDABLE_POPULAR_CACHE[model_id] = ok
    return ok


#: model_id -> whether it can be turned into code. Filled on first use.
_BUILDABLE_POPULAR_CACHE: Dict[str, bool] = {}


def _only_buildable_popular(entries):
    """Drop popular architectures that cannot be turned into code (f5tts)."""
    if not isinstance(entries, dict):
        return entries
    return {mid: meta for mid, meta in entries.items() if _buildable_popular(mid)}


def _only_buildable(entries):
    """Drop catalogue entries that cannot be turned into code."""
    if not isinstance(entries, dict):
        return entries
    return {tid: meta for tid, meta in entries.items() if _buildable_template(tid)}


@app.get("/api/templates/categories/{category}")
def get_templates_by_category(category: str):
    """Get templates for a specific category"""
    try:
        templates = _only_buildable(template_service.get_templates_by_category(category))
        popular_models = _only_buildable_popular(template_service.get_popular_models_by_category(category))
        module_templates = _only_real_modules(template_service.get_module_templates_by_category(category))
        
        return {
            "category": category,
            "templates": templates,
            "popular_models": popular_models,
            "module_templates": module_templates
        }
    except Exception:
        logger.exception("get_templates_by_category failed")
        raise HTTPException(status_code=500, detail="Error fetching templates")


@app.get("/api/popular-models")
def get_all_popular_models():
    """Get all popular models across categories"""
    try:
        return {"models": template_service.get_all_popular_models()}
    except Exception:
        logger.exception("get_all_popular_models failed")
        raise HTTPException(status_code=500, detail="Error fetching popular models")


@app.get("/api/popular-models/{model_id}")
def get_popular_model_details(model_id: str):
    """Get detailed information about a specific popular model"""
    try:
        model = template_service.get_popular_model_by_id(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Popular model '{model_id}' not found")
        
        # Get customization options for the model's template
        customization_options = template_service.get_customization_options(model.template_id)
        
        return {
            "model": model,
            "customization_options": customization_options
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("get_popular_model_details failed")
        raise HTTPException(status_code=500, detail="Error fetching model details")


# Template Instantiation Endpoints
@app.post("/api/templates/{template_id}/instantiate")
def instantiate_template(template_id: str, request: InstantiateTemplateRequest):
    """Instantiate a template with custom configuration"""
    try:
        config = request.config

        # Validate configuration
        errors = template_service.validate_customization(template_id, config)
        if errors:
            raise HTTPException(status_code=400, detail=f"Configuration errors: {'; '.join(errors)}")
        
        # Instantiate template
        hierarchical_graph = template_service.instantiate_template(template_id, config)
        
        return {
            "success": True,
            "graph": hierarchical_graph.dict(),
            "template_id": template_id,
            "config": config
        }
    except HTTPException:
        raise
    except (TypeError, AttributeError) as e:
        # A config VALUE the template cannot use is the caller's mistake, and it
        # used to be a 500 echoing the raw exception ("'str' object has no
        # attribute 'items'"). ValueError is deliberately NOT here: with `config`
        # typed, the remaining ValueErrors come from the template builders
        # themselves, and a template bug has to stay a logged 500.
        raise HTTPException(status_code=400, detail=f"Invalid template configuration: {e}")
    except Exception:
        logger.exception("instantiate_template failed unexpectedly template_id=%s", template_id)
        raise HTTPException(status_code=500, detail="Error instantiating template")


@app.post("/api/popular-models/{model_id}/instantiate")
def instantiate_popular_model(model_id: str, request: InstantiatePopularModelRequest):
    """Instantiate a popular model with optional customizations.

    Accepts either `customizations` (preferred, semantic) or `config` (legacy
    key the frontend has sent historically). Whichever is present is treated
    as the overrides applied on top of the model's default config.
    """
    # An unknown id is a 404 here for the same reason it already is on
    # GET /api/popular-models/{model_id}: the two disagreed, so the identical bad
    # id answered 404 on the read and 500 on the instantiate.
    if not template_service.get_popular_model_by_id(model_id):
        raise HTTPException(status_code=404, detail=f"Popular model '{model_id}' not found")
    try:
        customizations = request.customizations or request.config or {}

        hierarchical_graph = template_service.instantiate_popular_model(model_id, customizations)

        return {
            "success": True,
            "graph": hierarchical_graph.dict(),
            "model_id": model_id,
            "customizations": customizations
        }
    except HTTPException:
        raise
    except (TypeError, AttributeError) as e:
        # Same split as instantiate_template: a bad customization value is 400, a
        # ValueError from the template registry stays a logged 500 (the only
        # client-facing one — an unknown model id — is pre-checked above).
        raise HTTPException(status_code=400, detail=f"Invalid customizations: {e}")
    except Exception:
        logger.exception("instantiate_popular_model failed unexpectedly model_id=%s", model_id)
        raise HTTPException(status_code=500, detail="Error instantiating popular model")


# Model Customization Endpoints
@app.post("/api/models/customize")
def customize_model(request: CustomizeModelRequest):
    """Apply customizations to an existing model"""
    try:
        base_graph_data = request.base_graph
        customizations = request.customizations

        if not base_graph_data:
            raise HTTPException(status_code=400, detail="base_graph is required")

        # Convert to HierarchicalGraph object
        base_graph = HierarchicalGraph(**base_graph_data)

        # Apply customizations
        customized_graph = template_service.customize_model(base_graph, customizations)

        return {"graph": customized_graph.dict()}
    except HTTPException:
        raise
    except (ValueError, TypeError, AttributeError) as e:
        # HierarchicalGraph(**base_graph) rejects a malformed graph with a
        # pydantic ValidationError (a ValueError). That is a client error, not the
        # 500 it used to be.
        raise HTTPException(status_code=400, detail=f"Malformed graph: {_malformed_graph_detail(e)}")
    except Exception:
        logger.exception("customize_model failed unexpectedly")
        raise HTTPException(status_code=500, detail="Error customizing model")


@app.get("/api/templates/{template_id}/customization-options")
def get_customization_options(template_id: str):
    """Get available customization options for a template"""
    try:
        options = template_service.get_customization_options(template_id)
        if not options:
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
        
        return {"options": options}
    except HTTPException:
        raise
    except Exception:
        logger.exception("get_customization_options failed")
        raise HTTPException(status_code=500, detail="Error fetching customization options")


# Shape Inference Endpoints
@app.post("/api/models/infer-shapes")
def infer_shapes(request: GraphPayloadRequest):
    """Infer output shapes for all nodes in the graph"""
    try:
        graph_dict = request.graph
        if not graph_dict:
            raise HTTPException(status_code=400, detail="Graph data is required")
        # An HTTPException, so the clause below passes it through as a 503.
        shapes = load_pytorch_service("core.shape_inference", "Shape inference")

        graph_dict = _convert_to_hierarchical_graph(graph_dict)
        input_shape = request.input_shape or [1, 3, 224, 224]

        if "modules" in graph_dict or "node_hierarchy" in graph_dict:
            hierarchical_graph = HierarchicalGraph(**graph_dict)
            
            # infer_hierarchical_shapes returns a DICT ({"shapes": ..., "errors": ...}),
            # not a 2-tuple. Unpacking it yielded its KEYS, so this endpoint used to
            # answer {"shapes": "shapes", "output_shape": "errors"} for every
            # hierarchical graph. It has no output shape to give — that comes from the
            # dry forward pass in verification_service.
            inferred = shapes.infer_hierarchical_shapes(hierarchical_graph, input_shape)
            result = {
                "success": True,
                "shapes": inferred.get("shapes", {}),
                "output_shape": None,
                "errors": inferred.get("errors") or [],
            }
        else:
            result = shapes.infer_node_shapes(
                Graph(**graph_dict), input_shape
            )
            # Make sure success field exists for frontend compatibility
            if "success" not in result:
                result["success"] = len(result.get("errors", [])) == 0
        
        return result
    except HTTPException:
        # This clause was missing, and it was the only endpoint in the file
        # without it: the 400 raised above for an empty graph was swallowed by
        # `except Exception` and re-emitted as
        # `500 "Internal error during shape inference: 400: Graph data is required"`.
        raise
    except (ValueError, TypeError, AttributeError) as e:
        raise HTTPException(status_code=400, detail=f"Shape inference error: {str(e)}")
    except Exception:
        logger.exception("infer_shapes failed unexpectedly")
        raise HTTPException(status_code=500, detail="Internal error during shape inference")


# Code Generation Endpoints
@app.post("/api/models/generate-code")
def generate_code(request: GenerateCodeRequest):
    """Generate PyTorch code from graph representation"""
    try:
        graph = request.graph
        _require_nodes(graph, "Graph data")

        input_shape = request.input_shape or [1, 3, 224, 224]
        class_name = request.class_name or "GeneratedModel"

        # The product's promise is runnable PyTorch. Handing back code built from a
        # graph the validator rejects — Linear(in_features=-5) generated a file that
        # raises "negative dimension" on the first line that touches it — breaks that
        # promise while looking like it kept it.
        _refuse_if_invalid(graph, input_shape, "turned into code")

        # Prefer modular codegen: factor repeated blocks into reusable ×N modules
        # (e.g. F5-TTS -> DiTBlock ×22 via nn.ModuleList). Works whether the
        # canvas sends a flat graph or a frontend display-grouped one (the
        # modularizer flattens groups first). Falls back to the legacy
        # reconstruction when nothing is groupable.
        if isinstance(graph, dict) and graph.get("nodes") is not None \
                and "root_nodes" not in graph and "modules" not in graph:
            try:
                from services.graph_modularizer import modularize_for_codegen
                hg_mod = modularize_for_codegen(graph)
            except Exception:
                hg_mod = None
            if hg_mod is not None:
                code = code_generation_service.generate_hierarchical_code(
                    hg_mod, input_shape, class_name
                )
                # Report what the export could NOT reproduce. Imported graphs contain
                # library-private composites whose bodies are not in the graph, so the
                # generated file is a partial reconstruction; the caller must be able to
                # say so rather than presenting it as the model.
                unsupported = code_generation_service.unreproducible_modules(
                    hg_mod.root_nodes, hg_mod.modules
                )
                return {
                    "code": code,
                    "summary": _calculate_model_summary(graph, input_shape),
                    "unsupported": unsupported,
                    "complete": not unsupported,
                    "metadata": {
                        "generated_at": _now_iso8601(),
                        "generator_version": "2.0.0",
                        "graph_type": "modular",
                    },
                }

        graph = _convert_to_hierarchical_graph(graph)

        # Detect graph type and generate code
        unsupported = []
        if "modules" in graph or "node_hierarchy" in graph:
            hierarchical_graph = HierarchicalGraph(**graph)
            code = code_generation_service.generate_hierarchical_code(
                hierarchical_graph, input_shape, class_name
            )
            unsupported = code_generation_service.unreproducible_modules(
                hierarchical_graph.root_nodes, hierarchical_graph.modules
            )
        else:
            flat_graph = Graph(**graph)
            code = code_generation_service.generate_flat_code(
                flat_graph, input_shape, class_name
            )
            # The hierarchical branch above has always asked; this one never did,
            # so a flat graph answered `complete: true` even when a node could not
            # be reproduced. Nothing in the response said which node, because the
            # list it would have named was left empty. Silence read as success.
            unsupported = code_generation_service.unreproducible_modules(flat_graph.nodes)

        # Calculate model summary
        summary = _calculate_model_summary(graph, input_shape)

        return {
            "code": code,
            "summary": summary,
            "unsupported": unsupported,
            "complete": not unsupported,
            "metadata": {
                "generated_at": _now_iso8601(),
                "generator_version": "2.0.0",
                "graph_type": "hierarchical" if "modules" in graph else "flat"
            }
        }
    except HTTPException:
        raise
    except UnsafeGraphInput as e:
        # A layer the graph names but the registry does not know is a client error.
        # It used to surface as a 500 ("layer operation 'FooBar' is not a supported
        # operation"), i.e. a server fault for the caller's typo.
        raise HTTPException(status_code=400, detail=str(e))
    except (ValueError, TypeError, AttributeError) as e:
        raise HTTPException(status_code=400, detail=f"Malformed graph: {_malformed_graph_detail(e)}")
    except Exception:
        logger.exception("generate_code failed unexpectedly")
        raise HTTPException(status_code=500, detail="Error generating code")


# Graph endpoints and module wrappers: present on the canvas, but not something
# code can be generated FROM. A graph of nothing but these has no model in it.
_NON_LAYER_NODE_TYPES = frozenset({"inputNode", "outputNode"})
_NON_LAYER_LAYER_IDS = frozenset({"input", "output"})


def _malformed_graph_detail(exc: Exception) -> str:
    """Say WHAT is wrong with the payload without quoting the payload back.

    `str(pydantic.ValidationError)` embeds `input_value=…` for every failure, so a
    plain f-string here echoed submitted request content into the response body.
    That only ever reaches the requester, but it is the second, unredacted path in
    a service that deliberately redacts the messages a custom validator produces
    for exactly this reason — and having one rule with an exception nobody wrote
    down is how the first leak happens.

    Field path and error type are what a caller needs to fix their request; the
    value they just sent is not.
    """
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    try:
        parts = [
            f"{'.'.join(str(piece) for piece in error.get('loc', ())) or '(body)'}: "
            f"{error.get('type', 'invalid')}"
            for error in errors()
        ]
    except Exception:  # pragma: no cover - defensive; never fail while reporting a failure
        return "the payload does not match the expected graph shape"
    return "; ".join(parts) or "the payload does not match the expected graph shape"


def _require_nodes(graph: Any, what: str = "graph") -> None:
    """Refuse a request that has no model in it.

    `if not graph` only caught a missing or empty body: `{"nodes": [], "edges": []}`
    is a truthy dict, so an empty canvas sailed through code generation and came
    back as a do-nothing nn.Module stamped `complete: true` with zero warnings.

    "No model in it" means no computational layer, not zero nodes. A canvas holding
    only an Input node — or only an Input and an Output — produced exactly the same
    `def forward(self, x): return x` stamped complete:true, which is the symptom the
    finding is named after one step removed from the payload it quoted.
    """
    if not graph:
        raise HTTPException(status_code=400, detail=f"{what} is required")
    if not isinstance(graph, dict):
        return

    nodes = graph.get("nodes")
    if nodes is None:
        nodes = graph.get("root_nodes")
    if not isinstance(nodes, list):
        return

    def _is_layer(node: Any) -> bool:
        if not isinstance(node, dict):
            return True  # not ours to judge; the parser will reject it
        if node.get("type") in _NON_LAYER_NODE_TYPES:
            return False
        layer_id = (node.get("data") or {}).get("layerType", {})
        layer_id = (layer_id.get("id") if isinstance(layer_id, dict) else "") or ""
        return layer_id.strip().lower() not in _NON_LAYER_LAYER_IDS

    if not any(_is_layer(node) for node in nodes):
        raise HTTPException(
            status_code=400,
            detail=f"{what} has no layers — there is nothing to generate.",
        )


def _structural_validation(graph: Dict[str, Any], input_shape: List[int]) -> Dict[str, Any]:
    """The one place that decides whether a graph is structurally sound.

    Shared by /api/models/validate and by the two endpoints that used to trust
    their input independently — which is how a graph the validator flagged could
    still be handed to a user as a zip.
    """
    converted = _convert_to_hierarchical_graph(graph)
    if "modules" in converted or "node_hierarchy" in converted:
        hierarchical_graph = HierarchicalGraph(**converted)
        errors = model_validation_service.validate_hierarchical_graph(
            hierarchical_graph, input_shape
        )
        root_graph = Graph(
            nodes=hierarchical_graph.root_nodes, edges=hierarchical_graph.root_edges
        )
    else:
        root_graph = Graph(**converted)
        errors = model_validation_service.validate_flat_graph(root_graph, input_shape)

    # Warnings used to be computed and discarded here ("Could add warnings in the
    # future"), which is how a graph with no Input node — validated against an
    # assumed [1, 3, 224, 224] nobody chose — came back as an unqualified
    # valid:true. `valid` still means "no errors"; the warnings say what the tick
    # does not cover.
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": model_validation_service.warnings_for_graph(root_graph),
    }


def _refuse_if_invalid(graph: Dict[str, Any], input_shape: List[int], action: str) -> None:
    """422 rather than shipping something the validator already knows is broken.

    A payload too malformed to even parse into a Graph is the caller's mistake, so
    it answers 400 here rather than escaping as an unhandled 500. pydantic's
    ValidationError is a ValueError, which is why one clause covers both.
    """
    try:
        result = _structural_validation(graph, input_shape)
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=400, detail=f"Malformed graph: {_malformed_graph_detail(exc)}")
    if not result["valid"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    f"This model cannot be {action}: it does not pass validation. "
                    "Fix the problems below and try again."
                ),
                "errors": result["errors"],
                "warnings": result["warnings"],
            },
        )


# Model Validation Endpoints
@app.post("/api/models/validate")
def validate_model(request: GraphPayloadRequest):
    """Validate model for common errors and inconsistencies"""
    try:
        graph = request.graph
        if not graph:
            raise HTTPException(status_code=400, detail="Graph data is required")

        input_shape = request.input_shape or [1, 3, 224, 224]
        return _structural_validation(graph, input_shape)
    except HTTPException:
        raise
    except (ValueError, TypeError, AttributeError) as e:
        # A malformed payload is the caller's mistake, not a server fault. This
        # used to be a 500 carrying the raw exception text ("'str' object has no
        # attribute 'get'"), which poisons error monitoring and leaks internals.
        raise HTTPException(status_code=400, detail=f"Malformed graph: {_malformed_graph_detail(e)}")
    except Exception as e:
        logger.exception("validate_model failed unexpectedly")
        raise HTTPException(status_code=500, detail="Error validating model")


# ─── Verification + Export Package ──────────────────────────────────────
@app.post("/api/models/verify", dependencies=[Depends(get_current_user)])
def verify_model_endpoint(request: VerifyModelRequest):
    """Run shape inference + a synthetic torch forward pass on the graph.

    Body: { graph, input_shape?, run_forward? }
    """
    graph = request.graph
    if not graph:
        raise HTTPException(status_code=400, detail="graph is required")
    # An empty canvas used to come back ok:true, stage:"complete", layer_count:0.
    # "Verified" is the strongest claim this product makes about a model, and the
    # builder gates Export Code on it, so letting it mean "nothing here, and that
    # ran fine" is the same lie Q-m3 is about — just on the plane the user trusts
    # most.
    _require_nodes(graph, "graph")
    verification = load_pytorch_service("services.verification_service", "Model verification")
    try:
        return verification.verify_model(graph, input_shape=request.input_shape, run_forward=request.run_forward)
    except HTTPException:
        raise
    except UnsafeGraphInput as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (ValueError, TypeError, AttributeError) as exc:
        # verify_model catches every torch failure internally and reports it as
        # ok:false, so what escapes here comes from reading the payload itself —
        # a client error, and it used to be a 500 echoing the raw exception.
        raise HTTPException(status_code=400, detail=f"Malformed graph: {_malformed_graph_detail(exc)}")
    except Exception:
        logger.exception("verify_model failed unexpectedly")
        raise HTTPException(status_code=500, detail="Verification failed")


@app.post("/api/models/export-package", dependencies=[Depends(get_current_user)])
def export_package_endpoint(request: ExportPackageRequest):
    """Return a zip of model.py + sample_run.py + README + requirements.

    Gated. Every plane used to trust its input independently, so a graph that
    validation flagged still exported: a Linear(in_features=-5) canvas produced a
    well-formed zip whose sample_run.py died with "Trying to create tensor with
    negative dimension -5" on first run. Two checks now stand in the way — the
    structural validator, and the same synthetic forward pass the Validate button
    uses. Passing both is what "runnable export" has to mean.
    """
    graph = request.graph
    _require_nodes(graph, "graph")
    input_shape = request.input_shape or [1, 3, 224, 224]
    model_name = request.class_name or request.model_name or "GeneratedModel"
    # sanitize the model_name into a valid python identifier so the package
    # imports cleanly on the user's side.
    safe_name = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in model_name)
    if not safe_name or safe_name[0].isdigit():
        safe_name = f"Model_{safe_name}" if safe_name else "GeneratedModel"

    _refuse_if_invalid(graph, input_shape, "exported")
    # Structural problems are reported above without PyTorch; the runnable check needs it.
    service = load_pytorch_service("services.verification_service", "Package export")

    try:
        verification = service.verify_model(graph, input_shape=input_shape, run_forward=True)
    except UnsafeGraphInput as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=400, detail=f"Malformed graph: {_malformed_graph_detail(exc)}")
    except Exception:
        logger.exception("export pre-flight verification failed unexpectedly")
        raise HTTPException(status_code=500, detail="Export pre-flight failed")

    if not verification.get("ok"):
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "This model cannot be exported: it does not run. The export would "
                    "have handed you code that crashes on the first forward pass."
                ),
                "stage": verification.get("stage"),
                "errors": verification.get("errors") or [],
            },
        )

    try:
        zip_bytes, filename = service.export_package(graph, input_shape, safe_name)
    except UnsafeGraphInput as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=400, detail=f"Malformed graph: {_malformed_graph_detail(exc)}")
    except Exception:
        logger.exception("package export failed unexpectedly")
        raise HTTPException(status_code=500, detail="Package export failed")
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/models/export-onnx", dependencies=[Depends(get_current_user)])
def export_onnx_endpoint(request: ExportOnnxRequest):
    """Export the model graph to ONNX, run onnx shape inference + checker.

    Body: { graph, input_shape?, model_name?, opset? }

    Returns JSON:
      {
        "ok": bool,
        "errors": [...],
        "node_shapes": { onnx_name: [dims] },
        "onnx_base64": "<base64 .onnx bytes>",
        "onnx_size_bytes": int,
        "filename": str,
      }
    The frontend can decode onnx_base64 to trigger a file download or
    drag it into https://netron.app for visualization.
    """
    graph = request.graph
    if not graph:
        raise HTTPException(status_code=400, detail="graph is required")
    # Same gate as the code routes. Without it an empty canvas exported a
    # well-formed .onnx containing a single Identity op, ok:true — a downloadable
    # artifact asserting the user had a model. Independent verification caught
    # this one: the first pass gated generate-code, code/export and export-package
    # and missed that ONNX export has its own entry point.
    _require_nodes(graph, "graph")
    model_name = request.model_name or request.class_name or "GeneratedModel"
    onnx_export = load_pytorch_service("services.onnx_export", "ONNX export")
    try:
        return onnx_export.export_onnx(
            graph,
            input_shape=request.input_shape,
            model_name=model_name,
            opset=request.opset,
        )
    except UnsafeGraphInput as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (ValueError, TypeError, AttributeError) as exc:
        # export_onnx reports every onnx/torch failure as ok:false, so an escaping
        # exception came from reading the payload. `opset` is typed now, which is
        # what used to make {"opset": "abc"} a bare unhandled 500 (the int() cast
        # sat outside this try).
        raise HTTPException(status_code=400, detail=f"Malformed graph: {_malformed_graph_detail(exc)}")
    except Exception:
        logger.exception("onnx export failed unexpectedly")
        raise HTTPException(status_code=500, detail="ONNX export failed")



@app.post("/api/code/parse")
def parse_code(request: CodeParseRequest):
    """Parse PyTorch model code and convert to graph format"""
    try:
        graph_data = code_parser_service.parse_pytorch_code(request)
        return graph_data
    except (ValueError, SyntaxError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Code parsing error: {str(e)}")
    except Exception:
        logger.exception("parse_code failed unexpectedly")
        raise HTTPException(status_code=500, detail="Internal error during code parsing")


@app.post("/api/code/export")
def export_code(request: ExportCodeRequest):
    """Export model graph as various code formats"""
    try:
        export_format = request.format  # pytorch, onnx, tensorflow

        if export_format == "pytorch":
            return generate_code(request)
        else:
            raise HTTPException(status_code=400, detail=f"Export format '{export_format}' not supported yet")
    except HTTPException:
        raise
    except Exception:
        logger.exception("export_code failed unexpectedly")
        raise HTTPException(status_code=500, detail="Error exporting code")


# Module Template Endpoints
@app.get("/api/module-templates")
def get_module_templates():
    """Get available module templates across all categories"""
    try:
        all_modules = {}
        for category in ["transformer", "computer_vision", "nlp", "generative"]:
            modules = template_service.get_module_templates_by_category(category)
            all_modules.update(modules)
        
        return {"module_templates": all_modules}
    except Exception:
        logger.exception("get_module_templates failed")
        raise HTTPException(status_code=500, detail="Error fetching module templates")


@app.post("/api/module-templates/{template_id}/instantiate")
def instantiate_module_template(template_id: str, request: InstantiateModuleTemplateRequest):
    """Create a module instance from a template"""
    try:
        params = request.parameters
        instance_id = request.instance_id or f"{template_id}_instance"

        module_template = template_service.get_module_template_by_id(template_id)
        if not module_template:
            raise HTTPException(status_code=404, detail=f"Module template '{template_id}' not found")
        
        # Create module instance (simplified for now)
        module_instance = {
            "id": instance_id,
            "template_id": template_id,
            "name": module_template.name,
            "parameters": {**module_template.default_params, **params},
            "internal_graph": module_template.internal_graph.dict()
        }
        
        return {"module": module_instance}
    except HTTPException:
        raise
    except Exception:
        logger.exception("instantiate_module_template failed unexpectedly template_id=%s", template_id)
        raise HTTPException(status_code=500, detail="Error instantiating module")


@app.get("/api/canonical-templates")
def list_canonical_templates():
    """Every canonical block the registry can build, so a client can offer them.

    There was no way to ask. The registry holds 306 blocks and the only route to any of
    them was `/instantiate` with an id you had to know already, so the IDE's template
    picker could offer the 73 file-shaped architectures and none of these — and the
    catalogue block that DOES advertise 15 of them names ids the registry has never had
    (they 404 on instantiate). A listing is what lets a client show what actually exists.

    Names only. Instantiating 306 blocks to describe them would cost seconds and the
    caller is choosing, not building.
    """
    from services.canonical_template_registry import CanonicalTemplateRegistry

    try:
        contracts = CanonicalTemplateRegistry().list_contracts()
    except Exception:
        logger.exception("list_canonical_templates failed")
        raise HTTPException(status_code=500, detail="Error listing canonical templates")

    templates = []
    for contract in contracts:
        tid = getattr(contract, "template_id", None) or getattr(contract, "id", None)
        if not tid:
            continue
        templates.append(
            {
                "id": str(tid),
                "name": str(getattr(contract, "name", "") or tid),
                "family": str(getattr(contract, "family", "") or ""),
            }
        )
    templates.sort(key=lambda t: t["id"])
    return {"templates": templates, "count": len(templates)}


@app.post("/api/canonical-templates/{template_id}/instantiate")
def instantiate_canonical_template(template_id: str, request: InstantiateTemplateRequest):
    """Instantiate a paper-inventory / canonical template (e.g. 'vit_block')
    into its real layer subgraph via the canonical template registry, and
    return flat nodes/edges the canvas can place directly. This is the path
    the v2 palette uses so template blocks drop real, validatable layers
    instead of an opaque module shell."""
    config = request.config
    instance_id = request.instance_id or template_id
    try:
        from services.canonical_template_registry import CanonicalTemplateRegistry
        registry = CanonicalTemplateRegistry()
        # A block the generator cannot build is not offered by list_contracts, and must
        # not be reachable by id either — otherwise "removed" means "hidden", and the
        # palette or an old link still drops something on the canvas that can never
        # become a .py.
        if template_id not in registry.buildable_ids():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"'{template_id}' is not offered: it uses operations the code "
                    "generator has no entry for, so it could be drawn and never run."
                ),
            )
        nodes, edges, entry_node_id, exit_node_id, applied_config = registry.instantiate(
            template_id=template_id,
            instance_id=instance_id,
            config=config,
        )
        # Inside the try on purpose: the response below sits outside it, so a call
        # placed there would escape this route's guarded handlers as a bare 500.
        input_shape = registry.seed_input_shape(
            template_id=template_id, nodes=nodes, entry_node_id=entry_node_id, edges=edges
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except (TypeError, AttributeError) as e:
        # A config value the template cannot use ("'str' object has no attribute
        # 'get'") is the caller's mistake, not the 500 it used to be.
        raise HTTPException(status_code=400, detail=f"Invalid template configuration: {e}")
    except Exception:
        logger.exception("instantiate_canonical_template failed unexpectedly template_id=%s", template_id)
        raise HTTPException(status_code=500, detail="Error instantiating template")

    return {
        "nodes": [
            {"id": n.node_id, "op": n.op, "kind": n.kind, "params": n.params}
            for n in nodes
        ],
        "edges": [
            {"id": e.edge_id, "source": e.source_node_id, "target": e.target_node_id}
            for e in edges
        ],
        "entry_node_id": entry_node_id,
        "exit_node_id": exit_node_id,
        # A block has no Input node, so a client cannot size the example tensor from
        # the graph it is handed. Without this the generator declared an ImageNet
        # tensor for a token-sequence block and the trace died on its first LayerNorm.
        "input_shape": input_shape,
        "config": applied_config,
    }


@app.get("/api/papers/template-inventory")
def get_paper_template_inventory():
    """Return canonical template inventory for the v2 layer palette.

    The historical paper pipeline has been removed. The palette still uses this
    compatibility route to load the canonical registry inventory.
    """
    try:
        from services.canonical_template_registry import CanonicalTemplateRegistry

        registry = CanonicalTemplateRegistry()
        dynamic_ids = set(registry.list_dynamic_template_ids())
        contracts = registry.list_contracts()
        templates = [
            {
                "template_id": contract.template_id,
                "name": contract.name,
                "source": "dynamic" if contract.template_id in dynamic_ids else "builtin",
                "required_nodes": len(contract.required_nodes),
                "required_edges": len(contract.required_edges),
                "training_minimums": len(contract.training_minimums),
                "input_expectation": contract.input_expectation,
            }
            for contract in contracts
        ]
        return {
            "summary": {
                "template_total": len(templates),
                "template_builtin": len(templates) - len(dynamic_ids),
                "template_dynamic": len(dynamic_ids),
                "mapping_total": 0,
                "mapping_dynamic": 0,
                "staging_pending": 0,
                "staging_approved": 0,
                "staging_rejected": 0,
            },
            "templates": templates,
            "dynamic_template_ids": sorted(dynamic_ids),
            "dynamic_mappings": [],
        }
    except Exception:
        logger.exception("get_paper_template_inventory failed")
        raise HTTPException(status_code=500, detail="Error fetching template inventory")


@app.get("/api/layer-types")
def get_layer_types():
    """Get available layer types for the model builder"""
    return {
        "layers": [
            {
                "id": "conv2d",
                "name": "Conv2D",
                "category": "convolution",
                "description": "2D convolution layer",
                "parameters": [
                    {"name": "in_channels", "type": "number", "required": True, "description": "Number of input channels"},
                    {"name": "out_channels", "type": "number", "required": True, "description": "Number of output channels"},
                    {"name": "kernel_size", "type": "number", "required": True, "description": "Size of convolution kernel"},
                    {"name": "stride", "type": "number", "default": 1, "description": "Stride of convolution"},
                    {"name": "padding", "type": "number", "default": 0, "description": "Padding applied to input"},
                    {"name": "bias", "type": "boolean", "default": True, "description": "If True, adds learnable bias"}
                ]
            },
            {
                "id": "linear",
                "name": "Linear",
                "category": "linear",
                "description": "Fully connected linear layer",
                "parameters": [
                    {"name": "in_features", "type": "number", "required": True, "description": "Size of input features"},
                    {"name": "out_features", "type": "number", "required": True, "description": "Size of output features"},
                    {"name": "bias", "type": "boolean", "default": True, "description": "If True, adds learnable bias"}
                ]
            },
            {
                "id": "relu",
                "name": "ReLU",
                "category": "activation",
                "description": "Rectified Linear Unit activation",
                "parameters": [
                    {"name": "inplace", "type": "boolean", "default": False, "description": "If True, do operation in-place"}
                ]
            },
            {
                "id": "batchnorm2d",
                "name": "BatchNorm2D",
                "category": "normalization",
                "description": "2D batch normalization",
                "parameters": [
                    {"name": "num_features", "type": "number", "required": True, "description": "Number of features"},
                    {"name": "eps", "type": "number", "default": 1e-5, "description": "Value added for numerical stability"},
                    {"name": "momentum", "type": "number", "default": 0.1, "description": "Value used for running mean and variance"}
                ]
            },
            {
                "id": "maxpool2d",
                "name": "MaxPool2D",
                "category": "pooling",
                "description": "2D max pooling layer",
                "parameters": [
                    {"name": "kernel_size", "type": "number", "required": True, "description": "Size of pooling window"},
                    {"name": "stride", "type": "number", "description": "Stride of pooling window"},
                    {"name": "padding", "type": "number", "default": 0, "description": "Padding applied to input"}
                ]
            },
            {
                "id": "dropout",
                "name": "Dropout",
                "category": "regularization",
                "description": "Dropout layer for regularization",
                "parameters": [
                    {"name": "p", "type": "number", "default": 0.5, "min": 0, "max": 1, "description": "Probability of dropout"},
                    {"name": "inplace", "type": "boolean", "default": False, "description": "If True, do operation in-place"}
                ]
            },
            {
                "id": "flatten",
                "name": "Flatten",
                "category": "utility",
                "description": "Flatten input tensor",
                "parameters": [
                    {"name": "start_dim", "type": "number", "default": 1, "description": "First dimension to flatten"},
                    {"name": "end_dim", "type": "number", "default": -1, "description": "Last dimension to flatten"}
                ]
            },
            {
                "id": "lstm",
                "name": "LSTM",
                "category": "recurrent",
                "description": "Long Short-Term Memory layer",
                "parameters": [
                    {"name": "input_size", "type": "number", "required": True, "description": "Number of input features"},
                    {"name": "hidden_size", "type": "number", "required": True, "description": "Number of hidden units"},
                    {"name": "num_layers", "type": "number", "default": 1, "description": "Number of LSTM layers"},
                    {"name": "bidirectional", "type": "boolean", "default": False, "description": "If True, bidirectional LSTM"}
                ]
            },
            {
                "id": "embedding",
                "name": "Embedding",
                "category": "embedding",
                "description": "Embedding layer for discrete tokens",
                "parameters": [
                    {"name": "num_embeddings", "type": "number", "required": True, "description": "Size of vocabulary"},
                    {"name": "embedding_dim", "type": "number", "required": True, "description": "Size of embeddings"},
                    {"name": "padding_idx", "type": "number", "description": "Index of padding token"}
                ]
            }
        ]
    }

# Helper Functions
def _convert_to_hierarchical_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Convert flat graph dictionary with parentIds into hierarchical structure"""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    # Check if there's any hierarchy indication
    has_modules = any(n.get("type", "") == "moduleNode" or n.get("data", {}).get("parentId") for n in nodes)
    if not has_modules:
        return graph
        
    modules = {}
    node_hierarchy = {}
    
    # 1. Register all modules
    for n in nodes:
        if n.get("type", "") == "moduleNode":
            module_id = n["id"]
            parent_id = n.get("data", {}).get("parentId")
            modules[module_id] = {
                "module_id": module_id,
                "parent_id": parent_id,
                "internal_nodes": [],
                "internal_edges": [],
                "parameters": n.get("data", {}).get("parameters", {}),
                "input_node_id": None,
                "output_node_id": None
            }
            if parent_id:
                node_hierarchy[module_id] = parent_id
            
    # 2. Categorize nodes as root or internal
    root_nodes = []
    for n in nodes:
        if n.get("type", "") == "moduleNode":
            root_nodes.append(n)
            continue
            
        parent_id = n.get("data", {}).get("parentId")
        if parent_id and parent_id in modules:
            modules[parent_id]["internal_nodes"].append(n)
            node_hierarchy[n["id"]] = parent_id
            
            # Simple detection of input/output bounds for internal graph execution
            layer_name = n.get("data", {}).get("layerType", {}).get("name", "").lower()
            if layer_name == "input":
                modules[parent_id]["input_node_id"] = n["id"]
            elif layer_name == "output":
                modules[parent_id]["output_node_id"] = n["id"]
        else:
            root_nodes.append(n)
            
    # 3. Categorize edges
    root_edges = []
    for e in edges:
        src = e.get("source")
        tgt = e.get("target")
        src_parent = node_hierarchy.get(src)
        tgt_parent = node_hierarchy.get(tgt)
        
        # Internal module edges
        if src_parent and tgt_parent and src_parent == tgt_parent:
            modules[src_parent]["internal_edges"].append(e)
        else:
            # Cross-boundary or root-level edges
            root_edges.append(e)
            
    return {
        "root_nodes": root_nodes,
        "root_edges": root_edges,
        "modules": modules,
        "node_hierarchy": node_hierarchy
    }

def _now_iso8601() -> str:
    """The moment this response was generated.

    Was the literal "2024-01-01T00:00:00Z" at both generate-code return sites, so
    every response claimed to have been produced on New Year 2024 — while the README
    inside the very same export zip carried a real timestamp.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _dimension(params: Dict[str, Any], key: str, default: Any, node: Dict[str, Any]) -> int:
    """One positive integer from a stored parameter, or refuse to guess.

    Two ways this went wrong. A negative value multiplied straight through, so a
    Linear(in_features=-5) canvas reported `total_parameters: -16` while the same
    response said `complete: true` — a summary asserting the model has minus
    sixteen parameters is not a display glitch, it is proof the graph is broken.
    And the canvas stores a kernel as 3, [3, 3] or the string "3,3"; the string
    form reached `in_channels * out_channels * kernel_size[0]`, where
    `kernel_size[0]` is the *character* "3", so int*str repeated a string and
    str*str raised TypeError — a 500 on a perfectly ordinary layer.
    """
    raw = params.get(key, default)
    if isinstance(raw, str):
        raw = raw.split(",")[0].strip()
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else default
        if isinstance(raw, str):
            raw = raw.strip()
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        raise ValueError(
            f"Node {node.get('id', '?')}: {key} is {params.get(key, default)!r}, "
            "which is not a number"
        )
    if value <= 0:
        raise ValueError(
            f"Node {node.get('id', '?')}: {key} must be positive, got {value}"
        )
    return value


def _calculate_model_summary(graph: Dict[str, Any], input_shape: List[int]) -> Dict[str, Any]:
    """Calculate model summary statistics"""
    nodes = graph.get("nodes", [])
    if "root_nodes" in graph:
        nodes = graph["root_nodes"]

    # Count parameters (simplified estimation)
    total_params = 0
    trainable_params = 0

    for node in nodes:
        layer_type = node.get("data", {}).get("layerType", {}).get("name", "")
        params = node.get("data", {}).get("parameters", {})

        # Rough parameter estimation
        if layer_type == "Linear":
            in_features = _dimension(params, "in_features", 0, node)
            out_features = _dimension(params, "out_features", 0, node)
            has_bias = params.get("bias", True)
            layer_params = in_features * out_features + (out_features if has_bias else 0)
            total_params += layer_params
            trainable_params += layer_params
        elif layer_type == "Conv2d":
            in_channels = _dimension(params, "in_channels", 0, node)
            out_channels = _dimension(params, "out_channels", 0, node)
            kernel_h = _dimension(params, "kernel_size", 3, node)
            kernel_w = kernel_h
            raw_kernel = params.get("kernel_size", 3)
            if isinstance(raw_kernel, str) and "," in raw_kernel:
                kernel_w = int(float(raw_kernel.split(",")[1].strip()))
            elif isinstance(raw_kernel, (list, tuple)) and len(raw_kernel) > 1:
                kernel_w = int(float(raw_kernel[1]))
            layer_params = in_channels * out_channels * kernel_h * kernel_w
            total_params += layer_params
            trainable_params += layer_params
        elif layer_type == "Embedding":
            num_embeddings = _dimension(params, "num_embeddings", 0, node)
            embedding_dim = _dimension(params, "embedding_dim", 0, node)
            layer_params = num_embeddings * embedding_dim
            total_params += layer_params
            trainable_params += layer_params

    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "model_size_mb": round(model_size_mb, 2),
        "estimated_memory_mb": round(model_size_mb * 4, 2),  # Rough estimation
        "layer_count": len(nodes)
    }


# LAST, and it must stay last: Starlette matches routes in registration order and
# a mount on "/" matches everything, so anything registered below this line would
# be shadowed by the static build. StaticFiles itself raises if the directory does
# not exist, which is the loud failure we want from a stale TENSORVIZ_STATIC_DIR.
# html=True serves index.html for "/" — the app routes with a MemoryRouter in IDE
# mode, so no deep path is ever requested.
if _static_dir:
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="app")


# `python -m uvicorn server_v2:app` loads this module and never reaches __main__.
# The guard refuses a non-loopback listen at server startup, before the socket
# is created. Loopback and a set TENSORVIZ_LOCAL_TOKEN still start.
from bind_policy import install_uvicorn_bind_guard

install_uvicorn_bind_guard()


if __name__ == "__main__":
    import os
    import uvicorn
    from bind_policy import ensure_safe_bind
    from services.ide.local_browser_wire import MAX_WIRE_BYTES
    # Default 8765 because some local tools (e.g. VS Code plugin helpers)
    # bind 127.0.0.1:8000 and silently shadow a public bind.
    port = int(os.environ.get("BACKEND_PORT", "8765"))
    uvicorn.run(app, host=ensure_safe_bind(os.environ.get("HOST", "127.0.0.1")), port=port,
                ws_max_size=MAX_WIRE_BYTES)
