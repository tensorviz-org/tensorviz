"""Host-independent source/capture projection. Never reads or executes project files."""
from __future__ import annotations

import ast
import json
from typing import Any, Callable, Dict, List, Optional

from models.graph_models import CodeParseRequest
from services.ide import crossfile, multifile, projection
from services.ide.module_index import ModuleIndex
from services.ingest.document import to_document
from utils.code_parser import CodeParserService, _is_self_super_call

_parser = CodeParserService()


class ProjectionError(ValueError):
    def __init__(self, status_code: int, detail: Any):
        super().__init__(str(detail))
        self.status_code = status_code
        self.detail = detail


def _parse_workspace_class(
    file: str,
    source: str,
    cls: str,
    extra_module_bases: Optional[set],
    super_forward: Optional[tuple] = None,
    known_attributes: Optional[set] = None,
    constructor_kinds: Optional[Dict[str, bool]] = None,
    measured_input_shape: Optional[List[int]] = None,
    review_only: bool = False,
) -> Dict[str, Any]:
    """The parse_fn the cross-file expander runs against each supplied source.

    Mirrors ``_parse_class``'s selection and error mapping, but raises
    ``ClassParseRefused`` so the expander can demote a broken DEPENDENCY to a
    diagnostic — only the root's failure is a request failure.

    ``measured_input_shape`` is supplied for the ROOT class only (see
    ``multifile_project``): it is what a previous trace of *that* class ran on,
    and a dependency drawn with it would show an Input node for a tensor that
    never entered it.
    """
    try:
        parsed = _parser.parse_pytorch_code(
            CodeParseRequest(
                code=source, file=file, measured_input_shape=measured_input_shape
            ),
            extra_module_bases=extra_module_bases,
            super_forward=super_forward,
            extra_layer_names=known_attributes,
            constructor_kinds=constructor_kinds,
        )
    except SyntaxError as error:
        raise crossfile.ClassParseRefused(
            str(error.msg),
            line=error.lineno or 1,
            column=(error.offset or 1) - 1,
            syntax_error=True,
        ) from error
    chosen = next(
        (entry for entry in parsed.get("graphs") or [] if entry.get("name") == cls),
        None,
    )
    inherited = None
    if chosen is None and review_only:
        # A review can show the explicitly requested subclass's own declarations
        # without claiming that its external base has been resolved or executed.
        definitions = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == cls]
        target = definitions[0] if len(definitions) == 1 else None
        bases = {ast.unparse(base) for base in target.bases if isinstance(base, (ast.Name, ast.Attribute))} if target else set()
        if bases:
            forward = next((node for node in target.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward"), None)
            calls_super = forward is not None and any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "forward" and _is_self_super_call(node.func.value, cls) for node in ast.walk(forward))
            inherited = ("@inherited-forward", forward is None) if forward is None or calls_super else None
            parsed = _parser.parse_pytorch_code(CodeParseRequest(code=source, file=file),
                extra_module_bases=set(extra_module_bases or ()) | bases,
                super_forward=(inherited[0], inherited[0]) if inherited else None,
                constructor_kinds=constructor_kinds)
            chosen = next((entry for entry in parsed.get("graphs") or [] if entry.get("name") == cls), None)
    if not parsed.get("success"):
        raise crossfile.ClassParseRefused(str(parsed.get("error") or "could not parse"))
    if chosen is None or chosen.get("graph") is None:
        if cls not in projection.class_names(source):
            raise crossfile.ClassParseRefused(f"{cls} is not in {file}", class_missing=True)
        # An unresolved/conditional base can hide a real class from the static
        # parser. Keep it eligible for the canonical capture fallback below.
        raise crossfile.ClassParseRefused(f"{cls} in {file} cannot be projected from source")
    graph = chosen["graph"]
    result = {
        "nodes": [node.model_dump() for node in graph.nodes],
        "edges": [edge.model_dump() for edge in graph.edges],
    }
    if inherited:
        node_id, implicit = inherited
        result["nodes"].insert(1, {"id": node_id, "type": "layerNode", "position": {"x": 250, "y": 200},
            "data": {"layerType": {"id": "custom_code", "name": "custom_code", "parameters": []},
                     "parameters": {"_class": f"{', '.join(sorted(bases))}.forward · inherited", "_opaque": True, "_review_inherited": True}}})
        if implicit:
            # A missing override inherits forward, not declaration-order wiring.
            # Local declarations remain visible but unconnected until proven.
            result["edges"] = [{"id": "inherited-input", "source": "input", "target": node_id},
                               {"id": "inherited-output", "source": node_id, "target": "output"}]
    return result


def project_workspace(
    request: multifile.MultifileProjectionRequest,
    persist: Optional[Callable[[str, str, Dict[str, Any]], tuple[int, str]]] = None,
) -> Dict[str, Any]:
    """The versioned multi-file source projection (CURRENT_TASK.md steps 1-3).

    The extension supplies the whole candidate source set — dirty editor buffers
    preferred over disk — and gets the root class expanded through its workspace
    imports: qualified instance ids (``enc1.fc``), per-file spans with digests,
    the dependency set the projection actually consumed, and coded diagnostics
    for everything that deliberately stayed opaque (cycles cut at one card,
    dynamic/computed classes pinned but unresolved, external/framework ops left
    to the op registry).

    Invariants that preceded resolution and outlive it: path safety, digest
    honesty, version pinning and the explicit budget rules are all enforced
    deterministically BEFORE any parse begins, and nothing in the source set is
    imported, executed or written — the trace passthrough, when present, is the
    worker envelope the extension produced in the user's own interpreter.
    """
    if request.protocol_version != multifile.IDE_PROJECTION_PROTOCOL_VERSION:
        raise ProjectionError(
            status_code=400,
            detail={
                "error": "protocol_version_mismatch",
                "expected": multifile.IDE_PROJECTION_PROTOCOL_VERSION,
                "got": request.protocol_version,
                "message": (
                    "the extension and this backend disagree about the projection "
                    "contract; update both to the same build"
                ),
            },
        )
    try:
        sources = multifile.validate_sources(request.sources)
        root = multifile.validate_root(request.root, sources)
    except multifile.MultifileError as error:
        raise ProjectionError(
            status_code=400, detail={"error": error.code, "message": error.detail}
        ) from error

    # A workspace candidate set can include setup helpers for other models.
    # Captured graphs use only source files attested by this invocation; source
    # projection still uses the complete candidate set when no capture exists.
    result = (request.trace_result or {}).get("result")
    diagnostics = result.get("diagnostics") if isinstance(result, dict) else None
    canonical = diagnostics.get("semantic_capture") if isinstance(diagnostics, dict) else None
    if canonical is not None:
        from services.semantic.contracts import SourceSnapshot
        try:
            captured_source = SourceSnapshot.model_validate(canonical["source"])
        except (TypeError, ValueError, KeyError):
            raise ProjectionError(400, detail={"error": "trace_rejected",
                "message": "The semantic capture source snapshot is invalid."}) from None
        captured_paths = {file.path for file in captured_source.files if file.role == "source"}
        # Retain the requested root even when the capture names another model so
        # the existing source/model identity check rejects the mismatch.
        sources = {file: entry for file, entry in sources.items()
                   if file == root.file or file in captured_paths}

    source_map = {path: entry.source for path, entry in sources.items()}

    def parse_one(file: str, source: str, cls: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        # The remembered measurement belongs to the class the projection is
        # ABOUT, and to no other: a dependency expanded into the same document is
        # fed by its parent's activations, not by the root's example input.
        return _parse_workspace_class(
            file,
            source,
            cls,
            *args,
            **kwargs,
            measured_input_shape=(
                root.measured_input_shape
                if file == root.file and cls == root.cls
                else None
            ),
            review_only=request.review_only and file == root.file and cls == root.cls,
        )

    try:
        from services.jax_support.discovery import target as jax_target
        selected_jax = jax_target(source_map, root.file, root.cls)
        if selected_jax is not None:
            # JAX's callable program comes from Jaxpr. Source alone supplies
            # identity and setup, never invented operations or connections.
            expansion = crossfile.WorkspaceExpansion(
                graph={'nodes': [{'id': 'jax-source', 'type': 'layerNode', 'position': {'x': 0, 'y': 0},
                    'data': {'layerType': {'id': 'custom_code', 'name': 'custom_code', 'parameters': []},
                             'parameters': {'_class': root.cls, '_opaque': True,
                                            '_label': root.cls + ' · JAX · Not traced'}}}], 'edges': []},
                spanmap={}, dependencies=tuple(sorted(source_map)), diagnostics=())
        else:
            expansion = crossfile.project_workspace(
                ModuleIndex.build(source_map), source_map, root_file=root.file,
                root_cls=root.cls, parse_fn=parse_one)
    except crossfile.ClassParseRefused as error:
        # A canonical capture is sufficient for presentation even when static
        # lowering cannot represent this valid Python model.
        result = (request.trace_result or {}).get("result") or {}
        canonical = (result.get("diagnostics") or {}).get("semantic_capture")
        if canonical is not None and not error.syntax_error and not error.class_missing:
            expansion = crossfile.WorkspaceExpansion(
                graph={"nodes": [], "edges": []}, spanmap={},
                dependencies=tuple(sorted(source_map)), diagnostics=(),
            )
        elif error.syntax_error:
            raise ProjectionError(
                status_code=400,
                detail={
                    "error": "syntax_error",
                    "message": error.message,
                    "line": error.line,
                    "column": error.column,
                },
            ) from error
        else:
            status = 404 if error.class_missing else 422
            raise ProjectionError(status_code=status, detail=error.message) from error

    if request.review_only and request.trace_result is not None:
        raise ProjectionError(400, detail="Git review accepts source only, not a trace result.")
    graph_id = projection.graph_id_for(root.file, root.cls)
    if request.review_only:
        graph_id = "git-review:" + graph_id + ":" + multifile.projection_digest({file: entry.digest for file, entry in sources.items()})
    graph, spanmap, trace_error, traced = expansion.graph, expansion.spanmap, None, False
    document = None
    if request.trace_result is not None:
        try:
            document, spanmap, trace_error = projection.project_trace(
                expansion.graph,
                expansion.spanmap,
                request.trace_result,
                cls=root.cls,
                rel_file=root.file,
                graph_id=graph_id,
                source_files=set(source_map),
                source_texts=source_map,
                source_digests={file: entry.digest for file, entry in sources.items()},
            )
        except projection.TraceRejected as error:
            detail: Dict[str, Any] = {"error": "trace_rejected", "message": str(error)}
            if error.trace_error is not None:
                detail["trace_error"] = error.trace_error
            raise ProjectionError(status_code=400, detail=detail) from error
        traced = True

    review_class = projection.review_class(source_map[root.file], root.cls, root.file)
    # Dependency evidence is part of the structural baseline, including files
    # outside the edited class. This permits display context, never execution.
    review_class["structureDigest"] = projection.digest_of(review_class["structureDigest"] + repr(sorted(
        (file, sources[file].digest) for file in expansion.dependencies if file != root.file)))
    # Static expansion is not a Python dependency closure. Compare the supplied
    # source candidates verified against this run, and disclose partial coverage.
    review_class["dependencyDigests"] = {file: entry.digest for file, entry in sources.items() if file != root.file}
    if canonical is not None:
        review_class["dependencyScope"] = captured_source.dependencyScope
        if captured_source.dependencyScope == "complete":
            review_class["dependencyDigests"] = {file.path: file.digest for file in captured_source.files
                                                 if file.role == "source" and file.path != root.file}
    if document is None:
        document = to_document(root.cls, graph, graph_id)
        # Navigation to an assignment is source evidence, not a semantic layer.
        spanmap.update({name: {**fact["span"], "digest": fact["digest"]}
                        for name, fact in review_class.get("assignments", {}).items()})
        if request.review_only:
            spanmap = {**{name: fact["span"] for name, fact in review_class["members"].items()}, **spanmap}
    from .code_focus import enrich_focus_spans
    spanmap = enrich_focus_spans(spanmap, document, source_map)
    document.semantic.metadata["reviewClass"] = review_class
    if traced:
        capture = ((request.trace_result or {}).get("result", {}).get("diagnostics") or {}).get("semantic_capture")
        if capture:
            # ModuleList/ModuleDict may never be invoked. Their registered type
            # and ordered keys still prove how a literal subscript resolves.
            definitions = {item["id"]: item["qualifiedName"] for item in capture["definitions"]}
            paths = [(path, definitions[module["definitionId"]]) for module in capture["modules"] for path in module["paths"]]
            children = {}
            for path, _ in paths:
                parent, _, key = path.rpartition(".")
                children.setdefault(parent, []).append(key)
            containers = {path: dict(definition=definition, keys=children.get(path, [])) for path, definition in paths
                          if definition in {"torch.nn.modules.container.Sequential", "torch.nn.modules.container.ModuleList",
                                            "torch.nn.modules.container.ModuleDict"}}
            # A missing bounded inventory refuses indexed mapping; it never
            # changes the authoritative capture or claims a partial inventory.
            if len(json.dumps(containers).encode()) <= 256 * 1024:
                document.semantic.metadata["registeredContainers"] = containers
    document_json = document.model_dump(mode="json", by_alias=True)
    revision, project_id = persist(graph_id, root.cls, document_json) if persist and not request.review_only else (0, "standalone")

    digests = {path: entry.digest for path, entry in sources.items()}
    diagnostics = [dict(entry) for entry in expansion.diagnostics]
    if traced:
        measured = {node.node_id for node in document.semantic.nodes
                    if node.params.get("_shape_provenance") == "observed" and "_out_shape" in node.params}
        measured.update(fact["nodeId"] for fact in document.semantic.metadata.get("sourceCorrespondence", [])
                        if fact["status"] == "observed")
        for diagnostic in diagnostics:
            if diagnostic["code"] == multifile.EXTERNAL_PACKAGE and diagnostic.get("node") in measured:
                diagnostic["detail"] += (
                    " Boundary shapes were measured for this node in this run;"
                    " the static definition remains unresolved. Captured contents may be depth-limited."
                )
    if request.truncated:
        diagnostics.append(multifile.truncation_diagnostic())
    # A captured layer can come from a file static expansion never reached (a
    # class passed as a constructor argument). Its spans consumed that file too.
    consumed = sorted(sources) if request.review_only else sorted(
        {*expansion.dependencies, *(span["file"] for span in spanmap.values() if span.get("file") in digests)})

    return {
        "cls": root.cls,
        "document": document_json,
        "spanmap": multifile.with_file_digests(spanmap, digests),
        "source_digest": projection.digest_of(source_map[root.file]),
        "graph_id": graph_id,
        "revision": revision,
        "traced": traced,
        "trace_error": trace_error,
        "project_id": project_id,
        "protocol_version": multifile.IDE_PROJECTION_PROTOCOL_VERSION,
        "projection_digest": multifile.projection_digest(digests),
        "dependencies": [{"file": path, "digest": digests[path]} for path in consumed],
        "diagnostics": diagnostics,
        "truncated": request.truncated,
    }
