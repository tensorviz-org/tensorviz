"""Discover review targets from immutable source text, without importing models."""
from __future__ import annotations

import ast
import copy
import hashlib
from typing import List

from pydantic import BaseModel, Field

from . import multifile
from .module_index import ModuleIndex, package_of_file
from ..ingest.module_discovery import module_declarations


class ReviewIndexRequest(BaseModel):
    protocol_version: int
    sources: List[multifile.MultifileSourceEntry] = Field(max_length=multifile.MAX_SOURCES)
    truncated: bool = False


def index_review(request: ReviewIndexRequest) -> dict:
    if request.protocol_version != multifile.IDE_PROJECTION_PROTOCOL_VERSION:
        raise multifile.MultifileError("protocol_version_mismatch", "Unsupported review protocol")
    sources = {file: entry.source for file, entry in multifile.validate_sources(request.sources).items()}
    index = ModuleIndex.build(sources)
    trees, notes = {}, []
    for file, source in sources.items():
        try:
            trees[file] = ast.parse(source)
        except SyntaxError as error:
            notes.append({"file": file, "detail": f"could not parse {file}:{error.lineno or 1}"})
    declarations = module_declarations(sources, index=index)
    imports = {}
    for module, record in index.modules.items():
        dependencies = {record.file, *record.wildcard_dependencies}
        for bindings in record.bindings.values():
            for binding in bindings:
                for name in [binding.target_module, f"{binding.target_module}.{binding.target_name}"]:
                    if name in index.modules:
                        dependencies.add(index.modules[name].file)
        # Include conditional local imports as potential dependencies without
        # claiming which branch executes. Optional external imports alone are
        # not a discovery failure and do not establish a missing dependency.
        for node in ast.walk(trees.get(record.file, ast.Module(body=[], type_ignores=[]))):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = ModuleIndex._resolve_relative(node.module or '', node.level, package_of_file(record.file))
                targets = [base, *(f'{base}.{alias.name}' for alias in node.names)] if base else []
            else:
                continue
            dependencies.update(index.modules[target].file for target in targets if target in index.modules)
        imports[record.file] = dependencies
        for diagnostic in (*record.dynamic_diagnostics, *record.import_diagnostics):
            notes.append({'file': record.file, 'detail': f"Line {diagnostic['line']}: {diagnostic['detail']}"})

    def dependencies(file):
        reached, pending = set(), [file]
        while pending:
            item = pending.pop()
            if item in reached:
                continue
            reached.add(item)
            pending.extend(imports.get(item, set()) - reached)
        return sorted(reached)

    models = []
    for record in index.modules.values():
        tree = trees.get(record.file)
        if tree is None:
            continue
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        for info in record.classes.values():
            node = classes[info.name]
            if (info.file, info.line, info.name) not in declarations:
                if any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward" for item in node.body):
                    notes.append({"file": info.file, "detail": f"{info.name}: model inheritance could not be proven from local source."})
                continue
            # A unique unchanged declaration body and import context can prove a
            # rename. Similar topology, declaration order and generated IDs cannot.
            normalized = copy.deepcopy(node)
            normalized.name = "__review_model__"
            context = [item for item in tree.body if isinstance(item, (ast.Import, ast.ImportFrom))]
            fingerprint = hashlib.sha256((ast.dump(normalized) + repr([ast.dump(item) for item in context])).encode()).hexdigest()
            models.append({"file": info.file, "cls": info.name, "line": info.line,
                           "endLine": info.end_line, "fingerprint": fingerprint,
                           "dependencies": dependencies(info.file)})
    return {"models": sorted(models, key=lambda item: (item["file"], item["line"])),
            "notes": notes, "truncated": request.truncated}
