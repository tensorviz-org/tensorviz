"""Bounded lexical dependencies from host-supplied source, without executing it.

These are source relationships, never a claim that a branch executed, a default
was used, an attribute has one runtime value, or imports cannot be overridden.
"""
from __future__ import annotations

import ast
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from types import SimpleNamespace
from typing import Annotated, Literal
from threading import RLock
from time import monotonic

from pydantic import Field, StrictInt, StrictStr, model_validator

from ..ide.multifile import normalize_workspace_path
from ..validation.contracts import Contract, Digest
from .source_bindings import source_binding

VERSION = 1
MAX_FILES, MAX_BYTES, MAX_NODES, MAX_RECORDS, MAX_RELATIONS = 16, 1024 * 1024, 50000, 8192, 32768
PAGE_BYTES, EXCERPT_BYTES = 48 * 1024, 2048
_CACHE_BYTES, _CACHE_NODES, _CACHE_SECONDS = 2 * MAX_BYTES, 2 * MAX_NODES, 300


class SourceContextRefused(ValueError):
    pass


class SourceEntry(Contract):
    file: Annotated[StrictStr, Field(min_length=1, max_length=512)]
    source: Annotated[StrictStr, Field(max_length=256 * 1024)]
    digest: Digest


class SourceQuery(Contract):
    file: Annotated[StrictStr, Field(min_length=1, max_length=512)]
    symbol: Annotated[StrictStr, Field(min_length=1, max_length=256)] | None = None
    line: Annotated[StrictInt, Field(ge=1)] | None = None
    anchorIds: Annotated[list[Annotated[StrictStr, Field(max_length=80)]], Field(min_length=1, max_length=16)] | None = None
    indexId: Digest | None = None
    direction: Literal["upstream", "downstream", "both"] = "both"
    depth: Annotated[StrictInt, Field(ge=0, le=8)] = 4
    maxEntities: Annotated[StrictInt, Field(ge=1, le=500)] = 100
    offset: Annotated[StrictInt, Field(ge=0, le=100000)] = 0
    limit: Annotated[StrictInt, Field(ge=1, le=100)] = 50
    assumeValue: Annotated[StrictInt, Field(ge=-9007199254740991, le=9007199254740991)] | None = None

    @model_validator(mode="after")
    def selector(self):
        if self.assumeValue is not None and (self.anchorIds is not None or self.direction != "downstream"):
            raise ValueError("configuration changes require a downstream symbol selector")
        if self.anchorIds is not None:
            if self.indexId is None or self.symbol is not None or self.line is not None or len(set(self.anchorIds)) != len(self.anchorIds):
                raise ValueError("exact index anchors required")
        elif self.symbol is None or any(not part.isidentifier() for part in self.symbol.split(".")):
            raise ValueError("a source symbol is required")
        return self


class SourceContextRequest(Contract):
    sources: Annotated[list[SourceEntry], Field(min_length=1, max_length=MAX_FILES)]
    query: SourceQuery


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _bytes(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def _validated_sources(sources):
    total, names = 0, set()
    if not 1 <= len(sources) <= MAX_FILES:
        raise SourceContextRefused("source_limit")
    for entry in sources:
        normalize_workspace_path(entry.file)
        data = entry.source.encode("utf-8")
        total += len(data)
        if len(data) > 256 * 1024 or total > MAX_BYTES:
            raise SourceContextRefused("source_limit")
        if entry.file in names or hashlib.sha256(data).hexdigest() != entry.digest:
            raise SourceContextRefused("source_identity_mismatch")
        names.add(entry.file)
    return total


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return prefix + "." + node.attr if prefix else None
    return None


@dataclass
class Scope:
    file: str
    kind: str
    name: str
    parent: Scope | None = None
    bindings: dict = field(default_factory=lambda: defaultdict(list))
    declarations: set = field(default_factory=set)
    receiver: str | None = None
    attributes: dict = field(default_factory=lambda: defaultdict(list))


class SourceContextIndex:
    def __init__(self, sources, *, parse=None):
        self.sources, self.trees, self.lines, self.modules, self.roots = {}, {}, {}, defaultdict(list), {}
        self.records, self.nodes, self.scopes, self.parents = {}, {}, {}, {}
        self.references, self.imports, self.diagnostics, self.relations = [], {}, [], []
        self.incoming, self.outgoing = defaultdict(list), defaultdict(list)
        _validated_sources(sources)
        total_nodes = 0
        for entry in sorted(sources, key=lambda entry: entry.file):
            if parse is None:
                tree = ast.parse(entry.source)
                nodes = sum(1 for _ in ast.walk(tree))
            else:
                tree, nodes = parse(entry)
            total_nodes += nodes
            if total_nodes > MAX_NODES:
                raise SourceContextRefused("source_index_limit")
            self.sources[entry.file], self.trees[entry.file] = entry.model_copy(deep=True), tree
            self.lines[entry.file] = entry.source.split("\n")
            module = entry.file[:-3].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[:-9]
            self.modules[module].append(entry.file)
            self.roots[entry.file] = Scope(entry.file, "module", "<module>")
        self.node_count = total_nodes
        self.index_id = _digest([VERSION, sorted((file, entry.digest) for file, entry in self.sources.items())])
        for file, tree in self.trees.items():
            self.parents.update({child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)})
            self._visit(tree, self.roots[file])
        self._resolve()
        self._dependencies()
        from .source_documentation import SourceDocumentation
        self.documentation = SourceDocumentation(self)

    def _record(self, node, scope, kind, name, binding=None):
        if len(self.records) >= MAX_RECORDS or len(name) > 256 or len(scope.name) > 512:
            raise SourceContextRefused("source_index_limit")
        entry = self.sources[scope.file]
        lines = self.lines[entry.file]
        qualified = scope.name
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            qualified = name if scope.kind == "module" else scope.name + (".<locals>." if scope.kind == "function" else ".") + name
        found = source_binding(self.index_id, SimpleNamespace(path=entry.file, digest=entry.digest), lines, node,
                               "definition" if kind == "definition" else "call" if kind in ("call", "expression") else "argument", qualified)
        def column(line, offset):
            return len(lines[line - 1].encode()[:offset].decode().encode("utf-16-le")) // 2
        anchor = found["anchors"][0] if found else dict(file=entry.file, digest=entry.digest,
            startLine=node.lineno, endLine=node.end_lineno, startColumn=column(node.lineno, node.col_offset),
            endColumn=column(node.end_lineno, node.end_col_offset), scope=qualified, tokenDigest=None)
        identity = "source:" + _digest([self.index_id, kind, name, anchor])
        item = dict(id=identity, kind=kind, name=name, anchor=anchor)
        if binding:
            item["bindingKind"] = binding
        excerpt = "\n".join(lines[node.lineno - 1:node.end_lineno])
        if len(excerpt.encode()) <= EXCERPT_BYTES:
            item["excerpt"] = excerpt
        else:
            item["omitted"] = "excerpt_bytes"
        self.records[identity], self.nodes[node], self.scopes[node] = item, identity, scope
        return identity

    def _bind(self, node, scope, name, kind):
        identity = self._record(node, scope, "definition", name, kind)
        scope.bindings[name].append(identity)
        if kind == "assignment":
            parent = self.parents.get(node)
            value = parent.value if isinstance(parent, (ast.Assign, ast.AnnAssign)) else None
            if isinstance(value, ast.Constant) and type(value.value) in (int, float, bool, str, type(None)):
                literal = ast.get_source_segment(self.sources[scope.file].source, value)
                if literal and len(literal.encode()) <= 256:
                    self.records[identity]["literalSource"] = literal
        return identity

    def _visit(self, node, scope):
        self.scopes[node] = scope
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            self._bind(node, scope, node.name, kind)
            # Defaults, bases and decorators belong to the enclosing scope.
            headers = [*node.decorator_list]
            if kind == "class":
                headers += [*node.bases, *(keyword.value for keyword in node.keywords)]
            else:
                headers += [*node.args.defaults, *(value for value in node.args.kw_defaults if value is not None)]
            for child in headers:
                self._visit(child, scope)
            name = node.name if scope.kind == "module" else scope.name + (".<locals>." if scope.kind == "function" else ".") + node.name
            inner = Scope(scope.file, kind, name, scope)
            # Declarations cannot be interpreted as ordinary local assignments.
            def declarations(statement):
                if isinstance(statement, (ast.Global, ast.Nonlocal)):
                    inner.declarations.update(statement.names)
                elif not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    for child in ast.iter_child_nodes(statement):
                        declarations(child)
            for child in node.body:
                declarations(child)
            if kind == "function":
                args = node.args
                positional = [*args.posonlyargs, *args.args]
                if scope.kind == "class" and positional and not any(_dotted(item) == "staticmethod" for item in node.decorator_list):
                    inner.receiver = positional[0].arg
                for parameter in [*args.posonlyargs, *args.args, *args.kwonlyargs, *([args.vararg] if args.vararg else []), *([args.kwarg] if args.kwarg else [])]:
                    self._bind(parameter, inner, parameter.arg, "parameter")
                defaults = [*zip(positional[len(positional) - len(args.defaults):], args.defaults), *zip(args.kwonlyargs, args.kw_defaults)]
                for parameter, default in defaults:
                    if default is not None and default in self.nodes:
                        self._edge(self.nodes[default], self.nodes[parameter], "default", "default_only")
            for child in node.body:
                self._visit(child, inner)
            return
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            self.diagnostics.append(dict(file=scope.file, line=node.lineno, reason="nested_expression_scope"))
            scope = Scope(scope.file, "opaque", scope.name + ".<expression>", scope)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    scope.declarations.add("*")
                    self.diagnostics.append(dict(file=scope.file, line=node.lineno, reason="star_import"))
                    continue
                name = alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)
                identity = self._bind(alias, scope, name, "import")
                module = alias.name if isinstance(node, ast.Import) else node.module or ""
                if isinstance(node, ast.ImportFrom) and node.level:
                    parts = scope.file.split("/")[:-1]
                    module = ".".join([*parts[:len(parts) - node.level + 1], *([module] if module else [])]) if node.level <= len(parts) else "<unresolved-relative-import>"
                self.imports[identity] = (module, None if isinstance(node, ast.Import) else alias.name,
                                          not isinstance(node, ast.Import) or bool(alias.asname))
            return
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                self.references.append((node, scope, self._record(node, scope, "reference", node.id)))
            else:
                self._bind(node, scope, node.id, "assignment" if isinstance(node.ctx, ast.Store) else "deletion")
        elif isinstance(node, ast.Attribute):
            name = _dotted(node)
            if name:
                identity = self._record(node, scope, "reference", name)
                self.references.append((node, scope, identity))
                if not isinstance(node.ctx, ast.Load):
                    self.records[identity]["resolution"] = "attribute_write"
                    owner = self._attribute_owner(scope, name.split(".")[0])
                    if owner:
                        owner.attributes[name.partition(".")[2]].append(identity)
        elif isinstance(node, ast.Call):
            name = _dotted(node.func) or "<dynamic-call>"
            self._record(node, scope, "call", name)
            if name in ("exec", "eval", "globals", "locals", "vars", "setattr", "delattr"):
                self.diagnostics.append(dict(file=scope.file, line=node.lineno, reason="dynamic_namespace"))
                scope.declarations.add("*")
                self.roots[scope.file].declarations.add("*")
        elif isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.Subscript)):
            self._record(node, scope, "expression", type(node).__name__)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar, ast.MatchMapping)):
            name = getattr(node, "name", None) or getattr(node, "rest", None)
            if name:
                self._bind(node, scope, name, "conditional_binding")
        if isinstance(node, (ast.AugAssign, ast.NamedExpr, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
            self.diagnostics.append(dict(file=scope.file, line=node.lineno, reason="unsupported_binding_form"))
        for child in ast.iter_child_nodes(node):
            self._visit(child, scope)

    def _edge(self, source, target, relation, certainty="lexical", argument_path=None):
        if source == target:
            return
        if len(self.relations) >= MAX_RELATIONS:
            raise SourceContextRefused("source_index_limit")
        edge = dict(kind="relation", source=source, target=target, relation=relation, certainty=certainty)
        if argument_path is not None:
            edge["argumentPath"] = argument_path
        self.relations.append(edge)
        self.outgoing[source].append(edge)
        self.incoming[target].append(edge)

    def _lookup(self, scope, name):
        current = scope
        while current:
            if current.kind == "opaque" or name in current.declarations or "*" in current.declarations:
                return [], "unresolved_scope"
            if name in current.bindings:
                found = current.bindings[name]
                return found, "lexical" if len(found) == 1 else "ambiguous"
            parent = current.parent
            # A method's bare names do not resolve through its class namespace.
            while parent and parent.kind == "class":
                parent = parent.parent
            current = parent
        return [], "unresolved_name"

    @staticmethod
    def _attribute_owner(scope, receiver):
        while scope:
            if scope.receiver == receiver and scope.parent and scope.parent.kind == "class":
                return scope.parent
            if receiver in scope.bindings:
                return None  # A nested parameter/local shadows the outer receiver.
            scope = scope.parent
        return None

    def _import_targets(self, module, name):
        files = self.modules.get(module, [])
        if len(files) != 1:
            return []
        if "*" in self.roots[files[0]].declarations:
            return []
        return self.roots[files[0]].bindings.get(name, [])

    def _resolve(self):
        for identity, (module, name, _) in self.imports.items():
            found = self._import_targets(module, name) if name else []
            if found:
                for target in found:
                    self._edge(target, identity, "import", "lexical" if len(found) == 1 else "ambiguous")
            elif len(self.modules.get(".".join(filter(None, [module, name])), [])) != 1:
                self.records[identity]["resolution"] = "unresolved_import"
                self.diagnostics.append(dict(file=self.records[identity]["anchor"]["file"],
                    line=self.records[identity]["anchor"]["startLine"], reason="unresolved_import", module=module))
        for node, scope, identity in self.references:
            item = self.records[identity]
            if item.get("resolution") == "attribute_write":
                continue
            parts = item["name"].split(".")
            found, resolution = self._lookup(scope, parts[0])
            if len(parts) > 1:
                imported = self.imports.get(found[0]) if len(found) == 1 else None
                if imported and (imported[1] is None or len(self.modules.get(".".join(filter(None, imported[:2])), [])) == 1):
                    module, member, alias = imported
                    if member is not None:
                        module = ".".join(filter(None, [module, member]))
                    qualified = ".".join([module, *parts[1:]]) if alias else ".".join(parts)
                    prefix, _, symbol = qualified.rpartition(".")
                    found = self._import_targets(prefix, symbol)
                    resolution = "lexical" if len(found) == 1 else "ambiguous" if found else "unresolved_import"
                else:
                    owner = self._attribute_owner(scope, parts[0])
                    name = ".".join(parts[1:])
                    found = [*owner.attributes.get(name, []), *owner.bindings.get(name, [])] if owner else []
                    resolution = "possible_attribute" if found else "unresolved_attribute"
            item["resolution"] = resolution
            for definition in found:
                self._edge(definition, identity, "reference", "possible" if resolution == "possible_attribute" else resolution)

    def _dependencies(self):
        for node, identity in self.nodes.items():
            if self.records[identity]["kind"] == "definition":
                continue
            child, parent = node, self.parents.get(node)
            while parent is not None:
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    break
                if isinstance(parent, (ast.Assign, ast.AnnAssign)) and child is parent.value:
                    targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
                    for target in targets:
                        for part in ast.walk(target):
                            destination = self.nodes.get(part)
                            if destination and (self.records[destination]["kind"] == "definition" or isinstance(part, ast.Attribute)):
                                self._edge(identity, destination, "assignment")
                    break
                destination = self.nodes.get(parent)
                if destination:
                    argument_path = None
                    if isinstance(parent, ast.Call):
                        if child in parent.args:
                            argument_path = ["args", parent.args.index(child)]
                        elif isinstance(child, ast.keyword):
                            argument_path = ["kwargs", child.arg] if child.arg else ["unpacking"]
                    self._edge(identity, destination, "argument" if argument_path else "expression", argument_path=argument_path)
                    break
                child, parent = parent, self.parents.get(parent)

    def query(self, query):
        if query.file not in self.sources:
            raise SourceContextRefused("source_unavailable")
        if query.indexId is not None and query.indexId != self.index_id:
            raise SourceContextRefused("source_index_changed")
        if query.anchorIds:
            if any(identity not in self.records for identity in query.anchorIds):
                raise SourceContextRefused("unknown_source_anchor")
            seeds = query.anchorIds
        else:
            matches = [identity for identity, item in self.records.items() if item["anchor"]["file"] == query.file
                and item["name"] == query.symbol and (query.line is None or item["anchor"]["startLine"] == query.line)]
            # A graph class may carry its module-qualified declaration name.
            # Match that exact supplied module and scope, never a name suffix.
            module = query.file[:-3].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[:-9]
            if not matches and query.line is not None and query.symbol.startswith(module + "."):
                qualified = query.symbol[len(module) + 1:]
                matches = [identity for identity, item in self.records.items() if item["kind"] == "definition"
                    and item["anchor"]["file"] == query.file and item["anchor"]["startLine"] == query.line
                    and item["anchor"]["scope"] == qualified]
            definitions = [identity for identity in matches if self.records[identity]["kind"] == "definition"]
            seeds = definitions or matches
            if not seeds:
                raise SourceContextRefused("unknown_source_symbol")
        selected, pending = {}, deque()
        for identity in seeds[:query.maxEntities]:
            selected[identity] = 0
            pending.append(identity)
        while pending:
            identity = pending.popleft()
            if selected[identity] == query.depth:
                continue
            adjacent = [*(self.incoming[identity] if query.direction != "downstream" else []),
                        *(self.outgoing[identity] if query.direction != "upstream" else [])]
            for edge in adjacent:
                other = edge["source"] if edge["target"] == identity else edge["target"]
                if other not in selected and len(selected) < query.maxEntities:
                    selected[other] = selected[identity] + 1
                    pending.append(other)
        items = [self.records[identity] for identity in selected]
        boundary_count = 0
        for edge in self.relations:
            inside_source, inside_target = edge["source"] in selected, edge["target"] in selected
            if inside_source and inside_target:
                items.append(edge)
            elif inside_source and query.direction != "upstream" or inside_target and query.direction != "downstream":
                boundary_count += 1
                inside, outside = (edge["source"], edge["target"]) if inside_source else (edge["target"], edge["source"])
                items.append(dict(kind="boundary", anchorId=outside, via=edge,
                    reason="depth_limit" if selected[inside] == query.depth else "entity_limit"))
        result = dict(version=VERSION, indexId=self.index_id, evidence="source_only",
            scope="supplied_python_files",
            files=[dict(file=file, digest=entry.digest) for file, entry in sorted(self.sources.items())],
            assumptions=["Lexical relationships do not prove execution order, runtime values or import behavior.",
                         "Unresolved attributes, dynamic namespaces and omitted files require source inspection or new evidence."],
            limitations=["Default-value edges apply only when the caller omits that argument; receiver-attribute edges are possible relationships.",
                         "This is not a call graph or complete control-flow analysis. Type annotations and dynamic namespace resolution are not indexed."],
            diagnostics=self.diagnostics[:32], omittedDiagnostics=max(0, len(self.diagnostics) - 32),
            traversal=dict(entities=len(selected), boundaries=boundary_count, omittedSeeds=max(0, len(seeds) - query.maxEntities)),
            total=len(items), items=[], nextOffset=None)
        effects = {}
        documentation = self.documentation.query(matches if not query.anchorIds and query.line is not None else seeds)
        if documentation is not None:
            result["documentation"] = documentation
        if query.assumeValue is not None:
            from .source_change import SourceChangeAnalysis
            try:
                change = SourceChangeAnalysis(self, seeds, query.assumeValue)
            except ValueError:
                raise SourceContextRefused("configuration_change_unavailable") from None
            result["changeAnalysis"] = change.metadata
            # Evaluate the same bounded selection before paging so analysis-limit
            # outcomes cannot depend on the requested page size or offset.
            effects = {item["id"]: change.effect(item["id"]) for item in items if "id" in item}
        size = _bytes(result) + 128
        for item in items[query.offset:query.offset + query.limit]:
            effect = effects.get(item.get("id"))
            if effect is not None:
                item = dict(item, configurationChange=effect)
            added = _bytes(item) + 1
            if size + added > PAGE_BYTES:
                break
            result["items"].append(item)
            size += added
        if size > PAGE_BYTES or query.offset < len(items) and not result["items"]:
            raise SourceContextRefused("source_result_limit")
        if query.offset + len(result["items"]) < len(items):
            result["nextOffset"] = query.offset + len(result["items"])
        return result


class _SourceCache:
    """Only validated source data is reusable; every inventory change relinks it.

    Bounds apply independently to parsed files and complete indexes. ASTs are read
    without mutation; consumers receive copies of the bounded query result. No
    hash-only lookup, workspace read, runtime state or execution is cached here.
    """
    def __init__(self):
        self.parsed, self.indexes = OrderedDict(), OrderedDict()
        self.lock = RLock()

    def clear(self):
        with self.lock:
            self.parsed.clear()
            self.indexes.clear()

    @staticmethod
    def _prune(cache, count):
        now = monotonic()
        for key, (expires, _, _, _) in list(cache.items()):
            if expires <= now:
                del cache[key]
        while (len(cache) > count or sum(item[1] for item in cache.values()) > _CACHE_BYTES
               or sum(item[2] for item in cache.values()) > _CACHE_NODES):
            cache.popitem(last=False)

    def _parse(self, entry):
        # AST object identity participates in scope/parent maps. Equal contents
        # at different paths must never share the same nodes within an index.
        key = (VERSION, entry.file, entry.digest)
        found = self.parsed.get(key)
        if found is not None:
            self.parsed.move_to_end(key)
            return found[3], found[2]
        tree = ast.parse(entry.source)
        nodes = sum(1 for _ in ast.walk(tree))
        if nodes > MAX_NODES:
            raise SourceContextRefused("source_index_limit")
        self.parsed[key] = (monotonic() + _CACHE_SECONDS, len(entry.source.encode()), nodes, tree)
        self._prune(self.parsed, 32)
        return tree, nodes

    def query(self, request):
        # Validate supplied bytes even for a hit; a claimed digest grants no access.
        size = _validated_sources(request.sources)
        key = (VERSION, MAX_NODES, MAX_RECORDS, MAX_RELATIONS, EXCERPT_BYTES,
               tuple(sorted((entry.file, entry.digest) for entry in request.sources)))
        with self.lock:
            self._prune(self.parsed, 32)
            self._prune(self.indexes, 2)
            found = self.indexes.get(key)
            if found is not None:
                index = found[3]
                self.indexes.move_to_end(key)
            else:
                index = SourceContextIndex(request.sources, parse=self._parse)
                self.indexes[key] = (monotonic() + _CACHE_SECONDS, size, index.node_count, index)
                self._prune(self.indexes, 2)
            return deepcopy(index.query(request.query))


_SOURCE_CACHE = _SourceCache()


def query_source_context(payload):
    request = SourceContextRequest.model_validate(payload)
    return _SOURCE_CACHE.query(request)
