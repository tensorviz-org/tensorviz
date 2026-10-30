"""Project one class's static parse into the document the canvas renders.

The AST parser (``utils/code_parser.py``, M1) already reads a class into nodes,
edges, spans and dataflow. What it does not do — and must not, because the web
importer depends on its current ids — is give those nodes identities that
survive an edit. That is this module's whole job (task.md D6):

* a **module node** is identified by its qualified attribute path (``l1``,
  ``blocks.0.l1``), so renaming ``self.l1`` to ``self.proj`` is exactly one node
  removed and one added, and *nothing else moves*;
* a **functional or opaque node** is identified by ``<fn>@<owner>#<ordinal>``
  — ``add@Net#0``, ``python@Net#0`` — anchored on a hash of the statement that
  produced it. The parser's own ids are ``add_1``/``python_0``, numbered from a
  counter shared by every op in the class, so inserting one call at the top of
  ``forward`` renumbered every node after it and the canvas lost every position.

``owner`` is the qualified attribute path of the module whose ``forward``
created the op, and the class name when that path is empty (the root). The AST
tier only ever sees the root, so it always reads as the class name; the M4 trace
tier will supply ``blocks.0`` and friends for the same rule.

Line numbers are deliberately absent from every identity. Inserting a blank line
must move spans, not cards — so spans live in the spanmap, keyed by an id that
did not move.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

from core.op_registry import spec_for
from models.architecture_models import GraphDocumentV2
from services.ingest.document import to_document

#: Nodes the parser synthesises for the graph's edges, which no source line
#: produced and which therefore carry no span.
BOUNDARY_IDS = ("input", "output")

#: What the ingest side calls the same two nodes (``services/ingest/normalise.py``).
#: They are renamed on the way in so one graph does not carry two vocabularies.
_TRACE_BOUNDARIES = {"input_node": "input", "output_node": "output"}

#: The worker envelope version this module understands (task.md §6.3). Declared here
#: rather than imported from ``services.ingest.worker`` because importing that package
#: pulls in torch, and this module is on the hot path of every keystroke. The two are
#: kept in lockstep by ``tests/capture/test_ingest_worker.py``.
WORKER_PROTOCOL_VERSION = 1

#: Keys the AST tier owns even after a trace has run: where the statement is, what it
#: reads as, and whether it sits inside an ``if``/``for``. The trace cannot know any of
#: them — it sees ops, not syntax — so they are copied across rather than recomputed.
_AST_KEYS = (
    "_lineno", "_end_lineno", "_col_offset", "_end_col_offset",
    "_raw_call", "_control_flow", "_stmt_hash", "_decl_file", "_decl_line",
)


#: How much of a sha256 a span digest keeps. Long enough that a collision is not
#: a practical concern for statements inside one file, short enough to read in a
#: log next to the node id it belongs to.
_DIGEST_CHARS = 12


class ProjectionError(ValueError):
    """The request named a class this source does not define."""


class TraceRejected(ValueError):
    """A trace envelope this module will not merge.

    Separate from ``ProjectionError`` because the two mean different things to the
    caller: a ProjectionError says the request named a class that is not there, a
    TraceRejected says the worker's output cannot be trusted — usually a protocol
    version this backend was not built against. Merging it anyway would produce a
    document that claims to be executed while being partly guesswork.

    ``trace_error`` carries the exception the run itself raised, in the same shape
    ``Projection.error`` uses, whenever the envelope had one. A refusal that names
    only the missing graph tells the user the trace failed, which they watched
    happen; the exception underneath it is the half that says what to change.
    """

    def __init__(self, message: str, *, trace_error: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.trace_error = trace_error


@dataclass(frozen=True)
class Projection:
    """What ``/api/ide/parse`` and ``/api/ide/project`` return."""

    document: GraphDocumentV2
    spanmap: Dict[str, Dict[str, Any]]
    source_digest: str
    graph_id: str
    #: False until a trace has been merged in (M4). The app badges shapes stale
    #: while this is False, so it is a fact about the projection, not a hint.
    traced: bool
    #: The exception that stopped the run, ``{type, message, node_id?, file, line}``
    #: (task.md §6.3), or None when nothing raised. A partial trace without this is
    #: indistinguishable from a model whose second half is dead code: both draw the
    #: same dimmed cards, and only this says which one the user is looking at.
    error: Optional[Dict[str, Any]] = None


def digest_of(text: str) -> str:
    """The full sha256 of a buffer, as the spanmap's ``source_digest``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utf16_columns(source: str) -> Callable[[int, int], int]:
    """``column(line, byte offset)``: an AST column as UTF-16 code units.

    Python's AST counts UTF-8 bytes; VS Code, the Canvas and the capture's source
    anchors count UTF-16 code units, so every span this backend emits does too.
    Lines split as the AST numbers them: str.splitlines() also breaks at form
    feeds, U+0085 and U+2028, and would read an earlier line's text.
    """
    lines = re.split(r"\r\n?|\n", source)

    def column(line: int, offset: int) -> int:
        text = lines[line - 1] if 0 < line <= len(lines) else ""
        if text.isascii():
            return offset
        return len(text.encode("utf-8")[:offset].decode("utf-8", "ignore").encode("utf-16-le")) // 2

    return column


def utf16_spans(spans: Dict[str, Dict[str, Any]], sources: Mapping[str, str]) -> Dict[str, Dict[str, Any]]:
    """AST-tier spans with UTF-16 columns, each converted against its own file."""
    columns: Dict[str, Callable[[int, int], int]] = {}
    converted: Dict[str, Dict[str, Any]] = {}
    for node_id, span in spans.items():
        file = span.get("file")
        if file in sources and file not in columns:
            columns[file] = utf16_columns(sources[file])
        column = columns.get(file)
        converted[node_id] = span if column is None else {
            **span,
            "start": [span["start"][0], column(*span["start"])],
            "end": [span["end"][0], column(*span["end"])],
        }
    return converted


def review_class(source: str, cls: str, rel_file: str) -> Dict[str, Any]:
    """Source facts missed by layer lowering, without evaluating defaults or bodies."""
    tree = ast.parse(source)
    target = next(n for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == cls)
    init = next((n for n in target.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    defaults = {}
    if init:
        positional = init.args.posonlyargs + init.args.args
        defaults = {arg.arg: ast.unparse(value) for arg, value in zip(
            positional[len(positional) - len(init.args.defaults):], init.args.defaults)}
        defaults.update({arg.arg: ast.unparse(value) for arg, value in zip(
            init.args.kwonlyargs, init.args.kw_defaults) if value is not None})
    column = utf16_columns(source)
    span = dict(file=rel_file, start=[target.lineno, column(target.lineno, target.col_offset)],
                end=[target.end_lineno, column(target.end_lineno, target.end_col_offset)], kind="decl")
    declarations: Dict[str, List[Dict[str, Any]]] = {}
    assignments: Dict[str, List[Dict[str, Any]]] = {}
    resets: List[int] = []

    def literal_target(node):
        if isinstance(node, ast.Name):
            return node.id == "self"
        if isinstance(node, ast.Attribute):
            return literal_target(node.value)
        if isinstance(node, ast.Subscript):
            index = node.slice
            return (not isinstance(node.value, ast.Name) and literal_target(node.value)
                    and isinstance(index, ast.Constant) and type(index.value) in (int, str))
        return False

    def path_segments(node):
        if isinstance(node, ast.Name):
            return []
        if isinstance(node, ast.Attribute):
            return [*path_segments(node.value), dict(kind="attribute", key=node.attr)]
        return [*path_segments(node.value), dict(kind="index", key=node.slice.value)]

    def collect(statements, guards=()):
        for statement in statements:
            call = statement.value if isinstance(statement, ast.Expr) else None
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "__init__"
                    and isinstance(call.func.value, ast.Call) and isinstance(call.func.value.func, ast.Name)
                    and call.func.value.func.id == "super"):
                resets.append(statement.lineno)
            targets = statement.targets if isinstance(statement, ast.Assign) else (
                [statement.target] if isinstance(statement, ast.AnnAssign) and statement.value is not None else [])
            for attribute in targets:
                direct = (isinstance(attribute, ast.Attribute) and isinstance(attribute.value, ast.Name)
                          and attribute.value.id == "self")
                if not direct and not (isinstance(attribute, (ast.Attribute, ast.Subscript)) and literal_target(attribute)):
                    continue
                name = attribute.attr if direct else ast.unparse(attribute)[5:]
                entries = declarations if direct else assignments
                entries.setdefault(name, []).append(dict(
                    expression=ast.dump(statement.value, include_attributes=False), guards=guards,
                    path=path_segments(attribute),
                    value=ast.unparse(statement.value),
                    title=ast.unparse(statement.value.func) if isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, (ast.Name, ast.Attribute)) else "Assignment",
                    span=dict(file=rel_file, start=[statement.lineno, column(statement.lineno, statement.col_offset)],
                              end=[statement.end_lineno, column(statement.end_lineno, statement.end_col_offset)], kind="decl")))
            # Retain branch/loop conditions without selecting or executing a branch.
            # Nested functions/classes bind a different scope and are not members here.
            if isinstance(statement, (ast.If, ast.While, ast.For)):
                condition = ast.dump(statement.iter if isinstance(statement, ast.For) else statement.test, include_attributes=False)
                collect(statement.body, guards + ((type(statement).__name__, condition, "body"),))
                collect(statement.orelse, guards + ((type(statement).__name__, condition, "else"),))
            # Handler, context and case bodies bind members too, with the guards
            # review_declarations gives the same statements.
            elif isinstance(statement, ast.With):
                condition = repr([ast.dump(item, include_attributes=False) for item in statement.items])
                collect(statement.body, guards + (("With", condition, "body"),))
            elif isinstance(statement, (ast.Try, ast.TryStar)):
                kind = type(statement).__name__
                collect(statement.body, guards + ((kind, "", "body"),))
                for handler in statement.handlers:
                    condition = ast.dump(handler.type, include_attributes=False) if handler.type else ""
                    collect(handler.body, guards + ((kind, condition, "except"),))
                collect(statement.orelse, guards + ((kind, "", "else"),))
                collect(statement.finalbody, guards + ((kind, "", "finally"),))
            elif isinstance(statement, ast.Match):
                subject = ast.dump(statement.subject, include_attributes=False)
                for case in statement.cases:
                    condition = ast.dump(ast.match_case(pattern=case.pattern, guard=case.guard, body=[]), include_attributes=False)
                    collect(case.body, guards + (("Match", subject, condition),))

    if init:
        collect(init.body)
    members = {name: dict(
        digest=digest_of(repr([(entry["expression"], entry["guards"]) for entry in entries])),
        span=entries[0]["span"], ambiguous=len(entries) != 1,
    ) for name, entries in declarations.items()}
    # Formatting, comments and docstrings do not change this comparison evidence.
    for node in ast.walk(target):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:]
    facts = dict(defaults=defaults, members=members, digest=digest_of(ast.dump(target, include_attributes=False)), span=span)
    if assignments:
        # A later write to an ancestor (or a base initializer) can replace the
        # object targeted here. Keep the statement, but never pick its result.
        def ambiguous(name, entries):
            line = entries[0]["span"]["start"][0]
            return (len(entries) != 1 or any(reset > line for reset in resets)
                    or any((name.startswith(prefix + ".") or name.startswith(prefix + "["))
                           and any(entry["span"]["start"][0] >= line for entry in writes)
                           for prefix, writes in {**declarations, **assignments}.items()))

        uncertain = {name for name, entries in assignments.items() if ambiguous(name, entries)}
        facts["assignments"] = {"@assignment:" + name: dict(
            target=name, path=entries[0]["path"], digest=digest_of(repr([(entry["expression"], entry["guards"]) for entry in entries])),
            span=entries[0]["span"], ambiguous=name in uncertain, conditional=any(entry["guards"] for entry in entries),
            title=entries[0]["title"] if name not in uncertain else "Multiple assignments",
            value=entries[0]["value"] if name not in uncertain else "Multiple assignments",
        ) for name, entries in assignments.items()}
    # The rest of the file must be identical before a previous capture can give
    # these assignments architectural context. Never ignore imports, forward
    # changes, conditional writes, multiple targets or unknown mutation forms.
    structure = deepcopy(tree)
    structure_target = next(n for n in ast.walk(structure) if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == cls)
    structure_init = next((n for n in structure_target.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    removable = {tuple(fact["span"]["start"]) for fact in facts.get("assignments", {}).values()
                 if not fact["ambiguous"] and not fact["conditional"]}
    if structure_init:
        structure_init.body = [n for n in structure_init.body if not ((n.lineno, column(n.lineno, n.col_offset)) in removable
            and (isinstance(n, ast.AnnAssign) or isinstance(n, ast.Assign) and len(n.targets) == 1))]
    facts["structureDigest"] = digest_of(ast.dump(structure, include_attributes=False))
    return facts


def graph_id_for(rel_file: str, cls: str) -> str:
    """The CAS anchor an AI turn's ``base_revision`` points at (task.md D7).

    Derived, never stored: the same file and class always name the same record,
    so a second window — or a later session — upserts into it instead of minting
    a rival. ``rel_file`` is workspace-relative and separator-normalised so the
    same file opened on Windows and on macOS agrees with itself.
    """
    normalised = rel_file.replace("\\", "/").strip("/")
    return "ide:" + hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:12] + "#" + cls


def source_review_graph_id(graph_id: str) -> str:
    """Keep single-file candidates independent of live multi-file refreshes."""
    return "source-review:" + graph_id


def class_names(source: str) -> List[str]:
    """Every class defined at any depth in ``source``, in source order."""
    return [n.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef)]


def init_attributes(source: str, cls: str) -> Set[str]:
    """The ``self.<attr>`` names ``cls.__init__`` binds.

    This is the D6 module-node rule stated in the only place it is knowable:
    a node id that is one of these came from a declaration, and keeps its name;
    everything else came from ``forward`` and is renamed to ``fn@owner#ordinal``.

    It re-reads the source with ``ast`` rather than asking the parser, because
    ``CodeParserService`` does not report which of its node ids were attributes
    — and guessing from the id's shape misfiles an attribute a user happened to
    call ``add_1``.
    """
    tree = ast.parse(source)
    target = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls), None
    )
    if target is None:
        raise ProjectionError(f"{cls} is not defined in this file")
    init = next(
        (n for n in target.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None
    )
    if init is None:
        return set()

    names: Set[str] = set()

    def record(node: ast.AST) -> None:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            names.add(node.attr)

    for statement in ast.walk(init):
        if isinstance(statement, ast.Assign):
            for element in statement.targets:
                record(element)
        elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
            record(statement.target)
    return names


def module_path_of(rel_file: str) -> str:
    """The dotted module path of the file being projected.

    Workspace-relative already, so this is a spelling change and not a guess —
    the same path ``services/ingest/normalise.import_path_for`` computes for the
    executed tier, arrived at from the other side.
    """
    text = rel_file.replace("\\", "/").strip("/")
    if text.endswith(".py"):
        text = text[:-3]
    return text.replace("/", ".")


def workspace_classes(source: str, rel_file: str) -> Dict[str, str]:
    """``{class name: the module that defines it}`` for one buffer (task.md M6).

    Two syntactic sources and no execution: the classes this file defines (whose
    module is the file itself) and the names it imports from somewhere. A block
    a user keeps in ``models/blocks.py`` is the second case, and it is the one
    the parser cannot see at all — it reads one file, so ``self.blk = MyBlock(16)``
    reaches it as a call to a name it has never heard of.

    Nothing here decides whether a name is torch's. That question is answered by
    the op registry at the call site: a node whose id resolves to a real OpSpec is
    left alone however it was imported, so ``from torch.nn import Linear`` stays a
    Linear rather than becoming somebody's custom block.
    """
    found: Dict[str, str] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return found
    own = module_path_of(rel_file)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            found.setdefault(node.name, own)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                found.setdefault(alias.asname or alias.name, node.module)
    return found


def adopt_custom_class(node: Dict[str, Any], *, classes: Dict[str, str]) -> None:
    """Retype a node the registry does not know but this file does.

    The result is the bare ``custom_code`` placeholder plus ``_class`` and
    ``_import_path``. The op stays bare on purpose: ``graph_input_safety``'s
    ``_LAYER_OPERATION`` admits only ``nn.``/``torch.nn.``/``torch.`` prefixes, so
    a dotted user path could never cross into codegen — and the identity of the
    class belongs in the parameters, where every plane preserves it.
    """
    data = node.setdefault("data", {})
    layer_type = data.setdefault("layerType", {})
    params = dict(data.get("parameters") or {})
    layer_id = _layer_id(node)
    name = str(params.get("_class") or layer_type.get("name") or "")
    if not name or name not in classes:
        return
    if layer_id != "custom_code" and spec_for(layer_id) is not None:
        return
    params["_class"] = name
    params["_import_path"] = classes[name]
    data["parameters"] = params
    layer_type["id"] = "custom_code"
    layer_type["name"] = "custom_code"


def _params(node: Dict[str, Any]) -> Dict[str, Any]:
    return dict(((node.get("data") or {}).get("parameters") or {}))


def _layer_id(node: Dict[str, Any]) -> str:
    return str((((node.get("data") or {}).get("layerType")) or {}).get("id") or "op").lower()


def _statement_text(node: Dict[str, Any]) -> str:
    """What a node's identity and its span digest hash over.

    ``_raw_call`` is ``ast.unparse`` output, so it is already normalised for
    whitespace and comments: reformatting the statement does not move the node,
    while changing what it computes does.
    """
    params = _params(node)
    return str(params.get("_raw_call") or "")


def _sort_key(node: Dict[str, Any], index: int) -> Tuple[int, int, int]:
    params = _params(node)
    line = params.get("_lineno")
    column = params.get("_col_offset")
    return (
        int(line) if isinstance(line, int) else 1 << 30,
        int(column) if isinstance(column, int) else 0,
        index,
    )


def identity_map(nodes: List[Dict[str, Any]], *, cls: str, attributes: Set[str]) -> Dict[str, str]:
    """``{parser id: projection id}`` for one class's nodes (D6).

    Boundary and attribute nodes keep the id they have. Everything else is
    ordered by where it stands in the source — not by the order the parser
    happened to insert it into the node list — and numbered per ``(fn, owner)``,
    so a new ``add`` never renumbers an unrelated ``python`` node.
    """
    mapping: Dict[str, str] = {}
    ordinals: Dict[Tuple[str, str], int] = {}
    ordered = sorted(
        ((node, index) for index, node in enumerate(nodes)),
        key=lambda pair: _sort_key(pair[0], pair[1]),
    )
    for node, _index in ordered:
        node_id = str(node.get("id"))
        if node_id in BOUNDARY_IDS or node_id in attributes or _params(node).get("_review_inherited"):
            mapping[node_id] = node_id
            continue
        key = (_layer_id(node), cls)
        ordinal = ordinals.get(key, 0)
        ordinals[key] = ordinal + 1
        mapping[node_id] = f"{key[0]}@{key[1]}#{ordinal}"
    return mapping


def _rename(graph: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    nodes = []
    for node in graph.get("nodes") or []:
        renamed = dict(node)
        renamed["id"] = mapping.get(str(node.get("id")), str(node.get("id")))
        nodes.append(renamed)
    edges = []
    occurrences = {}
    for edge in graph.get("edges") or []:
        source = mapping.get(str(edge.get("source")), str(edge.get("source")))
        target = mapping.get(str(edge.get("target")), str(edge.get("target")))
        renamed = dict(edge)
        renamed["source"] = source
        renamed["target"] = target
        key = (source, target)
        ordinal = occurrences.get(key, 0)
        occurrences[key] = ordinal + 1
        renamed["id"] = f"edge_{source}_{target}" + (f"_{ordinal}" if ordinal else "")
        edges.append(renamed)
    return {"nodes": nodes, "edges": edges}


def _annotate(node: Dict[str, Any], *, attributes: Set[str]) -> None:
    """Stamp the provenance keys the canvas and the AI harness read (§6.1).

    They go in ``data.parameters`` under a leading underscore because that is the
    only per-node channel that survives the torch constructors, code generation,
    the modularizer and the frontend's save allowlist (task.md §3.3).
    """
    data = node.setdefault("data", {})
    params = dict(data.get("parameters") or {})
    node_id = str(node.get("id"))
    if _layer_id(node) == "modulelist":
        # A static member list is descriptive only. Expansion requires actual
        # child nodes, which this tier has not materialized or wired.
        data.update({
            "isGroup": True,
            "expanded": False,
            "moduleInstanceId": node_id,
            "moduleTemplateId": "ModuleList",
            "moduleLayerCount": 0,
            "moduleMembers": [],
        })
    if node_id not in BOUNDARY_IDS:
        statement = _statement_text(node)
        if statement:
            params["_stmt_hash"] = hashlib.sha256(statement.encode("utf-8")).hexdigest()[
                :_DIGEST_CHARS
            ]
        params["_kind"] = (
            "class"
            if node_id in attributes
            else ("statement" if _layer_id(node) == "python" else "function")
        )
    # "Static" means read off the source, never executed — the canvas badges it
    # stale on that basis. M4's trace merge is what replaces it with "observed".
    params["_shape_provenance"] = "static"
    data["parameters"] = params


def _spanmap(
    nodes: List[Dict[str, Any]], *, attributes: Set[str], rel_file: str
) -> Dict[str, Dict[str, Any]]:
    """``{node_id: {file, start, end, kind, digest}}`` (task.md §6.2).

    ``kind`` is what makes two-way navigation legible: a ``decl`` span is the
    ``__init__`` line that declared the layer, a ``call`` span is the ``forward``
    line that ran it. Inserting a line above ``forward`` moves every ``call``
    span and no ``decl`` span, which is exactly the difference the test asserts.
    """
    spans: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        node_id = str(node.get("id"))
        if node_id in BOUNDARY_IDS:
            continue
        params = _params(node)
        line = params.get("_lineno")
        if not isinstance(line, int):
            continue
        end_line = params.get("_end_lineno")
        column = params.get("_col_offset")
        end_column = params.get("_end_col_offset")
        spans[node_id] = {
            "file": str(params.get("_source_file") or rel_file),
            "start": [line, int(column) if isinstance(column, int) else 0],
            "end": [
                int(end_line) if isinstance(end_line, int) else line,
                int(end_column) if isinstance(end_column, int) else 0,
            ],
            "kind": "decl" if node_id in attributes else "call",
            "digest": params.get("_stmt_hash", ""),
        }
    return spans


# ── the executed tier (task.md M4) ───────────────────────────────────────────


def _same_file(span_file: str, rel_file: str) -> bool:
    """Whether a worker span (absolute) points at the file being projected (relative).

    The worker runs in the user's interpreter with the workspace's own layout and
    reports absolute paths; the editor talks in workspace-relative ones and never
    tells the backend where the workspace is. Comparing by suffix is what bridges
    them, and it is safe here because the only question asked is "is this the file
    the user is looking at", where a false positive would need a second file whose
    whole relative path is a suffix of this one.
    """
    left = os.path.normpath(span_file).replace(os.sep, "/")
    right = os.path.normpath(rel_file).replace(os.sep, "/").lstrip("./")
    return left == right or left.endswith("/" + right)


def _match_workspace_file(
    span_file: str, rel_file: str, source_files: Optional[Set[str]]
) -> Optional[str]:
    """Which supplied file a trace span points at, or None if it points elsewhere.

    Single-file callers pass ``source_files=None`` and the answer is the root or
    nothing — the historic behaviour. Multi-file projections pass the whole
    supplied set, so a frame in ``models/blocks.py`` is as much the user's own
    line as one in the root.
    """
    if source_files is None:
        return rel_file if _same_file(span_file, rel_file) else None
    for rel in sorted(source_files):
        if _same_file(span_file, rel):
            return rel
    return None


def _check_envelope(
    trace: Dict[str, Any],
    *,
    rel_file: str,
    source_files: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """The worker envelope, or a loud refusal (task.md §6.3)."""
    version = trace.get("protocol_version")
    if version != WORKER_PROTOCOL_VERSION:
        raise TraceRejected(
            f"trace worker protocol {version!r} is not {WORKER_PROTOCOL_VERSION}; "
            "the extension and this backend are from different builds"
        )
    result = trace.get("result")
    if isinstance(result, dict) and isinstance(result.get("diagnostics"), dict) and result["diagnostics"].get("semantic_capture") is not None:
        # The canonical recorder can retain a valid partial/identity capture
        # even when the old visual recorder had nothing to draw. Validation is
        # mandatory in project_trace before this reaches a document.
        return trace
    if not isinstance(result, dict) or not isinstance(result.get("graph"), dict):
        raise _nothing_to_draw(
            "trace envelope carries no result graph", trace, rel_file=rel_file,
            source_files=source_files,
        )
    if not (result["graph"].get("nodes") or []):
        raise _nothing_to_draw(
            "trace envelope carries an empty graph", trace, rel_file=rel_file,
            source_files=source_files,
        )
    return trace


def _nothing_to_draw(
    fallback: str, trace: Dict[str, Any], *, rel_file: str,
    source_files: Optional[Set[str]] = None,
) -> TraceRejected:
    """The refusal for an envelope with no graph in it, carrying the reason there is none.

    A run that produced nothing nearly always produced an exception saying why —
    a constructor that wanted an argument, a checkpoint that was not on disk — and
    the commonest of them, ``__init__() missing 1 required positional argument``,
    names the fix in its own text. That is the only actionable half of the
    envelope, and it sits right next to the empty half; refusing on the empty half
    alone throws it away and leaves the user with "the trace failed", which they
    already knew. ``fallback`` still stands for a worker that reported no error
    either, because "no graph and no reason" is itself worth saying out loud.
    """
    error = _trace_error(trace, {}, rel_file=rel_file, source_files=source_files)
    if error is None:
        return TraceRejected(fallback)
    named = ": ".join(
        str(part) for part in (error.get("type"), error.get("message")) if part
    )
    return TraceRejected(
        f"the model never ran: {named}" if named else fallback, trace_error=error
    )


def _trace_identity(
    nodes: List[Dict[str, Any]], materialised: Set[str], *, cls: str
) -> Dict[str, str]:
    """``{trace id: projection id}`` (D6) for one executed graph.

    Module nodes already ARE their qualified attribute path, so they keep their id
    and line up with the AST's by construction. Only the nodes the tracer invented
    — a residual add, a functional flatten, a global pool — need renaming, because
    ``add1``/``add2`` are numbered in fold order and would shuffle the moment an
    earlier statement started producing a merge.
    """
    mapping: Dict[str, str] = {}
    ordinals: Dict[Tuple[str, str], int] = {}
    for node in sorted(nodes, key=_trace_sort_key):
        node_id = str(node.get("id"))
        if node_id in _TRACE_BOUNDARIES:
            mapping[node_id] = _TRACE_BOUNDARIES[node_id]
            continue
        if node_id not in materialised:
            mapping[node_id] = node_id
            continue
        owner = str((node.get("data") or {}).get("parentId") or cls)
        key = (_layer_id(node), owner)
        ordinal = ordinals.get(key, 0)
        ordinals[key] = ordinal + 1
        mapping[node_id] = f"{key[0]}@{key[1]}#{ordinal}"
    return mapping


def _trace_sort_key(node: Dict[str, Any]) -> Tuple[int, str]:
    params = _params(node)
    line = params.get("_decl_line")
    if not isinstance(line, int):
        line = params.get("_source_line")
    return (int(line) if isinstance(line, int) else 1 << 30, str(node.get("id")))


def _covering(ast_id: str, trace_ids: Set[str]) -> Optional[str]:
    """The executed node that stands for ``ast_id``, if one does.

    Exact match first, then a descendant: ``self.blocks = nn.ModuleList(...)`` is one
    AST node and the trace renders its instances as ``blocks.0``. The declaration is
    not unreached — it is on the canvas under a more specific name — so it must not
    be drawn a second time as a dimmed ghost.
    """
    if ast_id in trace_ids:
        return ast_id
    prefix = ast_id + "."
    candidates = sorted(t for t in trace_ids if t.startswith(prefix))
    return candidates[0] if candidates else None


def _trace_error(
    trace: Dict[str, Any],
    mapping: Dict[str, str],
    *,
    rel_file: str,
    source_files: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """The exception that stopped the run, in the canvas's own terms.

    ``node_id`` goes through the same identity mapping the nodes did, or the card
    it names would not be on the graph it is meant to point at. ``file``/``line``
    are the innermost frame inside the file being projected — the torch frames
    under it are true and useless, and the user cannot edit them — falling back to
    the innermost frame there is when the raise happened somewhere else entirely,
    because naming another file is still better than naming none.
    """
    error = trace.get("error")
    if not isinstance(error, dict):
        return None
    frames = [frame for frame in (error.get("frames") or []) if isinstance(frame, dict)]
    own = next(
        (
            f
            for f in reversed(frames)
            if _match_workspace_file(str(f.get("file") or ""), rel_file, source_files)
        ),
        frames[-1] if frames else None,
    )
    node_id = error.get("node_id")
    surfaced: Dict[str, Any] = {
        "type": error.get("type"),
        "message": error.get("message"),
        "node_id": mapping.get(str(node_id), str(node_id)) if node_id else None,
    }
    if own is not None:
        surfaced["file"] = own.get("file")
        surfaced["line"] = own.get("line")
    return surfaced


def _merge(
    ast_graph: Dict[str, Any],
    ast_spans: Dict[str, Dict[str, Any]],
    trace: Dict[str, Any],
    *,
    cls: str,
    rel_file: str,
    source_files: Optional[Set[str]] = None,
    source_digests: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Fold an executed graph and a parsed one into the document the canvas draws.

    The rule (task.md §5), in the order it is applied:

    * the **trace wins** the node set, the edges, the shapes and the repeat counts —
      it is a record of what Python did, and the parser is a guess about what it
      would do;
    * the **AST supplies** the declaration span, the raw call text and
      ``_control_flow`` for every node it recognises, because those are syntax and
      the tracer never sees syntax;
    * a node the AST found and the trace never reached stays, dimmed
      (``_shape_provenance: "unreached"``). A trace must never silently delete a card
      the user can see in their own file — an untaken ``if`` branch is exactly the
      case, and answering "that layer does not exist" would be a lie.
    """
    result = _check_envelope(trace, rel_file=rel_file, source_files=source_files)["result"]
    trace_graph = result["graph"]
    diagnostics = result.get("diagnostics") or {}
    join = None
    if diagnostics.get("semantic_capture") is not None:
        from services.semantic.source_join import CaptureJoin
        try:
            join = CaptureJoin(diagnostics["semantic_capture"], model_file=rel_file, class_name=cls,
                               source_digests=source_digests or {})
        except (TypeError, ValueError, KeyError):
            raise TraceRejected("The semantic capture is invalid or does not match the projected saved source.") from None
    materialised = set(diagnostics.get("materialised_ops") or [])

    trace_nodes = list(trace_graph.get("nodes") or [])
    mapping = _trace_identity(trace_nodes, materialised, cls=cls)
    renamed = _rename(
        {"nodes": trace_nodes, "edges": trace_graph.get("edges") or []}, mapping
    )
    trace_spans = {
        mapping.get(node_id, node_id): span
        for node_id, span in (trace.get("spans") or {}).items()
    }
    trace_ids = {str(node.get("id")) for node in renamed["nodes"]}

    # Lines something actually ran on, keyed by (file, line): a multi-file
    # projection spans several of the user's files, and line 34 of blocks.py must
    # not cover an unrelated statement at line 34 of net.py. Single-file callers
    # pass no source set and reduce to the historic root-only behaviour.
    executed_lines: Set[Tuple[str, int]] = set()
    for span in trace_spans.values():
        if span.get("kind") != "frame":
            continue
        matched = _match_workspace_file(str(span.get("file") or ""), rel_file, source_files)
        if matched is None:
            continue
        start = span.get("start")
        if isinstance(start, list) and start and isinstance(start[0], int):
            executed_lines.add((matched, int(start[0])))

    ast_by_id = {str(node.get("id")): node for node in ast_graph["nodes"]}
    covers: Dict[str, str] = {}  # trace id -> the AST id it stands for
    unreached: List[Dict[str, Any]] = []
    #: AST nodes whose line ran under another node's name. Their wiring is still the
    #: only description of how the graph joins up there, so `_merge_edges` splices
    #: through them rather than letting their edges vanish with them.
    absorbed: Set[str] = set()
    for ast_id, node in ast_by_id.items():
        if ast_id in BOUNDARY_IDS:
            continue
        fact = join.correspondence(node) if join is not None else None
        known_entity = fact is not None and (fact.moduleId is not None or fact.status == "state")
        covering = _covering(ast_id, trace_ids)
        if known_entity and covering != ast_id:
            # A child invocation cannot stand for its registered parent instance.
            # Keep both identities even when the legacy drawing omitted the parent.
            covering = None
        if covering is not None:
            covers.setdefault(covering, ast_id)
            continue
        line = _params(node).get("_lineno")
        node_span = ast_spans.get(ast_id) or {}
        node_file = str(node_span.get("file") or rel_file)
        if not known_entity and isinstance(line, int) and (node_file, line) in executed_lines:
            # Ran, under another node's name. Remember it: an edge that passed
            # THROUGH this node still has to reach whatever it fed.
            absorbed.add(ast_id)
            continue
        unreached.append(node)

    nodes: List[Dict[str, Any]] = []
    for node in renamed["nodes"]:
        node_id = str(node.get("id"))
        ast_node = ast_by_id.get(covers.get(node_id, node_id))
        if ast_node is not None:
            data = node.setdefault("data", {})
            params = dict(data.get("parameters") or {})
            ast_params = _params(ast_node)
            for key in _AST_KEYS:
                if key in ast_params and key not in params:
                    params[key] = ast_params[key]
            if "_kind" in ast_params:
                params["_kind"] = ast_params["_kind"]
            data["parameters"] = params
        nodes.append(node)

    spans: Dict[str, Dict[str, Any]] = dict(trace_spans)
    for trace_id, ast_id in covers.items():
        if ast_id in ast_spans:
            # The declaration in the file the user has open beats a frame in torch's.
            spans[trace_id] = ast_spans[ast_id]
    for node in unreached:
        node_id = str(node.get("id"))
        data = node.setdefault("data", {})
        params = dict(data.get("parameters") or {})
        params["_shape_provenance"] = "unreached"
        data["parameters"] = params
        nodes.append(node)
        if node_id in ast_spans:
            spans[node_id] = ast_spans[node_id]

    edges = _merge_edges(renamed["edges"], ast_graph["edges"], nodes, absorbed=absorbed)
    if join is not None:
        # Source/trace wiring still uses the legacy adapter until reversible
        # projection lands. Coverage never comes from membership in that drawing.
        nodes = [join.decorate(node) if str(node.get("id")) not in BOUNDARY_IDS else node for node in nodes]
    return (
        {"nodes": nodes, "edges": edges},
        spans,
        _trace_error(trace, mapping, rel_file=rel_file, source_files=source_files),
    )


def _merge_edges(
    trace_edges: List[Dict[str, Any]],
    ast_edges: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]],
    *,
    absorbed: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Trace edges, plus the AST's wiring for every node the trace never reached.

    A node with no observed shape has no observed edges either, and the importer
    gives such a node a boundary edge at each end purely so it is not floating —
    which renders an untaken ``if`` branch as a second path from input to output.
    The parser knows better: it read ``x = self.up(x)`` and then ``self.down(x)``.
    So where the AST has real wiring for a dimmed node, it replaces those made-up
    boundary edges rather than being drawn beside them.
    """
    present = {str(node.get("id")) for node in nodes}
    roles = {(str(e.get("source")), str(e.get("target"))): e.get("data")
             for e in ast_edges if (e.get("data") or {}).get("role") == "residual"}
    ast_edges = _splice(ast_edges, absorbed or set())
    dimmed = {
        str(node.get("id"))
        for node in nodes
        if _params(node).get("_shape_provenance") == "unreached"
    }
    wanted = [
        edge
        for edge in ast_edges
        if dimmed & {str(edge.get("source")), str(edge.get("target"))}
        and str(edge.get("source")) in present
        and str(edge.get("target")) in present
    ]
    replaced = {
        node_id
        for node_id in dimmed
        if any(node_id in (str(e.get("source")), str(e.get("target"))) for e in wanted)
    }

    edges: List[Dict[str, Any]] = []
    for edge in trace_edges:
        source, target = str(edge.get("source")), str(edge.get("target"))
        boundary = {source, target} & set(BOUNDARY_IDS)
        if boundary and (replaced & {source, target}):
            continue
        edges.append({**edge, "data": roles[(source, target)]} if (source, target) in roles else edge)

    drawn = {(str(e.get("source")), str(e.get("target"))) for e in edges}
    for edge in wanted:
        pair = (str(edge.get("source")), str(edge.get("target")))
        if pair in drawn:
            continue
        drawn.add(pair)
        edges.append(dict(edge))
    return edges


def _splice(edges: List[Dict[str, Any]], absorbed: Set[str]) -> List[Dict[str, Any]]:
    """AST wiring with the un-drawn nodes bridged through rather than cut out.

    A node whose statement ran inside another node's frame is not drawn — the trace
    already accounts for it. But it can sit in the MIDDLE of the parser's wiring, and
    the both-endpoints-present filter below then drops the edges on either side of it,
    silently orphaning whatever it joined.

    `self.embed(tokens) + self.pos[:, :n]` is exactly that: the parser emits
    `embed -> add <- pos` and `add -> encoder`, the tracer accounts for the add inside
    the encoder's frame, and `pos` — a parameter the tracer can never reach, because
    `nn.Parameter` has no forward — was left hanging off the canvas with no edge at all.
    Splicing turns `pos -> add -> encoder` into `pos -> encoder`, which is what the
    source says: the parameter feeds what the encoder receives.

    Bounded by the number of absorbed nodes, so a chain of them terminates.
    """
    if not absorbed:
        return edges
    out = [dict(e) for e in edges]
    for _ in range(len(absorbed) + 1):
        bridged: List[Dict[str, Any]] = []
        changed = False
        for edge in out:
            target = str(edge.get("target"))
            if target not in absorbed:
                bridged.append(edge)
                continue
            # every edge leaving the absorbed node becomes a continuation of this one
            for onward in out:
                if str(onward.get("source")) != target:
                    continue
                nxt = str(onward.get("target"))
                if nxt == str(edge.get("source")):
                    continue  # never splice a node onto itself
                bridged.append({
                    "id": f"{edge.get('id')}~{onward.get('id')}",
                    "source": edge.get("source"),
                    "target": onward.get("target"),
                })
                changed = True
        seen = set()
        deduped = []
        for edge in bridged:
            pair = (str(edge.get("source")), str(edge.get("target")))
            if pair in seen:
                continue
            seen.add(pair)
            deduped.append(edge)
        out = deduped
        if not changed:
            break
    return [e for e in out if str(e.get("source")) not in absorbed
            and str(e.get("target")) not in absorbed]

def project_trace(ast_graph, ast_spans, trace, *, cls, rel_file, graph_id,
                  source_digests, source_files=None, source_texts=None):
    result = _check_envelope(trace, rel_file=rel_file, source_files=source_files)["result"]
    diagnostics = result.get("diagnostics") or {}
    if not isinstance(diagnostics, dict):
        raise TraceRejected("The trace diagnostics are malformed.")
    payload = diagnostics.get("semantic_capture")
    if payload is not None:
        from services.semantic.document import capture_document, DocumentBudgetExceeded
        try:
            document, spans = capture_document(payload, name=cls, graph_id=graph_id,
                model_file=rel_file, source_digests=source_digests,
                source_graph=ast_graph, source_spans=ast_spans, source_texts=source_texts)
        except DocumentBudgetExceeded:
            raise TraceRejected("The capture is retained, but its graph exceeds the display transport budget.") from None
        except (TypeError, ValueError, KeyError):
            raise TraceRejected("The semantic capture is invalid or does not match the projected saved source.") from None
        error = _trace_error(trace, {}, rel_file=rel_file, source_files=source_files)
        if error is not None:
            error["node_id"] = document.semantic.metadata["capture"]["failureNodeId"]
        return document, spans, error
    merged, spans, error = _merge(ast_graph, ast_spans, trace, cls=cls, rel_file=rel_file,
                                 source_digests=source_digests, source_files=source_files)
    return to_document(cls, merged, graph_id), spans, error


def project(
    ast_graph: Dict[str, Any],
    trace: Optional[Dict[str, Any]] = None,
    *,
    cls: str,
    rel_file: str,
    source: str,
) -> Projection:
    """Merge the static parse (and, from M4, a trace) into one projection.

    ``ast_graph`` is ``{nodes, edges}`` as ``CodeParserService`` produced it for
    ``cls``; it is never re-derived here. ``trace`` is the worker envelope of
    task.md §6.3 — ``{protocol_version, result, spans, error, worker}`` — exactly as
    it came off the worker's stdout.
    """
    attributes = init_attributes(source, cls)
    classes = workspace_classes(source, rel_file)
    nodes = list(ast_graph.get("nodes") or [])
    mapping = identity_map(nodes, cls=cls, attributes=attributes)
    renamed = _rename({"nodes": nodes, "edges": ast_graph.get("edges") or []}, mapping)
    for node in renamed["nodes"]:
        adopt_custom_class(node, classes=classes)
        _annotate(node, attributes=attributes)
    spans = utf16_spans(_spanmap(renamed["nodes"], attributes=attributes, rel_file=rel_file), {rel_file: source})

    graph_id = graph_id_for(rel_file, cls)
    if trace is None:
        return Projection(
            document=to_document(cls, renamed, graph_id),
            spanmap=spans,
            source_digest=digest_of(source),
            graph_id=graph_id,
            traced=False,
        )

    document, spans, error = project_trace(renamed, spans, trace, cls=cls, rel_file=rel_file,
                                         graph_id=graph_id, source_digests={rel_file: digest_of(source)},
                                         source_texts={rel_file: source})
    return Projection(
        document=document,
        spanmap=spans,
        source_digest=digest_of(source),
        graph_id=graph_id,
        traced=True,
        error=error,
    )
