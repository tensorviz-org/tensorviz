"""Graph edits, written back as the smallest possible source patch (task.md M5).

Until now the graph followed the file. This is the other direction: a canvas
edit becomes a patch to the one statement it is about, and everything else in
the file — comments, blank lines, the author's own spacing, the line the cursor
is on — is byte-identical afterwards.

That is why this is LibCST and not ``ast``. ``ast.unparse`` would regenerate the
whole module in its own style, so changing ``out_features`` would rewrite every
line of the file and the diff would be unreviewable. LibCST keeps every token it
did not touch, so rendering the modified module and diffing it against the
original line by line yields exactly the lines the edit changed. The diff *is*
the "restricted to the touched statements" step: no line outside a touched
statement can differ, because nothing else was rebuilt.

**This module never writes a file.** It returns ranges and replacement text; the
extension applies them as a ``WorkspaceEdit``, which is what puts the change in
the undo stack and in front of the user (task.md §8.5 — FastAPI's only workspace
write is ``.tensorviz/state.sqlite3``).

The refusals are the feature. A graph is a lossy view of code, and there are
edits the canvas can express that no single-statement patch can make honestly:

``control_flow_region``
    the statement lives inside an ``if``/``for``/``while``/``with``, so editing
    it changes one branch of something the graph drew as one node — or, for a
    removal, the attribute is still *read* inside one (a ModuleList driven by a
    ``for`` loop), so deleting its declaration would leave code reading it;
``dynamic_construction``
    the node has no statement to edit at all — it was built by a loop, a
    ``setattr`` or a comprehension, and there is no ``self.x = ...`` to patch;
``non_literal_value``
    the parameter is written as an expression (``cfg.dim``), so overwriting it
    with a number would silently delete the indirection the author chose. The
    IDE reveals the range instead;
``unsupported_operation``
    the operation has no single-statement form at all (rewiring two layers that
    both already exist, a ``set_param`` on a node that is a call rather than a
    declaration).

One limitation is deliberate and worth stating: two operations in one batch
whose statements *nest* — a change to a call and a change to a call inside it —
compose only through ``_Plan.current``, so the outer replacement is built from
the state the inner one already saw. Every pair the canvas can produce today is
either disjoint or reads through that path (a Sequential's children are
siblings; the Sequential itself is not a node), and ``remove_layer`` refuses any
node without a declaration of its own for the same reason.

``spanmap_stale`` is the fifth reason and the only one that is not per-op: it
means the spanmap was computed against a different buffer, so every span in the
request points at the wrong line. The router answers 409 for it before this
module is reached — a partial application against stale spans would corrupt the
file, and there is nothing to salvage from the request.
"""

from __future__ import annotations

import ast
import difflib
import keyword
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

#: Refusal reasons. Closed set: the extension shows a different message per
#: reason and the frontend decides whether to reveal a range from it.
CONTROL_FLOW_REGION = "control_flow_region"
DYNAMIC_CONSTRUCTION = "dynamic_construction"
NON_LITERAL_VALUE = "non_literal_value"
UNSUPPORTED_OPERATION = "unsupported_operation"
SPANMAP_STALE = "spanmap_stale"
MALFORMED_OPERATION = "malformed_operation"

#: Nodes the projection synthesises for the graph's boundary. They have no
#: declaration; ``forward``'s own parameter is what an edge from ``input``
#: actually reads.
BOUNDARY_IDS = ("input", "output")

#: Statements whose body is conditional or repeated. A node inside one of these
#: is drawn once and runs zero, one or many times, so a patch to it is not the
#: edit the user asked for.
_CONTROL_FLOW = tuple(
    node
    for node in (
        cst.If,
        cst.For,
        cst.While,
        cst.With,
        cst.Try,
        getattr(cst, "TryStar", None),
    )
    if node is not None
)


class ApplyError(RuntimeError):
    """The edit produced source this module will not hand back.

    Raised only for a bug in here — a transform that renders code Python cannot
    parse. It is loud on purpose: returning the edit anyway would put broken
    source in the user's buffer through a WorkspaceEdit they did not review.
    """


@dataclass(frozen=True)
class ApplyResult:
    """What ``/api/ide/apply`` returns, before the router adds the new spanmap."""

    #: ``[{range: {start: [line, col], end: [line, col]}, newText}]`` — 1-based
    #: lines and 0-based columns, exactly as the spanmap counts (task.md §6.2).
    edits: List[Dict[str, Any]]
    #: ``[{op_id, reason, message, span}]``. A refusal is not an error: the rest
    #: of the batch still applies, and the canvas must not mutate for this op.
    refused: List[Dict[str, Any]]
    #: The whole buffer as it would be after ``edits``. The router re-parses it
    #: to return a spanmap the client can anchor its next edit to.
    source: str


# ── reading the module ───────────────────────────────────────────────────────


@dataclass
class _Site:
    """One addressable place in the source, with the context that can refuse it."""

    #: The statement line the edit would land on.
    statement: cst.BaseStatement
    #: The call this site is about, when there is one (``nn.Linear(...)``).
    call: Optional[cst.Call]
    #: True when an ``if``/``for``/``while``/``with``/``try`` stands between the
    #: enclosing function and this statement.
    control_flow: bool


@dataclass
class _Index:
    """One class, read once, indexed the three ways the operations need it."""

    module: cst.Module
    positions: Any
    file: str
    #: The ``class <cls>`` node itself, so a rename stays inside it.
    class_node: Optional[cst.ClassDef] = None
    #: ``(line, column)`` of a statement -> that statement's site.
    by_position: Dict[Tuple[int, int], _Site] = field(default_factory=dict)
    #: line -> site, for a span whose column does not land on the statement.
    by_line: Dict[int, _Site] = field(default_factory=dict)
    #: ``(line, column)`` of a ``Call`` -> its site. A Sequential child's span is
    #: the child call, not a statement of its own.
    calls_by_position: Dict[Tuple[int, int], _Site] = field(default_factory=dict)
    #: attribute name -> the ``self.<attr> = <call>`` site in ``__init__``.
    declarations: Dict[str, _Site] = field(default_factory=dict)
    #: attribute name -> every ``self.<attr>(...)`` site in ``forward``.
    call_sites: Dict[str, List[_Site]] = field(default_factory=dict)
    #: attribute name -> every ``self.<attr>`` read in ``forward``, called or
    #: not, with whether it stands inside control flow. A ModuleList is iterated
    #: (``for block in self.blocks:``) rather than called, so it appears here and
    #: not in ``call_sites``; a removal that only looked at calls deleted the
    #: declaration and left the loop reading an attribute that no longer exists.
    reads: Dict[str, List[Tuple[cst.Attribute, bool]]] = field(default_factory=dict)
    #: The last statement of ``__init__``, for a layer with no anchor.
    init_tail: Optional[cst.BaseStatement] = None
    #: ``forward``'s first non-self parameter — what an edge from ``input`` reads.
    forward_input: Optional[str] = None
    #: ``forward``'s statements, in order, so a layer can be put at the front.
    forward_head: Optional[cst.BaseStatement] = None
    #: The last top-level ``import``/``from`` line, which a new import goes after.
    import_tail: Optional[cst.BaseStatement] = None
    #: The first top-level statement, for a file that imports nothing yet.
    module_head: Optional[cst.BaseStatement] = None
    #: Every name the module's top-level imports already bind.
    imported_names: Set[str] = field(default_factory=set)

    def range_of(self, node: cst.CSTNode) -> Dict[str, Any]:
        location = self.positions[node]
        return {
            "file": self.file,
            "start": [location.start.line, location.start.column],
            "end": [location.end.line, location.end.column],
        }

    def site_for(self, span: Mapping[str, Any]) -> Optional[_Site]:
        start = span.get("start") or []
        if len(start) < 2:
            return None
        key = (int(start[0]), int(start[1]))
        return (
            self.by_position.get(key)
            or self.calls_by_position.get(key)
            or self.by_line.get(key[0])
        )


class _Reader(cst.CSTVisitor):
    """Walk one class, recording every place an operation can address.

    The walk carries two pieces of context a flat visitor cannot have: whether a
    compound statement stands between here and the enclosing function (which is
    what ``control_flow_region`` means), and whether we are inside ``__init__``
    or ``forward`` (which decides whether a ``self.x`` is a declaration or a use).
    """

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, index: _Index, cls: str) -> None:
        super().__init__()
        self.index = index
        self.cls = cls
        self._in_class = 0
        self._function: Optional[str] = None
        self._depth = 0
        self._statement: Optional[cst.BaseStatement] = None

    # ── context ──────────────────────────────────────────────────────────────

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        if node.name.value == self.cls:
            self._in_class += 1
            self.index.class_node = node
        return True

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        if node.name.value == self.cls:
            self._in_class -= 1

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if not self._in_class:
            return True
        self._function = node.name.value
        self._depth = 0
        if node.name.value == "forward":
            params = [p.name.value for p in node.params.params if p.name.value != "self"]
            self.index.forward_input = params[0] if params else None
            body = getattr(node.body, "body", None)
            self.index.forward_head = body[0] if body else None
        return True

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        if self._in_class and self._function == node.name.value:
            self._function = None

    def on_visit(self, node: cst.CSTNode) -> bool:
        if isinstance(node, _CONTROL_FLOW) and self._function:
            self._depth += 1
        return super().on_visit(node)

    def on_leave(self, node: cst.CSTNode) -> None:
        if isinstance(node, _CONTROL_FLOW) and self._function:
            self._depth -= 1
        super().on_leave(node)

    # ── the index ────────────────────────────────────────────────────────────

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> bool:
        if not self._in_class or not self._function:
            return True
        site = _Site(statement=node, call=None, control_flow=self._depth > 0)
        location = self.get_metadata(PositionProvider, node)
        self.index.by_position[(location.start.line, location.start.column)] = site
        self.index.by_line.setdefault(location.start.line, site)
        self._statement = node
        if self._function == "__init__":
            self.index.init_tail = node
            self._record_declaration(node, site)
        return True

    def leave_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> None:
        if self._statement is node:
            self._statement = None

    def visit_Call(self, node: cst.Call) -> bool:
        if not self._in_class or not self._function or self._statement is None:
            return True
        location = self.get_metadata(PositionProvider, node)
        self.index.calls_by_position.setdefault(
            (location.start.line, location.start.column),
            _Site(statement=self._statement, call=node, control_flow=self._depth > 0),
        )
        if self._function == "forward":
            attribute = _self_attribute(node.func)
            if attribute:
                self.index.call_sites.setdefault(attribute, []).append(
                    _Site(statement=self._statement, call=node, control_flow=self._depth > 0)
                )
        return True

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        # Every read of ``self.<attr>`` in forward, not only the ones that are
        # calls: a removal has to know about the ones it cannot rewrite. There is
        # no `self._statement` guard here because a read can sit in the header of
        # a compound statement (`for block in self.blocks:`), which is exactly the
        # case that was being missed.
        if not self._in_class or self._function != "forward":
            return True
        attribute = _self_attribute(node)
        if attribute:
            self.index.reads.setdefault(attribute, []).append((node, self._depth > 0))
        return True

    def _record_declaration(self, node: cst.SimpleStatementLine, site: _Site) -> None:
        """``self.<attr> = <call>`` in ``__init__``.

        The call is stamped onto the site the position index already holds rather
        than onto a second one: a span resolves through ``by_position``, and a
        declaration whose constructor lived on a rival object answered "this is a
        call in forward, not a declaration" for every layer in the file.
        """
        for small in node.body:
            if not isinstance(small, cst.Assign) or len(small.targets) != 1:
                continue
            attribute = _self_attribute(small.targets[0].target)
            if not attribute:
                continue
            if isinstance(small.value, cst.Call):
                site.call = small.value
            self.index.declarations[attribute] = site


def _self_attribute(node: cst.CSTNode) -> Optional[str]:
    """``self.fc1`` -> ``"fc1"``; anything else -> None."""
    if (
        isinstance(node, cst.Attribute)
        and isinstance(node.value, cst.Name)
        and node.value.value == "self"
    ):
        return node.attr.value
    return None


def _read(source: str, cls: str, file: str) -> _Index:
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError as error:
        # Re-raised as Python's own SyntaxError so the router answers a half-typed
        # buffer the same way /api/ide/parse does, with the same shape the editor
        # already turns into a Diagnostic. LibCST's exception is not a SyntaxError.
        raise SyntaxError(
            error.message,
            (file, error.raw_line, error.raw_column + 1, ""),
        ) from error
    # unsafe_skip_copy keeps the metadata keyed on the very nodes `module` holds,
    # so a node found through PositionProvider is the same object the transformer
    # below sees as `original_node`. Without it the two trees are copies and every
    # identity check silently fails.
    wrapper = MetadataWrapper(module, unsafe_skip_copy=True)
    index = _Index(module=module, positions=wrapper.resolve(PositionProvider), file=file)
    wrapper.visit(_Reader(index, cls))
    _index_imports(index)
    return index


def _index_imports(index: _Index) -> None:
    """Record where a new import goes and which names are already bound.

    Read straight off ``module.body`` rather than in ``_Reader``: an import is a
    property of the FILE, and the reader is deliberately scoped to one class.
    """
    for statement in index.module.body:
        if index.module_head is None:
            index.module_head = statement
        if not isinstance(statement, cst.SimpleStatementLine):
            continue
        for small in statement.body:
            if not isinstance(small, (cst.Import, cst.ImportFrom)):
                continue
            index.import_tail = statement
            names = getattr(small, "names", None)
            if isinstance(names, cst.ImportStar) or names is None:
                continue
            for alias in names:
                if alias.asname is not None and isinstance(alias.asname.name, cst.Name):
                    index.imported_names.add(alias.asname.name.value)
                elif isinstance(alias.name, cst.Name):
                    index.imported_names.add(alias.name.value)
                elif isinstance(alias.name, cst.Attribute):
                    # `import torch.nn` binds `torch`, not `torch.nn`.
                    root = alias.name
                    while isinstance(root, cst.Attribute):
                        root = root.value
                    if isinstance(root, cst.Name):
                        index.imported_names.add(root.value)


def _ensure_import(index: _Index, plan: _Plan, module_path: str, class_name: str) -> None:
    """Put ``from <module_path> import <class_name>`` at the top, if it is missing.

    A user class added to the canvas is declared as ``self.x = MyBlock(...)``, and
    a declaration whose name is not bound is a NameError the moment the file is
    run — so the import is part of the same edit, not a second thing to remember
    (task.md M6). Placed after the last existing import so the file keeps the
    shape its author gave it.
    """
    if class_name in index.imported_names:
        return
    statement = cst.parse_statement(f"from {module_path} import {class_name}")
    if index.import_tail is not None:
        plan.insert_after(index.import_tail, statement)
        return
    if index.module_head is not None:
        plan.insert_before(index.module_head, statement)
        return
    raise _Refused(
        UNSUPPORTED_OPERATION,
        f"this file has no statements to put `from {module_path} import {class_name}` beside.",
    )


# ── writing values ───────────────────────────────────────────────────────────

_PARAM_NAMES_CACHE: Dict[str, List[str]] = {}


def positional_param_names(layer_type: str) -> List[str]:
    """Constructor parameter names in positional order, for ``nn.<layer_type>``.

    Delegates to ``CodeParserService`` so a positional argument is named here
    exactly as the parser named it when it read the same call into a node — if
    the two disagreed, ``set_param('out_features')`` would rewrite the wrong
    argument of ``nn.Linear(784, 128)``. Imported lazily: that module pulls in
    torch, and everything else here is pure syntax.
    """
    if layer_type not in _PARAM_NAMES_CACHE:
        from utils.code_parser import CodeParserService

        _PARAM_NAMES_CACHE[layer_type] = list(
            CodeParserService()._positional_param_names(layer_type)
        )
    return _PARAM_NAMES_CACHE[layer_type]


def _call_class(call: cst.Call) -> str:
    """Keep custom dotted names distinct from torch constructor signatures."""
    func = call.func
    if isinstance(func, cst.Attribute):
        if isinstance(func.value, cst.Name) and func.value.value == "nn":
            return func.attr.value
        return cst.Module([]).code_for_node(func)
    if isinstance(func, cst.Name):
        return func.value
    return ""


def _is_literal(module: cst.Module, node: cst.CSTNode) -> bool:
    try:
        ast.literal_eval(module.code_for_node(node).strip())
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False
    return True


def _literal_expression(value: Any) -> cst.BaseExpression:
    """A JSON value as the source text a human would have typed.

    Strings get one extra chance to be what they look like: the canvas keeps
    every parameter field's value as text, so ``out_features`` arrives as
    ``"256"`` and writing ``nn.Linear(784, '256')`` would be a bug the user only
    finds when they run it. A string that does not read as a literal (``"same"``)
    stays a string.
    """
    if isinstance(value, str):
        text = value.strip()
        try:
            ast.literal_eval(text)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return cst.parse_expression(repr(value))
        return cst.parse_expression(text)
    return cst.parse_expression(repr(value))



def _missing_constructor_arguments(layer_op: str, params: Mapping[str, Any]) -> List[str]:
    """Required constructor arguments the caller did not supply.

    Read off the torch class's own signature rather than guessed at, so
    ``nn.GELU()`` — valid with nothing — is never confused with ``nn.Linear()``,
    which is not. Fifty-nine of the registry's ops cannot be built with no
    arguments and the rest can, and only the class knows which is which.

    Nothing is constructed to find out. ``inspect.signature`` reads the parameters
    without allocating a weight, which matters because a caller can ask for a
    layer far too large to instantiate just to check whether it is well formed.

    An op the registry does not know, or a class this torch does not have, yields
    an empty list: this is a check that refuses what it is sure about, never one
    that refuses what it cannot see.
    """
    try:
        import inspect

        import torch.nn as nn

        from core.op_registry import spec_for

        spec = spec_for(layer_op)
        if spec is None or not spec.canonical.startswith("nn."):
            return []
        cls = getattr(nn, spec.canonical.split(".", 1)[1], None)
        if cls is None:
            return []
        signature = inspect.signature(cls)
    except Exception:  # pragma: no cover - a registry or torch that cannot answer
        return []
    supplied = {str(name) for name in params}
    return [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in supplied
    ]


def _is_torch_nn_class(class_name: str) -> bool:
    """Is ``nn.<class_name>`` a real nn.Module class in this torch?

    Looking like a class is not being one. A capitalised identifier is all the
    caller's op name has to be for `nn.NotAThing()` to be written into __init__,
    and the user then has a file that cannot even be imported — the same failure
    the missing-argument check below exists to prevent, one line earlier.

    A torch that cannot be imported answers True: this refuses what it is sure
    about, and an unreadable torch is not something it is sure about.
    """
    try:
        import torch.nn as nn
    except Exception:  # pragma: no cover - a torch that cannot be imported
        return True
    cls = getattr(nn, class_name, None)
    return isinstance(cls, type) and issubclass(cls, nn.Module)


def _constructor_call(layer_op: str, params: Mapping[str, Any]) -> Optional[cst.Call]:
    """``nn.<Class>(<params>)`` for a palette op, or None when it is not one.

    ``layer_op`` arrives as whatever the caller had — ``dropout`` from a canvas
    layerType id, ``nn.Dropout`` from a plan, ``torch.nn.Dropout`` from the
    registry — so it is resolved through the op registry's canonical name, the
    same lookup ``code_parser`` uses when reading such a call back.
    """
    canonical = layer_op
    try:
        from core.op_registry import spec_for

        spec = spec_for(layer_op)
        if spec is not None:
            canonical = spec.canonical
    except Exception:  # pragma: no cover - registry import is not this op's story
        pass
    class_name = canonical.split(".")[-1]
    if not class_name.isidentifier() or not class_name[0].isupper():
        return None
    if not _is_torch_nn_class(class_name):
        return None
    missing = _missing_constructor_arguments(layer_op, params)
    if missing:
        # Refuse rather than write it. `nn.Linear()` is valid Python and cannot be
        # constructed, so the only check downstream — `ast.parse` on the rendered
        # source — passes it, `refused` comes back empty and the editor reports a
        # successful edit over a file that no longer runs. A refusal naming the
        # arguments is the answer; writing something that parses is not.
        needed = "argument" if len(missing) == 1 else "arguments"
        raise _Refused(
            UNSUPPORTED_OPERATION,
            f"{class_name} needs the {needed} {', '.join(missing)}. Writing it without "
            "them would leave a file that parses and does not run, so it is refused "
            "instead. Supply them and try again.",
        )
    arguments = [
        _keyword_argument(str(name), value)
        for name, value in params.items()
        if not str(name).startswith("_") and str(name) != "name" and str(name).isidentifier()
    ]
    return cst.Call(
        func=cst.Attribute(value=cst.Name("nn"), attr=cst.Name(class_name)),
        args=_comma_normalised(arguments),
    )


#: ``p=0.1``, not ``p = 0.1``. LibCST's default for a synthesised keyword
#: argument is spaced, which is PEP 8's rule for assignment statements and its
#: explicit counter-example for keyword arguments — and it would show up as a
#: formatting change in the user's diff.
_TIGHT_EQUAL = cst.AssignEqual(
    whitespace_before=cst.SimpleWhitespace(""),
    whitespace_after=cst.SimpleWhitespace(""),
)


def _keyword_argument(name: str, value: Any) -> cst.Arg:
    return cst.Arg(value=_literal_expression(value), keyword=cst.Name(name), equal=_TIGHT_EQUAL)


def _comma_normalised(args: Sequence[cst.Arg]) -> List[cst.Arg]:
    """Give every argument but the last a trailing comma, as LibCST requires."""
    out: List[cst.Arg] = []
    for index, argument in enumerate(args):
        last = index == len(args) - 1
        out.append(argument.with_changes(comma=cst.MaybeSentinel.DEFAULT if last else cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))))
    return out


# ── the plan ─────────────────────────────────────────────────────────────────


@dataclass
class _Pending:
    """A layer named by ``add_layer`` and not yet given a place in the file."""

    node_id: str
    call: cst.Call
    placed: bool = False
    # Original source anchor shared by an ordered chain of newly inserted layers.
    anchor_id: Optional[str] = None
    #: ``(module, Class)`` when this layer is the user's own class, so the same
    #: edit that declares it also imports it.
    import_: Optional[Tuple[str, str]] = None


class _Plan:
    """Every change to make, keyed by the identity of the node it replaces.

    Collected before anything is rendered so that two operations on the same
    call compose (the second reads the first's result) and so the whole batch is
    applied in one pass — which is what makes the diff minimal.
    """

    def __init__(self) -> None:
        self.replace: Dict[int, cst.CSTNode] = {}
        self.remove: Set[int] = set()
        self.after: Dict[int, List[cst.BaseStatement]] = {}
        self.before: Dict[int, List[cst.BaseStatement]] = {}

    def current(self, node: cst.CSTNode) -> cst.CSTNode:
        return self.replace.get(id(node), node)

    def set(self, node: cst.CSTNode, replacement: cst.CSTNode) -> None:
        self.replace[id(node)] = replacement

    def drop(self, node: cst.CSTNode) -> None:
        self.remove.add(id(node))

    def insert_after(self, node: cst.CSTNode, statement: cst.BaseStatement) -> None:
        self.after.setdefault(id(node), []).append(statement)

    def insert_before(self, node: cst.CSTNode, statement: cst.BaseStatement) -> None:
        self.before.setdefault(id(node), []).append(statement)

    @property
    def empty(self) -> bool:
        return not (self.replace or self.remove or self.after or self.before)


class _Writer(cst.CSTTransformer):
    def __init__(self, plan: _Plan) -> None:
        super().__init__()
        self.plan = plan

    def on_leave(self, original_node, updated_node):  # type: ignore[override]
        key = id(original_node)
        if key in self.plan.remove:
            return cst.RemoveFromParent()
        if key in self.plan.replace:
            updated_node = self.plan.replace[key]
        before = self.plan.before.get(key)
        after = self.plan.after.get(key)
        if before or after:
            return cst.FlattenSentinel([*(before or []), updated_node, *(after or [])])
        return updated_node


# ── operations ───────────────────────────────────────────────────────────────


class _Refused(Exception):
    """One operation cannot be made as a source edit. Carries what to tell the user."""

    def __init__(self, reason: str, message: str, span: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.span = span


def _required(op: Mapping[str, Any], field: str) -> str:
    """Read a field the operation cannot be carried out without.

    A field the client left out is the client's mistake, and ``refused`` is where
    the canvas already looks for it. Reading it straight out of the dictionary
    raised a ``KeyError`` that escaped the per-operation handler, so one malformed
    operation turned the whole batch into a 500 and the user saw "tensorViz
    backend error" with no node named and the rest of their edits lost.
    """
    value = op.get(field)
    if value is None or str(value).strip() == "":
        kind = str(op.get("op") or op.get("action") or "") or "this operation"
        raise _Refused(
            MALFORMED_OPERATION,
            f"{kind} needs `{field}`, and this one does not carry it.",
        )
    return str(value)


def _site_for_node(
    index: _Index, spanmap: Mapping[str, Mapping[str, Any]], node_id: str
) -> Tuple[_Site, Mapping[str, Any]]:
    span = spanmap.get(node_id)
    if not span:
        raise _Refused(
            DYNAMIC_CONSTRUCTION,
            f"{node_id} has no declaration in this file — it was built at runtime.",
        )
    site = index.site_for(span)
    if site is None:
        raise _Refused(
            DYNAMIC_CONSTRUCTION,
            f"{node_id}'s span does not point at a statement in this buffer.",
            dict(span),
        )
    if site.control_flow:
        raise _Refused(
            CONTROL_FLOW_REGION,
            f"{node_id} is inside an if/for/while/with — editing it would change one branch of a node the canvas draws once.",
            dict(span),
        )
    return site, span


def _declaration(index: _Index, spanmap, node_id: str) -> Tuple[_Site, Mapping[str, Any]]:
    site, span = _site_for_node(index, spanmap, node_id)
    if site.call is None:
        raise _Refused(
            UNSUPPORTED_OPERATION,
            f"{node_id} is a call in forward, not a layer declaration; there is no constructor to edit.",
            dict(span),
        )
    return site, span


def _apply_set_param(index: _Index, plan: _Plan, spanmap, op: Mapping[str, Any]) -> None:
    node_id = _required(op, "node_id")
    param = _required(op, "param")
    if op.get("value") is None:
        # An absent value is not a value. Writing `None` into the constructor
        # produces source that still parses, so nothing catches it here or in the
        # editor — the user finds out when they run their own file and read a
        # traceback about a layer they never touched. Refusing is the only
        # reading of an incomplete request that cannot damage their model.
        raise _Refused(
            MALFORMED_OPERATION,
            f"set_param needs a `value` to write into {node_id}.{param}, and this one does not carry it.",
        )
    site, span = _declaration(index, spanmap, node_id)
    call = plan.current(site.call)
    assert isinstance(call, cst.Call)

    arguments = list(call.args)
    index_of: Optional[int] = None
    for position, argument in enumerate(arguments):
        if argument.keyword is not None and argument.keyword.value == param:
            index_of = position
            break
    if index_of is None:
        class_name = _call_class(call)
        names = positional_param_names(class_name) if "." not in class_name else []
        if "." in class_name and not names and any(a.keyword is None for a in arguments):
            # The parser's arg_i is a source position, not a guessed torch
            # signature for a workspace class with the same final name.
            position = int(param[4:]) if param.startswith("arg_") and param[4:].isdigit() else -1
            positional = [p for p, a in enumerate(arguments) if a.keyword is None]
            if any(a.star for a in arguments) or not 0 <= position < len(positional):
                raise _Refused(
                    UNSUPPORTED_OPERATION,
                    f"{node_id}.{param} cannot be mapped to a custom constructor's positional arguments; "
                    "edit an existing keyword or a literal arg_i instead.",
                    dict(span),
                )
            index_of = positional[position]
        if param in names:
            wanted = names.index(param)
            positional = [p for p, a in enumerate(arguments) if a.keyword is None and not a.star]
            if wanted < len(positional):
                index_of = positional[wanted]

    value = _literal_expression(op.get("value"))
    if index_of is None:
        # The author never wrote this argument, so there is nothing to preserve
        # and nothing to refuse: name it explicitly rather than guessing a slot.
        arguments = _comma_normalised(
            [*arguments, cst.Arg(value=value, keyword=cst.Name(param), equal=_TIGHT_EQUAL)]
        )
        plan.set(site.call, call.with_changes(args=arguments))
        return

    existing = arguments[index_of].value
    if not _is_literal(index.module, existing):
        raise _Refused(
            NON_LITERAL_VALUE,
            f"{node_id}.{param} is written as `{index.module.code_for_node(existing).strip()}`; "
            "overwriting it would delete the expression the author wrote.",
            index.range_of(existing),
        )
    arguments[index_of] = arguments[index_of].with_changes(value=value)
    plan.set(site.call, call.with_changes(args=arguments))


def _apply_replace_layer(index: _Index, plan: _Plan, spanmap, op: Mapping[str, Any]) -> None:
    node_id = _required(op, "node_id")
    new_layer_op = _required(op, "new_layer_op")
    site, span = _declaration(index, spanmap, node_id)
    replacement = _constructor_call(new_layer_op, op.get("params") or {})
    if replacement is None:
        raise _Refused(
            UNSUPPORTED_OPERATION,
            f"{op['new_layer_op']} does not resolve to an nn.Module class.",
            dict(span),
        )
    plan.set(site.call, replacement)
    new_id = op.get("new_node_id")
    if new_id and str(new_id) != node_id:
        _rename_attribute(index, plan, node_id, str(new_id))


def _apply_group(index: _Index, plan: _Plan, spanmap, op: Mapping[str, Any]) -> None:
    """Extract a proven, consecutive unary chain into a named module class.

    Declaration and call order must agree. Every selected attribute has exactly
    its declaration and one forward call; aliases, extra reads, branches and
    other methods are refused. All validation precedes changes to the plan.
    """
    names = op.get("node_ids")
    group = _required(op, "node_id")
    if (not isinstance(names, list) or not 2 <= len(names) <= 128
            or any(not isinstance(name, str) or not name.isidentifier() for name in names)
            or len(set(names)) != len(names)):
        raise _Refused(MALFORMED_OPERATION, "Select at least two distinct declared layers to group.")
    if not group.isidentifier() or keyword.iskeyword(group):
        raise _Refused(MALFORMED_OPERATION, "The block name must be a Python attribute name.")
    scope = index.class_node
    if scope is None or scope not in index.module.body or _walk_self_attributes(scope, group):
        raise _Refused(UNSUPPORTED_OPERATION, "That block name already exists or the class is unavailable.")
    # Use the file's established torch.nn binding; never capture an unrelated nn.
    tree = ast.parse(index.module.code)
    nn_name = None
    for item in tree.body:
        if isinstance(item, ast.Import):
            for alias in item.names:
                if alias.name == "torch.nn" and alias.asname:
                    nn_name = alias.asname
        if isinstance(item, ast.ImportFrom) and item.module == "torch":
            for alias in item.names:
                if alias.name == "nn":
                    nn_name = alias.asname or "nn"
    if nn_name is None:
        raise _Refused(UNSUPPORTED_OPERATION, "Grouping needs an explicit torch.nn import in this file.")
    if any(isinstance(item, ast.Name) and item.id == nn_name and isinstance(item.ctx, ast.Store)
           for item in ast.walk(tree)):
        raise _Refused(UNSUPPORTED_OPERATION, "The torch.nn import is rebound in this file.")
    if any(isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
           and item.func.id in {"getattr", "setattr", "delattr", "vars"} for item in ast.walk(tree)):
        raise _Refused(UNSUPPORTED_OPERATION, "Dynamic attribute access prevents a safe block refactor.")
    methods = {item.name.value: item for item in scope.body.body if isinstance(item, cst.FunctionDef)}
    init = methods.get("__init__")
    forward = methods.get("forward")
    if not init or not forward or not isinstance(init.body, cst.IndentedBlock) or not isinstance(forward.body, cst.IndentedBlock):
        raise _Refused(UNSUPPORTED_OPERATION, "Grouping needs explicit __init__ and forward method bodies.")
    declarations = [_declaration(index, spanmap, name)[0] for name in names]
    if any(index.declarations.get(name) is not site for name, site in zip(names, declarations)):
        raise _Refused(UNSUPPORTED_OPERATION, "Select layers with their own declarations in this class.")
    declarations.sort(key=lambda site: index.positions[site.statement].start.line)
    ordered = [next(name for name in names if index.declarations.get(name) is site) for site in declarations]
    init_body = list(init.body.body)
    statements = [site.statement for site in declarations]
    if any(statement not in init_body for statement in statements):
        raise _Refused(CONTROL_FLOW_REGION, "Select layers declared directly in __init__.")
    start = init_body.index(statements[0])
    if init_body[start:start + len(statements)] != statements:
        raise _Refused(UNSUPPORTED_OPERATION, "Select consecutive declarations so initialization order stays unchanged.")
    for site in declarations:
        # Moving a constructor into a new class must not lose a local binding
        # from the old __init__, or change which self an expression refers to.
        constructor = ast.parse(index.module.code_for_node(site.call), mode="eval").body
        try:
            for argument in [*constructor.args, *(item.value for item in constructor.keywords)]:
                ast.literal_eval(argument)
        except (ValueError, TypeError):
            raise _Refused(UNSUPPORTED_OPERATION, "Grouping currently requires literal constructor arguments; a local expression needs a manual refactor.")
    calls = []
    for name in ordered:
        sites = index.call_sites.get(name, [])
        if len(sites) != 1 or sites[0].control_flow or len(_walk_self_attributes(scope, name)) != 2:
            raise _Refused(UNSUPPORTED_OPERATION, f"{name} must have one forward call and no other references to become a block.")
        call = sites[0].call
        if call is None or len(call.args) != 1 or call.args[0].keyword or call.args[0].star:
            raise _Refused(UNSUPPORTED_OPERATION, "A block requires one positional input per layer.")
        calls.append(sites[0])
    first_call = calls[0].call
    assert first_call is not None
    # Two supported spellings: self.b(self.a(x)), or consecutive x = self.a(x).
    nested = all(calls[i].call.args[0].value is calls[i - 1].call for i in range(1, len(calls)))
    forward_body = list(forward.body.body)
    if nested:
        outer = calls[-1].call
        replacement = cst.Call(cst.Attribute(cst.Name("self"), cst.Name(group)), args=first_call.args)
    else:
        chain = [site.statement for site in calls]
        if any(statement not in forward_body for statement in chain):
            raise _Refused(CONTROL_FLOW_REGION, "Select a chain directly in forward, outside control flow.")
        offset = forward_body.index(chain[0])
        if forward_body[offset:offset + len(chain)] != chain:
            raise _Refused(UNSUPPORTED_OPERATION, "Select consecutive calls in the same order as their declarations.")
        variable = first_call.args[0].value
        if not isinstance(variable, cst.Name):
            raise _Refused(UNSUPPORTED_OPERATION, "The chain must pass one named tensor between its layers.")
        for site in calls:
            statement = site.statement
            call = site.call
            if (not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1
                    or not isinstance(statement.body[0], cst.Assign)
                    or len(statement.body[0].targets) != 1
                    or not isinstance(statement.body[0].targets[0].target, cst.Name)
                    or statement.body[0].targets[0].target.value != variable.value
                    or statement.body[0].value is not call
                    or not isinstance(call.args[0].value, cst.Name)
                    or call.args[0].value.value != variable.value):
                raise _Refused(UNSUPPORTED_OPERATION, "The chain must pass the same named tensor through each selected layer.")
        outer = first_call
        replacement = cst.Call(cst.Attribute(cst.Name("self"), cst.Name(group)), args=first_call.args)

    # No plan mutations above this line: a refusal is all-or-nothing.
    first = statements[0]
    # Refuse compound statements rather than discarding their other effects.
    if any(not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1
           or not isinstance(statement.body[0], cst.Assign) or len(statement.body[0].targets) != 1
           for statement in statements):
        raise _Refused(UNSUPPORTED_OPERATION, "Each declaration must occupy its own assignment statement.")
    assignment = first.body[0]
    class_base = "".join(part.capitalize() for part in group.split("_") if part) + "Block"
    class_name = class_base
    bound = {item.id for item in ast.walk(tree) if isinstance(item, ast.Name)}
    bound.update(item.name for item in ast.walk(tree) if isinstance(item, (ast.ClassDef, ast.FunctionDef)))
    bound.update(index.imported_names)
    suffix = 2
    while class_name in bound:
        class_name = f"{class_base}{suffix}"
        suffix += 1
    forward_expression = "x"
    for name in ordered:
        forward_expression = f"self.{name}({forward_expression})"
    extracted = cst.parse_statement(
        f"class {class_name}({nn_name}.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "    def forward(self, x):\n"
        f"        return {forward_expression}\n"
    )
    extracted_init = extracted.body.body[0]
    extracted_init = extracted_init.with_changes(body=extracted_init.body.with_changes(
        body=[*extracted_init.body.body, *(statement.deep_clone() for statement in statements)]))
    extracted = extracted.with_changes(
        leading_lines=[cst.EmptyLine(), cst.EmptyLine()],
        body=extracted.body.with_changes(body=[extracted_init, extracted.body.body[1]]))
    plan.insert_after(scope, extracted)
    plan.set(first, first.with_changes(leading_lines=[], trailing_whitespace=cst.TrailingWhitespace(), body=[assignment.with_changes(
        targets=[cst.AssignTarget(cst.Attribute(cst.Name("self"), cst.Name(group)))], value=cst.Call(cst.Name(class_name)))]))
    for statement in statements[1:]:
        plan.drop(statement)
    plan.set(outer, replacement)
    if not nested:
        # Preserve comments on removed forward assignments above the surviving call.
        first_statement = calls[0].statement
        lines = list(first_statement.leading_lines)
        for site in calls[1:]:
            lines.extend(site.statement.leading_lines)
            if site.statement.trailing_whitespace.comment:
                lines.append(cst.EmptyLine(comment=site.statement.trailing_whitespace.comment))
            plan.drop(site.statement)
        rewritten = first_statement.deep_replace(outer, replacement).with_changes(leading_lines=lines)
        plan.set(first_statement, rewritten)


def _apply_remove_layer(index: _Index, plan: _Plan, spanmap, op: Mapping[str, Any]) -> None:
    node_id = _required(op, "node_id")
    site, span = _site_for_node(index, spanmap, node_id)
    # Only a layer with a statement of its own can be deleted by deleting a
    # statement. A functional node's statement is `return x + self.l2(h)` and a
    # Sequential child's is the whole `self.net = nn.Sequential(...)`; dropping
    # either would delete far more than the card the user selected.
    if index.declarations.get(node_id) is not site:
        raise _Refused(
            UNSUPPORTED_OPERATION,
            f"{node_id} has no declaration of its own — it is written inside another statement, "
            "and deleting that statement would remove more than this layer.",
            dict(span),
        )
    for call_site in index.call_sites.get(node_id, []):
        if call_site.control_flow:
            raise _Refused(
                CONTROL_FLOW_REGION,
                f"{node_id} is called inside an if/for/while/with; removing it would change one branch only.",
                dict(span),
            )
    # A read this operation is not going to rewrite outlives the declaration it
    # reads, and the file stops running: `for block in self.blocks:` after
    # `self.blocks = ...` is gone raises AttributeError on the first forward
    # pass. The canvas cannot show a loop as anything but one card, so there is
    # no honest single-statement edit here — refusing keeps the promise that a
    # deletion never hands back source that cannot run.
    rewritten = {id(call_site.call.func) for call_site in index.call_sites.get(node_id, [])}
    for read, in_control_flow in index.reads.get(node_id, []):
        if id(read) in rewritten:
            continue
        if in_control_flow:
            raise _Refused(
                CONTROL_FLOW_REGION,
                f"self.{node_id} is read inside an if/for/while/with in forward, not called there; "
                "deleting the declaration would leave that code reading an attribute that no longer exists.",
                index.range_of(read),
            )
        raise _Refused(
            UNSUPPORTED_OPERATION,
            f"self.{node_id} is read in forward somewhere this edit cannot rewrite; "
            "deleting the declaration would leave a file that no longer runs.",
            index.range_of(read),
        )
    plan.drop(site.statement)
    for call_site in index.call_sites.get(node_id, []):
        call = call_site.call
        assert call is not None
        if not call.args:
            raise _Refused(
                UNSUPPORTED_OPERATION,
                f"self.{node_id}() takes no input, so there is nothing to pass on in its place.",
                dict(span),
            )
        passthrough = call.args[0].value
        statement = call_site.statement
        # `x = self.fc1(x)` becomes `x = x` when fc1 goes; that is not an edit a
        # reader would accept, so the whole line goes with it.
        if _is_identity_assignment(statement, call, passthrough):
            plan.drop(statement)
            continue
        plan.set(call, passthrough)


def _is_identity_assignment(
    statement: cst.BaseStatement, call: cst.Call, passthrough: cst.BaseExpression
) -> bool:
    if not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1:
        return False
    small = statement.body[0]
    if not isinstance(small, cst.Assign) or len(small.targets) != 1:
        return False
    target = small.targets[0].target
    return (
        small.value is call
        and isinstance(target, cst.Name)
        and isinstance(passthrough, cst.Name)
        and target.value == passthrough.value
    )


def _renameable(index: _Index, spanmap, node_id: str) -> None:
    """Refuse a rename of a layer no statement in this file declares.

    A rename only rewrites the `self.<name>` it can see, which is the whole edit
    when a `self.<name> = ...` line is one of them. A layer registered by
    `setattr` or built in a comprehension is named by a string instead, so the
    walk finds the call in `forward`, renames that, and leaves the construction
    behind — the user is handed a file that raises AttributeError on its first
    forward pass, and the canvas was told the edit succeeded. `set_param` and
    `replace_layer` already refuse that node through `_declaration`; a rename is
    no more expressible than they are. Control flow is deliberately not checked
    here: a rename touches every use, so it stays correct wherever the
    declaration happens to stand.
    """
    span = spanmap.get(node_id)
    if not span:
        raise _Refused(
            DYNAMIC_CONSTRUCTION,
            f"{node_id} has no declaration in this file — it was built at runtime, "
            "so renaming its uses would leave the construction naming the old attribute.",
        )
    if index.site_for(span) is None:
        raise _Refused(
            DYNAMIC_CONSTRUCTION,
            f"{node_id}'s span does not point at a statement in this buffer.",
            dict(span),
        )


def _rename_attribute(index: _Index, plan: _Plan, old: str, new: str) -> None:
    """``self.<old>`` -> ``self.<new>`` at every reference inside the class.

    Not guarded by ``control_flow_region``: a rename touches every use, so it is
    correct wherever the declaration happens to stand. The guard exists for edits
    that change one branch of something drawn as one node; this changes none.
    """
    scope = index.class_node
    if scope is None:
        raise _Refused(DYNAMIC_CONSTRUCTION, "this file does not define the class being edited.")
    # A card can be renamed to anything the user can type, and `self.class` is
    # not source. Refused rather than written, because the render below would
    # produce a file Python cannot parse and the user would get a 500 with no
    # way to tell whether their buffer had been touched.
    if not new.isidentifier() or keyword.iskeyword(new):
        raise _Refused(
            UNSUPPORTED_OPERATION,
            f"`{new}` cannot be an attribute name in Python, so self.{new} is not something "
            "this edit can write.",
        )
    if new in index.declarations:
        raise _Refused(
            UNSUPPORTED_OPERATION,
            f"self.{new} already exists in this class; renaming onto it would merge two layers.",
        )
    renamed = 0
    for node in _walk_self_attributes(scope, old):
        plan.set(node, node.with_changes(attr=cst.Name(new)))
        renamed += 1
    if renamed == 0:
        raise _Refused(DYNAMIC_CONSTRUCTION, f"self.{old} is not written anywhere in this file.")


def _walk_self_attributes(scope: cst.CSTNode, name: str) -> List[cst.Attribute]:
    found: List[cst.Attribute] = []

    class _Find(cst.CSTVisitor):
        def visit_Attribute(self, node: cst.Attribute) -> bool:
            if _self_attribute(node) == name:
                found.append(node)
            return True

    scope.visit(_Find())
    return found


def _custom_class(params: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    """``(module, Class)`` for a node that names one of the user's own classes.

    ``_import_path``/``_class`` are written by the projection and by the trace
    worker (task.md M6). Both are checked as identifiers here because they end up
    interpolated into source: graph data is data, never code.
    """
    module_path = params.get("_import_path")
    class_name = params.get("_class")
    if not isinstance(module_path, str) or not isinstance(class_name, str):
        return None
    parts = module_path.split(".")
    if not parts or not all(part.isidentifier() for part in parts):
        return None
    if not class_name.isidentifier():
        return None
    return module_path, class_name


def _apply_add_layer(pending: Dict[str, _Pending], op: Mapping[str, Any]) -> None:
    node_id = _required(op, "node_id")
    params = op.get("params") or {}
    custom = _custom_class(params)
    if custom is not None:
        # The user's own class: `MyBlock(...)`, not `nn.MyBlock(...)`. The registry
        # has no spec for it and must not grow one (task.md D5) — what makes it
        # writable is that the node says where the class lives.
        pending[node_id] = _Pending(
            node_id=node_id,
            call=cst.Call(
                func=cst.Name(custom[1]),
                args=_comma_normalised(
                    [
                        _keyword_argument(str(name), value)
                        for name, value in params.items()
                        if not str(name).startswith("_")
                        and str(name) != "name"
                        and str(name).isidentifier()
                    ]
                ),
            ),
            import_=custom,
        )
        return
    call = _constructor_call(_required(op, "layer_op"), params)
    if call is None:
        raise _Refused(
            UNSUPPORTED_OPERATION,
            f"{op['layer_op']} does not resolve to an nn.Module class, so it cannot be declared in __init__.",
        )
    pending[node_id] = _Pending(node_id=node_id, call=call)


def _place(
    index: _Index, plan: _Plan, spanmap, layer: _Pending, anchor_id: str
) -> None:
    """Declare ``layer`` after ``anchor_id`` and call it on what the anchor produced.

    Two inserted lines and nothing else touched, which is the shape a reviewer can
    read: one ``self.<id> = nn.<Op>(...)`` under the anchor's own declaration, one
    ``<v> = self.<id>(<v>)`` under the statement that produced ``<v>``. When the
    anchor's result is not bound to a plain name — ``return self.fc2(x)``, or a
    call nested inside a bigger expression — there is no line to insert after, so
    the anchor's call is wrapped in place instead: one changed line rather than
    two new ones, and still exactly one statement.
    """
    if not layer.node_id.isidentifier() or keyword.iskeyword(layer.node_id):
        raise _Refused(MALFORMED_OPERATION, "The new layer name must be a Python attribute name.")
    if index.class_node is None or _walk_self_attributes(index.class_node, layer.node_id):
        raise _Refused(UNSUPPORTED_OPERATION, f"self.{layer.node_id} already exists; choose a new layer name.")
    declaration = cst.parse_statement(
        f"self.{layer.node_id} = {_render_call(index.module, layer.call)}"
    )
    if anchor_id in BOUNDARY_IDS:
        _place_at_head(index, plan, layer, declaration)
        return

    anchor_decl, _ = _declaration(index, spanmap, anchor_id)
    plan.insert_after(anchor_decl.statement, declaration)

    sites = index.call_sites.get(anchor_id) or []
    if not sites:
        raise _Refused(
            UNSUPPORTED_OPERATION,
            f"self.{anchor_id} is never called in forward, so there is no place to run {layer.node_id}.",
        )
    site = sites[-1]
    if site.control_flow:
        raise _Refused(
            CONTROL_FLOW_REGION,
            f"self.{anchor_id} is called inside an if/for/while/with; inserting after it would only run on one branch.",
        )
    bound = _assigned_name(site)
    if bound is not None:
        plan.insert_after(
            site.statement,
            cst.parse_statement(f"{bound} = self.{layer.node_id}({bound})"),
        )
        return
    call = site.call
    assert call is not None
    plan.set(
        call,
        cst.Call(
            func=cst.Attribute(value=cst.Name("self"), attr=cst.Name(layer.node_id)),
            args=[cst.Arg(value=plan.current(call))],
        ),
    )


def _place_at_head(
    index: _Index, plan: _Plan, layer: _Pending, declaration: cst.BaseStatement
) -> None:
    """The anchor is the graph's ``input`` boundary: run first, on forward's own
    parameter. ``input`` has no declaration, so the constructor goes at the end of
    ``__init__`` — after every layer it might read a config value from."""
    if index.init_tail is None or index.forward_head is None or index.forward_input is None:
        raise _Refused(
            UNSUPPORTED_OPERATION,
            "this class has no __init__/forward pair to insert a layer into.",
        )
    plan.insert_after(index.init_tail, declaration)
    parameter = index.forward_input
    plan.insert_before(
        index.forward_head,
        cst.parse_statement(f"{parameter} = self.{layer.node_id}({parameter})"),
    )


def _assigned_name(site: _Site) -> Optional[str]:
    """The plain name ``site``'s statement binds its call's result to, if any."""
    statement = site.statement
    if not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1:
        return None
    small = statement.body[0]
    if not isinstance(small, cst.Assign) or len(small.targets) != 1:
        return None
    if small.value is not site.call:
        return None
    target = small.targets[0].target
    return target.value if isinstance(target, cst.Name) else None


def _render_call(module: cst.Module, call: cst.Call) -> str:
    return module.code_for_node(call)


# ── the entry point ──────────────────────────────────────────────────────────


def apply_operations(
    source: str,
    spanmap: Mapping[str, Mapping[str, Any]],
    ops: Sequence[Mapping[str, Any]],
    *,
    cls: str,
    file: str = "",
) -> ApplyResult:
    """Turn graph operations into source edits for one class.

    ``ops`` is the ``AgentGraphOperation`` vocabulary as plain dictionaries, plus
    ``rename_layer``. Dictionaries rather than the pydantic union on purpose:
    ``rename_layer`` is not in ``ai/contracts.py`` and adding it there would
    change the generated TypeScript contract and the AI harness's closed lists
    for an operation only the IDE plane can perform (task.md §8.2).

    ``insert_between`` is not handled here. ``semantic_actions.compile_plan_items``
    expands it into ``remove_connection`` + ``add_layer`` + two ``connect``s, and
    that is the form this understands — the expansion needs an ``ArchitectureGraph``
    which this endpoint deliberately does not receive (its inputs are a buffer, a
    spanmap and operations; the backend never re-derives the graph to edit code).
    """
    index = _read(source, cls, file)
    plan = _Plan()
    refused: List[Dict[str, Any]] = []
    pending: Dict[str, _Pending] = {}
    # remove_connection is only meaningful as half of an insertion on an edge.
    # It is held until the batch is done rather than refused on sight, because
    # the connect that justifies it comes later in the same list.
    deferred: List[Tuple[Mapping[str, Any], Tuple[str, str]]] = []

    for op in ops:
        kind = str(op.get("op") or op.get("action") or "")
        op_id = str(op.get("op_id") or kind or "op")
        try:
            if kind == "group_layers":
                if len(ops) != 1:
                    raise _Refused(UNSUPPORTED_OPERATION, "Review a grouping separately from other edits.")
                _apply_group(index, plan, spanmap, op)
            elif kind == "set_param":
                _apply_set_param(index, plan, spanmap, op)
            elif kind == "replace_layer":
                _apply_replace_layer(index, plan, spanmap, op)
            elif kind == "remove_layer":
                _apply_remove_layer(index, plan, spanmap, op)
            elif kind == "rename_layer":
                rename_from = _required(op, "node_id")
                rename_to = _required(op, "new_node_id")
                _renameable(index, spanmap, rename_from)
                _rename_attribute(index, plan, rename_from, rename_to)
            elif kind == "add_layer":
                _apply_add_layer(pending, op)
            elif kind == "connect":
                _apply_connect(index, plan, spanmap, pending, deferred, op)
            elif kind == "remove_connection":
                deferred.append(
                    (op, (str(op.get("source_node_id") or ""), str(op.get("target_node_id") or "")))
                )
            else:
                raise _Refused(
                    UNSUPPORTED_OPERATION,
                    f"{kind or 'this operation'} has no single-statement source form.",
                )
        except _Refused as refusal:
            refused.append(
                {
                    "op_id": op_id,
                    "reason": refusal.reason,
                    "message": refusal.message,
                    "span": refusal.span,
                }
            )

    for op, _pair in deferred:
        refused.append(
            {
                "op_id": str(op.get("op_id") or "remove_connection"),
                "reason": UNSUPPORTED_OPERATION,
                "message": "removing an edge on its own is not a source edit; the layers themselves say what runs.",
                "span": None,
            }
        )
    written: Set[str] = set()
    for layer in pending.values():
        if not layer.placed or layer.import_ is None or layer.import_[1] in written:
            continue
        written.add(layer.import_[1])
        try:
            _ensure_import(index, plan, *layer.import_)
        except _Refused as refusal:
            refused.append(
                {
                    "op_id": layer.node_id,
                    "reason": refusal.reason,
                    "message": refusal.message,
                    "span": None,
                }
            )

    for layer in pending.values():
        if not layer.placed:
            refused.append(
                {
                    "op_id": layer.node_id,
                    "reason": UNSUPPORTED_OPERATION,
                    "message": f"{layer.node_id} was added but never connected, so there is no line to declare it on.",
                    "span": None,
                }
            )

    if plan.empty:
        return ApplyResult(edits=[], refused=refused, source=source)

    rendered = index.module.visit(_Writer(plan)).code
    try:
        ast.parse(rendered)
    except SyntaxError as error:  # pragma: no cover - a bug in this module
        raise ApplyError(f"the edit produced source that does not parse: {error}") from error
    return ApplyResult(edits=_line_edits(source, rendered), refused=refused, source=rendered)


def _apply_connect(
    index: _Index,
    plan: _Plan,
    spanmap,
    pending: Dict[str, _Pending],
    deferred: List[Tuple[Mapping[str, Any], Tuple[str, str]]],
    op: Mapping[str, Any],
) -> None:
    source_id = str(op.get("source_node_id") or "")
    target_id = str(op.get("target_node_id") or "")
    layer = pending.get(target_id)
    if layer is not None and not layer.placed:
        anchor_id = source_id
        predecessor = pending.get(source_id)
        if predecessor is not None:
            if not predecessor.placed or predecessor.anchor_id is None:
                raise _Refused(UNSUPPORTED_OPERATION, "Connect a new layer to the source before inserting another layer after it.")
            anchor_id = predecessor.anchor_id
        # Insertions at a statement append in order; nested calls compose through
        # plan.current. Newly created layers therefore reuse their original anchor.
        _place(index, plan, spanmap, layer, anchor_id)
        layer.placed = True
        layer.anchor_id = anchor_id
        deferred[:] = [entry for entry in deferred if entry[1][0] != source_id]
        return
    if pending.get(source_id) is not None or layer is not None:
        # The other half of an insertion: placing the layer between the two
        # statements already wired it. Nothing left to write.
        deferred[:] = [entry for entry in deferred if entry[1][1] != target_id]
        return
    raise _Refused(
        UNSUPPORTED_OPERATION,
        f"connecting {source_id or '?'} to {target_id or '?'} rewires two layers that both already exist; "
        "forward decides the order, and there is no single statement to change.",
    )


def _line_edits(original: str, rendered: str) -> List[Dict[str, Any]]:
    """The changed line ranges, and only those.

    LibCST is lossless, so every line outside a statement the plan touched is
    byte-identical between the two strings and the matcher reports it as
    ``equal``. That is what makes a ``set_param`` come out as one edit rather
    than a whole-file replacement — and it is why this diffs rendered output
    instead of trying to compute a range per operation.
    """
    old = original.splitlines(keepends=True)
    new = rendered.splitlines(keepends=True)
    edits: List[Dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        edits.append(
            {
                "range": {"start": [i1 + 1, 0], "end": _end_position(old, i2)},
                "newText": "".join(new[j1:j2]),
            }
        )
    return edits


def _end_position(lines: Sequence[str], index: int) -> List[int]:
    """The start of line ``index`` (0-based) as a 1-based ``[line, column]``.

    One past the last line is the end of the document, which only exists as a
    position of its own when the file ends with a newline; otherwise it is the
    end of the last line's text.
    """
    if index < len(lines):
        return [index + 1, 0]
    if not lines:
        return [1, 0]
    if lines[-1].endswith(("\n", "\r")):
        return [len(lines) + 1, 0]
    return [len(lines), len(lines[-1])]
