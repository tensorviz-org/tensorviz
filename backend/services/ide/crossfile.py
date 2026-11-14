"""Recursive cross-file static projection (step 3 of "P0 Reliable Multi-File Model
Graphs", CURRENT_TASK.md).

What the single-buffer tier could not know, this module does: which ``nn.Module``
subclass each ``self.x = SomeBlock(...)`` call site refers to, and therefore what
that submodule's own layers are. It composes the pieces the earlier steps froze:

* the **module index** (step 2) answers the "which class" question per call site,
  with canonical ``<module>::<name>`` keys and coded diagnostics instead of guesses;
* the **parser** turns each class into its own ``{nodes, edges}`` — including
  classes that only read as models through a cross-file base (``class
  Mixer(Block)``), admitted via ``extra_module_bases``;
* the **projection primitives** (identity map, annotate, spanmap) keep every node
  on the D6 identity rules; this module adds the instance half of the rule:
  repeated instances of one class yield distinct qualified ids — ``enc1.fc`` and
  ``enc2.fc`` share the *definition* but never the *node*.

Hard rules:

* **Nothing executes.** Sources come in as text; the index and the parser read
  them as syntax and nothing else.
* **Cycles are cut, not recursed.** An expansion path that returns to a class
  already on its ancestor chain renders one opaque card with a CYCLE_BOUNDARY
  diagnostic. The graph is always finite (each level consumes one class from the
  chain, and the source set is finite).
* **Unresolved stays visible.** A call the index cannot place (a factory result,
  an unresolved wildcard import, a dynamic getattr) keeps its card, marked opaque, with a
  diagnostic saying exactly why.
* **Spans are cross-file.** A node's span points at the file its statement lives
  in; the response adds that file's digest beside it.
* **Inheritance is composed, not dropped.** A class built on a workspace-local
  base (``class WideBlock(Block)``) projects the members it actually runs with:
  the base's ``nn.Module`` assignments and forward flow, composed through the
  module index in MRO order. ``super().__init__`` and ``super().forward`` calls
  are followed deterministically, and a subclass's own declaration wins over a
  same-named inherited one. A base that cannot be composed — dynamic, ambiguous,
  external, cyclic, or one of several nn.Module bases — degrades to the class's
  own members plus a coded diagnostic; nothing is executed and nothing guessed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

from core.op_registry import spec_for
from core.source_ops import SOURCE_OPS as _SOURCE_OPS
from services.ide import projection
from services.ide.module_index import ClassInfo, ModuleIndex, module_of_file
from services.ide.multifile import (
    CYCLE_BOUNDARY,
    DEPENDENCY_SYNTAX_ERROR,
    UNRESOLVED_DYNAMIC,
    UNRESOLVED_IMPORT,
    UNSUPPORTED_INHERITANCE,
)
from utils.code_parser import _is_self_super_call
from ..torch_imports import canonical_name, torch_imports
from ..ingest.module_discovery import TORCH_MODULE_BASES

#: Child layout inside an expanded module frame. These mirror the ingest tier's
#: constants (services/ingest/normalise.py) so a statically expanded module looks
#: like a traced one; kept local because the ingest package is import-heavy and
#: this endpoint is on the keystroke hot path.
_CHILD_INSET_X = 24.0
_CHILD_INSET_Y = 64.0
_CHILD_PITCH_Y = 150.0

#: Base spellings that read as nn.Module without any resolution.
_NN_MODULE_BASES = frozenset({"nn.Module", "Module", "torch.nn.Module"})


class ClassParseRefused(ValueError):
    """A file/class the parser would not turn into a graph, with the locator the
    client renders.

    For the ROOT this is a request failure (the router maps it to the same 4xx
    family ``/api/ide/parse`` uses); for a dependency it is demoted to a
    diagnostic instead, because one broken file must never erase the rest of
    the graph.
    """

    def __init__(
        self,
        message: str,
        *,
        line: int = 1,
        column: int = 0,
        syntax_error: bool = False,
        class_missing: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column
        self.syntax_error = syntax_error
        self.class_missing = class_missing


#: parse_fn(file, source, cls, extra_bases, super_forward, known_attributes, constructor_kinds) ->
#:     {"nodes": [...], "edges": [...]}. Raises ClassParseRefused when the class
#: cannot become a graph. ``super_forward`` is the (entry, exit) node pair of
#: the next forward in the resolved inheritance chain, or None;
#: ``known_attributes`` are attribute names a composed base guarantees exist
#: (the subclass's own forward may legally call them without declaring them).
#: ``constructor_kinds`` retains dotted class candidates and excludes proven
#: helpers before parsing their forward flow.
ParseFn = Callable[
    [str, str, str, Optional[Set[str]], Optional[Tuple[str, str]], Optional[Set[str]], Dict[str, bool]],
    Dict[str, Any],
]


@dataclass(frozen=True)
class WorkspaceExpansion:
    """The whole result of a cross-file projection, before packaging."""

    graph: Dict[str, Any]
    spanmap: Dict[str, Dict[str, Any]]
    #: Sorted workspace-relative files whose text the projection actually
    #: consumed: parsed files plus every file a resolution consulted — a re-export
    #: hop counts, because editing an ``__init__`` can re-route a binding.
    dependencies: Tuple[str, ...]
    diagnostics: Tuple[Dict[str, Any], ...]


def _module_class_closure(index: ModuleIndex, torch_aliases=None) -> Set[str]:
    """The class keys that ARE nn.Module subclasses, by ancestor-chain resolution.

    A class qualifies when any base is an nn.Module spelling or resolves through
    the index to a class that qualifies. Base cycles (``A(B)`` / ``B(A)``) simply
    fail: neither can be shown to be a module without executing.
    """
    memo: Dict[str, bool] = {}
    visiting: Set[str] = set()

    def is_module(key: str) -> bool:
        if key in memo:
            return memo[key]
        if key in visiting:
            return False
        visiting.add(key)
        record_module, _, name = key.partition("::")
        record = index.modules.get(record_module)
        info = record.classes.get(name) if record is not None else None
        answer = False
        if info is not None:
            for base in info.bases:
                if base in _NN_MODULE_BASES or canonical_name(base, (torch_aliases or {}).get(record_module, {})) in TORCH_MODULE_BASES:
                    answer = True
                    break
                result = index.resolve(record_module, base)
                if result.ok and isinstance(result.definition, ClassInfo):
                    if is_module(result.definition.key):
                        answer = True
                        break
        visiting.discard(key)
        memo[key] = answer
        return answer

    for record in index.modules.values():
        for info in record.classes.values():
            is_module(info.key)
    return {key for key, value in memo.items() if value}


def _init_call_targets(source: str, cls: str) -> Dict[str, str]:
    """``{local name: callee text}`` for init-level ``name = call(...)`` lines.

    ``block_cls = load_block_class(kind)`` is how dynamic module selection reads;
    recognising it turns the honest diagnostic for the later ``block_cls(...)``
    from "no such import" into "this was computed at runtime".
    """
    found: Dict[str, str] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return found
    target = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls),
        None,
    )
    if target is None:
        return found
    init = next(
        (n for n in target.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    )
    if init is None:
        return found
    for statement in init.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
            for target_name in statement.targets:
                if isinstance(target_name, ast.Name):
                    found[target_name.id] = ast.unparse(statement.value.func)
    return found


def _super_calls(source: str, cls: str) -> Tuple[bool, bool, bool, bool]:
    """(has_init, calls_super_init, has_forward, calls_super_forward) for ``cls``.

    Static admission test for inheritance composition: only the zero-argument
    ``super().__init__(...)`` / ``super().forward(...)`` spellings (and the
    equivalent explicit two-argument form, which names the same next class) are
    followed. Anything else — ``Base.__init__(self, ...)``, delegated helpers,
    conditional calls — is outside this tier and degrades to a diagnostic.

    ``calls_super_init`` is stricter than a walk: the call must be an
    unconditional top-level ``super().__init__(...)`` statement of ``__init__``
    — only then is the base's ``__init__`` provably run. ``calls_super_forward``
    tolerates conditionality the way the parser tolerates conditional ops: the
    static graph over-approximates flow, honestly labelled.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return (False, False, False, False)
    target = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls),
        None,
    )
    if target is None:
        return (False, False, False, False)

    def is_super_call(node: ast.AST, attr: str) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr
            and _is_self_super_call(node.func.value, cls)
        )

    init = next(
        (n for n in target.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    )
    forward = next(
        (n for n in target.body if isinstance(n, ast.FunctionDef) and n.name == "forward"),
        None,
    )
    calls_super_init = init is not None and any(
        isinstance(statement, ast.Expr) and is_super_call(statement.value, "__init__")
        for statement in init.body
    )
    calls_super_forward = forward is not None and any(
        is_super_call(node, "forward") for node in ast.walk(forward)
    )
    return (init is not None, calls_super_init, forward is not None, calls_super_forward)


@dataclass
class _Composed:
    """One class's own graph merged with its composed base, per level.

    Node ids are in identity-mapped per-class space (attributes keep their
    names, forward-synthesised nodes carry ``fn@DefiningClass#ordinal``) and
    stay per-level until the instance qualification in ``_expand_class`` — so
    two instances of one class never share an inherited fn node. Edges are
    plain (source, target) dicts in the same space.
    """

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    attributes: Set[str]
    #: ``{local name: callee text}`` from every composed level's __init__.
    factories: Dict[str, str]


def _node_file(node: Dict[str, Any], fallback: str) -> str:
    """The file this node's statement lives in.

    For an inherited member that is the BASE's file, not the subclass's, so
    navigation lands on the line that actually declares the layer and candidate
    resolution runs against the base's imports.
    """
    params = (node.get("data") or {}).get("parameters") or {}
    source_file = params.get("_source_file")
    return str(source_file) if isinstance(source_file, str) and source_file else fallback


class _Expander:
    def __init__(
        self,
        index: ModuleIndex,
        sources: Mapping[str, str],
        parse_fn: ParseFn,
    ) -> None:
        self.index = index
        self.sources = sources
        self.parse_fn = parse_fn
        self.torch_aliases = {}
        for file, source in sources.items():
            try:
                self.torch_aliases[module_of_file(file)] = torch_imports(ast.parse(source))
            except SyntaxError:
                pass  # Source admission owns the located syntax diagnostic.
        self.module_keys = _module_class_closure(index, self.torch_aliases)
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self._entry_roles: Dict[str, Dict[str, Any]] = {}
        self.spans: Dict[str, Dict[str, Any]] = {}
        self.diagnostics: List[Dict[str, Any]] = []
        self._seen_diagnostics: Set[Tuple[Any, ...]] = set()
        self._edge_seen: Set[Tuple[str, str, int]] = set()
        self.consulted: Set[str] = set()
        #: Expanded wrapper id -> the qualified ids of its direct graph
        #: entries/exits (which may themselves be expanded wrappers).
        self.entries: Dict[str, List[str]] = {}
        self.exits: Dict[str, List[str]] = {}
        self._extra_bases_cache: Dict[str, Set[str]] = {}

    # ── diagnostics ───────────────────────────────────────────────────────────

    def _constructor_kinds(self, file: str, source: str, cls: str) -> Dict[str, bool]:
        """Dotted class candidates versus proven helpers, before parsing flow.

        Unknown calls remain absent: the parser retains class-shaped candidates
        as opaque, and _resolve_candidate supplies the resolution diagnostic.
        A resolved alias needs no capitalization convention to be a module.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {}  # The parse adapter owns located source-error refusals.
        target = next((n for n in ast.walk(tree)
                       if isinstance(n, ast.ClassDef) and n.name == cls), None)
        if target is None:
            return {}
        init = next((n for n in target.body if isinstance(n, ast.FunctionDef)
                     and n.name == "__init__"), None)
        kinds: Dict[str, bool] = {}
        for call in ast.walk(init) if init is not None else ():
            if not isinstance(call, ast.Call) or not isinstance(call.func, (ast.Name, ast.Attribute)):
                continue
            callee = ast.unparse(call.func)
            if not all(part.isidentifier() for part in callee.split(".")):
                continue
            result = self.index.resolve(module_of_file(file), callee, visited=self.consulted)
            if result.definition is not None:
                # Unknown ancestry can still be a module. Preserve its opaque
                # candidate so _resolve_candidate explains the unsupported base.
                kinds[callee] = (result.definition.key in self.module_keys
                                 or any(base != "object" for base in result.definition.bases))
            elif result.non_class:
                kinds[callee] = False
        return kinds

    def _emit(self, diagnostic: Dict[str, Any]) -> None:
        key = (
            diagnostic.get("code"),
            diagnostic.get("file"),
            diagnostic.get("line"),
            diagnostic.get("detail"),
        )
        if key in self._seen_diagnostics:
            return
        self._seen_diagnostics.add(key)
        self.diagnostics.append(diagnostic)

    # ── per-file extra bases ──────────────────────────────────────────────────

    def _extra_bases(self, file: str, module: str) -> Set[str]:
        """Base spellings in ``file`` that resolve to known nn.Module subclasses."""
        if file in self._extra_bases_cache:
            return self._extra_bases_cache[file]
        extra: Set[str] = set()
        try:
            tree = ast.parse(self.sources[file], filename=file)
        except (SyntaxError, ValueError):
            self._extra_bases_cache[file] = extra
            return extra
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                spelling = ast.unparse(base)
                if spelling in _NN_MODULE_BASES:
                    continue
                if canonical_name(spelling, self.torch_aliases.get(module, {})) in TORCH_MODULE_BASES:
                    extra.add(spelling)
                    continue
                result = self.index.resolve(module, spelling, visited=self.consulted)
                if (
                    result.ok
                    and isinstance(result.definition, ClassInfo)
                    and result.definition.key in self.module_keys
                ):
                    extra.add(spelling)
        self._extra_bases_cache[file] = extra
        return extra

    # ── the recursion ─────────────────────────────────────────────────────────

    # ── single local inheritance ──────────────────────────────────────────────

    def _select_base(
        self,
        file: str,
        cls: str,
        has_init: bool,
        calls_super_init: bool,
        chain: Tuple[str, ...],
    ) -> Optional[ClassInfo]:
        """The one composable base of ``cls``, or None after a coded diagnostic.

        Compose only when exactly one base spelling resolves through the module
        index to a workspace-local nn.Module subclass that is not already on the
        ancestor chain, and the subclass's __init__ provably runs the base's
        (its own ``super().__init__`` call, or no own ``__init__`` at all).
        Every other shape — multiple module bases, dynamic/ambiguous/external
        spellings, base cycles, explicit ``Base.__init__(self, ...)`` — keeps the
        subclass's own members only, with the reason on the record.
        """
        module = module_of_file(file)
        record = self.index.modules.get(module)
        info = record.classes.get(cls) if record is not None else None
        if info is None:
            return None
        spellings = [b for b in info.bases if b not in _NN_MODULE_BASES
                     and canonical_name(b, self.torch_aliases.get(module, {})) != "torch.nn.Module"]
        if not spellings:
            return None

        def emit(
            code: str, detail: str, import_text: str = "", *, refused: bool = True
        ) -> None:
            outcome = (
                f"{cls} keeps its own members only"
                if refused
                else f"{cls} composes only its nn.Module base's members"
            )
            self._emit(
                {
                    "code": code,
                    "severity": "error" if code == DEPENDENCY_SYNTAX_ERROR else "warning",
                    "file": file,
                    "line": info.line,
                    "import_text": import_text,
                    "detail": f"{detail}; {outcome}",
                }
            )

        # Per-spelling failures are held back: their honest outcome text depends
        # on whether composition is refused (own members only) or proceeds for
        # the one nn.Module base (only that base's members are composed).
        pending: List[Tuple[str, str, str]] = []
        resolved: List[ClassInfo] = []
        for spelling in spellings:
            result = self.index.resolve(module, spelling, visited=self.consulted)
            definition = (
                result.definition if isinstance(result.definition, ClassInfo) else None
            )
            if definition is not None and definition.key in self.module_keys:
                resolved.append(definition)
                continue
            diag = dict(result.diagnostic or {})
            if definition is not None:
                pending.append((
                    UNRESOLVED_IMPORT,
                    f"base `{spelling}` of {cls} resolves to {definition.key} in "
                    f"{definition.file}, which is not an nn.Module subclass",
                    "",
                ))
            else:
                pending.append((
                    str(diag.get("code") or UNRESOLVED_IMPORT),
                    f"base `{spelling}` of {cls} cannot be composed: "
                    + str(diag.get("detail") or "it could not be resolved statically"),
                    str(diag.get("import_text") or ""),
                ))
        if not resolved:
            for code, detail, import_text in pending:
                emit(code, detail, import_text)
            return None
        if len(resolved) > 1:
            emit(
                UNSUPPORTED_INHERITANCE,
                f"{cls} has several nn.Module bases "
                f"({', '.join(d.name for d in resolved)}); multiple inheritance is "
                "outside static composition",
            )
            for code, detail, import_text in pending:
                emit(code, detail, import_text)
            return None
        base = resolved[0]
        if base.key in chain:
            loop = " -> ".join(k.partition("::")[2] for k in chain + (base.key,))
            emit(
                CYCLE_BOUNDARY,
                f"{cls} inherits from {base.name}, closing an inheritance cycle "
                f"({loop})",
            )
            for code, detail, import_text in pending:
                emit(code, detail, import_text)
            return None
        if has_init and not calls_super_init:
            emit(
                UNSUPPORTED_INHERITANCE,
                f"{cls}.__init__ has no unconditional top-level super().__init__() "
                "statement, so the base's own `self.` assignments are not provably "
                "run; conditional calls and explicit `Base.__init__(self, ...)` "
                "spellings are not followed",
            )
            for code, detail, import_text in pending:
                emit(code, detail, import_text)
            return None
        for code, detail, import_text in pending:
            emit(code, detail, import_text, refused=False)
        return base

    def _compose_class(
        self, file: str, cls: str, chain: Tuple[str, ...]
    ) -> _Composed:
        """Own parse of ``cls`` merged with its composable base, recursively.

        Per level the parser sees exactly one class (with the ``super_forward``
        bridge when it is decidable: the base's flow reduces to exactly one
        entry and one exit); identity mapping runs per level keyed on the
        DEFINING class; the subclass's own declarations then win over same-named
        inherited ones. A base whose forward the subclass replaces keeps only
        its member-internal edges — its boundary edges described a flow the
        subclass overrode.
        """
        module = module_of_file(file)
        source = self.sources[file]
        extra = self._extra_bases(file, module)
        self.consulted.add(file)

        has_init, calls_super_init, has_forward, calls_super_forward = _super_calls(
            source, cls
        )
        base_info = self._select_base(file, cls, has_init, calls_super_init, chain)
        base: Optional[_Composed] = None
        if base_info is not None:
            try:
                base = self._compose_class(
                    base_info.file, base_info.name, chain + (base_info.key,)
                )
            except ClassParseRefused as error:
                self._emit(
                    {
                        "code": DEPENDENCY_SYNTAX_ERROR
                        if error.syntax_error
                        else UNRESOLVED_IMPORT,
                        "severity": "error" if error.syntax_error else "warning",
                        "file": base_info.file,
                        "line": error.line,
                        "import_text": "",
                        "detail": (
                            f"base class {base_info.key} of {cls} could not be parsed "
                            f"({error.message}); {cls} keeps its own members only"
                        ),
                    }
                )
                base = None

        bridge: Optional[Tuple[str, str]] = None
        if base is not None and calls_super_forward:
            entries = sorted(
                {
                    str(e.get("target"))
                    for e in base.edges
                    if str(e.get("source")) == "input"
                }
            )
            exits = sorted(
                {
                    str(e.get("source"))
                    for e in base.edges
                    if str(e.get("target")) == "output"
                }
            )
            if len(entries) == 1 and len(exits) == 1:
                bridge = (entries[0], exits[0])

        # The subclass's own forward may call members it does not declare
        # because the composed base's __init__ runs first; telling the parser
        # about them is what wires `self.fc(x)` to the inherited fc card.
        graph = self.parse_fn(
            file,
            source,
            cls,
            extra,
            bridge,
            set(base.attributes) if base is not None else None,
            self._constructor_kinds(file, source, cls),
        )
        attributes = projection.init_attributes(source, cls)
        own_nodes = [dict(node) for node in graph.get("nodes") or []]
        own_edges = [dict(edge) for edge in graph.get("edges") or []]
        own = projection._rename(
            {"nodes": own_nodes, "edges": own_edges},
            projection.identity_map(own_nodes, cls=cls, attributes=attributes),
        )
        own_factories = _init_call_targets(source, cls)
        if base is None:
            return _Composed(own["nodes"], own["edges"], attributes, own_factories)

        # Overrides win: an inherited member the subclass reassigns is dropped;
        # any surviving base edge that names it now flows through the subclass's
        # node of the same id, which is the dynamic-dispatch truth.
        overridden = base.attributes & attributes
        inherited_nodes = [
            node
            for node in base.nodes
            if str(node.get("id")) not in overridden
            and str(node.get("id")) not in projection.BOUNDARY_IDS
        ]
        combined_attributes = set(attributes) | (base.attributes - overridden)
        combined_nodes = inherited_nodes + own["nodes"]
        known = {str(node.get("id")) for node in combined_nodes} | set(
            projection.BOUNDARY_IDS
        )
        combined_edges: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        base_flow = base.edges
        if has_forward:
            # The subclass's own forward replaces the base flow's wiring to this
            # class's boundary; what survives is the member-internal flow.
            base_flow = [
                edge
                for edge in base.edges
                if str(edge.get("source")) not in projection.BOUNDARY_IDS
                and str(edge.get("target")) not in projection.BOUNDARY_IDS
            ]
        # Repeated operands are separate uses. Overlapping inherited/own flow
        # keeps the maximum multiplicity, rather than collapsing every pair.
        for flow in (base_flow, own["edges"]):
            occurrences = {}
            for edge in flow:
                source_id, target_id = str(edge.get("source")), str(edge.get("target"))
                if source_id not in known or target_id not in known:
                    continue
                pair = (source_id, target_id)
                ordinal = occurrences.get(pair, 0)
                occurrences[pair] = ordinal + 1
                combined_edges.setdefault(
                    (*pair, ordinal),
                    {**edge, "id": f"edge_{source_id}_{target_id}" + (f"_{ordinal}" if ordinal else "")},
                )
        factories = {**base.factories, **own_factories}
        return _Composed(
            combined_nodes, list(combined_edges.values()), combined_attributes, factories
        )

    # ── the recursion ─────────────────────────────────────────────────────────

    def expand_root(self, file: str, cls: str) -> None:
        root_key = f"{module_of_file(file)}::{cls}"
        self._expand_class(file, cls, instance="", parent=None, chain=(root_key,))

    def _expand_class(
        self,
        file: str,
        cls: str,
        *,
        instance: str,
        parent: Optional[str],
        chain: Tuple[str, ...],
    ) -> Tuple[List[str], List[str]]:
        """Parse one class, qualify its nodes under ``instance``, expand its
        custom submodules, and return its graph's (entry ids, exit ids).

        ``instance`` is the qualified attribute path of the object (``""`` for the
        root). ``chain`` is the ancestor class keys, root first — the working
        cycle detector: a candidate whose class is already on it is the boundary.
        """
        is_root = not instance
        depth = len(chain) - 1
        composed = self._compose_class(file, cls, chain)
        attributes = composed.attributes
        nodes = [dict(node) for node in composed.nodes]

        # Two composed LEVELS can synthesise identical per-class fn ids (the
        # base's forward and the subclass's own both call torch.relu, say), and
        # qualification drops the defining-class component — without care the
        # two statements fold into one node and their edges converge. A short
        # form claimed by more than one class-level id keeps its owner segment,
        # so every statement stays its own node and non-colliding ids (the
        # common case) keep the short `head@instance#ordinal` spelling.
        short_members: Dict[str, List[str]] = {}
        if instance:
            for node in nodes:
                class_id = str(node.get("id"))
                if (
                    class_id in projection.BOUNDARY_IDS
                    or class_id in attributes
                    or "@" not in class_id
                    or "#" not in class_id
                ):
                    continue
                head, _, tail = class_id.partition("@")
                _, _, ordinal = tail.partition("#")
                short_members.setdefault(f"{head}@{instance}#{ordinal}", []).append(
                    class_id
                )
        collided_shorts = {
            short for short, members in short_members.items() if len(members) > 1
        }

        def qualify(node_id: str) -> str:
            if node_id in projection.BOUNDARY_IDS:
                return node_id
            if node_id in attributes:
                return f"{instance}.{node_id}" if instance else node_id
            if instance and "@" in node_id:
                # Forward-synthesised nodes: the owner segment is the DEFINING
                # class (even an inherited one); the instance is what makes the
                # same declaration on two objects two different nodes.
                head, _, tail = node_id.partition("@")
                owner, _, ordinal = tail.partition("#")
                short = f"{head}@{instance}#{ordinal}"
                if short in collided_shorts:
                    return f"{head}@{instance}@{owner}#{ordinal}"
                return short
            return node_id

        renamed = projection._rename(
            {"nodes": nodes, "edges": composed.edges},
            {str(node.get("id")): qualify(str(node.get("id"))) for node in nodes},
        )

        qualified_attrs = {qualify(a) for a in attributes}
        boundaries = set(projection.BOUNDARY_IDS)
        entries: List[str] = []
        exits: List[str] = []
        own_node_ids: Set[str] = set()
        own_nodes: List[Dict[str, Any]] = []
        for position, node in enumerate(renamed["nodes"]):
            node_id = str(node.get("id"))
            if not is_root and node_id in boundaries:
                continue  # a nested class's own input/output are its parent's bridges
            node_id = str(node.get("id"))
            data = dict(node.get("data") or {})
            node["data"] = data
            data["parentId"] = parent
            data["depth"] = depth
            if not is_root:
                # Inside an expanded frame children lay out in one column under
                # the parent's origin; the canvas layout pass re-flows anyway.
                node["position"] = {
                    "x": _CHILD_INSET_X,
                    "y": _CHILD_INSET_Y + position * _CHILD_PITCH_Y,
                }
            projection._annotate(node, attributes=qualified_attrs)
            own_node_ids.add(node_id)
            own_nodes.append(node)

        local_edges: List[Tuple[str, str]] = []
        local_roles = {}
        for edge in renamed["edges"]:
            source_id, target_id = str(edge.get("source")), str(edge.get("target"))
            role = edge.get("data") or {}
            local_roles[(source_id, target_id)] = role
            if source_id == "input" and target_id in own_node_ids:
                entries.append(target_id)
                if not is_root and role.get("role") == "residual":
                    self._entry_roles[target_id] = role
            elif source_id in own_node_ids and target_id == "output":
                exits.append(source_id)
            if source_id in own_node_ids and target_id in own_node_ids:
                local_edges.append((source_id, target_id))
            elif is_root:
                # The root's own boundaries stay on the graph it draws.
                if source_id in own_node_ids or target_id in own_node_ids:
                    local_edges.append((source_id, target_id))

        # Register this level before recursing, so the parent card is in the
        # graph before its members — the order the canvas adapters assume.
        for node in own_nodes:
            self.nodes.append(node)
            self.spans.update(
                projection._spanmap(
                    [node], attributes=qualified_attrs, rel_file=_node_file(node, file)
                )
            )

        for node in own_nodes:
            node_id = str(node.get("id"))
            if node_id not in qualified_attrs:
                continue
            defining_file = _node_file(node, file)
            self._resolve_candidate(
                node,
                file=defining_file,
                module=module_of_file(defining_file),
                chain=chain,
                factories=composed.factories,
            )

        # Count uses within this class. Re-emitting the same bridge from an
        # enclosing class must not duplicate it, but repeated operands must survive.
        occurrences = {}
        def emit_edge(source_id, target_id, data):
            pair = (source_id, target_id)
            ordinal = occurrences.get(pair, 0)
            occurrences[pair] = ordinal + 1
            self._emit_edge(source_id, target_id, data, ordinal)

        bridge = not is_root  # a nested class's boundary edges are the parent's
        for source_id, target_id in local_edges:
            if bridge and (source_id in boundaries or target_id in boundaries):
                continue
            for real_source in self._deep_exits(source_id):
                for real_target in self._deep_entries(target_id):
                    if (real_source, real_target) != (source_id, target_id):
                        emit_edge(real_source, real_target, local_roles.get((source_id, target_id))
                                        or (self._entry_roles.get(real_target) if real_target != target_id else None))
            emit_edge(source_id, target_id, local_roles.get((source_id, target_id)))

        return entries, exits

    def _deep_entries(self, node_id: str) -> List[str]:
        hits = self.entries.get(node_id)
        if not hits:
            return [node_id]
        out: List[str] = []
        for hit in hits:
            out.extend(self._deep_entries(hit))
        return out

    def _deep_exits(self, node_id: str) -> List[str]:
        hits = self.exits.get(node_id)
        if not hits:
            return [node_id]
        out: List[str] = []
        for hit in hits:
            out.extend(self._deep_exits(hit))
        return out

    # ── candidate handling ────────────────────────────────────────────────────

    def _resolve_candidate(
        self,
        node: Dict[str, Any],
        *,
        file: str,
        module: str,
        chain: Tuple[str, ...],
        factories: Mapping[str, str],
    ) -> None:
        data = node["data"]
        params = dict(data.get("parameters") or {})
        node_id = str(node.get("id"))
        layer_id = str((data.get("layerType") or {}).get("id") or "").lower()
        raw_call = str(params.get("_raw_call") or "")
        callee = raw_call.split("(", 1)[0].strip()
        if not callee:
            return
        result = self.index.resolve(module, callee, visited=self.consulted)
        definition = result.definition if isinstance(result.definition, ClassInfo) else None

        # Resolution answers before the op registry does: the parser lowercases
        # the CALL CLASS NAME into the layer id, so a workspace class whose name
        # merely collides with an op id (`class Sub(...)`, `class Add`) would
        # otherwise be skipped as a "registry op" — never expanded, never
        # diagnosed. Only a callee the index cannot ground in a definition falls
        # through to the registry check, which keeps `nn.Linear(...)` and its
        # siblings silent exactly as the single-file tier always has.
        span = self.spans.get(node_id) or {}
        start = span.get("start") or [params.get("_lineno") or 0, 0]
        at_line = int(start[0]) if start else 0

        def opaque_diagnostic(
            code: str, detail: str, import_text: str = "", severity: str = ""
        ) -> None:
            # An opaque card is not automatically a warning. The resolver already
            # decides how bad each reason is — a workspace file the caller failed to
            # supply is a warning, the framework having no registry spec for one of
            # its own classes is information — and re-stamping every one of them as
            # "warning" here is what put a yellow badge on `nn.TransformerEncoder`
            # saying torch.nn was unavailable.
            self._emit(
                {
                    "code": code,
                    "severity": severity
                    or ("error" if code == DEPENDENCY_SYNTAX_ERROR else "warning"),
                    "file": file,
                    "line": at_line,
                    "import_text": import_text,
                    "detail": f"{detail} [node {node_id}]",
                    "node": node_id,
                }
            )
            self._mark_opaque(node, params, callee, definition)

        if definition is None or definition.key not in self.module_keys:
            if definition is not None:
                opaque_diagnostic(
                    UNRESOLVED_IMPORT,
                    f"{callee} resolves to {definition.key} in {definition.file}, but "
                    "that class is not an nn.Module subclass, so it cannot own layers; "
                    "the card stays opaque rather than pretending to be a module",
                )
                return
            if layer_id != "custom_code" and spec_for(layer_id) is not None:
                return  # a registry op; the existing single-file semantics win
            if layer_id in _SOURCE_OPS:
                # `nn.Parameter` and `nn.Buffer` are canonical framework constructs, not
                # third-party classes that stayed opaque: they have no forward to expand,
                # which is the definition of a source, not a failure to resolve. They have
                # no op-registry spec either — nothing infers a shape THROUGH them — so
                # they fell to the external-package branch and every parameter on the
                # canvas carried a warning telling the user its own class was unavailable.
                # Imported rather than respelled, for the reason verification_service
                # gives: a fourth spelling of this set is how the first three drifted.
                return
            if callee in factories:
                opaque_diagnostic(
                    UNRESOLVED_DYNAMIC,
                    f"`{callee}` is computed at runtime by {factories[callee]}(...) in "
                    "this __init__; static projection keeps the node opaque rather "
                    "than guessing a class for it",
                )
                return
            base = dict(result.diagnostic or {})
            opaque_diagnostic(
                str(base.get("code") or UNRESOLVED_IMPORT),
                str(base.get("detail") or f"{callee} could not be resolved statically"),
                import_text=str(base.get("import_text") or ""),
                severity=str(base.get("severity") or ""),
            )
            return

        if definition.key in chain:
            loop = " -> ".join(k.partition("::")[2] for k in chain + (definition.key,))
            opaque_diagnostic(
                CYCLE_BOUNDARY,
                f"{callee} closes an import/expansion cycle ({loop}); the boundary "
                "is one opaque card, not infinite recursion",
            )
            return

        # Resolved and not cyclic: expand the class under this instance.
        params["_class"] = definition.name
        params.pop("_opaque", None)
        params["_import_path"] = definition.module
        params["_defined_in"] = definition.file
        params["_def_line"] = definition.line
        data["parameters"] = params
        data["layerType"] = {"id": "custom_code", "name": "custom_code", "parameters": []}
        try:
            child_entries, child_exits = self._expand_class(
                definition.file,
                definition.name,
                instance=node_id,
                parent=node_id,
                chain=chain + (definition.key,),
            )
        except ClassParseRefused as error:
            code = DEPENDENCY_SYNTAX_ERROR if error.syntax_error else UNRESOLVED_IMPORT
            detail = (
                f"the static parser could not build a graph for {definition.key}: "
                f"{error.message}"
            )
            self._emit(
                {
                    "code": code,
                    "severity": "error" if code == DEPENDENCY_SYNTAX_ERROR else "warning",
                    "file": definition.file,
                    "line": error.line,
                    "import_text": "",
                    "detail": f"{detail} [node {node_id}]",
                    "node": node_id,
                }
            )
            self._mark_opaque(node, params, callee, definition)
            return

        member_ids = [
            str(candidate.get("id"))
            for candidate in self.nodes
            if (candidate.get("data") or {}).get("parentId") == node_id
        ]
        node["type"] = "moduleNode"
        data["isGroup"] = True
        data["expanded"] = False
        data["moduleType"] = "custom"
        data["displayName"] = node_id.split(".")[-1] or definition.name
        data["moduleInstanceId"] = node_id
        data["moduleTemplateId"] = definition.key
        data["moduleLayerCount"] = len(member_ids)
        data["moduleMembers"] = member_ids
        data["moduleEntryNodeId"] = child_entries[0] if child_entries else None
        data["moduleExitNodeId"] = child_exits[-1] if child_exits else None
        if child_entries:
            self.entries[node_id] = child_entries
        if child_exits:
            self.exits[node_id] = child_exits

    def _mark_opaque(
        self,
        node: Dict[str, Any],
        params: Dict[str, Any],
        callee: str,
        definition: Optional[ClassInfo],
    ) -> None:
        data = node["data"]
        params["_class"] = str(params.get("_class") or callee.split(".")[-1])
        if definition is not None:
            params["_import_path"] = definition.module
            params["_defined_in"] = definition.file
        params["_opaque"] = True
        data["parameters"] = params
        data["layerType"] = {"id": "custom_code", "name": "custom_code", "parameters": []}
        data.setdefault("displayName", str(node.get("id")).split(".")[-1])

    def _emit_edge(self, source_id: str, target_id: str, data=None, ordinal: int = 0) -> None:
        pair = (source_id, target_id, ordinal)
        if pair in self._edge_seen:
            return
        self._edge_seen.add(pair)
        self.edges.append(
            {"id": f"edge_{source_id}_{target_id}" + (f"_{ordinal}" if ordinal else ""),
             "source": source_id, "target": target_id, "data": data}
        )

    # ── assembly ──────────────────────────────────────────────────────────────

    def finish(self) -> WorkspaceExpansion:
        # Even an export that supplies no used class can change which name wins.
        # Keep every examined wildcard source in the host's freshness guards.
        for record in self.index.modules.values():
            if record.file in self.consulted:
                self.consulted.update(record.wildcard_dependencies)
        for record_name in sorted(self.index.modules):
            record = self.index.modules[record_name]
            if record.file not in self.consulted:
                continue
            for diagnostic in list(record.import_diagnostics) + list(record.dynamic_diagnostics):
                self._emit(dict(diagnostic))
        severity_rank = {"error": 0, "warning": 1, "info": 2}
        self.diagnostics.sort(
            key=lambda d: (
                severity_rank.get(str(d.get("severity")), 1),
                str(d.get("file") or ""),
                int(d.get("line") or 0),
                str(d.get("code") or ""),
            )
        )
        files = set(self.consulted)
        for diagnostic in self.diagnostics:
            if diagnostic.get("file"):
                files.add(str(diagnostic["file"]))
        return WorkspaceExpansion(
            graph={"nodes": self.nodes, "edges": self.edges},
            # Columns convert once, here, so each file's lines are split a single time.
            spanmap=projection.utf16_spans(self.spans, self.sources),
            dependencies=tuple(sorted(files)),
            diagnostics=tuple(self.diagnostics),
        )


def project_workspace(
    index: ModuleIndex,
    sources: Mapping[str, str],
    *,
    root_file: str,
    root_cls: str,
    parse_fn: ParseFn,
) -> WorkspaceExpansion:
    """Expand ``root_cls`` from ``root_file`` against the whole source set.

    Raises ``ClassParseRefused`` only for the root: a dependency class that
    refuses degrades to a diagnostic on its wrapper card instead, so one broken
    file never erases the rest of the model.
    """
    expander = _Expander(index, sources, parse_fn)
    expander.expand_root(root_file, root_cls)
    return expander.finish()
