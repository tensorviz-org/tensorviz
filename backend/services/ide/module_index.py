"""A deterministic static index over the supplied workspace source set
(CURRENT_TASK.md step 2, "P0 Reliable Multi-File Model Graphs").

AST only and graph-schema independent: the index answers "which class definition
does this name refer to" from source text, never from importing it. Parsing the
sources must not execute them, must not read the disk, and must not write
anything — the same promise ``POST /api/ide/multifile/project`` makes about the
set these modules come from.

What it indexes:

* **canonical modules** — a file ``models/blocks.py`` is module
  ``models.blocks``; its package's ``models/__init__.py`` is module ``models``.
  Two paths that canonicalise to the same module are a conflict, surfaced as an
  ``AMBIGUOUS_BINDING`` diagnostic rather than a coin-flip winner.
* **class definitions** — top-level ``class`` statements, keyed by
  ``<module>::<name>`` (so ``models.blocks::Block`` and ``models.heads::Block``
  are different definitions that never overwrite one another);
* **bindings** — what each module-level import statement makes a local name
  mean, for the supported forms:

      from pkg.blocks import Block            # absolute from-import
      from pkg.blocks import Block as Alias   # aliased
      from .blocks import Block               # relative, any level, ``from . import``
      import pkg.blocks as blocks             # module-as-value
      from pkg import ReExported              # through an unambiguous __init__ re-export
      from .blocks import *                   # proven local public names or literal __all__

* **per-module notes** — syntax errors (isolated: one bad file never erases the
  rest of the index), wildcard imports, conditional imports, and dynamic
  constructs (``importlib.import_module`` calls, computed ``getattr``) that a
  static index is right not to guess.

Resolution never guesses. A name with no binding, two candidate bindings, a
cycle, or a module outside the source set produces a coded diagnostic (the same
closed codes as the projection contract) — the class stays opaque and visible,
not silently dropped or quietly picked.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Tuple

from services.ide import multifile
from .source_decorators import known_decorators

_MAX_WILDCARD_EXPORTS = 4096
_MAX_WILDCARD_BINDINGS = 16384


@dataclass(frozen=True)
class ClassInfo:
    """One class definition, canonically addressed (``module::name``)."""

    module: str
    name: str
    file: str
    line: int
    end_line: int
    column: int
    end_column: int
    bases: Tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.module}::{self.name}"

    def as_dict(self) -> Dict[str, object]:
        return {
            "module": self.module,
            "name": self.name,
            "file": self.file,
            "line": self.line,
            "end_line": self.end_line,
            "column": self.column,
            "end_column": self.end_column,
            "bases": list(self.bases),
            "key": self.key,
        }


@dataclass(frozen=True)
class Binding:
    """What one module-level import statement makes a local name mean.

    ``target_module``/``target_name`` describe the *imported* thing; whether the
    binding ends up a class or a module reference is decided at resolve time —
    ``from . import blocks`` and ``from .blocks import Block`` both pass through
    here.
    """

    kind: str  # "name" | "module" | "wildcard"
    name: str
    target_module: str
    target_name: Optional[str]
    file: str
    line: int
    import_text: str


@dataclass(frozen=True)
class ModuleRecord:
    """Everything the index knows about one supplied source file."""

    file: str
    module: str
    package: str  # the package the module sits IN ('' for top-level files)
    syntax_error: Optional[Dict[str, object]]
    classes: Mapping[str, ClassInfo] = field(default_factory=dict)
    bindings: Mapping[str, Tuple[Binding, ...]] = field(default_factory=dict)
    dynamic_diagnostics: Tuple[Dict[str, object], ...] = ()
    import_diagnostics: Tuple[Dict[str, object], ...] = ()  # wildcard, etc.
    has_conditional_imports: bool = False
    functions: Tuple[str, ...] = ()
    bound_names: Tuple[str, ...] = ()
    explicit_exports: Optional[Tuple[str, ...]] = None
    exports_dynamic: bool = False
    uncertain_names: Tuple[str, ...] = ()
    wildcard_dependencies: Tuple[str, ...] = ()
    unresolved_wildcard: bool = False


@dataclass(frozen=True)
class ResolveResult:
    """Either a definition or a coded diagnostic; never both, never silent."""

    definition: Optional[ClassInfo]
    diagnostic: Optional[Dict[str, object]]
    # A proven function is different from a missing/unsupported constructor.
    non_class: bool = False

    @property
    def ok(self) -> bool:
        return self.definition is not None


def module_of_file(path: str) -> str:
    """The canonical dotted module path of a normalised workspace file."""
    parts = path[: -len(".py")].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def package_of_file(path: str) -> str:
    """The package a file sits IN: ``models/net.py`` → ``models``."""
    parts = path.split("/")[:-1]
    return ".".join(parts)


def _diag(
    code: str,
    severity: str,
    file: str,
    line: int,
    import_text: str,
    detail: str,
) -> Dict[str, object]:
    return {
        "code": code,
        "severity": severity,
        "file": file,
        "line": line,
        "import_text": import_text,
        "detail": detail,
    }


# Deeper ASTs (long generated operator chains) exceed the recursion limit of
# ast.NodeVisitor-based passes. Hand-written model code is far below this.
_MAX_AST_DEPTH = 200


def _deepest_line_beyond(tree: ast.AST, limit: int) -> Optional[int]:
    """Line of the first node nested deeper than ``limit``, else None. Iterative."""
    stack = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > limit:
            return getattr(node, "lineno", 1) or 1
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return None


class ModuleIndex:
    """Built once per projection request over the validated source set."""

    def __init__(self, modules: Dict[str, ModuleRecord]) -> None:
        self.modules = modules

    # ── build ────────────────────────────────────────────────────────────────

    @classmethod
    def build(cls, sources: Mapping[str, str]) -> "ModuleIndex":
        """Index the given ``{path: source}`` map. Paths are re-normalised here
        even though the contract validated them: an index built from a path that
        escapes the set is a path the resolver could escape the set with, and
        the cheap re-check is the whole defence in depth.
        """
        staged: Dict[str, Dict[str, object]] = {}
        for path, source in sources.items():
            clean = multifile.normalize_workspace_path(path)
            module = module_of_file(clean)
            record = staged.setdefault(module, {"module": module, "paths": []})
            record["paths"].append(clean)

        modules: Dict[str, ModuleRecord] = {}
        for module in sorted(staged):
            # Deterministic conflict handling: every path canonicalising to one
            # module gets a record entry that resolves as ambiguous, so nothing
            # with two spellings sneaks one definition past the other.
            paths = staged[module]["paths"]
            if len(paths) > 1:
                modules[module] = ModuleRecord(
                    file=paths[0],
                    module=module,
                    package=package_of_file(paths[0]),
                    syntax_error=None,
                    import_diagnostics=(
                        _diag(
                            multifile.AMBIGUOUS_BINDING,
                            "error",
                            paths[0],
                            1,
                            "",
                            "paths " + " and ".join(sorted(paths))
                            + f" both map to module {module}; duplicate conflicting "
                            "spellings are refused rather than merged",
                        ),
                    ),
                )
                continue
            modules[module] = cls._index_one(paths[0], module, sources[paths[0]])
        index = cls(modules)
        index._expand_wildcards()
        return index

    @classmethod
    def _index_one(cls, path: str, module: str, source: str) -> ModuleRecord:
        package = package_of_file(path)
        try:
            tree = ast.parse(source, filename=path)
            too_deep = _deepest_line_beyond(tree, _MAX_AST_DEPTH)
            if too_deep is not None:
                # Recursive AST visitors downstream would hit Python's recursion
                # limit and fail the whole scan; isolate this file like a parse error.
                raise SyntaxError("nesting is too deep to analyze statically", (path, too_deep, 1, None))
        except (SyntaxError, RecursionError, MemoryError) as error:
            if not isinstance(error, SyntaxError):
                error = SyntaxError("nesting is too deep to analyze statically", (path, 1, 1, None))
            # Isolate, never erase: the module index entry exists so the resolver
            # can say exactly WHICH dependency is broken while the rest stay live.
            return ModuleRecord(
                file=path,
                module=module,
                package=package,
                syntax_error={
                    "message": str(error.msg),
                    "line": error.lineno or 1,
                    "column": (error.offset or 1) - 1,
                },
                import_diagnostics=(
                    _diag(
                        multifile.DEPENDENCY_SYNTAX_ERROR,
                        "error",
                        path,
                        error.lineno or 1,
                        "",
                        f"{path} does not parse: {error.msg} (line {error.lineno or 1})",
                    ),
                ),
            )

        classes: Dict[str, ClassInfo] = {}
        bindings: Dict[str, List[Binding]] = {}
        import_diagnostics: List[Dict[str, object]] = []
        dynamic_diagnostics: List[Dict[str, object]] = []
        conditional = False
        functions: List[str] = []

        for statement in tree.body:
            if isinstance(statement, ast.ClassDef):
                classes[statement.name] = ClassInfo(
                    module=module,
                    name=statement.name,
                    file=path,
                    line=statement.lineno,
                    end_line=statement.end_lineno or statement.lineno,
                    column=statement.col_offset,
                    end_column=statement.end_col_offset or statement.col_offset,
                    bases=tuple(ast.unparse(base) for base in statement.bases),
                )
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(statement.name)
            elif isinstance(statement, ast.Import):
                for alias in statement.names:
                    name = alias.asname or alias.name.split(".")[0]
                    target = alias.name if alias.asname else alias.name.split(".")[0]
                    if alias.asname is None and "." in alias.name:
                        # `import a.b.c` binds the top package only; attribute
                        # chains walk modules at resolve time.
                        target = alias.name.split(".")[0]
                    bindings.setdefault(name, []).append(
                        Binding(
                            kind="module",
                            name=name,
                            target_module=target,
                            target_name=None,
                            file=path,
                            line=statement.lineno,
                            import_text=ast.get_source_segment(source, statement) or "",
                        )
                    )
            elif isinstance(statement, ast.ImportFrom):
                base = cls._resolve_relative(statement.module or "", statement.level, package)
                if base is None:
                    import_diagnostics.append(
                        _diag(
                            multifile.UNRESOLVED_IMPORT,
                            "warning",
                            path,
                            statement.lineno,
                            ast.get_source_segment(source, statement) or "",
                            "relative import escapes the workspace root",
                        )
                    )
                    continue
                import_diagnostics.extend(
                    cls._index_from_import(statement, base, path, source, bindings)
                )

        # Anything import-shaped BELOW the top level (inside if/try/def) is
        # conditional as far as this index is concerned: it does not create a
        # binding, but a failing lookup may mention it.
        top_level = set(tree.body)
        for node in ast.walk(tree):
            if node in top_level:
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                conditional = True
                break

        dynamic_diagnostics.extend(cls._dynamic_diagnostics(tree, source, path))
        dynamic_diagnostics.sort(key=lambda d: (int(d["line"]), str(d["code"])))
        names, exports, dynamic_exports, uncertain = cls._export_facts(tree, package)

        return ModuleRecord(
            file=path,
            module=module,
            package=package,
            syntax_error=None,
            classes=classes,
            bindings={name: tuple(items) for name, items in bindings.items()},
            dynamic_diagnostics=tuple(dynamic_diagnostics),
            import_diagnostics=tuple(import_diagnostics),
            has_conditional_imports=conditional,
            functions=tuple(functions),
            bound_names=names,
            explicit_exports=exports,
            exports_dynamic=dynamic_exports,
            uncertain_names=uncertain,
        )

    @staticmethod
    def _export_facts(tree: ast.Module, package=""):
        """Declared namespace only. Never evaluate an export expression or a factory.

        Function/class bodies are separate scopes; their local imports or assignments
        do not change module exports. Control-flow bindings and namespace mutation do.
        A literal __all__ is supported only when it has one untouched assignment.
        """
        def scope_nodes(root):
            # Iterative pre-order walk: deeply nested expressions (long operator
            # chains, generated code) must not hit Python's recursion limit.
            stack = [root]
            while stack:
                node = stack.pop()
                yield node
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    # Defaults, bases and decorators are evaluated in the enclosing
                    # scope, unlike the body. They can mutate an export declaration.
                    evaluated = list(getattr(node, "decorator_list", ()))
                    if isinstance(node, ast.ClassDef):
                        evaluated += list(node.bases) + [keyword.value for keyword in node.keywords]
                    else:
                        evaluated += list(node.args.defaults) + [item for item in node.args.kw_defaults if item is not None]
                        evaluated += [arg.annotation for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                                      if arg.annotation is not None]
                        if getattr(node, "returns", None) is not None:
                            evaluated.append(node.returns)
                    stack.extend(reversed(evaluated))
                    continue
                stack.extend(reversed(list(ast.iter_child_nodes(node))))

        def bound(node):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                return [node.name]
            if isinstance(node, ast.Import):
                return [alias.asname or alias.name.split(".")[0] for alias in node.names]
            if isinstance(node, ast.ImportFrom):
                return [alias.asname or alias.name for alias in node.names if alias.name != "*"]
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                return [node.id]
            return []

        counts: Dict[str, int] = {}
        uncertain = set()
        exports = None
        dynamic = False
        all_assignment = None
        for statement in tree.body:
            nodes = list(scope_nodes(statement))
            names = [name for node in nodes for name in bound(node)]
            for name in names:
                counts[name] = counts.get(name, 0) + 1
            direct = isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef,
                                             ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign))
            if not direct and (names or any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in nodes)):
                dynamic = True
                uncertain.update(names)
            if (isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and statement.decorator_list
                    and not known_decorators(tree, statement, package,
                                             dataclass=isinstance(statement, ast.ClassDef))):
                uncertain.add(statement.name)
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name == "__getattr__":
                dynamic = True
            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id in ("exec", "eval", "globals", "locals", "vars", "setattr", "delattr") for n in nodes):
                dynamic = True
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target] if isinstance(statement, ast.AnnAssign) else []
            if len(targets) == 1 and isinstance(targets[0], ast.Name) and targets[0].id == "__all__":
                value = statement.value
                if all_assignment is None and isinstance(value, (ast.List, ast.Tuple)) and all(
                        isinstance(item, ast.Constant) and isinstance(item.value, str) and item.value.isidentifier() for item in value.elts):
                    all_assignment = statement
                    exports = tuple(dict.fromkeys(item.value for item in value.elts))
                else:
                    dynamic = True
            if statement is not all_assignment and ("__all__" in names or any(isinstance(n, ast.Name) and n.id == "__all__" for n in nodes)):
                dynamic = True
        uncertain.update(name for name, count in counts.items() if count > 1)
        return tuple(sorted(counts)), exports, dynamic, tuple(sorted(uncertain))

    def _expand_wildcards(self) -> None:
        """Materialize only proven export names, preserving all dependency hops.

        Expansion is memoized per request. Cycles, missing sources and export
        computation stay opaque, including a potentially shadowed local binding.
        """
        cache: Dict[str, Tuple[Optional[Tuple[str, ...]], set]] = {}
        targets = {name: tuple(binding.target_module for binding in record.bindings.get("*", ()))
                   for name, record in self.modules.items()}

        def expand(module: str, stack: set):
            if module in cache:
                return cache[module]
            record = self.modules.get(module)
            if record is None:
                return None, set()
            dependencies = {record.file}
            if module in stack or len(stack) >= 64 or record.syntax_error or (
                    record.import_diagnostics and not record.classes and not record.bindings):
                return None, dependencies
            bindings = {name: list(items) for name, items in record.bindings.items() if name != "*"}
            diagnostics = list(record.import_diagnostics)
            unknown = False
            expanded_count = 0
            for star in record.bindings.get("*", ()):
                names, consulted = expand(star.target_module, stack | {module})
                dependencies.update(consulted)
                if names is None or "__all__" in names or expanded_count + len(names) > _MAX_WILDCARD_BINDINGS:
                    unknown = True
                    diagnostics.append(_diag(multifile.UNSUPPORTED_IMPORT_FORM, "warning", star.file,
                        star.line, star.import_text, "wildcard exports cannot be proven from the supplied local sources "
                        "(missing, dynamic, conditional, conflicting, cyclic or over-budget exports); affected names stay unresolved"))
                    continue
                expanded_count += len(names)
                for name in names:
                    bindings.setdefault(name, []).append(replace(star, kind="name", name=name, target_name=name))
            bound = set(record.bound_names) | set(bindings)
            names = record.explicit_exports if record.explicit_exports is not None else tuple(sorted(n for n in bound if not n.startswith("_")))
            if unknown or record.exports_dynamic or len(names) > _MAX_WILDCARD_EXPORTS or any(name not in bound and f"{module}.{name}" not in self.modules for name in names):
                names = None
            if names is not None:
                dependencies.update(self.modules[f"{module}.{name}"].file for name in names
                                    if name not in bound and f"{module}.{name}" in self.modules)
            self.modules[module] = replace(record, bindings={name: tuple(items) for name, items in bindings.items()},
                import_diagnostics=tuple(diagnostics), unresolved_wildcard=unknown,
                wildcard_dependencies=tuple(sorted(dependencies - {record.file})))
            cache[module] = names, dependencies
            return cache[module]

        for module in sorted(self.modules):
            expand(module, set())
        # A DFS cycle cut can return before its entire dependency set is known.
        # Close that set independently so editing any hop can break the cycle and
        # trigger refresh, regardless of which module sorted first.
        for module, record in list(self.modules.items()):
            pending = list(targets[module])
            seen = {module}
            dependencies = set(record.wildcard_dependencies)
            while pending:
                target = pending.pop()
                if target in seen or target not in self.modules:
                    continue
                seen.add(target)
                dependency = self.modules[target]
                dependencies.add(dependency.file)
                dependencies.update(dependency.wildcard_dependencies)
                pending.extend(targets[target])
            self.modules[module] = replace(record, wildcard_dependencies=tuple(sorted(dependencies - {record.file})))

    @classmethod
    def _index_from_import(
        cls,
        statement: ast.ImportFrom,
        base: str,
        path: str,
        source: str,
        bindings: Dict[str, List[Binding]],
    ) -> List[Dict[str, object]]:
        text = ast.get_source_segment(source, statement) or ""
        diagnostics: List[Dict[str, object]] = []
        for alias in statement.names:
            if alias.name == "*":
                bindings.setdefault("*", []).append(Binding("wildcard", "*", base, None, path, statement.lineno, text))
                continue
            name = alias.asname or alias.name
            bindings.setdefault(name, []).append(
                Binding(
                    kind="name",
                    name=name,
                    target_module=base,
                    target_name=alias.name,
                    file=path,
                    line=statement.lineno,
                    import_text=text,
                )
            )
        return diagnostics

    @staticmethod
    def _resolve_relative(base: str, level: int, package: str) -> Optional[str]:
        """Where ``from .x`` points: package, minus ``level - 1`` segments."""
        if level == 0:
            return base
        parts = package.split(".") if package else []
        keep = len(parts) - (level - 1)
        if keep < 0:
            return None
        head = ".".join(parts[:keep])
        return f"{head}.{base}" if head and base else head or base

    @classmethod
    def _dynamic_diagnostics(
        cls, tree: ast.AST, source: str, path: str
    ) -> List[Dict[str, object]]:
        """The computed-import constructs a static resolver may not guess."""
        diagnostics: List[Dict[str, object]] = []

        def line_text(node: ast.AST) -> str:
            segment = ast.get_source_segment(source, node) or ""
            return segment if len(segment) <= 160 else segment[:157] + "…"

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
            ):
                diagnostics.append(
                    _diag(
                        multifile.UNRESOLVED_DYNAMIC,
                        "warning",
                        path,
                        getattr(node, "lineno", 0),
                        line_text(node),
                        "importlib.import_module() computes its target at runtime; "
                        "the classes it returns stay opaque on the static graph",
                    )
                )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and not (isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str))
            ):
                diagnostics.append(
                    _diag(
                        multifile.UNRESOLVED_DYNAMIC,
                        "warning",
                        path,
                        getattr(node, "lineno", 0),
                        line_text(node),
                        "getattr() with a computed name is dynamic; the attribute "
                        "it picks stays unresolved on the static graph",
                    )
                )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("eval", "exec")
            ):
                diagnostics.append(
                    _diag(
                        multifile.UNRESOLVED_DYNAMIC,
                        "warning",
                        path,
                        getattr(node, "lineno", 0),
                        line_text(node),
                        f"{node.func.id}() runs source built at runtime; nothing "
                        "static may follow its result",
                    )
                )
        return diagnostics

    # ── resolution ───────────────────────────────────────────────────────────

    def resolve(
        self,
        from_module: str,
        expr: str,
        *,
        visited: Optional[set] = None,
    ) -> ResolveResult:
        """Resolve a class reference expression (``Name`` or dotted attribute
        path) inside ``from_module`` to a definition, or say exactly why not.

        Deterministic: same (index, from_module, expr) always yields the same
        result, and diagnostics carry the import statement's own text.

        ``visited``, when given, collects the FILE of every module record the
        resolution consults — including re-export hops and failure targets — so
        the projection can tell the host exactly which sources an answer (or a
        refusal) depended on.
        """
        record = self.modules.get(from_module)
        if record is None:
            return ResolveResult(
                None,
                _diag(
                    multifile.UNRESOLVED_IMPORT,
                    "warning",
                    "",
                    0,
                    "",
                    f"module {from_module} is not in the source set",
                ),
            )
        if visited is not None:
            visited.add(record.file)
        if record.syntax_error is not None:
            return ResolveResult(
                None,
                _diag(
                    multifile.DEPENDENCY_SYNTAX_ERROR,
                    "error",
                    record.file,
                    int(record.syntax_error["line"]),
                    "",
                    f"{record.file} does not parse: {record.syntax_error['message']}",
                ),
            )

        parts = expr.split(".")
        head = parts[0]
        start = self._resolve_name(record, head, set(), visited)
        if isinstance(start, ResolveResult):
            return start
        # start: ClassInfo (only valid alone) or a module name string
        if isinstance(start, ClassInfo):
            if len(parts) > 1:
                return ResolveResult(
                    None,
                    _diag(
                        multifile.UNRESOLVED_IMPORT,
                        "warning",
                        record.file,
                        0,
                        "",
                        f"attribute access on {head} (a class) is not a supported "
                        "binding target",
                    ),
                )
            return ResolveResult(start, None)

        module = start  # str
        for position, part in enumerate(parts[1:], start=1):
            found = self._resolve_class_in_module(module, part, visited)
            if isinstance(found, ClassInfo):
                if position != len(parts) - 1:
                    return ResolveResult(
                        None,
                        _diag(
                            multifile.UNRESOLVED_IMPORT,
                            "warning",
                            record.file,
                            0,
                            "",
                            f"{part} is a class, but {expr} continues past it",
                        ),
                    )
                return ResolveResult(found, None)
            if isinstance(found, str):
                module = found
                continue
            # Preserve the failure and its dependency location, as for bare names.
            assert isinstance(found, ResolveResult)
            return found
        return ResolveResult(
            None,
            _diag(
                multifile.UNRESOLVED_IMPORT,
                "warning",
                record.file,
                0,
                "",
                f"{expr} resolves to module {module}, not a class",
            ),
        )

    def _resolve_name(
        self, record: ModuleRecord, name: str, seen: set, visited: Optional[set] = None
    ) -> ClassInfo | str | ResolveResult:
        """Resolve one bare name in a module to a ClassInfo or a module name,
        following unambiguous re-export chains. Returns ResolveResult on failure.
        """
        if visited is not None:
            visited.add(record.file)
            visited.update(record.wildcard_dependencies)
        candidates = list(record.bindings.get(name, ()))
        local = record.classes.get(name)
        if record.unresolved_wildcard or (name in record.uncertain_names and (local is not None or candidates or name in record.functions) and len(candidates) < 2
                and not ((local is not None or name in record.functions) and candidates)):
            return ResolveResult(None, _diag(multifile.AMBIGUOUS_BINDING, "warning", record.file, 0, "",
                f"{name} may be rebound by an unresolved wildcard or another declaration in {record.module}; "
                "its class definition cannot be chosen statically"))
        if (local is not None or name in record.functions) and candidates:
            return ResolveResult(
                None,
                _diag(
                    multifile.AMBIGUOUS_BINDING,
                    "warning",
                    record.file,
                    candidates[0].line,
                    candidates[0].import_text,
                    f"{name} is both defined in {record.module} and imported there; "
                    "the binding is ambiguous and will not be guessed",
                ),
            )
        if local is not None:
            return local
        if name in record.functions and not candidates:
            return ResolveResult(
                None,
                _diag(multifile.UNRESOLVED_IMPORT, "warning", record.file, 0, "",
                      f"{name} is a function in {record.module}, not a class"),
                non_class=True,
            )
        if not candidates:
            return ResolveResult(
                None,
                _diag(
                    multifile.UNRESOLVED_IMPORT,
                    "warning",
                    record.file,
                    0,
                    "",
                    f"no import or local definition binds {name} in {record.module}"
                    + (
                        " (conditional/inner-function imports are not static bindings)"
                        if record.has_conditional_imports
                        else ""
                    ),
                ),
            )
        if len(candidates) > 1:
            targets = ", ".join(
                f"{c.target_module}.{c.target_name or ''}".rstrip(".") for c in candidates
            )
            return ResolveResult(
                None,
                _diag(
                    multifile.AMBIGUOUS_BINDING,
                    "warning",
                    record.file,
                    candidates[0].line,
                    candidates[0].import_text,
                    f"{name} is imported from more than one module ({targets})",
                ),
            )
        binding = candidates[0]
        mark = (record.module, name)
        if mark in seen:
            return ResolveResult(
                None,
                _diag(
                    multifile.CYCLE_BOUNDARY,
                    "warning",
                    binding.file,
                    binding.line,
                    binding.import_text,
                    f"re-export chain for {name} cycles back to {record.module}; "
                    "cut here rather than recursed",
                ),
            )
        seen.add(mark)

        if binding.kind == "module":
            target = binding.target_module
            if target in self.modules:
                return target
            return self._missing_module(record, binding, target)

        # kind == "name": chase the import target.
        return self._resolve_import_target(record, binding, seen, visited)

    def _resolve_import_target(
        self,
        record: ModuleRecord,
        binding: Binding,
        seen: set,
        visited: Optional[set] = None,
    ) -> ClassInfo | str | ResolveResult:
        target_module = binding.target_module
        target_name = binding.target_name or ""
        # Submodule reference: `from pkg import blocks` where pkg/blocks.py is in
        # the set.
        if f"{target_module}.{target_name}" in self.modules:
            return f"{target_module}.{target_name}"
        target_record = self.modules.get(target_module)
        if target_record is None:
            return self._missing_module(record, binding, target_module)
        if visited is not None:
            visited.add(target_record.file)
        if target_record.syntax_error is not None:
            return ResolveResult(
                None,
                _diag(
                    multifile.DEPENDENCY_SYNTAX_ERROR,
                    "error",
                    target_record.file,
                    int(target_record.syntax_error["line"]),
                    binding.import_text,
                    f"import target {target_record.file} does not parse: "
                    f"{target_record.syntax_error['message']}",
                ),
            )
        if target_record.import_diagnostics and not target_record.classes and not target_record.bindings:
            # e.g. the ambiguous-spellings placeholder
            first = target_record.import_diagnostics[0]
            return ResolveResult(None, dict(first))
        return self._resolve_name(target_record, target_name, seen, visited)

    def _resolve_class_in_module(
        self, module: str, name: str, visited: Optional[set] = None
    ) -> ClassInfo | str | ResolveResult:
        """Attribute access: a class, a submodule, or ResolveResult on failure."""
        record = self.modules.get(module)
        if f"{module}.{name}" in self.modules:
            return f"{module}.{name}"
        if record is None:
            return ResolveResult(
                None,
                _diag(
                    multifile.UNRESOLVED_IMPORT,
                    "warning",
                    "",
                    0,
                    "",
                    f"module {module} is not in the source set",
                ),
            )
        if visited is not None:
            visited.add(record.file)
        if record.syntax_error is not None:
            return ResolveResult(
                None,
                _diag(
                    multifile.DEPENDENCY_SYNTAX_ERROR,
                    "error",
                    record.file,
                    int(record.syntax_error["line"]),
                    "",
                    f"{record.file} does not parse: {record.syntax_error['message']}",
                ),
            )
        if record.import_diagnostics and not record.classes and not record.bindings:
            return ResolveResult(None, dict(record.import_diagnostics[0]))
        return self._resolve_name(record, name, set(), visited)

    #: The framework this product is about. Its absence from the source set is not a
    #: gap in the user's workspace — nobody vendors torch into their repo — so it is
    #: neither of the two cases below.
    _FRAMEWORK_ROOTS = ("torch",)

    def _missing_module(self, record: ModuleRecord, binding: Binding, module: str) -> ResolveResult:
        """Distinguish the framework, a not-supplied workspace module, and a package."""
        top = module.split(".")[0]
        workspace_ish = any(
            m == top or m.startswith(top + ".") for m in self.modules
        )
        if top in self._FRAMEWORK_ROOTS and not workspace_ish:
            # A torch class with no op-registry spec stays as one card because
            # nothing can infer a shape THROUGH it, not because torch is missing.
            # Saying "external to the workspace" about the framework reads as "your
            # torch.nn is unavailable", which is the same misreading the _SOURCE_OPS
            # branch in crossfile exists to stop for `nn.Parameter`.
            return ResolveResult(
                None,
                _diag(
                    multifile.EXTERNAL_PACKAGE,
                    "info",
                    binding.file,
                    binding.line,
                    binding.import_text,
                    f"{module} is the framework, not a workspace module: a class the op "
                    "registry has no shape spec for stays as one opaque card on the static "
                    "graph. Static analysis cannot inspect its contents or infer its shapes.",
                ),
            )
        if workspace_ish:
            code = multifile.UNRESOLVED_IMPORT
            severity = "warning"
            detail = (
                f"{module} is a workspace module but is not in the supplied source "
                "set (truncated set or missing file)"
            )
        else:
            code = multifile.EXTERNAL_PACKAGE
            severity = "info"
            detail = (
                f"{module} is external to the workspace; its classes stay opaque "
                "on the static graph unless they are canonical framework operations"
            )
        return ResolveResult(
            None,
            _diag(code, severity, binding.file, binding.line, binding.import_text, detail),
        )
