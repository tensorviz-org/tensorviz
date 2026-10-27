"""Turning a graph action into an SDK declaration, as a patch the host applies.

This is how a guess stops being a guess. The indexer infers that `make_loader`
feeds `TinyLM`; the user confirms it; this writes
`tensorviz.edge("make_loader", "TinyLM", "feeds")` into their entrypoint. The next
index reads that as `declared`, which outranks every rule, and nothing guesses
there again. See "Library code and user code" in `docs/project-graph-plan.md`.

**This module never writes a file.** It returns the new contents and the range to
replace, exactly as `services/ide/code_apply.py` does, because the extension is what
writes — through a native diff the user accepts or discards. A module that wrote
directly would bypass trust, undo, and the user's eyes at once.

Two disciplines it shares with `code_apply`:

* **Refusals are the feature.** A declaration naming a node that does not exist, a
  label that cannot be a string literal, or a file that does not parse is refused
  with a reason rather than written and left to fail at the next index.
* **The smallest possible patch.** Declarations are appended to the existing block
  of `tensorviz.*` calls when there is one, so a file does not grow a second cluster
  of them and a diff stays one line.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

from research.contracts import NodeKind, Relationship


class CodegenError(ValueError):
    """A declaration that would be wrong, or unwritable. Refused, never guessed at."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class DeclareNode:
    kind: str
    label: str


@dataclass(frozen=True)
class DeclareEdge:
    from_label: str
    to_label: str
    relationship: str = "feeds"


@dataclass(frozen=True)
class DeclareExperiment:
    name: str


Declaration = object  # one of the three above


@dataclass(frozen=True)
class SourceEdit:
    """What the host applies: whole new contents, plus the digest it was built on."""

    path: str
    contents: str
    #: sha256 of the source this was computed from. The host refuses to apply an
    #: edit whose base no longer matches, the same check `/api/ide/apply` makes.
    expected_digest: str
    #: 1-based line the statement landed on, so the diff can be revealed there.
    line: int
    statement: str


VALID_KINDS: Set[str] = set(NodeKind.__args__)  # type: ignore[attr-defined]
VALID_RELATIONSHIPS: Set[str] = set(Relationship.__args__)  # type: ignore[attr-defined]


def _quote(value: str) -> str:
    """A Python string literal, or a refusal.

    Refuses rather than escapes: a label containing a newline or both quote styles
    would produce a statement the user cannot read at a glance, and an unreadable
    declaration is worse than none.
    """
    if not value or not value.strip():
        raise CodegenError("empty_label", "a declaration needs a non-empty label")
    if "\n" in value or ("'" in value and '"' in value):
        raise CodegenError(
            "unwritable_label", f"{value!r} cannot be written as a readable string literal"
        )
    return repr(value)


def statement_for(declaration: Declaration) -> str:
    """The one line this declaration becomes."""
    if isinstance(declaration, DeclareNode):
        if declaration.kind not in VALID_KINDS:
            raise CodegenError(
                "unknown_kind",
                f"{declaration.kind!r} is not a node kind; expected one of {sorted(VALID_KINDS)}",
            )
        return f"tensorviz.node({_quote(declaration.kind)}, {_quote(declaration.label)})"
    if isinstance(declaration, DeclareEdge):
        if declaration.relationship not in VALID_RELATIONSHIPS:
            raise CodegenError(
                "unknown_relationship",
                f"{declaration.relationship!r} is not a relationship",
            )
        return (
            f"tensorviz.edge({_quote(declaration.from_label)}, "
            f"{_quote(declaration.to_label)}, {_quote(declaration.relationship)})"
        )
    if isinstance(declaration, DeclareExperiment):
        return f"tensorviz.experiment({_quote(declaration.name)})"
    raise CodegenError("unknown_declaration", f"cannot write {declaration!r}")


def is_tensorviz_call(stmt: ast.stmt) -> bool:
    """A bare `tensorviz.<something>(...)` statement at module level."""
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return False
    func = stmt.value.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "tensorviz"
    )


def _imports_tensorviz(tree: ast.Module) -> bool:
    return any(
        isinstance(stmt, ast.Import)
        and any(alias.name == "tensorviz" and alias.asname is None for alias in stmt.names)
        for stmt in tree.body
    )


def _parse(path: str, source: str) -> ast.Module:
    try:
        return ast.parse(source)
    except SyntaxError as error:
        # Writing into a file that does not parse would produce a second error on
        # top of the one the user is already looking at.
        raise CodegenError(
            "does_not_parse", f"{path} does not parse: {error.msg} (line {error.lineno})"
        ) from error


def _digest(source: str) -> str:
    import hashlib

    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def declare(
    path: str,
    source: str,
    declaration: Declaration,
    *,
    known_labels: Optional[Sequence[str]] = None,
) -> SourceEdit:
    """Add a declaration, and say where it went.

    `known_labels`, when given, is every label the current graph has. An edge naming
    something absent is refused: the contract rejects a dangling edge anyway, so
    writing one would produce source that makes the next index fail validation —
    a failure the user would have to debug rather than simply not being allowed to
    make.
    """
    tree = _parse(path, source)
    statement = statement_for(declaration)

    if isinstance(declaration, DeclareEdge) and known_labels is not None:
        present = set(known_labels)
        for endpoint in (declaration.from_label, declaration.to_label):
            if endpoint not in present:
                raise CodegenError(
                    "unknown_endpoint",
                    f"{endpoint!r} is not a node in this experiment",
                )

    if any(
        is_tensorviz_call(stmt) and _statement_text(stmt, source) == statement
        for stmt in tree.body
    ):
        raise CodegenError("already_declared", "this declaration is already in the file")

    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    existing = [stmt for stmt in tree.body if is_tensorviz_call(stmt)]
    needs_import = not _imports_tensorviz(tree)

    if existing:
        # Join the block that is already there, so a file never grows a second
        # cluster of declarations and the diff stays one line.
        after = existing[-1].end_lineno or existing[-1].lineno
        block: List[str] = [statement + "\n"]
    else:
        after = _after_header(tree)
        block = []
        if needs_import:
            block.append("import tensorviz\n")
            block.append("\n")
        block.append(statement + "\n")
        if after < len(lines):
            block.append("\n")

    new_lines = lines[:after] + block + lines[after:]
    line = after + len(block) - (1 if block[-1] == "\n" else 0)
    return SourceEdit(
        path=path,
        contents="".join(new_lines),
        expected_digest=_digest(source),
        line=line,
        statement=statement,
    )


def undeclare(path: str, source: str, statement: str) -> SourceEdit:
    """Remove one declaration, leaving everything else byte-identical.

    The counterpart to `declare`, and the reason a confirmed guess is not a one-way
    door: a user who accepts a suggestion and then disagrees can take it back the
    same way they added it.
    """
    tree = _parse(path, source)
    target = None
    for stmt in tree.body:
        if is_tensorviz_call(stmt) and _statement_text(stmt, source) == statement.strip():
            target = stmt
            break
    if target is None:
        raise CodegenError("not_declared", f"{statement!r} is not in {path}")

    lines = source.splitlines(keepends=True)
    start = target.lineno - 1
    end = target.end_lineno or target.lineno
    new_lines = lines[:start] + lines[end:]
    return SourceEdit(
        path=path,
        contents="".join(new_lines),
        expected_digest=_digest(source),
        line=target.lineno,
        statement=statement,
    )


def _statement_text(stmt: ast.stmt, source: str) -> str:
    lines = source.splitlines()
    start = stmt.lineno - 1
    end = stmt.end_lineno or stmt.lineno
    return "\n".join(lines[start:end]).strip()


def _after_header(tree: ast.Module) -> int:
    """The line to insert after: below the imports, or below a module docstring.

    Declarations belong with the imports rather than at the very top, because a
    module docstring is the first thing a reader sees and should stay that way.
    """
    imports = [s for s in tree.body if isinstance(s, (ast.Import, ast.ImportFrom))]
    if imports:
        return imports[-1].end_lineno or imports[-1].lineno
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(getattr(tree.body[0], "value", None), ast.Constant)
        and isinstance(tree.body[0].value.value, str)  # type: ignore[attr-defined]
    ):
        return tree.body[0].end_lineno or tree.body[0].lineno
    return 0
