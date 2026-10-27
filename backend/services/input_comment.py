"""Class-scoped ``# tensorviz: input=`` comments.

Torch-free on purpose. The capture worker and the drawing parser both read this
module, and the worker cannot import ``utils.code_parser`` (that pulls
``models``, which a user repository shadows).
"""
from __future__ import annotations

import ast
import re
from typing import List, Optional

INPUT_COMMENT = re.compile(r"#\s*tensorviz:\s*input\s*=\s*(?P<value>[^\n]+)")

_PLAIN = re.compile(r"\d+(\s*,\s*\d+)*")


def _line(source: str, match: re.Match) -> int:
    return source.count("\n", 0, match.start()) + 1


def comment_for_class(source: str, cls: Optional[str]) -> Optional[re.Match]:
    """The input comment that belongs to ``cls``.

    A comment inside a class body belongs to that class. A comment that sits
    before a class — including one immediately above it — belongs to the class
    that follows it, and to no other class. A comment with no following class
    stays the file-wide default used by classes that have no comment of their
    own (the trailing ``# tensorviz: input=`` fixtures).

    ``cls`` omitted returns the first comment in the file. A buffer that does
    not parse falls back to that same first comment so a mid-edit file does not
    lose the hint.
    """
    text = source or ""
    matches = list(INPUT_COMMENT.finditer(text))
    if not matches or cls is None:
        return matches[0] if matches else None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return matches[0]
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]

    def inside(node: ast.ClassDef, line: int) -> bool:
        return node.lineno <= line <= (node.end_lineno or node.lineno)

    target = next((node for node in classes if node.name == cls), None)
    if target is not None:
        for match in matches:
            if inside(target, _line(text, match)):
                return match
        preceding = None
        for match in matches:
            line = _line(text, match)
            if line >= target.lineno:
                break
            if any(inside(node, line) for node in classes):
                continue
            following = next((node for node in classes if node.lineno > line), None)
            if following is target:
                preceding = match
        if preceding is not None:
            return preceding
    for match in matches:
        line = _line(text, match)
        if any(inside(node, line) for node in classes):
            continue
        if any(node.lineno > line for node in classes):
            continue
        return match
    return None


def _ints(node: ast.AST) -> Optional[List[int]]:
    if isinstance(node, ast.Constant) and type(node.value) is int and node.value > 0:
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List)):
        dims: List[int] = []
        for item in node.elts:
            if not isinstance(item, ast.Constant) or type(item.value) is not int or item.value <= 0:
                return None
            dims.append(item.value)
        return dims or None
    return None


def _call_shape(expr: ast.AST) -> Optional[List[int]]:
    """Read a shape from ``torch.randn(2, 8)`` or ``torch.full((2, 8), 0)`` without evaluating it."""
    if not isinstance(expr, ast.Call):
        return None
    name = ast.unparse(expr.func)
    if name in {"torch.randn", "torch.rand", "torch.ones", "torch.zeros", "torch.empty"}:
        sized = next((item.value for item in expr.keywords if item.arg == "size"), None)
        if sized is not None:
            return _ints(sized)
        if len(expr.args) == 1 and isinstance(expr.args[0], (ast.Tuple, ast.List)):
            return _ints(expr.args[0])
        dims: List[int] = []
        for arg in expr.args:
            if not isinstance(arg, ast.Constant) or type(arg.value) is not int or arg.value <= 0:
                return None
            dims.append(arg.value)
        return dims or None
    if name == "torch.full" and expr.args:
        return _ints(expr.args[0])
    if name == "torch.randint" and len(expr.args) >= 3:
        return _ints(expr.args[2])
    return None


def plain_shapes(value: str) -> Optional[List[List[int]]]:
    """Every argument of an input comment, when each one is a static shape.

    ``None`` means the comment is not a list of shapes (an arbitrary expression).
    Callers must not guess a tensor over that comment.
    """
    parts = [part.strip() for part in (value or "").split(";") if part.strip()]
    if not parts:
        return None
    shapes: List[List[int]] = []
    for part in parts:
        if _PLAIN.fullmatch(part):
            shapes.append([int(dim) for dim in part.split(",")])
            continue
        try:
            expr = ast.parse(part, mode="eval").body
        except SyntaxError:
            return None
        shape = _call_shape(expr)
        if not shape:
            return None
        shapes.append(shape)
    return shapes


def declared_input_shape(source: str, cls: Optional[str] = None) -> Optional[List[int]]:
    """The first comma-separated shape a comment declares, or ``None``.

    ``cls`` scopes the comment to that class. Only the plain ``2,8`` form is
    read. ``torch.randint(...)`` needs the capture worker, which has torch;
    this function refuses it rather than half-reading the line.
    """
    match = comment_for_class(source, cls)
    if match is None:
        return None
    first = match.group("value").split(";")[0].strip()
    if not _PLAIN.fullmatch(first):
        return None
    dims = [int(part) for part in first.split(",")]
    return dims or None


def declared_example(source: str, cls: Optional[str]) -> Optional[tuple]:
    """``(shapes, line)`` for ``cls``.

    ``shapes`` is ``None`` when a comment exists but is not a static shape list.
    The whole result is ``None`` when this class declares nothing.
    """
    match = comment_for_class(source, cls)
    if match is None:
        return None
    return plain_shapes(match.group("value")), _line(source or "", match)
