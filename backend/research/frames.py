"""A function body as a dataflow graph: what it computes, and in what order.

The card for `LanguageModelTrainer` says "calls .backward() and an optimizer
.step()". True, and it is the whole of what the project graph knows about the one
place the training actually happens. Everything interesting is inside:

    logits = self.model(inputs)
    loss = F.cross_entropy(logits.reshape(...), targets.reshape(...))
    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    self.optimizer.step()

That is a pipeline — `inputs -> logits -> loss -> backward` — and it is invisible
because a function is drawn as one box.

**Variables are the edges.** A step that reads a name an earlier step wrote is
downstream of it. This is not a guess about intent: it is what the code does, and
Python's own scoping rules say which write a read refers to.

Pure AST, no imports, no execution — the same contract as the rest of `research/`.

What is deliberately NOT modelled: branches and loops as control flow. A `for` body
is flattened into the sequence, because a reader looking at a training loop wants
the pipeline, not a CFG — and a diagram that draws the back edge of every loop is a
diagram of Python, not of the model.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

#: Statements past this are not drawn. A 400-line `main` is a file to open, not a
#: frame to read, and the count is reported so the card can say so.
MAX_STEPS = 24

#: Names that carry no information as an edge: every step in a method reads `self`,
#: so an edge for it joins everything to everything.
IGNORED_READS = {"self", "cls", "super", "print", "len", "range", "enumerate", "zip"}

#: Calls that are not steps. Logging and assertions are things the code does ABOUT
#: the pipeline rather than parts of it, and a `print` per step doubles the frame
#: while saying nothing about what flows where.
IGNORED_CALLS = {"print", "assert", "log", "logger.info", "logger.debug", "tqdm", "time.time"}


@dataclass
class Step:
    """One statement of a function body, as the frame draws it."""

    id: str
    #: What the statement DOES, not what it says: `F.cross_entropy`, not the whole
    #: expression. The line is on the card for anyone who wants the rest.
    label: str
    lineno: int
    end_lineno: int
    #: The name this step assigns, if any. What later steps can read.
    writes: Optional[str] = None
    #: Names it consumes that an earlier step in this body wrote.
    reads: Set[str] = field(default_factory=set)
    #: True when the step is the gradient step or the optimizer step — the two that
    #: make a loop a training loop, and the two a reader looks for first.
    role: str = ""


def _call_name(node: ast.AST) -> Optional[str]:
    """`F.cross_entropy(...)` -> `F.cross_entropy`; `self.model(x)` -> `self.model`."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    parts: List[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    elif parts:
        pass
    else:
        return None
    return ".".join(reversed(parts))


def _reads(node: ast.AST) -> Set[str]:
    """Every plain name and attribute root the expression loads."""
    found: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            found.add(child.id)
        elif isinstance(child, ast.Attribute):
            root = child
            while isinstance(root, ast.Attribute):
                root = root.value  # type: ignore[assignment]
            if isinstance(root, ast.Name):
                found.add(root.id)
    return found - IGNORED_READS


def _target(stmt: ast.AST) -> Optional[str]:
    """The single name an assignment binds, or None for anything else."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id
    return None


def _role_of(label: str) -> str:
    tail = label.rsplit(".", 1)[-1]
    if tail == "backward":
        return "gradient"
    if tail in {"step", "zero_grad"}:
        return "optimizer"
    return ""


def _flatten(body: List[ast.stmt]) -> List[ast.stmt]:
    """Statements in source order, with loop and branch bodies inlined.

    A reader looking at a training loop wants the pipeline. Drawing the back edge of
    every `for` would be a diagram of Python rather than of the model.
    """
    out: List[ast.stmt] = []
    for stmt in body:
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try)):
            out.extend(_flatten(getattr(stmt, "body", [])))
            out.extend(_flatten(getattr(stmt, "orelse", [])))
        else:
            out.append(stmt)
    return out


def steps_of(source: str, qualname: str) -> List[Step]:
    """The steps of `qualname`'s body, in order, with their dataflow.

    `qualname` is `train` for a module-level function and `Trainer.train` for a
    method — the same shape the indexer names definitions by.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    target = _find(tree, qualname.split("."))
    if target is None:
        return []
    if isinstance(target, ast.ClassDef):
        target = _training_method(target)
        if target is None:
            return []

    steps: List[Step] = []
    written: Dict[str, str] = {}
    for index, stmt in enumerate(_flatten(target.body)):
        value = getattr(stmt, "value", None)
        if value is None:
            continue
        label = _call_name(value)
        if label is not None and (label in IGNORED_CALLS or label.rsplit(".", 1)[-1] in {"info", "debug", "warning"}):
            continue
        if label is None:
            # Not a call: an assignment from an expression. Worth a step only when it
            # names something later steps use, and named for what it binds.
            name = _target(stmt)
            if name is None:
                continue
            label = name
        writes = _target(stmt)
        reads = {name for name in _reads(value) if name in written}
        step = Step(
            id=f"s{index}",
            label=label,
            lineno=getattr(stmt, "lineno", 0),
            end_lineno=getattr(stmt, "end_lineno", 0) or getattr(stmt, "lineno", 0),
            writes=writes,
            reads=reads,
            role=_role_of(label),
        )
        steps.append(step)
        if writes:
            # Later reads of this name resolve to THIS step, so a name reassigned in
            # a loop chains through its own history rather than fanning out from the
            # first write.
            written[writes] = step.id
        if len(steps) >= MAX_STEPS:
            break
    _resolve(steps, written)
    return steps


def _resolve(steps: List[Step], _written: Dict[str, str]) -> None:
    """Turn each step's read NAMES into the ids of the steps that wrote them.

    Resolved against the most recent earlier write, which is what Python does.
    """
    latest: Dict[str, str] = {}
    for step in steps:
        step.reads = {latest[name] for name in step.reads if name in latest}
        if step.writes:
            latest[step.writes] = step.id


def _find(tree: ast.Module, parts: List[str]) -> Optional[ast.AST]:
    """The definition `parts` names, e.g. ['Trainer', 'train']."""
    scope: List[ast.stmt] = tree.body
    found: Optional[ast.AST] = None
    for part in parts:
        found = None
        for stmt in scope:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and stmt.name == part:
                found = stmt
                scope = stmt.body
                break
        if found is None:
            return None
    return found if isinstance(found, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else None


#: Method names that hold a loop, in the order a reader would look for one.
LOOP_METHODS = ("train", "fit", "run", "forward", "step", "evaluate", "__call__")


def _training_method(cls: ast.ClassDef) -> Optional[ast.AST]:
    """The method of `cls` whose body is worth drawing.

    A trainer is usually a CLASS — `LanguageModelTrainer`, `SFTTrainer` — and a class
    has no statements of its own worth a frame: its `__init__` assigns fields and the
    loop is somewhere else. The method that calls `.backward()` is the loop, and that
    is evidence rather than a naming guess; a name is the fallback.
    """
    methods = [
        stmt for stmt in cls.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for method in methods:
        for child in ast.walk(method):
            if isinstance(child, ast.Attribute) and child.attr == "backward":
                return method
    by_name = {method.name: method for method in methods}
    for name in LOOP_METHODS:
        if name in by_name:
            return by_name[name]
    return None
