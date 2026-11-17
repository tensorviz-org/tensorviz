"""tensorviz — declare what the indexer cannot infer honestly.

Local example SDK (see the SDK README). Three calls, all static-first: the
tensorViz indexer reads them from your source without running it, so they work
in a file that is never executed under the IDE. At runtime they record the declarations in-process and otherwise do nothing — no network, no
files, no torch import.

    import tensorviz

    tensorviz.experiment("wide-net")                 # this file is one experiment
    tensorviz.node("dataset", "raw-logs")            # a pipeline node the code hides
    tensorviz.node("transform", "tokenise")
    tensorviz.edge("raw-logs", "tokenise", "feeds")  # and how data flows between them

Arguments must be string literals: the indexer reads the AST, and a computed
label is exactly the kind of thing it cannot see. An explicit ``experiment``
declaration overrides automatic entrypoint discovery for its file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

__all__ = ["experiment", "node", "edge", "declarations", "reset", "NodeKind", "Relationship"]

__version__ = "0.1.0"

NodeKind = Literal[
    "project", "file", "function", "dataset", "transform", "model", "component",
    "trainer", "evaluator", "metric", "artifact", "checkpoint", "configuration", "generic",
]
Relationship = Literal[
    "contains", "imports", "calls", "reads", "transforms", "feeds", "trains",
    "evaluates", "produces", "logs", "runs_on", "derives_from",
]


@dataclass
class Declarations:
    experiment: Optional[str] = None
    nodes: List[tuple] = field(default_factory=list)
    edges: List[tuple] = field(default_factory=list)


_current = Declarations()


def experiment(name: str) -> str:
    """Name the experiment this file is. Returns ``name`` so it can be reused."""
    if not isinstance(name, str) or not name:
        raise TypeError("tensorviz.experiment(name) needs a non-empty string literal")
    _current.experiment = name
    return name


def node(kind: str, label: str, **params: object) -> str:
    """Declare a pipeline node. Returns ``label``, the handle ``edge`` uses."""
    if not isinstance(kind, str) or not isinstance(label, str) or not label:
        raise TypeError("tensorviz.node(kind, label) needs string literals")
    _current.nodes.append((kind, label, dict(params)))
    return label


def edge(from_label: str, to_label: str, relationship: str = "feeds") -> None:
    """Declare how data flows between two declared nodes."""
    for value in (from_label, to_label, relationship):
        if not isinstance(value, str) or not value:
            raise TypeError("tensorviz.edge(from, to, relationship) needs string literals")
    _current.edges.append((from_label, to_label, relationship))


def declarations() -> Declarations:
    """What this process has declared so far. For tests and future collectors."""
    return _current


def reset() -> None:
    global _current
    _current = Declarations()
