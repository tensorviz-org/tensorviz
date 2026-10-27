"""Stable ids for project-graph nodes and edges.

Every function here is pure and, critically, **contains no line numbers**. That is
not a style preference: it is what makes the realtime loop possible. A keystroke
that pushes a class forty lines down must leave its node id untouched, or every
re-index reads as "delete everything, add everything back" and the canvas strobes,
loses the user's layout, and forgets which frames were open.

Four layers, narrowest to widest:

  definition   `models/net.py::Net`   — a class or function, wherever it now sits
  experiment   `exp:3f9a…`           — one entrypoint plus its configuration
  node         `node:8c1e…`          — that definition, as used by that experiment
  edge         `edge:77b2…`          — a relationship between two nodes

The separation matters at the seams. Two experiments that both import `Net` share a
`definition_id` and differ in node id, which is exactly what "editing a shared file
highlights every dependent experiment" reads from. Two *instances* inside one
experiment differ by their attribute path. And a directory is never identity — a
project that moves from `demo/` to `examples/` keeps its experiment id as long as
its entrypoint and config do.
"""

from __future__ import annotations

import hashlib
import posixpath
from typing import Optional, Sequence


def normalise_path(rel_file: str) -> str:
    """Workspace-relative POSIX path, with no leading `./` and no backslashes.

    Refuses rather than repairs an empty path: a node whose file we cannot name is
    a node we cannot navigate to, and a silent `""` would surface much later as an
    unexplainable click that goes nowhere.
    """
    if not rel_file or not rel_file.strip():
        raise ValueError("a path is required")
    text = rel_file.replace("\\", "/").strip()
    text = posixpath.normpath(text)
    if text.startswith("./"):
        text = text[2:]
    return text


def _short(payload: str, prefix: str, width: int = 16) -> str:
    return f"{prefix}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:width]}"


def definition_id(rel_file: str, qualname: str) -> str:
    """Where a class or function is *defined*, independent of where it now sits.

    Readable on purpose — `models/net.py::Net` shows up in diffs, logs and test
    failures, and a hash there would cost more than it saved.
    """
    if not qualname or not qualname.strip():
        raise ValueError("a qualified name is required")
    return f"{normalise_path(rel_file)}::{qualname.strip()}"


def experiment_id(entrypoint_definition: str, config_digest: str = "") -> str:
    """One experiment: an entrypoint, plus the configuration that distinguishes runs.

    `config_digest` is what separates two experiments sharing one entrypoint — the
    same `train.py` invoked with different overrides. Empty when nothing
    distinguishes them.
    """
    if not entrypoint_definition or not entrypoint_definition.strip():
        raise ValueError("an entrypoint definition is required")
    return _short(f"{entrypoint_definition.strip()}\0{config_digest}", "exp")


def experiment_node_id(
    experiment: str, definition: str, instance_path: str = ""
) -> str:
    """One definition, as used by one experiment, at one attribute path.

    The experiment is part of the hash so two experiments never collide on a shared
    class; `instance_path` separates `enc1` from `enc2` inside a single one.
    """
    if not experiment or not definition:
        raise ValueError("experiment and definition are required")
    return _short(f"{experiment}\0{definition}\0{instance_path}", "node")


def edge_id(from_node_id: str, to_node_id: str, relationship: str) -> str:
    """Derived from its endpoints, so it moves when they do and never independently."""
    if not from_node_id or not to_node_id or not relationship:
        raise ValueError("both endpoints and a relationship are required")
    return _short(f"{from_node_id}\0{to_node_id}\0{relationship}", "edge")


def declared_experiment_id(name: str) -> str:
    """An SDK `tensorviz.experiment("name")` supplies its own identity.

    Namespaced under `sdk::` so a declared experiment and an inferred one that
    happen to share a name never collide, and so a declaration keeps its id when
    the file it sits in is renamed.
    """
    if not name or not name.strip():
        raise ValueError("a non-empty experiment name is required")
    return experiment_id(f"sdk::{name.strip()}")


def config_digest(values: Sequence[str]) -> str:
    """A stable digest over configuration strings, order-independent."""
    joined = "\0".join(sorted(v for v in values if v))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16] if joined else ""


def graph_id_for(root: str, entrypoint: str) -> Optional[str]:
    """The graph id for one experiment, derived from its root and entrypoint.

    Derived rather than stored, the same way `services/ide/projection.py` derives
    a per-class graph id, so two windows indexing one workspace name the same graph
    instead of minting rivals.
    """
    if not entrypoint:
        return None
    scope = f"{normalise_path(root)}\0{normalise_path(entrypoint)}" if root else normalise_path(entrypoint)
    return _short(scope, "graph")
