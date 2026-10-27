"""The graph as text: Mermaid, from the same document the canvas draws.

Three reasons this is worth having, and only one of them is the feature request.

**A graph in a README.** A research repository's diagram is out of date the day
after it is drawn. This one is generated from the source, so it is wrong only when
the code is.

**A graph in a bug report.** "The canvas looks wrong" is a screenshot; this is the
graph, in text, diffable.

**A graph anyone can read without a browser.** Including whoever is fixing the
canvas — most of the defects found in this project were found by reading the
indexer's output rather than by looking at the canvas, and the canvas had no such
output until now.

Derived from `ExperimentDocument`, deliberately, not from `ResearchGraph`: the
document is what the canvas draws, so an export taken from anywhere else is a
second opinion about what is on screen. If a node is hidden behind the structure
toggle it is absent from both.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

#: Mermaid node ids may not carry the punctuation our identities do (`node:1a2b`,
#: `Net~1a2b/encoder`), and quoting them is not enough — the id is also a target in
#: edge syntax. So ids are rewritten and the real label goes in the box.
_UNSAFE = re.compile(r"[^A-Za-z0-9_]")

#: Mermaid's own reserved words. `end` closes a subgraph, so a node id of `end`
#: truncates the diagram at that line and every node after it disappears.
_RESERVED = {"end", "graph", "subgraph", "click", "class", "style"}

#: The relationships that are the pipeline. Same set as research/contracts.py's
#: DATAFLOW_RELATIONSHIPS and the canvas's own, because an export that disagreed
#: with the canvas about which edges are the story would be a third opinion.
FLOW_RELATIONSHIPS = {
    "reads",
    "transforms",
    "feeds",
    "trains",
    "evaluates",
    "produces",
    "logs",
    "derives_from",
}


def _identifier(node_id: str, seen: Dict[str, str]) -> str:
    """A Mermaid-safe id for `node_id`, stable within one diagram."""
    if node_id in seen:
        return seen[node_id]
    base = _UNSAFE.sub("_", node_id) or "n"
    if base[0].isdigit() or base.lower() in _RESERVED:
        base = f"n_{base}"
    candidate = base
    # Two different identities can normalise to one name (`a.b` and `a_b`), and a
    # collision would silently merge two nodes into one box.
    taken = set(seen.values())
    suffix = 2
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    seen[node_id] = candidate
    return candidate


def _escape(text: str) -> str:
    """Label text Mermaid will not read as syntax."""
    return (
        str(text)
        .replace("\\", "")
        .replace('"', "'")
        .replace("\n", " ")
        .replace("[", "(")
        .replace("]", ")")
    )


def _field(holder: Any, name: str, default: Any = None) -> Any:
    """Read `name` off a pydantic model or a plain dict.

    Written as a lookup rather than `getattr(...) or holder[name]`, because `or`
    falls through on any EMPTY value — so a document with no edges, which is an
    ordinary document, took the dict branch and raised on a model.
    """
    if isinstance(holder, dict):
        return holder.get(name, default)
    return getattr(holder, name, default)


def to_mermaid(document: Any, *, title: str = "") -> str:
    """One experiment as a Mermaid flowchart.

    Left to right, matching the canvas. Dataflow is drawn solid and everything else
    dotted — the same distinction the canvas makes with a dashed, dimmed line, and
    for the same reason: `builds` is true of nearly every pair of definitions and
    says nothing about sequence.
    """
    nodes: Sequence[Any] = _field(document, "nodes")
    edges: Sequence[Any] = _field(document, "edges")

    lines: List[str] = []
    if title:
        # A `%%` comment rather than a Mermaid `title`, which not every renderer
        # supports and which GitHub draws inconsistently.
        lines.append(f"%% {_escape(title)}")
    lines.append("flowchart LR")

    seen: Dict[str, str] = {}
    children: Dict[str, List[Any]] = {}
    top: List[Any] = []
    for node in nodes:
        parent = _field(node, "parentNode")
        if parent:
            children.setdefault(parent, []).append(node)
        else:
            top.append(node)

    def label_of(node: Any) -> str:
        data = _field(node, "data") or {}
        params = data.get("parameters") or {}
        kind = params.get("_kind")
        label = data.get("label") or params.get("_label") or _node_id(node)
        return f"{_escape(label)}<br/><i>{_escape(kind)}</i>" if kind else _escape(label)

    def _node_id(node: Any) -> str:
        return _field(node, "id")

    def draw(node: Any, indent: str) -> None:
        lines.append(f'{indent}{_identifier(_node_id(node), seen)}["{label_of(node)}"]')

    for node in top:
        kids = children.get(_node_id(node)) or []
        if not kids:
            draw(node, "    ")
            continue
        # An expanded frame is a subgraph, which is what makes the drill-down
        # readable rather than a flat list of namespaced ids.
        lines.append(f'    subgraph {_identifier(_node_id(node), seen)}["{label_of(node)}"]')
        lines.append("    direction LR")
        for kid in kids:
            draw(kid, "        ")
        lines.append("    end")

    for edge in edges:
        source = _field(edge, "source")
        target = _field(edge, "target")
        if source not in seen or target not in seen:
            continue
        data = _field(edge, "data") or {}
        relationship = data.get("_relationship") or ""
        label = _field(edge, "label") or ""
        arrow = "-->" if relationship in FLOW_RELATIONSHIPS else "-.->"
        text = _escape(label or relationship)
        # `A -->|label| B`. The label goes AFTER the arrow, not inside it: splicing
        # it between the arrow's characters produces `--|feeds|>`, which Mermaid
        # does not parse and which renders as nothing at all.
        joiner = f"{arrow}|{text}|" if text else arrow
        lines.append(f"    {seen[source]} {joiner} {seen[target]}")

    return "\n".join(lines) + "\n"
