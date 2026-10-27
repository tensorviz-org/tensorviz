"""The project graph contract: typed nodes, typed edges, and what we know about each.

One `ResearchGraph` describes one experiment. A project is several of them plus the
files no experiment reaches — because a file the user can see must never vanish from
the picture just because nothing imports it.

Three rules hold the shape together:

* **Provenance is a field, not a decoration.** Every node and edge records *how we
  know it*: read from the AST (`static`), stated by the user through the SDK
  (`declared`), guessed by a naming or call-shape rule (`inferred`), or measured by
  actually running the model (`observed`, reserved for the trace worker). This is
  what lets the canvas draw a guess differently from a fact, and it is the arbiter
  when two producers claim the same node.

* **Roots and leaves are computed, never stored.** An entry node is one nothing
  *flows into*, which means the question is asked of dataflow relationships only.
  `contains`, `imports` and `calls` describe how code is arranged, not how data
  moves, so a file is never a root just because it contains a model.

* **Identity is content, not position.** `digest()` is order-independent, so
  re-indexing a file whose statements moved yields the same digest and the canvas
  does not redraw. Node ids come from `identity.py` and carry no line numbers.

Validation refuses a graph rather than repairing one: duplicate ids, an edge whose
endpoint does not exist, or a node scoped to a different experiment are all errors.
A lossy view of code is acceptable; a *wrong* one is not.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "research-graph/1"

#: What a node *is*. Deliberately small and closed: an unclassified definition
#: becomes `function` or `generic` rather than growing this list, because a kind
#: nobody can draw differently is not worth having.
NodeKind = Literal[
    "project",
    "file",
    "function",
    "dataset",
    "transform",
    "model",
    "component",
    "trainer",
    "evaluator",
    "metric",
    "artifact",
    "checkpoint",
    "configuration",
    "generic",
]

#: How two nodes relate. Split by meaning below into dataflow and structure.
Relationship = Literal[
    "contains",
    "builds",
    "imports",
    "calls",
    "reads",
    "transforms",
    "feeds",
    "trains",
    "evaluates",
    "produces",
    "logs",
    "runs_on",
    "derives_from",
]

#: How we know. Ordered weakest-to-strongest for merge precedence; see `outranks`.
Provenance = Literal["inferred", "static", "declared", "observed"]

_PROVENANCE_RANK: Dict[str, int] = {
    "inferred": 0,
    "static": 1,
    "declared": 2,
    "observed": 3,
}

#: Relationships along which data actually moves. Entry/terminal are computed over
#: these alone — `contains`/`imports`/`calls` say where code lives, not what flows.
DATAFLOW_RELATIONSHIPS: Set[str] = {
    "reads",
    "transforms",
    "feeds",
    "trains",
    "evaluates",
    "produces",
    "logs",
    "derives_from",
}

STRUCTURE_RELATIONSHIPS: Set[str] = {"contains", "builds", "imports", "calls", "runs_on"}

#: The subset the canvas hides behind "show structure". Narrower than
#: `STRUCTURE_RELATIONSHIPS` on purpose: these three describe where code *lives*, and
#: a researcher opening the canvas came for the pipeline rather than a directory
#: listing with arrows. `runs_on` is structural too — a configuration is not data
#: flowing anywhere — but it is part of the experiment's composition, and hiding it
#: would leave the configuration card drawn with no edge at all, which reads as a bug
#: rather than as a deliberate omission.
CODE_STRUCTURE_RELATIONSHIPS: Set[str] = {"contains", "imports", "calls"}


def outranks(candidate: str, incumbent: str) -> bool:
    """Does `candidate` provenance beat `incumbent` when both claim one node?

    Strictly greater, so an equal-strength second claim never silently replaces the
    first — ties are broken by producer order at the call site, which is declared,
    not emergent.
    """
    return _PROVENANCE_RANK[candidate] > _PROVENANCE_RANK[incumbent]


class SourceRef(BaseModel):
    """Where a node came from, in the shape the extension's spanmap already reads.

    Matches `services/ide/projection.py`'s span shape exactly (`file`, 1-based line
    with 0-based column, statement digest) so click-to-source needs no second
    format and no translation layer.
    """

    model_config = ConfigDict(extra="forbid")

    file: str = Field(min_length=1)
    start: Tuple[int, int]
    end: Tuple[int, int]
    #: sha256 of the statement text, truncated. Reformatting does not move a node;
    #: changing what it computes does.
    digest: Optional[str] = None


class ResearchNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    kind: NodeKind
    label: str = Field(min_length=1)
    #: Stable across edits that move the definition: `models/net.py::Net`. Two
    #: experiments using one class share this and differ in `id`.
    definition_id: Optional[str] = None
    source: Optional[SourceRef] = None
    params: Dict[str, object] = Field(default_factory=dict)
    #: Who emitted it — a classifier name, `tensorviz.sdk`, the trace worker.
    producer: str = "static"
    provenance: Provenance = "static"


class ResearchEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    from_node_id: str = Field(min_length=1)
    to_node_id: str = Field(min_length=1)
    relationship: Relationship
    source: Optional[SourceRef] = None
    producer: str = "static"
    provenance: Provenance = "static"
    #: What the source said, for an edge that wants to be read rather than merely
    #: followed: `{"attr": "transformer.h", "via": "ModuleList", "count":
    #: "config.n_layer"}`. Values are source text, never evaluated — `config.n_layer`
    #: is 12 only once a config exists, and printing 12 would claim a number nothing
    #: here ran to find out.
    params: Dict[str, str] = Field(default_factory=dict)


class GraphValidationError(ValueError):
    """A graph that would draw something untrue. Refused, never repaired."""


class ResearchGraph(BaseModel):
    """One experiment's typed graph."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str = CONTRACT_VERSION
    graph_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    nodes: List[ResearchNode] = Field(default_factory=list)
    edges: List[ResearchEdge] = Field(default_factory=list)

    # ── validation ──────────────────────────────────────────────────────────────

    def validate_graph(self) -> "ResearchGraph":
        """Refuse duplicate ids, dangling edges, and cross-experiment members.

        Returns self so it can be chained at construction sites.
        """
        seen_nodes: Set[str] = set()
        for node in self.nodes:
            if node.id in seen_nodes:
                raise GraphValidationError(f"duplicate node id {node.id!r}")
            seen_nodes.add(node.id)
            if node.graph_id != self.graph_id or node.experiment_id != self.experiment_id:
                raise GraphValidationError(
                    f"node {node.id!r} belongs to another graph or experiment"
                )

        seen_edges: Set[str] = set()
        for edge in self.edges:
            if edge.id in seen_edges:
                raise GraphValidationError(f"duplicate edge id {edge.id!r}")
            seen_edges.add(edge.id)
            if edge.graph_id != self.graph_id or edge.experiment_id != self.experiment_id:
                raise GraphValidationError(
                    f"edge {edge.id!r} belongs to another graph or experiment"
                )
            for endpoint in (edge.from_node_id, edge.to_node_id):
                if endpoint not in seen_nodes:
                    raise GraphValidationError(
                        f"edge {edge.id!r} names unknown node {endpoint!r}"
                    )
        return self

    # ── computed topology ───────────────────────────────────────────────────────

    def _dataflow_edges(self) -> List[ResearchEdge]:
        return [e for e in self.edges if e.relationship in DATAFLOW_RELATIONSHIPS]

    def entry_node_ids(self) -> List[str]:
        """Nodes that participate in dataflow but have nothing flowing into them."""
        flow = self._dataflow_edges()
        if not flow:
            return []
        participants = {e.from_node_id for e in flow} | {e.to_node_id for e in flow}
        targets = {e.to_node_id for e in flow}
        return sorted(participants - targets)

    def terminal_node_ids(self) -> List[str]:
        """Nodes that participate in dataflow but feed nothing further."""
        flow = self._dataflow_edges()
        if not flow:
            return []
        participants = {e.from_node_id for e in flow} | {e.to_node_id for e in flow}
        sources = {e.from_node_id for e in flow}
        return sorted(participants - sources)

    def dependents_of_definition(self, definition_id: str) -> List[str]:
        """Every node here instantiating one definition — "what else uses this class"."""
        return sorted(n.id for n in self.nodes if n.definition_id == definition_id)

    def node_by_id(self, node_id: str) -> Optional[ResearchNode]:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    # ── identity ────────────────────────────────────────────────────────────────

    def canonical(self) -> Dict[str, object]:
        """Order-independent content, for digesting and for diffing two indexes."""
        return {
            "contract_version": self.contract_version,
            "graph_id": self.graph_id,
            "experiment_id": self.experiment_id,
            "nodes": sorted(
                (
                    {
                        "id": n.id,
                        "kind": n.kind,
                        "label": n.label,
                        "definition_id": n.definition_id,
                        "params": _stable(n.params),
                        "producer": n.producer,
                        "provenance": n.provenance,
                    }
                    for n in self.nodes
                ),
                key=lambda item: item["id"],
            ),
            "edges": sorted(
                (
                    {
                        "id": e.id,
                        "from": e.from_node_id,
                        "to": e.to_node_id,
                        "relationship": e.relationship,
                        "producer": e.producer,
                        "provenance": e.provenance,
                    }
                    for e in self.edges
                ),
                key=lambda item: item["id"],
            ),
        }

    def digest(self) -> str:
        """Content hash. Same graph in a different order gives the same answer, so
        a re-index that changed nothing does not make the canvas redraw."""
        return hashlib.sha256(
            json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _stable(params: Mapping[str, object]) -> Dict[str, object]:
    """Params reduced to something JSON-stable, so a digest never depends on repr."""
    out: Dict[str, object] = {}
    for key in sorted(params):
        value = params[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = [v if isinstance(v, (str, int, float, bool)) or v is None else str(v) for v in value]
        elif isinstance(value, Mapping):
            out[key] = _stable(value)
        else:
            out[key] = str(value)
    return out


class UnreachedFile(BaseModel):
    """A file no experiment's import closure reaches. Still real, still shown."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(min_length=1)
    #: Set when the file could not be parsed, so the UI can say why rather than
    #: leaving a hole where the user knows a file exists.
    error: Optional[str] = None


class ExperimentStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: `active` when an SDK declaration or a `__main__` block proves it; `candidate`
    #: when only the filename suggested it and a human should confirm.
    state: Literal["active", "candidate"]
    #: True when `tensorviz.experiment(...)` named it.
    declared: bool = False


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    entrypoint: str = Field(min_length=1)
    #: The derived `sys.path` root this experiment's imports resolve against — its
    #: project. Two projects nested in one folder differ here, with no configuration.
    root: str = ""
    status: ExperimentStatus
    graph: ResearchGraph
    #: Every workspace file the closure walk reached, including the entrypoint.
    closure: List[str] = Field(default_factory=list)


class ProjectIndex(BaseModel):
    """Everything one workspace folder contains, as several experiments plus the rest."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str = CONTRACT_VERSION
    experiments: List[Experiment] = Field(default_factory=list)
    unreached: List[UnreachedFile] = Field(default_factory=list)
    #: True when the caller admitted its file set was already incomplete, so a
    #: partial graph never looks complete.
    truncated: bool = False
    #: Distinct derived project roots, sorted. One entry is the ordinary case.
    roots: List[str] = Field(default_factory=list)

    def digest(self) -> str:
        payload = {
            "contract_version": self.contract_version,
            "truncated": self.truncated,
            "roots": sorted(self.roots),
            "experiments": sorted(
                (
                    {
                        "experiment_id": e.experiment_id,
                        "entrypoint": e.entrypoint,
                        "root": e.root,
                        "graph": e.graph.digest(),
                    }
                    for e in self.experiments
                ),
                key=lambda item: item["experiment_id"],
            ),
            "unreached": sorted(u.file for u in self.unreached),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def sorted_unique(values: Sequence[str]) -> List[str]:
    return sorted(set(values))
