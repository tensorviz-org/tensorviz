"""Request/result contracts for model ingestion.

Ingestion turns a *model reference* (a library + architecture name, or a path to a
repository) into a flat ``ModelGraph`` payload the existing project store accepts.

Why flat and not ``HierarchicalGraph``: ``services/graph_document.py::graph_document_v2``
reads ``payload['nodes']``/``payload['edges']``. Fed a HierarchicalGraph
(``root_nodes``/``root_edges``/``modules``) it produces a document with ZERO semantic
nodes and dumps the payload into ``semantic.metadata.migration.legacyPayload`` — after
which every server-authoritative consumer sees an empty model (``ai/orchestrator.py``
builds its known-node set from ``document.semantic.nodes``, so every AI operation on an
imported graph would be rejected as referencing an unknown node). Emitting a flat
ModelGraph with ``data.parentId`` is the only shape that round-trips correctly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from ..semantic.contracts import Diagnostic


ShapeProvenance = Literal["observed", "unreached", "inferred"]
TraceTier = Literal["fake", "real", "tree-only", "jaxpr"]

#: Where the tensor the shapes were measured on came from. A trace is only as true
#: as its input, and the first two rungs are the only ones a person chose: the rest
#: are this service's own guesses, which run and produce shapes exactly as
#: convincingly. Spelled as a ``Literal`` for the same reason ``ShapeProvenance`` is
#: — the set is closed and a typo must fail validation rather than reach a card.
#:
#: How many candidates the search burned through getting there stays where it
#: already is, in ``timings_ms["input_candidates_tried"]``. Encoding the number in
#: this string would make the set open and every reader a parser.
InputProvenance = Literal[
    # The file's own `# tensorviz: input=...` line.
    "declared",
    # The caller's `--input` / `example_input=`, which outranks the file.
    "explicit",
    # `inputs.py::seeds_from_model` read it off the first shape-bearing layer —
    # a good guess (a Linear states its own width) and still a guess.
    "seed-from-first-layer",
    # The brute-force search found a tuple that survived a forward pass.
    "search-candidate",
    # The retained partial trace failed on an automatically chosen input.
    "failed-candidate",
    # A built-in default nobody chose: `inputs.py::recipe_for`'s per-library input.
    "assumed-default",
]


class ModelRef(BaseModel):
    """What to ingest. Either (library, arch) for Tier A or repo for Tier B."""

    library: Optional[str] = Field(
        default=None, description="transformers | diffusers | timm | torchvision"
    )
    arch: Optional[str] = Field(default=None, description="architecture / model name")
    repo: Optional[str] = Field(default=None, description="path to a local repository")
    # The IDE names a file and a class instead of searching for an entrypoint: the
    # user clicked "Open as Graph" on one `class` line, so there is nothing to rank
    # and nothing to guess. `file` also disambiguates two classes of the same name
    # in different files, which repo_scan's name-keyed index cannot.
    file: Optional[str] = Field(
        default=None, description="path to the .py file defining `cls` (IDE worker)"
    )
    cls: Optional[str] = Field(
        default=None, description="class defined in `file` to instantiate (IDE worker)"
    )
    entrypoint: Optional[str] = Field(
        default=None,
        description="class or factory name inside `repo`; when omitted the ranked "
        "top candidate is used and the alternatives are returned for confirmation",
    )
    depth: int = Field(default=2, ge=1, le=12, description="module depth to materialise")
    name: Optional[str] = Field(default=None, description="graph name override")
    overrides: Dict[str, Any] = Field(
        default_factory=dict, description="constructor kwarg overrides"
    )


class RepeatGroup(BaseModel):
    """One canonical block standing in for N structurally identical instances."""

    path: str
    count: int
    reason: Literal["module-list", "call-count", "fused-parameter"]


class ReuseReport(BaseModel):
    """Whether this run's result may be stored by the host's execution cache.

    Filled by ``services/ingest/reuse_audit.py`` for resolved-plan model runs only: a run
    that wrote a file, spawned a process, opened a socket, or read a file the plan's
    digests, the interpreter's versions and the backend digest do not cover is not
    storable. Reasons are a closed set and no path is reported."""

    storable: bool
    reasons: List[Literal["write", "untracked_workspace_read", "untracked_external_read", "subprocess", "network"]] = Field(
        default_factory=list)
    untrackedReads: int = 0


class IngestDiagnostics(BaseModel):
    """Everything an operator needs to judge how much to trust the graph."""

    tier: TraceTier
    failure_diagnostic: Optional[Diagnostic] = None
    raw_modules: int = 0
    emitted_nodes: int = 0
    dataflow_edges: int = 0
    aten_ops: int = 0
    # How many recorded aten ops carried a Python frame inside the traced roots. It
    # is compared against ``aten_ops`` rather than reported alone: a capture that
    # silently stops finding frames (a model whose forward runs entirely inside a
    # library) is the difference between "no op has a source line" and "the feature
    # is off", and only the ratio tells them apart.
    frames_captured: int = 0
    nodes_with_observed_shapes: int = 0
    repeat_groups: List[RepeatGroup] = Field(default_factory=list)
    tied_parameters: List[List[str]] = Field(default_factory=list)
    unreached_modules: List[str] = Field(default_factory=list)
    opaque_ops: List[str] = Field(default_factory=list)
    # Nodes the trace proved must exist but no nn.Module provides: functional merges
    # (add/mul/cat) and batch-preserving flattens. Reported so an operator can tell a
    # materialised node from one that came from a real module.
    materialised_ops: List[str] = Field(default_factory=list)
    # Extra nodes for the 2nd..Nth invocation of a reused stateless leaf.
    split_call_sites: List[str] = Field(default_factory=list)
    # Reused modules deliberately NOT split (weight sharing a flat graph cannot express),
    # which is why some arity violations can survive.
    refused_call_sites: List[str] = Field(default_factory=list)
    stubbed_dependencies: List[str] = Field(default_factory=list)
    example_input: List[List[int]] = Field(default_factory=list)
    # Whether anybody chose ``example_input``. None when no forward ran at all, which
    # is the honest answer for a tree-only result: there is no input to have a
    # provenance. Mirrored onto the Input node's params as ``_input_provenance`` so
    # the card can say it without the canvas having to carry the diagnostics.
    input_provenance: Optional[InputProvenance] = None
    # What the global CPU generator was seeded to for this run, so that "which graph
    # did I get" has an answer. Set only by ``ingest_source``; None everywhere else is
    # the honest reading — the fake tiers construct and trace under ``FakeTensorMode``,
    # which draws nothing, so there is no seed to report rather than a seed of zero.
    #
    # Declared for the same reason as ``series`` below: pydantic drops what it was not
    # told about, and the worker's JSON is the only way the extension ever sees this.
    seed: Optional[int] = None
    provenance: str = ""
    # The scalar a `--kind train_step` run differentiated, as a float. None for every
    # other kind, which is the honest answer: a forward pass has no loss, and 0.0
    # would read as one that had converged.
    loss: Optional[float] = None
    # P4 Watch: one entry per step of a ``--steps N`` run, as
    # ``{step, loss, per_node: {<node id>: {grad_norm, absmax, nan}}}``. Empty for
    # every run that did not ask for one, which is the honest answer — a single step
    # has no curve.
    #
    # Declared here because pydantic ignores what it is not told about: the worker
    # emits ``series`` inside ``diagnostics``, and without this field a record that
    # round-trips through the runs list comes back with the curve silently gone.
    # Nothing errors, nothing warns, and the sparkline simply stops existing on
    # reload — the failure mode that only shows up to a user, never to a test.
    #
    # Typed as loosely as the worker writes it. A model per entry would be a second
    # description of a shape that is already fixed by ``worker.py`` and asserted
    # there; two of them is how the two drift.
    series: List[Dict[str, Any]] = Field(default_factory=list)
    # Where a failed forward raised: (file, line, function), innermost last. The
    # message alone ("mat1 and mat2 shapes cannot be multiplied") names no line, and a
    # line is the entire content of the editor diagnostic the IDE files for it.
    error_frames: List[List[Any]] = Field(default_factory=list)
    # Which interpreter produced this result: {python, torch, elapsed_ms}. Empty when
    # ingestion ran in the backend process; filled by services/ingest/worker.py, whose
    # whole point is that the answer came from the USER's environment and not ours.
    worker: Dict[str, Any] = Field(default_factory=dict)
    # Immutable setup identity for explicit worker execution; never inferred from
    # the live graph or silently populated for legacy traces.
    execution: Optional[Dict[str, Any]] = None
    run_evidence: Optional[Dict[str, Any]] = None
    # Canonical saved-source capture from a resolved plan or prepared automatic call;
    # validated by the recorder before publication.
    semantic_capture: Optional[Dict[str, Any]] = None
    semantic_capture_error: Optional[Literal["recorder_error", "byte_limit", "source_mismatch"]] = None
    automatic_capture: Optional[Dict[str, Any]] = None
    # Execution-cache storability of a resolved-plan run (see ``ReuseReport``). None for
    # every other run. Declared because pydantic drops what it is not told about.
    reuse: Optional[ReuseReport] = None
    warnings: List[str] = Field(default_factory=list)
    timings_ms: Dict[str, float] = Field(default_factory=dict)


class EntrypointCandidate(BaseModel):
    name: str
    score: float
    file: str
    line: int
    reasons: List[str] = Field(default_factory=list)
    factories: List[str] = Field(default_factory=list)


class IngestResult(BaseModel):
    """A flat ModelGraph plus the evidence behind it."""

    ok: bool
    name: str
    graph: Dict[str, Any] = Field(default_factory=dict, description="{nodes, edges}")
    diagnostics: IngestDiagnostics
    candidates: List[EntrypointCandidate] = Field(
        default_factory=list, description="Tier B: ranked alternatives to confirm"
    )
    error: Optional[str] = None


TraceKind = Literal["model", "data", "train_step"]


class TraceRecord(BaseModel):
    """One traced *program*, whatever kind of program it was (task.md M9).

    ``IngestResult`` answers "what is this model". A record answers "what was run,
    when, and which invocation was it" — which is what a data pipeline and a training
    step have instead of an architecture, and what a list of runs is made of.

    Additive on purpose: ``graph`` and ``diagnostics`` are the result's own fields
    carried across by :meth:`of`, so every reader of an ``IngestResult`` keeps working
    unchanged and nothing has to be migrated to gain a second kind of trace.

    Who writes one: the **extension** composes a record from a worker envelope and
    appends it as one line to ``.tensorviz/runs/<id>.jsonl`` (task.md §6.5); this
    process only ever reads them back, and this model is what it validates them with.
    The worker's stdout envelope is unchanged (§6.3 fixes its five keys) — the two
    facts a record adds, *which kind was asked for* and *when*, are the caller's to
    know, not the subprocess's.

    An ``eval`` kind is task.md §10 material until a view consumes one: a kind in a
    ``Literal`` that nothing renders is a promise, not a feature.
    """

    kind: TraceKind
    target: str = Field(
        description="what was run, as '<file>::<class or callable>'",
        min_length=1,
    )
    graph: Dict[str, Any] = Field(default_factory=dict, description="{nodes, edges}")
    spans: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="task.md §6.2, as the worker computed it"
    )
    diagnostics: IngestDiagnostics
    run_id: str = Field(min_length=1, description="unique to this invocation")
    started_at: str = Field(
        min_length=1, description="ISO-8601 UTC instant the run started"
    )
    #: The worker envelope version this record was read from, kept so a record
    #: written by an older build is recognisable as one rather than misread.
    protocol_version: int = 1

    @classmethod
    def of(
        cls,
        *,
        kind: TraceKind,
        target: str,
        result: IngestResult,
        spans: Optional[Dict[str, Dict[str, Any]]] = None,
        run_id: str,
        started_at: str,
        protocol_version: int = 1,
    ) -> "TraceRecord":
        """Wrap one ``IngestResult`` as the record of the run that produced it."""
        return cls(
            kind=kind,
            target=target,
            graph=result.graph,
            spans=dict(spans or {}),
            diagnostics=result.diagnostics,
            run_id=run_id,
            started_at=started_at,
            protocol_version=protocol_version,
        )


class CatalogEntry(BaseModel):
    library: str
    arch: str
    kind: Literal["config", "factory", "registry"] = "registry"
