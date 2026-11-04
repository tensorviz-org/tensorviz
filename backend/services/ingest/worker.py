"""Trace one class in the USER's interpreter and print the result as JSON.

Run as ``python -m services.ingest.worker`` with ``backend`` on ``PYTHONPATH``,
from the interpreter the user's project actually uses. It needs **torch and pydantic
and nothing else** from that environment; every other dependency it touches belongs to
the user's own repository.

Why a subprocess at all (task.md §8.5, D3):

* it **executes the user's code**, which the FastAPI backend must never do — in the
  IDE the file under the cursor is half-written by definition, and an import runs it;
* it must run in the user's interpreter, where their torch build and their third-party
  packages are, which is not where the backend runs;
* ``repo_scan.import_with_stubs`` prepends ``sys.path`` and installs stub modules; a
  fresh process per trace is the only reliable way to unwind that. The worker runs in
  the user's interpreter so a native crash in user code cannot take down the editor backend.

**This process writes nothing to disk.** The extension owns ``.tensorviz/`` and
persists this stdout there; a worker that cached its own output would be writing into
a workspace it was handed by a code lens.

Contract (task.md §6.3): exactly one JSON object on stdout —
``{protocol_version, result, spans, error, worker}`` — and an exit code that says what
happened, so a caller can act without parsing prose:

    0  a trace ran
    2  the arguments were wrong
    3  the class could not be constructed (no result)
    4  the forward raised; ``result`` is the partial graph and ``error`` names the line

``--capture activations|grads`` (with ``--loss`` for the second) additionally measures
what flowed through the model and puts it on the nodes as ``_act_stats``/``_grad_stats``
(task.md M8). It is opt-in because it runs real tensors through the user's model, holds
them for the length of a forward and, for gradients, runs a backward: everything the
static tiers deliberately do not do. ``error`` then also carries the ``file`` and
``line`` of the innermost frame in the user's own code, which is the one an editor can
underline.

``--kind`` (task.md M9) chooses WHICH program is traced, and the envelope is the same
for all three:

* ``model`` (the default) — one forward of ``--cls``. Everything above describes it.
* ``data --callable make_loader`` — one batch out of the DataLoader that function
  returns. There is no module hierarchy in a data pipeline, so the nodes are the
  LINES of the user's ``__getitem__`` and the collate is the merge that turns N
  samples into a batch.
* ``train_step --callable train_step`` — the user's own training step, called with
  ``(model, batch, optimizer)``. It runs the forward, the backward and the optimiser
  step, so the graph is the model's while ``diagnostics.loss`` is the scalar the
  function returned and ``_grad_stats`` are a real step's gradients rather than a
  synthesised ``out.sum()``.

``--steps N`` (task.md P4, "Watch") runs the SAME ``--kind train_step`` call N times
in this one process and adds ``result.diagnostics.series`` — one entry per step, with
the loss the step returned and per-node gradient statistics. It answers "which module
went NaN at step 340", which no single step can. It is additive under ``result``: the
five envelope keys and ``protocol_version`` are untouched, so a reader written against
protocol 1 sees exactly what it saw before and simply never looks at the new key.

Anything printed by the user's own code goes to stderr, because stdout carries the
envelope and one stray ``print`` in a repository's module scope would corrupt it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import linecache
import os
import sys
import time
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional, Tuple

#: Bumped when the envelope below changes incompatibly. The extension refuses a
#: mismatch loudly rather than guessing which half it understands.
PROTOCOL_VERSION = 1

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CONSTRUCT_FAILED = 3
EXIT_TRACE_FAILED = 4

#: How much of a sha256 a span digest keeps — the same width the AST spanmap uses
#: (services/ide/projection.py), so the two are comparable in a log.
_DIGEST_CHARS = 12

#: Nodes the graph synthesises for its boundaries. No source line produced them.
_BOUNDARY = ("input_node", "output_node")


class _Envelope:
    """The one object this process prints. Built incrementally so that a failure
    halfway still prints everything already known."""

    def __init__(self) -> None:
        # The real stdout, captured before anything redirects it. Every failure path
        # below runs inside `redirect_stdout(sys.stderr)` — which is what keeps a
        # repository's module-scope print out of the envelope — and writing through
        # `sys.stdout` there would send the envelope itself to stderr.
        self._out = sys.stdout
        self.result: Optional[Dict[str, Any]] = None
        self.spans: Dict[str, Dict[str, Any]] = {}
        self.error: Optional[Dict[str, Any]] = None
        self.worker: Dict[str, Any] = {"python": sys.version.split()[0], "interpreter": sys.executable, "platform": sys.platform}

    def emit(self, code: int) -> int:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "result": self.result,
            "spans": self.spans,
            "error": self.error,
            "worker": self.worker,
        }
        self._out.write(json.dumps(payload, default=str))
        self._out.write("\n")
        self._out.flush()
        return code


def _line_digest(file: str, line: int) -> str:
    """A fingerprint of the source line a span points at.

    It is what lets a consumer notice that the file moved on underneath a cached
    trace without re-running one. Stripped, so re-indenting a block does not
    invalidate every span in it.
    """
    text = linecache.getline(file, line).strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]


def _spanmap(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """``{node_id: {file, start, end, kind, digest}}`` for what the trace observed.

    Two kinds are possible per node and the more useful one wins: a ``frame`` span is
    the statement in the user's own file that ran this node, a ``class`` span is where
    its class was defined. For an ``nn.Linear`` the class span points into torch, so a
    reader who clicks the card would leave their own project — hence frame first.
    """
    spans: Dict[str, Dict[str, Any]] = {}
    for node in graph.get("nodes") or []:
        node_id = str(node.get("id"))
        if node_id in _BOUNDARY:
            continue
        params = ((node.get("data") or {}).get("parameters")) or {}
        file, line, kind = params.get("_decl_file"), params.get("_decl_line"), "frame"
        if not file or not isinstance(line, int):
            file, line, kind = params.get("_source_file"), params.get("_source_line"), "class"
        if not file or not isinstance(line, int):
            continue
        spans[node_id] = {
            "file": str(file),
            "start": [line, 0],
            "end": [line, 0],
            "kind": kind,
            "digest": _line_digest(str(file), line),
        }
    return spans


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="services.ingest.worker",
        description="Trace one nn.Module class and print a JSON envelope on stdout.",
    )
    parser.add_argument("--file", required=True, help="path to the .py file defining --cls")
    parser.add_argument("--prepare-inputs", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prepare-config-defaults", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--cls",
        default=None,
        help="the class to construct and trace. Required for --kind model and "
        "--kind train_step; --kind data has no class",
    )
    parser.add_argument(
        "--kind",
        default="model",
        choices=("model", "data", "train_step"),
        help="which kind of program to trace (task.md M9). `model` runs one forward "
        "of --cls; `data` pulls one batch from the DataLoader --callable returns; "
        "`train_step` calls --callable with (model, batch, optimizer) so the "
        "forward, the backward and the optimiser step are all one traced run",
    )
    parser.add_argument(
        "--callable",
        dest="callable_",
        default=None,
        help="the function --kind data and --kind train_step run: `name` for one "
        "defined in --file, or `path/to/other.py:name` for one beside it",
    )
    parser.add_argument(
        "--input",
        default=None,
        help='example input: shapes ("2,8" or "2,8;2") or a Python expression '
        'evaluated with torch in scope ("torch.randint(0, 100, (1, 16))")',
    )
    parser.add_argument("--execution", default=None, help="JSON explicit execution setup")
    parser.add_argument("--resolved-execution", action="store_true", help="read the private host-resolved execution plan")
    parser.add_argument("--reuse-audit", action="store_true",
                        help="report whether this resolved-plan run may be stored by the host's execution cache "
                             "(diagnostics.reuse); sent only when the cache is on")
    parser.add_argument("--candidate-execution", action="store_true", help="run the exact private reviewed candidate")
    parser.add_argument("--capture-preparation", action="store_true", help="read private saved facts before automatic input selection")
    parser.add_argument("--diagnostic-job", default=None, help="JSON diagnostic job; requires explicit execution")
    parser.add_argument("--workspace", default=None, help="trusted workspace root for execution files")
    parser.add_argument("--tensor-inputs", default=None, help="JSON positional tensor descriptions: shape, dtype, fill; model only")
    parser.add_argument(
        "--tier",
        default="real",
        choices=("real", "fake"),
        help="real runs real tensors so data-dependent control flow traces; "
        "fake runs under FakeTensorMode and allocates nothing",
    )
    parser.add_argument("--depth", type=int, default=2, help="module depth to materialise")
    parser.add_argument(
        "--roots",
        action="append",
        default=None,
        help="import/frame-capture root; repeatable. Defaults to the package root of --file",
    )
    parser.add_argument(
        "--overrides",
        default=None,
        help="JSON object of constructor keyword arguments",
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="skip per-op source frame capture (see scripts/bench_frame_capture.py)",
    )
    parser.add_argument(
        "--capture",
        action="append",
        default=None,
        choices=("activations", "grads"),
        help="measure what flows through the model; repeatable. `activations` adds "
        "_act_stats to every module node, `grads` adds _grad_stats and needs --loss. "
        "Real tier only, and it holds real tensors alive: use a batch of 1 or 2",
    )
    parser.add_argument(
        "--loss",
        default=None,
        help='the scalar to differentiate, as a Python expression over `out` (the '
        'forward\'s return) and `torch` — e.g. "out.sum()". Required by '
        "--capture grads and meaningless without it",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=f"Watch (task.md P4): run the --kind train_step callable this many times "
        f"(1..{MAX_WATCH_STEPS}) and report result.diagnostics.series — one entry per "
        "step, with its loss and per-node gradient statistics. Omitted, one step runs "
        "and there is no series",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="accepted for symmetry with the CLI contract; the envelope is always JSON",
    )
    return parser


def _fail(envelope: _Envelope, exc: BaseException, node_id: Optional[str] = None) -> None:
    """Record an exception in the envelope, with the frames a reader can act on."""
    import traceback

    from ..validation.setup import ExecutionRequirement, exception_causes
    causes = exception_causes(exc)
    frames = [
        {"file": frame.filename, "line": frame.lineno, "name": frame.name}
        for frame in traceback.extract_tb(causes[-1].__traceback__)
    ]
    missing_module = next((cause.name for cause in causes if isinstance(cause, ModuleNotFoundError)), None)
    import re
    if not isinstance(missing_module, str) or len(missing_module) > 256 or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", missing_module, re.ASCII):
        missing_module = None
    envelope.error = {
        "type": type(exc).__name__,
        "message": str(exc) or repr(exc),
        "frames": frames,
        "node_id": node_id,
        "cause_types": [type(cause).__name__ for cause in causes],
        "missing_module": missing_module,
    }
    requirement = next((cause for cause in causes if isinstance(cause, ExecutionRequirement)), None)
    if requirement is None and missing_module:
        requirement = ExecutionRequirement("import", "missing_dependency", ["model"], "review_dependencies",
                                           [missing_module.split(".")[0]])
    if requirement is not None:
        from ..semantic.contracts import Diagnostic
        envelope.error["diagnostic"] = Diagnostic.model_validate(requirement.diagnostic).model_dump(mode="json")
        checkpoint = getattr(requirement, 'checkpoint_compatibility', None)
        if checkpoint is not None:
            envelope.worker['checkpointCompatibility'] = checkpoint


#: A capture holds real activations alive and runs a real backward. Anything past a
#: couple of samples is a training run, not a debugging aid, and the difference is
#: the user's memory — so a bigger batch is reported rather than silently obeyed.
MAX_CAPTURE_BATCH = 2


def _capture_usage_error(args: Any, capture: Dict[str, bool]) -> Optional[str]:
    """Why this combination of ``--capture``/``--loss``/``--tier`` cannot be run.

    Every one of these is a flag the caller passed that this process would
    otherwise have to ignore, and an ignored flag reads to the user as an answer:
    "this model has no gradients" instead of "you did not ask for any".
    """
    if not (capture["activations"] or capture["grads"]):
        if args.loss:
            return "--loss does nothing without --capture grads"
        return None
    if args.tier != "real":
        return (
            f"--capture needs --tier real; statistics of a {args.tier} tensor are "
            "statistics of nothing"
        )
    if capture["grads"] and not args.loss:
        return "--capture grads needs --loss, e.g. --loss 'out.sum()'"
    return None


#: The most steps one Watch may take. The bound is the payload, not the time: every
#: step carries statistics for every watched card, and this envelope is one JSON
#: object that has to cross a pipe, be held in the extension's memory and be appended
#: to a run file. 200 steps is enough to see a loss curve bend and a gradient blow up;
#: 20,000 is a training run, which is the user's own script's job and not this
#: process's.
MAX_WATCH_STEPS = 200

#: How deep into the module tree per-step statistics go. The card the user watches is
#: a card on the canvas, and the canvas draws to this depth; going deeper would
#: multiply the payload by the leaf count of a model nobody is looking at leaf-by-leaf
#: across 200 steps. Applied independently of ``--depth``, so raising that to inspect
#: a subtree cannot silently multiply the series with it.
WATCH_DEPTH = 2

#: And a ceiling on the width, because depth alone does not bound it: a model with 300
#: top-level children has 300 cards at depth 1. Repeat groups already collapse most of
#: those (a ModuleList of 24 identical blocks renders one card), so this is the last
#: resort rather than the usual limit — and it is reported as a warning rather than
#: applied quietly, because a card missing from the strip must not read as a card with
#: no gradient.
MAX_WATCH_NODES = 64

#: The optimiser a `--kind train_step` run hands the user's function when it takes
#: one. Plain SGD at a small step: the trace is about what the step DID (a loss, a
#: set of gradients), and the update rule changes none of that while every other
#: optimiser would need arguments this process would have to invent.
TRAIN_STEP_LR = 0.01


def _kind_usage_error(args: Any) -> Optional[str]:
    """Why this ``--kind`` cannot be run with these flags (task.md M9).

    Each rule below is a flag that would otherwise be silently ignored, and an
    ignored flag reads to the user as an answer: a `--kind data` run that quietly
    dropped `--cls` looks like a pipeline that has no model in it.
    """
    if args.steps is not None:
        # Checked before anything else about the kind, because a caller who asked for
        # 500 steps and got 1 would read the single point it printed as the answer to
        # a question it never ran.
        if args.kind != "train_step":
            return (
                "--steps watches a training loop; it belongs to --kind train_step. "
                f"--kind {args.kind} runs once"
            )
        if args.steps < 1 or args.steps > MAX_WATCH_STEPS:
            return (
                f"--steps must be between 1 and {MAX_WATCH_STEPS}; {args.steps} steps "
                "of per-node statistics is a payload no reader of this envelope can "
                "carry. Run your own training script for a longer curve"
            )
    if args.kind == "model":
        if not args.cls:
            return "--kind model needs --cls"
        if args.callable_:
            return "--callable belongs to --kind data / --kind train_step"
        return None
    if not args.callable_:
        return f"--kind {args.kind} needs --callable"
    if args.kind == "data":
        if args.cls:
            return "--kind data traces a DataLoader, not a class; drop --cls"
        if args.tier != "real":
            return "--kind data needs --tier real; a DataLoader reads real bytes"
        if args.capture or args.loss:
            return "--capture and --loss measure a model; a data pipeline has neither"
        return None
    # train_step
    if not args.cls:
        return "--kind train_step needs --cls: the model the step trains"
    if args.tier != "real":
        return "--kind train_step needs --tier real; a fake tensor has no gradient"
    if args.loss:
        return (
            "--loss does nothing for --kind train_step: the loss is the one the "
            "callable itself returns"
        )
    if "grads" in (args.capture or []):
        return (
            "--capture grads is implied by --kind train_step; the step runs its own "
            "backward and a second one would double the gradients it reports"
        )
    return None


def _load_callable(spec: str, default_file: str) -> Tuple[Any, str, str]:
    """``(function, its file, its name)`` for one ``--callable``.

    Two spellings, because the two kinds want different ones. ``make_loader`` is
    defined in the file the user clicked on, so a bare name resolves in ``--file``.
    A training step is written in a script BESIDE the model it trains — the fixture
    is exactly this — so ``scripts/train_step.py:train_step`` names its own file
    while ``--file``/``--cls`` keep meaning what they have always meant.

    Split on the last ``:`` only when the left half looks like a path, so a Windows
    drive letter is not mistaken for a file/attribute separator.
    """
    from . import repo_scan as repo
    from .service import package_root

    if ".py:" in spec:
        file_part, _, attr = spec.rpartition(":")
    else:
        file_part, attr = "", spec
    if not attr:
        raise ValueError(f"--callable {spec!r} names no function")
    file = os.path.abspath(os.path.expanduser(file_part or default_file))
    if not os.path.isfile(file):
        raise ValueError(f"--callable {spec!r} points at {file}, which is not a file")
    root = package_root(file)
    module, _stubbed = repo.import_with_stubs(repo.module_path_for(root, file), [root])
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise ValueError(f"{attr!r} is not a callable in {file}")
    return fn, file, attr


def _trace_data(args: Any, trace_module: Any, roots: Tuple[str, ...]) -> Any:
    """One batch out of the DataLoader ``--callable`` returns (task.md M9).

    The loader is BUILT outside the trace and one batch pulled inside it: opening a
    file handle is not a step of the pipeline, and `__getitem__` is. What comes back
    is a graph whose nodes are the lines of the user's own dataset — the only
    provenance a plain function has — with the collate as the merge that turns N of
    them into a batch.
    """
    from .normalise import build_callable_graph
    from .spec import IngestResult

    started = time.perf_counter()
    fn, file, attr = _load_callable(args.callable_, args.file)
    loader = fn()
    traced = trace_module.trace_callable(lambda: next(iter(loader)), frame_roots=roots)
    graph, diagnostics = build_callable_graph(attr, traced.ops, roots=roots)
    diagnostics.provenance = f"{args.file}::{attr}"
    diagnostics.error_frames = [list(frame) for frame in traced.error_frames]
    diagnostics.timings_ms = {
        "trace_ms": round(traced.elapsed_ms, 1),
        "build_ms": round((time.perf_counter() - started) * 1000 - traced.elapsed_ms, 1),
    }
    if traced.error:
        diagnostics.warnings.append(
            f"the pipeline stopped early ({traced.error}); the graph is what ran "
            "before that point"
        )
    return IngestResult(
        ok=traced.error is None,
        name=attr,
        graph=graph,
        diagnostics=diagnostics,
        error=traced.error,
    )


def _trace_train_step(
    args: Any,
    trace_module: Any,
    ingest_source: Any,
    ModelRef: Any,
    overrides: Dict[str, Any],
    activations: bool,
) -> Tuple[Any, Any, Any]:
    """``(result, capture, step)`` for one call of the user's training step.

    The graph is the MODEL's, produced by the same ``ingest_source`` every other
    trace uses — the difference is one line deep inside it: what gets called. A
    ``step_scope`` (services/ingest/trace.py) replaces ``model(batch)`` with
    ``train_step(model, batch, optimizer)``, so the forward hooks see the same
    forward while the backward and the optimiser step happen around it. That is why
    the gradients are a real training step's and not a synthesised ``out.sum()``.

    The fourth return value is the ``session``: the model, the batch, the optimiser
    and the user's function, as the traced call actually saw them. Watch (``--steps``)
    needs all four and has no other way to reach any of them — the model is
    constructed several frames below inside ``ingest_source`` and the batch is
    whichever example input the shape search settled on.
    """
    import torch

    fn, _file, _attr = _load_callable(args.callable_, args.file)
    session: Dict[str, Any] = {"fn": fn}

    def step(model: Any, call_args: Tuple[Any, ...], _call_kwargs: Dict[str, Any]) -> Any:
        batch = call_args[0] if call_args else None
        # One optimiser for the life of this process rather than one per call: a
        # watch of N steps is N steps of the same run, and a fresh optimiser each
        # step would throw away whatever state a stateful one had accumulated.
        # Identical behaviour at --steps 1, and for the plain SGD below at any N.
        if session.get("model") is not model:
            session["optimizer"] = torch.optim.SGD(model.parameters(), lr=TRAIN_STEP_LR)
        session["model"] = model
        session["batch"] = batch
        return fn(model, batch, session["optimizer"])

    spec = trace_module.CaptureSpec(
        activations=activations, grads=True, external_backward=True
    )
    with trace_module.capture_scope(spec) as capture, trace_module.step_scope(step) as taken:
        result = ingest_source(
            ModelRef(file=args.file, cls=args.cls, depth=args.depth, overrides=overrides),
            tier="real",
            example_input=args.input,
            roots=args.roots,
            capture_frames=not args.no_frames,
        )
    returned = taken.returned
    if isinstance(returned, torch.Tensor) and returned.numel() == 1:
        result.diagnostics.loss = float(returned.detach())
    else:
        result.diagnostics.warnings.append(
            f"{args.callable_} returned {type(returned).__name__}, not a one-element "
            "tensor; the run has no loss to report"
        )
    return result, capture, taken, session


def _watch_nodes(graph: Dict[str, Any], model: Any) -> Tuple[List[Tuple[str, Any]], int]:
    """``([(node id, module)], how many were dropped)`` — the cards a series measures.

    Read off the GRAPH rather than off ``model.named_modules()``, and that is the
    whole trick: the graph is already the depth-limited, repeat-collapsed set of
    cards the canvas draws, its node ids ARE the module paths (services/ingest/
    normalise.py sets both), and so a series entry lands on a card without anyone
    inventing a second identity for a module. A model whose 24 identical blocks
    collapsed to one card is watched as one card, which is also the only reason 200
    steps of a real network is a payload rather than a denial of service.

    The root module (path ``""``) is deliberately not watched. Its gradient is the
    whole model's, so it would go NaN whenever ANY child did — a strip that is amber
    everywhere names nothing.
    """
    named = dict(model.named_modules())
    chosen: List[Tuple[str, Any]] = []
    for node in graph.get("nodes") or []:
        data = node.get("data") or {}
        depth = data.get("depth")
        path = str(((data.get("parameters") or {}).get("_module_path")) or "")
        if not path or not isinstance(depth, int) or depth > WATCH_DEPTH:
            continue
        module = named.get(path)
        if module is not None:
            chosen.append((path, module))
    return chosen[:MAX_WATCH_NODES], max(0, len(chosen) - MAX_WATCH_NODES)


def _grad_summary(module: Any) -> Optional[Dict[str, Any]]:
    """``{grad_norm, absmax, nan}`` over one card's parameter gradients, or None.

    Parameter gradients rather than the activation gradients ``_grad_stats`` carries,
    because the question a watch answers is about the CARD ("block.3.attn.out went
    NaN"), and a card's parameters are its own by ``recurse=True``: a module card
    covers what it contains, including the leaves below the depth the canvas draws.

    ``None`` when nothing had a gradient this step — a card with no parameters, or
    one the backward never reached. Absent is not the same fact as zero, and a strip
    that drew 0.0 for a card nothing measured would be lying about a healthy layer.

    Every number here is computed over the FINITE elements only, and not out of
    tidiness: ``json.dumps`` writes ``NaN`` and ``Infinity`` for those two floats,
    which are not JSON — ``JSON.parse`` in the extension throws on the whole
    envelope. So an overflowing gradient must never reach the payload AS a float; it
    reaches it as the ``nan`` count, which is what the reader acts on anyway.
    """
    import torch

    total = 0.0
    absmax = 0.0
    nan = 0
    measured = False
    for parameter in module.parameters(recurse=True):
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        measured = True
        with torch.no_grad():
            flat = grad.detach().reshape(-1).double()
            nan += int(torch.isnan(flat).sum().item())
            finite = flat[torch.isfinite(flat)]
            if finite.numel():
                total += float(finite.pow(2).sum().item())
                absmax = max(absmax, float(finite.abs().max().item()))
    if not measured:
        return None
    # Doubles do not overflow for any finite float32 gradient; the guard is for the
    # float64 model that manages it anyway. The norm is at least the largest element,
    # so absmax is a floor and not a guess.
    norm = total ** 0.5 if total != float("inf") else absmax
    return {"grad_norm": norm, "absmax": absmax, "nan": nan}


def _watch_step_loss(returned: Any) -> Optional[float]:
    """The scalar a step returned, or None if it is not one a reader can plot.

    A NaN loss is reported as ``None`` for the same reason as above — ``NaN`` is not
    JSON — and it is not the loss that answers "what went wrong" anyway: the
    per-node ``nan`` counts name the module, which the loss never can.
    """
    import torch

    if isinstance(returned, torch.Tensor) and returned.numel() == 1:
        value = float(returned.detach())
        return value if value == value and abs(value) != float("inf") else None
    return None


def _watch(
    args: Any, result: Any, session: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """``(series, warnings)`` for ``--steps N`` (task.md P4).

    Step 1 is the step that was already traced — the graph, the spans and
    ``diagnostics.loss`` all describe it — and steps 2..N are the same call again,
    on the same model, with the SAME batch. That is a probe and not training, and it
    is what the UI has to say: this runs your ``train_step`` N times in a fresh
    process, it does not run your training script.

    A step that raises stops the watch and says so, rather than losing the graph and
    the steps that did run: the failure IS the answer often enough (a model that
    overflows at step 40 raises at step 41) that discarding it would be the worst
    behaviour available.
    """
    nodes, dropped = _watch_nodes(result.graph, session["model"])
    warnings: List[str] = []
    if dropped:
        warnings.append(
            f"watch reports the first {MAX_WATCH_NODES} cards at depth "
            f"{WATCH_DEPTH} or less; {dropped} more carry no series"
        )

    def per_node() -> Dict[str, Dict[str, Any]]:
        measured = {}
        for path, module in nodes:
            summary = _grad_summary(module)
            if summary is not None:
                measured[path] = summary
        return measured

    series: List[Dict[str, Any]] = [
        {"step": 1, "loss": result.diagnostics.loss, "per_node": per_node()}
    ]
    for index in range(2, int(args.steps) + 1):
        try:
            returned = session["fn"](
                session["model"], session["batch"], session["optimizer"]
            )
        except Exception as exc:
            warnings.append(
                f"watch stopped after step {index - 1} of {args.steps}: "
                f"{type(exc).__name__}: {exc}"
            )
            break
        series.append(
            {"step": index, "loss": _watch_step_loss(returned), "per_node": per_node()}
        )
    return series, warnings


def _train_step_batch_error(args: Any) -> Optional[str]:
    """Why a training step has no batch to run on, or None.

    Checked BEFORE the trace rather than after, because without a recipe
    ``_introspect_and_trace`` falls back to searching input shapes — and each
    attempt would call the user's ``train_step``, take an optimiser step and change
    the weights the next attempt measures.
    """
    from .service import _input_from_comment, parse_input_spec

    if args.input and parse_input_spec(args.input) is not None:
        return None
    # The class the run traces, so the check reads the declaration the run will use.
    if _input_from_comment(args.file, args.cls) is not None:
        return None
    return (
        "--kind train_step needs a batch: pass --input (e.g. --input 2,8) or put a "
        "`# tensorviz: input=...` comment directly above the model class"
    )


def _roots_of(args: Any) -> Tuple[str, ...]:
    """The directories whose Python frames count as the user's own code.

    The same rule the frame capture itself uses (``ingest_source``), restated here
    because the error mapping runs after that call has returned. ``package_root``
    is imported inside the function because importing ``.service`` at module scope
    would pull torch into a process that has not decided to load it yet.
    """
    from .service import package_root

    roots = [os.path.abspath(root) for root in (args.roots or []) if root]
    return tuple(roots or [package_root(args.file)])


def _attach_stats(graph: Dict[str, Any], stats: Dict[str, Dict[str, Any]]) -> None:
    """Put ``_act_stats``/``_grad_stats`` on the nodes their modules produced.

    Keyed by ``_module_path`` — the qualified attribute path — because that is the
    one identity shared by ``named_modules()``, the trace and the graph. A
    materialised node (a residual ``add``) has no module behind it and therefore no
    statistics, which is correct rather than missing: nothing hooked it.
    """
    if not stats:
        return
    for node in graph.get("nodes") or []:
        params = (node.get("data") or {}).get("parameters")
        if not isinstance(params, dict):
            continue
        measured = stats.get(str(params.get("_module_path") or ""))
        if not measured:
            continue
        if "act" in measured:
            params["_act_stats"] = measured["act"]
        if "grad" in measured:
            params["_grad_stats"] = measured["grad"]


def _batch_warning(result: Any) -> List[str]:
    """Say so when the capture ran on a batch it was not designed for."""
    shapes = result.diagnostics.example_input or []
    first = shapes[0] if shapes else []
    if first and int(first[0]) > MAX_CAPTURE_BATCH:
        return [
            f"capture ran on a batch of {int(first[0])}; {MAX_CAPTURE_BATCH} or fewer "
            "is what it is sized for, and the statistics hold that many activations "
            "in memory at once"
        ]
    return []


def _node_at(spans: Dict[str, Dict[str, Any]], file: str, line: int) -> Optional[str]:
    """The node whose span is exactly this line of this file.

    ``frame`` spans first: they are lines of the user's own ``forward``, while a
    ``class`` span points at wherever the class was defined — for an ``nn.Linear``,
    inside torch, which no traceback frame of the user's should ever match by luck.
    """
    for wanted in ("frame", "class"):
        for node_id in sorted(spans):
            span = spans[node_id]
            if span.get("kind") != wanted:
                continue
            if str(span.get("file")) == file and int(span.get("start", [0])[0]) == line:
                return node_id
    return None


def _error_site(
    frames: List[Dict[str, Any]],
    spans: Dict[str, Dict[str, Any]],
    roots: Tuple[str, ...],
) -> Dict[str, Any]:
    """Where a failed forward went wrong, as a line and as a node (task.md M8).

    The *innermost* frame under the user's roots is the line to underline: the
    torch frames below it are true and useless, and the frames above it are the
    call site rather than the fault.

    The node is a second question and a weaker one. A module that was entered and
    never produced an output is *evidence* of where the run stopped, and that is
    why captured invocation failures take precedence over a line match. This walk
    — outward from the innermost frame until a line matches a span, M4's per-op frames read backwards — is the
    fallback for a forward that raised somewhere no module owns, where the only
    thing connecting the failure to a card is the line they share.
    """
    own = [
        frame
        for frame in frames
        if str(frame.get("file") or "")
        and os.path.abspath(str(frame["file"])).startswith(tuple(roots))
    ]
    if not own:
        return {}
    site: Dict[str, Any] = {"file": own[-1]["file"], "line": int(own[-1]["line"])}
    for frame in reversed(own):
        node_id = _node_at(spans, str(frame["file"]), int(frame["line"]))
        if node_id:
            site["node_id"] = node_id
            break
    return site


def main(argv: Optional[List[str]] = None, *, envelope: Optional[_Envelope] = None) -> int:
    from .worker_lifetime import guard_parent_pipe
    guard_parent_pipe()
    # The docstring promise "writes nothing to disk" covers the workspace
    # bytecode cache too: without this, importing the user's file leaves a
    # ``__pycache__`` beside it.
    sys.dont_write_bytecode = True
    envelope = envelope if envelope is not None else _Envelope()
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_:
        # argparse writes its own message to stderr and raises SystemExit(2). The
        # caller still gets an envelope, because a caller that only knows how to read
        # one would otherwise see an empty stdout and no reason.
        envelope.error = {
            "type": "UsageError",
            "message": "invalid arguments; see stderr",
            "frames": [],
            "node_id": None,
        }
        return envelope.emit(EXIT_USAGE if exit_.code else EXIT_OK)

    if args.candidate_execution and not args.prepare_inputs and not args.prepare_config_defaults:
        from .candidate_worker import run_candidate
        return run_candidate(args, envelope)

    capture_spec = {
        "activations": "activations" in (args.capture or []),
        "grads": "grads" in (args.capture or []),
    }
    # Kind rules first: `--kind train_step --capture grads` is a train_step usage
    # error ("the step runs its own backward"), and the capture rule below would
    # otherwise answer it with the wrong sentence ("--capture grads needs --loss").
    usage = _kind_usage_error(args) or _capture_usage_error(args, capture_spec)
    execution = None
    diagnostic_job = None
    resolved_plan = None
    plan_digest = None
    capture_preparation = None
    preparation_digest = None
    if args.capture_preparation and args.resolved_execution:
        usage = "Choose a resolved execution or automatic capture preparation."
    if args.capture_preparation and not usage:
        try:
            from .execution_spec import workspace_file
            from ..semantic.execution_plan import consume_preparation_file, verify_saved_plan
            from pathlib import Path
            if (args.kind != "model" or args.execution is not None or args.diagnostic_job is not None
                    or args.input is not None or args.tensor_inputs is not None or args.overrides
                    or args.tier not in ("real", "fake") or args.capture or args.loss is not None or not args.workspace):
                raise ValueError("competing automatic capture arguments")
            capture_preparation, preparation_digest = consume_preparation_file()
            if (capture_preparation.source.className != args.cls
                    or workspace_file(args.workspace, capture_preparation.source.modelFile) != Path(args.file).resolve(strict=True)):
                raise ValueError("automatic capture target mismatch")
            verify_saved_plan(capture_preparation, args.workspace)
            request = capture_preparation.request
            args.input = request.input
            args.tensor_inputs = json.dumps(request.tensorInputs) if request.tensorInputs is not None else None
            args.overrides = json.dumps(request.overrides) if request.overrides is not None else None
            envelope.worker.update(preparationId=capture_preparation.id, preparationDigest=preparation_digest)
        except Exception:
            usage = "The saved capture preparation is invalid or changed. Prepare it again before running."
    if args.resolved_execution and not usage:
        try:
            from .execution_spec import workspace_file
            from ..semantic.execution_plan import consume_plan_file, verify_saved_plan
            if (args.kind != "model" or args.execution is not None or args.diagnostic_job is not None
                    or args.input is not None or args.tensor_inputs is not None or args.overrides
                    or args.tier != "real" or args.capture or args.loss is not None or not args.workspace):
                raise ValueError("competing execution arguments")
            resolved_plan, plan_digest = consume_plan_file()
            from pathlib import Path
            if (resolved_plan.source.className != args.cls
                    or workspace_file(args.workspace, resolved_plan.source.modelFile) != Path(args.file).resolve(strict=True)):
                raise ValueError("execution target mismatch")
            verify_saved_plan(resolved_plan, args.workspace)
            execution, diagnostic_job = resolved_plan.execution, resolved_plan.job
            envelope.worker.update(planId=resolved_plan.id, planDigest=plan_digest)
        except Exception:
            usage = "The saved execution plan is invalid or changed. Resolve it again before running."
    if args.diagnostic_job is not None and args.execution is None:
        usage = "A diagnostic job requires explicit execution setup."
    if args.execution is not None and not usage:
        try:
            from .execution_spec import validate_execution_spec
            if (args.kind != "model" or args.input is not None or args.tensor_inputs is not None
                    or args.tier != "real" or args.capture or args.loss is not None
                    or not args.workspace or len(args.execution) > 32_768):
                raise ValueError("Explicit execution requires a workspace, real model tier and no other input flags.")
            execution = validate_execution_spec(json.loads(args.execution))
            if args.diagnostic_job is not None:
                from .execution_spec import validate_execution_job
                if len(args.diagnostic_job) > 16_384:
                    raise ValueError("Diagnostic job exceeds 16 KiB.")
                diagnostic_job = validate_execution_job(json.loads(args.diagnostic_job))
        except (ValueError, TypeError, RecursionError) as exc:
            usage = str(exc) if isinstance(exc, ValueError) else "Invalid explicit execution request."
    if args.prepare_inputs and (resolved_plan is None or resolved_plan.origin != "explicit"
            or diagnostic_job is not None or args.candidate_execution):
        usage = "Input preparation needs a private model construction plan without a saved profile or diagnostic job."
    if args.prepare_config_defaults and not args.prepare_inputs:
        usage = "Config defaults require private input preparation; they cannot alter a capture request."
    tensor_inputs = None
    if args.tensor_inputs is not None and not usage:
        try:
            from .tensor_inputs import validate_tensor_inputs

            if args.kind != "model" or args.input is not None:
                raise ValueError("--tensor-inputs requires --kind model without --input.")
            if len(args.tensor_inputs) > 16_384:
                raise ValueError("Tensor input descriptions are too large.")
            tensor_inputs = validate_tensor_inputs(json.loads(args.tensor_inputs))
        except (ValueError, TypeError, RecursionError):
            usage = "Invalid --tensor-inputs: use bounded shape/dtype/fill descriptions for a model without --input."
    if usage:
        # Exit 2 rather than running anyway with half the flags honoured: a user
        # who asked for gradients and silently got activations would read the
        # missing numbers as "this model has none".
        sys.stderr.write(usage + "\n")
        envelope.error = {
            "type": "UsageError",
            "message": usage,
            "frames": [],
            "node_id": None,
        }
        return envelope.emit(EXIT_USAGE)

    if args.prepare_inputs:
        from .graph_example import run_probe
        return run_probe(args, envelope, resolved_plan, plan_digest)

    started = time.perf_counter()
    # Importing torch and the user's repository both print; neither may touch the
    # envelope's channel.
    with redirect_stdout(sys.stderr):
        try:
            import torch  # noqa: F401  (reported in `worker`, and a missing one is the error)

            from .service import ingest_source
            from .spec import ModelRef
        except Exception as exc:  # torch or pydantic missing from the user's env
            _fail(envelope, exc)
            return envelope.emit(EXIT_CONSTRUCT_FAILED)

        envelope.worker["torch"] = torch.__version__
        overrides: Dict[str, Any] = {}
        if args.overrides:
            try:
                overrides = json.loads(args.overrides)
                if not isinstance(overrides, dict):
                    raise ValueError("--overrides must be a JSON object")
            except Exception as exc:
                _fail(envelope, exc)
                envelope.worker["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
                return envelope.emit(EXIT_USAGE)

        from . import trace as trace_module

        # None for the kinds that measure nothing; a data pipeline has no module to
        # hook and therefore no statistics to attach.
        capture: Any = None
        # None unless --steps asked for a watch. Kept out of `result` until after it
        # is dumped, because `series` is additive UNDER `result` in the JSON and not a
        # field of the IngestResult model — a diagnostics model that silently drops
        # unknown keys (it does) would eat it on the way through.
        series: Optional[List[Dict[str, Any]]] = None
        from .execution_sources import workspace_imports
        if args.tier == "fake":
            # FakeTensorMode() imports torch._dynamo (~1,200 modules, ~0.6 s) the first
            # time it is built. Import it here, before the workspace import hook, which
            # would otherwise resolve a path for every one of those modules.
            import torch._dynamo  # noqa: F401
        imported_sources = []
        # Execution cache: which files the user's import, construction and forward read
        # (resolved-plan model runs the host asked about only, so a run with the cache off
        # does no extra work). Installed after torch and the capture internals are
        # imported, so a fresh worker and a pre-started one observe the same window.
        reuse_audit = None
        if args.reuse_audit and resolved_plan is not None and args.kind == "model" and not args.prepare_inputs:
            from .reuse_audit import ReadAudit
            reuse_audit = ReadAudit.for_plan(resolved_plan, args.workspace)
        reuse_report = None
        try:
            if args.kind == "data":
                result = _trace_data(args, trace_module, _roots_of(args))
            elif args.kind == "train_step":
                batch_error = _train_step_batch_error(args)
                if batch_error:
                    sys.stderr.write(batch_error + "\n")
                    envelope.error = {
                        "type": "UsageError",
                        "message": batch_error,
                        "frames": [],
                        "node_id": None,
                    }
                    return envelope.emit(EXIT_USAGE)
                result, capture, _taken, session = _trace_train_step(
                    args,
                    trace_module,
                    ingest_source,
                    ModelRef,
                    overrides,
                    capture_spec["activations"],
                )
                if args.steps is not None:
                    # Inside the redirect: steps 2..N call the user's function again,
                    # and a `print` in their training step must not land in the
                    # envelope. Inside the try for the same reason every trace is —
                    # except that _watch swallows a failing step itself, so a watch
                    # that dies at step 40 still prints the graph and 39 steps.
                    series, watch_warnings = _watch(args, result, session)
                    result.diagnostics.warnings.extend(watch_warnings)
            else:
                spec = trace_module.CaptureSpec(
                    activations=capture_spec["activations"],
                    grads=capture_spec["grads"],
                    loss=args.loss,
                )
                # The capture is a property of this process's next trace, not an
                # argument of ingest_source: the call below is shared with the web
                # importers, which must keep the signature they have (see
                # services/ingest/trace.py::capture_scope).
                # Torch 2.14 creates its inductor cache directory at
                # ``torch._dynamo.package`` import. That import is lazy and would
                # otherwise land inside the audit window, so a plain model looks
                # like it wrote a file. Paying for it here keeps a fresh worker
                # and a standby worker on the same side of the window.
                import torch._dynamo.package  # noqa: F401
                if reuse_audit is not None:
                    reuse_audit.start()
                try:
                    with workspace_imports(args.workspace if execution is not None or capture_preparation is not None else None) as imported_sources, trace_module.capture_scope(spec) as capture:
                        result = ingest_source(
                            ModelRef(
                                file=args.file, cls=args.cls, depth=args.depth, overrides=overrides
                            ),
                            tier=args.tier,
                            example_input=args.input,
                            tensor_inputs=tensor_inputs,
                            execution=execution,
                            diagnostic_job=diagnostic_job,
                            workspace=args.workspace,
                            roots=args.roots,
                            capture_frames=not args.no_frames,
                            resolved_plan=resolved_plan,
                            capture_preparation=capture_preparation,
                        )
                finally:
                    if reuse_audit is not None:
                        reuse_report = reuse_audit.stop()
        except Exception as exc:
            if execution is not None or capture_preparation is not None:
                envelope.worker["importedSources"] = imported_sources
            _fail(envelope, exc)
            if args.tier == "fake" and capture_preparation is not None:
                _fake_limitation(envelope, trace_module.fake_tensor_limitation(exc), "construction", trace_module)
            envelope.worker["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
            return envelope.emit(EXIT_CONSTRUCT_FAILED)

    if execution is not None and result.diagnostics.execution is not None:
        result.diagnostics.execution["importedSources"] = imported_sources
        if resolved_plan is not None:
            current = True
            sources = {f.path: f.digest for f in resolved_plan.source.files if f.role == "source"}
            imported_match = all(sources.get(item["file"]) == item["digest"] for item in imported_sources)
            try:
                verify_saved_plan(resolved_plan, args.workspace)
                current = imported_match
            except Exception:
                current = False
            if not imported_match:
                result.diagnostics.semantic_capture = None
                result.diagnostics.semantic_capture_error = "source_mismatch"
            result.diagnostics.execution.update(resolvedPlan=resolved_plan.model_dump(mode="json"),
                planDigest=plan_digest, planCurrent=current)
            if reuse_report is not None:
                from .spec import ReuseReport
                result.diagnostics.reuse = ReuseReport(**reuse_report)

    if capture_preparation is not None:
        current = True
        sources = {f.path: f.digest for f in capture_preparation.source.files if f.role == "source"}
        imported_match = all(sources.get(item["file"]) == item["digest"] for item in imported_sources)
        try:
            verify_saved_plan(capture_preparation, args.workspace)
            current = imported_match
        except Exception:
            current = False
        if not imported_match:
            result.diagnostics.semantic_capture = None
            result.diagnostics.semantic_capture_error = "source_mismatch"
        result.diagnostics.automatic_capture = dict(preparation=capture_preparation.model_dump(mode="json"),
            preparationDigest=preparation_digest, current=current, importedSources=imported_sources)

    if capture is not None and (
        capture_spec["activations"] or capture_spec["grads"] or args.kind == "train_step"
    ):
        result.diagnostics.warnings.extend(capture.warnings)
        result.diagnostics.warnings.extend(_batch_warning(result))
    result.diagnostics.worker = dict(envelope.worker)
    envelope.result = result.model_dump(mode="json")
    envelope.spans = _spanmap(result.graph)
    if capture is not None:
        _attach_stats(envelope.result["graph"], capture.stats)
    envelope.worker["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    envelope.result["diagnostics"]["worker"] = dict(envelope.worker)
    if series is not None:
        # Additive under `result`, next to `worker` and for the same reason: the five
        # envelope keys and protocol_version 1 are the contract (task.md §6.3), and a
        # protocol-1 reader that reads `result.diagnostics.loss` and the graph finds
        # both exactly where they were. Nothing it reads changed value, changed type
        # or moved; it simply never looks at this key.
        envelope.result["diagnostics"]["series"] = series

    if result.error:
        # The forward raised. The graph up to that point is real and the caller wants
        # it — the node the trace stopped at is exactly what the diagnostic points at.
        frames = [
            {"file": str(frame[0]), "line": int(frame[1]), "name": str(frame[2])}
            for frame in result.diagnostics.error_frames
            if len(frame) >= 3
        ]
        site = _error_site(frames, envelope.spans, roots=_roots_of(args))
        envelope.error = {
            "type": "TraceError",
            "message": result.error,
            "frames": frames,
            "node_id": _failure_node(result, site),
            "file": site.get("file"),
            "line": site.get("line"),
        }
        if result.diagnostics.failure_diagnostic is not None:
            from ..semantic.contracts import Diagnostic
            envelope.error["diagnostic"] = Diagnostic.model_validate(result.diagnostics.failure_diagnostic).model_dump(mode="json")
        if result.diagnostics.fake_tensor is not None:
            envelope.error["fakeTensor"] = dict(result.diagnostics.fake_tensor["limitation"], phase="forward")
        return envelope.emit(EXIT_TRACE_FAILED)
    return envelope.emit(EXIT_OK)


def _fake_limitation(envelope: _Envelope, limitation: Optional[Dict[str, str]], phase: str, trace_module: Any) -> None:
    """Mark a canonical fake-tier failure that is a fake-tensor limitation.

    ``error.fakeTensor`` is what the host keys its single real re-run on; the message is
    replaced by one that says the model needs real tensor values. Failures that are not
    a fake-tensor limitation keep the envelope exactly as ``_fail`` wrote it.
    """
    if limitation is None or envelope.error is None:
        return
    envelope.error["message"] = trace_module.fake_limitation_message(limitation)
    envelope.error["fakeTensor"] = dict(limitation, phase=phase)


def _failure_node(result: Any, site: dict) -> Optional[str]:
    evidence = result.diagnostics.run_evidence
    if not isinstance(evidence, dict):
        return _first_unreached(result) or site.get('node_id')
    failure = evidence.get('failure')
    if failure:
        owner = failure.get('modulePath')
    else:
        calls = [call for call in evidence.get('moduleCalls', [])
                 if call.get('disposition') in ('terminal', 'exception_chain')]
        owner = calls[-1].get('modulePath') if calls else None
    # A root-level functional failure need not have a materialised runtime node.
    # In particular, the next module may never have been called at all.
    if not owner:
        return None
    candidates = [str(node['id']) for node in result.graph.get('nodes', [])
                  if ((node.get('data') or {}).get('parameters') or {}).get('_module_path') == owner]
    return candidates[0] if len(candidates) == 1 else None


def _first_unreached(result: Any) -> Optional[str]:
    """The first node with no observed shape — where the trace stopped.

    Read off the graph rather than off ``diagnostics.unreached_modules``, which only
    lists modules the trace never *entered*. The module that raised was entered (its
    pre-hook fired) and simply never produced an output, so it is absent from that
    list and is precisely the node the reader must be sent to.
    """
    for node in (result.graph.get("nodes") or []):
        params = ((node.get("data") or {}).get("parameters")) or {}
        if params.get("_shape_provenance") == "unreached":
            return str(node.get("id"))
    return None


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
