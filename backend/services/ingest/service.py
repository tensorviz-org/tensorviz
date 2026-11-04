"""Ingestion orchestration: a ``ModelRef`` in, a flat ``ModelGraph`` out.

The tier ladder exists because tracing is not universally possible, and a silently
incomplete graph is worse than an honest one:

* **fake**      forward under ``FakeTensorMode`` — edges + observed shapes, 0 bytes.
  Survives 14 of 16 architectures.
* **real**      the same trace on real (small) CPU tensors. This is the escalation for
  data-dependent control flow: Mixture-of-Experts routing calls ``idx.tolist()``, which a
  fake tensor cannot answer.
* **tree-only** static introspection alone — complete hierarchy, parameter counts and
  repeat groups, but no dataflow edges and no shapes. Always works, never lies.

The tree is always built, and the trace only *decorates* it. That ordering is what keeps a
Mixture-of-Experts model honest: a trace visits 2 of 8 experts, so a trace-derived graph
would silently omit six of them, while the tree shows all eight (as one canonical block
with ``repeat_count=8``) and the untraced modules are marked ``unreached``.
"""

from __future__ import annotations

import ast
import os
import re
import sys
import time
import tokenize
from contextlib import contextmanager, nullcontext
from ..python_compat import chdir
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from . import inputs as input_search
from . import input_declarations
from ..validation.setup import package_root, ExecutionRequirement
from ..semantic.contracts import Diagnostic
from . import repo_scan as repo
from . import resolvers
from .introspect import apply_call_counts, build_tree
from .normalise import build_graph
from .spec import EntrypointCandidate, IngestDiagnostics, IngestResult, ModelRef
from .materialise import Plan, register, splittable_leaves
from .trace import resolve_dataflow, trace_model


class IngestError(RuntimeError):
    """A user-facing ingestion failure (bad reference, unresolvable arguments)."""


def ingest(ref: ModelRef) -> IngestResult:
    if ref.file and ref.cls:
        return ingest_source(ref)
    if ref.repo:
        return _ingest_repo(ref)
    if ref.library and ref.arch:
        return _ingest_library(ref)
    raise IngestError("provide either (library, arch) or repo")


# ────────────────────────────────────────────────────────────── Tier A ───────────
def _ingest_library(ref: ModelRef) -> IngestResult:
    name = ref.name or f"{ref.library}:{ref.arch}"
    timings: Dict[str, float] = {}
    started = time.perf_counter()

    with FakeTensorMode(allow_non_fake_inputs=True):
        model = resolvers.resolve(ref.library, ref.arch, ref.overrides)
        timings["build_ms"] = round((time.perf_counter() - started) * 1000, 1)
        recipe = input_search.recipe_for(ref.library, ref.arch, model)
        graph, diagnostics, _error = _introspect_and_trace(
            # A library recipe is tensorViz's own answer for that architecture —
            # ImageNet's 1x3x224x224 for torchvision, sixteen token ids for
            # transformers. Nobody asked for it, so it is reported as what it is.
            name, model, ref.depth, recipe, timings,
            recipe_provenance="assumed-default",
        )

    diagnostics.provenance = f"{ref.library} registry/config: {ref.arch}"
    diagnostics.timings_ms = timings
    return IngestResult(ok=True, name=name, graph=graph, diagnostics=diagnostics)


# ────────────────────────────────────────────────────────────── Tier B ───────────
def _ingest_repo(ref: ModelRef) -> IngestResult:
    started = time.perf_counter()
    scan = repo.scan_repo(ref.repo)
    ranked = repo.rank_entrypoints(scan)
    timings: Dict[str, float] = {"scan_ms": round((time.perf_counter() - started) * 1000, 1)}

    if not ranked:
        raise IngestError(f"no nn.Module subclasses found under {ref.repo}")
    pick = ref.entrypoint or ranked[0].name
    if pick not in scan.module_classes:
        raise IngestError(
            f"{pick!r} is not an nn.Module subclass in this repository; "
            f"candidates: {[c.name for c in ranked]}"
        )

    kind, expr, kwargs, provenance = repo.resolve_arguments(scan, pick)
    name = ref.name or f"{scan.root.rstrip('/').split('/')[-1]}:{pick}"
    if kind is None:
        # Honest partial result: the caller can pick another entrypoint or supply args.
        return IngestResult(
            ok=False, name=name,
            diagnostics=_empty_diagnostics(provenance, timings),
            candidates=ranked,
            error=f"could not determine constructor arguments for {pick!r}: {provenance}",
        )

    build_started = time.perf_counter()
    with FakeTensorMode(allow_non_fake_inputs=True):
        try:
            model, stubbed = repo.instantiate(
                scan, pick, kind, expr, {**kwargs, **ref.overrides}
            )
        except repo.StubbedDependencyError:
            raise
        except Exception as exc:
            # Name the rung that produced the arguments — an incomplete config is the
            # usual cause and the raw exception is often blank (minGPT's
            # get_default_config() deliberately leaves vocab_size unset, so GPT's own
            # assert fires with no message at all).
            raise IngestError(
                f"constructing {pick!r} failed with arguments from {provenance}: "
                f"{type(exc).__name__}: {exc or '<no message>'}. "
                "Pass explicit values via `overrides`, or choose another entrypoint."
            ) from exc
        timings["build_ms"] = round((time.perf_counter() - build_started) * 1000, 1)
        graph, diagnostics, _error = _introspect_and_trace(
            name, model, ref.depth, None, timings
        )

    diagnostics.provenance = provenance
    diagnostics.stubbed_dependencies = stubbed
    diagnostics.timings_ms = timings
    return IngestResult(
        ok=True, name=name, graph=graph, diagnostics=diagnostics, candidates=ranked
    )


# ─────────────────────────────────────────────────── the IDE worker path ─────────
#: The seed every ``tier='real'`` trace runs under, so that opening the same class
#: twice draws the same graph. Fixed rather than configurable: a data-dependent
#: branch is steered by *declaring the input* (``# tensorviz: input=torch.full(...)``),
#: which says which graph you want, and a ``--seed`` flag would offer a lottery
#: ticket instead. 0 matches the value ai/domain/candidate_validation.py already
#: seeds its own model work with.
_TRACE_SEED = 0


@contextmanager
def _seeded(seed: int | None = _TRACE_SEED):
    """Pin the global CPU generator for the duration, then put it back exactly.

    Three draws reach a real trace and all three change the drawn graph, so all
    three have to be inside this: the user's own ``__init__`` (``nn.Linear`` weights),
    the example input (``inputs.py`` and the ``2,8`` shorthand both build ``randn``
    tensors), and the forward itself. Only the first of those rules out the tidier
    fix — ``nn.Linear.__init__`` takes no ``generator`` argument, so a local
    ``torch.Generator`` cannot reach an arbitrary user constructor, and seeding the
    global stream is the only mechanism that does.

    ``default_generator.manual_seed`` rather than ``torch.manual_seed``, which
    "sets the seed for generating random numbers on all devices" while
    ``fork_rng(devices=[])`` saves only the CPU one — measured on this machine, the
    pair leaves MPS reseeded after the block exits. Weight init and ``randn`` on CPU
    draw from ``default_generator``, so seeding it alone is both sufficient and the
    only part the fork can hand back.

    The restore is not a nicety. ``ingest_source`` runs in-process — the FastAPI
    server calls it (api/routers/ingest.py) and the suite calls it in the same
    process as 1600 other tests — so a bare ``manual_seed`` would make an unrelated
    caller's stream depend on whether somebody had opened a graph first.
    """
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.default_generator.manual_seed(seed)
        yield


def parse_input_spec(value: str) -> Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
    """``"2,8"`` or ``"2,8;2"`` -> random tensors of those shapes; anything else is
    evaluated as a Python expression in a namespace holding ``torch``.

    Both forms are needed. Shapes cover the common case in one flag; the expression
    is the only way to say ``torch.randint(0, 50257, (1, 16))``, and a language model
    fed floats fails inside its embedding with an error about the wrong dtype rather
    than about the input.
    """
    text = value.strip()
    if not text:
        return None
    shapes = [part.strip() for part in text.split(";") if part.strip()]
    if all(re.fullmatch(r"\d+(\s*,\s*\d+)*", shape) for shape in shapes):
        tensors = tuple(
            torch.randn(*[int(dim) for dim in shape.split(",")]) for shape in shapes
        )
        return tensors, {}
    namespace: Dict[str, Any] = {"torch": torch}
    value_ = eval(text, namespace)  # noqa: S307 - the user's own worker, their own file
    if isinstance(value_, tuple):
        return value_, {}
    return (value_,), {}


def _scoped_input(source: str, cls: Optional[str]) -> Optional[str]:
    """The ``# tensorviz: input=`` value that belongs to ``cls``, or None.

    A bare ``search()`` takes the first match in the file and hands it to whichever
    class is being traced. In a file with two models that is simply the wrong tensor:
    ``custom/net.py`` declares ``2,16`` for ``Custom`` and ``Transposed`` needs a 3-D
    input, so ``Transposed`` was fed its file-mate's shape and died in ``transpose``.

    ``input_declarations`` holds the one rule setup and the drawing follow too: in a
    file with several model classes, a class owns the declaration directly above or
    inside it, and one elsewhere belongs to none of them. It is imported relatively
    and is stdlib-only, never ``utils.code_parser``: that module would pull
    ``models.graph_models`` into the worker, which runs with the USER's repository
    first on ``sys.path``, and a workspace ``models/`` package would shadow it. The
    worker has no module index, so it proves which classes are modules from this
    file's own imports, as the index does for a file on its own, and counts a
    subclass of a model class in this file, whose base another file may prove.
    """
    try:
        own, _stray = input_declarations.applicable(source, cls)
    except (SyntaxError, tokenize.TokenError, RecursionError):
        # A buffer mid-edit still deserves whatever the file declares, so fall back
        # to the file-wide search rather than losing the hint. Source nested too deep
        # for ast to build cannot be imported either; the import should say so, not a
        # RecursionError from here.
        match = input_declarations.INPUT_COMMENT.search(source)
        return match.group("value") if match else None
    return own[0][0] if own else None


#: What a construction failure should tell the user to DO. Every failure used to end
#: with "Pass constructor arguments via `overrides`.", which is the right advice for
#: exactly one of the causes below and nonsense for the rest — a circular import is
#: not fixed by an argument, and neither is a class that constructs itself.
_MISSING_ARGS = re.compile(
    r"missing \d+ required (?:positional|keyword-only) arguments?: (?P<names>.+)"
)


def construction_reason(exc: BaseException) -> str:
    """A stable, non-sensitive code for why a model could not be built.

    Returned to the extension so the page can say the right sentence without parsing
    a Python message — and carrying no user source, paths or values.
    """
    text = str(exc)
    if isinstance(exc, ExecutionRequirement):
        return exc.diagnostic["code"]
    if isinstance(exc, RecursionError):
        return "recursive_construction"
    if isinstance(exc, ImportError):
        if "partially initialized module" in text or "circular import" in text:
            return "circular_import"
        return "import_failed"
    if isinstance(exc, TypeError) and _MISSING_ARGS.search(text):
        return "missing_argument"
    return "construct_failed"


def _render_kwargs(kwargs: Dict[str, Any]) -> str:
    """``hidden=128, act='relu'`` — Python source, because it is going into a Python file."""
    return ", ".join(f"{name}={value!r}" for name, value in kwargs.items())


def _construction_advice(
    cls: str,
    exc: BaseException,
    inside: Optional[Tuple[str, str]] = None,
    proposal: Optional[repo.ArgumentProposal] = None,
) -> str:
    """One actionable sentence per reason, rather than one sentence for all of them.

    ``inside`` is ``(model, argument)`` when ``cls`` is not the model but a config
    object being built for the model's declaration. The line to fix is then that
    declaration with the config call filled in — a new line naming the config class
    would be attributed to IT by ``declared_init`` and never read for the model.

    ``proposal`` is what the repository already passes to this constructor. It fills
    the skeleton's blanks and is ATTRIBUTED, so the user ratifies a value they can
    check rather than one that appeared from nowhere — and it is never constructed
    from: the shipped counter-case is ``examples/tensorviz-sweep``'s ``BasicBlock``,
    whose author declared a seventh value matching none of its six call sites.
    """
    reason = construction_reason(exc)
    if reason == "missing_argument":
        match = _MISSING_ARGS.search(str(exc))
        names = match.group("names") if match else "its arguments"
        missing = exc.diagnostic["names"] if isinstance(exc, ExecutionRequirement) else [
            name.strip().strip(chr(39))
            for name in names.replace(" and ", ",").split(",")
            if name.strip()
        ]
        if isinstance(exc, ExecutionRequirement):
            names = ", ".join(missing) or "its arguments"
        # A contested proposal fills nothing. The blank is what the user has to look at
        # to choose, and the line they accept carries no ranking with it.
        filled = (
            {k: v for k, v in proposal.kwargs.items() if k in missing}
            if proposal is not None and not proposal.alternatives
            else {}
        )
        skeleton = ", ".join(
            f"{name}={filled[name]!r}" if name in filled else f"{name}="
            for name in missing
        )
        call, where = f"{cls}({skeleton})", "above the class"
        if inside is not None:
            call, where = f"{inside[0]}({inside[1]}={call})", "in the init declaration"
        if filled:
            return (
                f"{cls} needs {names}. This repository already builds it with "
                f"{_render_kwargs(filled)} ({proposal.provenance}); declare that {where} "
                f"with `# tensorviz: init={call}` — change the value if it is not the "
                "one you mean — then validate again."
            )
        if proposal is not None:
            ranked = [(proposal.kwargs, proposal.provenance), *proposal.alternatives]
            return (
                f"{cls} needs {names}. This repository builds it {len(ranked)} different "
                "ways and tensorViz will not choose between them: "
                + "; ".join(f"{_render_kwargs(kw)} ({prov})" for kw, prov in ranked)
                + f". Declare the one you mean {where} with `# tensorviz: init={call}`, "
                "then validate again."
            )
        return (
            f"{cls} needs {names}. Declare them {where} with "
            f"`# tensorviz: init={call}`, then validate again."
        )
    if reason == "circular_import":
        return (
            "Two of its modules import each other at module scope, so neither can be "
            "imported first. Move one of the two imports inside the function that uses "
            "it. No constructor argument can fix this."
        )
    if reason == "recursive_construction":
        return (
            f"{cls} constructs itself, directly or through another class, so building "
            "it never terminates. No constructor argument can fix this."
        )
    if reason == "import_failed":
        return (
            "Its module could not be imported. Check that the dependency is installed "
            "in the selected interpreter."
        )
    return "Check the explicit model profile and the underlying exception in tensorViz Output."


def _input_from_comment(
    file: str, cls: Optional[str] = None
) -> Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
    try:
        # "-sig" drops the BOM of a file saved as UTF-8 with BOM. Python imports such a
        # file, but ast refuses the BOM, and source ast refuses gets the file-wide search.
        with open(file, encoding="utf-8-sig", errors="ignore") as handle:
            source = handle.read()
    except OSError:
        return None
    value = _scoped_input(source, cls)
    if value is None:
        return None
    return parse_input_spec(value)


#: ``# tensorviz: init=Net(hidden=8, depth=2)``. Duplicated from ``utils.code_parser``,
#: which the worker may not import for the reason ``_scoped_input`` above gives, and
#: guarded by a drift test (``test_ide_init_comment``), which compares both the pattern
#: and what the two readers do with a table of declarations.
_INIT_COMMENT = re.compile(r"#\s*tensorviz:\s*init\s*=\s*(?P<value>[^\n]+)")


class ConfigCall(NamedTuple):
    """``ModelConfig()`` written as the VALUE of an init declaration argument.

    The commonest real ``__init__`` takes a config OBJECT, and no literal describes
    one: ``Net(cfg=8)`` is not what the user means, and JSON — which is what
    ``--overrides`` is — has no spelling for "a default ModelConfig". So the call is
    carried, unevaluated, to the one tier that can build it: the worker, which has
    just imported the user's module and is about to run the model's own constructor
    out of it anyway.

    A plain ``NamedTuple`` on purpose. ``utils.code_parser`` holds its own copy of
    this grammar (see ``_INIT_COMMENT`` above for why it may not import this module),
    and tuples compare by value, so its copy and this one are equal without either
    importing the other — which is what ``test_ide_init_comment``'s drift guard
    checks.
    """

    name: str
    kwargs: Dict[str, Any]


def _read_config_call(node: ast.Call, argument: str) -> Tuple[Optional[ConfigCall], Optional[str]]:
    """``ModelConfig(width=8)`` -> a ``ConfigCall``; see the parser's twin.

    One call, on a bare name, with literal keyword arguments. Not a general
    expression evaluator: a dotted name, a positional argument or a call nested
    inside this one is refused with the reason, because each of those is a step
    towards evaluating arbitrary source to decide how to run the user's source.
    """
    if not isinstance(node.func, ast.Name):
        return None, (
            f"`{argument}` must name a class to build, e.g. `{argument}=ModelConfig()`. "
            "Import the name into this file; tensorViz will not resolve a dotted path."
        )
    if node.args:
        return None, (
            f"Name the arguments: write `{argument}={node.func.id}(width=8)`, "
            "not positional values."
        )
    fields: Dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            return None, "Name the arguments: `**kwargs` cannot be read from a comment."
        try:
            fields[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError):
            return None, (
                f"`{keyword.arg}` must be a literal value (a number, string, bool, list, "
                f"tuple or dict). tensorViz builds one {node.func.id}(...), not an expression."
            )
    return ConfigCall(node.func.id, fields), None


def _read_init_call(value: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """``"Net(hidden=8)"`` -> ``("Net", {"hidden": 8}, None)``; see the parser's twin.

    Keyword arguments whose values are literals, or one constructor call to build the
    config object a real ``__init__`` takes. ``ast`` is stdlib, so reading the line
    here costs this file no new import and cannot be shadowed by the user's repository.
    """
    try:
        expr = ast.parse(value.strip(), mode="eval").body
    except SyntaxError:
        return "", {}, "tensorViz could not read this init declaration. Write it as a call, e.g. `# tensorviz: init=Net(hidden=8)`."
    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
        return "", {}, "An init declaration must be a call on the class name, e.g. `# tensorviz: init=Net(hidden=8)`."
    cls = expr.func.id
    if expr.args:
        return cls, {}, f"Name the arguments: write `# tensorviz: init={cls}(hidden=8)`, not positional values."
    overrides: Dict[str, Any] = {}
    for keyword in expr.keywords:
        if keyword.arg is None:
            return cls, {}, "Name the arguments: `**kwargs` cannot be read from a comment."
        if isinstance(keyword.value, ast.Call):
            call, error = _read_config_call(keyword.value, keyword.arg)
            if error is not None:
                return cls, {}, error
            overrides[keyword.arg] = call
            continue
        try:
            overrides[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError):
            return cls, {}, (
                f"`{keyword.arg}` must be a literal value (a number, string, bool, list, "
                "tuple or dict) or a call that builds one, e.g. "
                f"`{keyword.arg}=ModelConfig()`. tensorViz will not evaluate an arbitrary "
                "expression to build your model."
            )
    return cls, overrides, None


def declared_init(source: str, cls: Optional[str] = None) -> Optional[Tuple[str, Dict[str, Any], Optional[str]]]:
    """The declaration ``source`` carries for ``cls``, or ``None``."""
    for match in _INIT_COMMENT.finditer(source or ""):
        name, overrides, error = _read_init_call(match.group("value"))
        # See the parser's twin: an unattributable line is reported, not filtered.
        if cls is not None and name and name != cls:
            continue
        return name, overrides, error
    return None


def _init_from_comment(file: str, cls: str) -> Dict[str, Any]:
    """Constructor kwargs the file declares for ``cls``; ``{}`` when it declares none.

    A declaration carrying an ``error`` yields ``{}`` on purpose: the construction then
    fails the same way it would have without the line, and the editor is the surface
    that tells the user why their line was not used (P0's diagnostic). Guessing past a
    refused declaration would build a model the user did not ask for.
    """
    try:
        with open(file, encoding="utf-8", errors="ignore") as handle:
            found = declared_init(handle.read(), cls)
    except OSError:
        return {}
    if found is None or found[2] is not None:
        return {}
    return found[1]


def _propose_init(
    file: str, cls: str, root: str, exc: BaseException
) -> Optional[repo.ArgumentProposal]:
    """Constructor arguments the repository already supplies for ``cls``, or ``None``.

    Deliberately lazy and deliberately narrow. ``scan_repo`` walks every ``.py`` under
    the workspace — ~935 ms over 387 files — and this path runs on a SAVE, so it must
    cost nothing for the classes that construct, which is nearly all of them. Two gates
    ensure that: the failure has to be about a missing argument, and the file must not
    already answer the question.

    ``declared_init`` rather than ``_init_from_comment``: the latter returns ``{}`` both
    for "no declaration" and for "a declaration we refused", so anything chained onto it
    would answer a user who wrote ``# tensorviz: init=Net(8)`` with somebody else's
    value — burying the one diagnostic that tells them to name their arguments. A
    declaration of any kind is terminal here: the fix is the line they already wrote.
    """
    if construction_reason(exc) != "missing_argument":
        return None
    try:
        with open(file, encoding="utf-8", errors="ignore") as handle:
            if declared_init(handle.read(), cls) is not None:
                return None
    except OSError:
        return None
    return repo.propose_arguments(repo.scan_repo(root), cls)


def _config_builder(cls: str):
    """A ``repo.instantiate`` resolver that turns a ``ConfigCall`` into an object.

    The name is looked up on the module that defines the model, which is the module
    the user wrote the declaration in: a config class defined beside the model, or
    imported at its top with ``from .config import ModelConfig``, is bound there
    either way. A dotted path is not resolved — see ``_read_config_call``.

    Every failure here is an ``IngestError`` with advice aimed at the CONFIG, not at
    the model. A config that itself needs an argument is the case worth getting
    right: the user has written a declaration, it is very nearly correct, and
    "constructing Net failed" would send them back to the line they already wrote.
    """

    def resolve(argument: str, value: Any, module: Any) -> Any:
        if not isinstance(value, ConfigCall):
            return value
        factory = getattr(module, value.name, None)
        if factory is None:
            raise IngestError(
                f"the init declaration for {cls} builds `{value.name}(...)`, but "
                f"{value.name} is not defined or imported in {module.__name__}. "
                f"Import it there, or name a class that is."
            )
        try:
            return factory(**value.kwargs)
        except Exception as exc:
            raise IngestError(
                f"building `{value.name}(...)` for {cls} failed: "
                f"{type(exc).__name__}: {exc or '<no message>'}. "
                f"{_construction_advice(value.name, exc, inside=(cls, argument))}"
            ) from exc

    return resolve


def _construction_fake_mode(tier: str, execution: Optional[Dict[str, Any]]) -> Optional[FakeTensorMode]:
    """The one ``FakeTensorMode`` a fake-tier run builds its model, inputs and forward in.

    Only the fake tier without an explicit execution has one: building the model inside
    the mode allocates no parameter storage at all (``llama_1b`` would otherwise
    materialise 3.4 GB of weights only to trace their shapes). The real tier, explicit
    execution (which is real-only) and a caller that is already inside a fake mode (the
    library and repository importers) get None and run exactly as before.
    """
    if tier != "fake" or execution is not None:
        return None
    from .trace import fake_mode_active
    if fake_mode_active():
        return None
    return FakeTensorMode(allow_non_fake_inputs=True)


def ingest_source(
    ref: ModelRef,
    *,
    tier: str = "real",
    example_input: Optional[str] = None,
    tensor_inputs: Optional[List[Dict[str, Any]]] = None,
    execution: Optional[Dict[str, Any]] = None,
    diagnostic_job: Optional[Dict[str, Any]] = None,
    workspace: Optional[str] = None,
    roots: Optional[Sequence[str]] = None,
    capture_frames: bool = True,
    resolved_plan: Any = None,
    capture_preparation: Any = None,
) -> IngestResult:
    """Ingest one class the caller names by file (task.md M4).

    The difference from Tier B is that nothing is guessed. The user clicked "Open as
    Graph" on one ``class`` line, so there is no entrypoint to rank and no
    constructor ladder to climb: the class is constructed with ``overrides`` and
    nothing else, and a class that needs an argument says so instead of being
    silently built from somebody else's config.

    ``tier='real'`` by default, which is the whole point of running in the user's
    interpreter: ``if x.sum() > 0`` takes a branch instead of raising, and the graph
    describes what Python actually did.
    """
    if not ref.file or not ref.cls:
        raise IngestError("ingest_source needs both `file` and `cls`")
    file = os.path.abspath(os.path.expanduser(ref.file))
    if not os.path.isfile(file):
        raise IngestError(f"{ref.file} is not a file")

    if execution is not None and execution.get('framework') == 'jax':
        from ..jax_support.service import ingest_jax
        return ingest_jax(ref, execution, workspace, resolved_plan, diagnostic_job)

    search_roots = [os.path.abspath(root) for root in (roots or [])] or [package_root(file)]
    timings: Dict[str, float] = {}
    started = time.perf_counter()
    path_before = list(sys.path)
    capture_sources = None
    fake_mode = _construction_fake_mode(tier, execution)

    scan = repo.RepoScan(search_roots[0])
    # sys.path is prepended by the repository importer and never unwound there, which
    # is harmless in a throwaway worker and a leak anywhere else. Unwinding it in a
    # finally is what makes this function safe to call in-process — the tests do, in
    # the same process as 1600 others.
    try:
        source_setup = resolved_plan if resolved_plan is not None else capture_preparation
        if (tier == "real" or tier == "fake" and capture_preparation is not None) and workspace and source_setup is not None:
            from ..semantic.source_bindings import CapturedSourceEvidence
            capture_sources = CapturedSourceEvidence(source_setup.source, workspace)
        if execution is not None:
            from .execution_spec import validate_execution_spec, workspace_directory
            execution = validate_execution_spec(execution)
            if not workspace:
                raise ValueError("Explicit execution requires a workspace.")
            directory = workspace_directory(workspace, execution['cwd'])
            # A saved profile names its model relative to cwd, including namespace
            # packages. Use that same identity in capture and its setup factory.
            import_root = directory if resolved_plan is not None and resolved_plan.origin == 'profile' else None
            directory_context = chdir(directory)
        else:
            directory_context = chdir(workspace) if capture_preparation is not None else nullcontext()
        with _seeded(None if execution is not None else _TRACE_SEED), directory_context:
            try:
                with capture_sources.construction.observe() if capture_sources is not None else nullcontext():
                    overrides = dict(ref.overrides) or _init_from_comment(file, ref.cls)
                    if diagnostic_job is not None and execution is None:
                        raise ValueError("A diagnostic job requires explicit execution setup.")
                    prepared = None
                    if execution is not None:
                        from .execution_setup import prepare_execution
                        if not workspace or tier != "real" or tensor_inputs is not None or example_input is not None:
                            raise ValueError("Explicit execution requires a workspace, real tier and no legacy input.")
                        if diagnostic_job is not None:
                            from .execution_spec import validate_execution_job
                            validate_execution_job(diagnostic_job)
                        prepared = prepare_execution(file, ref.cls, workspace, execution, overrides, import_root=import_root)
                        model, stubbed = prepared.model, []
                    else:
                        model, stubbed = repo.instantiate(
                            scan, ref.cls, "class", ref.cls, overrides, file=file,
                            resolve=_config_builder(ref.cls), construction_mode=fake_mode,
                        )
            except repo.StubbedDependencyError:
                raise
            except IngestError:
                # The config builder already said which part failed and what to write;
                # re-wrapping would bury that under "constructing Net failed" and the
                # generic advice for an unclassified exception.
                raise
            except Exception as exc:
                # Explicit `overrides` is the caller having already answered; proposing a
                # different value over the top of it would be arguing with the request.
                proposal = (
                    None if ref.overrides
                    else _propose_init(file, ref.cls, search_roots[0], exc)
                )
                raise IngestError(
                    f"constructing {ref.cls!r} from {ref.file} failed: "
                    f"{type(exc).__name__}: {exc or '<no message>'}. "
                    f"{_construction_advice(ref.cls, exc, proposal=proposal)}"
                ) from exc
            timings["build_ms"] = round((time.perf_counter() - started) * 1000, 1)

            # The fake tier's inputs and forward run under the SAME mode the model was
            # built in, so its parameters, its inputs and every intermediate are fake
            # tensors of one mode. Nothing to enter for the real tier.
            with fake_mode if fake_mode is not None else nullcontext():
                recipe = (prepared.args, prepared.kwargs) if prepared else None
                recipe_provenance = "explicit" if prepared else None
                if tensor_inputs is not None:
                    from .tensor_inputs import make_tensor_inputs

                    recipe = make_tensor_inputs(tensor_inputs)
                    recipe_provenance = "explicit"
                elif example_input:
                    recipe = parse_input_spec(example_input)
                    recipe_provenance = "explicit"
                if recipe is None:
                    recipe = _input_from_comment(file, ref.cls)
                    recipe_provenance = "declared" if recipe is not None else None

                name = ref.name or ref.cls
                semantic_context = None
                if resolved_plan is not None and prepared is not None:
                    from ..semantic.recorder import CaptureContext
                    from ..semantic.execution_plan import applicability_digest
                    semantic_context = CaptureContext(resolved_plan.source, applicability_digest(resolved_plan),
                        resolved_plan.id, resolved_plan.captureLimits,
                        prepared.evidence['inputs']['digest'], prepared.evidence['state']['digest'], source_evidence=capture_sources)
                graph, diagnostics, trace_error = _introspect_and_trace(
                    name,
                    model,
                    ref.depth,
                    recipe,
                    timings,
                    tier=tier,
                    frame_roots=tuple(search_roots) if capture_frames else (),
                    workspace_roots=tuple(search_roots),
                    strict_recipe=True,
                    seed_inputs=True,
                    recipe_provenance=recipe_provenance,
                    preserve_mode=prepared is not None,
                    grad_enabled=execution['gradEnabled'] if execution is not None else False,
                    semantic_context=semantic_context,
                    capture_preparation=capture_preparation,
                    capture_sources=capture_sources,
                )
            if prepared is not None:
                diagnostics.execution = prepared.evidence

    finally:
        sys.path[:] = path_before
        if capture_sources is not None:
            capture_sources.clear()
    diagnostics.provenance = f"{ref.file}::{ref.cls}"
    diagnostics.stubbed_dependencies = stubbed
    diagnostics.seed = diagnostics.execution["seed"] if diagnostics.execution is not None else _TRACE_SEED
    diagnostics.timings_ms = timings
    return IngestResult(
        ok=True, name=name, graph=graph, diagnostics=diagnostics, error=trace_error
    )


def scan_only(repo_path: str) -> List[EntrypointCandidate]:
    """Rank a repository's entrypoints without importing or executing anything."""
    return repo.rank_entrypoints(repo.scan_repo(repo_path), limit=5)


# ───────────────────────────────────────────────────────────── internals ─────────
def _introspect_and_trace(
    name: str,
    model: torch.nn.Module,
    depth: int,
    recipe: Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]],
    timings: Dict[str, float],
    *,
    tier: str = "fake",
    frame_roots: Sequence[str] = (),
    workspace_roots: Sequence[str] = (),
    strict_recipe: bool = False,
    seed_inputs: bool = False,
    recipe_provenance: Optional[str] = None,
    preserve_mode: bool = False,
    grad_enabled: bool = False,
    semantic_context: Any = None,
    capture_preparation: Any = None,
    capture_sources: Any = None,
) -> Tuple[Dict[str, Any], Any, Optional[str]]:
    """Build the tree, then try to decorate it with a trace. Never raises for a failed
    trace — degrades to tree-only.

    ``strict_recipe`` is the IDE's mode: the caller supplied the input, so a forward
    that raises is a fact to report at that line, not a reason to go and try eighty
    other shapes. The third return value is that trace error, or None.

    ``recipe_provenance`` says where ``recipe`` came from, because only the caller
    knows: a file's own declaration and a ``--input`` flag arrive here as the same
    tuple of tensors. When the recipe is absent or does not survive, the search
    below answers for itself, and the answer distinguishes the seed read off the
    model's first layer from the eighty-candidate brute force behind it — both are
    guesses, but only one of them is an informed one.

    ``workspace_roots`` is separate from ``frame_roots`` on purpose: which classes are
    the USER's is a fact about the workspace, and it must not change because the caller
    turned per-op frame capture off (task.md M6)."""
    # Canonical whenever the caller prepared a canonical capture. The real tier always
    # could; the fake tier only reaches here with a capture preparation since the
    # worker started accepting one for it (fake-tier slice 2). The legacy fake run
    # (no preparation) still draws the folded graph below.
    canonical = semantic_context is not None or capture_preparation is not None
    intro = None if canonical else build_tree(model, depth)

    traced = None
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = {}
    tried = 0
    provenance: Optional[str] = None
    trace_started = time.perf_counter()

    def context_for(args, kwargs, candidate, provenance):
        if capture_preparation is None:
            return semantic_context
        from ..semantic.recorder import automatic_context
        return automatic_context(capture_preparation, model, args, kwargs, candidate, provenance, capture_sources)

    if recipe is not None:
        args, kwargs = recipe
        tried = 1
        provenance = recipe_provenance
        traced = trace_model(model, args, kwargs, tier=tier, frame_roots=frame_roots,
                             preserve_mode=preserve_mode, record_evidence=preserve_mode,
                             grad_enabled=grad_enabled, semantic_context=context_for(args, kwargs, 1, provenance))
        if (traced.error or not traced.io) and not strict_recipe:
            traced = None
            provenance = None

    if traced is None:
        arity = input_search.required_arity(model)
        if arity is None:
            raise ExecutionRequirement("binding", "unknown_signature", ["inputs"], "provide_inputs")
        if arity > 3:
            raise ExecutionRequirement("binding", "missing_argument", ["inputs"], "provide_inputs")
        # `candidates_for` puts the model's own seeds first, and only for a
        # single-argument forward; past that count the tuples are the generic pool.
        seeded = len(input_search.seeds_from_model(model)) if arity == 1 else 0
        for tried, factories in enumerate(
            input_search.candidates_for(model, arity)[: input_search.MAX_CANDIDATES], start=1
        ):
            attempt_args = tuple(make() for make in factories)
            attempt = trace_model(
                model, attempt_args, {}, tier=tier, frame_roots=frame_roots,
                semantic_context=context_for(attempt_args, {}, tried, _searched(tried, seeded))
            )
            if attempt.error is None and attempt.io:
                traced, args, kwargs = attempt, attempt_args, {}
                provenance = _searched(tried, seeded)
                break
            if attempt.fake_limitation is not None:
                # The model needs tensor values; no other input shape changes that.
                traced, args, kwargs = attempt, attempt_args, {}
                provenance = "failed-candidate"
                break
            # A partial trace from a data-dependent failure is still the best result for
            # that input, so keep the deepest one we saw.
            if traced is None or len(attempt.io) > len(traced.io):
                traced, args, kwargs = attempt, attempt_args, {}
                provenance = "failed-candidate"
    timings["trace_ms"] = round((time.perf_counter() - trace_started) * 1000, 1)
    timings["input_candidates_tried"] = tried
    if capture_preparation is not None and traced is not None and traced.semantic_capture is not None:
        traced.semantic_capture["automatic"]["candidatesTried"] = tried

    if canonical:
        # The saved source lane and canonical projector own this graph. Building
        # the old folded graph here would duplicate work and reintroduce its
        # unbounded op/reuse buffers even when canonical limits were exhausted.
        payload = traced.semantic_capture if traced is not None else None
        diagnostics = IngestDiagnostics(
            input_provenance=provenance, tier=tier,
            aten_ops=len(payload["operations"]) if payload is not None else 0,
            example_input=[list(t.shape) for t in (*args, *kwargs.values()) if hasattr(t, "shape")],
            error_frames=[list(frame) for frame in traced.error_frames] if traced is not None else [],
            run_evidence=traced.evidence if traced is not None else None,
            semantic_capture=payload,
            semantic_capture_error=traced.semantic_error if traced is not None else "recorder_error",
            failure_diagnostic=Diagnostic.model_validate(traced.diagnostic) if traced is not None and traced.diagnostic else None,
        )
        if traced is not None and traced.fake_limitation is not None:
            diagnostics.fake_tensor = dict(limitation=traced.fake_limitation)
        if traced is not None and traced.block_templates is not None:
            diagnostics.block_templates = traced.block_templates
        return {"nodes": [], "edges": []}, diagnostics, traced.error if traced is not None else None

    if traced is not None and traced.call_counts:
        apply_call_counts(intro, traced.call_counts)
    if traced is not None:
        _attach_frames(intro, traced.ops)

    if traced is None or not traced.io:
        graph, diagnostics = build_graph(
            name, intro, {}, {}, set(), set(), _entry_shape((*args, *kwargs.values())), roots=workspace_roots,
            input_provenance=provenance,
        )
        diagnostics.input_provenance = provenance
        diagnostics.tier = "tree-only"
        diagnostics.run_evidence = traced.evidence if traced is not None else None
        diagnostics.semantic_capture = traced.semantic_capture if traced is not None else None
        diagnostics.semantic_capture_error = traced.semantic_error if traced is not None else None
        diagnostics.failure_diagnostic = Diagnostic.model_validate(traced.diagnostic) if traced is not None and traced.diagnostic else None
        diagnostics.warnings.append(
            "no example input produced a forward pass; hierarchy only, no edges or shapes"
        )
        return graph, diagnostics, (traced.error if traced is not None else None)

    # Materialise what the fold would otherwise discard: a node for each functional
    # merge, and one node per call site of a reused stateless leaf. Without this an
    # imported ResNet is CYCLIC (relu -> conv2 -> bn2 -> relu) and every model with a
    # skip connection fails the runtime arity check. See services/ingest/materialise.py.
    splittable, refused = splittable_leaves(intro, traced.reuse_scopes)
    rendered_set = set(intro.rendered)
    plan = Plan(
        splittable=splittable,
        flatten=True,
        taken=frozenset(rendered_set),
        # A module that is rendered but whose children are not: its internal merge is
        # real structure that would otherwise fan into the card itself.
        composites=frozenset(
            path
            for path in rendered_set
            if not intro.nodes[path].is_leaf
            and not any(intro.parent.get(child) == path for child in rendered_set)
        ),
    )
    seeds = (
        {identity: "input_node" for identity in traced.input_ids}
        if seed_inputs
        else None
    )
    pairs = resolve_dataflow(traced.ops, set(intro.rendered), intro.parent, plan, seeds)
    register(intro, plan, traced.io, traced.visited)
    synthetic_params = {
        node_id: dict(syn.params) for node_id, syn in plan.nodes.items() if syn.params
    }
    graph, diagnostics = build_graph(
        name, intro, traced.io, traced.call_counts, pairs, traced.visited,
        _entry_shape((*args, *kwargs.values())), synthetic_params, roots=workspace_roots,
        input_provenance=provenance,
    )
    diagnostics.input_provenance = provenance
    diagnostics.materialised_ops = sorted(
        n for n, s in plan.nodes.items() if s.kind in ("merge", "flatten", "pool")
    )
    diagnostics.split_call_sites = sorted(
        n for n, s in plan.nodes.items() if s.kind == "split"
    )
    diagnostics.refused_call_sites = refused
    diagnostics.tier = tier
    diagnostics.aten_ops = len(traced.ops)
    diagnostics.frames_captured = traced.frames_captured
    diagnostics.example_input = [list(t.shape) for t in (*args, *kwargs.values()) if hasattr(t, "shape")]
    diagnostics.error_frames = [list(frame) for frame in traced.error_frames]
    diagnostics.run_evidence = traced.evidence
    diagnostics.semantic_capture = traced.semantic_capture
    diagnostics.semantic_capture_error = traced.semantic_error
    diagnostics.failure_diagnostic = Diagnostic.model_validate(traced.diagnostic) if traced.diagnostic else None
    if traced.error:
        diagnostics.warnings.append(
            f"trace stopped early ({traced.error}); modules after that point are marked "
            "unreached and carry no observed shape"
        )
    return graph, diagnostics, traced.error


def _attach_frames(intro: Any, ops: List[Dict[str, Any]]) -> None:
    """Give every rendered module the user frame its FIRST op ran in.

    First, not last: a module entered once runs many aten ops and they all share the
    call line, while a module the user calls twice in one forward reports the line it
    was reached from first — which is where the reader expects the card to point.
    Modules whose ops carried no user frame (everything a library owns) keep None,
    which the normaliser then omits rather than inventing.
    """
    for op in ops:
        frame = op.get("frame")
        owner = op.get("owner") or ""
        node = intro.nodes.get(owner)
        if node is None or node.decl_file is not None or not frame:
            continue
        node.decl_file, node.decl_line = str(frame[0]), int(frame[1])


def _searched(tried: int, seeded: int) -> str:
    """Which half of the candidate list the winning tuple came out of.

    ``candidates_for`` returns the model's own seeds first and the generic pool
    after them, so the 1-based position of the winner is the whole distinction —
    and it matters, because a seed states the model's real width while a pool hit
    is a shape that merely happened not to raise.
    """
    return "seed-from-first-layer" if tried <= seeded else "search-candidate"


def _entry_shape(args: Tuple[Any, ...]) -> Optional[List[int]]:
    for value in args:
        if hasattr(value, "shape"):
            return [int(d) for d in value.shape]
    return None


def _empty_diagnostics(provenance: str, timings: Dict[str, float]):
    from .spec import IngestDiagnostics

    return IngestDiagnostics(tier="tree-only", provenance=provenance, timings_ms=timings)
