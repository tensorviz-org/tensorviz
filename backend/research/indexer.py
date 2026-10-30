"""Python sources in, one typed graph per experiment out. Nothing is ever executed.

The extension supplies `{relative path: source text}` — dirty editor buffers included
— and gets back a `ProjectIndex`. There is no `os.walk` here and no `open()`: the
workspace belongs to the extension, which is the only component that knows about
trust, unsaved buffers and `.gitignore`.

Three decisions carry most of the behaviour.

**The project root is derived, not configured.** `python demo/exp/train.py` puts
`demo/exp` on `sys.path`, so `data.toy` inside it means `demo/exp/data/toy.py`, not
`data/toy.py` at the workspace root. `_resolve_import` tries the importer's own
directory first and walks up, and the prefix that finally resolved *is* that
entrypoint's root. A monorepo separates into projects for free, with no marker files
and nothing for the user to declare.

**Nothing disappears.** A file that does not parse is still a `file` node carrying
its error. A file no experiment imports is listed as unreached. A class nobody can
classify is `generic`, never dropped. The user can see these files in their editor;
a graph that silently omits them is lying by omission.

**Declared beats discovered.** An SDK `tensorviz.experiment("name")` outranks a
`__main__` block, which outranks a filename that merely looks like an entrypoint —
that last one is a *candidate* the user should confirm rather than an assertion.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from research import identity
from research.classify import (
    DEFAULT_CLASSIFIER,
    _trains,
    BodyScan,
    Classification,
    Classifier,
    UsageSite,
    dotted,
    literal,
    scan_body,
)
from services.ingest.input_declarations import applicable
from research.contracts import (
    DATAFLOW_RELATIONSHIPS,
    Experiment,
    ExperimentStatus,
    ProjectIndex,
    ResearchEdge,
    ResearchGraph,
    ResearchNode,
    SourceRef,
    UnreachedFile,
)

#: Filenames that suggest an entrypoint without proving one. Anchored on whole
#: words rather than prefixes: `train.*` also matched `trainer.py`, which turned a
#: perfectly ordinary module into a junk experiment in the rail.
ENTRYPOINT_NAME = re.compile(
    r"^(train|run|main|experiment|finetune|sft|sweep|pretrain|eval|evaluate|demo|benchmark)"
    r"([_-][a-z0-9_-]*)?\.py$"
)

SDK_CALLS = {
    "tensorviz.experiment",
    "tensorviz.node",
    "tensorviz.edge",
    #: How a module says what it needs to be BUILT (step 4.7). Four of nanoGPT's
    #: five modules take a config object, so without this the most common shape in
    #: research code cannot be validated by any declaration the user can write.
    "tensorviz.init",
    #: The example tensor a forward runs on, when no layer states it.
    "tensorviz.input",
}


@dataclass
class Definition:
    name: str
    kind: str
    reason: str
    provenance: str
    lineno: int
    end_lineno: int
    col: int
    end_col: int
    references: Set[str] = field(default_factory=set)
    #: Names this definition's body CONSTRUCTS — `Block(config)` inside GPT.
    #:
    #: Kept apart from `references`, which is every name loaded and therefore far
    #: too loose to mean composition: GPT's references include `B`, `C`, `T` and
    #: every other local. A call is the difference between mentioning a class and
    #: building one, and composition is what the canvas draws.
    constructs: Set[str] = field(default_factory=set)
    #: The same constructions, with the words the source used around them.
    sites: List["ConstructionSite"] = field(default_factory=list)
    params: Dict[str, object] = field(default_factory=dict)
    digest: str = ""


@dataclass
class SdkDeclaration:
    call: str
    args: List[object]
    lineno: int
    end_lineno: int
    col: int
    end_col: int
    #: Named arguments, kept BY NAME rather than flattened into ``args``.
    #: ``experiment``/``node``/``edge`` are positional vocabularies, so flattening
    #: cost them nothing; ``init`` names constructor parameters, and a parameter
    #: whose name is thrown away is a parameter that cannot be passed.
    kwargs: Dict[str, object] = field(default_factory=dict)


@dataclass
class FileFacts:
    rel_file: str
    source_digest: str
    module: str = ""
    imports: Set[str] = field(default_factory=set)
    #: Relative imports (`from .data import X`), already turned into workspace paths
    #: without an extension. Kept apart from `imports` because they need no sys.path
    #: search: a relative import is lexical, so it resolves from the importer's own
    #: package directory alone. Dropping them — which is what handling only
    #: `level == 0` did — makes every package's internals invisible, and inside a
    #: package relative imports are the norm rather than the exception.
    relative_targets: Set[str] = field(default_factory=set)
    definitions: List[Definition] = field(default_factory=list)
    usages: List[UsageSite] = field(default_factory=list)
    sdk: List[SdkDeclaration] = field(default_factory=list)
    has_main_block: bool = False
    #: The module body does work at import time — a call, a loop, a comprehension —
    #: rather than only defining things. A plain script has this; a library does not.
    runs_at_import: bool = False
    #: It imports something from this workspace, so the work it does is this
    #: project's work rather than a standalone utility's.
    imports_workspace: bool = False
    #: The module body itself trains — `.backward()` then an optimizer `.step()` at
    #: top level rather than inside a function. nanoGPT's train.py is written this
    #: way, and without noticing it the canonical PyTorch training script produced a
    #: graph with no trainer and therefore no pipeline at all.
    trains_at_module_level: bool = False
    #: Some other file in the workspace imports THIS one. A module other code pulls
    #: in is a library, whatever it is called — `evaluation/evaluate.py` matching an
    #: entrypoint name does not make it one.
    imported_by_others: bool = False
    error: Optional[str] = None


# ── one file ────────────────────────────────────────────────────────────────────


def analyse_file(
    rel_file: str, source: str, classifier: Classifier = DEFAULT_CLASSIFIER
) -> FileFacts:
    """Parse one file. A syntax error is recorded, never raised."""
    rel = identity.normalise_path(rel_file)
    facts = FileFacts(
        rel_file=rel,
        source_digest=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        module=rel[:-3].replace("/", ".") if rel.endswith(".py") else "",
    )
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        facts.error = f"{error.msg} (line {error.lineno})"
        return facts

    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            facts.imports.update(alias.name for alias in stmt.names)
        elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and stmt.module:
            facts.imports.add(stmt.module)
            # `from data import toy` may name a module rather than a symbol.
            facts.imports.update(f"{stmt.module}.{alias.name}" for alias in stmt.names)
        elif isinstance(stmt, ast.ImportFrom) and stmt.level:
            base = os.path.dirname(rel)
            for _ in range(stmt.level - 1):
                base = os.path.dirname(base)
            prefix = f"{base}/{stmt.module.replace('.', '/')}" if stmt.module else base
            prefix = prefix.lstrip("/")
            facts.relative_targets.add(prefix)
            # `from . import data` names a module; `from .data import X` may name
            # either a symbol in data.py or a submodule data/X.py. Both are offered
            # and whichever exists wins, because guessing between them needs the
            # file set this function deliberately does not have.
            facts.relative_targets.update(
                f"{prefix}/{alias.name}".lstrip("/") for alias in stmt.names
            )
        elif isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            facts.definitions.append(_definition(stmt, classifier))
        else:
            if _is_main_block(stmt):
                facts.has_main_block = True
            elif _does_work(stmt):
                facts.runs_at_import = True

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            _collect_sdk(node, facts)

    facts.usages = classifier.usage_sites(tree)

    # `# tensorviz: input=` is the class-scope input authority and the one a user
    # actually writes — rung 2 of the precedence ladder, shipped long before the SDK
    # existed. The project graph only ever looked for the SDK's `tensorviz.input()`,
    # so a model declaring its input exactly as the docs say was told "nothing here
    # says what shape to run through it" and offered a declaration for something
    # already declared.
    #
    # Class-scoped, by the rule setup, the drawing and the worker follow
    # (`input_declarations`): in a file with several models, a model owns the comment
    # directly above or inside it, and one elsewhere in the file declares for none.
    # This file's own models are the proof of which classes are models.
    #
    # Whether it can be RESOLVED to a shape is a separate question — the expression
    # form (`torch.randint(1, 16, (1, 10))`) needs torch, which this tier will not
    # import. But "is an input declared" is answerable without evaluating it, and
    # that is the only question the verdict asks.
    _promote_local_subclasses(facts)

    models = {(d.lineno, d.name) for d in facts.definitions if d.kind == "model"}
    for definition in facts.definitions:
        if definition.kind == "model" and applicable(source, definition.name, tree=tree, proven=lambda: models)[0]:
            definition.params["_input_comment"] = True

    module_body = [
        stmt for stmt in tree.body if not isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    facts.trains_at_module_level = _trains(scan_body(module_body))
    return facts


def _promote_local_subclasses(facts: FileFacts) -> None:
    """A class that subclasses one of this file's models is a model too.

    `classify_class` matches the literal base name — `nn.Module`,
    `LightningModule` — which is right for a class that says so and blind to one
    that says it through a class of its own. DeepSeek-V3 defines `Linear(nn.Module)`
    and then `ColumnParallelLinear(Linear)` and `RowParallelLinear(Linear)`; both
    came out `generic`, "a class with no recognised role", and they are the building
    blocks of its attention. Every serious repository defines its own base class, so
    this is ordinary rather than exotic.

    Iterated to a fixed point, because the chain is not always one link:
    `A(nn.Module)`, `B(A)`, `C(B)` is a normal shape.

    Same file only. A base imported from another module needs the name resolved
    through that file's imports, which is a different piece of work — recorded in
    the validation notes rather than half-done here.
    """
    kinds = {definition.name: definition.kind for definition in facts.definitions}
    changed = True
    while changed:
        changed = False
        for definition in facts.definitions:
            if definition.kind != "generic":
                continue
            bases = definition.params.get("bases")
            if not isinstance(bases, list):
                continue
            for base in bases:
                if kinds.get(str(base)) in ("model", "dataset"):
                    definition.kind = kinds[str(base)]
                    # Says which class carried it, so a reader can follow the chain
                    # rather than wonder why this one counts.
                    definition.reason = f"subclasses {base}, which is a {kinds[str(base)]}"
                    definition.provenance = "static"
                    kinds[definition.name] = definition.kind
                    changed = True
                    break


def _definition(stmt: ast.AST, classifier: Classifier) -> Definition:
    scan: BodyScan = scan_body(stmt.body)  # type: ignore[attr-defined]
    if isinstance(stmt, ast.ClassDef):
        result: Classification = classifier.classify_class(stmt, scan) or Classification(
            "generic", "unclassified"
        )
        params: Dict[str, object] = {"bases": [dotted(b) or "?" for b in stmt.bases]}
        # What building this class would require, read from `__init__` (step 5.9).
        # Recorded for every class, not only ones that look unconstructible: the
        # canvas has to be able to say "needs ndim, bias" BEFORE anything runs, and
        # asking at validate time would make the refusal arrive after the button.
        required = _ctor_required(stmt)
        if required:
            params["_ctor_required"] = required
    else:
        result = classifier.classify_function(stmt.name, scan, stmt) or Classification(
            "function", "unclassified"
        )
        params = {
            "arguments": [a.arg for a in stmt.args.args if a.arg not in ("self", "cls")]  # type: ignore[attr-defined]
        }
    if scan.logs_metrics:
        params["metrics"] = sorted(scan.logs_metrics)
    params.update(result.params)
    return Definition(
        name=stmt.name,  # type: ignore[attr-defined]
        kind=result.kind,
        reason=result.reason,
        provenance=result.provenance,
        lineno=stmt.lineno,  # type: ignore[attr-defined]
        end_lineno=getattr(stmt, "end_lineno", None) or stmt.lineno,  # type: ignore[attr-defined]
        col=stmt.col_offset,  # type: ignore[attr-defined]
        end_col=getattr(stmt, "end_col_offset", None) or 0,
        references=scan.references - {stmt.name},  # type: ignore[attr-defined]
        # Bare names only. `F.layer_norm` and `GPT2LMHeadModel.from_pretrained` are
        # calls into a library, not this project building one of its own parts.
        constructs={
            call for call in scan.calls if "." not in call and call != stmt.name  # type: ignore[attr-defined]
        },
        sites=_construction_sites(stmt) if isinstance(stmt, ast.ClassDef) else [],
        params=params,
        digest=hashlib.sha256(
            f"{stmt.name}:{result.kind}".encode("utf-8")  # type: ignore[attr-defined]
        ).hexdigest()[:12],
    )


def _is_main_block(stmt: ast.AST) -> bool:
    return (
        isinstance(stmt, ast.If)
        and isinstance(stmt.test, ast.Compare)
        and dotted(stmt.test.left) == "__name__"
        and any(
            isinstance(c, ast.Constant) and c.value == "__main__"
            for c in stmt.test.comparators
        )
    )


def _does_work(stmt: ast.AST) -> bool:
    """Does this top-level statement DO something, rather than declare something?

    Constants, imports, type aliases and `__all__` are declarations; a call, a loop
    or a `with` is work. Used to recognise a plain script as a real entrypoint rather
    than a filename guess.

    `try` and `if` are deliberately NOT work, though they were until indexing
    `pytorch/torchtitan` and `huggingface/trl`. A top-level `try` is almost always
    `try: import flash_attn / except ImportError:`, and a top-level `if` is
    `if TYPE_CHECKING:` or a version check — both are statements about the
    environment, not a program. Counting them made 120 of torchtitan's library
    modules and 89 of trl's look like entrypoints, which is most of how a 596-file
    repository reported 262 experiments. A file that really runs calls something, and
    that is `Expr(Call)`, `For`, `While` or `With`.
    """
    if isinstance(stmt, (ast.For, ast.While, ast.With, ast.AsyncWith)):
        return True
    if isinstance(stmt, ast.Expr):
        return isinstance(stmt.value, ast.Call)
    if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        value = getattr(stmt, "value", None)
        # `x = f()` is work; `X = 1` and `X = [1, 2]` are configuration.
        return value is not None and isinstance(value, (ast.Call, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp))
    return False


def _ctor_required(cls: ast.ClassDef) -> List[str]:
    """Parameter names ``__init__`` has no default for, in order.

    ``self`` excluded, and ``*args``/``**kwargs`` ignored: a class that takes only
    those is constructible with nothing, which is a different statement from one
    that demands two values nobody can guess.

    A class with no ``__init__`` of its own requires nothing here. That is a
    reading of THIS class, not of its bases, and it is deliberately not chased up
    the MRO: resolving inheritance means resolving imports, and a wrong answer
    about what a constructor needs is worse than an absent one.
    """
    init = next(
        (
            stmt
            for stmt in cls.body
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "__init__"
        ),
        None,
    )
    if init is None:
        return []
    positional = [a.arg for a in init.args.posonlyargs] + [a.arg for a in init.args.args]
    positional = [name for name in positional if name not in ("self", "cls")]
    without_default = positional[: len(positional) - len(init.args.defaults)] if init.args.defaults else positional
    required = list(without_default)
    for arg, default in zip(init.args.kwonlyargs, init.args.kw_defaults):
        if default is None:
            required.append(arg.arg)
    return required


@dataclass(frozen=True)
class ConstructionSite:
    """One place a definition builds another, with the words the source used.

    `GPT builds Block` is true and thin. The source says where it lives, what
    holds it and how many there are:

        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))

    which is `transformer.h`, a `ModuleList`, `config.n_layer` of them. A reader
    looking at the canvas should be able to reconstruct that line, because the
    canvas is not a replacement for the code — it is a way to see what the code
    already says.

    `count` is the repeat expression AS WRITTEN, never its value: `config.n_layer`
    is 12 only once the config exists, and printing 12 would be claiming a number
    nothing here ran to find out.
    """

    name: str
    #: Attribute path from `self`, dict keys included: `transformer.h`.
    attr: str = ""
    #: The container holding it: `ModuleList`, `ModuleDict`, `Sequential`.
    via: str = ""
    #: The repeat expression as source text, or "" when it is built once.
    count: str = ""


def _attribute_path(target: ast.expr) -> str:
    """`self.transformer` -> `transformer`. Anything else -> ""."""
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
        if target.value.id == "self":
            return target.attr
    return ""


def _construction_sites(cls: ast.ClassDef) -> List[ConstructionSite]:
    """Every `self.x = ...Cls(...)` in a class body, with its context.

    Walks the assigned value rather than matching one shape, because real code
    nests: a ModuleDict of a dict of a ModuleList of a comprehension is one
    statement in nanoGPT and three levels here. What is tracked through the walk
    is only what a reader would say out loud — the attribute, the container, and
    the repeat — and a shape this does not recognise degrades to a bare name,
    which is what the graph said before any of this existed.
    """
    sites: List[ConstructionSite] = []

    def walk(node: ast.expr, attr: str, via: str, count: str) -> None:
        if isinstance(node, ast.Call):
            callee = dotted(node.func) or ""
            tail = callee.rsplit(".", 1)[-1]
            if tail in {"ModuleList", "Sequential", "ModuleDict", "ParameterList"}:
                for argument in node.args:
                    walk(argument, attr, tail, count)
                for keyword in node.keywords:
                    if keyword.arg:
                        walk(keyword.value, f"{attr}.{keyword.arg}" if attr else keyword.arg, tail, count)
                return
            if tail == "dict":
                # `dict(h=..., ln_f=...)` names each member, and the name is what a
                # reader calls it: `transformer.h`, not `transformer[3]`.
                for keyword in node.keywords:
                    if keyword.arg:
                        walk(keyword.value, f"{attr}.{keyword.arg}" if attr else keyword.arg, via, count)
                return
            if "." not in callee and callee:
                sites.append(ConstructionSite(name=callee, attr=attr, via=via, count=count))
                return
            # A library call — `nn.Embedding(...)`. Its arguments cannot hold one of
            # this project's classes in any shape worth reporting.
            return
        if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            repeat = ""
            for generator in node.generators:
                iterated = generator.iter
                if (
                    isinstance(iterated, ast.Call)
                    and (dotted(iterated.func) or "").rsplit(".", 1)[-1] == "range"
                    and iterated.args
                ):
                    repeat = _source_text(iterated.args[-1])
                else:
                    repeat = _source_text(iterated)
            walk(node.elt, attr, via, repeat or count)
            return
        if isinstance(node, (ast.List, ast.Tuple)):
            for element in node.elts:
                walk(element, attr, via, count)
            return
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                name = key.value if isinstance(key, ast.Constant) and isinstance(key.value, str) else None
                walk(value, f"{attr}.{name}" if name and attr else (name or attr), via, count)
            return

    for statement in ast.walk(cls):
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            attr = _attribute_path(statement.targets[0])
            walk(statement.value, attr, "", "")
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            walk(statement.value, _attribute_path(statement.target), "", "")
    return sites


def _source_text(node: ast.expr) -> str:
    """The expression as written, short enough for an edge label."""
    try:
        text = ast.unparse(node)
    except Exception:  # pragma: no cover - unparse is total on parsed trees
        return ""
    return text if len(text) <= 40 else ""


def _collect_sdk(node: ast.Call, facts: FileFacts) -> None:
    """`tensorviz.experiment|node|edge(...)` read straight from the AST.

    Literals only, and deliberately so: the indexer reads the call without running
    it, and a computed label is exactly the thing it cannot see. A non-literal
    argument makes the declaration unreadable rather than wrong.
    """
    name = dotted(node.func)
    if name not in SDK_CALLS:
        return
    args: List[object] = []
    for arg in node.args:
        value = literal(arg)
        if value is None:
            return
        args.append(value)
    kwargs: Dict[str, object] = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        value = literal(kw.value)
        if value is not None:
            kwargs[kw.arg] = value
            args.append(value)
    facts.sdk.append(
        SdkDeclaration(
            call=name.split(".")[-1],
            args=args,
            kwargs=kwargs,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            col=node.col_offset,
            end_col=node.end_col_offset or 0,
        )
    )


# ── import resolution, and the project root that falls out of it ────────────────


def source_roots(files: Set[str]) -> List[str]:
    """Directories that behave as though they were on `sys.path`.

    A directory holding a top-level package — it contains `pkg/__init__.py` but is
    not itself a package — is one. That is what makes the `src/` layout work:
    `from voxcpm.model import X` resolves under `src/` because `src/voxcpm/` is a
    package and `src/` is not.

    Derived from the file set rather than configured, for the same reason project
    roots are. Without it a src-layout repository resolves NOTHING — VoxCPM's
    training script reached none of the model code it imports, so the graph showed a
    trainer with no model.
    """
    packages = {
        os.path.dirname(rel) for rel in files if os.path.basename(rel) == "__init__.py"
    }
    roots = {""}
    for package in packages:
        parent = os.path.dirname(package)
        if parent not in packages:
            roots.add(parent)
    return sorted(roots, key=lambda r: (-r.count("/") if r else 1, r))


def _resolve_import(
    module: str, importer: str, files: Set[str], roots: Optional[Sequence[str]] = None
) -> Optional[Tuple[str, str]]:
    """Resolve `data.toy` the way Python would for a script, and say from where.

    Returns `(resolved file, the prefix that made it resolve)`. That prefix is the
    directory this import behaved as though were on `sys.path` — the evidence the
    project root is derived from. None means third party, which is not an error.

    The importer's own directory and its ancestors are tried first, because that is
    what running a script does; declared source roots follow.
    """
    path = module.replace(".", "/")
    directory = os.path.dirname(importer)
    prefixes: List[str] = []
    while True:
        prefixes.append(directory)
        if not directory:
            break
        directory = os.path.dirname(directory)
    for extra in roots or ():
        if extra not in prefixes:
            prefixes.append(extra)
    for prefix in prefixes:
        base = f"{prefix}/{path}" if prefix else path
        if f"{base}.py" in files:
            return f"{base}.py", prefix
        if f"{base}/__init__.py" in files:
            return f"{base}/__init__.py", prefix
    return None


def _import_closure(
    start: str, facts: Mapping[str, FileFacts], files: Set[str], roots: Sequence[str]
) -> Tuple[Set[str], Optional[str]]:
    """Every workspace file this entrypoint reaches, plus its derived project root.

    The root is `None` when the entrypoint resolved no workspace import at all and
    therefore proved nothing about its own project; see the end of this function.

    The root is the longest directory prefix shared by the entrypoint and every
    prefix its imports resolved against — the smallest thing that contains the whole
    experiment.

    Worked through:

    * flat repo — `train.py` importing `models.net`, resolving at `""`. Shared
      prefix `""`. The project is the repository.
    * monorepo — `packages/alpha/train.py` importing `data.toy`, resolving at
      `packages/alpha`. Shared prefix `packages/alpha`, so two sibling projects
      separate without anyone declaring anything.
    * src layout — `scripts/train.py` importing `voxcpm.model`, resolving at `src`.
      Shared prefix `""`. Neither `scripts` nor `src` is the project on its own;
      the thing containing both is.
    """
    seen = {start}
    frontier = [start]
    found: Set[str] = set()
    while frontier:
        rel = frontier.pop()
        current = facts.get(rel)
        if current is None:
            continue
        for module in sorted(current.imports):
            resolved = _resolve_import(module, rel, files, roots)
            if resolved is None:
                continue
            target, prefix = resolved
            found.add(prefix)
            for init in _package_inits(target, files) | {target}:
                if init not in seen:
                    seen.add(init)
                    frontier.append(init)
        for target in sorted(_relative_files(current, files)):
            for init in _package_inits(target, files) | {target}:
                if init not in seen:
                    seen.add(init)
                    frontier.append(init)
    # Nothing resolved: this file imports the standard library and third parties and
    # nothing else, so it offers no evidence about where its project begins. Its own
    # folder is not the answer — `scripts/`, `.github/` and `trl/experimental/bco/`
    # are folders inside a project, and treating each as a project of its own is
    # what fragmented three of the corpus repositories. `None` says "no evidence"
    # and lets index_project settle it against what the rest of the repository
    # proved, which is the only place that is knowable.
    if not found:
        return seen, None
    # The project is the smallest thing containing both the entrypoint and whatever
    # its imports resolved against. A src-layout repo puts the script in `scripts/`
    # and the package under `src/`, and calling either one "the project" is wrong.
    #
    # The entrypoint's own directory is safe to use here BECAUSE there is evidence
    # beside it: the shared prefix can only be shallower than the deepest thing in
    # the list. A file deep inside a package contributes a deep directory and the
    # prefix its imports resolved against pulls the answer back out. It is when
    # `found` is empty — above — that its directory would be the whole answer, and
    # that is the case this function refuses to guess at.
    root = _common_directory([os.path.dirname(start), *sorted(found)])
    return seen, root


def _package_inits(rel: str, files: Set[str]) -> Set[str]:
    """Every `__init__.py` Python would execute to reach `rel`.

    Importing `whisper.transcribe` runs `whisper/__init__.py` first — that is what
    makes a package a package — and a library's `__init__.py` is very often where it
    re-exports the things worth drawing.

    Found by indexing `openai/whisper`, which reported **zero models in an ASR
    repository**. Its entrypoint is `whisper/__main__.py` -> `.transcribe`, and every
    `from .model import Whisper` in the package is inside `if TYPE_CHECKING:` — a
    typing-only import that does not run. The one import that does run is in
    `whisper/__init__.py`, which the closure never walked, so `model.py` was in no
    experiment's closure and its eleven `nn.Module` classes were simply absent.
    """
    found: Set[str] = set()
    directory = os.path.dirname(rel)
    while directory:
        candidate = f"{directory}/__init__.py"
        if candidate in files:
            found.add(candidate)
        directory = os.path.dirname(directory)
    return found


def _relative_files(facts: FileFacts, files: Set[str]) -> Set[str]:
    """Workspace files a file's relative imports name."""
    found: Set[str] = set()
    for prefix in facts.relative_targets:
        if f"{prefix}.py" in files:
            found.add(f"{prefix}.py")
        elif f"{prefix}/__init__.py" in files:
            found.add(f"{prefix}/__init__.py")
    return found


def _is_ancestor(ancestor: str, directory: str) -> bool:
    """Is `directory` inside `ancestor`? The workspace root contains everything."""
    return not ancestor or directory.startswith(f"{ancestor}/")


def _common_directory(paths: Sequence[str]) -> str:
    parts = [p.split("/") if p else [] for p in paths]
    if not parts:
        return ""
    shared: List[str] = []
    for segments in zip(*parts):
        if len(set(segments)) != 1:
            break
        shared.append(segments[0])
    return "/".join(shared)


# ── experiments ─────────────────────────────────────────────────────────────────


#: Directory names whose contents are tests by universal Python convention —
#: what `pytest` and `unittest discover` collect, not something inferred.
TEST_DIRECTORIES = frozenset({"test", "tests"})


def _is_test(rel_file: str) -> bool:
    """Would a test runner collect this file?

    Not a guess about intent: `pytest` collects `test_*.py` and `*_test.py`, and
    `conftest.py` is its own configuration file. torchtitan has 111 such files with
    a `if __name__ == "__main__": unittest.main()` block apiece, and every one of
    them was an "experiment" — 111 entries in a rail whose whole job is to say what
    this repository trains.
    """
    base = os.path.basename(rel_file)
    if base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py":
        return True
    return bool(TEST_DIRECTORIES.intersection(rel_file.split("/")[:-1]))


def _is_tooling(rel_file: str) -> bool:
    """Does this file live in a dot-directory?

    `.github/scripts`, `.claude/skills/*/scripts`. Repository plumbing that ships
    beside the source without being it — torchtitan offered four of these as
    experiments and three of them as separate *projects*, because a script with no
    workspace imports derives its own directory as a root.
    """
    return any(segment.startswith(".") for segment in rel_file.split("/")[:-1])


def _experiment_for(facts: FileFacts) -> Optional[Tuple[str, str, str, bool]]:
    """`(experiment_id, label, state, declared)` for a file, or None.

    Precedence is the whole point: a declaration is an assertion, a `__main__` block
    is strong evidence, and a filename is a suggestion the user must confirm.

    Underneath the precedence sits one question the rungs used to ask too late:
    **is this file a module?** A file another workspace file imports is something
    the project uses, not something the project runs, and that is true however much
    work its body does at import time. Asking it only on the filename rung is how
    `trl` reported 216 experiments and `torchtitan` 262 — a library module with a
    `logger = getLogger(__name__)` line outranked the evidence against it.
    """
    for decl in facts.sdk:
        if decl.call == "experiment" and decl.args and isinstance(decl.args[0], str):
            name = decl.args[0]
            return identity.declared_experiment_id(name), name, "active", True

    # Everything below this line is evidence; a declaration above it is an assertion,
    # so a deliberately-named `test_my_idea.py` experiment is still reachable.
    if _is_test(facts.rel_file) or _is_tooling(facts.rel_file):
        return None

    definition = identity.definition_id(facts.rel_file, "__main__")
    label = _label(facts.rel_file)
    if facts.has_main_block:
        # `python -m pkg.module` is a real way to run something, so a guard in an
        # imported file is not nothing — but it is not the project's entrypoint
        # either. A candidate says so and waits to be confirmed.
        state = "candidate" if facts.imported_by_others else "active"
        return identity.experiment_id(definition), label, state, False
    # A plain script: no guard, but the module body calls into the project rather
    # than only defining things. Plenty of research code is written exactly this
    # way — nanoGPT's train.py has no guard — and treating it as a mere filename
    # candidate made a repository's real entrypoint invisible. Something importing
    # it settles the question the other way.
    if facts.runs_at_import and facts.imports_workspace and not facts.imported_by_others:
        return identity.experiment_id(definition), label, "active", False
    # A filename is the weakest evidence there is, so it loses to the strongest
    # counter-evidence: something else imports this file, which makes it a module.
    if facts.imported_by_others:
        return None
    # It must also at least look like a program. A file of nothing but literal
    # assignments is configuration whatever it is called — nanoGPT's `config/`
    # directory is seven files named `train_*.py` and `eval_*.py`, and treating
    # them as candidate experiments buried the three real entrypoints.
    looks_like_code = bool(facts.definitions or facts.imports or facts.runs_at_import)
    if not looks_like_code:
        return None
    if ENTRYPOINT_NAME.match(os.path.basename(facts.rel_file)):
        return identity.experiment_id(definition), label, "candidate", False
    return None


def _label(rel_file: str) -> str:
    return os.path.basename(rel_file)[:-3] or rel_file


# ── graph assembly ──────────────────────────────────────────────────────────────


def _build_graph(
    experiment_id: str,
    graph_id: str,
    entrypoint: str,
    closure: Set[str],
    facts: Mapping[str, FileFacts],
) -> ResearchGraph:
    nodes: List[ResearchNode] = []
    edges: List[ResearchEdge] = []
    by_label: Dict[str, ResearchNode] = {}
    file_nodes: Dict[str, ResearchNode] = {}
    # The entrypoint itself plus what it imports by name — the files the author
    # reached for, as opposed to everything those files in turn pulled in.
    primary_files: Set[str] = {entrypoint}
    entry_facts = facts.get(entrypoint)
    # Computed once per graph, not once per import. `source_roots` derives the
    # sys.path-like roots from the file set alone, so recomputing it inside a loop
    # over imports asks the same question of the same set thousands of times: on a
    # 415-file workspace it ran 74,217 times and called posixpath.basename 4.5
    # million, which was 3.1 of the 9.8 seconds a profile of one index reported.
    closure_files = set(closure)
    closure_roots = source_roots(closure_files)
    if entry_facts is not None:
        for module in entry_facts.imports:
            resolved = _resolve_import(module, entrypoint, closure_files, closure_roots)
            if resolved is not None:
                primary_files.add(resolved[0])

    def add_node(
        definition: str,
        kind: str,
        label: str,
        *,
        instance: str = "",
        source: Optional[SourceRef] = None,
        params: Optional[Dict[str, object]] = None,
        producer: str = "static",
        provenance: str = "static",
    ) -> ResearchNode:
        node = ResearchNode(
            id=identity.experiment_node_id(experiment_id, definition, instance),
            graph_id=graph_id,
            experiment_id=experiment_id,
            kind=kind,  # type: ignore[arg-type]
            label=label,
            definition_id=definition,
            source=source,
            params=params or {},
            producer=producer,
            provenance=provenance,  # type: ignore[arg-type]
        )
        existing = node_ids.get(node.id)
        if existing is not None:
            # One id is one node. Edges have been deduplicated here since the
            # beginning; nodes were not, and a second node with the same id is not
            # a drawing glitch — `validate_graph` refuses the whole graph, so one
            # repeated identity costs the entire repository. Identities carry no
            # line numbers on purpose (that is what lets a node survive an edit
            # that moves it), so anything the source says twice under one name
            # arrives here twice: a variable assigned in both branches of an
            # `if`, a name rebound after a `try/except ImportError` fallback.
            return existing
        nodes.append(node)
        node_ids[node.id] = node
        return node

    node_ids: Dict[str, ResearchNode] = {}
    edge_ids: Set[str] = set()

    def add_edge(
        a: ResearchNode,
        b: ResearchNode,
        relationship: str,
        provenance: str = "static",
        producer: str = "static",
        params: Optional[Dict[str, str]] = None,
    ) -> None:
        edge = ResearchEdge(
            id=identity.edge_id(a.id, b.id, relationship),
            graph_id=graph_id,
            experiment_id=experiment_id,
            from_node_id=a.id,
            to_node_id=b.id,
            relationship=relationship,  # type: ignore[arg-type]
            producer=producer,
            provenance=provenance,  # type: ignore[arg-type]
            params=params or {},
        )
        # A set, not a scan. `any(e.id == edge.id for e in edges)` made every insert
        # walk every edge already placed — 53,666 inserts over a growing list ran
        # that generator 20.5 million times. The de-duplication it performs is
        # exactly a set membership test, and the edge list stays ordered because
        # the set only decides whether to append.
        if edge.id not in edge_ids:
            edge_ids.add(edge.id)
            edges.append(edge)

    # files, then the definitions they contain
    for rel in sorted(closure):
        current = facts.get(rel)
        if current is None:
            continue
        file_node = add_node(
            identity.definition_id(rel, "<file>"),
            "file",
            rel,
            params={"error": current.error} if current.error else {},
            provenance="static",
        )
        file_nodes[rel] = file_node

        for definition in current.definitions:
            node = add_node(
                identity.definition_id(rel, definition.name),
                definition.kind,
                definition.name,
                source=SourceRef(
                    file=rel,
                    start=(definition.lineno, definition.col),
                    end=(definition.end_lineno, definition.end_col),
                    digest=definition.digest,
                ),
                params={**definition.params, "_reason": definition.reason},
                producer="torch-ecosystem",
                provenance=definition.provenance,
            )
            by_label.setdefault(definition.name, node)
            add_edge(file_node, node, "contains")

        # usage sites: `trainer = SFTTrainer(...)` is a node even though the user
        # defined no class. Without this a modern HF repo renders as one box.
        # One variable is one node, however many times it is assigned. Mamba's
        # benchmark picks its tokenizer in an if/else — two
        # `tokenizer = AutoTokenizer.from_pretrained(...)` statements — and the
        # program still has one tokenizer. Grouping here rather than relying on
        # add_node's guard is what keeps the LABEL honest: a hint read off the
        # first branch's string literal ("EleutherAI/gpt-neox-20b") would name the
        # node after a model it only sometimes is.
        grouped: Dict[str, List[UsageSite]] = {}
        for usage in current.usages:
            grouped.setdefault(usage.binding, []).append(usage)

        for binding, group in grouped.items():
            first = group[0]
            hints = [use.params.get("_label_hint") for use in group]
            label = str(hints[0]) if hints[0] and all(h == hints[0] for h in hints) else binding
            params = {**first.params, "_reason": first.reason, "_callee": first.callee}
            if label == binding:
                params.pop("_label_hint", None)
            if len(group) > 1:
                # Said on the card, because a reader looking for the line that
                # made this node would otherwise find one of several and not know
                # the others exist.
                params["_assignments"] = len(group)
            node = add_node(
                identity.definition_id(rel, f"<usage>{binding}"),
                first.kind,
                label,
                source=SourceRef(
                    file=rel,
                    start=(first.lineno, first.col),
                    end=(first.end_lineno, first.end_col),
                ),
                params=params,
                producer="torch-ecosystem",
                provenance="static",
            )
            by_label.setdefault(binding, node)
            add_edge(file_node, node, "contains")

    # A file whose module body trains is itself the trainer. Without this a script
    # that runs its loop at top level — which is how a great deal of research code
    # is written — yields models and datasets with nothing joining them up.
    for rel in sorted(closure):
        current = facts.get(rel)
        if current is None or not current.trains_at_module_level or rel not in file_nodes:
            continue
        node = add_node(
            identity.definition_id(rel, "<module-trainer>"),
            "trainer",
            f"{os.path.basename(rel)[:-3]} loop",
            source=SourceRef(file=rel, start=(1, 0), end=(1, 0)),
            params={"_reason": "the module body calls .backward() and an optimizer .step()"},
            producer="torch-ecosystem",
            provenance="static",
        )
        by_label.setdefault(node.label, node)
        add_edge(file_nodes[rel], node, "contains")

    # imports between files in the closure
    files = set(closure)
    file_roots = source_roots(files)
    for rel in sorted(closure):
        current = facts.get(rel)
        if current is None or rel not in file_nodes:
            continue
        for module in sorted(current.imports):
            resolved = _resolve_import(module, rel, files, file_roots)
            if resolved and resolved[0] in file_nodes and resolved[0] != rel:
                add_edge(file_nodes[rel], file_nodes[resolved[0]], "imports")
        for relative in sorted(_relative_files(current, files)):
            if relative in file_nodes and relative != rel:
                add_edge(file_nodes[rel], file_nodes[relative], "imports")

    # dataflow from keyword arguments — the shape library-driven code actually uses
    from research.classify import KWARG_RELATIONSHIPS

    for rel in sorted(closure):
        current = facts.get(rel)
        if current is None:
            continue
        for usage in current.usages:
            target = by_label.get(usage.binding)
            if target is None:
                continue
            for kwarg, referenced in sorted(usage.name_kwargs.items()):
                relationship = KWARG_RELATIONSHIPS.get(kwarg)
                source_node = by_label.get(referenced)
                if relationship is None or source_node is None:
                    continue
                if relationship == "trains":
                    add_edge(target, source_node, "trains", provenance="static")
                else:
                    add_edge(source_node, target, relationship, provenance="static")

    # SDK declarations: the user's own words, ranked above anything inferred
    for rel in sorted(closure):
        current = facts.get(rel)
        if current is None or rel not in file_nodes:
            continue
        for decl in current.sdk:
            span = SourceRef(
                file=rel,
                start=(decl.lineno, decl.col),
                end=(decl.end_lineno, decl.end_col),
            )
            if decl.call in {"init", "input"} and decl.args:
                # Attaches to the node it NAMES, which may be declared in another
                # file of the closure: `tensorviz.init("GPT", ...)` is written where
                # the experiment is assembled, not inside model.py. A declaration
                # naming a node this index does not have is dropped rather than
                # inventing one — it is a statement about a class, and a class that
                # is not here cannot be described.
                target = by_label.get(str(decl.args[0]))
                if target is not None:
                    if decl.call == "init":
                        target.params["_init_args"] = dict(decl.kwargs)
                        target.params["_init_declared_at"] = f"{rel}:{decl.lineno}"
                    else:
                        target.params["_input_declared"] = dict(decl.kwargs)
                        target.params["_input_declared_at"] = f"{rel}:{decl.lineno}"
                continue
            if decl.call == "node" and len(decl.args) >= 2:
                kind, label = str(decl.args[0]), str(decl.args[1])
                node = add_node(
                    identity.definition_id(rel, f"<declared>{label}"),
                    kind,
                    label,
                    source=span,
                    params={"_reason": f"declared as {kind} by tensorviz.node()"},
                    producer="tensorviz.sdk",
                    provenance="declared",
                )
                by_label[label] = node
                add_edge(file_nodes[rel], node, "contains", provenance="declared")

    for rel in sorted(closure):
        current = facts.get(rel)
        if current is None:
            continue
        for decl in current.sdk:
            if decl.call == "edge" and len(decl.args) >= 2:
                a = by_label.get(str(decl.args[0]))
                b = by_label.get(str(decl.args[1]))
                relationship = str(decl.args[2]) if len(decl.args) > 2 else "feeds"
                if a is not None and b is not None:
                    add_edge(a, b, relationship, provenance="declared", producer="tensorviz.sdk")

    # Composition: one definition building another (`self.mlp = MLP(config)`).
    #
    # This is the model's own structure, and without it nanoGPT drew LayerNorm,
    # CausalSelfAttention, MLP and Block as four disconnected islands — a canvas
    # that could not show what `self.h = nn.ModuleList([Block(config) ...])` says
    # plainly two lines into the source.
    #
    # `static`, not `inferred`: the call is in the code, read rather than guessed.
    # What is deliberately NOT claimed is how many — a comprehension over
    # `range(config.n_layer)` builds twelve Blocks, and the graph says GPT builds
    # Block, because twelve is a value only running the code would know.
    for rel in sorted(closure):
        current = facts.get(rel)
        if current is None:
            continue
        for definition in current.definitions:
            owner = by_label.get(definition.name)
            if owner is None:
                continue
            # One edge per pair, carrying the clearest site: a construction the
            # source named (`transformer.h`) says more than one it did not, and two
            # `LayerNorm`s in one class are one edge with the first attribute rather
            # than two edges the canvas would draw on top of each other.
            best: Dict[str, ConstructionSite] = {}
            for site in definition.sites:
                if site.name not in by_label or by_label[site.name].id == owner.id:
                    continue
                current = best.get(site.name)
                if current is None or (not current.attr and site.attr):
                    best[site.name] = site
            for built in sorted(definition.constructs):
                target = by_label.get(built)
                # Only names this graph actually holds: `super`, `range` and every
                # other builtin in the scan simply never match a node.
                if target is None or target.id == owner.id:
                    continue
                site = best.get(built)
                add_edge(
                    owner,
                    target,
                    "builds",
                    provenance="static",
                    params={
                        key: value
                        for key, value in (
                            ("attr", site.attr if site else ""),
                            ("via", site.via if site else ""),
                            ("count", site.count if site else ""),
                        )
                        if value
                    },
                )

    # Which definitions each definition's body mentions — how "outermost" is decided.
    referenced_by: Dict[str, Set[str]] = {}
    for rel in sorted(closure):
        current = facts.get(rel)
        if current is None:
            continue
        for definition in current.definitions:
            referenced_by[definition.name] = set(definition.references)

    # a small, honest inferred spine, so a plain-PyTorch project has a root and a leaf
    _infer_spine(by_label, nodes, edges, primary_files, referenced_by, add_edge)

    _disambiguate_labels(nodes)

    return ResearchGraph(
        graph_id=graph_id, experiment_id=experiment_id, nodes=nodes, edges=edges
    ).validate_graph()


def _disambiguate_labels(nodes: List[ResearchNode]) -> None:
    """Two nodes with one label are indistinguishable on a canvas.

    Real projects do this constantly: VoxCPM keeps `audio_vae.py` beside
    `audio_vae_v2.py`, so `Snake1d` and `CausalEncoder` each appear twice with
    identical names. The ids differ, so the graph is correct — but a reader cannot
    tell which card is which, which is a different kind of wrong.

    Only the duplicated labels are qualified, so the common case stays clean.
    """
    counts: Dict[str, int] = {}
    for node in nodes:
        counts[node.label] = counts.get(node.label, 0) + 1
    for node in nodes:
        if counts.get(node.label, 0) < 2 or node.source is None:
            continue
        stem = os.path.basename(node.source.file)
        if stem.endswith(".py"):
            stem = stem[:-3]
        node.label = f"{node.label} ({stem})"


def _infer_spine(
    by_label: Mapping[str, ResearchNode],
    nodes: Sequence[ResearchNode],
    edges: Sequence[ResearchEdge],
    primary_files: Set[str],
    referenced_by: Mapping[str, Set[str]],
    add_edge,
) -> None:
    """A last-resort dataflow spine, for a project that declared none.

    Marked `inferred`, because it is a rule guessing at intent rather than a fact
    read from the code — the canvas draws it differently for exactly that reason.

    Two rules keep it from doing harm. It only fires when a node has NO dataflow
    edge already, so it can never contradict something read from the source; and it
    picks the most plausible node of a kind rather than the first one seen, because
    a composed model's first `nn.Module` is a leaf (`MultiHeadAttention`) while the
    one a reader means is the outermost (`TinyLM`).
    """
    connected = {e.from_node_id for e in edges if e.relationship in DATAFLOW_RELATIONSHIPS}
    connected |= {e.to_node_id for e in edges if e.relationship in DATAFLOW_RELATIONSHIPS}

    def pick(kind: str) -> Optional[ResearchNode]:
        candidates = [n for n in nodes if n.kind == kind]
        if not candidates:
            return None
        # Prefer something the entrypoint imported DIRECTLY. A composed model's
        # parts arrive transitively (`TinyLM` pulls in `TransformerBlock`, which
        # pulls in `MultiHeadAttention`), and the one the author reached for by name
        # is the one a reader means. Without this the spine wired a leaf.
        direct = [
            n
            for n in candidates
            if n.source is not None and n.source.file in primary_files
        ]
        candidates = direct or candidates
        if kind == "model" and len(candidates) > 1:
            # The outermost model is the one no other model builds out of. nanoGPT's
            # model.py defines LayerNorm, CausalSelfAttention, MLP, Block and GPT in
            # that order, and taking the first wired the pipeline to a leaf.
            built_from = {
                name
                for other in candidates
                for name in referenced_by.get(other.label, set())
            }
            outermost = [n for n in candidates if n.label not in built_from]
            candidates = outermost or candidates
        return candidates[0]

    dataset, model = pick("dataset"), pick("model")
    trainer, evaluator = pick("trainer"), pick("evaluator")
    metric, checkpoint = pick("metric"), pick("checkpoint")
    configuration = pick("configuration")

    pairs = [
        (dataset, model, "feeds"),
        (dataset, trainer, "feeds"),
        (trainer, model, "trains"),
        (trainer, checkpoint, "produces"),
        (evaluator, model, "evaluates"),
        (evaluator, metric, "logs"),
        (configuration, trainer, "runs_on"),
    ]
    for a, b, relationship in pairs:
        if a is None or b is None or a.id == b.id:
            continue
        # Never overrule the source: if either endpoint already participates in
        # dataflow, the code has said something and a guess must not argue with it.
        if a.id in connected and b.id in connected:
            continue
        add_edge(a, b, relationship, provenance="inferred", producer="spine-rule")


# ── the entry point ─────────────────────────────────────────────────────────────


def index_project(
    sources: Mapping[str, str],
    *,
    truncated: bool = False,
    classifier: Classifier = DEFAULT_CLASSIFIER,
    parse_cache: Optional[Any] = None,
) -> ProjectIndex:
    """`{path: source}` in, a `ProjectIndex` out. No disk, no imports, no execution.

    `parse_cache` is an optional `(path, digest) -> facts` store — `research.admission.
    ParseCache` — that turns a re-index into a re-parse of only what changed. Parsing
    is 88–93% of the cost of a whole-repo index, so on a keystroke this is the
    difference between 1.5 seconds and 0.2 across 596 files. Passed in rather than
    held here, because this module owns no state: the caller decides how long a
    workspace stays warm, and a test gets a cold one by passing nothing.
    """
    facts = {}
    for rel, source in sorted(sources.items()):
        path = identity.normalise_path(rel)
        if parse_cache is None:
            facts[path] = analyse_file(rel, source, classifier)
            continue
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        cached = parse_cache.get(path, digest)
        if cached is None:
            cached = analyse_file(rel, source, classifier)
            parse_cache.put(path, digest, cached)
        # A copy, because the cross-file pass below WRITES to these facts —
        # `imports_workspace` and `imported_by_others` are decided per index — and a
        # cached object mutated by one index would carry that answer into the next,
        # where a file whose importer was deleted would still believe it is imported.
        facts[path] = replace(cached)
    files = set(facts)
    roots = source_roots(files)

    # Whether a file's imports reach into this workspace is only knowable once every
    # file is present, so it is decided here rather than in analyse_file.
    for rel, current in facts.items():
        for module in current.imports:
            resolved = _resolve_import(module, rel, files, roots)
            if resolved is None:
                continue
            current.imports_workspace = True
            target = facts.get(resolved[0])
            if target is not None and resolved[0] != rel:
                target.imported_by_others = True
        for relative in _relative_files(current, files):
            current.imports_workspace = True
            target = facts.get(relative)
            if target is not None and relative != rel:
                target.imported_by_others = True

    # Two passes, because the second needs an answer only the first can give. An
    # entrypoint that resolves no workspace import proves nothing about where its
    # project begins, and the honest place to look then is what the REST of the
    # repository proved: DeepSeek-V3 keeps every file under `inference/`, and its
    # `convert.py` imports nothing local — inventing a second root at the workspace
    # for that one file split a five-file repository into two projects.
    found: List[Tuple[str, str, str, str, bool, Set[str], Optional[str]]] = []
    reached: Set[str] = set()
    for rel in sorted(facts):
        candidate = _experiment_for(facts[rel])
        if candidate is None:
            continue
        experiment_id, label, state, declared = candidate
        closure, root = _import_closure(rel, facts, files, roots)
        reached |= closure
        found.append((rel, experiment_id, label, state, declared, closure, root))

    # Deepest first, then alphabetically — never `key=len` alone. Two roots of equal
    # length would otherwise be ordered by set iteration, which varies with
    # PYTHONHASHSEED, and both the absorb below and the fallback that picks an
    # enclosing root take the FIRST match. A digest that changes between runs of the
    # same source is the one defect a cache cannot survive.
    evidenced = sorted(
        {root for *_, root in found if root is not None},
        key=lambda root: (-len(root), root),
    )

    # A `src/` layout derives two roots and means one project. `src/voxcpm/cli.py`
    # resolves its imports at `src`, while `app.py` imports the same package from
    # outside and resolves at `""` — so the same codebase was offered as two
    # projects depending on which side of `src/` you entered it from.
    #
    # The question that separates that from a real monorepo is whether the outer
    # project already REACHES INTO the inner root's directory. VoxCPM's `app.py`
    # closure contains `src/voxcpm/core.py`, so `src` is somewhere the outer project
    # keeps its code, not a project of its own. cleanrl's `cleanrl/ppo_trxl/` — which
    # ships its own `pyproject.toml` — is reached by nothing outside it, and stays
    # separate. So do `packages/alpha` and `packages/beta`, which share no file with
    # anything above them.
    #
    # Comparing closures directly is not enough: `cli.py` reaches only the timestamp
    # helpers and `app.py` only the model, so a src layout can be genuinely disjoint
    # and still be one project.
    absorbed: Dict[str, str] = {}
    by_root: Dict[str, Set[str]] = {}
    for _, _, _, _, _, closure, root in found:
        if root is not None:
            by_root.setdefault(root, set()).update(closure)
    for root in evidenced:
        if not root:
            continue
        for ancestor in evidenced:
            if ancestor == root or not _is_ancestor(ancestor, root):
                continue
            if any(reached.startswith(f"{root}/") for reached in by_root.get(ancestor, ())):
                absorbed[root] = ancestor
                break
    evidenced = [root for root in evidenced if root not in absorbed]

    experiments: List[Experiment] = []
    for rel, experiment_id, label, state, declared, closure, root in found:
        while root in absorbed:
            root = absorbed[root]
        if root is None:
            root = next(
                (
                    known for known in evidenced
                    if not known or rel.startswith(f"{known}/")
                ),
                "",
            )
        graph_id = identity.graph_id_for(root, rel) or experiment_id
        experiments.append(
            Experiment(
                experiment_id=experiment_id,
                label=label,
                entrypoint=rel,
                root=root,
                status=ExperimentStatus(state=state, declared=declared),  # type: ignore[arg-type]
                graph=_build_graph(experiment_id, graph_id, rel, closure, facts),
                closure=sorted(closure),
            )
        )

    unreached = [
        UnreachedFile(file=rel, error=facts[rel].error)
        for rel in sorted(files - reached)
    ]
    return ProjectIndex(
        experiments=experiments,
        unreached=unreached,
        truncated=truncated,
        roots=sorted({e.root for e in experiments}),
    )
