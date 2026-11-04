"""Tier B: ingesting an arbitrary repository whose entrypoint is unknown.

Static AST discovery is a solved problem (0 parse errors over 2,752 files across nine real
research repositories). The hard parts, in the order they actually bite:

1. **Which class is the model.** A weighted score gets top-1 right on 6 of 9 repos. The
   remaining failures are structural, not tunable: training/loss wrappers are what
   ``train.py`` instantiates, so they legitimately outrank the backbone (dinov2's
   ``SSLMetaArch`` beats ``DinoVisionTransformer``). We therefore return the ranked TOP-5
   and let the caller confirm rather than silently auto-picking.
2. **Constructor arguments.** A five-rung deterministic ladder. Where it fires, the values
   are ground truth: nanoGPT -> 124.5M parameters (GPT-2 is 124M), DiT ``DiT_XL_2()`` ->
   675.1M (DiT-XL/2 is 675M).
3. **Missing third-party imports** — the real crux. Repo-wide import success on a bare
   torch environment is ~33%. Most failures are *training* dependencies imported at module
   scope but never touched while merely defining a model, so an unresolved import is
   stubbed and retried; anything the definition genuinely uses still fails loudly.

Importing a repository executes its code. That is inherent to the approach and acceptable
for a locally-run tool; it is not a sandbox.
"""

from __future__ import annotations

import ast
import importlib
import io
import os
import re
import sys
import time
import types
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from .spec import EntrypointCandidate
from ..validation.setup import module_path_for

ENTRY_SCRIPT = re.compile(r"(train|sample|infer|main|generate|demo|run|predict|eval)", re.I)
MODEL_NAME = re.compile(
    r"(Model|Net|Transformer|GPT|UNet|Unet|Diffusion|CLIP|ViT|Encoder|Decoder)$"
)
# A config tree dominated by these is a TRAINING wrapper, not the model itself.
TRAINING_ATTRS = {
    "crops", "optim", "train", "student", "teacher", "loss", "schedule", "lr",
    "epochs", "batch_size", "dataset", "dataloader", "ckpt", "checkpoint",
}
SKIP_DIRS = {".git", "node_modules", "__pycache__", "build", "dist", ".venv", "venv", ".tox"}
CONFIG_PARAM_NAMES = ("config", "cfg", "args", "opt", "opts", "params", "hparams")
MAX_STUBS = 32

#: What makes a directory a Python ENVIRONMENT, whatever it is called. Matching the two
#: literal names ".venv"/"venv" missed every other real spelling — `env`, `.env`, `env310`,
#: a conda prefix, a vendored `site-packages`, a `third_party/` checkout — and a user whose
#: virtualenv is named `env` paid for walking all of torch on every open and save: 11,075
#: files, 46 s, 4 GB of resident memory, and a Models view holding 1,033 library classes
#: with their own two models buried in it. These markers are what the interpreter itself
#: writes, so they identify an environment by structure rather than by fashion in naming.
ENV_MARKER_FILES = ("pyvenv.cfg",)
ENV_MARKER_DIRS = ("conda-meta", "site-packages")

#: Hard bounds on one walk. A scan is answered on every open and save from an editor that
#: passes no timeout, so "slow" here is indistinguishable from "hung" — and with root="~"
#: the walk ran past 180 s and 6 GB, still going after the client had given up. A partial
#: index the caller is TOLD is partial beats an unbounded one nobody can cancel.
MAX_SCAN_FILES = 4000
MAX_SCAN_SECONDS = 20.0


def _is_environment(base: str, dirs: Sequence[str], files: Sequence[str]) -> bool:
    """True when this directory is an installed Python environment rather than source."""
    if os.path.basename(base) in ENV_MARKER_DIRS:
        return True
    if any(marker in files for marker in ENV_MARKER_FILES):
        return True
    return any(marker in dirs for marker in ENV_MARKER_DIRS)


def _python_files(
    root: str,
    *,
    max_files: Optional[int] = MAX_SCAN_FILES,
    max_seconds: Optional[float] = MAX_SCAN_SECONDS,
) -> Tuple[List[str], bool]:
    """The .py files under ``root``, and whether a bound cut the walk short.

    Returns a list rather than yielding: the count is bounded by ``max_files``, and the
    caller needs the truncation flag as an answer, not as something to infer from how
    many paths happened to arrive.
    """
    started = time.monotonic()
    found: List[str] = []
    truncated = False
    for base, dirs, files in os.walk(root):
        if _is_environment(base, dirs, files):
            # Prune, do not merely skip: descending into site-packages is the whole cost.
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not name.endswith(".py"):
                continue
            if max_files is not None and len(found) >= max_files:
                return found, True
            found.append(os.path.join(base, name))
        if max_seconds is not None and time.monotonic() - started > max_seconds:
            truncated = True
            break
    return found, truncated


def _base_names(cls: ast.ClassDef) -> List[str]:
    names: List[str] = []
    for base in cls.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, (ast.Attribute, ast.Subscript)):
            try:
                names.append(ast.unparse(base))
            except Exception:
                names.append(getattr(base, "attr", ""))
    return names


def scan_source(source: str, *, include_unresolved_bases: bool = False) -> List[Dict[str, Any]]:
    """The ``nn.Module`` classes one buffer declares, with their 1-based line spans.

    Unlike ``scan_repo`` nothing is read from disk: the caller passes the editor's
    current text, so an unsaved edit resolves exactly what the user sees. Syntax errors
    propagate (the router maps them to the parse endpoint's 400 shape). A nested
    helper cannot replace a same-named module-level declaration in this index.
    """
    # The worker also imports constructor helpers from this module in isolation;
    # graph-index dependencies belong only to an actual discovery request.
    from .module_discovery import module_declarations

    classes: List[Dict[str, Any]] = []
    tree = ast.parse(source)
    candidates = module_declarations({"buffer.py": source})
    from services.jax_support.discovery import declarations as jax_declarations
    jax = jax_declarations({"buffer.py": source})
    candidates |= jax
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) or (isinstance(node, ast.FunctionDef) and ("buffer.py", node.lineno, node.name) in jax):
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "end_line": node.end_lineno or node.lineno,
                "bases": _base_names(node) if isinstance(node, ast.ClassDef) else [],
            })
    out = [info for info in classes if ("buffer.py", info["line"], info["name"]) in candidates]
    module_classes = {info["name"] for info in out}
    if include_unresolved_bases:
        # Explicit graph opening may select a library subclass the source-only
        # index cannot prove. Keep it labelled and top-level; the worker checks
        # the actual constructed type before capture. Normal indexes are unchanged.
        top_level = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        out.extend(dict(name=node.name, line=node.lineno, end_line=node.end_lineno or node.lineno,
                        bases=_base_names(node), unverified=True) for node in top_level.values()
                   if node.name not in module_classes and any(base != "object" for base in _base_names(node)))
    out.sort(key=lambda info: (info["line"], info["name"]))
    return out


def _literalness(call: ast.Call) -> Tuple[int, int]:
    values = list(call.args) + [kw.value for kw in call.keywords]
    literal = 0
    for value in values:
        try:
            ast.literal_eval(value)
            literal += 1
        except Exception:
            pass
    return len(values), literal


class RepoScan:
    """The static index of a repository."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.classes: Dict[str, Dict[str, Any]] = {}
        #: Every ``class`` statement in the tree, in scan order, duplicates included.
        #: ``classes`` above is keyed by name and so keeps only the last declaration of
        #: a repeated name; anything that has to answer *per file* — the workspace class
        #: listing behind the "Open as Graph" lens — reads this instead.
        self.class_occurrences: List[Dict[str, Any]] = []
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.calls: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.module_classes: set = set()
        self.contains: Dict[str, set] = defaultdict(set)
        self.contained: set = set()
        self.readme: str = ""
        self.file_count = 0
        #: A bound stopped the walk, so this index is PARTIAL. The caller has to be able
        #: to say so: an index silently missing the user's own model reads as "tensorViz
        #: cannot see my model" with nothing to act on.
        self.truncated = False


def scan_repo(root: str) -> RepoScan:
    from .module_discovery import module_declarations

    scan = RepoScan(os.path.abspath(root))
    paths, scan.truncated = _python_files(scan.root)
    sources = {}
    for path in paths:
        scan.file_count += 1
        try:
            source = open(path, encoding="utf-8", errors="ignore").read()
            tree = ast.parse(source)
        except Exception:
            continue
        rel = os.path.relpath(path, scan.root)
        sources[rel.replace(os.sep, "/")] = source
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                declared = {
                    "name": node.name, "file": rel, "line": node.lineno,
                    "bases": _base_names(node), "node": node,
                }
                scan.classes[node.name] = declared
                scan.class_occurrences.append(declared)
            elif isinstance(node, ast.FunctionDef):
                positional = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
                required = len(positional) - len(node.args.defaults)
                required += sum(1 for d in node.args.kw_defaults if d is None)
                returns = None
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Call):
                        func = sub.value.func
                        if isinstance(func, ast.Name):
                            returns = func.id
                        elif isinstance(func, ast.Attribute):
                            returns = func.attr
                scan.functions[node.name] = {
                    "name": node.name, "file": rel, "line": node.lineno,
                    "required": max(0, required), "returns": returns,
                }
            elif isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else (
                    func.attr if isinstance(func, ast.Attribute) else None
                )
                if name:
                    total, literal = _literalness(node)
                    scan.calls[name].append(
                        {"file": rel, "line": node.lineno, "n": total,
                         "literal": literal, "node": node}
                    )

    candidates = module_declarations(sources)
    from services.jax_support.discovery import declarations as jax_declarations
    jax = jax_declarations(sources)
    candidates |= jax
    for file, source in sources.items():
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and (file, node.lineno, node.name) in jax:
                scan.class_occurrences.append(dict(name=node.name, file=file, line=node.lineno, bases=[], node=node))
    for entry in scan.class_occurrences:
        entry["is_module"] = (entry["file"].replace(os.sep, "/"), entry["line"], entry["name"]) in candidates
    # The legacy entrypoint API is name-keyed; never point it at a same-named helper.
    scan.module_classes = {name for name, info in scan.classes.items() if info["is_module"]}

    for name, info in scan.classes.items():
        for node in ast.walk(info["node"]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                target = node.func.id
                if target in scan.module_classes and target != name:
                    scan.contains[name].add(target)
    scan.contained = {c for group in scan.contains.values() for c in group}

    for candidate in ("README.md", "README.rst", "readme.md", "README.MD", "README.txt"):
        path = os.path.join(scan.root, candidate)
        if os.path.exists(path):
            scan.readme += open(path, encoding="utf-8", errors="ignore").read()
    return scan


def _config_attrs(cls_node: ast.ClassDef) -> Tuple[set, List[str]]:
    """Attributes read off a config parameter inside ``__init__``. This recovers a config
    class even when the parameter carries no type annotation, which is the norm."""
    init = next(
        (n for n in cls_node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    )
    if init is None:
        return set(), []
    names = {a.arg for a in init.args.args if a.arg in CONFIG_PARAM_NAMES}
    attrs = set()
    for node in ast.walk(init):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in names:
                attrs.add(node.attr)
    return attrs, sorted(names)


def rank_entrypoints(scan: RepoScan, limit: int = 5) -> List[EntrypointCandidate]:
    out: List[EntrypointCandidate] = []
    for name in sorted(scan.module_classes):
        info = scan.classes.get(name)
        if info is None:
            continue
        score = 0.0
        reasons: List[str] = []
        sites = scan.calls.get(name, [])
        if any(ENTRY_SCRIPT.search(os.path.basename(s["file"])) for s in sites):
            score += 4
            reasons.append("entry-script call site")
        if re.search(rf"\b{re.escape(name)}\s*\(", scan.readme):
            score += 3
            reasons.append("README instantiation")
        if name not in scan.contained:
            score += 2
            reasons.append("containment root")
        if any("test" in s["file"].lower() for s in sites):
            score += 2
            reasons.append("test call site")
        n_contains = len(scan.contains.get(name, ()))
        if n_contains:
            score += 0.3 * min(n_contains, 10)
            reasons.append(f"contains {n_contains} modules")
        if MODEL_NAME.search(name):
            score += 1
            reasons.append("model-shaped name")
        factories = [
            f["name"] for f in scan.functions.values()
            if f["returns"] == name and f["required"] == 0
        ]
        if factories:
            # Some models are ONLY ever built through a model-zoo function; without this
            # signal dinov2's DinoVisionTransformer never reaches the top 5 at all.
            score += 3
            reasons.append(f"{len(factories)} zero-arg factory function(s)")
        attrs, _ = _config_attrs(info["node"])
        overlap = attrs & TRAINING_ATTRS
        if len(overlap) >= 2:
            score -= 3
            reasons.append(f"training-wrapper attrs {sorted(overlap)[:3]}")
        out.append(
            EntrypointCandidate(
                name=name, score=round(score, 2), file=info["file"], line=info["line"],
                reasons=reasons, factories=factories[:4],
            )
        )
    out.sort(key=lambda c: -c.score)
    return out[:limit]


def _required_init_args(init: ast.FunctionDef) -> List[str]:
    """The ``__init__`` parameters with no default — what a bare call is missing.

    Keyword-only arguments count too: counting ``args.args`` alone reported vit-pytorch's
    ``ViT`` as "all defaulted" when it requires seven kw-only arguments. Shared with
    ``propose_arguments`` so the rung that picks a value and the rung that enumerates
    them cannot disagree about which arguments the question is even about.
    """
    positional = [a.arg for a in init.args.args if a.arg != "self"]
    n_pos_required = len(positional) - len(init.args.defaults)
    kwonly_required = [
        a.arg for a, d in zip(init.args.kwonlyargs, init.args.kw_defaults) if d is None
    ]
    return positional[:n_pos_required] + kwonly_required


def _rank_site(site: Dict[str, Any]) -> Tuple[int, int]:
    """Entry scripts, then tests, then everything else; most literal first."""
    base = os.path.basename(site["file"])
    bucket = 0 if ENTRY_SCRIPT.search(base) else (1 if "test" in base.lower() else 2)
    return bucket, -site["literal"]


def resolve_arguments(scan: RepoScan, name: str) -> Tuple[Optional[str], Any, Dict[str, Any], str]:
    """The deterministic ladder. Returns ``(kind, expr, kwargs, provenance)``.

    Type-hint synthesis is deliberately NOT a rung: it instantiates ~78% of arbitrary
    classes but the dimensions are fiction (a synthesised CLIP came out with
    ``num_heads=0``), and a plausible-but-wrong architecture is worse than a refusal.
    """
    info = scan.classes.get(name)
    if info is None:
        return None, None, {}, f"unknown class {name!r}"

    # R1 — a zero-required-argument factory function. Highest precision: it encodes the
    # author's own hyperparameters.
    factories = [
        f for f in scan.functions.values() if f["returns"] == name and f["required"] == 0
    ]
    if factories:
        chosen = sorted(factories, key=lambda f: -len(f["name"]))[0]
        return ("factory", chosen["name"], {},
                f"R1 factory {chosen['name']}() at {chosen['file']}:{chosen['line']}")

    init = next(
        (n for n in info["node"].body
         if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    )
    if init is None:
        return "class", name, {}, "R2 no __init__ (defaults only)"

    required = _required_init_args(init)

    if not required:
        return "class", name, {}, "R2 all __init__ parameters defaulted"

    # R2b — a zero-argument config classmethod on the class itself.
    for member in info["node"].body:
        if isinstance(member, ast.FunctionDef) and re.match(
            r"(get_)?default_config$|default_cfg$|get_config$", member.name
        ):
            if not [a for a in member.args.args if a.arg not in ("self", "cls")]:
                return ("classmethod", (name, member.name), {},
                        f"R2b {name}.{member.name}() at {info['file']}:{member.lineno}")

    # R3 — a config class recovered by harvesting attribute reads off the config parameter
    # and matching them against a repo-local zero-argument class's field set.
    if len(required) == 1:
        attrs, param_names = _config_attrs(info["node"])
        if attrs:
            best, coverage = None, 0.0
            for cname, cinfo in scan.classes.items():
                fields = {
                    t.target.id for t in cinfo["node"].body
                    if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)
                }
                fields |= {
                    t.targets[0].id for t in cinfo["node"].body
                    if isinstance(t, ast.Assign) and t.targets
                    and isinstance(t.targets[0], ast.Name)
                }
                if not fields:
                    continue
                cinit = next(
                    (n for n in cinfo["node"].body
                     if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
                    None,
                )
                if cinit is not None:
                    cpos = [a.arg for a in cinit.args.args if a.arg != "self"]
                    if len(cpos) - len(cinit.args.defaults) > 0:
                        continue
                score = len(attrs & fields) / max(1, len(attrs))
                if score > coverage:
                    best, coverage = cname, score
            if best and coverage >= 0.6:
                param = param_names[0] if param_names else "config"
                return ("class_with_config", (name, best), {},
                        f"R3 config {best}() covers {coverage:.0%} of "
                        f"{len(attrs)} attrs read off '{param}'")

    # R4 — an all-literal instantiation somewhere in the repo, entry scripts first.
    for site in sorted(scan.calls.get(name, []), key=_rank_site):
        if site["n"] and site["n"] == site["literal"]:
            kwargs: Dict[str, Any] = {}
            for kw in site["node"].keywords:
                try:
                    kwargs[kw.arg] = ast.literal_eval(kw.value)
                except Exception:
                    pass
            if kwargs and set(required).issubset(kwargs):
                return ("class", name, kwargs,
                        f"R4 all-literal call site {site['file']}:{site['line']}")

    # R4b — an all-literal instantiation shown in the README. Research READMEs almost
    # always print the canonical hyperparameters.
    pattern = rf"{re.escape(name)}\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)"
    for match in re.finditer(pattern, scan.readme):
        try:
            call = ast.parse(f"{name}({match.group(1)})", mode="eval").body
            if call.args:
                continue  # positional -> ambiguous
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        except Exception:
            continue
        if kwargs and set(required).issubset(kwargs):
            return ("class", name, kwargs,
                    f"R4b all-literal README instantiation ({len(kwargs)} kwargs)")

    return (None, None, {},
            f"no deterministic source for required args {sorted(required)} "
            "(type-hint synthesis refused: dimensions would be fictional)")


class ArgumentProposal(NamedTuple):
    """Constructor arguments the repository itself already supplies for a class.

    ``kwargs`` is the top-ranked candidate; ``alternatives`` holds the other DISTINCT
    ones behind it. Both are carried on purpose. A caller handed only the winner reports
    one call site while hiding every site that contradicts it — measured on this
    repository's own ``BasicBlock``, whose six sites re-spelled as keywords yield
    ``{'cin': 3, 'cout': 8}`` from ``train.py`` and silently drop ``(32, 32)``,
    ``(32, 64, 2)`` and ``(64, 128, 2)``.
    """

    kwargs: Dict[str, Any]
    provenance: str
    alternatives: List[Tuple[Dict[str, Any], str]]


def _site_kwargs(call: ast.Call, init: ast.FunctionDef) -> Optional[Dict[str, Any]]:
    """Every argument an all-literal call site passes, named — or ``None``.

    Positional arguments are mapped onto ``__init__``'s own parameter names. R4 reads
    ``node.keywords`` alone, so an all-positional site contributes nothing to it: that
    is a fact about how the author spelled the call, not about what the call says, and
    it is why ``BasicBlock`` is refused with a blank today. ``*args`` and ``**kwargs``
    are refused rather than guessed at, because neither carries the names.
    """
    names = [a.arg for a in init.args.posonlyargs + init.args.args if a.arg != "self"]
    if len(call.args) > len(names):
        return None
    kwargs: Dict[str, Any] = {}
    for name, value in zip(names, call.args):
        try:
            kwargs[name] = ast.literal_eval(value)
        except Exception:
            return None
    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        try:
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)
        except Exception:
            return None
    return kwargs


def propose_arguments(
    scan: RepoScan, name: str, limit: int = 5
) -> Optional[ArgumentProposal]:
    """What the repository already passes to ``name.__init__``, ranked, or ``None``.

    The ladder's enumerating twin. Two differences from ``resolve_arguments``, and each
    exists because the answer here is *offered to a user to ratify* rather than passed
    to a constructor:

    * **It stops at no rung.** Only the rungs whose answer can be WRITTEN DOWN are
      walked — R4 call sites and R4b the README. R1 (a factory), R2b (a config
      classmethod) and R3 (a recovered config class) each name a way to *build* the
      object, and ``# tensorviz: init=Cls(...)`` has no spelling for that.
    * **It does not collapse.** The module docstring's rule for entrypoints — "we
      therefore return the ranked TOP-5 and let the caller confirm rather than silently
      auto-picking" — applies with more force to an argument value, because the thing
      that survives the user accepting it is a line of their own source carrying no
      ranking, no provenance and no runner-up.

    Only the REQUIRED arguments are proposed. A call site's optional values are its
    author's taste, and every extra one is another value the reader has to audit before
    ratifying a line they did not write.
    """
    info = scan.classes.get(name)
    if info is None:
        return None
    init = next(
        (n for n in info["node"].body
         if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    )
    if init is None:
        return None
    required = _required_init_args(init)
    if not required:
        return None

    candidates: List[Tuple[Dict[str, Any], str]] = []
    seen: set = set()

    def offer(kwargs: Optional[Dict[str, Any]], provenance: str) -> None:
        if not kwargs or not set(required).issubset(kwargs):
            return
        chosen = {key: kwargs[key] for key in required}
        # repr, not a frozenset: a value may be a list or a dict, and two sites that
        # agree are one candidate however their arguments were spelled.
        fingerprint = repr(sorted(chosen.items(), key=lambda item: item[0]))
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        candidates.append((chosen, provenance))

    for site in sorted(scan.calls.get(name, []), key=_rank_site):
        if site["n"] and site["n"] == site["literal"]:
            offer(
                _site_kwargs(site["node"], init),
                f"R4 all-literal call site {site['file']}:{site['line']}",
            )

    pattern = rf"{re.escape(name)}\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)"
    for match in re.finditer(pattern, scan.readme):
        try:
            call = ast.parse(f"{name}({match.group(1)})", mode="eval").body
        except Exception:
            continue
        if isinstance(call, ast.Call):
            offer(_site_kwargs(call, init), "R4b all-literal README instantiation")

    if not candidates:
        return None
    return ArgumentProposal(candidates[0][0], candidates[0][1], candidates[1:limit])


class _StubModule(types.ModuleType):
    """Stands in for a missing third-party dependency that the model definition imports
    but never uses. Permissive on attribute access and calls so a decorator or a constant
    read at import time resolves."""

    def __getattr__(self, key: str) -> Any:  # noqa: D105
        if key.startswith("__"):
            raise AttributeError(key)
        child = _StubModule(f"{self.__name__}.{key}")
        child.__path__ = []  # type: ignore[attr-defined]
        setattr(self, key, child)
        return child

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D105
        return _StubModule(self.__name__)

    def __iter__(self):  # noqa: D105
        return iter(())


def import_with_stubs(module_path: str, roots: Sequence[str]) -> Tuple[Any, List[str]]:
    """Import ``module_path`` from a repository, stubbing unresolved dependencies.

    Two hazards, both hit in practice:

    * **Name collision with the backend's own modules.** Research repositories very often
      have a top-level ``models.py``, ``utils.py`` or ``core.py`` — exactly the names this
      backend already occupies. Since the backend imports its own ``models`` package at
      startup, a plain ``import_module("models")`` returns THAT one and the lookup fails
      with ``module 'models' has no attribute 'DiT_XL_2'``. So any already-imported module
      holding the name, whose file lives outside the repository, is temporarily evicted for
      the duration of the import and restored afterwards.
    * **Missing third-party imports.** Mostly training dependencies pulled in at module
      scope but never used to define the model, so an unresolved import is stubbed and the
      import retried. The stub must register the FULL dotted name and every parent WITH
      ``__path__`` or ``from pkg.sub import X`` fails with "'pkg' is not a package".
    """
    repo_root = os.path.abspath(roots[0]) if roots else ""
    for root in reversed([r for r in roots if os.path.isdir(r)]):
        absolute = os.path.abspath(root)
        if absolute in sys.path:
            sys.path.remove(absolute)
        sys.path.insert(0, absolute)  # the repository must win over the backend

    top = module_path.split(".")[0]
    shadowed: Dict[str, Any] = {}
    for key in list(sys.modules):
        if key != top and not key.startswith(top + "."):
            continue
        existing = sys.modules[key]
        origin = getattr(existing, "__file__", None) or ""
        if isinstance(existing, _StubModule):
            continue
        if not origin or not os.path.abspath(origin).startswith(repo_root):
            shadowed[key] = sys.modules.pop(key)

    stubbed: List[str] = []
    installed: List[str] = []
    noise = io.StringIO()
    try:
        with redirect_stdout(noise), redirect_stderr(noise):
            for _ in range(MAX_STUBS):
                try:
                    return importlib.import_module(module_path), stubbed
                except ModuleNotFoundError as exc:
                    missing = exc.name or ""
                    if not missing or missing in stubbed or missing == top:
                        raise
                    parts = missing.split(".")
                    for i in range(1, len(parts) + 1):
                        dotted = ".".join(parts[:i])
                        if not isinstance(sys.modules.get(dotted), _StubModule):
                            stub = _StubModule(dotted)
                            stub.__path__ = []  # type: ignore[attr-defined]
                            sys.modules[dotted] = stub
                            installed.append(dotted)
                    stubbed.append(missing)
            return importlib.import_module(module_path), stubbed
    finally:
        # Give the backend its own modules back. The imported object is already held by
        # the caller, so dropping the repository's entry here is harmless.
        for key in [k for k in sys.modules if k == top or k.startswith(top + ".")]:
            if key in shadowed:
                continue
            sys.modules.pop(key, None)
        sys.modules.update(shadowed)
        # And take the stubs back out. Left behind they make the whole process believe the
        # package is installed: after one repo import stubbed ``timm``, /api/ingest/libraries
        # reported ``timm: true`` and every later timm request died on
        # "type object '_StubModule' has no attribute 'forward'". Leaving them also emptied
        # the stub list on the SECOND import of the same repo, which is the list the
        # diagnostics report. The imported module holds its own reference, so anything
        # already bound still resolves.
        for dotted in installed:
            if isinstance(sys.modules.get(dotted), _StubModule):
                sys.modules.pop(dotted, None)


class StubbedDependencyError(RuntimeError):
    """A stubbed dependency turned out to be load-bearing.

    Stubbing an unresolved import is only safe when the model definition never *uses* it.
    When it does — DiT builds real ``timm`` layers, denoising-diffusion-pytorch builds
    ``einops`` layers — the stub reaches real code and fails somewhere far away with an
    unhelpful message like ``zeros(): argument 'size' failed to unpack``. Naming the
    packages to install turns that into something the user can act on.

    Raised for the loud failure above AND for the silent one: a construction that
    *succeeds* while keeping a stub as part of the model (see ``surviving_stubs``).
    """

    def __init__(self, stubbed: List[str], cause: Any) -> None:
        self.stubbed = stubbed
        names = sorted({s.split(".")[0] for s in stubbed})
        detail = cause if isinstance(cause, str) else f"{type(cause).__name__}: {cause}"
        super().__init__(
            f"this model genuinely uses {', '.join(names)}, which is not installed — "
            f"install and retry: pip install {' '.join(names)}. "
            f"Underlying error: {detail}"
        )


def surviving_stubs(model: Any) -> List[Tuple[str, str]]:
    """``(attribute path, stubbed dotted name)`` for every stub a CONSTRUCTED model kept.

    A construction that does not raise is not evidence that the stub went unused.
    ``_StubModule.__call__`` returns another ``_StubModule``, so
    ``self.backbone = timm.create_model(...)`` succeeds; and because ``_StubModule`` is a
    ``types.ModuleType`` rather than an ``nn.Module``, ``nn.Module.__setattr__`` files it
    in ``__dict__`` instead of ``_modules``, so ``named_modules()`` never yields it and
    the whole backbone is missing from the graph with nothing recorded. This module's
    contract is that a silently incomplete graph is worse than an honest one, so the
    caller refuses rather than reporting success.

    Any surviving stub counts, including one held for a non-architectural reason (a
    logger, a config object): there is no way to tell those apart from a dropped
    backbone, and the remedy — install the package and retry — is the same either way.

    The package name comes off the stub itself rather than off this import's stub list.
    Those lists went empty on the SECOND ingest of a repository in one process, because
    ``import_with_stubs`` used to leave its stubs in ``sys.modules`` and the retry then
    imported cleanly — measured on the running server, which is long-lived. That leak is
    closed, but reading the name off the object is what makes this independent of it.
    """
    if isinstance(model, _StubModule):
        return [("<the model itself>", model.__name__)]
    named = getattr(model, "named_modules", None)
    scopes = list(named()) if callable(named) else [("", model)]
    return [
        (f"{prefix}.{key}" if prefix else key, value.__name__)
        for prefix, scope in scopes
        for key, value in vars(scope).items()
        if isinstance(value, _StubModule)
    ]


def instantiate(
    scan: RepoScan, name: str, kind: str, expr: Any, kwargs: Dict[str, Any],
    file: Optional[str] = None,
    resolve: Optional[Callable[[str, Any, Any], Any]] = None,
    construction_mode: Any = None,
) -> Tuple[Any, List[str]]:
    """Import the defining module and construct the model. Executes repository code.

    ``file`` overrides the name-keyed lookup below with the path the caller
    already knows (the IDE worker always does).

    ``resolve`` is called with ``(name, value, module)`` for each keyword argument
    just before the call. It exists for an argument that can only be BUILT inside
    the imported module — ``# tensorviz: init=Net(cfg=ModelConfig())`` names a class
    that lives in the user's file, so there is nothing to pass until that file has
    been imported, and importing it here a second time would run its module scope
    twice. This function stays ignorant of what the values mean; the caller that
    wrote the grammar decides.

    ``construction_mode`` (a context manager, e.g. a ``FakeTensorMode``) is entered
    around the construction only, never around the import: module-level code runs
    exactly as it does without it, so module constants stay real tensors and a later
    real run in the same process sees the module it would have seen anyway. None
    (the default) constructs exactly as before.
    """
    if file is not None:
        module_path = module_path_for(scan.root, file)
    else:
        if kind == "factory":
            rel = next(f["file"] for f in scan.functions.values() if f["name"] == expr)
        elif kind in ("class_with_config", "classmethod"):
            rel = scan.classes[expr[0]]["file"]
        else:
            rel = scan.classes[name]["file"]
        module_path = rel[:-3].replace(os.sep, ".")
    roots = [scan.root, os.path.join(scan.root, "src")]
    module, stubbed = import_with_stubs(module_path, roots)

    def construct() -> Any:
        if kind == "factory":
            return getattr(module, expr)()
        if kind == "classmethod":
            cls_name, method = expr
            cls = getattr(module, cls_name)
            return cls(getattr(cls, method)())
        if kind == "class_with_config":
            cls_name, cfg_name = expr
            cfg_holder = module
            if not hasattr(module, cfg_name):
                cfg_rel = scan.classes[cfg_name]["file"]
                cfg_holder, _ = import_with_stubs(cfg_rel[:-3].replace(os.sep, "."), roots)
            return getattr(module, cls_name)(getattr(cfg_holder, cfg_name)())
        values = kwargs or {}
        if resolve is not None:
            values = {key: resolve(key, value, module) for key, value in values.items()}
        target = getattr(module, name)
        from ..validation.setup import bind_call
        bind_call(target, (), values, "init")
        return target(**values)

    try:
        if construction_mode is None:
            model = construct()
        else:
            with construction_mode:
                model = construct()
    except Exception as exc:
        if stubbed:
            raise StubbedDependencyError(stubbed, exc) from exc
        raise

    survivors = surviving_stubs(model)
    if survivors:
        paths = [path for path, _ in survivors]
        raise StubbedDependencyError(
            sorted(set(stubbed) | {dotted for _, dotted in survivors}),
            f"construction succeeded but {', '.join(paths)} "
            f"{'is' if len(paths) == 1 else 'are'} a stub, so that part of the model "
            "does not exist and would be missing from the graph without any sign of it",
        )
    return model, stubbed
