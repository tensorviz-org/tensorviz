"""Deciding what a piece of code *is*, behind the seam an adapter will later fill.

Today there is exactly one implementation and no registry, no plugin loading and no
precedence engine — `docs/research-graph-adapters.md` records why that layer is
deliberately deferred. What is here now is the shape it will take, so adding the
second implementation is opening a seam rather than unpicking forty lines of
`if`/`elif` welded into the indexer.

Two kinds of recognition, and the second is the one that matters:

* **Definitions** — `class Net(nn.Module)`, `def train(...)`. Classic PyTorch code
  subclasses and defines, so base classes and call shapes answer it.

* **Usage** — `trainer = SFTTrainer(...)`, `ds = load_dataset("imdb")`. Most modern
  research code *instantiates library classes* rather than subclassing them, so a
  repo built on HF or Lightning has almost no classifiable definitions at all. A
  recogniser that only reads `ClassDef`/`FunctionDef` renders such a project as a
  single `generic` box. Both halves live here.

Every rule states itself. `Classification.reason` carries the sentence the canvas
shows on hover — "classified as `dataset` because it subclasses
`torch.utils.data.Dataset`" — because a guess the user cannot interrogate is a guess
they cannot correct, and correcting it (via an SDK declaration) is the only input
channel a read-only surface has.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Set

from research.contracts import NodeKind, Provenance

# ── what a scan of one body collected ───────────────────────────────────────────


@dataclass
class BodyScan:
    """Names referenced, calls made, and a few specific facts worth singling out."""

    references: Set[str] = field(default_factory=set)
    calls: Set[str] = field(default_factory=set)
    saves_checkpoint: bool = False
    logs_metrics: Set[str] = field(default_factory=set)


@dataclass
class Classification:
    kind: NodeKind
    #: Why, in the words the canvas shows on hover. Never empty.
    reason: str
    provenance: Provenance = "static"
    params: Dict[str, object] = field(default_factory=dict)


@dataclass
class UsageSite:
    """A library class instantiated or a library function called, as a graph node.

    `binding` is the variable it was assigned to, which is how edges find it again:
    `trainer = SFTTrainer(model=m, train_dataset=ds)` binds `trainer`, and `ds` in
    its keyword arguments names another binding in the same file.
    """

    binding: str
    callee: str
    kind: NodeKind
    reason: str
    lineno: int
    end_lineno: int
    col: int
    end_col: int
    #: Keyword arguments whose values are plain names, for edge inference.
    name_kwargs: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, object] = field(default_factory=dict)


class Classifier(Protocol):
    """The seam. An adapter is an implementation of this plus an edit contract."""

    name: str

    def claims(self, imports: Set[str]) -> bool:
        """Does this classifier apply to a file importing these modules?"""

    def classify_class(self, node: ast.ClassDef, scan: BodyScan) -> Optional[Classification]:
        ...

    def classify_function(
        self, name: str, scan: BodyScan, node: Optional[ast.AST] = None
    ) -> Optional[Classification]:
        ...

    def usage_sites(self, tree: ast.AST) -> List[UsageSite]:
        """Library classes instantiated or called at module or function scope."""


# ── the one implementation: the PyTorch ecosystem ───────────────────────────────

#: `nn.` members that are not Modules. Everything else public in `torch.nn` is one,
#: so this is a deny-list rather than the impossible allow-list of every layer torch
#: ships — and a new layer class in a future torch release is recognised for free.
NOT_MODULES = {"Parameter", "UninitializedParameter", "UninitializedBuffer", "functional", "init", "utils"}


def _is_torch_nn_class(base: str) -> bool:
    """Is `base` a class out of `torch.nn`?"""
    for prefix in ("nn.", "torch.nn."):
        if base.startswith(prefix):
            name = base[len(prefix):]
            return bool(name) and "." not in name and name not in NOT_MODULES and name[:1].isupper()
    return False


MODEL_BASES = {
    "nn.Module",
    "torch.nn.Module",
    "Module",
    "pl.LightningModule",
    "LightningModule",
    "pytorch_lightning.LightningModule",
    "lightning.LightningModule",
    "PreTrainedModel",
    "transformers.PreTrainedModel",
}

DATASET_BASES = {
    "Dataset",
    "data.Dataset",
    "torch.utils.data.Dataset",
    "utils.data.Dataset",
    "IterableDataset",
    "torch.utils.data.IterableDataset",
}

CONFIG_CALLS = {
    "json.load",
    "yaml.safe_load",
    "yaml.load",
    "toml.load",
    "argparse.ArgumentParser",
    "ArgumentParser",
    "OmegaConf.load",
    "OmegaConf.create",
}

TRANSFORM_CALLS = {"Compose", "transforms.Compose", "T.Compose", "A.Compose"}

LOGGING_CALLS = {
    "wandb.log",
    "mlflow.log_metric",
    "mlflow.log_metrics",
    "writer.add_scalar",
    "self.log",
    "accelerator.log",
}

CHECKPOINT_CALLS = {
    "torch.save",
    "save_checkpoint",
    "save_pretrained",
    "accelerator.save_state",
}

#: Instantiating one of these *is* the thing, even though the user defined no class.
#: This is the half a definition-only recogniser misses entirely.
USAGE_CLASSES: Dict[str, NodeKind] = {
    # trainers
    "Trainer": "trainer",
    "SFTTrainer": "trainer",
    "DPOTrainer": "trainer",
    "PPOTrainer": "trainer",
    "GRPOTrainer": "trainer",
    "RewardTrainer": "trainer",
    "Seq2SeqTrainer": "trainer",
    "Accelerator": "trainer",
    # data
    "DataLoader": "dataset",
    "TensorDataset": "dataset",
    "ConcatDataset": "dataset",
    # configuration
    "TrainingArguments": "configuration",
    "Seq2SeqTrainingArguments": "configuration",
    "SFTConfig": "configuration",
    "DPOConfig": "configuration",
    "GRPOConfig": "configuration",
    "PPOConfig": "configuration",
    "LoraConfig": "configuration",
    "BitsAndBytesConfig": "configuration",
    "GenerationConfig": "configuration",
}

USAGE_FUNCTIONS: Dict[str, NodeKind] = {
    "load_dataset": "dataset",
    "load_from_disk": "dataset",
    "from_pretrained": "model",
    "AutoModel.from_pretrained": "model",
    "AutoModelForCausalLM.from_pretrained": "model",
    "AutoModelForSequenceClassification.from_pretrained": "model",
    "AutoTokenizer.from_pretrained": "transform",
    "get_peft_model": "model",
    "load_metric": "metric",
    "evaluate.load": "metric",
}

#: Keyword arguments that mean "this node feeds that one", and with what relationship.
KWARG_RELATIONSHIPS: Dict[str, str] = {
    "train_dataset": "feeds",
    "eval_dataset": "evaluates",
    "dataset": "feeds",
    "data_collator": "transforms",
    "tokenizer": "transforms",
    "processing_class": "transforms",
    "model": "trains",
    "ref_model": "trains",
    "args": "runs_on",
    "peft_config": "runs_on",
    "optimizers": "runs_on",
}

_EVAL_NAME = re.compile(r"^(eval|evaluate|validate|test)(_|$)")
_DATA_NAME = re.compile(r"(^|_)(data|dataset|loader|dataloader|batch|corpus|tokenize|tokenise)s?($|_)")
_CONFIG_NAME = re.compile(r"(^|_)(config|cfg|args|hparams)($|_)")
_METRIC_NAME = re.compile(r"(^|_)(metric|accuracy|f1|score|loss)s?($|_)")


def dotted(node: ast.AST) -> Optional[str]:
    """`a.b.c` for an attribute or name chain; None for anything computed."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _tail(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def literal(node: ast.AST) -> Optional[object]:
    """A literal value, or None. Never evaluates; never imports."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)) and all(
        isinstance(e, ast.Constant) for e in node.elts
    ):
        return [e.value for e in node.elts]  # type: ignore[attr-defined]
    return None


class BodyVisitor(ast.NodeVisitor):
    """Collects calls, references and a few flagged facts from one body."""

    def __init__(self) -> None:
        self.scan = BodyScan()

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = dotted(node.func)
        if name:
            self.scan.calls.add(name)
            if name in CHECKPOINT_CALLS or _tail(name) in {"save_checkpoint", "save_pretrained"}:
                self.scan.saves_checkpoint = True
            if name in LOGGING_CALLS or _tail(name) in {"add_scalar", "log_metric"}:
                key = literal(node.args[0]) if node.args else None
                if isinstance(key, str):
                    self.scan.logs_metrics.add(key)
        root = _root_name(node.func)
        if root:
            self.scan.references.add(root)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            self.scan.references.add(node.id)

    def visit_With(self, node: ast.With) -> None:  # noqa: N802
        for item in node.items:
            expr = item.context_expr
            name = dotted(expr.func) if isinstance(expr, ast.Call) else dotted(expr)
            if name:
                self.scan.calls.add(name)
        self.generic_visit(node)


def _root_name(node: ast.AST) -> Optional[str]:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def scan_body(body: Sequence[ast.stmt]) -> BodyScan:
    visitor = BodyVisitor()
    for stmt in body:
        visitor.visit(stmt)
    return visitor.scan


class TorchEcosystemClassifier:
    """torch, Lightning, HuggingFace and TRL — the mainstream research stack."""

    name = "torch-ecosystem"

    def claims(self, imports: Set[str]) -> bool:
        #: Claims everything for now: it is the only implementation, and refusing a
        #: file because it imports nothing recognised would leave it unclassified
        #: rather than `generic`.
        return True

    # definitions ───────────────────────────────────────────────────────────────

    def classify_class(self, node: ast.ClassDef, scan: BodyScan) -> Optional[Classification]:
        bases = {dotted(b) or "" for b in node.bases}
        hit = bases & MODEL_BASES
        if hit:
            return Classification("model", f"subclasses {sorted(hit)[0]}")
        # Anything else out of `torch.nn` is a Module by construction. Whisper
        # overrides three of them — `class Linear(nn.Linear)`, `LayerNorm`, `Conv1d`
        # — to change one line of forward, and all three came out "a class with no
        # recognised role" in a repository whose whole subject is a model.
        layer = sorted(base for base in bases if _is_torch_nn_class(base))
        if layer:
            return Classification("model", f"subclasses {layer[0]}")
        hit = bases & DATASET_BASES
        if hit:
            return Classification("dataset", f"subclasses {sorted(hit)[0]}")
        settings = settings_of(node)
        if settings and (
            lowered_name := node.name.lower()
        ).endswith(("config", "cfg", "settings", "args", "params", "hparams")):
            return Classification(
                "configuration",
                f"declares {len(settings)} setting{'' if len(settings) == 1 else 's'}",
                params={"settings": settings},
            )
        lowered = node.name.lower()
        if _trains(scan):
            return Classification("trainer", "calls .backward() and an optimizer .step()")
        if lowered.endswith("trainer"):
            return Classification("trainer", "named like a trainer")
        if lowered.endswith("evaluator"):
            return Classification("evaluator", "named like an evaluator")
        if lowered.endswith("metric"):
            return Classification("metric", "named like a metric")
        if lowered.endswith(("config", "cfg", "settings")):
            return Classification("configuration", "named like a configuration")
        return Classification("generic", "a class with no recognised role", provenance="inferred")

    def classify_function(
        self, name: str, scan: BodyScan, node: Optional[ast.AST] = None
    ) -> Optional[Classification]:
        lowered = name.lower()
        if _trains(scan):
            return Classification("trainer", "calls .backward() and an optimizer .step()")
        if scan.saves_checkpoint:
            # Writing a checkpoint does not make something a trainer. `save_checkpoint`
            # is a utility the trainer calls, and classifying it as a trainer put a
            # second, wrong trainer in the graph for the spine rule to wire up.
            return Classification("checkpoint", "writes a checkpoint")
        if _EVAL_NAME.match(lowered):
            return Classification("evaluator", "named like an evaluation step")
        if "torch.no_grad" in scan.calls and any(c.endswith(".eval") for c in scan.calls):
            return Classification("evaluator", "runs under torch.no_grad() after .eval()")
        if scan.calls & TRANSFORM_CALLS or any(_tail(c) == "Compose" for c in scan.calls):
            return Classification("transform", "builds a Compose pipeline")
        if _DATA_NAME.search(lowered) or any(_tail(c) == "DataLoader" for c in scan.calls):
            return Classification("dataset", "builds or names a data loader")
        if scan.calls & CONFIG_CALLS or _CONFIG_NAME.search(lowered):
            settings = settings_of(node) if node is not None else {}
            if settings:
                return Classification(
                    "configuration",
                    f"declares {len(settings)} setting{'' if len(settings) == 1 else 's'}",
                    params={"settings": settings},
                )
            return Classification("configuration", "reads configuration")
        if _METRIC_NAME.search(lowered):
            return Classification("metric", "named like a metric")
        return Classification("function", "a function with no recognised role", provenance="inferred")

    # usage ─────────────────────────────────────────────────────────────────────

    def usage_sites(self, tree: ast.AST) -> List[UsageSite]:
        """`trainer = SFTTrainer(...)` and friends — the pipeline, not its innards.

        Module level and the bodies of top-level functions, because research scripts
        routinely build the pipeline inside `main()`. But NOT inside a class: a local
        in `GPT.from_pretrained()` is that class's own business, and surfacing it put
        nanoGPT's internal `model_hf` on the canvas as a project-level model. On a
        large repository that leak would flood the graph with private variables.
        """
        scoped: List[ast.AST] = []
        for stmt in getattr(tree, "body", []):
            if isinstance(stmt, ast.ClassDef):
                continue
            scoped.append(stmt)

        sites: List[UsageSite] = []
        for root in scoped:
            for node in ast.walk(root):
                if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                    continue
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                    continue
                binding = node.targets[0].id
                call = node.value
                callee = dotted(call.func)
                if not callee:
                    continue
                kind, reason = _usage_kind(callee)
                if kind is None:
                    continue
                name_kwargs: Dict[str, str] = {}
                params: Dict[str, object] = {}
                for kw in call.keywords:
                    if kw.arg is None:
                        continue
                    value = literal(kw.value)
                    if value is not None:
                        params[kw.arg] = value
                    elif isinstance(kw.value, ast.Name):
                        name_kwargs[kw.arg] = kw.value.id
                first = literal(call.args[0]) if call.args else None
                label_hint = first if isinstance(first, str) else None
                if label_hint:
                    params["_label_hint"] = label_hint
                sites.append(
                    UsageSite(
                        binding=binding,
                        callee=callee,
                        kind=kind,
                        reason=reason,
                        lineno=node.lineno,
                        end_lineno=node.end_lineno or node.lineno,
                        col=node.col_offset,
                        end_col=node.end_col_offset or 0,
                        name_kwargs=name_kwargs,
                        params=params,
                    )
                )
        return sites


#: How many settings a card shows before it stops being readable. A config with
#: eighty fields is a file to open, not a card to read.
MAX_SETTINGS = 12


def settings_of(node: ast.AST) -> Dict[str, object]:
    """The hyperparameters a definition declares, as literals.

    Three shapes, because those are the three that actually appear:

    * a dataclass — `n_layer: int = 12`. nanoGPT's `GPTConfig` is seven of these,
      and a card that says "named like a configuration" while discarding them tells
      the reader nothing they did not already know from the name;
    * a class body of plain assignments — the same thing without the decorator;
    * `parser.add_argument("--lr", default=3e-4)` inside a function.

    Literals only. A default computed at runtime (`default=os.cpu_count()`) is a
    value this cannot know without running the code, and showing the expression
    text instead would be a different kind of lie.
    """
    found: Dict[str, object] = {}

    if isinstance(node, ast.ClassDef):
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                value = literal(stmt.value) if stmt.value is not None else None
                if value is not None:
                    found[stmt.target.id] = value
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                value = literal(stmt.value)
                if isinstance(target, ast.Name) and value is not None:
                    found[target.id] = value

    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call):
            continue
        if _tail(dotted(inner.func) or "") != "add_argument":
            continue
        flag = literal(inner.args[0]) if inner.args else None
        if not isinstance(flag, str):
            continue
        name = flag.lstrip("-").replace("-", "_")
        default = None
        for kw in inner.keywords:
            if kw.arg == "default":
                default = literal(kw.value)
        if name:
            found[name] = default

    return dict(list(found.items())[:MAX_SETTINGS])


def _usage_kind(callee: str) -> tuple:
    tail = _tail(callee)
    if callee in USAGE_FUNCTIONS:
        return USAGE_FUNCTIONS[callee], f"calls {callee}()"
    if tail in USAGE_FUNCTIONS:
        return USAGE_FUNCTIONS[tail], f"calls {tail}()"
    if callee in USAGE_CLASSES:
        return USAGE_CLASSES[callee], f"instantiates {callee}"
    if tail in USAGE_CLASSES:
        return USAGE_CLASSES[tail], f"instantiates {tail}"
    if tail == "from_pretrained" or callee.endswith(".from_pretrained"):
        return "model", "loads a pretrained model"
    return None, ""


def _trains(scan: BodyScan) -> bool:
    """Backward pass, then an optimizer step. The signature of a training loop.

    Matched on the call's tail rather than a dotted suffix, because a chained call
    loses its receiver: `scaler.scale(loss).backward()` resolves to a bare
    `backward`, and requiring `.backward` missed nanoGPT's loop entirely — which is
    to say it missed the canonical PyTorch training script.
    """
    return any(_tail(c) == "backward" for c in scan.calls) and any(
        _tail(c) == "step" for c in scan.calls
    )


#: The single classifier in use. Phase D of the plan turns this into a registry.
DEFAULT_CLASSIFIER: Classifier = TorchEcosystemClassifier()
