"""Static introspection of an instantiated ``nn.Module``.

Everything here reads the live object and never executes its ``forward``, so it cannot
fail on control flow. That property is what makes it the always-available fallback: a
model whose trace dies still yields a complete hierarchy.

Three independent sources of a repeat count, all measured as necessary:

* ``module-list``   — N structurally identical ``nn.ModuleList``/``Sequential`` siblings.
* ``call-count``    — one module *called* N times in a Python loop. ALBERT with
  ``num_hidden_groups=1`` has 28 modules but 89 hook events, 13 of them firing exactly
  6x; without this rule its six layers collapse to ``repeat_count=1`` and a 6-layer
  model renders as a 1-layer model.
* ``fused-parameter`` — modern Mixture-of-Experts layers do not use modules at all. An
  8-expert Mixtral (transformers 5.x) has only 19 ``named_modules``; the experts exist
  solely as a leading axis on fused 3-D parameters (``gate_up_proj`` of shape
  ``(8, 256, 64)``). Without this rule an 8-expert sparse layer renders as one dense box.
"""

from __future__ import annotations

import ast
import functools
import inspect
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .spec import RepeatGroup

# Python containers, not architectural concepts. They get no card: their children are
# hoisted to the container's own parent and do not consume a depth level. Left in, a
# Stable Diffusion UNet renders four meaningless "ModuleList" boxes with all the
# interesting skip topology hidden inside them.
TRANSPARENT_CONTAINERS = (nn.ModuleList, nn.Sequential, nn.ModuleDict)

# Leaf classes that should stay a plain layer card even though they may have children.
LEAF_LIKE = {
    "Linear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose2d", "Embedding",
    "LayerNorm", "GroupNorm", "BatchNorm1d", "BatchNorm2d", "RMSNorm", "Dropout",
    "GELU", "ReLU", "SiLU", "Softmax", "Flatten", "Identity", "Tanh", "Sigmoid",
    "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d",
}

# Expected parameter rank per op family. A leaf holding a parameter of rank one HIGHER
# than expected is carrying a fused instance axis (the MoE case).
_EXPECTED_PARAM_RANK = {"Linear": 2, "Embedding": 2, "Conv1d": 3, "Conv2d": 4, "Conv3d": 5}


@functools.lru_cache(maxsize=256)
def _class_lines(file: str) -> Dict[str, int]:
    """``{class name: definition line}`` for one file, read with ``ast``.

    The fallback for the line number when ``inspect`` cannot find the source at
    all. Cached per file because a model has many instances of few classes.
    """
    try:
        with open(file, encoding="utf-8", errors="ignore") as handle:
            tree = ast.parse(handle.read())
    except (OSError, SyntaxError, ValueError):
        return {}
    lines: Dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            lines.setdefault(node.name, node.lineno)
    return lines


#: ``# tensorviz: out_shape=(2, 16)`` written inside a class body. The user's
#: answer for a block the trace never reached: the class states what it produces
#: so the cards downstream of it are not all left unknown (task.md M6).
_OUT_SHAPE_COMMENT = re.compile(r"#\s*tensorviz:\s*out_shape\s*=\s*\(?(?P<value>[^)\n]*)\)?")


@functools.lru_cache(maxsize=256)
def _class_annotations(file: str) -> Dict[str, str]:
    """``{class name: the out_shape text declared in its body}`` for one file.

    Read with ``ast`` for the class's line range and with plain text for the
    comment itself, because a comment is not in the tree. Cached per file for the
    same reason ``_class_lines`` is: a model has many instances of few classes.
    """
    try:
        with open(file, encoding="utf-8", errors="ignore") as handle:
            source = handle.read()
        tree = ast.parse(source)
    except (OSError, SyntaxError, ValueError):
        return {}
    lines = source.splitlines()
    found: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        for line in lines[node.lineno - 1 : end]:
            match = _OUT_SHAPE_COMMENT.search(line)
            if match:
                found.setdefault(node.name, match.group("value").strip())
                break
    return found


def declared_out_shape(file: str, cls: str) -> Optional[Tuple[str, Optional[List[int]]]]:
    """``(the declaration as written, its dims)`` for a class, or None.

    The dims are None unless EVERY axis is an integer literal. ``(B, D)`` names
    the axes and no numbers, so it is a declaration this can record and not a
    shape it can hand on — inventing a number for ``B`` would put a size the user
    never wrote in front of every node downstream, which is the one thing a
    declared shape must not do.
    """
    raw = _class_annotations(file).get(cls)
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    dims: List[int] = []
    for part in parts:
        try:
            dims.append(int(part))
        except ValueError:
            return raw, None
    return raw, (dims or None)


def class_source(cls: type) -> Tuple[Optional[str], Optional[int]]:
    """``(file, first line)`` of a class definition, or ``(None, None)``.

    None-safe by contract, because every way of asking can fail for a reason that
    is not the caller's fault. ``inspect`` is tried first and answers for every
    ordinary class; it fails for a C-implemented one, and — the case that bites
    here — for any class whose module the repository importer has since evicted
    from ``sys.modules`` (``repo_scan.import_with_stubs`` restores the backend's
    own ``models`` package on the way out, and ``inspect.getfile`` resolves a
    class through ``sys.modules[cls.__module__]``).

    So the fallback asks the class itself: a Python method carries the defining
    module's namespace in ``__globals__``, which holds ``__file__`` whether or not
    the module is still importable. The line then comes from parsing that file.

    A user's own ``nn.Module`` is the case that matters: "reveal this card in the
    editor" needs the file the class was *defined* in, which is not the file the
    graph was opened from once a model imports its blocks from elsewhere.
    """
    try:
        file = inspect.getsourcefile(cls)
        if file:
            return file, int(inspect.getsourcelines(cls)[1])
    except (TypeError, OSError, IndexError):
        pass
    file = None
    for attribute in ("__init__", "forward"):
        function = getattr(cls, attribute, None)
        namespace = getattr(function, "__globals__", None)
        candidate = (namespace or {}).get("__file__")
        if isinstance(candidate, str):
            file = candidate
            break
    if file is None:
        return None, None
    return file, _class_lines(file).get(cls.__name__)


class ModuleNode:
    """One entry in the materialised module tree."""

    # Widened for the IDE (task.md M4): a card that cannot say where its class was
    # written cannot be revealed in an editor. __slots__ is not optional here — the
    # assignment below raises AttributeError without the two new names.
    __slots__ = (
        "path", "cls", "depth", "parent", "is_leaf", "params", "own_params",
        "source_file", "source_line", "decl_file", "decl_line",
        "capture_depth_limit",
    )

    def __init__(self, path: str, mod: nn.Module, depth: int, parent: Optional[str]):
        self.path = path
        self.cls = type(mod).__name__
        self.depth = depth
        self.parent = parent
        self.is_leaf = not list(mod.children())
        self.capture_depth_limit: Optional[int] = None
        self.params = sum(p.numel() for p in mod.parameters())
        self.own_params = sum(p.numel() for p in mod.parameters(recurse=False))
        self.source_file, self.source_line = class_source(type(mod))
        # Where this node RAN, in the file being projected — filled from the tracer's
        # per-op frames (task.md M4), and distinct from the class definition above:
        # an nn.Linear is defined in torch and declared on line 22 of the user's file,
        # and the editor wants the second one.
        self.decl_file: Optional[str] = None
        self.decl_line: Optional[int] = None

    def renders_as_module(self) -> bool:
        return not self.is_leaf and self.cls not in LEAF_LIKE


def structural_signature(mod: nn.Module) -> str:
    """A deterministic fingerprint of a subtree's SHAPE — class names plus parameter
    shapes, never values. Two siblings with the same signature are the same block."""
    parts = []
    for name, sub in mod.named_modules():
        shapes = [tuple(t.shape) for _, t in sub.named_parameters(recurse=False)]
        parts.append(f"{name}:{type(sub).__name__}:{shapes}")
    return "|".join(parts)


def constructor_kwargs(mod: nn.Module) -> Dict[str, Any]:
    """Recover a module's real constructor arguments by intersecting its ``__init__``
    signature with its own attributes.

    Required, not cosmetic: with only a class name the backend layer builder dies with
    ``GroupNorm.__init__() missing 2 required positional arguments: 'num_groups' and
    'num_channels'``. Verified output for a real SD UNet leaf:
    ``{'num_groups': 32, 'num_channels': 320, 'eps': 1e-05, 'affine': True}``.
    """
    out: Dict[str, Any] = {}
    try:
        sig = inspect.signature(type(mod).__init__)
    except (ValueError, TypeError):
        return out
    for pname, p in sig.parameters.items():
        if pname in ("self", "args", "kwargs"):
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        val = getattr(mod, pname, None)
        if val is None:
            continue
        if isinstance(val, (torch.Size, tuple)):
            val = list(val)
        if isinstance(val, bool) or isinstance(val, (int, float, str)):
            out[pname] = val
        elif isinstance(val, list) and all(
            isinstance(i, (int, float, bool, str)) for i in val
        ):
            out[pname] = val
    return out


def tied_parameter_groups(model: nn.Module) -> List[List[str]]:
    """Parameters shared by identity (tied embeddings). Deterministic: ``id()`` aliasing.

    ``remove_duplicate=False`` is essential — ``named_parameters()`` DEDUPLICATES shared
    tensors by default, so with the default a tied weight is reported under one name only
    and tying becomes undetectable.
    """
    by_id: Dict[int, List[str]] = defaultdict(list)
    for name, param in model.named_parameters(remove_duplicate=False):
        by_id[id(param)].append(name)
    return [sorted(set(v)) for v in by_id.values() if len(set(v)) > 1]


def fused_parameter_counts(mod: nn.Module) -> Optional[int]:
    """If a leaf owns a parameter of rank one higher than its op family expects, return
    the leading dimension as an instance count. This is the only way to see modern MoE
    experts, which are not modules at all."""
    expected = _EXPECTED_PARAM_RANK.get(type(mod).__name__)
    best: Optional[int] = None
    for _, p in mod.named_parameters(recurse=False):
        if p.dim() < 3:
            continue
        if expected is not None and p.dim() != expected + 1:
            continue
        if expected is None and p.dim() < 3:
            continue
        lead = int(p.shape[0])
        if lead > 1 and (best is None or lead == best):
            best = lead
    return best


class Introspection:
    """The materialised tree plus every repeat signal we can derive statically."""

    def __init__(self) -> None:
        self.nodes: Dict[str, ModuleNode] = {}
        self.parent: Dict[str, str] = {}
        self.repeats: Dict[str, RepeatGroup] = {}
        self.modules: Dict[str, nn.Module] = {}
        self.tied: List[List[str]] = []

    @property
    def rendered(self) -> List[str]:
        return [p for p in self.nodes if p != ""]

    def top_level(self) -> List[str]:
        return [p for p in self.rendered if not self.parent.get(p)]

    def descendants(self, path: str) -> int:
        prefix = path + "."
        return sum(1 for q in self.nodes if q.startswith(prefix))


def build_tree(model: nn.Module, max_depth: int) -> Introspection:
    """Walk the module tree to ``max_depth``, hoisting transparent containers and
    recording every repeat group we can see without executing anything."""
    out = Introspection()
    out.tied = tied_parameter_groups(model)
    named = dict(model.named_modules())

    def visit(mod: nn.Module, path: str, parent: Optional[str], depth: int) -> None:
        transparent = isinstance(mod, TRANSPARENT_CONTAINERS) and path != ""
        if not transparent:
            out.nodes[path] = ModuleNode(path, mod, depth, parent)
            out.modules[path] = mod
            if parent is not None:
                out.parent[path] = parent
            fused = fused_parameter_counts(mod)
            if fused:
                out.repeats[path] = RepeatGroup(
                    path=path, count=fused, reason="fused-parameter"
                )

        children = list(mod.named_children())
        if not children:
            return
        # A transparent container neither renders nor costs a depth level.
        child_depth = depth if transparent else depth + 1
        child_parent = parent if transparent else path
        if not transparent and depth >= max_depth:
            out.nodes[path].capture_depth_limit = max_depth
            return

        if isinstance(mod, TRANSPARENT_CONTAINERS) and len(children) > 1:
            groups: Dict[str, List[Tuple[str, nn.Module]]] = defaultdict(list)
            for cname, child in children:
                groups[structural_signature(child)].append((cname, child))
            for members in groups.values():
                cname, child = members[0]
                cpath = f"{path}.{cname}" if path else cname
                if len(members) > 1:
                    out.repeats[cpath] = RepeatGroup(
                        path=cpath, count=len(members), reason="module-list"
                    )
                visit(child, cpath, child_parent, child_depth)
            return

        for cname, child in children:
            visit(child, f"{path}.{cname}" if path else cname, child_parent, child_depth)

    visit(model, "", None, 0)
    # keep only modules that actually exist under their qualified name
    out.modules = {p: named[p] for p in out.modules if p in named}
    return out


def apply_call_counts(intro: Introspection, call_counts: Dict[str, int]) -> None:
    """Promote loop-called modules to repeat groups.

    A module invoked N>1 times in a Python loop is N layers even though the tree shows one.

    Restricted to COMPOSITE modules on purpose. A repeated *leaf* is reuse, not depth:
    ResNet's ``BasicBlock`` holds a single ``self.relu`` and applies it twice, so a naive
    rule stamps "ReLU x2" on every block — describing one activation as two layers. ALBERT's
    repeated unit is a whole layer group (composite), so it still fires."""
    for path, count in call_counts.items():
        if count < 2 or path not in intro.nodes:
            continue
        if path in intro.repeats:
            continue
        node = intro.nodes[path]
        if node.is_leaf or node.cls in LEAF_LIKE:
            continue
        # A parent frame and its children all fire N times; attribute the repeat to the
        # OUTERMOST such module so we do not stamp xN on every descendant.
        parent = intro.parent.get(path)
        if parent and call_counts.get(parent, 0) == count:
            continue
        intro.repeats[path] = RepeatGroup(path=path, count=count, reason="call-count")
