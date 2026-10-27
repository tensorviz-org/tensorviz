"""POLICY for turning traced functional ops and reused call sites into real nodes.

``trace.py`` still owns the transparency rule. This file decides only two things, and is
consulted exactly where the fold would otherwise have thrown information away:

* which functional op earns a node (``wants_merge``)
* which reused leaf earns one node per call site (``splittable_leaves``)

Why this is needed at all — measured on raw ``depth=2`` torchvision imports before the
change (see scratchpad/baseline.py):

    resnet18      56 nodes  62 edges   7 arity violations  CYCLIC
    resnet50      80 nodes  95 edges   8 arity violations  CYCLIC
    mobilenet_v2 115 nodes 107 edges   5 arity violations
    densenet121   78 nodes 612 edges  58 arity violations
    vgg16         18 nodes  17 edges   0                      <- purely sequential

Zero merge nodes existed in any of them. Two distinct causes:

1. A functional merge in a module's own forward (``out += identity``,
   ``torch.cat(features, 1)``) was transparent, so its several producers were unioned
   into the CONSUMER — giving a DenseNet norm 17 inbound edges and MobileNet's next conv
   2. The consumer is a real ``nn.Module`` that accepts one input, so the runtime check
   failed with "arity error: Conv2d received 2 inputs but expects exactly 1".
2. torchvision's BasicBlock calls ``self.relu`` TWICE. One name meant one node, so the
   node collected edges from both call sites — and since the second call happens after
   ``conv2``, the graph closed a cycle: relu -> conv2 -> bn2 -> relu. An imported ResNet
   was therefore not even a DAG.

Reverting the feature is deleting this file and the ``plan=`` argument at its two call
sites; ``dataflow_edges`` keeps the original behaviour by passing no plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

# Aten base name -> (layerType id, display name). The id is what BOTH consumers key on:
# verification_service._MERGE_OP_KEYS (which routes to _apply_merge instead of the
# single-input arity check) and code_generation._MERGE_OPS (which emits
# torch.add(a, b) / torch.cat([...], dim=)).
#
# Deliberately NOT included, having checked both consumers:
#   sub, div  — the op registry calls them merges, but they are absent from
#               _MERGE_OP_KEYS, so a materialised node would fall through to
#               build_layer and die with "Unknown layer type 'torch.sub'".
#   stack     — registry arity is "one", so it would take an arity warning and an
#               incorrect analytic shape.
MERGE_OPS: Dict[str, Tuple[str, str]] = {
    "add": ("add", "Add"),
    "mul": ("mul", "Mul"),
    "cat": ("cat", "Cat"),
    "concat": ("cat", "Cat"),
}

# A functional reshape that is exactly a batch-preserving flatten. Predicated on the OP,
# never on shapes alone: deciding from shapes would fire on a [B,T,C] -> [B*T,C] view and
# manufacture a wrong node on transformer imports, where dim 0 changes.
FLATTEN_OPS = {"view", "reshape", "flatten", "_unsafe_view"}

# Functional global average pooling. MobileNetV2 calls nn.functional.adaptive_avg_pool2d
# instead of holding a module, and it lowers to aten.mean.dim — measured
# [1, 1280, 7, 7] -> [1, 1280, 1, 1]. With no node the classifier received the unpooled
# map flattened to 1x62720 instead of 1x1280. Materialised as a real AdaptiveAvgPool2d,
# whose layerType id the registry and codegen already understand (resnet18's own avgpool
# module produces exactly that id today).
POOL_OPS = {"mean", "adaptive_avg_pool2d", "_adaptive_avg_pool2d"}


def _frame_of(op: Optional[Dict[str, Any]]) -> Optional[Tuple[str, int]]:
    """``op['frame']`` as a ``(file, line)`` pair, or None.

    Defensive about the shape because the frame arrives from JSON in the worker
    path, where a tuple has become a list. A malformed entry is dropped rather
    than propagated: a node with no line is honest, a node with a wrong line
    sends the reader to someone else's code.
    """
    frame = (op or {}).get("frame")
    if isinstance(frame, (tuple, list)) and len(frame) == 2:
        file, line = frame
        if isinstance(file, str) and isinstance(line, int):
            return (file, line)
    return None


@dataclass
class Synthetic:
    """A node the trace proved must exist."""

    node_id: str
    kind: str  # "merge" | "flatten" | "split"
    layer_id: str  # layerType.id, e.g. "add"
    label: str  # layerType.name / subtitle, e.g. "Add"
    parent: Optional[str]
    origin: Optional[str] = None  # for "split": the module path this call site shares
    params: Dict[str, Any] = field(default_factory=dict)
    out_shape: Optional[List[int]] = None
    # (file, line) of the user frame the op ran in, when the tracer captured one
    # (task.md M4). A materialised node has no nn.Module and therefore no class to
    # ask, so this is the ONLY thing that can point a residual add at the
    # ``return x + self.l2(h)`` the reader wrote.
    frame: Optional[Tuple[str, int]] = None


@dataclass
class Plan:
    """Accumulates what to materialise while the fold runs.

    ``splittable`` is decided up front from the trace's reuse scopes; the merge and
    flatten decisions are made per op as the fold reaches them.
    """

    splittable: FrozenSet[str] = frozenset()
    flatten: bool = False
    taken: FrozenSet[str] = frozenset()
    # Rendered modules that HAVE children which are not themselves rendered.
    composites: FrozenSet[str] = frozenset()
    nodes: Dict[str, Synthetic] = field(default_factory=dict)
    _counters: Dict[str, int] = field(default_factory=dict)

    # ---- call sites -----------------------------------------------------------------

    def call_site(self, owner: str, index: int, op: Optional[Dict[str, Any]] = None) -> str:
        """Node id for the ``index``-th invocation of ``owner`` within its parent.

        Call 1 keeps the plain module path, so a model with no reuse is byte-identical
        and every existing node id is preserved.
        """
        if index <= 1 or owner not in self.splittable:
            return owner
        # Deterministic, and NOT via _unique: this is asked once per aten op inside the
        # call, so a counter would mint a new id per op and shatter the call site.
        node_id = f"{owner}_call{index}"
        while node_id in self.taken:
            node_id += "_"
        if node_id not in self.nodes:
            self.nodes[node_id] = Synthetic(
                node_id=node_id,
                kind="split",
                layer_id="",  # filled from the origin module at registration
                label="",
                parent=None,  # filled from the origin module at registration
                origin=owner,
                # The class is shared with call 1; the CALL is not. A second
                # ``self.relu(...)`` is a different line, and that line is the only
                # thing distinguishing the two cards in the editor.
                frame=_frame_of(op),
            )
        return node_id

    # ---- functional merges ----------------------------------------------------------

    def wants_merge(self, op: Dict[str, Any]) -> bool:
        return op.get("fn", "") in MERGE_OPS

    def is_composite(self, owner: str) -> bool:
        """True when ``owner`` is a multi-child module rendered without its children.

        At depth 1 a DenseNet's ``_DenseLayer`` is one card, so the ``torch.cat`` in its
        forward is attributed to it and it collected up to 17 inbound edges. A genuine
        leaf (a ReLU) never contains a merge, so this keeps the change off leaves.
        """
        return owner in self.composites

    def merge_node(self, op: Dict[str, Any], frame: Optional[str], sources: List[str]) -> str:
        layer_id, label = MERGE_OPS[op["fn"]]
        node_id = self._unique(self._stem(frame, layer_id))
        params: Dict[str, Any] = {}
        if layer_id == "cat":
            # torch.cat's dim is positional in aten, so it is not always in kwargs.
            # Default to 1 (channels) rather than -1: every architecture that
            # concatenates feature maps here (DenseNet, U-Net) does it on channels, and
            # a wrong dim silently changes the analytic shape.
            params["dim"] = op.get("dim", 1)
        self.nodes[node_id] = Synthetic(
            node_id=node_id,
            kind="merge",
            layer_id=layer_id,
            label=label,
            parent=frame,
            params=params,
            out_shape=op.get("out_shape"),
            frame=_frame_of(op),
        )
        return node_id

    # ---- functional pooling ---------------------------------------------------------

    def wants_pool(self, op: Dict[str, Any]) -> bool:
        """A rank-4 spatial reduction to 1x1, i.e. a global average pool.

        Deliberately narrow. ``aten.mean`` is also how a plain ``tensor.mean()`` lowers,
        so requiring rank 4 in AND out, an unchanged batch and channel count, and spatial
        dims that were larger than 1 and are now exactly 1, keeps this off every reduction
        that is not a global pool.
        """
        if not self.flatten or op.get("fn", "") not in POOL_OPS:
            return False
        src, dst = op.get("in_shape"), op.get("out_shape")
        if not src or not dst or len(src) != 4 or len(dst) != 4:
            return False
        return (
            dst[0] == src[0]
            and dst[1] == src[1]
            and dst[2] == 1
            and dst[3] == 1
            and (src[2] > 1 or src[3] > 1)
        )

    def pool_node(self, op: Dict[str, Any], frame: Optional[str], sources: List[str]) -> str:
        node_id = self._unique(self._stem(frame, "avgpool"))
        self.nodes[node_id] = Synthetic(
            node_id=node_id,
            kind="pool",
            layer_id="adaptiveavgpool2d",
            label="AdaptiveAvgPool2d",
            parent=frame,
            params={"output_size": 1},
            out_shape=op.get("out_shape"),
            frame=_frame_of(op),
        )
        return node_id

    # ---- flatten --------------------------------------------------------------------

    def wants_flatten(self, op: Dict[str, Any]) -> bool:
        """A batch-preserving flatten, e.g. torchvision's ``torch.flatten(x, 1)``.

        Without this an imported ResNet reaches ``fc`` still shaped [1, 512, 1, 1] and
        dies with "mat1 and mat2 shapes cannot be multiplied (512x1 and 512x1000)" — so
        the arity fix alone would not make it runnable.
        """
        if not self.flatten or op.get("fn", "") not in FLATTEN_OPS:
            return False
        src, dst = op.get("in_shape"), op.get("out_shape")
        if not src or not dst or len(src) < 2 or len(dst) != 2:
            return False
        product = 1
        for dim in src[1:]:
            product *= dim
        return dst[0] == src[0] and dst[1] == product

    def flatten_node(self, op: Dict[str, Any], frame: Optional[str], sources: List[str]) -> str:
        node_id = self._unique(self._stem(frame, "flatten"))
        self.nodes[node_id] = Synthetic(
            node_id=node_id,
            kind="flatten",
            layer_id="flatten",
            label="Flatten",
            parent=frame,
            params={"start_dim": 1},
            out_shape=op.get("out_shape"),
            frame=_frame_of(op),
        )
        return node_id

    # ---- helpers --------------------------------------------------------------------

    def _stem(self, frame: Optional[str], layer_id: str) -> str:
        return f"{frame}.{layer_id}" if frame else layer_id

    def _unique(self, stem: str) -> str:
        """Deterministic, collision-free ids: add1, add2, ... within a frame."""
        self._counters[stem] = self._counters.get(stem, 0) + 1
        candidate = f"{stem}{self._counters[stem]}"
        while candidate in self.taken or (
            candidate in self.nodes and self.nodes[candidate].kind != "split"
        ):
            self._counters[stem] += 1
            candidate = f"{stem}{self._counters[stem]}"
        return candidate


# Classes safe to duplicate per call site: stateless, so two call sites cannot disagree
# about weights. A parameterised module reused in a loop is weight SHARING, which a
# flat graph cannot express by copying.
STATELESS = {
    "ReLU", "ReLU6", "GELU", "SiLU", "ELU", "LeakyReLU", "Hardswish", "Hardsigmoid",
    "Hardtanh", "Mish", "Softplus", "Tanh", "Sigmoid", "Identity", "Dropout",
    "Flatten", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d",
}


def splittable_leaves(intro: Any, reuse_scopes: List[FrozenSet[str]]) -> Tuple[FrozenSet[str], List[str]]:
    """Decide, per reuse scope, whether its members may be split into call sites.

    Scope-level rather than per-node on purpose. In ``for _ in range(4): x = relu(fc(x))``
    the scope is {relu, fc}; splitting only the stateless relu would leave ``fc`` newly
    receiving 4 inbound edges — turning a clean graph into a broken one. So a scope
    qualifies only if EVERY reused member is a stateless rendered leaf; otherwise nothing
    in it splits and the refusal is reported rather than hidden.
    """
    ok: Set[str] = set()
    refused: List[str] = []
    for scope in reuse_scopes:
        # A member that is not rendered cannot be split and is not a refusal: a collapsed
        # repeat instance (resnet18's layer1.1) has no child nodes at all, so its ops fold
        # into the instance's own card, which already accepts several inputs.
        members = sorted(n for n in scope if n in intro.nodes)
        if not members:
            continue
        qualifies = True
        for name in members:
            node = intro.nodes[name]
            if not node.is_leaf or node.cls not in STATELESS:
                qualifies = False
                break
            if node.params:  # a stateless class should have none; belt and braces
                qualifies = False
                break
            if name in intro.repeats:  # already presented as a repeat group
                qualifies = False
                break
        if qualifies:
            ok.update(members)
        else:
            refused.extend(members)
    return frozenset(ok), sorted(set(refused) - ok)


def register(intro: Any, plan: Plan, io: Dict[str, Dict[str, Any]], visited: Set[str]) -> None:
    """Make every synthetic node a citizen of the module tree, before layout runs.

    ``build_graph`` iterates ``intro.rendered`` and reads ``intro.nodes``/``intro.parent``,
    so a node that is only mentioned in an edge would produce an edge to nowhere.
    """
    import torch.nn as nn

    from .introspect import ModuleNode

    placeholder = nn.Identity()
    for syn in list(plan.nodes.values()):
        if syn.kind == "split":
            origin = intro.nodes.get(syn.origin)
            if origin is None:
                continue
            node = ModuleNode(syn.node_id, placeholder, origin.depth, origin.parent)
            node.cls = origin.cls
            node.source_file, node.source_line = origin.source_file, origin.source_line
            if syn.frame is not None:
                node.decl_file, node.decl_line = syn.frame
            intro.nodes[syn.node_id] = node
            if origin.parent:
                intro.parent[syn.node_id] = origin.parent
            # Same nn.Module: a second call site of a stateless layer IS the same layer,
            # so constructor kwargs and observed shapes must match.
            module = intro.modules.get(syn.origin)
            if module is not None:
                intro.modules[syn.node_id] = module
            if syn.origin in io:
                io[syn.node_id] = io[syn.origin]
            if syn.origin in visited:
                visited.add(syn.node_id)
            continue

        parent = syn.parent if syn.parent in intro.nodes else None
        depth = (intro.nodes[parent].depth + 1) if parent else 1
        node = ModuleNode(syn.node_id, placeholder, depth, parent)
        node.cls = syn.label
        # nn.Identity is the placeholder that gave this node its slots, so without
        # this the card would claim to be defined in torch/nn/modules/linear.py.
        node.source_file, node.source_line = (syn.frame or (None, None))
        node.decl_file, node.decl_line = (syn.frame or (None, None))
        intro.nodes[syn.node_id] = node
        if parent:
            intro.parent[syn.node_id] = parent
        # No entry in intro.modules: there is no nn.Module behind a functional op, and
        # build_graph already treats a missing module as "no constructor kwargs".
        io[syn.node_id] = {"in": None, "out": syn.out_shape}
        visited.add(syn.node_id)
