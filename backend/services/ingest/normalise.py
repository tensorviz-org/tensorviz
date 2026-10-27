"""Turn an introspected + traced model into a flat ``ModelGraph`` the store accepts.

Four canvas-contract details that were each found the hard way. Getting any of them wrong
produces a graph that looks plausible in JSON and is broken on screen:

1. ``data.parameters`` is the CONSTRUCTOR-KWARG namespace — the backend layer builder
   splats it into ``nn.<Class>(**params)``. Importer metadata placed there bare becomes
   ``SiLU.__init__() got an unexpected keyword argument 'module_path'``. It must be
   ``_``-prefixed, which ``core/layer_builder.py`` strips (``if k != 'name' and not
   k.startswith('_')``) and ``services/code_generation.py`` also skips. A sibling
   ``data.meta`` key is NOT an alternative: the client filters node ``data`` through a
   15-key allowlist (``graphDocumentAdapter.ts``) that has no ``meta``, so it renders once
   and is dropped on the first save.
2. A module card takes its title from ``data.moduleTemplateId`` and its sub-layer count
   from ``data.moduleLayerCount`` (``canvas/ModuleShapeUtil.tsx``), never from
   ``layerType.name`` — which the client always recomputes from the semantic ``op``. Omit
   them and every block renders as "module / 0 layers".
3. A module child's position is stored RELATIVE to its module's origin
   (``canvas/useTldrawBridge.ts``). Emitting ``(0,0)`` stacks every child on the parent's
   top-left corner.
4. The importer MUST emit real coordinates. ``reconcileModuleView`` only runs dagre when
   ``relayout: true``, which defaults to false and is false on the saved-graph load path,
   so ``graphNeedsLayout`` never gets a chance to rescue a graph saved at the origin — the
   cards stay stacked permanently.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from .introspect import Introspection, declared_out_shape
from .spec import IngestDiagnostics, RepeatGroup

# Layout pitch must exceed the rendered card footprint (200x120 logical px for the tldraw
# layer/input/output shapes) or adjacent cards overlap by construction.
STRIDE_X = 380
STRIDE_Y = 300
ORIGIN_X = 80
ORIGIN_Y = 80
CHILD_INSET_X = 24.0
CHILD_INSET_Y = 64.0
CHILD_PITCH_Y = 150.0


def _shape_text(shape: Optional[List[int]]) -> Optional[Union[str, List[int]]]:
    # [] is a measured scalar, not a missing shape.
    return "x".join(str(d) for d in shape) if shape else shape


def import_path_for(file: Optional[str], roots: Sequence[str]) -> Optional[str]:
    """The dotted module path ``file`` is imported by, or None if it is not the
    user's code at all (task.md M6).

    "Under one of the workspace roots" is the whole definition of "the user's own
    class". An ``nn.Linear`` resolves to a file inside site-packages and is
    therefore not one, which is what keeps ``custom_code`` from swallowing every
    torch layer. The rule is ``repo_scan.module_path_for``'s, called with the
    roots the caller already used to import the model, so the path this records
    is the path that would import the class back.

    Imported inside the function: ``repo_scan`` pulls the repository importer in,
    and the tree-only tier has no reason to load it.
    """
    if not file or not roots:
        return None
    from .repo_scan import module_path_for

    for root in roots:
        try:
            return module_path_for(root, file)
        except (ValueError, OSError):
            continue
    return None


def layered_positions(
    tops: List[str], pairs: Set[Tuple[str, str]], parent: Dict[str, str]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    """Longest-path layering over the TOP-LEVEL DAG.

    Dataflow pairs mostly connect deep children (``down_blocks.0.resnets.0 ->
    up_blocks.3.resnets.0``), so they must first be folded up to their top-level ancestor.
    Skip that fold and the top-level graph is edgeless, every node lands at rank 0, and the
    whole model renders as one vertical column.
    """
    topset = set(tops)

    def top_of(path: str) -> Optional[str]:
        current: Optional[str] = path
        for _ in range(64):
            if current in topset:
                return current
            if current is None:
                return None
            current = parent.get(current)
        return None

    successors: Dict[str, Set[str]] = defaultdict(set)
    indegree: Dict[str, int] = {t: 0 for t in tops}
    for src, dst in pairs:
        s, d = top_of(src), top_of(dst)
        if s and d and s != d and d not in successors[s]:
            successors[s].add(d)
            indegree[d] += 1

    rank: Dict[str, int] = {t: 0 for t in tops}
    queue = [t for t in tops if indegree[t] == 0]
    seen = 0
    while queue:
        node = queue.pop(0)
        seen += 1
        for nxt in successors[node]:
            rank[nxt] = max(rank[nxt], rank[node] + 1)
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if seen < len(tops):  # a cycle should be impossible on a forward trace; stay safe
        for index, path in enumerate(sorted(tops)):
            rank[path] = index

    slots: Dict[int, int] = defaultdict(int)
    positions: Dict[str, Dict[str, float]] = {}
    for path in sorted(tops, key=lambda p: (rank[p], p)):
        r = rank[path]
        positions[path] = {
            "x": float(ORIGIN_X + r * STRIDE_X),
            "y": float(ORIGIN_Y + slots[r] * STRIDE_Y),
        }
        slots[r] += 1
    return positions, rank


def build_graph(
    name: str,
    intro: Introspection,
    io: Dict[str, Dict[str, Any]],
    call_counts: Dict[str, int],
    pairs: Set[Tuple[str, str]],
    visited: Set[str],
    entry_shape: Optional[List[int]],
    synthetic_params: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    roots: Sequence[str] = (),
    input_provenance: Optional[str] = None,
) -> Tuple[Dict[str, Any], IngestDiagnostics]:
    """Assemble the flat ModelGraph plus diagnostics.

    ``synthetic_params`` carries the constructor kwargs for nodes that have no
    ``nn.Module`` behind them (a materialised ``cat`` needs its ``dim``, a ``Flatten``
    its ``start_dim``); everything else derives its params from the live object.

    ``roots`` are the workspace directories the model was imported from (the IDE
    worker's, and only its — the web importers pass none). A class defined under
    one of them is the USER's, so its node is the ``custom_code`` placeholder
    carrying the module path that would import it back, rather than an op named
    after a class no registry has ever heard of (task.md M6).

    ``input_provenance`` qualifies ``entry_shape``: whether anybody chose the tensor
    it describes. It is written next to the shape rather than beside it in the
    diagnostics because the card that draws the shape reads the node, not the
    diagnostics, and the two must not be able to drift apart.
    """
    from .introspect import constructor_kwargs

    rendered = set(intro.rendered)
    tops = intro.top_level()
    positions, rank = layered_positions(tops, pairs, intro.parent)
    max_rank = max(rank.values(), default=0)

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    opaque: List[str] = []
    unreached: List[str] = []
    shaped = 0

    nodes.append(
        {
            "id": "input_node",
            "type": "inputNode",
            "position": {"x": float(ORIGIN_X - STRIDE_X), "y": float(ORIGIN_Y)},
            "data": {
                "layerType": {"id": "input", "name": "Input", "parameters": []},
                "parameters": {
                    "name": "Input",
                    "shape": ",".join(str(d) for d in (entry_shape or [1])),
                    "_subtitle": "Input",
                    **({"_input_provenance": input_provenance} if input_provenance else {}),
                },
            },
        }
    )

    siblings: Dict[Optional[str], List[str]] = defaultdict(list)
    for path in sorted(rendered):
        siblings[intro.parent.get(path)].append(path)

    for path in sorted(rendered):
        node = intro.nodes[path]
        observed = io.get(path) or {}
        repeat = intro.repeats.get(path)
        is_module = node.renders_as_module()
        module = intro.modules.get(path)

        params: Dict[str, Any] = constructor_kwargs(module) if module is not None else {}
        if synthetic_params and path in synthetic_params:
            params.update(synthetic_params[path])
        params["_module_path"] = path
        params["_class"] = node.cls
        params["_params_M"] = round(node.params / 1e6, 4)
        # Where the reader can go and look at this node (task.md §6.1). For a real
        # module that is the file its CLASS was written in — torch's own for an
        # nn.Linear, the user's for their own block; for a materialised functional
        # op it is the forward() line the tracer caught it on. Both answer the same
        # question ("show me this card in the editor"), so they share the key.
        if node.source_file:
            params["_source_file"] = node.source_file
        if isinstance(node.source_line, int):
            params["_source_line"] = node.source_line
        # The statement in the user's own file that produced this node, when the
        # tracer caught a frame for it. This is what "reveal in editor" jumps to.
        if node.decl_file:
            params["_decl_file"] = node.decl_file
        if isinstance(node.decl_line, int):
            params["_decl_line"] = node.decl_line
        in_text = _shape_text(observed.get("in"))
        out_text = _shape_text(observed.get("out"))
        if in_text is not None:
            params["_in_shape"] = in_text
        if out_text is not None:
            params["_out_shape"] = out_text
            shaped += 1
        if repeat:
            params["_repeat_count"] = repeat.count
            params["_repeat_reason"] = repeat.reason
        calls = call_counts.get(path, 0)
        if calls > 1:
            params["_call_sites"] = calls
        if path in visited:
            params["_shape_provenance"] = "observed" if out_text is not None else "unreached"
        else:
            params["_shape_provenance"] = "unreached"
            unreached.append(path)
        if node.is_leaf and node.cls not in _KNOWN_LEAVES:
            params["_opaque"] = True
            opaque.append(node.cls)
        # A block the trace never entered has no observed shape, and everything
        # downstream of it then has none either. A declaration in its own source
        # is the only other thing that can answer, and it is marked as such:
        # `declared` is never presented as a measurement.
        if out_text is None and module is not None and node.source_file:
            declared = declared_out_shape(node.source_file, node.cls)
            if declared is not None:
                raw, dims = declared
                params["_declared_out_shape"] = raw
                params["_shape_provenance"] = "declared"
                if dims:
                    params["_out_shape"] = _shape_text(dims)

        # The user's own class. `op` must stay the bare placeholder id:
        # graph_input_safety._LAYER_OPERATION admits only `nn.`/`torch.nn.`/`torch.`
        # prefixes, so a dotted user path could never cross into codegen at all,
        # and a bare class name would claim an operation the registry does not
        # have. The identity of the class rides in `_class`/`_import_path`.
        # `module is not None` is what distinguishes a real class from a node the
        # tracer materialised. A residual `add` is given the FRAME it ran in as its
        # source (services/ingest/materialise.register), which is the user's own
        # file — so without this every merge in a workspace model would call itself
        # one of the user's classes and lose its `torch.add` op.
        import_path = (
            import_path_for(node.source_file, roots) if module is not None else None
        )
        if import_path:
            params["_import_path"] = import_path
            params["_opaque"] = True

        title = node.cls + (f" x{repeat.count}" if repeat else "")
        params["_subtitle"] = title

        parent_path = intro.parent.get(path) or None
        if parent_path:
            index = siblings[parent_path].index(path)
            position = {"x": CHILD_INSET_X, "y": CHILD_INSET_Y + index * CHILD_PITCH_Y}
        else:
            position = positions.get(path, {"x": 0.0, "y": 0.0})

        nodes.append(
            {
                "id": path,
                "type": "moduleNode" if is_module else "layerNode",
                "position": position,
                "data": {
                    "layerType": (
                        {"id": "custom_code", "name": "custom_code", "parameters": []}
                        if import_path
                        else {"id": node.cls.lower(), "name": title, "parameters": []}
                    ),
                    "parameters": params,
                    "parameterCount": node.params,
                    **({"outputTensor": observed["output_tensor"]}
                       if path in visited and observed.get("output_tensor") is not None else {}),
                    "parentId": parent_path,
                    "depth": node.depth,
                    "expanded": False,
                    "isGroup": is_module,
                    "moduleType": "custom" if is_module else "builtin",
                    "displayName": path.split(".")[-1] or name,
                    # moduleInstanceId is what makes a module card INTERACTIVE. Both
                    # canvas/ModuleShapeUtil.tsx and components/ModuleNode.tsx guard their
                    # expand/reset handlers with `if (!moduleInstanceId ...) return`, and
                    # neither falls back to the shape id — so without this field
                    # "Expand Block" and double-click silently do nothing, and the card's
                    # subtitle renders as the literal word "module".
                    # ModelBuilder's handleFlatToggleExpand matches
                    # `n.id === key || n.data.moduleInstanceId === key`, so the node's own
                    # qualified path is a valid key.
                    "moduleInstanceId": path,
                    "moduleTemplateId": title,
                    "moduleLayerCount": intro.descendants(path),
                    **({"moduleCapture": {
                        "status": "depth-limited", "maxDepth": node.capture_depth_limit,
                    }} if is_module and node.capture_depth_limit is not None else {}),
                    **({"repeatCount": repeat.count} if repeat else {}),
                },
            }
        )

    # The model's own return value. The tracer records the root module under the empty
    # name, so `io[""]["out"]` is the shape the forward pass actually produced — the one
    # number a reader most wants off this card, and until now the only card that never
    # showed one. The Input terminal has always drawn its shape; Output was built with no
    # shape field at all, so it could not have displayed one however the run went.
    exit_dims = (io.get("") or {}).get("out")
    output_parameters: Dict[str, Any] = {"name": "Output", "_subtitle": "Output"}
    if exit_dims is not None:
        # Two spellings on purpose, each matching what already reads it. `shape` is the
        # card's own field and is comma-joined exactly like the Input terminal's, so the
        # two ends of the graph do not disagree about how a shape is written. `_out_shape`
        # is the layer convention (`2x8`) that the tensor-evidence and overlay paths read,
        # so a measured output counts as measured everywhere rather than only looking right.
        output_parameters["shape"] = ",".join(str(d) for d in exit_dims)
        output_parameters["_out_shape"] = _shape_text(exit_dims)
        output_parameters["_shape_provenance"] = "observed"

    nodes.append(
        {
            "id": "output_node",
            "type": "outputNode",
            "position": {"x": float(ORIGIN_X + (max_rank + 1) * STRIDE_X), "y": float(ORIGIN_Y)},
            "data": {
                "layerType": {"id": "output", "name": "Output", "parameters": []},
                "parameters": output_parameters,
                **({"outputTensor": io[""]["output_tensor"]}
                   if (io.get("") or {}).get("output_tensor") is not None else {}),
            },
        }
    )

    drawn = set()
    for index, (src, dst) in enumerate(sorted(pairs)):
        edges.append({"id": f"e{index}", "source": src, "target": dst})
        drawn.add((src, dst))
    sources = [t for t in tops if rank.get(t, 0) == 0]
    sinks = [t for t in tops if rank.get(t, 0) == max_rank]
    # A seeded fold (services/ingest/trace.resolve_dataflow) already emits the
    # input_node edges it proved, so the rank-based ones below would draw a second
    # arrow between the same two cards.
    for index, path in enumerate(sources):
        if ("input_node", path) not in drawn:
            edges.append({"id": f"e_in{index}", "source": "input_node", "target": path})
    for index, path in enumerate(sinks):
        if (path, "output_node") not in drawn:
            edges.append({"id": f"e_out{index}", "source": path, "target": "output_node"})

    diagnostics = IngestDiagnostics(
        tier="fake",
        raw_modules=len(intro.nodes),
        emitted_nodes=len(nodes),
        dataflow_edges=len(pairs),
        nodes_with_observed_shapes=shaped,
        repeat_groups=sorted(intro.repeats.values(), key=lambda r: r.path),
        tied_parameters=intro.tied,
        unreached_modules=sorted(unreached)[:64],
        opaque_ops=sorted(set(opaque)),
    )
    return {"nodes": nodes, "edges": edges}, diagnostics


# Leaf classes the authored-graph op registry already understands. Anything else becomes an
# opaque card that carries its OBSERVED shapes, so the graph stays shape-correct without an
# analytic rule for a library-private composite.
_KNOWN_LEAVES = {
    # linear / conv
    "Linear", "LazyLinear", "Bilinear", "Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "Embedding", "EmbeddingBag",
    # normalisation
    "LayerNorm", "GroupNorm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "RMSNorm",
    "InstanceNorm1d", "InstanceNorm2d", "LocalResponseNorm",
    # activations — the full torch.nn set, so a standard op is never reported opaque
    "GELU", "ReLU", "ReLU6", "LeakyReLU", "SiLU", "ELU", "CELU", "SELU", "GLU", "Mish",
    "PReLU", "RReLU", "Hardswish", "Hardsigmoid", "Hardtanh", "Hardshrink", "Softplus",
    "Softsign", "Softshrink", "Tanh", "Tanhshrink", "Sigmoid", "LogSigmoid", "Softmax",
    "Softmax2d", "LogSoftmax", "Softmin", "Threshold",
    # shape / pooling / regularisation
    "Flatten", "Unflatten", "Identity", "Dropout", "Dropout1d", "Dropout2d", "Dropout3d",
    "AlphaDropout", "MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d",
    "AvgPool3d", "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
    "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "Upsample", "UpsamplingBilinear2d",
    "PixelShuffle", "PixelUnshuffle", "ZeroPad2d", "ReflectionPad2d", "ReplicationPad2d",
    "ConstantPad2d",
    # recurrent / attention
    "LSTM", "GRU", "RNN", "LSTMCell", "GRUCell", "RNNCell", "MultiheadAttention",
    # functional merges materialised by services/ingest/materialise.py. The authored-graph
    # registry understands all three (torch.add / torch.mul / torch.cat), so they are
    # known leaves, not opaque cards.
    "Add", "Mul", "Cat",
}


# ── traced callables (task.md M9) ─────────────────────────────────────────────
# A data pipeline has no nn.Module hierarchy, so `build_graph` above — which reads
# an `Introspection` — cannot describe one. What a `Dataset.__getitem__` has instead
# is LINES: every tensor op it performs ran on a line of the user's own file, and
# `trace.trace_callable` records exactly that. So the graph is one node per source
# line that touched a tensor, grouped under the function they ran in.

#: aten ops that turn N samples into one batch. `torch.utils.data` collates with
#: `torch.stack` (and `cat` for a batch-sampler), and that op is the pipeline's
#: single most important node: it is where the per-sample shape becomes the batch
#: shape the model is asked for.
COLLATE_OPS = ("stack", "cat", "concat")


def _op_shapes(op: Dict[str, Any]) -> Dict[str, str]:
    """`_in_shape`/`_out_shape` for one recorded op, when the tracer kept them."""
    params: Dict[str, str] = {}
    for key in ("in_shape", "out_shape"):
        text = _shape_text(op.get(key))
        if text:
            params[f"_{key}"] = text
    return params


def build_callable_graph(
    name: str,
    ops: List[Dict[str, Any]],
    *,
    roots: Sequence[str] = (),
) -> Tuple[Dict[str, Any], IngestDiagnostics]:
    """A flat ModelGraph for one traced function call.

    Three rules, and each is the answer to a question a module graph never asks:

    * **a node is a source line**, not a module. ``x * 0.5`` on line 12 is a node
      whose ``_source_line`` is 12; a second sample re-running line 12 is the same
      node, because it is the same code. Only ops that carried a frame under
      ``roots`` become nodes — the ops inside torch's own collate did not come from
      anything the reader can click on.
    * **the enclosing function is a display frame.** ``__getitem__``'s ops are drawn
      inside a card named for it, which is grouping and not computation: it has no
      edges of its own. A display frame groups the ops; it is not a step that ran.
    * **collation is a merge.** Sample 2 runs the same lines as sample 1, so the
      "two or more distinct producers" test ``resolve_dataflow`` uses for a residual
      cannot fire here — one line is one producer however many samples pass through
      it. The test that does hold is the one that matters: a ``stack``/``cat`` that
      ran OUTSIDE the user's code on tensors the user's lines produced is the
      collate, and it is where per-sample becomes per-batch.
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    # (file, line) -> node id, in first-seen order: execution order is the only
    # ordering a straight-line function has.
    line_nodes: Dict[Tuple[str, int], str] = {}
    frame_nodes: Dict[str, str] = {}
    frame_children: Dict[str, List[str]] = defaultdict(list)
    node_params: Dict[str, Dict[str, Any]] = {}
    node_titles: Dict[str, str] = {}
    tops: List[str] = []
    merges: List[str] = []
    producer: Dict[int, Set[str]] = {}
    pairs: List[Tuple[str, str]] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    def emit(src: str, dst: str) -> None:
        if src != dst and (src, dst) not in seen_pairs:
            seen_pairs.add((src, dst))
            pairs.append((src, dst))

    for op in ops:
        frame = op.get("frame")
        fn = str(op.get("fn") or "op")
        sources: Set[str] = set()
        for tid in op.get("in_ids") or []:
            sources |= producer.get(tid, set())

        if isinstance(frame, (tuple, list)) and len(frame) == 2:
            file, line = str(frame[0]), int(frame[1])
            key = (file, line)
            node_id = line_nodes.get(key)
            if node_id is None:
                node_id = f"line_{len(line_nodes)}_{fn}"
                line_nodes[key] = node_id
                owner = str(op.get("frame_fn") or "call")
                frame_id = frame_nodes.get(owner)
                if frame_id is None:
                    frame_id = f"frame_{owner}"
                    frame_nodes[owner] = frame_id
                    tops.append(frame_id)
                frame_children[frame_id].append(node_id)
                params: Dict[str, Any] = {
                    "_source_file": file,
                    "_source_line": line,
                    # The statement that produced it IS the statement it was written
                    # on here, so both provenance keys answer with the same line. Reveal-in-editor
                    # reads `_decl_*` first (task.md §6.1) and must not come up empty.
                    "_decl_file": file,
                    "_decl_line": line,
                    "_kind": "statement",
                    "_shape_provenance": "observed",
                    # No registry op is being claimed: this is a recorded aten call,
                    # not an authored layer, and an opaque card is how the canvas
                    # already draws "real, observed, and not one of ours".
                    "_opaque": True,
                    "_subtitle": f"{fn} · line {line}",
                    **_op_shapes(op),
                }
                import_path = import_path_for(file, roots)
                if import_path:
                    params["_import_path"] = import_path
                node_params[node_id] = params
                node_titles[node_id] = fn
            else:
                # A later op on the same line refines what the line's shapes are only
                # if it brought some; it never renames the node.
                node_params[node_id].update(_op_shapes(op))
            for src in sources:
                emit(src, node_id)
            for tid in op.get("out_ids") or []:
                producer[tid] = {node_id}
            continue

        if fn in COLLATE_OPS and sources:
            node_id = f"collate_{len(merges)}"
            merges.append(node_id)
            tops.append(node_id)
            node_params[node_id] = {
                "_subtitle": fn,
                "_kind": "statement",
                "_shape_provenance": "observed",
                "_opaque": True,
                **_op_shapes(op),
            }
            node_titles[node_id] = fn
            for src in sources:
                emit(src, node_id)
            for tid in op.get("out_ids") or []:
                producer[tid] = {node_id}
            continue

        # Everything else torch did on the way is transparent: it propagates its
        # inputs' producers rather than claiming to be a step of the pipeline.
        for tid in op.get("out_ids") or []:
            producer[tid] = producer.get(tid, set()) | sources

    nodes.append(
        {
            "id": "input_node",
            "type": "inputNode",
            "position": {"x": float(ORIGIN_X - STRIDE_X), "y": float(ORIGIN_Y)},
            "data": {
                "layerType": {"id": "input", "name": "Input", "parameters": []},
                "parameters": {"name": "Input", "_subtitle": "Input"},
            },
        }
    )

    for rank, top in enumerate(tops):
        position = {"x": float(ORIGIN_X + rank * STRIDE_X), "y": float(ORIGIN_Y)}
        children = frame_children.get(top, [])
        if children:
            nodes.append(
                {
                    "id": top,
                    "type": "moduleNode",
                    "position": position,
                    "data": {
                        "layerType": {"id": "module", "name": "module", "parameters": []},
                        "parameters": {"_subtitle": top.replace("frame_", ""), "_kind": "function"},
                        "parentId": None,
                        "depth": 0,
                        "expanded": False,
                        "isGroup": True,
                        "moduleType": "custom",
                        "displayName": top.replace("frame_", ""),
                        "moduleInstanceId": top,
                        "moduleTemplateId": top.replace("frame_", ""),
                        "moduleLayerCount": len(children),
                    },
                }
            )
            for index, child in enumerate(children):
                nodes.append(
                    {
                        "id": child,
                        "type": "layerNode",
                        "position": {
                            "x": CHILD_INSET_X,
                            "y": CHILD_INSET_Y + index * CHILD_PITCH_Y,
                        },
                        "data": {
                            "layerType": {
                                "id": node_titles[child],
                                "name": node_titles[child],
                                "parameters": [],
                            },
                            "parameters": node_params[child],
                            "parentId": top,
                            "depth": 1,
                            "expanded": False,
                            "isGroup": False,
                            "moduleType": "builtin",
                            "displayName": node_titles[child],
                        },
                    }
                )
        else:
            nodes.append(
                {
                    "id": top,
                    "type": "layerNode",
                    "position": position,
                    "data": {
                        "layerType": {
                            "id": node_titles[top],
                            "name": node_titles[top],
                            "parameters": [],
                        },
                        "parameters": node_params[top],
                        "parentId": None,
                        "depth": 0,
                        "expanded": False,
                        "isGroup": False,
                        "moduleType": "builtin",
                        "displayName": node_titles[top],
                    },
                }
            )

    nodes.append(
        {
            "id": "output_node",
            "type": "outputNode",
            "position": {"x": float(ORIGIN_X + (len(tops) + 1) * STRIDE_X), "y": float(ORIGIN_Y)},
            "data": {
                "layerType": {"id": "output", "name": "Output", "parameters": []},
                "parameters": {"name": "Output", "_subtitle": "Output"},
            },
        }
    )

    # An op node's edges belong to its display frame: the frame is grouping, the
    # line inside it is the computation, and an edge drawn to the group would say
    # the group computed something.
    inbound: Set[str] = set()
    outbound: Set[str] = set()
    for index, (src, dst) in enumerate(pairs):
        edges.append({"id": f"e{index}", "source": src, "target": dst})
        inbound.add(dst)
        outbound.add(src)
    entries = [n for n in line_nodes.values() if n not in inbound] or []
    exits = [n for n in ([*line_nodes.values(), *merges]) if n not in outbound]
    for index, node_id in enumerate(entries):
        edges.append({"id": f"e_in{index}", "source": "input_node", "target": node_id})
    for index, node_id in enumerate(exits):
        edges.append({"id": f"e_out{index}", "source": node_id, "target": "output_node"})

    diagnostics = IngestDiagnostics(
        tier="real",
        raw_modules=0,
        emitted_nodes=len(nodes),
        dataflow_edges=len(pairs),
        aten_ops=len(ops),
        frames_captured=sum(1 for op in ops if op.get("frame")),
        nodes_with_observed_shapes=sum(
            1 for params in node_params.values() if params.get("_out_shape")
        ),
        materialised_ops=sorted(merges),
        provenance=name,
    )
    return {"nodes": nodes, "edges": edges}, diagnostics
