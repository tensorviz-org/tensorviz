"""A measurement outranks an inference, in the validator as well as in the merge.

The projection draws a node per ``nn.Module`` call and per statement it can name; the
wiring between them is reconstructed, not measured. TinyViT's ``PatchEmbed`` does::

    x = self.proj(x)
    x = x.flatten(2).transpose(1, 2)
    return self.norm(x)

and the reshape in the middle belongs to a nested module, so the merged graph joins the
Conv2d straight to the LayerNorm. The shape engine then read the LayerNorm's input as
``[1, 64, 8, 8]``, compared 8 against ``normalized_shape=64`` and reported *"LayerNorm
normalized_shape must match trailing input dimensions"* — on a layer torch had just run,
whose measured output the same node was carrying.

The engine's failure there is a statement about the graph, not about the model. Where a
node already carries a shape something wrote down after running it, that shape is the
answer.

Verified end to end on 2026-09-12: the real worker on TinyViT, through `project()`, into
`/api/validate_shapes` — five errors before, zero after.
"""

from __future__ import annotations

from typing import Any, Dict, List

from api.routers.validate import ValidationRequest, validate_shapes


def _node(node_id: str, op: str, params: Dict[str, Any], kind: str = "layerNode") -> Dict[str, Any]:
    return {
        "id": node_id,
        "type": kind,
        "position": {"x": 0, "y": 0},
        "data": {"layerType": {"id": op.lower(), "name": op}, "parameters": params},
    }


def _request(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> ValidationRequest:
    return ValidationRequest(nodes=nodes, edges=edges)


def _errors(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return validate_shapes(_request(nodes, edges)).get("errors") or []


# The reshape between them is inside a nested module, so the graph has no node for it:
# exactly the gap that produced the false positive.
CONV_THEN_NORM_EDGES = [
    {"id": "e0", "source": "input", "target": "proj"},
    {"id": "e1", "source": "proj", "target": "norm"},
    {"id": "e2", "source": "norm", "target": "output"},
]


def test_a_measured_layer_is_not_failed_on_a_shape_the_graph_could_not_supply() -> None:
    nodes = [
        _node("input", "Input", {"shape": "1,3,32,32"}, kind="inputNode"),
        _node("proj", "Conv2d", {"in_channels": 3, "out_channels": 64, "kernel_size": 4, "stride": 4, "_out_shape": "1x64x8x8"}),
        _node("norm", "LayerNorm", {"normalized_shape": 64, "_out_shape": "1x64x64"}),
        _node("output", "Output", {}, kind="outputNode"),
    ]
    messages = [e.get("message") for e in _errors(nodes, CONV_THEN_NORM_EDGES)]
    assert not any("normalized_shape" in (m or "") for m in messages), (
        "the LayerNorm ran and recorded 1x64x64; the engine's inability to derive that "
        f"from a graph missing the reshape is not the model's fault. got {messages}"
    )


def test_an_unmeasured_layer_with_a_real_mismatch_is_still_an_error() -> None:
    """The guard is "something measured this", not "skip the check"."""
    nodes = [
        _node("input", "Input", {"shape": "1,3,32,32"}, kind="inputNode"),
        _node("proj", "Conv2d", {"in_channels": 3, "out_channels": 64, "kernel_size": 4, "stride": 4}),
        _node("norm", "LayerNorm", {"normalized_shape": 64}),
        _node("output", "Output", {}, kind="outputNode"),
    ]
    messages = [e.get("message") for e in _errors(nodes, CONV_THEN_NORM_EDGES)]
    assert any("normalized_shape" in (m or "") for m in messages), (
        "with nothing measured, the engine's arithmetic is the only answer there is and "
        f"it must still be reported. got {messages}"
    )


def test_a_measured_shape_feeds_the_nodes_after_it() -> None:
    # Accepting the recorded shape must also publish it, or the next node starves and the
    # cascade comes back one step later.
    nodes = [
        _node("input", "Input", {"shape": "1,3,32,32"}, kind="inputNode"),
        _node("proj", "Conv2d", {"in_channels": 3, "out_channels": 64, "kernel_size": 4, "stride": 4, "_out_shape": "1x64x8x8"}),
        _node("norm", "LayerNorm", {"normalized_shape": 64, "_out_shape": "1x64x64"}),
        _node("head", "Linear", {"in_features": 64, "out_features": 10}),
        _node("output", "Output", {}, kind="outputNode"),
    ]
    edges = CONV_THEN_NORM_EDGES[:2] + [
        {"id": "e2", "source": "norm", "target": "head"},
        {"id": "e3", "source": "head", "target": "output"},
    ]
    answer = validate_shapes(_request(nodes, edges))
    assert not (answer.get("errors") or []), answer.get("errors")
    assert answer.get("shapes", {}).get("norm") == [1, 64, 64], (
        "the measured shape has to reach the report, not just silence the error"
    )
