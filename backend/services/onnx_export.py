"""ONNX export and validation service.

Builds a torch.nn.Module wrapper from the canvas graph, exports it to ONNX,
runs onnx shape inference + checker, and returns per-node shapes + the raw
ONNX bytes so the frontend can offer a download / open in Netron.
"""
from __future__ import annotations

import base64
import io
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from core.layer_builder import build_layer, topological_sort
from services.verification_service import (
    _build_synthetic_input,
    _coerce_input_shape,
    _expand_module_templates,
    _flatten_for_torch,
    _reconstruct_hierarchy,
    _resolve_layer_class,
)

logger = logging.getLogger(__name__)


class _GraphModule(nn.Module):
    """Thin nn.Module wrapper that replays the canvas graph in topological order.
    This makes the graph torch.onnx.export()-able as a single module.
    """

    def __init__(
        self,
        layers: Dict[str, nn.Module],
        order: List[str],
        edges: List[Any],
        node_types: Dict[str, str],
    ) -> None:
        super().__init__()
        self._order = order
        self._edge_pairs: List[Tuple[str, str]] = [
            (e.source if hasattr(e, "source") else e["source"],
             e.target if hasattr(e, "target") else e["target"])
            for e in edges
        ]
        self._node_types = node_types
        # Register as proper submodules (required for ONNX tracing to capture weights)
        for node_id, layer in layers.items():
            safe_attr = _safe_attr(node_id)
            setattr(self, safe_attr, layer)
        self._id_to_attr = {nid: _safe_attr(nid) for nid in layers}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tensors: Dict[str, torch.Tensor] = {}
        last = x
        for node_id in self._order:
            node_type = self._node_types.get(node_id, "layerNode")
            if node_type == "inputNode":
                tensors[node_id] = x
                last = x
                continue
            inputs = [
                tensors[src]
                for src, tgt in self._edge_pairs
                if tgt == node_id and src in tensors
            ]
            t_in = inputs[0] if len(inputs) == 1 else (
                torch.cat(inputs, dim=-1) if inputs else last
            )
            if node_type == "layerNode" and node_id in self._id_to_attr:
                layer = getattr(self, self._id_to_attr[node_id])
                t_out = layer(t_in)
                tensors[node_id] = t_out
                last = t_out
            else:
                tensors[node_id] = t_in
                last = t_in
        return last


def _safe_attr(node_id: str) -> str:
    """Convert a node id (may contain hyphens/dots) to a valid Python attribute name."""
    return "layer_" + "".join(c if c.isalnum() else "_" for c in node_id)


def export_onnx(
    graph_payload: Dict[str, Any],
    input_shape: Any = None,
    model_name: str = "GeneratedModel",
    opset: int = 17,
) -> Dict[str, Any]:
    """Export the graph to ONNX and run onnx validation.

    Returns::

        {
          "ok": bool,
          "errors": [...],
          "node_shapes": { onnx_node_name: [output_shapes] },
          "onnx_base64": "<base64-encoded .onnx bytes>",
          "onnx_size_bytes": int,
          "filename": str,
          "trace": str | None,
        }
    """
    try:
        import onnx
        import onnx.checker
        import onnx.shape_inference
    except ImportError:
        return {
            "ok": False,
            "errors": [{"message": "onnx is not installed. Run: pip install onnx"}],
            "node_shapes": {},
            "onnx_base64": None,
            "onnx_size_bytes": 0,
            "filename": f"{model_name}.onnx",
            "trace": None,
        }

    result: Dict[str, Any] = {
        "ok": False,
        "errors": [],
        "node_shapes": {},
        "onnx_base64": None,
        "onnx_size_bytes": 0,
        "filename": f"{model_name}.onnx",
        "trace": None,
    }

    shape = _coerce_input_shape(input_shape, [1, 3, 224, 224])
    graph_payload = _expand_module_templates(graph_payload)
    graph_payload = _reconstruct_hierarchy(graph_payload)

    # ── Build layers ─────────────────────────────────────────────────────
    try:
        flat = _flatten_for_torch(graph_payload)
        leaf_layers: Dict[str, nn.Module] = {}
        for n in flat.nodes:
            if n.type != "layerNode":
                continue
            cls_name = _resolve_layer_class(n.data.layerType)
            leaf_layers[n.id] = build_layer(cls_name, n.data.parameters or {})

        order = topological_sort(flat.nodes, flat.edges)
        node_types = {n.id: n.type for n in flat.nodes}
        model = _GraphModule(leaf_layers, order, flat.edges, node_types)
        model.eval()
    except Exception as exc:
        result["errors"].append({"message": f"Model build failed: {exc}"})
        result["trace"] = traceback.format_exc()
        return result

    # ── Export to ONNX ───────────────────────────────────────────────────
    try:
        # Always use float for ONNX tracing; int tensors break matmul in export.
        safe_shape = [max(1, int(d)) for d in shape]
        dummy = torch.randn(*safe_shape)
        buf = io.BytesIO()
        torch.onnx.export(
            model,
            (dummy,),
            buf,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            export_params=True,
            do_constant_folding=True,
        )
        onnx_bytes = buf.getvalue()
    except Exception as exc:
        result["errors"].append({"message": f"ONNX export failed: {exc}"})
        result["trace"] = traceback.format_exc()
        return result

    # ── ONNX shape inference ─────────────────────────────────────────────
    try:
        onnx_model = onnx.load_from_string(onnx_bytes)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        # Collect shapes: intermediate tensors + declared inputs + outputs
        shapes: Dict[str, Any] = {}
        for vi in list(onnx_model.graph.value_info) + list(onnx_model.graph.input) + list(onnx_model.graph.output):
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
                shape_dims = [
                    d.dim_value if d.HasField("dim_value") else str(d.dim_param)
                    for d in vi.type.tensor_type.shape.dim
                ]
                shapes[vi.name] = shape_dims
        result["node_shapes"] = shapes
    except Exception as exc:
        result["errors"].append({"message": f"ONNX shape inference failed: {exc}"})
        logger.warning("ONNX shape inference failed: %s", exc)

    # ── ONNX checker ─────────────────────────────────────────────────────
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as exc:
        result["errors"].append({"message": f"ONNX checker: {exc}"})
    except Exception as exc:
        result["errors"].append({"message": f"ONNX checker error: {exc}"})

    # ── Package result ───────────────────────────────────────────────────
    result["ok"] = len(result["errors"]) == 0
    result["onnx_base64"] = base64.b64encode(onnx_bytes).decode("ascii")
    result["onnx_size_bytes"] = len(onnx_bytes)
    return result
