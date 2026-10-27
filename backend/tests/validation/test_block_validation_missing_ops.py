"""Unit + integration validation for newly-covered builder blocks.

This focuses on blocks that previously had no OpSpec entry and verifies both:
1) control-plane strict validation (CanonicalValidationService)
2) data-plane validation path (verify_model stage 1, run_forward disabled)
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from models.architecture_models import (
    ArchitectureEdge,
    ArchitectureGraph,
    ArchitectureNode,
)
from services.canonical_validation import CanonicalValidationService
from services.verification_service import verify_model


@dataclass(frozen=True)
class BlockCase:
    node_id: str
    op: str
    input_shape: list[int]
    params: dict


UNIT_CASES = [
    BlockCase(
        node_id="fractionalmaxpool2d",
        op="nn.FractionalMaxPool2d",
        input_shape=[1, 3, 16, 16],
        params={"output_size": [8, 8], "kernel_size": 2},
    ),
    BlockCase(
        node_id="channelshuffle",
        op="nn.ChannelShuffle",
        input_shape=[1, 4, 8, 8],
        params={"groups": 2},
    ),
    BlockCase(
        node_id="unfold",
        op="nn.Unfold",
        input_shape=[1, 3, 8, 8],
        params={"kernel_size": 2, "stride": 2},
    ),
    BlockCase(
        node_id="fold",
        op="nn.Fold",
        input_shape=[1, 12, 16],
        params={"output_size": [8, 8], "kernel_size": 2, "stride": 2},
    ),
    BlockCase("crossentropyloss", "nn.CrossEntropyLoss", [4, 10], {}),
    BlockCase("mseloss", "nn.MSELoss", [4, 10], {}),
    BlockCase("l1loss", "nn.L1Loss", [4, 10], {}),
    BlockCase("bceloss", "nn.BCELoss", [4, 10], {}),
    BlockCase("bcewithlogitsloss", "nn.BCEWithLogitsLoss", [4, 10], {}),
    BlockCase("nllloss", "nn.NLLLoss", [4, 10], {}),
    BlockCase("kldivloss", "nn.KLDivLoss", [4, 10], {}),
    BlockCase("smoothl1loss", "nn.SmoothL1Loss", [4, 10], {}),
    BlockCase("huberloss", "nn.HuberLoss", [4, 10], {}),
    BlockCase("tripletmarginloss", "nn.TripletMarginLoss", [4, 10], {}),
    BlockCase("cosineembeddingloss", "nn.CosineEmbeddingLoss", [4, 10], {}),
    BlockCase("ctcloss", "nn.CTCLoss", [8, 4, 12], {}),
]


def _control_graph(case: BlockCase) -> ArchitectureGraph:
    layer_node_id = "node1"
    nodes = [
        ArchitectureNode(
            node_id="input",
            kind="input",
            op="Input",
            params={"shape": case.input_shape},
        ),
        ArchitectureNode(
            node_id=layer_node_id,
            kind="layer",
            op=case.op,
            params=case.params,
        ),
        ArchitectureNode(
            node_id="output",
            kind="output",
            op="Output",
            params={},
        ),
    ]
    edges = [
        ArchitectureEdge(
            edge_id="e1",
            source_node_id="input",
            source_port="y",
            target_node_id=layer_node_id,
            target_port="x",
        ),
        ArchitectureEdge(
            edge_id="e2",
            source_node_id=layer_node_id,
            source_port="y",
            target_node_id="output",
            target_port="x",
        ),
    ]
    return ArchitectureGraph(
        graph_id=f"g_{case.node_id}",
        name=f"g_{case.node_id}",
        revision=1,
        nodes=nodes,
        edges=edges,
    )


def _data_payload(case: BlockCase) -> dict:
    class_name = case.op.removeprefix("nn.")
    shape_str = ",".join(str(v) for v in case.input_shape)
    return {
        "nodes": [
            {
                "id": "input",
                "type": "inputNode",
                "position": {"x": 0.0, "y": 0.0},
                "data": {
                    "layerType": {"id": "input", "name": "Input", "parameters": []},
                    "parameters": {"shape": shape_str},
                },
            },
            {
                "id": "node1",
                "type": "layerNode",
                "position": {"x": 120.0, "y": 0.0},
                "data": {
                    "layerType": {"id": case.node_id, "name": class_name, "parameters": []},
                    "parameters": case.params,
                },
            },
            {
                "id": "output",
                "type": "outputNode",
                "position": {"x": 240.0, "y": 0.0},
                "data": {
                    "layerType": {"id": "output", "name": "Output", "parameters": []},
                    "parameters": {},
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "node1"},
            {"id": "e2", "source": "node1", "target": "output"},
        ],
    }


@pytest.mark.parametrize("case", UNIT_CASES, ids=lambda c: c.node_id)
def test_unit_block_control_plane_validation(case: BlockCase):
    service = CanonicalValidationService()
    result = service.validate(_control_graph(case), strict=True)
    assert result.valid, f"{case.node_id} failed strict control-plane validation: {result}"


@pytest.mark.parametrize("case", UNIT_CASES, ids=lambda c: c.node_id)
def test_unit_block_data_plane_validation(case: BlockCase):
    result = verify_model(
        graph_payload=_data_payload(case),
        input_shape=case.input_shape,
        run_forward=False,
    )
    assert result.get("ok") is True, f"{case.node_id} failed data-plane validation: {result}"
    assert not result.get("errors"), f"{case.node_id} returned validation errors: {result.get('errors')}"


INTEGRATION_CASES = [
    {
        "name": "unfold_to_fold",
        "input_shape": [1, 3, 8, 8],
        "layers": [
            ("u1", "nn.Unfold", {"kernel_size": 2, "stride": 2}),
            ("f1", "nn.Fold", {"output_size": [8, 8], "kernel_size": 2, "stride": 2}),
        ],
    },
    {
        "name": "channelshuffle_to_fractionalmaxpool",
        "input_shape": [1, 4, 16, 16],
        "layers": [
            ("c1", "nn.ChannelShuffle", {"groups": 2}),
            ("p1", "nn.FractionalMaxPool2d", {"output_size": [8, 8], "kernel_size": 2}),
        ],
    },
    {
        "name": "linear_to_mseloss",
        "input_shape": [1, 16],
        "layers": [
            ("l1", "nn.Linear", {"in_features": 16, "out_features": 8}),
            ("loss", "nn.MSELoss", {}),
        ],
    },
]


def _control_integration_graph(case: dict) -> ArchitectureGraph:
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": case["input_shape"]}),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}),
    ]
    edges = []
    prev = "input"
    edge_idx = 1
    for node_id, op, params in case["layers"]:
        nodes.append(ArchitectureNode(node_id=node_id, kind="layer", op=op, params=params))
        edges.append(
            ArchitectureEdge(
                edge_id=f"e{edge_idx}",
                source_node_id=prev,
                source_port="y",
                target_node_id=node_id,
                target_port="x",
            )
        )
        prev = node_id
        edge_idx += 1
    edges.append(
        ArchitectureEdge(
            edge_id=f"e{edge_idx}",
            source_node_id=prev,
            source_port="y",
            target_node_id="output",
            target_port="x",
        )
    )
    return ArchitectureGraph(
        graph_id=f"int_{case['name']}",
        name=f"int_{case['name']}",
        revision=1,
        nodes=nodes,
        edges=edges,
    )


def _data_integration_payload(case: dict) -> dict:
    shape_str = ",".join(str(v) for v in case["input_shape"])
    nodes = [
        {
            "id": "input",
            "type": "inputNode",
            "position": {"x": 0.0, "y": 0.0},
            "data": {
                "layerType": {"id": "input", "name": "Input", "parameters": []},
                "parameters": {"shape": shape_str},
            },
        }
    ]
    edges = []
    prev = "input"
    edge_idx = 1
    for idx, (node_id, op, params) in enumerate(case["layers"], start=1):
        nodes.append(
            {
                "id": node_id,
                "type": "layerNode",
                "position": {"x": float(idx * 120), "y": 0.0},
                "data": {
                    "layerType": {"id": node_id, "name": op.removeprefix("nn."), "parameters": []},
                    "parameters": params,
                },
            }
        )
        edges.append({"id": f"e{edge_idx}", "source": prev, "target": node_id})
        prev = node_id
        edge_idx += 1
    nodes.append(
        {
            "id": "output",
            "type": "outputNode",
            "position": {"x": float((len(case["layers"]) + 1) * 120), "y": 0.0},
            "data": {
                "layerType": {"id": "output", "name": "Output", "parameters": []},
                "parameters": {},
            },
        }
    )
    edges.append({"id": f"e{edge_idx}", "source": prev, "target": "output"})
    return {"nodes": nodes, "edges": edges}


@pytest.mark.parametrize("case", INTEGRATION_CASES, ids=lambda c: c["name"])
def test_integration_block_control_plane_validation(case: dict):
    service = CanonicalValidationService()
    result = service.validate(_control_integration_graph(case), strict=True)
    assert result.valid, f"{case['name']} failed strict control-plane validation: {result}"


@pytest.mark.parametrize("case", INTEGRATION_CASES, ids=lambda c: c["name"])
def test_integration_block_data_plane_validation(case: dict):
    result = verify_model(
        graph_payload=_data_integration_payload(case),
        input_shape=case["input_shape"],
        run_forward=False,
    )
    assert result.get("ok") is True, f"{case['name']} failed data-plane validation: {result}"
    assert not result.get("errors"), f"{case['name']} returned validation errors: {result.get('errors')}"
