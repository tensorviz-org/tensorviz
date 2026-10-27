"""A layer parameter is whatever the canvas stored, not always a bare int.

The editor writes an asymmetric kernel as [3, 3] and a typed one as "3" or
"3,3". `_validate_layer_parameters` compared those raw values against 0, which
raised TypeError, which the endpoint's bare `except` returned as HTTP 500 — so
the project could not be validated at all. That matters now that
/api/models/validate gates the "export as Python" button on the project list:
an unvalidatable graph is an unexportable one, even though codegen emits it
fine.
"""
import pytest

from models.graph_models import Graph, Node, LayerData, LayerType, Edge
from utils.model_validation import ModelValidationService


def node(node_id: str, type_id: str, parameters: dict) -> Node:
    return Node(
        id=node_id,
        type="layerNode",
        position={"x": 0.0, "y": 0.0},
        data=LayerData(
            layerType=LayerType(id=type_id, name=type_id, parameters=[]),
            parameters=parameters,
        ),
    )


@pytest.mark.parametrize(
    "parameters",
    [
        {"kernel_size": 3, "in_channels": 3, "out_channels": 8},
        {"kernel_size": [3, 3], "in_channels": 3, "out_channels": 8},
        {"kernel_size": (3, 5), "in_channels": 3, "out_channels": 8},
        {"kernel_size": "3", "in_channels": "3", "out_channels": "8"},
        {"kernel_size": "3,3", "in_channels": 3, "out_channels": 8},
    ],
    ids=["int", "list", "tuple", "strings", "comma-string"],
)
def test_every_stored_form_of_a_valid_kernel_passes(parameters):
    graph = Graph(nodes=[node("n1", "conv2d", parameters)], edges=[])

    result = ModelValidationService().validate_model_graph(graph)

    assert result["errors"] == []


@pytest.mark.parametrize(
    "parameters, expected",
    [
        ({"kernel_size": 0, "in_channels": 3, "out_channels": 8}, "kernel_size"),
        ({"kernel_size": [3, 0], "in_channels": 3, "out_channels": 8}, "kernel_size"),
        ({"kernel_size": "-1", "in_channels": 3, "out_channels": 8}, "kernel_size"),
        ({"kernel_size": 3, "in_channels": 3}, "out_channels"),
    ],
    ids=["zero", "zero-in-list", "negative-string", "missing"],
)
def test_a_non_positive_value_is_still_reported(parameters, expected):
    # The tolerance above must not become blanket permissiveness: a zero hidden
    # inside a list is exactly the typo this check exists to catch.
    graph = Graph(nodes=[node("n1", "conv2d", parameters)], edges=[])

    result = ModelValidationService().validate_model_graph(graph)

    assert any(expected in message for message in result["errors"]), result["errors"]


def test_an_unreadable_value_is_not_invented_into_an_error():
    # "same as input" is not a number this function can judge. Reporting it as
    # non-positive would block a graph the shape engine and codegen both handle.
    graph = Graph(
        nodes=[node("n1", "linear", {"in_features": "same as input", "out_features": 10})],
        edges=[],
    )

    result = ModelValidationService().validate_model_graph(graph)

    assert result["errors"] == []
