"""Untrusted graph data must never become executable source tokens."""

from __future__ import annotations

import ast

import pytest

from core.graph_input_safety import UnsafeGraphInput
from models.graph_models import Graph, LayerData, LayerType, Node
from services.code_generation import CodeGenerationService


UNSAFE_PARAMETER_KEY = (
    "inplace=False); __import__('builtins').print('UNSAFE_GRAPH_INPUT'); #"
)


def _flat_relu(parameters) -> Graph:
    return Graph(
        nodes=[
            Node(
                id="relu",
                type="layerNode",
                position={"x": 0, "y": 0},
                data=LayerData(
                    layerType=LayerType(
                        id="relu",
                        name="ReLU",
                        parameters=[],
                    ),
                    parameters=parameters,
                ),
            )
        ],
        edges=[],
    )


def test_codegen_rejects_unsafe_parameter_key_before_emitting_source():
    with pytest.raises(UnsafeGraphInput):
        CodeGenerationService().generate_flat_code(
            _flat_relu({UNSAFE_PARAMETER_KEY: 0}),
            input_shape=[1, 4],
            class_name="GeneratedModel",
        )


def test_constructor_keys_are_checked_against_supported_layer_signature():
    with pytest.raises(UnsafeGraphInput, match="not accepted"):
        CodeGenerationService().generate_flat_code(
            _flat_relu({"unknown_constructor_argument": 1}),
            input_shape=[1, 4],
            class_name="GeneratedModel",
        )


def test_normal_supported_layer_still_generates_valid_python():
    source = CodeGenerationService().generate_flat_code(
        _flat_relu({"inplace": True}),
        input_shape=[1, 4],
        class_name="GeneratedModel",
    )

    ast.parse(source)
    assert "self.relu = nn.ReLU(inplace=True)" in source
