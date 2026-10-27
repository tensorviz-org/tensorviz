"""Generated code for functional ``torch.*`` nodes has to actually run.

``core/layer_builder.py`` has long refused an ``nn.`` constructor missing a
required positional argument. The functional path had no equivalent, so a node
left at its palette defaults emitted calls that are syntactically fine and raise
the moment they execute:

    torch.reshape(x)                    TypeError: missing 'shape'
    torch.split(x, dim=1)               TypeError: missing 'split_size_or_sections'
    torch.chunk(x, dim=1)               TypeError: missing 'chunks'
    torch.transpose(x, dims=[1, 3])     TypeError: no 'dims' keyword

The last one is not a missing param at all — this app stores transpose's axes as a
2-list under ``dims`` while torch's signature is ``transpose(input, dim0, dim1)``.

Every expectation below is checked against real torch, so the tests cannot drift
from the library they describe.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from core.graph_input_safety import UnsafeGraphInput
from models.graph_models import Edge, Graph, LayerData, LayerType, Node
from services.code_generation import CodeGenerationService


def _node(node_id: str, op: str, params: dict, op_id: str | None = None) -> Node:
    return Node(
        id=node_id,
        type="layerNode",
        position={"x": 0, "y": 0},
        data=LayerData(
            layerType=LayerType(id=op_id or node_id.split("-")[0], name=op, parameters=[]),
            parameters=params,
        ),
    )


def _graph(op: str, params: dict) -> Graph:
    """Input -> op, the smallest graph that exercises _emit_op."""
    nodes = [
        _node("input-1", "Input", {"shape": "1,64,32,32"}),
        _node("target-1", op, params, op_id=op.rsplit(".", 1)[-1].lower()),
    ]
    return Graph(nodes=nodes, edges=[Edge(id="e1", source="input-1", target="target-1")])


def _generate(op: str, params: dict) -> str:
    return CodeGenerationService().generate_flat_code(
        _graph(op, params), [1, 64, 32, 32], "SliceProbe")


def _forward(code: str, x: torch.Tensor) -> torch.Tensor:
    """Run the generated module. The `if __name__` example block does not execute
    on import, so this is just the class definition plus one forward pass."""
    namespace: dict = {}
    exec(compile(code, "<generated>", "exec"), namespace)
    return namespace["SliceProbe"]()(x)


# ── the guard ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("op,params,needed", [
    ("torch.reshape", {}, "shape"),
    ("torch.narrow", {"dim": 1}, "start"),
    ("torch.narrow", {}, "dim"),
    ("torch.split", {"dim": 1}, "split_size_or_sections"),
    ("torch.chunk", {"dim": 1}, "chunks"),
])
def test_export_refuses_a_functional_op_missing_a_required_argument(op, params, needed):
    with pytest.raises(UnsafeGraphInput) as err:
        _generate(op, params)
    # The message has to name the parameter, or it is not actionable.
    assert needed in str(err.value)


def test_a_param_left_blank_counts_as_missing():
    # '' is exactly what a palette default leaves behind, and the kwargs builder
    # skips it — so the guard has to treat it as absent rather than supplied.
    with pytest.raises(UnsafeGraphInput):
        _generate("torch.reshape", {"shape": ""})


def test_interpolate_needs_size_or_scale_factor():
    with pytest.raises(UnsafeGraphInput) as err:
        _generate("interpolate", {"mode": "nearest"})
    assert "size" in str(err.value) and "scale_factor" in str(err.value)
    # …and either one alone is enough.
    assert "F.interpolate(" in _generate(
        "interpolate", {"scale_factor": 2, "mode": "nearest"})
    assert "F.interpolate(" in _generate(
        "interpolate", {"size": [64, 64], "mode": "nearest"})


# ── what gets emitted, cross-checked against torch ────────────────────────────

def test_narrow_emits_a_call_torch_accepts():
    code = _generate("torch.narrow", {"dim": 1, "start": 0, "length": 16})
    assert "torch.narrow(" in code
    for part in ("dim=1", "start=0", "length=16"):
        assert part in code, part
    # The emitted keyword form is one torch really takes.
    assert list(torch.narrow(torch.zeros(1, 64, 32, 32), dim=1, start=0, length=16).shape) \
        == [1, 16, 32, 32]


def test_interpolate_is_emitted_from_the_functional_namespace():
    code = _generate("interpolate", {"scale_factor": 2, "mode": "nearest"})
    assert "F.interpolate(" in code
    # torch.interpolate does NOT exist — emitting it would be a NameError/AttributeError.
    assert "torch.interpolate(" not in code
    assert not hasattr(torch, "interpolate")
    assert list(F.interpolate(torch.zeros(1, 4, 8, 8), scale_factor=2, mode="nearest").shape) \
        == [1, 4, 16, 16]


def test_transpose_dims_are_remapped_to_torchs_signature():
    code = _generate("torch.transpose", {"dims": [1, 3]})
    assert "dim0=1" in code and "dim1=3" in code
    assert "dims=" not in code, "emitting dims= would be a TypeError"
    with pytest.raises(TypeError):
        torch.transpose(torch.zeros(1, 64, 32, 32), dims=[1, 3])


def test_permute_keeps_dims_because_torch_uses_that_name():
    code = _generate("torch.permute", {"dims": [0, 2, 3, 1]})
    assert "dims=" in code
    assert list(torch.permute(torch.zeros(1, 64, 32, 32), dims=(0, 2, 3, 1)).shape) \
        == [1, 32, 32, 64]


def test_the_axes_typed_into_the_panel_generate_a_call_that_runs():
    """The Properties field for permute/transpose is free text, so its value
    arrives as the comma string the Input node's shape uses. Emitted verbatim that
    was `torch.permute(x, dims='0,2,3,1')` — a TypeError — so the field could not
    be used even after it was renamed from `dim` to the name everything reads."""
    perm = _generate("torch.permute", {"dims": "0,2,3,1"})
    assert "dims=[0, 2, 3, 1]" in perm
    assert list(_forward(perm, torch.zeros(1, 64, 32, 32)).shape) \
        == list(torch.permute(torch.zeros(1, 64, 32, 32), (0, 2, 3, 1)).shape) \
        == [1, 32, 32, 64]

    trans = _generate("torch.transpose", {"dims": "1,3"})
    assert "dim0=1" in trans and "dim1=3" in trans
    assert "dims=" not in trans, "emitting dims= would be a TypeError"
    assert list(_forward(trans, torch.zeros(1, 64, 32, 32)).shape) == [1, 32, 32, 64]


def test_the_dim_the_old_palette_left_behind_is_not_emitted():
    """Neither torch.permute nor torch.transpose has a `dim` keyword at all, and a
    graph saved while the palette exposed one still carries it next to the `dims`
    the user has since filled in. Emitting both is a TypeError even though each
    value is well formed on its own."""
    code = _generate("torch.permute", {"dim": "0,2,1", "dims": "0,2,3,1"})
    assert "dim=" not in code
    assert list(_forward(code, torch.zeros(1, 64, 32, 32)).shape) == [1, 32, 32, 64]


def test_a_fully_specified_reshape_still_generates():
    code = _generate("torch.reshape", {"shape": "1,64,1024"})
    assert "torch.reshape(" in code and "shape=" in code


def test_the_generated_module_is_valid_python():
    import ast
    ast.parse(_generate("torch.narrow", {"dim": 1, "start": 0, "length": 16}))
