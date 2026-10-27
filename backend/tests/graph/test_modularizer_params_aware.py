"""Repeat detection must compare constructor params, not just op types.

Folding keeps only instance 0 (``canonical_spine = members[0:period]``) and emits
``nn.ModuleList([Block() for _ in range(N)])``. Keying repeats on op type alone
therefore collapsed materially different spans onto the first span's parameters and
produced code that *runs* while computing a different architecture.

The trigger is ordinary: a 3-layer MLP is ``Linear(16,32) ReLU Linear(32,32) ReLU``
— two op-identical spans of 2, which meets ``MIN_SPAN`` — and the generated file
raised ``mat1 and mat2 shapes cannot be multiplied (1x32 and 16x32)``. This is the
path behind ``POST /api/models/generate-code`` (``server_v2.py:315``), so it broke
user-facing Export Code, not only AI-authored graphs. It also caused the
``unet_w64_d4_block`` export-parity gap, whose xfail entry this fix removed.
"""

from __future__ import annotations

import torch

from services.code_generation import CodeGenerationService
from services.graph_modularizer import modularize_for_codegen


def _node(nid, tid, name, params, typ="layerNode"):
    return {"id": nid, "type": typ, "position": {"x": 0, "y": 0},
            "data": {"layerType": {"id": tid, "name": name, "parameters": []},
                     "parameters": params}}


def _chain(nodes):
    ids = [n["id"] for n in nodes]
    return {"nodes": nodes,
            "edges": [{"id": f"e{i}", "source": ids[i], "target": ids[i + 1]}
                      for i in range(len(ids) - 1)]}


def _mlp(widths):
    """input -> [Linear(w_in,w_out) ReLU]* -> Linear(last,4) -> output"""
    nodes = [_node("input_node", "input", "Input", {"shape": f"1,{widths[0][0]}"}, "inputNode")]
    for i, (a, b) in enumerate(widths):
        nodes.append(_node(f"l{i}", "linear", "Linear", {"in_features": a, "out_features": b}))
        nodes.append(_node(f"r{i}", "relu", "ReLU", {}))
    nodes.append(_node("head", "linear", "Linear",
                       {"in_features": widths[-1][1], "out_features": 4}))
    nodes.append(_node("output_node", "output", "Output", {}, "outputNode"))
    return _chain(nodes)


def _runs(hierarchical, input_shape, class_name="M"):
    code = CodeGenerationService().generate_hierarchical_code(hierarchical, input_shape, class_name)
    ns: dict = {}
    exec(compile(code, "generated", "exec"), ns)
    return ns[class_name]()(torch.randn(*input_shape))


# ── the bug ──────────────────────────────────────────────────────────────────
def test_spans_differing_only_in_width_are_not_folded():
    """The exact repro: 16->32 then 32->32 must not become one block reused twice."""
    hierarchical = modularize_for_codegen(_mlp([(16, 32), (32, 32)]))
    if hierarchical is not None:
        for module in hierarchical.modules.values():
            linears = [n for n in module.internal_nodes
                       if n.data.layerType.name == "Linear"]
            widths = {(n.data.parameters.get("in_features"),
                       n.data.parameters.get("out_features")) for n in linears}
            assert len(widths) <= 1, f"folded mismatched widths into one block: {widths}"


def test_the_generated_code_for_a_deeper_mlp_actually_runs():
    """What the user experiences. Before the fix this raised
    'mat1 and mat2 shapes cannot be multiplied (1x32 and 16x32)'."""
    flat = _mlp([(16, 32), (32, 32)])
    hierarchical = modularize_for_codegen(flat)
    if hierarchical is None:
        return  # flat fallback is the safe outcome; nothing to assert here
    assert tuple(_runs(hierarchical, [1, 16]).shape) == (1, 4)


# ── must not over-fragment ───────────────────────────────────────────────────
def test_genuinely_identical_blocks_still_fold():
    """Guards the feature this module exists for — an F5-TTS DiTBlock x22 must not
    degrade into 22 inlined copies because the signature got too strict."""
    hierarchical = modularize_for_codegen(_mlp([(32, 32)] * 3))
    assert hierarchical is not None, "identical blocks stopped folding"
    counts = [(n.data.parameters or {}).get("repeat_count")
              for n in hierarchical.root_nodes]
    assert 3 in counts, f"expected a x3 module, got repeat_counts {counts}"
    assert tuple(_runs(hierarchical, [1, 32]).shape) == (1, 4)


def test_a_display_name_difference_does_not_split_a_run():
    """Per-instance display names ("DiTBlock 1"/"DiTBlock 2") are not constructor
    args; letting them into the signature would silently stop all real folding."""
    flat = _mlp([(32, 32)] * 3)
    for i, node in enumerate(flat["nodes"]):
        node["data"]["parameters"]["name"] = f"Block {i}"
        node["data"]["parameters"]["_subtitle"] = f"instance {i}"
    hierarchical = modularize_for_codegen(flat)
    assert hierarchical is not None, "display-only keys split the repeat run"
    assert 3 in [(n.data.parameters or {}).get("repeat_count") for n in hierarchical.root_nodes]
