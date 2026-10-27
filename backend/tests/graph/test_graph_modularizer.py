"""Phase-2 modular codegen: a flat repeated graph is factored into reusable ×N
module classes for code generation, while the verify/validate gates keep seeing
the flat graph (so this never touches them).

The headline case is the full 267-node F5-TTS template: its 22 DiT blocks (which
read both the sequence and the global timestep embedding `t_emb`) must become a
single multi-input `DitBlock` instantiated 22× via nn.ModuleList.
"""
import ast

import pytest

from models.graph_models import HierarchicalGraph, ModuleDefinition, Node, Edge, LayerData, LayerType
from services.code_generation import CodeGenerationService
from services.generative_templates import _build_f5tts_structure
from services.graph_modularizer import modularize_for_codegen


@pytest.fixture
def f5_flat():
    s = _build_f5tts_structure()
    return {"nodes": s["root_nodes"], "edges": s["root_edges"]}


def _frontend_grouped(flat):
    """Mimic the canvas payload: the Phase-1 frontend display-groups (moduleNode
    wrappers + parentId children + synthetic display edges to module ids). This
    is what the Export buttons actually POST — not the raw flat graph."""
    nodes = {n["id"]: n for n in flat["nodes"]}
    groups = {
        "auto-mod-0": ("ConvNeXt-V2", [i for i in nodes if i.startswith("cnx")]),
        "auto-mod-1": ("Conv1D", ["cpe_c1", "cpe_m1", "cpe_c2", "cpe_m2"]),
        "auto-mod-2": ("DiT", [i for i in nodes if i.startswith("dit")]),
        "auto-mod-3": ("SinusPositionEmbedding", ["t_sinus", "t_lin1", "t_silu"]),
    }
    parent, mod_nodes = {}, []
    for mid, (label, members) in groups.items():
        for m in members:
            parent[m] = mid
        mod_nodes.append({"id": mid, "type": "moduleNode", "position": {"x": 0, "y": 0},
                          "data": {"layerType": {"id": "module", "name": "Module", "parameters": []},
                                   "parameters": {"name": label}, "moduleInstanceId": mid}})
    out_nodes = mod_nodes + [
        {**n, "data": {**n["data"], **({"parentId": parent[n["id"]]} if n["id"] in parent else {})}}
        for n in flat["nodes"]
    ]
    fold = lambda i: parent.get(i, i)
    synth, seen = [], set()
    for e in flat["edges"]:
        fs, ft = fold(e["source"]), fold(e["target"])
        if fs == ft or (fs == e["source"] and ft == e["target"]):
            continue
        if (fs, ft) in seen:
            continue
        seen.add((fs, ft))
        synth.append({"id": f"s_{fs}_{ft}", "source": fs, "target": ft, "data": {"synthetic": True}})
    return {"nodes": out_nodes, "edges": flat["edges"] + synth}


def test_modularizer_detects_dit_and_convnext_blocks(f5_flat):
    hg = modularize_for_codegen(f5_flat)
    assert hg is not None
    mods = hg.modules

    dit = next((m for m in mods.values() if m.parameters.get("repeat_count") == 22), None)
    assert dit is not None, "DiT stack not factored"
    # ONE canonical block (not 22×): 10 spine nodes + the off-spine AdaLayerNorm-
    # Zero modulation Linear pulled in by the off-spine expansion.
    assert len(dit.internal_nodes) == 11
    assert any(n.id.endswith("_mod") for n in dit.internal_nodes)
    assert dit.is_modular_block
    # Multi-input: the block reads the sequence (x) AND the global t_emb (cond0).
    assert dit.input_arg_names == ["x", "cond0"]
    assert len(dit.conditioning_source_ids) == 1

    cnx = next((m for m in mods.values() if m.parameters.get("repeat_count") == 4), None)
    assert cnx is not None, "ConvNeXt stack not factored"
    assert len(cnx.internal_nodes) == 7
    assert cnx.input_arg_names == ["x"]             # single-input block
    assert cnx.conditioning_source_ids == []


def test_modular_codegen_emits_reusable_blocks(f5_flat):
    hg = modularize_for_codegen(f5_flat)
    code = CodeGenerationService().generate_hierarchical_code(hg, [1, 256, 100], "F5TTS")

    ast.parse(code)  # must be syntactically valid Python

    # Reusable classes + ModuleList instantiation, not 220 inlined layers.
    assert "class DitBlock(nn.Module):" in code
    assert "nn.ModuleList([DitBlock() for _ in range(22)])" in code
    assert "nn.ModuleList([ConvnextV2Block() for _ in range(4)])" in code

    # The DiT block's forward takes BOTH the sequence and the conditioning, and
    # the main loop threads the sequence while holding the conditioning constant.
    assert "def forward(self, x, cond0):" in code
    assert "for _blk in self.dit_block:" in code
    assert "_blk(out_dit_block, out_t_lin2)" in code

    # Compression: a flat dump of 267 nodes would be ~280+ lines (incl. the
    # emitted custom-op classes the modular form still beats it comfortably).
    assert len(code.splitlines()) < 200


def test_exported_f5tts_imports_and_runs(f5_flat):
    """The exported F5-TTS code is RUNNABLE: custom ops (GRN/RoPE/AdaLayerNorm/
    SinusoidalPosEmbedding) are emitted as real shape-preserving nn.Modules, the
    model takes its 4 inputs, and a forward returns the predicted mel shape."""
    import torch

    hg = modularize_for_codegen(f5_flat)
    code = CodeGenerationService().generate_hierarchical_code(hg, [1, 256, 100], "F5TTS")

    # Custom-op classes are defined; merge ops are functional, not bogus nn.
    for cls in ("class GRN", "class RotaryEmbedding", "class AdaLayerNorm", "class SinusoidalPosEmbedding"):
        assert cls in code
    assert "nn.Add(" not in code and "nn.Concat(" not in code
    assert "def forward(self, in_text, in_noisy, in_cond, in_time):" in code

    ns: dict = {}
    exec(code, ns)
    model = ns["F5TTS"]()
    y = model(
        torch.randint(0, 257, (1, 256)),  # text token ids (Embedding-fed)
        torch.randn(1, 256, 100),          # noisy mel
        torch.randn(1, 256, 100),          # reference mel
        torch.randn(1, 256),               # timestep
    )
    assert list(y.shape) == [1, 256, 100]


def test_canvas_grouped_payload_exports_runnable_modular_code(f5_flat):
    """Regression: the Export buttons POST the FRONTEND display-grouped graph,
    not the flat one. The modularizer must flatten those groups and re-detect,
    producing the modular ×N form — NOT the literal frontend groups with a
    broken main forward ('Cycle detected ...')."""
    import torch

    grouped = _frontend_grouped(f5_flat)
    hg = modularize_for_codegen(grouped)
    assert hg is not None, "modularizer bailed on a grouped canvas payload"
    # Re-detected the real ×N blocks (not the 4 literal frontend groups).
    assert any(m.is_modular_block and m.parameters.get("repeat_count") == 22 for m in hg.modules.values())
    assert any(m.is_modular_block and m.parameters.get("repeat_count") == 4 for m in hg.modules.values())

    code = CodeGenerationService().generate_hierarchical_code(hg, [1, 256, 100], "F5TTS")
    assert "Cycle detected" not in code      # the bug this guards against
    assert "nn.ModuleList" in code
    assert "def forward(self, in_text, in_noisy, in_cond, in_time):" in code

    ns: dict = {}
    exec(code, ns)
    y = ns["F5TTS"]()(
        torch.randint(0, 257, (1, 256)),
        torch.randn(1, 256, 100),
        torch.randn(1, 256, 100),
        torch.randn(1, 256),
    )
    assert list(y.shape) == [1, 256, 100]


def test_no_repeats_returns_none():
    """A graph with no repeated block is left flat (caller falls back)."""
    flat = {
        "nodes": [
            {"id": "i", "type": "inputNode", "position": {"x": 0, "y": 0},
             "data": {"layerType": {"id": "input", "name": "Input", "parameters": []}, "parameters": {}}},
            {"id": "l", "type": "layerNode", "position": {"x": 0, "y": 0},
             "data": {"layerType": {"id": "linear", "name": "Linear", "parameters": []},
                      "parameters": {"in_features": 8, "out_features": 8}}},
            {"id": "o", "type": "outputNode", "position": {"x": 0, "y": 0},
             "data": {"layerType": {"id": "output", "name": "Output", "parameters": []}, "parameters": {}}},
        ],
        "edges": [{"id": "i_l", "source": "i", "target": "l"},
                  {"id": "l_o", "source": "l", "target": "o"}],
    }
    assert modularize_for_codegen(flat) is None


def test_real_hierarchical_graph_returns_none(f5_flat):
    """A genuine backend HierarchicalGraph ({root_nodes, modules}, no flat
    'nodes' key) is left for the caller — only flat / frontend-grouped {nodes,
    edges} payloads are modularized here."""
    assert modularize_for_codegen({"root_nodes": f5_flat["nodes"], "root_edges": f5_flat["edges"], "modules": {}}) is None


def test_frontend_grouped_is_flattened_not_bailed(f5_flat):
    """A flat graph carrying frontend moduleNode wrappers is flattened + re-
    detected (the old behaviour bailed to None — that was the export bug)."""
    grouped = _frontend_grouped(f5_flat)
    hg = modularize_for_codegen(grouped)
    assert hg is not None and len(hg.modules) >= 2


def test_non_modular_module_still_emits_single_instance():
    """A hierarchical module WITHOUT is_modular_block keeps the legacy behaviour
    (one instance, no nn.ModuleList) — guards existing templates/nanoDiT."""
    lt_lin = LayerType(id="linear", name="Linear", parameters=[])
    blk = Node(id="enc__layer_0", type="layerNode", position={"x": 0, "y": 0},
               data=LayerData(layerType=lt_lin, parameters={"in_features": 8, "out_features": 8}))
    mod = ModuleDefinition(module_id="enc", parent_id=None, internal_nodes=[blk],
                           internal_edges=[], input_node_id="enc__layer_0", output_node_id="enc__layer_0",
                           parameters={"repeat_count": 6, "repeat_block_label": "Linear"})
    mod_node = Node(id="enc", type="layerNode", position={"x": 0, "y": 0},
                    data=LayerData(layerType=LayerType(id="module", name="Module", parameters=[]), parameters={}))
    hg = HierarchicalGraph(root_nodes=[mod_node], root_edges=[], modules={"enc": mod})
    code = CodeGenerationService().generate_hierarchical_code(hg, [1, 8], "M")
    assert "nn.ModuleList" not in code      # legacy modules are not auto-looped
    assert "self.enc = Enc(" in code        # single instance (legacy behaviour)
    assert "for _blk in" not in code        # no modular forward loop
