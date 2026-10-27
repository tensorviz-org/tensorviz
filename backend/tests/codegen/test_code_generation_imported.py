"""Code generation must not present a partial reconstruction as the model.

An imported model's blocks are library classes (``ResnetBlock2D``, ``Timesteps``,
``CrossAttnDownBlock2D``) whose bodies are not in the graph. Generating a passthrough for
them produced a file that imports, runs, and computes something completely different, with
nothing anywhere to say so — and the user did not author the model, so they had no way to
notice. These tests pin the two things that make the export honest: unique class names, and
a refusal rather than a silent identity.
"""

from __future__ import annotations

import ast
import re

from models.graph_models import (
    Edge,
    Graph,
    HierarchicalGraph,
    LayerData,
    LayerType,
    ModuleDefinition,
    Node,
)
from services.code_generation import CodeGenerationService


def _module_node(node_id: str, cls: str, imported: bool) -> Node:
    params = {"num_channels": 320}
    if imported:
        params.update({"_module_path": node_id, "_class": cls})
    return Node(
        id=node_id,
        type="moduleNode",
        position={"x": 0, "y": 0},
        data=LayerData(
            layerType=LayerType(id=cls.lower(), name=cls, parameters=[]),
            parameters=params,
        ),
    )


def _hierarchical(node_ids, cls="ResnetBlock2D", imported=True, with_defs=True):
    """A graph of module nodes whose ModuleDefinitions are EMPTY — the imported shape."""
    nodes = [_module_node(nid, cls, imported) for nid in node_ids]
    modules = {}
    if with_defs:
        modules = {
            nid: ModuleDefinition(module_id=nid, parent_id=None, internal_nodes=[], internal_edges=[])
            for nid in node_ids
        }
    edges = [
        Edge(id=f"e{i}", source=node_ids[i], target=node_ids[i + 1])
        for i in range(len(node_ids) - 1)
    ]
    return HierarchicalGraph(root_nodes=nodes, root_edges=edges, modules=modules)


# ── class naming ─────────────────────────────────────────────────────────────
def test_dotted_module_paths_do_not_collide_into_one_class_name():
    """The old derivation split only on -/_/space then dropped any part failing isalnum(),
    so `down_blocks.0` lost `blocks.0` to the dot and became `Down` — as did every sibling.
    An imported UNet defined the same class several times and the last one silently won.
    """
    svc = CodeGenerationService()
    ids = ["down_blocks.0", "down_blocks.1", "down_blocks.2", "up_blocks.3", "transformer.h.0"]
    names = [svc._module_id_to_class_name(i) for i in ids]
    assert len(set(names)) == len(names), f"collision: {names}"
    assert names[0] == "DownBlocks0"
    assert names[-1] == "TransformerH0"


def test_internal_capitalisation_is_preserved():
    """`capitalize()` lowercased the rest of each part, turning DiTBlock into Ditblock."""
    assert CodeGenerationService()._module_id_to_class_name("DiTBlock") == "DiTBlock"


def test_generated_file_defines_each_class_exactly_once():
    code = CodeGenerationService().generate_hierarchical_code(
        _hierarchical(["down_blocks.0", "down_blocks.1", "down_blocks.2"]), [1, 4, 64, 64], "M"
    )
    defined = re.findall(r"^class (\w+)\(", code, re.M)
    duplicates = [c for c in set(defined) if defined.count(c) > 1]
    assert not duplicates, f"class defined more than once: {duplicates}"


# ── honesty ──────────────────────────────────────────────────────────────────
def test_imported_modules_with_no_body_raise_instead_of_returning_their_input():
    """The whole point. A passthrough here means the exported model runs and produces
    numbers from the wrong architecture."""
    code = CodeGenerationService().generate_hierarchical_code(
        _hierarchical(["down_blocks.0", "down_blocks.1"]), [1, 4, 64, 64], "M"
    )
    assert code.count("raise NotImplementedError") == 2
    assert not re.search(r"^\s+return x$", code, re.M), "an imported stub still passes through"
    ast.parse(code)


def test_authored_modules_keep_their_passthrough_placeholder():
    """Hand-authored graphs (e.g. the VAE templates) rely on a shape-preserving passthrough
    the user is expected to fill in; that behaviour must not change."""
    code = CodeGenerationService().generate_hierarchical_code(
        _hierarchical(["encoder_block"], cls="VAEEncoder", imported=False), [1, 3, 64, 64], "M"
    )
    assert "raise NotImplementedError" not in code


def test_unreproducible_modules_are_reported_even_when_a_definition_exists():
    """`modules[node_id]` existing is not the same as being reproducible: an empty
    ModuleDefinition still generates a passthrough. Reporting only the missing-definition
    case let an imported UNet claim a complete export while emitting 38 passthroughs."""
    svc = CodeGenerationService()
    graph = _hierarchical(["down_blocks.0", "down_blocks.1"], with_defs=True)
    reported = svc.unreproducible_modules(graph.root_nodes, graph.modules)
    assert {r["node_id"] for r in reported} == {"down_blocks.0", "down_blocks.1"}
    assert all(r["imported"] for r in reported)
    assert reported[0]["source_class"] == "ResnetBlock2D"


def test_a_reproducible_module_is_not_reported():
    """Guards the inverse: a module with real internals must not be flagged, or the warning
    becomes noise and gets ignored."""
    svc = CodeGenerationService()
    inner = Node(
        id="fc", type="layerNode", position={"x": 0, "y": 0},
        data=LayerData(layerType=LayerType(id="linear", name="Linear", parameters=[]),
                       parameters={"in_features": 8, "out_features": 8}),
    )
    graph = _hierarchical(["block"], with_defs=False)
    graph.modules = {
        "block": ModuleDefinition(module_id="block", parent_id=None, internal_nodes=[inner],
                                  internal_edges=[])
    }
    assert svc.unreproducible_modules(graph.root_nodes, graph.modules) == []


def test_the_generated_file_carries_the_warning_itself():
    """The file is what leaves the product — it gets downloaded and run somewhere the UI
    banner cannot follow it, so the warning has to be in the source."""
    code = CodeGenerationService().generate_hierarchical_code(
        _hierarchical(["down_blocks.0"]), [1, 4, 64, 64], "M"
    )
    assert "INCOMPLETE EXPORT" in code
    assert "ResnetBlock2D" in code.splitlines()[4]
    assert code.splitlines()[0].startswith("#"), "the banner must precede the imports"
