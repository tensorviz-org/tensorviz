"""Tier: code -> graph -> code stability (docs/tensorviz-plan.md §9, task.md M1).

The IDE pivot rests on one property: the graph is a faithful projection of the
source. If parsing a file and regenerating it produces a DIFFERENT graph the
second time round, then every canvas edit is applied to a model the user is not
looking at. This tier makes that property executable.

The loop is parse -> generate -> parse. **Equality** is deliberately weaker than
byte identity, because generated code is not the user's code:

  * the same multiset of ``(layerType.id, non-underscore parameters)`` per node —
    ``_``-prefixed keys are provenance (which file, which line), and the second
    pass reads a generated file, so of course they differ;
  * the same edge set after mapping node ids through the attribute names the
    generator mints (``CodeGenerationService._node_id_to_layer_name``) — node ids
    themselves need not survive, the wiring must.

Coverage is 6 hand-written registry-only fixtures plus the deterministic registry
stride sample ``test_export_roundtrip.py`` uses, seeded through the generator so
the tier starts from code. Entries that cannot complete the loop in M1 are named
in ``KNOWN_ROUNDTRIP_GAPS`` with the reason, and the test asserts the reason it
observes IS the reason recorded — a gap that has closed fails as a stale key,
rather than quietly staying listed.

Run: KMP_DUPLICATE_LIB_OK=TRUE <viz-python> -m pytest -m roundtrip tests/test_code_roundtrip.py -q
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("AUTH_DISABLED", "true")


import pytest  # noqa: E402

from core.graph_input_safety import UnsafeGraphInput  # noqa: E402
from models.graph_models import CodeParseRequest, Graph  # noqa: E402
from services.canonical_template_registry import CanonicalTemplateRegistry  # noqa: E402
from services.canonical_validation import CanonicalValidationService  # noqa: E402
from services.code_generation import CodeGenerationService  # noqa: E402
from services.verification_service import _arch_node_to_flat  # noqa: E402
from scripts.run_template_validation_sweep import _build_graph_for_sequence  # noqa: E402
from utils.code_parser import CodeParserService  # noqa: E402

pytestmark = pytest.mark.roundtrip

CLASS_NAME = "RoundtripModel"

_PARSER = CodeParserService()
_GENERATOR = CodeGenerationService()
_REGISTRY = CanonicalTemplateRegistry()
_VALIDATOR = CanonicalValidationService()


# ── the six hand-written fixtures (registry ops only) ────────────────────────
FIXTURES: Dict[str, Tuple[str, List[int]]] = {
    "mlp": ("""import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
""", [1, 8]),

    "sequential": ("""import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 4),
        )
    def forward(self, x):
        return self.net(x)
""", [1, 8]),

    "cnn_with_functional_ops": ("""import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.bn(self.conv(x))))
        x = x.flatten(1)
        return self.fc(x)
""", [1, 3, 8, 8]),

    "residual_merge": ("""import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 4)
        self.l2 = nn.Linear(4, 4)
    def forward(self, x):
        h = self.l1(x)
        return x + self.l2(h)
""", [2, 4]),

    "two_branches_concatenated": ("""import torch
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        self.head = nn.Linear(8, 2)
    def forward(self, x):
        return self.head(torch.cat([self.a(x), self.b(x)], 1))
""", [2, 4]),

    "recurrent": ("""import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(10)
        self.lstm = nn.LSTM(10, 20, batch_first=True)
        self.fc = nn.Linear(20, 5)
    def forward(self, x):
        x = self.norm(x)
        x, _ = self.lstm(x)
        return self.fc(x)
""", [2, 7, 10]),
}


# ── the corpus: the same deterministic registry sample test_export_roundtrip
# walks, seeded through the generator so this tier starts from source. ───────
def _registry_sample(n_target: int = 12) -> List[str]:
    ids = sorted(contract.template_id for contract in _REGISTRY.all_contracts())
    return ids[:: max(1, len(ids) // n_target)]


CORPUS = _registry_sample()


# ── known gaps. Each names WHY the loop cannot close in M1; the test checks the
# reason it observes against the one recorded, so a closed gap fails loudly. ──
OPAQUE_STATEMENT = "opaque_statement"
CUSTOM_MODULE = "custom_module"
SEED_UNGENERATABLE = "seed_ungeneratable"

KNOWN_ROUNDTRIP_GAPS: Dict[str, str] = {
    # The generated file defines a GaussianNoise nn.Module of its own and the
    # model holds an instance of it. Parsing that back gives a `custom_code` node
    # with a class name and no module to import it from, because the class was
    # defined in the file being read rather than imported into it. M6 gave
    # custom_code a codegen branch and this gap stayed open: the branch writes an
    # import, and there is nothing here to import from.
    "latent_diffusion_block": CUSTOM_MODULE,
    # There is no code to start from: message passing needs a connectivity input
    # the canvas cannot carry, so codegen refuses the template itself. Documented
    # as an export gap in test_export_roundtrip.KNOWN_EXPORT_GAPS too.
    "gnn_mean_1hop_block": SEED_UNGENERATABLE,
}


# ── equality ─────────────────────────────────────────────────────────────────
def _hashable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


def _node_signature(node) -> Tuple[Any, ...]:
    """What a node IS, independent of what it is called: its op and the
    parameters that op was given. `_`-prefixed keys are provenance about the file
    the node was read from, which necessarily changes between the two passes."""
    semantic = {
        key: _hashable(value)
        for key, value in (node.data.parameters or {}).items()
        if not key.startswith("_")
    }
    return (node.data.layerType.id, tuple(sorted(semantic.items())))


def _nodes(graph) -> Counter:
    return Counter(_node_signature(node) for node in graph.nodes)


def _edges(graph) -> set:
    name = CodeGenerationService._node_id_to_layer_name
    return {(name(edge.source), name(edge.target)) for edge in graph.edges}


def _top_level(result: Dict[str, Any]):
    """The class the file is about — the one no other class instantiates. The
    same choice the frontend makes (apiService.parseCode)."""
    assert result.get("success"), result
    usable = [entry for entry in result["graphs"] if entry.get("graph") is not None]
    assert usable, result
    chosen = next((entry for entry in usable if entry.get("is_top_level")), usable[0])
    return chosen["graph"]


def _parse(code: str, input_shape: List[int]):
    return _top_level(
        _PARSER.parse_pytorch_code(
            CodeParseRequest(code=code, input_shape=input_shape, file="roundtrip.py")
        )
    )


def _refusal_reason(graph) -> str:
    ops = {node.data.layerType.id for node in graph.nodes}
    if "python" in ops:
        return OPAQUE_STATEMENT
    if "custom_code" in ops:
        return CUSTOM_MODULE
    return "unexplained"


def _roundtrip(code: str, input_shape: List[int]) -> Tuple[str, Any]:
    """(outcome, detail). outcome is 'ok' with the two graphs, or the gap reason
    with a human-readable detail."""
    first = _parse(code, input_shape)
    try:
        regenerated = _GENERATOR.generate_flat_code(first, input_shape, CLASS_NAME)
    except UnsafeGraphInput as exc:
        return _refusal_reason(first), str(exc)
    second = _parse(regenerated, input_shape)
    return "ok", (first, second)


def _assert_same_model(first, second) -> None:
    missing = _nodes(first) - _nodes(second)
    added = _nodes(second) - _nodes(first)
    assert not missing and not added, (
        f"regenerating changed the ops: lost {list(missing)}, gained {list(added)}"
    )
    assert _edges(first) == _edges(second), (
        f"regenerating changed the wiring: "
        f"lost {sorted(_edges(first) - _edges(second))}, "
        f"gained {sorted(_edges(second) - _edges(first))}"
    )


# ── fixtures ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_hand_written_fixture_survives_parse_generate_parse(name):
    code, input_shape = FIXTURES[name]
    outcome, detail = _roundtrip(code, input_shape)
    assert outcome == "ok", f"{name}: {outcome} — {detail}"
    _assert_same_model(*detail)


def test_a_fixture_actually_exercises_a_merge_node():
    """Guard against the fixtures degenerating into six straight chains: the
    dataflow half of M1 is the part a round-trip can silently stop covering."""
    code, input_shape = FIXTURES["residual_merge"]
    graph = _parse(code, input_shape)
    assert any(node.data.layerType.id == "add" for node in graph.nodes)

    code, input_shape = FIXTURES["two_branches_concatenated"]
    graph = _parse(code, input_shape)
    assert any(node.data.layerType.id == "cat" for node in graph.nodes)


# ── corpus ───────────────────────────────────────────────────────────────────
def _seed_code(template_id: str) -> Tuple[str, List[int]]:
    """Generated source for one canonical template — the corpus's starting point.
    Raises UnsafeGraphInput when the template cannot be written as code at all."""
    graph, _ = _build_graph_for_sequence(
        _REGISTRY, _VALIDATOR, [template_id], 0, enable_adapters=False
    )
    input_shape = [
        int(dim)
        for node in graph.nodes
        if node.kind == "input"
        for dim in node.params.get("shape", [])
    ]
    flat = Graph(
        nodes=[_arch_node_to_flat(node) for node in graph.nodes],
        edges=[
            {"id": edge.edge_id, "source": edge.source_node_id, "target": edge.target_node_id}
            for edge in graph.edges
        ],
    )
    return _GENERATOR.generate_flat_code(flat, input_shape, CLASS_NAME), input_shape


@pytest.mark.parametrize("template_id", CORPUS)
def test_corpus_template_survives_parse_generate_parse(template_id):
    expected_gap = KNOWN_ROUNDTRIP_GAPS.get(template_id)
    try:
        code, input_shape = _seed_code(template_id)
    except UnsafeGraphInput as exc:
        assert expected_gap == SEED_UNGENERATABLE, (
            f"{template_id}: no code to round-trip and no gap recorded — {exc}"
        )
        return

    assert expected_gap != SEED_UNGENERATABLE, (
        f"{template_id} is recorded as {SEED_UNGENERATABLE} but generated fine"
    )

    outcome, detail = _roundtrip(code, input_shape)
    if outcome == "ok":
        assert expected_gap is None, (
            f"{template_id} is recorded as a {expected_gap} gap but round-trips — "
            "remove the stale key"
        )
        _assert_same_model(*detail)
        return

    assert expected_gap == outcome, (
        f"{template_id}: round-trip stopped with {outcome!r} "
        f"(recorded: {expected_gap!r}) — {detail}"
    )


# ── the refusal, which is the other half of the promise ─────────────────────
def _placeholder_graph(op: str, parameters: Dict[str, Any]):
    """A one-node graph holding a placeholder, built rather than parsed.

    Parsing a fixture would decide for us which provenance keys the node carries,
    and the whole question here is which keys make a node writable. Building it
    states the case exactly."""
    from models.graph_models import Graph

    return Graph(
        **{
            "nodes": [
                {
                    "id": "i",
                    "type": "inputNode",
                    "position": {"x": 0.0, "y": 0.0},
                    "data": {
                        "layerType": {"id": "input", "name": "Input", "parameters": []},
                        "parameters": {"shape": "2,4,4"},
                    },
                },
                {
                    "id": "n",
                    "type": "layerNode",
                    "position": {"x": 0.0, "y": 0.0},
                    "data": {
                        "layerType": {"id": op, "name": op, "parameters": []},
                        "parameters": parameters,
                    },
                },
            ],
            "edges": [{"id": "e1", "source": "i", "target": "n"}],
        }
    )


def test_generation_refuses_a_placeholder_it_cannot_write_and_names_its_line():
    """A graph the generator cannot write must not be written badly, and the
    refusal points at `file:line` — the only useful thing to say about code that
    lives in somebody's file. server_v2 maps UnsafeGraphInput to HTTP 400, so this
    is what /api/models/generate-code answers.

    **M6 moved this line, and the move is the point.** Until M6 an opaque
    statement was the example of something unwritable; now it carries its own
    source text and is regenerated from it, so the refusal is no longer about the
    KIND of node. It is about whether the node carries the one thing that makes it
    writable: a module to import the class from, or the statement's own text.
    Neither of the two below carries either.
    """
    for op, parameters in (
        # The user's class, with no module to import it from. Reimplementing it
        # from its name would be a different model wearing the same word.
        ("custom_code", {"_class": "MyBlock", "_source_file": "models/net.py", "_lineno": 8}),
        # A statement whose text was never recorded. There is nothing to write.
        ("python", {"_source_file": "models/net.py", "_lineno": 8}),
    ):
        with pytest.raises(UnsafeGraphInput) as excinfo:
            _GENERATOR.generate_flat_code(
                _placeholder_graph(op, parameters), [2, 4, 4], CLASS_NAME
            )
        assert "models/net.py:8" in str(excinfo.value), op


def test_the_same_node_is_written_once_it_says_where_its_class_lives():
    """The other side of the refusal above, in the same test file, because a
    refusal nobody can satisfy is indistinguishable from a feature that does not
    work. Given `_import_path`, the generated file imports the real class and
    calls the real constructor — it never defines a copy of it."""
    code = _GENERATOR.generate_flat_code(
        _placeholder_graph(
            "custom_code",
            {
                "_class": "MyBlock",
                "_import_path": "models.blocks",
                "_raw_call": "MyBlock(16)",
                "_source_file": "models/net.py",
                "_lineno": 8,
            },
        ),
        [2, 4, 4],
        CLASS_NAME,
    )
    assert "from models.blocks import MyBlock" in code
    assert "self.n = MyBlock(16)" in code
    # The argument matters: MyBlock(16) and MyBlock() are different models, and a
    # generator that dropped it would still produce a file that imports and runs.
    assert "class MyBlock" not in code


def test_every_recorded_gap_names_a_template_in_the_corpus():
    """A gap keyed on a template the corpus no longer walks documents nothing."""
    stale = set(KNOWN_ROUNDTRIP_GAPS) - set(CORPUS)
    assert not stale, f"KNOWN_ROUNDTRIP_GAPS references templates not in the corpus: {stale}"
