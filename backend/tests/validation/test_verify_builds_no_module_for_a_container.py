"""`/api/models/verify` BUILDS the model; a container is not a thing it can build.

Same misconception as `test_verify_builds_no_module_for_a_source.py`, one category over.
Open `examples/tensorviz-sweep/models/vit_demo.py` (`TinyViT`) on the in-editor canvas and
the card standing for `self.blocks` came back with a red panel:

    ModuleList — build error: Invalid parameters for ModuleList: dict is not a Module subclass

and `examples/tensorviz-test/models/tiny_transformer.py` with

    TransformerEncoder — build error: missing 1 required positional argument: 'encoder_layer'

Both models are correct. An `nn` container HOLDS submodules rather than computing anything,
so its constructor argument is other `nn.Module`s — and a graph payload is JSON. It can
carry what the members ARE (the previous commit moved that to `_modules`, display data the
card reads to say "6 × Block"), but never the objects themselves; `nn.TransformerEncoder`'s
`encoder_layer` is a local variable in the user's `__init__` and reaches the payload not at
all. So a container is not *sometimes* unbuildable here: it is structurally unbuildable,
always, and reporting that as a fault in the user's model is the product blaming them for
its own reading.

Leaving them out of `leaf_layers` drops them into the forward's identity/`_out_shape`
branch — where `custom_code` and the merge ops already land — which is what makes the whole
CLASS of failure go away rather than one key. The browser page had been escaping it by
accident: its cross-file expander re-types an unresolvable `nn.ModuleList` to `custom_code`
(`services/ide/crossfile.py::_mark_opaque`) and `verify_model` skips placeholder ops, so the
same model verified clean there and failed on the canvas.

The ModuleList semantic-kind regression is covered by
`test_code_to_graph_projection.py`; an edge-free list is excluded before build.
"""
from __future__ import annotations


import pytest


from models.graph_models import CodeParseRequest  # noqa: E402
from services.ide import projection  # noqa: E402
from utils.code_parser import CodeParserService  # noqa: E402

# `TinyViT`'s container and the loop that walks it, reduced to what the bug needs.
MODULE_LIST = '''
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class TinyViT(nn.Module):
    def __init__(self, dim=8, depth=6):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.l1(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)
'''

# `TinyTransformer`'s container: the submodule it needs is a LOCAL, so no amount of
# care with parameter KEYS reaches it. This is the case that needs the skip.
TRANSFORMER_ENCODER = '''
import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(self, d_model=8, n_heads=2, n_layers=2):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x):
        h = self.l1(x)
        h = self.encoder(h)
        return self.head(h)
'''


def _project(source: str, cls: str):
    """The static tier exactly as `POST /api/ide/parse` runs it."""
    parsed = CodeParserService().parse_pytorch_code(
        CodeParseRequest(code=source, file="m.py")
    )
    entry = next(e for e in parsed["graphs"] if e["name"] == cls)
    graph = entry["graph"]
    return projection.project(
        {
            "nodes": [n.model_dump() for n in graph.nodes],
            "edges": [e.model_dump() for e in graph.edges],
        },
        cls=cls,
        rel_file="m.py",
        source=source,
    )


def _build_errors(verification) -> list:
    return [
        e for e in (verification.get("errors") or [])
        if "build error" in str(e.get("message", ""))
    ]


def test_the_container_set_is_not_a_second_spelling_of_an_existing_one():
    """These are containers, not merge ops, not custom passthroughs, not sources —
    four sets that already exist and each mean something different. And none of them
    is a registered op, which is why the leaf loop is the only place to say so."""
    from core.op_registry import spec_for
    from services.verification_service import (
        _CONTAINER_OP_KEYS,
        _CUSTOM_PASSTHROUGH_OPS,
        _MERGE_OP_KEYS,
    )

    assert _CONTAINER_OP_KEYS & (_MERGE_OP_KEYS | _CUSTOM_PASSTHROUGH_OPS) == set()
    assert [key for key in _CONTAINER_OP_KEYS if spec_for(key) is not None] == []
    # `nn.Transformer` takes scalars, builds from them and computes. It is a layer.
    assert "transformer" not in _CONTAINER_OP_KEYS


@pytest.mark.parametrize(
    "source, cls, shape",
    [
        pytest.param(MODULE_LIST, "TinyViT", [1, 8], id="ModuleList"),
        pytest.param(TRANSFORMER_ENCODER, "TinyTransformer", [1, 4, 8], id="TransformerEncoder"),
    ],
)
def test_a_container_the_static_tier_drew_is_not_reported_as_a_build_error(source, cls, shape):
    """The user-visible bug, end to end: parse the file the way the canvas does, hand
    the result to the verify path the way the canvas does, and read the panel."""
    from services.verification_service import (
        architecture_graph_to_runtime_payload,
        verify_model,
    )

    payload = architecture_graph_to_runtime_payload(_project(source, cls).document.semantic)
    errors = _build_errors(verify_model(payload, shape, run_forward=True))
    assert errors == [], errors


def test_a_layer_that_really_is_misconfigured_is_still_reported():
    """The skip must not become a blanket amnesty: `build error` has to keep meaning
    something, or this trades a false alarm for a silent one."""
    from services.verification_service import verify_model

    graph = {
        "nodes": [
            {"id": "input", "type": "inputNode", "position": {"x": 0, "y": 0},
             "data": {"layerType": {"id": "input", "name": "Input", "parameters": []},
                      "parameters": {"shape": "1,8"}}},
            {"id": "l1", "type": "layerNode", "position": {"x": 1, "y": 0},
             "data": {"layerType": {"id": "linear", "name": "Linear", "parameters": []},
                      "parameters": {"in_features": 8, "out_features": -3}}},
            {"id": "output", "type": "outputNode", "position": {"x": 2, "y": 0},
             "data": {"layerType": {"id": "output", "name": "Output", "parameters": []},
                      "parameters": {}}},
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "l1"},
            {"id": "e2", "source": "l1", "target": "output"},
        ],
    }
    assert _build_errors(verify_model(graph, [1, 8], run_forward=True)) != []
