"""Guardrail: every frontend block must resolve to an OpSpec.

This is the Step B1 gate from the validation plan: no palette block can ship
without registry coverage in the control/data validation stack.
"""

from __future__ import annotations

import re
from pathlib import Path

from core.op_registry import spec_for


def _layer_type_ids() -> list[str]:
    layer_types_path = (
        Path(__file__).parent.resolve().parents[2]
        / "canvas"
        / "src"
        / "data"
        / "layerTypes.ts"
    )
    text = layer_types_path.read_text(encoding="utf-8")
    ids = re.findall(r"id:\s*'([^']+)'", text)
    # Drop non-op ids.
    excluded = {
        "input",
        "output",
        "module",
        "io",
        "conv",
        "linear",
        "activation",
        "pool",
        "norm",
        "recurrent",
        "embedding",
        "dropout",
        "tensor",
        "loss",
        "misc",
    }
    return [layer_id for layer_id in ids if layer_id not in excluded]


def test_frontend_palette_ids_resolve_to_registry_specs():
    missing = [layer_id for layer_id in _layer_type_ids() if spec_for(layer_id) is None]
    assert not missing, f"Missing OpSpec coverage for frontend blocks: {missing}"

