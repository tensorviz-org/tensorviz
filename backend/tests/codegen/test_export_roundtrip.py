"""Tier-5 export round-trip parity (docs/tensorviz-plan.md §5.4, §9 Tier 5, §11.6).

The moat's final promise: *exported code runs and reproduces the verified shapes*.
For a template we (1) instantiate it into a graph, (2) run ``verify_model`` to get the
authoritative output shape, (3) generate + package the code exactly as the product's
``/api/models/export-package`` endpoint does, (4) exec the generated ``model.py``,
instantiate the model, forward a dummy input, and (5) assert the executed output shape
equals the verified output shape.

Coverage is the 6 headline gallery models (ViT, BERT, GPT-2, T5, ResNet, GAN) plus a
deterministic stride-sample of the canonical registry (sorted ids, fixed stride — no
randomness, so the set is reproducible run to run). Only templates offered by the
product are round-trip candidates. Archived unsupported templates must be excluded
from the catalogue and refuse export explicitly.

Run: KMP_DUPLICATE_LIB_OK=TRUE <viz-python> -m pytest tests/test_export_roundtrip.py -q
"""

import io
import os
import zipfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # must precede torch import
os.environ.setdefault("AUTH_DISABLED", "true")



import pytest
import torch

from services.template_service import TemplateService
from services.verification_service import verify_model, export_package, _arch_node_to_flat
from services.canonical_template_registry import CanonicalTemplateRegistry
from services.canonical_validation import CanonicalValidationService
from scripts.run_template_validation_sweep import _build_graph_for_sequence

MODEL_NAME = "RoundtripModel"

# The 6 headline gallery models (docs/tensorviz-plan.md §5.4). Instantiated via TemplateService as
# full hierarchical graphs, using their declared input shapes.
GALLERY = ["vit_base", "bert_base", "gpt2_small", "t5_base", "resnet50", "biggan"]

_REGISTRY = CanonicalTemplateRegistry()
_VALIDATOR = CanonicalValidationService()


def _registry_sample(n_target: int = 12) -> list:
    """Deterministic stride sample of the canonical registry: sort every contract id,
    take every ``step``-th (step derived from a target count). No randomness — the same
    ids come out every run."""
    ids = sorted(c.template_id for c in _REGISTRY.list_contracts())
    step = max(1, len(ids) // n_target)
    return ids[::step]


SAMPLE = _registry_sample()



def _cap(shape):
    """Bound spatial / sequence dims so registry-sample forwards stay small & fast.
    The same capped shape is used for BOTH verify and exec, so parity is preserved."""
    s = [int(v) for v in shape]
    if len(s) == 4:
        s[2], s[3] = min(s[2], 32), min(s[3], 32)
    if len(s) == 3:
        s[1] = min(s[1], 32)
    return s


def _pins_absolute_spatial_size(nodes) -> bool:
    """True when a node PINS an absolute spatial target (nn.Upsample `size=[H, W]`).

    A template's parameters are resolved at INSTANTIATE time against the input its
    contract assumes — 224x224 for a rank-4 block, see
    CanonicalTemplateRegistry._default_shape_from_expectation — and `instantiate`
    is never told the real graph input. Capping the input afterwards is shape-neutral
    for channel/feature params, but not for a pinned size: that branch still resizes
    to its literal H,W while every sibling branch shrinks, so the concat downstream
    sees ranks the template never described. Those templates run at their declared
    shape instead (still the same shape for verify and exec, so parity holds)."""
    for n in nodes:
        size = (getattr(n, "params", None) or {}).get("size")
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            return True
    return False


def _gallery_graph(model_id: str):
    """(graph_payload, input_shape) for a headline gallery model."""
    hg = TemplateService().instantiate_popular_model(model_id, {})
    input_shape = None
    for n in hg.root_nodes:
        if n.type == "inputNode":
            raw = (n.data.parameters or {}).get("shape")
            if isinstance(raw, str) and raw.strip():
                input_shape = [int(v) for v in raw.split(",")]
            elif isinstance(raw, (list, tuple)) and raw:
                input_shape = [int(v) for v in raw]
    return hg.model_dump(), input_shape


def _registry_graph(template_id: str):
    """(graph_payload, input_shape) for a canonical registry template: wrap it in the
    product's own single-input->template->output graph, flattened to the canvas node
    shape verify_model/export consume."""
    graph, _ = _build_graph_for_sequence(_REGISTRY, _VALIDATOR, [template_id], 0, enable_adapters=False)
    pinned = _pins_absolute_spatial_size(graph.nodes)
    input_shape = None
    for n in graph.nodes:
        if n.kind == "input":
            declared = [int(v) for v in n.params.get("shape", [])]
            input_shape = declared if pinned else _cap(declared)
            n.params["shape"] = input_shape
    nodes = [_arch_node_to_flat(n) for n in graph.nodes]
    edges = [{"id": e.edge_id, "source": e.source_node_id, "target": e.target_node_id} for e in graph.edges]
    return {"nodes": nodes, "edges": edges}, input_shape


def _build_dummy(shape, dtype: str):
    safe = [max(1, int(d)) for d in shape] or [1]
    if dtype == "long":
        return torch.randint(0, 100, tuple(safe))
    return torch.randn(*safe)


def _exec_forward(model_py: str, input_shape):
    """exec the generated model.py, instantiate the model, forward a dummy input, and
    return the output tensor's shape. Tries float, retrying with long int indices when
    the model leads with an Embedding (matches verify_model's dtype convention)."""
    ns: dict = {}
    exec(compile(model_py, "<generated model.py>", "exec"), ns)  # noqa: S102 - code under test
    model = ns[MODEL_NAME]()
    model.eval()
    last_err = None
    for dtype in ("float", "long"):
        x = _build_dummy(input_shape, dtype)
        try:
            with torch.no_grad():
                out = model(x)
            break
        except RuntimeError as e:
            last_err = e
            if "indices" in str(e) and dtype == "float":
                continue  # Embedding needs Long — retry
            raise
    else:  # pragma: no cover - defensive
        raise last_err
    if not torch.is_tensor(out):
        raise AssertionError(f"exported model returned {type(out).__name__}, not a single tensor")
    return list(out.shape)


def roundtrip(kind: str, template_id: str):
    """instantiate -> verify_model (verified shape) -> generate+package code -> exec ->
    forward dummy input (executed shape). Returns (verified_shape, executed_shape).
    Raises on every verification, generation or execution failure."""
    payload, input_shape = _gallery_graph(template_id) if kind == "gallery" else _registry_graph(template_id)
    assert input_shape, f"{template_id}: could not determine an input shape"

    result = verify_model(payload, input_shape)
    assert result.get("ok"), f"{template_id}: verify_model failed at stage {result.get('stage')}: {result.get('errors')}"
    verified_shape = result.get("output_shape")

    zip_bytes, _ = export_package(payload, input_shape, MODEL_NAME)
    model_py = zipfile.ZipFile(io.BytesIO(zip_bytes)).read(f"{MODEL_NAME}/model.py").decode("utf-8")
    executed_shape = _exec_forward(model_py, input_shape)
    return verified_shape, executed_shape


def _params():
    return [pytest.param(kind, template_id, id=template_id)
            for kind, ids in (("gallery", GALLERY), ("registry", SAMPLE)) for template_id in ids]


@pytest.mark.parametrize("kind,template_id", _params())
def test_export_roundtrip(kind, template_id):
    verified_shape, executed_shape = roundtrip(kind, template_id)
    assert executed_shape == verified_shape, (
        f"{template_id}: exported code output {executed_shape} != verified {verified_shape}"
    )


def test_export_coverage():
    for kind, ids in (("gallery", GALLERY), ("registry", SAMPLE)):
        for template_id in ids:
            test_export_roundtrip(kind, template_id)


def test_archived_graph_network_refuses_export_instead_of_emitting_nonexistent_layers():
    from core.graph_input_safety import UnsafeGraphInput
    template_id = "gnn_mean_1hop_block"
    assert template_id in {contract.template_id for contract in _REGISTRY.all_contracts()}
    assert template_id not in {contract.template_id for contract in _REGISTRY.list_contracts()}
    payload, input_shape = _registry_graph(template_id)
    assert verify_model(payload, input_shape)["ok"] is False
    with pytest.raises(UnsafeGraphInput, match="neighbor_sampler.*not a supported operation"):
        export_package(payload, input_shape, MODEL_NAME)
