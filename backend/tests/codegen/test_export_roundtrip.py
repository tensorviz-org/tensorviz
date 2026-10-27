"""Tier-5 export round-trip parity (docs/tensorviz-plan.md §5.4, §9 Tier 5, §11.6).

The moat's final promise: *exported code runs and reproduces the verified shapes*.
For a template we (1) instantiate it into a graph, (2) run ``verify_model`` to get the
authoritative output shape, (3) generate + package the code exactly as the product's
``/api/models/export-package`` endpoint does, (4) exec the generated ``model.py``,
instantiate the model, forward a dummy input, and (5) assert the executed output shape
equals the verified output shape.

Coverage is the 6 headline gallery models (ViT, BERT, GPT-2, T5, ResNet, GAN) plus a
deterministic stride-sample of the canonical registry (sorted ids, fixed stride — no
randomness, so the set is reproducible run to run). Templates that don't cleanly
round-trip yet are documented in KNOWN_EXPORT_GAPS with a one-line reason and marked
xfail(strict=False); the suite is GREEN today and each gap auto-flips to XPASS the
moment its codegen/verify gap is closed. Nothing is silently skipped.

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
    ids come out every run so KNOWN_EXPORT_GAPS stays stable."""
    ids = sorted(c.template_id for c in _REGISTRY.all_contracts())
    step = max(1, len(ids) // n_target)
    return ids[::step]


SAMPLE = _registry_sample()


# ── Known export-parity gaps. xfail(strict=False) so each auto-flips to XPASS when its
# underlying codegen/verify gap is closed. Reasons distinguish the cause honestly:
# codegen emits a nonexistent nn class, verify can't build a custom op, an exec error, or
# an analytic/torch shape drift. "The block is inherently multi-input" was a fifth cause
# until 2026-08-26; it turned out to describe nothing — see the FPN note below. ────────
KNOWN_EXPORT_GAPS = {
    # --- registry: verify_model cannot produce a shape (not an export-parity candidate yet) ---
    "gnn_mean_1hop_block": "verify build fails: message passing needs a connectivity input "
                           "the canvas cannot carry, so neighbor_sampler / message_mlp / "
                           "mean_aggregate have no faithful torch.nn form",
    # CLOSED: "aspp_block" — "verify forward fails: parallel dilated branches yield
    # mismatched sizes before concat". Not a template bug: every dilated branch pads to
    # preserve size and the block round-trips at its declared 224x224. `_cap` below was
    # shrinking the graph input AFTER instantiate had resolved image_up's
    # `size=[224, 224]` against the contract's assumed shape, so only that branch stayed
    # at 224. See _pins_absolute_spatial_size.
    # CLOSED: "detr_fpn_l4_head_block" / "fcos_fpn_l6_head_block" / "yolo_fpn_l4_head_block"
    # — "verify arity error: FPN head Conv2d fed N pyramid inputs (multi-input block)".
    # Not inherently multi-input: the block has ONE external input and the fan-in was
    # internal, which merge ops already express. The template simply had no merge — it
    # wired all L pyramid levels straight into a Conv2d. template_expansion_pack now
    # concatenates them on the channel axis first. Folding then had to be corrected too:
    # graph_modularizer collapsed the L identical level convs into one ModuleList node
    # and deduped their L edges into the concat down to one.
    # CLOSED: "latent_diffusion_block" — "verify build fails: custom op 'gaussian_noise'
    # has no torch.nn class". It needs no class: x + sigma*eps is shape-preserving, so
    # the dry forward passes it through and code_generation emits a real GaussianNoise
    # module, the same split the GRN/RoPE/adaLN ops already use.
    # CLOSED: "tfm_enc_rmsnorm_silu_h8_block" — "verify build fails: custom op 'rmsnorm'
    # has no torch.nn class". It was never a custom op: nn.RMSNorm is real (torch >= 2.4)
    # and the templates emit the lowercase id "rmsnorm", which build_layer could not match
    # to a class name. verification_service._LAYER_ID_TO_CLASS now maps it like every other
    # palette id.
    # CLOSED: "unet_w64_d4_block" — "exported UNet conv in_channels don't track the
    # skip-concat widening (channel mismatch on forward)". The cause was
    # graph_modularizer folding on op type alone, so conv spans that differed only in
    # in_channels collapsed onto the first span's params. Repeat detection is now
    # params-aware (services/graph_modularizer.py `_sig`) and this passes, so the
    # entry is removed rather than left to rot as a permanent XPASS.
}


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
    Raises on any failure so a gap template trips its xfail."""
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
    out = []
    for kind, ids in (("gallery", GALLERY), ("registry", SAMPLE)):
        for tid in ids:
            marks = []
            if tid in KNOWN_EXPORT_GAPS:
                marks.append(pytest.mark.xfail(strict=False, reason=KNOWN_EXPORT_GAPS[tid]))
            out.append(pytest.param(kind, tid, id=tid, marks=marks))
    return out


@pytest.mark.parametrize("kind,template_id", _params())
def test_export_roundtrip(kind, template_id):
    verified_shape, executed_shape = roundtrip(kind, template_id)
    assert executed_shape == verified_shape, (
        f"{template_id}: exported code output {executed_shape} != verified {verified_shape}"
    )


def test_export_coverage(capsys):
    """Independent walk of the selected templates: prints N/M round-trip and is the real
    regression gate — fails if any NON-gap template stops round-tripping."""
    selected = [("gallery", t) for t in GALLERY] + [("registry", t) for t in SAMPLE]
    total = len(selected)
    passed = 0
    unexpected_fail, unexpected_pass = [], []
    for kind, tid in selected:
        try:
            verified, executed = roundtrip(kind, tid)
            ok = executed == verified
        except Exception:
            ok = False
        passed += int(ok)
        if not ok and tid not in KNOWN_EXPORT_GAPS:
            unexpected_fail.append(tid)
        if ok and tid in KNOWN_EXPORT_GAPS:
            unexpected_pass.append(tid)

    with capsys.disabled():
        print(f"\n[Tier-5 export round-trip] {passed}/{total} templates round-trip "
              f"({len(KNOWN_EXPORT_GAPS)} known gaps documented)")
        if unexpected_pass:
            print("FIXED (prune from KNOWN_EXPORT_GAPS): " + ", ".join(sorted(unexpected_pass)))

    assert not unexpected_fail, f"Non-gap templates regressed (exported shape != verified): {unexpected_fail}"
    selected_ids = {t for _, t in selected}
    stale = set(KNOWN_EXPORT_GAPS) - selected_ids
    assert not stale, f"KNOWN_EXPORT_GAPS references templates not in the selected set: {stale}"
