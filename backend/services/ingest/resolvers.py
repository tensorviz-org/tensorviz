"""Library adapters: an architecture name -> an instantiated ``nn.Module``, offline.

Instantiation is effectively free because every model is built under
``FakeTensorMode``/meta: a Llama-7B-shaped model (6.7B parameters) materialises in 6.9 ms
using 0 bytes of storage, and Stable Diffusion 3's 2B-parameter MMDiT in 57 ms. Nothing
here downloads weights, touches the network, or needs a GPU — architectures come from the
libraries' own config classes and registries.

``torchvision`` ships with the backend requirements and needs no extra install, so it is
the zero-configuration default (121 architectures). ``transformers``, ``diffusers`` and
``timm`` are optional extras and are imported lazily; a missing one degrades to "that
library is unavailable" rather than breaking the server.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .spec import CatalogEntry

LIBRARIES = ("torchvision", "transformers", "diffusers", "timm")


class MissingLibraryError(RuntimeError):
    """An optional extra this architecture needs is not installed in this deployment.

    Distinguished from a genuine ingestion failure so the API can answer 422 with the
    install command instead of a 500 whose detail ("RuntimeError: transformers is not
    installed (optional extra)") tells the caller nothing they can act on.
    """

    def __init__(self, library: str) -> None:
        self.library = library
        super().__init__(
            f"{library} is not installed (optional extra) — "
            f"install and retry: pip install {library}"
        )


def _module(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def available_libraries() -> Dict[str, bool]:
    return {
        "torchvision": _module("torchvision") is not None,
        "transformers": _module("transformers") is not None,
        "diffusers": _module("diffusers") is not None,
        "timm": _module("timm") is not None,
    }


# ─────────────────────────────────────────────────────────────── catalog ─────────
def catalog(library: Optional[str] = None) -> List[CatalogEntry]:
    """Every architecture we can instantiate, per library. Cheap: registry reads only."""
    out: List[CatalogEntry] = []
    want = (library,) if library else LIBRARIES

    if "torchvision" in want:
        tv = _module("torchvision.models")
        if tv is not None:
            try:
                for arch in tv.list_models():
                    out.append(CatalogEntry(library="torchvision", arch=arch, kind="registry"))
            except Exception:
                pass

    if "timm" in want:
        timm = _module("timm")
        if timm is not None:
            try:
                for arch in timm.list_models():
                    out.append(CatalogEntry(library="timm", arch=arch, kind="registry"))
            except Exception:
                pass

    if "transformers" in want:
        tf = _module("transformers")
        if tf is not None:
            try:
                from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

                for arch in sorted(CONFIG_MAPPING_NAMES):
                    out.append(CatalogEntry(library="transformers", arch=arch, kind="config"))
            except Exception:
                pass

    if "diffusers" in want:
        df = _module("diffusers")
        if df is not None:
            for arch in _DIFFUSERS_MODELS:
                if hasattr(df, arch):
                    out.append(CatalogEntry(library="diffusers", arch=arch, kind="config"))

    return out


_DIFFUSERS_MODELS = (
    "UNet2DModel", "UNet2DConditionModel", "AutoencoderKL", "AutoencoderTiny",
    "SD3Transformer2DModel", "FluxTransformer2DModel", "DiTTransformer2DModel",
    "Transformer2DModel", "ControlNetModel", "PriorTransformer",
)


# ───────────────────────────────────────────────────────────── resolution ────────
def resolve(library: str, arch: str, overrides: Optional[Dict[str, Any]] = None) -> nn.Module:
    """Instantiate ``arch`` from ``library``. Caller supplies the fake/meta context."""
    overrides = dict(overrides or {})
    if library == "torchvision":
        return _resolve_torchvision(arch, overrides)
    if library == "timm":
        return _resolve_timm(arch, overrides)
    if library == "transformers":
        return _resolve_transformers(arch, overrides)
    if library == "diffusers":
        return _resolve_diffusers(arch, overrides)
    raise ValueError(f"unknown library {library!r}; expected one of {LIBRARIES}")


def _resolve_torchvision(arch: str, overrides: Dict[str, Any]) -> nn.Module:
    tv = _module("torchvision.models")
    if tv is None:
        raise RuntimeError("torchvision is not installed")
    # weights=None keeps it architecture-only: no download, no checkpoint.
    return tv.get_model(arch, weights=None, **overrides)


def _resolve_timm(arch: str, overrides: Dict[str, Any]) -> nn.Module:
    timm = _module("timm")
    if timm is None:
        raise MissingLibraryError("timm")
    overrides.setdefault("pretrained", False)
    return timm.create_model(arch, **overrides)


def _resolve_transformers(arch: str, overrides: Dict[str, Any]) -> nn.Module:
    """Build from the CONFIG class only — no hub access, no weights.

    ``AutoModel.from_config`` covers the common case; where an architecture has no
    auto-model mapping we fall back to its own ``*Model`` class.
    """
    tf = _module("transformers")
    if tf is None:
        raise MissingLibraryError("transformers")
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if arch not in CONFIG_MAPPING:
        raise ValueError(f"unknown transformers architecture {arch!r}")
    config = CONFIG_MAPPING[arch](**overrides)
    try:
        return tf.AutoModel.from_config(config)
    except Exception:
        for attr in (f"{arch.title().replace('_', '')}Model", f"{arch.upper()}Model"):
            cls = getattr(tf, attr, None)
            if cls is not None:
                return cls(config)
        raise


def _resolve_diffusers(arch: str, overrides: Dict[str, Any]) -> nn.Module:
    df = _module("diffusers")
    if df is None:
        raise MissingLibraryError("diffusers")
    cls = getattr(df, arch, None)
    if cls is None:
        raise ValueError(f"unknown diffusers model {arch!r}")
    preset = dict(_DIFFUSERS_PRESETS.get(arch, {}))
    preset.update(overrides)
    return cls(**preset)


# Small, explicit presets so the common diffusers models come out at a recognisable size
# instead of the library's tiny test defaults (a bare AutoencoderKL() is 0.6M parameters).
_DIFFUSERS_PRESETS: Dict[str, Dict[str, Any]] = {
    "UNet2DConditionModel": {
        "sample_size": 64,
        "in_channels": 4,
        "out_channels": 4,
        "block_out_channels": (320, 640, 1280, 1280),
        "down_block_types": ("CrossAttnDownBlock2D",) * 3 + ("DownBlock2D",),
        "up_block_types": ("UpBlock2D",) + ("CrossAttnUpBlock2D",) * 3,
        "cross_attention_dim": 768,
        "layers_per_block": 2,
    },
    "AutoencoderKL": {
        "in_channels": 3,
        "out_channels": 3,
        "block_out_channels": (128, 256, 512, 512),
        "down_block_types": ("DownEncoderBlock2D",) * 4,
        "up_block_types": ("UpDecoderBlock2D",) * 4,
        "latent_channels": 4,
        "layers_per_block": 2,
    },
    "SD3Transformer2DModel": {
        "num_layers": 24,
        "attention_head_dim": 64,
        "num_attention_heads": 24,
        "caption_projection_dim": 1536,
        "joint_attention_dim": 4096,
        "pooled_projection_dim": 2048,
    },
}
