"""Finding an example input for a model whose forward signature is unknown.

For an arbitrary repository nobody tells us the shapes ``forward`` expects. Guessing from
names is unreliable; *searching* is not, because a forward pass under ``FakeTensorMode``
costs 10-300 ms and allocates nothing. So we enumerate a small ordered candidate space and
take the first tuple that completes a forward — a candidate that runs IS by construction a
real execution of that model.

Measured on real repositories: vit-pytorch found ``(1,3,224,224)`` on candidate 1,
nanoGPT ``(1,16)`` tokens on candidate 6, ddpm's UNet ``(1,3,224,224),(1,)`` on candidate 1,
and DiT's three-argument ``(latent, timestep, class_label)`` signature on candidate 41 with
no prior knowledge of it.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _image(channels: int, size: int) -> Callable[[], torch.Tensor]:
    return lambda: torch.randn(1, channels, size, size)


def _tokens(length: int, vocab: int = 1000) -> Callable[[], torch.Tensor]:
    return lambda: torch.randint(0, vocab, (1, length))


def _scalar_float() -> torch.Tensor:
    return torch.tensor([1.0])


def _scalar_int() -> torch.Tensor:
    return torch.tensor([1])


# Ordered cheapest/likeliest first. Position 0 is the primary activation, positions 1+ are
# conditioning arguments (a diffusion timestep, a class label, an encoder state).
_POOLS: Tuple[Tuple[Callable[[], torch.Tensor], ...], ...] = (
    (
        _image(3, 224), _image(3, 64), _image(3, 32), _image(4, 32), _image(4, 64),
        _image(1, 28), _tokens(16), _tokens(8),
    ),
    (
        _scalar_float, _scalar_int, _tokens(16),
        lambda: torch.randn(1, 77, 768), _image(4, 32),
    ),
    (
        _scalar_int, _scalar_float, lambda: torch.randn(1, 77, 768), _tokens(16),
    ),
)

MAX_CANDIDATES = 80


def required_arity(model: nn.Module) -> Optional[int]:
    """Known required positional arity; None needs an explicit invocation.

    Variadic or unavailable signatures are unknown, never proof of zero inputs.
    Required keyword-only parameters cannot be satisfied by positional guessing.
    """
    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return None
    count = 0
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            return None
        if param.kind == param.KEYWORD_ONLY:
            if param.default is inspect._empty:
                return None
            continue
        if param.default is inspect._empty:
            count += 1
    return count


def seed_from_model(model: nn.Module) -> Optional[Callable[[], torch.Tensor]]:
    """The single likeliest input; `seeds_from_model` is the full ordered list."""
    seeds = seeds_from_model(model)
    return seeds[0] if seeds else None


def _seed_from_model_original(model: nn.Module) -> Optional[Callable[[], torch.Tensor]]:
    """An input read off the model's own first layer, rather than guessed.

    The search below is a good answer for a repository nobody can introspect, but it
    is the wrong first move for the commonest model there is. Its pool holds images
    and integer token tensors and no plain 2-D float activation at all, so a model
    whose first layer is ``nn.Linear(16, 16)`` — an MLP, the simplest thing anyone
    writes — can never be given ``randn(1, 16)``. Every candidate failed, the search
    fell through to the 1x3x224x224 image below, and the user was shown
    "mat1 and mat2 shapes cannot be multiplied (672x224 and 16x16)" for a model that
    is perfectly correct.

    The first layer that consumes the input already states what it takes, and the
    model is constructed by the time we are called, so this reads it instead of
    guessing. Returns None when the first layer says nothing useful, and the search
    then runs exactly as before.
    """
    for module in model.modules():
        if module is model or isinstance(
            module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)
        ):
            continue
        if isinstance(module, nn.Linear):
            width = int(module.in_features)
            return lambda: torch.randn(1, width)
        if isinstance(module, nn.Embedding):
            vocab = int(module.num_embeddings)
            return lambda: torch.randint(0, max(1, vocab), (1, 16))
        if isinstance(module, nn.Conv1d):
            channels = int(module.in_channels)
            return lambda: torch.randn(1, channels, 32)
        if isinstance(module, nn.Conv2d):
            channels = int(module.in_channels)
            return lambda: torch.randn(1, channels, 32, 32)
        if isinstance(module, nn.Conv3d):
            channels = int(module.in_channels)
            return lambda: torch.randn(1, channels, 8, 32, 32)
        # Anything else (activations, norms, dropout) says nothing about the input
        # width on its own, so keep looking rather than guessing from it.
    return None


#: Sequence lengths tried for a recurrent or embedding first layer, likeliest first.
#: The width is known exactly — the layer states it — but the length never is, and a
#: model that flattens the sequence (RNN -> Flatten -> Linear) only accepts the one
#: length its Linear was sized for. So the length is searched and the width is not.
_SEQ_LENGTHS: Tuple[int, ...] = (10, 16, 8, 32, 4, 64)


def seeds_from_model(model: nn.Module) -> List[Callable[[], torch.Tensor]]:
    """Every input the model's own first layer justifies, likeliest first.

    A list rather than one answer because sequence models need a length as well as a
    width: `nn.RNN(input_size=8)` says what each step carries and nothing about how
    many steps, and a downstream `Flatten -> Linear(160)` silently requires exactly
    ten of them. Trying a handful of lengths costs a few fake forwards and is the
    difference between an RNN tracing and not.
    """
    for module in model.modules():
        if module is model or isinstance(
            module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)
        ):
            continue
        if isinstance(module, nn.Linear):
            width = int(module.in_features)
            return [lambda: torch.randn(1, width)]
        if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
            width = int(module.input_size)
            batch_first = bool(getattr(module, "batch_first", False))
            def make(length: int, w: int = width, bf: bool = batch_first):
                return (lambda: torch.randn(1, length, w)) if bf else (lambda: torch.randn(length, 1, w))
            return [make(length) for length in _SEQ_LENGTHS]
        if isinstance(module, nn.Embedding):
            vocab = max(1, int(module.num_embeddings))
            def tokens(length: int, v: int = vocab):
                return lambda: torch.randint(0, v, (1, length))
            return [tokens(length) for length in _SEQ_LENGTHS]
        if isinstance(module, nn.Conv1d):
            channels = int(module.in_channels)
            return [lambda: torch.randn(1, channels, length) for length in (32,)]
        if isinstance(module, nn.Conv2d):
            channels = int(module.in_channels)
            return [lambda: torch.randn(1, channels, 32, 32)]
        if isinstance(module, nn.Conv3d):
            channels = int(module.in_channels)
            return [lambda: torch.randn(1, channels, 8, 32, 32)]
        # Anything else (activations, norms, dropout) says nothing about the input
        # width on its own, so keep looking rather than guessing from it.
    return []


def candidates_for(
    model: nn.Module, arity: int
) -> List[Tuple[Callable[[], torch.Tensor], ...]]:
    """The ordered search space, with the model's own answer tried first."""
    pool = candidates(arity)
    if arity != 1:
        # The seed describes the primary activation only. With two or more required
        # arguments the conditioning ones still have to be searched, and putting a
        # half-known tuple first would only cost a failed forward.
        return pool
    seeds = seeds_from_model(model)
    if not seeds:
        return pool
    return [(seed,) for seed in seeds] + pool


def candidates(arity: int) -> List[Tuple[Callable[[], torch.Tensor], ...]]:
    if arity <= 0:
        return [()]
    if arity == 1:
        return [(a,) for a in _POOLS[0]]
    if arity == 2:
        return [(a, b) for a in _POOLS[0] for b in _POOLS[1]]
    return [(a, b, c) for a in _POOLS[0] for b in _POOLS[1] for c in _POOLS[2]]


def recipe_for(library: str, arch: str, model: nn.Module) -> Optional[
    Tuple[Tuple[Any, ...], Dict[str, Any]]
]:
    """A known-good input for a library architecture, skipping the search entirely."""
    if library == "diffusers":
        name = type(model).__name__
        if name == "UNet2DConditionModel":
            dim = getattr(getattr(model, "config", None), "cross_attention_dim", 768) or 768
            return (torch.randn(1, 4, 64, 64), torch.tensor([1.0]), torch.randn(1, 77, dim)), {}
        if name in ("AutoencoderKL", "AutoencoderTiny"):
            return (torch.randn(1, 3, 64, 64),), {}
        if name == "SD3Transformer2DModel":
            return (), {
                "hidden_states": torch.randn(1, 16, 32, 32),
                "encoder_hidden_states": torch.randn(1, 77, 4096),
                "pooled_projections": torch.randn(1, 2048),
                "timestep": torch.tensor([1.0]),
            }
        if name == "UNet2DModel":
            return (torch.randn(1, 3, 32, 32), torch.tensor([1.0])), {}
    if library == "transformers":
        return (torch.randint(0, 1000, (1, 16)),), {}
    if library in ("torchvision", "timm"):
        return (torch.randn(1, 3, 224, 224),), {}
    return None
