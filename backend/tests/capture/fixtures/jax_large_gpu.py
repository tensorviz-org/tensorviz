"""Full upstream text architectures, random BF16 parameters, real CUDA arrays.

Opt-in only: these models allocate tens of gigabytes. No checkpoint is downloaded.
"""
import dataclasses
import functools

import jax
import jax.numpy as jnp
from flax import nnx

CASES = ('qwen3_32b', 'gemma4_31b', 'gemma4_26b_a4b')


@jax.jit
def net(**kwargs):
    return build()(**kwargs)


@functools.lru_cache(maxsize=1)
def model_for(case):
    assert jax.config.jax_default_device.platform == 'gpu', 'GPU setup is required'
    if case == 'qwen3_32b':
        from tunix.models.qwen3 import model as upstream
        config = upstream.ModelConfig.qwen3_32b()
        constructor = upstream.Qwen3
    elif case in ('gemma4_31b', 'gemma4_26b_a4b'):
        from tunix.models.gemma4 import model as upstream
        config = getattr(upstream.ModelConfig, case)()
        constructor = upstream.Gemma4
    else:
        raise ValueError(case)
    # Only precision changes. Depth, width, vocabulary and expert counts are
    # the upstream published text configuration; Gemma's text_only defaults true.
    config = dataclasses.replace(config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    return constructor(config, rngs=nnx.Rngs(0))


def inputs(case, size=64, mode='eager'):
    model = model_for(case)
    tokens = 'input_tokens' if case.startswith('qwen') else 'tokens'
    values = {tokens: jnp.ones((1, size), dtype=jnp.int32),
        'positions': jnp.arange(size)[None], 'cache': None,
        'attention_mask': jnp.tril(jnp.ones((1, size, size), dtype=jnp.bool_))}
    if mode == 'decode':
        capacity = size * 2
        values.update(cache=model.init_cache(1, capacity, jnp.bfloat16),
            attention_mask=(jnp.arange(capacity)[None, :] <= jnp.arange(size)[:, None])[None])
        _, cache = jax.block_until_ready(model(**values))
        values = {tokens: jnp.ones((1, 1), dtype=jnp.int32), 'positions': jnp.array([[size]]),
            'cache': cache, 'attention_mask': (jnp.arange(capacity) <= size)[None, None, :]}
    return values


def build(case='qwen3_32b', size=64, mode='eager'):
    model = model_for(case)
    if mode == 'jit':
        # NNX returns updated model state; donate it to avoid duplicating >60 GB
        # of parameters. Parameters stay dynamic rather than XLA constants.
        return functools.partial(nnx.jit(lambda model, **kwargs: model(**kwargs), donate_argnums=(0,)), model)
    return model
