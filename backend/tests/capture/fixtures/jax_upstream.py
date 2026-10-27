"""Small configurations of unmodified upstream models, for local acceptance.

Dependencies and source revisions are checked by jax_upstream_acceptance.py.
These are randomly initialized architecture tests, not pretrained evaluations.
"""
import dataclasses
import functools

import jax
import jax.numpy as jnp


# A setup factory supplies the initialized upstream callable for this entry.
@jax.jit
def net(**inputs):
    return build()(**inputs)


def inputs(case, size=4):
    if case.endswith('_cache'):
        base = case.removesuffix('_cache')
        model = build(base, size)
        cache = model.init_cache(1, size * 2, jnp.float32)
        prefill = inputs(base, size)
        prefill.update(cache=cache, attention_mask=(jnp.arange(size * 2)[None, :] <= jnp.arange(size)[:, None])[None])
        _, cache = model(**prefill)
        token_key = 'tokens' if base.startswith('tunix_gemma4') else 'input_tokens'
        return {token_key: jnp.ones((1, 1), dtype=jnp.int32), 'positions': jnp.array([[size]]),
                'cache': cache, 'attention_mask': (jnp.arange(size * 2) <= size)[None, None, :]}
    if case.startswith('vit'):
        return dict(image=jnp.ones((1, size * 4, size * 4, 3)))
    if case == 'siglip2':
        return dict(image=jnp.ones((1, size * 4, size * 4, 3)),
                    text=jnp.ones((1, size), dtype=jnp.int32))
    tokens = jnp.ones((1, size), dtype=jnp.int32)
    if case.startswith('tunix_gemma4'):
        return dict(tokens=tokens, positions=jnp.arange(size)[None, :],
                    attention_mask=jnp.tril(jnp.ones((1, size, size), dtype=jnp.bool_)))
    if case.startswith('tunix'):
        return dict(input_tokens=tokens, positions=jnp.arange(size)[None, :],
                    cache=None, attention_mask=jnp.tril(jnp.ones((1, size, size), dtype=jnp.bool_)))
    return dict(tokens=tokens, return_hidden_states=True)


def build(case='gemma4', size=4):
    case = case.removesuffix('_cache')
    if case.startswith('gemma'):
        from gemma import gm
        if case == 'gemma3':
            original = gm.nn.Gemma3_1B()
        else:
            original = (gm.nn.Gemma4_26B_A4B if case == 'gemma4_moe' else gm.nn.Gemma4_E2B)(text_only=True)
        changes = dict(num_embed=128, embed_dim=32, hidden_dim=64, num_heads=4,
            head_dim=8, num_kv_heads=2, attention_types=tuple(dict.fromkeys(original.config.attention_types))[:2],
            sliding_window_size=8, vision_encoder=None)
        if case != 'gemma3':
            changes.update(num_global_kv_heads=2, global_key_size=8,
                per_layer_input_dim=8, kv_cache_sharing_config=None,
                override_kv_shared_ffw_hidden=None, audio_encoder=None)
        if case == 'gemma4_moe':
            changes.update(num_experts=4, expert_dim=32, top_k_experts=2, moe_dense_hidden_dim=64)
        model = original.clone(config=dataclasses.replace(original.config, **changes), dtype=jnp.float32)
    elif case.startswith('vit'):
        from big_vision.models import vit
        model = vit.Model(num_classes=8, patch_size=(4, 4), width=32, depth=2,
                          mlp_dim=64, num_heads=4, scan=case == 'vit_scan')
    elif case == 'siglip2':
        from big_vision.models.proj.image_text import two_towers
        model = two_towers.Model(out_dim=16, bias_init=-10,
            image=dict(patch_size=(4, 4), width=32, depth=2, mlp_dim=64, num_heads=4, pool_type='map'),
            text=dict(width=32, depth=2, mlp_dim=64, num_heads=4, vocab_size=128))
    elif case.startswith('tunix_gemma4'):
        from flax import nnx
        from tunix.models.gemma4 import model as gemma4
        config = gemma4.ModelConfig.gemma4_e2b()
        config = dataclasses.replace(config, num_layers=2, num_embed=128,
            embed_dim=32, hidden_dim=64, num_heads=4, head_dim=8, num_kv_heads=2,
            num_global_kv_heads=2, global_key_size=8, sliding_window_size=8,
            frac_shared_layers=0, per_layer_input_dim=8,
            attention_pattern=(gemma4.AttentionType.LOCAL_SLIDING, gemma4.AttentionType.GLOBAL),
            param_dtype=jnp.float32)
        if case.endswith('_moe'):
            config = dataclasses.replace(config, enable_moe=True, num_experts=4,
                expert_dim=32, num_experts_per_tok=2, moe_dense_hidden_dim=64)
        return gemma4.Gemma4(config, rngs=nnx.Rngs(0))
    elif case.startswith('tunix'):
        from flax import nnx
        from tunix.models.qwen3 import model as qwen
        config = qwen.ModelConfig(num_layers=2, vocab_size=128, embed_dim=32, hidden_dim=64,
            num_heads=4, head_dim=8, num_kv_heads=2, rope_theta=1_000_000, norm_eps=1e-6)
        if case == 'tunix_qwen3_remat':
            config.remat_config = qwen.RematConfig.DECODER
        return qwen.Qwen3(config, rngs=nnx.Rngs(0))
    else:
        raise ValueError(f'Unknown acceptance case: {case}')
    variables = model.init(jax.random.key(0), **inputs(case, size))
    return functools.partial(model.apply, variables)
