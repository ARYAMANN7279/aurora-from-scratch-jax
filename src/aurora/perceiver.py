"""Perceiver level aggregator and level decoder (handouts/stage04_perceiver.md, 2.3-2.5).

Attention runs across levels, separately at every patch position (vmap over T).
"""
import jax
import jax.numpy as jnp

from . import encodings as enc
from .attention import _w, block, init_block


def init(key, D, L, n_blocks=1, dtype=None):
    dtype = dtype or jnp.asarray(0.0).dtype
    kz, ka, kq, kd = jax.random.split(key, 4)
    return {
        "latents": jax.random.normal(kz, (L, D), dtype),
        "agg": [init_block(k, D, dtype) for k in jax.random.split(ka, n_blocks)],
        "dec_q": {"W": _w(kq, D, D, dtype), "b": jnp.zeros(D, dtype)},
        "dec": [init_block(k, D, dtype) for k in jax.random.split(kd, n_blocks)],
    }


def aggregate(p, tokens, H, mask=None):
    """tokens (C, T, D) [+ level mask (C,)] -> latents (L, T, D)."""
    def per_position(kv):
        z = p["latents"]
        for b in p["agg"]:
            z = block(b, z, kv, H, mask)
        return z

    return jax.vmap(per_position)(tokens.transpose(1, 0, 2)).transpose(1, 0, 2)


def decode(p, latents, C_out, H):
    """latents (L, T, D) -> tokens (C_out, T, D) for output levels 0 .. C_out-1."""
    D = latents.shape[-1]
    q0 = enc.level(C_out, D).astype(latents.dtype) @ p["dec_q"]["W"] + p["dec_q"]["b"]

    def per_position(kv):
        x = q0
        for b in p["dec"]:
            x = block(b, x, kv, H)
        return x

    return jax.vmap(per_position)(latents.transpose(1, 0, 2)).transpose(1, 0, 2)
