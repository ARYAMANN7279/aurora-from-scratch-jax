"""Per-variable patch embedding and unembedding heads (handouts/stage03_embed.md, 2.4-2.6).

Parameters are a plain pytree:
    {"embed": {var: {"W": (P*P, D), "b": (D,)}},
     "head":  {var: {"W": (D, P*P), "b": (P*P,)}},
     "enc":   {"pos" | "area" | "level" | "time": {"W": (D, D), "b": (D,)}}}
"""
import jax
import jax.numpy as jnp

from . import encodings as enc
from .levels import decompose, n_levels
from .patches import patchify, unpatchify

VARS = ("u", "v", "p", "omega")
ENCODINGS = ("pos", "area", "level", "time")


def _linear(key, n_in, n_out, dtype):
    return {"W": jax.random.normal(key, (n_in, n_out), dtype) / jnp.sqrt(n_in), "b": jnp.zeros(n_out, dtype)}


def init(key, D, P, variables=VARS, dtype=None):
    if D % 4:
        raise ValueError("D must be divisible by 4")
    dtype = dtype or jnp.asarray(0.0).dtype
    k_e, k_h, k_c = jax.random.split(key, 3)
    ke = jax.random.split(k_e, len(variables))
    kh = jax.random.split(k_h, len(variables))
    kc = jax.random.split(k_c, len(ENCODINGS))
    return {
        "embed": {v: _linear(k, P * P, D, dtype) for v, k in zip(variables, ke)},
        "head": {v: _linear(k, D, P * P, dtype) for v, k in zip(variables, kh)},
        "enc": {n: _linear(k, D, D, dtype) for n, k in zip(ENCODINGS, kc)},
    }


def encoding_terms(params, N, P, C):
    """Learned-linear-mapped encodings: position (T, D), area (D,), level (C, D)."""
    lin = lambda name, x: x @ params["enc"][name]["W"] + params["enc"][name]["b"]
    D = params["enc"]["pos"]["W"].shape[0]
    return (lin("pos", enc.position(enc.patch_centres(N, P), D)),
            lin("area", enc.area(N, P, D)),
            lin("level", enc.level(C, D)))


def to_levels(x, bands=True):
    """(V, N, N) -> (V, C, N, N)."""
    return decompose(x) if bands else x[:, None]


def embed(params, x, variables, P, bands=True, encodings=True):
    """x: (V, N, N) with channels in the order of `variables` -> tokens (C, T, D)."""
    N = x.shape[-1]
    C = n_levels(N) if bands else 1
    D = params["enc"]["pos"]["W"].shape[0]
    tok = jnp.zeros((C, (N // P) ** 2, D), dtype=params["enc"]["pos"]["W"].dtype)
    if variables:
        xl = to_levels(x, bands)
        for i, v in enumerate(variables):
            tok = tok + patchify(xl[i], P) @ params["embed"][v]["W"] + params["embed"][v]["b"]
    if encodings:
        pos, ar, lev = encoding_terms(params, N, P, C)
        tok = tok + pos[None] + ar[None, None] + lev[:, None]
    return tok


def unembed_levels(params, tokens, variables, P, N):
    """tokens (C, T, D) -> (V, C, N, N) per-level reconstructions."""
    return jnp.stack([unpatchify(tokens @ params["head"][v]["W"] + params["head"][v]["b"], P, N)
                      for v in variables])


def unembed(params, tokens, variables, P, N):
    """tokens (C, T, D) -> (V, N, N): per-variable heads, then the sum over levels."""
    return unembed_levels(params, tokens, variables, P, N).sum(axis=1)


def n_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
