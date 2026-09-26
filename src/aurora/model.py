"""Full model: embed -> Perceiver aggregate -> Swin U-Net backbone -> Perceiver decode -> heads
(handouts/stage07_model.md, section 1). One-step simulator Phi(X^{t-1}, X^t) -> X^{t+1}."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import backbone as bb
from .embed import VARS, embed, unembed
from .embed import init as init_embed
from . import encodings as enc
from . import perceiver
from .levels import n_levels


class ModelConfig(NamedTuple):
    D: int = 64
    P: int = 4
    L: int = 3
    heads: int = 4
    backbone: bb.Config = bb.Config(64, (2, 2, 2), (2, 4, 8), (2, 2, 2), (8, 4, 2), (3, 4, 4))


def init(key, mc=ModelConfig(), dtype=None):
    dtype = dtype or jnp.asarray(0.0).dtype
    if mc.backbone.D != mc.D:
        raise ValueError("backbone width must equal D")
    k = jax.random.split(key, 5)
    p = {"emb": init_embed(k[0], mc.D, mc.P, VARS, dtype=dtype),
         "per": perceiver.init(k[1], mc.D, mc.L, dtype=dtype),
         "bb": bb.init(k[2], mc.backbone, dtype=dtype),
         "frame": 0.02 * jax.random.normal(k[3], (2, mc.D), dtype),
         "mid": {"W": bb._tn(k[4], (2 * mc.D, mc.D), dtype), "b": jnp.zeros(mc.D, dtype)}}
    for v in VARS:  # zero heads: the model is persistence at initialisation
        p["emb"]["head"][v] = jax.tree_util.tree_map(jnp.zeros_like, p["emb"]["head"][v])
    return p


def apply(p, x_prev, x_cur, variables, dt, mc=ModelConfig()):
    """x_prev, x_cur: (V, N, N) with channels in the order of `variables` -> (4, N, N) in VARS order."""
    if not variables:
        raise ValueError("at least one input variable is required")
    N = x_cur.shape[-1]
    C, T = n_levels(N), (N // mc.P) ** 2
    tok = (embed(p["emb"], x_prev, variables, mc.P) + p["frame"][0]
           + embed(p["emb"], x_cur, variables, mc.P) + p["frame"][1])
    z = perceiver.aggregate(p["per"], tok, mc.heads)                                   # (L, T, D)
    t_emb = enc.lead_time(jnp.asarray(dt, x_cur.dtype), mc.D).astype(x_cur.dtype)
    y = bb.apply(p["bb"], z.reshape(1, mc.L * T, mc.D), t_emb[None],
                 (mc.L, N // mc.P, N // mc.P), mc.backbone)                            # (1, L*T, 2D)
    y = y[0].reshape(mc.L, T, 2 * mc.D) @ p["mid"]["W"] + p["mid"]["b"]
    delta = unembed(p["emb"], perceiver.decode(p["per"], y, C, mc.heads), VARS, mc.P, N)
    base = jnp.zeros((len(VARS), N, N), x_cur.dtype)
    for i, v in enumerate(variables):
        base = base.at[VARS.index(v)].set(x_cur[i])
    return base + delta
