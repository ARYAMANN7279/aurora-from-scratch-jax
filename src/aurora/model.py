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
    derive: tuple = ()  # E11: ((mean of u, v, p, ω), (std ...)) -> keep ω, derive u, v, p from it


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
    """x_prev, x_cur: (V, H, W) with channels in the order of `variables` -> (4, H, W) in VARS order.

    Square periodic fields are the experiments' case. A rectangular grid and a non-periodic
    backbone (E6) are also handled; position encodings follow the backbone's periodicity in x, y."""
    if not variables:
        raise ValueError("at least one input variable is required")
    H, W = x_cur.shape[-2:]
    Wr = None if W == H else W
    per = tuple(mc.backbone.periodic[1:]) == (True, True)
    C, T = n_levels(min(H, W)), (H // mc.P) * (W // mc.P)
    tok = (embed(p["emb"], x_prev, variables, mc.P, periodic=per) + p["frame"][0]
           + embed(p["emb"], x_cur, variables, mc.P, periodic=per) + p["frame"][1])
    z = perceiver.aggregate(p["per"], tok, mc.heads)                                   # (L, T, D)
    t_emb = enc.lead_time(jnp.asarray(dt, x_cur.dtype), mc.D).astype(x_cur.dtype)
    y = bb.apply(p["bb"], z.reshape(1, mc.L * T, mc.D), t_emb[None],
                 (mc.L, H // mc.P, W // mc.P), mc.backbone)                            # (1, L*T, 2D)
    y = y[0].reshape(mc.L, T, 2 * mc.D) @ p["mid"]["W"] + p["mid"]["b"]
    delta = unembed(p["emb"], perceiver.decode(p["per"], y, C, mc.heads), VARS, mc.P, H, Wr)
    base = jnp.zeros((len(VARS), H, W), x_cur.dtype)
    for i, v in enumerate(variables):
        if v in VARS:  # extra embedded inputs (e.g. E2's presence masks) have no residual
            base = base.at[VARS.index(v)].set(x_cur[i])
    out = base + delta
    return derive_from_omega(out, *mc.derive) if mc.derive else out


def derive_from_omega(x, mean, std):
    """(4, N, N) normalised (u, v, p, ω) -> four mutually consistent fields fixed by the predicted ω on
    the 2π-periodic grid (handouts/e11_omega_only.md): ω without its mean and Nyquist modes (zero in
    the data), psi = -lap^-1 ω, u = psi_y, v = -psi_x, lap p = 2(u_x v_y - u_y v_x); zero-mean
    u, v, p, as in the data."""
    N = x.shape[-1]
    if x.shape[-2] != N:
        raise ValueError("derive_from_omega needs a square periodic grid")
    mean, std = jnp.asarray(mean, x.dtype), jnp.asarray(std, x.dtype)
    k = jnp.fft.fftfreq(N, 1.0 / N).astype(x.dtype)
    ky, kx = jnp.meshgrid(k, k, indexing="ij")
    k2 = kx**2 + ky**2
    k2inv = jnp.where(k2 > 0, 1.0 / jnp.where(k2 > 0, k2, 1.0), 0.0)
    keep = (k2 > 0) & (jnp.abs(kx) < N / 2) & (jnp.abs(ky) < N / 2)  # a periodic ω has zero mean; drop Nyquist
    wh = jnp.where(keep, jnp.fft.fft2(x[3] * std[3] + mean[3]), 0)
    w = jnp.real(jnp.fft.ifft2(wh))
    ph = wh * k2inv
    uh, vh = 1j * ky * ph, -1j * kx * ph
    d = lambda fh, kk: jnp.real(jnp.fft.ifft2(1j * kk * fh))
    rhs = jnp.fft.fft2(2.0 * (d(uh, kx) * d(vh, ky) - d(uh, ky) * d(vh, kx)))
    phys = jnp.stack([jnp.real(jnp.fft.ifft2(uh)), jnp.real(jnp.fft.ifft2(vh)), jnp.real(jnp.fft.ifft2(-rhs * k2inv))])
    phys = jnp.concatenate([phys, w[None]])
    return ((phys - mean[:, None, None]) / std[:, None, None]).astype(x.dtype)
