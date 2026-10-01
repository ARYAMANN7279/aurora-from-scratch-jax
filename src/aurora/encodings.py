"""Fourier-feature encodings (handouts/stage03_embed.md, section 2.3)."""
import math

import jax.numpy as jnp

AREA_RANGE = (0.01, 100.0)
LEVEL_RANGE = (0.5, 64.0)
TIME_RANGE = (0.05, 100.0)


def fourier(s, D, lmin, lmax):
    """s (...,) -> (..., D): sin/cos of 2 pi s / lambda_i, D/2 wavelengths geometric in [lmin, lmax]."""
    lam = jnp.geomspace(lmin, lmax, D // 2)
    ang = 2 * math.pi * jnp.asarray(s)[..., None] / lam
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


def patch_centres(N, P, W=None, periodic=True):
    """(T, 2) patch-centre angles (x_c, y_c), row-major over [py, px], on an N x W grid (W defaults
    to N). In [0, 2 pi) on a periodic grid; in [0, pi) on a non-periodic one, so that its two edges
    get different codes (E6)."""
    W = N if W is None else W
    span = 2 * math.pi if periodic else math.pi
    cy = span * (jnp.arange(N // P) + 0.5) * P / N
    cx = span * (jnp.arange(W // P) + 0.5) * P / W
    yc, xc = jnp.meshgrid(cy, cx, indexing="ij")
    return jnp.stack([xc.ravel(), yc.ravel()], axis=-1)


def position(centres, D):
    """(T, 2) -> (T, D): sin(m x), cos(m x), sin(m y), cos(m y) for m = 1..D/4 (periodic)."""
    m = jnp.arange(1, D // 4 + 1)
    x = centres[:, :1] * m
    y = centres[:, 1:] * m
    return jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=-1)


def area(N, P, D, W=None):
    a = (2 * math.pi * P / N) ** 2 if W is None else (2 * math.pi * P / N) * (2 * math.pi * P / W)
    return fourier(jnp.log(a), D, *AREA_RANGE)


def level(C, D):
    return fourier(jnp.arange(C, dtype=jnp.asarray(0.0).dtype), D, *LEVEL_RANGE)


def lead_time(dt, D):
    return fourier(dt, D, *TIME_RANGE)
