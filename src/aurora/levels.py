"""Levels = square-shell octave bands (handouts/stage03_embed.md, section 2.1)."""
import math

import jax.numpy as jnp
import numpy as np


def n_levels(N):
    return int(round(math.log2(N))) - 1


def band_index(N, W=None):
    """(N, W//2+1) int array: band of each rfft2 mode, -1 for the Nyquist row/column.

    W defaults to N (square). On a rectangular N x W grid both wavenumbers are measured in units of
    the shorter side, so the bands are square shells in physical wavenumber (E6); on a square grid
    this is exactly the original definition."""
    W = N if W is None else W
    s = min(N, W)
    ky = np.abs(np.fft.fftfreq(N, 1.0 / N)) * (s / N)
    kx = np.fft.rfftfreq(W, 1.0 / W) * (s / W)
    n = np.maximum(ky[:, None], kx[None, :])
    idx = np.where(n < 2, 0, np.floor(np.log2(np.maximum(n, 1))).astype(np.int64))
    return np.where(n < s / 2, idx, -1)


def band_masks(N, dtype=None, W=None):
    idx = band_index(N, W)
    m = np.stack([idx == j for j in range(n_levels(min(N, N if W is None else W)))])
    return jnp.asarray(m, dtype=dtype or jnp.asarray(0.0).dtype)


def decompose(x):
    """(..., H, W) -> (..., C, H, W) band-pass fields that sum to x (Nyquist removed)."""
    H, W = x.shape[-2:]
    xh = jnp.fft.rfft2(x, norm="forward")
    m = band_masks(H, x.dtype, W=None if W == H else W)
    return jnp.fft.irfft2(xh[..., None, :, :] * m, s=(H, W), norm="forward")


def recompose(xb):
    return xb.sum(axis=-3)
