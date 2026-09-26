"""Levels = square-shell octave bands (handouts/stage03_embed.md, section 2.1)."""
import math

import jax.numpy as jnp
import numpy as np


def n_levels(N):
    return int(round(math.log2(N))) - 1


def band_index(N):
    """(N, N//2+1) int array: band of each rfft2 mode, -1 for the Nyquist row/column."""
    ky = np.abs(np.fft.fftfreq(N, 1.0 / N))
    kx = np.fft.rfftfreq(N, 1.0 / N)
    n = np.maximum(ky[:, None], kx[None, :]).astype(np.int64)
    idx = np.where(n < 2, 0, np.floor(np.log2(np.maximum(n, 1))).astype(np.int64))
    return np.where(n <= N // 2 - 1, idx, -1)


def band_masks(N, dtype=None):
    idx = band_index(N)
    m = np.stack([idx == j for j in range(n_levels(N))])
    return jnp.asarray(m, dtype=dtype or jnp.asarray(0.0).dtype)


def decompose(x):
    """(..., N, N) -> (..., C, N, N) band-pass fields that sum to x (Nyquist removed)."""
    N = x.shape[-1]
    xh = jnp.fft.rfft2(x, norm="forward")
    m = band_masks(N, x.dtype)
    return jnp.fft.irfft2(xh[..., None, :, :] * m, s=(N, N), norm="forward")


def recompose(xb):
    return xb.sum(axis=-3)
