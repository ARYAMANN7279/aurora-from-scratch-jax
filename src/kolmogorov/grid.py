"""Spectral grid on the doubly periodic box [0, 2pi)^2.

Conventions (handouts/stage01_solver.md, section 3):
  * physical arrays are indexed [y, x];
  * rfft2 / irfft2 with norm="forward", so coefficients are Fourier amplitudes;
  * spectral arrays have shape (N, N//2 + 1): ky along axis 0, kx >= 0 along axis 1.
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


def rfft2(f):
    return jnp.fft.rfft2(f, norm="forward")


def irfft2(fh, n):
    return jnp.fft.irfft2(fh, s=(n, n), norm="forward")


def real_dtype():
    """float64 when jax_enable_x64 is on, else float32."""
    return jnp.asarray(0.0).dtype


@dataclass(frozen=True, eq=False)
class Grid:
    N: int
    kx: jnp.ndarray      # (N, N//2+1), integer-valued
    ky: jnp.ndarray
    k2: jnp.ndarray
    k2inv: jnp.ndarray   # 1/|k|^2, 0 at the mean mode
    mask: jnp.ndarray    # strict square 2/3 rule, mean mode excluded
    weight: jnp.ndarray  # Parseval weight: 1 on kx=0 and kx=N/2, else 2
    shell: jnp.ndarray   # integer shell index round(|k|)
    n_shells: int
    x: jnp.ndarray       # (N,)
    y: jnp.ndarray       # (N,)

    @property
    def K(self):
        """Largest integer < N/3: last shell complete along the axes."""
        return (self.N - 1) // 3


def make_grid(N, dtype=None):
    if N % 2:
        raise ValueError("N must be even")
    dtype = dtype or real_dtype()
    ky1 = np.fft.fftfreq(N, 1.0 / N)
    kx1 = np.fft.rfftfreq(N, 1.0 / N)
    KY, KX = np.meshgrid(ky1, kx1, indexing="ij")
    k2 = KX**2 + KY**2
    k2inv = np.where(k2 > 0, 1.0 / np.where(k2 > 0, k2, 1.0), 0.0)
    mask = (3 * np.abs(KX) < N) & (3 * np.abs(KY) < N) & (k2 > 0)
    weight = np.full(k2.shape, 2.0)
    weight[:, 0] = 1.0
    weight[:, N // 2] = 1.0
    shell = np.floor(np.sqrt(k2) + 0.5).astype(np.int32)
    xs = 2 * np.pi * np.arange(N) / N
    c = lambda a: jnp.asarray(a, dtype=dtype)
    return Grid(N=N, kx=c(KX), ky=c(KY), k2=c(k2), k2inv=c(k2inv), mask=c(mask),
                weight=c(weight), shell=jnp.asarray(shell), n_shells=int(shell.max()) + 1,
                x=c(xs), y=c(xs))


def resample(fh, n_out):
    """Change resolution by copying Fourier coefficients (zero-pad or truncate).

    Keeps |kx|, |ky| < min(N_in, N_out)/2; the Nyquist modes of the smaller grid are dropped
    (they are zero for every masked state).
    """
    n_in = fh.shape[0]
    K = min(n_in, n_out) // 2
    out = jnp.zeros((n_out, n_out // 2 + 1), dtype=fh.dtype)
    out = out.at[:K, :K].set(fh[:K, :K])
    if K > 1:
        out = out.at[n_out - (K - 1):, :K].set(fh[n_in - (K - 1):, :K])
    return out
