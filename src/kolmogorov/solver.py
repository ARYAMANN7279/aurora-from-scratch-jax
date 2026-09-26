"""Lawson integrating-factor RK4 and the chunked scan driver (handout section 3)."""
import math

import jax
import jax.numpy as jnp

from .diagnostics import make_diagnostics, shell_spectrum
from .grid import irfft2, resample, rfft2
from .ops import forcing_hat, nonlinear, velocity

SAVE_DT = 0.1


def choose_m(N, prm):
    """Smallest integer m with dt = 0.1/m <= 0.5 (2 pi / N) / U, U = 1.5 A / lambda."""
    U = 1.5 * prm.A / prm.lam
    bound = 0.5 * (2 * math.pi / N) / U
    m = max(1, math.ceil(SAVE_DT / bound) - 1)
    while SAVE_DT / m > bound:
        m += 1
    return m


def make_step(g, prm, dt):
    L = -prm.nu * g.k2 - prm.alpha
    E1 = jnp.exp(L * dt)
    E2 = jnp.exp(L * (dt / 2))
    fh = forcing_hat(g, prm)

    def G(wh):
        return -nonlinear(g, wh) + fh

    def step(wh):
        a = G(wh)
        wa = E2 * (wh + (dt / 2) * a)
        b = G(wa)
        wb = E2 * wh + (dt / 2) * b
        c = G(wb)
        wc = E1 * wh + dt * E2 * c
        d = G(wc)
        return g.mask * (E1 * wh + (dt / 6) * (E1 * a + 2 * E2 * (b + c) + d))

    return step


def make_advance(g, prm, dt, n_steps):
    """jit function: state -> state after n_steps."""
    step = make_step(g, prm, dt)

    @jax.jit
    def advance(wh):
        return jax.lax.scan(lambda w, _: (step(w), None), wh, None, length=n_steps)[0]

    return advance


def cfl(g, wh, dt):
    u, v = velocity(g, wh)
    return dt * jnp.max(jnp.abs(u) + jnp.abs(v)) * g.N / (2 * math.pi)


def make_chunk(g, prm, m, n_saves, snap_N=64):
    """jit function advancing n_saves save intervals of m steps (dt = 0.1/m).

    Per save it returns the diagnostics at the save time, the shell spectrum, a snap_N^2 float32
    vorticity snapshot, and the trapezoid integrals of the E and Z budget right-hand sides over
    the preceding save interval (accumulated every step).
    """
    dt = SAVE_DT / m
    step = make_step(g, prm, dt)
    diag, rates = make_diagnostics(g, prm)

    def inner_body(carry, _):
        wh, IE, IZ, rE, rZ = carry
        wh = step(wh)
        rE2, rZ2 = rates(wh)
        IE = IE + (dt / 2) * (rE + rE2)
        IZ = IZ + (dt / 2) * (rZ + rZ2)
        return (wh, IE, IZ, rE2, rZ2), None

    def save_body(wh, _):
        rE, rZ = rates(wh)
        zero = jnp.zeros((), dtype=rE.dtype)
        (wh, IE, IZ, _, _), _ = jax.lax.scan(inner_body, (wh, zero, zero, rE, rZ), None, length=m)
        E, Z, Pi, P, Q = diag(wh)
        out = dict(E=E, Z=Z, Pi=Pi, P=P, Q=Q, cfl=cfl(g, wh, dt), IE=IE, IZ=IZ,
                   spec=shell_spectrum(g, wh),
                   snap=irfft2(resample(wh, snap_N), snap_N).astype(jnp.float32))
        return wh, out

    @jax.jit
    def chunk(wh):
        return jax.lax.scan(save_body, wh, None, length=n_saves)

    return chunk


def random_ic(g, seed=0, kmax=8, energy=1.0):
    """Real white noise (numpy default_rng(seed)), keep 1 <= |k| <= kmax, masked, scaled to E."""
    import numpy as np

    rng = np.random.default_rng(seed)
    wh = rfft2(jnp.asarray(rng.standard_normal((g.N, g.N)), dtype=g.k2.dtype))
    keep = (g.k2 >= 1) & (g.k2 <= kmax**2)
    wh = g.mask * jnp.where(keep, wh, 0)
    E = 0.5 * jnp.sum(g.weight * jnp.abs(wh) ** 2 * g.k2inv)
    return wh * jnp.sqrt(energy / E)
