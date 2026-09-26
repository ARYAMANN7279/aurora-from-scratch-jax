"""Spectral operators: streamfunction, velocity, nonlinear term, forcing, pressure."""
from typing import NamedTuple

import jax.numpy as jnp

from .grid import irfft2, rfft2


class Params(NamedTuple):
    nu: float
    alpha: float = 0.1
    A: float = 1.0
    kf: int = 4

    @property
    def lam(self):
        return self.nu * self.kf**2 + self.alpha

    @property
    def E_lam(self):
        return self.A**2 / (4 * self.lam**2)


def psi_hat(g, wh):
    return wh * g.k2inv


def vel_hat(g, wh):
    ph = psi_hat(g, wh)
    return 1j * g.ky * ph, -1j * g.kx * ph


def velocity(g, wh):
    uh, vh = vel_hat(g, wh)
    return irfft2(uh, g.N), irfft2(vh, g.N)


def advection(g, wh):
    """Physical u*omega_x + v*omega_y (no dealiasing applied to the product)."""
    u, v = velocity(g, wh)
    wx = irfft2(1j * g.kx * wh, g.N)
    wy = irfft2(1j * g.ky * wh, g.N)
    return u * wx + v * wy


def nonlinear(g, wh):
    """N_hat = M * rfft2(u omega_x + v omega_y)."""
    return g.mask * rfft2(advection(g, wh))


def nonlinear_unmasked(g, wh):
    return rfft2(advection(g, wh))


def forcing_phys(g, prm):
    fy = -prm.A * prm.kf * jnp.cos(prm.kf * g.y)
    return jnp.broadcast_to(fy[:, None], (g.N, g.N))


def forcing_hat(g, prm):
    return g.mask * rfft2(forcing_phys(g, prm))


def sin_kf_hat(g, prm):
    s = jnp.broadcast_to(jnp.sin(prm.kf * g.y)[:, None], (g.N, g.N))
    return rfft2(s)


def pressure_hat(g, wh):
    """Solve lap p = 2(u_x v_y - u_y v_x) with p_hat[0,0] = 0; product masked like N_hat."""
    uh, vh = vel_hat(g, wh)
    ux = irfft2(1j * g.kx * uh, g.N)
    uy = irfft2(1j * g.ky * uh, g.N)
    vx = irfft2(1j * g.kx * vh, g.N)
    vy = irfft2(1j * g.ky * vh, g.N)
    rhs = g.mask * rfft2(2.0 * (ux * vy - uy * vx))
    return -rhs * g.k2inv
