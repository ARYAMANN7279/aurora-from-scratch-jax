"""Integral diagnostics, budgets and shell spectrum (handout section 4)."""
import jax.numpy as jnp

from .ops import forcing_hat, sin_kf_hat, vel_hat


def inner(g, ah, bh):
    """Domain mean <a b> from half-plane coefficients (Parseval weights)."""
    return jnp.sum(g.weight * jnp.real(ah * jnp.conj(bh)))


def norm(g, ah):
    return jnp.sqrt(inner(g, ah, ah))


def make_diagnostics(g, prm):
    fh = forcing_hat(g, prm)
    sh = sin_kf_hat(g, prm)

    def diag(wh):
        a2 = g.weight * jnp.abs(wh) ** 2
        E = 0.5 * jnp.sum(a2 * g.k2inv)
        Z = 0.5 * jnp.sum(a2)
        Pi = 0.5 * jnp.sum(a2 * g.k2)
        uh, _ = vel_hat(g, wh)
        P = prm.A * inner(g, uh, sh)
        Q = inner(g, wh, fh)
        return E, Z, Pi, P, Q

    def rates(wh):
        E, Z, Pi, P, Q = diag(wh)
        rE = P - 2 * prm.nu * Z - 2 * prm.alpha * E
        rZ = Q - 2 * prm.nu * Pi - 2 * prm.alpha * Z
        return rE, rZ

    return diag, rates


def shell_spectrum(g, wh):
    e = 0.5 * g.weight * jnp.abs(wh) ** 2 * g.k2inv
    return jnp.zeros(g.n_shells, dtype=e.dtype).at[g.shell.ravel()].add(e.ravel())
