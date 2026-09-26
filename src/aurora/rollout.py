"""Autoregressive rollout and losses (handouts/stage08_rollout.md, section 1)."""
import jax
import jax.numpy as jnp

from . import model


def rollout(p, x_prev, x_cur, n, dt, mc=model.ModelConfig()):
    """Feed predictions back n times. x_prev, x_cur: (4, N, N) (all variables) -> (n, 4, N, N)."""
    def body(carry, _):
        a, b = carry
        c = model.apply(p, a, b, model.VARS, dt, mc)
        return (b, c), c

    return jax.lax.scan(body, (x_prev, x_cur), None, length=n)[1]


def mae(pred, target, weights=None):
    """sum_v w_v * mean|pred_v - target_v| / sum_v w_v; the variable axis is -3."""
    per_var = jnp.mean(jnp.abs(pred - target), axis=tuple(i for i in range(pred.ndim) if i != pred.ndim - 3))
    w = jnp.ones_like(per_var) if weights is None else jnp.asarray(weights, pred.dtype)
    return jnp.sum(w * per_var) / jnp.sum(w)


def mse(pred, target):
    return jnp.mean((pred - target) ** 2)
