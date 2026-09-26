"""Stage 8 gates Q1-Q4 (handouts/stage08_rollout.md, section 2). float64."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from aurora import model
from aurora.rollout import mae, rollout

VARS = model.VARS
MC = model.ModelConfig()


def report(name, value, tol, op="<="):
    print(f"\n{name}: {value:.3e} (pass if {op} {tol:.0e})")


def setup(seed=0):
    p = model.init(jax.random.PRNGKey(seed), MC)
    leaves, tree = jax.tree_util.tree_flatten(p)
    ks = jax.random.split(jax.random.PRNGKey(seed + 1), len(leaves) + 2)
    p = jax.tree_util.tree_unflatten(tree, [a + 0.02 * jax.random.normal(k, a.shape) for a, k in zip(leaves, ks)])
    return p, jax.random.normal(ks[-2], (4, 64, 64)), jax.random.normal(ks[-1], (4, 64, 64))


def test_Q1_Q2_rollout_equals_explicit_loop():
    p, a, b = setup()
    n = 3
    ys = jax.jit(partial(rollout, n=n, dt=0.2, mc=MC))(p, a, b)
    x0, x1 = a, b
    err = 0.0
    for k in range(n):
        x0, x1 = x1, model.apply(p, x0, x1, VARS, 0.2, MC)
        err = max(err, float(jnp.max(jnp.abs(ys[k] - x1))))
    report("Q1 rollout vs explicit loop", err, 1e-12)
    assert ys.shape == (n, 4, 64, 64) and err <= 1e-12
    one = rollout(p, a, b, 1, 0.2, MC)[0]
    assert float(jnp.max(jnp.abs(one - model.apply(p, a, b, VARS, 0.2, MC)))) <= 1e-12


def test_Q3_Q4_mae():
    rng = np.random.default_rng(0)
    pred, tgt = rng.standard_normal((2, 5, 4, 8, 8)), rng.standard_normal((2, 5, 4, 8, 8))
    w = np.array([1.0, 2.0, 0.5, 3.0])
    ref_u = np.mean(np.abs(pred - tgt))
    ref_w = (w * np.abs(pred - tgt).mean(axis=(0, 1, 3, 4))).sum() / w.sum()
    e3 = max(abs(float(mae(jnp.asarray(pred), jnp.asarray(tgt))) - ref_u),
             abs(float(mae(jnp.asarray(pred), jnp.asarray(tgt), w)) - ref_w))
    report("Q3 mae vs numpy", e3, 1e-14)
    assert e3 <= 1e-14
    g = jax.grad(lambda x: mae(x, jnp.asarray(tgt), w))(jnp.asarray(pred))
    count = pred.size / 4
    ref_g = w[None, None, :, None, None] * np.sign(pred - tgt) / (count * w.sum())
    e4 = float(np.max(np.abs(np.asarray(g) - ref_g)))
    report("Q4 mae gradient", e4, 1e-14)
    assert e4 <= 1e-14
