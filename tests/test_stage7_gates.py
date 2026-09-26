"""Stage 7 gates M1-M6 (handouts/stage07_model.md, section 2). float64."""
from functools import partial

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

from aurora import model

VARS = model.VARS
MC = model.ModelConfig()


def report(name, value, tol, op="<="):
    print(f"\n{name}: {value:.3e} (pass if {op} {tol:.0e})")


def frames(N, seed=0, V=4):
    k = jax.random.split(jax.random.PRNGKey(seed))
    return jax.random.normal(k[0], (V, N, N)), jax.random.normal(k[1], (V, N, N))


def perturbed(seed=0, scale=0.05):
    p = model.init(jax.random.PRNGKey(seed), MC)
    leaves, tree = jax.tree_util.tree_flatten(p)
    ks = jax.random.split(jax.random.PRNGKey(seed + 100), len(leaves))
    return jax.tree_util.tree_unflatten(tree, [a + scale * jax.random.normal(k, a.shape) for a, k in zip(leaves, ks)])


def test_M1_identity_at_init():
    p = model.init(jax.random.PRNGKey(0), MC)
    for N in (64, 128):
        a, b = frames(N)
        assert jnp.array_equal(model.apply(p, a, b, VARS, 0.2, MC), b)


def test_M2_M6_variable_subsets():
    p = perturbed()
    a, b = frames(64)
    for S in (VARS, ("u", "v"), ("omega",), ("u", "v", "p")):
        idx = jnp.array([VARS.index(v) for v in S])
        y = model.apply(p, a[idx], b[idx], S, 0.2, MC)
        assert y.shape == (4, 64, 64)
        if S == ("u", "v"):
            zero_heads = jax.tree_util.tree_map(lambda x: x, p)
            for v in VARS:
                zero_heads["emb"]["head"][v] = jax.tree_util.tree_map(jnp.zeros_like, p["emb"]["head"][v])
            base = model.apply(zero_heads, a[idx], b[idx], S, 0.2, MC)
            assert jnp.array_equal(base[2], jnp.zeros((64, 64))) and jnp.array_equal(base[3], jnp.zeros((64, 64)))
            assert jnp.array_equal(base[0], b[0]) and jnp.array_equal(base[1], b[1])


def test_M3_two_resolutions_under_jit():
    p = perturbed(1)
    f = jax.jit(partial(model.apply, variables=VARS, dt=0.2, mc=MC))
    for N in (64, 128):
        a, b = frames(N, 2)
        assert f(p, a, b).shape == (4, N, N)


def test_M4_translation_equivariance_128():
    """Amendment A1: with the absolute position encoding zeroed (the only position-dependent term),
    the rest of the model must commute with a 16-patch translation."""
    p = perturbed(2)
    p["emb"]["enc"]["pos"] = jax.tree_util.tree_map(jnp.zeros_like, p["emb"]["enc"]["pos"])
    a, b = frames(128, 3)
    sh = lambda z: jnp.roll(z, (64, 64), axis=(-2, -1))
    err = float(jnp.max(jnp.abs(model.apply(p, sh(a), sh(b), VARS, 0.2, MC) - sh(model.apply(p, a, b, VARS, 0.2, MC)))))
    report("M4 translation equivariance (128^2, 64-point shift)", err, 1e-10)
    assert err <= 1e-10


def test_M5_gradient():
    p = perturbed(4)
    a, b = frames(64, 5)
    target = frames(64, 6)[0]
    loss = lambda prm: jnp.mean((model.apply(prm, a, b, VARS, 0.2, MC) - target) ** 2)
    flat, unravel = jax.flatten_util.ravel_pytree(p)
    g, _ = jax.flatten_util.ravel_pytree(jax.grad(loss)(p))
    gmed = float(jnp.median(jnp.abs(g)))
    worst = 0.0
    for i in np.random.default_rng(0).choice(flat.size, 5, replace=False):
        h = 1e-4
        e = jnp.zeros_like(flat).at[i].set(h)
        fd = (loss(unravel(flat + e)) - loss(unravel(flat - e))) / (2 * h)
        worst = max(worst, abs(float(fd - g[i])) / max(abs(float(g[i])), gmed))
    report("M5 grad vs finite differences (mixed)", worst, 1e-6)
    assert worst <= 1e-6
