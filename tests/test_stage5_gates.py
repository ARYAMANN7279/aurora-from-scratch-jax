"""Stage 5 gates W1-W9 (handouts/stage05_swin.md, section 3).

Reference: tests/fixtures/swin3d_ref.npz, produced by experiments/stage5/make_reference.py from
Aurora's own Swin3DTransformerBlock (microsoft-aurora 2.0.1, see experiments/stage5/reference_env.txt).
"""
import os

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pytest

from aurora import swin

REF = np.load(os.path.join(os.path.dirname(__file__), "fixtures", "swin3d_ref.npz"))
CASES = ["a_noshift", "a_shift", "b_noshift", "b_shift"]
AURORA_PERIODIC = (False, False, True)  # Aurora's warped=True


def report(name, value, tol, op="<="):
    print(f"\n{name}: {value:.3e} (pass if {op} {tol:.0e})")


def case(name, dtype=jnp.float64):
    m = REF[f"{name}/meta"]
    B, C, H, W, D, heads = (int(v) for v in m[:6])
    ws, ss = tuple(int(v) for v in m[6:9]), tuple(int(v) for v in m[9:12])
    sd = {k.split("/w/")[1]: REF[k] for k in REF.files if k.startswith(f"{name}/w/")}
    p = jax.tree_util.tree_map(lambda a: a.astype(dtype), swin.from_torch_state(sd))
    x = jnp.asarray(REF[f"{name}/x"], dtype)
    c = jnp.asarray(REF[f"{name}/c"], dtype)
    return p, x, c, (C, H, W), heads, ws, ss


def test_W1_window_partition_matches_aurora():
    for name in CASES:
        _, x, _, (C, H, W), _, ws, _ = case(name)
        B = x.shape[0]
        win = swin.window_partition(x.reshape(B, C, H, W, -1), ws)
        assert np.array_equal(np.asarray(win), REF[f"{name}/windows"])
        assert jnp.array_equal(swin.window_reverse(win, ws, B, C, H, W), x.reshape(B, C, H, W, -1))


def test_W2_shift_mask_matches_aurora():
    for name in ("a_shift", "b_shift"):
        _, _, _, res, _, ws, ss = case(name)
        ours = np.asarray(swin.shift_mask(res, ws, ss, AURORA_PERIODIC))
        theirs = REF[f"{name}/mask"] == 0.0
        assert ours.shape == theirs.shape and np.array_equal(ours, theirs)


@pytest.mark.parametrize("dtype,key,tol", [(jnp.float32, "y32", 1e-5), (jnp.float64, "y64", 1e-12)])
def test_W3_W4_block_matches_aurora(dtype, key, tol):
    worst = 0.0
    for name in CASES:
        p, x, c, res, heads, ws, ss = case(name, dtype)
        y = swin.block(p, x, c, res, heads, ws, ss, periodic=AURORA_PERIODIC)
        ref = REF[f"{name}/{key}"]
        err = float(np.max(np.abs(np.asarray(y, np.float64) - ref)) / np.max(np.abs(ref)))
        print(f"\n  {name}: {err:.3e}")
        worst = max(worst, err)
    report(f"{'W3' if dtype == jnp.float32 else 'W4'} block vs Aurora ({jnp.dtype(dtype).name})", worst, tol)
    assert worst <= tol


def periodic_setup(seed=0):
    B, res, D, heads, ws = 2, (2, 8, 8), 32, 4, (2, 4, 4)
    p = swin.init_block(jax.random.PRNGKey(seed), D, D)
    k = jax.random.split(jax.random.PRNGKey(seed + 1), 4)
    for n in ("norm1", "norm2"):  # non-trivial modulation
        p[n]["W"] = 0.1 * jax.random.normal(k[0 if n == "norm1" else 1], p[n]["W"].shape)
    x = jax.random.normal(k[2], (B, 2 * 8 * 8, D))
    c = jax.random.normal(k[3], (B, D))
    return p, x, c, res, heads, ws


def test_W5_periodic_shift_equals_rolled_unshifted_block():
    p, x, c, res, heads, ws = periodic_setup()
    ss = (0, 2, 2)
    y = swin.block(p, x, c, res, heads, ws, ss)
    B, D = x.shape[0], x.shape[-1]
    roll = lambda z, s: jnp.roll(z.reshape(B, *res, D), s, axis=(1, 2, 3)).reshape(B, -1, D)
    y_ref = roll(swin.block(p, roll(x, (0, -2, -2)), c, res, heads, ws, (0, 0, 0)), (0, 2, 2))
    err = float(jnp.max(jnp.abs(y - y_ref)))
    report("W5 periodic shifted == roll o unshifted o roll^-1", err, 1e-12)
    assert err <= 1e-12


def test_W6_translation_by_whole_windows_commutes():
    p, x, c, res, heads, ws = periodic_setup(3)
    B, D = x.shape[0], x.shape[-1]
    tr = lambda z: jnp.roll(z.reshape(B, *res, D), (4, 4), axis=(2, 3)).reshape(B, -1, D)
    worst = 0.0
    for ss in ((0, 0, 0), (0, 2, 2), (1, 2, 2)):
        y1 = swin.block(p, tr(x), c, res, heads, ws, ss)
        y2 = tr(swin.block(p, x, c, res, heads, ws, ss))
        worst = max(worst, float(jnp.max(jnp.abs(y1 - y2))))
    report("W6 translation equivariance (whole windows)", worst, 1e-12)
    assert worst <= 1e-12


def test_W7_identity_at_init():
    p = swin.init_block(jax.random.PRNGKey(0), 32, 32)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 128, 32))
    c = jax.random.normal(jax.random.PRNGKey(2), (2, 32))
    for ss in ((0, 0, 0), (1, 2, 2)):
        assert jnp.array_equal(swin.block(p, x, c, (2, 8, 8), 4, (2, 4, 4), ss), x)


def test_W8_gradient():
    p, x, c, res, heads, ws = periodic_setup(5)
    loss = lambda prm: jnp.mean(swin.block(prm, x, c, res, heads, ws, (1, 2, 2)) ** 2)
    flat, unravel = jax.flatten_util.ravel_pytree(p)
    g, _ = jax.flatten_util.ravel_pytree(jax.grad(loss)(p))
    worst = 0.0
    for i in np.random.default_rng(0).choice(flat.size, 5, replace=False):
        h = 1e-6
        e = jnp.zeros_like(flat).at[i].set(h)
        fd = (loss(unravel(flat + e)) - loss(unravel(flat - e))) / (2 * h)
        worst = max(worst, abs(float(fd - g[i])) / max(abs(float(g[i])), 1e-10))
    report("W8 grad vs finite differences (rel)", worst, 1e-6)
    assert worst <= 1e-6


def test_W9_adjust_and_divisibility():
    for row in REF["adjust"]:
        ws, ss, res, w2, s2 = (tuple(int(v) for v in row[i:i + 3]) for i in range(0, 15, 3))
        assert swin.adjust(ws, ss, res) == (w2, s2)
    p = swin.init_block(jax.random.PRNGKey(0), 16, 16)
    with pytest.raises(ValueError):
        swin.block(p, jnp.zeros((1, 2 * 6 * 8, 16)), jnp.zeros((1, 16)), (2, 6, 8), 2, (2, 4, 4), (0, 0, 0))
