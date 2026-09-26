"""Stage 4 gates K1-K8 (handouts/stage04_perceiver.md, section 3). float64."""
from functools import partial

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pytest

import importlib

emb = importlib.import_module("aurora.embed")
from aurora import n_params, perceiver
from aurora.attention import init_mha, mha

VARS = ("u", "v", "p", "omega")


def report(name, value, tol, op="<="):
    print(f"\n{name}: {value:.3e} (pass if {op} {tol:.0e})")


def rand(shape, seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape))


def ref_mha(p, q_in, kv_in, H, mask=None):
    """NumPy loop reference: explicit heads, explicit softmax."""
    P = {k: np.asarray(v) for k, v in p.items()}
    q_in, kv_in = np.asarray(q_in), np.asarray(kv_in)
    Q, D = q_in.shape
    dh = D // H
    q, k, v = q_in @ P["Wq"], kv_in @ P["Wk"], kv_in @ P["Wv"]
    out = np.zeros((Q, D))
    for h in range(H):
        sl = slice(h * dh, (h + 1) * dh)
        for i in range(Q):
            s = np.array([q[i, sl] @ k[j, sl] / np.sqrt(dh) for j in range(kv_in.shape[0])])
            if mask is not None:
                s = np.where(np.asarray(mask), s, -np.inf)
            w = np.exp(s - s.max())
            w /= w.sum()
            out[i, sl] = sum(w[j] * v[j, sl] for j in range(kv_in.shape[0]))
    return out @ P["Wo"] + P["bo"]


@pytest.mark.parametrize("H", [1, 4])
def test_K1_mha_matches_reference(H):
    D = 16
    p = init_mha(jax.random.PRNGKey(0), D, jnp.float64)
    p["bo"] = p["bo"] + 0.1
    q, kv = rand((3, D), 1), rand((7, D), 2)
    mask = jnp.array([True, False, True, True, False, True, True])
    e1 = float(jnp.max(jnp.abs(mha(p, q, kv, H) - ref_mha(p, q, kv, H))))
    e2 = float(jnp.max(jnp.abs(mha(p, q, kv, H, mask) - ref_mha(p, q, kv, H, mask))))
    report(f"K1 mha vs reference, H={H} (unmasked / masked)", max(e1, e2), 1e-12)
    assert max(e1, e2) <= 1e-12


def setup(D=32, L=3, seed=0):
    return perceiver.init(jax.random.PRNGKey(seed), D, L), 4


def test_K2_level_permutation_invariance():
    p, H = setup()
    tok = rand((6, 10, 32))
    perm = jnp.array([3, 0, 5, 1, 4, 2])
    err = float(jnp.max(jnp.abs(perceiver.aggregate(p, tok, H) - perceiver.aggregate(p, tok[perm], H))))
    report("K2 level-permutation invariance", err, 1e-12)
    assert err <= 1e-12


def test_K3_position_equivariance():
    p, H = setup()
    tok = rand((5, 12, 32))
    perm = jnp.asarray(np.random.default_rng(3).permutation(12))
    z = perceiver.aggregate(p, tok, H)
    zp = perceiver.aggregate(p, tok[:, perm], H)
    y = perceiver.decode(p, z, 4, H)
    yp = perceiver.decode(p, zp, 4, H)
    err = max(float(jnp.max(jnp.abs(z[:, perm] - zp))), float(jnp.max(jnp.abs(y[:, perm] - yp))))
    report("K3 position equivariance", err, 1e-12)
    assert err <= 1e-12


def test_K4_masked_padding_has_no_effect():
    p, H = setup()
    tok = rand((5, 10, 32))
    pad = jnp.concatenate([tok, 1e3 * rand((3, 10, 32), 9)])
    mask = jnp.array([True] * 5 + [False] * 3)
    err = float(jnp.max(jnp.abs(perceiver.aggregate(p, tok, H) - perceiver.aggregate(p, pad, H, mask))))
    report("K4 masked padding vs unpadded", err, 1e-12)
    assert err <= 1e-12
    under_jit = jax.jit(partial(perceiver.aggregate, H=H))(p, pad, mask=mask)
    assert float(jnp.max(jnp.abs(under_jit - perceiver.aggregate(p, tok, H)))) <= 1e-12


def test_K5_any_C_to_L3_and_back_under_jit():
    p, H = setup()
    agg = jax.jit(partial(perceiver.aggregate, H=H))
    for C in (3, 5, 6, 8):
        z = agg(p, rand((C, 16, 32), C))
        assert z.shape == (3, 16, 32)
        for C_out in (3, 5, 8):
            assert jax.jit(partial(perceiver.decode, C_out=C_out, H=H))(p, z).shape == (C_out, 16, 32)


def test_K6_softmax_rows_and_empty_mask():
    p = init_mha(jax.random.PRNGKey(0), 8, jnp.float64)
    q, kv = rand((2, 8)), rand((4, 8), 1)
    dh = 4
    k = (kv @ p["Wk"]).reshape(4, 2, dh).transpose(1, 0, 2)
    qq = (q @ p["Wq"]).reshape(2, 2, dh).transpose(1, 0, 2)
    w = jax.nn.softmax(qq @ k.transpose(0, 2, 1) / jnp.sqrt(dh), axis=-1)
    err = float(jnp.max(jnp.abs(w.sum(-1) - 1)))
    report("K6 softmax row sums", err, 1e-14)
    assert err <= 1e-14
    with pytest.raises(ValueError):
        mha(p, q, kv, 2, jnp.zeros(4, bool))


def test_K7_gradient_of_full_autoencoder():
    P, D, N, H = 4, 16, 16, 2
    k1, k2 = jax.random.split(jax.random.PRNGKey(7))
    params = {"emb": emb.init(k1, D, P), "per": perceiver.init(k2, D, 3)}
    x = rand((4, N, N), 5)
    C = 3

    def loss(prm):
        tok = emb.embed(prm["emb"], x, VARS, P)
        out = perceiver.decode(prm["per"], perceiver.aggregate(prm["per"], tok, H), C, H)
        return jnp.mean((emb.unembed(prm["emb"], out, VARS, P, N) - x) ** 2)

    flat, unravel = jax.flatten_util.ravel_pytree(params)
    g, _ = jax.flatten_util.ravel_pytree(jax.grad(loss)(params))
    worst = 0.0
    for i in np.random.default_rng(0).choice(flat.size, 5, replace=False):
        h = 1e-6
        e = jnp.zeros_like(flat).at[i].set(h)
        fd = (loss(unravel(flat + e)) - loss(unravel(flat - e))) / (2 * h)
        worst = max(worst, abs(float(fd - g[i])) / max(abs(float(g[i])), 1e-10))
    report("K7 grad vs finite differences (rel)", worst, 1e-6)
    assert worst <= 1e-6


def test_K8_init_and_param_count():
    D, L = 128, 3
    a = perceiver.init(jax.random.PRNGKey(0), D, L)
    b = perceiver.init(jax.random.PRNGKey(0), D, L)
    assert all(jnp.array_equal(x, y) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))
    assert n_params(a) == 25 * D * D + 21 * D + L * D == 412672
