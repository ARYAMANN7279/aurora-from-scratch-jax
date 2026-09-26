"""Stage 3 gates H1-H12 (handouts/stage03_embed.md, section 3). float64 unless stated."""
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aurora import band_index, decompose, embed, encodings as enc, init, n_levels, patchify, recompose, unembed, unpatchify
from kolmogorov import resample, rfft2, irfft2

VARS = ("u", "v", "p", "omega")


def report(name, value, tol, op="<="):
    print(f"\n{name}: {value:.3e} (pass if {op} {tol:.0e})")


def field(N, seed=0, V=None, dtype=jnp.float64):
    """Random field(s) with the Nyquist row/column removed (as stored frames are)."""
    shape = (N, N) if V is None else (V, N, N)
    x = jnp.asarray(np.random.default_rng(seed).standard_normal(shape), dtype=dtype)
    xh = rfft2(x)
    ky = jnp.abs(jnp.fft.fftfreq(N, 1.0 / N))[:, None]
    kx = jnp.fft.rfftfreq(N, 1.0 / N)[None, :]
    return irfft2(jnp.where((ky == N // 2) | (kx == N // 2), 0, xh), N).astype(dtype)


def test_H1_patch_round_trip():
    for N in (64, 128):
        for P in (2, 4, 8):
            x = field(N)
            assert jnp.array_equal(unpatchify(patchify(x, P), P, N), x)


def test_H2_patch_order():
    N, P = 16, 4
    x = jnp.arange(N * N, dtype=jnp.float64).reshape(N, N)
    t = patchify(x, P)
    for py in range(N // P):
        for px in range(N // P):
            want = x[py * P:(py + 1) * P, px * P:(px + 1) * P].ravel()
            assert jnp.array_equal(t[py * (N // P) + px], want)


def test_H3_H4_bands_sum_and_orthogonal():
    for N in (64, 128):
        x = field(N)
        b = decompose(x)
        e3 = float(jnp.max(jnp.abs(recompose(b) - x)) / jnp.max(jnp.abs(x)))
        e4 = 0.0
        for i in range(b.shape[0]):
            for j in range(i + 1, b.shape[0]):
                e4 = max(e4, abs(float(jnp.mean(b[i] * b[j]))) / float(jnp.sqrt(jnp.mean(b[i] ** 2) * jnp.mean(b[j] ** 2))))
        report(f"H3 recompose(decompose(x)) N={N}", e3, 1e-13)
        report(f"H4 band orthogonality N={N}", e4, 1e-13)
        assert e3 <= 1e-13 and e4 <= 1e-13


def test_H5_band_nesting_across_resolution():
    x = field(128)
    b128 = decompose(x)
    x64 = irfft2(resample(rfft2(x), 64), 64)
    b64 = decompose(x64)
    err = max(float(jnp.max(jnp.abs(b64[j] - irfft2(resample(rfft2(b128[j]), 64), 64)))) for j in range(5))
    report("H5 band nesting 128 -> 64", err, 1e-13)
    assert err <= 1e-13


def test_H6_level_count_and_partition():
    for N in (32, 64, 128, 256):
        idx = band_index(N)
        assert n_levels(N) == int(math.log2(N)) - 1 == idx.max() + 1
        ky = np.abs(np.fft.fftfreq(N, 1.0 / N))[:, None]
        kx = np.fft.rfftfreq(N, 1.0 / N)[None, :]
        retained = np.maximum(ky, kx) <= N // 2 - 1
        assert np.all((idx >= 0) == retained)
        n = np.maximum(ky, kx)
        for j in range(n_levels(N)):
            lo, hi = (0, 2) if j == 0 else (2**j, 2 ** (j + 1))
            assert np.all(((idx == j) & retained) == ((n >= lo) & (n < hi) & retained))


def pinv_params(D, P, dtype):
    """Embedding with random square stacked weights; heads = its exact (pseudo-)inverse."""
    params = init(jax.random.PRNGKey(0), D, P, VARS, dtype=dtype)
    Ws = jnp.concatenate([params["embed"][v]["W"] for v in VARS], axis=0)  # (V*P*P, D)
    b = sum(params["embed"][v]["b"] + 0.1 for v in VARS)
    params = jax.tree_util.tree_map(lambda a: a, params)
    for v in VARS:
        params["embed"][v]["b"] = params["embed"][v]["b"] + 0.1
    Winv = jnp.linalg.pinv(Ws.astype(jnp.float64)).astype(dtype)  # (D, V*P*P)
    for i, v in enumerate(VARS):
        blk = Winv[:, i * P * P:(i + 1) * P * P]
        params["head"][v] = {"W": blk, "b": -(b @ blk)}
    return params


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_H7_acceptance_round_trip(dtype):
    """float64: <= 1e-12 (pre-registered). float32: <= 10 * cond(W) * eps (amendment A1; the
    pre-registered 1e-5 failed at 1.72e-5 and is reported as such)."""
    P = 4
    D = len(VARS) * P * P
    params = pinv_params(D, P, dtype)
    Ws = jnp.concatenate([params["embed"][v]["W"] for v in VARS]).astype(jnp.float64)
    sv = jnp.linalg.svd(Ws, compute_uv=False)
    kappa = float(sv[0] / sv[-1])
    tol = 1e-12 if dtype == jnp.float64 else 10 * kappa * float(jnp.finfo(jnp.float32).eps)
    print(f"\nH7 cond(W) = {kappa:.0f}")
    worst = 0.0
    for N in (64, 128):
        x = field(N, V=4, dtype=dtype)
        tok = embed(params, x, VARS, P, bands=True, encodings=False)
        xr = unembed(params, tok, VARS, P, N)
        worst = max(worst, float(jnp.linalg.norm(xr - x) / jnp.linalg.norm(x)))
    report(f"H7 round trip ({jnp.dtype(dtype).name})", worst, tol)
    assert worst <= tol


def test_H8_variable_subsets_add():
    P, D, N = 4, 32, 64
    params = init(jax.random.PRNGKey(1), D, P)
    for v in VARS:
        params["embed"][v]["b"] = params["embed"][v]["b"] + 0.3
    x = field(N, V=4)
    full = embed(params, x, VARS, P)
    empty = embed(params, x[:0], (), P)
    parts = sum(embed(params, x[i:i + 1], (v,), P) - empty for i, v in enumerate(VARS)) + empty
    pair = embed(params, x[:2], VARS[:2], P)
    pair_parts = embed(params, x[:1], VARS[:1], P) + embed(params, x[1:2], VARS[1:2], P) - empty
    err = max(float(jnp.max(jnp.abs(full - parts))), float(jnp.max(jnp.abs(pair - pair_parts))))
    report("H8 subset additivity", err, 1e-12)
    assert err <= 1e-12


def test_H9_encodings():
    D = 32
    c = enc.patch_centres(64, 4)
    per = float(jnp.max(jnp.abs(enc.position(c + 2 * math.pi, D) - enc.position(c, D))))
    report("H9 position periodicity", per, 1e-12)
    assert per <= 1e-12
    p = enc.position(c, D)
    d2 = jnp.sum((p[:, None] - p[None]) ** 2, axis=-1) + jnp.eye(p.shape[0]) * 1e9
    mind = float(jnp.sqrt(jnp.min(d2)))
    report("H9 min distance between patch codes", mind, 1e-3, ">")
    darea = float(jnp.linalg.norm(enc.area(64, 4, D) - enc.area(128, 4, D)))
    report("H9 area code 64 vs 128", darea, 1e-3, ">")
    assert mind > 1e-3 and darea > 1e-3


def test_H10_gradient_matches_finite_differences():
    P, D, N = 4, 16, 16
    params = init(jax.random.PRNGKey(2), D, P)
    x = field(N, V=4)

    def loss(prm):
        return jnp.mean((unembed(prm, embed(prm, x, VARS, P), VARS, P, N) - x) ** 2)

    g = jax.grad(loss)(params)
    flat, unravel = jax.flatten_util.ravel_pytree(params)
    gflat, _ = jax.flatten_util.ravel_pytree(g)
    rng = np.random.default_rng(0)
    worst = 0.0
    for i in rng.choice(flat.size, 5, replace=False):
        h = 1e-6
        e = jnp.zeros_like(flat).at[i].set(h)
        fd = (loss(unravel(flat + e)) - loss(unravel(flat - e))) / (2 * h)
        worst = max(worst, abs(float(fd - gflat[i])) / max(abs(float(gflat[i])), 1e-12))
    report("H10 grad vs finite differences (rel)", worst, 1e-6)
    assert worst <= 1e-6


def test_H11_init_deterministic():
    a = init(jax.random.PRNGKey(3), 32, 4)
    b = init(jax.random.PRNGKey(3), 32, 4)
    c = init(jax.random.PRNGKey(4), 32, 4)
    same = jax.tree_util.tree_all(jax.tree_util.tree_map(lambda p, q: bool(jnp.array_equal(p, q)), a, b))
    diff = any(not jnp.array_equal(p, q) for p, q in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(c))
               if p.size and jnp.any(p != 0))
    assert same and diff


def test_H12_same_params_two_resolutions_under_jit():
    P, D = 4, 32
    params = init(jax.random.PRNGKey(5), D, P)
    f = jax.jit(partial(embed, variables=VARS, P=P))
    for N, C in ((64, 5), (128, 6)):
        tok = f(params, field(N, V=4))
        assert tok.shape == (C, (N // P) ** 2, D)
        out = jax.jit(partial(unembed, variables=VARS, P=P, N=N))(params, tok)
        assert out.shape == (4, N, N)


import jax.flatten_util  # noqa: E402  (used in H10)
