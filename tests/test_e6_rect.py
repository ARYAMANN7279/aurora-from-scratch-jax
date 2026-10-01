"""E6 gates R1-R6 (handouts/e6_realpde_sim2real.md, section 3): rectangular, non-periodic fields.
The square periodic path must be bitwise unchanged (R1, R3, R4); the new case must be exact (R2)
and must run under jit with masked shifts (R5, R6). float64 unless stated."""
import math

import jax
import jax.numpy as jnp
import numpy as np

from aurora import band_index, decompose, encodings as enc, model, n_levels, patchify, recompose, unpatchify
from aurora import backbone as bb


def band_index_original(N):
    """band_index as it stood before E6 (git 7a16f3f), kept here as the reference."""
    ky = np.abs(np.fft.fftfreq(N, 1.0 / N))
    kx = np.fft.rfftfreq(N, 1.0 / N)
    n = np.maximum(ky[:, None], kx[None, :]).astype(np.int64)
    idx = np.where(n < 2, 0, np.floor(np.log2(np.maximum(n, 1))).astype(np.int64))
    return np.where(n <= N // 2 - 1, idx, -1)


def test_R1_square_bands_unchanged():
    for N in (8, 16, 32, 64, 128, 512):
        assert np.array_equal(band_index(N), band_index_original(N)), N


def test_R2_rectangular_bands_exact():
    H, W = 32, 64
    x = jnp.asarray(np.random.default_rng(0).standard_normal((3, H, W)))
    xb = decompose(x)
    assert xb.shape == (3, n_levels(min(H, W)), H, W) == (3, 4, H, W)
    keep = jnp.asarray(band_index(H, W) >= 0)
    ref = jnp.fft.irfft2(jnp.fft.rfft2(x, norm="forward") * keep, s=(H, W), norm="forward")
    err = float(jnp.max(jnp.abs(recompose(xb) - ref)))
    print(f"\nR2 recompose vs Nyquist-free field: {err:.2e} (pass if <= 1e-12)")
    assert err <= 1e-12
    # only the Nyquist row (ky = H/2) and column (kx = W/2) are dropped
    idx = band_index(H, W)
    assert (idx[H // 2] == -1).all() and (idx[:, -1] == -1).all()
    assert (idx[: H // 2, :-1] >= 0).all() and (idx[H // 2 + 1:, :-1] >= 0).all()


def test_R3_patches_square_unchanged_and_rectangular_exact():
    for N, P in ((64, 4), (16, 2)):
        x = jnp.asarray(np.random.default_rng(1).standard_normal((2, N, N)))
        assert jnp.array_equal(unpatchify(patchify(x, P), P, N), x)
    x = jnp.asarray(np.random.default_rng(2).standard_normal((2, 32, 64)))
    t = patchify(x, 2)
    assert t.shape == (2, 16 * 32, 4)
    assert jnp.array_equal(unpatchify(t, 2, 32, 64), x)


def test_R4_encodings_square_unchanged_nonperiodic_distinct():
    c_old = 2 * math.pi * (jnp.arange(16) + 0.5) * 4 / 64
    yc, xc = jnp.meshgrid(c_old, c_old, indexing="ij")
    assert jnp.array_equal(enc.patch_centres(64, 4), jnp.stack([xc.ravel(), yc.ravel()], axis=-1))
    assert jnp.array_equal(enc.area(64, 4, 16), enc.fourier(jnp.log((2 * math.pi * 4 / 64) ** 2), 16, *enc.AREA_RANGE))
    c = enc.patch_centres(32, 2, 64, periodic=False)
    assert c.shape == (16 * 32, 2)
    assert float(c.max()) < math.pi and float(c.min()) > 0
    pos = enc.position(c, 16)
    # without wrap-around, the first and last columns of a row get different codes
    first, last = pos[0], pos[31]
    assert float(jnp.max(jnp.abs(first - last))) > 0.5


def _rect_config():
    m = model.ModelConfig()
    return m._replace(P=2, backbone=m.backbone._replace(periodic=(False, False, False)))


def test_R5_rectangular_model_is_persistence_at_init():
    mc = _rect_config()
    p = model.init(jax.random.PRNGKey(0), mc, dtype=jnp.float64)
    rng = np.random.default_rng(3)
    for variables in (("u", "v"), ("u", "v", "p")):
        xm = jnp.asarray(rng.standard_normal((len(variables), 32, 64)))
        x0 = jnp.asarray(rng.standard_normal((len(variables), 32, 64)))
        y = jax.jit(lambda a, b: model.apply(p, a, b, variables, 0.02, mc))(xm, x0)
        assert y.shape == (4, 32, 64)
        for i, v in enumerate(variables):
            assert jnp.array_equal(y[model.VARS.index(v)], x0[i])


def test_R6_rectangular_model_trains_and_masks_shifts():
    mc = _rect_config()
    assert bb.stage_resolutions((mc.L, 16, 32), 3) == [(3, 16, 32), (3, 8, 16), (3, 4, 8)]
    p = model.init(jax.random.PRNGKey(1), mc, dtype=jnp.float32)
    rng = np.random.default_rng(4)
    xm, x0, xp = (jnp.asarray(rng.standard_normal((2, 2, 32, 64)), jnp.float32) for _ in range(3))
    fwd = jax.vmap(lambda q, a, b: model.apply(q, a, b, ("u", "v"), 0.02, mc), in_axes=(None, 0, 0))
    loss = lambda q: jnp.mean((fwd(q, xm, x0)[:, :2] - xp) ** 2)
    val, g = jax.jit(jax.value_and_grad(loss))(p)
    gn = float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g))))
    print(f"\nR6 loss {float(val):.4f}, gradient norm {gn:.3e}")
    assert np.isfinite(float(val)) and gn > 0
