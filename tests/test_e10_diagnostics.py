"""E10 gates G1-G5 (handouts/e10_diagnostics.md, section 4), plus a check that the head-input path
reproduces model.apply. G1 and G2 need E1's target data and weights and are skipped without them."""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "e10"))
import diag  # noqa: E402
from aurora import model  # noqa: E402
from aurora.embed import VARS  # noqa: E402

HAVE = os.path.isdir(diag.DATA) and os.path.isdir(diag.RUNS)


def _toy(n=6, seed=0):
    mc = model.ModelConfig()
    p = model.init(jax.random.PRNGKey(seed), mc, dtype=jnp.float32)
    k = jax.random.split(jax.random.PRNGKey(seed + 1), 2 * len(VARS) + 1)
    for i, v in enumerate(VARS):  # non-zero heads, so the model is not persistence
        p["emb"]["head"][v]["W"] = 0.05 * jax.random.normal(k[2 * i], p["emb"]["head"][v]["W"].shape)
        p["emb"]["head"][v]["b"] = 0.05 * jax.random.normal(k[2 * i + 1], p["emb"]["head"][v]["b"].shape)
    x = jax.random.normal(k[-1], (n + 2, 4, 64, 64))
    return mc, p, x[:-2], x[1:-1], x[2:]


def _heads(p):
    return {v: (p["emb"]["head"][v]["W"], p["emb"]["head"][v]["b"]) for v in VARS}


def test_head_path_matches_apply():
    mc, p, xm, x0, _ = _toy(3)
    S, C = diag.head_inputs(p, xm, x0, mc)
    y = diag.predict(S, C, _heads(p), x0, mc)
    ref = jnp.stack([model.apply(p, xm[i], x0[i], VARS, diag.DT, mc) for i in range(3)])
    assert float(jnp.max(jnp.abs(y - ref))) < 1e-5


def test_g3_parseval():
    x = np.random.default_rng(0).normal(size=(3, 64, 64))
    assert abs(diag.shell_power(x, diag.shells(64)).sum() / (x**2).sum() - 1) < 1e-6


def test_g4_g5_refit():
    mc, p, xm, x0, xp = _toy()
    S, C = diag.head_inputs(p, xm, x0, mc)
    mse = lambda h: float(jnp.mean((diag.predict(S, C, h, x0, mc) - xp) ** 2))
    h1 = diag.refit(S, C, x0, xp, mc)
    h2 = diag.refit(S, C, x0, xp, mc)
    assert mse(h1) <= mse(_heads(p))  # G5
    p2 = jax.tree_util.tree_map(lambda a: a, p)
    for v in VARS:
        p2["emb"]["head"][v] = {"W": h1[v][0], "b": h1[v][1]}
    h3 = diag.refit(S, C, x0, xp, mc)  # refit of the refit model: same tokens, so the same heads
    assert abs(mse(h3) / mse(h1) - 1) < 1e-6 and abs(mse(h2) / mse(h1) - 1) < 1e-6  # G4


@pytest.mark.skipif(not HAVE, reason="needs E1's target data")
def test_g2_truth_physics():
    a = np.load(os.path.join(diag.DATA, "seed_0002", "frames_64.npy"))[::50]
    D, C = diag.physics(a)
    assert D < 1e-4 and C < 1e-4, (D, C)


@pytest.mark.skipif(not HAVE, reason="needs E1's weights and target data")
def test_g1_reproduces_e1():
    for row in diag.e1_rows():
        if not os.path.exists(os.path.join(diag.RUNS, row["config_hash"], "params.npz")):
            pytest.skip("weights missing")
    rows = diag.e1_rows()
    assert len(rows) == 30
    for row in rows[::7]:  # the full G1 runs in diag.py over all 30 (stored_rel_err beside rel_err)
        mc, p, mu, sd = diag.load_model(row)
        norm = lambda a: jnp.asarray(((a - mu[:, None, None]) / sd[:, None, None]).astype(np.float32))
        xm, x0, xp = diag.pairs(diag.EVAL, norm)
        S, C = diag.head_inputs(p, xm, x0, mc)
        e = diag.predict(S, C, _heads(p), x0, mc) - xp
        rel = float(jnp.sqrt(jnp.sum(e**2) / jnp.sum(xp**2)))
        assert abs(rel / row["metrics"]["val/re300_kf4"]["rel_err"] - 1) < 1e-4, (row["name"], rel)
