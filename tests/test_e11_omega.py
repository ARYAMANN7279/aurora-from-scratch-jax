"""E11 gates D1-D5 (handouts/e11_omega_only.md, section 3): predicting ω only and deriving u, v, p.
D1's bitwise comparison with the previous commit's harness is run by hand (recorded in the E11 report);
here D1 checks the default. D2 needs E1's target data and is skipped without it."""
import copy
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aurora import harness, model
from aurora.embed import VARS
from test_harness_options import BASE, data_root, run  # noqa: F401  (the Stage 11 tiny dataset fixture)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRUE = os.path.join(ROOT, "data", "stage2", "e1_v1", "re300_kf4", "seed_0002", "frames_64.npy")


def ddx(f, axis):
    N = f.shape[-1]
    k = np.fft.fftfreq(N, 1.0 / N).reshape([N if a == axis % f.ndim else 1 for a in range(f.ndim)])
    return np.real(np.fft.ifft(1j * k * np.fft.fft(f, axis=axis), axis=axis))


def physics(phys):
    """(4, N, N) physical fields -> (divergence ratio, curl mismatch), as E10 measures them."""
    u, v, w = (np.asarray(phys[i], np.float64) for i in (0, 1, 3))
    ux, uy, vx, vy = ddx(u, -1), ddx(u, -2), ddx(v, -1), ddx(v, -2)
    g = np.sqrt(np.sum(ux**2 + uy**2 + vx**2 + vy**2))
    return np.linalg.norm(ux + vy) / g, np.linalg.norm(w - (vx - uy)) / np.linalg.norm(w)


def test_D1_default_is_off():
    assert model.ModelConfig().derive == ()
    assert harness.model_config(BASE["model"]).derive == ()


@pytest.mark.skipif(not os.path.exists(TRUE), reason="needs E1's target data")
def test_D2_derives_the_true_fields():
    a = np.load(TRUE)[::50].astype(np.float32)
    mu, sd = a.mean(axis=(0, 2, 3)), a.std(axis=(0, 2, 3))
    xn = (a - mu[:, None, None]) / sd[:, None, None]
    y = np.asarray(jax.vmap(lambda x: model.derive_from_omega(x, tuple(mu), tuple(sd)))(jnp.asarray(xn)))
    rel = [np.linalg.norm(y[:, i] - xn[:, i]) / np.linalg.norm(xn[:, i]) for i in range(4)]
    print(f"\nD2: derived u, v, p, ω relative errors {rel}")
    assert rel[0] < 1e-5 and rel[1] < 1e-5 and rel[2] < 1e-3 and rel[3] < 1e-5


def _model(derive, seed=0, N=32):
    mc = harness.model_config(BASE["model"])
    mu, sd = (0.1, -0.2, 0.3, 0.05), (1.1, 0.9, 0.4, 3.0)
    if derive:
        mc = mc._replace(derive=(mu, sd))
    p = model.init(jax.random.PRNGKey(seed), mc, dtype=jnp.float32)
    k = jax.random.split(jax.random.PRNGKey(seed + 1), 9)
    for i, v in enumerate(VARS):  # non-zero heads, so the output is not persistence
        p["emb"]["head"][v]["W"] = 0.1 * jax.random.normal(k[2 * i], p["emb"]["head"][v]["W"].shape)
    x = jax.random.normal(k[8], (2, 4, N, N))
    return mc, p, x, mu, sd


def test_D3_output_is_physical():
    mc, p, x, mu, sd = _model(True)
    y = model.apply(p, x[0], x[1], VARS, 0.2, mc)
    phys = np.asarray(y) * np.asarray(sd)[:, None, None] + np.asarray(mu)[:, None, None]
    D, C = physics(phys)
    print(f"\nD3: divergence ratio {D:.1e}, curl mismatch {C:.1e}")
    assert D < 1e-4 and C < 1e-4


def test_D4_gradients():
    mc, p, x, _, _ = _model(True)
    target = jax.random.normal(jax.random.PRNGKey(9), x[1].shape)
    loss = lambda q: jnp.mean((model.apply(q, x[0], x[1], VARS, 0.2, mc)[3] - target[3]) ** 2)
    g = jax.jit(jax.grad(loss))(p)
    assert all(bool(jnp.all(jnp.isfinite(a))) for a in jax.tree_util.tree_leaves(g))
    assert float(jnp.abs(g["emb"]["head"]["omega"]["W"]).sum()) > 0
    for v in ("u", "v", "p"):
        assert float(jnp.abs(g["emb"]["head"][v]["W"]).sum()) == 0 and float(jnp.abs(g["emb"]["head"][v]["b"]).sum()) == 0
    assert any(float(jnp.abs(a).sum()) > 0 for a in jax.tree_util.tree_leaves(g["bb"]))
    # with the loss over all four (derived) fields the u, v, p heads still get nothing
    loss4 = lambda q: jnp.mean((model.apply(q, x[0], x[1], VARS, 0.2, mc) - target) ** 2)
    g4 = jax.grad(loss4)(p)
    assert all(float(jnp.abs(g4["emb"]["head"][v]["W"]).sum()) == 0 for v in ("u", "v", "p"))


@pytest.mark.parametrize("fields", [None, ["omega"]])
def test_D5_arms_run(data_root, tmp_path, fields):  # noqa: F811
    c = copy.deepcopy(BASE)
    c["model"]["derive_from_omega"] = True
    if fields:
        c["train"]["loss_fields"] = fields
    c["eval"]["rollout_steps"] = 3
    row = run(c, data_root, tmp_path, "w")
    m = row["metrics"]["val/re100_kf4"]
    assert np.isfinite(row["final_loss"]) and np.isfinite(m["rel_err"]) and np.isfinite(m["rollout_drift"])


def test_loss_fields_need_mse(data_root, tmp_path):  # noqa: F811
    c = copy.deepcopy(BASE)
    c["train"].update(loss="mae", loss_fields=["omega"])
    with pytest.raises(ValueError):
        run(c, data_root, tmp_path, "m")
