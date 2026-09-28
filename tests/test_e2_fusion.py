"""E2 gates (handouts/e2_heterogeneous_fusion.md, section 4) on a tiny dataset."""
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aurora import fusion, model
from aurora.embed import VARS
from aurora.harness import model_config
from kdata import finalize, generate_dataset

SMALL = {"D": 16, "P": 4, "L": 2, "heads": 2,
         "backbone": {"enc_depths": [1, 1], "enc_heads": [1, 2], "dec_depths": [1, 1],
                      "dec_heads": [2, 1], "window": [2, 4, 4]}}
MC = model_config(SMALL)


def _perturbed(p, s=0.05):
    leaves, tdef = jax.tree_util.tree_flatten(p)
    ks = jax.random.split(jax.random.PRNGKey(7), len(leaves))
    return jax.tree_util.tree_unflatten(tdef, [l + s * jax.random.normal(k, l.shape, l.dtype) for l, k in zip(leaves, ks)])


def _band_limited(key, shape, n):
    x = jax.random.normal(key, shape)
    return fusion.spectral_resample(fusion.spectral_resample(x, n // 2), n)


def test_spectral_round_trip():
    x = _band_limited(jax.random.PRNGKey(0), (4, 32, 32), 32)
    y = fusion.spectral_resample(fusion.spectral_resample(x, 64), 32)
    assert float(jnp.max(jnp.abs(x - y))) < 1e-5


def test_per_source_single_source_equals_base():
    src = [{"name": "A", "vars": list(VARS)}]
    p = _perturbed(fusion.init(jax.random.PRNGKey(0), MC, "per_source", src))
    base = dict(p, emb=p["emb"]["A"], per=p["per"]["A"])
    x0, x1 = (_band_limited(jax.random.PRNGKey(i), (4, 32, 32), 32) for i in (1, 2))
    y = fusion.predict(p, x0, x1, src[0], "per_source", 32, 0.2, MC)
    np.testing.assert_array_equal(np.asarray(y), np.asarray(model.apply(base, x0, x1, VARS, 0.2, MC)))


def test_pad_with_zero_mask_embedding_equals_base():
    """pad-and-mask on a source already at n_max with every variable equals the base model when the
    mask embeddings are zero."""
    src = {"name": "A", "vars": list(VARS)}
    p = _perturbed(fusion.init(jax.random.PRNGKey(0), MC, "pad", [src]))
    for m in fusion.MASKS:
        p["emb"]["embed"][m] = jax.tree_util.tree_map(jnp.zeros_like, p["emb"]["embed"][m])
    x0, x1 = (_band_limited(jax.random.PRNGKey(i), (4, 32, 32), 32) for i in (1, 2))
    y = fusion.predict(p, x0, x1, src, "pad", 32, 0.2, MC)
    ref = model.apply(p, x0, x1, VARS, 0.2, MC)
    assert float(jnp.max(jnp.abs(y - ref))) < 1e-5


def test_loss_ignores_missing_variables():
    """A source without p and omega: for every method, the prediction (and so the loss and the
    metrics) covers exactly its own variables at its own resolution."""
    src = {"name": "B", "vars": ["u", "v"]}
    for method in fusion.METHODS:
        p = _perturbed(fusion.init(jax.random.PRNGKey(0), MC, method, [src]))
        x0, x1 = (_band_limited(jax.random.PRNGKey(i), (2, 16, 16), 16) for i in (1, 2))
        y = fusion.predict(p, x0, x1, src, method, 32, 0.2, MC)
        assert y.shape == (2, 16, 16)


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    r = tmp_path_factory.mktemp("e2")
    ds = dict(name="tiny", regimes=[dict(Re=100, kf=4, N_gen=64, seeds=[0, 1, 2]), dict(Re=300, kf=4, N_gen=64, seeds=[0, 1, 2])],
              T_spin=1.0, T_rec=12.0, save_dt=0.2, N_out=[16, 32])
    generate_dataset(ds, str(r / "tiny"), verbose=False)
    finalize(str(r / "tiny"))
    return str(r)


@pytest.mark.parametrize("method", fusion.METHODS)
def test_end_to_end(method, data_root, tmp_path):
    cfg = {"name": f"e2_tiny_{method}", "prediction": "gate: runs end to end", "method": method,
           "data": {"dataset": "tiny", "sources": [
               {"name": "A", "tag": "re100_kf4", "N": 32, "vars": list(VARS), "train_seeds": [0, 1], "val_seeds": [2]},
               {"name": "B", "tag": "re300_kf4", "N": 16, "vars": ["u", "v"], "train_seeds": [0, 1], "val_seeds": [2]}]},
           "model": copy.deepcopy(SMALL),
           "train": {"steps": 3, "batch": 2, "lr": 1e-3, "schedule": "cosine", "seed": 0},
           "eval": {"rollout_steps": 5}}
    row = fusion.run(cfg, data_root=data_root, runs_root=str(tmp_path / "runs"), results_path=str(tmp_path / "r.jsonl"))
    for s in ("A", "B"):
        m = row["metrics"][f"val/{s}"]
        assert np.isfinite(m["rel_err"]) and np.isfinite(m["rollout_drift"]) and m["persistence"] > 0
    bad = copy.deepcopy(cfg)
    bad["data"]["sources"][0]["val_seeds"] = [1]
    with pytest.raises(ValueError):
        fusion.run(bad, data_root=data_root, runs_root=str(tmp_path / "runs"), results_path=str(tmp_path / "r.jsonl"))
