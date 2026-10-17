"""E8 gates H1-H4 (handouts/e8_followups.md, section 3): the harness options, on the Stage 11 tiny dataset. Scan
chunks reproduce the step-by-step loop, with and without pushforward (H1); the many-start drift equals a
rollout-by-rollout computation (H2); the defaults add no metric (H3); the attention options run end to end through
the harness (H4)."""
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aurora import harness, model
from aurora.rollout import rollout
from kdata import finalize, generate_dataset, traj_dir

DS = dict(name="tiny", regimes=[dict(Re=100, kf=4, N_gen=64), dict(Re=300, kf=4, N_gen=64)],
          seeds=[0, 1, 2], T_spin=1.0, T_rec=11.0, save_dt=0.2, N_out=[16, 32])  # 56 frames: room for a rollout from frame 50
BASE = {
    "name": "tiny_opts", "prediction": "gate test: harness options",
    "data": {"dataset": "tiny", "N": 32, "stride": 1,
             "train": [{"tag": "re100_kf4", "seeds": [0, 1]}, {"tag": "re300_kf4", "seeds": [0, 1]}],
             "test": [], "val": [{"tag": "re100_kf4", "seeds": [2]}]},
    "model": {"D": 16, "P": 4, "L": 2, "heads": 2,
              "backbone": {"enc_depths": [1, 1], "enc_heads": [1, 2], "dec_depths": [1, 1],
                           "dec_heads": [2, 1], "window": [2, 4, 4]}},
    "train": {"steps": 7, "batch": 2, "lr": 1e-3, "schedule": "cosine", "loss": "mse",
              "pushforward": False, "seed": 0, "init_from": None, "lora_rank": None},
    "eval": {"splits": ["val"], "rollout_steps": 0},
}


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    r = tmp_path_factory.mktemp("h")
    generate_dataset(DS, str(r / "tiny"), verbose=False)
    finalize(str(r / "tiny"))
    return str(r)


def run(cfg, data_root, tmp, tag):
    return harness.run(cfg, data_root=data_root, runs_root=str(tmp / f"runs_{tag}"),
                       results_path=str(tmp / f"results_{tag}.jsonl"))


def params(row, tmp, tag):
    return np.load(tmp / f"runs_{tag}" / row["config_hash"] / "params.npz")["flat"]


@pytest.mark.parametrize("pf", [False, True])
def test_H1_scan_chunks_reproduce_the_loop(data_root, tmp_path, pf):
    """7 steps: three chunks of 2 and one step of the loop, against 7 steps of the loop, two regimes drawn."""
    a = copy.deepcopy(BASE)
    a["train"]["pushforward"] = pf
    b = copy.deepcopy(a)
    b["train"]["scan_chunk"] = 2
    ra, rb = run(a, data_root, tmp_path, "a"), run(b, data_root, tmp_path, "b")
    pa, pb = params(ra, tmp_path, "a"), params(rb, tmp_path, "b")
    d = float(np.max(np.abs(pa - pb)) / np.max(np.abs(pa)))
    print(f"\nH1 pushforward={pf}: scan vs loop, max parameter difference {d:.1e} (relative)")
    assert d <= 1e-5
    assert abs(rb["final_loss"] - ra["final_loss"]) <= 1e-5 * abs(ra["final_loss"])
    ea, eb = ra["metrics"]["val/re100_kf4"]["rel_err"], rb["metrics"]["val/re100_kf4"]["rel_err"]
    assert abs(eb - ea) <= 1e-4 * ea


def test_H2_many_start_drift_is_rollout_by_rollout(data_root, tmp_path):
    c = copy.deepcopy(BASE)
    c["eval"].update(drift_every=2, drift_steps=3, drift_from=1)
    r = run(c, data_root, tmp_path, "c")
    m = r["metrics"]["val/re100_kf4"]
    flat = jnp.asarray(params(r, tmp_path, "c"))
    mc = harness.model_config(c["model"])
    template = model.init(jax.random.PRNGKey(0), mc, dtype=jnp.float32)
    p = jax.flatten_util.ravel_pytree(template)[1](flat)
    z = np.load(tmp_path / "runs_c" / r["config_hash"] / "params.npz")
    v = np.load(f"{traj_dir(data_root + '/tiny', 're100_kf4', 2)}/frames_32.npy")
    v = ((v - z["mean"][:, None, None]) / z["std"][:, None, None]).astype(np.float32)
    starts = list(range(1, v.shape[0] - 3, 2))
    E = []
    for s in starts:
        pred = np.asarray(rollout(p, jnp.asarray(v[s - 1]), jnp.asarray(v[s]), 3, 0.2, mc))
        tr = v[s + 1:s + 4]
        E.append(np.sqrt(((pred - tr) ** 2).sum(axis=(1, 2, 3)) / (tr**2).sum(axis=(1, 2, 3))))
    E = np.array(E)
    print(f"\nH2 {len(starts)} rollouts; by step {np.round(E.mean(0), 4)}")
    assert m["drift_rollouts"] == len(starts) >= 3
    assert np.allclose(m["rollout_err_by_step"], E.mean(0), rtol=1e-4)
    assert np.isclose(m["rollout_drift_many"], E.mean(), rtol=1e-4)
    assert np.isclose(m["rollout_drift_many_long"], E.mean(), rtol=1e-4)
    # one start, at frame 50: the short many-start drift is then the existing rollout_drift over rollout_steps steps
    c["eval"].update(rollout_steps=2, drift_every=10, drift_from=50)
    m2 = run(c, data_root, tmp_path, "d")["metrics"]["val/re100_kf4"]
    assert m2["drift_rollouts"] == 1
    assert np.isclose(m2["rollout_drift_many"], m2["rollout_drift"], rtol=1e-5)
    assert np.isclose(m2["rollout_drift_many"], np.mean(m2["rollout_err_by_step"][:2]), rtol=1e-6)
    assert np.isclose(m2["rollout_drift_many_long"], np.mean(m2["rollout_err_by_step"]), rtol=1e-6)


def test_H3_defaults_add_no_metric(data_root, tmp_path):
    r = run(BASE, data_root, tmp_path, "e")
    assert set(r["metrics"]["val/re100_kf4"]) == {"rel_err", "persistence", "ratio"}
    assert 0 < r["train_s"] <= r["wall_s"]


def test_H4_attention_options_through_the_harness(data_root, tmp_path):
    c = copy.deepcopy(BASE)
    c["model"]["backbone"].update(rel_pos_bias=True, qk_norm=True)
    c["train"]["scan_chunk"] = 3
    r = run(c, data_root, tmp_path, "f")
    assert np.isfinite(r["final_loss"]) and r["n_params"] > run(BASE, data_root, tmp_path, "g")["n_params"]


def test_H5_training_curve_leaves_training_unchanged(data_root, tmp_path):
    """Final-report follow-up: logging a training curve changes no parameter, and its last point is the final error."""
    c = copy.deepcopy(BASE)
    c["train"]["steps"] = 7
    d = copy.deepcopy(c)
    d["train"]["curve_every"] = 3
    rc, rd = run(c, data_root, tmp_path, "h"), run(d, data_root, tmp_path, "i")
    assert np.array_equal(params(rc, tmp_path, "h"), params(rd, tmp_path, "i"))
    assert "curve" not in rc and [x["step"] for x in rd["curve"]] == [0, 3, 6, 7]
    assert rd["curve"][0]["train_loss"] is None and all(x["train_loss"] > 0 for x in rd["curve"][1:])
    assert np.isclose(rd["curve"][-1]["val/re100_kf4"], rd["metrics"]["val/re100_kf4"]["rel_err"], rtol=1e-6)
    d["train"]["scan_chunk"] = 2
    with pytest.raises(ValueError):
        run(d, data_root, tmp_path, "j")
    d["train"].pop("scan_chunk")
    for interval in (0, -1, 0.5, True):
        d["train"]["curve_every"] = interval
        with pytest.raises(ValueError, match="positive integer"):
            run(d, data_root, tmp_path, "invalid")
