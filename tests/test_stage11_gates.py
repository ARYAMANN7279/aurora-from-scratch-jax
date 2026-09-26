"""Stage 11 gates X1-X7 (handouts/stage11_harness.md, section 3), on a tiny dataset."""
import copy
import json
import os

import numpy as np
import pytest

from aurora import harness
from kdata import finalize, generate_dataset

DS = dict(name="tiny", regimes=[dict(Re=100, kf=4, N_gen=64), dict(Re=300, kf=4, N_gen=64)],
          seeds=[0, 1, 2], T_spin=1.0, T_rec=2.0, save_dt=0.2, N_out=[16, 32])
BASE = {
    "name": "tiny_run", "prediction": "gate test: runs end to end",
    "data": {"dataset": "tiny", "N": 32, "stride": 1,
             "train": [{"tag": "re100_kf4", "seeds": [0, 1]}], "test": ["re300_kf4"],
             "val": [{"tag": "re100_kf4", "seeds": [2]}]},
    "model": {"D": 16, "P": 4, "L": 2, "heads": 2,
              "backbone": {"enc_depths": [1, 1], "enc_heads": [1, 2], "dec_depths": [1, 1],
                           "dec_heads": [2, 1], "window": [2, 4, 4]}},
    "train": {"steps": 3, "batch": 2, "lr": 1e-3, "schedule": "cosine", "loss": "mse",
              "pushforward": False, "seed": 0, "init_from": None, "lora_rank": None},
    "eval": {"splits": ["val", "test"], "rollout_steps": 0},
}


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    r = tmp_path_factory.mktemp("h")
    generate_dataset(DS, str(r / "tiny"), verbose=False)
    finalize(str(r / "tiny"))
    return str(r)


def run(cfg, data_root, tmp, tag="a"):
    return harness.run(cfg, data_root=data_root, runs_root=str(tmp / f"runs_{tag}"),
                       results_path=str(tmp / f"results_{tag}.jsonl"))


def test_X1_config_hash():
    a = copy.deepcopy(BASE)
    b = json.loads(json.dumps(BASE, sort_keys=True))
    b["name"], b["prediction"] = "other", "reworded"
    assert harness.config_hash(a) == harness.config_hash(b)
    c = copy.deepcopy(BASE)
    c["train"]["steps"] = 4
    assert harness.config_hash(c) != harness.config_hash(a)


def test_X2_X3_refusals(data_root, tmp_path):
    c = copy.deepcopy(BASE)
    c["prediction"] = "  "
    with pytest.raises(ValueError):
        run(c, data_root, tmp_path)
    c = copy.deepcopy(BASE)
    c["data"]["train"] = ["re100_kf4", "re300_kf4"]
    with pytest.raises(ValueError):
        run(c, data_root, tmp_path)


def test_X4_X5_one_row_and_determinism(data_root, tmp_path):
    r1 = run(BASE, data_root, tmp_path, "a")
    r2 = run(BASE, data_root, tmp_path, "b")
    rows = [json.loads(line) for line in open(tmp_path / "results_a.jsonl")]
    assert len(rows) == 1
    for k in ("config_hash", "name", "prediction", "git_commit", "started", "wall_s", "n_params",
              "n_trainable", "final_loss", "metrics", "config"):
        assert k in rows[0]
    assert set(rows[0]["metrics"]) == {"val/re100_kf4", "test/re300_kf4"}
    assert r1["metrics"] == r2["metrics"] and r1["final_loss"] == r2["final_loss"]
    p1 = np.load(tmp_path / "runs_a" / r1["config_hash"] / "params.npz")["flat"]
    p2 = np.load(tmp_path / "runs_b" / r2["config_hash"] / "params.npz")["flat"]
    assert np.array_equal(p1, p2)


def test_X6_expand():
    cfgs = harness.expand({"base": BASE, "grid": {"train.seed": [0, 1, 2], "model.L": [1, 2]}})
    assert len(cfgs) == 6
    assert len({c["name"] for c in cfgs}) == 6 and len({harness.config_hash(c) for c in cfgs}) == 6
    assert {(c["train"]["seed"], c["model"]["L"]) for c in cfgs} == {(s, L) for s in (0, 1, 2) for L in (1, 2)}


def test_X7_init_from_and_lora(data_root, tmp_path):
    a = run(BASE, data_root, tmp_path, "c")
    runs = tmp_path / "runs_c"
    b = copy.deepcopy(BASE)
    b["train"].update(steps=0, init_from=a["config_hash"])
    rb = harness.run(b, data_root=data_root, runs_root=str(runs), results_path=str(tmp_path / "r.jsonl"))
    pa = np.load(runs / a["config_hash"] / "params.npz")["flat"]
    pb = np.load(runs / rb["config_hash"] / "params.npz")["flat"]
    assert np.array_equal(pa, pb)
    c = copy.deepcopy(BASE)
    c["train"].update(steps=2, init_from=a["config_hash"], lora_rank=2)
    rc = harness.run(c, data_root=data_root, runs_root=str(runs), results_path=str(tmp_path / "r.jsonl"))
    assert 0 < rc["n_trainable"] < 0.1 * rc["n_params"]
