"""E1 tooling checks (handouts/e1_volume_diversity.md): design counts, per-regime seeds, and a tiny
pretrain -> fine-tune chain through the harness with the normalisation carried over."""
import copy
import json
import os
import sys

import numpy as np

from aurora import harness
from kdata import finalize, generate_dataset, make_splits, read_manifest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "e1"))
import e1_design as E  # noqa: E402


def test_design_counts():
    need = E.seeds_needed()
    assert sum(need.values()) == 326
    assert need[E.TARGET] == 6 and need[(200, 3)] == 64 and need[(160, 3)] == 16 and need[(125, 4)] == 4
    pre, ft, sc = E.run_configs(harness.config_hash)
    assert (len(pre), len(ft), len(sc)) == (27, 27, 3)
    tgt = E.tag(E.TARGET)
    for p, f in zip(pre, ft):
        assert all(x["tag"] != tgt for x in p["data"]["train"])
        assert p["data"]["test"] == [{"tag": tgt, "seeds": E.EVAL_SEEDS}]
        assert f["train"]["init_from"] == harness.config_hash(p) and f["data"]["norm_from_init"]
        assert f["data"]["val"] == p["data"]["test"]
        n_traj = sum(len(x["seeds"]) for x in p["data"]["train"])
        assert f"_T{n_traj}_" in p["name"]
    assert all(s["train"]["steps"] == E.STEPS_PRE + E.STEPS_FT for s in sc)
    assert len({harness.config_hash(c) for c in pre + ft + sc}) == 57


def test_per_regime_seeds(tmp_path):
    cfg = E.dataset_config({r: 64 for r in E.all_regimes()}, name="t", T_spin=0.2, T_rec=0.4, N_out=(16,))
    cfg["regimes"] = [dict(cfg["regimes"][0], seeds=[0, 1]), dict(cfg["regimes"][1], seeds=[0])]
    generate_dataset(cfg, str(tmp_path / "t"), verbose=False)
    man = finalize(str(tmp_path / "t"))
    assert sorted((t["tag"], t["seed"]) for t in man["trajectories"]) == [("re125_kf3", 0), ("re300_kf4", 0),
                                                                           ("re300_kf4", 1)]
    assert read_manifest(str(tmp_path / "t"))["config"]["regimes"][0]["seeds"] == [0, 1]


def test_pretrain_finetune_chain(tmp_path):
    ds = dict(name="e1t", regimes=[dict(Re=100, kf=4, N_gen=64, seeds=[0, 1, 2]), dict(Re=300, kf=4, N_gen=64, seeds=[0])],
              T_spin=1.0, T_rec=15.0, save_dt=0.2, N_out=[32])
    generate_dataset(ds, str(tmp_path / "e1t"), verbose=False)
    finalize(str(tmp_path / "e1t"))
    small = {"D": 16, "P": 4, "L": 2, "heads": 2,
             "backbone": {"enc_depths": [1, 1], "enc_heads": [1, 2], "dec_depths": [1, 1],
                          "dec_heads": [2, 1], "window": [2, 4, 4]}}
    pre = E._run("p", "gate: pretrain", [{"tag": "re300_kf4", "seeds": [0]}], [{"tag": "re100_kf4", "seeds": [1, 2]}],
                 [], ["test"], 3, 1e-3, 0, batch=2, model=small, dataset="e1t", N=32)
    ft = E._run("f", "gate: finetune", [{"tag": "re100_kf4", "seeds": [0]}], [], [{"tag": "re100_kf4", "seeds": [1, 2]}],
                ["val"], 2, 1e-4, 0, init_from=harness.config_hash(pre), batch=2, model=small, dataset="e1t", N=32)
    kw = dict(data_root=str(tmp_path), runs_root=str(tmp_path / "runs"), results_path=str(tmp_path / "r.jsonl"))
    rp = harness.run(pre, **kw)
    rf = harness.run(ft, **kw)
    assert "test/re100_kf4" in rp["metrics"] and "rollout_drift" in rp["metrics"]["test/re100_kf4"]
    assert "val/re100_kf4" in rf["metrics"]
    zp = np.load(tmp_path / "runs" / rp["config_hash"] / "params.npz")
    zf = np.load(tmp_path / "runs" / rf["config_hash"] / "params.npz")
    np.testing.assert_array_equal(zp["mean"], zf["mean"])
    np.testing.assert_array_equal(zp["std"], zf["std"])
    # the target is refused as training data of a run that holds it out
    bad = copy.deepcopy(pre)
    bad["data"]["train"].append({"tag": "re100_kf4", "seeds": [0]})
    try:
        make_splits(str(tmp_path / "e1t"), bad["data"]["train"], bad["data"]["test"])
        raise AssertionError("leak not refused")
    except ValueError:
        pass
    assert len(open(tmp_path / "r.jsonl").read().splitlines()) == 2
    json.loads(open(tmp_path / "r.jsonl").readline())
