"""E3–E5 designs (experiments/designs.py) checked against their specs: run counts, no held-out
regime in pretraining, every fine-tune pointing at its own pretrain run, and every E4 variant
building a model that runs."""
import os
import sys

import jax
import jax.numpy as jnp

from aurora import harness, model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import designs as D  # noqa: E402


def _tags(split):
    return {x["tag"] for x in split}


def test_e3():
    pre, ft, sc = D.e3(harness.config_hash)
    assert (len(pre), len(ft), len(sc)) == (3, 15, 15)
    for p in pre:
        assert not _tags(p["data"]["train"]) & _tags(p["data"]["test"])
        assert len(p["data"]["train"]) == 16 and len(p["data"]["test"]) == 5
    hashes = {harness.config_hash(p): p["train"]["seed"] for p in pre}
    for f in ft:
        assert hashes[f["train"]["init_from"]] == f["train"]["seed"] and f["data"]["norm_from_init"]
    assert len({harness.config_hash(c) for c in pre + ft + sc}) == 33


def test_e4_variants_build_and_run():
    runs = D.e4()
    assert len(runs) == 27 and len({harness.config_hash(c) for c in runs}) == 27
    by = {c["name"].rsplit("_s", 1)[0]: c for c in runs if c["train"]["seed"] == 0}
    assert by["e4_pf_compute"]["train"]["steps"] * 2 == by["e4_pf_steps"]["train"]["steps"]
    assert by["e4_noshift"]["model"]["backbone"]["shift"] is False and by["e4_mae"]["train"]["loss"] == "mae"
    x = jnp.zeros((4, 32, 32), jnp.float32)
    for name in ("e4_base", "e4_noshift", "e4_depth6", "e4_L1", "e4_L6"):
        m = dict(by[name]["model"], D=16, heads=2)
        m["backbone"] = dict(m["backbone"], enc_heads=[1, 2, 2], dec_heads=[2, 2, 1])
        mc = harness.model_config(m)
        p = model.init(jax.random.PRNGKey(0), mc, dtype=jnp.float32)
        y = model.apply(p, x, x, model.VARS, 0.2, mc)
        assert y.shape == (4, 32, 32)


def test_e5():
    pre, ft, sc = D.e5(harness.config_hash)
    assert (len(pre), len(ft), len(sc)) == (3, 24, 12)
    for p in pre:
        assert "re1000_kf4" not in _tags(p["data"]["train"])
    for f in ft:
        assert f["data"]["train"][0]["tag"] == "re1000_kf4"
        assert not set(f["data"]["train"][0]["seeds"]) & set(D.E5_EVAL)
        assert (f["train"]["lora_rank"] == 4) == ("_lora_" in f["name"])
    assert len({harness.config_hash(c) for c in pre + ft + sc}) == 39


def _row(rows, c, key, **vals):
    rows.setdefault(harness.config_hash(c), {"metrics": {}})["metrics"][key] = vals


def test_analysis_e3_e4_e5_on_planted_rows():
    import numpy as np

    import analyse_e345 as A

    rng = np.random.default_rng(0)
    jit = lambda: 1 + 0.002 * rng.standard_normal()
    # E3: zero-shot ratio rises with distance, the advantage falls but stays positive
    rows = {}
    pre, ft, sc = D.e3(harness.config_hash)
    for p in pre:
        for r in D.E3_TESTS:
            _row(rows, p, f"test/{D.tag(r)}", ratio=0.4 + 0.3 * np.log(r[0] / 250) * jit())
    for c in ft + sc:
        t = c["data"]["train"][0]["tag"]
        d = np.log(int(t[2:].split("_")[0]) / 250)
        e = 0.2 * (1 - (0.25 - 0.15 * d)) if c in ft else 0.2
        _row(rows, c, f"val/{t}", rel_err=e * jit(), rollout_drift=2 * e * jit())
    assert A.e3(rows)["verdicts"] == {f"E3-P{i}": "HIT" for i in range(1, 5)}
    # E4: every planted effect inside its predicted range
    eff = {"base": (0, 0), "noshift": (0.1, 0.1), "depth6": (0.08, 0.05), "depth24": (0.0, 0.0),
           "L1": (0.08, 0.05), "L6": (0.01, 0.0), "pf_steps": (0.0, -0.12), "pf_compute": (0.02, 0.02),
           "mae": (0.05, 1.0)}
    rows = {}
    for c in D.e4():
        one, drift = eff[c["name"].rsplit("_s", 1)[0][3:]]
        for t in ("re100_kf4", "re300_kf4"):
            _row(rows, c, f"val/{t}", rel_err=0.1 * (1 + one) * jit(), rollout_drift=0.5 * (1 + drift) * jit())
    res = A.e4(rows)
    assert res["verdicts"] == {f"E4-P{i}": "HIT" for i in range(1, 6)}
    # E5: gains rise with k, LoRA falls behind, the advantage over scratch shrinks
    rows = {}
    pre, ft, sc = D.e5(harness.config_hash)
    for p in pre:
        _row(rows, p, "test/re1000_kf4", rel_err=0.2)
    gain_full = {1: 0.02, 2: 0.05, 4: 0.08, 6: 0.10}
    ratio = {1: 0.9, 2: 0.7, 4: 0.5, 6: 0.3}
    adv = {1: 0.4, 2: 0.3, 4: 0.2, 6: 0.1}
    for c in ft + sc:
        k = len(c["data"]["train"][0]["seeds"])
        ef = 0.2 * (1 - gain_full[k])
        e = 0.2 * (1 - gain_full[k] * ratio[k]) if c["train"]["lora_rank"] else ef
        e = ef / (1 - adv[k]) if c in sc else e
        _row(rows, c, "val/re1000_kf4", rel_err=e * jit())
    assert A.e5(rows)["verdicts"] == {f"E5-P{i}": "HIT" for i in range(1, 5)}
